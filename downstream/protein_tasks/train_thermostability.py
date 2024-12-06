import os
import csv
import copy
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random
from transformers import Trainer, TrainingArguments, BertTokenizer, EsmTokenizer, EsmModel, AutoConfig, AutoModelForSequenceClassification, EarlyStoppingCallback


import torch
import transformers
import sklearn
import scipy
import numpy as np
from torch.utils.data import Dataset
from torchmetrics.utilities import dim_zero_cat
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
root_path = os.getcwd()
sys.path.append(root_path)
sys.path.append(parent_dir)


from model.dnabert2.bert_layers import BertForSequenceClassification as DNABERT2ForClassification
from model.ntv2.modeling_esm import EsmForSequenceClassification as NTv2ForSequenceClassification
from model.rnafm.modeling_rnafm import RnaFmForSequenceClassification
from model.esm.modeling_esm import EsmForSequenceClassification
from model.lucagplm.v2_0.modeling_gplm import LucaOneForSequenceClassification

# from peft import LoraConfig, get_peft_model

early_stopping = EarlyStoppingCallback(early_stopping_patience=20)

@dataclass
class ModelArguments:
    tokenizer_name_or_path: Optional[str] = field(default="facebook/esm1b_t33_650M_UR50S")
    model_name_or_path: Optional[str] = field(default="facebook/esm1b_t33_650M_UR50S")
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    
    # use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    # use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    # lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"}) # 8
    # lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    # lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    # lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default="train.csv", metadata={"help": "Path to the training data."})
    data_val_path: str = field(default="valid.csv", metadata={"help": "Path to the training data."})
    data_test_path: str = field(default="test.csv", metadata={"help": "Path to the test data. is list"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    tokenizer_cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1022, metadata={"help": "Maximum sequence length."})
    output_dir: str = field(default="output")
    freeze_backbone: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    model_type: str = field(default='esm-1b')
    token_type: str = field(default='esm-1b')
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=5000)
    eval_steps: int = field(default=5000)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    load_best_model_at_end: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=8)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    
    log_dir: str = field(default="output")
    
    report_to: str = field(default="wandb")
    stage: str = field(default='0')

    train_from_scratch: bool = field(default=False)
    attn_implementation: str = field(default="eager")

    seq_type: str = field(default=None)
    trunc_type: str = field(default=None)
    lucaone_args: object = field(default=None) # ?
    lucaone_model_args: object = field(default=None) # ?
    


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed, seed = {args.seed}")



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        
        # data must in the format of [text, label, *]
        texts = [d[0].upper() for d in data]
        labels = [d[1] for d in data]
        labels = [json.loads(label) for label in labels if isinstance(label, str)]


        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)
            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        self.input_ids = texts
        self.labels = labels
        self.num_labels = 585

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def gene_seq_replace(seq):
    '''
    Nucleic acid gene replace: A->1, U/T->2, C->3, G->4, N->5
    :param seq:
    :return:
    '''
    new_seq = ""
    for ch in seq:
        if ch in ["A", "a"]:
            new_seq += "1"
        elif ch in ["T", "U", "t", "u"]:
            new_seq += "2"
        elif ch in ["C", "c"]:
            new_seq += "3"
        elif ch in ["G", "g"]:
            new_seq += "4"
        else: # unknown
            new_seq += "5"
    return new_seq



@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels"))
        
        if self.args.model_type == 'lucaOne':
            input_ids = [gene_seq_replace(input_id) for input_id in input_ids]

        sequences = self.tokenizer(
            input_ids, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.model_max_length
        )
        if self.args.model_type == 'lucaOne':
            sequences['token_type_ids'] = torch.zeros_like(sequences['attention_mask'])
            sequences['position_ids'] = None

        labels = torch.Tensor(labels) # .int()
        sequences['labels'] = labels
        return sequences


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    return {
        "valid_spearman": spearmanr(logits, labels)[0],
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): 
        logits = logits[0]
    if logits.ndim == 2: 
        logits = logits.flatten()
    return calculate_metric_with_sklearn(logits, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)

    # load tokenizer
    print(f"token type is : {training_args.token_type}")
    if training_args.token_type in ["esm-1b"]:
        cache_dir = os.path.join(training_args.tokenizer_cache_dir, model_args.tokenizer_name_or_path.split("/")[-1])
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
    print(f"tokenizer is : {tokenizer}")
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

    
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, args=training_args)
    print(f'# train: {len(train_dataset)}, val:{len(val_dataset)}')


    # load model
    model_name_or_path = training_args.cache_dir
    print(f"train_dataset.num_labels: {train_dataset.num_labels}")
    print(f"model type is : {training_args.model_type}")
    print(f'Loading pretrained ckpts from {model_name_or_path}')
    
    if training_args.model_type in ["esm-1b", "esm-2"]:      
        model = EsmForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    elif training_args.model_type == "dnabert2":
        model = DNABERT2ForClassification.from_pretrained(
            model_name_or_path,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
            use_alibi=model_args.use_alibi
        )
    elif training_args.model_type == "ntv2":
        model = NTv2ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    elif training_args.model_type in ["rna-fm", "beacon"]:
        model = RnaFmForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    elif training_args.model_type in ["lucaone"]:
        model = LucaOneForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    else:
        raise f"Model Type {training_args.model_type} Is Not Supported."


    if training_args.freeze_backbone:
        if training_args.model_type in ['esm-2', "esm-1b"]:
            no_optim = ["embeddings", "encoder"]
        elif training_args.model_type in ['dnabert2', "ntv2", "rna-fm", "beacon"]:
            no_optim = ["encoder", "embeddings.position_embeddings.weight"]
        elif training_args.model_type in ["lucaOne"]:
            no_optim = ["lucaone"]
        else:
            raise "Wrong Model Type"

        for n, p in model.named_parameters():
            if any(nd in n for nd in no_optim):
                p.requires_grad = False
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"gradient parameter: {n}")


    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )

    trainer.train()

    if training_args.save_model:
        trainer.save_state()
    
    if training_args.eval_and_save_results:
        data_test_list = data_args.data_test_path.replace(" ", "").split(",")
        print(f"data_test_list = {len(data_test_list)}")

        for data_test_name in data_test_list:
            print(f"evaluating data_test_name = {data_test_name}")
            test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                            data_path=os.path.join(data_args.data_path, data_test_name), 
                                            kmer=data_args.kmer)
            results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
            os.makedirs(results_path, exist_ok=True)
            results_test = trainer.evaluate(eval_dataset=test_dataset)
            with open(os.path.join(results_path, f"{data_test_name}_results.json"), "w") as f:
                json.dump(results_test, f, indent=4)




if __name__ == "__main__":
    train()
