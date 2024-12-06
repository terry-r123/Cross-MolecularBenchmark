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
# from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

# from model.lucagplm.v0_2.lucaone_gplm import LucaGPLM as LucaGPLMV0_2
# from model.lucagplm.v0_2.lucaone_gplm_config import LucaGPLMConfig as LucaGPLMConfigV0_2
# from model.lucagplm.v0_2.alphabet import Alphabet as AlphabetV0_2
# from model.lucagplm.v2_0.lucaone_gplm import LucaGPLM as LucaGPLMV2_0
# from model.lucagplm.v2_0.lucaone_gplm_config import LucaOneConfig
# from model.lucagplm.v2_0.alphabet import Alphabet as AlphabetV2_0
# from model.lucagplm.v2_0.modeling_gplm import LucaOneForSequenceClassification

# from peft import LoraConfig, get_peft_model
# change for LucaOne

early_stopping = EarlyStoppingCallback(early_stopping_patience=20)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/esm1b_t33_650M_UR50S")
    # use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    # use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    # use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    # lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"}) # 8
    # lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    # lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    # lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    # tokenizer_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")


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
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1022, metadata={"help": "Maximum sequence length."})
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
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default="wandb")
    stage: str = field(default='0')
    model_type: str = field(default='esm-1b')
    token_type: str = field(default='esm-1b')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    freeze_backbone: bool = field(default=False)
    seq_type: str = field(default=None)
    trunc_type: str = field(default=None)
    # lucaone_args: object = field(default=None) # ?
    # lucaone_model_args: object = field(default=None) # ?
    dataloader_num_workers: int = field(default=8) # debug using 0



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



@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels"))
        
        # if self.args.model_type == 'LucaOne':
        #     input_ids = [gene_seq_replace(input_id) for input_id in input_ids]
        sequences = self.tokenizer(
            input_ids, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.model_max_length
        )
        # if self.args.model_type == 'LucaOne':
        #     sequences['token_type_ids'] = torch.zeros_like(sequences['attention_mask'])
            # sequences['position_ids'] = None
        labels = torch.Tensor(labels) # .int()
        sequences['labels'] = labels
        return sequences


"""
Compute metrics used for huggingface trainer.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    pred = dim_zero_cat(logits)
    target = dim_zero_cat(labels)

    order = pred.argsort(descending=True, dim=1)


    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)

    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()

    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)

    all_order = pred.flatten().argsort(descending=True)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]

    precision = precision.flatten()
    recall = recall.flatten()
    
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return { "fmax": all_f1.max().item()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)


def convert_lmdb_to_csv():
    data_root = "/home/bingxing2/ailab/group/ai4bio/public/multi-omics/protein/downstream/EC/AF2/normal"
    train_path = os.path.join(data_root, "train")
    valid_path = os.path.join(data_root, "valid")
    test_path = os.path.join(data_root, "test")

    import lmdb
    env = lmdb.open(train_path)
    txn = env.begin()


    with open("train.csv", "w") as fw:
        csv_writer = csv.writer(fw)

        csv_writer.writerow(["text","label","name","chain"])

        length = int(txn.get(str("length").encode()).decode())
        for index in range(length):
            item = json.loads(txn.get(str(index).encode()).decode())
            csv_writer.writerow([item["seq"],item["label"],item["name"],item["chain"]])    
    
    env = lmdb.open(valid_path)
    txn = env.begin()

    with open("valid.csv", "w") as fw:
        csv_writer = csv.writer(fw)

        csv_writer.writerow(["text","label","name","chain"])

        length = int(txn.get(str("length").encode()).decode())
        for index in range(length):
            item = json.loads(txn.get(str(index).encode()).decode())
            csv_writer.writerow([item["seq"],item["label"],item["name"],item["chain"]])    
   
    env = lmdb.open(test_path)
    txn = env.begin()


    with open("test.csv", "w") as fw:
        csv_writer = csv.writer(fw)

        csv_writer.writerow(["text","label","name","chain"])

        length = int(txn.get(str("length").encode()).decode())
        for index in range(length):
            item = json.loads(txn.get(str(index).encode()).decode())
            csv_writer.writerow([item["seq"],item["label"],item["name"],item["chain"]])    
    

    exit()



def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)

    # load tokenizer
    print(f"token type is : {training_args.token_type}")
    if training_args.token_type in ["esm-1b"]:
        cache_dir = os.path.join(training_args.cache_dir, "esm1b_t33_650M_UR50S")
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
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
    cache_dir = os.path.join(training_args.cache_dir, model_args.model_name_or_path.split("/")[-1])
    print(f"train_dataset.num_labels: {train_dataset.num_labels}")
    print(f"model type is : {training_args.model_type}")
    print(f'Loading pretrained ckpts from {cache_dir}')
    
    if training_args.model_type in ["esm-1b", "esm-2_150M"]:      
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    elif training_args.model_type == "dnabert2":
        model = DNABERT2ForClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
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
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    elif training_args.model_type in ["rna-fm", "beacon"]:
        model = RnaFmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            num_labels=train_dataset.num_labels,
            ignore_mismatched_sizes=True, 
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
    else:
        assert f"Model Type {training_args.model_type} Is Not Supported."



    if training_args.freeze_backbone:
        if training_args.model_type in ['esm2']:
            no_optim = ["encoder"]
        elif training_args.model_type in ['esm-1b']:
            no_optim = ["encoder", "embeddings.position_embeddings.weight"]
        elif training_args.model_type in ["LucaOne"]:
            no_optim = ["lucaone"]
        else:
            no_optim = ["embeddings", "encoder"]
        for n,p in model.named_parameters():
            if any(nd in n for nd in no_optim):
                p.requires_grad = False
        for n,p in model.named_parameters():
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
    exit()
    trainer.train()

    
    if training_args.save_model:
        trainer.save_state()
    data_test_list = data_args.data_test_path.replace(" ", "").split(",")
    print(f"data_test_list = {len(data_test_list)}")


    if training_args.eval_and_save_results:
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
