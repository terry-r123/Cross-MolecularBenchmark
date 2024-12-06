import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModelForSequenceClassification, EarlyStoppingCallback

import torch
import transformers
import sklearn
import scipy
import numpy as np
import re
from torch.utils.data import Dataset
from torchmetrics.utilities import dim_zero_cat
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from model.rnalm.modeling_rnalm import RnaLmForSequenceClassification
from model.rnalm.rnalm_config import RnaLmConfig
from model.esm.modeling_esm import EsmForSequenceClassification
from model.esm.esm_config import EsmConfig
from model.dnabert2.bert_layers import BertForSequenceClassification as DNABERT2ForClassification
from model.prompt_dnabert2.bert_layers import BertForSequenceClassificationPromptTokenAvg
from model.prompt_dnabert2.tokenization_6mer import DNATokenizer
from model.dnabert1.dnabert_layer import DNABertForSequenceClassification as DNABERT1ForClassification
from model.ntv2.modeling_esm import EsmForSequenceClassification as NTv2ForSequenceClassification
from model.hyenadna.tokenization_hyena import HyenaDNATokenizer
from model.hyenadna.modeling_hyena import HyenaDNAForSequenceClassification
from model.rnafm.modeling_rnafm import RnaFmForSequenceClassification
from model.rnabert.modeling_rnabert import RnaBertForSequenceClassification
from model.rnamsm.modeling_rnamsm import RnaMsmForSequenceClassification
from model.splicebert.modeling_splicebert import SpliceBertForSequenceClassification
from model.utrbert.modeling_utrbert import UtrBertForSequenceClassification
from model.utrlm.modeling_utrlm import UtrLmForSequenceClassification
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
from model.rnalm.rnalm_tokenizer import RnaLmTokenizer

# change for LucaOne
from collections import OrderedDict
from transformers import AutoTokenizer, PretrainedConfig
from model.lucagplm.v0_2.lucaone_gplm import LucaGPLM as LucaGPLMV0_2
from model.lucagplm.v0_2.lucaone_gplm_config import LucaGPLMConfig as LucaGPLMConfigV0_2
from model.lucagplm.v0_2.alphabet import Alphabet as AlphabetV0_2
# from model.lucagplm.v2_0.lucaone_gplm import LucaGPLM as LucaGPLMV2_0
from model.lucagplm.v2_0.lucaone_gplm_config import LucaOneConfig
from model.lucagplm.v2_0.alphabet import Alphabet as AlphabetV2_0
from model.lucagplm.v2_0.modeling_gplm import LucaOneForSequenceClassification

from peft import LoraConfig, get_peft_model
# change for LucaOne

os.environ["WANDB_PROJECT"] = "cds_ec"

early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"}) # 8
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    #lr_scheduler_type: str = field(default="cosine_with_restarts")
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default="wandb")
    metric_for_best_model : str = field(default="fmax")
    stage: str = field(default='0')
    model_type: str = field(default='dna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
# change for LucaOne
    freeze_backbone: bool = field(default=False)
    seq_type: str = field(default=None)
    trunc_type: str = field(default=None)
    lucaone_args: object = field(default=None) # ?
    lucaone_model_args: object = field(default=None) # ?
    dataloader_num_workers: int = field(default=8) # debug using 0
# change for LucaOne

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def remove_non_acgt_chars(sequence):
    pattern = '[^ACGT]'
    cleaned_sequence = re.sub(pattern, '', sequence)
    return cleaned_sequence

def replace_consecutive_ns(sequence, n=10):
    pattern = 'N' * n + '+'
    return re.sub(pattern, 'N', sequence)


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

def count_bases(sequence):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'Others': 0}
    total_chars = len(sequence)
    others_count = 0
    max_percentage = 0
    for char in sequence:
        if char in counts:
            counts[char] += 1
        else:
            counts['Others'] += 1
    for char, count in counts.items():
        percentage = (count / total_chars) * 100
        if percentage > 0 and char == 'Others':
            # pdb.set_trace()
            max_percentage = max(percentage, max_percentage)
            print(f"{char}: {percentage:.2f}%, sequence = {sequence}")
            others_count += 1
    return others_count, max_percentage

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

# change for LucaOne
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

def clean_seq(protein_id, seq, return_rm_index=False):
    seq = seq.upper()
    new_seq = ""
    has_invalid_char = False
    invalid_char_set = set()
    return_rm_index_set = set()
    for idx, ch in enumerate(seq):
        if 'A' <= ch <= 'Z' and ch not in ['J']:
            new_seq += ch
        else:
            invalid_char_set.add(ch)
            return_rm_index_set.add(idx)
            has_invalid_char = True
    if has_invalid_char:
        print("id: %s. Seq: %s" % (protein_id, seq))
        print("invalid char set:", invalid_char_set)
        print("return_rm_index:", return_rm_index_set)
    if return_rm_index:
        return new_seq, return_rm_index_set
    return new_seq
# change for LucaOne


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()        
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 3:
            # data is in the format of [text, label]
            texts = [d[0].upper().replace("U", "T") for d in data]
            labels = [d[2] for d in data]
            import ast
            labels = [json.loads(label) for label in labels if isinstance(label, str)]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        
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
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels"))
        if self.args.model_type == 'LucaOne':
            input_ids = [gene_seq_replace(input_id) for input_id in input_ids]
        sequences = self.tokenizer(
            input_ids, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.model_max_length
        )
        if self.args.model_type == 'LucaOne':
            sequences['token_type_ids'] = torch.zeros_like(sequences['attention_mask'])
            # sequences['position_ids'] = None
        labels = torch.Tensor(labels) # .int()
        sequences['labels'] = labels
        return sequences

"""
Manually calculate the mse and r^2.
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
    return {
    "fmax": all_f1.max().item()
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)

    # load tokenizer
    print(training_args.model_type)
    if training_args.model_type == 'rnalm' or  training_args.model_type == 'ntv2':
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type in ['esm2', 'esm-1b', 'rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif 'dna-esm' in training_args.model_type or 'rna-esm' in training_args.model_type:
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'hyenadna':
        tokenizer = HyenaDNATokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type == 'prompt-6mer':
        tokenizer = DNATokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            Model=training_args.model_max_length,
            cache_dir=training_args.cache_dir,
        )
    elif training_args.model_type == 'BEACON-B':
        if training_args.token_type != 'single':
            tokenizer = EsmTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=training_args.token_type
                )
        else:
            tokenizer = RnaLmTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
                token_type=training_args.token_type
                )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    print(tokenizer)
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    if 'mer' in training_args.token_type:
        data_args.kmer=int(training_args.token_type[0])

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     kmer=data_args.kmer)
    # test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
    #                                  data_path=os.path.join(data_args.data_path, data_args.data_test_path), 
    #                                  kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    # holdout_HSPE1 = SupervisedDataset(tokenizer=tokenizer,
    #                                 data_path=os.path.join(data_path, "holdout_HSPE1.csv"))
    # holdout_SNHG6 = SupervisedDataset(tokenizer=tokenizer,
    #                                 data_path=os.path.join(data_path, "holdout_SNHG6.csv"))
    # holdout_WHAMMP2 = SupervisedDataset(tokenizer=tokenizer,
    #                                 data_path=os.path.join(data_path, "holdout_WHAMMP2.csv"))
    #data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)}')#,test:{len(test_dataset)}')

    # load model
    if training_args.model_type == 'rnalm':
        print(training_args.model_type)
        if training_args.train_from_scratch:
            print('Train from scratch')
            config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                )
            print(config)
            model =  RnaLmForSequenceClassification(
                config,
                )
        else:
            print(f'Loading {training_args.model_type} model')
            model =  RnaLmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                )
    elif training_args.model_type == 'rna-fm':      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaFmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'rnabert':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif training_args.model_type == 'rnamsm':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaMsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )        
    elif 'splicebert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = SpliceBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )       
    elif 'utrbert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrBertForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )  
    elif 'utr-lm' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrLmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )           
    elif training_args.model_type == 'hyenadna':
        if training_args.train_from_scratch:
            pass
        else:
            model = HyenaDNAForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
            )
    elif training_args.model_type == 'dnabert1':

        if training_args.train_from_scratch:
            pass
        else:
            print(f'Loading {training_args.model_type} model')  
            print(train_dataset.num_labels)
            model = DNABERT1ForClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                
            )
    elif training_args.model_type == 'dnabert2':

        if training_args.train_from_scratch:
            pass
        else:
            print(f'Loading {training_args.model_type} model')    
            print(train_dataset.num_labels)
            model = DNABERT2ForClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                use_alibi=model_args.use_alibi,            
            )
    elif training_args.model_type == "ntv2":
        if training_args.train_from_scratch:
            pass
        else:
            print(f'Loading {training_args.model_type} model')
            model = NTv2ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
            )
    elif training_args.model_type == 'prompt-6mer' or training_args.model_type == 'prompt-bpe':
        print(f'Loading {training_args.model_type} model')  
        print(train_dataset.num_labels)
        model = BertForSequenceClassificationPromptTokenAvg.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            use_alibi=model_args.use_alibi,
        )
    elif 'esm2' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading protein model {training_args.model_type} model')
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            ignore_mismatched_sizes=True, # change config for RNA downtasks
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
        print(model.config)
        
        # Access the embedding layer
        embedding_layer = model.esm.embeddings
        # Initialize the embedding layer with random values
        # torch.nn.init.normal_(embedding_layer.weight, mean=0.0, std=1.0)
        for embedding_layer in [model.esm.embeddings.word_embeddings]:
            print(embedding_layer)
            if isinstance(embedding_layer, torch.nn.Embedding):
                print(f"init embedding_layer: {embedding_layer}")
                embedding_layer.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if embedding_layer.padding_idx is not None:
                    embedding_layer.weight.data[embedding_layer.padding_idx].zero_()
    elif 'dna-esm' in training_args.model_type or 'rna-esm' in training_args.model_type:
        if training_args.train_from_scratch:
            print('Loading esm model')
            print('Train from scratch')
            config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels)
            model = transformers.AutoModelForSequenceClassification.from_config(
                config
                )
        else:
            print(training_args.model_type)
            print(f'Loading Nucleotide model {training_args.model_type} model')
            model = EsmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
            ) 
    elif training_args.model_type =='esm-1b':
        print(f"train_dataset.num_labels: {train_dataset.num_labels}")
        print(f'Loading protein model {training_args.model_type} model')
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            ignore_mismatched_sizes=True, # change config for RNA downtasks
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )

       # Access the embedding layer
        embedding_layer = model.esm.embeddings
        # Initialize the embedding layer with random values
        # torch.nn.init.normal_(embedding_layer.weight, mean=0.0, std=1.0)
        for embedding_layer in [model.esm.embeddings.word_embeddings]:
            print(embedding_layer)
            if isinstance(embedding_layer, torch.nn.Embedding):
                print(f"init embedding_layer: {embedding_layer}")
                embedding_layer.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                if embedding_layer.padding_idx is not None:
                    embedding_layer.weight.data[embedding_layer.padding_idx].zero_()
    elif training_args.model_type == 'BEACON-B':
        if training_args.train_from_scratch:
            print('Loading BEACON-B model')
            print(f'Train from scratch {training_args.model_type} model')
            config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
                attn_implementation=training_args.attn_implementation,)
            model = RnaLmForSequenceClassification(config)
        else:
            print(f'args.model_name_or_path: {model_args.model_name_or_path}')    
            model = RnaLmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
                attn_implementation=training_args.attn_implementation,
            )
# change for LucaOne
    elif training_args.model_type =='LucaOne':
        print(f'training_args.lucaone_args: \n{training_args.lucaone_args}')
        print(f'Loading {training_args.model_type} model')

        model = LucaOneForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=train_dataset.num_labels,
        )
        if model_args.use_lora and not training_args.freeze_backbone:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=list(model_args.lora_target_modules.split(",")),
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
# change for LucaOne
    print(model.config)
    print(model)
    
    # from torchviz import make_dot
    # from torch.utils.data import DataLoader

    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # for _, inputs in enumerate(train_loader):
    #     outputs = model(input_ids=inputs["input_ids"],
    #                     labels=inputs["labels"],
    #                     attention_mask=inputs["attention_mask"],
    #                     )
    #     break
    # import sys
    # sys.setrecursionlimit(20000)
    # make_dot(outputs.logits, params=dict(model.named_parameters())).render("dnabert_model", format="pdf")
    
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
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    data_test_list = data_args.data_test_path.replace(" ", "").split(",")
    print(f"data_test_list = {len(data_test_list)}")
    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        for data_test in data_test_list:
            data_test_name = data_test
            print(f"evaluating data_test_name = {data_test_name}")
            test_dataset = SupervisedDataset(tokenizer=tokenizer, args=training_args,
                                            data_path=os.path.join(data_args.data_path, data_test_name), 
                                            kmer=data_args.kmer)
            results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
            results = trainer.evaluate(eval_dataset=test_dataset)
            os.makedirs(results_path, exist_ok=True)
            results_test = trainer.evaluate(eval_dataset=test_dataset)
            with open(os.path.join(results_path, f"{data_test}_results.json"), "w") as f:
                json.dump(results_test, f, indent=4)
       


if __name__ == "__main__":
    train()
