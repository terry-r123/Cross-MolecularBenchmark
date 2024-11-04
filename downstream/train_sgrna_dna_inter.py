import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback

import torch
import transformers
import sklearn
import scipy
import numpy as np
import re
from torch.utils.data import Dataset

import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)


from model.dnabert2.bert_layers import BertModel as DNABERT2Model
from model.ntv2.modeling_esm import EsmModel as NTv2Model
from model.ntv2.esm_config import EsmConfig as NTv2Config 
from model.rnalm.modeling_rnalm import RnaLmModel
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.configuration_rnafm import RnaFmConfig
from model.rnafm.modeling_rnafm import RnaFmModel
from model.multi_molecule.modeling_multi import OmicsOmicsInter, RNADNAInter
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from model.multi_molecule.modeling_naive import ConvolutionalNetwork, ResNet, LSTM, RNADNANaiveClassifier
from typing import Optional

@dataclass
class ModelArguments:
    omics1_model_name_or_path: Optional[str] = field(default="")
    omics2_model_name_or_path: Optional[str] = field(default="")
    dna_model_name_or_path: Optional[str] = field(default="")
    rna_model_name_or_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_rna_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    omics1_kmer: int = field(default=-1, metadata={"help": "k-mer for input omics1ein sequence. -1 means not using k-mer."})
    dna_kmer: int = field(default=-1, metadata={"help": "k-mer for input DNA sequence. -1 means not using k-mer."})
    omics2_kmer: int = field(default=-1, metadata={"help": "k-mer for input RNA sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    omics1_model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    omics2_model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard")
    metric_for_best_model : str = field(default="spearman")
    stage: str = field(default='0')
    omics1_model_type: str = field(default='esm2')
    dna_model_type: str = field(default='dnabert2')
    rna_model_type: str = field(default='rna-fm')
    omics1_model_type: str = field(default='rna-fm')
    omics2_model_type: str = field(default='rna-fm')
    omics1_token_type: str = field(default='single')
    dna_token_type: str = field(default='bpe')
    rna_token_type: str = field(default='single')
    omics1_token_type: str = field(default='single')
    omics2_token_type: str = field(default='single')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    frozen_backbone: bool = field(default=False)
    init_embedding: bool = field(default=False)
    use_features: bool = field(default=False, metadata={"help": "whether to use additional features"})
    weight: Optional[str] = field(default=None)
    #weight: List[float] = field(default_factory=lambda: [0.0478, 0.9522])
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

def check_freeze(model):
    for name, param in model.named_parameters():       
        if param.requires_grad:
            print(f"{name} is not frozen")
        else:
            print(f"{name} is frozen")


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, args,
                 omics1_tokenizer: transformers.PreTrainedTokenizer, 
                 omics2_tokenizer: transformers.PreTrainedTokenizer, 
                 omics1_kmer: int = -1,  omics2_kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 12:
            omics1 = [d[1].upper().replace("U", "T") for d in data]  # sgrna           
            omics2 = [d[6].upper().replace("U", "T") for d in data]  # dna          
            labels = [float(d[11]) for d in data]
            features =  [d[2:6] + d[7:11] for d in data]

        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        labels = np.array(labels)
        labels = labels.tolist()
       
        if omics1_kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            #omics1 = load_or_generate_kmer(data_path.replace('.csv', '_omics1.csv'), omics1, kmer)
            omics1 = load_or_generate_kmer(data_path.replace('.csv', '_omics1.csv'), omics1, omics1_kmer)
            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        if omics2_kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            #omics1 = load_or_generate_kmer(data_path.replace('.csv', '_omics1.csv'), omics1, kmer)
            omics2 = load_or_generate_kmer(data_path.replace('.csv', '_omics2.csv'), omics2, omics2_kmer)
            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        # ensure tokenier
        if torch.distributed.get_rank() in [0, -1]:
            print(type(omics1[0]))
            print(omics1[0])
            test_example = omics1_tokenizer.tokenize(omics1[0])
            print(test_example)
            print(len(test_example))
            print(omics1_tokenizer(omics1[0]))

        self.omics1 = omics1
        self.omics2 = omics2
        self.labels = labels
        self.num_labels = 1
        self.features = features

    def __len__(self):
        return len(self.omics1)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        features = torch.tensor([list(map(int, list(f))) for f in self.features[i]], dtype=torch.float32)
        #return dict(input_ids=self.omics1[i],omics2_input_ids=self.omics2[i],labels=self.labels[i],features=self.features[i])
        return dict(input_ids=self.omics1[i],omics2_input_ids=self.omics2[i],labels=self.labels[i],features=features)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, omics1_tokenizer, omics2_tokenizer, args):
        self.omics1_tokenizer = omics1_tokenizer
        self.omics2_tokenizer = omics2_tokenizer

        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        omics1, omics2, labels, features = tuple([instance[key] for instance in instances] for key in ("input_ids", "omics2_input_ids","labels","features"))
        omics1_output = self.omics1_tokenizer(omics1, padding='longest', max_length=self.omics1_tokenizer.model_max_length, truncation=True, return_tensors='pt')
        omics2_output = self.omics2_tokenizer(omics2, padding='longest', max_length=self.omics2_tokenizer.model_max_length, truncation=True, return_tensors='pt')
        omics1_input_ids = omics1_output["input_ids"]
        omics1_attention_mask = omics1_output["attention_mask"]
        omics2_input_ids = omics2_output["input_ids"]
        omics2_attention_mask = omics2_output["attention_mask"]
        
        features = torch.stack(features)
        labels = torch.Tensor(labels).float()
        
        return dict(
            input_ids=omics1_input_ids,
            labels=labels,
            attention_mask=omics1_attention_mask,
            omics2_input_ids=omics2_input_ids,
            omics2_attention_mask=omics2_attention_mask,
            features=features,
        )


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "mse": sklearn.metrics.mean_squared_error(labels, logits),
        "spearman" : scipy.stats.spearmanr(labels, logits)[0],
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
    omics1_tokenizer=None
    omics2_tokenizer=None
    # load omics1 tokenizer
    if training_args.omics1_model_type in ['rnalm', 'cnn', 'resnet', 'lstm']:
        omics1_tokenizer = EsmTokenizer.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.omics1_model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.omics1_model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        omics1_tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.omics1_model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        omics1_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.omics1_model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    # load omics2 tokenizer
    if training_args.omics2_model_type in ['rnalm', 'cnn', 'resnet', 'lstm']:
        omics2_tokenizer = EsmTokenizer.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.omics2_model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.rna_model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        omics2_tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.omics2_model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        omics2_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.omics2_model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    
    if 'mer' in training_args.omics1_token_type:
        data_args.omics1_kmer=int(training_args.omics1_token_type[0])
    if 'mer' in training_args.omics2_token_type:
        data_args.omics2_kmer=int(training_args.omics2_token_type[0])
    # define datasets and data collator

    train_dataset = SupervisedDataset(omics1_tokenizer=omics1_tokenizer,  omics2_tokenizer=omics2_tokenizer, 
                                    args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      omics1_kmer=data_args.omics1_kmer, omics2_kmer=data_args.dna_kmer)
    val_dataset = SupervisedDataset(omics1_tokenizer=omics1_tokenizer,  omics2_tokenizer=omics2_tokenizer,   args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     omics1_kmer=data_args.omics1_kmer, omics2_kmer=data_args.dna_kmer)
    test_dataset = SupervisedDataset(omics1_tokenizer=omics1_tokenizer,  omics2_tokenizer=omics2_tokenizer,  args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path), 
                                     omics1_kmer=data_args.omics1_kmer, omics2_kmer=data_args.dna_kmer)
    data_collator = DataCollatorForSupervisedDataset(omics1_tokenizer=omics1_tokenizer,  omics2_tokenizer=omics2_tokenizer, args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    
    # load model
    if 'beacon' in training_args.omics1_model_type:
        print(f'Loading {training_args.omics1_model_type} model')
        omics1_model = RnaLmModel.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            attn_implementation=training_args.attn_implementation,
            )
        omics1_config = RnaLmConfig.from_pretrained(model_args.omics1_model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                )
    elif training_args.omics1_model_type == 'rna-fm':      
        print(f'Loading {training_args.omics1_model_type} model')
        omics1_model = RnaFmModel.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
        )     
        omics1_config = RnaFmConfig.from_pretrained(model_args.omics1_model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                )
    elif training_args.omics1_model_type == 'cnn':  # Add this condition for CNN
        print(f'Loading {training_args.omics1_model_type} model')
        omics1_model = ConvolutionalNetwork(
            input_dim=33,  # Match with ESMTokenizer vocab size
            hidden_dims=[1024, 1024],
            kernel_size=5,
            stride=1,
            padding=2
        )
        # Manually define omics1_config as a simple configuration object for consistency
        omics1_config = {
            'num_labels': train_dataset.num_labels,
            'hidden_size': 1024, 
            'problem_type': "regression"
        }
    elif training_args.omics1_model_type == 'resnet':  # Add this condition for ResNet
        print(f'Loading {training_args.omics1_model_type} model')
        omics1_model = ResNet(
            input_dim=33, 
            hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512], 
            kernel_size=3,
            stride=1,
            padding=1,
            activation="gelu",  
            short_cut=False,
            concat_hidden=False,
            layer_norm=True,
            dropout=0.1
        )
        omics1_config = {
            'num_labels': train_dataset.num_labels,
            'hidden_size': 512,  
            'problem_type': "regression"
        }
    elif training_args.omics1_model_type == 'lstm':  # Add this condition for LSTM
        print(f'Loading {training_args.omics1_model_type} model')
        omics1_model = LSTM(
            input_dim=33,
            hidden_dim=640,  
            num_layers=3,  
            activation='tanh',  
            layer_norm=True, 
            dropout=0.1 
        )
        omics1_config = {
            'num_labels': train_dataset.num_labels,
            'hidden_size': 1280,  # Set to match the hidden_dim * 2 (bidirectional LSTM)
            'problem_type': "regression"
        }

    if training_args.omics2_model_type == "dnabert2":
        print(f'Loading {training_args.omics2_model_type} model')
        omics2_model = DNABERT2Model.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            use_alibi=model_args.use_alibi,
            problem_type="regression",

        )
        omics2_config = AutoConfig.from_pretrained(model_args.omics2_model_name_or_path,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
            )
    elif "nt_v2" in training_args.omics2_model_type:
        print(f'Loading {training_args.omics2_model_type} model')
        omics2_model = NTv2Model.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
        omics2_config = NTv2Config.from_pretrained(model_args.omics2_model_name_or_path,
                    num_labels=train_dataset.num_labels,
                    problem_type="regression",
                    )
    elif training_args.omics2_model_type == 'cnn':  # Add this condition for CNN
        print(f'Loading {training_args.omics2_model_type} model')
        omics2_model = ConvolutionalNetwork(
            input_dim=33,  # Example input dim, modify as needed
            hidden_dims=[1024, 1024],  # Example hidden dims, modify as needed
            kernel_size=5,
            stride=1,
            padding=2
        )
        # Manually define omics2_config for the CNN model
        omics2_config = {
            'num_labels': train_dataset.num_labels,
            'hidden_size': 1024,  # Set to match the hidden_dims of the CNN model
            'problem_type': "regression"
        }
    elif training_args.omics2_model_type == 'resnet':
        print(f'Loading {training_args.omics1_model_type} model')
        omics2_model = ResNet(
            input_dim=33, 
            hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512], 
            kernel_size=3,
            stride=1,
            padding=1,
            activation="gelu",  
            short_cut=False,
            concat_hidden=False,
            layer_norm=True,
            dropout=0.1
        )
        omics2_config = {
            'num_labels': train_dataset.num_labels,
            'hidden_size': 512,  
            'problem_type': "regression"
        }
    elif training_args.omics2_model_type == 'lstm':  # Add this condition for LSTM
        print(f'Loading {training_args.omics1_model_type} model')
        omics2_model = LSTM(
            input_dim=33,
            hidden_dim=640,  
            num_layers=3,  
            activation='tanh',  
            layer_norm=True, 
            dropout=0.1 
        )
        omics2_config = {
            'num_labels': train_dataset.num_labels,
            'hidden_size': 1280,  # Set to match the hidden_dim * 2 (bidirectional LSTM)
            'problem_type': "regression"
        }
    
    if training_args.omics1_model_type in ['cnn', 'resnet','lstm'] and training_args.omics2_model_type in ['cnn','resnet','lstm']:
        print(f'training_agr.use_features: {training_args.use_features}')
        model = RNADNANaiveClassifier(omics1_config, omics2_config, omics1_model, omics2_model, training_args.use_features)
    else: 
        model = RNADNAInter(omics1_config, omics2_config, omics1_model, omics2_model, training_args.use_features)
    if training_args.frozen_backbone:
        for name, param in model.named_parameters():
            if ('classifier' not in name) and ('pooler' not in name) and ('position_embeddings' not in name):
                param.requires_grad = False
        if torch.distributed.get_rank() in [0, -1]:
            check_freeze(model)    
    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=omics1_tokenizer,
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

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        
        os.makedirs(results_path, exist_ok=True)
        results_test = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results_test, f, indent=4)

if __name__ == "__main__":
    train()
