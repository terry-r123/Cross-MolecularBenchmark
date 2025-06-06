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

from model.rnalm.modeling_rnalm import RnaLmModel
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.configuration_rnafm import RnaFmConfig
from model.rnafm.modeling_rnafm import RnaFmModel
from model.multi_molecule.modeling_multi import OmicsOmicsInter
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
early_stopping = EarlyStoppingCallback(early_stopping_patience=10)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from model.multi_molecule.modeling_naive import ConvolutionalNetwork, ResNet, LSTM, MultiOmicsNaiveClassifier
from typing import Optional
from torchinfo import summary

@dataclass
class ModelArguments:
    omics1_model_name_or_path: Optional[str] = field(default="")
    dna_model_name_or_path: Optional[str] = field(default="")
    omics2_model_name_or_path: Optional[str] = field(default="")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_omics2_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    omics1_kmer: int = field(default=-1, metadata={"help": "k-mer for input protein sequence. -1 means not using k-mer."})
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
    metric_for_best_model : str = field(default="matthews_correlation")
    stage: str = field(default='0')
    omics1_model_type: str = field(default='esm2')
    dna_model_type: str = field(default='dnabert2')
    omics2_model_type: str = field(default='rna-fm')
    omics1_token_type: str = field(default='single')
    dna_token_type: str = field(default='bpe')
    omics2_token_type: str = field(default='single')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    frozen_backbone: bool = field(default=False)
    init_embedding: bool = field(default=False)

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

        if len(data[0]) == 7:
            omics1 = [d[-2].upper() for d in data]  #protein
            omics2 = [d[-3].upper().replace("U", "T") for d in data]    #rna
            labels = [float(d[-1]) for d in data]
            
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        labels = np.array(labels)
        labels = labels.tolist()
        
        if omics2_kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            #omics1 = load_or_generate_kmer(data_path.replace('.csv', '_prot.csv'), omics1, kmer)
            omics2 = load_or_generate_kmer(data_path.replace('.csv', '_rna.csv'), omics2, omics2_kmer)
            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        # ensure tokenier
        print(type(omics1[0]))
        print(omics1[0])
        test_example = omics1_tokenizer.tokenize(omics1[0])
        print(test_example)
        print(len(test_example))
        print(omics1_tokenizer(omics1[0]))

        self.omics1 = omics1
        self.omics2 = omics2
        self.labels = labels
        self.num_labels = 2

    def __len__(self):
        return len(self.omics2)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        return dict(input_ids=self.omics1[i],omics2_input_ids=self.omics2[i],labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, omics1_tokenizer, omics2_tokenizer, args):
        self.omics1_tokenizer = omics1_tokenizer
        self.omics2_tokenizer= omics2_tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        omics1, omics2, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "omics2_input_ids","labels"))
        omics1_output = self.omics1_tokenizer(omics1, padding='longest', max_length=self.omics1_tokenizer.model_max_length, truncation=True, return_tensors='pt')
        omics2_output = self.omics2_tokenizer(omics2, padding='longest', max_length=self.omics2_tokenizer.model_max_length, truncation=True, return_tensors='pt')
        omics1_input_ids = omics1_output["input_ids"]
        omics1_attention_mask = omics1_output["attention_mask"]
        omics2_input_ids = omics2_output["input_ids"]
        omics2_attention_mask = omics2_output["attention_mask"]
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=omics1_input_ids,
            labels=labels,
            attention_mask=omics1_attention_mask,
            omics2_input_ids=omics2_input_ids,
            omics2_attention_mask=omics2_attention_mask
        )

"""
Manually calculate the spearman.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
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

    # load tokenizer based on model types
    if any(model_type in training_args.omics1_model_type for model_type in ['esm', 'cnn', 'resnet', 'lstm']):
        omics1_tokenizer = EsmTokenizer.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    if training_args.omics2_model_type in ['rnalm', 'cnn', 'resnet', 'lstm']:
        omics2_tokenizer = EsmTokenizer.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.omics2_model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        omics2_tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        omics2_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    
    if 'mer' in training_args.omics2_token_type:
        data_args.omics2_kmer=int(training_args.omics2_token_type[0])

    # define datasets and data collator
    train_dataset = SupervisedDataset(omics1_tokenizer=omics1_tokenizer, omics2_tokenizer=omics2_tokenizer, 
                                    args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      omics1_kmer=data_args.omics1_kmer,  omics2_kmer=data_args.omics2_kmer)
    val_dataset = SupervisedDataset(omics1_tokenizer=omics1_tokenizer,  omics2_tokenizer=omics2_tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     omics1_kmer=data_args.omics1_kmer, omics2_kmer=data_args.omics2_kmer)
    test_dataset = SupervisedDataset(omics1_tokenizer=omics1_tokenizer, omics2_tokenizer=omics2_tokenizer, args=training_args,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path), 
                                     omics1_kmer=data_args.omics1_kmer, omics2_kmer=data_args.omics2_kmer)
    data_collator = DataCollatorForSupervisedDataset(omics1_tokenizer=omics1_tokenizer, omics2_tokenizer=omics2_tokenizer, args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # Print out sample tokenized inputs to determine input sizes    
    sample_batch = data_collator([train_dataset[0]])  # Get a batch of size 1 from the train dataset

    # Access the tokenized inputs
    sample_omics1_input = sample_batch['input_ids'].long()
    sample_omics2_input = sample_batch['omics2_input_ids'].long()

    print(f"Sample omics1 input shape: {sample_omics1_input.shape}")  # Example output: (batch_size, sequence_length)
    print(f"Sample omics2 input shape: {sample_omics2_input.shape}") 

    # load model based on omics1_model_type and omics2_model_type
    if 'esm' in training_args.omics1_model_type:
        print(f'Loading {training_args.omics1_model_type} model')
        omics1_model =  EsmModel.from_pretrained(
            model_args.omics1_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            )
        omics1_config =  AutoConfig.from_pretrained(
            model_args.omics1_model_name_or_path,
                    num_labels=train_dataset.num_labels,
                    problem_type="single_label_classification",
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
            'problem_type': "single_label_classification"
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
            'problem_type': "single_label_classification"
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
            'problem_type': "single_label_classification"
        }

    if 'beacon' in training_args.omics2_model_type:
        print(f'Loading {training_args.omics2_model_type} model')
        omics2_model = RnaLmModel.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            attn_implementation=training_args.attn_implementation,
            )
    elif training_args.omics2_model_type == 'rna-fm':      
        print(f'Loading {training_args.omics2_model_type} model')
        omics2_model = RnaFmModel.from_pretrained(
            model_args.omics2_model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
        )     
        omics2_config = RnaFmConfig.from_pretrained(model_args.omics2_model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="single_label_classification",
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
            'problem_type': "single_label_classification"
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
            'problem_type': "single_label_classification"
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
            'problem_type': "single_label_classification"
        }
    
    
    if training_args.omics1_model_type in ['cnn', 'resnet', 'lstm'] and training_args.omics2_model_type in ['cnn', 'resnet', 'lstm']:
        model = MultiOmicsNaiveClassifier(omics1_config, omics2_config, omics1_model, omics2_model)
    else:
        model = OmicsOmicsInter(omics1_config, omics2_config, omics1_model, omics2_model)
    if training_args.frozen_backbone:
        for name, param in model.named_parameters():
            if ('classifier' not in name) and ('pooler' not in name) and ('position_embeddings' not in name):
                param.requires_grad = False
        if torch.distributed.get_rank() in [0, -1]:
            check_freeze(model)

    # Print the architecture of the combined MultiOmicsNaiveClassifier model
    print("Full MultiOmicsNaiveClassifier Model Architecture:")
    print(model)       
    # # Print model architecture using torchinfo.summary
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    summary(model) #, [(sample_omics1_input.shape), (sample_omics2_input.shape)]

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
