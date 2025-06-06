import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
from scipy.stats import spearmanr, pearsonr
import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import random
#from dnabert2_source.bert_layers_enhancer_design import BertForSequenceClassificationAllPromptTokenAvg
import sys
import pdb

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)

from torch.utils.data import DataLoader
from tqdm import tqdm
# from model.esm.flashesm_layers import EsmForSequenceClassification
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModelForSequenceClassification, EarlyStoppingCallback

from model.dnabert2.bert_layers import BertForSequenceClassification as DNABERT2ForClassification
# from model.dnabert1.dnabert_layer import DNABertForSequenceClassification as DNABERT1ForClassification
from model.ntv2.modeling_esm import EsmForSequenceClassification as NTv2ForSequenceClassification
from model.hyenadna.tokenization_hyena import HyenaDNATokenizer
from model.hyenadna.modeling_hyena import HyenaDNAForSequenceClassification
from model.rnafm.modeling_rnafm import RnaFmForSequenceClassification
from model.esm.modeling_esm import EsmForSequenceClassification
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnalm.modeling_rnalm import RnaLmForSequenceClassification 
from model.rnalm.rnalm_tokenizer import RnaLmTokenizer
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer

# change for LucaOne
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

os.environ["WANDB_PROJECT"] = "enhancer_design"

early_stopping = EarlyStoppingCallback(early_stopping_patience=100)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def delete_finetune(root_dir):
    file_list = ["pytorch_model.bin", "training_args.bin" ]
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            if "checkpoint" in dir_name:
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"目录 {dir_path} 已成功删除。")
                except OSError as e:
                    print(f"无法删除目录 {dir_path}: {e}")
        for file in files:
            if file in file_list:
                filename = os.path.join(root, file)
                try:
                    os.remove(filename)
                    print(f"文件 {filename} 已成功删除。")
                except OSError as e:
                    print(f"无法删除目录 {filename}: {e}")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    token_ratio: float = field(default=1.0, metadata={"help": "token ratio"})
    # use_alibi: bool = field(default=True)
    use_bpe: bool = field(default=True)
    with_prompt_num: int = field(default=0, metadata={"help": "with prompt num"})
    #model_type: Optional[str] = field(default="baseline")
    token_position: Optional[str] = field(default="final")
    model_max_length_2: int = field(default=512, metadata={"help": "Maximum sequence length."})
    delete_n: bool = field(default=False, metadata={"help": "data delete N"})



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
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    report_to: str = field(default="wandb")
    metric_for_best_model: str = field(default="PCC_test_mean")
    model_type: str = field(default='esm')
    token_type: str = field(default='single')
    train_from_scratch: bool = field(default=False)
    attn_implementation : str = field(default="eager")
# change for LucaOne
    freeze_backbone: bool = field(default=False)
    seq_type: str = field(default=None)
    trunc_type: str = field(default=None)
    lucaone_args: object = field(default=None) # ?
    lucaone_model_args: object = field(default=None) # ?
    dataloader_num_workers: int = field(default=8) # debug using 0
# change for LucaOne


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    # kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    # if os.path.exists(kmer_path):
    #     logging.warning(f"Loading k-mer from {kmer_path}...")
    #     with open(kmer_path, "r") as f:
    #         kmer = json.load(f)
    # else:        
    logging.warning(f"Generating k-mer...")
    kmer = [generate_kmer_str(text, k) for text in texts]
        # with open(kmer_path, "w") as f:
        #     logging.warning(f"Saving k-mer to {kmer_path}...")
        #     json.dump(kmer, f)
        
    return kmer


def truncate_middle(seq, max_length):
    if len(seq) <= max_length-2:
        return seq
    start = (len(seq) - max_length) // 2
    end = start + max_length
    return seq[start+1:end-1]

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
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 data_args,
                 kmer: int = -1,
                 ):

        super(SupervisedDataset, self).__init__()
        # if data_args.use_bpe:
        #     prompt_tokenizer = "[BPE]"
        # else:
        #     prompt_tokenizer = "[KMER]"
        # if data_args.use_alibi:
        #     prompt_positon = "[ALIBI]"
        # else:
        #     prompt_positon = "[APE]"
        # load data from the disk
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [truncate_middle(d[0].upper().replace("U", "T"), tokenizer.model_max_length) for d in data]
            labels = [int(d[1]) for d in data]
            self.num_labels = 1
        # elif len(data[0]) >= 3:
        #     # data is in the format of [text1, text2, label]
        #     logging.warning("Perform sequence-pair classification...")
        #     texts = [truncate_middle(d[0], tokenizer.model_max_length) for d in data]
        #     labels = [[int(d[1]), int(d[2]), int(d[3])] for d in data]
        else:
            texts = [d[1].upper() for d in data]
            labels = [[float(d[4]), float(d[5])] for d in data]
            self.num_labels = 2
        

        if kmer != -1:
            # only write file on the first process
            # if torch.distributed.get_rank() not in [0, -1]:
            #     torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            # if torch.distributed.get_rank() == 0:
            #     torch.distributed.barrier()
        #############################################################

# change for LucaOne
        self.model_type = data_args.model_type
        if data_args.model_type == 'LucaOne':
            print(type(texts[1]))
            print(texts[1])
            seq_encoded_list = []
            attention_mask_list = []
            for seq in texts:
                # seq process
                if data_args.seq_type == "prot": # t
                    processed_seq = clean_seq(seq_id, seq)
                    # print(f'processed_seq: {processed_seq}')
                else:
                    processed_seq = seq.strip()
                
                # print(f'data_args.lucaone_args["max_length"]: {data_args.lucaone_args["max_length"]}')
                truncation_seq_length = data_args.lucaone_args['max_length'] - 2
                if len(processed_seq) > truncation_seq_length: # exclude [CLS] [SEP]
                    if data_args.trunc_type == "left":
                        print("Truncate a sequence to retain its right side.")
                        processed_seq = processed_seq[-truncation_seq_length:]
                    else:
                        print("Truncate a sequence to retain its left side.")
                        processed_seq = processed_seq[:truncation_seq_length]
                # print(f'trunc_processed_seq: {processed_seq}')

                if data_args.seq_type == "gene":
                    processed_seq = gene_seq_replace(processed_seq)
                    # seqs = [processed_seq] # for
                    self.seq_types = data_args.seq_type
                    # print(processed_seq)
                    # print(len(processed_seq))
                    # print(processed_seq[0])
                    token_seq = tokenizer.encode(processed_seq)
                    seq_encoded_list.append(token_seq)
                    attention_mask_list.append(torch.ones(len(token_seq) + 2))
                    # print(f"seq_encoded_list: {seq_encoded_list}")
                else:
                    # seqs = [processed_seq]
                    self.seq_types = data_args.seq_type
                    token_seq = tokenizer.encode(processed_seq)
                    seq_encoded_list.append(token_seq)
                    attention_mask_list.append(torch.ones(len(token_seq) + 2))
            self.input_ids = seq_encoded_list
            print(f'self.input_ids: \n{self.input_ids[1]}')
            print(f'labels: \n{labels[1]}')
        else:   
# change for LucaOne
            temp_count_max = 0
            temp_count_all = 0
            texts_delete = []
            token_test = texts[0]
            print(token_test)
            print(tokenizer.tokenize(token_test))
            print(tokenizer(token_test))
            output = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="longest",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                )

            self.input_ids = output["input_ids"]
        
# change for LucaOne
        if data_args.model_type == 'LucaOne':
            self.attention_mask = attention_mask_list
            print(f'self.attention_mask[1]: \n{self.attention_mask[1]}')
# change for LucaOne
        else:
            self.attention_mask = output["attention_mask"]
    
        # self.input_ids = input_ids
        # self.attention_mask = attention_mask
        # pdb.set_trace()

        # print(self.input_ids[0], len(self.input_ids[0]))
        # print(self.attention_mask[0], len(self.attention_mask[0]))
        self.labels = labels
        #print(self.labels)
        print(f"number of regression labels: {self.num_labels}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
# change for LucaOne
        if self.args.model_type == 'LucaOne':
            batch_size = len(instances)
            input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask"))
            # attention_mask = torch.stack(attention_mask)
            if self.args.lucaone_args['max_length'] and self.args.lucaone_args['max_length'] > 0:
                seq_encoded_list = [encoded[:self.args.lucaone_args['max_length']] for encoded in input_ids]
            max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
            processed_seq_len = max_len + int(self.tokenizer.prepend_bos) + int(self.tokenizer.append_eos)
            # print(f'processed_seq_len: {processed_seq_len}')
            # for input
            input_ids = torch.empty(
                (
                    batch_size,
                    processed_seq_len,
                ),
                dtype=torch.int64,
            )
            input_ids.fill_(self.tokenizer.padding_idx)

            position_ids = None
            if not self.args.lucaone_model_args['no_position_embeddings']:
                position_ids = torch.empty(
                    (
                        batch_size,
                        processed_seq_len,
                    ),
                    dtype=torch.int64,
                )
                position_ids.fill_(self.tokenizer.padding_idx)

            token_type_ids = None
            if not self.args.lucaone_model_args['no_token_type_embeddings']:
                token_type_ids = torch.empty(
                    (
                        batch_size,
                        processed_seq_len,
                    ),
                    dtype=torch.int64,
                )
                token_type_ids.fill_(self.tokenizer.padding_idx)

            seq_types = [self.args.seq_type for _ in range(batch_size)]
            for i, (seq_type, seq_encoded) in enumerate(
                    zip(seq_types, seq_encoded_list)
            ):
                if self.tokenizer.prepend_bos:
                    input_ids[i, 0] = self.tokenizer.cls_idx
                seq = torch.tensor(seq_encoded, dtype=torch.int64)
                input_ids[i, int(self.tokenizer.prepend_bos): len(seq_encoded) + int(self.tokenizer.prepend_bos)] = seq
                if self.tokenizer.append_eos:
                    input_ids[i, len(seq_encoded) + int(self.tokenizer.prepend_bos)] = self.tokenizer.eos_idx

                if not self.args.lucaone_model_args['no_position_embeddings']:
                    cur_len = int(self.tokenizer.prepend_bos) + len(seq_encoded) + int(self.tokenizer.append_eos)
                    for idx in range(0, cur_len):
                        position_ids[i, idx] = idx
                if not self.args.lucaone_model_args['no_token_type_embeddings']:
                    if seq_type == "gene":
                        type_value = 0
                    else:
                        type_value = 1
                    cur_len = int(self.tokenizer.prepend_bos) + len(seq_encoded) + int(self.tokenizer.append_eos)
                    for idx in range(0, cur_len):
                        token_type_ids[i, idx] = type_value

            encoding = {"input_ids": input_ids, "token_type_ids": token_type_ids, "position_ids": position_ids}
            
            # # for chain B in PPI etc.
            # if seq_type == "prot":
            #     new_encoding = {}
            #     for item in encoding.items():
            #         new_encoding[item[0] + "_b"] = item[1]
            #     encoding = new_encoding
            
            repr_layers = list(range(self.args.lucaone_model_args["num_hidden_layers"] + 1))

            attention_mask = torch.stack(attention_mask)
            labels = torch.Tensor(labels).float()  
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                labels=labels,
                # return_contacts=True,
                # return_dict=True,
                # repr_layers=repr_layers,
            )
# change for LucaOne
        else:
            input_ids,  labels, attention_mask = tuple(
                [instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
            # input_ids = [[i[0] for i in inputs], [i[1] for i in inputs]]
            # attention_mask = [[i[0] for i in attentions], [i[1] for i in attentions]]
            labels = torch.Tensor(labels).float()
            input_ids = torch.stack(input_ids)

            attention_mask = torch.stack(attention_mask)

            
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    #predictions = np.argmax(logits, axis=-1)
    # print(f"cm_logits: \n{logits}")
    # print(f"cm_labels: \n{labels}")
    # print(f"cm_logits: {logits.shape}")
    # print(f"cm_labels: {labels.shape}")
    # print(f"cm_logits[:,0]: \n{logits[:,0]}")
    # print(f"cm_labels[:,0]: \n{labels[:,0]}")
    # print(f"cm_logits[:,1]: \n{logits[:,1]}")
    # print(f"cm_labels[:,1]: \n{labels[:,1]}")
    return {
        #"accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        #"f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        #"matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        #"precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        #"recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
        "mse": sklearn.metrics.mean_squared_error(labels, logits),
        "PCC_test_development": pearsonr(logits[:,0], labels[:,0])[0],
        "PCC_test_housekeeping": pearsonr(logits[:,1], labels[:,1])[0],
        "PCC_test_mean": np.mean([pearsonr(logits[:,0], labels[:,0])[0], pearsonr(logits[:,1], labels[:,1])[0]]),
        "SCC_test_development": spearmanr(logits[:,0], labels[:,0])[0],
        "SCC_test_housekeeping": spearmanr(logits[:,1], labels[:,1])[0],
    }


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


# change for LucaOne
"""
Load LucaOne model
"""
def load_model(log_filepath, model_dirpath):
    '''
    create tokenizer, model config, model
    :param log_filepath:
    :param model_dirpath:
    :return:
    '''
    strs = model_dirpath.split("llm/models/")
    # if len(strs) > 1:
    #     download_trained_checkpoint_lucaone(os.path.join(strs[0], "llm/"))
    with open(log_filepath, "r") as rfp:
        for line_idx, line in enumerate(rfp):
            if line_idx == 0:
                try:
                    args_info = json.loads(line.strip(), encoding="UTF-8")
                except Exception as e:
                    args_info = json.loads(line.strip())
                break
    print("Model dirpath: %s" % model_dirpath)
    assert model_dirpath is not None and os.path.exists(model_dirpath)
    # create tokenizer
    tokenizer_dir = os.path.join(model_dirpath, "tokenizer")
    assert os.path.exists(tokenizer_dir)
    if args_info["tokenization"]:
        print("AutoTokenizer, tokenizer dir: %s" % tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            do_lower_case=args_info["do_lower_case"],
            truncation_side=args_info["truncation"]
        )
    elif args_info["model_type"] in ["lucaone_gplm"]:
        print("Alphabet, vocab path: %s" % tokenizer_dir)
        if "/v0.2/" in model_dirpath:
            tokenizer = AlphabetV0_2.from_predefined("gene_prot")
        elif "/v2.0/" in model_dirpath:
            tokenizer = AlphabetV2_0.from_predefined("gene_prot")
        else:
            raise Exception("Not support version=%s" % model_dirpath)
    else:
        print("BertTokenizer, vocab path: %s" % tokenizer_dir)
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_dir,
            do_lower_case=args_info["do_lower_case"],
            truncation_side=args_info["truncation"])
    # model config
    model_config: PretrainedConfig = LucaOneConfig.from_json_file(os.path.join(model_dirpath, "config.json"))
    return args_info, model_config, tokenizer
# change for LucaOne

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type == 'hyenadna':
        tokenizer = HyenaDNATokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type in ['esm2', 'esm-1b', 'rna-fm_frozen', 'rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
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
# change for LucaOne
    elif training_args.model_type =='LucaOne':
        if training_args.seq_type not in ['gene', 'prot'] and training_args.trunc_type not in ['left', 'right']:
            raise RuntimeError('Please set seq_type and trunc_type when using LucaOne model')
        lucaone_global_log_filepath = "/home/bingxing2/ailab/group/ai4bio/public/multi-omics/lucaone/llm/logs/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/logs.txt"
        lucaone_global_model_dirpath = "/home/bingxing2/ailab/group/ai4bio/public/multi-omics/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step5600000/"

        lucaone_global_args_info, lucaone_global_model_config, lucaone_global_tokenizer = load_model(lucaone_global_log_filepath, lucaone_global_model_dirpath)
        lucaone_global_args_info["max_length"] = training_args.model_max_length

        training_args.lucaone_args = lucaone_global_args_info
        training_args.lucaone_model_args = lucaone_global_model_config.to_dict()

        tokenizer = lucaone_global_tokenizer
# change for LucaOne
    elif training_args.model_type in ['cnn', 'resnet', 'lstm']:
        from model.supervised_model.tokenizer import create_rna_tokenizer
        tokenizer = create_rna_tokenizer(
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=True,
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
    train_dataset = SupervisedDataset(tokenizer=tokenizer,                                    
                                      data_path=os.path.join(data_args.data_path, data_args.data_train_path), 
                                      kmer=data_args.kmer,
                                      data_args=training_args)
    val_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, data_args.data_val_path), 
                                     kmer=data_args.kmer,
                                     data_args=training_args)
    test_dataset = SupervisedDataset(tokenizer=tokenizer,
                                     data_path=os.path.join(data_args.data_path, data_args.data_test_path),
                                     kmer=data_args.kmer,
                                     data_args=training_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,args=training_args)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # 确定eval_steps
    total_training_steps = len(train_dataset)  * training_args.num_train_epochs // (training_args.per_device_train_batch_size * torch.cuda.device_count() * training_args.gradient_accumulation_steps)
    eval_steps = max(1, total_training_steps // 40)
    
    # 更新TrainingArguments来包含这些参数
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = eval_steps
    training_args.save_steps = eval_steps  # 如果你也希望每5%保存一次模型
    training_args.logging_steps = eval_steps

    if training_args.model_type == "esm":
        print(f'Loading {training_args.model_type} model')
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
    if training_args.model_type == "dnabert2":
        model = DNABERT2ForClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            use_alibi=model_args.use_alibi,
            problem_type="regression"
        )
    elif training_args.model_type == "esm":
        print(f'Loading {training_args.model_type} model')
        model = EsmForRegression512concat.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
    elif training_args.model_type == "ntv2":
        print(f'Loading {training_args.model_type} model')
        model = NTv2ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
    elif training_args.model_type == "hyenadna":
        print(f'Loading {training_args.model_type} model')
        model = HyenaDNAForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
        )
    elif 'rna-fm' in training_args.model_type:      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        if "frozen" in training_args.model_type:
            model = RnaFmEncoderPoolingForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                trust_remote_code=True,
                wEncoder=model_args.wEncoder,
                encoder_hidden_size=model_args.encoder_hidden_size,
                encoder_num_hidden_layers=model_args.encoder_num_hidden_layers,
                encoder_num_attention_heads=model_args.encoder_num_attention_heads,
                encoder_intermediate_size=model_args.encoder_intermediate_size,
                pooling_type=model_args.pooling_type,
                encoder_classifer_size=model_args.encoder_classifer_size,
            )
        else:
            print(f"model_args.model_name_or_path: {model_args.model_name_or_path}")
            model = RnaFmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                trust_remote_code=True,
            )
    elif 'esm2' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading protein model {training_args.model_type} model')
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
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
                problem_type="regression",
                trust_remote_code=True,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
            ) 
    elif training_args.model_type == 'BEACON-B':
        if training_args.train_from_scratch:
            print('Loading BEACON-B model')
            print(f'Train from scratch {training_args.model_type} model')
            config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
                attn_implementation=training_args.attn_implementation,
                num_labels=train_dataset.num_labels,
                )
            model = RnaLmForSequenceClassification(config)
        else:
            print(f'args.model_name_or_path: {model_args.model_name_or_path}')    
            model = RnaLmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                trust_remote_code=True,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
                attn_implementation=training_args.attn_implementation,
            )
    elif training_args.model_type =='esm-1b':
        print(f"train_dataset.num_labels: {train_dataset.num_labels}")
        print(f'Loading protein model {training_args.model_type} model')
        model = EsmForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
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
# change for LucaOne
    elif training_args.model_type =='LucaOne':
        print(f'training_args.lucaone_args: \n{training_args.lucaone_args}')
        print(f'Loading {training_args.model_type} model')

        model = LucaOneForSequenceClassification.from_pretrained(
            lucaone_global_model_dirpath,
            num_labels=train_dataset.num_labels,
            problem_type="regression",
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
    elif training_args.model_type in ['cnn', 'resnet', 'lstm']:
        from model.supervised_model.configuration_supervisedmodel import SupervisedModelConfig
        from model.supervised_model.modeling_supervisedmodel import SupervisedModelForSequenceClassification

        config = SupervisedModelConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                )
        model = SupervisedModelForSequenceClassification(config)
    print(model.config)
    print(model)

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
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

    delete_finetune(training_args.output_dir)


if __name__ == "__main__":
    train()