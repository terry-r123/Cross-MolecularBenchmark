import os
import csv
import json
import copy
import logging
import pdb


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



"""
TODO

"""

def train():
    pass


if __name__ == "__main__":
    train()