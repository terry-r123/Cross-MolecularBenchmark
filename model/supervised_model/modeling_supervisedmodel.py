from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .configuration_supervisedmodel import SupervisedModelConfig
from transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput


class RNAResNetBlock(nn.Module):
    

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation="gelu"):
        super(RNAResNetBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding, bias=False)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride, padding, bias=False)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, input,mask):

        identity = input
        input = input * mask
        out = self.conv1(input.transpose(1, 2)).transpose(1, 2).contiguous()
        out = self.layer_norm1(out)
        out = self.activation(out)

        out = out * mask
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2).contiguous()
        out = self.layer_norm2(out)

        out += identity
        out = self.activation(out)

        return out
        
class RNAResNet(nn.Module):
    '''
        class: RNAResNet
        input_dim: 4
        hidden_dims: [512, 512, 512, 512, 512, 512, 512, 512]
        layer_norm: True
        dropout: 0.1
    '''
    def __init__(self, input_dim=4, hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512], kernel_size=3, stride=1, padding=1,
                 activation="gelu", short_cut=False, concat_hidden=False, layer_norm=True,
                 dropout=0.1):
        super(RNAResNet, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.dims = list(hidden_dims)

        self.embedding = nn.Embedding(input_dim,hidden_dims[0])
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dims[0])
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(RNAResNetBlock(self.dims[i], self.dims[i + 1], kernel_size,
                                                         stride, padding, activation))
        # Attention Pooling
        self.mapping = nn.Linear(hidden_dims[-1], 1)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input,mask):
        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        input = input * mask
        
        hiddens = []
        layer_input = input
        
        for layer in self.layers:
            hidden = layer(layer_input,mask)
            hiddens.append(hidden)
            layer_input = hidden
        hidden = hiddens[-1]
        # Do Attention pooling
        weights = self.mapping(hidden)
        weights = weights + (1 - mask) * -1e9
        weights = self.softmax(weights.squeeze(-1))
        hidden = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return hidden


class RNAConvolutionalNetwork(nn.Module):
    '''
        class: RNAConvolutionalNetwork
        input_dim: 4
        hidden_dims: [1024, 1024]
        kernel_size: 5
        padding: 2
    '''
    def __init__(self, input_dim=4, hidden_dims=[1024, 1024], kernel_size=5, stride=1, padding=2,
                activation='relu'):
        super(RNAConvolutionalNetwork, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim =  hidden_dims[-1]
        self.embedding = nn.Embedding(input_dim, input_dim)
        self.dims = [input_dim] + list(hidden_dims)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                nn.Conv1d(self.dims[i], self.dims[i+1], kernel_size, stride, padding)
            )

        # Attention Pooling
        self.mapping = nn.Linear(hidden_dims[-1], 1)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, input,mask):
        hiddens = []
        input = self.embedding(input)
        input = input * mask
        layer_input = input
        for layer in self.layers:
            
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2).contiguous()
            hidden = self.activation(hidden)
            hidden = hidden * mask
            hiddens.append(hidden)
            layer_input = hidden

        hidden = hiddens[-1]
        # Do Attention pooling

        weights = self.mapping(hidden)
        weights = weights + (1 - mask) * -1e9
        weights = self.softmax(weights.squeeze(-1))
        hidden = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        
        return hidden


class RNALSTM(nn.Module): 
    '''
        class: RNALSTM
        input_dim: 4
        hidden_dim: 640
        num_layers: 3
    ''' 
    def __init__(self, input_dim=4, hidden_dim=640, num_layers=3, activation='tanh', layer_norm=False, 
                dropout=0):
        super(RNALSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim*2
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=True)

        # attention pooling
        self.mapping = nn.Linear(hidden_dim * 2, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,  input,mask):
        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        
        lengths = (mask.squeeze(-1)).sum(dim=1)
        input = input * mask
        
        lengths, sorted_idx = lengths.sort(descending=True)
        _, original_idx = sorted_idx.sort()

        embedded = input[sorted_idx]
        
        # Pack the sequence
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        


        packed_output, (hidden, cell) = self.lstm(packed_embedded)


        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        

        output = output[original_idx]
        weights = self.mapping(output)
        weights = weights + (1 - mask) * -1e9
        weights = self.softmax(weights.squeeze(-1))
        hidden = torch.sum(output * weights.unsqueeze(-1), dim=1)
        return hidden

MODEL_CLASS={"cnn":RNAConvolutionalNetwork,"resnet":RNAResNet,"lstm":RNALSTM}

class SupervisedModelForSequenceClassification(PreTrainedModel):

    config_class = SupervisedModelConfig
    base_model_prefix = "SupervisedModel"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        supervised_model_type = config.supervised_model_type
        vocab_size = config.vocab_size

        self.feature_extractor = MODEL_CLASS[supervised_model_type](input_dim=vocab_size)

        self.hidden_dim = self.feature_extractor.output_dim 

        # MLP layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_labels)
        )

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                return_dict=None,
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        attention_mask = attention_mask.unsqueeze(-1).float()
        hidden = self.feature_extractor(input_ids,attention_mask)

        logits = self.classifier(hidden)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            #print(self.config.problem_type)
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        #print('return_dict',return_dict)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class Classifier(nn.Module):
    def __init__(self, supervised_model_type, num_classes,vocab_size=9):
        super(Classifier, self).__init__()
        self.feature_extractor_1 = MODEL_CLASS[supervised_model_type](input_dim=vocab_size)
        self.feature_extractor_2 = MODEL_CLASS[supervised_model_type](input_dim=vocab_size)
        
        self.hidden_dim = self.feature_extractor_1.output_dim 
        self.num_classes = num_classes

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, input_ids_1,mask_1,input_ids_2,mask_2):
        mask_1 = mask_1.unsqueeze(-1).float()
        mask_2 = mask_2.unsqueeze(-1).float()
        hidden_1 = self.feature_extractor_1(input_ids_1,mask_1)
        hidden_2 = self.feature_extractor_2(input_ids_2,mask_2)
        hidden = torch.cat([hidden_1, hidden_2], dim=-1)
        logits = self.mlp(hidden)
        return logits

def tokenize(seq):
    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "U": 3,
        "N": 4
    }
    ids = []
    for s in seq:
        ids.append(mapping[s])
    return torch.tensor(ids).unsqueeze(0)

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

def create_rna_tokenizer():
    """
    Creates a PreTrainedTokenizerFast for RNA sequences, tokenizing at the nucleotide level.
    """
    # Define the RNA vocabulary
    vocab = {
        "A": 0,   # Adenine
        "C": 1,   # Cytosine
        "G": 2,   # Guanine
        "T": 3,   # T
        "N": 4,   # Any nucleotide (unknown)
        "[UNK]": 5,
        "[PAD]": 6,
    }

    # Create a tokenizer using the Tokenizers library
    base_tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    base_tokenizer.pre_tokenizer = Split('', 'isolated')  # Split every character (nucleotide)

    # Create a PreTrainedTokenizerFast tokenizer with the RNA vocab
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token=None,
        sep_token=None,
        bos_token=None,
        eos_token=None,
        cache_dir=None,
        model_max_length=4,
        padding_side="right",
        use_fast=True,
    )

    return tokenizer


# import torchsummary
if __name__ == "__main__":
    model = Classifier("cnn", num_classes=2,vocab_size=9)

    seq1 = "AGCUGUCUAUGUCUAGGAC"
    seq2 = "AGCUGUCUAUGUCUAGGAC"

    tokenizer = create_rna_tokenizer()
    input_ids1 = tokenizer(seq1)
    
    input_ids1 = tokenize(seq1)
    input_ids2 = tokenize(seq2)
    mask_1 = torch.ones_like(input_ids1)
    mask_2 = torch.ones_like(input_ids2)
    
    output = model(input_ids1,mask_1,input_ids2,mask_2)
    
    print("Finish testing")
    
