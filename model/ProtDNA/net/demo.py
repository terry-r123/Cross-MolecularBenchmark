from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        out = self.conv1(input.transpose(1, 2)).transpose(1, 2)
        out = self.layer_norm1(out)
        out = self.activation(out)

        out = out * mask
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
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
        self.mapping = nn.Linear(hidden_dims[-1],hidden_dims[-1])


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
        output = self.mapping(hidden)
        return output
        


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
        self.mapping = nn.Linear(hidden_dims[-1], hidden_dims[-1])


    def forward(self, input,mask):
        hiddens = []
        input = self.embedding(input)
        input = input * mask
        layer_input = input
        for layer in self.layers:
            
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
            hidden = self.activation(hidden)
            hidden = hidden * mask
            hiddens.append(hidden)
            layer_input = hidden

        hidden = hiddens[-1]
        # Do Attention pooling

        output = self.mapping(hidden)
        
        return output


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
        self.output_dim = hidden_dim
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
        self.mapping = nn.Linear(hidden_dim * 2, hidden_dim)

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
        output = self.mapping(output)
        
        return output

MODEL_CLASS={"cnn":RNAConvolutionalNetwork,"resnet":RNAResNet,"lstm":RNALSTM}

class Classifier(nn.Module):
    def __init__(self, model_type,output_dim,vocab_size=9):
        super(Classifier, self).__init__()
        self.feature_extractor_1 = MODEL_CLASS[model_type](input_dim=vocab_size)
        self.feature_extractor_2 = MODEL_CLASS[model_type](input_dim=vocab_size)
        
        self.hidden_dim = self.feature_extractor_1.output_dim 
        self.output_dim = output_dim

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim , self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, input_ids_1,mask_1,input_ids_2,mask_2,input_ids_3,mask_3):
        mask_1 = mask_1.unsqueeze(-1).float()
        mask_2 = mask_2.unsqueeze(-1).float()
        mask_3 = mask_3.unsqueeze(-1).float()
        hidden_1 = self.feature_extractor_1(input_ids_1,mask_1)
        hidden_2 = self.feature_extractor_1(input_ids_2,mask_2)
        hidden_3 = self.feature_extractor_2(input_ids_3,mask_3)
        
        return logits


class DNAClassifier(nn.Module):
    def __init__(self, model_type,output_dim,vocab_size=9):
        super(DNAClassifier, self).__init__()
        self.feature_extractor_1 = MODEL_CLASS[model_type](input_dim=vocab_size)      
        self.hidden_dim = self.feature_extractor_1.output_dim 
        self.output_dim = output_dim

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim , self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, input_ids_1,mask_1,input_ids_2,mask_2):

        mask_1 = mask_1.unsqueeze(-1).float()
        mask_2 = mask_2.unsqueeze(-1).float()
        hidden_1 = self.feature_extractor_1(input_ids_1,mask_1)
        hidden_2 = self.feature_extractor_1(input_ids_2,mask_2)

        logits_1 = self.mlp(hidden_1)
        logits_2 = self.mlp(hidden_2)

        logits = torch.cat([logits_1, logits_2], dim=1)
        return logits


class ProteinClassifier(nn.Module):
    def __init__(self, model_type,output_dim,vocab_size=9):
        super(ProteinClassifier, self).__init__()
        self.feature_extractor_1 = MODEL_CLASS[model_type](input_dim=vocab_size)
       
        self.hidden_dim = self.feature_extractor_1.output_dim 
        self.output_dim = output_dim

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim , self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, input_ids_1,mask_1):
        mask_1 = mask_1.unsqueeze(-1).float()
        hidden_1 = self.feature_extractor_1(input_ids_1,mask_1)
        logits_1 = self.mlp(hidden_1)

        return logits_1
