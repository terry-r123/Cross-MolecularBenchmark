from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


class ResNetBlock(nn.Module):
    

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, activation="gelu"):
        super(ResNetBlock, self).__init__()
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

    def forward(self, input, mask):

        identity = input
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
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
        
class ResNet(nn.Module):
    '''
        class: ResNet
        input_dim: 4
        hidden_dims: [512, 512, 512, 512, 512, 512, 512, 512]
        layer_norm: True
        dropout: 0.1
    '''
    def __init__(self, input_dim=4, hidden_dims=[512, 512, 512, 512, 512, 512, 512, 512], kernel_size=3, stride=1, padding=1,
                 activation="gelu", short_cut=False, concat_hidden=False, layer_norm=True,
                 dropout=0.1):
        super(ResNet, self).__init__()
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
            self.layers.append(ResNetBlock(self.dims[i], self.dims[i + 1], kernel_size,
                                                         stride, padding, activation))
        # Attention Pooling
        self.mapping = nn.Linear(hidden_dims[-1], 1)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input, mask):
        input = self.embedding(input)
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.dropout:
            input = self.dropout(input)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
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


class ConvolutionalNetwork(nn.Module):
    '''
        class: ConvolutionalNetwork
        input_dim: 4
        hidden_dims: [1024, 1024]
        kernel_size: 5
        padding: 2
    '''
    def __init__(self, input_dim=4, hidden_dims=[1024, 1024], kernel_size=5, stride=1, padding=2,
                activation='relu'):
        super(ConvolutionalNetwork, self).__init__()
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

    def forward(self, input, mask):
        hiddens = []
        # input = input.long()
        input = self.embedding(input)

        # Expand the mask dimensions to match the input's dimensions for multiplication
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)  # Expanding mask from (batch_size, sequence_length) to (batch_size, sequence_length, 1)
        input = input * mask
        layer_input = input
        for layer in self.layers:
            # print(f"Input shape before conv1d: {layer_input.shape}")
            hidden = layer(layer_input.transpose(1, 2)).transpose(1, 2)
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


class LSTM(nn.Module): 
    '''
        class: LSTM
        input_dim: 4
        hidden_dim: 640
        num_layers: 3
    ''' 
    def __init__(self, input_dim=4, hidden_dim=640, num_layers=3, activation='tanh', layer_norm=False, 
                dropout=0):
        super(LSTM, self).__init__()
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
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)
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

MODEL_CLASS={"cnn":ConvolutionalNetwork,"resnet":ResNet,"lstm":LSTM}

# class Classifier(nn.Module):
#     def __init__(self, model_type, num_classes,vocab_size=9):
#         super(Classifier, self).__init__()
#         self.feature_extractor_1 = MODEL_CLASS[model_type](input_dim=vocab_size)
#         self.feature_extractor_2 = MODEL_CLASS[model_type](input_dim=vocab_size)
        
#         self.hidden_dim = self.feature_extractor_1.output_dim 
#         self.num_classes = num_classes

#         # MLP layers
#         self.mlp = nn.Sequential(
#             nn.Linear(self.hidden_dim * 2, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.num_classes)
#         )

#     def forward(self, input_ids_1,mask_1,input_ids_2,mask_2):
#         mask_1 = mask_1.unsqueeze(-1).float()
#         mask_2 = mask_2.unsqueeze(-1).float()
#         hidden_1 = self.feature_extractor_1(input_ids_1,mask_1)
#         hidden_2 = self.feature_extractor_2(input_ids_2,mask_2)
#         hidden = torch.cat([hidden_1, hidden_2], dim=-1)
#         logits = self.mlp(hidden)
#         return logits


class MultiOmicsNaiveClassifier(nn.Module):
    def __init__(self, omics1_config, omics2_config, omics1_model, omics2_model, weight=None):
        super().__init__()
        self.num_labels = omics1_config['num_labels']
        self.omics1_model = omics1_model
        self.omics2_model = omics2_model

        # Classifier that concatenates omics1 and omics2 hidden sizes and predicts final output
        self.classifier = nn.Linear(omics1_config['hidden_size'] + omics2_config['hidden_size'], self.num_labels)

        self.omics1_config = omics1_config
        self.omics2_config = omics2_config
        self.weight = weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        omics2_input_ids: Optional[torch.Tensor] = None,
        omics2_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.omics1_config.get('use_return_dict', True)

        # Forward pass through omics1_model (ConvolutionalNetwork)
        omics1_out = self.omics1_model(input_ids, attention_mask)  # Pass input_ids and attention_mask as inputs

        # Forward pass through omics2_model (ConvolutionalNetwork)
        omics2_out = self.omics2_model(omics2_input_ids, omics2_attention_mask)  # Pass omics2_input_ids and omics2_attention_mask as inputs

        # Concatenate outputs from omics1 and omics2 models
        final_input = torch.cat([omics1_out, omics2_out], dim=-1)

        # Compute logits
        logits = self.classifier(final_input)

        loss = None
        if labels is not None:
            if self.omics1_config['problem_type'] is None:
                if self.num_labels == 1:
                    self.omics1_config['problem_type'] = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.omics1_config['problem_type'] = "single_label_classification"
                else:
                    self.omics1_config['problem_type'] = "multi_label_classification"

            if self.omics1_config['problem_type'] == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)

            elif self.omics1_config['problem_type'] == "single_label_classification":
                if self.weight is not None:
                    self.weight = torch.tensor(self.weight, device=logits.device)
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif self.omics1_config['problem_type'] == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + (omics1_out[2:] if len(omics1_out) > 2 else ())
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class RNARNANaiveClassifier(nn.Module):
    def __init__(self, omics1_config, omics2_config, omics1_model, omics2_model, use_features, weight=None):
        super().__init__()
        self.num_labels = omics1_config['num_labels']
        self.omics1_model = omics1_model
        self.omics2_model = omics2_model
        self.omics1_config = omics1_config
        self.omics2_config = omics2_config
        
        self.use_features = use_features  # Flag to indicate whether to use additional features

        # If using additional features, create a feature embedding layer
        if use_features:
            self.feature_embedding = nn.Linear(25, 100)
            self.classifier = nn.Linear(omics1_config['hidden_size'] + omics2_config['hidden_size'] + 100, self.num_labels)
        else:
            self.classifier = nn.Linear(omics1_config['hidden_size'] + omics2_config['hidden_size'], self.num_labels)
        
        self.weight = weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        omics2_input_ids: Optional[torch.Tensor] = None,
        omics2_attention_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,  # Include features argument
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.omics1_config.get('use_return_dict', True)

        omics1_out = self.omics1_model(input_ids, attention_mask)

        omics2_out = self.omics2_model(omics2_input_ids, omics2_attention_mask)

        # Concatenate outputs from omics1 and omics2 models
        final_input = torch.cat([omics1_out, omics2_out], dim=-1)

        # If using additional features, embed and concatenate them to the final input
        if self.use_features:
            features = self.feature_embedding(features.float())  # Convert features to float and embed
            final_input = torch.cat([final_input, features], dim=-1)  # Concatenate features

        # Compute logits
        logits = self.classifier(final_input)

        loss = None
        if labels is not None:
            if self.omics1_config['problem_type'] is None:
                if self.num_labels == 1:
                    self.omics1_config['problem_type'] = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.omics1_config['problem_type'] = "single_label_classification"
                else:
                    self.omics1_config['problem_type'] = "multi_label_classification"

            if self.omics1_config['problem_type'] == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)

            elif self.omics1_config['problem_type'] == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif self.omics1_config['problem_type'] == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + (omics1_out[2:] if len(omics1_out) > 2 else ())
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class RNADNANaiveClassifier(nn.Module):
    def __init__(self, omics1_config, omics2_config, omics1_model, omics2_model, use_features, weight=None):
        super().__init__()
        self.num_labels = omics1_config['num_labels']
        self.omics1_model = omics1_model
        self.omics2_model = omics2_model
        self.omics1_config = omics1_config
        self.omics2_config = omics2_config
        
        self.use_feature = use_features
        if use_features:
            self.feature_embedding = nn.Linear(23, 1)   # features.shape[2]=23 num of seq length
            self.classifier = nn.Linear(omics1_config['hidden_size'] + omics2_config['hidden_size'] + 8, self.num_labels)   # features.shape[1]=8 num of stacked vectors
        else:
            self.classifer = nn.Linear(omics1_config['hidden_size']+omics2_config['hidden_size'], self.num_labels)
        
        print(f"Input size of classifier: {self.classifier.in_features}")
        self.weight = weight
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        omics2_input_ids: Optional[torch.Tensor] = None,
        omics2_attention_mask: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,  # Include features argument
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.omics1_config.get('use_return_dict', True)

        omics1_out = self.omics1_model(input_ids, attention_mask)
        omics2_out = self.omics2_model(omics2_input_ids, omics2_attention_mask)
        final_input = torch.cat([omics1_out, omics2_out], dim=-1)
        if self.use_feature:
            features = self.feature_embedding(features).squeeze(dim=-1)
            final_input = torch.cat([final_input, features], dim=-1)

        logits = self.classifier(final_input)

        loss = None
        if labels is not None:
            if self.omics1_config['problem_type'] is None:
                if self.num_labels == 1:
                    self.omics1_config['problem_type'] = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.omics1_config['problem_type'] = "single_label_classification"
                else:
                    self.omics1_config['problem_type'] = "multi_label_classification"

            if self.omics1_config['problem_type'] == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)

            elif self.omics1_config['problem_type'] == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif self.omics1_config['problem_type'] == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + (omics1_out[2:] if len(omics1_out) > 2 else ())
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    

# def tokenize(seq):
#     mapping = {
#         "A": 0,
#         "C": 1,
#         "G": 2,
#         "U": 3,
#         "N": 4
#     }
#     ids = []
#     for s in seq:
#         ids.append(mapping[s])
#     return torch.tensor(ids).unsqueeze(0)

# import torchsummary
# if __name__ == "__main__":
#     model = Classifier("cnn", num_classes=2,vocab_size=9)

#     seq1 = "AGCUGUCUAUGUCUAGGAC"
#     seq2 = "AGCUGUCUAUGUCUAGGAC"
    
#     input_ids1 = tokenize(seq1)
#     input_ids2 = tokenize(seq2)
#     mask_1 = torch.ones_like(input_ids1)
#     mask_2 = torch.ones_like(input_ids2)
    
#     output = model(input_ids1,mask_1,input_ids2,mask_2)
    
#     print("Finish testing")
    
