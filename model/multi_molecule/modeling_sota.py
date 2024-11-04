from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


class AttLayer(nn.Module):
    def __init__(self, attention_dim):
        super(AttLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(attention_dim, attention_dim))
        self.b = nn.Parameter(torch.Tensor(attention_dim))
        self.u = nn.Parameter(torch.Tensor(attention_dim, 1))
        nn.init.normal_(self.W)
        nn.init.normal_(self.b)
        nn.init.normal_(self.u)

    def forward(self, x):
        uit = torch.tanh(torch.matmul(x, self.W) + self.b)
        ait = torch.matmul(uit, self.u).squeeze(-1)
        ait = torch.softmax(ait, dim=1)
        ait = ait.unsqueeze(-1)
        weighted_input = x * ait
        return weighted_input.sum(1)

class MultiOmicsSOTAModel(nn.Module):
    def __init__(self, num_labels, vocab_size=4, embedding_dim=100, seq_len_en=3000, seq_len_pr=2000, num_filters=64):
        super(MultiOmicsSOTAModel, self).__init__()

        # Embeddings for enhancers and promoters
        self.enhancer_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.promoter_embedding = nn.Embedding(vocab_size, embedding_dim)

        # CNN Layers for Enhancer and Promoter sequences
        self.enhancer_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=60, padding='same')
        self.enhancer_pool = nn.MaxPool1d(kernel_size=30, stride=30)

        self.promoter_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=40, padding='same')
        self.promoter_pool = nn.MaxPool1d(kernel_size=20, stride=20)

        # GRU and Attention Layers
        self.enhancer_gru = nn.GRU(input_size=num_filters, hidden_size=50, batch_first=True, bidirectional=True)
        self.promoter_gru = nn.GRU(input_size=num_filters, hidden_size=50, batch_first=True, bidirectional=True)

        self.enhancer_attention = AttLayer(100)
        self.promoter_attention = AttLayer(100)

        # Dense and output layers
        self.classifier = nn.Linear(100 * 4, num_labels)  # 50 * 4 due to concatenation of attention outputs

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
        return_dict = return_dict if return_dict is not None else None
        # Embedding lookup
        enhancer_embeds = self.enhancer_embedding(input_ids[:,1:-1]).permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        promoter_embeds = self.promoter_embedding(omics2_input_ids[:,1:-1]).permute(0, 2, 1)

        # Conv1D and Pooling
        enhancer_features = self.enhancer_pool(F.relu(self.enhancer_conv(enhancer_embeds)))
        promoter_features = self.promoter_pool(F.relu(self.promoter_conv(promoter_embeds)))

        # GRU + Attention
        enhancer_features = enhancer_features.permute(0, 2, 1)  # (batch_size, seq_len, num_filters)
        promoter_features = promoter_features.permute(0, 2, 1)

        enhancer_gru_out, _ = self.enhancer_gru(enhancer_features)
        promoter_gru_out, _ = self.promoter_gru(promoter_features)

        enhancer_att = self.enhancer_attention(enhancer_gru_out)
        promoter_att = self.promoter_attention(promoter_gru_out)

        # Matching Heuristics: Concatenation, Subtraction, Multiplication
        diff = torch.abs(enhancer_att - promoter_att)
        mult = enhancer_att * promoter_att
        concat_output = torch.cat([enhancer_att, promoter_att, diff, mult], dim=-1)

        # Classification layer
        logits = self.classifier(concat_output)

        # Optionally compute loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )