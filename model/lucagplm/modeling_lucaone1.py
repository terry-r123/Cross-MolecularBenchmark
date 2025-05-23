# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from warnings import warn

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
import torch.autograd as autograd
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging

from .configuration_lucaone import LucaOneConfig

from ..common.pooling import create_pooler

logger = logging.get_logger(__name__)

# class LucaOneForSequenceClassification(PreTrainedModel):
    
#     config_class = LucaOneConfig
#     base_model_prefix = "lucaone"
#     supports_gradient_checkpointing = True
#     # _no_split_modules = ["RnaFmLayer", "RnaFmEmbeddings"]
#     def __init__(self, config, lucaone_model):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config

#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#         self.lucaone = lucaone_model
    
#     def forword(
#             self,
#             input_ids: Optional[torch.Tensor] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             global_attention_mask: Optional[torch.Tensor] = None,
#             token_type_ids: Optional[torch.Tensor] = None,
#             position_ids: Optional[torch.Tensor] = None,
#             head_mask: Optional[torch.Tensor] = None,
#             inputs_embeds: Optional[torch.Tensor] = None,
#             output_keys: Optional[dict[str, set[str]]] = None,
#             labels: Optional[dict[str, dict[str, torch.Tensor]]] = None,
#             input_ids_b: Optional[torch.Tensor] = None,
#             attention_mask_b: Optional[torch.Tensor] = None,
#             global_attention_mask_b: Optional[torch.Tensor] = None,
#             token_type_ids_b: Optional[torch.Tensor] = None,
#             position_ids_b: Optional[torch.Tensor] = None,
#             head_mask_b: Optional[torch.Tensor] = None,
#             inputs_embeds_b: Optional[torch.Tensor] = None,
#             output_keys_b: Optional[dict[str, set[str]]] = None,
#             labels_b: Optional[dict[str, dict[str, torch.Tensor]]] = None,
#             pair_label: Optional[dict[str, dict[str, torch.Tensor]]] = None,
#             pair_output_keys: Optional[dict[str, set[str]]] = None,
#             output_hidden_states: Optional[dict[str, set[str]]] = None,
#             output_attentions: Optional[dict[str, set[str]]] = None,
#             need_head_weights: Optional[bool] = None,
#             return_contacts: Optional[bool] = None,
#             repr_layers: Optional[list[int]] = None,
#             return_dict: Optional[bool] = None,
#             use_last_layer_norm: Optional[bool] = True
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
#         outputs = self.lucaone(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             global_attention_mask=global_attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_keys=output_keys,
#             labels=labels,
#             input_ids_b=input_ids_b,
#             attention_mask_b=attention_mask_b,
#             global_attention_mask_b=global_attention_mask_b,
#             token_type_ids_b=token_type_ids_b,
#             position_ids_b=position_ids_b,
#             head_mask_b=head_mask_b,
#             inputs_embeds_b=inputs_embeds_b,
#             output_keys_b=output_keys_b,
#             labels_b=labels_b,
#             pair_label=pair_label,
#             pair_output_keys=pair_output_keys,
#             output_hidden_states=output_hidden_states,
#             output_attentions=output_attentions,
#             need_head_weights=need_head_weights,
#             return_contacts=return_contacts,
#             repr_layers=repr_layers,
#             return_dict=return_dict,
#             use_last_layer_norm=use_last_layer_norm
#         )

#         pooled_output = outputs.hidden_states
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#             #print(self.config.problem_type)
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         #print('return_dict',return_dict)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class LucaOneForSequenceClassification(PreTrainedModel):
    
    # config_class = LucaOneConfig
    # base_model_prefix = "lucaone"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["RnaFmLayer", "RnaFmEmbeddings"]
    def __init__(self, lucaone_config: LucaOneConfig, lucaone_model, num_labels, problem_type):
        super().__init__(lucaone_config)
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.lucaone_config = lucaone_config

        print(f'lucaone_config: \n{lucaone_config}')
        self.classifier = nn.Linear(lucaone_config.hidden_size, num_labels)

        # Initialize weights and apply final processing
        self._init_weights(self.classifier)

        self.lucaone = lucaone_model
    
    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            # global_attention_mask: Optional[torch.Tensor] = None,
            # token_type_ids: Optional[torch.Tensor] = None,
            # position_ids: Optional[torch.Tensor] = None,
            # head_mask: Optional[torch.Tensor] = None,
            # inputs_embeds: Optional[torch.Tensor] = None,
            # output_keys: Optional[dict[str, set[str]]] = None,
            labels: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            # input_ids_b: Optional[torch.Tensor] = None,
            # attention_mask_b: Optional[torch.Tensor] = None,
            # global_attention_mask_b: Optional[torch.Tensor] = None,
            # token_type_ids_b: Optional[torch.Tensor] = None,
            # position_ids_b: Optional[torch.Tensor] = None,
            # head_mask_b: Optional[torch.Tensor] = None,
            # inputs_embeds_b: Optional[torch.Tensor] = None,
            # output_keys_b: Optional[dict[str, set[str]]] = None,
            # labels_b: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            # pair_label: Optional[dict[str, dict[str, torch.Tensor]]] = None,
            # pair_output_keys: Optional[dict[str, set[str]]] = None,
            # output_hidden_states: Optional[dict[str, set[str]]] = None,
            # output_attentions: Optional[dict[str, set[str]]] = None,
            # need_head_weights: Optional[bool] = None,
            # return_contacts: Optional[bool] = None,
            # repr_layers: Optional[list[int]] = None,
            # return_dict: Optional[bool] = None,
            # use_last_layer_norm: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        outputs = self.lucaone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_keys=output_keys,
            labels=labels,
            input_ids_b=input_ids_b,
            attention_mask_b=attention_mask_b,
            global_attention_mask_b=global_attention_mask_b,
            token_type_ids_b=token_type_ids_b,
            position_ids_b=position_ids_b,
            head_mask_b=head_mask_b,
            inputs_embeds_b=inputs_embeds_b,
            output_keys_b=output_keys_b,
            labels_b=labels_b,
            pair_label=pair_label,
            pair_output_keys=pair_output_keys,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            need_head_weights=need_head_weights,
            return_contacts=return_contacts,
            repr_layers=repr_layers,
            return_dict=return_dict,
            use_last_layer_norm=use_last_layer_norm
        )

        pooled_output = outputs.hidden_states
        print(f'pooled_output: {pooled_output.shape}')
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"
            #print(self.problem_type)
            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        #print('return_dict',return_dict)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# def rotate_half(x):
#     x1, x2 = x.chunk(2, dim=-1)
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(x, cos, sin):
#     cos = cos[:, :, : x.shape[-2], :]
#     sin = sin[:, :, : x.shape[-2], :]

#     return (x * cos) + (rotate_half(x) * sin)

# class RMSELoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.eps = eps

#     def forward(self, yhat, y):
#         loss = torch.sqrt(self.mse(yhat, y) + self.eps)
#         return loss


# class MCRMSELoss(nn.Module):
#     def __init__(self, num_scored=3):
#         super().__init__()
#         self.rmse = RMSELoss()
#         self.num_scored = num_scored

#     def forward(self, yhat, y):
#         score = 0
#         for i in range(self.num_scored):
#             score += self.rmse(yhat[:, :, i], y[:, :, i]) / self.num_scored
#         return score
        
# class RnaFmPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = RnaFmConfig
#     base_model_prefix = "rnafm"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["RnaFmLayer", "RnaFmEmbeddings"]

#     # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
#     def _init_weights(self, module: nn.Module):
#         """Initialize the weights"""
#         if isinstance(module, nn.Linear):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)


# class RnaFmModel(RnaFmPreTrainedModel):
#     """
#     Examples:
#         >>> from multimolecule import RnaFmConfig, RnaFmModel, RnaTokenizer
#         >>> config = RnaFmConfig()
#         >>> model = RnaFmModel(config)
#         >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
#         >>> input = tokenizer("ACGUN", return_tensors="pt")
#         >>> output = model(**input)
#     """

#     def __init__(self, config: RnaFmConfig, add_pooling_layer: bool = True):
#         super().__init__(config)
#         self.pad_token_id = config.pad_token_id
#         self.embeddings = RnaFmEmbeddings(config)
#         self.encoder = RnaFmEncoder(config)
#         self.pooler = RnaFmPooler(config) if add_pooling_layer else None

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings

#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value

#     def _prune_heads(self, heads_to_prune):
#         """
#         Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
#         class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def forward(
#         self,
#         input_ids: Tensor | torch.Tensor,
#         attention_mask: Tensor | None = None,
#         position_ids: Tensor | None = None,
#         head_mask: Tensor | None = None,
#         inputs_embeds: Tensor | None = None,
#         encoder_hidden_states: Tensor | None = None,
#         encoder_attention_mask: Tensor | None = None,
#         past_key_values: Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...] | None = None,
#         use_cache: bool | None = None,
#         output_attentions: bool | None = None,
#         output_hidden_states: bool | None = None,
#         return_dict: bool | None = None,
#     ) -> Tuple[Tensor, ...] | BaseModelOutputWithPoolingAndCrossAttentions:
#         r"""
#         encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors
#             of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
#             Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

#             If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
#             don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
#             `decoder_input_ids` of shape `(batch_size, sequence_length)`.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         """
#         # print(f"input_ids:\n{input_ids}")
#         # print(f"attention_mask:\n{attention_mask}")
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if self.config.is_decoder:
#             use_cache = use_cache if use_cache is not None else self.config.use_cache
#         else:
#             use_cache = False

        
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         if input_ids is not None:
#             self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         batch_size, seq_length = input_shape
#         device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore[union-attr]

#         # past_key_values_length
#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

#         if attention_mask is None:
#             attention_mask = (
#                 input_ids.ne(self.pad_token_id)
#                 if self.pad_token_id is not None
#                 else torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
#             )

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#             encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_extended_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         if inputs_embeds is None:
#             inputs_embeds = self.embeddings(
#                 input_ids=input_ids,
#                 position_ids=position_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 past_key_values_length=past_key_values_length,
#             )
#         # print(f"inputs_embeds: {inputs_embeds}")
#         # print(f"attention_mask: {attention_mask}")
#         # print(f"inputs_embeds: {inputs_embeds.shape}")
#         # print(f"attention_mask: {attention_mask.shape}")
#         encoder_outputs = self.encoder(
#             inputs_embeds,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             return (sequence_output, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             past_key_values=encoder_outputs.past_key_values,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#             cross_attentions=encoder_outputs.cross_attentions,
#             # attention_mask=attention_mask,
#         )


# class RnaFmForMaskedLM(RnaFmPreTrainedModel):
#     """
#     Examples:
#         >>> from multimolecule import RnaFmConfig, RnaFmForMaskedLM, RnaTokenizer
#         >>> config = RnaFmConfig()
#         >>> model = RnaFmForMaskedLM(config)
#         >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
#         >>> input = tokenizer("ACGUN", return_tensors="pt")
#         >>> output = model(**input)
#     """

#     _tied_weights_keys = ["lm_head.decoder.weight"]

#     def __init__(self, config: RnaFmConfig):
#         super().__init__(config)
#         if config.is_decoder:
#             logger.warning(
#                 "If you want to use `RnaFmForMaskedLM` make sure `config.is_decoder=False` for "
#                 "bi-directional self-attention."
#             )
#         self.rnafm = RnaFmModel(config, add_pooling_layer=False)
#         self.lm_head = MaskedLMHead(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Tensor | torch.Tensor,
#         attention_mask: Tensor | None = None,
#         position_ids: Tensor | None = None,
#         head_mask: Tensor | None = None,
#         inputs_embeds: Tensor | None = None,
#         encoder_hidden_states: Tensor | None = None,
#         encoder_attention_mask: Tensor | None = None,
#         labels: Tensor | None = None,
#         output_attentions: bool | None = None,
#         output_hidden_states: bool | None = None,
#         return_dict: bool | None = None,
#     ) -> Tuple[Tensor, ...] | MaskedLMOutput:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
#             loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#         """

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         output = self.lm_head(outputs, labels)
#         logits, loss = output.logits, output.loss

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return MaskedLMOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class RnaFmForPretraining(RnaFmPreTrainedModel):
#     """
#     Examples:
#         >>> from multimolecule import RnaFmConfig, RnaFmForPretraining, RnaTokenizer
#         >>> config = RnaFmConfig()
#         >>> model = RnaFmForPretraining(config)
#         >>> tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
#         >>> input = tokenizer("ACGUN", return_tensors="pt")
#         >>> output = model(**input)
#     """

#     _tied_weights_keys = ["head.predictions.decoder.weight"]

#     def __init__(self, config: RnaFmConfig):
#         super().__init__(config)
#         if config.is_decoder:
#             logger.warning(
#                 "If you want to use `RnaFmForPretraining` make sure `config.is_decoder=False` for "
#                 "bi-directional self-attention."
#             )
#         self.rnafm = RnaFmModel(config, add_pooling_layer=False)
#         self.pretrain_head = RnaFmPreTrainingHeads(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_output_embeddings(self):
#         return self.pretrain_head.predictions.decoder

#     def set_output_embeddings(self, embeddings):
#         self.pretrain_head.predictions.decoder = embeddings

#     def forward(
#         self,
#         input_ids: Tensor | torch.Tensor,
#         attention_mask: Tensor | None = None,
#         position_ids: Tensor | None = None,
#         head_mask: Tensor | None = None,
#         inputs_embeds: Tensor | None = None,
#         encoder_hidden_states: Tensor | None = None,
#         encoder_attention_mask: Tensor | None = None,
#         labels: Tensor | None = None,
#         labels_contact: Tensor | None = None,
#         output_attentions: bool | None = None,
#         output_hidden_states: bool | None = None,
#         return_dict: bool | None = None,
#     ) -> Tuple[Tensor, ...] | RnaFmForPretrainingOutput:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
#             config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
#             loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
#         """

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             output_attentions=True,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         logits, contact_map = self.pretrain_head(outputs, attention_mask, input_ids)

#         loss = None
#         if any(x is not None for x in [labels, labels_contact]):
#             loss_mlm = loss_contact = 0
#             if labels is not None:
#                 loss_mlm = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
#             if labels_contact is not None:
#                 loss_contact = F.mse_loss(contact_map.view(-1), labels_contact.view(-1))
#             loss = loss_mlm + loss_contact

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return RnaFmForPretrainingOutput(
#             loss=loss,
#             logits=logits,
#             contact_map=contact_map,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )



# class RnaFmEmbeddings(nn.Module):
#     """
#     Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
#     """

#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

#         if config.emb_layer_norm_before:
#             self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         else:
#             self.layer_norm = None
#         # position_ids (1, len position emb) is contiguous in memory and exported when serialized
#         self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
#         self.register_buffer(
#             "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
#         )

#         self.padding_idx = config.pad_token_id
#         if self.position_embedding_type == "absolute":
#             self.position_embeddings = nn.Embedding(
#                 config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
#             )
#         else:
#             self.position_embeddings = None
#         self.token_dropout = config.token_dropout
#         self.mask_token_id = config.mask_token_id

#     def forward(
#         self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
#     ):
#         if position_ids is None:
#             if input_ids is not None:
#                 # Create the position ids from the input token ids. Any padded tokens remain padded.
#                 position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
#             else:
#                 position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
#             # This is a bug in the original implementation
#             position_ids += 1

#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)

#         embeddings = inputs_embeds

#         if self.token_dropout:
#             embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
#             mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all RNAFM model training runs
#             src_lengths = attention_mask.sum(-1)
#             mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
#             embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
#                 embeddings.dtype
#             )

#         if self.position_embedding_type == "absolute":
#             position_embeddings = self.position_embeddings(position_ids)
#             embeddings = embeddings + position_embeddings

#         if self.layer_norm is not None:
#             embeddings = self.layer_norm(embeddings)
#         if attention_mask is not None:
#             embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
#         return embeddings

#     def create_position_ids_from_inputs_embeds(self, inputs_embeds):
#         """
#         We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

#         Args:
#             inputs_embeds: Tensor

#         Returns: Tensor
#         """
#         input_shape = inputs_embeds.size()[:-1]
#         sequence_length = input_shape[1]

#         position_ids = torch.arange(
#             self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
#         )
#         return position_ids.unsqueeze(0).expand(input_shape)


# class RotaryEmbedding(torch.nn.Module):
#     """
#     Rotary position embeddings based on those in
#     [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
#     matrices which depend on their relative positions.
#     """

#     def __init__(self, dim: int):
#         super().__init__()
#         # Generate and save the inverse frequency buffer (non trainable)
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
#         self.register_buffer("inv_freq", inv_freq)

#         self._seq_len_cached = None
#         self._cos_cached = None
#         self._sin_cached = None

#     def _update_cos_sin_tables(self, x, seq_dimension=2):
#         seq_len = x.shape[seq_dimension]

#         # Reset the tables if the sequence length has changed,
#         # or if we're on a new device (possibly due to tracing for instance)
#         if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
#             self._seq_len_cached = seq_len
#             t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
#             freqs = torch.outer(t, self.inv_freq)
#             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

#             self._cos_cached = emb.cos()[None, None, :, :]
#             self._sin_cached = emb.sin()[None, None, :, :]

#         return self._cos_cached, self._sin_cached

#     def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
#         self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

#         return (
#             apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
#             apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
#         )


# class RnaFmEncoder(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.config = config
#         self.layer = nn.ModuleList([RnaFmLayer(config) for _ in range(config.num_hidden_layers)])
#         self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.gradient_checkpointing = False

#     def forward(
#         self,
#         hidden_states: Tensor,
#         attention_mask: torch.FloatTensor | None = None,
#         head_mask: torch.FloatTensor | None = None,
#         encoder_hidden_states: torch.FloatTensor | None = None,
#         encoder_attention_mask: torch.FloatTensor | None = None,
#         past_key_values: Tuple[Tuple[torch.FloatTensor, ...], ...] | None = None,
#         use_cache: bool | None = None,
#         output_attentions: bool = False,
#         output_hidden_states: bool = False,
#         return_dict: bool = True,
#     ) -> Tuple[Tensor, ...] | BaseModelOutputWithPastAndCrossAttentions:
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

#         if self.gradient_checkpointing and self.training and use_cache:
#             logger.warning_once(
#                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#             )
#             use_cache = False

#         next_decoder_cache = () if use_cache else None
#         for i, layer_module in enumerate(self.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

#             layer_head_mask = head_mask[i] if head_mask is not None else None
#             past_key_value = past_key_values[i] if past_key_values is not None else None

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     layer_module.__call__,
#                     hidden_states,
#                     attention_mask,
#                     layer_head_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     past_key_value,
#                     output_attentions,
#                 )
#             else:
#                 layer_outputs = layer_module(
#                     hidden_states,
#                     attention_mask,
#                     layer_head_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     past_key_value,
#                     output_attentions,
#                 )

#             hidden_states = layer_outputs[0]
#             if use_cache:
#                 next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)  # type: ignore[operator]
#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)  # type: ignore[operator]
#                 if self.config.add_cross_attention:
#                     all_cross_attentions = all_cross_attentions + (layer_outputs[2],)  # type: ignore[operator]

#         if self.emb_layer_norm_after:
#             hidden_states = self.emb_layer_norm_after(hidden_states)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     hidden_states,
#                     next_decoder_cache,
#                     all_hidden_states,
#                     all_self_attentions,
#                     all_cross_attentions,
#                 ]
#                 if v is not None
#             )
#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=next_decoder_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             cross_attentions=all_cross_attentions,
#         )


# class RnaFmLayer(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.seq_len_dim = 1
#         self.attention = RnaFmAttention(config)
#         self.is_decoder = config.is_decoder
#         self.add_cross_attention = config.add_cross_attention
#         if self.add_cross_attention:
#             if not self.is_decoder:
#                 raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
#             self.crossattention = RnaFmAttention(config)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.intermediate = RnaFmIntermediate(config)
#         self.output = RnaFmOutput(config)

#     def forward(
#         self,
#         hidden_states: Tensor,
#         attention_mask: torch.FloatTensor | None = None,
#         head_mask: torch.FloatTensor | None = None,
#         encoder_hidden_states: torch.FloatTensor | None = None,
#         encoder_attention_mask: torch.FloatTensor | None = None,
#         past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
#         output_attentions: bool = False,
#     ) -> Tuple[Tensor, ...]:
#         # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
#         self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
#         self_attention_outputs = self.attention(
#             hidden_states,
#             attention_mask,
#             head_mask,
#             output_attentions=output_attentions,
#             past_key_value=self_attn_past_key_value,
#         )
#         attention_output = self_attention_outputs[0]

#         # if decoder, the last output is tuple of self-attn cache
#         if self.is_decoder:
#             outputs = self_attention_outputs[1:-1]
#             present_key_value = self_attention_outputs[-1]
#         else:
#             outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

#         cross_attn_present_key_value = None
#         if self.is_decoder and encoder_hidden_states is not None:
#             if not hasattr(self, "crossattention"):
#                 raise AttributeError(
#                     f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
#                     " with cross-attention layers by setting `config.add_cross_attention=True`"
#                 )

#             # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
#             cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
#             cross_attention_outputs = self.crossattention(
#                 attention_output,
#                 attention_mask,
#                 head_mask,
#                 encoder_hidden_states,
#                 encoder_attention_mask,
#                 cross_attn_past_key_value,
#                 output_attentions,
#             )
#             attention_output = cross_attention_outputs[0]
#             outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

#             # add cross-attn cache to positions 3,4 of present_key_value tuple
#             cross_attn_present_key_value = cross_attention_outputs[-1]
#             present_key_value = present_key_value + cross_attn_present_key_value

#         layer_output = self.feed_forward_chunk(attention_output)

#         outputs = (layer_output,) + outputs

#         # if decoder, return the attn key/values as the last output
#         if self.is_decoder:
#             outputs = outputs + (present_key_value,)

#         return outputs

#     def feed_forward_chunk(self, attention_output):
#         attention_output_ln = self.layer_norm(attention_output)
#         intermediate_output = self.intermediate(attention_output_ln)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output


# class RnaFmAttention(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.self = RnaFmSelfAttention(config)
#         self.output = RnaFmSelfOutput(config)
#         self.pruned_heads: set = set()
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

#     def prune_heads(self, heads):
#         if len(heads) == 0:
#             return
#         heads, index = find_pruneable_heads_and_indices(
#             heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
#         )

#         # Prune linear layers
#         self.self.query = prune_linear_layer(self.self.query, index)
#         self.self.key = prune_linear_layer(self.self.key, index)
#         self.self.value = prune_linear_layer(self.self.value, index)
#         self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

#         # Update hyper params and store pruned heads
#         self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
#         self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
#         self.pruned_heads = self.pruned_heads.union(heads)

#     def forward(
#         self,
#         hidden_states: Tensor,
#         attention_mask: torch.FloatTensor | None = None,
#         head_mask: torch.FloatTensor | None = None,
#         encoder_hidden_states: torch.FloatTensor | None = None,
#         encoder_attention_mask: torch.FloatTensor | None = None,
#         past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
#         output_attentions: bool = False,
#     ) -> Tuple[Tensor, ...]:
#         hidden_states_ln = self.layer_norm(hidden_states)
#         self_outputs = self.self(
#             hidden_states_ln,
#             attention_mask,
#             head_mask,
#             encoder_hidden_states,
#             encoder_attention_mask,
#             past_key_value,
#             output_attentions,
#         )
#         attention_output = self.output(self_outputs[0], hidden_states)
#         outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
#         return outputs


# class RnaFmSelfAttention(nn.Module):
#     def __init__(self, config: RnaFmConfig, position_embedding_type: str | None = None):
#         super().__init__()
#         if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
#             raise ValueError(
#                 f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
#                 f"heads ({config.num_attention_heads})"
#             )

#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_dropout)
#         self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
#         self.rotary_embeddings = None
#         if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#             self.max_position_embeddings = config.max_position_embeddings
#             self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
#         elif self.position_embedding_type == "rotary":
#             self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

#         self.is_decoder = config.is_decoder

#     def transpose_for_scores(self, x: Tensor) -> Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states: Tensor,
#         attention_mask: torch.FloatTensor | None = None,
#         head_mask: torch.FloatTensor | None = None,
#         encoder_hidden_states: torch.FloatTensor | None = None,
#         encoder_attention_mask: torch.FloatTensor | None = None,
#         past_key_value: Tuple[torch.FloatTensor, torch.FloatTensor] | None = None,
#         output_attentions: bool = False,
#     ) -> Tuple[Tensor, ...]:
#         mixed_query_layer = self.query(hidden_states)

#         # If this is instantiated as a cross-attention module, the keys
#         # and values come from an encoder; the attention mask needs to be
#         # such that the encoder's padding tokens are not attended to.
#         is_cross_attention = encoder_hidden_states is not None

#         if is_cross_attention and past_key_value is not None:
#             # reuse k,v, cross_attentions
#             key_layer = past_key_value[0]
#             value_layer = past_key_value[1]
#             attention_mask = encoder_attention_mask
#         elif is_cross_attention:
#             key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
#             value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
#             attention_mask = encoder_attention_mask
#         elif past_key_value is not None:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))
#             key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
#             value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
#         else:
#             key_layer = self.transpose_for_scores(self.key(hidden_states))
#             value_layer = self.transpose_for_scores(self.value(hidden_states))

#         query_layer = self.transpose_for_scores(mixed_query_layer)

#         query_layer = query_layer * self.attention_head_size**-0.5

#         if self.is_decoder:
#             # if cross_attention save Tuple(Tensor, Tensor) of all cross attention key/value_states.
#             # Further calls to cross_attention layer can then reuse all cross-attention
#             # key/value_states (first "if" case)
#             # if uni-directional self-attention (decoder) save Tuple(Tensor, Tensor) of
#             # all previous decoder key/value_states. Further calls to uni-directional self-attention
#             # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#             # if encoder bi-directional self-attention `past_key_value` is always `None`
#             past_key_value = (key_layer, value_layer)

#         if self.position_embedding_type == "rotary":
#             query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)  # type: ignore[misc]

#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # type: ignore[attr-defined]

#         if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#             seq_length = hidden_states.size()[1]
#             position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
#             position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
#             distance = position_ids_l - position_ids_r
#             positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
#             positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

#             if self.position_embedding_type == "relative_key":
#                 relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#                 attention_scores = attention_scores + relative_position_scores
#             elif self.position_embedding_type == "relative_key_query":
#                 relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#                 relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
#                 attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

#         if attention_mask is not None:
#             # Apply the attention mask is (precomputed for all layers in RnaFmModel forward() function)
#             attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs.to(value_layer.dtype), value_layer)  # type: ignore[attr-defined]

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

#         if self.is_decoder:
#             outputs = outputs + (past_key_value,)
#         return outputs


# class RnaFmSelfOutput(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout)

#     def forward(self, hidden_states, input_tensor):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = hidden_states + input_tensor
#         return hidden_states


# class RnaFmIntermediate(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states: Tensor) -> Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states


# class RnaFmOutput(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout)

#     def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = hidden_states + input_tensor
#         return hidden_states


# # Copied from transformers.models.bert.modeling_bert.BertPooler
# class RnaFmPooler(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()

#     def forward(self, hidden_states: Tensor) -> Tensor:
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output


# class RnaFmPreTrainingHeads(nn.Module):
#     def __init__(self, config: RnaFmConfig):
#         super().__init__()
#         self.contact = ContactPredictionHead(config)
#         self.predictions = MaskedLMHead(config)

#     def forward(
#         self,
#         outputs: BaseModelOutputWithPastAndCrossAttentions | Tuple[Tensor, ...],
#         attention_mask: Tensor | None = None,
#         input_ids: Tensor | torch.Tensor | None = None,
#     ) -> Tuple[Tensor, Tensor]:
#         logits = self.predictions(outputs)
#         contact_map = self.contact(torch.stack(outputs[-1], 1), attention_mask, input_ids)
#         return logits, contact_map


# @dataclass
# class RnaFmForPretrainingOutput(ModelOutput):
#     loss: torch.FloatTensor | None = None
#     logits: torch.FloatTensor = None
#     contact_map: torch.FloatTensor | None = None
#     hidden_states: Tuple[torch.FloatTensor, ...] | None = None
#     attentions: Tuple[torch.FloatTensor, ...] | None = None


# def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
#     """
#     Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
#     are ignored. This is modified from fairseq's `utils.make_positions`.

#     Args:
#         x: Tensor x:

#     Returns: Tensor
#     """
#     # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
#     mask = input_ids.ne(padding_idx).int()
#     incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
#     return incremental_indices.long() + padding_idx

# class RnaFmForSequenceClassification(RnaFmPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config

#         self.rnafm = RnaFmModel(config)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         #return_dict = None
#         outputs = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#             #print(self.config.problem_type)
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         #print('return_dict',return_dict)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

# class RnaFmForNucleotideLevel(RnaFmPreTrainedModel):
#     # include Degradation and SpliceAI
#     def __init__(self, config, tokenizer=None):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
    
#         self.rnafm = RnaFmModel(config)
       
#         self.tokenizer = tokenizer
#         if self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
#             self.classifier_a = nn.Linear(config.hidden_size, config.num_labels)
#             self.classifier_t = nn.Linear(config.hidden_size, config.num_labels)
#             self.classifier_c = nn.Linear(config.hidden_size, config.num_labels)
#             self.classifier_g = nn.Linear(config.hidden_size, config.num_labels)
#             self.classifier_n = nn.Linear(config.hidden_size, config.num_labels)
#             self.classifer_dict = {
#                 'A': self.classifier_a,
#                 'T': self.classifier_t,
#                 'C': self.classifier_c,
#                 'G': self.classifier_g,
#                 'N': self.classifier_n,
#                 }
#         else:
#             self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         # Initialize weights and apply final processing
#         self.post_init()


#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         weight_mask: Optional[bool] = None,
#         post_token_length: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         outputs = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             # token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         final_input= outputs[0]

#         ### init mappint tensor
#         ori_length = weight_mask.shape[1]
#         batch_size = final_input.shape[0]
#         cur_length = int(final_input.shape[1])

#         if self.config.token_type == 'single':
#             assert attention_mask.shape==weight_mask.shape==post_token_length.shape
#             mapping_final_input = final_input
#         elif self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
#             logits = torch.zeros((batch_size, ori_length, self.num_labels), dtype=final_input.dtype, device=final_input.device)
#             nucleotide_indices = {nucleotide: (input_ids == self.tokenizer.encode(nucleotide, add_special_tokens=False)[0]).nonzero() for nucleotide in 'ATCGN'}
#             mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
#             for bz in range(batch_size):
#                 start_index = 0
#                 for i, length in enumerate(post_token_length[bz]): #astart from [cls]
#                     mapping_final_input[bz,start_index:start_index + int(length.item()), :] = final_input[bz,i,:]
#                     start_index += int(length.item())
#             for nucleotide, indices in nucleotide_indices.items(): # indices:[bzid,seqid]
#                 #print(nucleotide, indices) 
#                 if indices.numel() > 0:  
#                     bz_indices, pos_indices = indices.split(1, dim=1)
#                     bz_indices = bz_indices.squeeze(-1) 
#                     pos_indices = pos_indices.squeeze(-1)
#                     nucleotide_logits = self.classifer_dict[nucleotide](mapping_final_input[bz_indices, pos_indices])
#                     nucleotide_logits = nucleotide_logits.to(logits.dtype)
#                     logits.index_put_((bz_indices, pos_indices), nucleotide_logits)
#             # for bz in range(batch_size): #exclude cls,sep token
#             #     ori_text = original_texts[bz]
#             #     ori_text = ori_text.replace(" ", "")
#             #     #print(len(ori_text))
#             #     for pos_id in range(1,ori_length-1):
#             #         logits[bz,pos_id,:] = self.classifer_dict[ori_text[pos_id-1]](mapping_final_input[bz,pos_id, :])
#         elif self.config.token_type == '6mer':
#             mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
#             mapping_final_input[:,0,:] = final_input[:,0,:] #[cls] token
#             for bz in range(batch_size):
#                 value_length = torch.sum(attention_mask[bz,:]==1).item()
#                 for i in range(1,value_length-1): #exclude cls,sep token
#                     mapping_final_input[bz,i:i+6,:] += final_input[bz,i]
#                 mapping_final_input[bz,value_length+5-1,:] = final_input[bz,value_length-1,:] #[sep] token
#         #print(mapping_final_input.shape,weight_mask.shape)
#         mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
#         if self.config.token_type == '6mer' or self.config.token_type =='single': 
#             logits = self.classifier(mapping_final_input)
#         #print(logits.shape)
        
#         loss = None
#         if labels is not None:
#             logits = logits[:, 1:1+labels.size(1), :]
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MCRMSELoss()
                
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 #logits = logits[:, 1:1+labels.size(1), :]
#                 loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1).long())
#             # elif self.config.problem_type == "multi_label_classification":
#             #     loss_fct = BCEWithLogitsLoss()
#             #     loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

# class RnaFmForCRISPROffTarget(RnaFmPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
    
#         self.rnafm = RnaFmModel(config)


#         self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         target_input_ids: Optional[torch.Tensor] = None,
#         target_attention_mask: Optional[torch.Tensor] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
#         sgrna_out = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             # token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )[1]
#         target_out = self.rnafm(
#             target_input_ids,
#             attention_mask=target_attention_mask,
#             # token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )[1]
#         final_input = torch.cat([sgrna_out,target_out],dim=-1)
#         logits = self.classifier(final_input)
#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#             #print(self.config.problem_type)
#             if self.config.problem_type == "regression":
#                 loss_fct = nn.MSELoss()
#                 if self.num_labels == 1:
                    
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + sgrna_out[2:]
#             return ((loss,) + output) if loss is not None else output
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=None,
#             attentions=None,
#         )
# class RnaFmForStructuralimputation(RnaFmPreTrainedModel):
#     # include Degradation and SpliceAI
#     def __init__(self, config, tokenizer=None):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config
    
#         self.rnafm = RnaFmModel(config)
       
#         self.tokenizer = tokenizer
#         if self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
#             self.down_mlp_a = nn.Linear(config.hidden_size, config.hidden_size)
#             self.down_mlp_t = nn.Linear(config.hidden_size, config.hidden_size)
#             self.down_mlp_c = nn.Linear(config.hidden_size, config.hidden_size)
#             self.down_mlp_g = nn.Linear(config.hidden_size, config.hidden_size)
#             self.down_mlp_n = nn.Linear(config.hidden_size, config.hidden_size)
#             self.down_mlp_dict = {
#                 'A': self.down_mlp_a,
#                 'T': self.down_mlp_t,
#                 'C': self.down_mlp_c,
#                 'G': self.down_mlp_g,
#                 'N': self.down_mlp_n,
#                 }
#         else:
#             self.down_mlp = nn.Linear(config.hidden_size, config.hidden_size)
#         self.embedding_struct = nn.Linear(1,config.hidden_size)
#         self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
#         # Initialize weights and apply final processing
#         self.post_init()


#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         struct: Optional[torch.Tensor] = None,
#         weight_mask: Optional[bool] = None,
#         post_token_length: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         outputs = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             # token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         final_input= outputs[0]

#         ### init mappint tensor
#         ori_length = weight_mask.shape[1]
#         batch_size = final_input.shape[0]
#         cur_length = int(final_input.shape[1])

#         if self.config.token_type == 'single':
#             assert attention_mask.shape==weight_mask.shape==post_token_length.shape
#             mapping_final_input = final_input
#         elif self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
#             inter_input = torch.zeros((batch_size, ori_length, self.config.hidden_size), dtype=final_input.dtype, device=final_input.device)
#             nucleotide_indices = {nucleotide: (input_ids == self.tokenizer.encode(nucleotide, add_special_tokens=False)[0]).nonzero() for nucleotide in 'ATCGN'}
#             mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
#             for bz in range(batch_size):
#                 start_index = 0
#                 for i, length in enumerate(post_token_length[bz]): #astart from [cls]
#                     mapping_final_input[bz,start_index:start_index + int(length.item()), :] = final_input[bz,i,:]
#                     start_index += int(length.item())
#             for nucleotide, indices in nucleotide_indices.items(): # indices:[bzid,seqid]
#                 #print(nucleotide, indices) 
#                 if indices.numel() > 0:  
#                     bz_indices, pos_indices = indices.split(1, dim=1)
#                     bz_indices = bz_indices.squeeze(-1) 
#                     pos_indices = pos_indices.squeeze(-1)
#                     nucleotide_logits = self.down_mlp_dict[nucleotide](mapping_final_input[bz_indices, pos_indices])
#                     nucleotide_logits = nucleotide_logits.to(inter_input.dtype)
#                     inter_input.index_put_((bz_indices, pos_indices), nucleotide_logits)
#             #mapping_final_input = inter_input[:,1:-1,:]
#         elif self.config.token_type == '6mer':
#             mapping_final_input = torch.zeros((batch_size, ori_length, final_input.shape[-1]), dtype=final_input.dtype, device=final_input.device)
#             mapping_final_input[:,0,:] = final_input[:,0,:] #[cls] token
#             for bz in range(batch_size):
#                 value_length = torch.sum(attention_mask[bz,:]==1).item()
#                 for i in range(1,value_length-1): #exclude cls,sep token
#                     mapping_final_input[bz,i:i+6,:] += final_input[bz,i]
#                 mapping_final_input[bz,value_length+5-1,:] = final_input[bz,value_length-1,:] #[sep] token
#         #print(mapping_final_input.shape,weight_mask.shape)
#         mapping_final_input = mapping_final_input * weight_mask.unsqueeze(2)
        
#         if self.config.token_type == '6mer' or self.config.token_type =='single': 
#             mapping_final_input = self.down_mlp(mapping_final_input)[:,1:-1,:] # exclude <cls> and <eos>
#         elif self.config.token_type == 'bpe' or  self.config.token_type=='non-overlap':
#             mapping_final_input = mapping_final_input[:,1:-1,:]
#         # print(labels.shape)
#         # print(struct.shape,mapping_final_input.shape)
#         struct_input = self.embedding_struct(struct.unsqueeze(-1))
        
#         final_input = torch.cat([mapping_final_input,struct_input], dim=-1)

#         logits = self.classifier(final_input)
#         label_mask = struct== -1

    
#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#             if self.config.problem_type == "regression":
#                 loss_fct = nn.MSELoss()
#                 # print()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits[label_mask].squeeze(), labels.squeeze())
#                     # print(loss)
#                 else:
#                     loss = loss_fct(logits[label_mask], labels)

#         if not return_dict:
#             output = (logits[label_mask],) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output       
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits[label_mask],
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class RnaFmEncoderPoolingForSequenceClassification(RnaFmPreTrainedModel):
#     def __init__(
#         self,
#         config,
#         # wEncoder=False,
#         # encoder_hidden_size=0,
#         # encoder_num_hidden_layers=0,
#         # encoder_num_attention_heads=0,
#         # encoder_intermediate_size=0,
#         # encoder_hidden_act="gelu",
#         # encoder_hidden_dropout=0.1,
#         # encoder_attention_dropout=0.1,
#         # encoder_layer_norm_eps=1e-12,
#         # pooling_type=None,
#         # encoder_classifer_size=0
#         ):
#         super().__init__(config)
#         print(config)
#         self.num_labels = config.num_labels
#         self.config = config

#         self.rnafm = RnaFmModel(config)

#         if config.wEncoder:
#             import copy

#             self.embedding2 = nn.Linear(config.hidden_size, config.encoder_hidden_size)

#             matrix_encoder_config = copy.deepcopy(config)

#             matrix_encoder_config.hidden_size = config.encoder_hidden_size
#             matrix_encoder_config.num_hidden_layers = config.encoder_num_hidden_layers
#             matrix_encoder_config.num_attention_heads = config.encoder_num_attention_heads
#             matrix_encoder_config.intermediate_size = config.encoder_intermediate_size
#             # config.hidden_act=self.encoder_hidden_act
#             # config.hidden_dropout=self.encoder_hidden_dropout
#             # config.attention_dropout=self.encoder_attention_dropout
#             # config.layer_norm_eps=self.encoder_layer_norm_eps

#             self.encoder2 = RnaFmEncoder(matrix_encoder_config)
#             # self.emb_layer_norm_after = nn.LayerNorm(matrix_encoder_config.hidden_size, eps=config.layer_norm_eps)
        
#         if config.pooling_type:
#             self.matrix_pooler = create_pooler(pooling_type=config.pooling_type, hidden_size=matrix_encoder_config.hidden_size)
#             self.dropout = nn.Dropout(matrix_encoder_config.hidden_dropout)
#             self.encoder_hidden_layer = nn.Linear(matrix_encoder_config.hidden_size, config.encoder_classifer_size, bias=True)
#             self.encoder_intermediate_act_fn = ACT2FN[matrix_encoder_config.hidden_act]
#             self.classifier = nn.Linear(config.encoder_classifer_size, matrix_encoder_config.num_labels)
#         else:
#             self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         # print(f"input_ids:\n{input_ids.shape}")
#         # print(f"attention_mask:\n{attention_mask.shape}")
        
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         #return_dict = None
#         outputs = self.rnafm(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[0]
#         # print(f"pooled_output:\n{pooled_output.shape}")

#         # attention_mask = outputs[6] #
#         encoder2_output = self.embedding2(pooled_output)

        
#         input_shape = encoder2_output.size()[:-1]

        
#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

#         # encoder2_output = self.encoder2(
#         #     input_ids=None,
#         #     attention_mask=attention_mask, #
#         #     position_ids=position_ids,
#         #     head_mask=head_mask,
#         #     inputs_embeds=encoder2_output,
#         #     output_attentions=output_attentions,
#         #     output_hidden_states=output_hidden_states,
#         #     return_dict=return_dict,
#         # )

#         # print(f"encoder2_output:\n{encoder2_output.shape}")
#         # print(f"extended_attention_mask:\n{extended_attention_mask.shape}")
#         # print(f"head_mask:\n{head_mask}")
#         # print(f"output_attentions:\n{output_attentions}")
#         # print(f"output_hidden_states:\n{output_hidden_states}")
#         encoder2_output = self.encoder2(
#             encoder2_output,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
        
#         # print(f"encoder2_output:\n{encoder2_output}")

#         # if self.emb_layer_norm_after:
#         #     encoder2_output[1] = self.emb_layer_norm_after(encoder2_output[1])
#         pooled_output2 = self.matrix_pooler(encoder2_output[0], mask=attention_mask)
#         pooled_output2 = self.dropout(pooled_output2)
#         pooled_output2 = self.encoder_hidden_layer(pooled_output2)
#         pooled_output2 = self.encoder_intermediate_act_fn(pooled_output2)
#         logits = self.classifier(pooled_output2)

#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#             #print(self.config.problem_type)
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         #print('return_dict',return_dict)
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

# class RnaFmForRegression512concat(RnaFmPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.config = config

#         self.rnafm = RnaFmModel(config)
#         self.classifier = nn.Linear(config.hidden_size*12, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = None

#         # print(f"input_ids: {input_ids}")
#         # print(f"attention_mask: {attention_mask}")
#         # print(f"input_ids: {input_ids.shape}")
#         # print(f"attention_mask: {attention_mask.shape}")

#         output1 = self.rnafm(
#             input_ids[:,:512],
#             attention_mask=attention_mask[:,:512],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output2 = self.rnafm(
#             input_ids[:,512:1024],
#             attention_mask=attention_mask[:,512:1024],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output3 = self.rnafm(
#             input_ids[:,1024:1536],
#             attention_mask=attention_mask[:,1024:1536],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output4 = self.rnafm(
#             input_ids[:,1536:2048],
#             attention_mask=attention_mask[:,1536:2048],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output5 = self.rnafm(
#             input_ids[:,2048:2560],
#             attention_mask=attention_mask[:,2048:2560],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output6 = self.rnafm(
#             input_ids[:,2560:3072],
#             attention_mask=attention_mask[:,2560:3072],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output7 = self.rnafm(
#             input_ids[:,3072:3584],
#             attention_mask=attention_mask[:,3072:3584],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output8 = self.rnafm(
#             input_ids[:,3584:4096],
#             attention_mask=attention_mask[:,3584:4096],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output9 = self.rnafm(
#             input_ids[:,4096:4608],
#             attention_mask=attention_mask[:,4096:4608],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output10 = self.rnafm(
#             input_ids[:,4608:5120],
#             attention_mask=attention_mask[:,4608:5120],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output11 = self.rnafm(
#             input_ids[:,5120:5632],
#             attention_mask=attention_mask[:,5120:5632],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         output12 = self.rnafm(
#             input_ids[:,5632:],
#             attention_mask=attention_mask[:,5632:],
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )[0][:,0,:]
#         # print(output12.shape)

#         outputs = torch.cat((output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12),dim = -1)
#         logits = self.classifier(outputs)

#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#             #print(self.config.problem_type)
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         #print('return_dict',return_dict)
#         if not return_dict:
#             output = (logits,) # + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
