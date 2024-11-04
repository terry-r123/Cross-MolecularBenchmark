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
from typing import Optional, Union, Tuple

class ProteinRNAInter(nn.Module):
    def __init__(self, prot_config, rna_config, prot_model, rna_model):
        super().__init__()
        self.num_labels = prot_config.num_labels
        self.prot_model = prot_model
        self.rna_model = rna_model
        
        self.classifier = nn.Linear(prot_config.hidden_size + rna_config.hidden_size, prot_config.num_labels)

        self.prot_config = prot_config
        self.rna_config = rna_config
        # Initialize weights and apply final processing
        #self.post_init()


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
        rna_input_ids: Optional[torch.Tensor] = None,
        rna_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.prot_config.use_return_dict
        
        
        prot_out = self.prot_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        rna_out = self.rna_model(
            rna_input_ids,
            attention_mask=rna_attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        final_input = torch.cat([prot_out,rna_out],dim=-1)
        logits = self.classifier(final_input)
        loss = None
        if labels is not None:
            if self.prot_config.problem_type is None:
                if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.prot_config.problem_type = "single_label_classification"
            elif self.prot_config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + prot_out[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class RNARNAInter(nn.Module):
    def __init__(self, omics1_config, omics2_config, omics1_model, omics2_model, use_features):
        super().__init__()
        self.num_labels = omics1_config.num_labels
        self.omics1_model = omics1_model
        self.omics2_model = omics2_model
        
        

        self.omics1_config = omics1_config
        self.omics2_config = omics2_config

        if use_features:
            self.feature_embedding = nn.Linear(25, 100)
            self.classifier = nn.Linear(omics1_config.hidden_size + omics2_config.hidden_size+100, omics1_config.num_labels)
        else:
            
            self.classifier = nn.Linear(omics1_config.hidden_size + omics2_config.hidden_size, omics1_config.num_labels)
        # Initialize weights and apply final processing
        # self.post_init()
        self.use_features = use_features

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
        features: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.omics1_config.use_return_dict
        
        
        omics1_out = self.omics1_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        omics2_out = self.omics2_model(
            omics2_input_ids,
            attention_mask=omics2_attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        final_input = torch.cat([omics1_out,omics2_out],dim=-1)
        if self.use_features:
            features = self.feature_embedding(features.float())
            final_input = torch.cat([final_input, features], dim=-1)
            #breakpoint()
        logits = self.classifier(final_input)
        loss = None
        if labels is not None:
            if self.omics1_config.problem_type is None:
                if self.num_labels == 1:
                    self.omics1_config.problem_type = "regression"
                if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.omics1_config.problem_type = "single_label_classification"
                else:
                    self.omics1_config.problem_type = "multi_label_classification"
            if self.omics1_config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:    
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.omics1_config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.omics1_config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + omics1_out[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
class RNADNAInter(nn.Module):
    def __init__(self, omics1_config, omics2_config, omics1_model, omics2_model, use_features):
        super().__init__()
        self.num_labels = omics1_config.num_labels
        self.omics1_model = omics1_model
        self.omics2_model = omics2_model
        
        

        self.omics1_config = omics1_config
        self.omics2_config = omics2_config

        if use_features:
            self.feature_embedding = nn.Linear(23, 1)
            self.classifier = nn.Linear(omics1_config.hidden_size + omics2_config.hidden_size+8, omics1_config.num_labels)
        else:
            
            self.classifier = nn.Linear(omics1_config.hidden_size + omics2_config.hidden_size, omics1_config.num_labels)
        # Initialize weights and apply final processing
        # self.post_init()
        self.use_features = use_features

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
        features: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.omics1_config.use_return_dict
        
        
        omics1_out = self.omics1_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        omics2_out = self.omics2_model(
            omics2_input_ids,
            attention_mask=omics2_attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        final_input = torch.cat([omics1_out,omics2_out],dim=-1)
        if self.use_features:
            features = self.feature_embedding(features).squeeze(dim=-1)

            final_input = torch.cat([final_input, features], dim=-1)

        logits = self.classifier(final_input)
        loss = None
        if labels is not None:
            if self.omics1_config.problem_type is None:
                if self.num_labels == 1:
                    self.omics1_config.problem_type = "regression"
                if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.omics1_config.problem_type = "single_label_classification"
                else:
                    self.omics1_config.problem_type = "multi_label_classification"
            if self.omics1_config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:    
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.omics1_config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.omics1_config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + omics1_out[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class OmicsOmicsInter(nn.Module):
    def __init__(self, omics1_config, omics2_config, omics1_model, omics2_model,weight=None):
        super().__init__()
        self.num_labels = omics1_config.num_labels
        self.omics1_model = omics1_model
        self.omics2_model = omics2_model
        
        self.classifier = nn.Linear(omics1_config.hidden_size + omics2_config.hidden_size, omics1_config.num_labels)

        self.omics1_config = omics1_config
        self.omics2_config = omics2_config
        # Initialize weights and apply final processing
        # self.post_init()

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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.omics1_config.use_return_dict
        
        
        omics1_out = self.omics1_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        omics2_out = self.omics2_model(
            omics2_input_ids,
            attention_mask=omics2_attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]
        final_input = torch.cat([omics1_out,omics2_out],dim=-1)
        logits = self.classifier(final_input)
        loss = None
        if labels is not None:
            if self.omics1_config.problem_type is None:
                if self.num_labels == 1:
                    self.omics1_config.problem_type = "regression"
                if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.omics1_config.problem_type = "single_label_classification"
                else:
                    self.omics1_config.problem_type = "multi_label_classification"
            if self.omics1_config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:    
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.omics1_config.problem_type == "single_label_classification":
                if self.weight is not None:
                    self.weight = torch.tensor(self.weight, device=logits.device)
                    print(self.weight)
                loss_fct = CrossEntropyLoss(weight=self.weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.omics1_config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + omics1_out[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )