import sys

import os
import json
import time, gzip, pickle
import numpy as np
import random
import string
from tqdm import tqdm

from rdkit import Chem

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pdb as depdb
import transformers
import torch.nn.functional as F

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}


def pad_to_fixed_length(tensor, target_length=256, padding_value=0):
    current_length = tensor.size(0)
    if current_length < target_length:

        padding = (0, target_length - current_length)
        tensor = F.pad(tensor, padding, value=padding_value)
    else:
        tensor = tensor[:target_length]
    return tensor

class ProtDNADataset(Dataset):
    def __init__(self, data_json_path, mmcif_dir, feats_dir, dna_tokenizer=None, protein_tokenizer=None):
        
        self.protein_chain_info = json.load(open(data_json_path, 'r'))
        self.mmcif_dir = mmcif_dir
        self.feats_dir = feats_dir

        self._chain_ids = list(self.protein_chain_info.keys())
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }
        self.dna_tokenizer = dna_tokenizer
        self.protein_tokenizer = protein_tokenizer

        
    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]
    
    def __len__(self):
        return len(self.protein_chain_info)

    def __getitem__(self, idx):

        pro_dna_chain_key = self.idx_to_chain_id(idx)
        
        pro_dna_chain_info =  self.protein_chain_info[pro_dna_chain_key]
        name = pro_dna_chain_key.split('_')[0]
        protein_chain = pro_dna_chain_info['protein_chain_id']
        dna_chain = pro_dna_chain_info['dna_chain']
        dna_reverse_chain = pro_dna_chain_info['dna_reverse_chain']

        protein_chain_name = f"{name}_protein_{protein_chain}"
        dna_chain_name = f"{name}_dna_{dna_chain}"
        dna_reverse_chain_name = f"{name}_dna_{dna_reverse_chain}"


        temp_data_path = os.path.join(self.feats_dir, f'{name}.npy')
        data = np.load(temp_data_path, allow_pickle=True).item()
        # dna input
        dna_seq = data[dna_chain_name]['feat']['seq'].replace('X', 'N')
        dna_seq_reverse = data[dna_reverse_chain_name]['feat']['seq'].replace('X', 'N')
        dna_tokens = self.dna_tokenizer.tokenize(dna_seq)
        dna_tokens_len = torch.tensor([[len(token) for token in dna_tokens]])
        dna_token_input_ids = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq], add_special_tokens=False)['input_ids']) # .to(torch.device("cuda:0" ))
        dna_token_attention_mask = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq], add_special_tokens=False)['attention_mask'])

        dna_reverse_tokens = self.dna_tokenizer.tokenize(dna_seq_reverse)
        dna_reverse_tokens_len = torch.tensor([[len(token) for token in dna_reverse_tokens]])
        dna_reverse_token_input_ids = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq_reverse], add_special_tokens=False)['input_ids']) # .to(torch.device("cuda:0" ))
        dna_token_reverse_attention_mask = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq_reverse], add_special_tokens=False)['attention_mask'])

        weight_mask = torch.ones(dna_token_input_ids.shape[0], len(dna_seq))
        reverse_weight_mask = torch.ones(dna_reverse_token_input_ids.shape[0], len(dna_seq_reverse))
        cls_token_len = torch.zeros(dna_token_input_ids.shape[0],1)
        post_token_length = torch.cat([cls_token_len, dna_tokens_len], dim=1)
        reverse_post_token_length = torch.cat([cls_token_len, dna_reverse_tokens_len], dim=1)
        # protein input
        protein_seq = data[protein_chain_name]['feat']['seq']
        protein_input_ids = torch.tensor(self.protein_tokenizer.batch_encode_plus([protein_seq], add_special_tokens=False)['input_ids'])
        protein_input_attention_mask = torch.tensor(self.protein_tokenizer.batch_encode_plus([protein_seq], add_special_tokens=False)['attention_mask'])

        batch = {}
        batch['dna_token_input_ids'] = dna_token_input_ids[0]
        batch['dna_reverse_token_input_ids'] = dna_reverse_token_input_ids[0]
        batch['dna_token_type_ids'] = torch.ones_like(dna_token_input_ids[0])

        batch['dna_token_attention_mask'] = dna_token_attention_mask[0]
        batch['dna_token_reverse_attention_mask'] = dna_token_reverse_attention_mask[0]
        batch['dna_reverse_token_type_ids'] = torch.ones_like(dna_reverse_token_input_ids[0])

        batch['weight_mask'] = weight_mask[0]
        batch['reverse_weight_mask'] = reverse_weight_mask[0]
        batch['post_token_length'] = post_token_length[0]
        batch['reverse_post_token_length'] = reverse_post_token_length[0]

        # protein
        batch['protein_input_ids'] = protein_input_ids[0]
        batch['protein_input_attention_mask'] = protein_input_attention_mask[0]
        batch['protein_token_type_ids'] = torch.zeros_like(protein_input_ids[0])
        ## label

        batch['pro_atom_pos'] = torch.tensor(data[protein_chain_name]['label']['all_atom_positions'])
        batch['pro_atom_ref_pos'] = torch.tensor(data[protein_chain_name]['feat']['ref_pos'])
        batch['pro_atom_ref_type'] = torch.tensor(data[protein_chain_name]['feat']['ref_element'])
        batch['pro_atom_mask'] = torch.tensor(data[protein_chain_name]['label']['all_atom_mask'])
        batch['pro_atom_ref_aatype'] = torch.tensor(data[protein_chain_name]['feat']['ref_element'])
        batch['pro_seq_mask'] = torch.tensor(data[protein_chain_name]['feat']['seq_mask'])
        batch['pro_residue_atom_lens'] = torch.tensor(data[protein_chain_name]['label']['all_token_to_atom_nums'])
        batch['pro_residue_indices'] = torch.tensor(data[protein_chain_name]['feat']['residue_index'])
        batch['pro_pseudo_beta'] = torch.tensor(data[protein_chain_name]['label']['pseudo_beta'])
        batch['pro_pseudo_beta_mask'] = torch.tensor(data[protein_chain_name]['label']['pseudo_beta_mask'])
        batch['pro_restype'] = torch.tensor(data[protein_chain_name]['feat']['restype'])
        batch['pro_ref_token2atom_idx'] = torch.tensor(data[protein_chain_name]['feat']['ref_token2atom_idx'])
        # batch['pro_label_atom_ids'] = torch.tensor(data[protein_chain_name]['label']['label_atom_ids'])
        mapped_atom_ids = []
        for atom in data[protein_chain_name]['label']['label_atom_ids']:
            if atom_order.get(atom) is None:
                print(pro_dna_chain_key, name)
            #     random_idx = random.randint(0,len(self.protein_chain_info)-1)
            #     return self.__getitem__(random_idx)
            mapped_atom_ids.append(atom_order.get(atom, 0)) # temp
        # mapped_atom_ids = np.vectorize(atom_order.get)(data[protein_chain_name]['label']['label_atom_ids'])
        batch['pro_label_atom_ids'] = torch.tensor(mapped_atom_ids)

        batch['dna_atom_pos'] = torch.tensor(data[dna_chain_name]['label']['all_atom_positions'])
        batch['dna_atom_ref_pos'] = torch.tensor(data[dna_chain_name]['feat']['ref_pos'])
        batch['dna_atom_ref_type'] = torch.tensor(data[dna_chain_name]['feat']['ref_element'])
        batch['dna_atom_mask'] = torch.tensor(data[dna_chain_name]['label']['all_atom_mask'])
        batch['dna_atom_ref_aatype'] = torch.tensor(data[dna_chain_name]['feat']['ref_element'])
        batch['dna_seq_mask'] = torch.tensor(data[dna_chain_name]['feat']['seq_mask'])
        batch['dna_residue_atom_lens'] = torch.tensor(data[dna_chain_name]['label']['all_token_to_atom_nums'])
        batch['dna_residue_indices'] = torch.tensor(data[dna_chain_name]['feat']['residue_index'])
        batch['dna_pseudo_beta'] = torch.tensor(data[dna_chain_name]['label']['pseudo_beta'])
        batch['dna_pseudo_beta_mask'] = torch.tensor(data[dna_chain_name]['label']['pseudo_beta_mask'])
        batch['dna_restype'] = torch.tensor(data[dna_chain_name]['feat']['restype'])
        batch['dna_ref_token2atom_idx'] = torch.tensor(data[dna_chain_name]['feat']['ref_token2atom_idx'])
        # batch['dna_label_atom_ids'] = torch.tensor(data[dna_chain_name]['label']['label_atom_ids'])

        batch['dna_reverse_atom_pos'] = torch.tensor(data[dna_reverse_chain_name]['label']['all_atom_positions'])
        batch['dna_reverse_atom_ref_pos'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_pos'])
        batch['dna_reverse_atom_ref_type'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_element'])
        batch['dna_reverse_atom_mask'] = torch.tensor(data[dna_reverse_chain_name]['label']['all_atom_mask'])
        batch['dna_reverse_atom_ref_aatype'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_element'])
        batch['dna_reverse_seq_mask'] = torch.tensor(data[dna_reverse_chain_name]['feat']['seq_mask'])
        batch['dna_reverse_residue_atom_lens'] = torch.tensor(data[dna_reverse_chain_name]['label']['all_token_to_atom_nums'])
        batch['dna_reverse_residue_indices'] = torch.tensor(data[dna_reverse_chain_name]['feat']['residue_index'])
        batch['dna_reverse_pseudo_beta'] = torch.tensor(data[dna_reverse_chain_name]['label']['pseudo_beta'])
        batch['dna_reverse_pseudo_beta_mask'] = torch.tensor(data[dna_reverse_chain_name]['label']['pseudo_beta_mask'])
        batch['dna_reverse_restype'] = torch.tensor(data[dna_reverse_chain_name]['feat']['restype'])
        batch['dna_reverse_ref_token2atom_idx'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_token2atom_idx'])
        # batch['dna_reverse_label_atom_ids'] = torch.tensor(data[dna_reverse_chain_name]['label']['label_atom_ids'])

        # concat protein + dna + dna_reverse
        batch['atom_pos'] = torch.cat([batch['pro_atom_pos'], batch['dna_atom_pos'], batch['dna_reverse_atom_pos']], dim=0)
        batch['atom_ref_pos'] = torch.cat([batch['pro_atom_ref_pos'], batch['dna_atom_ref_pos'], batch['dna_reverse_atom_ref_pos']], dim=0)
        batch['atom_ref_type'] = torch.cat([batch['pro_atom_ref_type'], batch['dna_atom_ref_type'], batch['dna_reverse_atom_ref_type']], dim=0)
        batch['atom_mask'] = torch.cat([batch['pro_atom_mask'], batch['dna_atom_mask'], batch['dna_reverse_atom_mask']], dim=0)
        batch['atom_ref_aatype'] = torch.cat([batch['pro_atom_ref_aatype'], batch['dna_atom_ref_aatype'], batch['dna_reverse_atom_ref_aatype']], dim=0)
        batch['seq_mask'] = torch.cat([batch['pro_seq_mask'], batch['dna_seq_mask'], batch['dna_reverse_seq_mask']], dim=0)
        batch['pseudo_beta'] = torch.cat([batch['pro_pseudo_beta'], batch['dna_pseudo_beta'], batch['dna_reverse_pseudo_beta']], dim=0)
        batch['pseudo_beta_mask'] = torch.cat([batch['pro_pseudo_beta_mask'], batch['dna_pseudo_beta_mask'], batch['dna_reverse_pseudo_beta_mask']], dim=0)
        batch['residue_atom_lens'] = torch.cat([batch['pro_residue_atom_lens'], batch['dna_residue_atom_lens'], batch['dna_reverse_residue_atom_lens']], dim=0)

        pro_residue_indices = batch['pro_residue_indices']
        dna_residue_indices = batch['dna_residue_indices']
        dna_reverse_residue_indices = batch['dna_reverse_residue_indices']

        max_pro = pro_residue_indices.max() + 1  
        max_dna = max_pro + dna_residue_indices.max() + 1
        dna_residue_indices = dna_residue_indices + max_pro
        dna_reverse_residue_indices = dna_reverse_residue_indices + max_dna

        batch['residue_indices'] = torch.cat([pro_residue_indices, dna_residue_indices, dna_reverse_residue_indices], dim=0)
        batch['restype'] = torch.cat([batch['pro_restype'], batch['dna_restype'], batch['dna_reverse_restype']], dim=0)
        batch['ref_token2atom_idx'] = torch.cat([batch['pro_ref_token2atom_idx'], batch['dna_ref_token2atom_idx'], batch['dna_reverse_ref_token2atom_idx']], dim=0)
        # batch['label_atom_id'] = torch.cat([batch['pro_label_atom_id'], batch['dna_label_atom_id'], batch['dna_reverse_label_atom_id']], dim=0)
        batch['label_atom_ids'] = batch['pro_label_atom_ids']
        batch['pro_token_len'] = torch.tensor(len(batch['pro_restype']))
        batch['pro_atom_len'] = torch.tensor(len(batch['pro_label_atom_ids']))
        # if batch['seq_mask'].shape[0] != batch['weight_mask'].shape[0] + batch['reverse_weight_mask'].shape[0] + len(protein_seq):
        #     print(name)
        batch['file_id'] = pro_dna_chain_key
        batch['id'] = idx
        if len(batch['restype']) > 648:
            print(name, len(batch['restype']))
            random_idx = random.randint(0,len(self.protein_chain_info)-1)
            return self.__getitem__(random_idx)


        return batch
    
    def process(self, data_info):
        pass
 
    def crop(self, total_feats, complex_label_dict):
        pass
    
    def collate_fn(self, batch):
        pass



class ProtDNADataset_single(Dataset):
    def __init__(self, data_json_path, mmcif_dir, feats_dir, dna_tokenizer=None, protein_tokenizer=None):
        
        self.protein_chain_info = json.load(open(data_json_path, 'r'))
        self.mmcif_dir = mmcif_dir
        self.feats_dir = feats_dir

        self._chain_ids = list(self.protein_chain_info.keys())
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }
        self.dna_tokenizer = dna_tokenizer
        self.protein_tokenizer = protein_tokenizer

        
    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]
    
    def __len__(self):
        return len(self.protein_chain_info)

    def __getitem__(self, idx):

        pro_dna_chain_key = self.idx_to_chain_id(idx)
        
        pro_dna_chain_info =  self.protein_chain_info[pro_dna_chain_key]
        name = pro_dna_chain_key.split('_')[0]
        protein_chain = pro_dna_chain_info['protein_chain_id']
        dna_chain = pro_dna_chain_info['dna_chain']
        dna_reverse_chain = pro_dna_chain_info['dna_reverse_chain']

        protein_chain_name = f"{name}_protein_{protein_chain}"
        dna_chain_name = f"{name}_dna_{dna_chain}"
        dna_reverse_chain_name = f"{name}_dna_{dna_reverse_chain}"


        temp_data_path = os.path.join(self.feats_dir, f'{name}.npy')
        data = np.load(temp_data_path, allow_pickle=True).item()
        # dna input
        dna_seq = data[dna_chain_name]['feat']['seq'].replace('X', 'N')
        dna_seq_reverse = data[dna_reverse_chain_name]['feat']['seq'].replace('X', 'N')
        dna_tokens = self.dna_tokenizer.tokenize(dna_seq)
        dna_tokens_len = torch.tensor([[len(token) for token in dna_tokens]])
        dna_token_input_ids = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq])['input_ids']) # .to(torch.device("cuda:0" ))
        dna_token_attention_mask = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq])['attention_mask'])

        dna_reverse_tokens = self.dna_tokenizer.tokenize(dna_seq_reverse)
        dna_reverse_tokens_len = torch.tensor([[len(token) for token in dna_reverse_tokens]])
        dna_reverse_token_input_ids = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq_reverse])['input_ids']) # .to(torch.device("cuda:0" ))
        dna_token_reverse_attention_mask = torch.tensor(self.dna_tokenizer.batch_encode_plus([dna_seq_reverse])['attention_mask'])

        weight_mask = torch.ones(dna_token_input_ids.shape[0], len(dna_seq))
        reverse_weight_mask = torch.ones(dna_reverse_token_input_ids.shape[0], len(dna_seq_reverse))
        cls_token_len = torch.zeros(dna_token_input_ids.shape[0],1)
        post_token_length = torch.cat([cls_token_len, dna_tokens_len], dim=1)
        reverse_post_token_length = torch.cat([cls_token_len, dna_reverse_tokens_len], dim=1)
        # protein input
        protein_seq = data[protein_chain_name]['feat']['seq']
        protein_input_ids = torch.tensor(self.protein_tokenizer.batch_encode_plus([protein_seq])['input_ids'])
        protein_input_attention_mask = torch.tensor(self.protein_tokenizer.batch_encode_plus([protein_seq])['attention_mask'])

        batch = {}
        batch['dna_token_input_ids'] = dna_token_input_ids[0]
        batch['dna_reverse_token_input_ids'] = dna_reverse_token_input_ids[0]
        batch['dna_token_type_ids'] = torch.ones_like(dna_token_input_ids[0])

        batch['dna_token_attention_mask'] = dna_token_attention_mask[0]
        batch['dna_token_reverse_attention_mask'] = dna_token_reverse_attention_mask[0]
        batch['dna_reverse_token_type_ids'] = torch.ones_like(dna_reverse_token_input_ids[0])

        batch['weight_mask'] = weight_mask[0]
        batch['reverse_weight_mask'] = reverse_weight_mask[0]
        batch['post_token_length'] = post_token_length[0]
        batch['reverse_post_token_length'] = reverse_post_token_length[0]

        # protein
        batch['protein_input_ids'] = protein_input_ids[0]
        batch['protein_input_attention_mask'] = protein_input_attention_mask[0]
        batch['protein_token_type_ids'] = torch.zeros_like(protein_input_ids[0])
        ## label

        batch['pro_atom_pos'] = torch.tensor(data[protein_chain_name]['label']['all_atom_positions'])
        batch['pro_atom_ref_pos'] = torch.tensor(data[protein_chain_name]['feat']['ref_pos'])
        batch['pro_atom_ref_type'] = torch.tensor(data[protein_chain_name]['feat']['ref_element'])
        batch['pro_atom_mask'] = torch.tensor(data[protein_chain_name]['label']['all_atom_mask'])
        batch['pro_atom_ref_aatype'] = torch.tensor(data[protein_chain_name]['feat']['ref_element'])
        batch['pro_seq_mask'] = torch.tensor(data[protein_chain_name]['feat']['seq_mask'])
        batch['pro_residue_atom_lens'] = torch.tensor(data[protein_chain_name]['label']['all_token_to_atom_nums'])
        batch['pro_residue_indices'] = torch.tensor(data[protein_chain_name]['feat']['residue_index'])
        batch['pro_pseudo_beta'] = torch.tensor(data[protein_chain_name]['label']['pseudo_beta'])
        batch['pro_pseudo_beta_mask'] = torch.tensor(data[protein_chain_name]['label']['pseudo_beta_mask'])
        batch['pro_restype'] = torch.tensor(data[protein_chain_name]['feat']['restype'])
        batch['pro_ref_token2atom_idx'] = torch.tensor(data[protein_chain_name]['feat']['ref_token2atom_idx'])
        # batch['pro_label_atom_ids'] = torch.tensor(data[protein_chain_name]['label']['label_atom_ids'])
        mapped_atom_ids = []
        for atom in data[protein_chain_name]['label']['label_atom_ids']:
            if atom_order.get(atom) is None:
                print(pro_dna_chain_key, name)
            #     random_idx = random.randint(0,len(self.protein_chain_info)-1)
            #     return self.__getitem__(random_idx)
            mapped_atom_ids.append(atom_order.get(atom, 0)) # temp
        # mapped_atom_ids = np.vectorize(atom_order.get)(data[protein_chain_name]['label']['label_atom_ids'])
        batch['pro_label_atom_ids'] = torch.tensor(mapped_atom_ids)

        batch['dna_atom_pos'] = torch.tensor(data[dna_chain_name]['label']['all_atom_positions'])
        batch['dna_atom_ref_pos'] = torch.tensor(data[dna_chain_name]['feat']['ref_pos'])
        batch['dna_atom_ref_type'] = torch.tensor(data[dna_chain_name]['feat']['ref_element'])
        batch['dna_atom_mask'] = torch.tensor(data[dna_chain_name]['label']['all_atom_mask'])
        batch['dna_atom_ref_aatype'] = torch.tensor(data[dna_chain_name]['feat']['ref_element'])
        batch['dna_seq_mask'] = torch.tensor(data[dna_chain_name]['feat']['seq_mask'])
        batch['dna_residue_atom_lens'] = torch.tensor(data[dna_chain_name]['label']['all_token_to_atom_nums'])
        batch['dna_residue_indices'] = torch.tensor(data[dna_chain_name]['feat']['residue_index'])
        batch['dna_pseudo_beta'] = torch.tensor(data[dna_chain_name]['label']['pseudo_beta'])
        batch['dna_pseudo_beta_mask'] = torch.tensor(data[dna_chain_name]['label']['pseudo_beta_mask'])
        batch['dna_restype'] = torch.tensor(data[dna_chain_name]['feat']['restype'])
        batch['dna_ref_token2atom_idx'] = torch.tensor(data[dna_chain_name]['feat']['ref_token2atom_idx'])
        # batch['dna_label_atom_ids'] = torch.tensor(data[dna_chain_name]['label']['label_atom_ids'])

        batch['dna_reverse_atom_pos'] = torch.tensor(data[dna_reverse_chain_name]['label']['all_atom_positions'])
        batch['dna_reverse_atom_ref_pos'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_pos'])
        batch['dna_reverse_atom_ref_type'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_element'])
        batch['dna_reverse_atom_mask'] = torch.tensor(data[dna_reverse_chain_name]['label']['all_atom_mask'])
        batch['dna_reverse_atom_ref_aatype'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_element'])
        batch['dna_reverse_seq_mask'] = torch.tensor(data[dna_reverse_chain_name]['feat']['seq_mask'])
        batch['dna_reverse_residue_atom_lens'] = torch.tensor(data[dna_reverse_chain_name]['label']['all_token_to_atom_nums'])
        batch['dna_reverse_residue_indices'] = torch.tensor(data[dna_reverse_chain_name]['feat']['residue_index'])
        batch['dna_reverse_pseudo_beta'] = torch.tensor(data[dna_reverse_chain_name]['label']['pseudo_beta'])
        batch['dna_reverse_pseudo_beta_mask'] = torch.tensor(data[dna_reverse_chain_name]['label']['pseudo_beta_mask'])
        batch['dna_reverse_restype'] = torch.tensor(data[dna_reverse_chain_name]['feat']['restype'])
        batch['dna_reverse_ref_token2atom_idx'] = torch.tensor(data[dna_reverse_chain_name]['feat']['ref_token2atom_idx'])
        # batch['dna_reverse_label_atom_ids'] = torch.tensor(data[dna_reverse_chain_name]['label']['label_atom_ids'])

        # concat protein + dna + dna_reverse
        batch['atom_pos'] = torch.cat([batch['pro_atom_pos'], batch['dna_atom_pos'], batch['dna_reverse_atom_pos']], dim=0)
        batch['atom_ref_pos'] = torch.cat([batch['pro_atom_ref_pos'], batch['dna_atom_ref_pos'], batch['dna_reverse_atom_ref_pos']], dim=0)
        batch['atom_ref_type'] = torch.cat([batch['pro_atom_ref_type'], batch['dna_atom_ref_type'], batch['dna_reverse_atom_ref_type']], dim=0)
        batch['atom_mask'] = torch.cat([batch['pro_atom_mask'], batch['dna_atom_mask'], batch['dna_reverse_atom_mask']], dim=0)
        batch['atom_ref_aatype'] = torch.cat([batch['pro_atom_ref_aatype'], batch['dna_atom_ref_aatype'], batch['dna_reverse_atom_ref_aatype']], dim=0)
        batch['seq_mask'] = torch.cat([batch['pro_seq_mask'], batch['dna_seq_mask'], batch['dna_reverse_seq_mask']], dim=0)
        batch['pseudo_beta'] = torch.cat([batch['pro_pseudo_beta'], batch['dna_pseudo_beta'], batch['dna_reverse_pseudo_beta']], dim=0)
        batch['pseudo_beta_mask'] = torch.cat([batch['pro_pseudo_beta_mask'], batch['dna_pseudo_beta_mask'], batch['dna_reverse_pseudo_beta_mask']], dim=0)
        batch['residue_atom_lens'] = torch.cat([batch['pro_residue_atom_lens'], batch['dna_residue_atom_lens'], batch['dna_reverse_residue_atom_lens']], dim=0)

        pro_residue_indices = batch['pro_residue_indices']
        dna_residue_indices = batch['dna_residue_indices']
        dna_reverse_residue_indices = batch['dna_reverse_residue_indices']

        max_pro = pro_residue_indices.max() + 1  
        max_dna = max_pro + dna_residue_indices.max() + 1
        dna_residue_indices = dna_residue_indices + max_pro
        dna_reverse_residue_indices = dna_reverse_residue_indices + max_dna

        batch['residue_indices'] = torch.cat([pro_residue_indices, dna_residue_indices, dna_reverse_residue_indices], dim=0)
        batch['restype'] = torch.cat([batch['pro_restype'], batch['dna_restype'], batch['dna_reverse_restype']], dim=0)
        batch['ref_token2atom_idx'] = torch.cat([batch['pro_ref_token2atom_idx'], batch['dna_ref_token2atom_idx'], batch['dna_reverse_ref_token2atom_idx']], dim=0)
        # batch['label_atom_id'] = torch.cat([batch['pro_label_atom_id'], batch['dna_label_atom_id'], batch['dna_reverse_label_atom_id']], dim=0)
        batch['label_atom_ids'] = batch['pro_label_atom_ids']
        batch['pro_token_len'] = torch.tensor(len(batch['pro_restype']))
        batch['pro_atom_len'] = torch.tensor(len(batch['pro_label_atom_ids']))
        # if batch['seq_mask'].shape[0] != batch['weight_mask'].shape[0] + batch['reverse_weight_mask'].shape[0] + len(protein_seq):
        #     print(name)
        batch['file_id'] = pro_dna_chain_key
        batch['id'] = idx
        if len(batch['restype']) > 648:
            print(name, len(batch['restype']))
            random_idx = random.randint(0,len(self.protein_chain_info)-1)
            return self.__getitem__(random_idx)


        return batch
    
    def process(self, data_info):
        pass
 
    def crop(self, total_feats, complex_label_dict):
        pass
    
    def collate_fn(self, batch):
        pass




