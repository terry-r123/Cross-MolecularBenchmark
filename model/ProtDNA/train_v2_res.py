import argparse
import os
import csv
import sys
sys.path.insert(0, './')
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import numpy as np
import time
import random
from einops import rearrange, repeat, reduce, einsum, pack, unpack
from safetensors.torch import load_file
import pdb as depdb

import transformers
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import shutil
import io
import tempfile

import numpy as np
import pandas as pd
from config import prodnafold_config

from common.np import protein
# from nt.nt_v2_100m_multi.modeling_esm import EsmModel, DNALmForNucleotideLevel
from net.demo import ProteinClassifier, DNAClassifier
# from esm2.esm2_t30_150M_UR50D import 


from model_dis_resnet import  EDM, ProDNAFold
from transformers import AutoModel, AutoModelForTokenClassification
from evaluation_utils.eval_protein import eval_pdb_file
# from openfold.utils.validation_metrics import KabschRMSD, KabschLigRMSD, compute_metrics
from openfold.utils.rmsd_utils import  Ligand_KabschRMSD, Protein_KabschRMSD, KabschRMSD

from prodna_dataset_v2_res import ProtDNADataset

def exists(val):
    return val is not None

      
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    
    name = (args.experiment_name +
        f'_Diffusion_'
        f'lr{args.lr}_'
        f'bs{args.batch_size}_'
        f'dbs{args.diffusion_batch_size}_'
        f'epo{args.num_train_epochs}_'
        f'warm{args.warmup_steps}epoch_'
        f'backbone_blocks{args.num_blocks_back}_'
        f'enc_blocks{args.num_blocks_enc}_'
        f'dec_blocks{args.num_blocks_dec}_'
        f'dit_blocks{args.num_blocks_dit}_'
        f'train_crop_size{args.train_crop_size}_'
    )
    print(name)
    # modify the args.output_dir with name
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, name)
    
    # fix the random seed
    args.seed = args.seed
    set_seed(args)
     

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers, log_with='wandb', step_scheduler_with_optimizer=False)

    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # 0. load config
    config = prodnafold_config
    
    config.globals.c_s = 256
    config.data.batch_size = 1
    config.data.num_workers = 4
    config.data.prefetch_factor = 1

    config.data.train.crop_size = 256
    config.data.eval.crop_size = None

    config.diffusion.num_blocks_dec = args.num_blocks_dec
    config.diffusion.num_blocks_enc = args.num_blocks_enc
    config.diffusion.num_blocks_dit = args.num_blocks_dit
    # loss
    # config.loss.dihedral.weight = 0.0
    # config.loss.distogram.weight = 0.5
    # config.loss.masked_msa.weight = 0.5

    generator = torch.Generator(device=accelerator.device)

    try:
        generator.manual_seed(args.seed + int(os.environ.get('RANK')))
    except:
        generator.manual_seed(args.seed)

    # 1. load dataset and dataloader
    #    dataset
    # dna_tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.dna_model_name_or_path,
    #     model_max_length=256,
    #     padding_side="right",
    #     use_fast=True,
    #     trust_remote_code=True,
    # )
    # protein_tokenizer =  transformers.AutoTokenizer.from_pretrained(args.protein_model_name_or_path)
    # 2. load model
    
    # dna_backbone = DNALmForNucleotideLevel.from_pretrained(
    #     args.dna_model_name_or_path,
    #     num_labels=256,
    #     tokenizer=dna_tokenizer)
    # dna_backbone = dna_backbone.to(accelerator.device)


    # protein_backbone = AutoModelForTokenClassification.from_pretrained(args.protein_model_name_or_path,
    #                                                                 num_labels=256)
    # protein_backbone = protein_backbone.to(accelerator.device)
    # navie net tokenizr
    
    dna_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./ProtDNA/net/dna_tokenizer",
        model_max_length=512,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    protein_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./ProtDNA/net/protein_tokenizer",
        model_max_length=648,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    dna_backbone = DNAClassifier(model_type=args.model_type, output_dim=256, vocab_size=dna_tokenizer.vocab_size)
    protein_backbone = ProteinClassifier(model_type=args.model_type, output_dim=256, vocab_size=protein_tokenizer.vocab_size)

    model = ProDNAFold(config)
    
    model = model.to(accelerator.device)
    no_decay = ["bias", "LayerNorm.weight"]

    # model_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # dna_backbone_named_params = [(n, p) for n, p in dna_backbone.named_parameters() if p.requires_grad]
    # protein_backbone_named_params = [(n, p) for n, p in protein_backbone.named_parameters() if p.requires_grad]

    model_named_params = [(n, p) for n, p in model.named_parameters()]
    dna_backbone_named_params = [(n, p) for n, p in dna_backbone.named_parameters()]
    protein_backbone_named_params = [(n, p) for n, p in protein_backbone.named_parameters()]

    dna_backbone_params = count_parameters(dna_backbone)
    protein_backbone_params = count_parameters(protein_backbone)
    model_params = count_parameters(model)

    print(f"DNA Backbone param: {dna_backbone_params}")
    print(f"Protein Backbone param: {protein_backbone_params}")
    print(f"diffusion: {model_params}")
    total_params = dna_backbone_params + protein_backbone_params + model_params
    print(f"total: {total_params}")

    model_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
    ]

    dna_backbone_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in dna_backbone_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in dna_backbone_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr_backbone,
        },
    ]
    protein_backbone_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in protein_backbone_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in protein_backbone_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr_backbone, 
        },
    ]

    optimizer_grouped_parameters = (
        dna_backbone_optimizer_grouped_parameters +
        protein_backbone_optimizer_grouped_parameters +
        model_optimizer_grouped_parameters
    )

    optimizer = optim.AdamW(optimizer_grouped_parameters, betas=(args.beta1, args.beta2))


    base_path = args.base_path
    train_data_json_path = os.path.join(base_path, "final", args.train_data_dir)
    val_data_json_path = os.path.join(base_path, "final", args.val_data_dir)
    test_data_json_path = os.path.join(base_path, "final", args.test_data_dir)
    mmcif_dir = os.path.join(base_path, "output_v2")
    feats_dir = os.path.join(base_path, "output_v2")


    
    train_dataset = ProtDNADataset(train_data_json_path, mmcif_dir, feats_dir, dna_tokenizer, protein_tokenizer)
    val_dataset = ProtDNADataset(val_data_json_path, mmcif_dir, feats_dir, dna_tokenizer, protein_tokenizer)
    test_dataset = ProtDNADataset(test_data_json_path, mmcif_dir, feats_dir, dna_tokenizer, protein_tokenizer)
    
    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    per_steps_one_epoch = len(train_dataset) // accelerator.num_processes
    num_warmup_steps = args.warmup_epochs * per_steps_one_epoch
    num_training_steps = args.num_train_epochs * per_steps_one_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    dna_backbone, protein_backbone, model, optimizer, train_loader,  scheduler= accelerator.prepare(dna_backbone, protein_backbone, model, optimizer, train_loader, scheduler)

    start_epoch = 0

    if accelerator.is_main_process:
        if args.wandb:
            wandb.init(project='ProtDNA', id=args.wandb_id)
            wandb.run.name = name
            wandb.run.save()
            wandb.watch(model)
            print(name)
    
    steps = (start_epoch) * per_steps_one_epoch

    best_metric = None
    current_output_dir = None
    train_loss_dict = {}
    best_lddt_ca = -float('inf')
    best_epoch = 0
    for epoch in range(args.num_train_epochs): 
        train_loss_dict['diffusion'] = []
        train_loss_dict['dis_loss'] = []
        accelerator.wait_for_everyone()
        model.train()
        dna_backbone.train()
        protein_backbone.train()

        for batch in tqdm(train_loader, disable= not accelerator.is_main_process, desc=f"Training Epoch {epoch}"):

            steps += 1
            dna_token_input_ids = batch['dna_token_input_ids']
            dna_token_attention_mask = batch['dna_token_attention_mask']
            weight_mask = batch['weight_mask']
            post_token_length = batch['post_token_length']
            dna_reverse_token_input_ids = batch['dna_reverse_token_input_ids']
            dna_token_reverse_attention_mask = batch['dna_token_reverse_attention_mask']
            reverse_weight_mask = batch['reverse_weight_mask']
            reverse_post_token_length = batch['reverse_post_token_length']
            
            # .module
            
            dna_feats = dna_backbone(input_ids_1=dna_token_input_ids,mask_1=dna_token_attention_mask,input_ids_2=dna_reverse_token_input_ids,mask_2=dna_token_reverse_attention_mask
                                    )

            protein_input_ids = batch['protein_input_ids']
            protein_input_attention_mask = batch['protein_input_attention_mask']
            protein_feats = protein_backbone(protein_input_ids, protein_input_attention_mask)
            
            atom_info = {}
            
            atom_info['atom_pos'] = batch['atom_pos'].repeat(args.diffusion_batch_size,1,1)
            atom_info['atom_ref_pos'] = batch['atom_ref_pos'].repeat(args.diffusion_batch_size,1,1)
            atom_info['atom_ref_type'] = batch['atom_ref_type'].repeat(args.diffusion_batch_size,1)
            atom_info['atom_mask'] = batch['atom_mask'].repeat(args.diffusion_batch_size,1)
            atom_info['atom_ref_aatype'] = batch['atom_ref_aatype'].repeat(args.diffusion_batch_size,1)
            atom_info['seq_mask'] = batch['seq_mask'].repeat(args.diffusion_batch_size,1)
            atom_info['residue_atom_lens'] = batch['residue_atom_lens'].repeat(args.diffusion_batch_size,1)
            atom_info['residue_indices'] = batch['residue_indices']
            atom_info['pseudo_beta_mask'] = batch['pseudo_beta_mask']
            atom_info['pseudo_beta'] = batch['pseudo_beta']

                
            loss, loss_dict = model(atom_info, dna_feats, protein_feats, diffusion_batch_size=args.diffusion_batch_size)
            
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # if accelerator.sync_gradients:
            #     dna_backbone_params = [p for p in dna_backbone.parameters() if p.requires_grad]
            #     protein_backbone_params = [p for p in protein_backbone.parameters() if p.requires_grad]
            #     model_params = [p for p in model.parameters() if p.requires_grad]
            #     all_trainable_params = dna_backbone_params + protein_backbone_params + model_params

            #     accelerator.clip_grad_norm_(all_trainable_params, args.max_grad_norm)
                     
            train_loss_dict['diffusion'].append(loss_dict['diffusion'].cpu().detach().numpy())
            train_loss_dict['dis_loss'].append(loss_dict['dis_loss'].cpu().detach().numpy())
            
            accelerator.wait_for_everyone() 

        accelerator.wait_for_everyone()
        if epoch % 10 == 0 and accelerator.is_main_process:
            if args.output_dir is not None:
                current_output_dir = os.path.join(args.output_dir, f'step_epoch_{epoch}')
                if not os.path.exists(current_output_dir):
                    os.makedirs(current_output_dir)
                accelerator.save_state(current_output_dir)
                print(f'save_state to {current_output_dir}')
        print(f"eval over, {accelerator.device}")
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
         



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DATA DIRECTORIES
    parser.add_argument("--base_path", type=str, help="Directory containing data")
    parser.add_argument("--train_data_dir", type=str, help="Directory containing training pdb files")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Directory containing validation pdb files")
    parser.add_argument("--test_data_dir", type=str, default=None, help="Directory containing validation pdb files")
    parser.add_argument("--model_type", type=str, default=None, help="Directory containing validation pdb files")
    # HYPERPARAMETERS

    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--lr_backbone', type=float, default=3e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight Decay")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2")
    parser.add_argument('--warmup_steps', type=int, default=10, help="Number of warmup epochs")
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Number of warmup epochs")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--diffusion_batch_size', type=int, default=8, help="Diffusion Batch size")
    parser.add_argument('--seed', type=int, default=3407, help="Random seed")
    parser.add_argument('--num_blocks_back', type=int, default=48, help="Number of backbone blocks")
    parser.add_argument('--num_blocks_enc', type=int, default=3, help="Number of encoder blocks")
    parser.add_argument('--num_blocks_dec', type=int, default=24, help="Number of decoder blocks")
    parser.add_argument('--num_blocks_dit', type=int, default=3, help="Number of dit blocks")
    parser.add_argument('--train_crop_size', type=int, default=None, help="maximum length of residue sequences")
    parser.add_argument('--num_workers', type=int, default=0, help="maximum length of residue sequences")

    # TRAINING SETTINGS
    parser.add_argument(
        "--wandb_id", type=str, default=None,
        help="ID of a previous run to be resumed"
    )
    parser.add_argument("--checkpoint_every_epoch", action="store_true", default=False, help="Whether to checkpoint at the end of every training epoch")
    parser.add_argument("--wandb", action="store_true", default=False, help="Whether to log metrics to Weights & Biases")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the current experiment for wandb logging")

    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Maximum gradient norm for gradient clipping")

    # OUTPUT SETTINGS
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model checkpoints")

    # pro and dna model
    parser.add_argument("--dna_model_name_or_path", 
                        type=str,
                        default="./ProtDNA/nt/nt_v2_100m_multi",
                        help="Directory containing dna pretrained model")
    parser.add_argument("--protein_model_name_or_path", 
                        type=str,
                        default="./ProtDNA/esm2/esm2_t30_150M_UR50D",
                        help="Directory containing protein pretrained model")
    args = parser.parse_args()

    main(args)
