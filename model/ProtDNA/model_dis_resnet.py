from functools import partial
from pickle import NONE
from turtle import forward
import argparse
import os
import csv
import sys
import time
from sympy import true
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

import pdb as depdb
from typing import Tuple, Optional


from diffusion_module.diffusion_module_opt import DiffusionModule,repeat_consecutive_with_lens

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def add(m1, m2, inplace):
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


# def convert_ids_to_atoms(ids):
#     reverse_atom_encoding = {}
#     for key, value in atom_encoding.items():
#         if value not in reverse_atom_encoding:
#             reverse_atom_encoding[value] = [key]
#         else:
#             reverse_atom_encoding[value].append(key)

#     atom_symbols = [reverse_atom_encoding.get(id, ['None'])[0] for id in ids]
    
#     return atom_symbols

def exists(val):
    return val is not None

def unique_with_indices(lst):
    seen = {}
    unique_lst = []
    indices = []

    for idx, item in enumerate(lst):
        if item not in seen:
            seen[item] = len(unique_lst)  # Store the index in the unique list
            unique_lst.append(item)
            indices.append(idx)

    return unique_lst, indices
    
class WeightedRigidAlign(nn.Module):
    """ Algorithm 28 """
    def forward(
        self,
        pred_coords,       # predicted coordinates
        true_coords,       # true coordinates
        weights,             # weights for each atom
        mask = None    # mask for variable lengths
    ):

        batch_size, num_points, dim = pred_coords.shape

        if exists(mask):
            # zero out all predicted and true coordinates where not an atom

            pred_coords = torch.where(mask.unsqueeze(-1), pred_coords, 0.)
            true_coords = torch.where(mask.unsqueeze(-1), true_coords, 0.)
            weights = torch.where(mask, weights, 0.)

        # Take care of weights broadcasting for coordinate dimension
        weights = rearrange(weights, 'b n -> b n 1')

        # Compute weighted centroids
        true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)

        # Center the coordinates
        true_coords_centered = true_coords - true_centroid
        pred_coords_centered = pred_coords - pred_centroid

        if num_points < (dim + 1):
            print(
                "Warning: The size of one of the point clouds is <= dim+1. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        # Compute the weighted covariance matrix
        cov_matrix = einsum(weights * true_coords_centered, pred_coords_centered, 'b n i, b n j -> b i j')

        # Compute the SVD of the covariance matrix
        U, S, V = torch.svd(cov_matrix.float())
        U_T = U.transpose(-2, -1)

        # Catch ambiguous rotation by checking the magnitude of singular values
        if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
            print(
                "Warning: Excessively low rank of "
                + "cross-correlation between aligned point clouds. "
                + "`WeightedRigidAlign` cannot return a unique rotation."
            )

        det = torch.det(einsum(V, U_T, 'b i j, b j k -> b i k').float())
        # Ensure proper rotation matrix with determinant 1
        diag = torch.eye(dim, dtype=det.dtype, device=det.device)[None].repeat(batch_size, 1, 1)
        diag[:, -1, -1] = det
        rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

        # Apply the rotation and translation
        # af3
        true_aligned_coords = einsum(rot_matrix, true_coords_centered, 'b i j, b n j -> b n i') + true_centroid
        # our
        # true_aligned_coords = einsum(rot_matrix, true_coords_centered, 'b i j, b n j -> b n i') + pred_centroid
        true_aligned_coords.detach_()

        return true_aligned_coords

class CentreRandomAugmentation(nn.Module):
    """ Algorithm 19 """

    def __init__(self, trans_scale = 1.0):
        super().__init__()
        self.trans_scale = trans_scale
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def device(self):
        return self.dummy.device

    def forward(
        self,
        coords,
        mask = None
    ):
        """
        coords: coordinates to be augmented
        """
        batch_size = coords.shape[0]

        # Center the coordinates
        # Accounting for masking

        if mask is not None:

            coords = torch.where(mask.unsqueeze(-1), coords, 0.)
            num = reduce(coords, 'b n c -> b c', 'sum')
            den = reduce(mask.float(), 'b n -> b', 'sum')

            coords_mean = num.unsqueeze(1)/(den.clamp(min = 1.).unsqueeze(-1).unsqueeze(-1))
        else:
            coords_mean = coords.mean(dim = 1, keepdim = True)

        centered_coords = coords - coords_mean

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(batch_size)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(batch_size)
        translation_vector = rearrange(translation_vector, 'b c -> b 1 c')

        # Apply rotation and translation
        augmented_coords = einsum(centered_coords, rotation_matrix, 'b n i, b j i -> b n j') + translation_vector

        return augmented_coords

    def _random_rotation_matrix(self, batch_size):
        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device = self.device()) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device = self.device())
        rotation_matrix = repeat(eye, 'i j -> b i j', b = batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] - sin_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 0, 2] = cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] + sin_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] + cos_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 1, 2] = sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] - cos_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    def _random_translation_vector(self, batch_size):
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device = self.device()) * self.trans_scale
        return translation_vector
class EDM(nn.Module):
    def __init__(
        self,
        atom_encoder_depth =2,
        atom_decoder_depth =2,
        token_transformer_depth=6,
        P_mean = -1.2,
        P_std = 1.5,
        sigma_data = 16,
        sigma_min = 4e-4, 
        sigma_max = 160, 
        rho = 7,
        S_churn = 0.8, 
        S_min = 1.0,
        S_noise = 1.003,
        S_step = 1.5,
        num_sample_steps = 201,
        use_deepspeed_attention=False,
        generator = None,
    ):
        super().__init__()

        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.S_churn = S_churn
        self.S_min = S_min
        self.S_noise = S_noise
        self.S_step = S_step

        self.num_sample_steps = num_sample_steps

        self.net = DiffusionModule(atom_encoder_depth=atom_encoder_depth,atom_decoder_depth=atom_decoder_depth,token_transformer_depth=token_transformer_depth,use_deepspeed_attention=use_deepspeed_attention)

        self.weighted_rigid_align = WeightedRigidAlign()
        self.centre_random_augmentation = CentreRandomAugmentation()
        
        self.generator = generator
    
    def forward(self, 
            atom_pos = None, 
            atom_ref_type = None, 
            atom_mask = None,
            atom_ref_pos = None, 
            atom_ref_aatype = None,
            seq_mask = None,
            single_fea = None, 
            pair_fea = None, 
            residue_atom_lens = None,
            residue_indices = None,
            ligand_atom_type = None,
            ligand_len = None,
            protein_len = None):
        if atom_pos is not None:
            return self.calculate_loss(   
                       atom_pos = atom_pos, 
                       atom_ref_type = atom_ref_type, 
                       atom_mask = atom_mask,
                       atom_ref_pos = atom_ref_pos,
                       atom_ref_aatype= atom_ref_aatype, 
                       seq_mask = seq_mask,
                       single_fea = single_fea, 
                       pair_fea = pair_fea, 
                       residue_atom_lens = residue_atom_lens,
                       residue_indices = residue_indices,
                       ligand_atom_type = ligand_atom_type,
                       ligand_len = ligand_len,
                       protein_len = protein_len)
        else:
            return self.sample(
                atom_ref_type = atom_ref_type,
                atom_ref_pos=atom_ref_pos,
                atom_ref_aatype=atom_ref_aatype,
                seq_mask = seq_mask ,
                single_fea = single_fea, 
                pair_fea = pair_fea,
                residue_atom_lens = residue_atom_lens,
                residue_indices = residue_indices,
                ligand_atom_type = ligand_atom_type,
                ligand_len = ligand_len,
                protein_len = protein_len
            )

    def calculate_loss(self, 
                       atom_pos = None, 
                       atom_ref_type = None, 
                       atom_mask = None,
                       atom_ref_pos = None,
                       atom_ref_aatype=None, 
                       seq_mask = None,
                       single_fea = None, 
                       pair_fea = None, 
                       residue_atom_lens = None,
                       residue_indices = None,
                       ligand_atom_type = None,
                       ligand_len = None,
                       protein_len = None):
        N, L = atom_pos.shape[:2]
        device = atom_pos.device
        
        atom_pos = self.centre_random_augmentation(atom_pos,atom_mask.bool())
        
        # Noise distribution
        rnd_normal = torch.randn([N, 1, 1], device=device,generator=self.generator)
        sigma = self.sigma_data * ((rnd_normal * self.P_std + self.P_mean).exp())
        
        # Add noise to positions
        atom_pos_noisy = atom_pos + torch.randn(atom_pos.shape,device = atom_pos.device,generator=self.generator) * sigma

        # get net predict
        denoised_atom_pos = self.net(sigma, atom_pos_noisy, atom_ref_type,atom_ref_pos,atom_ref_aatype, atom_mask, seq_mask, \
                                     single_fea, pair_fea, residue_atom_lens,residue_indices, ligand_atom_type, ligand_len = ligand_len, protein_len = protein_len)

        align_weights = torch.ones(N, L).to(device = device)
        # ligand weight 11, 0819
        # start_indices = protein_len.long()
        # end_indices = (start_indices + ligand_len).long()
        # indices = torch.arange(align_weights.size(1), device=device).unsqueeze(0)
        # mask = (indices >= start_indices.unsqueeze(1)) & (indices < end_indices.unsqueeze(1))
        # align_weights[mask] += 10
        
        # atom_pos_aligned = self.weighted_rigid_align(
        #     atom_pos,
        #     denogised_atom_pos,
        #     alin_weights,
        #     mask = atom_mask.bool()
        # )
        atom_pos_aligned = atom_pos

        loss_dict = {}

        loss_pos = F.mse_loss(denoised_atom_pos, atom_pos_aligned, reduction = 'none') / 3.
        

        loss_pos = loss_pos*(align_weights.unsqueeze(-1))
        
        loss_weights = (sigma ** 2 + self.sigma_data ** 2) / ((sigma  *  self.sigma_data) ** 2) # + --> *
        
        loss_pos = loss_pos * loss_weights

        mse_pos = loss_pos[atom_mask.bool()].sum(dim = -1).mean()

        loss_dict['pos'] = mse_pos
        return loss_dict
    
    def sample_schedule(self, num_sample_steps = None, device = None):

        num_sample_steps = num_sample_steps if num_sample_steps else self.num_sample_steps

        N = num_sample_steps

        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = device, dtype = torch.float32)

        sigmas = self.sigma_data * (self.sigma_max ** inv_rho + (steps / (N - 1)) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])]) # t_N = 0
        
        return sigmas

    @torch.no_grad()
    def sample(
        self, 
        atom_ref_type = None,
        atom_ref_pos = None,
        atom_ref_aatype=None,
        seq_mask  = None,
        single_fea = None, 
        pair_fea = None,
        residue_atom_lens = None,
        residue_indices = None,
        ligand_atom_type = None,
        ligand_len = None,
        protein_len = None
    ):
        device = atom_ref_type.device
        shape = list(atom_ref_type.shape)
        shape.append(3)

        atom_mask = torch.BoolTensor(np.ones((shape[0], shape[1]))).to(device=device)
        
        sigmas = self.sample_schedule(self.num_sample_steps, device = device)

        # atom position is noise at the beginning
        atom_pos = sigmas[0] * torch.randn(shape, device = device)


        # Main sampling loop.
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])): # 0, ..., N-1
            
            atom_pos = self.centre_random_augmentation(atom_pos,atom_mask.bool())

            gamma  = self.S_churn if sigma_next > self.S_min else 0

            sigma = sigma.unsqueeze(-1).repeat(shape[0], 1, 1)
            sigma_next = sigma_next.unsqueeze(-1).repeat(shape[0], 1, 1)
            
            t_hat = sigma * (gamma + 1)
            
            # scaled noise
            atom_eps = self.S_noise * ((t_hat ** 2 - sigma ** 2).sqrt()) * torch.randn(shape, device = device) # stochastic sampling

            # add noise
            atom_pos_hat = atom_pos + atom_eps

            # get model predict
            denoised_atom_pos = self.net(t_hat, atom_pos_hat, atom_ref_type, atom_ref_pos,atom_ref_aatype,atom_mask,seq_mask, \
                                         single_fea, pair_fea, residue_atom_lens,residue_indices, ligand_atom_type =ligand_atom_type, ligand_len = ligand_len, protein_len = protein_len)
            
            denoised_cur = (atom_pos_hat - denoised_atom_pos) / t_hat #BUG

            # follow edm
            atom_pos_next = atom_pos_hat + self.S_step * (sigma_next - t_hat) * denoised_cur

            atom_pos = atom_pos_next

        return atom_pos

# pair resnet from alphafold1

class ResidualBlock(nn.Module):
    def __init__(self, dim=128, dilation=1):
        super(ResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim) 
        self.elu1 = nn.ELU()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)

        self.ln2 = nn.LayerNorm(dim) 
        self.elu2 = nn.ELU()

        self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)

        self.ln3 = nn.LayerNorm(dim)
        self.elu3 = nn.ELU()

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)

        self.ln4 = nn.LayerNorm(dim)

    def forward(self, x):

        identity = x 

        out = self.ln1(x) 
        out = self.elu1(out)

        out = out.permute(0, 3, 1, 2) 
        out = self.conv1(out)

        out = out.permute(0, 2, 3, 1)
        out = self.ln2(out)
        out = self.elu2(out)

        out = out.permute(0, 3, 1, 2) 
        out = self.dilated_conv(out)

        out = out.permute(0, 2, 3, 1)
        out = self.ln3(out)
        out = self.elu3(out)

        out = out.permute(0, 3, 1, 2)
        out = self.conv2(out)

        out = out.permute(0, 2, 3, 1)
        out += identity 
        out = self.ln4(out)
        out = F.elu(out)

        return out


class PairNetwork(nn.Module):
    def __init__(self, dim=128, num_blocks=8, dilation_cycle=[1, 2, 4, 8]):
        super(PairNetwork, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = dilation_cycle[i % len(dilation_cycle)]
            self.blocks.append(ResidualBlock(dim, dilation))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

        

class PairModel(nn.Module):
    def __init__(
        self,
        tf_dim = 256,
        c_z = 128,
        max_relative_idx = 32,
        use_chain_relative = True,
        max_relative_chain = 2,
        relpos_k = 32,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(PairModel, self).__init__()

        self.tf_dim = tf_dim


        self.c_z = c_z
        # self.c_m = c_m
        self.relpos_k = relpos_k
        self.linear_tf_z_i = nn.Linear(tf_dim, c_z)
        self.linear_tf_z_j = nn.Linear(tf_dim, c_z)
        # self.linear_tf_m = nn.Linear(tf_dim, c_m)


        # RPE stuff
        # self.max_relative_idx = max_relative_idx
        # self.use_chain_relative = use_chain_relative
        # self.max_relative_chain = max_relative_chain

        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = nn.Linear(self.no_bins, c_z)

        self.pair_resnet = PairNetwork()

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """

        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        ) 
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def forward(
        self, tf,ri,inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Dict containing
                "target_feat":
                    Features of shape [*, N_res, tf_dim]
                "residue_index":
                    Features of shape [*, N_res]
                "msa_feat":
                    Features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )
        
        pair_emb = self.pair_resnet(pair_emb)
        return pair_emb


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = nn.Linear(self.c_z, self.no_bins)

    def _forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
    
    def forward(self, z): 
        # if(is_fp16_enabled()):
        #     with torch.cuda.amp.autocast(enabled=False):
        #         return self._forward(z.float())
        # else:
            # return self._forward(z)
        return self._forward(z)

class ProDNAFold(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.DNAbackbone = DNAbackbone(config)
        # self.probackbone = Probackbone(config)
        
        self.num_blocks_enc = self.config.diffusion.num_blocks_enc
        self.num_blocks_dec = self.config.diffusion.num_blocks_dec
        self.num_blocks_dit = self.config.diffusion.num_blocks_dit

        self.pairmodel = PairModel()
        self.token_pair_channel = 128
        self.diff_model = EDM(atom_encoder_depth=self.num_blocks_enc,atom_decoder_depth=self.num_blocks_dec,token_transformer_depth=self.num_blocks_dit)
        self.dis_head = DistogramHead(self.token_pair_channel, 64)  ## 64 bins

        self.position_embeddings = nn.Embedding(1024, 256)

        # EDM(atom_encoder_depth=args.num_blocks_enc,
        # atom_decoder_depth=args.num_blocks_dec,token_transformer_depth=args.num_blocks_dit,generator = generator,use_deepspeed_attention = False)

    def forward(self, batch, dna_feats, protein_feats, mode='train', diffusion_batch_size=6):

        atom_pos = batch['atom_pos']
        atom_ref_pos = batch['atom_ref_pos']
        atom_ref_type = batch['atom_ref_type']
        atom_mask = batch['atom_mask']
        atom_ref_aatype = batch['atom_ref_aatype']
        seq_mask = batch['seq_mask']
        residue_atom_lens = batch['residue_atom_lens']
        residue_indices = batch['residue_indices']

        # dna_len = batch['dna_atom_len']
        # protein_atom_len = batch['protein_atom_len']
        if dna_feats.shape[1] + protein_feats.shape[1] == residue_indices.shape[1]:
            single_feat =  torch.cat([dna_feats, protein_feats], dim=-2) # b,token,heddien_size
        else:
            single_feat =  torch.cat([dna_feats, protein_feats[:,1:-1]], dim=-2) # b,token,heddien_size
        single_feat = single_feat + self.position_embeddings(residue_indices)
        # single_feat= protein_feats[:,1:-1]

        pair_feats = self.pairmodel(single_feat, residue_indices)# b, token, token, hidden_size

        pair_feats = pair_feats.repeat(diffusion_batch_size, 1, 1, 1)
        single_feat = single_feat.repeat(diffusion_batch_size, 1, 1)
        residue_indices = residue_indices.repeat(diffusion_batch_size, 1)

        if mode == 'train':
            loss_diffusion = self.diff_model(atom_pos = atom_pos,atom_ref_pos=atom_ref_pos, atom_ref_type = atom_ref_type, atom_mask = atom_mask,atom_ref_aatype=atom_ref_aatype, \
                                        seq_mask = seq_mask, single_fea =single_feat, pair_fea =pair_feats, residue_atom_lens = residue_atom_lens,residue_indices = residue_indices,\
                                            ligand_atom_type=None, ligand_len=None,  protein_len=None)
        else:
            pos = self.diff_model(atom_pos = None,atom_ref_pos=atom_ref_pos, atom_ref_type = atom_ref_type, atom_mask = atom_mask,atom_ref_aatype=atom_ref_aatype, \
                                        seq_mask = seq_mask, single_fea =single_feat, pair_fea =pair_feats, residue_atom_lens = residue_atom_lens,residue_indices = residue_indices,\
                                            ligand_atom_type=None, ligand_len=None,  protein_len=None)
            return pos

        distogram_logits = self.dis_head(pair_feats)
        dis_loss = distogram_loss(logits=distogram_logits,
                                  pseudo_beta=batch['pseudo_beta'],
                                  pseudo_beta_mask=batch['pseudo_beta_mask']) 
        loss_dict = {}
        loss_dict['diffusion'] = loss_diffusion['pos']
        loss_dict['dis_loss'] = dis_loss
        loss = self.config.loss.diffusion *loss_diffusion['pos'] + self.config.loss.distogram*dis_loss
        return loss, loss_dict