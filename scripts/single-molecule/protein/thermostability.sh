#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p vip_gpu_ailab 
#SBATCH -A ai4agr


module load anaconda/2021.11 cuda/11.7.0 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda11.x
source activate /home/bingxing2/ailab/scxlab0067/.conda/envs/zhangqianyuan

TOKENIZER_NAME_OR_PATH=facebook/esm1b_t33_650M_UR50S
MODE_MAX_LENGTH=1022
FREEZE_BACKBONE=False

### MODEL ESM1b
MODEL_NAME_OR_PATH=facebook/esm1b_t33_650M_UR50S
MODEL_TYPE=esm-1b
CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/hanwenwei/huggingface/esm1b_t33_650M_UR50S

### MODEL ESM2
# MODEL_NAME_OR_PATH=facebook/esm2_t30_150M_UR50D
# MODEL_TYPE=esm-2
# CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/hanwenwei/huggingface/esm2_t30_150M_UR50D

### MODEL DNABERT2
# MODEL_NAME_OR_PATH=zhihan1996/DNABERT2-2-117M
# MODEL_TYPE=dnabert2
# CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/public/multi-omics/DNA/model/opensource/DNABERT2

### MODEL NTV2
# MODEL_NAME_OR_PATH=InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# MODEL_TYPE=ntv2
# CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/public/multi-omics/DNA/model/opensource/nt/nt_v2_100m_multi

### MODEL RNAFM
# MODEL_NAME_OR_PATH=multimolecule/rnafm
# MODEL_TYPE=rna-fm
# CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/public/multi-omics/RNA/model/opensource/rna-fm

### MODEL BEACON
# MODEL_NAME_OR_PATH=ours/BEACON
# MODEL_TYPE=beacon
# CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/public/multi-omics/RNA/model/ours/BEACON-B

### MODEL LUCAONE
# MODEL_NAME_OR_PATH=Yuanfei/LucaOne
# MODEL_TYPE=lucaone
# CACHE_DIR=/home/bingxing2/ailab/group/ai4bio/public/multi-omics/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step5600000/



python downstream/protein_tasks/train_ec.py \
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH \
    --tokenizer_cache_dir /home/bingxing2/ailab/group/ai4bio/hanwenwei/huggingface/ \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path /home/bingxing2/ailab/group/ai4bio/public/multi-omics/protein/downstream/thermostability/ \
    --cache_dir $CACHE_DIR  \
    --model_max_length $MODE_MAX_LENGTH \
    --model_type $MODEL_TYPE \
    --freeze_backbone $FREEZE_BACKBONE \

