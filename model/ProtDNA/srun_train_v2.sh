#!/bin/bash

# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

TRAIN_DATA_DIR=prot_dna_metadata-filter_dna-train.json
VAL_DATA_DIR=prot_dna_metadata-filter_dna-val.json
test_data_dir=prot_dna_metadata-filter_dna-test.json

EXPERIMENT_NAME=protdna-new_data-224-8-dna2+esm-100epoch
WANDB_ID=$EXPERIMENT_NAME
OUTPUT_DIR=./outputs/$EXPERIMENT_NAME
ACCELERATE_CONFIG_FILE=./dna_model/accelerator_template_1.json

LR=0.0001
WARMUP_STEPS=1000
warmup_epochs=5
BATCH_SIZE=1
DIFFUSION_BATCH_SIZE=8
SEED=47
BLOCKS_BACK=6
BLOCKS_ENC=2
BLOCKS_DEC=2
BLOCKS_DIT=4
TRAIN_CROP_SIZE=768
NUM_TRAIN_EPOCHS=100
MAX_GRAD_NORM=10
PYTHON_FILE=./ProtDNA/train_v2_backbone.py


export TMPDIR_PRFX=./

for seed in 42
do
 
srun -p ai4bio --quotatype=spot --gres=gpu:1 accelerate launch --config_file $ACCELERATE_CONFIG_FILE  --main_process_port 21233  $PYTHON_FILE \
    --base_path "./datasets/deepdps/mmcif_feats_v2" \
    --train_data_dir $TRAIN_DATA_DIR \
    --val_data_dir $VAL_DATA_DIR \
    --test_data_dir $test_data_dir \
    --num_blocks_back $BLOCKS_BACK \
    --num_blocks_enc $BLOCKS_ENC \
    --num_blocks_dec $BLOCKS_DEC \
    --num_blocks_dit $BLOCKS_DIT \
    --max_grad_norm $MAX_GRAD_NORM \
    --lr $LR \
    --lr_backbone 0.00003 \
    --warmup_epochs $warmup_epochs \
    --batch_size $BATCH_SIZE \
    --diffusion_batch_size $DIFFUSION_BATCH_SIZE \
    --seed $SEED \
    --train_crop_size $TRAIN_CROP_SIZE \
    --experiment_name  $EXPERIMENT_NAME \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --wandb_id $WANDB_ID \
    --num_workers 4 \
    --dna_model_name_or_path "./ProtDNA/nt/nt_v2_100m_multi" \
    --protein_model_name_or_path "./ProtDNA/esm-1b/esm1b_t33_650M_UR50S"
done