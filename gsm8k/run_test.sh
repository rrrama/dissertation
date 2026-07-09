#!/bin/bash

# Set environment variables if needed

# Define variables for model paths
MODEL_NAME="meta-llama/Llama-2-7b-hf"
EXPERIMENT_NAME="gsm8k_training_lora"
EPOCHS=2
LR="3e-5"
SEED=42

# Construct the adapter path based on training output structure
ADAPTER_PATH="outputs_lora/gsm8k_training_lora/Llama-2-7b-hf/ep_6/lr_0.0003/seed_42"
#"outputs_lora/${EXPERIMENT_NAME}/${MODEL_NAME##*/}/ep_${EPOCHS}/lr_${LR}/seed_${SEED}"

CUDA_VISIBLE_DEVICES=1 python test.py \
    --model_name_or_path ${MODEL_NAME} \
    --adapter_name_or_path ${ADAPTER_PATH} \
    --data_name "gsm8k" \
    --model_max_length 512 \
    --batch_size 16 \
    --full_precision True