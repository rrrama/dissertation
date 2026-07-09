#!/bin/bash

# Run from this script's directory so train.py resolves regardless of launch cwd
cd "$(dirname "$0")" || exit 1

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1,2  # Specify which GPUs to use

for rank in 512
do
    for epoch in 6
    do
        for lr in 5e-2
        do
            torchrun --standalone --nproc_per_node=3 train.py \
                --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
                --data_name "gsm8k" \
                --tuning_type "lorta" \
                --rank 128 \
                --lora_alpha 4 \
                --learning_rate $lr \
                --num_train_epochs $epoch \
                --per_device_train_batch_size 2 \
                --gradient_accumulation_steps 10 \
                --gradient_checkpointing True \
                --output_dir "outputs_llada_lorta_${rank}_${epoch}_${lr}" \
                --expt_name "llada_gsm8k_training" \
                --model_max_length 512 \
                --logging_steps 10 \
                --save_strategy "steps" \
                --save_steps 100 \
                --warmup_ratio 0.03 \
                --lr_scheduler_type "cosine" \
                --seed 42 \
                --report_to "wandb" \
                --logging_strategy "steps" \
                --eval_steps 100 \
                --mask_id 126336 \
                --gen_length 256 \
                --diffusion_steps 256 \
                --block_length 256 \
                --remasking "low_confidence" \
                --gen_temperature 0.0
        done
    done
done
