#!/bin/bash

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0  # Specify which GPU to use


for rank in 512
do
    for epoch in 6
    do
        for lr in 5e-2
        do
            python train.py \
                --model_name_or_path "meta-llama/Llama-2-7b-hf" \
                --data_name "gsm8k" \
                --tuning_type "lorta" \
                --lora_init True \
                --rank 128 \
                --lora_alpha 4 \
                --learning_rate $lr \
                --num_train_epochs $epoch \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 16 \
                --output_dir "outputs_lorta_${rank}_${epoch}_${lr}" \
                --expt_name "gsm8k_training" \
                --model_max_length 512 \
                --logging_steps 10 \
                --save_strategy "steps" \
                --save_steps 100 \
                --warmup_ratio 0.03 \
                --lr_scheduler_type "cosine" \
                --seed 42 \
                --report_to "wandb" \
                --logging_strategy "steps" \
                --eval_steps 100 
        done
    done
done