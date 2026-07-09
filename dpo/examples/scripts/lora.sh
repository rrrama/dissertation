for epochs in 1 # 5
do
    for algorithm in "erm" # "clamped"
    do
        for tolerance in 0.8 # 0.5 0.0
        do
            for r in 1 2 4 8
            do
                CUDA_VISIBLE_DEVICES=1 python dpof.py --per_device_eval_batch_size 1 --lr_scheduler_type "cosine" --beta 0.1 --max_prompt_length 1024 --max_length 1536  --learning_rate 5e-5 --algorithm ${algorithm} --optim paged_adamw_32bit --dataset orca --train_epochs $epochs --model_name_or_path=meta-llama/Llama-2-7B-hf --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --logging_steps 10 --eval_steps 50 --output_dir=dpo_intel --warmup_steps 200 --report_to wandb --bf16 --load_in_4bit --logging_first_step --no_remove_unused_columns --use_peft --lora_r=$r --lora_alpha=16 --loss_tolerance $tolerance
            done
       done
    done
done