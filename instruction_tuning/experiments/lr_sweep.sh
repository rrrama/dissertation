rank=235
for lr in 5e-4 5e-3 1e-4 5e-3 5e-2 1e-2 5e-5 5e-5
do
    CUDA_VISIBLE_DEVICES=0 python finetune/lora.py --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" --precision "bf16-true" --rank $rank --tensor_lora 'True' --joint_layers 'True' --joint_qkvp 'True' --joint_heads 'True' --init_scale 0.0 --alpha 16 --lora_dropout 0.05 --learning_rate $lr
done