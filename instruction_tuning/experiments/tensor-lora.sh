for r in 235
do
    python finetune/lora.py --checkpoint_dir "checkpoints/openlm-research/open_llama_7b" --precision "bf16-true" --rank $r --tensor_lora 'True' --joint_layers 'True' --joint_qkvp 'True' --joint_heads 'True' --init_scale 0.0 --alpha 16 --lora_dropout 0.05 --learning_rate 0.0003
done