for r in 8 16
do
    python finetune/lora.py --precision "bf16-true" --rank $r --tensor_lora 'True' --joint_layers 'True' --joint_qkvp 'True' --init_scale 0.0
done