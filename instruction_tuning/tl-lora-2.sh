for r in 32 64
do
    CUDA_VISIBLE_DEVICES=1 python finetune/lora.py --precision "bf16-true" --rank $r --tensor_lora 'True' --joint_layers 'True' --joint_qkvp 'True' --init_scale 0.0
done