for r in 1
do
    CUDA_VISIBLE_DEVICES=1 python finetune/lora.py --precision "bf16-true" --rank $r --joint_layers 'False'
done