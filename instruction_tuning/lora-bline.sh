for r in 1
do
    CUDA_VISIBLE_DEVICES=1 python finetune/lora.py --checkpoint_dir "checkpoints/openlm-research/open_llama_7b" --precision "bf16-true" --rank $r --joint_qkvp 'False'
done