python scripts/download.py --repo_id meta-llama/Llama-2-13b-hf --access_token ''
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf