python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
pip install sentencepiece
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf