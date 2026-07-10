#!/bin/bash
#
#SBATCH --job-name=llada_lorta
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:lovelace_l40:3
#SBATCH --time=12:00:00
#SBATCH --mem=120G

source /etc/profile.d/modules.sh
source $SHARE/u5751903/lorta_venv/bin/activate

export HF_HOME=$SHARE/u5751903/models/
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install wandb
pip install -e $SHARE/u5751903/lorta/peft --no-deps

srun bash llada/run_training.sh
