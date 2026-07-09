#!/bin/bash
#
#SBATCH --job-name=llada_lorta
#SBATCH --partition=gecko
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --mem=120G

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
conda activate llada
module load CUDA/12.6.2

export HF_HOME=/dcs/large/u5751903/models/
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies on run
pip install wandb
# Local LorTA-enabled peft (editable) so LorTaConfig is importable.
# --no-deps: env already has compatible torch/transformers/etc; avoid upgrades.
pip install -e /dcs/pg25/u5751903/dev/lorta/peft --no-deps

srun bash llada/run_training.sh
