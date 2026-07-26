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

# Install the peft fork from THIS checkout, not a sibling clone, so the tuners
# (lorta/nalorta) match the train.py being run.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
pip install -e "$REPO_ROOT/peft" --no-deps || exit 1
python -c "from peft import LorTaConfig, NALorTaConfig; import peft; print('peft from', peft.__file__)" || exit 1

srun bash llada/run_training.sh
