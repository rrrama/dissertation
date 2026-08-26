#!/bin/bash
#
# EXPERIMENT_DESIGN.md §9, run as one job. §10 gates every experiment on this
# exiting 0, so submit it before Tier 0 and read the table it prints.
#
# Not routed through batch_train.py: it is not a run, it produces no adapter and
# no benchmark result, and it has no config to sweep.
#
# One GPU. Only §9.5 needs it -- §9.2 and §9.4 are CPU-only shape and count
# checks -- but the job loads the 8B model four times either way, once per arm,
# and that is what the hour is for.
#
# Submit from `llada/`, which is where every other script here has to run from
# too (see the `cd` below):
#     cd llada && sbatch jobs/correctness_checks.sbatch
#
# The log paths below are relative to the submit directory and slurm fails the
# job outright if they do not resolve, so submitting from elsewhere fails fast
# rather than half-working.
#
#SBATCH --job-name=llada_correctness
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/correctness-%j.out
#SBATCH --error=logs/correctness-%j.err

source /etc/profile.d/modules.sh
source $SHARE/u5751903/lorta_venv/bin/activate

export HF_HOME=$SHARE/u5751903/models/
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install wandb pyyaml

# The peft fork from THIS checkout, so the tuners being checked are the ones the
# experiments will run.
LLADA_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
pip install -e "$(dirname "$LLADA_DIR")/peft" --no-deps || exit 1

# `llada/`, not the repo root. The repo root holds a legacy `gsm8k/` directory
# (the old Llama harness), and `datasets.load_dataset("gsm8k", ...)` resolves
# local paths before the hub -- so from the root it finds that directory, finds
# no parquet in it, and dies with "No (supported) data files found in gsm8k".
# Every sbatch this project generates cds here for the same reason.
cd "$LLADA_DIR"

# §9.5 scores the committed frozen eval set, so it has to exist. Generating it is
# a one-off (`frozen_eval.py --write`), and it is committed, so this is a missing
# prerequisite rather than something to generate here -- a set built on the fly
# would differ from the one every run is scored against.
python -c "
import os, sys
from frozen_eval import FROZEN_EVAL_FILE
if not os.path.exists(FROZEN_EVAL_FILE):
    sys.exit(
        f'{FROZEN_EVAL_FILE} is missing. Run `python frozen_eval.py --write` from '
        'llada/ and commit it before running the correctness checks.'
    )
" || exit 1

srun python correctness_checks.py
