#!/bin/bash
#
# EXPERIMENT_DESIGN.md §9, run as one job. §10 gates every experiment on this
# exiting 0, so run it before Tier 0 and read the table it prints.
#
# Not routed through batch_train.py: it is not a run, it produces no adapter and
# no benchmark result, and it has no config to sweep.
#
# One GPU. Only §9.5 needs it -- §9.2 and §9.4 are CPU-only shape and count
# checks -- but the job loads the 8B model four times either way, once per arm,
# and that is what the hour is for.
#
# Run from `llada/`, which is where every other script here has to run from too
# (see the `cd` below):
#
#     cd llada && bash jobs/correctness_checks.sh
#     cd llada && CUDA_VISIBLE_DEVICES=1 bash jobs/correctness_checks.sh
#
# Re-clearing the §10 gate on new hardware is the first thing to do after
# bootstrapping a rented instance: it is the cheapest check that the pinned
# environment reproduces the cluster's numerics.

set -euo pipefail

LLADA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$LLADA_DIR")"

# One GPU unless told otherwise; the checks are not distributed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Storage roots, venv, HF_HOME, WANDB_MODE. `set +u` because some venv activate
# scripts touch $PS1 unguarded, which is fatal under `set -u`.
set +u
source "$REPO_ROOT/scripts/env.sh"
set -u

# No `pip install` here any more: scripts/bootstrap.sh owns the environment, once
# per instance, and asserts that peft resolves into this checkout. Re-assert it
# cheaply rather than reinstalling -- the tuners being checked must be the ones
# the experiments will run.
python -c "
import peft, sys
from peft import LorTaConfig, NALorTaConfig, NARAConfig
if not peft.__file__.startswith('$REPO_ROOT/peft'):
    sys.exit(f'peft resolves to {peft.__file__}, outside $REPO_ROOT/peft. '
             'Run: bash scripts/bootstrap.sh')
print('peft from', peft.__file__)
" || exit 1

# `llada/`, not the repo root. The repo root holds a legacy `gsm8k/` directory
# (the old Llama harness), and `datasets.load_dataset("gsm8k", ...)` resolves
# local paths before the hub -- so from the root it finds that directory, finds
# no parquet in it, and dies with "No (supported) data files found in gsm8k".
# Every job script this project generates cds here for the same reason.
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
        f'{FROZEN_EVAL_FILE} is missing. Run \`python frozen_eval.py --write\` from '
        'llada/ and commit it before running the correctness checks.'
    )
" || exit 1

# No `exec`: pipefail makes the pipeline report python's exit status, which is
# what §10 gates on, and exec'ing into a pipeline would report tee's instead.
mkdir -p logs
python correctness_checks.py 2>&1 | tee "logs/correctness-$(date +%Y%m%d-%H%M%S).log"
