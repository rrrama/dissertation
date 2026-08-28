#!/bin/bash
# Resolve the LoRTA environment. Sourced by bootstrap.sh, by the per-run job
# scripts batch_train.py generates, and by hand:
#
#     source scripts/env.sh
#
# Idempotent and side-effect-free apart from exporting variables and activating
# the venv, so sourcing it twice is harmless.
#
# Two roots, because a rented instance has two disks (SLURM_MIGRATION.md §2):
#
#   LORTA_PERSIST_ROOT  volume storage. The repo, adapters, checkpoints,
#                       benchmark results, logs. Losing it loses the experiment.
#   LORTA_SCRATCH_ROOT  instance storage. venv, pip cache, HF model weights,
#                       datasets cache. Losing it costs a bootstrap.
#
# Precedence, highest first:
#   1. variables already exported in the calling shell
#   2. $LORTA_REPO_ROOT/.lorta.env   (written by bootstrap.sh; per-instance, untracked)
#   3. the defaults below
#
# That ordering is what makes this zero-faff: bootstrap writes .lorta.env once,
# and every later shell and job picks the same roots up with no flags.

# Repo root, from this script's own location -- not $PWD, since jobs cd to llada/.
LORTA_REPO_ROOT="${LORTA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LORTA_REPO_ROOT

# Per-instance overrides. Sourced with `set -a` so plain `KEY=value` lines export
# without needing `export` on each, but only for keys not already set: an explicit
# `LORTA_SCRATCH_ROOT=/mnt/x scripts/bootstrap.sh` must win over a stale file.
if [ -f "$LORTA_REPO_ROOT/.lorta.env" ]; then
    _lorta_saved="$(export -p | grep -E '^declare -x LORTA_|^declare -x HF_HOME' || true)"
    set -a
    # shellcheck disable=SC1091
    . "$LORTA_REPO_ROOT/.lorta.env"
    set +a
    eval "$_lorta_saved"
    unset _lorta_saved
fi

export LORTA_PERSIST_ROOT="${LORTA_PERSIST_ROOT:-$(dirname "$LORTA_REPO_ROOT")}"

# /scratch is the usual mount name for instance storage on rented boxes; fall back
# to /tmp, which is at least local and not the volume.
if [ -z "${LORTA_SCRATCH_ROOT:-}" ]; then
    if [ -d /scratch ] && [ -w /scratch ]; then
        LORTA_SCRATCH_ROOT=/scratch/lorta
    else
        LORTA_SCRATCH_ROOT=/tmp/lorta
    fi
fi
export LORTA_SCRATCH_ROOT

# Outputs default *inside the repo* when the repo is already on persistent
# storage -- that is where they live today, and moving them would orphan
# llada/outputs/baseline and llada/outputs/lorta_vs_nalorta. Only when the repo
# sits outside the persistent root do they get relocated onto it.
if [ -z "${LORTA_OUTPUTS_ROOT:-}" ]; then
    case "$LORTA_REPO_ROOT/" in
        "$LORTA_PERSIST_ROOT"/*) LORTA_OUTPUTS_ROOT="$LORTA_REPO_ROOT/llada/outputs" ;;
        *)                       LORTA_OUTPUTS_ROOT="$LORTA_PERSIST_ROOT/lorta-outputs" ;;
    esac
fi
export LORTA_OUTPUTS_ROOT

export LORTA_VENV="${LORTA_VENV:-$LORTA_SCRATCH_ROOT/venv}"

# Model weights and datasets on instance storage. ~16 GB for LLaDA-8B; volume
# storage bills continuously and these are re-downloadable. Override HF_HOME to
# the persistent root if the volume is large and cheap -- that trades one
# bootstrap step for standing storage cost, and nothing else changes.
export HF_HOME="${HF_HOME:-$LORTA_SCRATCH_ROOT/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$LORTA_SCRATCH_ROOT/pip-cache}"

# Runtime settings that were in every #SBATCH preamble.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# The tokenizers fast-path forks under the dataloader and warns on every batch.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ -f "$LORTA_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$LORTA_VENV/bin/activate"
fi

lorta_env_describe() {
    cat <<EOF
LORTA_REPO_ROOT     $LORTA_REPO_ROOT
LORTA_PERSIST_ROOT  $LORTA_PERSIST_ROOT
LORTA_SCRATCH_ROOT  $LORTA_SCRATCH_ROOT
LORTA_OUTPUTS_ROOT  $LORTA_OUTPUTS_ROOT
LORTA_VENV          $LORTA_VENV
HF_HOME             $HF_HOME
WANDB_MODE          $WANDB_MODE
EOF
}
