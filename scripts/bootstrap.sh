#!/bin/bash
#
# One-shot setup for a fresh rented GPU instance. Idempotent -- re-running it on
# a half-finished instance completes it rather than starting over, and re-running
# it on a finished one is a fast no-op that reprints the environment.
#
#     bash scripts/bootstrap.sh
#
# On a box whose persistent volume is mounted somewhere non-obvious:
#
#     bash scripts/bootstrap.sh --persist-root /mnt/volume --scratch-root /scratch
#
# The roots are written to .lorta.env (untracked) so every later shell and every
# generated job script picks them up with no flags. That file is the only
# per-instance state; delete it to re-resolve from scratch.
#
# What this replaces: the `pip install` lines that ran inside every single sbatch
# job, because each slurm job landed on a fresh node. One instance, one venv,
# installed once (SLURM_MIGRATION.md §6).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="GSAI-ML/LLaDA-8B-Instruct"
PYTHON_BIN="python3"
TORCH_INDEX=""
REQUIREMENTS=""
SKIP_MODEL=0
FORCE=0

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --persist-root PATH   Volume storage: repo, adapters, checkpoints, results.
  --scratch-root PATH   Instance storage: venv, pip cache, HF weights.
  --outputs-root PATH   Override where outputs/ lives (default: inside the repo
                        when the repo is on persistent storage).
  --python BIN          Interpreter to build the venv from (default: python3).
  --requirements FILE   Pin file (default: scripts/requirements.lock.txt, the
                        pip freeze of the cluster venv). No fallback: see §6.2.
  --torch-index URL     Extra pip index for a specific CUDA build, e.g.
                        https://download.pytorch.org/whl/cu118
  --model REPO          Base model to prefetch (default: GSAI-ML/LLaDA-8B-Instruct).
  --skip-model          Skip the model/dataset prefetch. Fast re-run; the first
                        training job then downloads ~16 GB itself.
  --force               Rebuild the venv from scratch.
  -h, --help            This message.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --persist-root) export LORTA_PERSIST_ROOT="$2"; shift 2 ;;
        --scratch-root) export LORTA_SCRATCH_ROOT="$2"; shift 2 ;;
        --outputs-root) export LORTA_OUTPUTS_ROOT="$2"; shift 2 ;;
        --python)       PYTHON_BIN="$2"; shift 2 ;;
        --requirements) REQUIREMENTS="$2"; shift 2 ;;
        --torch-index)  TORCH_INDEX="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --skip-model)   SKIP_MODEL=1; shift ;;
        --force)        FORCE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '    \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1. Roots
# --------------------------------------------------------------------------- #
step "Resolving storage roots"

export LORTA_REPO_ROOT="$REPO_ROOT"
# `set +u` around the source: env.sh activates the venv if one already exists, and
# some venv activate scripts touch $PS1 unguarded, which is fatal under `set -u`.
set +u
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"
set -u

mkdir -p "$LORTA_PERSIST_ROOT" "$LORTA_SCRATCH_ROOT" "$LORTA_OUTPUTS_ROOT" \
         "$HF_HOME" "$PIP_CACHE_DIR"
lorta_env_describe | sed 's/^/    /'

case "$LORTA_REPO_ROOT/" in
    "$LORTA_PERSIST_ROOT"/*) ;;
    *) warn "the repo is NOT under LORTA_PERSIST_ROOT."
       warn "outputs go to $LORTA_OUTPUTS_ROOT, but uncommitted work in the repo"
       warn "dies with the instance. Commit and push before you shut down." ;;
esac

# Adapters are small; checkpoints are not. A 6-epoch run keeps several optimizer
# states, and four run concurrently (SLURM_MIGRATION.md §5.2).
avail_gb=$(df -BG --output=avail "$LORTA_OUTPUTS_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [ "${avail_gb:-0}" -lt 50 ]; then
    warn "only ${avail_gb}G free on $LORTA_OUTPUTS_ROOT (want >=50G for checkpoints)"
else
    ok "${avail_gb}G free for outputs"
fi

# --------------------------------------------------------------------------- #
# 2. GPUs
# --------------------------------------------------------------------------- #
step "Checking GPUs"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
               --format=csv,noheader | sed 's/^/    /'
    n_gpus=$(nvidia-smi --list-gpus | wc -l)
    ok "$n_gpus GPU(s) visible"
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "${gpu_mem:-0}" -lt 40000 ]; then
        warn "$((gpu_mem/1024)) GB per GPU. The Tier 0 configs were sized for 48 GB"
        warn "(~33 GB peak at per_device_train_batch_size 4). Repartition the batch"
        warn "across all arms identically before running -- SLURM_MIGRATION.md §8.5."
    fi
    # The lock file pins torch 2.8.0, whose PyPI wheels bundle CUDA 12.8. A CUDA 12
    # runtime needs driver >= 525; below that torch installs cleanly and then dies
    # at the first .cuda() with "CUDA driver version is insufficient", hours later.
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
                   | head -1 | cut -d. -f1)
    if [ "${driver_major:-0}" -lt 525 ]; then
        warn "driver $driver_major is below 525; the pinned torch 2.8.0 (cu12.8)"
        warn "will not run. Pick an image with a newer driver, or install a cu11"
        warn "build with --torch-index https://download.pytorch.org/whl/cu118"
    else
        ok "driver $driver_major supports the pinned cu12.8 build"
    fi
else
    die "nvidia-smi not found; this is not a GPU instance (or the driver is missing)"
fi

# --------------------------------------------------------------------------- #
# 3. Virtualenv
# --------------------------------------------------------------------------- #
step "Building the virtualenv at $LORTA_VENV"
if [ "$FORCE" = 1 ] && [ -d "$LORTA_VENV" ]; then
    warn "--force: removing existing venv"
    rm -rf "$LORTA_VENV"
fi
if [ ! -f "$LORTA_VENV/bin/activate" ]; then
    "$PYTHON_BIN" -m venv "$LORTA_VENV" || die "venv creation failed (need python3-venv?)"
    ok "created from $("$PYTHON_BIN" --version 2>&1)"
else
    ok "already present"
fi
set +u
# shellcheck disable=SC1091
source "$LORTA_VENV/bin/activate"
set -u
python -m pip install --upgrade pip setuptools wheel >/dev/null
ok "pip $(pip --version | awk '{print $2}')"

# --------------------------------------------------------------------------- #
# 4. Dependencies
# --------------------------------------------------------------------------- #
step "Installing dependencies"
REQUIREMENTS="${REQUIREMENTS:-$REPO_ROOT/scripts/requirements.lock.txt}"
[ -f "$REQUIREMENTS" ] || die "requirements file not found: $REQUIREMENTS
    This is a pip freeze of the cluster venv every existing result was measured
    under. There is no fallback on purpose: guessed pins that merely import are
    how a silent environment mismatch gets into the comparison."

# A `pip freeze` of the cluster venv contains its own editable VCS install of the
# peft fork:
#
#   -e git+ssh://git@github.com/rrrama/dissertation.git@<sha>#egg=peft&subdirectory=peft
#
# Installing that here would be wrong three times over: it needs an SSH key with
# access to a private repo, it resolves *with* deps and so may move the pins below,
# and it installs the fork from a fixed remote commit rather than from the checkout
# being run -- exactly the "not a sibling clone" failure the sbatch preamble
# already warned about. The fork is installed from $REPO_ROOT/peft immediately
# after this, so strip any VCS/editable line here.
#
# Nothing is lost by dropping it: every dependency peft declares (numpy, packaging,
# psutil, pyyaml, torch, transformers, tqdm, accelerate, safetensors,
# huggingface_hub) is pinned explicitly in the lock file already.
FILTERED="$LORTA_SCRATCH_ROOT/requirements.filtered.txt"
grep -vE '^[[:space:]]*(-e[[:space:]]|#)|#egg=|@[[:space:]]+(git|file|https?):' \
    "$REQUIREMENTS" > "$FILTERED" || true
n_dropped=$(( $(wc -l < "$REQUIREMENTS") - $(wc -l < "$FILTERED") ))
if [ "$n_dropped" -gt 0 ]; then
    ok "dropped $n_dropped editable/VCS line(s); peft comes from this checkout below"
fi

pip_args=(install -r "$FILTERED")
# torch 2.8.0 on PyPI bundles CUDA 12.8 (the nvidia-*-cu12 pins in the lock file
# are its own dependencies), so no extra index is needed for Ada/Ampere/Hopper.
# --torch-index exists for a box that needs a different CUDA build.
if [ -n "$TORCH_INDEX" ]; then
    pip_args+=(--extra-index-url "$TORCH_INDEX")
fi
pip "${pip_args[@]}"
ok "installed from $(basename "$REQUIREMENTS")"

# The peft fork from THIS checkout, not PyPI and not a sibling clone: lorta /
# nalorta / nara live only here, and --no-deps stops it dragging in a transformers
# that disagrees with the pin above.
step "Installing the vendored peft fork (editable, --no-deps)"
pip install -e "$REPO_ROOT/peft" --no-deps
python - <<'PY' || die "the peft fork did not import correctly"
import peft
# All four tuning_type values the configs use must resolve, not just peft itself:
# a stray PyPI peft imports fine and then fails inside _build_peft_config.
from peft import LoraConfig, LorTaConfig, NALorTaConfig, NARAConfig  # noqa: F401
print(f"    peft {peft.__version__} from {peft.__file__}")
PY
# Assert it resolves into this checkout rather than a stray site-packages copy --
# the thing the per-job `pip install -e` was really defending against.
peft_path="$(python -c 'import peft; print(peft.__file__)')"
case "$peft_path" in
    "$REPO_ROOT"/peft/*) ok "resolves into this checkout" ;;
    *) die "peft resolves to $peft_path, outside $REPO_ROOT/peft" ;;
esac

step "Verifying torch sees the GPUs"
python - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"devices {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"    cuda:{i}  {p.name}  {p.total_memory / 1e9:.0f} GB")
PY
ok "torch/CUDA healthy"

# --------------------------------------------------------------------------- #
# 5. Prefetch weights and data
# --------------------------------------------------------------------------- #
# Serially, before any job runs. Four concurrent jobs each discovering a cold
# cache on their first forward is wasted bandwidth at best and a corrupted
# partial download at worst.
if [ "$SKIP_MODEL" = 1 ]; then
    step "Skipping model prefetch (--skip-model)"
else
    step "Prefetching $MODEL into $HF_HOME (~16 GB, once per instance)"
    MODEL="$MODEL" python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    os.environ["MODEL"],
    token=os.environ.get("HF_TOKEN"),
    # Weights + config + tokenizer + the trust_remote_code modeling files.
    ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.pth"],
)
print(f"    cached at {path}")
PY
    ok "model cached"

    step "Prefetching GSM8K"
    # From llada/, never the repo root: the root holds a legacy gsm8k/ directory
    # and load_dataset resolves local paths before the hub, so from there this
    # dies with "No (supported) data files found in gsm8k". Every job script cds
    # here for the same reason (OUTSTANDING.md).
    (cd "$REPO_ROOT/llada" && python - <<'PY'
from datasets import load_dataset
for split in ("train", "test"):
    d = load_dataset("gsm8k", "main", split=split)
    print(f"    gsm8k/{split}: {len(d)} examples")
PY
    )
    ok "dataset cached"
fi

# --------------------------------------------------------------------------- #
# 6. Repo-side prerequisites
# --------------------------------------------------------------------------- #
step "Checking repo prerequisites"
(cd "$REPO_ROOT/llada" && python - <<'PY'
import os, sys
from frozen_eval import FROZEN_EVAL_FILE
if not os.path.exists(FROZEN_EVAL_FILE):
    sys.exit(f"    {FROZEN_EVAL_FILE} is missing. It is committed, so this is a bad "
             "checkout; regenerating it would produce a different set from the one "
             "every result is scored against.")
print(f"    frozen eval set present: {FROZEN_EVAL_FILE}")
PY
) || die "frozen eval set missing"
ok "frozen eval set present"

if [ "${WANDB_MODE}" = "offline" ]; then
    ok "wandb offline (sync later with: wandb sync llada/wandb/offline-run-*)"
elif ! python -c "import wandb, sys; sys.exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
    warn "WANDB_MODE=$WANDB_MODE but no API key found; run 'wandb login'"
fi

# --------------------------------------------------------------------------- #
# 7. Record the environment
# --------------------------------------------------------------------------- #
step "Recording the environment"

cat > "$REPO_ROOT/.lorta.env" <<EOF
# Written by scripts/bootstrap.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Untracked.
# Sourced by scripts/env.sh; delete this file to re-resolve the roots.
LORTA_PERSIST_ROOT=$LORTA_PERSIST_ROOT
LORTA_SCRATCH_ROOT=$LORTA_SCRATCH_ROOT
LORTA_OUTPUTS_ROOT=$LORTA_OUTPUTS_ROOT
LORTA_VENV=$LORTA_VENV
HF_HOME=$HF_HOME
EOF
ok "wrote .lorta.env"

# The marker the supervisor's preflight looks for, and the record of what this
# instance actually resolved to -- which is the question you will be asking if a
# result from this box disagrees with one from the cluster.
pip freeze > "$LORTA_SCRATCH_ROOT/pip-freeze.txt"
python - <<PY
import json, os, subprocess, sys
def sh(*c):
    try:
        return subprocess.check_output(c, cwd="$REPO_ROOT", stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None
versions = {}
for m in ("torch", "transformers", "peft", "datasets", "accelerate", "numpy"):
    try:
        versions[m] = __import__(m).__version__
    except Exception:
        versions[m] = None
import torch
marker = {
    "bootstrapped": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "repo_root": "$REPO_ROOT",
    "git_hash": sh("git", "rev-parse", "HEAD"),
    "git_dirty": bool(sh("git", "status", "--porcelain")),
    "python": sys.version.split()[0],
    "versions": versions,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "roots": {
        "persist": "$LORTA_PERSIST_ROOT",
        "scratch": "$LORTA_SCRATCH_ROOT",
        "outputs": "$LORTA_OUTPUTS_ROOT",
        "hf_home": "$HF_HOME",
    },
    "requirements": "$(basename "$REQUIREMENTS")",
}
with open("$LORTA_SCRATCH_ROOT/.lorta-bootstrap.json", "w") as f:
    json.dump(marker, f, indent=2)
print("    " + json.dumps(marker["versions"]))
PY
ok "wrote $LORTA_SCRATCH_ROOT/.lorta-bootstrap.json and pip-freeze.txt"

cat <<EOF

$(printf '\033[1mBootstrap complete.\033[0m')

Every new shell needs the environment; .lorta.env means it takes no arguments:

    source scripts/env.sh

Then, from llada/ (never the repo root):

    cd llada
    bash jobs/correctness_checks.sh                          # the §10 gate; run this first
    python batch_train.py --mode all --config configs/tier0_lr.yaml --dry-run
    tmux new -s tier0
    python batch_train.py --mode all --config configs/tier0_lr.yaml

EOF
