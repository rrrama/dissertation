#!/usr/bin/env python3
"""Storage roots for the LoRTA harness — the Python side of ``scripts/env.sh``.

A rented instance has two disks and they must not be confused
(SLURM_MIGRATION.md §2):

  ``PERSIST_ROOT``  volume storage. The repo, adapters, checkpoints, benchmark
                    results, logs. Losing it loses the experiment.
  ``SCRATCH_ROOT``  instance storage. venv, pip cache, HF weights, dataset cache.
                    Losing it costs a bootstrap.

Resolution order, highest first: an exported environment variable, then
``$REPO_ROOT/.lorta.env`` (written by ``scripts/bootstrap.sh``), then the
defaults below. That is the same order ``scripts/env.sh`` uses, and the defaults
are deliberately identical -- bash needs the roots before any Python runs, so the
duplication is unavoidable. **If you change a default here, change it there too.**

Stdlib only, and no import of torch/transformers, so ``batch_train.py`` keeps the
property that made it usable on a login node.
"""

import os

LLADA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LLADA_DIR)
PEFT_DIR = os.path.join(REPO_ROOT, "peft")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
ENV_FILE = os.path.join(REPO_ROOT, ".lorta.env")


def _load_env_file(path=ENV_FILE):
    """Parse bootstrap.sh's ``.lorta.env`` (plain ``KEY=value`` lines).

    Not exported into ``os.environ``: a value here must lose to one the caller
    exported, and mutating the environment would make that ordering invisible.
    """
    values = {}
    if not os.path.exists(path):
        return values
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


_FILE_ENV = _load_env_file()


def _setting(name, default):
    value = os.environ.get(name) or _FILE_ENV.get(name)
    return value if value else default


def _default_scratch():
    # /scratch is the usual mount name for instance storage on rented boxes.
    if os.path.isdir("/scratch") and os.access("/scratch", os.W_OK):
        return "/scratch/lorta"
    return "/tmp/lorta"


PERSIST_ROOT = _setting("LORTA_PERSIST_ROOT", os.path.dirname(REPO_ROOT))
SCRATCH_ROOT = _setting("LORTA_SCRATCH_ROOT", _default_scratch())
VENV_DIR = _setting("LORTA_VENV", os.path.join(SCRATCH_ROOT, "venv"))
HF_HOME = _setting("HF_HOME", os.path.join(SCRATCH_ROOT, "hf"))

# Outputs default *inside the repo* when the repo is already on persistent
# storage -- that is where they live today, and relocating them would orphan the
# existing llada/outputs/baseline and llada/outputs/lorta_vs_nalorta. Only a repo
# checked out somewhere ephemeral gets its outputs moved onto the volume.
_repo_is_persistent = os.path.abspath(REPO_ROOT).startswith(
    os.path.abspath(PERSIST_ROOT) + os.sep
)
OUTPUTS_ROOT = _setting(
    "LORTA_OUTPUTS_ROOT",
    os.path.join(LLADA_DIR, "outputs")
    if _repo_is_persistent
    else os.path.join(PERSIST_ROOT, "lorta-outputs"),
)

# Written by scripts/bootstrap.sh; the supervisor's preflight refuses to dispatch
# without it, because its absence means the venv was never verified against a GPU.
BOOTSTRAP_MARKER = os.path.join(SCRATCH_ROOT, ".lorta-bootstrap.json")


def short(path, base=None):
    """`path` relative to `base` when that is actually shorter, else absolute.

    Output directories used to be inside the repo, so a plain `relpath` always
    read as `outputs/<config>/<run>`. With OUTPUTS_ROOT on a separate volume the
    same call produces `../../../../mnt/volume/...`, which is worse than the
    absolute path in every way. Keeps existing `summary.json` entries unchanged
    for the in-repo layout.
    """
    base = base or LLADA_DIR
    relative = os.path.relpath(path, base)
    return path if relative.startswith(os.pardir) else relative


def describe():
    """The resolved roots, for printing and for `metadata.json`."""
    return {
        "repo_root": REPO_ROOT,
        "persist_root": PERSIST_ROOT,
        "scratch_root": SCRATCH_ROOT,
        "outputs_root": OUTPUTS_ROOT,
        "venv": VENV_DIR,
        "hf_home": HF_HOME,
    }


def format_description(indent="  "):
    return "\n".join(
        f"{indent}{k:<13} {v}" for k, v in describe().items()
    )


if __name__ == "__main__":
    print(format_description(indent=""))
