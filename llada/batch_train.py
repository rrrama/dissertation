#!/usr/bin/env python3
"""Config-driven launcher for LoRTA training / benchmarking runs.

A single YAML experiment config describes a run (or a *sweep* of runs). Any
list-valued field is expanded into the cartesian product of runs -- `seed` is
not special, it is just another hyperparameter that can be listed. For each
point in the product this script:

  1. creates ``outputs/<config_name>/<run>/``,
  2. writes a frozen, fully-resolved per-run ``config.yaml`` (no lists) and a
     ``metadata.json`` (git hash, timestamp, dataset, versions),
  3. generates one ``job.sbatch`` from a template based on
     ``run_training.sbatch`` and submits it with ``sbatch`` (one job per run,
     scheduled in parallel).

Each per-run job runs ``torchrun train.py --config <frozen config.yaml>``
(train mode) or ``python train.py --mode benchmark ...`` (benchmark mode).

Re-running a config in train mode is safe: runs that already have a saved
adapter are skipped, so only the missing points of a sweep are submitted. Pass
``--overwrite`` to retrain (and clobber) finished runs.

Usage:
    python batch_train.py --mode train     --config configs/run001_baseline.yaml
    python batch_train.py --mode benchmark  --config configs/run001_baseline.yaml

This script only depends on PyYAML + the stdlib so it can run on a login node
without importing torch/transformers.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml

LLADA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_ROOT = os.path.join(LLADA_DIR, "outputs")

# Keys that configure the batch/slurm layer rather than the training run. They
# are still written into the frozen config (harmless -- train.py ignores keys it
# does not recognise) but are consumed here for job generation.
SLURM_META_KEYS = {
    "nproc_per_node",
    "gres",
    "partition",
    "cpus_per_task",
    "job_name",
    "sbatch_time",
    "sbatch_mem",
}

# Shorter labels for common sweep fields when naming run directories.
NAME_ABBREV = {
    "learning_rate": "lr",
    "num_train_epochs": "ep",
    "rank": "rank",
    "lora_alpha": "alpha",
    "seed": "seed",
    "tuning_type": "",
    "per_device_train_batch_size": "bs",
    "gradient_accumulation_steps": "ga",
}

SBATCH_TEMPLATE = """#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres={gres}
#SBATCH --time={sbatch_time}
#SBATCH --mem={sbatch_mem}
#SBATCH --output={log_dir}/slurm-%j.out
#SBATCH --error={log_dir}/slurm-%j.err

source /etc/profile.d/modules.sh
source $SHARE/u5751903/lorta_venv/bin/activate

export HF_HOME=$SHARE/u5751903/models/
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install wandb pyyaml
pip install -e $SHARE/u5751903/lorta/peft --no-deps

cd {llada_dir}
srun {run_cmd}
"""


# --------------------------------------------------------------------------- #
# Config expansion                                                            #
# --------------------------------------------------------------------------- #


def expand_runs(config):
    """Expand every list-valued field into the cartesian product of runs.

    Returns a list of ``(run_config, varying)`` tuples where ``run_config`` is
    a fully-resolved dict (no lists) and ``varying`` holds only the fields that
    differ across runs (used for naming).
    """
    list_fields = [k for k, v in config.items() if isinstance(v, list)]
    if not list_fields:
        return [(dict(config), {})]

    value_lists = [config[k] for k in list_fields]
    runs = []
    for combo in itertools.product(*value_lists):
        varying = dict(zip(list_fields, combo))
        run_cfg = dict(config)
        run_cfg.update(varying)
        runs.append((run_cfg, varying))
    return runs


def _format_value(value):
    text = str(value)
    for bad in ("/", " ", ":", "="):
        text = text.replace(bad, "-")
    return text


def run_name(varying, idx):
    """Name a run directory by its varying hyperparameters, else index it."""
    if not varying:
        return "run_000"
    parts = []
    for key in sorted(varying):
        abbrev = NAME_ABBREV.get(key, key)
        parts.append(f"{abbrev}{_format_value(varying[key])}")
    name = "_".join(parts)
    if not name or len(name) > 80:
        return f"run_{idx:03d}"
    return name


def _unique_names(runs):
    """Assign a unique directory name to each run, falling back to indices."""
    proposed = [run_name(varying, idx) for idx, (_, varying) in enumerate(runs)]
    if len(set(proposed)) == len(proposed):
        return proposed
    # Collision -> use indexed names for everything so they stay comparable.
    return [f"run_{idx:03d}" for idx in range(len(runs))]


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #


def _git_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=LLADA_DIR, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _versions():
    versions = {}
    for module in ("torch", "transformers", "peft", "datasets"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None
    return versions


def build_metadata(run_cfg, varying):
    return {
        "git_hash": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": run_cfg.get("data_name", "gsm8k"),
        # A true content hash requires materialising the dataset; record the
        # name here and leave a slot for a fingerprint captured at train time.
        "dataset_hash": None,
        "versions": _versions(),
        "varying": varying,
        "python": sys.version.split()[0],
    }


# --------------------------------------------------------------------------- #
# Slurm job generation                                                        #
# --------------------------------------------------------------------------- #


def _set_gpu_count(gres, count):
    """Return `gres` with its trailing GPU count replaced by `count`.

    Preserves the GPU type, e.g. ``gpu:lovelace_l40:3`` -> ``gpu:lovelace_l40:1``.
    """
    parts = gres.split(":")
    if parts and parts[-1].isdigit():
        parts[-1] = str(count)
        return ":".join(parts)
    return gres


def _slurm_settings(run_cfg, mode):
    """Resolve slurm/GPU settings from the config, with plan defaults."""
    nproc = int(run_cfg.get("nproc_per_node", 3))
    gres = run_cfg.get("gres")
    if gres is None:
        gres = f"gpu:lovelace_l40:{nproc}"
    # Benchmark (diffusion sampling) is single-GPU; do not reserve the full set,
    # but keep the configured GPU type by only shrinking the count.
    if mode == "benchmark":
        nproc = 1
        gres = _set_gpu_count(gres, 1)
    return {
        "nproc_per_node": nproc,
        "gres": gres,
        "partition": run_cfg.get("partition", "gpu"),
        "cpus_per_task": run_cfg.get("cpus_per_task", 3),
        "sbatch_time": run_cfg.get("sbatch_time", "12:00:00"),
        "sbatch_mem": run_cfg.get("sbatch_mem", "120G"),
    }


def _run_command(mode, config_path, run_dir, slurm):
    if mode == "train":
        return (
            f"torchrun --standalone --nproc_per_node={slurm['nproc_per_node']} "
            f"train.py --config {config_path} --output_dir {run_dir}"
        )
    return (
        f"python train.py --mode benchmark "
        f"--config {config_path} --output_dir {run_dir}"
    )


def write_sbatch(mode, run_cfg, config_name, name, config_path, run_dir, log_dir):
    slurm = _slurm_settings(run_cfg, mode)
    job_name = run_cfg.get("job_name") or f"{config_name}-{name}-{mode}"
    script = SBATCH_TEMPLATE.format(
        job_name=job_name,
        partition=slurm["partition"],
        cpus_per_task=slurm["cpus_per_task"],
        gres=slurm["gres"],
        sbatch_time=slurm["sbatch_time"],
        sbatch_mem=slurm["sbatch_mem"],
        log_dir=log_dir,
        llada_dir=LLADA_DIR,
        run_cmd=_run_command(mode, config_path, run_dir, slurm),
    )
    sbatch_path = os.path.join(run_dir, f"job_{mode}.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(script)
    return sbatch_path


def submit(sbatch_path, no_submit):
    if no_submit:
        print(f"  [no-submit] would run: sbatch {sbatch_path}")
        return None
    if shutil.which("sbatch") is None:
        print(f"  [warn] sbatch not found; skipping submit of {sbatch_path}")
        return None
    result = subprocess.run(
        ["sbatch", sbatch_path], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [error] sbatch failed: {result.stderr.strip()}")
        return None
    job_line = result.stdout.strip()
    print(f"  submitted: {job_line}")
    return job_line


# --------------------------------------------------------------------------- #
# Modes                                                                       #
# --------------------------------------------------------------------------- #


def _is_trained(run_dir):
    """A run counts as trained once train.py has saved an adapter into it."""
    return os.path.exists(os.path.join(run_dir, "adapter_config.json"))


def _prepare_run_dir(exp_dir, name):
    run_dir = os.path.join(exp_dir, name)
    log_dir = os.path.join(run_dir, "logs")
    for sub in ("logs", "checkpoints", "samples"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir, log_dir


def _freeze_config(run_cfg, run_dir):
    """Write the fully-resolved, list-free per-run config."""
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=True, default_flow_style=False)
    return config_path


def write_summary(exp_dir, runs_info, mode):
    """Aggregate run listing (+ any existing benchmark results) into summary.json."""
    summary = {
        "mode": mode,
        "updated": datetime.now(timezone.utc).isoformat(),
        "num_runs": len(runs_info),
        "runs": [],
    }
    for info in runs_info:
        entry = {
            "run": info["name"],
            "run_dir": os.path.relpath(info["run_dir"], LLADA_DIR),
            "varying": info["varying"],
        }
        bench_path = os.path.join(info["run_dir"], "benchmark.json")
        if os.path.exists(bench_path):
            try:
                with open(bench_path) as f:
                    bench = json.load(f)
                entry["benchmark"] = bench.get("benchmark")
                entry["accuracy"] = bench.get("accuracy")
                entry["status"] = "done"
            except (OSError, json.JSONDecodeError):
                entry["status"] = "unreadable"
        else:
            entry["status"] = "pending"
        summary["runs"].append(entry)

    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


def run_train_mode(config, config_name, no_submit, overwrite):
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Expanded to {len(runs)} run(s) under {exp_dir}")
    runs_info = []
    skipped = []
    for idx, ((run_cfg, varying), name) in enumerate(zip(runs, names)):
        run_dir = os.path.join(exp_dir, name)
        # A finished run leaves an adapter behind. Re-submitting would clobber
        # it (train.py saves into the same dir), so skip unless asked not to.
        if _is_trained(run_dir) and not overwrite:
            skipped.append(name)
            runs_info.append({"name": name, "run_dir": run_dir, "varying": varying})
            continue

        run_dir, log_dir = _prepare_run_dir(exp_dir, name)
        config_path = _freeze_config(run_cfg, run_dir)
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(build_metadata(run_cfg, varying), f, indent=2)

        print(f"[{idx + 1}/{len(runs)}] {name}")
        sbatch_path = write_sbatch(
            "train", run_cfg, config_name, name, config_path, run_dir, log_dir
        )
        submit(sbatch_path, no_submit)
        runs_info.append({"name": name, "run_dir": run_dir, "varying": varying})

    if skipped:
        print(
            f"[skip] {len(skipped)}/{len(runs)} run(s) already trained; pass "
            f"--overwrite to retrain them: {', '.join(skipped)}"
        )

    write_summary(exp_dir, runs_info, "train")


def run_benchmark_mode(config, config_name, no_submit):
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)

    # Check all runs exist / are trained before launching anything.
    runs_info = []
    missing = []
    for (run_cfg, varying), name in zip(runs, names):
        run_dir = os.path.join(exp_dir, name)
        trained = _is_trained(run_dir)
        if not trained:
            missing.append(name)
        runs_info.append(
            {
                "name": name,
                "run_dir": run_dir,
                "varying": varying,
                "run_cfg": run_cfg,
                "trained": trained,
            }
        )

    if missing:
        print(
            f"[warn] {len(missing)}/{len(runs)} run(s) are not trained yet and "
            f"will be skipped: {', '.join(missing)}"
        )

    for idx, info in enumerate(runs_info):
        if not info["trained"]:
            continue
        run_dir = info["run_dir"]
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        config_path = os.path.join(run_dir, "config.yaml")
        print(f"[{idx + 1}/{len(runs)}] {info['name']}")
        sbatch_path = write_sbatch(
            "benchmark",
            info["run_cfg"],
            config_name,
            info["name"],
            config_path,
            run_dir,
            log_dir,
        )
        submit(sbatch_path, no_submit)

    # Refresh the aggregate; benchmark.json files appear as jobs finish, so
    # re-running this mode later picks up completed results.
    write_summary(exp_dir, runs_info, "benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "benchmark"], default="train")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Generate run dirs / sbatch scripts but do not call sbatch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Retrain runs that already have a saved adapter, overwriting them. "
            "By default such runs are skipped, so re-running a config resumes "
            "an incomplete sweep instead of clobbering finished LoRTAs."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    if args.mode == "train":
        run_train_mode(config, config_name, args.no_submit, args.overwrite)
    else:
        run_benchmark_mode(config, config_name, args.no_submit)


if __name__ == "__main__":
    main()
