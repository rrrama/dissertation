#!/usr/bin/env python3
"""Config-driven launcher for LoRTA training / benchmarking runs.

A single YAML experiment config describes a run (or a *sweep* of runs). Any
list-valued field is expanded into the cartesian product of runs -- `seed` is
not special, it is just another hyperparameter that can be listed. For each
point in the product this script:

  1. creates ``<OUTPUTS_ROOT>/<config_name>/<run>/``,
  2. writes a frozen, fully-resolved per-run ``config.yaml`` (no lists) and a
     ``metadata.json`` (git hash, timestamp, dataset, versions, storage roots),
  3. generates one ``job_<mode>.sh`` per phase and hands the lot to
     ``local_runner``, which runs them across the box's GPUs and blocks until
     they finish.

Each per-run job runs ``torchrun train.py --config <frozen config.yaml>``
(train mode) or ``torchrun train.py --mode benchmark ...`` (benchmark mode).

``--mode all`` gives each run *two phases in one GPU slot*: the benchmark starts
only if that run's own training exits 0. That is what the slurm ``afterok``
dependency used to do, so a sweep still goes from config to benchmarked adapters
without intervention.

**This blocks.** There is no queue to submit into any more -- run it under tmux::

    tmux new -s tier0
    python batch_train.py --mode all --config configs/tier0_lr.yaml

Re-running a config is safe and is also how you recover from a dead supervisor:
runs that already have a saved adapter are skipped, and a run killed mid-training
resumes from its last checkpoint. Pass ``--overwrite`` to retrain (and clobber)
finished runs; that also deletes each restarted run's ``benchmark*.json`` and
``checkpoint-*`` directories, so ``summary.json`` reads as pending rather than
reporting the superseded adapter's accuracy -- and so the retrain starts from
scratch instead of silently resuming the run you asked to discard.

``tuning_type: none`` marks an untuned-baseline run: no adapter, nothing to
train, benchmark only (train mode materialises its run directory and runs
nothing; ``all``/``benchmark`` run its benchmark unconditionally). Listing it
alongside real adapters -- ``tuning_type: ["none", "nara"]`` -- puts the base
model's score in the same ``summary.json`` under identical decoding settings.

Usage:
    python batch_train.py --mode train      --config configs/run001_baseline.yaml
    python batch_train.py --mode benchmark  --config configs/run001_baseline.yaml
    python batch_train.py --mode all        --config configs/run001_baseline.yaml
    python batch_train.py --mode summary    --config configs/run001_baseline.yaml
    python batch_train.py --mode all --config ... --dry-run   # plan only

This script depends on PyYAML + the stdlib only; it never imports
torch/transformers, so a config error surfaces in a second rather than after a
model load. See SLURM_MIGRATION.md for the design.
"""

import argparse
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml

import local_runner
import paths
from local_runner import Job, Phase

LLADA_DIR = paths.LLADA_DIR
REPO_DIR = paths.REPO_ROOT
PEFT_DIR = paths.PEFT_DIR
OUTPUTS_ROOT = paths.OUTPUTS_ROOT

# `tuning_type` value marking an untuned-baseline run: no adapter, benchmark
# only. Kept in sync with train.py's NO_ADAPTER by hand rather than imported --
# this module deliberately avoids importing train.py (and hence torch) so it can
# run on a login node.
NO_ADAPTER = "none"

# Keys that meant something to slurm and mean nothing now. They are left in the
# configs on purpose: those files are the experimental record, and deleting the
# keys would be a diff across every one of them for no behavioural gain. They are
# still written into the frozen config (harmless -- train.py ignores keys it does
# not recognise) and reported once per invocation so nobody tunes a dead knob.
#
# `nproc_per_node` is NOT here: it survives, and now means "GPUs this run holds
# out of the box's pool" rather than a request to a scheduler with a whole cluster
# behind it. See _run_settings.
RETIRED_KEYS = {
    "gres",
    "partition",
    "cpus_per_task",
    "job_name",
    "sbatch_time",
    "sbatch_time_benchmark",
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
    # Tier 0 sweeps these, and the unabbreviated fallback would name a run directory
    # `eta0.0_head_learning_rate0.0003_learning_rate0.0001` -- long enough to trip the
    # 80-character guard in `run_name` and fall back to `run_003`, which is exactly the
    # comparability §8's naming convention is trying to preserve.
    "eta": "eta",
    "head_learning_rate": "hlr",
    "fourier_m": "m",
}

JOB_TEMPLATE = """#!/bin/bash
#
# {job_name}
#
# Generated by batch_train.py -- do not edit; re-running batch_train.py rewrites
# it. Runnable by hand exactly as the supervisor runs it:
#
#     bash {script_name}                      # uses the defaults below
#     CUDA_VISIBLE_DEVICES=1 bash {script_name}
#
# local_runner overrides CUDA_VISIBLE_DEVICES, LORTA_RDZV_PORT and
# OMP_NUM_THREADS at dispatch; everything else is baked in here so this file is a
# faithful record of the run.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{default_devices}}}"
export LORTA_RDZV_PORT="${{LORTA_RDZV_PORT:-29500}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"

# Storage roots, venv, HF_HOME, WANDB_MODE. `set +u` because some venv activate
# scripts touch $PS1 unguarded, which is fatal under `set -u`.
set +u
source {repo_dir}/scripts/env.sh
set -u

# `llada/`, not the repo root: the root holds a legacy gsm8k/ directory and
# datasets.load_dataset resolves local paths before the hub, so from there this
# dies with "No (supported) data files found in gsm8k".
cd {llada_dir}

# Not `--standalone`: it picks the c10d rendezvous endpoint itself, and which port
# depends on the torch version -- older releases pin 29400, which two concurrent
# jobs on one box collide over. One job per node made that unreachable under
# slurm. Asking for c10d explicitly on a port local_runner probed is equivalent
# and version-independent.
exec torchrun \\
    --nnodes=1 \\
    --nproc_per_node={nproc_per_node} \\
    --rdzv-backend=c10d \\
    --rdzv-endpoint=localhost:"$LORTA_RDZV_PORT" \\
    --rdzv-id={rdzv_id} \\
    train.py {mode_flag}--config {config_path} --output_dir {run_dir}
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
    """Library versions, read from the bootstrap marker.

    Read from `.lorta-bootstrap.json` rather than by importing torch here: this
    module's whole reason for staying import-light is that a config error should
    surface in a second, not after a multi-second CUDA import. The marker is
    written by scripts/bootstrap.sh from the venv every job actually runs in, so
    it is the same answer -- and unlike the old login-node import, it is not
    `null` for everything. Falls back to importing if the marker is missing.
    """
    try:
        with open(paths.BOOTSTRAP_MARKER) as f:
            recorded = json.load(f).get("versions")
        if recorded:
            return recorded
    except (OSError, json.JSONDecodeError):
        pass

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
        # Which disks this run used. A result that cannot be found later is
        # usually a root that resolved differently than assumed.
        "roots": paths.describe(),
    }


# --------------------------------------------------------------------------- #
# Slurm job generation                                                        #
# --------------------------------------------------------------------------- #


def _run_settings(run_cfg, n_available=None):
    """Resolve how many GPUs this run holds.

    `nproc_per_node` used to be a request to a scheduler with a cluster behind it;
    it is now a claim on a pool of 2-4 devices, so the default drops from 3 to 1
    and `local_runner.preflight` refuses a config asking for more than exist
    rather than leaving it unschedulable forever.

    ``nproc_per_node: all`` means every GPU in the pool. That is the right setting
    for a config that expands to a *single* run -- `baseline.yaml` -- where there
    is nothing to run alongside it and splitting the decode across all devices is
    pure speedup. It is the wrong setting for a sweep with more runs than GPUs:
    scaling within a run is imperfect, so N runs at 1 GPU each finish no later
    than N runs serialised across N GPUs, and usually sooner. It also keeps these
    configs from hardcoding a device count the next box will not have.

    Benchmark sampling cannot be split within a question, but it is embarrassingly
    parallel *across* questions: each rank samples its own shard of the GSM8K test
    set (see diffusion_evaluate), so benchmarks use the same GPU count as training.

    There is no wall-clock budget any more, which retires `sbatch_time` and the
    whole class of failure where a 6-epoch run was killed at exactly its 8 h
    reservation. Training that overruns now simply keeps running, and a run that
    dies for any other reason resumes from its last checkpoint.
    """
    requested = run_cfg.get("nproc_per_node", 1)
    if isinstance(requested, str) and requested.strip().lower() == "all":
        if n_available is None:
            n_available = len(local_runner.visible_devices())
        # `or 1` so that a dry run on a machine with no GPUs still writes a
        # readable plan instead of dividing the world by zero.
        return {"nproc_per_node": n_available or 1}
    return {"nproc_per_node": int(requested)}


def write_job_script(
    mode, run_cfg, config_name, name, config_path, run_dir, log_dir, n_available=None
):
    """Write the per-run launch script for one phase. Returns its path.

    Same role, signature and call sites as the old `write_sbatch`; the artifact
    just moved from `job_<mode>.sbatch` to `job_<mode>.sh`.
    """
    settings = _run_settings(run_cfg, n_available)
    job_name = run_cfg.get("job_name") or f"{config_name}-{name}-{mode}"
    script_name = f"job_{mode}.sh"
    script = JOB_TEMPLATE.format(
        job_name=job_name,
        script_name=script_name,
        # Only a fallback for running this script by hand; local_runner passes the
        # devices it actually allocated.
        default_devices=",".join(str(i) for i in range(settings["nproc_per_node"])),
        nproc_per_node=settings["nproc_per_node"],
        rdzv_id=f"{config_name}-{name}-{mode}",
        repo_dir=REPO_DIR,
        llada_dir=LLADA_DIR,
        mode_flag="" if mode == "train" else "--mode benchmark ",
        config_path=config_path,
        run_dir=run_dir,
    )
    script_path = os.path.join(run_dir, script_name)
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def _report_retired_keys(config):
    """Say once that the slurm knobs in this config no longer do anything."""
    present = sorted(RETIRED_KEYS & set(config))
    if present:
        print(
            f"[note] ignoring retired slurm key(s): {', '.join(present)} "
            f"(no scheduler any more; see SLURM_MIGRATION.md §1.4)"
        )


# --------------------------------------------------------------------------- #
# Modes                                                                       #
# --------------------------------------------------------------------------- #


def _is_baseline(run_cfg):
    """True for the untuned baseline run (`tuning_type: none`) -- see NO_ADAPTER."""
    return run_cfg.get("tuning_type") == NO_ADAPTER


def _is_trained(run_dir):
    """A run counts as trained once train.py has saved an adapter into it."""
    return os.path.exists(os.path.join(run_dir, "adapter_config.json"))


def _is_benchmarkable(run_cfg, run_dir):
    """Can this run be benchmarked now?

    A baseline run always can: it has no adapter to wait for. Everything else
    needs train.py to have saved one.
    """
    return _is_baseline(run_cfg) or _is_trained(run_dir)


def _benchmark_result_files(run_dir):
    """Every benchmark result in a run directory.

    One file per benchmark, so dev accuracy, test accuracy and `eval_loss_frozen` on
    the same adapter stop overwriting each other. `gsm8k_accuracy` keeps the original
    `benchmark.json` name and schema (O5); the rest are `benchmark_{name}.json`.
    """
    return sorted(glob.glob(os.path.join(run_dir, "benchmark*.json")))


# The fields each benchmark contributes to its `summary.json` entry. `gsm8k_accuracy`
# lists `accuracy` alone, so its entries are byte-identical to what they were before
# other benchmarks existed. An unregistered benchmark still gets an entry -- name and
# status -- rather than being dropped from the summary silently.
BENCHMARK_SUMMARY_FIELDS = {
    "gsm8k_accuracy": ("accuracy",),
    "eval_loss_frozen": ("eval_loss_frozen", "per_t", "adapter_params"),
}


def _clear_stale_results(run_dir, dry_run):
    """Drop what `--overwrite` has invalidated: benchmark results and checkpoints.

    **Benchmark results.** train.py only rewrites those when a benchmark
    *finishes*, hours after the training it depends on. Until then the old result
    sits next to an adapter that is being replaced, and `write_summary` reports it
    as `status: done` with the previous accuracy -- indefinitely, if the rerun
    never lands. Removing it up front makes the run read as `pending`.

    **Checkpoints.** train.py now resumes from `checkpoint-*` when one is present
    (SLURM_MIGRATION.md §5). Leaving them behind would make `--overwrite` silently
    *resume* the run it was asked to discard -- the exact thing the flag exists to
    prevent, and invisible in the results afterwards. They must go in the same pass.

    Not called without `--overwrite`: that is the flag that says the run's outputs
    are expendable. A dry run only says what it would remove.
    """
    stale = _benchmark_result_files(run_dir)
    stale += sorted(glob.glob(os.path.join(run_dir, "checkpoint-*")))
    for path in stale:
        rel = paths.short(path)
        if dry_run:
            print(f"  [overwrite] would remove stale {rel}")
            continue
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        print(f"  [overwrite] removed stale {rel}")


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


def _materialise_run(exp_dir, name, run_cfg, varying):
    """Create the run directory and write its frozen config + metadata."""
    run_dir, log_dir = _prepare_run_dir(exp_dir, name)
    config_path = _freeze_config(run_cfg, run_dir)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(build_metadata(run_cfg, varying), f, indent=2)
    return run_dir, log_dir, config_path


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
            "run_dir": paths.short(info["run_dir"]),
            "varying": info["varying"],
        }
        result_files = _benchmark_result_files(info["run_dir"])
        if not result_files:
            entry["status"] = "pending"
            summary["runs"].append(entry)
            continue
        # One entry per (run, benchmark): the same adapter is scored on dev accuracy,
        # test accuracy and eval_loss_frozen, and collapsing those into one row was
        # what made them overwrite each other in the first place.
        for bench_path in result_files:
            bench_entry = dict(entry)
            try:
                with open(bench_path) as f:
                    bench = json.load(f)
                name = bench.get("benchmark")
                bench_entry["benchmark"] = name
                for key in BENCHMARK_SUMMARY_FIELDS.get(name, ()):
                    bench_entry[key] = bench.get(key)
                bench_entry["status"] = "done"
            except (OSError, json.JSONDecodeError):
                bench_entry["status"] = "unreadable"
            summary["runs"].append(bench_entry)

    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")


def run_train_mode(config, config_name, dry_run, overwrite, queue_opts):
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Expanded to {len(runs)} run(s) under {exp_dir}")
    n_available = len(queue_opts["devices"])
    runs_info = []
    jobs = []
    skipped = []
    baselines = []
    for idx, ((run_cfg, varying), name) in enumerate(zip(runs, names)):
        run_dir = os.path.join(exp_dir, name)
        # A finished run leaves an adapter behind. Re-submitting would clobber
        # it (train.py saves into the same dir), so skip unless asked not to.
        if _is_trained(run_dir) and not overwrite:
            skipped.append(name)
            runs_info.append({"name": name, "run_dir": run_dir, "varying": varying})
            continue

        if _is_baseline(run_cfg):
            # Nothing to train, but still materialise the run dir: `--mode
            # benchmark` reads the frozen config.yaml from it.
            run_dir, _, _ = _materialise_run(exp_dir, name, run_cfg, varying)
            baselines.append(name)
            runs_info.append({"name": name, "run_dir": run_dir, "varying": varying})
            continue

        run_dir, log_dir, config_path = _materialise_run(
            exp_dir, name, run_cfg, varying
        )

        print(f"[{idx + 1}/{len(runs)}] {name}")
        if overwrite:
            # The adapter this result was measured on is about to be replaced.
            _clear_stale_results(run_dir, dry_run)
        script = write_job_script(
            "train", run_cfg, config_name, name, config_path, run_dir, log_dir,
            n_available,
        )
        jobs.append(
            Job(
                name=name,
                run_dir=run_dir,
                phases=[Phase("train", script)],
                n_gpus=_run_settings(run_cfg, n_available)["nproc_per_node"],
            )
        )
        runs_info.append({"name": name, "run_dir": run_dir, "varying": varying})

    if skipped:
        print(
            f"[skip] {len(skipped)}/{len(runs)} run(s) already trained; pass "
            f"--overwrite to retrain them: {', '.join(skipped)}"
        )
    if baselines:
        print(
            f"[baseline] {len(baselines)}/{len(runs)} run(s) have "
            f"tuning_type: {NO_ADAPTER} and need no training; benchmark them with "
            f"--mode benchmark: {', '.join(baselines)}"
        )

    results = local_runner.run_queue(jobs, dry_run=dry_run, **queue_opts)
    write_summary(exp_dir, runs_info, "train")
    return results


def run_all_mode(config, config_name, dry_run, overwrite, queue_opts):
    """Run train + benchmark for every run, chained within one GPU slot.

    Unlike `--mode benchmark`, this does not require an adapter to exist on disk
    yet: the benchmark phase is gated on its own training exiting 0, so a fresh
    sweep goes from config to results in one command.
    """
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Expanded to {len(runs)} run(s) under {exp_dir}")
    n_available = len(queue_opts["devices"])
    runs_info = []
    jobs = []
    reused = []
    for idx, ((run_cfg, varying), name) in enumerate(zip(runs, names)):
        run_dir = os.path.join(exp_dir, name)
        info = {"name": name, "run_dir": run_dir, "varying": varying}
        print(f"[{idx + 1}/{len(runs)}] {name}")

        phases = []
        if _is_baseline(run_cfg):
            # No adapter to train, so the benchmark is unconditional -- nothing to
            # gate it on, and `--overwrite` has nothing to redo.
            print(f"  [baseline] tuning_type: {NO_ADAPTER}; benchmarking base model")
            run_dir, log_dir, config_path = _materialise_run(
                exp_dir, name, run_cfg, varying
            )
        elif _is_trained(run_dir) and not overwrite:
            # Adapter already on disk: nothing to train, so the run is a benchmark
            # phase on its own.
            reused.append(name)
            log_dir = os.path.join(run_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
        else:
            run_dir, log_dir, config_path = _materialise_run(
                exp_dir, name, run_cfg, varying
            )
            phases.append(
                Phase(
                    "train",
                    write_job_script(
                        "train", run_cfg, config_name, name, config_path, run_dir,
                        log_dir, n_available,
                    ),
                )
            )

        if overwrite:
            # Reachable from the baseline branch (its benchmark is unconditional)
            # and from the retrain branch; the "reused" branch above is entered
            # only when `overwrite` is false, so a kept adapter keeps its result.
            _clear_stale_results(run_dir, dry_run)

        # Appended after the train phase, so local_runner runs it only if training
        # exits 0 -- the `afterok` dependency, minus the scheduler.
        phases.append(
            Phase(
                "benchmark",
                write_job_script(
                    "benchmark", run_cfg, config_name, name, config_path, run_dir,
                    log_dir, n_available,
                ),
            )
        )
        jobs.append(
            Job(
                name=name,
                run_dir=run_dir,
                phases=phases,
                n_gpus=_run_settings(run_cfg, n_available)["nproc_per_node"],
            )
        )
        runs_info.append(info)

    if reused:
        print(
            f"[skip] {len(reused)}/{len(runs)} run(s) already trained; benchmarked "
            f"against the existing adapter (pass --overwrite to retrain): "
            f"{', '.join(reused)}"
        )

    results = local_runner.run_queue(jobs, dry_run=dry_run, **queue_opts)
    write_summary(exp_dir, runs_info, "all")
    return results


def run_summary_mode(config, config_name):
    """Refresh summary.json from whatever results are on disk. Runs nothing."""
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)
    if not os.path.isdir(exp_dir):
        print(f"[error] no output directory for this config: {exp_dir}")
        return
    runs_info = [
        {"name": name, "run_dir": os.path.join(exp_dir, name), "varying": varying}
        for (_, varying), name in zip(runs, names)
    ]
    write_summary(exp_dir, runs_info, "summary")


def run_benchmark_mode(config, config_name, dry_run, queue_opts):
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)

    # Check all runs exist / are trained before launching anything.
    n_available = len(queue_opts["devices"])
    runs_info = []
    jobs = []
    missing = []
    for (run_cfg, varying), name in zip(runs, names):
        run_dir = os.path.join(exp_dir, name)
        ready = _is_benchmarkable(run_cfg, run_dir)
        if not ready:
            missing.append(name)
        runs_info.append(
            {
                "name": name,
                "run_dir": run_dir,
                "varying": varying,
                "run_cfg": run_cfg,
                "ready": ready,
            }
        )

    if missing:
        print(
            f"[warn] {len(missing)}/{len(runs)} run(s) are not trained yet and "
            f"will be skipped: {', '.join(missing)}"
        )

    for idx, info in enumerate(runs_info):
        if not info["ready"]:
            continue
        run_dir = info["run_dir"]
        print(f"[{idx + 1}/{len(runs)}] {info['name']}")
        if _is_baseline(info["run_cfg"]):
            # A baseline run may never have been through train mode, so its run
            # dir and frozen config might not exist yet. Writing them here is
            # idempotent and keeps `--mode benchmark` usable on its own.
            print(f"  [baseline] tuning_type: {NO_ADAPTER}; benchmarking base model")
            run_dir, log_dir, config_path = _materialise_run(
                exp_dir, info["name"], info["run_cfg"], info["varying"]
            )
            info["run_dir"] = run_dir
        else:
            log_dir = os.path.join(run_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
        script = write_job_script(
            "benchmark",
            info["run_cfg"],
            config_name,
            info["name"],
            config_path,
            run_dir,
            log_dir,
            n_available,
        )
        jobs.append(
            Job(
                name=info["name"],
                run_dir=run_dir,
                phases=[Phase("benchmark", script)],
                n_gpus=_run_settings(info["run_cfg"], n_available)["nproc_per_node"],
            )
        )

    results = local_runner.run_queue(jobs, dry_run=dry_run, **queue_opts)
    # Refresh the aggregate; benchmark*.json files appear as jobs finish, and the
    # queue has drained by the time we get here, so this picks up everything that
    # landed in this invocation.
    write_summary(exp_dir, runs_info, "benchmark")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["train", "benchmark", "all", "summary"],
        default="train",
        help=(
            "train/benchmark run one job per run; 'all' runs both, with each "
            "benchmark gated on its own training exiting 0; 'summary' only "
            "refreshes summary.json from results on disk."
        ),
    )
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument(
        "--dry-run",
        "--no-submit",
        dest="dry_run",
        action="store_true",
        help=(
            "Materialise run dirs and job scripts, print the dispatch plan, but "
            "launch nothing. `--no-submit` is the old name and still works."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Retrain runs that already have a saved adapter, overwriting them, "
            "and delete the benchmark*.json and checkpoint-* of every run "
            "restarted this way (without which the retrain would resume the run "
            "it was asked to discard). By default such runs are skipped, so "
            "re-running a config resumes an incomplete sweep instead of "
            "clobbering finished LoRTAs."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated GPU ids to schedule onto, e.g. '0,2'. Defaults to "
            "every visible device (CUDA_VISIBLE_DEVICES, else all of them)."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=(
            "Cap on simultaneously running jobs. Defaults to GPU-limited. Lower it "
            "if host RAM, not GPU memory, is the binding constraint."
        ),
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=local_runner.DEFAULT_STAGGER_SECONDS,
        help=(
            "Seconds to leave between launches, so that concurrent 8B model loads "
            "do not spike host RAM together. 0 disables."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    print(paths.format_description())
    _report_retired_keys(config)

    # Resolved concretely here rather than inside run_queue, because
    # `nproc_per_node: all` has to mean "all of the pool this invocation is using",
    # which `--gpus` can narrow.
    devices = (
        [d.strip() for d in args.gpus.split(",") if d.strip()]
        if args.gpus
        else local_runner.visible_devices()
    )
    queue_opts = {
        "devices": devices,
        "max_concurrent": args.max_concurrent,
        "stagger_seconds": args.stagger,
    }

    if args.mode == "summary":
        run_summary_mode(config, config_name)
        return 0

    runner = {
        "train": run_train_mode,
        "all": run_all_mode,
    }.get(args.mode)
    if runner is not None:
        results = runner(config, config_name, args.dry_run, args.overwrite, queue_opts)
    else:
        results = run_benchmark_mode(config, config_name, args.dry_run, queue_opts)

    # Non-zero if any run failed, so a wrapper script or `&&` chain can tell.
    return local_runner.exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
