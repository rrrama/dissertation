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

``--mode all`` submits both per run and chains them with a slurm dependency
(``afterok``), so a sweep goes from config to benchmarked adapters without
intervention: every run trains in parallel and each benchmark starts as soon as
*its own* training exits 0.

Re-running a config in train mode is safe: runs that already have a saved
adapter are skipped, so only the missing points of a sweep are submitted. Pass
``--overwrite`` to retrain (and clobber) finished runs; that also deletes each
resubmitted run's ``benchmark.json``, so ``summary.json`` shows it as pending
rather than reporting the superseded adapter's accuracy for the ~20 h until the
new result lands.

``tuning_type: none`` marks an untuned-baseline run: no adapter, nothing to
train, benchmark only (train mode materialises its run directory and submits
nothing; ``all``/``benchmark`` submit its benchmark with no dependency). Listing
it alongside real adapters -- ``tuning_type: ["none", "nara"]`` -- puts the base
model's score in the same ``summary.json`` under identical decoding settings.

Usage:
    python batch_train.py --mode train      --config configs/run001_baseline.yaml
    python batch_train.py --mode benchmark  --config configs/run001_baseline.yaml
    python batch_train.py --mode all        --config configs/run001_baseline.yaml
    python batch_train.py --mode summary    --config configs/run001_baseline.yaml

This script only depends on PyYAML + the stdlib so it can run on a login node
without importing torch/transformers.
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import yaml

LLADA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(LLADA_DIR)
PEFT_DIR = os.path.join(REPO_DIR, "peft")
OUTPUTS_ROOT = os.path.join(LLADA_DIR, "outputs")

# `tuning_type` value marking an untuned-baseline run: no adapter, benchmark
# only. Kept in sync with train.py's NO_ADAPTER by hand rather than imported --
# this module deliberately avoids importing train.py (and hence torch) so it can
# run on a login node.
NO_ADAPTER = "none"

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

# Installed explicitly because the peft install below is --no-deps.
pip install wandb pyyaml
pip install -e {peft_dir} --no-deps

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


def _slurm_settings(run_cfg, mode):
    """Resolve slurm/GPU settings from the config, with plan defaults."""
    nproc = int(run_cfg.get("nproc_per_node", 3))
    gres = run_cfg.get("gres")
    if gres is None:
        gres = f"gpu:lovelace_l40:{nproc}"
    # Benchmark sampling cannot be split within a question, but it is embarrassingly
    # parallel *across* questions: each rank samples its own shard of the GSM8K test
    # set (see diffusion_evaluate), so benchmarks use the same GPU count as training.
    return {
        "nproc_per_node": nproc,
        "gres": gres,
        "partition": run_cfg.get("partition", "gpu"),
        "cpus_per_task": run_cfg.get("cpus_per_task", 3),
        "sbatch_time": run_cfg.get("sbatch_time", "12:00:00"),
        "sbatch_mem": run_cfg.get("sbatch_mem", "120G"),
    }


def _run_command(mode, config_path, run_dir, slurm):
    mode_flag = "" if mode == "train" else "--mode benchmark "
    return (
        f"torchrun --standalone --nproc_per_node={slurm['nproc_per_node']} "
        f"train.py {mode_flag}--config {config_path} --output_dir {run_dir}"
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
        peft_dir=PEFT_DIR,
        run_cmd=_run_command(mode, config_path, run_dir, slurm),
    )
    sbatch_path = os.path.join(run_dir, f"job_{mode}.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(script)
    return sbatch_path


def submit(sbatch_path, no_submit, after_job=None):
    """Submit a job, optionally gated on another job finishing successfully.

    Returns the slurm job id (as a string) so callers can chain dependencies,
    or None if nothing was submitted.

    ``after_job`` adds ``--dependency=afterok:<id>``: the job stays queued until
    the dependency exits 0, and ``--kill-on-invalid-dep`` cancels it outright if
    the dependency fails, rather than leaving it pending forever.
    """
    cmd = ["sbatch"]
    if after_job:
        cmd += [f"--dependency=afterok:{after_job}", "--kill-on-invalid-dep=yes"]
    cmd.append(sbatch_path)

    if no_submit:
        print(f"  [no-submit] would run: {' '.join(cmd)}")
        return None
    if shutil.which("sbatch") is None:
        print(f"  [warn] sbatch not found; skipping submit of {sbatch_path}")
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [error] sbatch failed: {result.stderr.strip()}")
        return None
    job_line = result.stdout.strip()
    print(f"  submitted: {job_line}")
    match = re.search(r"(\d+)", job_line)
    return match.group(1) if match else None


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


def _clear_stale_results(run_dir, no_submit):
    """Drop a previous run's benchmark.json, which `--overwrite` has invalidated.

    train.py only rewrites that file when a benchmark *finishes*, which is ~20 h
    after the training job it depends on. Until then the old result sits next to
    an adapter that is being replaced, and `write_summary` reports it as
    `status: done` with the previous accuracy -- indefinitely, if the resubmitted
    job never lands. Removing it up front makes the run read as `pending`, which
    is what it is.

    Not called without `--overwrite`: that is the flag that says the run's outputs
    are expendable. A dry run (`--no-submit`) only says what it would remove.
    """
    bench_path = os.path.join(run_dir, "benchmark.json")
    if not os.path.exists(bench_path):
        return
    rel = os.path.relpath(bench_path, LLADA_DIR)
    if no_submit:
        print(f"  [overwrite] would remove stale {rel}")
        return
    os.remove(bench_path)
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
            _clear_stale_results(run_dir, no_submit)
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
    if baselines:
        print(
            f"[baseline] {len(baselines)}/{len(runs)} run(s) have "
            f"tuning_type: {NO_ADAPTER} and need no training; benchmark them with "
            f"--mode benchmark: {', '.join(baselines)}"
        )

    write_summary(exp_dir, runs_info, "train")


def run_all_mode(config, config_name, no_submit, overwrite):
    """Submit train + benchmark for every run, chained by a slurm dependency.

    Unlike `--mode benchmark`, this does not require an adapter to exist on disk
    yet: the `afterok` dependency is what gates the benchmark, so a fresh sweep
    goes from config to results in one command.
    """
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Expanded to {len(runs)} run(s) under {exp_dir}")
    runs_info = []
    reused = []
    for idx, ((run_cfg, varying), name) in enumerate(zip(runs, names)):
        run_dir = os.path.join(exp_dir, name)
        info = {"name": name, "run_dir": run_dir, "varying": varying}
        print(f"[{idx + 1}/{len(runs)}] {name}")

        if _is_baseline(run_cfg):
            # No adapter to train, so the benchmark is unconditional -- no
            # dependency, and `--overwrite` has nothing to redo.
            print(f"  [baseline] tuning_type: {NO_ADAPTER}; benchmarking base model")
            run_dir, log_dir, config_path = _materialise_run(
                exp_dir, name, run_cfg, varying
            )
            train_job = None
        elif _is_trained(run_dir) and not overwrite:
            # Adapter already on disk: nothing to train, so the benchmark goes
            # straight onto the queue with no dependency.
            reused.append(name)
            log_dir = os.path.join(run_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
            train_job = None
        else:
            run_dir, log_dir, config_path = _materialise_run(
                exp_dir, name, run_cfg, varying
            )
            train_sbatch = write_sbatch(
                "train", run_cfg, config_name, name, config_path, run_dir, log_dir
            )
            train_job = submit(train_sbatch, no_submit)
            if no_submit:
                # Keep the dry run readable: show the dependency that a real
                # submit would attach.
                train_job = "<train_job_id>"
            elif train_job is None:
                print("  [error] train job not submitted; skipping its benchmark")
                runs_info.append(info)
                continue

        if overwrite:
            # Reachable from the baseline branch (its benchmark is unconditional)
            # and from the retrain branch; the "reused" branch above is entered
            # only when `overwrite` is false, so a kept adapter keeps its result.
            _clear_stale_results(run_dir, no_submit)
        bench_sbatch = write_sbatch(
            "benchmark", run_cfg, config_name, name, config_path, run_dir, log_dir
        )
        submit(bench_sbatch, no_submit, after_job=train_job)
        runs_info.append(info)

    if reused:
        print(
            f"[skip] {len(reused)}/{len(runs)} run(s) already trained; benchmarked "
            f"against the existing adapter (pass --overwrite to retrain): "
            f"{', '.join(reused)}"
        )

    write_summary(exp_dir, runs_info, "all")


def run_summary_mode(config, config_name):
    """Refresh summary.json from whatever results are on disk. Submits nothing."""
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


def run_benchmark_mode(config, config_name, no_submit):
    runs = expand_runs(config)
    names = _unique_names(runs)
    exp_dir = os.path.join(OUTPUTS_ROOT, config_name)

    # Check all runs exist / are trained before launching anything.
    runs_info = []
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
    parser.add_argument(
        "--mode",
        choices=["train", "benchmark", "all", "summary"],
        default="train",
        help=(
            "train/benchmark submit one job per run; 'all' submits both and "
            "chains the benchmark behind training with a slurm dependency; "
            "'summary' only refreshes summary.json from results on disk."
        ),
    )
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
            "Retrain runs that already have a saved adapter, overwriting them, "
            "and delete the benchmark.json of every run resubmitted this way. "
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
    elif args.mode == "all":
        run_all_mode(config, config_name, args.no_submit, args.overwrite)
    elif args.mode == "summary":
        run_summary_mode(config, config_name)
    else:
        run_benchmark_mode(config, config_name, args.no_submit)


if __name__ == "__main__":
    main()
