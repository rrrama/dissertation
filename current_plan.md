# Brief: LoRTA Training Run Management System

## Goal

Build a config-driven system for launching and tracking multiple LoRTA training
runs with different hyperparameters, on top of the existing `llada/train.py`.

A single entrypoint, **`llada/batch_train.py`**, runs in either `train` or
`benchmark` mode and takes a YAML config file on the command line:

```
python batch_train.py --mode train     --config configs/run001_baseline.yaml
python batch_train.py --mode benchmark  --config configs/run001_baseline.yaml
```

Config files are YAML. **Any list-valued field expands into a "tensor" of
runs** — the cartesian product across all list fields. `seed` is not special;
it is just another hyperparameter that can be listed like any other. Each point
in the product is one run.

## Architecture

### `batch_train.py` (new, single entrypoint)

1. Parse the experiment YAML.
2. Expand every list-valued field into the cartesian product of runs.
3. For each run, create `outputs/<config_name>/<run>/`, write a frozen
   per-run `config.yaml` (fully resolved, no lists) and a `metadata.json`
   (git hash, timestamp, dataset hash, versions).
4. **Generate and `sbatch`-submit one job per run** from a template derived
   from the existing `run_training.sbatch`. Runs go through the Slurm
   scheduler in parallel (one sbatch job each).
5. Each per-run job runs `torchrun train.py --config <frozen config.yaml>`.

### `train.py` (refactor existing)

- Extract the training body into an **importable function** that takes a
  resolved single-run config plus an explicit output directory.
- `__main__` reads one frozen `config.yaml` and calls that function.
- **Remove** the self-constructed `output_dir` path logic (currently
  `output_dir/expt_name/model/ep_N/lr_X/seed_N`) — the output directory is
  assigned by `batch_train.py` instead.

## Output structure

```
llada/
├── configs/
│   ├── run001_baseline.yaml
│   ├── run002_higher_lr.yaml
│   └── run003_rank32.yaml
├── outputs/
│   ├── run001_baseline/
│   │   ├── <run>/                    # one dir per point in the tensor
│   │   │   ├── config.yaml           # frozen, fully-resolved config for this run
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   ├── samples/              # sample generations, if applicable
│   │   │   └── metadata.json         # git hash, timestamp, dataset hash, versions
│   │   ├── <run>/
│   │   │   └── ...
│   │   └── summary.json              # aggregated results/notes across runs
│   ├── run002_higher_lr/
│   └── run003_rank32/
├── batch_train.py                    # single entrypoint (train / benchmark)
└── train.py                          # refactored: importable train fn + thin __main__
```

- **Run directory naming**: named by the varying hyperparameters
  (e.g. `lr5e-2_rank512`), falling back to indexed `run_000` names if names
  collide or become unwieldy.

## Train mode

Trains across different LoRTA architectures — `lora`, `lorta`, and `nalorta`
are already wired in `train.py`; keep the selection extensible. Final trained
weights and per-run info land in the `outputs/` structure above.

## Benchmark mode

Given a config file, check that all runs for the experiment have been trained,
then run the configured benchmark against each trained adapter and output a
usable format.

- For now, **reuse the existing `diffusion_evaluate` (GSM8K accuracy)**, gated
  behind a `benchmark:` config field with a single registered entry.
- Leave the registry seam in place so additional benchmarks can be plugged in
  later.

## Config-driven Slurm

- One sbatch job per run, generated from a template based on
  `run_training.sbatch`.
- Make GPU count config-driven: `nproc_per_node` and the `--gres` line come
  from the config, defaulting to the current value of `3`.

## Notes

Reuse as much of the original code as reasonably possible where helpful.
