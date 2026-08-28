# Migration — slurm cluster → rented GPU instance

How `batch_train.py` stops being a slurm submitter and becomes a local supervisor,
with the smallest possible blast radius on everything around it.

Scope: the launcher layer only. Nothing in this document changes what a run *is* —
no hyperparameters, no metrics, no output schema. `EXPERIMENT_DESIGN.md` and
`FROZEN_EVAL_SPEC.md` are untouched by it, and the unbuilt items in
`OUTSTANDING.md` (2.1 `gsm8k_accuracy_dev`, 2.3 `adapter_name_or_path`) are
deliberately out of scope; they land on top of this, in either order.

Last updated 2026-08-27.

---

## 0. The contract that must survive

The migration is only cheap if the seam it cuts is the one slurm already sits
behind. It nearly is. These stay **byte-identical in behaviour**:

- **The command line.** `python batch_train.py --mode {train,benchmark,all,summary}
  --config configs/X.yaml`, run from `llada/`, keeps every meaning it has today,
  including `--overwrite` and the resume-an-incomplete-sweep semantics.
- **The output tree.** `outputs/<config_name>/<run_name>/` holding `config.yaml`,
  `metadata.json`, `adapter_params.json`, `adapter_config.json`,
  `adapter_model.safetensors`, `benchmark*.json`, `logs/`.
- **`summary.json`.** Same schema, same `BENCHMARK_SUMMARY_FIELDS`, same
  `pending`/`done`/`unreadable` statuses.
- **`train.py`'s CLI and internals.** `--config` / `--mode` / `--output_dir`, and
  the fact that it is launched under `torchrun` with one process per GPU. The one
  exception is §5 (resume), which is four lines inside `run_training`.
- **Every config YAML in `configs/`.** They keep parsing and keep producing the
  same runs. Their slurm keys become inert rather than fatal (§1.4).

What is deleted, per the decision to drop slurm outright rather than keep a
backend abstraction:

`SBATCH_TEMPLATE`, `write_sbatch`, `submit`, `_slurm_settings`, `SLURM_META_KEYS`
(already dead code — defined at `batch_train.py:74`, referenced nowhere),
`llada/run_training.sbatch`, `llada/run_training.sh` (superseded by configs long
ago), and `llada/jobs/correctness_checks.sbatch` → rewritten as a plain
`llada/jobs/correctness_checks.sh`.

---

## 1. Replacement execution model

### 1.1 One foreground supervisor

`batch_train.py` gains a scheduler that **blocks**. You run it inside tmux:

```
tmux new -s tier0
cd llada && python batch_train.py --mode all --config configs/tier0_lr.yaml
# ctrl-b d to detach
```

It holds a pool of GPU slots, dispatches runs into free slots as subprocesses,
prints a status line as each starts and finishes, and exits when the queue drains.

No daemon, no queue file, no lockfile, no PID reaping. That is the entire reason
for choosing this shape: the scheduler's only durable state is *the output tree
that already exists*, so there is nothing to corrupt and nothing to migrate.

### 1.2 Crash recovery is the existing skip logic

If the supervisor dies — ssh drop outside tmux, Ctrl-C, instance reboot — you
recover by **re-running the identical command**. `_is_trained()` skips runs that
already saved an adapter; §5's resume picks up runs that have a checkpoint but no
adapter; runs that never started are dispatched fresh. This composes for free and
is worth more than a daemon.

Signal handling, so that Ctrl-C is not a footgun mid-training:

- children are spawned with `start_new_session=True`, so they are not in the
  supervisor's process group and do not take its SIGINT;
- **first Ctrl-C**: stop dispatching, print what is still running, wait;
- **second Ctrl-C**: SIGTERM the children, then exit.

### 1.3 The `afterok` dependency becomes `&&`

Today `--mode all` submits two sbatch jobs per run and chains them with
`--dependency=afterok:<id> --kill-on-invalid-dep=yes`. Locally, one run is **one
queue entry with two phases**: the supervisor runs `job_train.sh`, and only on
exit 0 runs `job_benchmark.sh` in the same GPU slot. A non-zero train exit marks
the run failed and the benchmark never runs — exactly `afterok` plus
`kill-on-invalid-dep`, with no scheduler involved.

The three special cases in `run_all_mode` map straight across:

| Case | Today | Local |
|---|---|---|
| `tuning_type: none` | benchmark, no dependency | queue entry with the train phase omitted |
| already trained, no `--overwrite` | benchmark, no dependency | same |
| train + benchmark | `afterok` chain | two phases in one slot |

Holding both phases in one slot also removes a real hazard: on slurm the
benchmark was independently scheduled, so it could not start with the base model
weights still cold. Here it inherits a warm page cache from its own training job.

### 1.4 Slurm keys go inert, not fatal — **built**

`sbatch_time`, `sbatch_time_benchmark`, `sbatch_mem`, `gres`, `partition`,
`cpus_per_task`, `job_name` appear in every config in `configs/`. Deleting them
from the YAMLs would be a pointless diff across five files that are otherwise the
experimental record. Instead `batch_train.py` keeps a `RETIRED_KEYS` set, ignores
them, and prints **once per invocation**:

```
[note] ignoring retired slurm keys: gres, partition, sbatch_mem, sbatch_time
```

`nproc_per_node` is the one that survives and gains teeth — it is now a request
against a real, small pool (§4).

There is no wall clock any more. That deletes a whole class of failure the code
comments agonise over (`_slurm_settings`, `batch_train.py:236-275`: "a wall-clock
kill loses the run rather than resuming it"). Delete those comments with the
function; §5 makes the point moot regardless.

---

## 2. Storage roots

Two volumes, two roots, both overridable:

| Root | Default | Holds | Cost if lost |
|---|---|---|---|
| `LORTA_PERSIST_ROOT` | the repo checkout's parent | repo, `outputs/` (adapters, checkpoints, benchmark results, logs), `wandb/` | the experiment |
| `LORTA_SCRATCH_ROOT` | `/scratch` if it exists, else `/tmp/lorta` | venv, pip cache, HF hub cache (`HF_HOME`), datasets cache | ~20 min and ~16 GB of re-download |

New module `llada/paths.py`, stdlib-only so `batch_train.py` keeps its
"no torch import" property:

```python
PERSIST_ROOT  = os.environ.get("LORTA_PERSIST_ROOT",  <repo parent>)
SCRATCH_ROOT  = os.environ.get("LORTA_SCRATCH_ROOT",  "/scratch" if isdir else "/tmp/lorta")
OUTPUTS_ROOT  = os.environ.get("LORTA_OUTPUTS_ROOT",  join(PERSIST_ROOT, "lorta-outputs"))
VENV_DIR      = os.environ.get("LORTA_VENV",          join(SCRATCH_ROOT, "venv"))
HF_HOME       = os.environ.get("HF_HOME",             join(SCRATCH_ROOT, "hf"))
```

Resolution order everywhere: **explicit env var → config key → derived default.**

`OUTPUTS_ROOT` gets its own override because the two plausible instance layouts
disagree. If the repo itself is checked out on the persistent volume, today's
`llada/outputs/` is already in the right place and the default should stay there;
if the repo is on instance storage and only the volume is persistent, outputs must
be relocated. Rather than guess, `paths.py` defaults `OUTPUTS_ROOT` to
`llada/outputs` **when the repo is already under `PERSIST_ROOT`**, and to
`$LORTA_PERSIST_ROOT/lorta-outputs` otherwise, and prints the resolved value on
every invocation. `batch_train.py` imports it instead of computing
`OUTPUTS_ROOT` at line 63; nothing else in that file changes, because every other
path is derived from it.

`.gitignore` needs `llada/outputs/` kept and a `paths.py`-driven note; if
`OUTPUTS_ROOT` moves outside the tree it is untracked anyway.

**Do not put `HF_HOME` on the persistent volume for cost reasons alone.** Model
weights are re-downloadable and read once per job at ~16 GB; volume storage bills
continuously. If the volume is cheap and large, overriding `HF_HOME` to it is a
one-line env change and saves the per-instance prefetch — that is the whole reason
it is an env var.

---

## 3. Changes, file by file

### New — `llada/paths.py` (~40 lines)
§2. Stdlib only. Also exposes `describe()` returning the resolved roots, printed
by `batch_train.py` at startup and written into `metadata.json`.

### New — `llada/local_runner.py` (~200 lines)
The supervisor. Everything genuinely new lives here so `batch_train.py`'s diff is
small and reviewable.

```python
@dataclass
class Job:
    name: str
    run_dir: str
    phases: list[tuple[str, str]]   # [("train", "job_train.sh"), ("benchmark", ...)]
    n_gpus: int

class GpuPool:      # acquire(n) -> [device ids] | None; release(devices)
def preflight(jobs) -> None          # §6.3; raises before anything is dispatched
def run_queue(jobs, pool, dry_run)   # blocks; returns per-job exit status
```

Dispatch loop: while jobs remain, take the first whose `n_gpus` fits the free
pool, spawn it, poll children every second, release slots on exit, advance a
finished job to its next phase in the same slot. First-fit **in submission order**,
not best-fit — a sweep's runs are near-identical in size, and preserving order
keeps the log readable and the run→GPU mapping reproducible.

### New — `scripts/env.sh`, `scripts/bootstrap.sh`, `scripts/requirements.lock.txt`
**Built** — see §6. `env.sh` is the shell-side root resolver; `paths.py` is the
Python-side one and reads the same variables with the same fallbacks. That
duplication is deliberate and unavoidable: bash needs the roots before any Python
runs. Keep the two default blocks in sync, or have `paths.py` shell out to
`env.sh` once if they drift.

### Rewritten — `llada/jobs/correctness_checks.sh`
Same body, `#SBATCH` header and `srun` dropped, `$SLURM_SUBMIT_DIR` → `$PWD`,
env from `paths.py` via the shared preamble. Its §9-gate role
(`EXPERIMENT_DESIGN.md §10`) is unchanged: `cd llada && bash jobs/correctness_checks.sh`.

### Edited — `llada/batch_train.py`

| Today | After |
|---|---|
| `SBATCH_TEMPLATE` | `JOB_TEMPLATE` (§3.1 below) |
| `write_sbatch(...) -> path` | `write_job_script(...) -> path` — same signature, same call sites, same per-run artifact, `job_{mode}.sh` instead of `job_{mode}.sbatch` |
| `submit(path, no_submit, after_job)` | deleted; call sites append a `Job` to a list |
| `_slurm_settings` | `_run_settings` — returns `{nproc_per_node}` and nothing else |
| `run_*_mode` bodies | unchanged in structure; each ends by handing its job list to `local_runner.run_queue` |
| `OUTPUTS_ROOT` (line 63) | from `paths.py` |
| `--no-submit` | keeps working as an alias for the new `--dry-run` |

New flags: `--gpus 0,2` (default: all visible), `--max-concurrent` (default:
GPU-limited), `--stagger` (seconds between launches, §4.4), `--dry-run`.

`main()` now returns an exit code — non-zero if any run failed — so a wrapper or
an `&&` chain can tell. And because `run_queue` blocks, `write_summary` runs
*after* the queue drains rather than at submit time, so `summary.json` reports
results instead of a wall of `pending`.

`_versions()` no longer imports torch: it reads the versions the bootstrap
recorded from the venv the jobs actually run in, falling back to importing if the
marker is absent. Same answer as the old code intended, without paying a CUDA
import to find out a config has a typo — and unlike the login-node original, not
`null` for everything.

`run_train_mode` / `run_benchmark_mode` / `run_all_mode` keep their existing
control flow verbatim — the baseline branch, the `_is_trained` skip, the
`_clear_stale_results` call, the `write_summary` at the end. Only the two lines
that build and submit an sbatch change per branch. That is what keeps this a
reviewable diff rather than a rewrite.

One free improvement: `_versions()` (line 206) imports torch/transformers/peft and
recorded `null` for all of them on the login node. `batch_train.py` now runs on the
GPU box inside the venv, so `metadata.json` starts carrying real versions.

#### 3.1 The generated job script

```bash
#!/bin/bash
# Generated by batch_train.py — outputs/<config>/<run>/job_train.sh
set -euo pipefail

# Overridable so this script is runnable by hand exactly as the supervisor ran it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-{default_devices}}"
export LORTA_RDZV_PORT="${LORTA_RDZV_PORT:-29500}"

export HF_HOME={hf_home}
export WANDB_MODE={wandb_mode}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS={omp_num_threads}

source {venv}/bin/activate
cd {llada_dir}

exec torchrun --standalone --nnodes=1 --nproc_per_node={nproc} \
     --rdzv-endpoint=localhost:$LORTA_RDZV_PORT \
     train.py {mode_flag}--config {config_path} --output_dir {run_dir}
```

Note what is *not* here: no `pip install`. On slurm every job reinstalled `wandb`,
`pyyaml` and `peft --no-deps` because each job landed on a fresh node. One
instance, one venv, installed once by the bootstrap — and the supervisor's
preflight asserts `peft.__file__` resolves into this checkout (§6.3), which is the
check that install was actually protecting.

`CUDA_VISIBLE_DEVICES` and the rendezvous port are injected by the supervisor at
dispatch, with the generated defaults as the manual-rerun fallback. Everything else
is baked in at generation time so the script is a faithful record of the run.

### Edited — `llada/train.py`
Only §5. Roughly 6 lines.

### Edited — `llada/configs/*.yaml`
`nproc_per_node: 3` → `1` in `baseline.yaml`, `lorta_vs_nalorta.yaml`,
`nara_vs_nalorta.yaml`, plus a comment noting the change of machine. See §4.3 —
this is a scheduling change, not an experimental one, but it *is* a change to
committed configs and should be its own commit with that stated.

---

## 4. GPU allocation

### 4.1 Assignment
The pool owns integer device ids `0..N-1`. A job asking for `k` GPUs gets `k`
free ids, exported as `CUDA_VISIBLE_DEVICES=1,2`; `torchrun --nproc_per_node=k`
then sees them as `cuda:0..cuda:k-1`. `train.py` already does the right thing —
it uses `training_args.device` per rank and explicitly avoids `device_map`
(`train.py:1149`, `1382`) — so no training code is aware of this.

### 4.2 Rendezvous port collisions — the one real trap
`torchrun --standalone` picks the c10d rendezvous endpoint itself. On recent torch
that is a free port; on older torch it is a **fixed 29400**, and two concurrent
jobs on one box then collide with `Address already in use`, intermittently,
partway into a sweep. Under slurm this could not happen — one job per node.

The supervisor binds port 0 on localhost to find a genuinely free port, releases
it, and passes it down as `LORTA_RDZV_PORT`. The race between probe and bind is
theoretical at this concurrency and fails loudly if it ever loses.

**`--standalone` had to go, not just be supplemented.** The first draft of this
plan kept it and added `--rdzv-endpoint`; that would have been a silent no-op,
because `--standalone` *overwrites* `rdzv_backend`, `rdzv_endpoint` and `rdzv_id`
after argument parsing rather than deferring to them. The generated scripts ask
for what standalone would have configured, explicitly:

```
torchrun --nnodes=1 --nproc_per_node=N \
         --rdzv-backend=c10d --rdzv-endpoint=localhost:$LORTA_RDZV_PORT \
         --rdzv-id=<config>-<run>-<mode>
```

Equivalent, and independent of which torch decides what `--standalone` means.

### 4.3 The 3-GPU configs — **updated**
Three of five configs requested `nproc_per_node: 3`. On a 4-GPU box that is one
job at a time with a GPU idle; on a 2-GPU box it never schedules at all. The
supervisor **fails fast** at preflight when any job asks for more GPUs than exist,
rather than deadlocking — a queue that silently waits forever is the worst outcome
here.

Since the rented box's GPU count is still unknown, hardcoding any number is wrong.
`nproc_per_node` therefore also accepts **`all`**, meaning every GPU in the pool
this invocation is using (`--gpus` narrows it):

| config | runs | setting | why |
|---|---|---|---|
| `baseline.yaml` | 1 | `all` | solo run; nothing to schedule beside it, and the decode shards across ranks |
| `lorta_vs_nalorta.yaml` | 2 | `1` | arms run concurrently and are time-matched |
| `nara_vs_nalorta.yaml` | 3 | `1` | as above |
| `tier0_lr.yaml` | 6 | `1` | already correct |
| `tier0_headlr_matrix.yaml` | 3 | `1` | already correct |

`1` for a sweep is not a compromise. Scaling within a run is imperfect, so N runs
at one GPU each finish no later than N runs serialised across N GPUs, and usually
sooner — `tier0_lr`'s own comment reached the same conclusion for the cluster
("6 single-GPU jobs schedule concurrently, so time to all six is shorter"), and it
holds more strongly with no queue to wait in.

`all` covers the opposite case: a single long run with nothing beside it, such as
Tier 1's last straggler, where splitting across every device is pure speedup.

### 4.4 Host-side contention
Three or four concurrent jobs on one box share what slurm used to isolate:

- **CPU RAM.** Each job loads 8B in bf16 with `low_cpu_mem_usage=True` — the peak
  is transient but real, ~16 GB per *rank*. Four concurrent single-GPU jobs can
  spike ~64 GB during load. Staggering dispatch by ~60 s makes the peaks miss each
  other. The supervisor does this; it costs nothing against an 8 h run.
- **Threads.** Default `OMP_NUM_THREADS` (= core count, per process) thrashes at
  4× oversubscription. Set it to `max(1, cores // total_ranks)` in the job script.
- **Dataloader workers** inherit the same problem if `dataloader_num_workers` is
  ever raised from its default.
- **Disk.** Four jobs writing checkpoints every 100 steps to one volume. §5 caps
  this with `save_total_limit`.

---

## 5. Checkpoint / resume across instance loss — **built**

Today `run_training` calls `trainer.train()` bare (`train.py:1216`) with
`save_strategy: steps, save_steps: 100`. So checkpoints **are already being
written** to `output_dir/checkpoint-N` — and never read. A killed run restarts
from step 0. On a preemptible rented instance that is the single largest risk in
this migration.

### 5.1 The change
```python
last = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
if last is not None:
    print(f"Resuming from {last}")
trainer.train(resume_from_checkpoint=last)
```
`get_last_checkpoint` returns `None` on a clean directory, so a fresh run is
unaffected. PEFT models checkpoint adapter weights plus optimizer/scheduler/RNG
state, so this restores the run rather than approximating it.

### 5.2 `save_total_limit`
Unset today, i.e. unbounded. `tier0_lr` at ~241 steps keeps 2 checkpoints; a
6-epoch Tier 1 run at ~120 steps/epoch keeps ~7, per run, times 12 runs, on the
persistent volume. Set `save_total_limit: 2` as a default in the generated config
(one to resume from, one in case the newest is torn mid-write).

### 5.3 `--overwrite` must delete checkpoints — do not miss this
`_clear_stale_results` currently removes `benchmark*.json` only. With resume
wired up, `--overwrite` on a run with checkpoints on disk would **silently resume
the run you asked to discard**, which is exactly the failure `--overwrite` exists
to prevent, and it would be invisible in the results. Extend it to remove
`checkpoint-*/` in the same pass, under the same `--dry-run` reporting.

### 5.4 Interaction with the skip logic
Unchanged and already correct: `_is_trained` keys on `adapter_config.json`, which
only appears after `trainer.save_model()`. A run killed at step 150 has checkpoints
but no adapter → not "trained" → re-dispatched → resumes at 150. The desired
behaviour falls out; it just needs §5.1 to exist.

### 5.5 Vestigial directory
`_prepare_run_dir` creates `checkpoints/` and `samples/` (`batch_train.py:407`).
HF writes to `output_dir/checkpoint-N`, not into `checkpoints/`, and nothing writes
`samples/`. Leave them — harmless, and repointing HF's checkpoint path is a
behavioural change for no gain. Worth one comment saying so.

---

## 6. Instance bootstrap — **built**

```
bash scripts/bootstrap.sh                                  # fresh instance, one command
bash scripts/bootstrap.sh --persist-root /mnt/volume       # volume mounted elsewhere
bash scripts/bootstrap.sh --skip-model --force             # rebuild the venv, keep weights
```

It resolves the roots and writes them to `.lorta.env` (untracked), so every later
shell is `source scripts/env.sh` with no arguments — that file is the whole
anti-faff mechanism. Idempotent: re-running on a half-finished instance completes
it, on a finished one it is a fast no-op that reprints the environment.

### 6.1 What it does
1. Resolve and create both roots (§2); print them.
2. `python -m venv $LORTA_VENV` on scratch if absent; `pip install -U pip`.
3. Install deps — pinned torch/transformers/datasets/accelerate/wandb/pyyaml, then
   `pip install -e $REPO/peft --no-deps`, exactly as the sbatch did.
4. `python -c "from peft import LorTaConfig, NALorTaConfig; import peft; print(peft.__file__)"`
   and assert the path is inside this checkout.
5. `nvidia-smi` → print device count, names, free memory.
6. Prefetch the base model into `HF_HOME` with `huggingface_hub.snapshot_download`,
   **serially, before any job runs.** Concurrent first-run downloads of the same
   repo from 4 jobs is a real corruption/duplication risk and wastes bandwidth.
7. Assert `frozen_eval.py`'s `FROZEN_EVAL_FILE` exists (it is committed; this is a
   checkout sanity check, the same one `correctness_checks.sbatch` does).
8. Write `$LORTA_SCRATCH_ROOT/.lorta-bootstrap.json` — timestamp, git hash,
   resolved versions — as the marker the preflight looks for.

### 6.2 The version pins — **captured**
`scripts/requirements.lock.txt` is a `pip freeze` of the working cluster venv.
This is the environment every existing result was measured under, so it is part of
the experimental record; commit it. There is deliberately **no fallback** — a
guessed pin set that merely imports is how a silent environment mismatch gets into
the comparison, so `bootstrap.sh` fails outright if the lock file is missing.

What the capture actually says, versus what was assumed before it existed:

| | assumed | actual |
|---|---|---|
| torch | 2.4.1 | **2.8.0** (cu12.8) |
| transformers | 4.44.2 | **4.57.6** |
| accelerate | 0.34.2 | **1.10.1** |
| datasets | 2.21.0 | **4.5.0** |
| numpy | `<2` | **2.0.2** |

The reasoning that produced the guess was wrong in an important way. The vendored
peft fork is `0.10.1.dev0` (~Apr 2024), and installing it `--no-deps` was read as
"nothing forces a compatible transformers, so it must be pinned to its own
vintage". In fact it has been running against transformers 4.57 the whole time —
the fork's tuners are self-contained enough not to care. `--no-deps` is still
right, but for the narrower reason that it stops peft's own loose requirements
from moving the pinned versions.

The lock file's one landmine: `pip freeze` recorded the fork as an editable VCS
install,

```
-e git+ssh://git@github.com/rrrama/dissertation.git@24d9b7c…#egg=peft&subdirectory=peft
```

which would need an SSH key for a private repo, resolve *with* deps, and install
the fork from a fixed remote commit instead of the checkout being run — the exact
"not a sibling clone" failure the old sbatch preamble warned about. `bootstrap.sh`
strips editable/VCS lines before installing and reports how many it dropped;
nothing is lost, because every dependency peft declares is pinned explicitly
elsewhere in the file.

### 6.3 Preflight
`local_runner.preflight()` runs before the first dispatch and raises on: missing
venv or bootstrap marker; `peft` resolving outside this checkout; `max(nproc_per_node)`
> visible GPUs; `FROZEN_EVAL_FILE` missing when a queued job needs it; less than
~50 GB free on `OUTPUTS_ROOT`. Six jobs each dying 30 s in, four hours after you
detached, is the outcome this is buying against.

---

## 7. Build order

Phases 1–4 are **built**; §8 is what remains, and it needs the hardware.

Each phase leaves the tree working.

1. **`paths.py` + the `correctness_checks.sh` rewrite** — no launcher changes.
   `scripts/{env.sh,bootstrap.sh,requirements.txt}` are already written; this
   phase adds the Python-side root resolver and drops the last `#SBATCH` header.
   Verify by hand: bootstrap the instance, then
   `cd llada && bash jobs/correctness_checks.sh`. Passing it re-clears
   `EXPERIMENT_DESIGN.md §10`'s gate on the new hardware, which needs doing anyway.
2. **`local_runner.py` + `batch_train.py` cutover** — sbatch out, job scripts and
   supervisor in. Acceptance met: `--dry-run` reproduces the pre-migration run
   names (`eta0_lr0.0001`, …), expansion counts and run dirs, and `summary.json`
   entries are byte-identical for the in-repo outputs layout.
3. **Resume + `save_total_limit` + `--overwrite` checkpoint cleanup** (§5). Small,
   independent, and the thing you least want to discover you need.
4. **Config `nproc_per_node` updates** (§4.3), as its own commit.
5. **First real sweep**: `tier0_lr` (6 × 1 GPU). It is the cheapest config that
   exercises train → benchmark chaining, concurrency, `eval_loss_frozen`, and the
   summary path together.

Phases 1–4 are all mechanical; the work is in getting phase 5 to run unattended
overnight.

---

## 8. Verify first, in this order

1. **A single run end to end**, one GPU, before any concurrency: does
   `torchrun … train.py` still reach step 1 with the new venv and pinned versions?
   This is where a transformers/`trust_remote_code` mismatch surfaces.
2. **Loss agreement.** `OUTSTANDING.md` already flags this for the O1/O2 refactor:
   the training loss curve should match a previous cluster run to ~1e-7. Now it is
   also the check that the new environment is the old environment. A large
   divergence here invalidates comparing anything measured on the cluster with
   anything measured after the move.
3. **Two concurrent runs** — the rendezvous port (§4.2) and the RAM spike (§4.4).
4. **Kill and resume**: SIGKILL a run past step 100, re-run the same command,
   confirm the log says `Resuming from checkpoint-100` and the wandb step counter
   continues rather than restarting.
5. **Single-GPU memory headroom.** `tier0_lr`'s comment estimates ~33 GB of 48 GB
   at batch 4 on an L40. Rented cards may have 24 GB (A10/L4/4090) or 80 GB
   (A100/H100). At 24 GB the Tier 0 batch size does not fit and
   `per_device_train_batch_size` × `gradient_accumulation_steps` must be
   repartitioned to hold the effective batch at 60 — which the O2 per-example
   lambda change explicitly permits, and which must be done identically across all
   arms.

---

## 9. Decisions taken

- **Slurm is removed, not abstracted.** No launcher interface, no second backend.
- **Foreground supervisor in tmux**, not a daemon. Durable state is the output
  tree; recovery is re-running the command.
- **Per-run job scripts are kept** (`job_train.sh`, `job_benchmark.sh`) in the same
  place the `.sbatch` files lived, so a run stays hand-reproducible and
  `write_sbatch`'s call sites survive as `write_job_script`.
- **Config YAMLs are not rewritten** beyond `nproc_per_node`; retired slurm keys
  are ignored with one note. `nproc_per_node` gained `all` so the configs need not
  name a device count the next box will not have (§4.3).
- **Stale slurm references remain in the other planning docs** — `OUTSTANDING.md`
  (O8's `sbatch_time` guidance, "generated sbatch scripts"), `IMPLEMENTATION_PLAN.md`,
  `FROZEN_EVAL_SPEC.md` §7, `current_plan.md`. Left alone deliberately: they are
  the experimental record and this document is the one that supersedes them on
  execution. Worth a pass before they are read as current.
- **wandb stays offline** (`WANDB_MODE=offline`), configurable via `paths.py`.
  Not changed here — it is orthogonal, and `check_wandb_progress.py` depends on it.

## 10. Open

- ~~The version pins do not exist yet~~ **Done** — `scripts/requirements.lock.txt`
  is captured (§6.2). Commit it; it is experimental record, not build config.
- **GPU model and count are unknown**, which §8.5 makes a live question for the
  Tier 0 batch geometry.
- **Instance preemption policy** — whether the rental can be reclaimed
  mid-run — decides whether §5's resume is a safety net or the normal path. If it
  is the normal path, `save_steps: 100` (~40 min of lost work on a Tier 1 run at
  ~8 h / 720 steps) is worth tightening.
