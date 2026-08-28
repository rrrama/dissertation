# Implementation Plan — NaLoRTA experiments

Companion to `EXPERIMENT_DESIGN.md`. Part A is what the design assumed the repo
had and it did not; Part B is the build order. Open decisions live in
`OUTSTANDING.md`, which is the file to read first.

Everything runs through `llada/batch_train.py` except the §9 correctness tests,
which are a single standalone job.

Last updated 2026-08-26.

---

# Part A — Findings

## A1. Resolved gaps

All of these blocked Tier 0 and are now closed. Kept as one-liners because the
*reason* each mattered is still the justification for the code that exists.

| # | Gap | Resolution |
|---|---|---|
| A1.1 | Adapter knobs unreachable from YAML — **RQ2 was not expressible in a config** | `eta` / `fourier_m` / `lora_dropout` via `ModelArguments` + `ADAPTER_DEFAULTS`; all default to `None` = "keep this adapter's own default", so pre-existing configs are unchanged |
| A1.2 | Adapter initialised outside the seed — `set_seed` runs in `Trainer.__init__`, after `get_peft_model` | `transformers.set_seed(training_args.seed)` right after `build_args`. Before this, §6's ≥3-seed protocol was measuring data order only |
| A1.3 | No dev split — every selection decision would have been made on test | `llada/splits.py`, 250 questions carved from GSM8K **train** |
| A1.4 | No frozen eval set and no eval loss at all — **Tier 0 had no metric** | `llada/frozen_eval.py`; see `FROZEN_EVAL_SPEC.md` |
| A1.5 | No parameter counter, and the claim is joint on accuracy *and* params | `llada/adapter_params.py` |
| A1.6 | One optimiser param group, so §5.2's head LR was unexpressible | `LladaSFTTrainer.create_optimizer` splits the groups the base implementation already built |
| A1.7 | One benchmark result slot per run — dev and test accuracy overwrote each other | `benchmark_{name}.json` for new benchmarks; `benchmark.json` keeps its name and schema |
| A1.9 | Benchmark mode never logs to W&B | **Won't fix** — O5, everything stays offline |

**A1.8 — benchmarks cannot target an external checkpoint. Still open**, as Phase
2.3. `ModelArguments.adapter_name_or_path` exists and is never read; that is the
hook. Tier 3 needs it, and under O11 so does Tier 1's epoch selection.

`ModelArguments.model_name_or_path`'s default also moved from `LLaDA-8B-Base` to
`LLaDA-8B-Instruct`, which is what every config already set and the only model in
scope.

Still hardcoded, all Tier 2 concerns, all plumbable the same way when needed:
`init_scale`, `lambda_source`, `input_mode`, `init_c`, `fnn_hidden_size_*`,
`scale_ab`, `pool_lambda`.

## A2. Design/code disagreements that are still live

### A2.5 §7's suggested ranks do not produce overlapping parameter counts
**Parked** — full table and the alternative rank sets in `OUTSTANDING.md` §O6. It
changes the frontier configs, not the headline ones. The short version: LoRTA at
r=32 is ~137 k and LoRA at r=32 is ~33.6 M, so RQ1(a)'s "accuracy at
iso-parameters" is not obtainable from R ∈ {8,16,32}. Keep R=32-for-everything as
the Tier 1 2×2 but do not call it iso-parameter.

### A2.7 §3.1's λ definition matches generation but not training
**Documented, not fixed.** At generation, `generate.py:166-171` sets
`response_mask` over the entire `gen_length` window, so λ = masked/256 exactly as
§3.1 says. At training the denominator is the *example's own answer length*, which
varies from ~30 to ~300 tokens. So it is not only the mask *geometry* that
mismatches (§3.1's Tier 3 flag) — the denominator's meaning shifts too. Fold into
the same note in the writeup.

### A2.8 Missing adapter features for Tier 2 / Tier 3
Genuinely new code, not plumbing. Built in Phase 5.

| RQ | Requirement | Status |
|---|---|---|
| RQ3 | Θ shared scalar vs per-rank vector | not implemented |
| RQ3 | MLP head on the CP backbone | not implemented (NaRA's `embedding_type: "mlp"` is the λ *embedding*, not the head) |
| RQ3 | Θ zero-init | not reachable — `init_scale` feeds kaiming's leaky-ReLU slope `a`, which never yields exactly zero |
| RQ4 | Θ rank restriction | not implemented |
| RQ6 | C_λ(λ) plots from checkpoints | no script |

`fourier_m` maps to `embedding_length` (NaLoRTA) / `embedding_dim` (NaRA); both
are already config fields, so the `m` sweep is pure config.

### Resolved disagreements

- **A2.1 §9.1's η=0 collapse test — dropped.** The RNG streams differ between
  `nalorta` and `lorta` (`NALorTaModel.inject_adapter` draws `Θ` and the Fourier
  **k** before `lora_A`; `LorTaModel` does not), so "bit-identical given the same
  seed" cannot hold as written. The collapse is evident from the code —
  `c_mask = 1 + eta * (...)` with `eta = 0` — and is not worth a test.
- **A2.2 λ pooling — fixed** (O2, option 3).
- **A2.3 NaRA's collapse knob is `c_scale`, not `eta` — plumbed.**
  `C(λ) = I_r + c_scale · F_φ(e(λ))` is functionally identical, so the YAML `eta`
  key maps onto `c_scale` for NaRA and `eta` for NaLoRTA. Note `scale_ab`
  (default `1.0`) multiplies the delta on top of `lora_alpha/r` and **must stay at
  `1.0`** for the `nara(η=0) ≡ lora` identity to hold exactly; it is hardcoded, so
  this holds by default.
- **A2.4 `lora_dropout` — decided**, see `OUTSTANDING.md` §O3. Applies when the
  Tier 1 configs are written.
- **A2.6 §9.3's per-slice formula — dropped.** Notation only (`Bᵀ` in the
  document vs `B` of shape `(r, head_dim)` in the code; four `Diag()`s vs one
  elementwise product). Same operation — fix the notation in the writeup, no test.
  Checking it is what turned up O1.

## A3. Operational notes that still bite

- **Everything must run from `llada/`, not the repo root.** Two confusing failure
  modes; see `OUTSTANDING.md` for both. Generated sbatch scripts already `cd`, so
  only hand-run commands hit this.
- **`configs/*.yaml` set `eval_steps: 100` and it is a no-op** — there is no
  `evaluation_strategy`, and the frozen eval is a callback rather than a Trainer
  evaluation loop, so it has its own key: `eval_loss_frozen_steps`. Two
  similarly-named knobs where one does nothing is a trap. Either delete
  `eval_steps` from the configs or know that only `eval_loss_frozen_steps` does
  anything.
- **`expand_runs` expands *any* list-valued field**, and it is a cartesian
  product over a single flat dict per file. Two consequences: adding e.g.
  `target_modules` to a YAML would silently fan out into runs (hence 2.6's
  no-expand set), and **a sweep whose LR depends on `tuning_type` cannot be one
  YAML** — hence one config file per family in Tier 1.
- **§8's run-name convention** (`{method}_r{rank}_lr{..}_hlr{..}_eta{..}_s{seed}`)
  differs from `run_name`, which names dirs by *varying* fields only — a
  fixed-rank sweep produces `lr0.0001_nalorta` rather than the full tuple. Cheap
  to align, worth doing for cross-sweep comparability (2.6).
- **Checkpoint quota.** `save_steps: 100` over ~750 steps with no
  `save_total_limit` keeps ~7 checkpoints. Trivial for LoRTA (~140 k params) but
  ~400 MB each for LoRA/NaRA including optimiser state — ~34 GB across the 12
  Tier 1 runs. Fine, but budget it. O11 depends on those checkpoints existing, so
  **do not add `save_total_limit`**.
- **Wall clocks and the 12 h Tier 1 requirement:** `OUTSTANDING.md` §O8.
- **`error.err`** (untracked, repo root) shows the baseline truncating
  mid-sentence at `<|eot_id|>` after ~20 tokens, so the current zero-shot number
  is probably not a fair one. Worth a look before Tier 1 fixes the baseline in
  stone.
- **`configs/lorta_vs_nalorta.yaml` uses `rank: 128` → α/r = 0.031.** It is the
  frozen record of an already-completed sweep, so leave it — but any number quoted
  from it is not comparable to the new runs, and those adapters also predate the
  O1/O2 fix.

---

# Part B — Implementation plan

## Phases 0, 1, 3 — done

- **0.1** seed the adapter init (A1.2). **0.2** plumb `eta` / `fourier_m` /
  `lora_dropout` (A1.1).
- **0.3 `llada/adapter_params.py`** — `count_trainable_adapter_params`,
  `adapter_param_breakdown`, `summarise_adapter_params`, plus
  `assert_fourier_projection_frozen`, which asserts §3.2's *premise* (that **k** is
  frozen, so a plain `requires_grad` filter implements the exclusion) rather than
  trusting it. If that ever stopped holding, every parameter count already written
  down would be wrong and nothing would say so.
- **0.4 `llada/splits.py`** — `derive_dev_indices` / `load_dev_indices` /
  `train_split` / `dev_split`. Two deliberate deviations: the data lives in
  `llada/split_data/` (a `splits/` directory beside `splits.py` shadows on
  `import splits`), and the indices are a **pure function** of
  `(DEV_SEED, DEV_SIZE, len(train))` rather than read from the committed file. The
  file is a tripwire — `load_dev_indices` re-derives and refuses to run on a
  mismatch, which is what would happen if the dataset were revised or a constant
  edited after runs had been selected on the old split.
- **0.5 `llada/frozen_eval.py`** — spec in `FROZEN_EVAL_SPEC.md`. The one Phase 0
  item that edits the training path, so it was written down before it was written.
- **1 — §9 correctness tests.** `llada/correctness_checks.py` +
  `llada/jobs/correctness_checks.sbatch`, 1 GPU, 1 h (the budget is four 8B loads,
  not the checks). Three checks after A2.1/A2.6 dropped two: §9.2 dW=0 at init via
  `delta.to_dense()`, §9.4 the param counter against closed-form counts derived
  from each *method's* definition rather than read off the tuner's shapes
  (137,344 / 138,368 / 33,554,432 / 34,227,968), and §9.5 frozen-eval determinism
  — two evals of one perturbed checkpoint, plus forward hooks on every `p > 0`
  dropout asserting `training=False`. `--checks delta,params` gives the two
  CPU-only ones. **Passed 2026-08-26**, clearing §10's gate.
- **3 — two-LR training** (A1.6). `create_optimizer` splits the groups the base
  implementation already built rather than rebuilding them, so the weight-decay
  partition, optimiser class and kwargs cannot drift from the installed
  transformers. Head params matched by name (`HEAD_PARAM_NAMES`): `lora_Theta`,
  `lora_mapper`, `lora_constant_c`, `lora_phi`. `head_learning_rate: None` = use
  `learning_rate`, returning the base optimiser untouched — which matters because
  lora and lorta have no head at all, and which under O10 is what every run now
  uses. Setting it on an adapter with no head raises rather than silently doing
  nothing.

Two things §9.5 does that the original spec did not, both because the obvious
version goes vacuous: it **perturbs `lora_B` first** (at init dW = 0, so a fresh
adapter would score the bare base model against itself), and it **hooks the
dropout modules** rather than comparing an eval-mode loss against a train-mode one
— the loss comparison is a proxy that silently stops testing anything whenever the
delta is too small to move a bf16 logit, while a missing `model.eval()` shows up
as `training=True` regardless of magnitude.

## Phase 2 — Evaluation plumbing

**2.1 Benchmark registry — half done.**
```
"eval_loss_frozen"    -> frozen (t, mask) loss on the dev set        (Tier 0)     DONE
"gsm8k_accuracy_dev"  -> generation accuracy, 250 dev questions      (Tier 1-2)   TODO
"gsm8k_accuracy"      -> generation accuracy, full 1319 test         (final only) DONE
```
`diffusion_evaluate` is already split-agnostic — it takes `test_set` — so the dev
variant is a two-line addition over `splits.dev_split`.

Landed alongside: `_load_model_with_adapter` extracted from
`_gsm8k_accuracy_benchmark` and shared, loading with `is_trainable=True` so that
`requires_grad` — the rule §3.2 counts by — survives into benchmark mode. Without
it `adapter_params` in a result file is `0`, which is a wrong number rather than a
missing one.

**2.2 One result file per benchmark — DONE.** A filename change, not a format
change: `gsm8k_accuracy` keeps writing `benchmark.json` with today's schema (plus
the additive `adapter_params`); only new benchmarks write
`benchmark_{name}.json`. `write_summary` globs `benchmark*.json` and emits one
entry per (run, benchmark); `_clear_stale_results` clears all of them.

**2.3 External adapter source — TODO, on the critical path.** In
`_gsm8k_accuracy_benchmark`, resolve the adapter from
`model_args.adapter_name_or_path` when set, else `output_dir` (fixes A1.8). In
`batch_train.py`, treat a run as benchmarkable if either the run dir has an
adapter or `adapter_name_or_path` is set. This is what makes Tier 3 a config
rather than a script, **and what lets Tier 1's checkpoints be scored for the
epoch curve** (O11).

**2.4 W&B in benchmark mode — dropped** (O5).

**2.5 `--benchmark` CLI override** on `--mode benchmark`, so the same run dirs can
be scored on dev and then on test without editing configs. Do it with 2.1.

**2.6 Naming.** Align `run_name` with §8; add `NAME_ABBREV` entries for `eta`,
`head_learning_rate`, `embedding_length` (done). Add a no-expand set to
`expand_runs` (open).

## Phase 4 — Run the experiments

All via `batch_train.py`, one config file per tier per family, all committed.

**4.1 Tier 0 — backbone LR only.** `configs/tier0_lr.yaml`: `tuning_type:
nalorta`, `eta: [0, 1.0]`, `learning_rate: [1e-4, 3e-4, 1e-3]` → **6 runs**, 1–2
epochs, `benchmark: eval_loss_frozen`, seed 42, `lora_dropout: 0.1`.
**Submitted and training.**

Per O4 the matrix family is fixed at `1e-4` and not swept, so it needs no
backbone-sweep file: its η=1 cell has no free hyperparameter left and its η=0 cell
(`lora`) is first trained in Tier 1. Both η cells of the CP family *are* swept, so
`lorta` — a headline 2×2 cell — is not the under-tuned baseline §5 warns about.
Extend the CP grid if a cell selects an endpoint (§7).

**~~4.1b Tier 0, head LR~~ — cut by O10.** All arms train single-LR.
`configs/tier0_headlr_cp.yaml` will not be written.

**~~4.1c Epoch probe~~ — cut by O11**, folded into 4.2: Tier 1's own
`save_steps: 100` checkpoints get scored on dev-250 post-hoc, giving the epoch
curve per arm instead of extrapolating one `nalorta` probe onto four methods.

Output of Tier 0: `configs/locked_lrs.yaml`, committed, recording the selected CP
LRs alongside the fixed `1e-4` for the matrix family.

**4.2 Tier 1 — the headline 2×2 (RQ1 + RQ2).** `eta: [0, 1.0]` × `seed: [42, 43,
44]` at R=32 for each family → 12 runs, with the non-noise-aware cells produced by
the η=0 collapse as §2 requires.

**One file per family** — `tier1_headline_cp.yaml`, `tier1_headline_matrix.yaml` —
because `expand_runs` is a cartesian product over one flat dict and
`learning_rate` cannot vary with `tuning_type` (A3). If 4.1 selects different LRs
for the two η cells of the CP family, that splits again into one file per cell.
**Write the selected LR into each file explicitly** rather than relying on a
default, so the config is the record of what ran. Also: `lora_dropout: 0.1` (O3),
`sbatch_time: "12:00:00"` (O8), no `save_total_limit` (O11).

Then score every run's checkpoints on `gsm8k_accuracy_dev` for the epoch curve,
and a single separate `--mode benchmark --benchmark gsm8k_accuracy` pass over the
final 2×2 only.

Frontier extension — `configs/tier1_frontier.yaml`, 2 additional ranks for `lorta`
and `nalorta`, 1 seed each; curve *shape* matters, not error bars. **See A2.5 /
O6 before fixing the rank sets.**

**4.3 Tier 3** (before Tier 2 if time is tight, per §10) —
`configs/tier3_sampler.yaml`: `adapter_name_or_path` over the Tier 1 run dirs ×
`diffusion_steps: [32, 64, 128, 256]` × `block_length: [16, 32, 64]`, `--mode
benchmark`, no training. Cheap in *training* cost only — it is still a full decode
per cell, so measure one cell's wall clock and budget from that, then drop cells
rather than seeds.

**4.4 Tier 2** — needs Phase 5. The η and `fourier_m` sweeps are pure config and
can run at any time.

## Phase 5 — Tier 2/3 adapter features (buildable while Tier 1 trains)

New `NALorTaConfig` fields plus logic in `nalorta/model.py:inject_adapter` /
`_compute_weights_from_tensor`. Each defaults to today's behaviour, so existing
checkpoints keep loading.

- `theta_mode: "per_rank" | "shared_scalar"` — RQ3's per-rank-vector test, the
  part of the contribution that is not merely parameter count. Shared scalar is
  `Θ` of shape `(m, 1)` broadcast over `r`.
- `theta_rank: int | None` — factorise `Θ = Θ₁Θ₂` to decouple Θ's capacity from
  the backbone R (RQ4's confound).
- `init_theta: "kaiming" | "zero"` — reachable zero-init (A2.8).
- `head_type: "linear" | "mlp"`, `head_hidden: int` — MLP head on the CP backbone,
  isolating noise-function expressivity from the backbone.

`llada/plot_c_lambda.py` (RQ6): load an adapter, sweep λ ∈ [0,1], plot
`c_mask(λ) = 1 + η·Θᵀφ(λ)` per rank and the induced per-layer scaling via `C_l`.
No GPU, no training — run it locally. Unblocked by O1: `C_h` now genuinely indexes
attention heads.

## New / changed files

| File | Status | Phase |
|---|---|---|
| `llada/adapter_params.py` | done | 0.3 |
| `llada/splits.py` + `split_data/gsm8k_dev_250.json` | done | 0.4 |
| `llada/frozen_eval.py` + `split_data/frozen_eval_v1.json` | done | 0.5 |
| `llada/correctness_checks.py` + `jobs/correctness_checks.sbatch` | done, passed | 1 |
| `llada/train.py` | seed, peft plumbing, benchmarks, optimiser groups | 0, 2, 3 |
| `llada/batch_train.py` | result files, wall clocks, naming — **`adapter_name_or_path` and `--benchmark` outstanding** | 2 |
| `llada/configs/tier0_lr.yaml` | done, training | 4.1 |
| `llada/configs/tier1_headline_{cp,matrix}.yaml` | **next** | 4.2 |
| `llada/plot_c_lambda.py` | new | 5 |
| `peft/.../nalorta/{config,model}.py` | Θ mode / rank / init, MLP head | 5 |
