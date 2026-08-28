# Outstanding — NaLoRTA experiments

Live tracker: what is **undecided**, **unbuilt**, or **deliberately left broken**.
`IMPLEMENTATION_PLAN.md` holds the build order; `EXPERIMENT_DESIGN.md` is the
design of record and this file records where we deviate from it.

Last updated 2026-08-26.

---

## Where the project is

The §9 correctness job passed, which clears §10's gate ("do not proceed past step
1"). `configs/tier0_lr.yaml` is training — 6 CP runs, `eta: [0, 1.0]` ×
`learning_rate: {1e-4, 3e-4, 1e-3}`, 1–2 epochs, scored on `eval_loss_frozen`.

Everything from here is aimed at **Tier 1**, which is where the compute actually
sits: 12 runs × ~8 h training, against Tier 0's ~3 h short runs.

**Critical path to Tier 1:**

1. **Build 2.1 and 2.3** — `gsm8k_accuracy_dev` and `adapter_name_or_path` in
   benchmark mode. Both unbuilt; Tier 1 cannot be scored without 2.1, and the
   epoch selection (O11) needs 2.3. Buildable now, while `tier0_lr` trains.
2. `tier0_lr` reports → read the selected CP backbone LR, **per η cell**. Extend
   the grid if a cell selects an endpoint (§7); do not accept an endpoint.
3. Write `configs/tier1_headline_cp.yaml` and `configs/tier1_headline_matrix.yaml`
   with the LRs spelled out explicitly, and submit.

The matrix family needs no sweep at all — O4 fixed it at `1e-4` — so `tier0_lr` is
the only Tier 0 result Tier 1 waits on.

**Smoke run: consider it done.** It was going to be a separate short job to
exercise the new O1/O2 forward path before 9 jobs committed to it. `tier0_lr` is
that exercise. One thing to check by eye before Tier 1: the training loss curve
should match an older run to ~1e-7 — the `compute_loss` split is arithmetically
identical but changed summation order, so a small drift is expected and a large
one is not.

**Run everything from `llada/`, never the repo root.** Two failure modes, both
confusing rather than clean: the root holds a legacy `gsm8k/` directory and
`datasets.load_dataset` resolves local paths before the hub (→
`DataFilesNotFoundError: No (supported) data files found in gsm8k`), and
`batch_train.py` opens `--config` relative to the working directory. Generated
sbatch scripts already `cd llada/`; only hand-run commands hit this.

---

## Open

### O3 — `lora_dropout` differs across arms

**Decided, not yet applied.** `lora_dropout: 0.1` everywhere. The per-adapter
defaults are 0.1 for lora/lorta/nalorta and 0.05 for nara, while §5 says LR is the
only thing that varies per method — an unstated per-method difference in a
comparison whose claim is "identical treatment except the adapter". The effect is
well inside §6's noise floor; the reason to fix it is the claim, not the effect.

Set explicitly in both Tier 0 configs already. **Applies when the Tier 1 configs
are written**, which is the last chance to get it right.

### O6 — Rank grids for the RQ1 frontier

**Parked until Tier 1 is running.** At r=32:

| Method | trainable params |
|---|---|
| LoRTA | ~137 k (A 131k + B 4k + C_l 1k + C_h 1k + C_m 128) |
| NaLoRTA (m=32) | ~138 k (+Θ 1k) |
| LoRA | ~33.6 M (32 layers × 4 matrices × 2·4096·32) |
| NaRA | ~34.2 M (+mapper ~674 k) |

The ~245× ratio is the headline and it is a good one. But it means **RQ1(a),
"accuracy at iso-parameters", is not obtainable** from R ∈ {8,16,32} across all
four methods — the two x-ranges never overlap. LoRA at r=1 (~1.05 M) is still ~8×
LoRTA at r=32.

Overlapping curves would need LoRTA/NaLoRTA at R ∈ {32,128,512} (~137 k → ~2.1 M)
against LoRA/NaRA at R ∈ {1,2,4} (~1.05 M → ~4.2 M). Keep R=32-for-everything as
the Tier 1 2×2 — it is the honest "matched rank" cell — but **do not call it
iso-parameter**.

### O9 — The ≥1-masked-token guarantee is batch-level, and reads as though it should not be

**Noted, deliberately not fixed.** `train.py`'s `compute_loss` guards against an
all-clean mask with

```python
if not masked_indices.any():
    first_response = response_mask.float().argmax(dim=1)
    masked_indices[arange(b), first_response] = response_mask.any(dim=1)
```

The condition is `.any()` over the **whole micro-batch**, the body writes to
**every row**. So it fires only when nothing anywhere is masked, and then forces a
token into rows that did not need one. In between — the common case — an
individual example can draw an all-clean mask and contribute an exact zero to the
batch mean.

At `batch_size: 2` and `t ~ U(eps, 1)` this is rare, and it is a small downward
bias rather than a divergence. Making it per-row would change the training
objective for every arm and every completed run, for an effect below §6's noise
floor. **It is written down because it looks like a bug on sight** — the next
person to read it will otherwise either "fix" it mid-project or spend an hour
deciding not to.

**Where it does matter:** the frozen eval applies the guarantee **per item**,
because an item with nothing masked is a zero that drags its bucket's mean down
while carrying no signal. That is a real train/metric divergence, confined to the
lowest `t` bucket, recorded per item as `forced` in `frozen_eval_v1.json`, and
written up in `FROZEN_EVAL_SPEC.md` §5a.

### O10 — The head-LR sweep is dropped

**Decided 2026-08-26.** §5.2 and §5 rule 3 call for a two-stage sweep: backbone LR
first, then a 3-point head-LR sweep for `nara` and `nalorta` at the selected
backbone LR. **The second stage is not being run.** Phase 4.1b, both files, is
cut; `configs/tier0_headlr_cp.yaml` will not be written. If
`tier0_headlr_matrix.yaml` is already queued it is harmless — 3 short runs — but
nothing waits on it.

Rationale: `head_learning_rate: None` means "use the backbone LR", which returns
the base optimiser untouched, and that is exactly the **middle cell** of the
proposed 0.3× / 1× / 3× grid. So skipping it is not leaving a hyperparameter
unset; it is taking the conventional single-LR setting and not spending 6 runs and
a serial queue round-trip to confirm it. The head/backbone split is implemented
(Phase 3) and stays available for Tier 2.

**Owed to the writeup, one sentence:** §5.2's separate parameter groups were
implemented but not swept; all arms trained single-LR. The exposure is that a
reader can ask whether the noise-aware arms were handicapped by a head LR tuned
for the backbone — note that this cuts *against* our method, not for it, which is
the safe direction for §5's stated failure mode.

### O11 — The epoch budget is selected post-hoc, not by a probe

**Decided 2026-08-26.** §10 step 4 and Phase 4.1c called for one long `nalorta`
run scored at each checkpoint to locate the epoch sweet spot, with every
subsequent run shortened to match. **That run is not being made as a separate
step.** Instead:

Tier 1 already saves every 100 steps with no `save_total_limit` (~7 checkpoints
per run), so the checkpoints get scored on dev-250 after the fact and the epoch
curve falls out as a byproduct — **per arm**, rather than extrapolated from one
`nalorta` probe onto four methods. Strictly more information, one less serial
step, and it needs exactly the build work (2.3) the probe needed anyway.

What it costs: if the peak is early, the tail epochs of all 12 runs are wasted
compute. The result is not lost — you select the checkpoint — and overfitting
becomes something you *observe* rather than something baked into the budget.

Consequence for O8: 12 long runs are now in flight simultaneously with no probe to
shorten them, which makes the wall-clock margin below the thing to get right.

### O8 — Slurm wall clocks (live operational constraint)

Measured on this cluster (3× L40): **6-epoch training ≈ 8 h**, **full
1319-question benchmark decode ≈ 6 h**. Dev-250 scales to ~1.1 h. These supersede
`BENCHMARK_PERF.md` entirely. All three predate the O1/O2 factored delta, which
should be cheaper but is untimed.

`_slurm_settings` (`batch_train.py:226`) now defaults to `08:00:00`.

**The edge:** a 6-epoch run *is* ~8 h — the default with no margin — and nothing
passes `resume_from_checkpoint`, so a wall-clock kill loses the run despite
`save_steps: 100` leaving checkpoints on disk. **Tier 1 configs must set
`sbatch_time: "12:00:00"`.** Frozen-eval benchmark jobs should go the other way:
~2000 forwards behind one model load, so `sbatch_time: "01:00:00"`.

Worth doing, not yet done: wire `resume_from_checkpoint`, which turns a
wall-clock kill from a lost run into a resubmit. More attractive now that O11 puts
12 long runs in flight at once.

---

## Fixed, with consequences that outlive the fix

### O1 / O2 — the CP-factored delta

Both fixed in one change, because they touched the same function and the fix for
one is what makes the other safe. Kept here only for what the writeup owes.

**O1 — `C_H` was applied to the wrong axis for q/k/v.** `dW` was built as
`cat_h[A · diag(C_h[h] ⊙ C_m ⊙ C_l) · B]`, giving an `(out, in)` matrix whose
head-structured axis was **axis 1 = `in_features`**. The targeted modules are
`q_proj`, `k_proj`, `v_proj`, `attn_out`:

| Module | Head axis | `C_H` indexed |
|---|---|---|
| `q_proj` / `k_proj` / `v_proj` | **output** | residual-stream slices ✗ |
| `attn_out` | **input** | attention heads ✓ |

So for 3 of the 4 adapted matrices, `C_H` modulated 32 arbitrary contiguous
128-wide slices of the residual stream, while the axis that *does* carry heads got
no per-head factor at all. It never crashed because `d_model == n_heads · d_head
== 4096` and LLaDA-8B has no GQA, so every matrix is square and every shape lines
up whichever orientation you assume. On a GQA model it would have failed loudly on
day one.

**O2 — λ was pooled over the micro-batch.** At generation pooling is exact:
`generate.py` masks the whole `gen_length` window for every row and every row
shares a schedule. At training it is not: `compute_loss` draws a separate `t` per
example and answer lengths differ, but `_mask_token_proportion` returned one
token-weighted scalar per micro-batch — so at `batch_size: 2` each example was
conditioned on the average of its own λ and an unrelated example's, off by up to
~0.5. NaRA never had this problem (`pool_lambda=False`), so **the two noise-aware
arms were being trained on different conditioning signals**, which lands directly
on RQ2.

**What landed.** The delta stopped being materialised. Both tuners carry factors,
not the product: `LorTaDelta` / `NALorTaDelta` hold `A` `(d_model, r)`, `B`
`(r, head_dim)`, `coef` `(batch_or_1, n_heads, r)` and a `head_axis` tag per
adapted matrix; `_apply_delta` contracts them against the activations in rank
space, with `attn_out` tagged `HEAD_AXIS_IN` and q/k/v `HEAD_AXIS_OUT`. The head
factor lands on the axis that carries heads *by construction* — there is no
orientation left to get wrong — and per-example λ becomes one elementwise multiply
on an `r`-wide tensor. `_mask_token_proportion` returns `[batch]`, mirroring
NaRA's `_lambda`. Both branches assert factor shapes against the base layer's real
`in_features`/`out_features`, and the merge path asserts `dW.shape ==
base.weight.shape` — the checks that would have caught O1.

Side effect: the dense form rebuilt ~8.6 GB of autograd graph per forward. The
factored form is `r`-wide throughout, so it is strictly cheaper.

**What this still means:**

- **RQ6's per-head claim is live again** — `C_h` now genuinely indexes attention
  heads — and `llada/plot_c_lambda.py` can be written against that.
- **The `lorta` baseline is publishable.** Shipping a headline 2×2 cell as "LoRTA"
  with the head factorisation misapplied was a real exposure.
- **RQ2 compares like with like**: both noise-aware arms now see per-example λ.
- **Adapters in `outputs/` predating the fix compute something different**, so
  their benchmark numbers are void. They used `rank: 128` / α/r = 0.031 and were
  already superseded by the α/r = 0.125 decision, so little is lost.
- `NALorTaConfig.pool_lambda` (default `False`) restores the old pooled scalar for
  reproducing those runs. `to_dense()` refuses a batched `coef` rather than
  silently merging row 0.

### O4 — LR grids

**Settled.** The matrix family (`nara`, and `lora` as its η=0 collapse) is **fixed
at `1e-4`** — no sweep. The CP family keeps §7's shifted grid `{1e-4, 3e-4, 1e-3}`,
swept at **both** η values rather than selecting on η=1 and reusing it, since
`lorta` is a headline 2×2 cell and §5's stated failure mode is winning because the
baseline was under-tuned. If both η cells select the same LR, say so — it makes the
η comparison cleaner, not weaker.

`1e-4` is both §7's matrix-grid midpoint and the conventional LoRA value, so it is
not an arbitrary pick. History: `configs/nara_vs_nalorta.yaml` records NaRA
diverging to all-NaN at lr 0.05 (262 of 263 checkpoint tensors non-finite; only
the frozen Fourier projection survived), so the *upper* end of the matrix grid was
the risky one. `_assert_adapter_finite` (`train.py:989`) is what now catches that
at the end of training rather than a day later in the benchmark.

**Owed to the writeup, one sentence:** fixing the matrix family's LR by fiat
deviates from §5 rule 1. The honest framing is that `1e-4` is the standard value
for two-factor adapters and the midpoint of the grid §7 proposed, and that the CP
arm — the one being advocated — is the one that got the search.

### O5 — W&B stays offline

**Settled.** `WANDB_MODE=offline` as `SBATCH_TEMPLATE` sets it, no `wandb sync`
epilogue, no `wandb.init` in benchmark mode (Phase 2.4 dropped — it would write
accuracy into an unsynced local directory, strictly worse than the JSON already
written). W&B is a local training-curve log only.

**The record of record for every reported number** is the run directory's
benchmark JSON plus `summary.json`. §8's required fields (`adapter_params`,
`eval_loss_frozen`, dev / test accuracy) live there.

### O7 — two plan defects, both silent

**Fixed in `IMPLEMENTATION_PLAN.md`.** (a) Tier 0 was ordered *behind* the epoch
probe, which needed the LR that Tier 0 exists to select — circular, and now moot
under O11. (b) §9.2's check was written against the pre-O1/O2 API and would have
failed on a type error rather than a wrong number; since §10 gates everything on
Phase 1, a check that cannot run stops Tier 0 as surely as one that fails. Now
asserts `delta.to_dense()` is exactly zero for every adapted module — which also
exercises the merge path and the `(out, in)` orientation — plus `lora_B` zero,
the property that makes it true.

Also settled and no longer worth arguing: α/r = 0.125 (`rank: 32`, `lora_alpha:
4`) in every new config; LLaDA-8B-Instruct is the only model in scope; §9.1 and
§9.3 dropped from the correctness job (see `IMPLEMENTATION_PLAN.md` A2.1, A2.6).

---

## Not yet built

| # | Component | Needed for | Status |
|---|---|---|---|
| 2.1 | `gsm8k_accuracy_dev` in `BENCHMARKS` | **Tier 1** | **build now** — two lines over `splits.dev_split`; `diffusion_evaluate` is already split-agnostic |
| 2.3 | Honour `adapter_name_or_path` in benchmark mode | **Tier 1 epoch selection (O11)**, Tier 3 | **build now** |
| 2.5 | `--benchmark` CLI override | scoring the same run dirs on dev then test | convenience, do with 2.1 |
| 2.6 | §8 run naming; no-expand set in `expand_runs` | comparability | partial — `eta`/`head_learning_rate`/`fourier_m` abbreviations exist; the no-expand set is still open |
| 5 | Θ mode / rank / init, MLP head | Tier 2 (RQ3, RQ4) | Tier 2 only |
| 5 | `llada/plot_c_lambda.py` | RQ6 | Tier 3 only — unblocked, O1 is fixed |

Everything else from the original checklist (0.3 `adapter_params.py`, 0.4
`splits.py`, 0.5 `frozen_eval.py`, 1 `correctness_checks.py`, 2.2 per-benchmark
result files, 3 `head_learning_rate`) is **done**. 2.4 was dropped by O5.

---

## Suggested order

1. **2.1 + 2.3 + 2.5**, now, while `tier0_lr` trains. This is the only thing
   between here and Tier 1.
2. `tier0_lr` reports → check the loss curve against an old run (~1e-7 drift, no
   more) → read the selected CP LR per η cell → `configs/locked_lrs.yaml`,
   recording the fixed `1e-4` for the matrix family alongside it.
3. Write `tier1_headline_cp.yaml` + `tier1_headline_matrix.yaml` — one file per
   family, since `expand_runs` is a cartesian product and `learning_rate` cannot
   vary with `tuning_type` in one YAML. `eta: [0, 1.0]` × `seed: [42, 43, 44]`,
   `rank: 32`, `lora_alpha: 4`, `lora_dropout: 0.1` (**O3**), `sbatch_time:
   "12:00:00"` (**O8**). Submit.
4. Score Tier 1 checkpoints on dev-250 for the epoch curve (**O11**), then a
   single `--mode benchmark --benchmark gsm8k_accuracy` pass over the final 2×2
   only.
5. **O6** once Tier 1 is running, for the frontier extension.
6. Tier 3 (free — no training), then Tier 2.

Worth ten minutes before Tier 1 fixes the baseline in stone: `error.err`
(untracked, repo root) shows the baseline truncating mid-sentence at `<|eot_id|>`
after ~20 tokens, which means the current zero-shot number is probably not a fair
one.
