# Experiment Design — Noise-Aware LoRTA (NaLoRTA) on LLaDA

Audience: coding agent working in this repo. This document specifies **what to run, in
what order, and what to record**. It does not specify implementation details of the
adapters themselves — those already exist.

**This is the design of record and is not edited to match what was built.** Where
execution deviates from it, the deviation and its justification live in
`OUTSTANDING.md`; `IMPLEMENTATION_PLAN.md` holds the build order. Sections below carry
inline pointers where that has happened. Every original **[DECIDE]** has now been
decided and the decision is written in place.

---

## 1. Claim under test

> Noise-conditioning can be integrated natively into a tensor-decomposed adapter,
> matching or exceeding MLP-based noise-aware LoRA (NaRA) at a small fraction of the
> trainable parameters.

This is a **joint** claim on two axes (accuracy, parameter count). Every headline result
must report both. A win on accuracy alone, or on parameters alone, is not the result.

## 2. Methods under comparison

| Key | Name | Factorisation | Noise-aware | Notes |
|---|---|---|---|---|
| `none` | Base model | — | — | Zero-shot LLaDA, no adapter |
| `lora` | LoRA | matrix (BA) | no | Classic baseline |
| `lorta` | LoRTA | CP (A,B,C_L,C_H,C_M) | no | Parameter-efficient baseline |
| `nara` | NaRA | matrix (BCA) | yes (MLP) | Main competitor |
| `nalorta` | **Ours** | CP + Θ | yes (linear-on-Fourier) | Proposed method |

**Critical structural property — exploit it.** Setting `eta = 0` collapses
`nalorta` → `lorta` and `nara` → `lora` *exactly*. Therefore the non-noise-aware cells
of the 2×2 must be produced by **the same code path with `eta=0`**, not by a separate
implementation. This eliminates implementation confounds from the central comparison and
must be stated as such in the writeup. Add an assertion test for it (see §9).

## 3. Definitions to fix once and reuse everywhere

### 3.1 λ (noise level)
λ is the **global mask fraction over the full generation window** (`gen_length` tokens),
i.e. `num_masked / gen_length`. Under semi-autoregressive block decoding with
`block_length=32`, `gen_length=256`, λ decreases monotonically 1 → 0 across the 8 blocks.

Known train/inference mismatch to be **documented, not fixed** in the first pass: LLaDA
training masks are i.i.d. scattered; inference masks at a given λ are *structured*
(earlier blocks clean, later blocks fully masked). λ matches; mask geometry does not.
The adapter conditions only on λ. Flagged as a Tier 3 ablation.

### 3.2 Trainable parameter count
Counted identically for all methods and reported in every results table.

- **Include:** all adapter tensors that receive gradients — LoRA `A,B`; LoRTA
  `A,B,C_L,C_H,C_M`; NaRA core-MLP weights+biases and its noise-embedding projection;
  NaLoRTA `Θ`.
- **Exclude:** frozen base-model weights; the sampled Fourier frequency vector **k**
  (sampled once, frozen, for both NaRA and NaLoRTA — exclude for both or include for
  both; default is exclude).
- Implement as a single shared utility `count_trainable_adapter_params(model)` used by
  every run and logged to W&B as `adapter_params`. Do not hand-compute per method.

### 3.3 Data splits
- **Train:** GSM8K train, minus the dev carve-out.
- **Dev:** 250 questions held out **from GSM8K train** (fixed seed, saved to disk as an
  index list so it is identical across all runs). Used for *all* model/hyperparameter
  selection.
- **Test:** full GSM8K test (1319). Touched **only** for the final Tier 1 numbers that
  appear in the dissertation. Never used for selection.

### 3.4 Frozen evaluation masks
Diffusion eval loss is dominated by the random draw of `t` and the random mask. Before
any runs, pre-sample a **fixed** set of `(t, mask)` pairs (suggest 8 t-values spanning
(0,1] × the dev set) and persist them. Every eval of every run uses this identical set.

This is the single biggest variance-reduction lever in the project and makes LR ranking
possible from short runs. Log as `eval_loss_frozen`.

## 4. Research questions

- **RQ1 — Frontier.** Accuracy vs trainable parameters for all four adapters on GSM8K.
  Deliverable: accuracy vs log-params curve, each method swept over rank. Report
  (a) accuracy at iso-parameters, (b) parameter ratio at iso-accuracy.
- **RQ2 — Is it noise-awareness or factorisation?** The 2×2 {matrix, CP} × {noise, no
  noise} via the `eta=0` collapse. Hypothesis: noise gain > 0 in both columns, and the
  gain under CP is not smaller than under matrix factorisation.
- **RQ3 — Is linear-on-Fourier expressive enough?** Ablations: η sweep; Fourier size `m`
  sweep; MLP head on the CP backbone (isolates noise-function expressivity from
  backbone); **shared scalar vs per-rank vector** (tests the claim that CP lets noise
  modulate layers/heads differentially — this is the part of the contribution that is
  not merely parameter count).
- **RQ4 — Role of R.** Note and state the confound: raising R raises *both* backbone
  capacity and the number of linear combinations available to Θ. Disentangle by fixing
  the backbone R and low-rank-restricting Θ (or vice versa).
- **RQ5 — Sampler robustness.** Vary `diffusion_steps` and `block_length` at eval time
  on existing checkpoints. Hypothesis: noise-aware adapters degrade more gracefully at
  low step counts. Zero training cost; potentially the most differentiating result.
- **RQ6 — What is learned.** Post-hoc plots of C_λ(λ) over λ∈[0,1] from saved
  checkpoints: monotonicity, saturation, per-rank specialisation, per-layer
  differentiation. Zero training cost.
- **RQ7 — Generality (optional).** Second dataset and/or LLaDA-1.5B. Only if Tier 1–2
  come out clean.

## 5. Hyperparameter protocol (validity-critical)

The four methods have gradient scales differing by orders of magnitude for *structural*
reasons: LoRTA's update is a product of five factors, so the gradient w.r.t. any factor
carries the product of the other four; Θ sits multiplicatively on top of all five,
scaled by η. CP-factorised adapters characteristically need a substantially higher LR
than a two-factor LoRA.

**The failure mode to avoid:** tuning our method carefully, taking NaRA's published LR,
and winning by 2 points. That result is not defensible.

Rules:
1. **Identical LR grid, identical budget, per-method selection on dev.** Report the
   selected LR for every method in a table in the writeup.
2. **Separate optimiser parameter groups** for the noise head (Θ for NaLoRTA, the MLP
   for NaRA) vs the backbone factors. A single global LR forces a compromise between
   components with different curvature and makes *both* noise-aware methods look worse
   than they are.
3. **Two-stage sweep, not a grid:** tune backbone LR with the head at a default, then
   sweep head LR at the chosen backbone LR. 3+3, not 3×3. — **Stage 2 is not run; see
   §7 and `OUTSTANDING.md` §O10.**
4. **Fix α/r across all runs** and sweep LR only. α/r multiplies the update and is
   degenerate with LR; sweeping both wastes a grid dimension. Current config has
   `lora_alpha: 4`, `rank: 32` → α/r = 0.125. State this in the writeup.
5. **Warmup and epochs are shared constants**, not tuned per method.

Budget context: effective batch = 2 × 10 × 3 = 60; ~7.5k train examples → ~125 steps per
epoch, ~750 steps over 6 epochs. This is small enough that LR and warmup interact
strongly. **Run one long baseline first** and inspect where dev accuracy peaks — if it
peaks at epoch 3, halve every subsequent run.

## 6. Evaluation protocol

Three tiers of evaluation, used at different stages. Do not use the expensive one early.

| Stage | Metric | Cost |
|---|---|---|
| LR ranking (Tier 0) | `eval_loss_frozen` on frozen (t, mask) set | cheap |
| Method selection (Tier 1–2) | GSM8K accuracy, generation, **250-question dev** | moderate |
| Final reported numbers | GSM8K accuracy, generation, **full 1319 test** | expensive |

Generation settings for accuracy eval (held constant unless the RQ varies them):
`gen_length=256`, `diffusion_steps=256`, `block_length=32`, `remasking=low_confidence`,
`gen_temperature=0.0`.

**Statistical bar:** GSM8K accuracy on an 8B model has enough seed variance that a 1–2
point gap is not a result. Any comparison appearing in the abstract needs **≥3 seeds**
(42, 43, 44) and must report mean ± std. Prefer cutting breadth over cutting seeds.

## 7. Run plan

### Tier 0 — Hyperparameter search (short runs, single seed 42)
- 4 methods × 3 backbone LRs, 1–2 epochs, `eval_loss_frozen` only, **no generation**.
- Then 3-point head-LR sweep for `nara` and `nalorta` only at their selected backbone LR.
- ≈18 short runs. Output: a locked LR table, committed to the repo.

**DECIDED — LR grid** (`OUTSTANDING.md` §O4). The CP family takes the shifted grid
`{1e-4, 3e-4, 1e-3}`, swept at **both** η values so `lorta` is not an under-tuned
baseline. The matrix family is **fixed at `1e-4`** — the midpoint of the proposed matrix
grid and the conventional LoRA value — and not swept, which is a deliberate deviation
from §5 rule 1 and owes one sentence in the hyperparameter table. The
extend-rather-than-accept rule still binds on the CP grid.

**DEVIATION — the head-LR sweep is not run** (`OUTSTANDING.md` §O10). Stage 2 of §5
rule 3 is cut; separate parameter groups are implemented but every arm trains
single-LR. Tier 0 is therefore **6 runs, not ≈18**.

### Tier 1 — Headline (RQ1 + RQ2)
- 2×2 at matched rank (R=32), full training, **3 seeds** → 12 runs.
  - `lorta` and `lora` cells produced via `eta=0` on the same code path.
- Frontier extension: 2 additional ranks for `lorta` and `nalorta`, 1 seed each. Curve
  *shape* matters here, not error bars.
- Dev accuracy for all; **full test only for the final 2×2**.

**DECIDED — rank sets, parked until Tier 1 is running** (`OUTSTANDING.md` §O6). R ∈
{8, 16, 32} does **not** give overlapping parameter counts across families — LoRTA at
r=32 is ~137 k against LoRA's ~33.6 M — so **RQ1(a) "accuracy at iso-parameters" is not
obtainable from it**. R=32-for-everything stays as the headline 2×2, but it is the
"matched rank" cell and must not be called iso-parameter.

### Tier 2 — Ablations (RQ3 + RQ4), single seed, our method only
- η sweep: {0, 0.1, 0.5, 1.0}.
- Fourier `m` sweep: {16, 32, 64}.
- Shared scalar vs per-rank vector.
- MLP head on CP backbone.
- Θ init: current `kaiming_uniform_(a=sqrt(5)*init_scale)` vs **zero-init**. Note: `a`
  is the leaky-ReLU negative slope, so `init_scale` *shrinks* the init as it grows —
  easy to misread as a multiplier. Since LoRTA already has B=0 (so dW=0 at init) this is
  a secondary question, hence Tier 2 not Tier 0.
- Θ rank restriction for the RQ4 confound.

None of these require re-running baselines.

### Tier 3 — Free results (no training)
- RQ5: sweep `diffusion_steps` ∈ {32, 64, 128, 256} and `block_length` ∈ {16, 32, 64}
  on existing Tier 1 checkpoints. Produces a second Pareto axis: accuracy vs latency.
- RQ6: load checkpoints, plot C_λ(λ) across λ∈[0,1], per rank and per layer.
- Optional: block-structured masking distribution at train time (§3.1 mismatch).

Total ≈40 training runs, most short.

## 8. Logging requirements

Every run logs to W&B: `method`, `rank`, `eta`, `fourier_m`, backbone LR, head LR, seed,
`adapter_params` (from the shared utility), `eval_loss_frozen`, dev accuracy, and — where
applicable — test accuracy. Checkpoints must be retained for Tier 3; do not delete
adapters after training.

Run naming: `{method}_r{rank}_lr{backbone_lr}_hlr{head_lr}_eta{eta}_s{seed}`.

## 9. Correctness tests to add before running anything

Implemented as `llada/correctness_checks.py`; **passed 2026-08-26**, clearing §10's
gate. Two of the five were dropped by decision (`IMPLEMENTATION_PLAN.md` A2.1, A2.6).

1. ~~**η=0 collapse.**~~ **DROPPED.** The RNG streams differ between the two tuners
   (`NALorTaModel.inject_adapter` draws Θ and the Fourier **k** before `lora_A`;
   `LorTaModel` does not), so "bit-identical given the same seed" cannot hold as
   written. The collapse is evident from `c_mask = 1 + eta * (...)` at `eta = 0`.
2. **dW = 0 at init.** Assert the adapter's contribution to the base weights is exactly
   zero at step 0 for every method (LoRTA's B=0 should guarantee this; verify rather
   than assume, and verify it holds for NaLoRTA too). *As built:* asserts
   `delta.to_dense()` is exactly zero per adapted module — which also exercises the
   merge path and the `(out, in)` orientation §O1 was about — plus `lora_B` zero.
3. ~~**Per-slice formula.**~~ **DROPPED** — notation only (`Bᵀ` here vs `B` of shape
   `(r, head_dim)` in the code; four `Diag()`s vs one elementwise product). Same
   operation; fix the notation in the writeup. Checking it is what turned up §O1.
4. **Parameter counter.** Unit-test `count_trainable_adapter_params` against hand-worked
   values for one small config per method.
5. **Frozen eval determinism.** Two evals of the same checkpoint return identical
   `eval_loss_frozen`. *As built:* perturbs `lora_B` first (at init dW = 0, so a fresh
   adapter would score the base model against itself) and hooks the dropout modules to
   assert `training=False`, rather than comparing an eval-mode loss against a
   train-mode one.

## 10. Order of work

1. Tests in §9. 2. Frozen eval set + dev split, persisted. 3. Parameter counter.
4. One long `nalorta` baseline to locate the epoch sweet spot. 5. Tier 0. 6. Tier 1.
7. Tier 3 (cheap, do before Tier 2 if time is tight). 8. Tier 2.

Do not proceed past step 1 if any assertion in §9 fails — raise it instead.

**CORRECTED ORDER.** Step 4 cannot precede step 5: the epoch probe is a single long
`nalorta` run, so it needs the learning rate Tier 0 exists to select (`OUTSTANDING.md`
§O7a). It is also cut as a separate step entirely (§O11) — Tier 1's own `save_steps`
checkpoints are scored on dev-250 post-hoc instead, which gives the epoch curve **per
arm** rather than extrapolating one `nalorta` probe onto four methods, and removes a
serial round-trip before the only expensive tier. Steps 2 and 3 are done; step 1 has
passed. The order actually being run is: **1, 2, 3 → 5 (Tier 0, 6 runs) → 6 (Tier 1,
with epoch selection folded in) → 7 → 8.**
