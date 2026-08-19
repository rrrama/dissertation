# LLaDA GSM8K benchmark — performance work

Handoff spec. Everything below is derived from reading the code at commit `42d81fa`
("added parallel benchmarking"); **none of it has been verified against a running
job**, because the cluster was not reachable from the machine where this was
written. Treat the numbers as estimates and the diagnosis in §5 as a hypothesis to
be confirmed, not a conclusion.

## 0. Context

`llada/train.py --mode benchmark` scores a trained adapter on the GSM8K test set
(1319 questions) using the LLaDA diffusion sampler in `llada/generate.py`. Each
question is decoded with `diffusion_steps=256`, `gen_length=256`,
`block_length=256` (so `num_blocks == 1`, 256 steps per question), batch size 1.

Commit `42d81fa` changed benchmark mode from a single `python train.py` process to
`torchrun --nproc_per_node=3`, with each rank taking an interleaved shard of the
test set. The observed wall clock went from ~28 h to ~20 h — a 1.4x speedup where
~3x was expected.

Config in use: `llada/configs/lorta_vs_nalorta.yaml` (two runs, `lorta` and
`nalorta`, on `GSAI-ML/LLaDA-8B-Instruct`, `gpu:lovelace_l40:3`).

## 1. What is *not* the problem

Checked and ruled out — don't spend time re-deriving these:

- **The sharding is correct.** `train.py:323-327` strides
  `range(rank, len(questions), world_size)`, and `_gather_predictions`
  (`train.py:285`) reassembles in test-set order. Each rank does ~440 questions.
- **The ranks are on separate GPUs.** `train.py:570` uses `training_args.device`
  (i.e. `cuda:LOCAL_RANK`), not `"cuda"`. Independently: bf16 LLaDA-8B is ~16 GB,
  so three ranks piled onto one 48 GB L40 would OOM rather than run slowly.

So the 1.4x is not a lost 3-way split. Each rank is doing 1/3 of the questions but
each question got roughly 2x slower. §5 covers why; §2-§3 are the much larger wins
and are worth doing regardless of how §5 resolves.

## 2. Remove the fp64 softmax (largest expected win)

**Status: implemented, unverified on hardware.** Done as a prerequisite of §4 —
at batch 16 the fp64 tensor below is ~5 GB and would OOM alongside the model.
The landed version is `_selected_token_prob` (`llada/generate.py`), which runs
`F.log_softmax(..., dtype=torch.float32)` over the rows in chunks of a fixed byte
budget (256 MB) and keeps only the gathered `(B, S)` column. An unchunked
`log_softmax` was the first attempt, but it still allocates the full `(B, S, V)`
fp32 tensor (~2.6 GB at B=16, S=320) just to read one element per row out of it,
and that allocation — not the weights — is what caps `benchmark_batch_size`.
Softmax rows are independent, so chunking is the same arithmetic per row.

**File:** `llada/generate.py:122-124`

```python
if remasking == "low_confidence":
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
```

`logits` is `(1, S, 126464)` with S ≈ 320 (prompt ~64 + gen_length 256). The cast
materialises ~324 MB; the softmax reads and writes it several times (~1 GB of
traffic) and runs ~40M `exp()` calls in fp64. The L40 runs FP64 at 1/64 of FP32
rate with no hardware transcendentals, so this is disproportionately expensive.

Sanity check on the magnitude: 28 h / 1319 questions / 256 steps ≈ **300 ms per
step**. An 8B bf16 forward at S=320 is ~5.1 TFLOP, i.e. ~50-80 ms on an L40. The
remaining ~220 ms/step is not otherwise accounted for.

`x0_p` is only ever used to *rank* candidates inside `torch.topk` at line 138, so
fp32 is more than sufficient, and the full `(1, S, V)` probability tensor never
needs to exist:

```python
if remasking == "low_confidence":
    logits_f32 = logits.float()
    x0_logit = logits_f32.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    x0_p = (x0_logit - torch.logsumexp(logits_f32, dim=-1)).exp()
```

`logsumexp` reduces to `(1, S)` rather than materialising a second `(1, S, V)`
tensor.

Note the fp64 in `add_gumbel_noise` (`generate.py:28-29`) is a *different* thing
and is deliberate per the reference implementation — but it is dead code in this
config, since `gen_temperature: 0.0` makes `add_gumbel_noise` return immediately
at line 26. Leave it alone.

**Expected:** ~3-4x on the sampling loop. This is the single biggest item here.

## 3. Remove the per-step device sync

**Status: implemented, unverified on hardware.** `get_num_transfer_tokens` now
moves `mask_num` to the CPU at line 41, so both the `k` and the `remainder[i]`
syncs are gone. Line 138 remains its only consumer.

**File:** `llada/generate.py:46-53` and `:138`

`get_num_transfer_tokens` builds `num_transfer_tokens` on `mask_index.device`
(CUDA). At line 138:

```python
_, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
```

`k` must be a Python int, so the 0-d CUDA tensor goes through `__index__` →
`.item()` → a full device sync. That happens 256 times per question and prevents
the CPU from ever running ahead of the GPU.

Fix: return `num_transfer_tokens` on CPU (it is a tiny `(B, steps)` int tensor and
is only ever indexed for a Python int). Add `.cpu()` before the return at line 53.
While there, `remainder[i]` at line 51 is also a CUDA tensor used as a slice bound
— same sync, but only once per question, so it is minor; moving `mask_num` to CPU
at line 41 fixes both.

Verify nothing else consumes `num_transfer_tokens` on-device — as of `42d81fa`
line 138 is the only use.

**Expected:** ~1.2-1.5x, and it makes step timings much easier to interpret.

## 4. Batch the sampler (largest remaining headroom)

**Status: implemented, unverified on hardware.** How it landed:

- `generate()` takes `(B, prompt_len)` prompts plus an `attention_mask`. The open
  question in this section — whether padding is genuinely free — resolved in our
  favour: `modeling_llada.py` folds a `(batch, seq)` `attention_mask` into the
  attention bias built by `get_bidirectional_attention_bias`, and
  `_scaled_dot_product_attention` passes `is_causal=False`. So masked padding is
  properly excluded. Positions are RoPE, i.e. relative, so the left shift of a
  padded prompt does not move it relative to its own tokens.
- Batches hold **only questions whose prompts tokenize to the same length**
  (`_uniform_length_batches`), which is what makes batched decoding exactly
  output-preserving. See §4a — batching by merely *similar* length is not exact
  under NA-LoRTA. Equal lengths also mean zero padding, and a padded position
  costs a full forward at each of the 256 steps.
- Ragged batches are still supported by `generate()` (**left**-padded, so
  `block_start`/`block_end` stay shared across the batch — the per-row block
  bookkeeping this section anticipated is not needed) but are not used by the
  benchmark. `generate()` asserts the pad id is not `mask_id`, since padding that
  looked like MASK would get decoded.
- Batch size is `benchmark_batch_size` (`TrainingArguments`, default 8); set it in
  the run config. It is a *cap*: real batches are limited by how many questions
  share a prompt length, so they are variable and often smaller. Rank 0 prints the
  realised batch count and mean at startup. `benchmark_batch_size: 1` reproduces
  the old path.

Unverified and worth watching on the first run: peak memory. With the chunked
confidence softmax (§2) the largest live tensor scaling with `B` is the model's own
bf16 `(B, S, V)` logits — ~1.3 GB at B=16, S=320, plus a fixed 256 MB chunk buffer,
on top of ~16 GB of weights. So the batch ceiling is set by the forward itself
rather than by the confidence computation; 32 is worth trying. Measure with
`torch.cuda.max_memory_allocated` rather than trusting this arithmetic.

`llada/check_batching.py` decodes the same questions batched and unbatched, diffs
the strings and reports both timings and peak memory — run it before trusting any
accuracy number (§"Correctness guardrails"). It deliberately picks questions that
*share* a prompt length, since the first N test questions would give mostly
singleton buckets and exercise nothing.

## 4a. Why batches must be uniform-length (NA-LoRTA)

Not in the original spec; found while implementing §4.

`NALorTaModel.forward` modulates the adapter weights by the proportion of the
input that is the MASK token (`peft/src/peft/tuners/nalorta/model.py:191-197`),
via `c_mask = 1 + eta * Theta^T phi(mask_proportion)`. `_mask_token_proportion`
returns **one scalar pooled over the whole batch**, and the resulting `c_mask` is
applied to every row's weights. So under NA-LoRTA a row's decode can depend on its
batch-mates. (Plain LoRTA is unaffected: `LorTaModel.forward` calls
`_compute_weights_from_tensor()` with no inputs.)

Uniform-length batching removes the dependence rather than bounding it. At any
sampling step every row holds the same *count* of masked tokens — they share the
`get_num_transfer_tokens` schedule — so the per-row proportion is
`masked / (prompt_len + gen_length)` and differs across rows only via
`prompt_len`. Equal prompt lengths ⇒ pooled scalar == per-row value, exactly.

Two things worth recording:

- **Training pools too, and includes padding.** `compute_loss` calls
  `model(input_ids=noisy_batch)` (`train.py:267`) with no `attention_mask`, at
  `per_device_train_batch_size: 2`. So during SFT the proportion is pooled over
  two examples *and* diluted by right-padding. Per-row exactness was never the
  trained regime; batch-1 eval is simply the reference the existing numbers use.
- **A per-row `c_mask` is cheap in principle, and was deliberately not done.**
  Since `c_mask` sits in a diagonal, `x @ (A diag(c) B) == ((x @ A) * c) @ B`, so
  per-row `c` costs a broadcast on a `(B, S, r)` tensor. It would need
  `nalorta/layer.py` to take the factors instead of a merged `dW` — a change to
  the training path, out of scope for a perf pass. Related: the merged form
  rebuilds every layer/matrix/head weight on *every one of the 256 steps*
  (~275 GFLOP/step at r=128, against a ~5.1 TFLOP forward), so the factored form
  would be a modest win in its own right.

---

*Original analysis, kept for reference:*

`generate()` is hardcoded to batch size 1 (`generate.py:86`, and the docstring
says "single (batch size 1) prompt"), and `diffusion_evaluate` calls it one
question at a time. A single 320-token forward pass leaves an L40 badly
underutilised; batching 8-16 questions per rank is worth more than the 3 GPUs are.

Most of the loop body is already batch-shaped (`for j in range(confidence.shape[0])`
at line 137, `mask_num` per-row at line 41). The work is:

- Left-pad ragged prompts within a batch and carry an attention mask, or bucket
  questions by prompt length so each batch is near-uniform. LLaDA is a masked
  diffusion model with bidirectional attention, so padding must be masked out
  properly rather than just ignored — confirm against `modeling_llada.py` how it
  handles an attention mask before assuming this is free.
- Make `block_start` / `block_end` per-row (line 131 `x0_p[:, block_end:] = -inf`
  assumes a shared prompt length).
- `x[transfer_index] = x0[transfer_index]` at line 141 already works batched.


**Expected:** ~3-5x, on top of §2 and §3.

## 5. Diagnose the per-rank regression (unresolved)

After §2-§4 this may stop mattering, but it is currently unexplained and should be
measured rather than guessed at.

The 1.4x aggregate with a correct 3-way split implies each rank runs ~2x slower per
question than the old single-process job did. Two candidate explanations, neither
confirmed:

1. **The comparison may not be apples-to-apples.** The tqdm bar at
   `train.py:330-334` uses `total=len(local_indices)`, so rank 0's ETA covers its
   own ~440-question shard — i.e. it *is* the whole-benchmark ETA and is directly
   comparable to the old 1319-question ETA. But if both the 28 h and 20 h figures
   are early-run ETAs rather than completed runs, tqdm's smoothed rate weights
   warm-up heavily (NFS model load from `$SHARE/u5751903/models/`, first-question
   CUDA warm-up), and some of the gap is an artefact. **Establish first whether
   28 h was ever a completed run.** Per §6 it cannot have been.
2. **CPU/sync contention.** With the §3 sync in place the process is
   latency-bound and spin-waits on a core. The sbatch template requests
   `--cpus-per-task=3` (`batch_train.py:77`, default 3 at `:216`) for what is now
   3 ranks instead of 1, and torchrun additionally forces `OMP_NUM_THREADS=1`,
   which the old `python train.py --mode benchmark` path did not. Fixing §3 should
   largely dissolve this; raising `cpus_per_task` (it is already a config key,
   `batch_train.py:54`) is a cheap independent test.

**Instrumentation to add** so the next run answers this directly:

- Log, once per rank at the start of `diffusion_evaluate`: `rank`, `world_size`,
  `torch.cuda.current_device()`, `model.device`, and `len(local_indices)`.
  Currently only rank 0 prints anything (`train.py:352`, `disable=rank != 0` at
  `:333`), so a misplaced or stalled rank is invisible.
- Time each question and log a running mean per rank (not just rank 0) — for
  example every 25 questions. Comparing rank 0's per-question mean against a
  1-GPU control run is the decisive measurement.

## 6. The job cannot finish at current speeds

`sbatch_time` defaults to `"12:00:00"` (`batch_train.py:217`, template at `:79`)
and `configs/lorta_vs_nalorta.yaml` does not override it. Both the 28 h and the
20 h benchmarks would be killed by Slurm at 12 h.

Either add `sbatch_time` to the config, or — better — land §2 and §3 first and
re-measure, since the target is to get comfortably under the existing limit.

Related: `_slurm_settings` (`batch_train.py:205-220`) no longer distinguishes
train from benchmark mode, so `mode` is now an unused parameter there. Harmless,
but worth removing if you touch that function.

## Suggested order

§2, §3 and §4 are now implemented (in that dependency order — §2 had to land with
§4 for memory reasons). Nothing has been run. What remains:

1. Run `check_batching.py` on one GPU with `--batch_size 8`. It answers three
   things at once: whether batched decodes are identical to unbatched, what the
   actual speedup is, and how big the same-length buckets are in practice (the
   thing that caps the realised batch size, and the main unknown in the §4
   estimate).
2. Sweep `benchmark_batch_size` (8 / 16 / 24) on one GPU for the largest that
   fits, watching `torch.cuda.max_memory_allocated`. Note the cap only binds if
   buckets are big enough to reach it.
3. Add the §5 instrumentation and re-run a short benchmark (e.g. 30 questions per
   rank) on 1 GPU and on 3 GPUs. §5 may well have dissolved with the §3 sync fix,
   but it is still unmeasured.
4. Recompute the projected full-run time and settle §6 (`sbatch_time`) against it.

## Correctness guardrails

The sampler's output must be unchanged by §2-§3 at `gen_temperature: 0.0`:

- §2 changes the *precision* of the confidence values used for ranking, not the
  ranking rule. Ties broken differently by fp32 vs fp64 are possible in principle;
  if you want certainty, decode a handful of questions before and after and diff
  the decoded strings.
- §3 is a pure device placement change and must be exactly output-preserving.
- §4 is not obviously output-preserving — padding changes what the bidirectional
  attention sees. Validate it by decoding the same questions batched and unbatched
  and diffing, before trusting any accuracy number produced with it.

Any accuracy figure produced during this work should be regarded as provisional
until a full run completes with the final code.
