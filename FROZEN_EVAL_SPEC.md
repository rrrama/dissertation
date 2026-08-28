# `eval_loss_frozen` — reference

What `llada/frozen_eval.py` and the `compute_loss` split actually do, and the
properties other code is allowed to rely on. Built and signed off 2026-08-25;
§9.5 (frozen-eval determinism) passed 2026-08-26.

`eval_loss_frozen` is Tier 0's only metric (§3.4, §6). It is the one piece of
Phase 0 that edits the training objective every run in the project shares, which
is why it has its own document.

---

## 1. The split

`compute_loss` keeps the sampling of `t` and the mask. Everything from
`noisy_batch` onward — the ≥1-masked-token guarantee, the `response_mask` forward
kwarg, the `1/p_mask` reweighting, the per-example answer-length normalisation —
lives in a pure function both training and eval call:

```python
def diffusion_masked_loss(
    model,
    input_ids,            # (b, l)  clean tokens
    response_mask,        # (b, l)  bool, labels != IGNORE_INDEX
    attention_mask,       # (b, l)
    t,                    # (b,)    per-example noise level in (0, 1]
    masked_indices,       # (b, l)  bool, subset of response_mask, >=1 True per row
    mask_id,
    diffusion_eps,
    pass_response_mask=True,
) -> tuple[torch.Tensor, torch.Tensor]:   # (b,) per-example loss, NOT the batch mean;
                                          # plus logits, for `return_outputs`
```

Extracted rather than reimplemented so the eval and the training objective cannot
drift: a second implementation would diverge the first time either was touched,
and **the Tier 0 metric would silently stop measuring the thing being trained** —
the loss would still be plausible, still monotone in adapter quality, still
deterministic. `compute_loss` is now the sampling block, one call, `.sum() / b`.

Four properties worth knowing:

**Returns `(b,)`, not a scalar.** The eval needs per-example values so its mean is
over the *eval set*, not an average of per-batch means; those differ whenever the
last batch is short.

**Arithmetically identical, not bitwise.** Today's `torch.sum` ran over one flat
vector of masked tokens; the new form sums per row then over rows, and float
addition is not associative. **Expect ~1e-7 relative drift on the loss curve and
do not go looking for a bug in it.**

**Takes `t`, not `p_mask`.** `p_mask = (1 - eps) * t + eps` is derived inside from
the trainer's own `diffusion_eps`. `t` is what §3.1 defines and what is worth
persisting; deriving `p_mask` keeps the eval faithful if `diffusion_eps` ever
changes rather than silently pinning an old value.

**`pass_response_mask` is explicit, not sniffed.** Everything adapted takes
`response_mask` as a forward kwarg; the bare base model (`tuning_type: none`) does
not, and would raise `TypeError`. Set `False` at exactly one call site — the
baseline branch of the frozen-eval benchmark — rather than inferred from
`special_peft_forward_args`, because inferring it is precisely how a wrapper that
hides that attribute turns into a silent fallback to a prompt+answer λ
denominator.

## 2. The frozen set

250 dev questions (`splits.dev_split`) × 8 noise levels = **2000 items**, fully
crossed. Crossing rather than one-`t`-per-question is the variance reduction §3.4
asks for: each adapter is scored on the same question at the same noise level
every time.

Noise levels are stratified midpoints, `t_i = (i + 0.5) / 8` for `i` in `0..7` →
`0.0625 … 0.9375`. Even spacing over the open interval, no `t→0` (nothing masked)
and no `t=1` (everything masked) degenerate endpoint. As a quadrature of the
`U(eps, 1]` training draws from, the mean over the eight buckets estimates the
same expectation the training objective takes.

Committed to `llada/split_data/frozen_eval_v1.json` — JSON rather than `.pt`
because it is ~2000 short integer lists, it wants to be diffable, and
`torch.load` on a committed binary is a liability for what is essentially a config
file.

```
{
  "version": 1,
  "tokenizer": <name_or_path>, "model_max_length": <int>,
  "dev_seed": 20260825, "dev_size": 250, "num_t": 8,
  "content_sha256": <hex>,
  "items": [
    {"dev_pos": int, "gsm8k_index": int, "t_index": int, "t": float,
     "masked_positions": [int, ...], "forced": bool},
    ...
  ]
}
```

`dev_pos` is the position *within the dev split* — the index into the
`SupervisedDataset` built over `dev_split(...)`, and the only one anything indexes
with. `gsm8k_index` is the absolute row in GSM8K train, carried for traceability
and never used to look anything up. Both are stored because conflating them is the
obvious bug here, and two differently-named fields cannot be conflated silently.

`masked_positions` are absolute token positions within *that example's own*
sequence. The collator right-pads (`DataCollatorForSupervisedDataset:306`), so
those are also the correct positions in any padded batch, with no offset
arithmetic. A padded `(2000, L)` tensor would instead bake in a batch composition
and break the moment batch size changed.

`content_sha256` covers the tokenizer name, `model_max_length`, the tokenised
`input_ids` of all 250 dev questions, **and each example's response boundary**
(first response position and response length). The boundary is in the hash
because the positions are only meaningful relative to it. Any change to
tokenisation, prompt template, or the dev split moves the hash and **the loader
refuses to run** — the alternative is scoring two adapters against different masks
and reporting the difference as a result.

The loader additionally asserts, per item, that every stored position lies inside
that example's response region and that the list is non-empty. The hash cannot
catch an indexing bug in the generator; this can, and it costs nothing.

Generation and evaluation both build the dataset the same way, which is what makes
the hash meaningful:

```python
raw  = load_dataset(data_args.data_name, "main", split="train")
data = SupervisedDataset(raw_data=dev_split(raw), tokenizer=tokenizer)
```

## 3. Determinism — what §9.5 asserts

**One batch is one question's eight noise levels.** All eight rows are the same
token sequence, so there is no padding at all, batch composition is fixed by
construction rather than by a batch-size setting, and sharding splits questions —
never batches — across ranks. 250 batches of 8.

That is structural on purpose. Independence from batch size **is not achievable by
careful accumulation**: padded rows change the shapes the attention and MLP
matmuls reduce over, and in bf16 that alone perturbs the result. Accumulation
order is a real problem but not the only one.

Together with:

- `model.eval()` — **`lora_dropout` is 0.1**, so without this the metric is
  stochastic and every Tier 0 comparison is noise. This is the failure §9.5 exists
  to catch, and it is asserted by hooking the dropout modules rather than by
  comparing losses.
- `torch.no_grad()`.
- Per-example losses accumulated as `float64` keyed by `(dev_pos, t_index)`,
  reduced **in sorted key order**. Averaging per-batch means, or summing in
  completion order under DDP, makes the result depend on sharding.
- Fixed item order, no shuffling, no `drop_last`.
- Under DDP: shard by *question*, gather `(question, [8 losses])` with
  `_gather_by_index` (`train.py:419`) — it already does the all-gather and
  in-order reassembly — then sum on rank 0.
- The loader asserts `pool_lambda` is `False` on the loaded adapter. It is the
  default since O2, but with pooling on, batching eight noise levels of one
  question would blend them into a single λ and the metric would measure something
  else entirely — quietly, and only for the noise-aware arms.

**Scope of the guarantee:** bit-identical across repeat runs on the same host and
world size, and unchanged across world sizes up to the final float64 reduction,
which the sorted-order accumulation pins. Comparisons across different hardware
get a tolerance, not equality.

## 4. The ≥1-masked-token guarantee differs between the two paths

`train.py:373` is `if not masked_indices.any()` — a **batch-level** check. It
fires only when nothing in the entire micro-batch is masked, and then sets the
first response token of *every* row. So an individual training example can
legitimately draw an all-clean mask and contribute exactly zero to the loss.

The frozen set's guarantee is **per item**: an item with nothing masked is not a
sample of anything, it is a zero that drags its bucket's mean down while carrying
no signal about the adapter. The generator forces one position for such an item
and records `forced: true`.

**A deliberate divergence, not a match.** It only bites the lowest bucket: at
`t = 0.0625` a 30-token answer draws an all-clean mask ~14% of the time, a
100-token answer ~0.1% — so expect a handful of forced items out of 250 at `t_0`
and essentially none above it. `forced` is stored per item so the size of the
effect is auditable rather than assumed; the count is printed at generation time.

The batch-level check itself is left as it is; see `OUTSTANDING.md` §O9 for why.

## 5. How it is reached

**As a benchmark.** `BENCHMARKS["eval_loss_frozen"]`, sharing
`_load_model_with_adapter` with the accuracy benchmarks — which loads with
`is_trainable=True`, because PEFT's default `inference_mode=True` clears every
`requires_grad` and `requires_grad` is exactly the rule §3.2 counts parameters by
(`adapter_params.py:40`). Without it `adapter_params` in a result file would be
`0`.

Result dict: `benchmark`, `tuning_type`, `eval_loss_frozen`, `per_t` (8 values),
`num_items`, `adapter_params`, `frozen_eval_sha256`, `source` → written to
`benchmark_eval_loss_frozen.json` (O5; `benchmark.json` keeps its name and schema
for `gsm8k_accuracy`).

Report the mean over all 2000 items as `eval_loss_frozen`, and also the mean per
`t` bucket — 8 numbers, free to compute, and the shape of loss-vs-noise is the
first place a noise-aware adapter should show an effect at all.

Slurm: 2000 forwards, not a 6 h decode — minutes of compute behind a model load
that costs more than the metric does. Set `sbatch_time: "01:00:00"` rather than
inheriting the 8 h default.

**As an in-training curve.** A `TrainerCallback` calls the same
`frozen_eval_loss` every `eval_loss_frozen_steps` optimizer steps and logs
`eval_loss_frozen` plus the 8 buckets to W&B alongside the training curve. New
`TrainingArguments` field, **default `None` = off**, so no existing config changes
behaviour; Tier 0 configs set `100`.

Both exist because for LR selection at a 1–2 epoch budget one endpoint is thin
evidence: a curve separates "this LR is wrong" from "this LR diverged at step 400
and partly recovered", which is a distinction Tier 0 exists to make. Note this is
**not** the Trainer's evaluation loop — there is no `eval_dataset`, and the
`eval_steps: 100` in `configs/*.yaml` is a no-op for want of an
`evaluation_strategy`.

The callback runs on the training model, which is DDP-wrapped and in train mode —
hence `model.eval()` / restore around it, and the same shard-and-gather path. Cost
is ~2000 forwards per firing against ~125 steps per epoch, i.e. two firings over a
Tier 0 run.
