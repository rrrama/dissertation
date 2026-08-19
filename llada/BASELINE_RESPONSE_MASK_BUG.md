# Baseline benchmark crash: `forward() got an unexpected keyword argument 'response_mask'`

Fixed on 2026-08-17. Written from reading the code and the failing job's `error.err`;
the fix is compile-checked and the orchestration dry run passes, but **it has not yet
been confirmed against a running job** — the cluster was not reachable from the machine
where this was written.

## Symptom

`batch_train.py --mode benchmark --config configs/baseline.yaml` (the untuned-baseline
run, `tuning_type: "none"`) died on every rank as soon as the first question started
decoding:

```
File "llada/generate.py", line 217, in generate
  logits = model(x, **model_kwargs).logits
TypeError: forward() got an unexpected keyword argument 'response_mask'
```

The model loaded, the dataset downloaded, the progress bar reached 0/440 — the failure
is in the sampler's first forward, ~1 min into a job that is otherwise ~20 h long.

## Cause

The noise-aware tuners (NA-LoRTA, NaRA) condition their adapter weights on the masked
proportion of the *answer*. That denominator is not recoverable from `input_ids` alone,
so the caller marks the response region with a `response_mask` kwarg on the forward.
`PeftModel` lists it in `special_peft_forward_args` (`peft/src/peft/peft_model.py:134`):
the kwarg reaches the tuner's forward hook and is stripped again before the base model
is called.

`generate()` relied on that stripping and passed `response_mask` unconditionally, on the
reasoning that it is therefore harmless for every tuning type. True — for a `PeftModel`.
The `tuning_type: "none"` baseline added in the "untuned baseline" work has no adapter by
construction (`train.py:_gsm8k_accuracy_benchmark`), so the sampler is handed the bare
`AutoModelForCausalLM`. LLaDA's `forward()` has no `response_mask` parameter and nothing
strips it, so it raised.

Every other run in the sweep passes through `PeftModel.from_pretrained`, which is why
`lorta` / `nalorta` / `nara` never hit this and the baseline hit it on its first ever run.

## Fix

`generate.py` now asks the model whether it strips the kwarg instead of assuming it does:

```python
model_kwargs = {}
if "response_mask" in getattr(model, "special_peft_forward_args", ()):
    ...
    model_kwargs["response_mask"] = response_mask
```

A `PeftModel` sets that attribute in `__init__`, so every adapted model still gets the
mask; the bare base model gets a clean `model(x)` call. The CFG branch is unaffected — it
concatenates whatever is in `model_kwargs`, which is simply empty for the baseline.

The training path (`train.py:compute_loss`) still passes `response_mask` unconditionally,
which is correct: `_build_peft_config` rejects `tuning_type: "none"` for training, so that
call always has a `PeftModel`.

### Why there is also a check in `train.py`

Making the kwarg conditional trades a loud failure for a quiet one. If an *adapted* model
ever stopped advertising `special_peft_forward_args` — hidden behind a DDP or `torch.compile`
wrapper, say — the sampler would silently omit the mask, and NA-LoRTA / NaRA do not raise
in that case: they warn once and fall back to a prompt+answer lambda denominator
(`nalorta/model.py:169`, `nara/model.py:199`). That would benchmark a differently
conditioned adapter and report the number as if nothing happened. `_gsm8k_accuracy_benchmark`
now raises on the adapter path if the wrapped model does not declare the kwarg.

## Related caveats on the baseline number (not bugs)

Neither of these breaks the run, but both depress the baseline's GSM8K accuracy, so a low
number is not by itself evidence about the base model's ability:

- **No chat template.** Nothing in `llada/` calls `apply_chat_template`; prompts are the
  raw `question + QUESTION_PROMPT`. That matches the SFT arms' training format, so the
  comparison is internally consistent, but it runs LLaDA-8B-**Instruct** outside the format
  it was instruction-tuned in.
- **No learned stopping.** The untuned model fills all 256 response positions, and
  `extract_answer_number` (`test.py:41`) falls back to "the last number in the string" when
  `"The final answer is: "` is absent — so trailing junk after a correct answer is scored as
  the answer.
