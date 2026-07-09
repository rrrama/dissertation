# Plan: Convert LoRTA to the LLaDA-8B model

Goal: make LoRTA (Low Rank Tensor Adaptation) work on **LLaDA-8B** (`GSAI-ML/LLaDA-8B-Base`),
a masked **diffusion** language model, instead of the current autoregressive Llama-2-7B target.

This document is written for an implementing agent that has NOT seen the prior investigation.
Read the "Background" section first; it explains why the changes are needed.

---

## Background

### How LoRTA works today
LoRTA factorizes ALL attention adapters into a small set of shared tensor factors instead of
per-module A/B matrices like LoRA. The factors are (see `peft/src/peft/tuners/lorta/model.py`):

- `model.lora_A`  — shape `(hidden_size, r)`
- `model.lora_B`  — shape `(r, head_dim)`
- `model.lora_C_l` — per-layer factor, shape `(num_hidden_layers, r)`
- `model.lora_C_h` — per-head factor, shape `(num_attention_heads, r)`
- `model.lora_C_m` — per-matrix factor, shape `(4, r)`  ← the "4" is q, k, v, o

`_compute_weights_from_tensor()` (`model.py:139`) reconstructs a per-module delta weight `dW` for
every (block, matrix) pair by, for each head, computing
`lora_A @ diag(lora_C_h[head] * lora_C_m[matrix] * lora_C_l[block]) @ lora_B`
and concatenating across heads. The dict is keyed by a module path built in
`_map_layer_to_adapter()` (`model.py:136`): `"{prefix}.{block_idx}.{qkvo_submodule}"`.

At forward time, `LorTaModel.forward` (`model.py:191`) recomputes the weights, and a pre-forward
hook (`weights_pre_forward_hook`, `model.py:322`) injects the right `dW` into each patched Linear
as the `adapter_weight` kwarg. The patched Linear (`lorta/layer.py:337`) then does
`result = base_layer(x) + dropout(x) @ dW.T * scaling`.

### What is hardwired to Llama (must change)
1. `model.py:297-303` reads `self.model.config.num_hidden_layers`, `num_attention_heads`,
   `hidden_size` directly. The tensor-build loops (`model.py:144-188`) read the same fields.
2. Model-type mappings in `peft/src/peft/utils/constants.py:153-171` are keyed on `"llama"`/
   `"roberta"` and define the block-container prefix + q/k/v/o submodule names.
3. The GSM8K harness `gsm8k/train.py` loads `AutoModelForCausalLM`, trains with HF `Trainer`
   (next-token cross-entropy over shifted labels), and evaluates with autoregressive `.generate()`.

### LLaDA-8B facts (verified from the HF repo config + modeling_llada.py)
- `model_type: "llada"`, `block_type: "llama"` → uses the **LLaDALlamaBlock**, which has
  **separate** `q_proj` / `k_proj` / `v_proj` Linear layers. The output projection is named
  **`attn_out`** (NOT `o_proj`). These projections sit directly on the block object (there is
  **no** `self_attn.` nesting like Llama has).
- Blocks are under **`model.transformer.blocks`** (when `block_group_size == 1`), NOT `model.layers`.
- `n_layers = 32`, `n_heads = 32`, `n_kv_heads = 32` → full multi-head attention, **no GQA**.
  This means LoRTA's "concat over `num_attention_heads`" is valid for k/v as well as q/o.
- `d_model = 4096`, head_dim = 4096/32 = 128. All four projections are 4096×4096.
- Config field names are `n_layers` / `n_heads` / `d_model` — it does **NOT** expose
  `num_hidden_layers` / `num_attention_heads` / `hidden_size`. So item #1 above will hard-crash.
- LLaDA is a **masked diffusion LM**: bidirectional attention (no causal mask), trained with a
  masked-token prediction (diffusion) loss, and generated with an **iterative diffusion sampler**
  — NOT `.generate()`. The MASK token id is **126336**. `eos`/`pad` token id is 126081.
- Must be loaded with `trust_remote_code=True` (custom modeling code; class `LLaDAModelLM`).

---

## Work items

### A. LoRTA / PEFT changes (small, mechanical — do these first, they unblock everything)

**A1. Add `"llada"` entries to the mappings** in `peft/src/peft/utils/constants.py` (~lines 153-171):
```python
TRANSFORMERS_MODELS_TO_LORTA_TARGET_MODULES_MAPPING["llada"] = ["q_proj", "k_proj", "v_proj", "attn_out"]
TRANSFORMERS_MODELS_TO_LORTA_PREFIX_MAPPING["llada"] = "model.transformer.blocks"
TRANSFORMERS_MODELS_TO_LORTA_QKVO_MAPPING["llada"] = {
    "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "attn_out",
}
```

**A2. Make config-field access architecture-agnostic** in `peft/src/peft/tuners/lorta/model.py`.
LLaDA config lacks `num_hidden_layers` / `num_attention_heads` / `hidden_size`. Add a small helper
(e.g. `_get_arch_dims(config)`) that resolves each across naming schemes:
- num layers:  `num_hidden_layers` → fallback `n_layers`
- num heads:   `num_attention_heads` → fallback `n_heads`
- hidden size: `hidden_size` → fallback `d_model`

Then replace the direct `self.model.config.<field>` reads at `model.py:297-303` AND inside the
tensor-build loops at `model.py:144-188` with the resolved values. Double-check `head_dim`
(`model.py:297`) uses the resolved hidden/heads (128 for LLaDA).

**A3. Verify hook wiring.** LLaDA's projections are plain `nn.Linear`, so the LoRTA `Linear`
wrapper and the `adapter_weight`-kwarg path (`lorta/layer.py:337`) should apply unchanged. Confirm
the pre-forward-hook lookup keys produced by `_map_layer_to_adapter` match the real module names,
i.e. `model.transformer.blocks.{i}.q_proj`, `...k_proj`, `...v_proj`, `...attn_out`. Fix the
prefix/qkvo mapping if the actual named_modules differ (print `key_list` in `inject_adapter`,
`model.py:231`).

### B. Training / eval harness — the real work (LLaDA is a diffusion model)

Recommendation: create a **new experiment dir `llada/`** (copy from `gsm8k/`) rather than editing
`gsm8k/train.py` in place, so the autoregressive Llama GSM8K path stays intact and the diffusion
loss doesn't get entangled with the CE loss. Decide with the user before starting if unsure.

**B4. Model + tokenizer loading.** Use
`AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Base", trust_remote_code=True,
torch_dtype=torch.bfloat16, ...)` and tokenizer with `trust_remote_code=True`. Remove the Llama
special-token resize block (`smart_tokenizer_and_embedding_resize`) — LLaDA has its own vocab and a
dedicated MASK token; do not add/resize tokens.

**B5. Replace the loss with the LLaDA masked-diffusion SFT objective.** The current next-token
CE-over-shifted-labels path does NOT apply. Implement (custom `Trainer.compute_loss` or a custom
loop): sample a masking ratio `t ~ U(0,1)` per sequence, replace **response** tokens with the MASK
id (126336) with prob `t` (keep prompt tokens unmasked), forward the model, compute cross-entropy
only on the masked positions, and reweight by `1/t`. Port the exact recipe from the reference
GSAI-ML/LLaDA repo (their SFT / pre-training loss). Prompt vs. response boundary comes from the same
label masking the current dataset builder already computes.

**B6. Replace eval/generation.** Swap the `.generate()` block (`gsm8k/train.py:337-360`) for LLaDA's
iterative diffusion sampler (parameters: generation length, number of sampling steps, block length,
remasking strategy — e.g. low-confidence remasking). Port from the reference LLaDA repo's `generate`
/ `get_log_likelihood` utilities. Keep the existing GSM8K answer-extraction + accuracy code
(`extract_answer_number`, `compute_accuracy` from `test.py`) for scoring.

**B7. Update the run script** (`gsm8k/run_training.sh` → new `llada/run_training.sh`):
`--model_name_or_path GSAI-ML/LLaDA-8B-Base`, target modules `q_proj k_proj v_proj attn_out`,
plus diffusion hyperparams (MASK id, sampling steps, gen length, block length). Note the current
script passes `target_modules=['q_proj','k_proj','v_proj','o_proj']` inside `train.py` — update to
`attn_out`.

### C. Validation

**C1. Smoke test the adapter build.** Run `get_peft_model(llada_model, LorTaConfig(...))`; print
trainable params. Confirm `_compute_weights_from_tensor` materializes a `dW` for all 32 blocks × 4
matrices and every `dW` is 4096×4096. Confirm hook keys match real module names.

**C2. End-to-end micro-run.** One training step with the diffusion loss (verify it runs and the loss
is finite), then a short sample to confirm eval produces text and answer extraction works.

---

## Suggested sequencing
1. A1 → A2 → A3 (unblocks loading + adapter injection; A2 is the one guaranteed hard-crash).
2. C1 (prove the adapter attaches to LLaDA before touching the loss).
3. B4 → B5 → B6 → B7 (the diffusion harness; B5/B6 are the bulk of the effort and are independent
   of LoRTA — lift from the GSAI-ML/LLaDA reference repo).
4. C2 (end-to-end).

## Key references
- LoRTA tensor build: `peft/src/peft/tuners/lorta/model.py:139` (`_compute_weights_from_tensor`),
  `:297-318` (parameter creation), `:322` (hook), `:136` (`_map_layer_to_adapter`).
- LoRTA patched Linear forward: `peft/src/peft/tuners/lorta/layer.py:337`.
- Model-type mappings: `peft/src/peft/utils/constants.py:153-171`.
- Current harness to adapt/copy: `gsm8k/train.py`, `gsm8k/test.py`, `gsm8k/run_training.sh`.
- LLaDA model: https://huggingface.co/GSAI-ML/LLaDA-8B-Base (config.json, modeling_llada.py) and
  the GSAI-ML/LLaDA GitHub repo for the SFT loss + diffusion sampler.
