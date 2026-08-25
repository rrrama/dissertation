"""The §9 correctness checks. `EXPERIMENT_DESIGN.md` §10: do not proceed past these.

Three checks survive; §9.1 and §9.3 were dropped by decision (`OUTSTANDING.md` A2.1,
A2.6 -- the first cannot hold as written because the RNG streams differ between tuners,
the second was a notation difference).

    §9.2  dW = 0 at init, for every method
    §9.4  the parameter counter, against independently-written formulae
    §9.5  eval_loss_frozen determinism

What each one is actually protecting:

  §9.2 underpins "the adapter starts as the identity". Since O1 it also exercises the
  CP delta's `(out, in)` orientation and the merge path, which is where the head axis
  was wrong for q/k/v -- a check that would have caught it (`OUTSTANDING.md` §O1).

  §9.4 is not arithmetic for its own sake: the parameter count is half the headline
  claim, and `count_trainable_adapter_params` is a `requires_grad` filter, so anything
  that quietly freezes or unfreezes a tensor changes every number in the RQ1 table with
  nothing else to notice. The expected values here are written from the *paper's*
  description of each method, not read back off the tuner's shapes.

  §9.5 is the one with a subtle failure mode. `lora_dropout` is 0.1, so an eval that
  forgets `model.eval()` returns a different number every time and every Tier 0
  comparison becomes noise -- while still looking like a perfectly ordinary loss. It is
  run against a *randomised* adapter, because at init dW = 0 (that is §9.2) and dropout
  on a zero delta changes nothing: the check would pass vacuously. For the same reason
  it also asserts that train mode *does* perturb the metric, so a green §9.5 means the
  check has teeth rather than that nothing was being tested.

Usage (needs the frozen eval set, so run `frozen_eval.py --write` first):
    python correctness_checks.py                 # all three
    python correctness_checks.py --checks params # no GPU needed
"""

import argparse
import gc
import sys

import torch
import transformers

from adapter_params import (
    FOURIER_PROJECTION,
    adapter_param_breakdown,
    count_trainable_adapter_params,
)
from train import _build_peft_config, _load_tokenizer, build_args

# The four adapted matrices (`peft/src/peft/utils/constants.py`), and the arms of the
# 2x2. `lora`/`lorta` are produced by `eta=0` in the experiments themselves, but they
# are separate tuners and each has its own init path, so each is checked.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "attn_out")
TUNING_TYPES = ("lora", "lorta", "nalorta", "nara")

# NaRA mapper widths (`NARAConfig.fnn_hidden_size_1` / `_2`). Repeated here rather than
# imported for the same reason the formulae below are written out: a check that reads
# its expectation from the code it is checking cannot fail.
NARA_HIDDEN_1, NARA_HIDDEN_2 = 256, 512


def arch_dims(config):
    """(layers, heads, hidden) from either naming scheme. LLaDA uses n_layers/d_model."""

    def resolve(*names):
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return value
        raise AttributeError(f"Could not resolve architecture dims; tried {names}")

    return (
        resolve("num_hidden_layers", "n_layers"),
        resolve("num_attention_heads", "n_heads"),
        resolve("hidden_size", "d_model"),
    )


def expected_adapter_params(tuning_type, dims, rank, fourier_m):
    """Trainable parameters per §3.2, from each method's definition.

    Written from the methods rather than from the tuners: LoRA is `A (r x in)` and
    `B (out x r)` per adapted matrix per layer; the CP methods share one `A (d x r)` and
    one `B (r x head_dim)` across the whole model and index layer / head / matrix
    through `C_l`, `C_h`, `C_m`; NA-LoRTA adds `Theta (m x r)`; NaRA adds the shared
    mapper MLP. The frozen Fourier projection **k** is excluded from all of them.

    Assumes every adapted matrix is `d x d`, which is true for LLaDA-8B (no GQA, and
    `d_model == n_heads * head_dim`) and is asserted against the real layers by
    `check_param_count` rather than taken on trust.
    """
    layers, heads, hidden = dims
    head_dim = hidden // heads
    r, m = rank, fourier_m
    matrix_family = len(TARGET_MODULES) * layers * 2 * hidden * r

    if tuning_type == "lora":
        return matrix_family
    if tuning_type == "nara":
        mapper = (
            (m * NARA_HIDDEN_1 + NARA_HIDDEN_1)
            + (NARA_HIDDEN_1 * NARA_HIDDEN_2 + NARA_HIDDEN_2)
            + (NARA_HIDDEN_2 * r * r + r * r)
        )
        return matrix_family + mapper

    # A + B + C_l + C_h + C_m. C_m is one coefficient per adapted matrix.
    cp_family = hidden * r + r * head_dim + layers * r + heads * r + len(TARGET_MODULES) * r
    if tuning_type == "lorta":
        return cp_family
    if tuning_type == "nalorta":
        return cp_family + m * r
    raise ValueError(f"No expected parameter count written for '{tuning_type}'")


# --------------------------------------------------------------------------- #
# §9.2 -- dW = 0 at init                                                       #
# --------------------------------------------------------------------------- #


def check_delta_zero(peft_model, tuning_type):
    """Every adapter's contribution to the base weights is exactly zero at step 0."""
    tuner = peft_model.base_model

    if tuning_type in ("lorta", "nalorta"):
        lora_b = tuner.model.lora_B
        if lora_b.count_nonzero().item() != 0:
            return False, f"lora_B has {lora_b.count_nonzero().item()} non-zero entries at init"

        # `to_dense()` is the merge path: it contracts the CP factors into an
        # `(out_features, in_features)` matrix and applies `head_axis`. Checking it
        # rather than the factors means a re-transposed or mis-tagged head axis shows
        # up here, not in a benchmark six hours later.
        deltas = tuner._compute_weights_from_tensor()
        if not deltas:
            return False, "_compute_weights_from_tensor() returned no deltas"
        bad = []
        for name, delta in deltas.items():
            dense = delta.to_dense()
            if dense.count_nonzero().item() != 0:
                bad.append(name)
        if bad:
            return False, f"{len(bad)}/{len(deltas)} deltas non-zero, e.g. {bad[0]}"
        return True, f"{len(deltas)} deltas exactly zero; lora_B zero"

    # lora / nara: per-layer B, zero-initialised. B = 0 is what makes the whole delta
    # zero (`BA` for LoRA, `B C(lambda) A` for NaRA), so it is the property to assert.
    modules = [
        (name, module)
        for name, module in peft_model.named_modules()
        if isinstance(getattr(module, "lora_B", None), torch.nn.ModuleDict)
        and len(module.lora_B) > 0
    ]
    if not modules:
        return False, "found no adapted module with a lora_B"
    bad = [
        name
        for name, module in modules
        for sub in module.lora_B.values()
        if sub.weight.count_nonzero().item() != 0
    ]
    if bad:
        return False, f"{len(bad)}/{len(modules)} lora_B non-zero, e.g. {bad[0]}"
    return True, f"{len(modules)} lora_B tensors exactly zero"


# --------------------------------------------------------------------------- #
# §9.4 -- the parameter counter                                               #
# --------------------------------------------------------------------------- #


def check_param_count(peft_model, tuning_type, dims, rank, fourier_m):
    """`count_trainable_adapter_params` against the independently-written formula."""
    square = [
        (name, module.base_layer.in_features, module.base_layer.out_features)
        for name, module in peft_model.named_modules()
        if isinstance(getattr(module, "base_layer", None), torch.nn.Linear)
        and any(name.endswith(target) for target in TARGET_MODULES)
    ]
    if not square:
        return False, f"found no adapted module ending in one of {TARGET_MODULES}"
    hidden = dims[2]
    odd = [entry for entry in square if entry[1] != hidden or entry[2] != hidden]
    if odd:
        # The formula multiplies out `2 * d * r` per adapted matrix. On a GQA model
        # k_proj/v_proj are narrower and every count in the RQ1 table would be wrong.
        return False, f"{len(odd)} adapted matrices are not {hidden}x{hidden}, e.g. {odd[0]}"

    expected = expected_adapter_params(tuning_type, dims, rank, fourier_m)
    actual = count_trainable_adapter_params(peft_model)
    if actual != expected:
        breakdown = adapter_param_breakdown(peft_model)
        largest = sorted(breakdown.items(), key=lambda kv: -kv[1])[:4]
        return False, (
            f"expected {expected}, counted {actual} (diff {actual - expected}); "
            f"largest trainable: {largest}"
        )

    frozen_fourier = [
        name for name in adapter_param_breakdown(peft_model) if FOURIER_PROJECTION in name
    ]
    if frozen_fourier:
        return False, f"the Fourier projection is trainable: {frozen_fourier}"
    return True, f"{actual} trainable parameters, as derived"


# --------------------------------------------------------------------------- #
# §9.5 -- frozen-eval determinism                                             #
# --------------------------------------------------------------------------- #


def _perturb_lora_b(peft_model, seed=0, scale=0.02):
    """Give the adapter a non-zero delta, so the eval is not scoring the base model.

    At init `lora_B` is zero and so is the whole delta (that is §9.2), which would make
    this check compare the bare base model against itself. Only `B` is perturbed: every
    other factor keeps its real initialisation, so the result is shaped like a trained
    adapter rather than noise.

    The magnitude deliberately does not matter to either half of the check below --
    equality is exact whatever the loss is, and the dropout check inspects module state
    rather than the number -- so there is nothing here to tune.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    touched = 0
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if not param.requires_grad or "lora_B" not in name:
                continue
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
            param.copy_((noise * scale).to(param.dtype).to(param.device))
            touched += 1
    return touched


def _dropout_modules(peft_model):
    """Adapter dropout that would actually randomise the metric (`p > 0`)."""
    return [
        (name, module)
        for name, module in peft_model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.p > 0
    ]


def check_frozen_eval_determinism(peft_model, tokenizer, data_args, training_args):
    """Two evals of one checkpoint agree, and dropout is genuinely disabled while they run.

    Two assertions, because the obvious one alone is weak. Repeat-equality catches
    non-deterministic accumulation and sharding, but it would also pass if the metric
    were stochastic in a way that happened not to fire, or if the adapter contributed
    nothing at all.

    So the second assertion inspects the mechanism directly: hook every `nn.Dropout` with
    `p > 0` and record `module.training` at the moment it is called. A missing
    `model.eval()` -- the failure §9.5 exists to catch, and the one that would turn every
    Tier 0 comparison into noise while still returning a plausible loss -- shows up here
    as `training=True`, whatever the loss happens to be. Comparing eval-mode and
    train-mode losses instead would have been a proxy that goes quietly vacuous whenever
    the delta is too small to move a bf16 logit.
    """
    from frozen_eval import build_dev_dataset, frozen_eval_loss, load_frozen_set

    record = load_frozen_set()
    dataset = build_dev_dataset(tokenizer, data_args)
    kwargs = dict(dataset=dataset, record=record)

    if _perturb_lora_b(peft_model) == 0:
        return False, "found no trainable lora_B to perturb; the eval would score the base model"

    dropouts = _dropout_modules(peft_model)
    if not dropouts:
        return False, (
            "no adapter dropout with p > 0, so this check cannot detect a missing "
            "model.eval(). Re-run with --lora_dropout 0.1 (the value every config uses)."
        )

    observed = []
    handles = [
        module.register_forward_hook(
            lambda mod, args, out: observed.append(mod.training)
        )
        for _, module in dropouts
    ]
    # Left in train mode on purpose: if `frozen_eval_loss` does not switch it, the hooks
    # will say so. Starting from eval mode would let a missing switch pass unnoticed.
    peft_model.train()
    try:
        first = frozen_eval_loss(peft_model, tokenizer, data_args, training_args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()

    if not observed:
        return False, f"{len(dropouts)} dropout module(s) exist but none was called"
    if any(observed):
        active = sum(1 for flag in observed if flag)
        return False, (
            f"{active}/{len(observed)} dropout calls ran in training mode during the "
            "eval, so eval_loss_frozen is stochastic and every Tier 0 comparison "
            "reduces to noise"
        )

    second = frozen_eval_loss(peft_model, tokenizer, data_args, training_args, **kwargs)
    if first["eval_loss_frozen"] != second["eval_loss_frozen"]:
        delta = abs(first["eval_loss_frozen"] - second["eval_loss_frozen"])
        return False, (
            f"two evals of one checkpoint differ by {delta:.3e} "
            f"({first['eval_loss_frozen']!r} vs {second['eval_loss_frozen']!r})"
        )

    return True, (
        f"identical across repeats ({first['eval_loss_frozen']:.9f}); "
        f"{len(observed)} dropout calls, all with training=False"
    )


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def _load_peft_model(tuning_type, args):
    from peft import get_peft_model

    config = {
        "model_name_or_path": args.model_name_or_path,
        "tuning_type": tuning_type,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "fourier_m": args.fourier_m,
        "lora_dropout": args.lora_dropout,
        "output_dir": ".",
        "report_to": "none",
        "model_max_length": args.model_max_length,
    }
    if tuning_type in ("nalorta", "nara"):
        config["eta"] = args.eta
    model_args, data_args, training_args = build_args(config)

    # Same seeding discipline as `run_training`: adapter init is part of what is being
    # checked, so it must not depend on whatever ran before.
    transformers.set_seed(training_args.seed)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        trust_remote_code=True,
    )
    dims = arch_dims(base.config)
    return get_peft_model(base, _build_peft_config(model_args)), dims, model_args, data_args, training_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks",
        default="delta,params,determinism",
        help="Comma-separated subset. 'delta,params' needs no GPU.",
    )
    parser.add_argument("--model_name_or_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--fourier_m", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument(
        "--determinism_tuning_type",
        default="nalorta",
        help="Which adapter §9.5 runs against. It tests the eval harness, not the "
        "adapter, so one arm is enough -- but it must be a noise-aware one, since "
        "those are the arms whose conditioning the frozen set has to reproduce.",
    )
    args = parser.parse_args()
    requested = [name.strip() for name in args.checks.split(",") if name.strip()]

    results = []
    per_arm = {"delta", "params"} & set(requested)
    for tuning_type in TUNING_TYPES if per_arm else ():
        print(f"\n=== {tuning_type} ===", flush=True)
        model, dims, _, _, _ = _load_peft_model(tuning_type, args)
        if "delta" in requested:
            results.append((f"§9.2 dW=0 at init [{tuning_type}]",) + check_delta_zero(model, tuning_type))
        if "params" in requested:
            results.append(
                (f"§9.4 parameter count [{tuning_type}]",)
                + check_param_count(model, tuning_type, dims, args.rank, args.fourier_m)
            )
        # Four 8B models will not sit in RAM together.
        del model
        gc.collect()

    if "determinism" in requested:
        print(f"\n=== {args.determinism_tuning_type} (determinism) ===", flush=True)
        model, _, model_args, data_args, training_args = _load_peft_model(
            args.determinism_tuning_type, args
        )
        model = model.to(training_args.device)
        tokenizer = _load_tokenizer(model_args, training_args)
        results.append(
            ("§9.5 frozen-eval determinism",)
            + check_frozen_eval_determinism(model, tokenizer, data_args, training_args)
        )

    if not results:
        print(
            f"--checks {args.checks!r} selected nothing. Valid names: "
            "delta, params, determinism."
        )
        return 1

    print("\n" + "=" * 78)
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * 78)

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(
            f"\n{len(failed)} check(s) FAILED: {', '.join(failed)}\n"
            "EXPERIMENT_DESIGN.md §10: do not proceed past step 1 -- raise it rather "
            "than working around it. A failure here means every number produced "
            "downstream is measuring something other than what it claims to."
        )
        return 1
    print(f"\nAll {len(results)} check(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
