"""Trainable-parameter accounting for the adapters under comparison.

The headline claim of this project is joint on accuracy *and* parameter count, so the
count is a reported number, not a diagnostic -- it belongs on the critical path with
the accuracy it is quoted beside. `EXPERIMENT_DESIGN.md` §3.2 fixes what counts:

    Trainable parameters = every tensor with `requires_grad=True` after
    `get_peft_model`, excluding the frozen Gaussian Fourier projection **k**.

That exclusion needs no special case here. Both noise-aware tuners create `lora_phi`'s
weight with `requires_grad=False` and re-freeze it in `_mark_only_adapters_as_trainable`
(`peft/src/peft/tuners/nalorta/model.py:570`), so a plain `requires_grad` filter already
implements the rule. `assert_fourier_projection_frozen` exists to keep that true: if a
refactor ever unfreezes **k**, every parameter count in the writeup silently gains ~2 k
parameters, and nothing else in the pipeline would notice.

No torch import beyond what the model already brings in, and no CUDA: this runs on a
login node against a CPU model just as happily as inside a training job.
"""

from typing import Dict

# Substring identifying the frozen Fourier projection in a parameter name. Both
# tuners name it `lora_phi` (NaRA reaches it through its embedding module, whose
# parameters are also frozen); §3.2 excludes it because it is drawn once and never
# trained, so counting it would inflate every noise-aware method equally and
# meaninglessly.
FOURIER_PROJECTION = "lora_phi"


def count_trainable_adapter_params(model) -> int:
    """Total trainable parameters in `model`, per §3.2.

    Counts `numel()` over `requires_grad` parameters. For every tuning type in scope
    the base model is fully frozen by `_mark_only_adapters_as_trainable`, so this is
    the adapter's parameter count rather than the model's -- but it is deliberately
    *not* filtered by name: a count that silently ignored an unfrozen base tensor
    would be the one number in the writeup that hides a broken run.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_param_breakdown(model) -> Dict[str, int]:
    """Per-tensor trainable counts, `{parameter_name: numel}`.

    The per-factor detail behind the aggregate -- `A`, `B`, `C_l`, `C_h`, `C_m`, `Theta`
    for the CP tuners -- which is what the RQ1 parameter table is built from, and what
    makes a surprising total diagnosable without re-running anything.
    """
    return {
        name: param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def assert_fourier_projection_frozen(model) -> None:
    """Fail if the frozen Fourier projection **k** ever becomes trainable.

    §3.2 excludes **k** from the count on the grounds that it is fixed. That is
    currently enforced in the tuners, not here, so this asserts the premise rather
    than the conclusion: if it stops holding, the counts change silently and every
    parameter number already written down becomes wrong by ~2 k.
    """
    trainable = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and FOURIER_PROJECTION in name
    ]
    if trainable:
        raise RuntimeError(
            f"The Gaussian Fourier projection is trainable ({', '.join(trainable)}), but "
            "EXPERIMENT_DESIGN.md §3.2 excludes it from the trainable-parameter count on "
            "the grounds that it is frozen. Either re-freeze it in the tuner's "
            "_mark_only_adapters_as_trainable, or change §3.2 and every parameter count "
            "quoted from it."
        )


def summarise_adapter_params(model) -> Dict:
    """The record written to `adapter_params.json` and logged alongside every result."""
    assert_fourier_projection_frozen(model)
    breakdown = adapter_param_breakdown(model)
    return {
        "adapter_params": sum(breakdown.values()),
        "breakdown": breakdown,
        "num_trainable_tensors": len(breakdown),
    }
