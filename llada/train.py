# Adapted from ../gsm8k/train.py for LLaDA-8B (GSAI-ML/LLaDA-8B-Base), a masked
# *diffusion* language model. The autoregressive next-token CE loss and the
# `.generate()` eval of the GSM8K harness do NOT apply here; both are replaced:
#   - training uses the LLaDA masked-diffusion SFT objective (see LladaSFTTrainer)
#   - evaluation uses the iterative diffusion sampler in generate.py, and runs
#     only in `--mode benchmark` (never inside the training loop)
# See ../LLADA_CONVERSION_PLAN.md (items B4-B7).

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, Optional, Sequence

import copy
import torch
import torch.nn.functional as F
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import Trainer
from tqdm import tqdm

from datasets import load_dataset
from test import (
    ANSWER_PROMPT,
    build_prompt,
    compute_accuracy,
    extract_answer_number,
    template_add_special_tokens,
)
from generate import generate, MASK_ID
from adapter_params import summarise_adapter_params
from splits import train_split
import wandb

IGNORE_INDEX = -100

# `tuning_type` value for the untuned baseline: no adapter at all, so the
# benchmark scores the bare base model. It is a benchmark-only pseudo-adapter --
# there is nothing to train, and no adapter is written to (or read from) the run
# directory. Putting it in a sweep (`tuning_type: ["none", "nara", ...]`) gives
# every other run a reference point under identical decoding settings.
NO_ADAPTER = "none"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the adapter. Used in evaluation or resuming from the checkpoint."
        },
    )
    rank: int = field(
        default=128,
        metadata={"help": "Rank of the LoRTA / LoRA adapters."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Adapter alpha."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access private models."},
    )
    tuning_type: str = field(
        default="lorta",
        metadata={
            "help": "Adapter to train: 'lora', 'lorta', 'nalorta' or 'nara'. "
            "'none' is the untuned-baseline pseudo-adapter: benchmark-only, it "
            "scores the bare base model (see NO_ADAPTER)."
        },
    )
    # --- Adapter hyperparameters shared across tuning types ---
    #
    # All three default to None, meaning "keep this adapter's own default"
    # (`ADAPTER_DEFAULTS`). That way naming a field in a YAML changes exactly the
    # runs that name it, and every config written before these existed still
    # resolves to the values that were hardcoded in `_build_peft_config`.
    eta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Strength of the noise-conditioning term, under one name for both "
            "noise-aware adapters: NA-LoRTA's `eta` in `I_r + eta * Theta^T phi(lambda)` "
            "and NaRA's `c_scale` in `I_r + c_scale * F_phi(e(lambda))`. `eta: 0` "
            "collapses nalorta -> lorta and nara -> lora exactly, which is how the "
            "non-noise-aware cells of the 2x2 are produced (EXPERIMENT_DESIGN.md §2). "
            "Ignored, with a warning, by 'lora' and 'lorta'."
        },
    )
    fourier_m: Optional[int] = field(
        default=None,
        metadata={
            "help": "Width of the Fourier embedding of lambda, under one name for both "
            "noise-aware adapters: NA-LoRTA's `embedding_length` and NaRA's "
            "`embedding_dim`. Must be even. Ignored, with a warning, by 'lora' and 'lorta'."
        },
    )
    lora_dropout: Optional[float] = field(
        default=None,
        metadata={
            "help": "Dropout on the adapter input. Per-adapter defaults differ (see "
            "ADAPTER_DEFAULTS); set this to hold it constant across a comparison."
        },
    )


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    save_total_limit: Optional[int] = field(
        default=2,
        metadata={
            "help": "Checkpoints to keep. Was unbounded, which was harmless when "
            "nothing read checkpoints back; now that training resumes from them "
            "(and several runs write to one volume concurrently) a 6-epoch run "
            "would keep ~7 optimizer states for no reason. Two, so that a "
            "checkpoint torn by a mid-write kill still leaves one to resume from."
        },
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    wandb_project: str = field(
        default="llada_gsm8k_training",
        metadata={"help": "Weights & Biases project name"},
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report results to wandb"},
    )
    # --- LLaDA diffusion hyperparameters ---
    mask_id: int = field(
        default=MASK_ID,
        metadata={"help": "MASK token id for LLaDA (126336)."},
    )
    diffusion_eps: float = field(
        default=1e-3,
        metadata={"help": "Lower bound on the masking probability during SFT."},
    )
    head_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": "Learning rate for the noise head (NA-LoRTA's Theta, NaRA's mapper) "
            "as distinct from the backbone factors. None means 'use `learning_rate`', "
            "which reproduces the single-group optimiser exactly, so lora/lorta are "
            "unaffected. EXPERIMENT_DESIGN.md §5.2: a single global LR forces a "
            "compromise between components with very different curvature and makes "
            "*both* noise-aware methods look worse than they are."
        },
    )
    eval_loss_frozen_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Compute `eval_loss_frozen` (frozen_eval.py) every N optimizer "
            "steps and log it beside the training curve. None disables it, so no "
            "existing config changes behaviour. Tier 0 should set it: selecting a "
            "learning rate from one end-of-run number cannot distinguish 'this LR is "
            "wrong' from 'this LR diverged at step 400 and partly recovered', and the "
            "metric is ~2000 forwards. Note this is NOT `eval_steps`, which drives the "
            "Trainer's own evaluation loop and is a no-op here (there is no "
            "eval_dataset). See FROZEN_EVAL_SPEC.md §7."
        },
    )
    # --- Diffusion sampler (eval) hyperparameters ---
    gen_length: int = field(
        default=256, metadata={"help": "Number of response tokens to sample."}
    )
    diffusion_steps: int = field(
        default=256, metadata={"help": "Number of diffusion sampling steps."}
    )
    block_length: int = field(
        default=32,
        metadata={
            "help": "Semi-autoregressive block length. Must divide gen_length, and "
            "the resulting block count must divide diffusion_steps. Keep it well "
            "below gen_length: a single full-length block lets EOS in the far tail "
            "win the low-confidence topk and truncate the answer (see "
            "configs/baseline.yaml)."
        },
    )
    remasking: str = field(
        default="low_confidence", metadata={"help": "'low_confidence' or 'random'."}
    )
    gen_temperature: float = field(
        default=0.0, metadata={"help": "Gumbel sampling temperature (0 = greedy)."}
    )
    benchmark_batch_size: int = field(
        default=8,
        metadata={
            "help": "Questions decoded simultaneously per rank in benchmark mode. "
            "A single 320-token forward badly underuses an L40; memory is the "
            "limit (the (B, S, vocab) logits dominate). 1 reproduces the old "
            "one-question-at-a-time path exactly."
        },
    )


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    add_special_tokens: bool = True,
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    # Each string is tokenized on its own, so `padding="longest"` pads it to its
    # own length -- i.e. never pads. The length therefore *is* the sequence
    # length. Deriving it from `ne(pad_token_id)` instead would undercount any
    # string that legitimately contains the pad token, and the chat template puts
    # `<|eot_id|>` inside the prompt (`test.build_prompt`) -- exactly such a token
    # on a tokenizer that pads with it. A short `source_len` leaves prompt tokens
    # unmasked in `labels` below, i.e. silently trains on the question.
    input_ids_lens = [ids.shape[0] for ids in input_ids]
    return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    add_special_tokens: bool = True,
) -> Dict:
    """Preprocess the data by tokenizing.

    `labels` marks the prompt tokens with IGNORE_INDEX; the remaining (response)
    tokens are the ones the diffusion loss masks and predicts.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, add_special_tokens)
        for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    source_ids = sources_tokenized["input_ids"]

    # Marking the prompt by *length* assumes tokenizing `source + target`
    # reproduces tokenizing `source` as a literal prefix. BPE can merge across a
    # concatenation boundary and break that, which shifts the mask by a token or
    # two -- training on the tail of the question, or dropping the head of the
    # answer out of the loss. Either way it is invisible in the loss curve, so
    # check it rather than trust it. Truncated examples are skipped: there the
    # prompt itself may have been cut off, and `label[:source_len]` already masks
    # the whole (shorter) row.
    for ids, source in zip(input_ids, source_ids):
        if ids.shape[0] >= source.shape[0] and not torch.equal(
            ids[: source.shape[0]], source
        ):
            raise ValueError(
                "Tokenizing prompt+answer did not reproduce the prompt's own "
                "tokenization as a prefix, so the IGNORE_INDEX label mask would "
                "not line up with the prompt. This usually means the prompt and "
                "the answer merge across their boundary; see test.build_prompt."
            )

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        # Same prompt format the benchmark decodes with (`test.build_prompt`), so
        # the adapters are scored inside the format they were trained in.
        sources = [build_prompt(tokenizer, example["question"]) for example in raw_data]
        targets = [
            f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT)
            for example in raw_data
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(
            sources, targets, tokenizer, template_add_special_tokens(tokenizer)
        )

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        # Recorded before padding: a position is real iff it is inside the
        # sequence's own length. Deriving the mask from token *identity*
        # (`input_ids.ne(pad_token_id)`) would be wrong here -- every target ends
        # with a genuine `eos_token` (see SupervisedDataset), so if the tokenizer
        # pads with EOS, that real, predicted token would be masked out as padding.
        lengths = torch.tensor([ids.shape[0] for ids in input_ids])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # long rather than bool, matching the dtype the sampler feeds LLaDA in
        # `generate.py` (the path that has been checked against modeling_llada.py)
        attention_mask = (
            torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
        ).long()
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    train_set = load_dataset(data_args.data_name, "main", split="train")
    # Hold out the 250-question dev split (EXPERIMENT_DESIGN.md 3.3). Training on it
    # would make `gsm8k_accuracy_dev` and `eval_loss_frozen` scores of memorisation
    # rather than of generalisation, and those are what every selection decision in
    # the project is made on.
    train_set = train_split(train_set)
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


# Parameter-name substrings identifying the *noise head* -- the part of an adapter that
# maps lambda to a modulation -- as opposed to the backbone factors it modulates. §5.2
# gives the two their own learning rates because their curvature differs sharply: the
# head sits multiplicatively on top of every backbone factor.
#
#   lora_Theta       NA-LoRTA's Theta, in `c_mask = 1 + eta * (phi(lambda) @ Theta)`
#   lora_mapper      NaRA's MLP, in `C(lambda) = I_r + c_scale * F_phi(e(lambda))`
#   lora_constant_c  NaRA's learned constant term alongside it
#   lora_phi         the lambda embedding. Frozen for `embedding_type: "fourier"` (both
#                    tuners), so it never reaches the optimiser today -- named here
#                    because if it ever becomes trainable (NaRA's "mlp" embedding) it is
#                    part of the noise function, not of the backbone.
#
# Everything else trainable is backbone: A, B, C_l, C_h, C_m for the CP tuners, and the
# per-layer lora_A / lora_B for the matrix ones.
HEAD_PARAM_NAMES = ("lora_Theta", "lora_mapper", "lora_constant_c", "lora_phi")


def _is_head_parameter(name: str) -> bool:
    return any(marker in name for marker in HEAD_PARAM_NAMES)


def diffusion_masked_loss(
    model,
    input_ids,
    response_mask,
    attention_mask,
    t,
    masked_indices,
    mask_id,
    diffusion_eps,
    pass_response_mask=True,
):
    """The LLaDA diffusion objective, given an already-chosen noise level and mask.

    Everything `compute_loss` does *after* sampling, factored out so that
    `frozen_eval.py` can score a checkpoint on a fixed `(t, mask)` set by calling the
    same code the run was trained with. A second implementation would drift the first
    time either was touched, and the Tier 0 metric would stop measuring the thing being
    trained without any number looking wrong (`FROZEN_EVAL_SPEC.md` §1).

    Returns `(per_example_loss, logits)`, where `per_example_loss` is `(b,)` rather than
    the batch mean: the eval's mean must be over the *eval set*, not an average of
    per-batch means, which differ whenever batches are unequal in size. `compute_loss`
    finishes the job with `.sum() / b`. The pair rather than the bare `(b,)` of the spec
    is for `Trainer.compute_loss`'s `return_outputs` contract, which needs the logits.

    `t` is the noise level, not `p_mask`: `p_mask` is derived here from the caller's
    `diffusion_eps`, so a change to `diffusion_eps` reaches the eval instead of leaving
    it pinned to whatever value was in force when the frozen set was generated.

    `masked_indices` must already satisfy the >=1-masked-token guarantee; the two call
    sites establish it differently and deliberately (`FROZEN_EVAL_SPEC.md` §5a).

    `pass_response_mask` is False for exactly one caller -- the bare base model
    (`tuning_type: none`), whose forward has no such kwarg. It is a parameter rather
    than something sniffed from `special_peft_forward_args` because sniffing is how an
    adapted model that mislays that attribute silently falls back to a prompt+answer
    lambda denominator and returns a plausible number for the wrong conditioning.
    """
    b, l = input_ids.shape
    p_mask = ((1 - diffusion_eps) * t + diffusion_eps)[:, None].expand(-1, l)  # (b, l)

    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    # Padding is excluded from attention. LLaDA attends *bidirectionally*, so
    # without this every real token of a short example attends to the padding
    # that its longer batch-mate forced onto the batch -- i.e. an example's
    # hidden states depend on who it was collated with. It also keeps the
    # padding out of the noise-aware adapters' mask-proportion statistic,
    # which takes the attention_mask into account when it is given one
    # (`peft/src/peft/tuners/nalorta/model.py:_mask_token_proportion`);
    # otherwise pad positions inflate that proportion's denominator without
    # ever contributing to its numerator.
    #
    # `response_mask` marks the answer region. The noise-aware adapters
    # (NA-LoRTA, NaRA) condition on the masked *proportion of the answer*,
    # and that denominator is not recoverable from `input_ids`: the response
    # holds unmasked tokens too. PEFT lists `response_mask` in
    # `special_peft_forward_args`, so it reaches the tuner and is stripped
    # before LLaDA is called; the other tuning types simply ignore it.
    extra = {"response_mask": response_mask} if pass_response_mask else {}
    logits = model(input_ids=noisy_batch, attention_mask=attention_mask, **extra).logits

    # 1/t reweighting and per-example answer-length normalisation
    answer_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1).expand(-1, l)

    token_loss = (
        F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        / p_mask[masked_indices]
        / answer_lengths[masked_indices]
    )

    # Scatter into a dense (b, l) and sum along the sequence, rather than
    # `index_add_` over row indices: the latter uses atomics on CUDA and is
    # therefore non-deterministic, which is exactly what `eval_loss_frozen` cannot
    # afford (FROZEN_EVAL_SPEC.md §5). Masked assignment plus a reduction is
    # deterministic and the (b, l) buffer is negligible next to the logits.
    per_token = torch.zeros(b, l, dtype=token_loss.dtype, device=token_loss.device)
    per_token[masked_indices] = token_loss
    return per_token.sum(dim=1), logits


class LladaSFTTrainer(Trainer):
    """Trainer implementing the LLaDA masked-diffusion SFT objective.

    Recipe (from the GSAI-ML/LLaDA SFT guidelines):
      1. Sample a per-sequence masking probability ``t ~ U(eps, 1)``.
      2. Replace *response* tokens with the MASK id with probability ``t``
         (prompt and padding tokens are never masked).
      3. Forward the (partially masked) sequence.
      4. Cross-entropy on the masked positions only, each token reweighted by
         ``1/t`` (the Monte-Carlo estimator of the diffusion ELBO) and by the
         per-example answer length, then averaged over the batch.
    """

    def __init__(self, *args, mask_id=MASK_ID, diffusion_eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_id = mask_id
        self.diffusion_eps = diffusion_eps

    def create_optimizer(self):
        """Give the noise head its own learning rate (EXPERIMENT_DESIGN.md §5.2).

        Built by splitting the groups `Trainer.create_optimizer` already made, rather
        than by rebuilding them: the weight-decay partition, the optimiser class and its
        kwargs are all version-dependent internals, and inheriting them means this
        cannot drift from whatever the installed transformers does. Splitting a group
        copies its dict and replaces `params`/`lr`, so every other default carries over.

        With `head_learning_rate` unset the base implementation is returned untouched --
        the single-LR path is preserved exactly, which matters because lora and lorta
        have no head at all and must not be perturbed by this existing.
        """
        optimizer = super().create_optimizer()

        head_lr = self.args.head_learning_rate
        if head_lr is None or head_lr == self.args.learning_rate:
            self._head_group_indices = ()
            return optimizer

        head_ids = {
            id(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad and _is_head_parameter(name)
        }
        if not head_ids:
            raise ValueError(
                f"head_learning_rate={head_lr} is set, but no trainable parameter of "
                f"tuning type in this run matches {HEAD_PARAM_NAMES}. For 'lora' and "
                "'lorta' that is expected -- they have no noise head -- and so is a "
                "config error: drop head_learning_rate. At eta=0 the head is multiplied "
                "by zero, so its LR cannot affect anything either."
            )

        groups, head_indices = [], []
        for group in optimizer.param_groups:
            head = [p for p in group["params"] if id(p) in head_ids]
            backbone = [p for p in group["params"] if id(p) not in head_ids]
            if backbone:
                groups.append({**group, "params": backbone})
            if head:
                head_indices.append(len(groups))
                groups.append({**group, "params": head, "lr": head_lr})
        optimizer.param_groups = groups
        self._head_group_indices = tuple(head_indices)

        num_head = sum(len(groups[i]["params"]) for i in head_indices)
        print(
            f"Optimiser: {len(groups)} param groups; {num_head} head tensor(s) at "
            f"lr={head_lr}, the rest at lr={self.args.learning_rate}"
        )
        return optimizer

    def log(self, logs, *args, **kwargs):
        """Log the head LR beside the backbone one.

        The Trainer reports `learning_rate` from the *first* param group, so with two
        groups on different schedules half the optimiser state would go unrecorded --
        and §8 requires the head LR in the run's log.
        """
        for index in getattr(self, "_head_group_indices", ()):
            logs["head_learning_rate"] = self.optimizer.param_groups[index]["lr"]
            break
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        b, l = input_ids.shape

        # response tokens = everything that is not prompt and not padding
        response_mask = labels != IGNORE_INDEX

        # per-sequence masking probability t ~ U(eps, 1)
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - self.diffusion_eps) * t + self.diffusion_eps  # (b,)
        p_mask = p_mask[:, None].expand(-1, l)  # (b, l)

        masked_indices = (
            torch.rand((b, l), device=input_ids.device) < p_mask
        ) & response_mask
        # Guarantee at least one masked token so the loss is well defined. Note this
        # is a *batch-level* check: it fires only when nothing anywhere in the
        # micro-batch is masked, so an individual example may legitimately draw an
        # all-clean mask and contribute exactly zero. The frozen eval set forces a
        # token per *item* instead, because a zero-loss item is not a sample of
        # anything -- a deliberate divergence, written up in FROZEN_EVAL_SPEC.md §5a.
        if not masked_indices.any():
            first_response = response_mask.float().argmax(dim=1)
            masked_indices[torch.arange(b, device=input_ids.device), first_response] = (
                response_mask.any(dim=1)
            )

        per_example_loss, logits = diffusion_masked_loss(
            model,
            input_ids=input_ids,
            response_mask=response_mask,
            attention_mask=inputs["attention_mask"],
            t=t,
            masked_indices=masked_indices,
            mask_id=self.mask_id,
            diffusion_eps=self.diffusion_eps,
        )
        loss = per_example_loss.sum() / b

        return (loss, logits) if return_outputs else loss


class FrozenEvalCallback(transformers.TrainerCallback):
    """Log `eval_loss_frozen` *during* training, not only at the end of it.

    Tier 0 selects learning rates, and one number at the end of a 1-2 epoch run is thin
    evidence for that: it cannot separate "this LR is wrong" from "this LR diverged and
    partly recovered", which is exactly the distinction the sweep exists to make. The
    metric is ~2000 forwards, so a curve is affordable (FROZEN_EVAL_SPEC.md §7).

    The authoritative number still comes from the `eval_loss_frozen` benchmark against
    the saved adapter -- this is the curve, not the record. Nothing is caught here that
    is not also caught there.

    Everything that can be wrong about the *setup* is checked in `__init__`, before
    training starts: a hash mismatch or a pooled lambda discovered 400 steps in would
    take the run down with it.
    """

    def __init__(self, model, tokenizer, data_args, training_args, every_n_steps):
        from frozen_eval import (
            assert_matches,
            build_dev_dataset,
            load_frozen_set,
            _assert_per_example_lambda,
        )

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.every_n_steps = int(every_n_steps)
        self.record = load_frozen_set()
        # Built once. Re-tokenising 250 questions on every firing would be wasted work,
        # and re-deriving the dev split would be a second place for it to drift.
        self.dataset = build_dev_dataset(tokenizer, data_args)
        assert_matches(self.record, self.dataset, tokenizer, training_args.model_max_length)
        _assert_per_example_lambda(model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every_n_steps <= 0 or state.global_step % self.every_n_steps != 0:
            return
        from frozen_eval import frozen_eval_loss

        # `model` is the Trainer's unwrapped model, not the DDP wrapper: this is a
        # forward-only pass under no_grad, so there is nothing for DDP to synchronise
        # and the wrapper's unused-parameter bookkeeping is pure cost. Every rank runs
        # it -- the gather inside is collective.
        result = frozen_eval_loss(
            model,
            self.tokenizer,
            self.data_args,
            self.training_args,
            dataset=self.dataset,
            record=self.record,
        )
        if state.is_world_process_zero:
            print(
                f"step {state.global_step} | eval_loss_frozen: "
                f"{result['eval_loss_frozen']:.6f}"
            )
            if wandb.run is not None:
                wandb.log(
                    {
                        "eval_loss_frozen": result["eval_loss_frozen"],
                        **{
                            f"eval_loss_frozen/t_{t:.4f}": loss
                            for t, loss in zip(result["t_values"], result["per_t"])
                        },
                    },
                    step=state.global_step,
                )


def _gather_by_index(local_indices, local_values, total, world_size, fill):
    """Re-assemble a full, in-order list from every rank's interleaved shard.

    `fill` is what an index nobody produced keeps -- `inf` for a prediction (never
    equal to a gold answer, so it scores as wrong), `""` for a completion.
    """
    values = [fill] * total
    if world_size <= 1 or not torch.distributed.is_initialized():
        for i, value in zip(local_indices, local_values):
            values[i] = value
        return values

    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, (local_indices, local_values))
    for indices, shard in gathered:
        for i, value in zip(indices, shard):
            values[i] = value
    return values


def _pad_token_id(tokenizer):
    """A token id to left-pad prompts with. Must not be the MASK id."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None or pad_id == MASK_ID:
        raise ValueError(
            f"tokenizer has no usable pad token for batched decoding (got {pad_id})"
        )
    return pad_id


def _stop_token_ids(tokenizer):
    """Token ids that end a response: EOS, plus the chat template's turn marker.

    The SFT targets end with `tokenizer.eos_token` (see `SupervisedDataset`), but
    under the chat template the model's *natural* terminator is `<|eot_id|>` --
    that is what closes an assistant turn, and it is what the untuned baseline
    emits (followed by `<|endoftext|>` filler). Both have to count as a stop, and
    `<|eot_id|>` especially: now that the model has been shown the multi-turn
    format, the filler after it can be a hallucinated *next turn* rather than
    padding, and any numbers in that turn would otherwise reach
    `extract_answer_number`'s "last number in the string" fallback.
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    # `convert_tokens_to_ids` returns the unk id (or None) for a token this
    # tokenizer does not have, so neither is evidence of a real `<|eot_id|>`.
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        ids.add(eot)
    return sorted(ids)


def _truncate_at_stop(gen_tokens, stop_ids):
    """Cut each response row at its first stop token -- the stand-in for stopping.

    Diffusion decoding has no early exit: `generate()` fills all `gen_length`
    positions for every question, so a model that finished its answer after 40
    tokens still emits the remaining 216. The model does say where it meant to
    stop -- decoding with `skip_special_tokens=True` merely *deletes* that marker
    instead of respecting it, which glues the trailing filler onto the answer and
    hands `extract_answer_number`'s "last number in the string" fallback
    (`test.py`) whatever number the filler happened to end on.

    Rows end at different places, so the result is a list of 1-D tensors rather than
    a rectangular batch. A row with no stop token is returned whole: cutting it at
    an arbitrary point would be a guess.
    """
    if not stop_ids:
        return list(gen_tokens)

    stops = torch.tensor(stop_ids, dtype=gen_tokens.dtype)
    rows = []
    for row in gen_tokens:
        hit = torch.isin(row, stops).nonzero().flatten()
        rows.append(row[: hit[0]] if hit.numel() else row)
    return rows


def _stack_prompts(prompt_ids, pad_id, device):
    """Stack 1-D prompt id tensors into (B, len) ids + an attention mask.

    Equal-length prompts (what `_uniform_length_batches` produces) need no
    padding and get `attention_mask=None`, which is not just tidier: LLaDA's
    `modeling_llada.py` tests `0.0 in attention_mask` on every forward, and that
    membership test is a device sync -- 256 of them per question.

    Ragged input is still supported, and is padded on the *left*: the sampler
    shares one block boundary across the batch, so every row's response region
    has to start at the same offset.
    """
    lengths = {ids.shape[0] for ids in prompt_ids}
    if len(lengths) == 1:
        return torch.stack(prompt_ids).to(device), None

    max_len = max(lengths)
    batch = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long)
    for row, ids in enumerate(prompt_ids):
        batch[row, max_len - ids.shape[0] :] = ids
        attention_mask[row, max_len - ids.shape[0] :] = 1
    return batch.to(device), attention_mask.to(device)


def _uniform_length_batches(local_indices, prompt_ids, batch_size):
    """Group a rank's questions into batches whose prompts are all the same length.

    Equal lengths mean zero padding, and padding is pure waste: a padded position
    still costs a full forward at each of the 256 sampling steps. It also lets
    `_stack_prompts` return `attention_mask=None`, dodging a per-forward device
    sync in LLaDA's modeling code.

    This also used to be what kept decoding output-preserving under NA-LoRTA,
    whose adapter weights are modulated by a mask proportion pooled to a single
    scalar over the batch: with a prompt+answer denominator that proportion
    varied across rows through the prompt length, so only equal lengths made the
    pooled scalar equal each row's own value. The denominator is now the answer
    only (`peft/src/peft/tuners/nalorta/model.py:_mask_token_proportion`), which
    at generation time is a constant `gen_length` for every row, and every row
    shares the transfer schedule and so the same masked *count* -- so the pooled
    scalar is already exact for any batch. Equal-length batching is now a
    performance choice rather than a correctness one.

    The price is that batches are capped by how many questions happen to share a
    length, so they are variable-sized and often smaller than `batch_size`.
    """
    by_length = {}
    for i in local_indices:
        by_length.setdefault(prompt_ids[i].shape[0], []).append(i)

    batches = []
    for length in sorted(by_length):
        group = by_length[length]
        batches += [group[k : k + batch_size] for k in range(0, len(group), batch_size)]
    return batches


@torch.no_grad()
def diffusion_evaluate(model, tokenizer, test_set, training_args):
    """Evaluate GSM8K accuracy using the LLaDA diffusion sampler.

    Two levels of parallelism, both over the test set (the 256 sampling steps
    themselves are inherently sequential):

      - across ranks: under torchrun each rank samples an interleaved shard of the
        questions on its own GPU, and the predictions are gathered back into
        test-set order before scoring;
      - within a rank: up to `benchmark_batch_size` questions are decoded in one
        batch, which is where most of the speedup is -- a batch-1 320-token
        forward leaves an L40 mostly idle.

    Both are exact: every question is decoded exactly as it would be alone on one
    GPU. Batches are restricted to questions of identical prompt length, which
    eliminates padding -- see `_uniform_length_batches`.

    Returns `(accuracy, predictions, gold_answers, completions)`, all four in
    test-set order. Scoring reads each response only up to its first stop token
    (`_truncate_at_stop`); `completions` holds the untruncated decode instead, so a
    later scoring change can be evaluated against stored output.
    """
    model.eval()

    # Identical to the prompts the SFT arms were trained on (`SupervisedDataset`).
    questions = [build_prompt(tokenizer, example["question"]) for example in test_set]
    answers = []
    for example in test_set["answer"]:
        ans = example.split("####")[-1].replace(",", "")
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answers.append(ans)

    world_size, rank = training_args.world_size, training_args.process_index
    # Interleaved (strided) shard rather than contiguous chunks: question cost
    # varies with prompt length, and striding keeps the ranks balanced so none
    # of them straggles at the end.
    local_indices = list(range(rank, len(questions), world_size))

    add_special = template_add_special_tokens(tokenizer)
    prompt_ids = {
        i: tokenizer(
            questions[i], return_tensors="pt", add_special_tokens=add_special
        )["input_ids"][0]
        for i in local_indices
    }
    pad_id = _pad_token_id(tokenizer)
    stop_ids = _stop_token_ids(tokenizer)
    batch_size = max(1, int(training_args.benchmark_batch_size))
    batches = _uniform_length_batches(local_indices, prompt_ids, batch_size)
    if rank == 0:
        mean_batch = len(local_indices) / max(1, len(batches))
        print(
            f"[rank {rank}] {len(local_indices)} questions -> {len(batches)} batches "
            f"(cap {batch_size}, mean {mean_batch:.1f}); batches hold equal-length "
            f"prompts only, so decoding is exact"
        )

    preds_by_index = {}
    completions_by_index = {}
    progress = tqdm(
        total=len(local_indices),
        desc=f"Evaluating (diffusion) [rank {rank}]",
        disable=rank != 0,
    )
    for batch_indices in batches:
        input_ids, attention_mask = _stack_prompts(
            [prompt_ids[i] for i in batch_indices], pad_id, model.device
        )
        out = generate(
            model,
            input_ids,
            attention_mask=attention_mask,
            steps=training_args.diffusion_steps,
            gen_length=training_args.gen_length,
            block_length=training_args.block_length,
            temperature=training_args.gen_temperature,
            remasking=training_args.remasking,
            mask_id=training_args.mask_id,
        )
        # left padding means the response region starts at the same offset in
        # every row of the batch. One host transfer for the whole batch:
        # `_truncate_at_stop` reads positions row by row, which on the GPU would be
        # a device sync per row.
        gen_tokens = out[:, input_ids.shape[1] :].cpu()
        # Kept verbatim for the record: all `gen_length` tokens, special tokens
        # left in. That is what `_write_benchmark_result` stores, so a future
        # scoring rule (text-level stop sequences, a different extractor) can be
        # tried offline against these instead of re-decoding the test set. Cutting
        # a stored completion at its first stop marker reproduces what was scored
        # below, up to any other special token inside the answer.
        raw_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)
        scored_batch = [
            tokenizer.decode(row, skip_special_tokens=True)
            for row in _truncate_at_stop(gen_tokens, stop_ids)
        ]
        for i, raw, decoded in zip(batch_indices, raw_batch, scored_batch):
            if rank == 0:
                print(decoded)
            preds_by_index[i] = extract_answer_number(decoded)
            completions_by_index[i] = raw
        progress.update(len(batch_indices))
    progress.close()

    local_preds = [preds_by_index[i] for i in local_indices]
    local_completions = [completions_by_index[i] for i in local_indices]
    ans_pred_list = _gather_by_index(
        local_indices, local_preds, len(questions), world_size, fill=float("inf")
    )
    completions = _gather_by_index(
        local_indices, local_completions, len(questions), world_size, fill=""
    )
    accuracy = compute_accuracy(answers, ans_pred_list)
    return accuracy, ans_pred_list, answers, completions


# --------------------------------------------------------------------------- #
# Config plumbing                                                              #
# --------------------------------------------------------------------------- #
#
# `batch_train.py` freezes a fully-resolved, single-run config to `config.yaml`
# in each run directory. That config is a flat dict that mixes fields for the
# three dataclasses below with batch/slurm-only keys (`nproc_per_node`, `gres`,
# `benchmark`, ...). We keep only the recognised dataclass fields, coerce their
# types (YAML can leave e.g. "5e-2" as a string), and build the dataclasses.


def _dataclass_field_names():
    names = set()
    for dc in (ModelArguments, DataArguments, TrainingArguments):
        for f in dataclass_fields(dc):
            names.add(f.name)
    return names


def _maybe_number(value):
    """Coerce YAML strings that denote numbers (e.g. "5e-2") to int/float."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def build_args(config: Dict):
    """Build (ModelArguments, DataArguments, TrainingArguments) from a dict."""
    valid = _dataclass_field_names()
    filtered = {k: _maybe_number(v) for k, v in config.items() if k in valid}
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    return parser.parse_dict(filtered)


# Per-adapter defaults for the shared hyperparameters on `ModelArguments`. These
# are exactly the values `_build_peft_config` used to hardcode, so a config that
# does not mention a field gets the behaviour it had before the field existed.
#
# `eta` and `fourier_m` are deliberately *not* given a single cross-method value:
# the two noise-aware adapters were tuned at different strengths (NA-LoRTA at
# 1.0, NaRA at its reference 0.1) and unifying them here would silently change
# what a plain `tuning_type: nara` run means. Set them explicitly in the YAML to
# hold them equal -- which is what the eta sweep and the eta=0 collapse do.
ADAPTER_DEFAULTS = {
    "lora": {"lora_dropout": 0.1},
    "lorta": {"lora_dropout": 0.1},
    "nalorta": {"lora_dropout": 0.1, "eta": 1.0, "fourier_m": 32},
    "nara": {"lora_dropout": 0.05, "eta": 0.1, "fourier_m": 64},
}


def _resolve_adapter_settings(model_args):
    """Resolve the shared adapter hyperparameters for this run's `tuning_type`.

    Returns a dict of the settings that apply to this adapter. Fields left unset
    in the config fall back to `ADAPTER_DEFAULTS`; fields that the selected
    adapter has no use for (`eta`/`fourier_m` on lora/lorta) are dropped with a
    warning rather than raising, so a sweep may list them across a mix of
    adapters without every non-noise-aware cell erroring out.

    The resolved values are written back onto `model_args` so that W&B records
    what the run actually used rather than the `None` the config carried.
    """
    if model_args.tuning_type not in ADAPTER_DEFAULTS:
        # 'none' (the benchmark-only baseline) and typos both land here. Neither
        # has an adapter to configure; let `_build_peft_config` raise the error
        # that actually says what to do about it.
        return {}

    defaults = ADAPTER_DEFAULTS[model_args.tuning_type]
    resolved = {}
    for name, default in defaults.items():
        value = getattr(model_args, name, None)
        resolved[name] = default if value is None else value

    for name in ("eta", "fourier_m"):
        if name not in defaults and getattr(model_args, name, None) is not None:
            logging.warning(
                f"`{name}` is set but tuning_type='{model_args.tuning_type}' has no "
                f"noise-conditioning term to apply it to; ignoring it. Note that "
                f"'{model_args.tuning_type}' is already the eta=0 case by construction."
            )

    for name, value in resolved.items():
        setattr(model_args, name, value)
    return resolved


def _build_peft_config(model_args):
    """Return the adapter config for the selected `tuning_type`."""
    # LLaDA target modules: q_proj / k_proj / v_proj / attn_out (no o_proj).
    llada_target_modules = ["q_proj", "k_proj", "v_proj", "attn_out"]
    if model_args.tuning_type == NO_ADAPTER:
        raise ValueError(
            f"tuning_type '{NO_ADAPTER}' is the untuned baseline and has nothing to "
            "train: it exists only for `--mode benchmark`, which scores the base "
            "model directly. Drop it from this config, or run the sweep with "
            "`batch_train.py --mode benchmark` / `--mode all` (both skip training "
            "for baseline runs)."
        )
    settings = _resolve_adapter_settings(model_args)

    if model_args.tuning_type == "lora":
        from peft import LoraConfig

        return LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=settings["lora_dropout"],
            target_modules=llada_target_modules,
            init_lora_weights=True,
        )
    elif model_args.tuning_type == "lorta":
        from peft import LorTaConfig

        return LorTaConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=llada_target_modules,
            lora_dropout=settings["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
        )
    elif model_args.tuning_type == "nalorta":
        from peft import NALorTaConfig

        return NALorTaConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=llada_target_modules,
            lora_dropout=settings["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            eta=settings["eta"],
            embedding_length=settings["fourier_m"],
        )
    elif model_args.tuning_type == "nara":
        from peft import NARAConfig

        # Reference hyperparameters from NaRA's own LLaDA config
        # (NaRA/config/nara/llada_instruct_nara_math14k.yaml), except for the scaling
        # convention: NaRA scales its delta by `scale_ab` alone and leaves `lora_alpha`
        # unused, while this uses PEFT's `lora_alpha / r` so the run is comparable to
        # lora/lorta/nalorta under the same sweep. That makes the reference `lr: 1e-4`
        # meaningless here -- sweep the learning rate rather than trusting either number.
        return NARAConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=llada_target_modules,
            lora_dropout=settings["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            embedding_dim=settings["fourier_m"],
            embedding_type="fourier",
            fnn_hidden_size_1=256,
            fnn_hidden_size_2=512,
            c_scale=settings["eta"],
            input_mode="nl",
        )
    raise ValueError(f"Unknown tuning_type: {model_args.tuning_type}")


def _load_tokenizer(model_args, training_args):
    # LLaDA has its own vocabulary and a dedicated MASK token; do NOT add or
    # resize special tokens (unlike the Llama GSM8K path).
    return transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
    )


def run_training(config: Dict, output_dir: str):
    """Train a single run whose hyperparameters come from `config`.

    `output_dir` is assigned by the caller (batch_train.py); this function does
    NOT construct its own nested output path.
    """
    from peft import get_peft_model

    os.makedirs(output_dir, exist_ok=True)
    config = dict(config)
    config.setdefault("output_dir", output_dir)
    model_args, data_args, training_args = build_args(config)
    training_args.output_dir = output_dir
    is_main_process = training_args.process_index == 0

    # Seed *here*, not implicitly via the Trainer. `Trainer.__init__` calls
    # `set_seed(args.seed)`, but it is constructed further down -- after
    # `get_peft_model` has already drawn `lora_A`, the `C_*` factors, `Theta` and
    # the frozen Fourier `k`. Left to the Trainer, `seed` therefore controls data
    # order only and the adapter's initialisation is not reproducible at all,
    # which quietly makes a multi-seed comparison measure something other than
    # seed variance. Seeding before the model is built covers both.
    transformers.set_seed(training_args.seed)

    # Resolve before `wandb.init` rather than leaving it to `_build_peft_config`
    # below: the run's config snapshot is taken from `vars(model_args)`, and an
    # unset field is still `None` at that point, so W&B would record `eta: None`
    # for every run instead of the value the adapter was actually built with.
    # Idempotent -- the resolved values are written back onto `model_args`.
    _resolve_adapter_settings(model_args)

    if is_main_process:
        wandb.init(
            project=training_args.wandb_project,
            config={**vars(model_args), **vars(training_args), **vars(data_args)},
        )

    # LLaDA ships custom modeling code (class LLaDAModelLM) -> trust_remote_code.
    # No device_map: under torchrun each process owns one GPU and the Trainer/
    # Accelerate wraps the model in DDP, replicating it per-GPU instead of
    # splitting layers across GPUs (which serializes compute across devices).
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        trust_remote_code=True,
    )

    # LLaDA's custom modeling code (LLaDAModelLM) never wires up HF's generic
    # PreTrainedModel.gradient_checkpointing_enable() hook, so leaving
    # training_args.gradient_checkpointing=True makes the Trainer raise
    # "LLaDAModelLM does not support gradient checkpointing." Instead, drive
    # LLaDAModel's own native activation checkpointing (equivalent whole-layer
    # torch.utils.checkpoint wrapping, see modeling_llada.py) directly, and
    # stop the Trainer from taking its unsupported generic path.
    if training_args.gradient_checkpointing:
        model.model.set_activation_checkpointing("whole_layer")
        training_args.gradient_checkpointing = False

    peft_config = _build_peft_config(model_args)
    model = get_peft_model(model, peft_config)

    # The parameter count is a *reported* number, not a diagnostic: the claim is joint
    # on accuracy and parameters, so it is captured next to the adapter it describes
    # rather than recomputed later from a config (which is how a rank or a target-module
    # list quietly stops matching the table it is quoted in). See adapter_params.py.
    param_summary = summarise_adapter_params(model)
    for name, count in param_summary["breakdown"].items():
        print(f"{name}: {count} parameters")
    print(f"Trainable adapter parameters: {param_summary['adapter_params']}")
    if is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "adapter_params.json"), "w") as f:
            json.dump(param_summary, f, indent=2)
        # Summary rather than a logged metric: it is one number per run, not a series.
        wandb.run.summary["adapter_params"] = param_summary["adapter_params"]

    tokenizer = _load_tokenizer(model_args, training_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # `output_dir` is assigned by batch_train.py; no nested path is constructed
    # here (it was previously expt_name/model/ep_N/lr_X/seed_N).

    trainer = LladaSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        mask_id=training_args.mask_id,
        diffusion_eps=training_args.diffusion_eps,
        **data_module,
    )

    # Off unless a config asks for it, so nothing already written changes behaviour.
    if training_args.eval_loss_frozen_steps:
        trainer.add_callback(
            FrozenEvalCallback(
                model=model,
                tokenizer=tokenizer,
                data_args=data_args,
                training_args=training_args,
                every_n_steps=training_args.eval_loss_frozen_steps,
            )
        )

    # Resume if this run was interrupted. `save_strategy: steps` has always been
    # writing `output_dir/checkpoint-N`; nothing ever read them back, so a killed
    # run restarted from step 0. On a rented instance that can be reclaimed
    # mid-run, that was the largest single risk in the migration off slurm
    # (SLURM_MIGRATION.md §5).
    #
    # `get_last_checkpoint` returns None on a clean directory, so a fresh run is
    # unaffected. A PEFT checkpoint carries optimizer, scheduler and RNG state
    # alongside the adapter weights, so this restores the run rather than
    # approximating it.
    #
    # Only reached when the run directory still has checkpoints: `--overwrite`
    # deletes them precisely so that it retrains instead of resuming.
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        print(f"Resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    # Nothing in the loss guards against divergence (the masking probability is
    # floored at `diffusion_eps` and the answer-length denominator is clamped, so
    # the objective itself cannot produce NaN -- only the weights can), and a
    # diverged adapter is indistinguishable from a good one until something looks
    # at the numbers. Read the file back on the main process: a fraction of a second
    # here, against ~20 h of benchmark decoding on an adapter that is all NaN. Only
    # rank 0 wrote it and so only rank 0 checks; torchrun tears down the other ranks
    # when this one exits non-zero.
    #
    # No evaluation here either: diffusion sampling is expensive and single-GPU, so
    # it is a separate job (`--mode benchmark`) run against the saved adapter.
    if is_main_process:
        from peft import load_peft_weights

        _assert_adapter_finite(
            load_peft_weights(training_args.output_dir, device="cpu"),
            training_args.output_dir,
        )
        print(f"Training finished; adapter saved to {training_args.output_dir}")
        wandb.finish()


# --------------------------------------------------------------------------- #
# Benchmark mode                                                              #
# --------------------------------------------------------------------------- #
#
# `benchmark` mode loads the base model + a trained adapter from a run
# directory and runs a registered benchmark against it. The registry seam
# (`BENCHMARKS`) lets additional benchmarks be plugged in later; for now the
# only entry reuses `diffusion_evaluate` (GSM8K accuracy).


def _benchmark_result_filename(bench_name: str) -> str:
    """One file per benchmark, so dev and test accuracy stop overwriting each other.

    `gsm8k_accuracy` keeps `benchmark.json`, with exactly the schema it already has:
    per O5 no existing file or field changes meaning, and `write_summary` keeps emitting
    the entries it emits today. Only the benchmarks added since get a suffixed name.
    """
    return "benchmark.json" if bench_name == "gsm8k_accuracy" else f"benchmark_{bench_name}.json"


def _write_benchmark_result(output_dir: str, result: Dict):
    path = os.path.join(output_dir, _benchmark_result_filename(result["benchmark"]))
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote benchmark result -> {path}")


def _examples(keys, n=5):
    """A few key names for an error message, rather than all 262 of them."""
    shown = ", ".join(sorted(keys)[:n])
    return f"{shown}, ... ({len(keys)} total)" if len(keys) > n else shown


def _assert_adapter_finite(state_dict, adapter_dir):
    """Reject an adapter whose weights diverged during training.

    A NaN adapter loads perfectly happily and then decodes 1319 GSM8K questions into
    garbage over ~20 h, so it is worth a fraction of a second to refuse it here. It is
    also checked *before* `_assert_adapter_loaded`, because NaN != NaN: a correctly
    loaded NaN tensor fails that function's `allclose` against the very file it came
    from, and gets reported as a loading bug that isn't there.
    """
    corrupt = [key for key, value in state_dict.items() if not torch.isfinite(value).all()]
    if corrupt:
        raise RuntimeError(
            f"{len(corrupt)}/{len(state_dict)} adapter tensor(s) in {adapter_dir} are "
            f"non-finite ({_examples(corrupt)}). The training run diverged -- this is a "
            "training problem, not a loading one; lower the learning rate and retrain."
        )


def _assert_adapter_loaded(model, adapter_dir):
    """Fail loudly if the saved adapter tensors did not land in the model.

    PEFT loads adapter weights with `load_state_dict(..., strict=False)` and no
    caller inspects the result, so a key-naming mismatch drops every tensor without
    raising. What is left is the freshly *initialised* adapter, whose `lora_B` is
    zero -- an exactly-zero delta, i.e. the bare base model. That silently
    benchmarks the wrong thing, and every adapter then scores identically, so
    check that the weights on disk really are the weights in the model.
    """
    from peft import load_peft_weights
    from peft.utils import get_peft_model_state_dict

    saved = load_peft_weights(adapter_dir, device="cpu")
    # Non-finite weights would be reported below as "present but not equal", i.e. as
    # this function's failure mode rather than their own; rule them out first.
    _assert_adapter_finite(saved, adapter_dir)
    # `get_peft_model_state_dict` is what wrote the file, so it names tensors the
    # same way on the way back out -- for LoRA (per-adapter ModuleDicts) as well as
    # for LoRTA (flat parameters). Comparing through it keeps this check agnostic
    # to each tuner's naming convention. save_embedding_layers=False: no embedding
    # layer is targeted here, and "auto" would probe the HF hub.
    current = get_peft_model_state_dict(model, save_embedding_layers=False)

    missing, mismatched = [], []
    for key, value in saved.items():
        loaded = current.get(key)
        if loaded is None:
            missing.append(key)
        elif not torch.allclose(loaded.detach().cpu().float(), value.cpu().float()):
            mismatched.append(key)

    if missing or mismatched:
        raise RuntimeError(
            f"Adapter in {adapter_dir} was not loaded into the model: "
            f"{len(missing)} key(s) absent from the model ({_examples(missing)}), "
            f"{len(mismatched)} key(s) present but not equal to the saved tensor "
            f"({_examples(mismatched)}). Benchmarking would score the base model instead."
        )


def _load_model_with_adapter(model_args, training_args, adapter_dir):
    """Base model + the adapter under test, on this rank's GPU, in eval mode.

    Shared by every benchmark. The `response_mask` check below matters more for
    `eval_loss_frozen` than for accuracy: a silent fallback to a prompt+answer lambda
    denominator would mis-condition the adapter and still return an entirely plausible
    loss, whereas a wrong accuracy at least tends to look wrong.

    Returns `(model, is_baseline)`.
    """
    from peft import PeftModel

    is_baseline = model_args.tuning_type == NO_ADAPTER

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        trust_remote_code=True,
    )
    if is_baseline:
        # No adapter to load, and nothing for `_assert_adapter_loaded` to check:
        # the base model *is* the thing being benchmarked here, not an accident.
        print(f"tuning_type='{NO_ADAPTER}': benchmarking the base model, no adapter")
    else:
        # `is_trainable=True` trains nothing -- every benchmark runs under `no_grad`
        # with the model in eval mode. It is here because PEFT's default
        # `inference_mode=True` clears `requires_grad` on the adapter, and
        # `requires_grad` is the rule EXPERIMENT_DESIGN.md §3.2 counts parameters by
        # (`adapter_params.py`). Without it `adapter_params` in every result file would
        # be 0, which is a wrong number rather than a missing one.
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        # Checked on the CPU model, before the .to(device) below, so it is cheap.
        _assert_adapter_loaded(model, adapter_dir)
        # `generate()` only passes `response_mask` to a model that advertises it will
        # strip it again (the bare base model's forward would raise a TypeError). If
        # that ever stops being true for an adapted model -- a wrapper hiding the
        # attribute, say -- the noise-aware tuners would not raise: they warn once and
        # silently fall back to a prompt+answer lambda denominator, i.e. quietly
        # benchmark a differently-conditioned adapter. Fail here instead.
        if "response_mask" not in getattr(model, "special_peft_forward_args", ()):
            raise RuntimeError(
                f"{type(model).__name__} does not declare `response_mask` in "
                "`special_peft_forward_args`, so the sampler cannot pass it (see "
                "generate.py). NA-LoRTA / NaRA would fall back to a prompt+answer "
                "lambda denominator and score something other than what was trained."
            )
    # `training_args.device` is this rank's own GPU (cuda:LOCAL_RANK under
    # torchrun); "cuda" would pile every rank onto cuda:0. No DDP wrapper: this
    # is pure inference, the ranks only talk to each other at the final gather.
    model = model.to(training_args.device)
    model.eval()
    return model, is_baseline


def _benchmark_adapter_params(model, is_baseline) -> int:
    """`adapter_params` for a result file. Zero for the baseline, which has no adapter."""
    return 0 if is_baseline else summarise_adapter_params(model)["adapter_params"]


def _gsm8k_accuracy_benchmark(model_args, data_args, training_args, adapter_dir):
    """Load base model + trained adapter and score GSM8K test accuracy."""
    model, is_baseline = _load_model_with_adapter(model_args, training_args, adapter_dir)

    tokenizer = _load_tokenizer(model_args, training_args)
    logging.warning("Downloading Data")
    test_set = load_dataset(data_args.data_name, "main", split="test")

    accuracy, ans_pred_list, answers, completions = diffusion_evaluate(
        model, tokenizer, test_set, training_args
    )
    label = "base model (no adapter)" if is_baseline else f"adapter: {adapter_dir}"
    print(f"{label} | GSM8K test accuracy: {100*accuracy:.2f}%")
    return {
        "benchmark": "gsm8k_accuracy",
        "tuning_type": model_args.tuning_type,
        "accuracy": accuracy,
        # An addition, not a change (O5): the claim is joint on accuracy and
        # parameters, so the count travels with the accuracy it is quoted beside
        # rather than being recomputed later from a config.
        "adapter_params": _benchmark_adapter_params(model, is_baseline),
        "num_examples": len(answers),
        "predictions": ans_pred_list,
        "ground_truth": answers,
        # Raw model output, one entry per test question, in test-set order:
        # everything the sampler produced, special tokens included. `accuracy`
        # above scores only the part before the first EOS, so this is the record
        # needed to re-derive a different scoring rule (stop sequences, a changed
        # `extract_answer_number`) without re-running the ~20 h decode.
        "completions": completions,
        "source": "benchmark",
    }


def _eval_loss_frozen_benchmark(model_args, data_args, training_args, adapter_dir):
    """Tier 0's metric: the training objective on a committed set of (t, mask) pairs.

    2000 forwards, so the model load costs more than the metric does -- Tier 0 configs
    should set `sbatch_time: "01:00:00"` rather than inherit the 8 h default (O8).
    """
    from frozen_eval import frozen_eval_loss

    model, is_baseline = _load_model_with_adapter(model_args, training_args, adapter_dir)
    tokenizer = _load_tokenizer(model_args, training_args)

    result = frozen_eval_loss(
        model,
        tokenizer,
        data_args,
        training_args,
        # The bare base model's forward has no `response_mask` kwarg. This is the one
        # call site that turns it off, and it is off because there is no adapter to
        # condition -- not because the attribute happened to be missing.
        pass_response_mask=not is_baseline,
    )
    label = "base model (no adapter)" if is_baseline else f"adapter: {adapter_dir}"
    per_t = ", ".join(f"{t:.4f}:{loss:.4f}" for t, loss in zip(result["t_values"], result["per_t"]))
    print(f"{label} | eval_loss_frozen: {result['eval_loss_frozen']:.6f}")
    print(f"  per t bucket: {per_t}")
    return {
        "benchmark": "eval_loss_frozen",
        "tuning_type": model_args.tuning_type,
        "adapter_params": _benchmark_adapter_params(model, is_baseline),
        "source": "benchmark",
        **result,
    }


# Registry seam: name -> callable(model_args, data_args, training_args, adapter_dir).
BENCHMARKS = {
    "gsm8k_accuracy": _gsm8k_accuracy_benchmark,
    "eval_loss_frozen": _eval_loss_frozen_benchmark,
}


def run_benchmark(config: Dict, output_dir: str):
    """Run the configured benchmark against the trained adapter in `output_dir`."""
    config = dict(config)
    config.setdefault("output_dir", output_dir)
    bench_name = config.get("benchmark", "gsm8k_accuracy")
    if bench_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{bench_name}'. Registered: {sorted(BENCHMARKS)}"
        )

    # The baseline run has no adapter by construction, so the existence check
    # below would reject exactly the run it is meant to protect.
    if config.get("tuning_type") != NO_ADAPTER:
        adapter_config = os.path.join(output_dir, "adapter_config.json")
        if not os.path.exists(adapter_config):
            raise FileNotFoundError(
                f"No trained adapter found in {output_dir} (missing adapter_config.json). "
                "Train this run before benchmarking."
            )

    model_args, data_args, training_args = build_args(config)
    training_args.output_dir = output_dir
    result = BENCHMARKS[bench_name](model_args, data_args, training_args, output_dir)
    # Every rank holds the same gathered result; only one of them writes it.
    if training_args.process_index == 0:
        _write_benchmark_result(output_dir, result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train or benchmark a single LoRTA run from a frozen config.yaml."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a frozen per-run config.yaml."
    )
    parser.add_argument("--mode", choices=["train", "benchmark"], default="train")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Run output directory. Defaults to the directory holding --config.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.config))

    if args.mode == "train":
        run_training(config, output_dir)
    else:
        run_benchmark(config, output_dir)


if __name__ == "__main__":
    main()
