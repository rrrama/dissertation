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
from test import extract_answer_number, compute_accuracy, ANSWER_PROMPT, QUESTION_PROMPT
from generate import generate, MASK_ID
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
        default="GSAI-ML/LLaDA-8B-Base",
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


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
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
    # --- Diffusion sampler (eval) hyperparameters ---
    gen_length: int = field(
        default=256, metadata={"help": "Number of response tokens to sample."}
    )
    diffusion_steps: int = field(
        default=256, metadata={"help": "Number of diffusion sampling steps."}
    )
    block_length: int = field(
        default=256, metadata={"help": "Semi-autoregressive block length."}
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
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing.

    `labels` marks the prompt tokens with IGNORE_INDEX; the remaining (response)
    tokens are the ones the diffusion loss masks and predicts.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [
            f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT)
            for example in raw_data
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

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
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


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
        # guarantee at least one masked token so the loss is well defined
        if not masked_indices.any():
            first_response = response_mask.float().argmax(dim=1)
            masked_indices[torch.arange(b, device=input_ids.device), first_response] = (
                response_mask.any(dim=1)
            )

        noisy_batch = torch.where(masked_indices, self.mask_id, input_ids)

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
        logits = model(
            input_ids=noisy_batch,
            attention_mask=inputs["attention_mask"],
            response_mask=response_mask,
        ).logits

        # 1/t reweighting and per-example answer-length normalisation
        answer_lengths = (
            response_mask.sum(dim=-1, keepdim=True).clamp(min=1).expand(-1, l)
        )

        token_loss = (
            F.cross_entropy(
                logits[masked_indices], input_ids[masked_indices], reduction="none"
            )
            / p_mask[masked_indices]
        )
        loss = torch.sum(token_loss / answer_lengths[masked_indices]) / b

        return (loss, logits) if return_outputs else loss


def _gather_predictions(local_indices, local_preds, total, world_size):
    """Re-assemble the full, in-order prediction list from every rank's shard."""
    preds = [float("inf")] * total
    if world_size <= 1 or not torch.distributed.is_initialized():
        for i, pred in zip(local_indices, local_preds):
            preds[i] = pred
        return preds

    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, (local_indices, local_preds))
    for indices, values in gathered:
        for i, pred in zip(indices, values):
            preds[i] = pred
    return preds


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
    """
    model.eval()

    questions = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
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

    prompt_ids = {
        i: tokenizer(questions[i], return_tensors="pt")["input_ids"][0]
        for i in local_indices
    }
    pad_id = _pad_token_id(tokenizer)
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
        # every row of the batch
        gen_tokens = out[:, input_ids.shape[1] :]
        decoded_batch = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for i, decoded in zip(batch_indices, decoded_batch):
            if rank == 0:
                print(decoded)
            preds_by_index[i] = extract_answer_number(decoded)
        progress.update(len(batch_indices))
    progress.close()

    local_preds = [preds_by_index[i] for i in local_indices]
    ans_pred_list = _gather_predictions(
        local_indices, local_preds, len(questions), world_size
    )
    accuracy = compute_accuracy(answers, ans_pred_list)
    return accuracy, ans_pred_list, answers


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
    if model_args.tuning_type == "lora":
        from peft import LoraConfig

        return LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=llada_target_modules,
            init_lora_weights=True,
        )
    elif model_args.tuning_type == "lorta":
        from peft import LorTaConfig

        return LorTaConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=llada_target_modules,
            lora_dropout=0.1,
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
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            embedding_length=32,
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
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            embedding_dim=64,
            embedding_type="fourier",
            fnn_hidden_size_1=256,
            fnn_hidden_size_2=512,
            c_scale=0.1,
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

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape} parameters")

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
    trainer.train()
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


def _write_benchmark_result(output_dir: str, result: Dict):
    path = os.path.join(output_dir, "benchmark.json")
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


def _gsm8k_accuracy_benchmark(model_args, data_args, training_args, adapter_dir):
    """Load base model + trained adapter and score GSM8K test accuracy."""
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
        model = PeftModel.from_pretrained(model, adapter_dir)
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

    tokenizer = _load_tokenizer(model_args, training_args)
    logging.warning("Downloading Data")
    test_set = load_dataset(data_args.data_name, "main", split="test")

    accuracy, ans_pred_list, answers = diffusion_evaluate(
        model, tokenizer, test_set, training_args
    )
    label = "base model (no adapter)" if is_baseline else f"adapter: {adapter_dir}"
    print(f"{label} | GSM8K test accuracy: {100*accuracy:.2f}%")
    return {
        "benchmark": "gsm8k_accuracy",
        "tuning_type": model_args.tuning_type,
        "accuracy": accuracy,
        "num_examples": len(answers),
        "predictions": ans_pred_list,
        "ground_truth": answers,
        "source": "benchmark",
    }


# Registry seam: name -> callable(model_args, data_args, training_args, adapter_dir).
BENCHMARKS = {
    "gsm8k_accuracy": _gsm8k_accuracy_benchmark,
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
