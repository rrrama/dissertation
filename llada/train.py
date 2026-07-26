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
        metadata={"help": "Type of tuning to use: 'lora' or 'lorta'."},
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
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


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

        logits = model(input_ids=noisy_batch).logits

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


@torch.no_grad()
def diffusion_evaluate(model, tokenizer, test_set, training_args):
    """Evaluate GSM8K accuracy using the LLaDA diffusion sampler (one Q at a time)."""
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

    ans_pred_list = []
    for question in tqdm(
        questions, total=len(questions), desc="Evaluating (diffusion)"
    ):
        input_ids = tokenizer(question, return_tensors="pt")["input_ids"].to(
            model.device
        )
        out = generate(
            model,
            input_ids,
            steps=training_args.diffusion_steps,
            gen_length=training_args.gen_length,
            block_length=training_args.block_length,
            temperature=training_args.gen_temperature,
            remasking=training_args.remasking,
            mask_id=training_args.mask_id,
        )
        gen_tokens = out[:, input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        print(decoded)
        ans_pred_list.append(extract_answer_number(decoded))

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

    # No evaluation here: diffusion sampling is expensive and single-GPU, so it
    # is a separate job (`--mode benchmark`) run against the saved adapter.
    if is_main_process:
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


def _gsm8k_accuracy_benchmark(model_args, data_args, training_args, adapter_dir):
    """Load base model + trained adapter and score GSM8K test accuracy."""
    from peft import PeftModel

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    tokenizer = _load_tokenizer(model_args, training_args)
    logging.warning("Downloading Data")
    test_set = load_dataset(data_args.data_name, "main", split="test")

    accuracy, ans_pred_list, answers = diffusion_evaluate(
        model, tokenizer, test_set, training_args
    )
    print(f"adapter: {adapter_dir} | GSM8K test accuracy: {100*accuracy:.2f}%")
    return {
        "benchmark": "gsm8k_accuracy",
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

    adapter_config = os.path.join(output_dir, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(
            f"No trained adapter found in {output_dir} (missing adapter_config.json). "
            "Train this run before benchmarking."
        )

    model_args, data_args, training_args = build_args(config)
    training_args.output_dir = output_dir
    result = BENCHMARKS[bench_name](model_args, data_args, training_args, output_dir)
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
