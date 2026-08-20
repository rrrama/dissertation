"""Check that batched diffusion sampling matches one-question-at-a-time sampling.

Batching is not obviously output-preserving. LLaDA attends bidirectionally, and
NA-LoRTA modulates its adapter weights by a mask proportion pooled over the whole
batch, so a row's decode can in principle depend on its batch-mates. The
benchmark defends against both by only batching questions whose prompts are the
same length (`train._uniform_length_batches`); this script decodes such a batch
both ways and diffs the strings, and times both so the speedup is measured rather
than estimated.

Questions are picked to *share a prompt length*, since taking the first N
questions would give mostly singleton buckets and exercise nothing. Run on one
GPU (no torchrun needed):

    python check_batching.py --config <run_dir>/config.yaml --batch_size 8
"""

import argparse
import os
import time

import torch
import transformers
import yaml
from datasets import load_dataset

from generate import generate
from test import build_prompt, template_add_special_tokens
from train import _load_tokenizer, _pad_token_id, _stack_prompts, build_args


def _decode(model, tokenizer, prompt_ids, pad_id, training_args, batch_size):
    """Decode `prompt_ids` in batches of `batch_size`; return strings + seconds."""
    decoded = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for k in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[k : k + batch_size]
        input_ids, attention_mask = _stack_prompts(chunk, pad_id, model.device)
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
        decoded += tokenizer.batch_decode(
            out[:, input_ids.shape[1] :], skip_special_tokens=True
        )
    torch.cuda.synchronize()
    return decoded, time.perf_counter() - start


def _pick_equal_length_questions(questions, tokenizer, wanted):
    """Return up to `wanted` tokenized prompts that all have the same length."""
    add_special = template_add_special_tokens(tokenizer)
    by_length = {}
    for q in questions:
        ids = tokenizer(q, return_tensors="pt", add_special_tokens=add_special)[
            "input_ids"
        ][0]
        by_length.setdefault(ids.shape[0], []).append(ids)
    length, group = max(by_length.items(), key=lambda kv: len(kv[1]))
    print(
        f"prompt-length buckets over {len(questions)} questions: "
        f"{len(by_length)} distinct lengths, largest bucket = {len(group)} "
        f"questions at length {length}"
    )
    return group[:wanted]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A frozen per-run config.yaml.")
    parser.add_argument(
        "--adapter_dir",
        default=None,
        help="Run directory holding the adapter. Defaults to the config's directory.",
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=300,
        help="How many test questions to search for a same-length bucket.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    from peft import PeftModel

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    adapter_dir = args.adapter_dir or os.path.dirname(os.path.abspath(args.config))

    model_args, data_args, training_args = build_args(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        token=model_args.token,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.to(training_args.device)
    model.eval()

    tokenizer = _load_tokenizer(model_args, training_args)
    test_set = load_dataset(data_args.data_name, "main", split="test")
    questions = [
        build_prompt(tokenizer, example["question"])
        for example in test_set.select(range(min(args.pool, len(test_set))))
    ]
    prompt_ids = _pick_equal_length_questions(questions, tokenizer, args.batch_size)
    if len(prompt_ids) < 2:
        raise SystemExit(
            "No two questions share a prompt length in this pool -- raise --pool."
        )

    pad_id = _pad_token_id(tokenizer)
    single, single_s = _decode(model, tokenizer, prompt_ids, pad_id, training_args, 1)
    batched, batched_s = _decode(
        model, tokenizer, prompt_ids, pad_id, training_args, len(prompt_ids)
    )

    mismatches = [i for i, (a, b) in enumerate(zip(single, batched)) if a != b]
    n = len(prompt_ids)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nbatch=1    : {single_s:7.1f}s  ({single_s / n:6.2f}s/question)")
    print(
        f"batch={n:<5d}: {batched_s:7.1f}s  ({batched_s / n:6.2f}s/question)"
        f"  -> {single_s / batched_s:.2f}x"
    )
    print(f"peak GPU memory: {peak:.1f} GiB")
    print(f"identical decodes: {n - len(mismatches)}/{n}")
    for i in mismatches:
        print(f"\n--- question {i} ---\n[batch=1]\n{single[i]}\n[batched]\n{batched[i]}")


if __name__ == "__main__":
    main()
