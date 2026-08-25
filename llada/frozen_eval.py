"""`eval_loss_frozen` -- Tier 0's only metric (`EXPERIMENT_DESIGN.md` §3.4, §6).

Diffusion eval loss is dominated by the random draw of `t` and of the mask, which is
enough noise to make learning-rate ranking from short runs impossible. So the draws are
made once, committed, and reused by every evaluation of every run: 250 dev questions ×
8 stratified noise levels, fully crossed, so each adapter is scored on the same question
at the same noise level every time.

The design and its sign-off are in `FROZEN_EVAL_SPEC.md`; the parts of it that are load-
bearing here:

  - The objective is *not* reimplemented. `train.diffusion_masked_loss` is the same
    function the run was trained with, and both paths call it (§1).
  - A batch is one question's eight noise levels. The rows are the same token sequence,
    so there is no padding, and batch composition is fixed by construction rather than
    by a batch-size setting -- which is what makes the metric reproducible in bf16,
    where padding changes matmul reduction shapes and therefore the result (§5).
  - The >=1-masked-token guarantee is per *item* here and per *batch* in training. That
    is deliberate and it is a divergence; see §5a and `forced` below.

Regenerating the committed set (needs a tokenizer, so: on the cluster, once):
    python llada/frozen_eval.py --write
"""

import argparse
import hashlib
import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence

import torch

from splits import DEV_SEED, DEV_SIZE, dev_split, load_dev_indices
from train import (
    IGNORE_INDEX,
    SupervisedDataset,
    _gather_by_index,
    _load_tokenizer,
    diffusion_masked_loss,
)

VERSION = 1
NUM_T = 8

# Seeds the mask draws. Distinct from DEV_SEED so that "which questions" and "which
# tokens within them" cannot be made to move together by editing one constant.
FROZEN_MASK_SEED = 20260826

# The `diffusion_eps` in force when the masks were drawn. The *evaluation* derives
# p_mask from `training_args.diffusion_eps` (FROZEN_EVAL_SPEC.md §2), so this is
# recorded for the audit trail rather than used: if the two ever differ, the masks were
# drawn under a marginally different distribution than they are reweighted by, which is
# worth knowing and not worth regenerating over (the difference at eps=1e-3 is ~0.1%).
GENERATION_DIFFUSION_EPS = 1e-3

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_data")
FROZEN_EVAL_FILE = os.path.join(_DATA_DIR, f"frozen_eval_v{VERSION}.json")


def noise_levels(num_t: int = NUM_T) -> List[float]:
    """Stratified midpoints of (0, 1]: `(i + 0.5) / num_t`.

    Even spacing with neither degenerate endpoint -- no `t -> 0` (nothing masked) and no
    `t = 1` (everything masked). As a quadrature of the `U(eps, 1]` that training draws
    from, the mean over the buckets estimates the same expectation the training
    objective takes, so `eval_loss_frozen` is comparable in magnitude to the loss curve.
    """
    return [(i + 0.5) / num_t for i in range(num_t)]


# --------------------------------------------------------------------------- #
# The dev set, tokenised exactly as training tokenises it                      #
# --------------------------------------------------------------------------- #


def build_dev_dataset(tokenizer, data_args) -> SupervisedDataset:
    """The 250 dev questions through the *training* preprocessing path.

    `SupervisedDataset` is reused rather than re-tokenised here so that the prompt
    template, the truncation and the prompt/answer boundary are the ones the run was
    trained under. The content hash below is only meaningful because both the generator
    and the evaluator come through this same constructor.
    """
    from datasets import load_dataset

    train_set = load_dataset(data_args.data_name, "main", split="train")
    return SupervisedDataset(raw_data=dev_split(train_set), tokenizer=tokenizer)


def _response_bounds(labels: torch.Tensor):
    """(first response position, response length) for one example.

    `preprocess` marks the prompt with IGNORE_INDEX and leaves the answer, so the
    response is a contiguous suffix. An empty response means the example was truncated
    inside its own prompt -- there is nothing to mask, nothing to predict, and it cannot
    be a dev item.
    """
    positions = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError(
            "A dev example has no response tokens: the prompt filled or exceeded "
            "model_max_length, so `labels` is IGNORE_INDEX throughout. There is nothing "
            "for the diffusion loss to mask. Raise model_max_length or drop the example "
            "from the split -- silently scoring it as zero would flatter every adapter "
            "equally and hide the truncation."
        )
    return int(positions[0]), int(positions.numel())


def content_sha256(dataset: SupervisedDataset, tokenizer, model_max_length: int) -> str:
    """Hash of everything the stored mask positions are relative to.

    Covers the tokenizer, `model_max_length`, each example's `input_ids`, **and** each
    example's response boundary. The boundary is in here because the positions are only
    meaningful relative to it: a change that moved the prompt/answer split would
    otherwise have to show up in `input_ids` to be caught, and "would otherwise have to"
    is not a safety argument.

    Any change to tokenisation, prompt template or the dev split moves the hash and
    `assert_matches` refuses to run -- because the alternative is scoring two adapters
    against different masks and reporting the difference as a result.
    """
    digest = hashlib.sha256()
    digest.update(f"frozen_eval_v{VERSION}\0".encode())
    digest.update(f"{tokenizer.name_or_path}\0".encode())
    digest.update(f"{model_max_length}\0".encode())
    for position in range(len(dataset)):
        example = dataset[position]
        start, length = _response_bounds(example["labels"])
        payload = json.dumps(
            [example["input_ids"].tolist(), start, length], separators=(",", ":")
        )
        digest.update(payload.encode())
        digest.update(b"\0")
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Generating and loading the frozen set                                        #
# --------------------------------------------------------------------------- #


def build_frozen_set(dataset: SupervisedDataset, tokenizer, model_max_length: int) -> Dict:
    """Draw the (t, mask) pairs once. Deterministic given the constants in this module."""
    gsm8k_indices = load_dev_indices()
    if len(dataset) != len(gsm8k_indices):
        raise RuntimeError(
            f"Dev dataset has {len(dataset)} rows but the split records "
            f"{len(gsm8k_indices)} indices."
        )

    rng = random.Random(FROZEN_MASK_SEED)
    levels = noise_levels()
    items, forced_total = [], 0

    for position in range(len(dataset)):
        example = dataset[position]
        # Raises, with the explanation, if the example was truncated inside its own
        # prompt and so has nothing maskable.
        _response_bounds(example["labels"])
        response = (example["labels"] != IGNORE_INDEX).nonzero(as_tuple=True)[0].tolist()
        for t_index, t in enumerate(levels):
            p_mask = (1 - GENERATION_DIFFUSION_EPS) * t + GENERATION_DIFFUSION_EPS
            masked = [pos for pos in response if rng.random() < p_mask]
            # Per-item, unlike training's per-batch guarantee (FROZEN_EVAL_SPEC.md §5a):
            # an item with nothing masked contributes an exact zero, which drags its
            # bucket's mean down while saying nothing about the adapter. Recorded rather
            # than merely done, so the size of the effect stays auditable.
            forced = not masked
            if forced:
                masked = [response[0]]
                forced_total += 1
            items.append(
                {
                    "dev_pos": position,
                    "gsm8k_index": int(gsm8k_indices[position]),
                    "t_index": t_index,
                    "t": t,
                    "masked_positions": masked,
                    "forced": forced,
                }
            )

    per_level = [
        sum(1 for item in items if item["forced"] and item["t_index"] == i)
        for i in range(len(levels))
    ]
    print(
        f"{len(items)} items ({len(dataset)} questions x {len(levels)} noise levels); "
        f"{forced_total} forced to one masked token, per t bucket: {per_level}"
    )

    return {
        "version": VERSION,
        "tokenizer": str(tokenizer.name_or_path),
        "model_max_length": int(model_max_length),
        "dev_seed": DEV_SEED,
        "dev_size": DEV_SIZE,
        "num_t": len(levels),
        "mask_seed": FROZEN_MASK_SEED,
        "diffusion_eps_at_generation": GENERATION_DIFFUSION_EPS,
        "content_sha256": content_sha256(dataset, tokenizer, model_max_length),
        "num_forced": forced_total,
        "items": items,
    }


def load_frozen_set(path: str = FROZEN_EVAL_FILE) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No frozen eval set at {path}. Generate it once with "
            "`python llada/frozen_eval.py --write` (needs the tokenizer, so run it "
            "where the model is available) and commit the result."
        )
    with open(path) as f:
        record = json.load(f)
    if record.get("version") != VERSION:
        raise RuntimeError(
            f"{path} is version {record.get('version')}, this code expects {VERSION}."
        )
    return record


def assert_matches(record: Dict, dataset: SupervisedDataset, tokenizer, model_max_length: int):
    """Refuse to score against masks that were drawn for different data.

    Two checks, doing different jobs. The hash catches a changed tokenizer, template,
    truncation length or dev split -- anything that moves what a position *means*. The
    per-item position check catches an indexing bug in the generator, which the hash
    cannot see because it would leave the data untouched.
    """
    actual = content_sha256(dataset, tokenizer, model_max_length)
    if actual != record["content_sha256"]:
        raise RuntimeError(
            "The frozen eval set was generated against different data: content hash "
            f"{record['content_sha256'][:12]}... on file, {actual[:12]}... now "
            f"(tokenizer {record['tokenizer']} -> {tokenizer.name_or_path}, "
            f"model_max_length {record['model_max_length']} -> {model_max_length}). "
            "The stored mask positions no longer mean what they meant, so any two "
            "adapters compared across this change are being scored on different masks."
        )

    # One position set per question rather than one per item: the same 250 examples
    # back 2000 items, and `response[p]` on a tensor is a device-free but per-element
    # Python round trip.
    allowed: Dict[int, set] = {}
    for item in record["items"]:
        dev_pos = item["dev_pos"]
        if dev_pos not in allowed:
            labels = dataset[dev_pos]["labels"]
            allowed[dev_pos] = set(
                (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0].tolist()
            )
        positions = item["masked_positions"]
        if not positions:
            raise RuntimeError(
                f"Item (dev_pos={item['dev_pos']}, t_index={item['t_index']}) has no "
                "masked positions, so its loss would be exactly zero."
            )
        if not set(positions) <= allowed[dev_pos]:
            raise RuntimeError(
                f"Item (dev_pos={item['dev_pos']}, t_index={item['t_index']}) masks "
                "positions outside its own response region. The loss masks only ever "
                "land in the answer, so this is an indexing bug in the generator -- "
                "most likely a dev-split position confused with a GSM8K row index."
            )


def _assert_per_example_lambda(model):
    """Reject `pool_lambda=True` -- it would make batch composition change the metric.

    A batch here is one question's eight noise levels (FROZEN_EVAL_SPEC.md §5). With
    lambda pooled over the batch, those eight would be averaged into one conditioning
    signal and every noise-aware adapter would be scored on a lambda it never sees.
    `pool_lambda` has defaulted to False since O2; this is here so that turning it back
    on to reproduce an old run fails loudly instead of quietly returning a number.
    """
    inner = getattr(model, "module", model)
    for name, config in getattr(inner, "peft_config", {}).items():
        if getattr(config, "pool_lambda", False):
            raise RuntimeError(
                f"Adapter '{name}' has pool_lambda=True. The frozen eval batches eight "
                "noise levels of one question together, so a pooled lambda would blend "
                "them into a single value and score the adapter on conditioning it was "
                "never trained with."
            )


# --------------------------------------------------------------------------- #
# The metric                                                                   #
# --------------------------------------------------------------------------- #


def _question_batch(dataset, items: Sequence[Dict], device):
    """One question's `num_t` noise levels as a batch. No padding: the rows are equal."""
    example = dataset[items[0]["dev_pos"]]
    ids, labels = example["input_ids"], example["labels"]
    n, length = len(items), ids.shape[0]

    masked_indices = torch.zeros(n, length, dtype=torch.bool)
    for row, item in enumerate(items):
        masked_indices[row, item["masked_positions"]] = True

    return {
        "input_ids": ids[None].repeat(n, 1).to(device),
        "response_mask": (labels != IGNORE_INDEX)[None].repeat(n, 1).to(device),
        # long, matching what `DataCollatorForSupervisedDataset` feeds training
        "attention_mask": torch.ones(n, length, dtype=torch.long, device=device),
        "t": torch.tensor([item["t"] for item in items], dtype=torch.float32, device=device),
        "masked_indices": masked_indices.to(device),
    }


@torch.no_grad()
def frozen_eval_loss(
    model,
    tokenizer,
    data_args,
    training_args,
    dataset: Optional[SupervisedDataset] = None,
    record: Optional[Dict] = None,
    pass_response_mask: bool = True,
) -> Dict:
    """`eval_loss_frozen` for one model, plus the per-`t` breakdown.

    Deterministic by construction rather than by care: `model.eval()` (without which
    `lora_dropout=0.1` alone makes every Tier 0 comparison noise), no grad, fixed item
    order, padding-free batches, and a final reduction with `math.fsum`, which is
    exactly rounded and therefore independent of the order the ranks report in.

    `dataset` and `record` are accepted so the in-training callback can build them once
    rather than re-tokenising 250 questions every time it fires.
    """
    if record is None:
        record = load_frozen_set()
    if dataset is None:
        dataset = build_dev_dataset(tokenizer, data_args)
    assert_matches(record, dataset, tokenizer, training_args.model_max_length)
    _assert_per_example_lambda(model)

    by_question: Dict[int, List[Dict]] = {}
    for item in record["items"]:
        by_question.setdefault(item["dev_pos"], []).append(item)
    for position, items in by_question.items():
        items.sort(key=lambda item: item["t_index"])
        if len(items) != record["num_t"]:
            raise RuntimeError(
                f"dev_pos={position} has {len(items)} items, expected {record['num_t']}: "
                "the frozen set is meant to be fully crossed."
            )
    questions = sorted(by_question)

    world_size, rank = training_args.world_size, training_args.process_index
    local = list(range(rank, len(questions), world_size))

    was_training = model.training
    model.eval()
    try:
        local_losses = []
        for slot in local:
            items = by_question[questions[slot]]
            batch = _question_batch(dataset, items, training_args.device)
            per_example, _ = diffusion_masked_loss(
                model,
                input_ids=batch["input_ids"],
                response_mask=batch["response_mask"],
                attention_mask=batch["attention_mask"],
                t=batch["t"],
                masked_indices=batch["masked_indices"],
                mask_id=training_args.mask_id,
                diffusion_eps=training_args.diffusion_eps,
                pass_response_mask=pass_response_mask,
            )
            # float64 on the host: the accumulation below must not depend on how the
            # questions were sharded, and bf16/float32 addition does.
            local_losses.append([float(value) for value in per_example.double().cpu()])
    finally:
        model.train(was_training)

    gathered = _gather_by_index(local, local_losses, len(questions), world_size, fill=None)
    missing = [i for i, value in enumerate(gathered) if value is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} question(s) produced no loss after the gather "
            f"(first: dev_pos={questions[missing[0]]}); the shards did not cover the set."
        )

    num_t = record["num_t"]
    per_t = [math.fsum(row[i] for row in gathered) / len(questions) for i in range(num_t)]
    total = math.fsum(value for row in gathered for value in row)
    num_items = len(questions) * num_t
    # Read off the record rather than recomputed from `noise_levels(num_t)`: identical
    # today, but the buckets reported must be the buckets that were scored.
    t_values = [item["t"] for item in by_question[questions[0]]]

    return {
        "eval_loss_frozen": total / num_items,
        "per_t": per_t,
        "t_values": t_values,
        "num_items": num_items,
        "num_questions": len(questions),
        "frozen_eval_sha256": record["content_sha256"],
    }


# --------------------------------------------------------------------------- #
# CLI: generate the committed set                                              #
# --------------------------------------------------------------------------- #


def _serialise(record: Dict) -> str:
    """Header pretty-printed, one item per line.

    `indent=2` would expand every `masked_positions` list one integer per line --
    ~120k lines for a file whose whole reason to be JSON rather than `.pt` is that it
    diffs. One line per item keeps it readable and keeps a regeneration's diff
    interpretable: the items that moved are the lines that moved.
    """
    header = {key: value for key, value in record.items() if key != "items"}
    lines = ["{"]
    for key, value in header.items():
        lines.append(f"  {json.dumps(key)}: {json.dumps(value)},")
    lines.append('  "items": [')
    items = record["items"]
    for index, item in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        lines.append("    " + json.dumps(item, separators=(",", ":")) + comma)
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _write(record: Dict, path: str = FROZEN_EVAL_FILE, force: bool = False):
    """Write the committed set, refusing to silently replace a different one.

    Regenerating over an existing frozen set is how two adapters end up scored against
    different masks with the difference reported as a result -- and unlike every other
    check in this module, nothing downstream can detect it after the fact, because the
    hash would move with the file. `--force` is the deliberate override.
    """
    payload = _serialise(record)
    if os.path.exists(path) and not force:
        with open(path) as f:
            existing = f.read()
        if existing != payload:
            with open(path) as f:
                old_hash = json.load(f).get("content_sha256", "?")
            raise RuntimeError(
                f"{path} already exists and differs from what would be written "
                f"(content_sha256 {old_hash[:12]}... -> {record['content_sha256'][:12]}...). "
                "Every eval_loss_frozen already recorded was scored against the file on "
                "disk; replacing it makes those numbers incomparable with everything "
                "measured afterwards, and nothing downstream can notice. Pass --force if "
                "that is what you mean to do."
            )
        print(f"{path} is already up to date.")
        return

    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(payload)
    print(f"Wrote {len(record['items'])} frozen eval items -> {path}")
    print(f"content_sha256 = {record['content_sha256']}")


if __name__ == "__main__":
    from train import DataArguments, ModelArguments, TrainingArguments, build_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the committed set.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing frozen set that differs. Invalidates every "
        "eval_loss_frozen already recorded against it.",
    )
    parser.add_argument("--model_name_or_path", default=ModelArguments.model_name_or_path)
    parser.add_argument("--data_name", default=DataArguments.data_name)
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=TrainingArguments.model_max_length,
        help="Must match what the runs train and evaluate under; it is in the hash.",
    )
    parser.add_argument("--token", default=None, help="HF token, if the model is private.")
    args = parser.parse_args()

    # Through `build_args` rather than constructed by hand, so the defaults here are the
    # same ones every run resolves -- the hash covers model_max_length, so a divergence
    # would look like a corrupted frozen set rather than like a mismatched default.
    model_args, data_args, training_args = build_args(
        {
            "model_name_or_path": args.model_name_or_path,
            "data_name": args.data_name,
            "model_max_length": args.model_max_length,
            "token": args.token,
            "output_dir": ".",
            "report_to": "none",
        }
    )

    tok = _load_tokenizer(model_args, training_args)
    dev = build_dev_dataset(tok, data_args)
    frozen = build_frozen_set(dev, tok, args.model_max_length)
    if args.write:
        _write(frozen, force=args.force)
    else:
        print("(pass --write to record it in " + FROZEN_EVAL_FILE + ")")
