"""The GSM8K dev carve-out (`EXPERIMENT_DESIGN.md` §3.3).

Without this, *every* selection decision in the project is made on the test set: the
training data is all of GSM8K train and the only accuracy benchmark scores all 1319 of
GSM8K test. §3.3 carves 250 questions out of **train** to select on, leaving test
untouched until the final numbers.

The carve-out is a deterministic function of `(DEV_SEED, DEV_SIZE, len(train))`, so it
is reproducible without any file having to survive. `split_data/gsm8k_dev_250.json` is
committed anyway, as an audit record and a tripwire: `load_dev_indices` re-derives the
indices and refuses to run if the committed file disagrees, which is what would happen
if the dataset were revised or the seed edited after runs had already been selected on
the old split.

Note on layout: the plan called for `llada/splits/`, but a `splits/` directory beside
`splits.py` shadows confusingly on `import splits`, so the data lives in
`llada/split_data/` instead.

Usage:
    from splits import train_split, dev_split
    train_set = train_split(load_dataset("gsm8k", "main", split="train"))

Regenerating the committed file (only ever needed once):
    python splits.py --write
"""

import json
import os
import random
from typing import List, Sequence

# Fixed once, on 2026-08-25, and not to be changed: every selection decision in the
# project is made against this split, so a new seed silently invalidates comparisons
# with anything already run.
DEV_SEED = 20260825
DEV_SIZE = 250

# GSM8K "main" train, as published. Asserted rather than assumed: the sample below is
# taken from `range(len(train))`, so a dataset revision that changed the row count
# would silently produce a *different* 250 questions under the same seed.
EXPECTED_TRAIN_SIZE = 7473

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split_data")
DEV_INDEX_FILE = os.path.join(_DATA_DIR, f"gsm8k_dev_{DEV_SIZE}.json")


def derive_dev_indices(train_size: int = EXPECTED_TRAIN_SIZE) -> List[int]:
    """The dev indices as a pure function of the constants above.

    `random.Random` is seeded explicitly and used for nothing else, so this is
    independent of global RNG state, of `transformers.set_seed`, and of whatever the
    caller did beforehand. Sorted on the way out: the order carries no meaning and a
    sorted list diffs readably in the committed file.
    """
    if train_size < DEV_SIZE:
        raise ValueError(f"Cannot carve {DEV_SIZE} dev questions out of {train_size} train rows.")
    return sorted(random.Random(DEV_SEED).sample(range(train_size), DEV_SIZE))


def load_dev_indices(train_size: int = EXPECTED_TRAIN_SIZE) -> List[int]:
    """The dev indices, cross-checked against the committed record.

    Derivation is the source of truth; the file is the tripwire. They can only
    disagree if the dataset changed underneath the project or a constant was edited
    after runs had been selected -- both of which invalidate every selection made so
    far, and neither of which should be discovered by noticing odd numbers later.
    """
    indices = derive_dev_indices(train_size)
    if os.path.exists(DEV_INDEX_FILE):
        with open(DEV_INDEX_FILE) as f:
            committed = json.load(f)
        recorded = committed["indices"]
        if recorded != indices:
            raise RuntimeError(
                f"The committed dev split ({DEV_INDEX_FILE}) does not match the split derived "
                f"from DEV_SEED={DEV_SEED}, DEV_SIZE={DEV_SIZE}, train_size={train_size}: "
                f"{len(set(recorded) ^ set(indices))} index/indices differ. Either the dataset "
                "was revised or a constant in splits.py was edited. Every selection decision "
                "made on the old split is invalid -- resolve this deliberately rather than "
                "deleting the file."
            )
        if committed.get("train_size") != train_size:
            raise RuntimeError(
                f"The committed dev split was derived from train_size="
                f"{committed.get('train_size')}, but the loaded dataset has {train_size} rows."
            )
    return indices


def _check_train_size(dataset) -> int:
    size = len(dataset)
    if size != EXPECTED_TRAIN_SIZE:
        raise RuntimeError(
            f"GSM8K train has {size} rows, expected {EXPECTED_TRAIN_SIZE}. The dev carve-out is "
            "sampled from range(len(train)), so a different row count means a different 250 "
            "questions under the same seed. Update EXPECTED_TRAIN_SIZE and regenerate the "
            "committed split only if you accept invalidating prior selections."
        )
    return size


def dev_split(train_dataset):
    """The 250 held-out questions. Selection and `eval_loss_frozen` run against these."""
    size = _check_train_size(train_dataset)
    return train_dataset.select(load_dev_indices(size))


def train_split(train_dataset):
    """GSM8K train minus the dev carve-out -- what every run actually trains on."""
    size = _check_train_size(train_dataset)
    held_out = set(load_dev_indices(size))
    return train_dataset.select([i for i in range(size) if i not in held_out])


def _write(indices: Sequence[int], train_size: int) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    record = {
        "dataset": "gsm8k/main:train",
        "train_size": train_size,
        "seed": DEV_SEED,
        "size": DEV_SIZE,
        "indices": list(indices),
        "note": (
            "Held out of training and used for all selection decisions (EXPERIMENT_DESIGN.md "
            "3.3). Derived by splits.derive_dev_indices; committed as an audit record."
        ),
    }
    with open(DEV_INDEX_FILE, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {DEV_SIZE} dev indices -> {DEV_INDEX_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Materialise the committed dev-index file from the constants in this module.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=EXPECTED_TRAIN_SIZE,
        help="Row count of GSM8K train to sample from (default: the published size).",
    )
    args = parser.parse_args()

    derived = derive_dev_indices(args.train_size)
    if args.write:
        _write(derived, args.train_size)
    else:
        print(f"{DEV_SIZE} dev indices from seed {DEV_SEED}: {derived[:5]} ... {derived[-5:]}")
        print(f"(pass --write to record them in {DEV_INDEX_FILE})")
