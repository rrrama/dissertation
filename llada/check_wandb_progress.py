#!/usr/bin/env python
"""Peek at an offline wandb run's progress without syncing to the cloud.

Usage: python check_wandb_progress.py [run_dir_or_glob]
Defaults to the most recently modified offline-run under ./wandb/.
"""
import glob
import os
import sys

from wandb.sdk.internal.datastore import DataStore
import wandb.proto.wandb_internal_pb2 as pb


def find_latest_run(pattern="wandb/offline-run-*/run-*.wandb"):
    candidates = glob.glob(os.path.join(os.path.dirname(__file__), pattern))
    if not candidates:
        raise SystemExit(f"No .wandb files found matching {pattern}")
    return max(candidates, key=os.path.getmtime)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    print(f"Reading {path}\n")

    ds = DataStore()
    ds.open_for_scan(path)

    last_history = None
    last_output = None
    last_stats = None

    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        which = record.WhichOneof("record_type")
        if which == "history":
            last_history = record.history
        elif which == "output_raw" and record.output_raw.line.strip():
            last_output = record.output_raw.line
        elif which == "stats":
            last_stats = record.stats

    if last_output:
        print("Last stdout/stderr line:")
        print(" ", last_output.strip())

    if last_history:
        print("\nLatest logged metrics:")
        for item in last_history.item:
            print(f"  {item.key} = {item.value_json}")
    else:
        print("\nNo metrics logged yet (waiting for first logging_steps interval).")

    if last_stats:
        print("\nLatest system stats:")
        for item in last_stats.item:
            if item.key in ("cpu", "memory_percent", "proc.memory.rssMB"):
                print(f"  {item.key} = {item.value_json}")


if __name__ == "__main__":
    main()
