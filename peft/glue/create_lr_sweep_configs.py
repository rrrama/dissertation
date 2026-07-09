import argparse
import json
import os


def generate_config(rank, task_name, model_name, lr_head, lr_lorta, epochs):
    # Select appropriate settings based on the model name
    max_seq_len = 128 if model_name == "roberta-large" else 512
    batch_size = 128 if model_name == "roberta-large" else 64  # Vera uses 32 bz x 4 gpus in roberta-large
    shared_dim = 1024 if model_name == "roberta-large" else 768

    config = {
        "do_train": True,
        "do_eval": True,
        "gradient_accumulation_steps": 1,
        "output_dir": "./output/model",
        "overwrite_output_dir": True,
        "logging_steps": 10,
        "logging_dir": "./output/log",
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "warmup_ratio": 0.06,
        "max_grad_norm": 1000.0,
        "weight_decay": 0.0,
        "shared_uv": 1,
        "shared_dim": shared_dim,
        "model_name_or_path": model_name,
        "per_device_train_batch_size": batch_size,
        "max_seq_length": max_seq_len,
        "mode": "lora",
        "lora_r": rank,
        "init_type": 1,
        "d_init_type": 94,
        "seed": 42,
        "task_name": task_name,
        "num_train_epochs": epochs,
        "classifier_lr": lr_head,
        "learning_rate": lr_lorta,
    }

    # Ensure directory exists
    filepath = f"configs/sweep/{model_name}/{task_name}/r={rank}/"
    os.makedirs(filepath, exist_ok=True)

    file_name = filepath + f"lr_{lr_lorta:.0E}_lrc_{lr_head:.0E}.json"
    with open(file_name, "w") as json_file:
        json.dump(config, json_file, indent=4)
        print(f"Generated config file: {file_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate configuration for model training.")
    parser.add_argument("--rank", type=int, default=4, help="The rank of LoRA layers.")
    parser.add_argument(
        "--model", type=str, default="roberta-base", help="The model name or path (e.g., roberta-base, roberta-large)."
    )
    parser.add_argument("--task", type=str, default=None, help="sst2 mrpc cola qnli rte or stsb.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs.")

    lr_grid = [5e-3, 1e-3, 1e-2, 5e-2, 1e-4]

    args = parser.parse_args()

    for lr_head in lr_grid:
        for lr_lorta in lr_grid:
            if args.task is None:
                for task in ["sst2", "mrpc", "cola", "qnli", "rte", "stsb"]:
                    generate_config(args.rank, task, args.model, lr_head, lr_lorta, args.epochs)
            else:
                generate_config(args.rank, args.task, args.model, lr_head, lr_lorta, args.epochs)


if __name__ == "__main__":
    main()
