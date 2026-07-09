import argparse
import json
import os


def generate_config(rank, task_name, model_name):
    # Settings for roberta-base
    base_task_settings = {
        "sst2": {"epochs": 60, "classifier_lr": 4.00e-3, "learning_rate": 4.00e-3},
        "mrpc": {"epochs": 30, "classifier_lr": 4.00e-3, "learning_rate": 1.00e-2},
        "cola": {"epochs": 80, "classifier_lr": 1.00e-3, "learning_rate": 1.00e-2},  # 1.00e-2  1.00e-02 in VeRA
        "qnli": {"epochs": 25, "classifier_lr": 4.00e-3, "learning_rate": 1.00e-2},
        "rte": {"epochs": 160, "classifier_lr": 1.00e-2, "learning_rate": 1.00e-2},  # 4.00e-3 in VeRA
        "stsb": {"epochs": 80, "classifier_lr": 5.00e-3, "learning_rate": 5.00e-2},  # 1.00e-2 1.00e-2 in VeRA
    }

    # Settings for roberta-large
    large_task_settings = {
        "sst2": {"epochs": 10, "classifier_lr": 6.00e-3, "learning_rate": 1.00e-2},
        "mrpc": {"epochs": 40, "classifier_lr": 3.00e-3, "learning_rate": 3.00e-2},
        "cola": {"epochs": 40, "classifier_lr": 6.00e-3, "learning_rate": 1.00e-2},
        "qnli": {"epochs": 20, "classifier_lr": 2.00e-4, "learning_rate": 1.00e-2},
        "rte": {"epochs": 40, "classifier_lr": 2.00e-3, "learning_rate": 2.00e-2},
        "stsb": {"epochs": 20, "classifier_lr": 2.00e-3, "learning_rate": 2.00e-2},
    }
    # Select appropriate settings based on the model name
    task_settings = large_task_settings if model_name == "roberta-large" else base_task_settings
    max_seq_len = 128 if model_name == "roberta-large" else 512
    batch_size = 64 if model_name == "roberta-large" else 64  # Vera uses 32 bz x 4 gpus in roberta-large
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
        "num_train_epochs": task_settings[task_name]["epochs"],
        "classifier_lr": task_settings[task_name]["classifier_lr"],
        "learning_rate": task_settings[task_name]["learning_rate"],
    }

    # Ensure directory exists
    if not os.path.exists("configs"):
        os.makedirs("configs")

    file_name = f"configs/{model_name}_{task_name}_r={rank}.json"
    print("writing file to: ", file_name)
    with open(file_name, "w") as json_file:
        json.dump(config, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Generate configuration for model training.")
    parser.add_argument("--rank", type=int, help="The rank of LoRA layers.")
    parser.add_argument("--model", type=str, help="The model name or path (e.g., roberta-base, roberta-large).")

    args = parser.parse_args()

    tasks = ["sst2", "mrpc", "cola", "qnli", "rte", "stsb"]
    for task in tasks:
        generate_config(args.rank, task, args.model)


if __name__ == "__main__":
    main()
