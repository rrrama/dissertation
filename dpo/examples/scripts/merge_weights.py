# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Dict, Optional
import json

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.trainer import DPOfTrainer as DPOTrainer

from safetensors.torch import load_file as safe_load_file

from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LorTaConfig, LoraConfig
import wandb


@dataclass
class ScriptArguments:
    max_length: int = field(default=1024, metadata={"help": "max length of each sample"})
    max_prompt_length: int = field(default=128, metadata={"help": "max length of each sample's prompt"})#128
    max_target_length: int = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )#128
    sanity_check: bool = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    output: str = field(default="./merged/", metadata={"help": "output path"})
    adapter_path: str = field(default="./dpo_intel/lorta-r-1/", metadata={"help": "adapter path"})
    adapter: str = field(default="lorta", metadata={"help": "adapter type"})


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()
   
    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    if hasattr(quantization_config, "bnb_4bit_compute_dtype"):
        quantization_config.bnb_4bit_compute_dtype=torch.bfloat16
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    #model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(f"{args.adapter_path}/adapter_config.json", "r") as f:
        adapter_config_dict = json.load(f)
    if args.adapter == "lorta":
        peft_config = LorTaConfig(**adapter_config_dict)
    else:
        peft_config = LoraConfig(**adapter_config_dict)
    model = get_peft_model(model, peft_config)
    # load adapters
    # loas adapter config
    # Step 4: Load the adapter weights
    adapter_weights = safe_load_file(f"{args.adapter_path}/adapter_model.safetensors")

    # Manually load the adapter weights
    missing_keys, unexpected_keys = model.load_state_dict(adapter_weights, strict=False)
    
    if unexpected_keys:
        print(f"Unexpected keys when loading adapter weights: {unexpected_keys}")


    model = model.merge_and_unload()
    model.save_pretrained(f"{args.output}")
    tokenizer.save_pretrained(f"{args.output}")