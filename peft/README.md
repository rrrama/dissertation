<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p LoRTA 🤗 PEFT</p></h1>
<h3 align="center">
    <p> Tensor Low rank adapters (LoRTA) implementation using HF's PEFT</p>
</h3>

PEFT is integrated with Transformers for easy model training and inference, Diffusers for conveniently managing different adapters, and Accelerate for distributed training and inference for really big models.

I have modified LoRA (`src/peft/tuners/lora`) to implement Low Rank Tensor Adapters (LoRTA). It can be found in `src/peft/tuners/lorta` folder.

Migrating a training script from LoRa to LorTA should be as easy as changing `LoraConfig` by `LorTaConfig` and using 
`LorTaLayer` (from peft.tuners.lorta) instead of `LoraLayer` (from peft.tuners.lora).

In so far I have tested it with LLAMA-2-7B on Alpaca, when finetuning all linear modules at each layer ('q_proj', 'k_proj', 'v_proj', 'o_proj'), and roberta-base in GLUE using experiment scripts from the [VeRA paper](https://arxiv.org/abs/2310.11454) openreview submission, found in the folder `instruct` and `glue`.

Supporting other models and tasks might require updating the `TRANSFORMERS_MODELS_TO_LORTA_QKVO_MAPPING` and `TRANSFORMERS_MODELS_TO_LORTA_TARGET_MODULES_MAPPING` in `src/peft/utils/constants.py`.

## Quickstart

Install PEFT locally:

```bash
pip install -r requirements.txt
pip install .
```

Try an example:
    
```bash
cd instruct
./lora_llama2_7b.sh
```
