# ⚡ LoRTA: Tensor Parametrizations


## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Install dependencies

```bash
pip install -r requirements.txt tokenizers sentencepiece
```

All of our bash scripts are in the folder experiments.

Download the alpaca dataset and LLAMA 7B model:
```bash
./experiments/prepare_llama7b_alpaca.sh
```

The baseline lora can be run using
```bash
./experiments/lora-bline.sh
```

## Llama Experiments

You have to run first 
```bash
./experiments/prepare_llama7b_alpaca.sh
```

If any of these give you an out of memory, pass --micro_batch_size 2 to the script.
Modify CUDA_VISIBLE_DEVICES to run on a different GPU.

### Learning Rate Grid Search
I set up a grid search of 8 values for the learning rate using rank=256 which aproximately matches the number of parameters in the original LORA with rank 1. Feel free to modify the script to try different values. The default is 3e-4 for LoRA and 4e-3 for VeRA.

```bash
./experiments/lr_sweep.sh
```
### Rank Sweep
I set up a grid search of 8 values for the rank ranging from 6 to 768. Feel free to modify the script to try different values. 

```bash
./experiments/rank_sweep.sh
```

## Other things I want to fine-tune/play around with

* `alpha`: this hyperparameter scales the update ($W' = W+\alpha/r * dW$). Default is 16, but it gets scaled inversely by the matrix rank in OG LORA.
* `lora_dropout`: adapter dropout - we might need less or even no regularization. Default is 0.05.