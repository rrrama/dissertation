import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from transformers import get_cosine_schedule_with_warmup

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    lazy_load,
    num_parameters,
    quantization,
    step_csv_logger,
)
from scripts.prepare_alpaca import generate_prompt

from eval.lm_eval_harness import EvalHarnessBase

import wandb
eval_interval = 100
save_interval = 1000
eval_iters = 100
eval_max_new_tokens = 100
log_interval = 1
devices = 1
# change this value to force a maximum sequence length
override_max_seq_length = None

# Hyperparameters
learning_rate = 3e-4
batch_size = 16
micro_batch_size = 4
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
max_iters = 51000 // micro_batch_size # train dataset size
weight_decay = 0.01
alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = True
lora_value = True
lora_projection = True
lora_mlp = False
lora_head = False
warmup_steps = int(0.1*max_iters*micro_batch_size//batch_size)
tensor_lora = True
joint_heads = True
joint_layers = True
joint_qk_vp = False
joint_qkvp = True
#stochastic_layer_threshold = 0.5
# evaluation
eval_tasks= []#"arc_challenge", "piqa", "hellaswag"]#"winogrande"]
num_fewshot=0

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/lora/alpaca"),
    precision: Optional[str] = None,
    rank: int = 8,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"]] = None,
    learning_rate: float = 3e-4,
    alpha: int = 16,
    tensor_lora: bool = True,
    joint_heads: bool = True,
    joint_layers: bool = True,
    joint_qk_vp: bool = False,
    joint_qkvp: bool = True,
    lora_dropout: float = 0.05,
    eval_tasks: Optional[List[str]] = [],#["arc_challenge", "piqa", "hellaswag"], # add  "hendrycksTest-*" to run MMLU
    num_fewshot: int = 0,
    init_scale: float = 1.0,
    micro_batch_size: int = 4,
    max_iters: int = 51000 // micro_batch_size,
    #stochastic_layer_threshold: float = 0.5,
):
    precision = precision or get_default_supported_precision(training=True)

    fabric_devices = devices
    if fabric_devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. "
                "Please set devices=1 when using the --quantization flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
        )
    else:
        strategy = "auto"
    hparams["lora_r"] = rank
    hparams["alpha"] = alpha
    hparams["learning_rate"] = learning_rate
    #hparams["stochastic_layer_threshold"] = stochastic_layer_threshold
    hparams["tensor_lora"] = tensor_lora
    hparams["joint_heads"] = joint_heads
    hparams["joint_layers"] = joint_layers
    hparams["joint_qk_vp"] = joint_qk_vp
    hparams["joint_qkvp"] = joint_qkvp
    hparams["eval_tasks"] = eval_tasks
    hparams["num_fewshot"] = num_fewshot
    hparams["lora_dropout"] = lora_dropout
    hparams["init_scale"] = init_scale
    hparams["micro_batch_size"] = micro_batch_size
    hparams["max_iters"] = max_iters
    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, quantize, rank, hparams=hparams)

def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path, quantize: Optional[str] = None, lora_r: int = 8, hparams=None):
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    if not any((lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    config = Config.from_name(
        name=checkpoint_dir.name,
        r=lora_r,
        alpha=hparams["alpha"],
        dropout=hparams["lora_dropout"],
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
        tensor_lora=hparams["tensor_lora"],
        joint_heads=hparams["joint_heads"],
        joint_layers=hparams["joint_layers"],
        joint_qk_vp=hparams["joint_qk_vp"],
        joint_qkvp=hparams["joint_qkvp"],
        init_scale=hparams["init_scale"],
        #stochastic_layer_threshold=stochastic_layer_threshold,
    )
    name = f"lora_r_{config.r}"
    if config.tensor_lora:
        name = "tensor_" + name
        if config.joint_heads:
            name+="_joint_heads"
        if config.joint_layers:
            name+="_joint_layers"
        if config.joint_qk_vp:
            name+="_joint_qk_vp"
        if config.joint_qkvp:
            name+="_joint_qkvp"
    wandb.init(config=hparams, project="lora", name=name)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=False), quantization(quantize):
        model = GPT(config)
    with lazy_load(checkpoint_path) as checkpoint:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
    non_trainable = num_parameters(model, requires_grad=False)
    lora = num_parameters(model, requires_grad=True)
    wandb.log({"params/lora":lora,  "params/frac_lora": lora/(non_trainable+lora), "params/non_trainable": non_trainable})
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if quantize and quantize.startswith("bnb."):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW(trainable_params, lr=hparams["learning_rate"], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=hparams["learning_rate"], weight_decay=weight_decay)

    model, optimizer = fabric.setup(model, optimizer)
    scheduler  = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, hparams["warmup_steps"]), num_training_steps=int(hparams["max_iters"]//hparams["gradient_accumulation_iters"]))

    fabric.seed_everything(1337 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, speed_monitor, scheduler)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = checkpoint_dir / f"{name}_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)
    wandb.save(str(save_path), policy='now')

    if len(eval_tasks)> 0:
        eval_harness = EvalHarnessBase(
            checkpoint_dir=str(checkpoint_dir),
            model=model,
            quantize=quantize,
        )

    #results = eval_harness.run_eval(
    #    eval_tasks=["hendrycksTest-*"], num_fewshot=5, use_cache=False
    #)
    #wandb.log(results['results'])

        results = eval_harness.run_eval(
            eval_tasks=eval_tasks, num_fewshot=num_fewshot, use_cache=False
        )
        wandb.log(results['results'])


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    scheduler,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)
    model.max_seq_length = max_seq_length

    #validate(fabric, model, val_data, tokenizer, longest_seq_length)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        #mark_only_lora_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, longest_seq_length))
        measured_flops = measure_flops(meta_model, x)
        #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num in range(hparams["max_iters"]):

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, longest_seq_length, longest_seq_ix if iter_num == 0 else None
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)
        #fabric.print(f"sum of grads Cl: {torch.sum(model.lora_C_l.grad)}")
        #fabric.print(model.lora_C_m.grad)
        #fabric.print(model.lora_C_h.grad)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            scheduler.step()
            # log lr
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step_count)

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )
            wandb.log({"loss": loss.item()}, step=step_count)

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, longest_seq_length)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
            wandb.log({"val_loss": val_loss.item()}, step=step_count)
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)


@torch.inference_mode()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, longest_seq_length)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8)
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_lora_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})



if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
