"""Iterative diffusion sampler for LLaDA.

Ported from the reference implementation in the GSAI-ML/LLaDA repository
(https://github.com/ML-GSAI/LLaDA, `generate.py`). LLaDA is a masked diffusion
language model: generation starts from a fully-masked response and iteratively
"unmasks" tokens over a number of sampling steps, keeping the highest-confidence
predictions at each step (low-confidence remasking).

The MASK token id for LLaDA-8B is 126336.

Unlike the reference implementation this sampler is batched: `prompt` may hold
several left-padded prompts, with `attention_mask` marking the padding. See
`generate()` for why the padding must be on the left.
"""

import numpy as np
import torch
import torch.nn.functional as F


MASK_ID = 126336


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling.

    The reference notes that low-precision Gumbel noise hurts quality, so this is
    intentionally done in float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Number of tokens to unmask ("transfer") at each step.

    A masked-diffusion sampling schedule with a linear noise schedule unmasks, in
    expectation, an equal number of tokens per step. This distributes the total
    number of masked tokens as evenly as possible across ``steps``.

    Returned on the CPU: the only consumer is ``torch.topk(..., k=...)``, which
    needs a Python int. Reading a CUDA element for `k` forces a device sync on
    every step of every question and stops the CPU running ahead of the GPU.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True).cpu()

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_num.device, dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=MASK_ID,
):
    """Generate completions for a batch of prompts.

    Prompts of differing lengths must be **left**-padded, so that every row's
    response region starts at the same offset. The sampler unmasks a block at a
    time and the block bounds are shared across the batch; right-padding would
    interleave padding with response positions and break that. LLaDA attends
    bidirectionally, so the padding must additionally be masked out via
    ``attention_mask`` -- otherwise every row's result depends on how much
    padding its batch-mates happened to need.

    Args:
        model: a callable that maps ``input_ids -> ModelOutput`` with a ``.logits``
            field (the PEFT-wrapped LLaDA model works directly). It must accept an
            ``attention_mask`` keyword when one is supplied here; LLaDA's
            `modeling_llada.py` folds it into the (bidirectional) attention bias.
        prompt: LongTensor of shape ``(B, prompt_len)``, left-padded. Padding must
            not use ``mask_id``, or it would be decoded as a response position.
        attention_mask: 0/1 LongTensor of shape ``(B, prompt_len)``; 0 marks left
            padding. ``None`` means "no padding" (e.g. a batch of one).
        steps: total number of sampling steps (must be divisible by the number of
            blocks, i.e. ``gen_length // block_length``).
        gen_length: number of response tokens to generate.
        block_length: semi-autoregressive block size; the response is filled in
            left-to-right blocks, each denoised over ``steps / num_blocks`` steps.
        temperature: Gumbel sampling temperature (0 = greedy/argmax).
        cfg_scale: classifier-free-guidance scale (0 disables CFG).
        remasking: 'low_confidence' or 'random'.
        mask_id: the MASK token id (126336 for LLaDA-8B).

    Returns:
        LongTensor of shape ``(B, prompt_len + gen_length)`` -- the (padded)
        prompts followed by the generated tokens.
    """
    device = getattr(model, "device", prompt.device)
    batch_size, prompt_len = prompt.shape

    x = torch.full(
        (batch_size, prompt_len + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = prompt.to(device)

    if attention_mask is None:
        model_kwargs = {}
    else:
        attention_mask = attention_mask.to(device)
        assert attention_mask.shape == prompt.shape, (
            f"attention_mask {tuple(attention_mask.shape)} does not match prompt "
            f"{tuple(prompt.shape)}"
        )
        assert not (x[:, :prompt_len].eq(mask_id) & attention_mask.eq(0)).any(), (
            "prompts must be padded with something other than mask_id"
        )
        # the response region is always attended to
        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, gen_length), dtype=attention_mask.dtype, device=device
                ),
            ],
            dim=1,
        )
        model_kwargs = {"attention_mask": full_attention_mask}

    prompt_index = x != mask_id

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by the number of blocks"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                cfg_kwargs = {
                    k: torch.cat([v, v], dim=0) for k, v in model_kwargs.items()
                }
                logits = model(x_, **cfg_kwargs).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, **model_kwargs).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                # Confidence is only ever used to *rank* candidates in the topk
                # below, so fp32 is ample. The reference does this softmax in
                # fp64, which materialises a (B, S, V) fp64 tensor -- ~5 GB at
                # B=16 -- and runs tens of millions of fp64 exp() on a GPU that
                # has no fp64 transcendentals. log_softmax(dtype=float32) upcasts
                # inside the kernel, so no fp32 copy of the logits is needed.
                log_p = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                x0_p = torch.squeeze(
                    torch.gather(log_p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                ).exp()
                del log_p
            elif remasking == "random":
                x0_p = torch.rand(
                    (x0.shape[0], x0.shape[1]), device=x0.device, dtype=torch.float32
                )
            else:
                raise NotImplementedError(remasking)

            # never unmask beyond the current block
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
