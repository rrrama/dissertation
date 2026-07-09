"""Iterative diffusion sampler for LLaDA.

Ported from the reference implementation in the GSAI-ML/LLaDA repository
(https://github.com/ML-GSAI/LLaDA, `generate.py`). LLaDA is a masked diffusion
language model: generation starts from a fully-masked response and iteratively
"unmasks" tokens over a number of sampling steps, keeping the highest-confidence
predictions at each step (low-confidence remasking).

The MASK token id for LLaDA-8B is 126336.
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
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=MASK_ID,
):
    """Generate a completion for a single (batch size 1) prompt.

    Args:
        model: a callable that maps ``input_ids -> ModelOutput`` with a ``.logits``
            field (the PEFT-wrapped LLaDA model works directly).
        prompt: LongTensor of shape ``(1, prompt_len)``.
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
        LongTensor of shape ``(1, prompt_len + gen_length)`` — the prompt followed
        by the generated tokens.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by the number of blocks"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
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
