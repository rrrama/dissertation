# Copyright 2023-present the HuggingFace Inc. team.
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
"""Scalar embeddings and the shared mapper MLP that turn a noise level into NaRA's `C`."""

from __future__ import annotations

import math

import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for scalar noise/mask levels.

    Vendored from `diffusers.models.embeddings` so that this fork does not depend on
    `diffusers`, whose recent releases require `peft>=0.17` and therefore refuse to
    import alongside this fork (0.10.1.dev0).

    The random projection is an `nn.Parameter` with `requires_grad=False` rather than a
    registered buffer. That is deliberate: `get_peft_model_state_dict` selects adapter
    tensors by the `lora_` substring in their *parameter* names, so a buffer would be
    dropped from the checkpoint and the projection silently redrawn on reload -- which
    reinterprets every noise level and makes the trained mapper meaningless, with no error.
    """

    def __init__(
        self,
        embedding_size: int = 256,
        scale: float = 1.0,
        log: bool = True,
        flip_sin_to_cos: bool = False,
    ):
        super().__init__()
        # Frozen random projection: drawn once at init and never trained.
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log:
            x = torch.log(x)

        x_proj = x[:, None] * self.weight[None, :] * 2 * math.pi

        if self.flip_sin_to_cos:
            return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPEmbedding(nn.Module):
    """Learnable scalar embedding: `Linear(1 -> embedding_dim) -> SiLU`."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, embedding_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch]
        return self.act(self.proj(x[:, None].to(dtype=self.proj.weight.dtype)))


class RawEmbedding(nn.Module):
    """Pass-through: the scalar itself is the embedding (output dim 1)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch]
        return x[:, None]


def build_embedding(embedding_type: str, embedding_dim: int) -> tuple[nn.Module, int]:
    """Construct the scalar embedding and report the feature width it produces.

    The width is the mapper's input dimension. For `"fourier"` the module concatenates
    sin and cos, so it is built at half width to produce `embedding_dim` features in
    total -- matching the reference implementation, whose `embed_dim` is the *total*.
    `scale=16.0` and no log are also the reference's settings, and both matter: the
    reference draws its projection at scale 16 over inputs in [0, 1], and `log=True`
    (the diffusers default) would send a fully-unmasked input at lambda=0 to -inf.
    """
    if embedding_type == "fourier":
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"`embedding_dim` must be even for the Gaussian Fourier embedding (got {embedding_dim}); "
                "it concatenates sin and cos to reach the requested width."
            )
        return GaussianFourierProjection(embedding_size=embedding_dim // 2, scale=16.0, log=False), embedding_dim
    if embedding_type == "mlp":
        return MLPEmbedding(embedding_dim), embedding_dim
    if embedding_type == "raw":
        return RawEmbedding(), 1
    raise ValueError(f"Unknown `embedding_type`: {embedding_type!r}. Expected 'fourier', 'mlp' or 'raw'.")


class NARAMapper(nn.Module):
    """The single globally shared MLP `F_phi` that maps an embedded noise level to `C`.

    `Linear(input_dim, h1) -> SiLU -> Linear(h1, h2) -> SiLU -> Linear(h2, r*r)`. One
    instance is shared by every adapted matrix in every layer, so `C` is computed once
    per forward and broadcast; it is by far the cheapest part of the adapter.
    """

    def __init__(self, r: int, input_dim: int, fnn_hidden_size_1: int, fnn_hidden_size_2: int, init_c: str):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"NARAMapper needs a positive input dimension, got {input_dim}")

        self.init_c = init_c
        self.model = nn.Sequential(
            nn.Linear(input_dim, fnn_hidden_size_1),
            nn.SiLU(),
            nn.Linear(fnn_hidden_size_1, fnn_hidden_size_2),
            nn.SiLU(),
            nn.Linear(fnn_hidden_size_2, r * r),
        )
        self.reset_c_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def reset_c_parameters(self) -> None:
        linears = [m for m in self.model.modules() if isinstance(m, nn.Linear)]

        if self.init_c == "zero_last":
            # Zeroing only the output layer makes F_phi(.) == 0 at step 0, hence C == I_r,
            # hence a NaRA adapter that is bit-identical to plain LoRA before any training.
            *others, last = linears
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
            for m in others:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif self.init_c == "kaiming_uniform_m":
            for m in linears:
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif self.init_c == "zero_all":
            for m in linears:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        else:
            raise ValueError(
                f"Unknown `init_c`: {self.init_c!r}. Expected 'zero_last', 'kaiming_uniform_m' or 'zero_all'."
            )
