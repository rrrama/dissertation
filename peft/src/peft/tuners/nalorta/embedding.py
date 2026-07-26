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
from __future__ import annotations

import math

import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for scalar noise/mask levels.

    Vendored from `diffusers.models.embeddings` so that this fork does not depend on
    `diffusers`, whose recent releases require `peft>=0.17` and therefore refuse to
    import alongside this fork (0.10.1.dev0). Parameter name (`weight`) and output
    layout are kept identical to the diffusers implementation so existing checkpoints
    keep loading.
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
