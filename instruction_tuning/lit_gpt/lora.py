# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

import lit_gpt
from lit_gpt.config import Config as BaseConfig
from lit_gpt.model import GPT as BaseModel
from lit_gpt.model import Block as BaseBlock
from lit_gpt.model import CausalSelfAttention as BaseCausalSelfAttention
from lit_gpt.model import KVCache
from lit_gpt.utils import map_old_state_dict_weights
from lit_gpt.model import apply_rope


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            # TODO (iusername): check that omitting this does not affect the behaviour of og LORA
            #self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
            #self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            #self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self, dW: torch.Tensor=None):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        # TODO (iusername): test weight merging
        #raise NotImplementedError("(iusername) weight merging not implemented")
        if self.r > 0 and not self.merged:
            if dW is None:
                # Merge the weights and mark it
                self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            else:
                self.linear.weight.data += dW.T * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor, dW: Optional[torch.Tensor] = None) -> torch.Tensor:
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        else:
            lora = (self.lora_dropout(x) @ dW) * self.scaling
            return pretrained + lora


class LoRAQKVLinear(LoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `lit_gpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(LoRALinear, self).__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            #self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r * sum(enable_lora), in_features)))  # (4, 128)
            enable_q, enable_k, enable_v = enable_lora
            self.kv_embd_size = self.linear.in_features // (n_head // n_query_groups)
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                self.linear.in_features * enable_q,
                self.kv_embd_size * enable_k,
                self.kv_embd_size * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            #self.lora_B = nn.Parameter(self.linear.weight.new_zeros(sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Compute the indices
            # Indices are needed to properly pad weight updates with zeros. If we want to fine-tune queries and values,
            # but not keys, then the weights update should be:
            #
            # [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
            #  [....................................],
            #  [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            #      ↑              ↑            ↑
            # ________________________________________
            # | query         | key       | value    |
            # ----------------------------------------
            self.lora_ind = []
            if enable_q:
                self.lora_ind.extend(range(0, self.linear.in_features))
            if enable_k:
                self.lora_ind.extend(range(self.linear.in_features, self.linear.in_features + self.kv_embd_size))
            if enable_v:
                self.lora_ind.extend(range(self.linear.in_features + self.kv_embd_size, self.linear.out_features))
            self.reset_parameters()

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)
        # Note: double transpose (in the beginning and in the end) is basically a guard for two-dimensional tensors
        # for example when we want to merge/unmerge LoRA weights and pretrained weights
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.linear.out_features))  # (64, 64, 384)
        result = result.view(-1, self.linear.out_features)  # (4096, 384)
        result = result.index_copy(
            1, torch.tensor(self.lora_ind, device=result.device), x.reshape(-1, sum(self.qkv_shapes))
        )  # (4096, 256)
        return result.view((*x.shape[:-1], self.linear.out_features)).transpose(0, 1)  # (64, 64, 384)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `lit_gpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.n_head == self.n_query_groups:
            return F.conv1d(input, weight, groups=sum(self.enable_lora))  # (B, C_output, T)

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)], dim=1  # (B, C_output', T)
        )  # (B, C_output, T)

    def merge(self, dQ: torch.Tensor=None, dK: torch.Tensor=None, dV: torch.Tensor=None):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""

        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        # TODO (iusername): test weight merging
        #raise NotImplementedError("(iusername) weight merging not implemented")
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            if dQ is None and dK is None and dV is None:
                delta_w = self.conv1d(
                    self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                    self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
                ).squeeze(
                    0
                )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
                # W = W + delta_W (merge)
                self.linear.weight.data += self.zero_pad(delta_w * self.scaling)  # (256, 128) after zero_pad (384, 128)
                self.merged = True
            elif dQ is not None and dK is not None and dV is not None:
                #print("dQ", dQ.shape)
                #print("dK", dK.shape)
                #print("dV", dV.shape)
                delta_w = torch.concatenate((dQ, dK, dV),  1)
                #print("delta_W", delta_w.shape)
                #print("weight", self.linear.weight.data.shape)
                #print(delta_w.shape)
                #print(self.linear.weight.data.shape)
                self.linear.weight.data += delta_w.T * self.scaling  # (256, 128) after zero_pad (384, 128)
                self.merged = True
            else:
                raise NotImplementedError("all dQ, dK, dV have to be passed, padding not implemented")

    def forward(self, x: torch.Tensor, dQ: torch.Tensor=None, dK:torch.Tensor=None, dV: torch.Tensor=None) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """
        # TODO (iusername): cleanup coments

        # Let's assume that: 
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        #print("x", x.shape)
        pretrained = self.linear(x)
        #print("pretrained", pretrained.shape)
        if dQ is not None and dK is not None and dV is not None:
            # TODO (iusername): implement this as a single multiplication
            x_d = self.lora_dropout(x)
            lora = torch.concatenate((x_d@dQ, x_d@dK, x_d@dV), 2) * self.scaling
            return pretrained + lora
        else:
            return pretrained


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


@dataclass
class Config(BaseConfig):
    """
    Args:
        tensor_lora: whether to use LoRA for tensors or the OG matrix version
        joint_heads: when using tensor lora, whether to jointly parametrize heads
        joint_layers: when using tensor lora, whether to jointly parametrize layers
        joint_qk_vp: when using tensor lora, whether to jointly parametrize q/k and v/p matrices
        joint_qkvp: when using tensor lora, whether to jointly parametrize q/k/v/p matrices
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        to_*: either apply LoRA to the specified weights or not
    """

    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    to_query: bool = False
    to_key: bool = False
    to_value: bool = False
    to_projection: bool = False
    to_mlp: bool = False
    to_head: bool = False
    tensor_lora: bool = True
    joint_heads: bool = True
    joint_layers: bool = True
    joint_qk_vp: bool = False
    joint_qkvp: bool = False
    init_scale: float = 1.0

    @property
    def mlp_class(self) -> Type:
        return getattr(lit_gpt.lora, self._mlp_class)


class GPT(BaseModel):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = LoRALinear(
            config.n_embd,
            config.padded_vocab_size,
            bias=False,
            r=(config.r if config.to_head else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None
        if not config.tensor_lora:
            print("Using original matrix low rank adapters")
            # original matrix low rank adapters
            # fine tuning all of Q, K, V, P
            self.lora_A = nn.Parameter(torch.empty((config.n_layer, 4, config.n_embd, config.r)))
            self.lora_B = nn.Parameter(torch.zeros((config.n_layer, 4, config.r, config.n_embd)))
            #print(f"lora_A: {self.lora_A.shape}, lora_B: {self.lora_B.shape}")
            for layer in range(config.n_layer):
                nn.init.kaiming_uniform_(self.lora_A[layer], a=math.sqrt(5)*config.init_scale)
        else:
            head_dim = config.n_embd // config.n_head
            if config.joint_heads:
                A_shape = (config.n_embd, config.r)
                B_shape = (config.r, head_dim)
            else:
                head_dim = config.n_embd
                A_shape = (config.n_embd, config.r)
                B_shape = (config.n_heads, config.r, head_dim)
            C_shape = (config.r, )
            if not config.joint_qkvp:
                A_shape = (4, ) + A_shape
                B_shape = (4, ) + B_shape 
                C_shape = (4, ) + C_shape
            if config.joint_layers:
                self.lora_C_l = nn.Parameter(torch.zeros((config.n_layer,)+ C_shape))
                nn.init.kaiming_uniform_(self.lora_C_l, a=math.sqrt(5)*config.init_scale)
            else:
                A_shape = (config.n_layer, ) + A_shape
                B_shape = (config.n_layer, ) + B_shape
                C_shape = (config.n_layer, ) + C_shape
            if config.joint_qk_vp:
                raise NotImplementedError("joint_qk_vp not implemented")
            if config.joint_heads:
                self.lora_C_h = nn.Parameter(torch.zeros((config.n_head,) + C_shape))
                nn.init.kaiming_uniform_(self.lora_C_h, a=math.sqrt(5)*config.init_scale)
            if self.config.joint_qkvp:
                self.lora_C_m = nn.Parameter(torch.zeros((4,)+ C_shape))
                nn.init.kaiming_uniform_(self.lora_C_m, a=math.sqrt(5)*config.init_scale)

            self.lora_A = nn.Parameter(torch.empty(A_shape))
            self.lora_B = nn.Parameter(torch.zeros(B_shape))
            if len(A_shape) == 2:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            elif len(A_shape) == 3:
                for i in range(A_shape[0]):
                    nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            elif len(A_shape) == 4:
                for i in range(A_shape[0]):
                    for j in range(A_shape[1]):
                        nn.init.kaiming_uniform_(self.lora_A[i, j], a=math.sqrt(5))
            #print(f"lora_A: {self.lora_A.shape}, lora_B: {self.lora_B.shape}")

    def forward(
        self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, lm_head_chunk_size: int = 0
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # (iusername) compute adapter weights for each layer on the fly
        for block_idx, block in enumerate(self.transformer.h):
            if not self.config.tensor_lora:
                dQ = self.lora_A[block_idx, 0] @ self.lora_B[block_idx, 0]
                dK = self.lora_A[block_idx, 1] @ self.lora_B[block_idx, 1]
                dV = self.lora_A[block_idx, 2] @ self.lora_B[block_idx, 2]
                dP = self.lora_A[block_idx, 3] @ self.lora_B[block_idx, 3]
                #print(f"sum of B is {torch.sum(self.lora_B[block_idx, 0])}")
                #print(dQ.shape, dK.shape, dV.shape, dP.shape)
            else:
                if self.config.joint_layers and self.config.joint_heads and self.config.joint_qkvp:
                    dQ = torch.cat([self.lora_A@torch.diag(self.lora_C_h[head_idx]*self.lora_C_m[0]*self.lora_C_l[block_idx])@self.lora_B for head_idx in range(self.config.n_head)], dim=1)
                    assert(dQ.shape == (self.config.n_embd, self.config.n_embd))
                    dK = torch.cat([self.lora_A @torch.diag(self.lora_C_h[head_idx]*self.lora_C_m[1]*self.lora_C_l[block_idx])@ self.lora_B for head_idx in range(self.config.n_head)], dim=1)
                    dV = torch.cat([self.lora_A @torch.diag(self.lora_C_h[head_idx]*self.lora_C_m[2]*self.lora_C_l[block_idx])@self.lora_B for head_idx in range(self.config.n_head)], dim=1) 
                    dP = torch.cat([self.lora_A @torch.diag(self.lora_C_h[head_idx]*self.lora_C_m[3]*self.lora_C_l[block_idx])@self.lora_B for head_idx in range(self.config.n_head)], dim=1)
                    #if torch.sum(torch.abs(self.lora_B))>0:
                    #print(f"sum of B: {torch.sum(self.lora_B)}")
                    #print(f"sum of C_m: {torch.sum(self.lora_C_m)}, sum of C_h: {torch.sum(self.lora_C_h)}, sum of C_l: {torch.sum(self.lora_C_l)}")
                    #assert(self.lora_C_m.requires_grad == True)
                    #assert(self.lora_C_h.requires_grad == True)
                    #assert(self.lora_C_l.requires_grad == True)
                    #assert(dQ.shape == (self.config.n_embd, self.config.n_embd))
                    #assert(dK.shape == (self.config.n_embd, self.config.n_embd))
                    #assert(dV.shape == (self.config.n_embd, self.config.n_embd))
                    #assert(dP.shape == (self.config.n_embd, self.config.n_embd))
                elif (not self.config.joint_layers) and self.config.joint_heads and self.config.joint_qkvp:
                    #print(f"shapes A {self.lora_A.shape}, B {self.lora_B.shape}, C_h {self.lora_C_h.shape}, C_m {self.lora_C_m.shape}")
                    dQ = torch.cat([self.lora_A[block_idx] @ torch.diag(self.lora_C_h[head_idx,block_idx]*self.lora_C_m[0, block_idx])@ self.lora_B[block_idx] for head_idx in range(self.config.n_head)], dim=1)
                    assert(dQ.shape == (self.config.n_embd, self.config.n_embd))
                    dK = torch.cat([self.lora_A[block_idx] @ torch.diag(self.lora_C_h[head_idx,block_idx]*self.lora_C_m[1, block_idx])@ self.lora_B[block_idx] for head_idx in range(self.config.n_head)], dim=1)
                    dV = torch.cat([self.lora_A[block_idx] @ torch.diag(self.lora_C_h[head_idx,block_idx]*self.lora_C_m[2, block_idx])@ self.lora_B[block_idx] for head_idx in range(self.config.n_head)], dim=1) 
                    dP = torch.cat([self.lora_A[block_idx] @ torch.diag(self.lora_C_h[head_idx,block_idx]*self.lora_C_m[3, block_idx])@ self.lora_B[block_idx] for head_idx in range(self.config.n_head)], dim=1)
                elif self.config.joint_layers and self.config.joint_heads and (not self.config.joint_qkvp):
                    #print(f"shapes A {self.lora_A.shape}, B {self.lora_B.shape}, C_h {self.lora_C_h.shape}, C_l {self.lora_C_l.shape}")
                    dQ = torch.cat([self.lora_A[0]@torch.diag(self.lora_C_h[head_idx, 0]*self.lora_C_l[block_idx, 0])@self.lora_B[0] for head_idx in range(self.config.n_head)], dim=1)
                    assert(dQ.shape == (self.config.n_embd, self.config.n_embd))
                    dK = torch.cat([self.lora_A[1] @torch.diag(self.lora_C_h[head_idx, 1]*self.lora_C_l[block_idx, 1])@ self.lora_B[1] for head_idx in range(self.config.n_head)], dim=1)
                    dV = torch.cat([self.lora_A[2] @torch.diag(self.lora_C_h[head_idx, 2]*self.lora_C_l[block_idx, 2])@self.lora_B[2] for head_idx in range(self.config.n_head)], dim=1) 
                    dP = torch.cat([self.lora_A[3] @torch.diag(self.lora_C_h[head_idx, 3]*self.lora_C_l[block_idx, 3])@self.lora_B[3] for head_idx in range(self.config.n_head)], dim=1)
                else:
                    raise NotImplementedError("Implemented joint heads and either joint qkvp or joint layers or the three of them at the same time. All other combinations are not implemented yet.")
            x = block(x, cos, sin, mask, input_pos, dQ, dK, dV, dP)
        x = self.transformer.ln_f(x)
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        return self.lm_head(x)  # (B, T, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {"lm_head.weight": "lm_head.linear.weight"}
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(BaseBlock):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        dQ: Optional[torch.Tensor] = None,
        dK: Optional[torch.Tensor] = None,
        dV: Optional[torch.Tensor] = None,
        dP: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos, dQ, dK, dV, dP)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x

class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(self, config: Config) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = LoRAQKVLinear(
            in_features=config.n_embd,
            out_features=shape,
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            enable_lora=(config.to_query, config.to_key, config.to_value),
            bias=config.bias,
            # for MQA/GQA support
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )
        # output projection
        self.proj = LoRALinear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_projection else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "attn.weight": "attn.linear.weight",
            "attn.bias": "attn.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        dQ: Optional[torch.Tensor] = None,
        dK: Optional[torch.Tensor] = None,
        dV: Optional[torch.Tensor] = None,
        dP: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x, dQ, dK, dV)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # repeat k and v if necessary
        if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
            # for MHA this is a no-op
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y, dP)


class GptNeoxMLP(lit_gpt.model.GptNeoxMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc.weight": "fc.linear.weight",
            "fc.bias": "fc.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LLaMAMLP(lit_gpt.model.LLaMAMLP):
    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.fc_1 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.fc_2 = LoRALinear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.proj = LoRALinear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def merge_lora_weights(model: GPT) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        for block_idx, block in enumerate(model.transformer.h):
            if not model.config.tensor_lora:
                dQ = model.lora_A[block_idx, 0] @ model.lora_B[block_idx, 0]
                dK = model.lora_A[block_idx, 1] @ model.lora_B[block_idx, 1]
                dV = model.lora_A[block_idx, 2] @ model.lora_B[block_idx, 2]
                dP = model.lora_A[block_idx, 3] @ model.lora_B[block_idx, 3]
            else:
                if model.config.joint_layers and model.config.joint_heads and model.config.joint_qkvp:
                    dQ = torch.cat([model.lora_A@torch.diag(model.lora_C_h[head_idx]*model.lora_C_m[0]*model.lora_C_l[block_idx])@model.lora_B for head_idx in range(model.config.n_head)], dim=1)
                    assert(dQ.shape == (model.config.n_embd, model.config.n_embd))
                    dK = torch.cat([model.lora_A @torch.diag(model.lora_C_h[head_idx]*model.lora_C_m[1]*model.lora_C_l[block_idx])@ model.lora_B for head_idx in range(model.config.n_head)], dim=1)
                    dV = torch.cat([model.lora_A @torch.diag(model.lora_C_h[head_idx]*model.lora_C_m[2]*model.lora_C_l[block_idx])@model.lora_B for head_idx in range(model.config.n_head)], dim=1) 
                    dP = torch.cat([model.lora_A @torch.diag(model.lora_C_h[head_idx]*model.lora_C_m[3]*model.lora_C_l[block_idx])@model.lora_B for head_idx in range(model.config.n_head)], dim=1)
                elif (not model.config.joint_layers) and model.config.joint_heads and model.config.joint_qkvp:
                    dQ = torch.cat([model.lora_A[block_idx] @ torch.diag(model.lora_C_h[block_idx,head_idx]*model.lora_C_m[block_idx,0])@ model.lora_B[block_idx] for head_idx in range(model.config.n_head)], dim=1)
                    assert(dQ.shape == (model.config.n_embd, model.config.n_embd))
                    dK = torch.cat([model.lora_A[block_idx] @ torch.diag(model.lora_C_h[block_idx, head_idx]*model.lora_C_m[block_idx,1])@ model.lora_B[block_idx] for head_idx in range(model.config.n_head)], dim=1)
                    dV = torch.cat([model.lora_A[block_idx] @ torch.diag(model.lora_C_h[block_idx,head_idx]*model.lora_C_m[block_idx,2])@ model.lora_B[block_idx] for head_idx in range(model.config.n_head)], dim=1) 
                    dP = torch.cat([model.lora_A[block_idx] @ torch.diag(model.lora_C_h[block_idx, head_idx]*model.lora_C_m[block_idx,3])@ model.lora_B[block_idx] for head_idx in range(model.config.n_head)], dim=1)
                elif model.config.joint_layers and model.config.joint_heads and (not model.config.joint_qkvp):
                    dQ = torch.cat([model.lora_A[0]@torch.diag(model.lora_C_h[0,head_idx]*model.lora_C_l[block_idx, 0])@model.lora_B[0] for head_idx in range(model.config.n_head)], dim=1)
                    assert(dQ.shape == (model.config.n_embd, model.config.n_embd))
                    dK = torch.cat([model.lora_A[1] @torch.diag(model.lora_C_h[1, head_idx]*model.lora_C_l[block_idx, 1])@ model.lora_B[1] for head_idx in range(model.config.n_head)], dim=1)
                    dV = torch.cat([model.lora_A[2] @torch.diag(model.lora_C_h[2, head_idx]*model.lora_C_l[block_idx, 2])@model.lora_B[2] for head_idx in range(model.config.n_head)], dim=1) 
                    dP = torch.cat([model.lora_A[3] @torch.diag(model.lora_C_h[3, head_idx]*model.lora_C_l[block_idx, 3])@model.lora_B[3] for head_idx in range(model.config.n_head)], dim=1)
            block.attn.attn.merge(dQ, dK, dV)
            block.attn.proj.merge(dP)
