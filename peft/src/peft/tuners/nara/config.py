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

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class LoftQConfig:
    """
    This is the sub-configuration class to store the configuration of a [`NARAModel`].

    Args:
        bits_pattern (`dict`): The mapping from layer names or regexp expression to bits which are different from the
            default bits specified by `bits`. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 2`}.
        bits (`int`): Quantization bits for LoftQ.
        iter (`int`): Alternating iterations for LoftQ.
        fake (`bool`): True: use fp16/fp32; used for first time to save weights. False: use bitsandbytes 4bit linear
            models. weights can't be saved. Recommend to set to True, save the weights and load the saved weights in 4
            bits.
    """

    loftq_bits: int = field(default=4, metadata={"help": "Quantization bits for LoftQ"})
    loftq_iter: int = field(default=1, metadata={"help": "Alternating iterations for LoftQ"})


@dataclass
class NARAConfig(PeftConfig):
    """
    Configuration for NaRA (Noise-aware Rank Adaptation).

    NaRA is LoRA with a noise-conditioned mixing matrix inserted between the two factors:

        h(x) = W_0 x + B . C(lambda) . A x . scaling
        C(lambda) = I_r + c_scale * F_phi(e(lambda))     # F_phi output reshaped to r x r

    `A` and `B` are ordinary per-layer LoRA factors. `F_phi` is one small MLP shared by
    every adapted matrix in every layer, so `C` is computed once per forward and broadcast.
    Its output layer is zero-initialised (`init_c="zero_last"`), so `C == I_r` at step 0
    and an untrained NaRA adapter is numerically identical to plain LoRA.

    **What lambda is.** The masked proportion of the *answer*: masked answer tokens over
    answer tokens. Its denominator is not recoverable from `input_ids` (the response holds
    unmasked tokens too), so the caller passes a `response_mask` kwarg to the forward;
    PEFT strips it before the base model is called. See `lambda_source`.

    **Scaling convention.** The delta is scaled by `lora_alpha / r` (PEFT's convention,
    shared with `lora`/`lorta`/`nalorta` so the four are comparable under one sweep),
    times `scale_ab`. The reference implementation instead scales by `scale_ab` alone and
    leaves `lora_alpha` unused, so its reported learning rates do not transfer: to
    reproduce upstream numerics exactly, set `lora_alpha = r`. Expect to sweep the
    learning rate rather than trusting either repo's number.

    **Batching.** `C` is computed per example (`[batch, r, r]`) by default, because rows
    of a training batch genuinely carry different masking probabilities. `pool_lambda=True`
    selects the cheaper single-`C`-per-batch behaviour, which is exact only when every row
    has the same lambda.

    Args:
        r (`int`):
            Lora attention dimension (the "rank"). Written `r_ab` in the reference
            implementation; `C` is `r x r`, so the mapper's output layer grows as `r^2`.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
        lora_dropout (`float`):
            The dropout probability for Lora layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for LoRA. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        use_rslora (`bool`):
            When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a> which
            sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it was proven to work better.
            Otherwise, it will use the original default value of `lora_alpha/r`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_lora_weights (`bool` | `Literal["gaussian", "loftq"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft. Passing 'gaussian' results in Gaussian
            initialization scaled by the LoRA rank for linear and layers. Setting the initialization to False leads to
            completely random initialization and is discouraged. Pass `'loftq'` to use LoftQ initialization.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `lora_alpha`.
        megatron_config (`Optional[dict]`):
            The TransformerConfig arguments for Megatron. It is used to create LoRA's parallel linear layer. You can
            get it like this, `core_transformer_config_from_args(get_args())`, these two functions being from Megatron.
            The arguments will be used to initialize the TransformerConfig of Megatron. You need to specify this
            parameter when you want to apply LoRA to the ColumnParallelLinear and RowParallelLinear layers of megatron.
        megatron_core (`Optional[str]`):
            The core module from Megatron to use, defaults to `"megatron.core"`.
        loftq_config (`Optional[LoftQConfig]`):
            The configuration of LoftQ. If this is not None, then LoftQ will be used to quantize the backbone weights
            and initialize Lora layers. Also pass `init_lora_weights='loftq'`. Note that you should not pass a
            quantized model in this case, as LoftQ will quantize the model itself.
        use_dora (`bool`):
            Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the weights
            into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the magnitude is
            handled by a separate learnable parameter. This can improve the performance of LoRA especially at low
            ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger overhead than pure
            LoRA, so it is recommended to merge weights for inference. For more information, see
            https://arxiv.org/abs/2402.09353.
        layer_replication(`List[Tuple[int, int]]`):
            Build a new stack of layers by stacking the original model layers according to the ranges specified. This
            allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will
            all have separate LoRA adapters attached to them.

    NaRA-specific args:
        c_scale (`float`):
            Weight on the mapper's contribution: `C = I_r + c_scale * F_phi(e(lambda))`. The
            reference configs use `0.1`.
        scale_ab (`float`):
            Extra multiplier on the delta, on top of PEFT's `lora_alpha / r`. Defaults to `1.0`.
        init_c (`str`):
            Mapper initialisation. `"zero_last"` (default) zeroes only the output layer, so
            `C == I_r` and the adapter starts identical to LoRA. `"kaiming_uniform_m"` and
            `"zero_all"` are the reference's other options and give up that property.
        fnn_hidden_size_1 / fnn_hidden_size_2 (`int`):
            Widths of the mapper's two hidden layers (reference: 256, 512).
        embedding_type (`str`):
            How the scalar lambda is embedded before the mapper: `"fourier"` (default, frozen
            Gaussian Fourier features), `"mlp"` (learnable) or `"raw"` (the scalar itself).
        embedding_dim (`int`):
            Width of that embedding; ignored for `"raw"`, and must be even for `"fourier"`.
        input_mode (`str`):
            `"nl"` (default) conditions on the noise level lambda. `"constant"` replaces the
            mapper with a single learnable `r x r` parameter -- a useful ablation that isolates
            "an extra `r x r` factor" from "an extra `r x r` factor that *knows the noise
            level*". `"nd"` and `"both"` (local mask density) are not implemented yet.
        lambda_source (`str`):
            Denominator of lambda. `"response"` (default) divides by the answer tokens and
            requires a `response_mask` kwarg on the forward. `"non_padding"` divides by every
            non-padding token, prompt included; it is offered only for comparison against
            NA-LoRTA's original behaviour, and caps lambda below 1.0 by a per-example factor.
            Note the reference implementation trains on the *sampled* masking probability and
            evaluates on the realised response-only fraction; this port uses the realised
            answer fraction at both ends, on purpose.
        pool_lambda (`bool`):
            Pool lambda to one scalar over the batch, giving a single `[r, r]` `C`, instead of
            the default per-example `[batch, r, r]`. Cheaper, and exact only when every row
            shares a lambda.
        train_a / train_b / train_mapper (`bool`):
            Ablation switches; each freezes the corresponding parameters.
        mapper_groups (`Optional[Dict[str, List[str]]]`):
            Per-group mappers. Not implemented yet; setting it raises.
        training_stage (`int`):
            Reference two-stage schedule (stage 1 freezes the mapper at `C == I`). Not
            implemented yet; only `2` is accepted.
        density_radius (`Optional[int]`):
            Window radius for the local mask density used by `input_mode="nd"/"both"`. Not
            implemented yet.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})

    # --- NaRA: the noise-conditioned mixing matrix C ---
    c_scale: float = field(
        default=1.0, metadata={"help": "Weight on the mapper term in `C = I_r + c_scale * F_phi(e(lambda))`"}
    )
    scale_ab: float = field(
        default=1.0,
        metadata={
            "help": "Extra multiplier on the delta, on top of PEFT's `lora_alpha / r`. Set `lora_alpha = r` "
            "to reproduce the reference implementation, which scales by `scale_ab` alone."
        },
    )
    init_c: Literal["zero_last", "kaiming_uniform_m", "zero_all"] = field(
        default="zero_last",
        metadata={
            "help": "Mapper init. 'zero_last' zeroes only the output layer so C == I_r and NaRA starts "
            "numerically identical to LoRA."
        },
    )
    fnn_hidden_size_1: int = field(default=256, metadata={"help": "Width of the mapper's first hidden layer"})
    fnn_hidden_size_2: int = field(default=512, metadata={"help": "Width of the mapper's second hidden layer"})

    # --- NaRA: how lambda is embedded ---
    embedding_type: Literal["fourier", "mlp", "raw"] = field(
        default="fourier", metadata={"help": "Scalar embedding of lambda: 'fourier' (frozen), 'mlp', or 'raw'"}
    )
    embedding_dim: int = field(
        default=64,
        metadata={"help": "Width of the lambda embedding; must be even for 'fourier', ignored for 'raw'"},
    )

    # --- NaRA: what the adapter is conditioned on ---
    input_mode: Literal["nl", "constant", "nd", "both"] = field(
        default="nl",
        metadata={
            "help": "'nl' conditions on the noise level; 'constant' uses a learnable r x r parameter with no "
            "noise input (ablation baseline). 'nd'/'both' (local mask density) are not implemented."
        },
    )
    lambda_source: Literal["response", "non_padding"] = field(
        default="response",
        metadata={
            "help": "Denominator of lambda. 'response' divides by the answer tokens and needs a `response_mask` "
            "kwarg on the forward; 'non_padding' divides by every non-padding token, prompt included."
        },
    )
    pool_lambda: bool = field(
        default=False,
        metadata={
            "help": "Pool lambda to one scalar over the batch (single [r, r] C) instead of the default "
            "per-example [batch, r, r]. Exact only when every row shares a lambda."
        },
    )

    # --- NaRA: ablation switches ---
    train_a: bool = field(default=True, metadata={"help": "Train the per-layer A factors"})
    train_b: bool = field(default=True, metadata={"help": "Train the per-layer B factors"})
    train_mapper: bool = field(
        default=True, metadata={"help": "Train the shared mapper (and the learnable parts of the embedding)"}
    )

    # --- NaRA: declared but not implemented (phase 2); present so adding them stays additive ---
    mapper_groups: Optional[dict[str, list[str]]] = field(
        default=None, metadata={"help": "Per-group mappers. Not implemented yet."}
    )
    training_stage: int = field(
        default=2, metadata={"help": "Reference two-stage schedule. Not implemented yet; only 2 is accepted."}
    )
    density_radius: Optional[int] = field(
        default=None, metadata={"help": "Window radius for local mask density. Not implemented yet."}
    )

    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses Rank-Stabilized LoRA doi.org/10.48550/arXiv.2312.03732"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool | Literal["gaussian", "loftq"] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing True (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing 'gaussian' results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to False leads to completely random initialization and is discouraged."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply LoRA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )
    # dict type is used when loading config.json
    loftq_config: Union[LoftQConfig, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` in this case."
            )
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference. For more information, "
                "see  https://arxiv.org/abs/2402.09353."
            )
        },
    )
    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.NARA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        if self.use_dora and self.megatron_config:
            raise ValueError("DoRA does not support megatron_core, please set `use_dora=False`.")

        # handle init_lora_weights and loftq_config
        if self.init_lora_weights == "loftq":
            import importlib

            if not importlib.util.find_spec("scipy"):
                raise ImportError("The required package 'scipy' is not installed. Please install it to continue.")
            if self.loftq_config is None:
                raise ValueError("`loftq_config` must be specified when `init_lora_weights` is 'loftq'.")

        # convert loftq_config to dict
        if self.loftq_config and not isinstance(self.loftq_config, dict):
            self.loftq_config = vars(self.loftq_config)

        # --- NaRA-specific validation ---
        if self.input_mode not in ("nl", "constant"):
            raise NotImplementedError(
                f"`input_mode={self.input_mode!r}` needs the local mask density, which is not implemented yet. "
                "Use 'nl' (noise level) or 'constant'."
            )
        if self.mapper_groups is not None:
            raise NotImplementedError("`mapper_groups` (per-group mappers) is not implemented yet.")
        if self.training_stage != 2:
            raise NotImplementedError(
                f"`training_stage={self.training_stage}` is not implemented yet; only stage 2 (the full method) "
                "is available. A stage-1 warm-up is equivalent to freezing the mapper from the training script."
            )
        if self.density_radius is not None:
            raise NotImplementedError("`density_radius` only applies to the unimplemented 'nd'/'both' input modes.")
        if self.use_dora:
            raise NotImplementedError("NaRA does not support DoRA.")
        if self.embedding_type == "fourier" and self.embedding_dim % 2 != 0:
            raise ValueError(
                f"`embedding_dim` must be even for `embedding_type='fourier'` (got {self.embedding_dim}); "
                "the embedding concatenates sin and cos to reach the requested width."
            )
