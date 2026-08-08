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
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_quantization_config,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties

from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import NARAConfig
from .embedding import NARAMapper, build_embedding
from .gptq import dispatch_gptq
from .layer import Conv2d, NARALayer, dispatch_default
from .tp_layer import dispatch_megatron


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


class NARAModel(BaseTuner):
    """
    Creates a NaRA (Noise-aware Rank Adaptation) model from a pretrained transformers model.

    NaRA is LoRA with a noise-conditioned mixing matrix between the two factors:

        h(x) = W_0 x + B . C(lambda) . A x . scaling
        C(lambda) = I_r + c_scale * F_phi(e(lambda))

    `A` and `B` are ordinary per-layer LoRA factors, held in this tuner's layers. `F_phi`
    (`self.model.lora_mapper`) and the lambda embedding (`self.model.lora_phi`) are single
    globally shared modules hung off the base model, so `C` is computed once per forward
    and broadcast to every adapted layer.

    **How lambda reaches the adapter.** Nothing in the training loop or the diffusion
    sampler knows this adapter exists. `forward` reads `input_ids` / `attention_mask` off
    the call, `_enable_peft_forward_hooks` catches the `response_mask` kwarg (which PEFT
    strips before the base model is called), and the resulting `C` is injected into every
    `NARALayer` by a forward pre-hook. This replaces the reference implementation's
    `model.set_context_state(...)` push, which required edits to both the loss loop and the
    256-step sampler and left layers reusing a stale `C` if anyone forgot to call it.

    **Why the globals are named `lora_*`.** `get_peft_model_state_dict` selects adapter
    tensors by the `lora_` substring, so a differently named parameter -- or a registered
    buffer, which the reference uses for its Fourier projection -- is dropped from the
    checkpoint and silently redrawn on reload.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`NARAConfig`]): The configuration of the NaRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The NaRA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import NARAConfig, get_peft_model

        >>> config = NARAConfig(
        ...     task_type="CAUSAL_LM",
        ...     r=32,
        ...     lora_alpha=32,
        ...     target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
        ...     lora_dropout=0.05,
        ...     c_scale=0.1,
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        >>> nara_model = get_peft_model(model, config)
        >>> out = nara_model(input_ids=input_ids, attention_mask=attention_mask, response_mask=response_mask)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`NARAConfig`]): The configuration of the NaRA model.
    """

    prefix: str = "lora_"

    # LLaDA mask token id (see llada/run_training.sh: --mask_id 126336)
    MASK_TOKEN_ID: int = 126336

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        # `C` for the forward currently in flight, read by the pre-hook on every adapted layer.
        self.c_matrix = None
        # Set by `_enable_peft_forward_hooks` for the duration of one forward; see `forward`.
        self._response_mask = None

    # ------------------------------------------------------------------ #
    # NaRA: the shared mapper, and computing C                            #
    # ------------------------------------------------------------------ #

    def inject_adapter(self, model: nn.Module, adapter_name: str):
        super().inject_adapter(model, adapter_name)
        self._create_nara_modules(adapter_name)
        # The base implementation marks trainables before the shared modules exist, so
        # redo it now that they do. It is idempotent.
        self._mark_only_adapters_as_trainable(self.model)

    def _create_nara_modules(self, adapter_name: str) -> None:
        """Build the globally shared parts of the adapter on the base model.

        Deliberately built in float32 even when the base model is bf16: the mapper is tiny
        (its cost is one forward per step, not per layer), and `C` feeds a sin/cos embedding
        and an `r x r` product where the extra precision is cheap insurance. `_apply_c` casts
        down to the activation dtype at the point of use.
        """
        config = self.peft_config[adapter_name]
        if getattr(self.model, "lora_mapper", None) is not None or getattr(self.model, "lora_phi", None) is not None:
            # A second adapter shares the first one's mapper; NaRA's C is global by construction.
            return

        if config.input_mode == "constant":
            # No noise input at all: C = I + c_scale * P for a single learnable P. Zero-init keeps
            # the "identical to LoRA at step 0" property that `init_c="zero_last"` gives the mapper.
            self.model.lora_constant_c = nn.Parameter(torch.zeros(config.r, config.r, dtype=torch.float32))
            return

        embedding, embedding_width = build_embedding(config.embedding_type, config.embedding_dim)
        self.model.lora_phi = embedding.to(dtype=torch.float32)
        self.model.lora_mapper = NARAMapper(
            r=config.r,
            input_dim=embedding_width,
            fnn_hidden_size_1=config.fnn_hidden_size_1,
            fnn_hidden_size_2=config.fnn_hidden_size_2,
            init_c=config.init_c,
        ).to(dtype=torch.float32)

    def _nara_config(self) -> NARAConfig:
        """The config driving the shared modules.

        NaRA's mapper is global, so with multiple adapters loaded there is still only one
        `C`; the first active adapter's config owns it.
        """
        return self.peft_config[self.active_adapters[0]]

    def _lambda(self, input_ids, attention_mask=None, response_mask=None) -> torch.Tensor:
        """The masked proportion of the answer, per example (`[batch]`).

        Masks only ever land in the response, so the numerator is answer-only either way.
        The denominator is what has to be right: dividing by prompt + answer yields
        `t * L_ans / (L_prompt + L_ans)` rather than `t`, which can never reach 1.0 (the
        fully-masked regime this adapter most needs to represent), compresses by a factor
        that varies with prompt length, and shifts between training and generation.
        """
        config = self._nara_config()
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        is_mask = input_ids == self.MASK_TOKEN_ID

        valid = None
        if config.lambda_source == "response":
            if response_mask is None:
                warnings.warn(
                    "NaRA is configured with `lambda_source='response'` but no `response_mask` was passed to "
                    "the forward; falling back to a prompt+answer denominator, which caps lambda below 1.0 by "
                    "a per-example factor. Pass `response_mask=...` (a boolean [batch, seq] tensor marking the "
                    "answer) to the model call, or set `lambda_source='non_padding'` to choose this deliberately."
                )
            else:
                valid = response_mask.bool()
                if valid.dim() == 1:
                    valid = valid[None, :]
        if valid is None:
            valid = torch.ones_like(is_mask)
        if attention_mask is not None:
            am = attention_mask.bool()
            valid = valid & (am[None, :] if am.dim() == 1 else am)

        if config.pool_lambda:
            # One C for the whole batch: exact only when every row shares a lambda.
            return ((is_mask & valid).sum() / valid.sum().clamp(min=1)).reshape(1)
        return (is_mask & valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)

    def _compute_c(self, input_ids=None, attention_mask=None, response_mask=None) -> Optional[torch.Tensor]:
        """`C(lambda)`, shaped `[r, r]` when lambda is pooled and `[batch, r, r]` otherwise."""
        config = self._nara_config()
        r = config.r

        if config.input_mode == "constant":
            p = self.model.lora_constant_c
            return torch.eye(r, dtype=p.dtype, device=p.device) + config.c_scale * p

        if input_ids is None:
            # Nothing to derive a noise level from (e.g. an `inputs_embeds`-only call).
            # `None` means "identity" downstream, i.e. this forward behaves as plain LoRA.
            return None

        param = next(self.model.lora_mapper.parameters())
        lam = self._lambda(input_ids, attention_mask, response_mask).to(device=param.device, dtype=param.dtype)

        # The reference reshapes the mapper output with `.view(R, R)`, which silently assumes
        # a batch of one -- which is why every upstream config ships `batch_size: 1`. Keeping
        # the batch dimension is what lets rows with different masking probabilities coexist.
        c = self.model.lora_mapper(self.model.lora_phi(lam)).view(-1, r, r)
        c = torch.eye(r, dtype=c.dtype, device=c.device) + config.c_scale * c
        return c[0] if config.pool_lambda else c

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask", None)
        # `response_mask` is stripped from kwargs by `PeftModel.forward` before we are called
        # (it is in `special_peft_forward_args`), so it arrives via `_enable_peft_forward_hooks`,
        # which runs first and stashes it on `self`.
        response_mask = getattr(self, "_response_mask", None)
        self.c_matrix = self._compute_c(input_ids, attention_mask, response_mask)
        return self.model.forward(*args, **kwargs)

    def _c_pre_forward_hook(self, target, args, kwargs):
        # Reads `self.c_matrix` at call time, so it always sees the value `forward` just computed
        # -- including on a gradient-checkpointing recompute, where the hook fires a second time.
        kwargs["adapter_c"] = self.c_matrix
        return args, kwargs

    def _check_new_adapter_config(self, config: NARAConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        return check_target_module_exists(lora_config, key)

    def _prepare_model(self, peft_config: NARAConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of NARALayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, NARALayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
            )
            layer = target
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
            layer = new_module

        # `update_layer` has set the PEFT-standard `lora_alpha / r`. NaRA's own `scale_ab` is an
        # extra factor on top, kept so the reference's numerics stay reachable (it scales by
        # `scale_ab` alone and ignores `lora_alpha`); it defaults to 1.0 and is then a no-op.
        if lora_config.scale_ab != 1.0:
            layer.scaling[adapter_name] *= lora_config.scale_ab

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = child.qweight if hasattr(child, "qweight") else child.weight
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            config = self.peft_config[active_adapter]
            if not config.train_a:
                for n, p in model.named_parameters():
                    if f"lora_A.{active_adapter}" in n or f"lora_embedding_A.{active_adapter}" in n:
                        p.requires_grad = False
            if not config.train_b:
                for n, p in model.named_parameters():
                    if f"lora_B.{active_adapter}" in n or f"lora_embedding_B.{active_adapter}" in n:
                        p.requires_grad = False

        # The mapper and the lambda embedding are shared across adapters, so the first active
        # adapter's config governs them (see `_nara_config`).
        train_mapper = self._nara_config().train_mapper
        for attr in ("lora_mapper", "lora_constant_c", "lora_phi"):
            module_or_param = getattr(self.model, attr, None)
            if module_or_param is None:
                continue
            params = [module_or_param] if isinstance(module_or_param, nn.Parameter) else module_or_param.parameters()
            for p in params:
                p.requires_grad = train_mapper
        if getattr(self.model, "lora_phi", None) is not None and self._nara_config().embedding_type == "fourier":
            # The Fourier projection is a fixed random feature map, drawn once. It is an
            # `nn.Parameter` only so that `lora_` in its name gets it into the checkpoint --
            # redrawing it on reload would reinterpret every noise level.
            for p in self.model.lora_phi.parameters():
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, NARALayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend([dispatch_aqlm, dispatch_awq, dispatch_gptq, dispatch_megatron, dispatch_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, NARALayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        """Broadcast this forward's `C` to every adapted layer.

        Two things are stitched together here, because neither entry point sees everything:
        this context receives `PeftModel.forward`'s `**kwargs` (so it is the only place
        `response_mask` is visible, before PEFT strips it), while `forward` receives
        `input_ids`/`attention_mask` (named parameters, which never reach here). The mask is
        therefore stashed on `self` and picked up by `forward`, which runs inside this context.
        """
        if kwargs.get("adapter_names", None) is not None:
            raise ValueError("NaRA does not support mixed adapter batches (`adapter_names`).")

        # Scoped to the context, so a forward taken outside it cannot silently reuse a stale mask.
        self._response_mask = kwargs.get("response_mask", None)

        hook_handles = []
        for module in self.modules():
            if isinstance(module, NARALayer):
                handle = module.register_forward_pre_hook(self._c_pre_forward_hook, with_kwargs=True)
                hook_handles.append(handle)

        try:
            yield
        finally:
            self._response_mask = None
            for handle in hook_handles:
                handle.remove()

    def _check_merge_allowed(self):
        """NaRA cannot be merged; see `NARALayer._no_merge`."""
        raise NotImplementedError(
            "NaRA adapters cannot be merged into the base weights: `C` depends on the noise level lambda, so "
            "`B C(lambda) A` is not a fixed delta. A merged model would only be equivalent at a single lambda. "
            "Use `unload()` to drop the adapter, or keep it unmerged."
        )

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        # The shared mapper / embedding live on the base model, not inside the replaced layers,
        # so unloading has to drop them explicitly or they linger as dead parameters.
        for attr in ("lora_mapper", "lora_phi", "lora_constant_c"):
            if getattr(self.model, attr, None) is not None:
                delattr(self.model, attr)

        return self.model

    def add_weighted_adapter(
        self,
        adapters,
        weights,
        adapter_name,
        combination_type="svd",
        svd_rank=None,
        svd_clamp=None,
        svd_full_matrices=True,
        svd_driver=None,
        density=None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """
        # Combining NaRA adapters would have to combine their mappers too, and the mapper is a
        # nonlinear function of lambda -- there is no weighted sum of `B C(lambda) A` products
        # that is itself a NaRA adapter. Refused rather than silently combining A/B only.
        raise NotImplementedError("NaRA does not support `add_weighted_adapter`.")

        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].r for adapter in adapters]
        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] == str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] == set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, NARALayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for LoRA A and B layers.
        valid_weights = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            valid_weights.append(math.sqrt(weight * target.scaling[adapter]))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        valid_weights = torch.tensor(valid_weights).to(lora_A_deltas[0].device)
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights)
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights, density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights, density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights, density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        # The per-layer A/B could be deleted, but the shared mapper cannot: it is owned by
        # whichever adapter was active first (`_nara_config`), so deleting that one would leave
        # the survivors conditioned by a mapper whose config no longer exists.
        raise NotImplementedError("NaRA does not support `delete_adapter`; its mapper is shared across adapters.")

        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, NARALayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
