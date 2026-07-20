# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LoRA wrappers for interatomic potential models.

This module extends the underlying LoRA registry with wrappers for e3nn layers
used by E(3) equivariant interatomic potentials.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Literal, TypeAlias

import torch
from torch import nn
from torch.func import functional_call

from nvalchemi.training.peft import _peft

LoRALayer = _peft.LoRALayer

LoRAWrappableLayer: TypeAlias = type[nn.Module]
LoRAWrapper: TypeAlias = type[_peft.LoRALayer]
LoRAWrapperRegistration: TypeAlias = tuple[LoRAWrappableLayer, LoRAWrapper]
LoRAWrapperRegistrations: TypeAlias = tuple[LoRAWrapperRegistration, ...]


__all__ = [
    "E3NNFullyConnectedLoRALayer",
    "EquivariantLoRALinear",
    "LoRAWrapper",
    "LoRAWrapperRegistration",
    "LoRAWrapperRegistrations",
    "LoRAWrappableLayer",
    "available_lora_wrappers",
    "register_builtin_lora_wrappers",
]

_BUILTIN_LORA_WRAPPERS_REGISTERED = False


def _import_e3nn() -> tuple[type[nn.Module], type[nn.Module], object, type[nn.Module]]:
    """Import e3nn objects needed by the equivariant LoRA wrappers."""
    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([slice])
        from e3nn import o3
        from e3nn.nn import Dropout
        from e3nn.nn._fc import _Layer
    except ImportError as exc:
        raise ImportError(
            "Equivariant LoRA requires e3nn. Install the e3nn dependency."
        ) from exc
    return o3.Linear, _Layer, o3, Dropout


def _e3nn_layer_cls(kind: Literal["linear", "fully_connected"]) -> type[nn.Module]:
    """Return an e3nn layer class by kind."""
    linear_cls, fc_layer_cls, _o3, _dropout_cls = _import_e3nn()
    if kind == "linear":
        return linear_cls
    elif kind == "fully_connected":
        return fc_layer_cls
    else:
        raise ValueError(f"Invalid e3nn layer kind: {kind}")

class EquivariantLoRALinear(nn.Module, LoRALayer):
    """LoRA layer for equivariant linear layers ``e3nn.o3.Linear``.

    The adapter is itself equivariant: ``base(x) + scale * B(A(x))`` where
    ``A`` and ``B`` are ``e3nn.o3.Linear`` maps through irreps common to the
    base layer's input and output.

    Parameters
    ----------
    base_layer : nn.Module
        An ``e3nn.o3.Linear`` instance to wrap.
    rank : int
        Adapter multiplicity per irrep.
    alpha : float
        LoRA scaling parameter.
    dropout : float, optional
        Adapter dropout probability. Defaults to ``0.0``.
    init : _peft.LoRAInit, optional
        The initialization method for the LoRA adapter.
    **kwargs : Any
        Additional keyword arguments.

    Attributes
    ----------
    mergeable : bool
        Required by ``LoRALayer``; whether this adapter can be folded into the
        wrapped base layer.
    enabled : bool
        Required by ``LoRALayer``; whether the LoRA residual is active during
        forward passes.

    Properties
    ----------
    lora_A : nn.Parameter
        Required by ``LoRALayer``; weight parameter of the input-side equivariant adapter layer.
    lora_B : nn.Parameter
        Required by ``LoRALayer``; weight parameter of the output-side equivariant adapter layer.
    """

    mergeable = True

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init: _peft.LoRAInit = "default",  # noqa: ARG002
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        linear_cls, _fc_layer_cls, _o3, dropout_cls = _import_e3nn()
        if not isinstance(base_layer, linear_cls):
            raise TypeError(
                "EquivariantLoRALinear can only wrap e3nn.o3.Linear layers."
            )
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.enabled = True

        # Initialize LoRA adapter layers.
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.irreps_in = base_layer.irreps_in
        self.irreps_out = base_layer.irreps_out
        self.adapter_irreps = self._build_adapter_irreps(
            self.irreps_in, self.irreps_out, self.rank
        )
        self.lora_A_layer = linear_cls(
            self.irreps_in,
            self.adapter_irreps,
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )
        self.lora_B_layer = linear_cls(
            self.adapter_irreps,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )
        self.lora_A_layer.to(
            device=base_layer.weight.device, dtype=base_layer.weight.dtype
        )
        self.lora_B_layer.to(
            device=base_layer.weight.device, dtype=base_layer.weight.dtype
        )
        self.lora_dropout = (
            dropout_cls(self.irreps_in, p=dropout) if dropout > 0.0 else nn.Identity()
        )
        # Initialize LoRA weights to small random values while keeping the
        # initial adapter residual at zero.
        with torch.no_grad():
            self.lora_A_layer.weight.normal_(mean=0.0, std=1e-3)
            self.lora_B_layer.weight.zero_()

        # Pre-compute instruction maps for efficient merged weight computation.
        self._build_instruction_maps()

    @property
    def lora_A(self) -> nn.Parameter:
        """Return the input-side LoRA parameter."""
        return self.lora_A_layer.weight

    @property
    def lora_B(self) -> nn.Parameter:
        """Return the output-side LoRA parameter."""
        return self.lora_B_layer.weight

    @classmethod
    def is_compatible(cls, base_layer: nn.Module) -> bool:
        """Return whether ``base_layer`` has shared input/output irreps."""
        try:
            cls._build_adapter_irreps(
                base_layer.irreps_in,
                base_layer.irreps_out,
                rank=1,
            )
        except (AttributeError, ValueError):
            return False
        return True

    @staticmethod
    def _build_adapter_irreps(
        irreps_in: object, irreps_out: object, rank: int
    ) -> object:
        """Build equivariant adapter irreps from input/output overlap.

        Following MACE's convention, each shared irrep type receives ``rank``
        copies.
        """
        _linear_cls, _fc_layer_cls, o3, _dropout_cls = _import_e3nn()
        irreps_in = o3.Irreps(irreps_in).simplify()
        irreps_out = o3.Irreps(irreps_out).simplify()
        in_irrep_types = {ir for _, ir in irreps_in}
        out_irrep_types = {ir for _, ir in irreps_out}

        if len(in_irrep_types) != len(irreps_in):
            raise ValueError(
                "cannot build equivariant LoRA adapter: input irreps are not unique."
            )
        if len(out_irrep_types) != len(irreps_out):
            raise ValueError(
                "cannot build equivariant LoRA adapter: output irreps are not unique."
            )

        shared_irrep_types = in_irrep_types & out_irrep_types
        if not shared_irrep_types:
            raise ValueError(
                "cannot build equivariant LoRA adapter: input and output irreps "
                f"share no common irreps ({irreps_in!s} -> {irreps_out!s})."
            )
        shared_irrep_types = sorted(shared_irrep_types, key=lambda ir: (ir.l, ir.p))
        return o3.Irreps([(rank, ir) for ir in shared_irrep_types])

    def _build_instruction_maps(self) -> None:
        """Build maps for composing A/B adapter paths into base paths."""
        # Maps input irrep index to ``(instruction_index, adapter_irrep_index,
        # path_weight)`` for the input-side adapter layer.
        self._A_path_by_input = {
            instr.i_in: (idx, instr.i_out, instr.path_weight)
            for idx, instr in enumerate(self.lora_A_layer.instructions)
        }
        # Maps ``(adapter_irrep_index, output_irrep_index)`` to
        # ``(instruction_index, path_weight)`` for the output-side adapter layer.
        self._B_path_by_adapter_output = {
            (instr.i_in, instr.i_out): (idx, instr.path_weight)
            for idx, instr in enumerate(self.lora_B_layer.instructions)
        }

    @staticmethod
    def _weight_blocks_by_instruction(linear: nn.Module) -> dict[int, torch.Tensor]:
        """Reshape flat ``linear.weight`` into per-instruction blocks.

        Parameters
        ----------
        linear : nn.Module
            An ``e3nn.o3.Linear`` instance.

        Returns
        -------
        dict[int, torch.Tensor]
            Instruction index to weight matrix with shape ``instr.path_shape``.
        """
        blocks = {}
        offset = 0
        for idx, instr in enumerate(linear.instructions):
            # e3nn stores all instruction weights in one flat vector. For
            # o3.Linear, each instruction weight mixes multiplicity channels
            # for one matching irrep path.
            size = instr.path_shape[0] * instr.path_shape[1]
            blocks[idx] = linear.weight[offset : offset + size].reshape(
                instr.path_shape
            )
            offset += size
        return blocks

    def _merged_weight(self) -> torch.Tensor:
        """Return base plus LoRA delta in flat ``o3.Linear.weight`` layout."""
        base_blocks = self._weight_blocks_by_instruction(self.base_layer)
        lora_A_blocks = self._weight_blocks_by_instruction(self.lora_A_layer)
        lora_B_blocks = self._weight_blocks_by_instruction(self.lora_B_layer)

        merged_blocks = []
        for base_idx, base_instr in enumerate(self.base_layer.instructions):
            if base_instr.i_in not in self._A_path_by_input:
                merged_blocks.append(base_blocks[base_idx])
                continue

            A_idx, i_mid, path_weight_A = self._A_path_by_input[base_instr.i_in]
            B_key = (i_mid, base_instr.i_out)
            if B_key not in self._B_path_by_adapter_output:
                merged_blocks.append(base_blocks[base_idx])
                continue

            B_idx, path_weight_B = self._B_path_by_adapter_output[B_key]
            ratio = path_weight_A * path_weight_B / base_instr.path_weight
            delta = lora_A_blocks[A_idx] @ lora_B_blocks[B_idx]
            merged_blocks.append(base_blocks[base_idx] + self.scaling * ratio * delta)

        return torch.cat([block.flatten() for block in merged_blocks])

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """Run the frozen base layer plus the equivariant LoRA residual."""
        out = self.base_layer(x, *args, **kwargs)
        if self.enabled:
            out = (
                out
                + self.lora_B_layer(self.lora_A_layer(self.lora_dropout(x)))
                * self.scaling
            )
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold the LoRA delta into the frozen base layer."""
        self.base_layer.weight.copy_(self._merged_weight())


class E3NNFullyConnectedLoRALayer(nn.Module, LoRALayer):
    """LoRA layer for fully connected scalar MLP layers ``e3nn.nn._fc._Layer``.

    These internal layers store weights in ``(in_features, out_features)``
    layout, so the LoRA delta is ``A @ B`` rather than the ``torch.nn.Linear``
    ``B @ A`` convention.

    Parameters
    ----------
    base_layer : nn.Module
        An ``e3nn.nn._fc._Layer`` instance to wrap.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling numerator.
    dropout : float, optional
        Adapter dropout probability. Defaults to ``0.0``.
    init : _peft.LoRAInit, optional
        The initialization method for the LoRA adapter.
    **kwargs : Any
        Additional keyword arguments.

    Attributes
    ----------
    mergeable : bool
        Required by ``LoRALayer``; whether this adapter can be folded into the
        wrapped base layer.
    enabled : bool
        Required by ``LoRALayer``; whether the LoRA residual is active during
        forward passes.
    lora_A : nn.Parameter
        Required by ``LoRALayer``; Input-side low-rank factor with shape ``(in_features, rank)``.
    lora_B : nn.Parameter
        Required by ``LoRALayer``; Output-side low-rank factor with shape ``(rank, out_features)``.
    """

    mergeable = True

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        init: _peft.LoRAInit = "default",
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        _linear_cls, fc_layer_cls, _o3, _dropout_cls = _import_e3nn()
        if not isinstance(base_layer, fc_layer_cls):
            raise TypeError(
                "E3NNFullyConnectedLoRALayer can only wrap e3nn.nn._fc._Layer."
            )
        if not hasattr(base_layer, "weight"):
            raise TypeError("e3nn fully connected layer has no weight parameter.")
        if dropout != 0.0:
            raise ValueError(
                "E3NNFullyConnectedLoRALayer does not support nonzero dropout "
                "because e3nn fully connected LoRA must be applied in weight "
                "space to match merge_into_base."
            )
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        weight = base_layer.weight
        in_features, out_features = int(weight.shape[0]), int(weight.shape[1])
        self._make_lora_params(
            in_features,
            out_features,
            weight,
            int(rank),
            float(alpha),
            dropout,
            init,
        )

    @classmethod
    def is_compatible(cls, base_layer: nn.Module) -> bool:
        """Return whether ``base_layer`` exposes a matrix weight."""
        try:
            _linear_cls, fc_layer_cls, _o3, _dropout_cls = _import_e3nn()
        except ImportError:
            return False
        return (
            isinstance(base_layer, fc_layer_cls)
            and getattr(getattr(base_layer, "weight", None), "ndim", None) == 2
        )

    def _merged_weight(self) -> torch.Tensor:
        """Return the base weight plus the low-rank adapter delta."""
        return self.base_layer.weight + (self.lora_A @ self.lora_B) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the frozen base layer plus the scalar LoRA residual."""
        if not self.enabled:
            return self.base_layer(x)

        # e3nn _Layer normalizes ``weight`` inside forward. Applying the LoRA
        # delta as an output residual would skip that normalization and would
        # not match ``merge_into_base``.
        # ``functional_call`` substitutes the effective weight for this call
        # without mutating the frozen base layer parameter.
        return functional_call(self.base_layer, {"weight": self._merged_weight()}, (x,))

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold the LoRA delta into the frozen base layer."""
        self.base_layer.weight.add_((self.lora_A @ self.lora_B) * self.scaling)


_BUILTIN_LORA_WRAPPER_FACTORIES = (
    (
        partial(_e3nn_layer_cls, "linear"),
        EquivariantLoRALinear,
    ),
    (
        partial(_e3nn_layer_cls, "fully_connected"),
        E3NNFullyConnectedLoRALayer,
    ),
)


def register_builtin_lora_wrappers() -> None:
    """Register ALCHEMI's built-in e3nn LoRA wrappers."""
    global _BUILTIN_LORA_WRAPPERS_REGISTERED
    if _BUILTIN_LORA_WRAPPERS_REGISTERED:
        return
    for layer_cls_factory, wrapper_cls in _BUILTIN_LORA_WRAPPER_FACTORIES:
        try:
            layer_cls = layer_cls_factory()
        except ImportError as exc:
            warnings.warn(
                f"Skipping built-in LoRA wrapper {wrapper_cls.__name__}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue
        _peft.register_lora_wrapper(layer_cls, wrapper_cls)
    _BUILTIN_LORA_WRAPPERS_REGISTERED = True


def available_lora_wrappers() -> LoRAWrapperRegistrations:
    """Return registered (layer, LoRA-wrapper) pairs."""
    register_builtin_lora_wrappers()
    return tuple(_peft._physicsnemo_peft.lora._LORA_WRAPPERS.items())
