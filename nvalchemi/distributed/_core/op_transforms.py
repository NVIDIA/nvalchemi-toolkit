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

"""Argument and output transforms for opaque-kernel adapters.

Declarative, zero-data marker dataclasses keyed by argument / output
position on an :class:`~nvalchemi.distributed._core.adapter.OpAdapter`.
They describe how a sharded/halo tensor must be reshaped around an
*opaque* kernel — one that bypasses ShardTensor's ``__torch_function__``
(Warp / Triton ``custom_op`` / ``@torch.jit.script``) and would
otherwise see only the per-rank local view.

* Argument transforms reshape an input *before* the kernel fires
  (:class:`GatherInputs`, :class:`GatherInputsFull`, :class:`SliceOwned`).
* Output transforms reshape a result *after* the kernel returns
  (:class:`ScatterOutputs`, :class:`AllReduceSum`, :class:`SliceOutputsOwned`).

Each is a frozen marker so future configuration (reduce op, slice
ranges) can land additively. The :class:`~...adapter.AdapterRegistry`
applies them around the wrapped op via the relevant collective.

Part of the upstream-candidate ``_core/`` surface; intentionally
domain-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

__all__ = [
    "ArgTransform",
    "OutputTransform",
    "GatherInputs",
    "GatherInputsFull",
    "SliceOwned",
    "ScatterOutputs",
    "AllReduceSum",
    "SliceOutputsOwned",
]


@dataclass(frozen=True)
class GatherInputs:
    """Halo-pad an owned-shape input ``(n_owned, *F)`` to
    ``(n_padded, *F)`` via :func:`halo_forward_exchange` before the
    kernel fires.

    Use when a kernel expects to see all rows the rank can route an edge
    to — i.e. owned plus halo. Inverse of :class:`SliceOwned`.
    """


@dataclass(frozen=True)
class GatherInputsFull:
    """Full-gather a sharded ``(n_owned + 1, *F)`` input to
    ``(n_global + 1, *F)`` via :func:`distributed_index_select` over
    ``[0, n_global)`` plus the local padding row.

    Sharded-storage analogue of :class:`GatherInputs`. Use for opaque
    kernels (Warp / Triton custom_ops bypassing
    ``__torch_function__``) that would otherwise be silently unwrapped
    to the per-rank view and read out of bounds on a global-index NL
    (e.g. AIMNet2's ``aimnet::conv_sv_2d_sp_*``).
    """


@dataclass(frozen=True)
class SliceOwned:
    """Slice a halo-padded ``(n_padded, *F)`` input to ``(n_owned, *F)``
    before the kernel fires.

    Use when each atom should contribute exactly once globally — Ewald
    structure-factor accumulation, PME charge spreading. The cross-rank
    sum happens at the output via :class:`AllReduceSum` rather than at
    the input via halo-correction.
    """


@dataclass(frozen=True)
class ScatterOutputs:
    """Halo-correct an output: ``halo_reverse_exchange + halo_forward_exchange``
    after the kernel returns.

    The fused-scatter pattern (e.g. MACE-cueq): kernel writes per-row
    values into halo rows; halo_reverse routes those partial contributions
    to the owners; halo_forward refreshes halo copies to match.
    """


@dataclass(frozen=True)
class AllReduceSum:
    """Sum the output's per-rank partial across the mesh via
    :func:`distributed_all_reduce`.

    Backward is symmetric (the all-reduce is its own adjoint), so this
    plugs cleanly into autograd. Pair with :class:`SliceOwned` on the
    input side — slice ensures each atom contributes once locally;
    AllReduceSum collects partials into a globally-correct result.
    """


@dataclass(frozen=True)
class SliceOutputsOwned:
    """Slice a sharded ``(n_global + 1, *F)`` output back to
    ``(n_owned + 1, *F)`` so downstream per-rank model code keeps its
    layout.

    Inverse of :class:`GatherInputsFull` on the output side; pair them
    when wrapping a sharded-storage kernel that needs the global view
    only inside its body.
    """


# Discriminated unions. Argument transforms apply to the kernel's
# input args; output transforms apply to the kernel's outputs.
ArgTransform = Union[GatherInputs, GatherInputsFull, SliceOwned]
OutputTransform = Union[ScatterOutputs, AllReduceSum, SliceOutputsOwned]
