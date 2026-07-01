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
"""MACE-free toy reproducing the ``@torch.jit.script`` + ShardTensor CUDA
illegal-memory-access (IMA) that scripted-op marshalling fixes.

The minimal trigger is
a ``@torch.jit.script`` op whose input is a **null-storage op-output**
ShardTensor (e.g. a scatter result) with autograd recording on. TorchScript
bypasses ``__torch_function__`` and its fused kernel reads the storage-less
wrapper's near-null ``data_ptr`` -> IMA. Reproducing it needs **>=2 layers**
so a later scripted op consumes the previous layer's scatter output (a
storage-less wrapper) rather than a from_local tensor (which has storage).

``ToyScriptedMPNNWrapper`` is a faithful F = -dE/dx MLIP (the dominant force
category): gather -> scripted message -> grad-aware halo scatter -> node
update, x2 layers, energy = readout(x).sum(), forces = -dE/dpositions. It
defaults to a scripted **submodule** (``scripted_block``) so the regression
exercises the zero-config auto-discovery safety net; auto-discovery
wraps the ``ScriptModule`` at ``DistributedModel`` setup. With marshalling
disabled (``NVALCHEMI_SCRIPTED_MARSHAL=off``) the forward IMAs; with it on the
forward runs clean.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import NeighborConfig, NeighborListFormat


@torch.jit.script
def scripted_msg(a: torch.Tensor, elen: torch.Tensor) -> torch.Tensor:
    """Module-level scripted message op (covers the e3nn ``spherical_harmonics``
    case — a scripted *function*, caught only by a declared ``JitAdapter``)."""
    return a * elen * elen


def plain_msg(a: torch.Tensor, elen: torch.Tensor) -> torch.Tensor:
    """Eager byte-equivalent of :func:`scripted_msg` (no IMA — control)."""
    return a * elen * elen


class _ScriptedMsgModule(nn.Module):
    """Scripted *submodule* form (covers the e3nn ``TensorProduct``
    ``_compiled_main_*`` case — a ``ScriptModule``, caught by auto-discovery)."""

    def forward(self, a: torch.Tensor, elen: torch.Tensor) -> torch.Tensor:
        return a * elen * elen


class ToyScriptedMPNNWrapper(nn.Module, BaseModelMixin):
    """2-layer halo MPNN with autograd forces and a scripted message op.

    Parameters
    ----------
    mode
        Which message op the scripted boundary uses:
        ``"function"`` (default) — the module-level ``@torch.jit.script``
        function (the faithful analog of e3nn's ``_spherical_harmonics``; trips
        the IMA without the fix and is cleared by a declared ``JitAdapter``);
        ``"submodule"`` — a ``torch.jit.ScriptModule`` (auto-discovered);
        ``"plain"`` — eager control (no IMA).
    """

    def __init__(
        self,
        hidden: int = 16,
        n_layers: int = 2,
        cutoff: float = 6.0,
        mode: str = "function",
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.mode = mode
        self.embed = nn.Embedding(100, hidden)
        self.msg_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )
        self.upd = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden, 1)
        self.scripted_block = torch.jit.script(_ScriptedMsgModule())
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=NeighborListFormat.COO, half_list=False
            ),
        )

    @property
    def embedding_shapes(self) -> dict:
        return {}

    @property
    def distribution_spec(self):
        # SPEC_MPNN_HALO: scatter-heavy MPNN halo policy (same as MACE). For the
        # module-level scripted function (``mode="function"``), declare a
        # marshalling JitAdapter — auto-discovery only catches ScriptModule
        # *submodules*, not module-level scripted *functions* (exactly why MACE
        # must declare its ``_spherical_harmonics`` adapter). ``__name__`` is
        # this module's import name, which the adapter re-imports at install.
        import dataclasses

        from nvalchemi.distributed.spec import SPEC_MPNN_HALO

        if self.mode != "function":
            return SPEC_MPNN_HALO
        from nvalchemi.distributed._core.adapter import JitAdapter

        d = SPEC_MPNN_HALO.distribution
        return dataclasses.replace(
            SPEC_MPNN_HALO,
            distribution=dataclasses.replace(
                d,
                third_party_helpers=d.third_party_helpers
                + (JitAdapter(__name__, "scripted_msg", mode="marshal"),),
            ),
        )

    def adapt_input(self, data, **kwargs):
        return {}

    def adapt_output(self, model_output, data):
        return model_output

    def compute_embeddings(self, data, **kwargs):
        return data

    def _msg(self, a: torch.Tensor, elen: torch.Tensor) -> torch.Tensor:
        if self.mode == "submodule":
            return self.scripted_block(a, elen)
        if self.mode == "function":
            return scripted_msg(a, elen)
        return plain_msg(a, elen)

    def forward(self, data, **kwargs):
        positions = data.positions
        if not positions.requires_grad:
            positions.requires_grad_(True)
        z = data.atomic_numbers.long()
        nl = data.neighbor_list.long()
        n_atoms = positions.shape[0]
        src = nl[:, 0]
        dst = nl[:, 1]
        valid = (src < n_atoms) & (dst < n_atoms)
        src = src[valid]
        dst = dst[valid]
        n_graphs = int(data.num_graphs)

        x = self.embed(z)
        hidden = x.shape[1]
        dst_exp = dst.unsqueeze(-1).expand(-1, hidden)
        for msg_lin, upd in zip(self.msg_lin, self.upd, strict=True):
            s = x[src]
            ev = positions[dst] - positions[src]
            elen = ev.norm(dim=-1, keepdim=True)
            msg = self._msg(msg_lin(s), elen)  # scripted op on halo ShardTensor
            agg = torch.zeros_like(x)
            # MUST capture the return — the grad-aware halo scatter handler
            # returns a fresh tensor; ``agg.scatter_add_(...)`` and dropping
            # the result silently severs the autodiff force graph.
            agg = agg.scatter_add_(0, dst_exp, msg)
            x = x + upd(agg)

        e_scalar = self.readout(x).sum()
        (grad_e,) = torch.autograd.grad(
            e_scalar, positions, allow_unused=True, retain_graph=True
        )
        forces = -grad_e if grad_e is not None else torch.zeros_like(positions)
        energy = (
            torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
            + e_scalar.detach()
        )
        out: OrderedDict = OrderedDict()
        out["energy"] = energy
        out["forces"] = forces
        return out
