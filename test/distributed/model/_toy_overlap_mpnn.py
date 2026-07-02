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
"""Opaque MPNN for the async-overlap gate: overlap is added by the SPEC only.

Unlike ``_toy_graph_parallel_mpnn`` (which calls ``refresh_neighbors`` inside its
forward), this model is a plain, distribution-agnostic MPNN — each message layer
is a bare scatter-add over ``x[sender]``, with **no DD verbs anywhere in the
model body**. The per-layer node-feature exchange is added *entirely* by the
wrapper's ``distribution_spec``, which declares an ``overlap_adapters`` set over
the message layers. That mirrors the real models (MACE / UMA / AIMNet2): the
model is opaque; only the spec changes.

Positions are folded into the initial node features so the (feature-only) message
still yields position-dependent forces.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nvalchemi.distributed.helpers import Scope, overlap_adapters, system_sum
from nvalchemi.distributed.spec import SPEC_MPNN_GP
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import NeighborConfig, NeighborListFormat


class _MessageLayer(nn.Module):
    """Bare message-passing layer: ``agg[dst] = Σ lin(x[src])``. No DD verbs —
    under domain decomposition a spec-declared OverlapAdapter wraps this forward
    to gather/borrow the sender features and split the edges; here it is just a
    scatter-add, so single-process runs it unchanged."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = self.lin(x[src])
        out = torch.zeros(
            x.shape[0], x.shape[1], dtype=x.dtype, device=x.device
        )
        return out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)


class _ToyOverlapMPNN(nn.Module):
    """Opaque 2-layer MPNN. ``forward(positions, atomic_numbers, edge_index)`` →
    per-atom energies. Positions enter via ``pos_proj`` so forces are non-trivial."""

    def __init__(self, hidden: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, hidden)
        self.pos_proj = nn.Linear(3, hidden)
        self.layers = nn.ModuleList([_MessageLayer(hidden) for _ in range(n_layers)])
        self.upd = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden, 1)

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.embed(atomic_numbers) + self.pos_proj(positions)
        for layer, upd in zip(self.layers, self.upd, strict=True):
            h = h + upd(layer(h, edge_index))
        return self.readout(h).squeeze(-1)


class ToyOverlapMPNNWrapper(nn.Module, BaseModelMixin):
    """BYO wrapper whose ``distribution_spec`` adds async overlap over the message
    layers — the only distributed-aware line in the whole model."""

    def __init__(self, hidden: int = 8, n_layers: int = 2, cutoff: float = 2.5) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.model = _ToyOverlapMPNN(hidden, n_layers)
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
        # The whole distributed story: name the message layers, add overlap. The
        # model body above is untouched. Same pattern for MACE/UMA/AIMNet2.
        return SPEC_MPNN_GP.with_adapters(*overlap_adapters(self.model.layers))

    def adapt_input(self, data, **kwargs):
        nl = data.neighbor_list.long()  # (E, 2): global senders, owned-local receivers
        return {
            "atomic_numbers": data.atomic_numbers.long(),
            "edge_index": nl.T,  # (2, E)
        }

    def adapt_output(self, model_output, data):
        return model_output

    def compute_embeddings(self, data, **kwargs):
        return data

    def forward(self, data, **kwargs):
        pos = data.positions
        n_graphs = int(data.num_graphs)
        batch_idx = data.batch_idx.long()
        kw = self.adapt_input(data)
        kw["positions"] = pos
        node_e = self.model(**kw)
        # Owned per-graph partial; the GP consolidation folds it into the global
        # energy and takes forces from it (the per-layer gather adjoint already
        # routes each owned atom's cross-rank gradient once).
        energy = system_sum(node_e, batch_idx, n_graphs, scope=Scope.LOCAL)
        out: OrderedDict = OrderedDict()
        out["energy"] = energy
        out["atomic_energies"] = node_e
        return out
