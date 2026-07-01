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
"""Toy graph-parallel MPNN: a minimal BYO model on ``SPEC_MPNN_GP``.

A faithful ``E = sum(node_energy)`` MLIP written only against the public
intent-verb helpers — :func:`refresh_neighbors` (all-gather the node features so
each edge sees its source) and :func:`system_sum` (owned per-graph energy sum +
cross-rank all-reduce). The wrapper carries no decomposition logic: outside a
distributed forward both verbs are the identity, so the same forward runs
single-process; under the graph-parallel policy the framework hands it owned
rows plus a ``neighbor_list`` whose senders are global ids and receivers are
owned-local, and the verbs resolve to the GP collectives.

Used to gate ``DistributedModel._call_graph_parallel`` end to end against a
single-process reference (energy + owned forces).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nvalchemi.distributed.helpers import Scope, refresh_neighbors, system_sum
from nvalchemi.distributed.spec import SPEC_MPNN_GP
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import NeighborConfig, NeighborListFormat


class ToyGraphParallelMPNNWrapper(nn.Module, BaseModelMixin):
    """2-layer message-passing MLIP for the graph-parallel strategy."""

    def __init__(
        self, hidden: int = 8, n_layers: int = 2, cutoff: float = 2.5
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(100, hidden)
        self.msg_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )
        self.upd = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden, 1)
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
        return SPEC_MPNN_GP

    def adapt_input(self, data, **kwargs):
        return {}

    def adapt_output(self, model_output, data):
        return model_output

    def compute_embeddings(self, data, **kwargs):
        return data

    def forward(self, data, **kwargs):
        pos = data.positions
        z = data.atomic_numbers.long()
        nl = data.neighbor_list.long()
        src, dst = nl[:, 0], nl[:, 1]
        n_graphs = int(data.num_graphs)
        batch_idx = data.batch_idx.long()

        pos_full = refresh_neighbors(pos)
        x = self.embed(z)
        hidden = x.shape[1]
        dst_exp = dst.unsqueeze(-1).expand(-1, hidden)
        for msg_lin, upd in zip(self.msg_lin, self.upd, strict=True):
            x_full = refresh_neighbors(x)
            edge_len = (pos_full[src] - pos[dst]).norm(dim=-1, keepdim=True)
            msg = msg_lin(x_full[src]) * edge_len
            agg = torch.zeros_like(x).scatter_add_(0, dst_exp, msg)
            x = x + upd(agg)

        node_e = self.readout(x).squeeze(-1)
        # Owned per-graph partial (no cross-rank reduce): the framework folds it
        # into the global energy and takes forces from it — the per-layer
        # node-gather adjoint already collects each owned atom's cross-rank
        # gradient, so an autograd-aware all-reduce here would over-count it.
        energy = system_sum(node_e, batch_idx, n_graphs, scope=Scope.LOCAL)
        out: OrderedDict = OrderedDict()
        out["energy"] = energy
        out["atomic_energies"] = node_e
        return out
