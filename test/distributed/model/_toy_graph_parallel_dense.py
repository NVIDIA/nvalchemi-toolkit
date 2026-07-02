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
"""Toy graph-parallel MPNN over a DENSE ``neighbor_matrix`` — the dense-nbmat
analogue of :mod:`_toy_graph_parallel_mpnn`.

Same ``E = sum(node_energy)`` contract and the same intent-verb helpers
(:func:`refresh_neighbors`, :func:`system_sum`), but the model consumes a dense
``[n, K]`` neighbour matrix (global sender columns, ``num_neighbors`` mask)
instead of a COO ``neighbor_list``. This is the toy that gates the framework's
dense-nbmat graph-parallel path (``_graph_parallel_owned_nbmat`` + the
``NeighborListFormat.MATRIX`` branch in ``_graph_partition_run_forward``) — the
same machinery PME real-space / AIMNet2 will ride.

Outside a distributed forward both verbs are identities and ``neighbor_matrix``
is the full ``[N, K]``, so the identical forward runs single-process; under the
graph-parallel policy the framework hands it owned receiver rows whose sender
columns are global ids into the all-gathered node set.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nvalchemi.distributed.helpers import Scope, refresh_neighbors, system_sum
from nvalchemi.distributed.spec import SPEC_MPNN_GP
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import NeighborConfig, NeighborListFormat


class ToyGraphParallelDenseWrapper(nn.Module, BaseModelMixin):
    """2-layer dense-``neighbor_matrix`` MLIP for the graph-parallel strategy."""

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
            # MATRIX format is what routes the framework onto the dense-nbmat GP
            # builder; the toy reads ``neighbor_matrix`` / ``num_neighbors``.
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=NeighborListFormat.MATRIX, half_list=False
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
        pos = data.positions  # owned receivers [n, 3] (leaf under GP)
        z = data.atomic_numbers.long()
        nbmat = data.neighbor_matrix.long()  # [n, K] global sender ids (padded)
        num = data.num_neighbors.long()  # [n] valid neighbours per receiver
        n_graphs = int(data.num_graphs)
        batch_idx = data.batch_idx.long()

        # All-gather the node features so each receiver's global senders resolve.
        pos_full = refresh_neighbors(pos)  # (N_global, 3)
        n_global = pos_full.shape[0]
        k = nbmat.shape[1]
        # Mask padded columns (num_neighbors) and clamp sender ids in range —
        # padded entries contribute zero, so the clamped index is never used.
        valid = (
            torch.arange(k, device=nbmat.device).unsqueeze(0) < num.unsqueeze(1)
        )  # (n, K)
        sender = nbmat.clamp(max=n_global - 1)  # (n, K)

        x = self.embed(z)  # (n, H)
        for msg_lin, upd in zip(self.msg_lin, self.upd, strict=True):
            x_full = refresh_neighbors(x)  # (N_global, H)
            # receiver = owned pos[i]; sender = pos_full[global id]
            edge_len = (pos_full[sender] - pos.unsqueeze(1)).norm(
                dim=-1, keepdim=True
            )  # (n, K, 1)
            msg = msg_lin(x_full[sender]) * edge_len  # (n, K, H)
            msg = msg * valid.unsqueeze(-1)  # zero padded neighbours
            x = x + upd(msg.sum(dim=1))

        node_e = self.readout(x).squeeze(-1)  # (n,)
        # Owned per-graph partial (no cross-rank reduce): the framework folds it
        # into the global energy and takes forces from it; the per-layer node
        # gather's reduce-scatter adjoint already collects cross-rank gradients.
        energy = system_sum(node_e, batch_idx, n_graphs, scope=Scope.LOCAL)
        out: OrderedDict = OrderedDict()
        out["energy"] = energy
        out["atomic_energies"] = node_e
        return out
