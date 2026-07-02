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
"""Toy full-geometry dense-``neighbor_matrix`` MLIP — gates the
``gp_replicate_geometry`` graph-parallel path (PME's fused-kernel shape).

Unlike :mod:`_toy_graph_parallel_dense` (owned rows + ``refresh_neighbors``), this
model's "kernel" indexes the position array directly (``pos[neighbor_matrix]``),
like PME's fused real-space+reciprocal kernel — so the framework runs it on the
**replicated full geometry** with the neighbour matrix masked to this rank's owned
receivers, emits a per-node energy under ``node_energy_key``, and takes forces by
autograd of the owned energy over the full-position leaf.

It is deliberately **single-pass / pairwise** (``E_i = ½ Σ_j w_i w_j e^{-r_ij}``):
owned-receiver masking is exact only for single-pass models (a multi-layer MPNN
would need per-layer feature all-gather instead), which matches PME (one pairwise
real-space sum + one mesh reciprocal, no iterated message passing). A per-node
``atomic_energies`` output lets the framework do the owned-aware reduction; the
plain ``energy`` (sum over all atoms) serves the single-process reference.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict

import torch
from torch import nn

from nvalchemi.distributed.helpers import Scope, system_sum
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import NeighborConfig, NeighborListFormat


def _build_spec():
    from nvalchemi.distributed.spec import (
        SPEC_MPNN_GP,
        CompilePolicy,
        ForceStrategy,
        OutputKind,
        OutputSpec,
        Reduce,
    )

    return dataclasses.replace(
        SPEC_MPNN_GP,
        outputs={
            "energy": OutputSpec(OutputKind.PER_GRAPH),
            "forces": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
            "atomic_energies": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
        },
        node_energy_key="atomic_energies",
        gp_replicate_geometry=True,
        compile=CompilePolicy(force_strategy=ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY),
    )


class ToyGraphParallelDenseFullWrapper(nn.Module, BaseModelMixin):
    """Single-pass pairwise MLIP over a dense ``neighbor_matrix`` (full geometry)."""

    def __init__(self, cutoff: float = 2.5) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.zval = nn.Embedding(100, 1)
        self._spec = _build_spec()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "atomic_energies"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=NeighborListFormat.MATRIX, half_list=False
            ),
        )

    @property
    def embedding_shapes(self) -> dict:
        return {}

    @property
    def distribution_spec(self):
        return self._spec

    def adapt_input(self, data, **kwargs):
        return {}

    def adapt_output(self, model_output, data):
        return model_output

    def compute_embeddings(self, data, **kwargs):
        return data

    def forward(self, data, **kwargs):
        pos = data.positions  # [N, 3] — full geometry under gp_replicate_geometry
        z = data.atomic_numbers.long()
        nbmat = data.neighbor_matrix.long()  # [N, K] global sender ids
        num = data.num_neighbors.long()  # [N] (masked to owned under GP)
        n_graphs = int(data.num_graphs)
        batch_idx = data.batch_idx.long()

        n = pos.shape[0]
        k = nbmat.shape[1]
        valid = torch.arange(k, device=nbmat.device).unsqueeze(0) < num.unsqueeze(1)
        sender = nbmat.clamp(max=n - 1)

        w = self.zval(z).squeeze(-1)  # [N] per-atom weight
        rij = (pos[sender] - pos.unsqueeze(1)).norm(dim=-1)  # [N, K]
        # Single-pass pairwise energy per receiver atom.
        pair = w.unsqueeze(1) * w[sender] * torch.exp(-rij) * valid  # [N, K]
        node_e = 0.5 * pair.sum(dim=1)  # [N]

        # Plain energy (all atoms) for the single-process reference; the framework
        # overrides it with the owned-aware sum of atomic_energies under GP.
        energy = system_sum(node_e, batch_idx, n_graphs, scope=Scope.LOCAL)
        out: OrderedDict = OrderedDict()
        out["energy"] = energy
        out["atomic_energies"] = node_e
        return out
