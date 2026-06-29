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
"""Toy *opaque* MACE-style MPNN gating the node-replicate graph-parallel path.

This is the BYO stand-in for a model you can't edit (upstream MACE): its forward
uses a single global ``edge_index`` for both the edge-vector read and the conv
scatter, has a NONLINEAR node-wise update after each conv, and calls **none** of
the DD intent verbs. The cross-rank recombine is injected purely by a declared
``ModuleForwardAdapter`` on the conv submodule — exactly the adapter pattern MACE
already uses for halo — which wraps the conv's partial-message output with
``scatter_to_owners`` (→ ``GraphReplicatePolicy.fold`` = all-reduce). Under the
node-replicate strategy every rank holds the full node set and a sharded edge
slice, so the all-reduce *before* the nonlinear update is what makes the result
correct.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from nvalchemi.distributed import scatter_to_owners
from nvalchemi.distributed._core.adapter import ModuleForwardAdapter
from nvalchemi.distributed._core.spec import DistributionSpec
from nvalchemi.distributed._core.storage_policy import GraphReplicatePolicy
from nvalchemi.distributed.output_kinds import OutputKind, OutputSpec, Reduce
from nvalchemi.distributed.spec import MLIPSpec
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.neighbors import NeighborConfig, NeighborListFormat

_GP_REPLICATE_OUTPUTS = {
    "energy": OutputSpec(OutputKind.PER_GRAPH),
    "forces": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
    "atomic_energies": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
}


class _ToyConv(nn.Module):
    """A message conv: gather senders, per-edge MLP, scatter-sum to receivers.

    Returns the per-receiver partial message sum ``[N, H]`` — partial because
    each rank holds only its edge slice. The declared adapter all-reduces it.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_len: torch.Tensor,
        sender: torch.Tensor,
        receiver: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        mji = self.lin(node_feats[sender]) * edge_len
        out = node_feats.new_zeros((n, node_feats.shape[1]))
        return out.index_add(0, receiver, mji)


class ToyOpaqueMPNNWrapper(nn.Module, BaseModelMixin):
    """2-layer MPNN whose conv recombine is injected by a declared adapter."""

    def __init__(
        self, hidden: int = 8, n_layers: int = 2, cutoff: float = 2.5
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(100, hidden)
        self.convs = nn.ModuleList([_ToyConv(hidden) for _ in range(n_layers)])
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
        # Node-replicate policy + one conv adapter per conv: the adapter wraps the
        # partial-message output with scatter_to_owners (the all-reduce recombine).
        conv_adapters = tuple(
            ModuleForwardAdapter(c, _make_replicate_conv_forward(c), label="gp_replicate_conv")
            for c in self.convs
        )
        return MLIPSpec(
            distribution=DistributionSpec(
                policy=GraphReplicatePolicy(), shard_fields=()
            ),
            outputs=dict(_GP_REPLICATE_OUTPUTS),
        ).with_adapters(*conv_adapters)

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
        sender, receiver = nl[:, 0], nl[:, 1]
        n = pos.shape[0]
        n_graphs = int(data.num_graphs)
        batch_idx = data.batch_idx.long()

        node_feats = self.embed(z)
        for conv, upd in zip(self.convs, self.upd, strict=True):
            edge_len = (pos[sender] - pos[receiver]).norm(dim=-1, keepdim=True)
            msg = conv(node_feats, edge_len, sender, receiver, n)
            node_feats = node_feats + upd(torch.nn.functional.silu(msg))

        atomic_e = self.readout(node_feats).squeeze(-1)
        energy = atomic_e.new_zeros(n_graphs).index_add(0, batch_idx, atomic_e)
        out: OrderedDict = OrderedDict()
        out["energy"] = energy
        out["atomic_energies"] = atomic_e
        return out


def _make_replicate_conv_forward(conv: _ToyConv):
    """Wrap a conv's forward so its partial-message output is recombined across
    ranks (``scatter_to_owners`` → all-reduce). Identity in single-process."""
    original = conv.forward

    def _forward(*args, **kwargs):
        return scatter_to_owners(original(*args, **kwargs))

    return _forward
