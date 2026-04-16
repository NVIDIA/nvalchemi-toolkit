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
"""
AtomicData and Batch: Graph-structured molecular data
=====================================================

This example walks through the full API of :class:`~nvalchemi.data.AtomicData`
and :class:`~nvalchemi.data.Batch`: construction, properties, indexing,
mutation, device movement, and serialization.
"""

import torch

from nvalchemi.data import AtomicData, Batch

# %%
# AtomicData — Construction
# --------------------------
# :class:`~nvalchemi.data.AtomicData` requires ``positions`` (shape ``[n_nodes, 3]``)
# and ``atomic_numbers`` (shape ``[n_nodes]``). All other fields are optional.

positions = torch.randn(4, 3)
atomic_numbers = torch.tensor([1, 6, 6, 1], dtype=torch.long)
data = AtomicData(positions=positions, atomic_numbers=atomic_numbers)

# With edges (e.g. bonds or neighbor list): provide ``neighbor_list`` shape ``[n_edges, 2]``.
neighbor_list = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
data_with_edges = AtomicData(
    positions=positions,
    atomic_numbers=atomic_numbers,
    neighbor_list=neighbor_list,
)
print(f"With edges: num_edges={data_with_edges.num_edges}")

# With system-level fields (e.g. energy, cell, pbc for periodicity):
data_with_system = AtomicData(
    positions=positions,
    atomic_numbers=atomic_numbers,
    energy=torch.tensor([[0.5]]),
    cell=torch.eye(3).unsqueeze(0),
    pbc=torch.tensor([[True, True, False]]),
)
print(f"System energy shape: {data_with_system.energy.shape}")

# %%
# AtomicData — Properties
# -----------------------
# Core properties: :attr:`~nvalchemi.data.AtomicData.num_nodes`,
# :attr:`~nvalchemi.data.AtomicData.num_edges`, :attr:`~nvalchemi.data.AtomicData.device`,
# :attr:`~nvalchemi.data.AtomicData.dtype`.

print(f"num_nodes={data.num_nodes}, num_edges={data.num_edges}")
print(f"device={data.device}, dtype={data.dtype}")

# Level-wise property views (dicts of set fields):
# :attr:`~nvalchemi.data.AtomicData.node_properties`,
# :attr:`~nvalchemi.data.AtomicData.edge_properties`,
# :attr:`~nvalchemi.data.AtomicData.system_properties`.
print("node_properties keys:", list(data.node_properties.keys()))
print("system_properties keys:", list(data_with_system.system_properties.keys()))

# %%
# AtomicData — Dict-like access and mutation
# ------------------------------------------
# Use :meth:`~nvalchemi.data.AtomicData.__getitem__` / :meth:`~nvalchemi.data.AtomicData.__setitem__`
# for attribute access by name.

assert data["positions"] is data.positions
data["positions"] = torch.randn(4, 3)
assert data.positions.shape == (4, 3)

# Add custom node/edge/system properties with
# :meth:`~nvalchemi.data.AtomicData.add_node_property`,
# :meth:`~nvalchemi.data.AtomicData.add_edge_property`,
# :meth:`~nvalchemi.data.AtomicData.add_system_property`.
data.add_node_property("custom_node_feat", torch.randn(4, 2))
data_with_edges.add_edge_property("edge_weights", torch.ones(data_with_edges.num_edges))
data_with_system.add_system_property("temperature", torch.tensor([[300.0]]))
print(
    "After add_*_property, 'custom_node_feat' in node_properties:",
    "custom_node_feat" in data.node_properties,
)

# %%
# AtomicData — Chemical hash and equality
# ----------------------------------------
# :attr:`~nvalchemi.data.AtomicData.chemical_hash` gives a structure/composition hash;
# :meth:`~nvalchemi.data.AtomicData.__eq__` compares by chemical hash.

h = data.chemical_hash
print(f"chemical_hash length: {len(h)}")
data2 = AtomicData(
    positions=data.positions.clone(), atomic_numbers=data.atomic_numbers.clone()
)
print(f"Same structure equal: {data == data2}")

# %%
# AtomicData — Device and clone
# ------------------------------
# :meth:`~nvalchemi.data.AtomicData.to` and :meth:`~nvalchemi.data.data.DataMixin.clone`
# (and ``.cpu()`` / ``.cuda()`` from the mixin) for device movement and copying.

on_cpu = data.to("cpu")
cloned = data.clone()
print(f"to('cpu').device: {on_cpu.device}, clone is new object: {cloned is not data}")

# %%
# AtomicData — Serialization
# ---------------------------
# Pydantic serialization: :meth:`~pydantic.BaseModel.model_dump` and
# :meth:`~pydantic.BaseModel.model_dump_json` (tensors become lists in JSON).

data_vanilla = AtomicData(
    positions=torch.randn(2, 3), atomic_numbers=torch.ones(2, dtype=torch.long)
)
d = data_vanilla.model_dump(exclude_none=True)
print("model_dump keys (sample):", list(d.keys())[:4])
json_str = data_vanilla.model_dump_json()
print(f"model_dump_json length: {len(json_str)}")

# %%
# Batch — Construction
# ---------------------
# Build a :class:`~nvalchemi.data.Batch` with
# :meth:`~nvalchemi.data.Batch.from_data_list`. Optionally pass ``device`` or
# ``exclude_keys`` to omit certain attributes.

data_list = [
    AtomicData(
        positions=torch.randn(2, 3),
        atomic_numbers=torch.ones(2, dtype=torch.long),
        energy=torch.tensor([[0.0]]),
    ),
    AtomicData(
        positions=torch.randn(3, 3),
        atomic_numbers=torch.ones(3, dtype=torch.long),
        energy=torch.tensor([[0.0]]),
    ),
    AtomicData(
        positions=torch.randn(1, 3),
        atomic_numbers=torch.ones(1, dtype=torch.long),
        energy=torch.tensor([[0.0]]),
    ),
]
batch = Batch.from_data_list(data_list)

# exclude_keys: e.g. skip a key when batching
data_with_extra = AtomicData(
    positions=torch.randn(2, 3),
    atomic_numbers=torch.ones(2, dtype=torch.long),
)
data_with_extra.add_node_property("skip_me", torch.zeros(2, 1))
batch_slim = Batch.from_data_list([data_with_extra], exclude_keys=["skip_me"])
print(f"Batch num_graphs={batch.num_graphs}, num_nodes={batch.num_nodes}")

# %%
# Batch — Size and shape properties
# ----------------------------------

print(f"num_graphs={batch.num_graphs}, batch_size={batch.batch_size}")
print(f"num_nodes_list={batch.num_nodes_list}, num_edges_list={batch.num_edges_list}")
print(
    f"batch_idx (graph index per node) shape: {batch.batch_idx.shape}, batch_ptr: {batch.batch_ptr.tolist()}"
)
print(f"max_num_nodes={batch.max_num_nodes}")

# %%
# Batch — Reconstructing graphs
# ------------------------------

first = batch.get_data(0)
last = batch.get_data(-1)
all_graphs = batch.to_data_list()
print(
    f"get_data(0).num_nodes={first.num_nodes}, get_data(-1).num_nodes={last.num_nodes}"
)
print(f"len(to_data_list())={len(all_graphs)}")

# %%
# Batch — Indexing (single graph, sub-batch, attribute)
# ------------------------------------------------------

one = batch[0]
sub = batch[1:3]
sub2 = batch[torch.tensor([0, 2])]
sub3 = batch[[0, 2]]
mask = torch.tensor([True, False, True])
sub4 = batch[mask]
positions_tensor = batch["positions"]
print(f"batch[0] num_nodes={one.num_nodes}, batch[1:3] num_graphs={len(sub)}")
print(f"batch[[0,2]] num_nodes_list={sub3.num_nodes_list}")
print(f"batch['positions'].shape={positions_tensor.shape}")

# %%
# Batch — Containment, length, iteration
# ---------------------------------------

print(f"'positions' in batch: {'positions' in batch}")
print(f"len(batch)={len(batch)}")
keys_from_iter = [k for k, _ in batch]
print(f"Keys from iteration (sample): {keys_from_iter[:3]}")

# %%
# Batch — Setting attributes and adding keys
# ------------------------------------------

batch.add_key(
    "node_feat",
    [torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)],
    level="node",
)
batch.add_key(
    "temperature",
    [torch.tensor([[300.0]]), torch.tensor([[350.0]]), torch.tensor([[400.0]])],
    level="system",
)
data_a = AtomicData(
    positions=torch.randn(2, 3),
    atomic_numbers=torch.ones(2, dtype=torch.long),
    neighbor_list=torch.tensor([[0, 1]], dtype=torch.long),
)
data_b = AtomicData(
    positions=torch.randn(3, 3),
    atomic_numbers=torch.ones(3, dtype=torch.long),
    neighbor_list=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
)
batch_with_edges = Batch.from_data_list([data_a, data_b])
batch_with_edges.add_key(
    "edge_attr",
    [torch.randn(1, 4), torch.randn(2, 4)],
    level="edge",
)
print(
    f"After add_key: 'node_feat' in batch, 'temperature' in batch, 'edge_attr' in batch_with_edges: {'node_feat' in batch, 'temperature' in batch, 'edge_attr' in batch_with_edges}"
)

# %%
# Batch — Append and append_data
# -------------------------------

# ``extra`` only carries the default atomic fields. It does not contain the
# custom ``node_feat`` or ``temperature`` keys added above. Current
# ``Batch.append()`` behavior keeps only the intersection of keys present in
# both batches within each shared storage group, so custom keys missing from
# the appended batch are dropped from the combined batch. The later round-trip
# section therefore reports ``node_feat=False`` by design: the key is already
# gone before ``to_data_list()`` runs.

extra = Batch.from_data_list(
    [
        AtomicData(
            positions=torch.randn(2, 3), atomic_numbers=torch.ones(2, dtype=torch.long)
        ),
    ]
)
batch.append(extra)
print(f"After append: num_graphs={batch.num_graphs}")
print(
    f"'node_feat' survived append: {'node_feat' in batch}"
    "  (expected: False — extra batch lacks it)"
)

batch.append_data(
    [
        AtomicData(
            positions=torch.randn(1, 3), atomic_numbers=torch.ones(1, dtype=torch.long)
        ),
    ]
)
print(
    f"After append_data: num_graphs={batch.num_graphs}, num_nodes_list={batch.num_nodes_list}"
)

# Re-add node_feat (dropped by append) so the round-trip check at the end
# can demonstrate that custom properties survive to_data_list → from_data_list.
batch.add_key(
    "node_feat",
    [torch.randn(n, 4) for n in batch.num_nodes_list],
    level="node",
)

# %%
# Batch — put and defrag
# -----------------------


def _tiny_graph(energy: float):
    return AtomicData(
        positions=torch.randn(2, 3),
        atomic_numbers=torch.ones(2, dtype=torch.long),
        energy=torch.tensor([[energy]]),
    )


buffer = Batch.empty(
    num_systems=40, num_nodes=80, num_edges=80, template=_tiny_graph(0.0)
)
print(
    f"Empty buffer: num_graphs={buffer.num_graphs}, system_capacity={buffer.system_capacity}"
)

src_batch = Batch.from_data_list([_tiny_graph(1.0), _tiny_graph(2.0)])
mask = torch.tensor([True, False])
copied_mask = torch.zeros(2, dtype=torch.bool)
dest_mask = torch.zeros(buffer.system_capacity, dtype=torch.bool)
buffer.put(src_batch, mask, copied_mask=copied_mask, dest_mask=dest_mask)
print(
    f"After put: buffer has {buffer.num_graphs} graphs; copied_mask={copied_mask.tolist()}"
)

src_batch.defrag(copied_mask=copied_mask)
print(f"After defrag: src_batch has {src_batch.num_graphs} graph(s)")
print(f"Remaining graph energy: {src_batch['energy']}")

# %%
# Batch — Device, clone, contiguous, pin_memory
# -----------------------------------------------

batch_cpu = batch.to("cpu")
batch_cloned = batch.clone()
batch_contig = batch.contiguous()
batch_pinned = batch.pin_memory()
print(
    f"to('cpu').device: {batch_cpu.device}, clone is new: {batch_cloned is not batch}"
)
print(f"pin_memory: {batch_pinned['positions'].is_pinned()}")

# %%
# Batch — Serialization
# ----------------------

flat = batch.model_dump()
print("model_dump keys (sample):", list(flat.keys())[:6])
flat_slim = batch.model_dump(exclude_none=True)
print(f"model_dump(exclude_none=True) has 'device': {'device' in flat_slim}")

# %%
# Batch — Computing neighbors
# ----------------------------
# :func:`~nvalchemi.neighbors.compute_neighbors` populates a neighbor list
# on a batch in-place.  This is the recommended way to prepare a batch for
# model evaluation outside a dynamics loop (inside a dynamics loop, the
# :class:`~nvalchemi.dynamics.hooks.NeighborListHook` handles this
# automatically).
#
# You can pass a scalar ``cutoff`` directly, or pass a ``config`` object
# from a model's :attr:`~nvalchemi.models.base.ModelConfig.neighbor_config`.
# The result is written into the batch as ``neighbor_matrix`` /
# ``num_neighbors`` (MATRIX format) or ``neighbor_list`` (COO format).

from nvalchemi.neighbors import compute_neighbors

device = "cuda" if torch.cuda.is_available() else "cpu"
# Build a small periodic batch on GPU.
periodic_data = AtomicData(
    positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]),
    atomic_numbers=torch.tensor([1, 1, 1], dtype=torch.long),
    cell=torch.diag(torch.tensor([5.0, 5.0, 5.0])).unsqueeze(0),
    pbc=torch.tensor([[True, True, True]]),
    device=device,
)
periodic_batch = Batch.from_data_list([periodic_data], device=device)

# Option 1: pass a cutoff directly (max_neighbors auto-estimated).
compute_neighbors(periodic_batch, cutoff=3.0)
print(f"neighbor_matrix shape: {periodic_batch.neighbor_matrix.shape}")
print(f"num_neighbors: {periodic_batch.num_neighbors.tolist()}")

# Option 2: pass a model's neighbor config (preferred with models).
# config = model.model_config.neighbor_config
# compute_neighbors(periodic_batch, config=config)

# %%
# Round-trip summary
# ------------------
#
# ``to_data_list()`` preserves the current batch state faithfully. Since
# ``node_feat`` was dropped earlier by ``append()`` when we added graphs that
# did not carry that key, it is expected to remain absent after the round-trip.
reconstructed = batch.to_data_list()
batch_again = Batch.from_data_list(reconstructed)
print(
    f"Round-trip: num_graphs {batch.num_graphs} -> {len(reconstructed)} -> {batch_again.num_graphs}"
)
print(
    f"First graph has 'node_feat' after round-trip: {'node_feat' in reconstructed[0].model_dump()}"
)
