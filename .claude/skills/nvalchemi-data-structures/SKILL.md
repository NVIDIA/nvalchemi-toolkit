---
name: nvalchemi-data-structures
description: How to use AtomicData and Batch — the core graph-based data structures for representing atomic systems and batching them for GPU computation.
---

# nvalchemi Data Structures

## Overview

`nvalchemi` represents atomic systems as graphs using two core classes:

- **`AtomicData`** — a single atomic system (molecule, crystal, etc.)
- **`Batch`** — an efficient container of multiple `AtomicData` objects stored as concatenated tensors

Both are Pydantic `BaseModel` subclasses with `DataMixin` for device/dtype operations.

```python
from nvalchemi.data import AtomicData, Batch
```

---

## AtomicData

### Construction

**Required fields:** `positions` `[n_nodes, 3]` and `atomic_numbers` `[n_nodes]`.

```python
import torch

# Minimal
data = AtomicData(
    positions=torch.randn(4, 3),
    atomic_numbers=torch.tensor([1, 6, 6, 1], dtype=torch.long),
)

# With edges (bonds or neighbor list)
data = AtomicData(
    positions=torch.randn(4, 3),
    atomic_numbers=torch.tensor([1, 6, 6, 1], dtype=torch.long),
    neighbor_list=torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long),
)

# With system-level fields (energy, cell, pbc)
data = AtomicData(
    positions=torch.randn(4, 3),
    atomic_numbers=torch.tensor([1, 6, 6, 1], dtype=torch.long),
    energy=torch.tensor([[0.5]]),
    cell=torch.eye(3).unsqueeze(0),       # [1, 3, 3]
    pbc=torch.tensor([[True, True, False]]),  # [1, 3]
)
```

**From ASE Atoms:**

```python
data = AtomicData.from_atoms(
    atoms,                    # ase.Atoms object
    energy_key="energy",      # key in atoms.info / atoms.calc
    forces_key="forces",
    device="cpu",
    dtype=torch.float32,
)
```

### Field reference

Fields are organized by level. All are optional except `positions` and `atomic_numbers`.

| Level  | Field              | Shape              | Notes                              |
|--------|--------------------|--------------------|-------------------------------------|
| Node   | `atomic_numbers`   | `[V]`             | Required, int64                     |
| Node   | `positions`        | `[V, 3]`          | Required, float                     |
| Node   | `atomic_masses`    | `[V]`             | Auto-populated from periodic table  |
| Node   | `atom_categories`  | `[V]`             | Defaults to zeros                   |
| Node   | `forces`           | `[V, 3]`          | eV/Angstrom                         |
| Node   | `velocities`       | `[V, 3]`          | Auto-initialized to zeros           |
| Node   | `momenta`          | `[V, 3]`          |                                     |
| Node   | `charges`          | `[V, 1]`          |                                     |
| Node   | `node_embeddings`  | `[V, H]`          |                                     |
| Node   | `kinetic_energies` | `[V, 1]`          |                                     |
| Edge   | `neighbor_list`    | `[E, 2]`          | COO format, int64                   |
| Edge   | `shifts`           | `[E, 3]`          | Cartesian displacements (`neighbor_list_shifts @ cell`) |
| Edge   | `neighbor_list_shifts` | `[E, 3]`       | Integer lattice image indices       |
| Edge   | `edge_embeddings`  | `[E, H]`          |                                     |
| Dense  | `neighbor_matrix`  | `[V, K]`          | Dense neighbor matrix (int64)       |
| Dense  | `neighbor_matrix_shifts` | `[V, K, 3]` | Periodic shifts for dense neighbors |
| Dense  | `num_neighbors`    | `[V]`             | Valid neighbor count per atom        |
| System | `cell`             | `[1, 3, 3]`       | Lattice vectors                     |
| System | `pbc`              | `[1, 3]`          | Periodic boundary conditions (bool) |
| System | `energy`           | `[1]`             | eV                                  |
| System | `stress`           | `[1, 3, 3]`       | eV/Angstrom^3                       |
| System | `virial`           | `[1, 3, 3]`       |                                     |
| System | `dipole`           | `[1, 3]`          |                                     |
| System | `charge`           | `[1]`             |                                     |
| System | `graph_embeddings` | `[1, H]`          |                                     |

Custom data can be stored in the `info: dict[str, torch.Tensor]` field.

### Properties

```python
data.num_nodes          # int — number of atoms
data.num_edges          # int — number of edges (0 if None)
data.device             # torch.device
data.dtype              # torch.dtype (of positions)
data.chemical_hash      # str — blake2s hash of structure/composition
data.node_properties    # dict of set node-level fields
data.edge_properties    # dict of set edge-level fields
data.system_properties  # dict of set system-level fields
```

### Dict-like access

```python
data["positions"]                # get attribute by name
data["positions"] = new_tensor   # set attribute by name
```

### Adding custom properties

```python
data.add_node_property("custom_feat", torch.randn(data.num_nodes, 4))
data.add_edge_property("edge_weights", torch.ones(data.num_edges))
data.add_system_property("temperature", torch.tensor([[300.0]]))
```

### Device, clone, serialization

```python
data.to("cuda")                         # move to device
data.to("cpu", dtype=torch.float64)     # move + cast
data.cpu()
data.cuda()
data.clone()                            # deep copy
data.model_dump(exclude_none=True)      # dict
data.model_dump_json()                  # JSON string
```

### Equality

Two `AtomicData` objects are equal if they have the same `chemical_hash`:

```python
data1 == data2  # compares by chemical_hash
```

---

## Batch

### Construction

```python
data_list = [
    AtomicData(positions=torch.randn(2, 3), atomic_numbers=torch.ones(2, dtype=torch.long)),
    AtomicData(positions=torch.randn(3, 3), atomic_numbers=torch.ones(3, dtype=torch.long)),
]
batch = Batch.from_data_list(data_list)

# Exclude specific keys
batch = Batch.from_data_list(data_list, exclude_keys=["velocities"])

# Pre-allocated empty buffer (for high-performance use)
buffer = Batch.empty(
    num_systems=40, num_nodes=80, num_edges=80,
    template=data_list[0],  # defines schema
)
```

### Size properties

```python
batch.num_graphs            # number of graphs
batch.batch_size            # alias for num_graphs
batch.num_nodes             # total nodes across all graphs
batch.num_edges             # total edges across all graphs
batch.batch_idx             # Tensor [num_nodes] — per-node graph index
batch.batch_ptr             # Tensor [num_graphs+1] — cumulative node counts
batch.num_nodes_list        # list[int] — per-graph node counts
batch.num_edges_list        # list[int] — per-graph edge counts
batch.num_nodes_per_graph   # Tensor — per-graph node counts
batch.num_edges_per_graph   # Tensor — per-graph edge counts
batch.max_num_nodes         # int — max nodes in any graph
batch.system_capacity       # int — max graphs for pre-allocated batches
```

### Indexing

```python
# Single graph -> AtomicData
batch[0]
batch[-1]
batch.get_data(0)

# Sub-batch -> Batch
batch[1:3]                          # slice
batch[torch.tensor([0, 2])]        # int tensor
batch[[0, 2]]                       # list
batch[torch.tensor([True, False, True])]  # bool mask

# Attribute -> Tensor
batch["positions"]                  # concatenated positions from all graphs

# Reconstruct all graphs
all_graphs = batch.to_data_list()   # list[AtomicData]
```

### Containment, length, iteration

```python
"positions" in batch       # True
len(batch)                 # num_graphs
for key, tensor in batch:  # iterate (key, value) pairs
    ...
```

### Mutation

```python
# Add a new key (one value per graph)
batch.add_key("node_feat", [torch.randn(2, 4), torch.randn(3, 4)], level="node")
batch.add_key("temperature", [torch.tensor([[300.0]]), torch.tensor([[350.0]])], level="system")
batch.add_key("edge_attr", [torch.randn(1, 4), torch.randn(2, 4)], level="edge")

# Overwrite an existing key
batch.add_key("node_feat", new_values, level="node", overwrite=True)

# Concatenate batches (in-place)
batch.append(other_batch)
batch.append_data([more_atomic_data])
```

### Pre-allocated buffer operations

For high-throughput workflows (e.g. streaming dynamics), use pre-allocated buffers:

```python
# Create buffer
buffer = Batch.empty(num_systems=40, num_nodes=80, num_edges=80, template=data)

# Copy selected graphs into buffer
mask = torch.tensor([True, False])           # which src graphs to copy
copied_mask = torch.zeros(2, dtype=torch.bool)  # updated in-place: which actually fit
dest_mask = torch.zeros(buffer.system_capacity, dtype=torch.bool)
buffer.put(src_batch, mask, copied_mask=copied_mask, dest_mask=dest_mask)

# Remove copied graphs from source (compact in-place)
src_batch.defrag(copied_mask=copied_mask)

# Reset buffer for reuse
buffer.zero()
```

### Device, clone, memory

```python
batch.to("cuda")
batch.cpu()
batch.cuda()
batch.clone()
batch.contiguous()     # make all tensors contiguous
batch.pin_memory()     # pin for async host-to-device transfer
```

### Serialization

```python
batch.model_dump()                    # flat dict of all tensors + metadata
batch.model_dump(exclude_none=True)   # drop None-valued keys
batch.model_dump_json()               # JSON string
```

### Distributed communication

`Batch` supports point-to-point distributed communication via `torch.distributed`. Data is sent in three phases: a metadata header (`num_graphs`, `num_nodes`, `num_edges`), per-group segment lengths, and bulk tensor data.

**Blocking send/recv:**

```python
import torch.distributed as dist

# Sender (rank 0)
batch.send(dst=1, tag=0, group=None)

# Receiver (rank 1) — template provides schema (keys, dtypes, group structure)
received = Batch.recv(src=0, device="cuda", template=template_batch, tag=0)
```

**Non-blocking send/recv:**

```python
# Sender — returns _BatchSendHandle
handle = batch.isend(dst=1, tag=0, group=None)
# ... do other work ...
handle.wait()  # block until all sends complete

# Receiver — returns _BatchRecvHandle
handle = Batch.irecv(src=0, device="cuda", template=template_batch, tag=0)
# ... do other work ...
received = handle.wait()  # block until data arrives, returns Batch
```

**Key details:**

- `template` is required on the receiver to know the attribute keys, dtypes, and group structure (atoms/edges/system). Cache it across calls.
- A 0-graph (sentinel) batch can be sent/received — only the metadata header is transmitted.
- `tag` is a base tag; it is incremented internally per group. Use distinct base tags for concurrent send/recv pairs.
- `empty_like(batch)` creates a 0-graph batch with the same schema — useful for sentinel signals.

```python
sentinel = Batch.empty_like(batch, device="cuda")  # 0-graph, same schema
sentinel.send(dst=1)  # signal "no more data"
```

### Round-trip

```python
reconstructed = batch.to_data_list()
batch_again = Batch.from_data_list(reconstructed)
```
