<!-- markdownlint-disable MD014 -->

(data_guide)=

# AtomicData and Batch

The ALCHEMI Toolkit represents molecular systems as **graphs**: atoms are nodes, and
optional edges (e.g. bonds or radius-cutoff neighbors) connect them. The
{py:class}`nvalchemi.data.AtomicData` class holds a single graph (one molecule or
structure), and {py:class}`nvalchemi.data.Batch` batches many such graphs into one
structure for efficient GPU-friendly training and inference.

## AtomicData: a single graph

{py:class}`nvalchemi.data.AtomicData` is a Pydantic model that stores:

- **Required**: `positions` (shape `[n_nodes, 3]`) and `atomic_numbers` (shape `[n_nodes]`).
- **Optional node-level**: e.g. `atomic_masses`, `forces`, `velocities`, `node_attrs`.
- **Optional edge-level**: `edge_index` (shape `[2, n_edges]`) and edge attributes such
as `shifts` for periodicity.
- **Optional system-level**: `energies`, `cell`, `pbc`, `stresses`, `virials`, etc.

All tensor fields use PyTorch tensors, so you can move them to GPU with `.to(device)` or
use the mixin method {py:meth}`nvalchemi.data.data.DataMixin.to` for device/dtype changes.

Example:

```python
import torch
from nvalchemi.data import AtomicData

positions = torch.randn(5, 3)
atomic_numbers = torch.tensor([1, 6, 6, 1, 8], dtype=torch.long)
data = AtomicData(positions=positions, atomic_numbers=atomic_numbers)

# Optional: add system-level labels
data = AtomicData(
    positions=positions,
    atomic_numbers=atomic_numbers,
    energies=torch.tensor([[0.0]]),
)
```

Properties such as `num_nodes`, `num_edges`, and `device` are available; optional
fields default to `None` when not provided.

## Batch: multiple graphs

{py:class}`nvalchemi.data.Batch` is built from a **list** of {py:class}`nvalchemi.data.AtomicData`
instances. Node tensors are concatenated along the first dimension; edge tensors are
concatenated with node-index offsets so each graph’s edges refer to the correct atoms.
System-level tensors are stacked so that the first dimension is the number of graphs.

- Build a batch: {py:meth}`nvalchemi.data.batch.Batch.from_data_list`\ (data_list).
- Access batch size: `num_graphs`, `num_nodes`, `num_edges`, `num_nodes_list`, `num_edges_list`.
- Recover a single graph: {py:meth}`nvalchemi.data.batch.Batch.get_data`\ (index).
- Recover all graphs: {py:meth}`nvalchemi.data.batch.Batch.to_data_list`\ ().

Example:

```python
import torch
from nvalchemi.data import AtomicData, Batch

data_list = [
    AtomicData(
        positions=torch.randn(2, 3),
        atomic_numbers=torch.ones(2, dtype=torch.long),
        energies=torch.zeros(1, 1),
    ),
    AtomicData(
        positions=torch.randn(3, 3),
        atomic_numbers=torch.ones(3, dtype=torch.long),
        energies=torch.zeros(1, 1),
    ),
]
batch = Batch.from_data_list(data_list)

print(batch.num_graphs, batch.num_nodes, batch.num_nodes_list)  # 2, 5, [2, 3]
first = batch.get_data(0)
again = batch.to_data_list()
```

### Indexing and selection

`Batch` supports bracket indexing that mirrors familiar Python and PyTorch
conventions. The type of index determines what you get back:

| Index type | Returns | Example |
|------------|---------|---------|
| `str` | The raw tensor attribute by name | `batch["positions"]` |
| `int` | A single {py:class}`~nvalchemi.data.AtomicData` (via `get_data`) | `batch[0]` |
| `slice` | A new {py:class}`~nvalchemi.data.Batch` with the selected graphs | `batch[1:3]` |
| `Tensor` / `list[int]` | A new {py:class}`~nvalchemi.data.Batch` with the selected graphs | `batch[torch.tensor([0, 2])]` |

When selecting multiple graphs (slice, tensor, or list), the underlying
{py:meth}`~nvalchemi.data.batch.Batch.index_select` method operates directly on the
concatenated storage --- it slices segments and adjusts `edge_index` offsets without
reconstructing individual `AtomicData` objects, so it is efficient even for large
batches.

```python
# Select a sub-batch of graphs 0 and 2
sub = batch[torch.tensor([0, 2])]
print(sub.num_graphs)  # 2

# String indexing accesses the raw concatenated tensor
all_positions = batch["positions"]  # shape (total_nodes, 3)
```

## Adding keys to a batch

You can add new tensor keys (e.g. model outputs or extra labels) at node, edge, or
system level with {py:meth}`nvalchemi.data.batch.Batch.add_key`. The new key is then
available on the underlying storage and when you call {py:meth}`nvalchemi.data.batch.Batch.get_data`
or {py:meth}`nvalchemi.data.batch.Batch.to_data_list`, so each {py:class}`nvalchemi.data.AtomicData`
gets the correct slice.

```python
batch.add_key("node_feat", [torch.randn(2, 4), torch.randn(3, 4)], level="node")
batch.add_key(
    "energies",
    [torch.tensor([[0.1]]), torch.tensor([[0.2]])],
    level="system",
    overwrite=True,
)
list_of_data = batch.to_data_list()
# list_of_data[i] now has "node_feat" and "energies" with the right shapes.
```

## Device and serialization

- **Device**: Use {py:meth}`nvalchemi.data.batch.Batch.to`\ (device) or the mixin
  {py:meth}`nvalchemi.data.data.DataMixin.to` on {py:class}`nvalchemi.data.AtomicData`.
  The batch implementation delegates to the underlying storage for efficiency.
- **Serialization**: {py:class}`nvalchemi.data.AtomicData` supports Pydantic
  serialization (e.g. `model_dump`, `model_dump_json`). Tensor fields are serialized
  to lists in JSON mode.

## How Batch stores data internally

When you call {py:meth}`nvalchemi.data.batch.Batch.from_data_list`, the resulting
`Batch` does not simply stack all tensors along a new "batch" axis. Different kinds
of data need different layouts, and the toolkit uses a storage model that reflects
this.

Every tensor attribute belongs to one of three **levels**:

| Level | Storage class | Shape convention | Examples |
|-----------|----------------------------|--------------------------------------|---------------------------------------------|
| **system** | {py:class}`~nvalchemi.data.level_storage.UniformLevelStorage` | First dim = number of graphs | `cell`, `pbc`, `energies`, `stresses` |
| **atoms** | {py:class}`~nvalchemi.data.level_storage.SegmentedLevelStorage` | Concatenated across graphs | `positions`, `atomic_numbers`, `forces` |
| **edges** | {py:class}`~nvalchemi.data.level_storage.SegmentedLevelStorage` | Concatenated across graphs | `edge_index`, `shifts`, `edge_embeddings` |

**Uniform storage** is straightforward: every graph contributes exactly one row, so
the i-th graph's data is always at index `i`. System-level properties like the
simulation cell or total energy work this way.

**Segmented storage** is designed for variable-length data. Positions, for example,
are concatenated into a single tensor of shape `(total_nodes, 3)`. To know where each
graph's atoms start and end, the storage tracks `segment_lengths` and a pointer array
`batch_ptr`. The i-th graph's nodes live at `positions[batch_ptr[i]:batch_ptr[i+1]]`.
Edge data works the same way, with node-index offsets automatically applied to
`edge_index` so that each graph's edges still point to the correct atoms in the
flattened array.

The mapping from attribute name to level is determined by a
{py:obj}`~nvalchemi.data.level_storage.DEFAULT_ATTRIBUTE_MAP`. When you add a new key with
{py:meth}`~nvalchemi.data.batch.Batch.add_key`, you explicitly specify the level
(`"node"`, `"edge"`, or `"system"`) so the batch knows how to slice it back out when
you call {py:meth}`~nvalchemi.data.batch.Batch.get_data`.

## Pre-allocated batches and the buffer API

For training and data loading, `from_data_list` creates a batch that fits its data
exactly. But in high-throughput dynamics simulations, you often need a **fixed-capacity
buffer** that you fill and drain without reallocating memory: this abstraction is
used in the dynamics pipeline abstraction for point-to-point data sample passing,
which bypasses the need for host and/or file I/O.

### Creating an empty buffer

{py:meth}`nvalchemi.data.batch.Batch.empty` allocates a batch with room for a
specified number of systems, nodes, and edges, but with zero graphs initially.
It requires a `template` ({py:class}`~nvalchemi.data.AtomicData` or
{py:class}`~nvalchemi.data.Batch`) that defines which keys to allocate and their
schema:

```python
template = AtomicData(
    positions=torch.zeros(1, 3),
    atomic_numbers=torch.zeros(1, dtype=torch.long),
    forces=torch.zeros(1, 3),
    energies=torch.zeros(1, 1),
    cell=torch.zeros(1, 3, 3),
    pbc=torch.zeros(1, 3, dtype=torch.bool),
)
buffer = Batch.empty(
    num_systems=64,
    num_nodes=4096,
    num_edges=32768,
    template=template,
    device="cuda",
)
```

All tensors are pre-allocated at the given capacity. The batch's `num_graphs` starts
at zero.

### Filling the buffer with `put`

{py:meth}`nvalchemi.data.batch.Batch.put` copies selected graphs from a source batch
into the buffer. A boolean `mask` selects which graphs to copy:

```python
# Copy the first two graphs from incoming_batch into buffer
mask = torch.tensor([True, True, False, False])
buffer.put(incoming_batch, mask)
```

The method performs capacity checks to make sure the incoming segments fit, and uses
optimized kernels for the data movement.

### Compacting with `defrag`

After graphs have been consumed (e.g. copied out to another stage), you remove them
with {py:meth}`nvalchemi.data.batch.Batch.defrag`. This compacts the remaining graphs
to the front of the buffer so that freed capacity is available again:

```python
# Mark which graphs have been copied out
copied_mask = torch.tensor([True, False, True])
buffer.defrag(copied_mask=copied_mask)
```

### Resetting with `zero`

{py:meth}`nvalchemi.data.batch.Batch.zero` resets the batch to zero graphs while
keeping the allocated memory in place --- useful at the start of a new epoch or
pipeline iteration.

These operations (`empty` / `put` / `defrag` / `zero`) form the backbone of the
dynamics pipeline's inflight batching, where systems enter and leave a running
simulation at different times.

## ASE Atoms interoperability

The [Atomic Simulation Environment (ASE)](https://ase-lib.org/about.html) is the
most widely-used Python library for representing and manipulating atomistic systems.
The toolkit provides a conversion path so you can move data between ASE and ALCHEMI
seamlessly.

### Converting ASE Atoms to AtomicData

{py:meth}`nvalchemi.data.AtomicData.from_atoms` accepts an `ase.Atoms` object and
returns an {py:class}`nvalchemi.data.AtomicData`:

```python
from ase.build import molecule
from nvalchemi.data import AtomicData

atoms = molecule("H2O")
data = AtomicData.from_atoms(atoms, device="cpu")
```

The conversion maps ASE fields to ALCHEMI fields:

| ASE source | AtomicData field | Notes |
|-------------------------------|----------------------|-----------------------------------------------|
| `atoms.numbers` | `atomic_numbers` | |
| `atoms.positions` | `positions` | |
| `atoms.get_pbc()` | `pbc` | Reshaped to `(1, 3)` |
| `atoms.get_cell()` | `cell` | Reshaped to `(1, 3, 3)` |
| `atoms.info[energy_key]` | `energies` | Default key: `"energy"` |
| `atoms.arrays[forces_key]` | `forces` | Default key: `"forces"` |
| `atoms.info[stress_key]` | `stresses` | Voigt vector converted to `(1, 3, 3)` matrix |
| `atoms.info[virials_key]` | `virials` | Voigt vector converted to `(1, 3, 3)` matrix |
| `atoms.get_tags()` | `atom_categories` | 0 = GAS, 1 = SURFACE, 2+ = BULK |
| `atoms.get_masses()` | `atomic_masses` | |
| `atoms.info` (remaining) | preserved | Filtered to tensors, arrays, and floats |

Keyword arguments (`energy_key`, `forces_key`, etc.) let you adapt to different
naming conventions in your ASE dataset.

### Building a Batch from a list of Atoms

There is no special bulk constructor --- compose the two operations:

```python
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch

atoms_list = [molecule("H2O"), molecule("CH4")]
batch = Batch.from_data_list([AtomicData.from_atoms(a) for a in atoms_list])
```

### Converting back to ASE Atoms

The core library does not provide a `to_atoms` method, since the reverse mapping is
application-specific (e.g. which `info` keys to preserve, how to handle missing
fields). The examples directory includes a utility function that demonstrates the
reconstruction:

```python
# From examples/04_ase_dynamics_example.py
from ase import Atoms

def data_to_atoms(data: AtomicData) -> Atoms:
    return Atoms(
        numbers=data.atomic_numbers.cpu().numpy(),
        positions=data.positions.cpu().numpy(),
        cell=data.cell.squeeze(0).cpu().numpy() if data.cell is not None else None,
        pbc=data.pbc.squeeze(0).cpu().numpy() if data.pbc is not None else None,
    )
```

```{tip}
Converting a ``Batch`` to ``ase.Atoms`` should convert to ``AtomicData`` first
via ``Batch.to_data_list``, and loop over individual ``AtomicData``
entries then.
```

## See also

- **Examples**: The gallery includes **AtomicData and Batch: Graph-structured molecular data**
  (``01_data_example.py``) for a runnable script.
- **API**: {py:mod}`nvalchemi.data` for the full API of AtomicData, Batch, and the
  zarr-based reader/writer and dataloader.
