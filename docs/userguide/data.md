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
- **Optional edge-level**: `neighbor_list` (shape `[n_edges, 2]`) and edge attributes such
as `shifts` (Cartesian displacements) and
`neighbor_list_shifts` (integer lattice indices) for periodicity.
- **Optional system-level**: `energy`, `cell`, `pbc`, `stress`, `virial`, etc.

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
    energy=torch.tensor([[0.0]]),
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
        energy=torch.zeros(1, 1),
    ),
    AtomicData(
        positions=torch.randn(3, 3),
        atomic_numbers=torch.ones(3, dtype=torch.long),
        energy=torch.zeros(1, 1),
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
concatenated storage --- it slices segments and adjusts `neighbor_list` offsets without
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
    "energy",
    [torch.tensor([[0.1]]), torch.tensor([[0.2]])],
    level="system",
    overwrite=True,
)
list_of_data = batch.to_data_list()
# list_of_data[i] now has "node_feat" and "energy" with the right shapes.
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
| **system** | {py:class}`~nvalchemi.data.level_storage.UniformLevelStorage` | First dim = number of graphs | `cell`, `pbc`, `energy`, `stress` |
| **atoms** | {py:class}`~nvalchemi.data.level_storage.SegmentedLevelStorage` | Concatenated across graphs | `positions`, `atomic_numbers`, `forces` |
| **edges** | {py:class}`~nvalchemi.data.level_storage.SegmentedLevelStorage` | Concatenated across graphs | `neighbor_list`, `shifts`, `neighbor_list_shifts`, `edge_embeddings` |

**Uniform storage** is straightforward: every graph contributes exactly one row, so
the i-th graph's data is always at index `i`. System-level properties like the
simulation cell or total energy work this way.

**Segmented storage** is designed for variable-length data. Positions, for example,
are concatenated into a single tensor of shape `(total_nodes, 3)`. To know where each
graph's atoms start and end, the storage tracks `segment_lengths` and a pointer array
`batch_ptr`. The i-th graph's nodes live at `positions[batch_ptr[i]:batch_ptr[i+1]]`.
Edge data works the same way, with node-index offsets automatically applied to
`neighbor_list` so that each graph's edges still point to the correct atoms in the
flattened array.

The mapping from attribute name to level is determined by a
{py:obj}`~nvalchemi.data.level_storage.DEFAULT_ATTRIBUTE_MAP`. When you add a new key with
{py:meth}`~nvalchemi.data.batch.Batch.add_key`, you explicitly specify the level
(`"node"`, `"edge"`, or `"system"`) so the batch knows how to slice it back out when
you call {py:meth}`~nvalchemi.data.batch.Batch.get_data`.

## Neighbor list formats

The framework supports two neighbor list representations, configured via
{py:class}`~nvalchemi.models.base.NeighborConfig` and populated by
{py:class}`~nvalchemi.hooks.NeighborListHook` at the ``BEFORE_COMPUTE``
stage.

| | **MATRIX format** | **COO format** |
|---|---|---|
| **Edge indices** | `neighbor_matrix` `[N, K]` int32 | `neighbor_list` `[E, 2]` int32 |
| **Per-atom counts** | `num_neighbors` `[N]` int32 | *(derived via `edge_ptr`)* |
| **CSR pointer** | *(not used)* | `edge_ptr` `[N+1]` int32 |
| **PBC shifts** | `neighbor_matrix_shifts` `[N, K, 3]` int32 | `neighbor_list_shifts` `[E, 3]` int32 |
| **Padding value** | `N` (total atoms in batch) | *(no padding — sparse)* |
| **Configured via** | `NeighborConfig(format="matrix")` | `NeighborConfig(format="coo")` |
| **Used by** | Analytical-force models (LJ, Ewald, PME) | GNN-based models (MACE, etc.) |

Here `N` is the total number of atoms in the batch, `K` is the maximum
number of neighbors per atom (``max_neighbors``), and `E` is the total
number of edges.

**MATRIX format** is a dense representation where each atom has a
fixed-width row of `K` neighbor indices.  Unused slots are filled with the
sentinel value `N` (total atoms in the batch).  Valid neighbors for atom `i`
are in `neighbor_matrix[i, :num_neighbors[i]]`.  This format avoids
dynamic allocation and is used by analytical-force models that iterate over
pair interactions.

**COO format** is a sparse representation where each edge is an `(i, j)`
pair in `neighbor_list`.  This format is used by GNN-based models that
operate on edge features.  The per-atom CSR pointer `edge_ptr` is derived
on demand via the {py:attr}`~nvalchemi.data.Batch.edge_ptr` property.

Both formats are populated automatically by
{py:class}`~nvalchemi.hooks.NeighborListHook`.  The format is controlled
by the `format` field in the model's
{py:class}`~nvalchemi.models.base.NeighborConfig`.

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
    energy=torch.zeros(1, 1),
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

| ASE source | Field | Notes |
|---|---|---|
| `atoms.numbers` | `atomic_numbers` | Always populated |
| `atoms.positions` | `positions` | Always populated |
| `atoms.get_pbc()` | `pbc` | Reshaped to `(1, 3)` |
| `atoms.get_cell()` | `cell` | Reshaped to `(1, 3, 3)` |
| `atoms.info[energy_key]` | `energy` | `None` if absent; `(1, 1)` |
| `atoms.arrays[forces_key]` | `forces` | `None` if absent |
| `atoms.info[stress_key]` | `stress` | `None` if absent; Voigt → `(1, 3, 3)` |
| `atoms.info[virials_key]` | `virial` | `None` if absent; Voigt → `(1, 3, 3)` |
| `atoms.info[dipole_key]` | `dipole` | `None` if absent; `(1, 3)` |
| `atoms.arrays[charges_key]` | `charges` | `None` if absent; `(N,)` |
| `atoms.info["charge"]` | `charge` | `None` if absent; from per-atom sum |
| `atoms.get_masses()` | `atomic_masses` | Always populated |
| `atoms.info` (remaining) | `info` | Arrays, lists, ints, floats kept; bools/strings dropped |

Optional label fields (`energy`, `forces`, `stress`, `virial`, `dipole`,
`charges`, `charge`) are populated **only** when present in the ASE
object; otherwise they remain `None`. The input `atoms` object is **not** mutated.

Keyword arguments (`energy_key`, `forces_key`, etc.) let you adapt to different
naming conventions in your ASE dataset.

### Atom categories

{py:class}`~nvalchemi.data.AtomicData` has an optional `atom_categories` field
(shape `[n_nodes]`) that classifies atoms using the
{py:class}`~nvalchemi._typing.AtomCategory` enum. This is used by dynamics hooks
such as {py:class}`~nvalchemi.dynamics.hooks.FreezeAtomsHook`, which freezes atoms
marked as `AtomCategory.SPECIAL`.

`from_atoms` does **not** set `atom_categories` automatically --- you assign it after
construction based on your specific workflow. For example, in a slab+adsorbate
system you can use ASE tags to identify which atoms to freeze:

```python
import torch
from ase.build import fcc111, molecule
from nvalchemi.data import AtomicData
from nvalchemi._typing import AtomCategory

slab = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)
co = molecule("CO")
co.translate([slab.cell[0, 0] / 2, slab.cell[1, 1] / 3,
              slab.positions[:, 2].max() + 1.8])
system = slab + co

data = AtomicData.from_atoms(system)
tags = torch.tensor(system.get_tags())
# tag 0 = adsorbate (free), tag >= 1 = slab (freeze)
data.atom_categories = torch.where(
    tags > 0, AtomCategory.SPECIAL.value, AtomCategory.GAS.value
)
```

The full set of available categories is documented in
{py:class}`~nvalchemi._typing.AtomCategory`. For simple binary cases (free vs
frozen), the convention is `GAS` (0) for free atoms and `SPECIAL` (-1) for
frozen atoms.

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
# From examples/basic/03_ase_integration.py
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
  (``basic/01_data_structures.py``) for a runnable script.
- **API**: {py:mod}`nvalchemi.data` for the full API of AtomicData, Batch, and the
  zarr-based reader/writer and dataloader.
