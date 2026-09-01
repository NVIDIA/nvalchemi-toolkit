<!-- markdownlint-disable MD014 -->

(data_guide)=

# AtomicData and Batch

The ALCHEMI Toolkit represents molecular systems as **graphs**: atoms are nodes, and
optional edges (e.g. bonds or radius-cutoff neighbors) connect them. The
{py:class}`nvalchemi.data.AtomicData` class holds a single graph (one molecule or
structure), and {py:class}`nvalchemi.data.Batch` batches many such graphs into one
structure for efficient GPU-friendly training and inference.

```{tip}
**AI coding assistant?** Load the ``nvalchemi-data-structures``
{ref}`agent skill <agent_skills>` for concise instructions on creating,
manipulating, and batching ``AtomicData`` objects.
```

## AtomicData: a single graph

{py:class}`nvalchemi.data.AtomicData` is a Pydantic model that stores:

- **Required**: `positions` (shape `[n_nodes, 3]`) and `atomic_numbers` (shape `[n_nodes]`).
- **Optional node-level**: e.g. `atomic_masses`, `forces`, `velocities`, `node_attrs`.
- **Optional edge-level**: `neighbor_list` (shape `[n_edges, 2]`) and edge attributes such
as `shifts` (Cartesian displacements) and
`neighbor_list_shifts` (integer lattice indices) for periodicity.
- **Optional system-level**: `energy`, `cell`, `pbc`, `stress`, `virial`, etc.

For stress, virial, and pressure sign conventions, see {ref}`conventions`.

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

### Grouping graphs

A batch can optionally tag contiguous runs of graphs as **groups** — a single
logical unit such as the images of one NEB path — via
{py:meth}`~nvalchemi.data.batch.Batch.set_group_layout`. This is unrelated to
the internal atoms/edges/system **storage groups** described later in *How
Batch stores data internally*.

```python
batch.set_group_layout(torch.tensor([4, 4, 1, 1, 1]))
print(batch.group_idx)  # tensor([0, 0, 1, 1, 1]) - normalized to dense IDs
```

The derived {py:attr}`~nvalchemi.data.batch.Batch.group_layout` property
lazily builds and caches a
{py:class}`~nvalchemi.data.group_layout.GroupLayout`, mapping between graph,
node, and group cardinalities (`graph_rank`, `node_to_group`, `group_ptr`,
`num_graphs_per_group`), plus mask/broadcast helpers `reduce_all`,
`reduce_any`, `broadcast`, `graph_mask`, and `selected_group_idx`. The cache
invalidates automatically whenever `group_idx` changes or graph membership
mutates (selection, `zero`, `put`, `defrag`, ...). A selection that leaves
group numbering non-dense must be repaired with
{py:meth}`~nvalchemi.data.batch.Batch.normalize_group_idx` before
`group_layout` is accessed again. `append()` requires both batches grouped or
both ungrouped and rebases labels, while `append_data()` is rejected on a
grouped batch.

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

A `Batch` organizes tensor fields by their cardinality within each atomic system.
Each field belongs to a **level**, which determines how many logical entities it
contains per system and how those values are packed across the batch. The three
built-in levels cover the usual atomic-data layout:

| Built-in level | Kind | Per-system cardinality | Examples |
|---|---|---:|---|
| `system` | Uniform | `1` | `cell`, `pbc`, `energy`, `stress` |
| `atoms` | Segmented | `N_i` atoms | `positions`, `atomic_numbers`, `forces` |
| `edges` | Segmented | `E_i` edges | `neighbor_list`, `shifts`, `edge_embeddings` |

A uniform level contributes one row per system. A segmented level may contribute a
different number of rows from each system. For example, all positions are packed
into one tensor, and an atom pointer records the slice owned by each system. Edge
data uses the same segmented representation with a separate pointer.

### Defining custom levels

Use {py:class}`~nvalchemi.data.LevelSchema` when data needs cardinalities beyond the
built-in atom, edge, and system levels. A schema can register:

- A custom **uniform** level with one entity per system.
- A custom **segmented** level with an arbitrary entity count per system.
- A **product** level representing the Cartesian product of two ordered segmented
  base levels.

The example below adds a custom `constraints` level to the schema and fills it with
a variable number of constraint records per system. Each record is represented here
by an eight-element feature vector.

```python
import torch

from nvalchemi.data import AtomicData, Batch, LevelSchema

data_list = [
    AtomicData(
        positions=torch.randn(num_atoms, 3),
        atomic_numbers=torch.ones(num_atoms, dtype=torch.long),
    )
    for num_atoms in (2, 3)
]

schema = LevelSchema()
schema.add_level("constraints", segmented=True)

print(schema.level_names)
# ('atoms', 'edges', 'system', 'constraints')
print(schema.level_kind("constraints"))  # 'segmented'

constraint_batch = Batch.from_data_list(data_list, attr_map=schema)
constraint_values = [torch.randn(4, 8), torch.randn(1, 8)]
constraint_batch.add_key(
    "constraint_features",
    constraint_values,
    level="constraints",
)

print(constraint_batch.level_ptr("constraints").tolist())  # [0, 4, 5]
print(constraint_batch.constraint_features.shape)  # torch.Size([5, 8])
```

The two systems contain four and one constraint records. These counts do not need to
match their atom or edge counts.

Registering a level defines its cardinality semantics; it does not automatically
assign arbitrary tensor fields to that level. There are three ways to classify a
custom field:

1. For a field already present on each `AtomicData`, call
   `schema.set("constraint_features", "constraints")` before constructing the
   batch.
2. Pass `field_levels={"constraint_features": "constraints"}` to
   {py:meth}`~nvalchemi.data.Batch.from_data_list`.
3. Add values after construction with
   `batch.add_key("constraint_features", values, level="constraints")`, as in the
   example above.

All three routes record field ownership in the batch-owned schema. The explicit
`field_levels` form is useful when the input objects should remain unchanged.

### Storing a Hessian

An atom-by-atom Hessian can use a separate product level whose left and right parents
are both the built-in `atoms` level:

```python
hessian_schema = LevelSchema()
hessian_schema.add_product_level(
    "atom_atom",
    left="atoms",
    right="atoms",
)

print(hessian_schema.level_kind("atom_atom"))  # 'product'
```

The `atom_atom` level can hold atom-pair blocks from a Hessian. For a system with
`N_i` atoms, a field on this level has logical shape `[N_i, N_i, 3, 3]`: the first
two axes select the ordered atom pair, and the final axes contain the Cartesian
second-derivative block. Product parent order is part of the definition; for
different parents, `left × right` and `right × left` have different logical axis
orders. Both parents must be registered ordinary segmented levels; products cannot
use a uniform or product parent.

Position Hessians returned by PyTorch normally interleave the atom and Cartesian
axes as `[N_i, 3, N_i, 3]`. Move the second atom axis next to the first before
storing the Cartesian `3 x 3` blocks on the product level:

```python
def quadratic_energy(positions):
    return positions.square().sum()


raw_hessians = [
    torch.func.hessian(quadratic_energy)(data.positions) for data in data_list
]
# Each raw Hessian has shape [N_i, 3, N_i, 3].
hessian_blocks = [
    hessian.permute(0, 2, 1, 3).contiguous() for hessian in raw_hessians
]
# Each stored value now has shape [N_i, N_i, 3, 3].

hessian_batch = Batch.from_data_list(data_list, attr_map=hessian_schema)
hessian_batch.add_key("hessian_blocks", hessian_blocks, level="atom_atom")

print(hessian_batch.level_ptr("atom_atom").tolist())  # [0, 4, 13]
print(hessian_batch.hessian_blocks.shape)  # torch.Size([13, 3, 3])
print(hessian_batch.get_data(0).hessian_blocks.shape)  # torch.Size([2, 2, 3, 3])
```

The two systems contain two and three atoms, so their Hessians contribute four and
nine atom-pair blocks. The supplied blocks are packed without padding inside the
batch and restored with their two atom axes by
{py:meth}`~nvalchemi.data.Batch.get_data` and
{py:meth}`~nvalchemi.data.Batch.to_data_list`. `Batch` stores and reconstructs these
values; it does not compute the Hessian.

| Field kind | Per-system logical shape | Packed `Batch` shape |
|---|---|---|
| Uniform | `[1, H]` | `[B, H]` |
| Segmented | `[S_i, H]` | `[sum(S_i), H]` |
| Product | `[L_i, R_i, H]` | `[sum(L_i * R_i), H]` |

{py:attr}`~nvalchemi.data.Batch.level_keys` reports every cardinality-resolved level
and its fields in schema order. A fieldless resolved level appears with an empty set.
{py:meth}`~nvalchemi.data.Batch.level_ptr` returns its cumulative per-system pointer
as an `int32` tensor. This limit applies to each materialized level in one in-memory
`Batch`; constructing a level whose packed cardinality exceeds the signed `int32`
range raises an error. Zarr uses `int64` pointers for dataset-wide offsets, so the
total dataset may contain more entities than a single `Batch`. The existing
{py:attr}`~nvalchemi.data.Batch.keys` property remains the node, edge, and system
compatibility view and does not list custom levels.

### Fieldless cardinality metadata

A level's cardinality can be known even when it owns no tensor field. This is useful
when a product field is the only value that exposes one of its parent dimensions:

```python
schema = LevelSchema()
schema.add_level("augmented", segmented=True)
schema.add_product_level(
    "atom_augmented",
    left="atoms",
    right="augmented",
)

augmented_batch = Batch.from_data_list(data_list, attr_map=schema)
augmented_batch.add_key(
    "atom_augmented_features",
    [torch.randn(2, 4, 8), torch.randn(3, 5, 8)],
    level="atom_augmented",
)

print(augmented_batch.level_keys["augmented"])          # set()
print(augmented_batch.level_ptr("augmented").tolist())  # [0, 4, 9]
```

In this example the augmented counts happen to be `N_i + 2`. The schema does not
store or evaluate that expression: the product tensors supply the concrete counts
four and five. Registered segmented and product levels are otherwise materialized
lazily, when a field or product axis establishes their cardinality. Zero-length
parent and product axes are valid.

Custom definitions and segment boundaries are preserved through reconstruction,
selection, append, device movement, buffering, and point-to-point batch transport.
Custom segmented payloads remain opaque: only the built-in `neighbor_list` receives
automatic atom-index offsets. A custom edge-like tensor that contains indices must
manage its own index semantics.

Custom point-to-point transport uses the receiver's template to determine message
order, field dtypes, and payload shapes. The sender and receiver must therefore use
independently matching templates; the transport does not negotiate or validate two
different layouts at runtime.

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

For a `Batch` template with custom levels, pass an element capacity for each
payload-bearing custom segmented or product level:

```python
custom_buffer = Batch.empty(
    num_systems=64,
    num_nodes=4096,
    num_edges=32768,
    template=hessian_batch,
    level_capacities={
        "atom_atom": 8192,
    },
    device="cuda",
)
```

Custom uniform levels use `num_systems`. Product capacity is explicit and is not
derived from either parent capacity. A fieldless segmented parent needs pointer
metadata but no element-buffer entry in `level_capacities`. Use
{py:meth}`~nvalchemi.data.Batch.empty_like` to create an empty buffer that
preserves an existing batch's complete materialized layout and capacities.

The generic buffer kernels support `bool`, `float32`, `float64`, `int32`, and
`int64` payloads, and built-in fields may use any of those dtypes. Custom uniform
levels use the same set. For custom segmented and product levels,
{py:meth}`~nvalchemi.data.Batch.put` currently accepts only `float32` payloads.
Unsupported or mismatched dtypes are rejected before any batch group is modified.
This narrower buffer-copy policy does not change the dtype support of ordinary,
tightly packed `Batch` objects.

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

## Persisting custom levels in Zarr

{py:class}`~nvalchemi.data.AtomicDataZarrWriter` persists a batch's complete level
schema. Custom fields are stored under `levels/<level>/`, while cumulative pointers
for resolved custom segmented and product levels are stored under
`meta/level_ptrs/`. Product payloads stay packed on disk and are restored to their
two logical parent axes when read.

The root `attrs["levels"]` entry contains locally versioned custom-level definitions.
This version applies only to that metadata; it is not a store-wide format version.
A batch containing only the built-in levels retains the existing `core/`, `custom/`,
and built-in pointer layout and does not create custom-level metadata.

The example below writes two ordinary systems, then adds a variable-length
`constraints` field. The pointer is a complete physical-store prefix pointer: the
two systems own two and three constraint rows, respectively.

```python
import torch

from nvalchemi.data import (
    AtomicData,
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    Batch,
    Dataset,
    LevelSchema,
)

systems = [
    AtomicData(
        positions=torch.zeros(num_atoms, 3),
        atomic_numbers=torch.ones(num_atoms, dtype=torch.long),
    )
    for num_atoms in (2, 3)
]

writer = AtomicDataZarrWriter("features.zarr")
writer.write(Batch.from_data_list(systems))

schema = LevelSchema()
schema.add_level("constraints", segmented=True)
constraint_ptr = torch.tensor([0, 2, 5], dtype=torch.int64)
constraint_values = torch.arange(40, dtype=torch.float32).reshape(5, 8)
writer.add_custom(
    "constraint_features",
    constraint_values,
    "constraints",
    attr_map=schema,
    level_ptrs={"constraints": constraint_ptr},
)

reader = AtomicDataZarrReader("features.zarr")
stored_schema = reader.level_schema
dataset = Dataset(reader, device="cpu")
loaded_batch = dataset.load_batches([[0, 1]])[0]

print(loaded_batch.level_ptr("constraints").tolist())  # [0, 2, 5]
print(loaded_batch.constraint_features.shape)  # torch.Size([5, 8])
print(loaded_batch.get_data(1).constraint_features.shape)  # torch.Size([3, 8])
print(loaded_batch.get_data(1).constraint_features[0].tolist())
# [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
```

`reader.level_schema` returns an independent schema. `Dataset` and
{py:class}`~nvalchemi.data.InMemoryDataset` propagate field-bearing custom levels
and fieldless parents whose counts can be recovered from product payload axes. A
level represented only by a stored pointer, with no field or product payload, is
preserved by Zarr but is not reconstructed by those dataset batching paths.

{py:meth}`~nvalchemi.data.AtomicDataZarrWriter.add_custom` can add a field on a new
custom level to an existing store by accepting its `attr_map` and complete physical
store prefix pointers through `level_ptrs`. Each pointer must have one entry more
than the number of physical samples and must cover deleted rows as well as active
samples. Definitions, pointers, and tensor conversion are validated before new
level groups or arrays are created. Once a definition and its pointers are stored,
later fields on the same level can reuse them.

Legacy stores are opened without migration or rewriting. Conversely, Toolkit
versions that predate custom levels do not know how to recover fields stored in the
new `levels/` groups, even though the store's legacy `core/` and `custom/` contents
retain their existing representation.

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
- **Transforms**: See the [Transforms section](datapipes_guide) of the Data
  Loading Pipeline guide for how to hook per-sample and per-batch transforms
  into {py:class}`~nvalchemi.data.datapipes.dataset.Dataset` and
  {py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader`.
