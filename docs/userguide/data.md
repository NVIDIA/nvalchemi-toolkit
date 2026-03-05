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
- Index by graph: `batch[i]` returns the i-th {py:class}`nvalchemi.data.AtomicData`;
  `batch[torch.tensor([0, 2])]` or `batch[slice(0, 2)]` returns a sub-{py:class}`nvalchemi.data.Batch`.

Example:

```python
from nvalchemi.data import AtomicData, Batch

data_list = [
    AtomicData(positions=torch.randn(2, 3), atomic_numbers=torch.ones(2, dtype=torch.long)),
    AtomicData(positions=torch.randn(3, 3), atomic_numbers=torch.ones(3, dtype=torch.long)),
]
batch = Batch.from_data_list(data_list)

print(batch.num_graphs, batch.num_nodes, batch.num_nodes_list)  # 2, 5, [2, 3]
first = batch.get_data(0)
again = batch.to_data_list()
```

## Adding keys to a batch

You can add new tensor keys (e.g. model outputs or extra labels) at node, edge, or
system level with {py:meth}`nvalchemi.data.batch.Batch.add_key`. The new key is then
available on the underlying storage and when you call {py:meth}`nvalchemi.data.batch.Batch.get_data`
or {py:meth}`nvalchemi.data.batch.Batch.to_data_list`, so each {py:class}`nvalchemi.data.AtomicData`
gets the correct slice.

```python
batch.add_key("node_feat", [torch.randn(2, 4), torch.randn(3, 4)], level="node")
batch.add_key("energies", [torch.tensor([[0.1]]), torch.tensor([[0.2]])], level="system")
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

## See also

- **Examples**: The gallery includes **AtomicData and Batch: Graph-structured molecular data**
  (``01_data_example.py``) for a runnable script.
- **API**: {py:mod}`nvalchemi.data` for the full API of AtomicData, Batch, and the
  zarr-based reader/writer and dataloader.
