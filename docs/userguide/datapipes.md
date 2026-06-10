<!-- markdownlint-disable MD014 -->

(datapipes_guide)=

# Data Loading Pipeline

The toolkit ships a data loading pipeline designed for GPU-accelerated atomistic
workloads. It is built from four composable pieces: a **Reader** that pulls raw
tensors from storage, a **Dataset** that validates them into
{py:class}`nvalchemi.data.AtomicData` objects, a **DataLoader** that batches them
into {py:class}`nvalchemi.data.Batch` objects, and an optional **Sampler** that
controls batching strategy. A **MultiDataset** can also compose several datasets
behind one global index space. Each layer adds exactly one concern, and you can
swap any of them independently.

```{note}
The ``datapipes`` abstraction is shared with ``physicsnemo``: there are some
specializations in ``nvalchemi`` for CSR-type data, but in the near-term
we will merge implementations.
```

```{tip}
**AI coding assistant?** Load the ``nvalchemi-data-storage``
{ref}`agent skill <agent_skills>` for concise instructions on writing,
reading, and loading atomic data through the Zarr-backed storage pipeline.
```

## Reader: raw tensor I/O

A {py:class}`~nvalchemi.data.datapipes.backends.base.Reader` is the storage-facing
layer of the pipeline. It returns plain `dict[str, torch.Tensor]` objects on CPU:
no `AtomicData` validation, no device transfers, no batching policy, and no
threading. That separation keeps storage backends focused on I/O and lets
samplers decide *which* samples to request without also needing to know how the
store should be read efficiently.

Readers expose two public loading methods:

- `read(index)`: Load one sample and return `(raw_tensor_dict, metadata)`.
- `read_many(indices)`: Load several samples in the requested order and return one
  `(raw_tensor_dict, metadata)` pair per requested index.

Both public methods attach per-sample metadata and optionally pin CPU tensors when
`pin_memory=True`. Index validity is a backend concern: for example, the Zarr
reader supports negative logical indices and maps them through its active-sample
mask, while another backend may choose different index semantics.

`read_many` has an ordered contract: results must align one-for-one with the
requested indices. Backends can reorder internally for physical I/O, but they must
restore the caller's requested order before returning.

Backend authors implement one or both raw loading hooks:

- `_load_sample(index) -> dict[str, torch.Tensor]`: Simple single-sample path.
- `_load_many_samples(indices) -> list[dict[str, torch.Tensor]]`: Batch-oriented
  path for amortizing I/O across many requested samples.
- `__len__() -> int`: Total number of available logical samples.

For simple formats, implementing `_load_sample` is enough; the base `Reader`
implements `read_many` by looping over `_load_sample`. Readers that only have an
efficient batch path can implement `_load_many_samples`; the base single-sample
hook can call it with a one-index request. For storage formats with high per-call
overhead or chunk locality, implement `_load_many_samples` so the backend can
sort, merge, cache, or otherwise coalesce physical reads before returning samples
in the caller's original order.

The built-in reader is
{py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader`, which
reads from the structured Zarr stores produced by the toolkit's
{py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`. The Zarr
layout uses separate groups for core fields, metadata, and custom attributes, and
supports soft-deletes via a validity mask.

`AtomicDataZarrReader` implements `_load_many_samples` as the fast path. Given a
shuffled list of logical indices, it maps them to physical sample positions, sorts
by physical order, groups reads by Zarr chunk locality, loads each array in
coalesced ranges or orthogonal selections, and then restores the caller's requested
sample order. This is why downstream code should prefer `read_many` for batches
instead of looping over `read`.

```{tip}
The writer supports per-group compression and chunking via
{py:class}`~nvalchemi.data.datapipes.ZarrWriteConfig`. See the
[Zarr Compression Tuning Guide](zarr_compression_guide) for codec
recommendations and storage estimates.
```

If your data lives in a different format (HDF5, LMDB, a collection of files), you
can subclass `Reader` and implement the hook that matches the backend. Everything
downstream --- Dataset, DataLoader, Sampler --- will work without changes.

```python
from collections.abc import Sequence

import torch

from nvalchemi.data.datapipes.backends.base import Reader


class MyReader(Reader):
    def __len__(self) -> int:
        return 10_000

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        # Good enough for simple formats or true random-access stores.
        return load_one_sample(index)


class MyBatchOptimizedReader(Reader):
    def __len__(self) -> int:
        return 10_000

    def _load_many_samples(
        self, indices: Sequence[int]
    ) -> list[dict[str, torch.Tensor]]:
        # Use backend-specific locality here, then return results in the same
        # logical order as ``indices``.
        return load_samples_with_coalesced_io(indices)
```

## Dataset: validation and prefetching

{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` wraps a Reader and adds two
responsibilities:

1. **Validation**: Raw dictionaries are validated into
   {py:class}`nvalchemi.data.AtomicData` objects, catching schema issues early.
   Pass `skip_validation=True` to bypass Pydantic validation when the backing
   store is already known to be well-formed (see
   [Read performance tuning](read_performance_tuning)).
2. **Async prefetching**: A background `ThreadPoolExecutor` loads and transfers
   samples to the target device ahead of time, reducing stalls while the model
   consumes previous batches.

The Dataset talks to readers through public `reader.read_many(...)`. This is true
even when a caller asks for one sample: single-sample Dataset access is
represented as a one-element read request, so batch-capable readers keep their
optimized path and Dataset does not need to know backend-specific private hooks.
Duck-typed readers can be used without inheriting from `Reader` if they implement
`read_many`, `__len__`, and `close`.

```python
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader

reader = AtomicDataZarrReader("/path/to/store.zarr")
dataset = Dataset(
    reader=reader,
    device="cuda:0",
    num_workers=1,
    skip_validation=True,  # use for trusted stores written by the toolkit
)

# Fetch a single sample (AtomicData on GPU)
data, metadata = dataset[0]
```

### Batch loading and CUDA stream prefetching

When called by a DataLoader, the Dataset can overlap host-to-device transfers with
compute. The DataLoader issues prefetch calls on non-default CUDA streams; the
Dataset records the transfer and synchronises the stream before returning the data.
This means the next batch can already be on the GPU while the model is processing
the current one.

The canonical synchronous batch API is
{py:meth}`~nvalchemi.data.datapipes.dataset.Dataset.load_batches`. It accepts a
sequence of batch-index lists and returns one {py:class}`nvalchemi.data.Batch` per
input list. Even for a single emitted batch, this path goes through one
`reader.read_many(...)` request so batch-capable readers can use the same
coalesced I/O implementation everywhere:

```python
batches = dataset.load_batches([[0, 4, 2], [8, 1, 3]])
batch0, batch1 = batches
```

For asynchronous loader iteration, the important path is fused prefetch:
{py:meth}`~nvalchemi.data.datapipes.dataset.Dataset.prefetch_fused_batches`
accepts several upcoming DataLoader batches, flattens their indices into one
`reader.read_many(...)` request, and then splits the loaded samples back into the
original batch boundaries. This improves I/O throughput without requiring the
sampler to choose storage-friendly windows.

### Lightweight metadata access

Samplers often need to know sample sizes (how many atoms? how many edges?) before
deciding which samples to group into a batch.
{py:meth}`~nvalchemi.data.datapipes.dataset.Dataset.get_metadata` returns
`(num_atoms, num_edges)` for a given index without constructing the full
`AtomicData`, keeping the overhead low.

## DataLoader: batching and iteration

{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader` ties the pipeline
together. It requests indices from a Sampler, fetches `AtomicData` objects from
the Dataset, and collates them into {py:class}`nvalchemi.data.Batch` objects for
consumption by the model.

```python
from nvalchemi.data.datapipes.dataloader import DataLoader

loader = DataLoader(
    dataset=dataset,
    batch_size=64,
    prefetch_factor=16,
    num_streams=2,
    use_streams=True,
    pin_memory=True,
)

for batch in loader:
    # batch is a Batch on the dataset's target device
    outputs = model(batch)
```

Key parameters:

| Parameter | Purpose |
|---------------------|--------------------------------------------------------------|
| `batch_size` | Number of graphs per batch |
| `prefetch_factor` | How many **batches** to fuse into each background read ([tuning guide](read_performance_tuning)) |
| `num_streams` | Number of CUDA streams used for overlapping transfers |
| `use_streams` | Whether to enable CUDA-stream prefetching when CUDA is available |
| `pin_memory` | Request page-locked CPU tensors from readers that support pinned memory |
| `sampler` | Controls index ordering (defaults to sequential or random) |
| `batch_sampler` | Supplies complete batches of indices and overrides `batch_size`, `shuffle`, and `sampler` |

Unlike PyTorch's `torch.utils.data.DataLoader`, this implementation returns
{py:class}`nvalchemi.data.Batch` objects (disjoint graphs with proper node-index
offsets) rather than generic collated tensors.

### Batch throughput and fused prefetch

`batch_size` controls the number of samples emitted to the training loop.
`prefetch_factor` controls how many emitted batches are fused into one background
backend read. For positive `prefetch_factor`, together they define the effective
read window:

```text
effective_read_window = batch_size * prefetch_factor
```

For example, `batch_size=64` and `prefetch_factor=16` produces batches of 64
graphs for the model, but the reader sees read requests of up to 1,024 logical
indices. The model-facing batch size stays unchanged; only the storage access
window grows.

This distinction is useful for graph-like data with shuffled access:

- Samplers remain semantic: they decide ordering and batch membership based on
  training needs, size limits, or distributed partitioning.
- Readers remain physical: they can exploit chunk locality, sort by physical
  position, merge adjacent ranges, and amortize per-call overhead.
- Dataset and DataLoader connect the two by converting several upcoming batches
  into one larger `read_many` request, then yielding the original batch sequence.

Use `prefetch_factor=0` to disable fused prefetch and issue one backend read per
emitted batch. This is useful for debugging or for stores where large read windows
do not help. For shuffled Zarr training reads, start with `prefetch_factor=16` or
`32`, then benchmark with `nvalchemi-io-test` on a representative store. Enable
`pin_memory=True` for CUDA training so the DataLoader requests page-locked CPU
tensors before asynchronous transfer. See
[Read performance tuning](read_performance_tuning) and the
[I/O benchmark tool](io_benchmark_section) for concrete commands.

## MultiDataset: composing datasets

{py:class}`~nvalchemi.data.datapipes.multidataset.MultiDataset` concatenates
multiple {py:class}`~nvalchemi.data.datapipes.dataset.Dataset` instances behind
one global index space. It follows the PhysicsNeMo multidataset indexing contract
while preserving the nvalchemi batch fast path: `load_batches(...)` routes each
global batch to the relevant child datasets and recombines mixed-child batches in
the requested sample order.

```python
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    DataLoader,
    Dataset,
    MultiDataset,
    MultiDatasetBatchSampler,
)

dataset_a = Dataset(AtomicDataZarrReader("dataset_a.zarr"), device="cuda:0")
dataset_b = Dataset(AtomicDataZarrReader("dataset_b.zarr"), device="cuda:0")
dataset = MultiDataset(dataset_a, dataset_b, output_strict=True)

batch_sampler = MultiDatasetBatchSampler.balanced(
    dataset,
    batch_size=64,
    epoch_policy="max_size",
    replacement=True,
)

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    prefetch_factor=16,
    pin_memory=True,
)
```

By default, `output_strict=True` requires all non-empty child datasets to expose
the same field names. Empty children are skipped when choosing the reference
field set. Use `output_strict=False` only when downstream code can handle
source-specific fields.

### Multidataset sampler policies

The multidataset samplers operate on global indices but allocate samples at the
child-dataset level:

| Sampler | Use case |
|---------------------|--------------------------------------------------------------|
| {py:class}`~nvalchemi.data.datapipes.samplers.MultiDatasetSampler` | Draw individual samples from child datasets at custom rates |
| {py:class}`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler` | Build batches with explicit or weighted per-dataset allocations |
| {py:meth}`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler.balanced` | Build dataset-balanced batches |

`samples_per_dataset` accepts integer counts or floating-point relative ratios.
For example, `[1.0, 3.0]` allocates roughly one quarter of each batch to the first
dataset and three quarters to the second dataset.

When `num_batches` is omitted, `epoch_policy` controls the default epoch length:

| `epoch_policy` | Behavior |
|----------------|----------|
| `"dataset_size"` | Preserve the historical default based on total dataset size |
| `"min_size"` | Stop when the smallest contributing dataset would be exhausted |
| `"max_size"` | Run until the largest contributing dataset is covered, oversampling smaller datasets when `replacement=True` |

Use `"max_size"` for balanced training over datasets of different sizes when you
want smaller datasets to be oversampled instead of dominated by the largest
dataset. Without replacement, `"max_size"` raises if oversampling would be
required.

For data-parallel training, the multidataset samplers can shard sample or batch
orders across ranks. See {ref}`distributed_manager_guide` for examples ranging
from the default `DDPHook` sampler injection to distributed
`MultiDatasetBatchSampler` composition.

## Transforms: per-sample and per-batch hooks

Both {py:class}`~nvalchemi.data.datapipes.dataset.Dataset` and
{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader` accept user-supplied
transforms that run on the output path. A *per-sample* transform operates on
individual {py:class}`~nvalchemi.data.AtomicData` objects (plus a metadata dict)
after device transfer but before collation, while a *per-batch* transform
operates on the collated {py:class}`~nvalchemi.data.Batch` before it is yielded
to the caller. Note, however, that {py:class}`~nvalchemi.data.transforms.Compose`
dispatches based on function arguments; their signatures should (must) match
the examples given below:

```python
# Per-sample: runs inside Dataset, one AtomicData at a time
def shift_positions(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    return data.replace(positions=data.positions + 1.0), metadata

dataset = Dataset(reader=reader, device="cuda:0", transforms=[shift_positions])

# Per-batch: runs inside Dataloader, once per yielded Batch.
def center_batch(batch: Batch) -> Batch:
    # compute the mean position per sample
    mean_pos = segmented_mean(batch.positions, batch.batch_idx)
    # subtract the mean per graph with broadcasting
    batch.positions = batch.positions - mean_pos.view(...)
    return batch

loader = DataLoader(dataset=dataset, batch_size=32, batch_transforms=[center_batch])
```

```{tip}
Prefer per-batch transforms over per-sample transforms for anything
compute-heavy. Per-sample transforms run once per graph on whatever device the
:class:`~nvalchemi.data.datapipes.dataset.Dataset` produces and cannot amortize
launch overhead across graphs. A vectorized per-batch transform that uses
segmented / scatter-reduce operations on the fully collated
:class:`~nvalchemi.data.Batch` will be significantly more efficient on GPU. Reserve
per-sample transforms for light, sample-specific bookkeeping (e.g. attaching
metadata, filtering keys) that genuinely cannot be batched.
```

Each sequence is composed left-to-right via
{py:class}`~nvalchemi.data.transforms.Compose`, so the output of transform *i*
becomes the input of transform *i + 1*. Transforms must **return** their
(possibly mutated) output ---- in-place mutation without a return value is not
supported. The two hooks are independent: you can use one, both, or neither.

```{note}
A single {py:class}`~nvalchemi.data.transforms.Compose` is either a sample
composition or a batch composition; mixing the two shapes inside one
sequence raises ``TypeError`` at construction. In practice you never
instantiate ``Compose`` yourself --- pass the list to ``Dataset(transforms=...)``
or ``DataLoader(batch_transforms=...)`` and the right wrapper is built for you.
```

## SizeAwareSampler: memory-safe batching

For datasets where systems vary widely in size --- a common situation in atomistic
ML --- a fixed `batch_size` can either waste GPU memory (when graphs are small) or
cause out-of-memory errors (when a few large graphs land in the same batch).

{py:class}`~nvalchemi.dynamics.sampler.SizeAwareSampler` solves this with
capacity-aware bin-packing:

```python
from nvalchemi.dynamics.sampler import SizeAwareSampler

sampler = SizeAwareSampler(
    dataset=dataset,
    max_atoms=4096,
    max_edges=32768,
    max_batch_size=64,
)
```

Instead of grouping a fixed count of graphs, the sampler fills each batch until
adding the next sample would exceed one of the capacity constraints (`max_atoms`,
`max_edges`, or `max_batch_size`). Internally it uses bin-packing by atom count: it
sorts samples into bins of similar size, then draws from bins in a way that
maximises GPU utilisation while respecting the limits.

### GPU memory heuristic

If you omit `max_atoms`, the sampler can estimate a safe limit from the GPU's
available memory fraction. This is useful for workloads where the optimal batch size
depends on the hardware.

### Inflight replacement

In dynamics pipelines, systems converge and leave the batch at different times.
{py:meth}`~nvalchemi.dynamics.sampler.SizeAwareSampler.request_replacement` finds a
new sample whose size fits the memory slot left by a graduated system, keeping the
batch full without reallocation.

## Putting it all together

A typical end-to-end setup:

```python
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.dynamics.sampler import SizeAwareSampler

reader = AtomicDataZarrReader("/path/to/store.zarr")
dataset = Dataset(
    reader=reader,
    device="cuda:0",
    num_workers=1,
    skip_validation=True,
)
sampler = SizeAwareSampler(dataset=dataset, max_atoms=4096)

loader = DataLoader(
    dataset=dataset,
    batch_size=64,      # upper bound; sampler may produce smaller batches
    sampler=sampler,
    prefetch_factor=16,
    num_streams=2,
    use_streams=True,
    pin_memory=True,
)

for batch in loader:
    # batch.num_atoms <= 4096 guaranteed
    outputs = model(batch)
```

## See also

- **Storage guide**: See {py:class}`~nvalchemi.data.AtomicDataZarrWriter` and
  {py:class}`~nvalchemi.data.AtomicDataZarrReader` for writing and reading Zarr stores.
- **API**: {py:mod}`nvalchemi.data` for the full datapipe API reference.
- **Dynamics**: The [Dynamics](dynamics_guide) guide shows how the DataLoader and
  SizeAwareSampler integrate with simulation pipelines.
- **Compression**: The [Zarr Compression Tuning Guide](zarr_compression_guide)
  covers how to configure compression and chunking when writing Zarr stores.
- **I/O benchmark**: The [I/O benchmark tool](io_benchmark_section) lets you
  measure write throughput, readback throughput, and compression ratios on
  synthetic data before choosing a configuration.
