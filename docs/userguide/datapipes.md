<!-- markdownlint-disable MD014 -->

(datapipes_guide)=

# Data Loading Pipeline

The toolkit ships a data loading pipeline designed for GPU-accelerated atomistic
workloads. It is built from four composable pieces: a **Reader** that pulls raw
tensors from storage, a **Dataset** that validates them into
{py:class}`nvalchemi.data.AtomicData` objects, a **DataLoader** that batches them
into {py:class}`nvalchemi.data.Batch` objects, and an optional **Sampler** that
controls batching strategy. Each layer adds exactly one concern, and you can swap
any of them independently.

```{note}
The ``datapipes`` abstraction is shared with ``physicsnemo``: there are some
specializations in ``nvalchemi`` for CSR-type data, but in the near-term
we will merge implementations.
```

## Reader: raw tensor I/O

A {py:class}`~nvalchemi.data.datapipes.backends.base.Reader` is the simplest
abstraction in the pipeline. It knows how to load a single sample from storage and
return it as a plain `dict[str, torch.Tensor]` --- no validation, no device
transfers, no threading. Readers are intentionally minimal so that adding a new
storage backend only requires implementing two methods:

- `_load_sample(index) -> dict[str, torch.Tensor]`: Reads one sample into CPU tensors.
- `__len__() -> int`: Returns the total number of available samples.

The built-in reader is
{py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader`, which
reads from the structured Zarr stores produced by the toolkit's
{py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`. The Zarr
layout uses separate groups for core fields, metadata, and custom attributes, and
supports soft-deletes via a validity mask.

If your data lives in a different format (HDF5, LMDB, a collection of files), you
can subclass `Reader` and implement `_load_sample` and `__len__`. Everything
downstream --- Dataset, DataLoader, Sampler --- will work without changes.

## Dataset: validation and prefetching

{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` wraps a Reader and adds two
responsibilities:

1. **Validation**: Raw dictionaries are validated into
   {py:class}`nvalchemi.data.AtomicData` objects, catching schema issues early.
2. **Async prefetching**: A background `ThreadPoolExecutor` loads and transfers
   samples to the target device ahead of time, so the GPU is never starved.

```python
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader

reader = AtomicDataZarrReader("/path/to/store.zarr")
dataset = Dataset(reader=reader, device="cuda:0", num_workers=4)

# Fetch a single sample (AtomicData on GPU)
data, metadata = dataset[0]
```

### CUDA stream prefetching

When called by a DataLoader, the Dataset uses
{py:meth}`~nvalchemi.data.datapipes.dataset.Dataset.prefetch` to overlap
host-to-device data transfers with compute. The DataLoader issues prefetch calls on
non-default CUDA streams; the Dataset records the transfer and synchronises the
stream before returning the data. This means the next batch is already on the GPU
while the model is processing the current one.

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
    batch_size=32,
    prefetch_factor=2,
    num_streams=2,
)

for batch in loader:
    # batch is a Batch on the dataset's target device
    outputs = model(batch)
```

Key parameters:

| Parameter | Purpose |
|---------------------|--------------------------------------------------------------|
| `batch_size` | Number of graphs per batch |
| `prefetch_factor` | How many **batches** to load ahead of the current one |
| `num_streams` | Number of CUDA streams used for overlapping transfers |
| `sampler` | Controls index ordering (defaults to sequential or random) |

Unlike PyTorch's `torch.utils.data.DataLoader`, this implementation returns
{py:class}`nvalchemi.data.Batch` objects (disjoint graphs with proper node-index
offsets) rather than generic collated tensors.

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
dataset = Dataset(reader=reader, device="cuda:0", num_workers=4)
sampler = SizeAwareSampler(dataset=dataset, max_atoms=4096)

loader = DataLoader(
    dataset=dataset,
    batch_size=64,      # upper bound; sampler may produce smaller batches
    sampler=sampler,
    prefetch_factor=2,
    num_streams=2,
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
