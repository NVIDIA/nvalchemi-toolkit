---
name: nvalchemi-data-storage
description: How to write, read, and load atomic data using nvalchemi's composable Zarr-backed storage pipeline (Writer, Reader, Dataset, DataLoader).
---

# nvalchemi Data Storage

## Overview

`nvalchemi` provides a composable pipeline for persisting and loading atomic data:

```
Writer                          Reader
(AtomicData/Batch -> Zarr)      (Zarr -> dict[str, Tensor])
                                    |
                                Dataset
                                (dict -> AtomicData, device transfer, prefetch)
                                    |
                                DataLoader
                                (AtomicData -> Batch, batching, iteration)
```

```python
from nvalchemi.data.datapipes import (
    AtomicDataZarrWriter,
    AtomicDataZarrReader,
    Dataset,
    DataLoader,
)
```

---

## Writing Data

`AtomicDataZarrWriter` serializes `AtomicData`, `list[AtomicData]`, or `Batch` into a Zarr store.

```python
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrWriter
import torch

writer = AtomicDataZarrWriter("dataset.zarr")

# Write a single system
data = AtomicData(
    positions=torch.randn(10, 3),
    atomic_numbers=torch.ones(10, dtype=torch.long),
    energy=torch.tensor([[0.5]]),
)
writer.write(data)

# Write a list of systems
writer.write([data1, data2, data3])

# Write a Batch
batch = Batch.from_data_list([data1, data2])
writer.write(batch)
```

### Appending to an existing store

```python
writer = AtomicDataZarrWriter("dataset.zarr")
writer.append(new_data)          # single AtomicData
writer.append([data1, data2])    # list
writer.append(batch)             # Batch
```

### Adding custom arrays

```python
writer.add_custom("my_feature", torch.randn(total_atoms, 32), level="atom")
```

### Deleting and defragmenting

```python
writer.delete([0, 2])   # soft-delete samples 0 and 2 (sets mask=False)
writer.defragment()      # rebuild store without deleted samples
```

### Zarr store layout

```
dataset.zarr/
├── meta/
│   ├── atoms_ptr       # int64 [N+1] — cumulative node counts
│   ├── edges_ptr       # int64 [N+1] — cumulative edge counts
│   ├── samples_mask    # bool [N] — False = deleted
│   ├── atoms_mask      # bool [V_total]
│   └── edges_mask      # bool [E_total]
├── core/               # AtomicData fields
│   ├── atomic_numbers
│   ├── positions
│   └── ...
├── custom/             # user-defined arrays
└── .zattrs             # root metadata
```

---

## Reading Data

### Low-level: AtomicDataZarrReader

Returns raw `dict[str, torch.Tensor]` per sample with metadata.

```python
from nvalchemi.data.datapipes import AtomicDataZarrReader

reader = AtomicDataZarrReader(
    "dataset.zarr",
    pin_memory=False,                  # pin tensors to page-locked memory
    include_index_in_metadata=True,    # add "index" key to metadata
)

# Access a sample
data_dict, metadata = reader[0]       # (dict[str, Tensor], dict)

len(reader)          # number of active (non-deleted) samples
reader.field_names   # list of field names in each sample
reader.close()       # release resources
reader.refresh()     # reload after external modifications
```

### Mid-level: Dataset

Wraps a `Reader` and constructs `AtomicData` objects, with device transfer and prefetching.

```python
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset

reader = AtomicDataZarrReader("dataset.zarr")
ds = Dataset(
    reader,
    device="cuda",       # target device ("auto" picks CUDA if available)
    num_workers=2,       # thread pool size for prefetching
)

# Get a sample
atomic_data, metadata = ds[0]   # AtomicData on target device

# Lightweight metadata (no full construction)
num_atoms, num_edges = ds.get_metadata(0)

len(ds)    # number of samples
ds.close()

# Context manager
with Dataset(reader, device="cuda") as ds:
    data, meta = ds[0]
```

### Prefetching with CUDA streams

```python
ds = Dataset(reader, device="cuda")

# Prefetch a single sample
stream = torch.cuda.Stream()
ds.prefetch(0, stream=stream)
atomic_data, meta = ds[0]   # waits for prefetch to complete

# Prefetch multiple samples
streams = [torch.cuda.Stream() for _ in range(4)]
ds.prefetch_batch([0, 1, 2, 3], streams=streams)

# Cancel pending prefetches
ds.cancel_prefetch()       # cancel all
ds.cancel_prefetch(0)      # cancel specific index
```

### High-level: DataLoader

Iterates over a `Dataset` in batches, producing `Batch` objects.

```python
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, DataLoader

reader = AtomicDataZarrReader("dataset.zarr")
ds = Dataset(reader, device="cuda", num_workers=4)

loader = DataLoader(
    ds,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    sampler=None,              # optional torch Sampler
    prefetch_factor=2,         # batches to prefetch ahead
    num_streams=4,             # CUDA streams for prefetching
    use_streams=True,          # enable stream prefetching
)

for batch in loader:
    # batch is a Batch with concatenated tensors on target device
    print(batch.num_graphs, batch.num_nodes)

len(loader)                    # number of batches
loader.set_epoch(epoch)        # for distributed sampler
```

---

## Custom Readers

Subclass `Reader` to support additional storage formats.

```python
from nvalchemi.data.datapipes.backends.base import Reader

class MyReader(Reader):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load raw tensor dict for a single sample."""
        ...

    def __len__(self) -> int:
        """Total number of samples."""
        ...

    # Optional overrides:
    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Per-sample metadata (default: empty dict)."""
        ...

    def _get_field_names(self) -> list[str]:
        """List of field names in each sample."""
        ...

    def close(self):
        """Release resources."""
        ...
```

Custom readers plug directly into `Dataset` and `DataLoader`:

```python
reader = MyReader("data/", pin_memory=True)
ds = Dataset(reader, device="cuda")
loader = DataLoader(ds, batch_size=16)
```

---

## Full Workflow Example

```python
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import (
    AtomicDataZarrWriter,
    AtomicDataZarrReader,
    Dataset,
    DataLoader,
)

# --- Write ---
data_list = [
    AtomicData(
        positions=torch.randn(n, 3),
        atomic_numbers=torch.ones(n, dtype=torch.long),
        energy=torch.tensor([[float(i)]]),
    )
    for i, n in enumerate([5, 8, 3, 12])
]

writer = AtomicDataZarrWriter("train.zarr")
writer.write(data_list)

# Append more later
writer.append(AtomicData(
    positions=torch.randn(6, 3),
    atomic_numbers=torch.ones(6, dtype=torch.long),
))

# --- Read & Train ---
reader = AtomicDataZarrReader("train.zarr")
ds = Dataset(reader, device="cuda", num_workers=4)
loader = DataLoader(ds, batch_size=2, shuffle=True, prefetch_factor=2)

for epoch in range(10):
    loader.set_epoch(epoch)
    for batch in loader:
        energy = batch["energy"]          # [batch_size, 1]
        positions = batch["positions"]    # [total_nodes, 3]
        # ... model forward pass ...
```
