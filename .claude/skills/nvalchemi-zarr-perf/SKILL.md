---
name: nvalchemi-zarr-perf
description: >
  Performance tuning for nvalchemi's Zarr-backed Reader, Dataset, and
  DataLoader pipeline. Use when configuring AtomicDataZarrReader, Dataset,
  DataLoader, ZarrWriteConfig, or nvalchemi-io-test for training/inference
  throughput, especially shuffled access, graph-like random access, fused
  prefetch, pinned memory, validation overhead, or Zarr chunk/shard choices.
---

# Zarr DataLoader Performance Tuning

Use this skill when optimizing nvalchemi Zarr reads or writing stores that will
later be read through the nvalchemi DataLoader.

## Current API model

The pipeline has clean ownership boundaries:

- `Reader`: storage I/O only. Returns raw CPU tensor dictionaries plus metadata.
- `Dataset`: validation, optional validation skipping, device transfer, and async
  prefetch orchestration. Its canonical explicit batch API is
  `load_batches(batch_index_lists)`.
- `DataLoader`: sampler/batch iteration, fused prefetch, stream usage, and batch
  construction.
- `MultiDataset`: global index composition over multiple Datasets while routing
  `load_batches` requests to child datasets.
- `Sampler` / `batch_sampler`: semantic sample order and batch membership. Do not
  rely on sampler windows to optimize storage I/O.

Reader public methods:

- `reader.read(index)`: one sample.
- `reader.read_many(indices)`: many samples, returned in the request order.

Reader backend hooks:

- `_load_sample(index)`: implement for simple single-sample formats.
- `_load_many_samples(indices)`: implement for batch-optimized formats.
- `__len__()`: total logical samples.

The base `Reader` owns metadata finalization and optional pinned memory. Index
validity is the concrete reader's responsibility. `AtomicDataZarrReader` supports
negative logical indices, maps through the active sample mask, and implements
`_load_many_samples` as the fast path.

## Recommended DataLoader setup

```python
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    Dataset,
    DataLoader,
)

reader = AtomicDataZarrReader("store.zarr")

dataset = Dataset(
    reader,
    device="cuda",
    num_workers=1,          # 1 is enough; concurrent Zarr reads contend
    skip_validation=True,   # safe when store was written by the toolkit
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    prefetch_factor=16,     # up to 64 * 16 = 1024 indices per backend read
    num_streams=2,
    use_streams=True,
    pin_memory=True,        # request pinned CPU tensors from the reader
)
```

Use `pin_memory=True` on `AtomicDataZarrReader(...)` directly only for manual
reader usage. For normal training, prefer `DataLoader(..., pin_memory=True)` so
the loader owns the transfer optimization.

## Key knobs

### `prefetch_factor` (DataLoader)

Controls how many emitted batches are fused into one backend read:

```text
effective_read_window = batch_size * prefetch_factor
```

For `batch_size=64, prefetch_factor=16`, the model still receives batches of 64
graphs, but the Zarr reader sees up to 1024 logical indices per `read_many`.

| Access pattern | Recommended `prefetch_factor` |
|----------------|------------------------------:|
| Sequential     |                           2-4 |
| Shuffled       |                         16-64 |
| Block-shuffle  |                           2-8 |

Use `prefetch_factor=0` to disable fused prefetch and issue one backend read per
emitted batch through `Dataset.load_batches([indices])`. This is useful for
debugging or for stores where larger windows do not help. Positive
`prefetch_factor` values use the async
`prefetch_fused_batches(...)` / `get_fused_batches()` path.

Manual batch reads should use:

```python
batches = dataset.load_batches([[0, 4, 2], [8, 1, 3]])
```

### `skip_validation` (Dataset)

Bypasses per-sample `AtomicData` Pydantic validation (~4 ms/sample).
Constructs `Batch` directly from raw tensor dicts via
`Batch.from_raw_dicts()`.

**Use when:** the store was written by `AtomicDataZarrWriter` or has been
validated externally.
**Do not use when:** the store contents are untrusted or from a third party.

### `num_workers` (Dataset)

Thread pool size for background Dataset prefetch work. Start with **1**.
Increase only if profiling shows CPU-side validation or device transfer is
underlapping and storage reads are not contending.

### `pin_memory` (DataLoader or Reader)

Pinned CPU tensors make async CPU-to-GPU transfer possible. Use with CUDA targets
and `use_streams=True`.

Normal path:

```python
loader = DataLoader(dataset, batch_size=64, pin_memory=True)
```

Manual reader path:

```python
reader = AtomicDataZarrReader("store.zarr", pin_memory=True)
data, metadata = reader.read(0)
```

## Writing stores for fast random reads

For shuffled training reads, avoid extremely large chunks unless reads are mostly
sequential. A practical starting point:

```python
from zarr.codecs import ZstdCodec

from nvalchemi.data.datapipes import (
    AtomicDataZarrWriter,
    ZarrWriteConfig,
    ZarrArrayConfig,
)

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        chunk_size=10_000,
        shard_size=500_000,
    ),
)
writer = AtomicDataZarrWriter("store.zarr", config=config)
```

Guidance:

- `chunk_size` is rows along dimension 0, not number of structures. Atom fields
  are stored on the total atom axis; edge fields on the total edge axis.
- Smaller chunks reduce single-sample read amplification but increase metadata
  and codec overhead.
- Sharding groups many chunks into fewer storage objects and is useful when small
  chunks would create too many files.
- Use `edge_chunk_size` / `edge_shard_size` in `nvalchemi-io-test` when edge
  arrays need different tuning from atom/system arrays.
- Zstd level 3 is a good default ratio/speed tradeoff. LZ4 is useful when write
  and decompression speed matter more than compression ratio.

## How the reader optimises random access

`AtomicDataZarrReader._load_many_samples(indices)` is the optimized path behind
public `reader.read_many(indices)`.

It currently:

1. Resolves logical indices through the active sample mask.
2. Sorts requests by physical sample index.
3. Groups physical positions by Zarr chunk locality.
4. Uses coalesced range reads when a small number of chunk-local runs exists.
5. Falls back to orthogonal selection for highly fragmented requests.
6. Restores the caller's original request order.

This is transparent to Dataset, DataLoader, and Samplers. Larger fused read
windows give the Zarr backend more indices to coalesce, which is why
`prefetch_factor` matters most for shuffled reads.

For multidataset training, use `MultiDatasetBatchSampler` or
`MultiDatasetBatchSampler.balanced(...)` to define semantic dataset mixing
rates.
`samples_per_dataset` may be integer counts or float ratios. Use
`epoch_policy="max_size", replacement=True` when smaller datasets should be
oversampled so the largest dataset does not dominate an epoch.

## Benchmark workflow

Use the current CLI subcommands:

```bash
# Self-contained write + read benchmark.
env COLUMNS=240 uv run nvalchemi-io-test roundtrip \
    -n 10000 \
    --read-mode batch \
    --read-order shuffle \
    --batch-size 64 \
    --prefetch-factor 16 \
    --pin-memory

# Sweep prefetch factors on the same access pattern.
for pf in 8 16 32 64 128; do
    env COLUMNS=240 uv run nvalchemi-io-test roundtrip \
        -n 10000 \
        --read-mode batch \
        --read-order shuffle \
        --batch-size 64 \
        --prefetch-factor "$pf" \
        --pin-memory
done

# Benchmark an existing store without rewriting it.
env COLUMNS=240 uv run nvalchemi-io-test read /path/to/store.zarr \
    --read-order shuffle \
    --batch-size 64 \
    --prefetch-factor 32 \
    --pin-memory

# Compare DataLoader fused reads against one-sample-at-a-time reads.
env COLUMNS=240 uv run nvalchemi-io-test read /path/to/store.zarr \
    --read-mode both \
    --read-order shuffle \
    --batch-size 64 \
    --prefetch-factor 32
```

Important benchmark semantics:

- `read-mode=batch` uses the public DataLoader path with fused prefetch.
- Benchmark batch mode uses `Dataset(skip_validation=True)` to focus on storage
  and batching throughput.
- `read-mode=single` calls `reader.read(index)` once per sample and is only a
  baseline for one-sample-at-a-time access.
- `batch_size` is the model-facing batch size.
- `prefetch_factor` controls the backend read window.
- Use `read-order=shuffle` to model fully shuffled training reads.
- Use `read-order=block-shuffle` to test partial locality.

## Diagnosing bottlenecks

1. Run `nvalchemi-io-test read` on an existing representative store.
2. Sweep `prefetch_factor` at the target `batch_size`.
3. Compare `read-mode=batch` against `read-mode=single`.
4. If batch mode is fast but training is slow, inspect validation, batching, and
   device-transfer overhead. Try `skip_validation=True`, `pin_memory=True`, and
   CUDA streams.
5. If batch mode is slow, inspect chunk/shard configuration, compression codec,
   filesystem metadata pressure, and read order.

## Quick checklist

- [ ] Use `Dataset(skip_validation=True)` for trusted toolkit-written stores.
- [ ] Use `DataLoader(pin_memory=True)` for CUDA training.
- [ ] Start with `batch_size=64`.
- [ ] Start with `prefetch_factor=16` or `32` for shuffled reads.
- [ ] Sweep `prefetch_factor=8,16,32,64,128` with `nvalchemi-io-test`.
- [ ] Keep sampler semantics independent from storage locality.
- [ ] Use `load_batches(...)` for explicit batch reads.
- [ ] Tune chunk/shard sizes on a representative store and filesystem.
- [ ] Use `read-mode=single` only as a baseline, not as the training path.
