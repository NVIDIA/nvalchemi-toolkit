---
name: nvalchemi-zarr-perf
description: >
  Performance tuning for nvalchemi's Zarr-backed DataLoader pipeline.
  Use when constructing or configuring AtomicDataZarrReader, Dataset,
  or DataLoader for training or inference and throughput matters —
  especially with shuffled access patterns, large datasets, or when
  profiling shows I/O or validation bottlenecks. Also use when writing
  Zarr stores that will later be read with random access.
---

# Zarr DataLoader Performance Tuning

## Defaults that give reasonable performance

```python
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    Dataset,
    DataLoader,
)

reader = AtomicDataZarrReader("store.zarr", pin_memory=True)

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
    prefetch_factor=16,     # fuse 16 batches into one read_many call
    num_streams=2,
    use_streams=True,
)
```

## Key knobs

### `prefetch_factor` (DataLoader)

Controls how many consecutive batches are fused into a single
`reader.read_many()` call. The reader sorts and merges the fused indices
into contiguous ranges, amortising ~2 ms of per-call Zarr overhead.

| Access pattern | Recommended `prefetch_factor` |
|----------------|------------------------------:|
| Sequential     |                             2 |
| Shuffled       |                         16–32 |

Larger values help shuffled reads dramatically (155 → 994 samples/s going
from pf=8 to pf=32 on a 10k-sample store). For sequential access the
reader already detects contiguous runs, so pf=2 suffices.

### `skip_validation` (Dataset)

Bypasses per-sample `AtomicData` Pydantic validation (~4 ms/sample).
Constructs `Batch` directly from raw tensor dicts via
`Batch.from_raw_dicts()`.

**Use when:** the store was written by `AtomicDataZarrWriter` or has been
validated externally.
**Do not use when:** the store contents are untrusted or from a third party.

### `num_workers` (Dataset)

Thread pool size for background reads. Keep at **1** — concurrent Zarr
decompression threads contend on CPU and reduce throughput.

### `pin_memory` (Reader)

Set `pin_memory=True` on the reader when the target device is CUDA.
Enables async host-to-device transfers via `use_streams=True`.

## Writing stores for fast random reads

When creating a Zarr store that will be read with shuffle:

```python
from nvalchemi.data.datapipes import AtomicDataZarrWriter, ZarrWriteConfig, ZarrArrayConfig

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        chunk_size=1024,
        shard_size=4096,
    ),
)
writer = AtomicDataZarrWriter("store.zarr", config=config)
```

- **`chunk_size=1024`** — balances per-chunk metadata cost against read
  amplification. Smaller chunks (16, 64) are slower due to metadata
  overhead.
- **`shard_size=4096`** — groups chunks into fewer storage objects,
  reducing filesystem metadata pressure.
- **`ZstdCodec(level=3)`** — good compression/speed tradeoff. LZ4 is
  faster to decompress but compresses less.

## How the reader optimises random access

`AtomicDataZarrReader.read_many()` automatically:

1. **Sorts** requested indices by physical position.
2. **Gap-merges** nearby indices into contiguous range reads (capped at
   8× read amplification to avoid decompressing huge unused spans).
3. Returns results in the caller's original request order.

This is transparent — no caller-side work needed. Larger batches (via
`prefetch_factor`) give the merge step more indices to coalesce, which is
why pf matters most for shuffled reads.

## Performance reference

10k-sample store, ~55 atoms/sys, chunk=1024, shard=4096, zstd-3,
batch_size=64:

| Configuration | Shuffled samples/s |
|---------------|-------------------:|
| pf=8, validated | ~80 |
| pf=8, skip_validation | ~155 |
| pf=16, skip_validation | ~309 |
| pf=32, skip_validation | ~994 |
| Sequential, pf=2 | ~4,000 |

## Diagnosing bottlenecks

Use `nvalchemi-io-test read` to measure raw reader throughput in isolation
(no validation, no device transfer):

```bash
nvalchemi-io-test read /path/to/store.zarr \
    --read-order shuffle --read-batch-size 1024
```

If raw reader throughput >> DataLoader throughput, the bottleneck is
validation or device transfer. Set `skip_validation=True`.

If raw reader throughput is already low, increase `--read-batch-size`
or check chunk/shard configuration.

## Quick checklist

- [ ] `pin_memory=True` on reader
- [ ] `skip_validation=True` if store is trusted
- [ ] `prefetch_factor=16` or higher for shuffled training
- [ ] `num_workers=1`
- [ ] `use_streams=True`, `num_streams=2`
- [ ] Store written with `chunk_size=1024`, `shard_size=4096`
- [ ] Codec: `ZstdCodec(level=3)` or `LZ4` for speed
