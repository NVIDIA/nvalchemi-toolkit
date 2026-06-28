<!-- markdownlint-disable MD014 -->

(zarr_compression_guide)=

# Zarr Compression Tuning

Zarr stores are the primary persistence format for atomic simulation data in the
toolkit. Configuring compression, chunking, sharding, and read windows correctly
can reduce disk usage and improve training-time I/O throughput. This guide covers
the configuration options, codec trade-offs, and practical recipes for common
workloads.

## Quick start

The simplest way to enable compression is to pass a
{py:class}`~nvalchemi.data.datapipes.ZarrWriteConfig` when creating a writer or
sink:

```python
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter
from zarr.codecs import ZstdCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(compressors=(ZstdCodec(level=3),)),
)
writer = AtomicDataZarrWriter("/data/example.zarr", config=config)
```

For dynamics trajectories, pass the same config to
{py:class}`~nvalchemi.dynamics.sinks.ZarrData`:

```python
from nvalchemi.dynamics.sinks import ZarrData

sink = ZarrData("/tmp/trajectory.zarr", config=config)
```

```{tip}
The configuration classes are Pydantic models, and you do not need to
import and construct them manually: you can pass a `dict` with the
same structure and keys and under the hood they will be validated
against the configuration classes. Using the classes explicitly is
helpful, however, when working with modern IDEs and language servers
as they tell you what arguments are required, defaults, etc.
```

## Configuration hierarchy

The toolkit organises Zarr arrays into three logical groups:

| Group | Contents | Default compression |
|-------|----------|---------------------|
| `meta` | Pointer arrays (`atoms_ptr`, `edges_ptr`), validity mask | None |
| `core` | Positions, forces, energy, atomic numbers, cell, pbc | None |
| `custom` | User-added arrays via `AtomicData.custom` | None |

{py:class}`~nvalchemi.data.datapipes.ZarrWriteConfig` lets you set different
{py:class}`~nvalchemi.data.datapipes.ZarrArrayConfig` for each group:

```python
config = ZarrWriteConfig(
    meta=ZarrArrayConfig(...),    # metadata arrays
    core=ZarrArrayConfig(...),    # core physics arrays
    custom=ZarrArrayConfig(...),  # user-added arrays
)
```

### Field overrides

For fine-grained control, `field_overrides` takes precedence over group defaults.
Resolution order:

```text
field_overrides["positions"]   →   if present, use this
         ↓ (not found)
core (group default)           →   if present, use this
         ↓ (not configured)
no compression (Zarr defaults)
```

```{tip}
Use `field_overrides` when a single array has different access patterns from
its group — for example, if positions need fast random access while other core
arrays are read sequentially.
```

## Codec comparison

Zarr v3 supports pluggable codecs via the `zarr.abc.codec.Codec` interface. The
toolkit writer accepts any codec supported by Zarr; the benchmark CLI exposes the
common choices `zstd`, `lz4`, and `blosc-zstd`.

| Codec | Class | Strengths | Weaknesses | Typical use |
|-------|-------|-----------|------------|-------------|
| Zstd | `zarr.codecs.ZstdCodec` | Good ratio, fast decompress | Moderate compress speed | General purpose, sequential data |
| Blosc/LZ4 | `zarr.codecs.BloscCodec(cname="lz4")` | Very fast compress+decompress | Lower ratio | Trajectories, real-time I/O |
| Blosc/Zstd | `zarr.codecs.BloscCodec(cname="zstd")` | Blosc blocking + Zstd ratio | Slightly more complex | Large arrays, balanced ratio/speed |

```{note}
Compression level controls the ratio/speed trade-off. Higher levels yield better
compression but slower writes. For Zstd, level 3 is a good default; level 5–9
improves ratio modestly at the cost of write throughput. For LZ4, the level
parameter has minimal effect---speed is consistently high.
```

### Blosc options

`BloscCodec` exposes codec name, compression level, shuffle, and blocksize through
its constructor. Keep these settings explicit in `ZarrWriteConfig` when you want
reproducible stores:

```python
from zarr.codecs import BloscCodec

compressor = BloscCodec(cname="zstd", clevel=5)
```

## Chunk size tuning

The `chunk_size` parameter in {py:class}`~nvalchemi.data.datapipes.ZarrArrayConfig`
controls the chunk length along **dimension 0** of the stored array. Other
dimensions use the full extent. Because atom-level fields (positions, forces,
atomic_numbers) are stored **concatenated** along the atom axis — not per
structure — dimension 0 is the total-atoms axis, not the number of structures.

### Target chunk size

The Zarr documentation recommends chunks of **at least 1 MB uncompressed** for good
throughput, particularly when using Blosc. Smaller chunks increase per-chunk
overhead (metadata, system calls, compression dictionary resets). Larger chunks
reduce the number of I/O operations for sequential reads but increase
**read amplification** for random access — reading a single 50-atom structure
(600 bytes of positions) from a 1 MB chunk wastes 99.9 % of the decompressed data.

| Access pattern | Recommended chunk target | Rationale |
|----------------|--------------------------|-----------|
| Sequential DataLoader | 1–4 MB | Amortises overhead across many samples |
| Trajectory capture (append, then sequential read) | 1 MB | Balances write latency and read throughput |
| Random access (visualisation, single-sample lookup) | 64–256 KB | Limits read amplification |

```{note}
Zarr v3 supports **sharding**, which decouples the read unit (chunk) from the
storage unit (shard). With sharding you can have small chunks for fine-grained
random access grouped into large shards for filesystem efficiency. Set
``shard_size`` on {py:class}`~nvalchemi.data.datapipes.ZarrArrayConfig` to
enable it — the shard size must be a multiple of the chunk size.
```

### Back-of-the-envelope formula

For a stored array whose rows have `trailing_dims` trailing dimensions and
dtype size `d` bytes:

$$
\text{bytes\_per\_row} = d \times \prod(\text{trailing\_dims})
$$

$$
\text{chunk\_size} = \left\lfloor \frac{\text{target\_bytes}}{\text{bytes\_per\_row}} \right\rfloor
$$

The following table gives concrete values for common arrays:

| Array | Trailing dims | Dtype | Bytes/row | chunk_size (1 MB) | chunk_size (4 MB) |
|-------|---------------|-------|-----------|-------------------|-------------------|
| positions `[V, 3]` | 3 | float32 | 12 | 83,333 | 333,333 |
| forces `[V, 3]` | 3 | float32 | 12 | 83,333 | 333,333 |
| atomic_numbers `[V]` | 1 | int64 | 8 | 125,000 | 500,000 |
| energy `[B]` | 1 | float64 | 8 | 125,000 | 500,000 |
| cell `[B, 3, 3]` | 9 | float32 | 36 | 27,778 | 111,111 |
| neighbor_list `[E, 2]` | 2 | int64 | 16 | 62,500 | 250,000 |
| shifts `[E, 3]` | 3 | float32 | 12 | 83,333 | 333,333 |

#### Positions Example

$$
\text{bytes\_per\_row} = 3 \times 4 = 12 \text{ bytes}
$$
$$
\text{chunk\_size} = \left\lfloor \frac{1{,}000{,}000}{12} \right\rfloor = 83{,}333
$$

#### Energy Example

$$
\text{bytes\_per\_row} = 1 \times 8 = 8 \text{ bytes}
$$
$$
\text{chunk\_size} = \left\lfloor \frac{1{,}000{,}000}{8} \right\rfloor = 125{,}000
$$

### Read amplification

When reading a single structure by index, the reader fetches the slice
`positions[atoms_ptr[i]:atoms_ptr[i+1], :]` --- typically about 50 rows
(600 bytes for float32 positions). With large chunks, most of the decompressed
data is discarded:

| chunk_size | Chunk bytes (positions) | Amplification (50-atom read) |
|------------|------------------------|------------------------------|
| 333,333 | 4 MB | 6,667× |
| 83,333 | 1 MB | 1,667× |
| 10,000 | 120 KB | 200× |

For purely sequential workloads, amplification does not matter because every row
is consumed. For shuffled training, amplification depends on the effective read
window: the DataLoader fuses `prefetch_factor` batches into one `read_many` call,
and the Zarr reader can group indices that share chunk locality. Larger chunks can
still hurt fully random single-sample access, so prefer smaller chunks or field
overrides when interactive lookup or visualization is a primary workload.

```{warning}
Atom-level fields (positions, forces, atomic_numbers) are stored as
**concatenated** arrays of shape `[V_total, ...]` where `V_total` is the sum of
atoms across all structures. The `chunk_size` parameter controls the number of
**rows** in each chunk, not the number of structures. System-level fields
(energy, cell, pbc) have one row per structure, so `chunk_size` directly equals
the number of structures per chunk.
```

## Storage estimation

The tables below assume 50 atoms per structure on average with ~200 edges
(a typical cutoff-based neighbour list). Edge arrays dominate storage; many
workflows recompute edges at load time via neighbour lists and omit them from
the store.

### Per-array breakdown (100k structures)

| Array | Shape | Dtype | Uncompressed |
|-------|-------|-------|-------------|
| positions | [5M, 3] | float32 | 60 MB |
| forces | [5M, 3] | float32 | 60 MB |
| atomic_numbers | [5M] | int64 | 40 MB |
| energy | [100k] | float64 | 0.8 MB |
| cell | [100k, 3, 3] | float32 | 3.6 MB |
| pbc | [100k, 3] | bool | 0.3 MB |
| stress | [100k, 3, 3] | float32 | 3.6 MB |
| virial | [100k, 3, 3] | float32 | 3.6 MB |
| dipole | [100k, 3] | float32 | 1.2 MB |
| neighbor_list | [20M, 2] | int64 | 320 MB |
| shifts | [20M, 3] | float32 | 240 MB |
| metadata (ptrs, masks) | — | mixed | 27 MB |
| **Total (with edges)** | | | **760 MB** |
| **Total (without edges)** | | | **200 MB** |

### Scaling by dataset size

| Component | 100k | 1M | 10M |
|-----------|------|-----|------|
| Node + system core | 173 MB | 1.7 GB | 17 GB |
| Edge arrays | 560 MB | 5.6 GB | 56 GB |
| Metadata | 27 MB | 267 MB | 2.7 GB |
| **Total (with edges)** | **760 MB** | **7.6 GB** | **76 GB** |
| **Total (without edges)** | **200 MB** | **2.0 GB** | **20 GB** |

### With compression

| Codec | Typical ratio | 100k | 1M | 10M |
|-------|---------------|------|-----|------|
| Zstd (level 3) | 2–4× | 190–380 MB | 1.9–3.8 GB | 19–38 GB |
| LZ4 | 1.5–2.5× | 300–510 MB | 3.0–5.1 GB | 30–51 GB |

```{note}
Actual ratios depend heavily on data characteristics. Smooth MD trajectories
(correlated frames) compress 4–6×; random equilibrium structures compress 2–3×.
Integer arrays (atomic numbers, pointers) often compress 5–10× due to repetition.
The estimates above include edge arrays; without edges, divide by ~3.8.

The [I/O benchmark tool](io_benchmark_section) uses purely random tensors, so
its measured ratios (~1.75× Zstd, ~1.63× LZ4) represent a worst case. Real
molecular data will compress significantly better.
```

### File count

Without sharding, each chunk becomes a separate file on local stores. A
Zarr store also contains one `zarr.json` metadata file per array and per
group, so the **total file count** across the whole store is the sum of
chunk files for every array plus metadata files (~20 for a typical store).

The table below shows **chunk files per array** for the positions array
(`[V_total, 3]` float32), which is representative of other atom-level arrays:

| chunk_size | 100k (V = 5M) | 1M (V = 50M) | 10M (V = 500M) |
|------------|--------------|--------------|----------------|
| 83,333 (1 MB) | 61 | 601 | 6,001 |
| 10,000 (120 KB) | 500 | 5,000 | 50,000 |

A typical store has ~10 chunked arrays, so **multiply by ~10** for total
chunk files, then add ~20 metadata files. At 100k systems with
`chunk_size=10,000`, the TUI reports **~4,500 total files**; at 100k with
`chunk_size=83,333`, it reports **~690 total files**.

**With sharding** (`shard_size=500,000`, `chunk_size=10,000`), the same
100k-system store drops to **~160 total files** — a 28× reduction — because
each shard file bundles 50 chunks.

Filesystem metadata overhead becomes significant above ~10,000 files per
array. If you need small chunks for random access at scale, enable sharding
with ``shard_size`` or use a cloud object store (S3, GCS via `FsspecStore`).

## Recipes

### Recipe 1: Sequential dataset (best compression)

Prioritise disk space over write speed. Use Zstd at a moderate level with large
chunks (~1 MB per chunk) for sequential reads.

```python
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter
from zarr.codecs import ZstdCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=5),),
        chunk_size=100_000,   # ~1.2 MB chunks for positions [V,3] f32
    ),
)
writer = AtomicDataZarrWriter("/data/example.zarr", config=config)
```

### Recipe 2: Dynamics trajectory (fast I/O)

Prioritise write throughput for real-time trajectory capture. Use LZ4 with
moderate chunks (~120 KB) to balance write latency and random-access readback.

```python
from nvalchemi.dynamics.sinks import ZarrData
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from zarr.codecs import BloscCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(BloscCodec(cname="lz4"),),
        chunk_size=10_000,    # ~120 KB chunks for positions [V,3] f32
    ),
)
sink = ZarrData("/tmp/trajectory.zarr", config=config)
```

### Recipe 3: Per-field override (mixed access patterns)

Use Zstd for most arrays but LZ4 with smaller chunks for positions (frequently
accessed for visualisation or neighbour list rebuilds).

```python
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter
from zarr.codecs import ZstdCodec, BloscCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        chunk_size=100_000,   # 1 MB chunks for sequential core arrays
    ),
    field_overrides={
        "positions": ZarrArrayConfig(
            compressors=(BloscCodec(cname="lz4"),),
            chunk_size=50_000,  # ~600 KB: smaller for random access
        ),
    },
)
writer = AtomicDataZarrWriter("/data/mixed.zarr", config=config)
```

### Recipe 4: Sparse data (skip empty chunks)

For datasets with many optional fields or sparse validity masks, disable writing
empty chunks to save space.

```python
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter
from zarr.codecs import ZstdCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        write_empty_chunks=False,
    ),
    custom=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        write_empty_chunks=False,
    ),
)
writer = AtomicDataZarrWriter("/data/sparse.zarr", config=config)
```

```{tip}
`write_empty_chunks=False` is especially useful for custom arrays that are only
populated for a subset of structures. Zarr will skip writing chunks that contain
only the fill value, reducing both disk usage and write time.
```

### Recipe 5: Sharded storage (large datasets)

For datasets with millions of structures, use sharding to keep small read-friendly
chunks while reducing the number of storage objects. The shard size must be a
multiple of the chunk size.

```python
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrWriter
from zarr.codecs import ZstdCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        chunk_size=10_000,     # 120 KB chunks for random access
        shard_size=500_000,    # 50 chunks per shard, ~6 MB per shard
    ),
)
writer = AtomicDataZarrWriter("/data/large.zarr", config=config)
```

```{tip}
Sharding is particularly valuable on local filesystems with large datasets
where file count can become a bottleneck. With 10M structures and
``chunk_size=10,000``, you would get 50,000 files per array without sharding
versus only 1,000 shard files with ``shard_size=500,000``.
```

(io_benchmark_section)=

## I/O benchmark tool

The toolkit ships a command-line benchmark for measuring Zarr write throughput,
readback throughput, and compression ratios on synthetic data. Use it to
validate storage configuration and readback strategy before committing to a
production workflow.

The CLI has two subcommands:

- **`roundtrip`** — generate synthetic data, write it to a temporary Zarr
  store, then read it back and report timing.
- **`read`** — benchmark read throughput against a pre-existing Zarr store,
  without writing anything.

Run `nvalchemi-io-test --help` to see the available subcommands. Use
`roundtrip` when you want the benchmark to create a temporary store, and use
`read` when you already have a representative store on the target filesystem.

### Running the roundtrip benchmark

```bash
# Install (if not already)
$ uv sync

# Basic: compare codec overhead across dataset sizes
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 \
    --codec zstd --level 3 \
    --chunk-size 83333 --edge-chunk-size 62500

# Compare fast batch readback against one-sample-at-a-time
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 \
    --read-mode both --batch-size 64 --prefetch-factor 8

# Model shuffled training reads against compressed stores
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 \
    --read-order shuffle --batch-size 64 --prefetch-factor 16
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 \
    --read-order block-shuffle --read-order-block-size 8192 \
    --batch-size 64 --prefetch-factor 16

# Fast codec with smaller chunks for trajectory-style workloads
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 --codec lz4 \
    --chunk-size 10000 --edge-chunk-size 10000

# Larger molecules with edge-specific chunking
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 \
    --min-atoms 100 --max-atoms 500 \
    --codec zstd --chunk-size 83333 --edge-chunk-size 62500

# With sharding enabled
$ nvalchemi-io-test roundtrip -n 1000 -n 10000 \
    --chunk-size 10000 --shard-size 500000 \
    --edge-chunk-size 10000 --edge-shard-size 500000

# Write a store to a specific directory for later read benchmarking
$ nvalchemi-io-test roundtrip -n 10000 --codec zstd \
    --chunk-size 1024 --shard-size 4096 \
    --output-dir /scratch/benchmark_stores/
```

Roundtrip options:

| Option | Default | Description |
|--------|---------|-------------|
| `-n` / `--num-systems` | 1000 10000 100000 | Dataset sizes to benchmark (repeatable) |
| `--min-atoms` | 10 | Minimum atoms per structure |
| `--max-atoms` | 100 | Maximum atoms per structure |
| `--codec` | — | Compression codec: `zstd`, `lz4`, or `blosc-zstd` |
| `--level` | 3 | Compression level |
| `--chunk-size` | — | Chunk size for node/system arrays |
| `--shard-size` | — | Shard size for node/system arrays |
| `--edge-chunk-size` | — | Chunk size for edge arrays (neighbor_list, shifts) |
| `--edge-shard-size` | — | Shard size for edge arrays |
| `--read-mode` | `batch` | Readback path to time: `batch`, `single`, or `both` |
| `--batch-size` | 64 | Number of samples per emitted DataLoader batch in `batch` mode |
| `--prefetch-factor` | 16 | Number of emitted batches to fuse into each backend read in `batch` mode |
| `--read-order` | `sequential` | Logical read order: `sequential`, `shuffle`, or `block-shuffle` |
| `--read-seed` | 0 | Random seed for shuffled read orders |
| `--read-order-block-size` | 8192 | Contiguous block size for `block-shuffle` read order |
| `--pin-memory` | `False` | Request pinned CPU tensors in batch read mode |
| `--output-dir` | — | Persist the written store(s) here instead of a temp directory |

### Read-only benchmark

Use `nvalchemi-io-test read` to benchmark against an existing Zarr store.
This isolates read performance from generation and write overhead, and lets
you test multiple read configurations against the same store without
rewriting it each time.

```bash
# Sequential read baseline
$ nvalchemi-io-test read /path/to/store.zarr

# Shuffled access at different read windows
$ nvalchemi-io-test read /path/to/store.zarr \
    --read-order shuffle --batch-size 64 --prefetch-factor 8
$ nvalchemi-io-test read /path/to/store.zarr \
    --read-order shuffle --batch-size 64 --prefetch-factor 64

# Compare batch vs. single-sample under shuffle
$ nvalchemi-io-test read /path/to/store.zarr \
    --read-mode both --read-order shuffle
```

Read options:

| Option | Default | Description |
|--------|---------|-------------|
| `PATH` | — | Path to an existing Zarr store (directory) |
| `--read-mode` | `batch` | `batch`, `single`, or `both` |
| `--batch-size` | 64 | Number of samples per emitted DataLoader batch |
| `--prefetch-factor` | 16 | Number of emitted batches to fuse into each backend read |
| `--read-order` | `sequential` | `sequential`, `shuffle`, or `block-shuffle` |
| `--read-seed` | 0 | Random seed for shuffled orders |
| `--read-order-block-size` | 8192 | Block size for `block-shuffle` |
| `--pin-memory` | `False` | Request pinned CPU tensors in batch read mode |

```{tip}
The ``read`` subcommand measures the public DataLoader read path by default:
``batch_size`` controls emitted batches, and ``prefetch_factor`` controls how
many emitted batches are fused into one backend read. Use ``single`` mode only
as a one-sample-at-a-time baseline.
```

```{note}
Benchmark `batch` mode uses `Dataset(skip_validation=True)` to focus on storage
and batching throughput for stores that are already trusted. If your training
pipeline keeps validation enabled, expect lower end-to-end throughput.
```

### Readback mode: batch vs. single sample

The benchmark reports write time plus a full-store readback. Readback uses the
batch path by default:

```bash
$ nvalchemi-io-test roundtrip -n 10000 --codec zstd \
    --chunk-size 83333
```

In `batch` mode the benchmark uses the toolkit
{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader` with fused prefetch.
The emitted batch size is controlled by `--batch-size`; the backend read window is
controlled by `--batch-size * --prefetch-factor`. The Zarr reader then receives
large `read_many(...)` requests and can coalesce physical I/O across the requested
indices.

Use `single` mode to time a one-sample-at-a-time access pattern:

```bash
$ nvalchemi-io-test roundtrip -n 10000 --read-mode single
```

Use `both` to emit one row per read path from the same written store:

```bash
$ nvalchemi-io-test roundtrip -n 10000 \
    --read-mode both --batch-size 64 --prefetch-factor 8
```

`batch` mode should be faster for DataLoader-style workloads because it amortises
Python dispatch, Zarr array indexing, chunk lookup, decompression setup, and
filesystem metadata access over many samples. `single` mode remains useful as a
baseline for debugging and for estimating the penalty paid by code that reads one
structure at a time.

### Read order: sequential vs. shuffled training access

For compressed Zarr stores, the logical index order can dominate throughput.
Sequential readback gives the Zarr reader mostly contiguous physical positions.
Fully shuffled readback models `DataLoader(shuffle=True)`: each emitted batch can
contain unrelated samples, but fused prefetch still gives the reader a larger
window of indices to sort and group by chunk locality.

Use `--read-order shuffle` to benchmark that worst-case training pattern:

```bash
$ nvalchemi-io-test roundtrip -n 10000 --codec zstd \
    --chunk-size 83333 --edge-chunk-size 62500 \
    --read-order shuffle
```

Use `--read-order block-shuffle` to model one locality-preserving training
order:

```bash
$ nvalchemi-io-test roundtrip -n 10000 --codec zstd \
    --chunk-size 83333 --edge-chunk-size 62500 \
    --read-order block-shuffle --read-order-block-size 8192
```

`block-shuffle` splits the index range into contiguous blocks of
`--read-order-block-size` samples, shuffles the *blocks*, and leaves the
indices inside each block in sequential order. For example, with 10,000
samples and a block size of 2,000 the reader sees five blocks in random
order, but within each block it reads indices 0–1,999, 2,000–3,999, etc.
sequentially.

This benchmark mode does **not** correspond to a specific DataLoader API;
it is a synthetic access pattern that helps you measure how much throughput
you recover when read locality is partially preserved. Compare
`block-shuffle` against `shuffle` to quantify the cost of fully random
access. In practice, a
{py:class}`~nvalchemi.dynamics.sampler.SizeAwareSampler` with bin-packing
can produce similar locality as a side-effect of grouping similarly-sized
systems.

```{note}
When `--read-mode both` is used, the two read paths run back-to-back against the
same freshly written store. This is useful for relative comparisons, but the
second mode may benefit from filesystem cache. For strict cold-cache numbers,
run `batch` and `single` in separate invocations with the same benchmark
configuration.
```

The following output illustrates the expected shape of the result table. Treat
numbers as machine- and store-specific; use the CLI on the target filesystem for
decisions.

```text
Zarr I/O Roundtrip Benchmark — no compression

  Systems   Read path   Read order   Batch   Prefetch   Read window   Write   Read   I/O/s
 ──────────────────────────────────────────────────────────────────────────────────────────
   10,000   batch       shuffle         64         32         2,048    0.54s   3.17s  2,695
```

(read_performance_tuning)=

## Read performance tuning

The benchmark commands above measure the public read paths: `batch` mode uses
the toolkit DataLoader with fused prefetch and `single` mode calls
`reader.read(...)` once per sample. In production, validation, batching, and
device-transfer overhead can dominate the end-to-end pipeline. This section
covers the knobs that matter most for read throughput, especially under shuffled
access patterns.

```{graphviz}
:caption: End-to-end read pipeline.

digraph read_pipeline {
    rankdir=LR
    compound=true
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=11 shape=box style="filled,rounded"]
    edge [fontname="Helvetica" fontsize=10]

    subgraph cluster_dataloader {
        label="DataLoader"
        style=rounded
        color="#4a90d9"
        fontcolor="#4a90d9"

        sampler [label="Sampler\n(indices)" fillcolor="#dce6f1"]
        fuse [label="Fuse\nprefetch_factor\nbatches" fillcolor="#f9e2ae"]
        sampler -> fuse [label="batch of\nindices"]
    }

    subgraph cluster_dataset {
        label="Dataset  (background thread)"
        style=rounded
        color="#5bb35b"
        fontcolor="#5bb35b"

        read_many [label="reader.read_many()\ncoalesced backend read" fillcolor="#dce6f1"]
        validate [label="AtomicData\nvalidation\n(Pydantic)" fillcolor="#fddede"]
        raw [label="raw tensor\ndicts" fillcolor="#d5f5d5"]
        batch_val [label="Batch.from_data_list()" fillcolor="#e8daef"]
        batch_raw [label="Batch.from_raw_dicts()" fillcolor="#e8daef"]

        read_many -> validate [label="skip_validation\n= False"]
        read_many -> raw [label="skip_validation\n= True"]
        validate -> batch_val
        raw -> batch_raw
    }

    subgraph cluster_consumer {
        label="Consumer"
        style=rounded
        color="#c0392b"
        fontcolor="#c0392b"

        device [label=".to(device)" fillcolor="#f9e2ae"]
        model [label="Model" fillcolor="#dce6f1"]
        device -> model
    }

    fuse -> read_many [label="N indices\n(N = pf \u00d7 bs)" lhead=cluster_dataset style=bold]
    batch_val -> device [ltail=cluster_dataset lhead=cluster_consumer style=bold]
    batch_raw -> device [ltail=cluster_dataset lhead=cluster_consumer style=bold]
}
```

### The read window: `prefetch_factor`

{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader` groups
`prefetch_factor` consecutive batches into a single
{py:meth}`~nvalchemi.data.datapipes.dataset.Dataset.prefetch_fused_batches` call.
The reader sees one large `read_many(...)` request containing up to
`prefetch_factor * batch_size` indices instead of many small calls, which lets
the Zarr backend coalesce random indices into larger physical reads.

The synchronous counterpart is
{py:meth}`~nvalchemi.data.datapipes.dataset.Dataset.load_batches`, which accepts
one or more batch-index lists and returns one
{py:class}`~nvalchemi.data.Batch` per list. `DataLoader` uses this same
batch-construction path when `prefetch_factor=0`; only the async double-buffered
prefetch is disabled. New code should prefer `load_batches(...)` for explicit
batch reads rather than calling older one-batch helpers directly.

Larger windows amortise per-call Zarr overhead across more samples.  For
shuffled training, a `prefetch_factor` of 16–32 is a good starting point, but
the best value depends on store size, chunking, compression, filesystem, and
whether pinned memory is enabled. Use the benchmark tool below on a
representative store before treating any value as a default for production.

```{tip}
For sequential access the reader already detects contiguous runs, so
``prefetch_factor=2`` is enough.  Increase it primarily when
``read_order=shuffle`` or ``read_order=block-shuffle``.
```

### Skipping validation: `skip_validation`

By default the {py:class}`~nvalchemi.data.datapipes.dataset.Dataset`
validates every loaded sample through
{py:class}`~nvalchemi.data.AtomicData` (Pydantic), which adds CPU overhead.
When the backing store is known to
contain well-formed data --- for example, stores written by the toolkit's
own writer --- you can bypass this:

```python
dataset = Dataset(reader=reader, device="cuda:0", skip_validation=True)
```

With `skip_validation=True` the Dataset constructs
{py:class}`~nvalchemi.data.Batch` objects directly from raw tensor
dictionaries via
{py:meth}`~nvalchemi.data.Batch.from_raw_dicts`, avoiding per-sample
Pydantic overhead entirely.

```{warning}
``skip_validation`` trusts the store contents.  Use it only with stores
produced by
{py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`
or stores whose schema you have already validated independently.
```

### How the Zarr reader coalesces random indices

The public `read_many` method delegates raw loading to the Zarr reader's
batch-oriented `_load_many_samples` hook. That hook applies several
backend-specific optimisations automatically:

1. **Resolve logical indices**: requested logical indices are mapped through the
   active-sample mask, so soft-deleted samples are skipped consistently.
2. **Sort by physical position**: requests are ordered by physical sample index
   so the underlying storage sees monotonic offsets where possible.
3. **Group by chunk locality**: samples that share Zarr chunks are grouped into
   range reads, with an amplification cap to avoid pathological over-reads when
   indices are very sparse.
4. **Fallback for fragmentation**: highly fragmented requests use orthogonal
   selections instead of many tiny range reads.

These optimisations are transparent: `read_many` still returns results in the
caller's original request order.

### Starting configurations

| Access pattern | `prefetch_factor` | `skip_validation` | Notes |
|----------------|------------------:|:-----------------:|-------|
| Sequential training | 2–4 | `False` or `True` | Small windows are usually enough because samples are already contiguous. |
| Shuffled training (trusted store) | 16–64 | `True` | Larger windows give the Zarr reader more indices to coalesce. |
| Shuffled training (untrusted store) | 16–64 | `False` | Keeps validation enabled, but validation can dominate end-to-end time. |
| Block-shuffle (block ≥ chunk) | 2–8 | `True` | Preserves some locality while still mixing batches. |

```{note}
Treat these as starting points, not throughput guarantees.  Benchmark with
``nvalchemi-io-test read`` or ``nvalchemi-io-test roundtrip`` using the same
read order, batch size, prefetch factor, compression, and storage backend you
expect in training.
```

## See also

- **Data pipeline**: The [Data Loading Pipeline](datapipes_guide) guide covers
  readers, datasets, and dataloaders.
- **Dynamics sinks**: The [Data Sinks](dynamics_sinks_guide) guide explains how
  `ZarrData` integrates with snapshot hooks.
- **API reference**:
  - {py:class}`~nvalchemi.data.datapipes.ZarrWriteConfig`
  - {py:class}`~nvalchemi.data.datapipes.ZarrArrayConfig`
  - {py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrWriter`
  - {py:class}`~nvalchemi.dynamics.sinks.ZarrData`
