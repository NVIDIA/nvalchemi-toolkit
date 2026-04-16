<!-- markdownlint-disable MD014 -->

(zarr_compression_guide)=

# Zarr Compression Tuning

Zarr stores are the primary persistence format for atomic simulation data in the
toolkit. Configuring compression and chunking correctly can reduce disk usage by
2–4× and significantly improve I/O throughput for data pipelines. This
guide covers the configuration options, codec trade-offs, and practical recipes
for common workloads.

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
toolkit has been tested with the following:

| Codec | Class | Strengths | Weaknesses | Typical use |
|-------|-------|-----------|------------|-------------|
| Zstd | `zarr.codecs.ZstdCodec` | Good ratio, fast decompress | Moderate compress speed | General purpose, sequential data |
| Blosc/LZ4 | `zarr.codecs.BloscCodec(cname="lz4")` | Very fast compress+decompress | Lower ratio | Trajectories, real-time I/O |
| Blosc/Zstd | `zarr.codecs.BloscCodec(cname="zstd")` | Blosc multithreading + Zstd ratio | Slightly more complex | Large arrays, parallel writes |
| Gzip | `zarr.codecs.GzipCodec` | Universal compatibility | Slow | Archival, interop |

```{note}
Compression level controls the ratio/speed trade-off. Higher levels yield better
compression but slower writes. For Zstd, level 3 is a good default; level 5–9
improves ratio modestly at the cost of write throughput. For LZ4, the level
parameter has minimal effect---speed is consistently high.
```

### Blosc multithreading

`BloscCodec` can use multiple threads internally, which helps when compressing
large chunks. By default it uses a single thread; pass `nthreads=4` (or similar)
if your workload benefits from parallel compression:

```python
from zarr.codecs import BloscCodec

compressor = BloscCodec(cname="zstd", clevel=5, nthreads=4)
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

**Example: positions (float32, shape [V, 3]), 1 MB target**

$$
\text{bytes\_per\_row} = 3 \times 4 = 12 \text{ bytes}
$$
$$
\text{chunk\_size} = \left\lfloor \frac{1{,}000{,}000}{12} \right\rfloor = 83{,}333
$$

**Example: energy (float64, shape [B]), 1 MB target**

$$
\text{bytes\_per\_row} = 1 \times 8 = 8 \text{ bytes}
$$
$$
\text{chunk\_size} = \left\lfloor \frac{1{,}000{,}000}{8} \right\rfloor = 125{,}000
$$

### Read amplification

When reading a single structure by index, the reader fetches the slice
`positions[atoms_ptr[i]:atoms_ptr[i+1], :]` — typically ~50 rows (600 bytes).
With large chunks, most of the decompressed data is discarded:

| chunk_size | Chunk bytes (positions) | Amplification (50-atom read) |
|------------|------------------------|------------------------------|
| 333,333 | 4 MB | 6,667× |
| 83,333 | 1 MB | 1,667× |
| 10,000 | 120 KB | 200× |

For purely sequential workloads (sequential DataLoader) amplification does not
matter — every row is consumed. For random-access workloads, prefer smaller
chunks or consider field overrides for frequently accessed arrays.

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

The toolkit ships a command-line benchmark for measuring Zarr write throughput
and compression ratios on synthetic data. Use it to validate configuration
choices before committing to a production workflow.

### Running the benchmark

```bash
# Install (if not already)
$ uv sync --all-extras

# Basic: compare codec overhead across dataset sizes
$ nvalchemi-io-test -n 1000 -n 10000 --codec zstd --level 3 --chunk-size 83333

# Fast codec with smaller chunks for trajectory-style workloads
$ nvalchemi-io-test -n 1000 -n 10000 --codec lz4 --chunk-size 10000

# Larger molecules with edge-specific chunking
$ nvalchemi-io-test -n 1000 -n 10000 --min-atoms 100 --max-atoms 500 \
    --codec zstd --chunk-size 83333 --edge-chunk-size 62500

# With sharding enabled
$ nvalchemi-io-test -n 1000 -n 10000 --codec zstd \
    --chunk-size 1000 --shard-size 10000
```

Key options:

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

### Example output

**Small molecules (10–100 atoms), Zstd level 3, 1 MB chunks:**

```text
nvalchemi Zarr I/O benchmark  atoms=10-100  config=zstd L3, chunk=83,333,
                                             edge_chunk=62,500
Pre-computed: 100,000 systems, 5,504,449 total atoms (avg 55.0),
              11,062,584 total edges (avg 110.6)
Estimated uncompressed: 484.9 MB

      Zarr I/O Benchmark — zstd L3, chunk=83,333, edge_chunk=62,500

              Avg     Avg      Raw     Disk                   Write
  Systems   atoms   edges     size     size  Ratio  Files      time  Systems/s
 ─────────────────────────────────────────────────────────────────────────────
    1,000      56     115   4.8 MB   2.8 MB  1.74x     36     0.14s     7,282
   10,000      55     112  47.1 MB  27.0 MB  1.75x     96     0.48s    20,736
  100,000      55     111 467.5 MB 267.7 MB  1.75x    691     4.66s    21,471
```

**Small molecules, LZ4, 120 KB chunks (trajectory-optimised):**

```text
nvalchemi Zarr I/O benchmark  atoms=10-100  config=lz4 L3, chunk=10,000,
                                             edge_chunk=10,000

      Zarr I/O Benchmark — lz4 L3, chunk=10,000, edge_chunk=10,000

              Avg     Avg      Raw     Disk                   Write
  Systems   atoms   edges     size     size  Ratio  Files      time  Systems/s
 ─────────────────────────────────────────────────────────────────────────────
    1,000      56     115   4.8 MB   3.0 MB  1.61x     76     0.12s     8,207
   10,000      55     112  47.1 MB  28.9 MB  1.63x    480     0.80s    12,446
  100,000      55     111 467.5 MB 287.5 MB  1.63x  4,509     8.10s    12,341
```

**Small molecules, sharded (chunk=10,000 inside shard=500,000):**

```text
nvalchemi Zarr I/O benchmark  atoms=10-100  config=chunk=10,000,
    shard=500,000, edge_chunk=10,000, edge_shard=500,000

      Zarr I/O Benchmark — chunk=10,000, shard=500,000,
                            edge_chunk=10,000, edge_shard=500,000

              Avg     Avg      Raw     Disk                   Write
  Systems   atoms   edges     size     size  Ratio  Files      time  Systems/s
 ─────────────────────────────────────────────────────────────────────────────
    1,000      56     115   4.8 MB   2.8 MB  1.73x     34     0.14s     6,998
   10,000      55     112  47.1 MB  27.0 MB  1.74x     46     0.63s    15,930
  100,000      55     111 467.5 MB 268.2 MB  1.74x    158     6.46s    15,471
```

Note the dramatic file count reduction with sharding: **4,509 → 158** at 100k
systems with the same chunk size, while compression ratio and disk size remain
essentially unchanged.

**Larger molecules (100–500 atoms), Zstd with edge-specific chunks:**

```text
nvalchemi Zarr I/O benchmark  atoms=100-500  config=zstd L3, chunk=83,333,
                                              edge_chunk=62,500
Pre-computed: 10,000 systems, 3,016,657 total atoms (avg 301.7),
              6,073,861 total edges (avg 607.4)
Estimated uncompressed: 263.5 MB

      Zarr I/O Benchmark — zstd L3, chunk=83,333, edge_chunk=62,500

              Avg     Avg      Raw     Disk                   Write
  Systems   atoms   edges     size     size  Ratio  Files      time  Systems/s
 ─────────────────────────────────────────────────────────────────────────────
    1,000     303     615  25.7 MB  15.4 MB  1.67x     66     0.21s     4,737
   10,000     302     607 254.7 MB 152.9 MB  1.67x    394     1.23s     8,138
```

```{note}
Zarr v3 defaults to ``ZstdCodec(level=0)`` when no compressor is specified.
The "Raw size" column reflects the data as written by the toolkit (including
Zarr metadata overhead), so even runs without an explicit ``--codec`` flag
will show some compression.
```

```{tip}
Run with ``--min-atoms`` and ``--max-atoms`` matching your actual dataset to get
realistic estimates. The benchmark uses uniform random atom counts; real-world
distributions may be skewed toward smaller or larger structures.
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
