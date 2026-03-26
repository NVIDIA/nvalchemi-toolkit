<!-- markdownlint-disable MD014 -->

(dynamics_sinks_guide)=

# Data Sinks

During a dynamics simulation you often need to record results: full trajectory
frames for post-processing, relaxed structures at convergence, or scalar time series
for monitoring. **Data sinks** are the pluggable storage backends that snapshot
hooks write into.

## What is a sink?

A sink is an object that accepts {py:class}`~nvalchemi.data.Batch` snapshots and
stores them. The dynamics module ships three implementations, each targeting a
different performance and persistence trade-off:

| Sink | Backing store | Typical use |
|------|---------------|-------------|
| {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` | GPU device memory | Maximum throughput; short trajectories or inter-stage data passing |
| {py:class}`~nvalchemi.dynamics.sinks.HostMemory` | Host RAM | Intermediate staging; moderate-length trajectories |
| {py:class}`~nvalchemi.dynamics.sinks.ZarrData` | Zarr store on disk | Persistent storage; long trajectories, post-processing, checkpointing |

## How sinks integrate with hooks

Sinks do not run on their own --- they are consumed by snapshot hooks. When a
{py:class}`~nvalchemi.dynamics.hooks.SnapshotHook` or
{py:class}`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` fires, it writes the
current batch state into its associated sink. The hook controls *when* data is
captured; the sink controls *where* it goes.

```python
from nvalchemi.dynamics.hooks import SnapshotHook, ConvergedSnapshotHook
from nvalchemi.dynamics.sinks import ZarrData, GPUBuffer

# Record the full trajectory to disk every 50 steps
trajectory_hook = SnapshotHook(
    sink=ZarrData("/path/to/trajectory.zarr"),
    frequency=50,
)

# Capture only converged structures in a GPU buffer
converged_hook = ConvergedSnapshotHook(
    sink=GPUBuffer(capacity=256),
)
```

## GPUBuffer

{py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` stores snapshots in GPU device
memory. This avoids device-to-host transfers entirely, making it the fastest option
when downstream consumers (e.g. the next stage in a fused pipeline) also live on
the GPU.

Because GPU memory is limited, `GPUBuffer` is best suited for short-lived data: a
few hundred frames, or converged structures that will be consumed and discarded
before the buffer fills.

## HostMemory

{py:class}`~nvalchemi.dynamics.sinks.HostMemory` moves snapshots to host RAM. This
is a middle ground: cheaper than GPU memory but still in-process, so there is no
disk I/O overhead. Use it when trajectories are too large for GPU memory but you
want to avoid disk writes during the simulation loop, deferring serialization to
after the run completes.

## ZarrData

{py:class}`~nvalchemi.dynamics.sinks.ZarrData` writes snapshots to a Zarr store on
disk. Zarr's chunked, compressed format handles large trajectories efficiently and
integrates with the toolkit's data loading pipeline --- the same
{py:class}`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader` used for
training data can read trajectory stores.

```python
from nvalchemi.dynamics.sinks import ZarrData

sink = ZarrData("/path/to/trajectory.zarr")
```

### Configuring compression

By default `ZarrData` writes uncompressed arrays. Pass a
{py:class}`~nvalchemi.data.datapipes.ZarrWriteConfig` (or a plain dict that
follows the same schema) to enable compression, custom chunking, or per-field
overrides:

```python
from nvalchemi.dynamics.sinks import ZarrData
from nvalchemi.data.datapipes import ZarrWriteConfig, ZarrArrayConfig
from zarr.codecs import ZstdCodec

config = ZarrWriteConfig(
    core=ZarrArrayConfig(
        compressors=(ZstdCodec(level=3),),
        chunk_size=512,
    ),
)
sink = ZarrData("/path/to/trajectory.zarr", config=config)
```

The `config` parameter accepts either a `ZarrWriteConfig` instance or a plain
dictionary --- the latter is automatically validated:

```python
from zarr.codecs import ZstdCodec

sink = ZarrData(
    "/path/to/trajectory.zarr",
    config={"core": {"compressors": (ZstdCodec(level=3),)}},
)
```

```{tip}
See the [Zarr Compression Tuning Guide](zarr_compression_guide) for codec
comparisons, chunk-size recommendations, and back-of-the-envelope storage
calculations.
```

ZarrData is the recommended choice for production workflows where results need to
survive the process, be shared across machines, or feed back into training.

## Putting it together

A typical dynamics setup combines multiple hooks and sinks to capture different
aspects of the simulation:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook
from nvalchemi.dynamics.hooks import (
    ConvergedSnapshotHook,
    LoggingHook,
    SnapshotHook,
)
from nvalchemi.dynamics.sinks import GPUBuffer, ZarrData

with FIRE(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[
        # Stop when converged
        ConvergenceHook.from_fmax(0.05),
        # Log scalars every 10 steps
        LoggingHook(backend="csv", log_path="hooks.csv", frequency=10),
        # Full trajectory to disk every 50 steps
        SnapshotHook(sink=ZarrData("/tmp/traj.zarr"), frequency=50),
        # Converged frames to GPU for downstream consumption
        ConvergedSnapshotHook(sink=GPUBuffer(capacity=256)),
    ],
) as opt:
    relaxed = opt.run(batch)
```

## See also

- **Hooks**: The [Hooks guide](dynamics_hooks_guide) covers the hook protocol and
  how to write custom hooks.
- **Data loading**: The [Data Loading Pipeline](datapipes_guide) guide shows how to
  read Zarr stores back for training or analysis.
- **API**: {py:mod}`nvalchemi.dynamics` for the full sinks API reference.
- **Compression tuning**: The [Zarr Compression Tuning Guide](zarr_compression_guide)
  covers codec choices, chunk sizing, and storage estimates.
- **I/O benchmark**: The [I/O benchmark tool](io_benchmark_section) lets you
  measure write throughput and compression ratios on synthetic data matching
  your workload.
