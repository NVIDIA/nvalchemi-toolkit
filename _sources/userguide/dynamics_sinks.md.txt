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
    interval=50,
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

ZarrData is the recommended choice for production workflows where results need to
survive the process, be shared across machines, or feed back into training.

## Putting it together

A typical dynamics setup combines multiple hooks and sinks to capture different
aspects of the simulation:

```python
from nvalchemi.dynamics import FIRE
from nvalchemi.dynamics.hooks import (
    ConvergenceHook,
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
        ConvergenceHook(fmax=0.05),
        # Log scalars every 10 steps
        LoggingHook(interval=10),
        # Full trajectory to disk every 50 steps
        SnapshotHook(sink=ZarrData("/tmp/traj.zarr"), interval=50),
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
