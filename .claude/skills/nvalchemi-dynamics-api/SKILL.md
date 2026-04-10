---
name: nvalchemi-dynamics-api
description: How to configure and run dynamics simulations, compose multi-stage pipelines (FusedStage, DistributedPipeline), use inflight batching, and manage data sinks.
---

# nvalchemi Dynamics API

## Overview

The dynamics API provides tools to discover available dynamics classes, configure them,
and scale simulations up (single GPU) and out (multi-rank pipelines).

```python
from nvalchemi.dynamics import (
    BaseDynamics,
    DemoDynamics,
    FusedStage,
    DistributedPipeline,
    ConvergenceHook,
    Hook,
    DynamicsStage,
    SizeAwareSampler,
    DataSink, GPUBuffer, HostMemory, ZarrData,
    hooks,
)
```

---

## Available dynamics classes

| Class | Description |
|-------|-------------|
| `BaseDynamics` | Abstract base — subclass to create integrators |
| `DemoDynamics` | Velocity Verlet reference implementation (testing only) |

To find all dynamics classes in a codebase, search for subclasses of `BaseDynamics`.

---

## Configuring a dynamics run

### Basic setup

```python
from nvalchemi.dynamics import DemoDynamics, ConvergenceHook
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.data import AtomicData, Batch
import torch

model = DemoModelWrapper()
dynamics = DemoDynamics(
    model=model,
    n_steps=1000,       # total steps for run()
    dt=0.5,             # timestep (fs)
)

# Create batch with required state
data = AtomicData(
    atomic_numbers=torch.tensor([6, 8, 1], dtype=torch.long),
    positions=torch.randn(3, 3),
)
batch = Batch.from_data_list([data])
batch.forces = torch.zeros(3, 3)
batch.energy = torch.zeros(1, 1)

result = dynamics.run(batch)
```

### With convergence detection

```python
dynamics = DemoDynamics(
    model=model,
    n_steps=10000,
    dt=0.5,
    convergence_hook=ConvergenceHook(
        criteria=[
            {"key": "fmax", "threshold": 0.05, "reduce_op": "max"},
            {"key": "energy_change", "threshold": 1e-6},
        ],
        frequency=1,
    ),
)
```

Shorthand for force-based convergence:

```python
dynamics = DemoDynamics(
    model=model,
    n_steps=10000,
    dt=0.5,
    convergence_hook=ConvergenceHook.from_fmax(threshold=0.05),
)
```

### With hooks

```python
from nvalchemi.dynamics.hooks import MaxForceClampHook, LoggingHook

dynamics = DemoDynamics(
    model=model,
    n_steps=1000,
    dt=0.5,
    hooks=[
        MaxForceClampHook(max_force=10.0),
        LoggingHook(frequency=100),
    ],
)
```

---

## Scaling up: FusedStage (single GPU, multiple stages)

`FusedStage` composes multiple dynamics stages on one GPU with a **single shared
model forward pass per step**. Samples migrate between stages via convergence.

### Composition with `+` operator

```python
# Stage 0: geometry optimization
opt = DemoDynamics(
    model=model, n_steps=100, dt=1.0,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
)

# Stage 1: production MD
md = DemoDynamics(model=model, n_steps=500, dt=0.5)

# Compose — auto-registers convergence hook to migrate status 0 → 1
fused = opt + md

result = fused.run(batch)  # runs until all samples reach exit_status
```

### Constructor (explicit)

```python
fused = FusedStage(
    sub_stages=[(0, opt), (1, md)],
    entry_status=0,          # initial sample status
    exit_status=2,           # auto-set to len(sub_stages) if -1
    compile_step=False,      # enable torch.compile
    compile_kwargs=None,     # kwargs for torch.compile
)
```

### With torch.compile

```python
fused = (opt + md)
fused.compile(fullgraph=True, mode="reduce-overhead")

with fused:  # lazy compilation on context entry
    result = fused.run(batch)
```

### How it works

1. Each sample has a `status` field (integer)
2. Each sub-stage processes only samples matching its status code
3. When a sub-stage's `ConvergenceHook` fires, converged samples' status increments
4. Samples at `exit_status` are graduated (no longer updated)
5. `run()` loops until all samples reach `exit_status` or sampler is exhausted

### Chaining more stages

```python
fused_3 = opt + md + analysis   # 3 sub-stages: status 0, 1, 2
fused_3 = fused + extra_stage   # append to existing FusedStage
```

---

## Scaling out: DistributedPipeline (multi-rank)

`DistributedPipeline` chains dynamics stages across multiple ranks using
`torch.distributed`. Each rank runs one stage; converged samples are sent
to the next rank.

### Composition with `|` operator

```python
opt_stage = DemoDynamics(model=model, n_steps=100, dt=1.0)   # rank 0
md_stage = DemoDynamics(model=model, n_steps=500, dt=0.5)    # rank 1

pipeline = opt_stage | md_stage

with pipeline:   # initializes torch.distributed + setup
    pipeline.run()
```

### Constructor (explicit)

```python
pipeline = DistributedPipeline(
    stages={0: opt_stage, 1: md_stage},
    synchronized=False,   # True for debugging (adds barriers)
)
```

### Communication modes

Control how inter-rank buffers synchronize:

```python
stage = DemoDynamics(
    model=model, n_steps=100, dt=0.5,
    comm_mode="async_recv",   # default: deferred blocking
    # comm_mode="sync",       # immediate blocking (debugging)
    # comm_mode="fully_async", # maximum overlap
)
```

The default `comm_mode` is `"async_recv"`. The three modes differ in when blocking occurs:

- `"sync"`: `irecv` completes inline in `_prestep_sync_buffers` — simplest, good for debugging
- `"async_recv"`: `irecv` is posted in `_prestep_sync_buffers` but `wait()` deferred to `_complete_pending_recv` — allows compute/communication overlap
- `"fully_async"`: both send and receive are deferred — maximum overlap, highest throughput; pending sends from the previous step are drained at the start of the next `_prestep_sync_buffers`

### Pre-allocated buffers

For high-throughput pipelines, pre-allocate send/recv buffers:

```python
from nvalchemi.dynamics.base import BufferConfig

buffer_cfg = BufferConfig(
    num_systems=100,   # max graphs
    num_nodes=5000,    # total node capacity
    num_edges=20000,   # total edge capacity
)

stage = DemoDynamics(
    model=model, n_steps=100, dt=0.5,
    buffer_config=buffer_cfg,
)
```

Buffers are **lazily initialized** on the first step using the first concrete batch as a template for attribute keys, dtypes, and shapes. This means the first step has slightly more overhead.

Adjacent stages must use identical `BufferConfig` values — this is validated in `DistributedPipeline.setup()`.

---

## Buffer semantics and communication

### Three buffer layers

The dynamics framework manages data flow through three layers:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Active batch** | `_CommunicationMixin.active_batch` | Working set being integrated |
| **Communication buffers** | `send_buffer` / `recv_buffer` | Pre-allocated `Batch.empty()` for zero-copy inter-rank transfer |
| **Overflow sinks** | `DataSink` list (priority-ordered) | Staging when active batch is full |

### Communication protocol (DistributedPipeline)

Each pipeline step follows a four-phase protocol:

1. `_prestep_sync_buffers()` — zeros send buffer, posts `irecv` from prior rank
2. `_complete_pending_recv()` — waits on deferred recv, routes into active batch, drains overflow sinks
3. `step()` — dynamics integration
4. `_poststep_sync_buffers(converged_indices)` — extracts converged into send buffer, sends to next rank

**Deadlock prevention:** when no samples converge, an empty send buffer is still sent so the downstream `irecv` completes.

### Back-pressure

When `send_buffer` has limited capacity (via `BufferConfig`):

- Only `min(converged_count, remaining_capacity)` samples are extracted
- Excess converged samples remain in the active batch as **no-ops** — their positions/velocities are saved before the integrator and restored after
- Without `BufferConfig`, all converged samples are sent without constraints (backward compat)

### Buffer lifecycle: put/defrag/zero

```python
# Pre-allocated buffer created via Batch.empty()
buffer = Batch.empty(num_systems=100, num_nodes=5000, num_edges=20000, template=batch)

# Copy selected graphs into buffer (Warp GPU kernels, float32 only)
mask = converged_mask  # bool tensor, True = copy this graph
buffer.put(src_batch, mask)

# Remove copied graphs from source in-place
src_batch.defrag()

# Reset buffer for reuse (preserves allocated memory)
buffer.zero()
```

**Important:** `Batch.put()` uses Warp GPU kernels that only handle float32 attributes. Adjacent pipeline stages must have identical `BufferConfig` values.

### Data routing methods

| Method | Purpose |
|--------|---------|
| `_recv_to_batch(incoming)` | Route received data through recv buffer into active batch |
| `_buffer_to_batch(incoming)` | Append to active batch, overflow to sinks if full |
| `_batch_to_buffer(mask)` | Copy graduated samples into send buffer, defrag active batch |
| `_overflow_to_sinks(batch)` | Write to first non-full sink in priority order |
| `_drain_sinks_to_batch()` | Pull from sinks back into active batch when room available |

---

## Inflight batching with SizeAwareSampler

For streaming workflows, `SizeAwareSampler` manages dataset access with
bin-packing for size-matched batching. As samples converge and leave
the batch, new samples are pulled from the dataset.

```python
from nvalchemi.dynamics import SizeAwareSampler

sampler = SizeAwareSampler(
    dataset=dataset,                # must have __len__, __getitem__, get_metadata
    max_atoms=1000,                 # max total atoms per batch (None = auto from GPU)
    max_edges=5000,                 # max total edges per batch
    max_batch_size=32,              # max graphs per batch
    bin_width=10,                   # group samples by atom count bins
    shuffle=False,
    max_gpu_memory_fraction=0.8,    # for auto max_atoms estimation
)

# Build initial batch via greedy bin-packing
batch = sampler.build_initial_batch()

# Request a replacement sample that fits constraints
replacement = sampler.request_replacement(num_atoms=50, num_edges=200)

# Check if all samples consumed
sampler.exhausted  # bool
```

### How inflight replacement works (`_refill_check`)

When `refill_frequency` triggers (every N steps), `_refill_check()`:

1. Identifies graduated graphs (`status >= exit_status`)
2. Writes graduated graphs to sinks
3. Extracts remaining graphs via `Batch.index_select`
4. Requests replacements from sampler (one per graduated slot, matching atom/edge budget)
5. Appends replacements via `Batch.append`
6. Rebuilds `status` (replacements get `0`) and `fmax` (replacements get `inf`) tensors

This produces a **new** `Batch` object (not in-place mutation). Returns `None` when the sampler is exhausted and no active samples remain.

### With FusedStage

```python
opt = DemoDynamics(
    model=model, n_steps=100, dt=1.0,
    sampler=sampler,
    refill_frequency=1,    # check for replacements every N steps
    convergence_hook=ConvergenceHook.from_fmax(0.05),
)
md = DemoDynamics(model=model, n_steps=500, dt=0.5)

fused = opt + md
with fused:
    result = fused.run()  # no batch arg — built from sampler
```

### With DistributedPipeline (first stage only)

```python
first_stage = DemoDynamics(
    model=model, n_steps=100, dt=1.0,
    sampler=sampler,
    refill_frequency=1,
)
pipeline = first_stage | second_stage
with pipeline:
    pipeline.run()  # first stage auto-refills from sampler
```

---

## Data sinks

Sinks store graduated/snapshot data. Used by `SnapshotHook` and the communication layer.

```python
from nvalchemi.dynamics import GPUBuffer, HostMemory, ZarrData
```

### DataSink interface

```python
class DataSink(ABC):
    def write(self, batch: Batch, mask: Tensor | None = None) -> None: ...
    def read(self) -> Batch: ...
    def zero(self) -> None: ...           # clear contents
    def drain(self) -> Batch: ...         # read() + zero()
    def is_full(self) -> bool: ...        # len(self) >= capacity
    def __len__(self) -> int: ...
    @property
    def capacity(self) -> int: ...
```

### Implementations

**GPUBuffer** — GPU-resident, pre-allocated.

```python
gpu_sink = GPUBuffer(
    capacity=100,       # max graphs
    max_atoms=5000,     # total node capacity
    max_edges=20000,    # total edge capacity
    device="cuda",
)
```

**HostMemory** — CPU-resident list.

```python
cpu_sink = HostMemory(capacity=1000)
```

**ZarrData** — Disk-backed persistent storage.

```python
zarr_sink = ZarrData(
    store="trajectory.zarr",   # path, S3 URI, or dict
    capacity=1_000_000,
)
```

### Using sinks with dynamics

```python
from nvalchemi.dynamics.hooks import SnapshotHook

dynamics = DemoDynamics(
    model=model, n_steps=1000, dt=0.5,
    sinks=[gpu_sink, zarr_sink],   # for pipeline communication
    hooks=[
        SnapshotHook(sink=zarr_sink, frequency=10),
    ],
)
```

---

## ConvergenceHook reference

```python
ConvergenceHook(
    criteria=[                              # AND semantics (all must pass)
        {
            "key": "fmax",                  # batch attribute to check
            "threshold": 0.05,              # convergence threshold
            "reduce_op": "max",             # min, max, norm, mean, sum
            "reduce_dims": -1,              # dimensions to reduce
            "custom_op": None,              # custom callable
        },
    ],
    source_status=0,                        # check samples with this status
    target_status=1,                        # migrate converged samples
    frequency=1,                            # check every N steps
)

# Shorthand
ConvergenceHook.from_fmax(
    threshold=0.05,
    source_status=None,
    target_status=None,
    frequency=1,
)
```

---

## Full workflow example

```python
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.dynamics import (
    DemoDynamics, ConvergenceHook, SizeAwareSampler, ZarrData,
)
from nvalchemi.dynamics.hooks import MaxForceClampHook, LoggingHook, SnapshotHook

# Model
model = DemoModelWrapper()

# Dataset + sampler for inflight batching
reader = AtomicDataZarrReader("structures.zarr")
dataset = Dataset(reader, device="cuda")
sampler = SizeAwareSampler(
    dataset=dataset, max_atoms=1000, max_edges=5000, max_batch_size=32,
)

# Output sink
output = ZarrData("results.zarr", capacity=100000)

# Stage 0: optimize
opt = DemoDynamics(
    model=model, n_steps=500, dt=1.0,
    sampler=sampler, refill_frequency=1,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
    hooks=[MaxForceClampHook(max_force=10.0)],
)

# Stage 1: MD
md = DemoDynamics(
    model=model, n_steps=1000, dt=0.5,
    convergence_hook=ConvergenceHook(
        criteria=[{"key": "fmax", "threshold": 0.01}],
    ),
    hooks=[
        LoggingHook(frequency=100),
        SnapshotHook(sink=output, frequency=50),
    ],
)

# Compose and run
fused = opt + md
with fused:
    fused.run()  # batch built from sampler, runs until exhausted
```
