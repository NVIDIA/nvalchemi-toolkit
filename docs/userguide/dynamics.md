<!-- markdownlint-disable MD014 -->

(dynamics_guide)=

# Dynamics: Optimization and Molecular Dynamics

The dynamics module provides a unified framework for running geometry optimizations
and molecular dynamics simulations on GPU. All simulation types share a common
execution loop --- hooks, model evaluation, convergence checking --- so you learn the
pattern once and apply it to any integrator.

```{tip}
It is important to keep in mind that ``nvalchemi`` follows a batch-first principle:
users should think and reason about dynamics workflows with multiple structures
simultaneously, as opposed to individual structures being processed sequentially.
```

## The execution loop

Every simulation is driven by {py:class}`~nvalchemi.dynamics.base.BaseDynamics`,
which defines a single `step()` that all integrators and optimizers follow. The
loop is broken into discrete stages, enumerated by
{py:class}`~nvalchemi.dynamics.base.DynamicsStage`:

| Stage | When it fires |
|-------|---------------|
| `BEFORE_STEP` | At the very beginning of a step, before any operations |
| `BEFORE_PRE_UPDATE` | Just before the integrator's first half-step |
| `AFTER_PRE_UPDATE` | After the first half-step completes |
| `BEFORE_COMPUTE` | Just before the model forward pass |
| `AFTER_COMPUTE` | After the model forward pass completes |
| `BEFORE_POST_UPDATE` | Just before the integrator's second half-step |
| `AFTER_POST_UPDATE` | After the second half-step completes |
| `AFTER_STEP` | At the very end of a step, after all operations |
| `ON_CONVERGE` | When a convergence criterion is met |

A single call to `step()` proceeds through these stages in order:

1. **BEFORE_STEP** hooks fire.
2. `pre_update(batch)` --- the integrator's first half-step (e.g. update velocities
   by half a timestep), bracketed by BEFORE/AFTER_PRE_UPDATE hooks.
3. `compute(batch)` --- the wrapped ML model evaluates forces (and stress, if
   needed), bracketed by BEFORE/AFTER_COMPUTE hooks.
4. `post_update(batch)` --- the integrator's second half-step (e.g. complete the
   velocity update with the new forces), bracketed by BEFORE/AFTER_POST_UPDATE hooks.
5. **AFTER_STEP** hooks fire (convergence checks, logging, ...).
6. Convergence is evaluated: converged systems fire **ON_CONVERGE** hooks and (in
   multi-stage pipelines) migrate to the next stage.

`run(batch, n_steps)` calls `step()` in a loop until all systems converge or
`n_steps` is reached. Every hook declares which
{py:class}`~nvalchemi.dynamics.base.DynamicsStage` stage it should fire at and at
what frequency, so you have fine-grained control over when callbacks execute.

## Using dynamics as a context manager

All dynamics objects (optimizers, integrators, fused stages) support Python's
context manager protocol. The `with` block manages a dedicated
`torch.cuda.Stream` for the simulation and ensures hooks are properly opened and
closed:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook

with FIRE(model=model, dt=0.1, n_steps=500, hooks=[ConvergenceHook.from_fmax(0.05)]) as opt:
    relaxed = opt.run(batch)
```

When you call `run()` without a `with` block, hook setup and teardown happen
automatically inside `run()`. The context manager form is useful when you need to
call `step()` manually or interleave dynamics with other operations while keeping
hook state (e.g. open log files) alive.

## Multi-stage pipelines with FusedStage

Real workflows often chain multiple simulation phases: relax a structure, then run
MD at increasing temperatures, then relax again. The
{py:class}`~nvalchemi.dynamics.base.FusedStage` abstraction lets you compose stages
with the `+` operator:

```python
from nvalchemi.dynamics import FIRE, NVTLangevin, ConvergenceHook

relax = FIRE(model=model, dt=0.1, n_steps=200, hooks=[ConvergenceHook.from_fmax(0.05)])
md = NVTLangevin(model=model, dt=1.0, temperature=300.0, friction=0.01, n_steps=5000)

pipeline = relax + md
with pipeline:
  pipeline.run(batch)
```

Systems start in the first stage (relaxation). As each system converges, it
automatically migrates to the next stage (MD). Different systems can be in different
stages simultaneously --- the batch is partitioned internally, and a single model
forward pass is shared across all active systems regardless of which stage they
belong to.

### Compiling with `torch.compile`

{py:class}`~nvalchemi.dynamics.base.FusedStage` can compile its entire step function
with `torch.compile` to reduce Python overhead and enable kernel fusion. Call
{py:meth}`~nvalchemi.dynamics.base.FusedStage.compile` after composing stages:

```python
fused = (relax + md).compile(fullgraph=True)
with fused:
    fused.run(batch)
```

`compile()` wraps the internal `_step_impl` method --- which includes hook dispatch,
masked sub-stage updates, and the shared model forward pass --- in a single compiled
graph. It returns the same instance, so you can chain it fluently.

You can also defer compilation by passing `compile_step=True` at construction time.
In that case, `torch.compile` is invoked lazily when the context manager is entered:

```python
fused = relax + md  # compile_step inherited from sub-stages or set explicitly
with fused:         # compilation happens here
    fused.run(batch)
```

Any keyword arguments accepted by `torch.compile` (e.g. `fullgraph`, `mode`,
`backend`) can be passed to `.compile()` or stored via `compile_kwargs` at
construction.

```{note}
Not all hooks are graph-break-free under `fullgraph=True`. Hooks that perform
Python-side control flow (e.g. logging, I/O) will introduce graph breaks. If you
need an unbroken graph, ensure your hooks are written with torch-compatible
operations only.
```

## Distributed pipelines

When a workflow needs more than one GPU --- for example, relaxing structures on one
device and running MD on another --- the
{py:class}`~nvalchemi.dynamics.base.DistributedPipeline` distributes stages across
ranks. Where `+` fuses stages onto a single GPU, the `|` operator (or a `stages`
dictionary) assigns one stage per rank and wires up inter-rank communication
automatically.

### Configuring a pipeline

Each rank owns a {py:class}`~nvalchemi.dynamics.base.BaseDynamics` (or
{py:class}`~nvalchemi.dynamics.base.FusedStage`) instance. Stages are collected in a
dictionary keyed by global rank and handed to
{py:class}`~nvalchemi.dynamics.base.DistributedPipeline`:

```python
from nvalchemi.dynamics import FIRE, NVTLangevin, DistributedPipeline
from nvalchemi.dynamics.base import BufferConfig

buffer_cfg = BufferConfig(num_systems=4, num_nodes=50, num_edges=0)

stages = {
    0: FIRE(model=model, buffer_config=buffer_cfg, ...),        # upstream — relaxation
    1: NVTLangevin(model=model, buffer_config=buffer_cfg, ...),  # downstream — MD
}

pipeline = DistributedPipeline(stages=stages, backend="nccl")
with pipeline:
    pipeline.run()
```

By default, `setup()` (called automatically by the context manager) sorts stages by
rank and wires `prior_rank` / `next_rank` between adjacent stages as a simple linear
chain. For more sophisticated topologies --- such as multiple independent
sub-pipelines running in the same job --- set `prior_rank` and `next_rank` explicitly
on each stage:

```python
stages = {
    # Sub-pipeline A: rank 0 → rank 1
    0: FIRE(model=model, buffer_config=buffer_cfg, prior_rank=None, next_rank=1, ...),
    1: NVTLangevin(model=model, buffer_config=buffer_cfg, prior_rank=0, next_rank=None, ...),
    # Sub-pipeline B: rank 2 → rank 3
    2: FIRE(model=model, buffer_config=buffer_cfg, prior_rank=None, next_rank=3, ...),
    3: NVTLangevin(model=model, buffer_config=buffer_cfg, prior_rank=2, next_rank=None, ...),
}
```

The first stage in each sub-pipeline typically owns a *sampler* that feeds new
structures into the chain; the last stage owns one or more *data sinks* that collect
converged results.

```{note}
Each rank currently communicates with at most one upstream and one downstream
neighbour (one-to-one topology). Fan-out (one-to-many) and fan-in (many-to-one)
patterns are planned for a future release.
```

### Sizing the buffer

NCCL point-to-point transfers require fixed-size tensors, so each communicating stage
pre-allocates a send buffer and a receive buffer whose dimensions are set by
{py:class}`~nvalchemi.dynamics.base.BufferConfig`. The three fields control how much
data a single transfer can carry:

| Field | What it controls |
|-------|------------------|
| `num_systems` | Maximum number of graphs (structures) per transfer. Determines throughput per step --- higher values move more data but consume more GPU memory. |
| `num_nodes` | Total atom capacity across all graphs in the buffer. Must be large enough for the worst-case combination of systems. For example, transferring up to 4 structures of at most 50 atoms each requires `num_nodes=200`. |
| `num_edges` | Total edge capacity. Set to **0** when the downstream model recomputes edges via its neighbor list (the common case). Only set a non-zero value if pre-computed edge attributes must be transferred. |

```python
from nvalchemi.dynamics.base import BufferConfig

# 4 structures, up to 200 atoms total, edges recomputed downstream
buffer_cfg = BufferConfig(num_systems=4, num_nodes=200, num_edges=0)
```

When the upstream stage has more converged samples than `num_systems` allows in a
single transfer, the excess stays in the active batch as a no-op until the next
step --- this is the back-pressure mechanism described below.

```{important}
Every pair of communicating stages **must** share an identical
{py:class}`~nvalchemi.dynamics.base.BufferConfig`.
`DistributedPipeline.setup()` validates this and raises an error on mismatch.
```

### Buffer synchronization

The diagram below shows how two adjacent ranks exchange data through pre-allocated
send and receive buffers during a single step. The upstream rank pushes converged
samples; the downstream rank pulls them into its active batch.

```{graphviz}
:caption: Buffer synchronization between two adjacent ranks in a DistributedPipeline.

digraph buffer_sync {
    rankdir=LR
    compound=true
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=11]
    edge [fontname="Helvetica" fontsize=10]

    subgraph cluster_upstream {
        label="Rank 0  (upstream)"
        style=rounded
        color="#4a90d9"
        fontcolor="#4a90d9"

        u_batch [label="active_batch" shape=box style=filled fillcolor="#dce6f1"]
        u_send  [label="send_buffer"  shape=box style=filled fillcolor="#f9e2ae"]
        u_sinks [label="sinks\n(overflow)" shape=box style=dashed]

        u_batch -> u_send [label="converged\nsamples" style=bold]
        u_batch -> u_sinks [label="excess\n(back-pressure)" style=dotted]
    }

    subgraph cluster_downstream {
        label="Rank 1  (downstream)"
        style=rounded
        color="#5bb35b"
        fontcolor="#5bb35b"

        d_recv  [label="recv_buffer"  shape=box style=filled fillcolor="#f9e2ae"]
        d_batch [label="active_batch" shape=box style=filled fillcolor="#dce6f1"]
        d_sinks [label="sinks\n(results)" shape=box style=dashed]

        d_recv -> d_batch [label="incoming\nsamples" style=bold]
        d_batch -> d_sinks [label="converged\nresults" style=bold]
        d_sinks -> d_batch [label="drain when\ncapacity available" style=dotted]
    }

    u_send -> d_recv [label="isend / irecv\n(NCCL)" style=bold color="#c0392b" fontcolor="#c0392b" penwidth=2]
}
```

A step proceeds as follows:

1. **Pre-step** --- The downstream rank zeros its receive buffer and posts an
   asynchronous `irecv` from its `prior_rank`. In `async_recv` mode (the default),
   the wait is deferred until later in the step; in `sync` mode it blocks
   immediately.
2. **Complete receive** --- The downstream rank waits on the pending receive,
   then routes incoming samples into its active batch (or overflow sinks if the
   batch is full).
3. **Step** --- Both ranks execute their respective integrator or optimizer on their
   active batches.
4. **Post-step** --- The upstream rank identifies converged samples, copies them into
   its send buffer (up to `BufferConfig` capacity), and issues an `isend`. An empty
   buffer is always sent to prevent deadlocks. The final stage routes converged
   samples to its sinks instead.

```{tip}
**Back-pressure**: when the send buffer is full, excess converged samples remain in
the upstream active batch as no-ops until buffer capacity opens up. This naturally
throttles fast producers without dropping data.
```

### Communication modes

The `comm_mode` parameter controls how aggressively communication overlaps with
computation:

| Mode | Behavior |
|------|----------|
| `sync` | Blocks on `irecv` immediately in the pre-step. Simplest to debug. |
| `async_recv` *(default)* | Posts `irecv` early, waits only when the data is needed. Overlaps receive with computation. |
| `fully_async` | Also defers `isend` completion to the next step's pre-step. Maximum overlap, highest throughput. |

### Launching

Distributed pipelines are launched with `torchrun` (or any `torch.distributed`
launcher):

```bash
torchrun --nproc_per_node=2 my_pipeline.py
```

`DistributedPipeline` calls `init_distributed()` on entry and coordinates
termination across ranks via an `all_reduce` on per-rank done flags.

```{seealso}
The {doc}`/examples/distributed/index` gallery contains end-to-end examples,
including multi-pipeline topologies and monitoring with persistent storage.
```

## What's next

```{toctree}
:maxdepth: 1

dynamics_simulations
dynamics_hooks
dynamics_sinks
```

- [Optimization and Integrators](dynamics_simulations) --- FIRE, NVE, NVT, NPT and
  their configuration.
- [Hooks](dynamics_hooks) --- the hook protocol, built-in hooks, and writing custom
  hooks.
- [Data Sinks](dynamics_sinks) --- recording trajectories and simulation results.

## See also

- **Examples**: ``basic/02_geometry_optimization.py`` demonstrates a complete relaxation
  workflow.
- **API**: See the {py:mod}`nvalchemi.dynamics` module for the full reference,
  including the hook protocol and distributed pipeline documentation.
- **Data guide**: The [AtomicData and Batch](data_guide) guide covers the input data
  structures consumed by dynamics.
