(introduction_guide)=

# Introduction to ALCHEMI Toolkit

NVIDIA ALCHEMI Toolkit is a **GPU-first Python framework** for building, running,
and deploying AI-driven atomic simulation workflows. It provides a unified interface
for machine-learned interatomic potentials (MLIPs), composable multi-stage simulation
pipelines, and high-throughput infrastructure that keeps your GPUs fully saturated
from prototype to production.

Whether you are relaxing a handful of crystals on a single GPU or screening millions
of candidate structures across a cluster, ALCHEMI Toolkit gives you the same
expressive API and handles the scaling for you.

The core design principles for `nvalchemi` are:

- Batched-first: run all workflows with multiple systems operating
  in parallel to amortize GPU usage.
- Flexibility and extensibility: users are able to insert their desired
  behaviors into workflows with minimal friction, and freely compose
  different elements to achieve what they need holistically.
- Production-quality: optimal developer and end-user experience through
  design choices like `pydantic`, `jaxtyping`, and support for `beartyping`
  to validate inputs (including shapes and data types), which provide
  a first-class experience using modern language server protocols like
  `pyright`, `ruff`, and `ty`.

## When to Use ALCHEMI Toolkit

ALCHEMI Toolkit is designed for GPU-accelerated workflows in computational chemistry
and materials science. Common use cases include:

- **High-throughput molecular dynamics** --- run thousands of structures
  simultaneously in a single batched simulation on one or more GPUs.
- **MLIP integration** --- wrap any PyTorch-based interatomic potential (MACE,
  AIMNet2, or bring your own) behind a standardized interface and immediately use it
  in dynamics workflows.
- **Geometry optimization** --- relax atomic structures to their minimum-energy
  configuration using GPU-accelerated optimizers (FIRE, FIRE2).
- **Multi-stage simulation pipelines** --- chain relaxation, equilibration, and
  production MD phases that share a single model forward pass, and run them
  on a single GPU.
- **Trajectory I/O** --- write and read variable-size atomic graph data
  efficiently with Zarr-backed storage.
- **Extendable Hook interface** --- arbitrary modify and extend the behavior
  of optimizers and dynamics using user-defined functionality.
- **Distributed workflows** --- scale your workflow by building production
  lines: use the {py:class}`~nvalchemi.dynamics.base.DistributedPipeline`
  to spread out your MD campaign across many GPUs and nodes.

```{tip}
ALCHEMI Toolkit follows a **batch-first** philosophy: every API is designed to
process many structures at once rather than looping over them one by one. If your
workflow touches more than a handful of atoms, you will benefit from batching.
```

- **Rapid prototyping** --- wrap a new MLIP in minutes with `BaseModelMixin`,
  compose it with existing force fields using the `+` operator, and plug it into
  any simulation workflow without modifying downstream code.
- **Batched geometry optimization** --- relax thousands of structures in a single
  GPU pass using FIRE or FIRE2, with automatic convergence monitoring.
- **Molecular dynamics** --- run NVE, NVT, or NPT ensembles at scale, driven by
  any supported MLIP (MACE, AIMNet2, or your own model).
- **Multi-stage pipelines** --- chain relaxation, equilibration, and production
  stages on a single GPU (`FusedStage`) or distribute them across many
  (`DistributedPipeline`).
- **High-throughput screening** --- use *inflight batching* to continuously replace
  converged samples, allowing asynchronous workflows to be easily built and
  scaled by users.
- **Dataset generation** --- capture trajectories to Zarr stores with zero-copy GPU
  buffering, then reload them through a CUDA-stream-prefetching `DataLoader` for
  model retraining or active-learning loops.

**GPU-first execution.**
Data structures live on the GPU by default. Neighbor lists, integrator
half-steps, and model forward passes all operate on device-resident tensors,
minimizing host--device transfers.

**Pydantic-backed validation.**
{py:class}`~nvalchemi.data.AtomicData` and configuration objects are Pydantic
models. Fields are validated on construction --- atom counts match tensor shapes,
periodic boundary conditions are consistent with cell vectors, and device/dtype
mismatches are caught early.

**Strong, semantic typing.**
Tensor shapes are annotated with jaxtyping (e.g.,
`Float[Tensor, "V 3"]` for node positions). Semantic type aliases such as
`NodePositions`, `Forces`, and `Energy` make function signatures
self-documenting.

**Composable simulation pipelines.**
Simulation stages compose with Python operators: `+` fuses stages on a single
GPU (sharing the model forward pass), and `|` distributes stages across ranks.
Hooks give fine-grained control over every point in the execution loop.

## Core Components

The toolkit is organized around four layers:

### Data: AtomicData and Batch

{py:class}`~nvalchemi.data.AtomicData` represents a single molecular system as
a graph (atoms as nodes, bonds or neighbors as edges).
{py:class}`~nvalchemi.data.Batch` concatenates many graphs into one
GPU-friendly structure with automatic index offsetting.

See the [data guide](data_guide) for details on construction, batching, and
inflight batching.

### Models: wrapping ML potentials

{py:class}`~nvalchemi.models.base.BaseModelMixin` provides a standardized
wrapper around any PyTorch MLIP. A
{py:class}`~nvalchemi.models.base.ModelCard` declares capabilities (e.g.,
forces, stress, periodic boundaries) and a
{py:class}`~nvalchemi.models.base.ModelConfig` controls runtime behavior.
The wrapper's `adapt_input` / `adapt_output` methods handle format conversion
so the model sees its native tensors and the toolkit sees
{py:class}`~nvalchemi.data.Batch`.

See the [models guide](models_guide) for supported models and how to wrap your
own.

### Dynamics: optimization and MD

The dynamics module provides integrators (NVE, NVT Langevin, NVT Nose--Hoover,
NPT, NPH) and optimizers (FIRE, FIRE2) that share a common execution loop.
Every step passes through a sequence of hook stages --- from `BEFORE_STEP`
to `ON_CONVERGE` --- giving you full control via callbacks.

See the [dynamics guide](dynamics_guide) for the execution loop, multi-stage
pipelines, and hook system.

### Data storage and loading

Zarr-backed writers and readers handle variable-size graph data with a
CSR-style layout. A custom `Dataset` and `DataLoader` support async
prefetching and CUDA-stream-aware loading for overlap of I/O with GPU
computation.

See the [data loading guide](datapipes_guide) for storage formats and
pipeline configuration.

ALCHEMI Toolkit is organized into a small set of tightly integrated modules:

| Module | Purpose | Key Types |
| :--- | :--- | :--- |
| [Data structures](data_guide) | Graph-based atomic representations with Pydantic validation | {py:class}`~nvalchemi.data.AtomicData`, {py:class}`~nvalchemi.data.Batch` |
| [Data loading](datapipes_guide) | Zarr-backed I/O with CUDA-stream prefetching | {py:class}`~nvalchemi.data.datapipes.AtomicDataZarrWriter`, {py:class}`~nvalchemi.data.datapipes.Reader`, {py:class}`~nvalchemi.data.datapipes.Dataset`, {py:class}`~nvalchemi.data.datapipes.DataLoader` |
| [Models](models_guide) | Unified MLIP interface and model composition | {py:class}`~nvalchemi.models.base.BaseModelMixin`, {py:class}`~nvalchemi.models.base.ModelCard`, {py:class}`~nvalchemi.models.composable.ComposableModelWrapper` |
| [Dynamics](dynamics_guide) | Integrators, hooks, and simulation orchestration | {py:class}`~nvalchemi.dynamics.base.BaseDynamics`, {py:class}`~nvalchemi.dynamics.base.FusedStage`, {py:class}`~nvalchemi.dynamics.base.DistributedPipeline` |
| [Hooks](dynamics_hooks_guide) | Pluggable callbacks at nine points per step | {py:class}`~nvalchemi.dynamics.base.Hook`, {py:class}`~nvalchemi.dynamics.hooks.NeighborListHook`, {py:class}`~nvalchemi.dynamics.hooks.SnapshotHook` |
| [Data sinks](dynamics_sinks_guide) | Trajectory capture to GPU buffer, host memory, or disk | {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer`, {py:class}`~nvalchemi.dynamics.sinks.HostMemory`, {py:class}`~nvalchemi.dynamics.sinks.ZarrData` |

## What's Next?

1. **[Install ALCHEMI Toolkit](install)** --- set up your environment with `uv` or `pip`.
2. **[Data structures](data_guide)** --- learn how `AtomicData` and `Batch` represent
   molecular systems as validated, GPU-resident graphs.
3. **[Wrap a model](models_guide)** --- connect your MLIP to the framework with
   `BaseModelMixin`.
4. **[Run a simulation](dynamics_guide)** --- build a dynamics pipeline and capture
   trajectories.
5. **Browse the examples** --- the gallery covers everything from basic relaxation to
   distributed multi-GPU production runs.
