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

- **Rapid prototyping** --- wrap a new MLIP in minutes with `BaseModelMixin`,
  compose it with existing force fields using the `+` operator, and plug it into
  any simulation workflow without modifying downstream code.
- **Batched geometry optimization** --- relax thousands of structures in a single
  GPU pass using FIRE or LBFGS, with automatic convergence monitoring.
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

## Core Components

ALCHEMI Toolkit is organized into a small set of tightly integrated modules:

| Module | Purpose | Key Types |
| :--- | :--- | :--- |
| [Data structures](data_guide) | Graph-based atomic representations with Pydantic validation | `AtomicData`, `Batch` |
| [Data loading](datapipes_guide) | Zarr-backed I/O with CUDA-stream prefetching | `Writer`, `Reader`, `Dataset`, `DataLoader` |
| [Models](models_guide) | Unified MLIP interface and model composition | `BaseModelMixin`, `ModelCard`, `ComposableModelWrapper` |
| [Dynamics](dynamics_guide) | Integrators, hooks, and simulation orchestration | `BaseDynamics`, `FusedStage`, `DistributedPipeline` |
| [Hooks](dynamics_hooks_guide) | Pluggable callbacks at nine points per step | `Hook`, `NeighborListHook`, `SnapshotHook` |
| [Data sinks](dynamics_sinks_guide) | Trajectory capture to GPU buffer, host memory, or disk | `GPUBuffer`, `HostMemory`, `ZarrData` |

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
