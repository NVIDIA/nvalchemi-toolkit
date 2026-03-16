(introduction_guide)=

# Introduction to ALCHEMI Toolkit

ALCHEMI Toolkit is a GPU-first deep-learning framework for atomic simulations.
It provides a unified, composable interface for running machine-learning
interatomic potential (MLIP) driven workflows --- geometry optimization,
molecular dynamics, and data generation --- with high throughput on NVIDIA GPUs.

The toolkit is built on PyTorch, validated with Pydantic, and strongly typed
with jaxtyping. Install it as `nvalchemi-toolkit` and import it as `nvalchemi`.

## When to Use ALCHEMI Toolkit

This package is designed for GPU-accelerated workflows in computational chemistry
and machine learning. Common use cases include:

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

## Design Principles

ALCHEMI Toolkit prioritizes performance, correctness, and usability:

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
forces, stresses, periodic boundaries) and a
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

## What's Next?

1. Follow the [installation guide](install_guide) to set up your environment.
2. Read the [data guide](data_guide) to understand how molecular systems are
   represented.
3. Walk through the [models guide](models_guide) to learn how to plug in an
   MLIP.
4. Explore the [dynamics guide](dynamics_guide) to run your first simulation.
5. Check the `examples/` directory for complete working scripts.
6. Copy the `.claude/skills` folder contents into your project
   or home directory to allow agents to access skills to accelerate
   workflows using `nvalchemi`.
