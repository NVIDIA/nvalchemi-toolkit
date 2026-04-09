# NVIDIA ALCHEMI Toolkit

**GPU-first Python framework for AI-driven atomic simulations.**

NVIDIA ALCHEMI Toolkit gives you a unified, composable API for
machine-learned interatomic potentials --- from single-GPU prototyping to
distributed high-throughput production. Wrap any MLIP, assemble multi-stage
simulation pipelines with Python operators, and let inflight batching keep
your hardware fully utilized.

---

## Who Is This For?

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Computational Chemists
Run batched geometry/cell optimization, molecular dynamics (NVE/NVT/NPT), or
multi-stage relaxation-equilibration-production workflows --- all from a
single Python script.

[Data structures →](data_guide)
:::

:::{grid-item-card} ML Researchers
Plug your own potential into the framework with `BaseModelMixin`, compose
it with existing force fields, and generate simulation data through
GPU-buffered trajectory capture.

[Model interface →](models_guide)
:::

:::{grid-item-card} HPC Engineers
Scale from one GPU to an entire node with `DistributedPipeline`. Inflight
batching and size-aware sampling handle load balancing automatically.

[Dynamics pipelines →](dynamics_guide)
:::
::::

[Get started →](userguide/about/install)

---

## Highlights

- **Bring your own model** --- wrap MACE, AIMNet2, or any PyTorch MLIP in a
  few lines with a standardized `ModelCard` interface.
- **Compose, don't configure** --- fuse stages on one GPU with `+`, distribute
  across GPUs with `|`, and inject behavior at nine hook points per step.
- **GPU-native data** --- `AtomicData` and `Batch` are Pydantic-validated,
  `jaxtyping`-annotated graph structures that live on-device.
- **Inflight batching** --- converged samples are replaced mid-run so the GPU
  never idles during high-throughput screening.
- **Zarr-backed I/O** --- write trajectories with zero-copy GPU buffering;
  reload through a CUDA-stream-prefetching `DataLoader`.

[Read the introduction →](introduction_guide)

---

## User Guide

```{toctree}
:maxdepth: 2

userguide/index
```

## Models

```{toctree}
:maxdepth: 2

models/index
```

## Examples

```{toctree}
:maxdepth: 2

examples/index
```

## API

```{toctree}
:maxdepth: 2

API <modules/index>
```
