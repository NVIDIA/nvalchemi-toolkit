<!-- markdownlint-disable MD033 MD007 -->

# NVIDIA ALCHEMI Toolkit

[![PyPI version](https://badge.fury.io/py/nvalchemi-toolkit.svg)](https://badge.fury.io/py/nvalchemi-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/NVIDIA/nvalchemi-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/NVIDIA/nvalchemi-toolkit)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://nvidia.github.io/nvalchemi-toolkit/)

## High-performance deep-learning framework for atomic simulations

NVIDIA ALCHEMI Toolkit is a GPU-first Python framework for building, running, and
deploying AI-driven atomic simulation workflows. It provides a unified interface for
machine-learned interatomic potentials (MLIPs), batched molecular dynamics, and
composable multi-stage simulation pipelines: all designed for high throughput on
NVIDIA GPUs.

### Key Features

- **Bring your own model** &mdash; wrap any MLIP (MACE, AIMNet2, or your own) with
  a standard `BaseModelMixin` that handles input/output adaptation, capability
  negotiation, and runtime control via `ModelConfig`
- **Graph-structured data** &mdash; `AtomicData` and `Batch` provide Pydantic-backed,
  GPU-resident graph representations with built-in serialization to Zarr
- **Composable dynamics** &mdash; subclass `BaseDynamics` for custom integrators;
  compose stages with `+` (single-GPU `FusedStage`) or `|` (multi-GPU
  `DistributedPipeline`)
- **Pluggable hook system** &mdash; nine insertion points per step for logging,
  safety checks, enhanced sampling, profiling, and convergence detection
- **Inflight batching** &mdash; `SizeAwareSampler` replaces graduated samples
  on the fly, maximizing GPU utilization across long-running pipelines
- **High-performance primitives** &mdash; built on
  [`nvalchemi-toolkit-ops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
  for GPU-optimized neighbor lists, dispersion, and electrostatics via
  NVIDIA `warp-lang`
- Agents as first-class citizens; includes core `SKILLS.md` library that
  teach agents how to use `nvalchemi` efficiently in agentic workflows.
  Simply copy the `.claude/skills` folder contents to your project repository
  or home directory depending on use case and agent platform (e.g. Claude
  Code, Cursor, OpenCode).

### Example Snippets

<details>
<summary>Build atomic data and run a batched forward pass</summary>

```python
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# Create two molecules
mol_a = AtomicData(
    positions=torch.randn(4, 3),
    atomic_numbers=torch.tensor([6, 6, 1, 1], dtype=torch.long),
)
mol_b = AtomicData(
    positions=torch.randn(3, 3),
    atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
)

# Batch for GPU-efficient inference
batch = Batch.from_data_list([mol_a, mol_b])

# Wrap a model and run
model = DemoModelWrapper(DemoModel())
outputs = model(batch)
print(outputs["energy"].shape)    # [2, 1] &mdash; one energy per system
print(outputs["forces"].shape)    # [7, 3] &mdash; one force vector per atom
```

</details>

<details>
<summary>Geometry optimization with convergence detection</summary>

```python
from nvalchemi.dynamics import DemoDynamics, ConvergenceHook
from nvalchemi.dynamics.hooks import LoggingHook, NaNDetectorHook

dynamics = DemoDynamics(
    model=model,
    n_steps=10_000,
    dt=0.5,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
    hooks=[LoggingHook(frequency=100), NaNDetectorHook()],
)
with dynamics:
    result = dynamics.run(batch)
```

</details>

<details>
<summary>Multi-stage pipeline: relax then MD (single GPU)</summary>

```python
from nvalchemi.dynamics import DemoDynamics

optimizer = DemoDynamics(model=model, dt=0.5)
md = DemoDynamics(model=model, dt=1.0)

# + fuses stages: one forward pass, masked updates per sub-stage
fused = optimizer + md
with fused:
    fused.run(batch)
```

</details>

<details>
<summary>Distributed pipeline across GPUs</summary>

```python
# Launch with: torchrun --nproc_per_node=2 my_pipeline.py
from nvalchemi.dynamics import DemoDynamics

optimizer = DemoDynamics(model=model, dt=0.5)
md = DemoDynamics(model=model, dt=1.0)

# | distributes stages: one dynamics per GPU rank
pipeline = optimizer | md
with pipeline:
    pipeline.run()
```

</details>

## Installation

The quickest way to install:

```bash
pip install nvalchemi-toolkit
```

For development:

```bash
git clone https://github.com/NVIDIA/nvalchemi-toolkit.git
cd nvalchemi-toolkit
uv sync --all-extras
```

Optional extras:

```bash
pip install nvalchemi-toolkit[mace]       # MACE model support
```

See the [Installation Guide](docs/userguide/about/install.md) for
detailed setup instructions.

## Contributions & Disclaimers

NVIDIA ALCHEMI Toolkit is in public beta. During this phase, the API is subject to
change. Feature requests, bug reports, and general feedback are welcome via
[GitHub Issues](https://github.com/NVIDIA/nvalchemi-toolkit/issues).

## License

Apache 2.0 &mdash; see [LICENSE](LICENSE) for details.
