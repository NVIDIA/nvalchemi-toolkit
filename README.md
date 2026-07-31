<!-- markdownlint-disable MD033 MD007 -->

# NVIDIA ALCHEMI Toolkit

[![PyPI version](https://badge.fury.io/py/nvalchemi-toolkit.svg)](https://badge.fury.io/py/nvalchemi-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/gh/NVIDIA/nvalchemi-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/NVIDIA/nvalchemi-toolkit)
[![Documentation](https://img.shields.io/badge/docs-github%20pages-blue)](https://nvidia.github.io/nvalchemi-toolkit/)

## High-performance deep-learning framework for atomic simulations

NVIDIA ALCHEMI Toolkit is a GPU-first Python framework for building, running, and
deploying AI-driven atomic simulation workflows. It provides a unified interface for
machine-learned interatomic potentials (MLIPs), batched molecular dynamics,
composable multi-stage simulation pipelines, and model training and fine-tuning:
all designed for high throughput on NVIDIA GPUs.

### Key Features

- **Batched GPU simulation**:
  - many systems batched together in one pass on a single GPU
  - molecular dynamics: NVE, NVT with Langevin or Nosé-Hoover chain
    thermostats, NPT and NPH
  - geometry relaxation: FIRE and FIRE2, with optional cell relaxation
  - `+` fuses stages so a relax-then-MD workflow costs one forward pass per
    step (`FusedStage`)
  - `SizeAwareSampler` swaps in new samples as others converge, keeping the
    GPU full on long runs

- **Training and fine-tuning** ([guide](docs/userguide/training.md)):
  - composable energy, force and stress losses with weight schedules
  - validation and restartable checkpoints
  - EMA and DDP hooks

- **Multi-GPU scaling** ([guide](docs/userguide/distributed.md)):
  - spatial domain decomposition splits one large system across ranks via
    `DomainParallel` and `ShardTensor`, with automatic halo exchange
  - `|` distributes pipeline stages across ranks (`DistributedPipeline`)

- **Interatomic potentials**:
  - wrappers for MACE, AIMNet2 and UMA
  - your own model via `BaseModelMixin`

- **Interaction terms**:
  - Ewald and particle mesh Ewald electrostatics, which read the per-atom
    charges an MLIP such as AIMNet2 predicts
  - DFT-D3(BJ) dispersion
  - Lennard-Jones, usable on its own as a lightweight potential

- **Data at scale**:
  - `AtomicData` and `Batch`, Pydantic-backed and GPU-resident
  - Zarr serialization for trajectories and datasets
  - `MultiDataset` mixes several datasets with balanced sampling;
    `InMemoryDataset` keeps a hot dataset resident
  - per-sample and per-batch transforms, CUDA-stream prefetching

- **Extensible by design**:
  - one `BaseModelMixin` wrapper, with capabilities declared through
    `ModelConfig`, works unchanged in dynamics, training and multi-GPU runs
  - custom integrators and optimizers by subclassing `BaseDynamics`
  - custom loss functions by subclassing `BaseLossFunction`, and custom
    training steps through `training_fn`
  - custom storage backends by subclassing `Reader`
  - your own hooks at nine per-step insertion points, for logging, safety
    checks, enhanced sampling, profiling and convergence detection
  - stack interaction terms onto a potential with `PipelineModelWrapper`,
    wiring outputs between steps so AIMNet2 supplies the charges Ewald reads

- **Agent skills**: task-specific API guides under `.claude/skills/`
  (see [Using with AI coding agents](#using-with-ai-coding-agents))

Built on [`nvalchemi-toolkit-ops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
for GPU-optimized neighbor lists and interaction kernels via NVIDIA `warp-lang`.

Upgrading from 0.1.0? See [CHANGELOG.md](CHANGELOG.md) for breaking changes and
migration snippets.

### Roadmap

Features planned for upcoming releases:

- **Generative models**: model-agnostic abstraction of generative models for
  the ALCHEMI Toolkit simulation pipeline
- **Crystal structure prediction (CSP) primitives**: composable, batched
  building blocks for molecular CSP workflows
- **Enhanced sampling**: GPU-resident collective variables and biasing methods
- **Model distillation**: pipeline for distilling large, accurate potentials
  into compact models for fast production inference
- **LoRA adapters**: parameter-efficient fine-tuning that maintains many
  specialized variants of one base potential without duplicating its weights
- **Hessians and phonons**: analytical second derivatives through automatic
  differentiation for vibrational and thermodynamic property prediction
- **Domain decomposition optimization**: continued performance improvement of
  spatial domain decomposition
- **Kernel improvements** at the
  [`nvalchemi-toolkit-ops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
  level

### Using with AI coding agents

The repository ships agent-facing guidance at two levels:

- **Skills** — task-specific API guides under `.claude/skills/`. Claude Code
  discovers them automatically when working inside a clone; other platforms
  are routed to the right `SKILL.md` by the table in `AGENTS.md`. To use
  them outside this repository, copy the folder contents into your
  project's or home skills directory.
- **`AGENTS.md`** — repository-wide guidance (setup, conventions, gotchas,
  and the skill routing table). Agents that follow the `AGENTS.md`
  convention (e.g. Codex, Cursor, OpenCode, GitHub Copilot) load it
  natively; the repository also ships a `CLAUDE.md` that imports it, so
  Claude Code gets the same guidance automatically.

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
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DemoDynamics, ConvergenceHook
from nvalchemi.dynamics.hooks import LoggingHook, NaNDetectorHook


# Dynamics reads and writes these per-step buffers, so allocate them up front.
def system(n_atoms: int, z: int) -> AtomicData:
    return AtomicData(
        positions=torch.randn(n_atoms, 3),
        atomic_numbers=torch.full((n_atoms,), z, dtype=torch.long),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
        velocities=torch.zeros(n_atoms, 3),
    )


batch = Batch.from_data_list([system(4, 6), system(3, 8)])

dynamics = DemoDynamics(
    model=model,
    n_steps=10_000,
    dt=0.5,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
    hooks=[
        LoggingHook(backend="csv", log_path="run.csv", frequency=100),
        NaNDetectorHook(),
    ],
)
with dynamics:
    result = dynamics.run(batch)
```

</details>

<details>
<summary>Multi-stage pipeline: relax then MD (single GPU)</summary>

```python
from nvalchemi.dynamics import DemoDynamics

optimizer = DemoDynamics(model=model, n_steps=500, dt=0.5)
md = DemoDynamics(model=model, n_steps=1_000, dt=1.0)

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

optimizer = DemoDynamics(model=model, n_steps=500, dt=0.5)
md = DemoDynamics(model=model, n_steps=1_000, dt=1.0)

# | distributes stages: one dynamics per GPU rank
pipeline = optimizer | md
with pipeline:
    pipeline.run()
```

</details>

<details>
<summary>Train a model with validation and checkpointing</summary>

```python
import torch
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStrategy,
    ValidationConfig,
    default_training_fn,
)

# Assumes `model` is a BaseModelMixin wrapper and `train_loader` /
# `val_loader` are nvalchemi DataLoaders (see the data pipeline guide).
device = torch.device("cuda")

# Compose an objective: weighted energy + force terms
loss_fn = 1.0 * EnergyMSELoss() + 10.0 * ForceMSELoss()

strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    ),
    num_steps=10_000,
    training_fn=default_training_fn,
    loss_fn=loss_fn,
    devices=[device],
    validation_config=ValidationConfig(
        validation_data=val_loader,
        validation_fn=default_training_fn,
        loss_fn=loss_fn,
        every_n_steps=500,
    ),
)
strategy.run(train_loader)
print(strategy.last_validation)
```

For a complete runnable script, see
[`examples/advanced/10_mace_training.py`](examples/advanced/10_mace_training.py).

</details>

<details>
<summary>Split one large system across GPUs (domain decomposition)</summary>

```python
# Launch with: torchrun --nproc_per_node=2 my_dd_run.py
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nvalchemi.distributed import DomainConfig, DomainParallel
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.models.mace import MACEWrapper

dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{dist.get_rank()}")
torch.cuda.set_device(device)
mesh = DeviceMesh(
    "cuda", list(range(dist.get_world_size())), mesh_dim_names=("domain",)
)

# The wrapper and the integrator are the same objects you would use on a
# single GPU; `batch` is the full system, built on rank 0 only.
wrapper = MACEWrapper.from_checkpoint("medium-0b2", device=device).eval()
integrator = NVTLangevin(
    model=wrapper, dt=0.5, temperature=300.0, friction=0.01, n_steps=200
)

# One DomainConfig + one wrap is the entire user-facing addition. Atoms are
# partitioned spatially; halo exchange and cross-rank reductions are automatic.
domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)
dynamics = DomainParallel(dynamics=integrator, config=domain_cfg, n_steps=200)

owned = dynamics.partition(batch if dist.get_rank() == 0 else None)
dynamics.run(owned)
dynamics.close()
```

For complete runnable scripts, see
[`examples/distributed/`](examples/distributed/) and the
[distributed guide](docs/userguide/distributed.md).

</details>

## Installation

The quickest way to install:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.nvidia.com \
  'nvalchemi-toolkit[cu13]'
```

For development:

```bash
git clone https://github.com/NVIDIA/nvalchemi-toolkit.git
cd nvalchemi-toolkit
uv sync --extra cu13
```

`cu13` is the default development CUDA variant. For CUDA 12 environments, run
`uv sync --extra cu12` instead and pass the same extra to `uv run`, for example
`uv run --extra cu12 pytest test/`. The Makefile does this automatically:
`make test CUDA_EXTRA=cu12`. CUDA-aligned optional extras follow the same
pattern, for example `uv sync --extra cu12 --extra mace` or
`make test CUDA_EXTRA=cu12 OPTIONAL_EXTRAS=mace`. To include documentation
dependencies, add `--group docs`. Avoid `uv sync --all-extras`, because the
CUDA variants are mutually exclusive.

Optional extras:

```bash
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.nvidia.com \
  'nvalchemi-toolkit[cu12]'               # Specify CUDA 12 version
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.nvidia.com \
  'nvalchemi-toolkit[cu13,mace]'          # MACE model support, CUDA 13
pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.nvidia.com \
  'nvalchemi-toolkit[cu12,mace]'          # MACE model support, CUDA 12
```

The `uma` extra is mutually exclusive with `mace` and the CUDA extras
(incompatible `e3nn` / `torch` pins) and resolves into its own environment.

See the [Installation Guide](docs/userguide/about/install.md) for
detailed setup instructions.

## Contributions & Disclaimers

NVIDIA ALCHEMI Toolkit is in public beta. During this phase, the API is subject to
change. Feature requests, bug reports, and general feedback are welcome via
[GitHub Issues](https://github.com/NVIDIA/nvalchemi-toolkit/issues).

## License

Apache 2.0 &mdash; see [LICENSE](LICENSE) for details.
