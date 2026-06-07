<!-- markdownlint-disable MD014 -->

(charged_mace_training_example)=

# Charged MACE Training Example

This guide walks through a complete model-training lifecycle using the ALCHEMI Toolkit. To anchor these concepts in a realistic workflow, we will build and train a charged MACE model using the MatPES r2SCAN dataset. The runnable end-to-end training script for this workflow can be found in
[`examples/advanced/10_mace_training.py`](../../examples/advanced/10_mace_training.py).

The ALCHEMI training workflow has the following structure:

```text
[Graph Data] ➔ [Model Architecture] ➔ [Supervised Objective] ➔ [Runtime Hooks] ➔ [Training Strategy]
```

The guide below focuses on the core ALCHEMI APIs: data pipes, model wrappers,
loss composition, hooks, validation config, optimizer config, and
{py:class}`~nvalchemi.training.TrainingStrategy`.

## 1. Data: MatPES r2SCAN Dataset

This pipeline reads MatPES r2SCAN structures from ALCHEMI-compatible Zarr splits.

Each sample contains the graph inputs needed by the model: atomic positions,
atom types, periodic boundary condition metadata, and supervised labels such as
energy, forces, and optionally partial charges.


## 2. Model: Charged MACE in Brief

While multiple MACE architectures support charge modeling, the variation described below is built specifically to demonstrate the ALCHEMI training pipeline. It is presented for pedagogical clarity as one possible way to integrate charge prediction and equilibration, rather than the definitive approach.

To capture short-range atomic interactions, we utilize MACE (equivariant many-body interatomic potentials). Standard MACE maps equivariant node and edge features to extract local atomic energy contributions.

The charged MACE variant here adds two readout heads on the final node features: one predicts raw per-atom charges, the other predicts redistribution weights. Charge equilibration follows the AIMNET2 strategy—the raw charges are adjusted so each structure's per-atom charges sum to the target total charge. The model exposes three outputs:
- `raw_charges`: Unconstrained per-atom charges from the raw charge head.
- `charge_weights`: Non-negative redistribution weights from the charge weight head.
- `charges`: Equilibrated per-atom charges after the AIMNET2-style projection.

```
    ┌──► Short-Range Energy Head ──────────────────────┐
    │                                                  ▼
 Final Node Features                             Total Energy ──► Autograd 
    │                                                  ▲         (Forces)
    ├──► Raw Charge Head ─┐                            │
    └──► Charge Weight ───┴─► Equilibrated ──► Ewald ──┘
            Head                Charges        Energy
```

Equilibrated `charges` are passed to a separate Ewald model, which computes long-range electrostatic energy from those charges. Total energy is the sum of short-range MACE energy and Ewald energy; forces come from autograd through that combined energy.

The configuration we train in this experiment is a 3.51M-parameter Charged MACE model in total (short-range backbone plus two charge heads).

## 3. Build the Data Pipelines

Next, we construct the executable pipeline that streams data from disk into batched tensors.

The {py:class}`~nvalchemi.data.datapipes.AtomicDataZarrReader` streams the raw data, and the DataLoader compiles individual graphs into batched tensors ready for GPU acceleration.

```python
from pathlib import Path
import torch
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
pin_memory = device.type == "cuda"

# Stream and compile training batches
reader = AtomicDataZarrReader(
    Path("/path/to/r2scan-2025.2-train.zarr"),
    pin_memory=pin_memory,
)
dataset = Dataset(reader, device=device, num_workers=1)
train_batches = DataLoader(dataset, batch_size=64, shuffle=True)

# Stream and compile validation batches
val_reader = AtomicDataZarrReader(
    Path("/path/to/r2scan-2025.2-valid.zarr"),
    pin_memory=pin_memory,
)
val_dataset = Dataset(val_reader, device=device, num_workers=1)
val_batches = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

This guide uses a fixed batch size of 32 for training. For this dataset, structure sizes can run up to 240 atoms, so an unlucky draw of larger graphs can still spike memory, especially with charged MACE and Ewald electrostatics. When memory is tighter or the size distribution is heavier-tailed, {py:class}`nvalchemi.data.datapipes.SizeAwareBatchSampler` is a useful alternative: it limits atom count in a batch so each batch stays within a chosen memory budget.

### Infer model metadata from the dataset

Before building the model, populate dataset-derived metadata in your Hydra config: `E0s` (keys give `atomic_numbers`; from structure-energy regression or isolated-atom DFT), `avg_num_neighbors`, and the ScaleShiftMACE pair `atomic_inter_shift` / `atomic_inter_scale`. Run `compute_mace_metadata.py` on the training split once, or supply equivalent values from your own preprocessing.

## 4. Defining the Multi-Objective Loss

The model fits three physical quantities here: energies, forces, and charges. The core training API represents this as a composed objective: individual loss terms can be added and scaled directly, with scalar weights balancing their relative importance.

```python
from nvalchemi.training import (
    BaseLossFunction,
    EnergyHuberLoss,
    ForceHuberLoss,
)

charge_loss: BaseLossFunction = build_charge_loss(...)
loss_fn = (
    EnergyHuberLoss(delta=0.01)
    + 10.0 * ForceHuberLoss(delta=0.01)
    + 0.1 * charge_loss
)
```

Under the hood, the `+` and `*` operators are syntactic sugar for constructing a
`ComposedLossFunction`. For two-stage weight schedule, replace fixed scalar weights with
loss-weight schedules such as `PiecewiseWeight` for a step change or
`LinearWeight`/`CosineWeight` for a ramp over steps or epochs.

Stress can also be used as a training target, for example by adding `StressHuberLoss(...)` or `StressMSELoss(...)` to the composed loss and enabling the model's `stress` output, but this example fits only energy, forces, and charges.

## 5. Assembling the TrainingStrategy

With our data loaders, architecture, and objectives defined, we unify them inside a {py:class}`nvalchemi.training.TrainingStrategy`. This orchestration object governs the model execution graph, backpropagation, optimizer states, validation, and runtime hooks.

```python
from nvalchemi.models.base import BaseModelMixin

# 1. Initialize the architecture
model: BaseModelMixin = build_model(...)
model.model_config.active_outputs = {"energy", "forces", "charges"}
```

`build_model(...)` is deliberately left recipe-specific here. The core requirement
is that the object follows the {py:class}`~nvalchemi.models.base.BaseModelMixin`
interface used by `TrainingStrategy`.

### Harnessing Runtime Hooks

To augment our core training loop without cluttering it, we leverage Hooks. For example, the {py:class}`~nvalchemi.hooks.NeighborListHook` handles the critical task of rebuilding the atomic interaction radius graph immediately before every forward pass.

```python
import torch

from nvalchemi.distributed import DistributedManager
from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import (
    CheckpointHook,
    DDPHook,
    EMAHook,
    OptimizerConfig,
    TrainingStage,
    TrainingStrategy,
    ValidationConfig,
    default_training_fn,
)

DistributedManager.initialize()
manager = DistributedManager()
device = torch.device(manager.device)

# 2. Attach lifecycle hooks
hooks = [
    DDPHook(backend="nccl", sampler_kwargs={"seed": 42}),
    EMAHook(model_key="main", decay=0.995, update_every=1),
    NeighborListHook(model.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD),
    CheckpointHook("outputs/checkpoints", step_interval=5000),
]

# 3. Configure strategy-owned validation
# Validation reuses BEFORE_FORWARD hooks, so the NeighborListHook above also
# prepares validation batches before model forward.
validation_config = ValidationConfig(
    validation_data=val_batches,
    validation_fn=default_training_fn,
    loss_fn=loss_fn,
    every_n_steps=5000,
    grad_mode="auto",
    use_ema="auto",
    name="validation",
)

# 4. Synthesize into an executable strategy
strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 5.0e-3},
        scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs={"T_max": 100_000, "eta_min": 5.0e-4},
    ),
    num_epochs=None,
    num_steps=100_000,
    training_fn=default_training_fn,
    loss_fn=loss_fn,
    devices=[device],
    distributed_manager=manager,
    hooks=hooks,
    validation_config=validation_config,
)

# Run the training loop
strategy.run(train_batches)
```

Runtime behavior such as distributed wrapping, EMA, neighbor-list construction,
checkpointing, and logging is attached through hooks. Validation is owned by
{py:class}`~nvalchemi.training.ValidationConfig`, not a validation hook, so the
same strategy can run training and held-out evaluation while respecting EMA and
distributed reductions.

Schedulers are attached through {py:class}`~nvalchemi.training.OptimizerConfig`. The concise choice above is PyTorch's `CosineAnnealingLR`; you can also pass other PyTorch schedulers, or custom `torch.optim.lr_scheduler.LRScheduler` subclasses, by changing `scheduler_cls` and `scheduler_kwargs`.

## 6. Configuring and Launching Training Runs

In this example, we use the Hydra config
[`examples/advanced/10_mace_training.yaml`](../../examples/advanced/10_mace_training.yaml).
The excerpt below highlights the main training-run fields; see the full config for
model architecture, dataset metadata, and additional defaults.

```yaml
# 10_mace_training.yaml (excerpt)
data:
  zarr_path: /path/to/r2scan-2025.2-train.zarr
  validation_zarr_path: /path/to/r2scan-2025.2-valid.zarr

training:
  steps: 100000
  batch_size: 32
  validation:
    enabled: true
    batch_size: 64
  ema:
    decay: 0.995
  checkpoint:
    dir: outputs/checkpoints
  loss:
    energy_weight: 1.0
    force_weight: 10.0
    charge_weight: 0.1
    # ...
  optimizer:
    lr: 0.005
  scheduler:
    type: cosine
  # ...

model:
  model_type: charged_mace
  dtype: float32
  # MACE architecture knobs
  hidden_irreps: "128x0e"
  num_interactions: 2
  num_bessel: 10
  max_ell: 3
  correlation: 3
  r_max: 5.0
  # Dataset-derived metadata (from compute_mace_metadata.py or equivalent)
  avg_num_neighbors: 31.26582868498974
  atomic_inter_shift: 0.0
  atomic_inter_scale: 1.8222970948949255
  E0s:
    1: -1.12256101
    6: -2.37153267
    8: -3.22079709
    # ... one entry per element in the training split

```

Then the single-GPU run can be launched with the config file name and any run-specific overrides on the command line:

```bash
uv run python examples/advanced/10_mace_training.py
```

For distributed training with data parallelism, launch on a node with torchrun to leverage the `DDPHook` that wraps the model in DDP at `TrainingStage.BEFORE_TRAINING`:

```bash
uv run torchrun --standalone --nproc_per_node=8 examples/advanced/10_mace_training.py
```

In a distributed configuration, `training.batch_size` represents the size per individual GPU. Your true global batch size scales directly to `training.batch_size * world_size`. Be sure to scale your learning rates, validation cadences, and checkpoint schedules accordingly.

## 8. Monitoring Validation During Training

We evaluate held-out batches according to `ValidationConfig`; the script sets this from `training.validation.every_epochs`. Each round reports component losses and a combined validation loss. Values are averaged over the validation batches on each rank; in distributed runs, those averages are reduced across processes.

Here is the validation loss over training for the charged MACE run.

![Validation metrics](../_static/userguide/validation_metrics-mace-260615.png)


## See Also

- [`examples/advanced/10_mace_training.py`](../../examples/advanced/10_mace_training.py)
  for the runnable entrypoint.
- [`examples/advanced/10_mace_training.yaml`](../../examples/advanced/10_mace_training.yaml)
  for the default Hydra config.
