---
name: nvalchemi-distributed-training
description: >-
  How to scale nvalchemi training across GPUs and nodes with
  DistributedManager, DDPHook, distributed samplers, and rank-safe
  validation, checkpointing, and reporting. Use when launching multi-GPU or
  multi-node training with torchrun or srun, wiring DistributedManager into
  TrainingStrategy, debugging rank, device, or process-group issues, or
  making logging and checkpoint writes rank-safe; for multi-rank dynamics
  pipelines see nvalchemi-dynamics-api.
---

# nvalchemi Distributed Training

## Overview

Use distributed training when one GPU is not enough for the dataset or the
epoch budget. Everything routes through one object: `DistributedManager`,
re-exported from PhysicsNeMo as `nvalchemi.distributed.DistributedManager`.
It holds rank, local rank, world size, device selection, and process-group
state behind a single handle; passing it to `TrainingStrategy` gives every
hook the same view of the runtime, so the same script runs unchanged on one
process or many.

This skill covers data-parallel TRAINING. Multi-rank dynamics pipelines
(`DistributedPipeline`) are covered by the `nvalchemi-dynamics-api` skill.
See `docs/userguide/distributed_training.md` for full details.

```python
from nvalchemi.distributed import DistributedManager
from nvalchemi.training import TrainingStrategy
from nvalchemi.training.hooks import DDPHook
```

---

## Minimal DDP pattern

A distributed script differs from a single-process one in two places: bring
up the runtime with `DistributedManager.initialize()`, then hand the
strategy a manager plus a `DDPHook`. The hook does the wiring during setup.

```python
import torch

from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.distributed import DistributedManager
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStrategy,
)
from nvalchemi.training.hooks import DDPHook

DistributedManager.initialize()
manager = DistributedManager()

dataset = Dataset(AtomicDataZarrReader("train.zarr"), device=manager.device)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4},
    ),
    loss_fn=EnergyMSELoss() + ForceMSELoss(),
    distributed_manager=manager,
    hooks=[DDPHook()],
    num_epochs=20,
)
strategy.run(train_loader)
```

`DistributedManager.initialize()` also supports single-process execution.
When the world size is one, `DDPHook` becomes a no-op, so the same script
runs unchanged locally or under a distributed launcher. Call
`DistributedManager.cleanup()` at the end of the script when the process
group should be torn down explicitly (see
`examples/intermediate/06_ddp_mlp_training.py`).

---

## Launching

Single node with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

The repo's runnable DDP example launches the same way:

```bash
uv run --extra cu12 torchrun --standalone --nproc_per_node=2 \
    examples/intermediate/06_ddp_mlp_training.py --backend auto
```

Multi-node with `torchrun`, one launcher per node:

```bash
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" train.py
```

Under Slurm, `srun` starts one task per rank; PhysicsNeMo's
`DistributedManager.initialize()` also reads Slurm environment variables,
so export the rendezvous address alongside the launch:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500
srun --nodes=2 --ntasks-per-node=8 --gpus-per-task=1 python train.py
```

Environment variables the launcher must provide (torchrun sets all of
these; the fallback helpers read them when `torch.distributed` is not yet
initialized):

| Variable      | Meaning                                          |
|---------------|--------------------------------------------------|
| `RANK`        | Global process rank across all nodes             |
| `WORLD_SIZE`  | Total number of processes                        |
| `LOCAL_RANK`  | Rank within the node; selects the local GPU      |
| `MASTER_ADDR` | Hostname or IP of the rendezvous (rank-0) node   |
| `MASTER_PORT` | TCP port for the rendezvous                      |

CPU note: the `gloo` backend works for CPU-only smoke tests. When
`DDPHook` (or `init_distributed`) brings up the process group itself, it
picks `nccl` when CUDA is available and `gloo` otherwise; pass
`DDPHook(backend="gloo")` to force CPU, which also keeps devices on CPU
instead of selecting `cuda:LOCAL_RANK`.

---

## What DDPHook wires up

`DDPHook` (`nvalchemi/training/hooks/ddp.py`) is a standard training hook
with `stage = TrainingStage.SETUP`. It acts in two phases and is a full
no-op whenever the resolved world size is one.

**Phase 1 — `prepare_strategy()`**, called before the strategy moves models
to devices:

- With `auto_init=True` (default), initializes `torch.distributed` via
  `init_distributed(manager, backend=...)` when nothing has done so yet.
- Resolves the rank-local device with `distributed_device(...)`; unless
  `backend == "gloo"`, that prefers `cuda:LOCAL_RANK`. It then calls
  `torch.cuda.set_device(device)` and rewrites `strategy.devices` so the
  strategy places models and batches on the right GPU.

**Phase 2 — hook dispatch at `TrainingStage.SETUP`:**

- Wraps target models in `torch.nn.parallel.DistributedDataParallel`.
  Targets are `model_keys` when given, otherwise every model that has an
  optimizer config; already-wrapped models are skipped. `device_ids` and
  `output_device` are set for CUDA models. Raises `RuntimeError` if
  `world_size > 1` but communication is not initialized.
- Injects a distributed sampler into `strategy.active_dataloader` when it
  exposes `dataset` and `sampler` attributes. For a nvalchemi `DataLoader`
  over a `Dataset` it installs `BatchSampler(DistributedSampler(...))`; for
  `MultiDataset` it installs `MultiDatasetBatchSampler` so per-dataset
  batch composition and rank sharding are handled together. It infers
  `num_replicas`, `rank`, `shuffle`, and `drop_last` from the manager and
  dataloader, with `seed=0`, before applying `sampler_kwargs` overrides.
  Dataloaders whose sampler already satisfies
  `DistributedSamplerProtocol` (exposes `num_replicas`, `rank`, and
  `set_epoch`) are preserved untouched.

Config options: `model_keys`, `find_unused_parameters`,
`broadcast_buffers` (both default to the manager's setting, else `False`),
`static_graph`, `process_group`, `backend`, `auto_init`, `sampler_cls`,
and `sampler_kwargs`. On exit the hook restores the original unwrapped
models and destroys the process group only if it created it.

```python
DDPHook(
    find_unused_parameters=True,
    sampler_kwargs={"shuffle": False, "seed": 1234},
)
```

---

## Rank utilities

`nvalchemi/training/distributed.py` provides structural helpers. Each one
consults an optional manager first, then live `torch.distributed` state,
then torchrun environment variables — so they are always safe to call.

| Helper                          | Returns / does                          | Fallback when not initialized       |
|---------------------------------|-----------------------------------------|-------------------------------------|
| `get_rank(manager)`             | Global rank                             | `RANK` env var, else `0`            |
| `get_world_size(manager)`       | World size                              | `WORLD_SIZE` env var, else `1`      |
| `get_local_rank(manager)`       | Node-local rank                         | `LOCAL_RANK` env var, else `0`      |
| `distributed_device(m, fb)`     | Rank-local device (`cuda:LOCAL_RANK`)   | `fallback` device (CPU-safe)        |
| `is_distributed_initialized(m)` | Whether communication is up             | `False`                             |
| `init_distributed(m, backend=)` | Bring up the process group              | No-op when `WORLD_SIZE <= 1`        |
| `destroy_distributed(m)`        | Tear down the process group             | No-op, returns `False`              |
| `barrier(m)`                    | Synchronize all ranks                   | No-op                               |
| `all_reduce(tensor, m, op=)`    | In-place all-reduce (default `SUM`)     | Returns tensor unchanged            |

```python
from nvalchemi.training.distributed import barrier, get_rank, get_world_size

if get_rank(manager) == 0:
    print(f"training on {get_world_size(manager)} ranks")
barrier(manager)
```

`nvalchemi/distributed.py` adds manager-level resolvers that check
`DistributedManager.is_initialized()` first, then `torch.distributed`,
then the environment:

- `resolve_world_size()` — world size; `WORLD_SIZE` env var, else `1`.
- `resolve_global_rank(global_rank=None)` — an explicit value wins;
  otherwise resolves like above from `RANK`, else `0`.
- `collective_device(fallback="cpu")` — device to stage tensors on for
  collectives; returns CPU when the backend is not `nccl`.

Every helper degrades to sensible single-process values, which is what
lets one script serve both local runs and `torchrun` launches.

---

## Validation and checkpointing across ranks

**Validation** (`nvalchemi/training/_validation.py`) runs on every rank.
`ValidationLoop` accumulates per-batch losses locally, then packs totals,
batch counts, and per-component sums into one float64 tensor and
all-reduces it (SUM) via `_distributed_sum_in_place` whenever
`is_distributed_initialized(...)` is true. Dividing summed totals by
summed counts yields globally averaged metrics, so every rank returns the
same summary dictionary; `summary["distributed_reduced"]` reports whether
the reduction ran. Because the reduction is a collective, every rank must
enter validation — never rank-gate the validation call itself.

**Checkpointing** (`nvalchemi/training/hooks/checkpoint.py` and
`nvalchemi/training/_checkpoint.py`): `CheckpointHook` defaults to
`rank_zero_only=True`, guarded as `ctx.global_rank != 0` in
`_should_save`, so only global rank 0 snapshots and writes. The write
itself is not a collective, so nonzero ranks never block on it. Before
saving, `_checkpoint_model` unwraps `DistributedDataParallel` via
`.module` so the checkpoint stores plain model weights; FSDP-wrapped
models raise `NotImplementedError` and should use
`torch.distributed.checkpoint` instead.

**Resume**: `TrainingStrategy.load_checkpoint(...)` restores weights,
optimizer/scheduler state, and runtime counters on every rank from the
shared checkpoint directory. Older checkpoints without a saved
`global_step_count` reconstruct it as `step_count * world_size` using the
current manager. Restart under the same world size so step counters and
sampler sharding stay consistent.

```python
strategy = TrainingStrategy(
    models=model,
    optimizer_configs=optimizer_config,
    loss_fn=loss_fn,
    validation_config=ValidationConfig(
        validation_data=val_loader, every_n_epochs=1
    ),
    distributed_manager=manager,
    hooks=[
        DDPHook(),
        CheckpointHook("runs/ddp/checkpoints", epoch_interval=1),
    ],
    num_epochs=20,
)
```

---

## Rank-safe reporting

Reporting is rank-aware out of the box; see the `nvalchemi-reporting`
skill's Distributed Reporting section for the full pattern.
`ReportingOrchestrator` exposes `global_rank` / `is_rank_zero` and skips a
reporter on nonzero ranks when either the orchestrator or the reporter
sets `rank_zero_only=True` (`RichReporter` defaults to `True`). Reporters
that declare `requires_all_ranks=True` — e.g. any reporter using
`rank_reduction="mean"` — are dispatched on every rank because the
reduction is a collective; only rank zero renders the result. Rank
reductions raise
`"Reporting rank reductions require DistributedManager to be initialized."`
outside an initialized run. For the dynamics `LoggingHook`, give each rank
a unique `log_path`; it writes directly and does not coordinate file
access.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `PhysicsNeMoUninitializedDistributedManagerWarning` (re-exported from `nvalchemi.distributed`) | `DistributedManager()` instantiated before the singleton was initialized | Call `DistributedManager.initialize()` once, before constructing the manager |
| `RuntimeError: DDPHook requires initialized distributed communication when world_size > 1.` | `WORLD_SIZE > 1` but no process group came up (e.g. `auto_init=False`, or the manager never initialized communication) | Launch with torchrun, initialize `torch.distributed` before `strategy.run()`, or provide an initialized `distributed_manager` |
| Script silently runs single-process; every rank helper returns 0/1 | `RANK`/`WORLD_SIZE` env vars missing — script started with plain `python` | Expected fallback; launch with `torchrun --nproc_per_node=N` for a real multi-process run |
| All ranks compute on `cuda:0`, or device-mismatch errors between model and batch | `DDPHook` skipped or added without a manager, so rank-local device selection never ran | Add `DDPHook()` with `distributed_manager=manager`; it sets `strategy.devices` to `cuda:LOCAL_RANK` in `prepare_strategy()` |
| Identical loss curves on every rank; no speedup from more GPUs | No distributed sampler — each rank trains the full dataset (custom iterable without `sampler`/`dataset` attributes, or hook missing) | Use the nvalchemi `DataLoader` so `DDPHook` can inject `DistributedSampler`/`MultiDatasetBatchSampler` |
| Hang at `barrier`/`all_reduce` (often at validation or a reducing reporter) | Rank divergence: some ranks skipped a collective, e.g. rank-gated control flow before validation reduction or a `rank_reduction` reporter | Ensure every rank reaches validation and all-rank reporters; only gate pure side effects (file writes, printing) by rank |
| NCCL errors or init failure on a CPU-only machine | `nccl` backend requested without CUDA | Use `DDPHook(backend="gloo")`; the auto-selection already falls back to `gloo` when CUDA is unavailable |

---

## Key files

- `docs/userguide/distributed_training.md` — primary narrative guide:
  basic pattern, sampler configuration, multidataset sharding.
- `nvalchemi/distributed.py` — `DistributedManager` re-export,
  `resolve_world_size`, `resolve_global_rank`, `collective_device`.
- `nvalchemi/training/distributed.py` — structural rank helpers with
  manager/torch/env fallback chain.
- `nvalchemi/training/hooks/ddp.py` — `DDPHook`: device selection, DDP
  wrapping, distributed sampler injection.
- `nvalchemi/training/_validation.py` — `ValidationLoop` and the
  all-reduced validation summary.
- `nvalchemi/training/_checkpoint.py` and
  `nvalchemi/training/hooks/checkpoint.py` — manifest checkpoints, DDP
  unwrapping, `rank_zero_only` write guard.
- `nvalchemi/hooks/reporting/` — rank-gated reporters and rank
  reductions (see `nvalchemi-reporting`).
- `examples/intermediate/06_ddp_mlp_training.py` — runnable single-node
  DDP training example with backend selection.
- `examples/distributed/` — dynamics-pipeline examples; reuse only their
  torchrun launch pattern for training work.
