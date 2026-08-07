<!-- markdownlint-disable MD014 -->

(distributed_manager_guide)=

# Distributed Training

Scaling a training run across multiple GPUs or nodes in ALCHEMI comes down to one
object: `DistributedManager`. It gathers the distributed runtime state a job
needs — process rank, local rank, world size, device selection, process groups,
and DistributedDataParallel defaults — behind a single handle, re-exported from
PhysicsNeMo as `nvalchemi.distributed.DistributedManager`.

Routing that state through one object buys a shared context: passing a manager to
{py:class}`~nvalchemi.training.TrainingStrategy` gives every ALCHEMI hook the same
view of the distributed runtime, so no hook has to read environment variables or
initialize communication on its own. The same script then runs unchanged whether
launched on one process or many. Advanced workflows can still drive
`torch.distributed` directly, but the manager is the recommended entry point.

## Basic pattern

A distributed script differs from a single-process one in only two places: you
bring up the runtime, and you hand the strategy a manager alongside a
{py:class}`~nvalchemi.training.hooks.DDPHook`. Call `DistributedManager.initialize()`
once to start the distributed runtime, construct a manager from it, and pass that
instance to the strategy. From there `DDPHook` does the wiring during setup: it
selects the rank-local device, wraps the optimized models in
`torch.nn.parallel.DistributedDataParallel`, and installs a distributed sampler
for supported dataloaders.

```python
from nvalchemi.distributed import DistributedManager
from nvalchemi.training import TrainingStrategy
from nvalchemi.training.hooks import DDPHook

DistributedManager.initialize()
manager = DistributedManager()

strategy = TrainingStrategy(
    ...,
    distributed_manager=manager,
    hooks=[
        DDPHook(),
    ],
)

strategy.run(train_loader)
```

Launch the script with the process launcher for your environment. For a simple
single-node PyTorch launch:

```bash
$ torchrun --nproc_per_node=4 train.py
```

```{note}
`DistributedManager.initialize()` also supports single-process execution. When
the world size is one, `DDPHook` becomes a no-op, so the same script runs
unchanged locally or under a distributed launcher.
```

For a complete single-node dummy training script, see
{doc}`/examples/intermediate/06_ddp_mlp_training`. It can be launched with:

```bash
$ uv run --extra cu12 torchrun --standalone --nproc_per_node=2 \
    examples/intermediate/06_ddp_mlp_training.py --backend auto
```

## Data loaders and samplers

The one part of distributed training that needs care beyond the manager is data
loading: each data-parallel rank must see a different slice of the training data,
or the ranks would redundantly train on the same samples. How that sharding is
arranged depends on the sampler, and the subsections below walk the cases in
increasing order of control — the automatic default, a custom distributed
sampler, and multi-dataset batch sampling.

### Automatic configuration via `DDPHook`

For regular `nvalchemi` data pipes, `DDPHook` installs the distributed sampler
during strategy setup with no extra configuration; what it installs depends on the
dataset type. For a single {py:class}`~nvalchemi.data.datapipes.Dataset`, the hook
wraps a {py:class}`~torch.utils.data.DistributedSampler` in a
{py:class}`~torch.utils.data.BatchSampler` so the loader keeps emitting complete
batches. For {py:class}`~nvalchemi.data.datapipes.MultiDataset`, it installs
{py:class}`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler` so the
per-dataset batch composition and rank sharding are handled together. Either way,
the hook infers `num_replicas`, `rank`, `shuffle`, and `drop_last` from the
distributed manager and dataloader, and uses `seed=0` unless overridden.

```python
from nvalchemi.data.datapipes import DataLoader, Dataset
from nvalchemi.distributed import DistributedManager
from nvalchemi.training import TrainingStrategy
from nvalchemi.training.hooks import DDPHook

DistributedManager.initialize()
manager = DistributedManager()

dataset = Dataset(reader, device=manager.device)
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
)

strategy = TrainingStrategy(
    ...,
    distributed_manager=manager,
    hooks=[DDPHook()],
)
strategy.run(train_loader)
```

This is the preferred starting point for a single dataset, and like the basic
pattern it stays single-process friendly: when `manager.world_size == 1`, `DDPHook`
leaves the loader unchanged.

Use `sampler_kwargs` to override arguments passed to the default sampler:

```python
DDPHook(
    sampler_kwargs={
        "shuffle": False,
        "seed": 1234,
    },
)
```

### Custom distributed sampler

When the default sampler does not fit, you can supply your own, and `DDPHook` gets
out of the way accordingly. If a dataloader already has a distributed-aware
sampler, the hook preserves it instead of replacing it. A sampler counts as
distributed-aware when it satisfies
{py:class}`~nvalchemi.data.datapipes.samplers.DistributedSamplerProtocol`: it
exposes `num_replicas`, `rank`, and `set_epoch(epoch)`. Native PyTorch
`DistributedSampler` satisfies this protocol.

For a sampler class or factory that accepts PyTorch-style distributed sampler
arguments, pass it to `DDPHook`. The hook supplies `num_replicas`, `rank`,
`shuffle`, `seed`, and `drop_last` defaults before applying your `sampler_kwargs`.

```python
DDPHook(
    sampler_cls=MyDistributedSampler,
    sampler_kwargs={
        "seed": 1234,
    },
)
```

If your sampler uses different constructor names, pass those names explicitly in
`sampler_kwargs`.

```python
DDPHook(
    sampler_cls=MyDistributedSampler,
    sampler_kwargs={
        "replicas": manager.world_size,
        "worker_rank": manager.rank,
    },
)
```

### Multidataset batch sampling

Training on several datasets at once raises a second question on top of sharding:
how batches are composed across the child datasets. When `DDPHook` sees a nvalchemi
`DataLoader` backed by `MultiDataset` and no custom sampler class was supplied, it
installs {py:class}`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler`
automatically, keeping per-dataset batch composition and distributed sharding in
the same sampler.

Pass `MultiDatasetBatchSampler` options through `DDPHook.sampler_kwargs` when you
need a specific allocation policy, such as balanced batches or a fixed number of
samples per child dataset. The hook still supplies `num_replicas`, `rank`,
`shuffle`, and `drop_last` defaults before applying your overrides. As with the
single-dataset case, `DDPHook` preserves an existing distributed-aware batch
sampler instead of replacing it, so manual construction remains available for
fully custom samplers.

```python
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    DataLoader,
    Dataset,
    MultiDataset,
)
from nvalchemi.distributed import DistributedManager
from nvalchemi.training import TrainingStrategy
from nvalchemi.training.hooks import DDPHook

DistributedManager.initialize()
manager = DistributedManager()

dataset = MultiDataset(
    Dataset(AtomicDataZarrReader("dataset_a.zarr"), device=manager.device),
    Dataset(AtomicDataZarrReader("dataset_b.zarr"), device=manager.device),
)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    prefetch_factor=16,
    pin_memory=True,
)

strategy = TrainingStrategy(
    ...,
    distributed_manager=manager,
    hooks=[
        DDPHook(
            sampler_kwargs={
                "epoch_policy": "max_size",
                "replacement": True,
                "seed": 1234,
            },
        ),
    ],
)
strategy.run(train_loader)
```

Internally, `MultiDatasetBatchSampler` first builds the global batch order
according to its per-dataset allocation policy, then splits that order across
data-parallel ranks. With `drop_last=False`, it pads the batch order so each rank
emits the same number of batches, matching PyTorch `DistributedSampler` behavior;
with `drop_last=True`, it truncates the uneven tail instead.

Whichever sampler is in play, call
{py:meth}`~nvalchemi.data.datapipes.dataloader.DataLoader.set_epoch` yourself, or
let {py:class}`~nvalchemi.training.TrainingStrategy` call it during training, so
distributed samplers reshuffle deterministically from epoch to epoch.

## API details

This guide covers the training-facing surface of the manager. For the complete
API, including process-group methods and the distributed configuration knobs, see
the
[PhysicsNeMo DistributedManager API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.distributed.html#physicsnemo.distributed.manager.DistributedManager).
