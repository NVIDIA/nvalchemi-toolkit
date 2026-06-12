<!-- markdownlint-disable MD014 -->

(distributed_manager_guide)=

# Distributed Training

`DistributedManager` is the recommended entry point for distributed runtime
state in ALCHEMI training workflows. ALCHEMI re-exports PhysicsNeMo's manager as
`nvalchemi.distributed.DistributedManager` so training code can use one object
for process rank, local rank, world size, device selection, process groups, and
DistributedDataParallel defaults.

You can still manage `torch.distributed` directly in advanced workflows. Passing
a `DistributedManager` to {py:class}`~nvalchemi.training.TrainingStrategy` gives
ALCHEMI hooks a shared view of the distributed runtime without each hook needing
to read environment variables or initialize communication on its own.

## Basic pattern

Initialize the manager before constructing it, then pass the instance into the
strategy. {py:class}`~nvalchemi.training.hooks.DDPHook` uses the manager during
setup to choose the rank-local device, wrap optimized models in
`torch.nn.parallel.DistributedDataParallel`, and install a distributed sampler
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

`DistributedManager.initialize()` also supports single-process execution. In
that case `DDPHook` is a no-op because the world size is one, so the same script
can run locally and under a distributed launcher.

For a complete single-node dummy training script, see
{doc}`/examples/intermediate/06_ddp_mlp_training`. It can be launched with:

```bash
$ uv run --extra cu12 torchrun --standalone --nproc_per_node=2 \
    examples/intermediate/06_ddp_mlp_training.py --backend auto
```

## Data loaders and samplers

Each data-parallel rank must see a different slice of the training data. The
right composition depends on whether you use the default sampler, a custom
sampler, or a sampler that already emits complete batches.

### Simple case: let DDPHook install the sampler

For standard dataloaders with a `dataset` and mutable `sampler`, use an ordinary
loader and let {py:class}`~nvalchemi.training.hooks.DDPHook` install
`torch.utils.data.DistributedSampler` during strategy setup. The hook infers
`num_replicas`, `rank`, `shuffle`, and `drop_last` from the distributed manager
and dataloader, and uses `seed=0` unless overridden.

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

This is the preferred starting point for a single dataset. The loader stays
single-process friendly: when `manager.world_size == 1`, `DDPHook` leaves it
unchanged.

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

If a dataloader already has a distributed-aware sampler, `DDPHook` preserves it
instead of replacing it. A sampler is considered distributed-aware when it
satisfies {py:class}`~nvalchemi.data.datapipes.samplers.DistributedSamplerProtocol`:
it exposes `num_replicas`, `rank`, and `set_epoch(epoch)`. Native PyTorch
`DistributedSampler` satisfies this protocol.

For a sampler class or factory that accepts PyTorch-style distributed sampler
arguments, pass it to `DDPHook`. The hook supplies `num_replicas`, `rank`,
`shuffle`, `seed`, and `drop_last` defaults before applying your
`sampler_kwargs`.

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

When a dataloader is constructed with `batch_sampler`, the sampler is already
responsible for emitting complete batches. In that case, `DDPHook` cannot safely
replace the sampler with a plain `DistributedSampler`; the batch sampler itself
must be distributed-aware.

Use {py:class}`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler` when
you need per-dataset batch composition and distributed sharding together. Pass
the initialized manager to the sampler so each rank receives a different shard of
the batch sequence.

```python
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    DataLoader,
    Dataset,
    MultiDataset,
    MultiDatasetBatchSampler,
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

batch_sampler = MultiDatasetBatchSampler.balanced(
    dataset,
    batch_size=64,
    epoch_policy="max_size",
    replacement=True,
    distributed_manager=manager,
    seed=1234,
)

train_loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    prefetch_factor=16,
    pin_memory=True,
)

strategy = TrainingStrategy(
    ...,
    distributed_manager=manager,
    hooks=[DDPHook()],
)
strategy.run(train_loader)
```

`MultiDatasetBatchSampler` first builds the global batch order according to its
per-dataset allocation policy, then splits that batch order across data-parallel
ranks. With `drop_last=False`, it pads the batch order so each rank emits the
same number of batches, matching PyTorch `DistributedSampler` behavior. With
`drop_last=True`, it truncates the uneven tail instead.

Use {py:meth}`~nvalchemi.data.datapipes.dataloader.DataLoader.set_epoch` or let
{py:class}`~nvalchemi.training.TrainingStrategy` call it during training so
distributed samplers reshuffle deterministically from epoch to epoch.

## API details

For the complete manager API, including process-group methods and distributed
configuration knobs, see the
[PhysicsNeMo DistributedManager API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.distributed.html#physicsnemo.distributed.manager.DistributedManager).
