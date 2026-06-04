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

## API details

For the complete manager API, including process-group methods and distributed
configuration knobs, see the
[PhysicsNeMo DistributedManager API](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.distributed.html#physicsnemo.distributed.manager.DistributedManager).
