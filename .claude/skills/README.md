# Agent Skills for `nvalchemi`

This folder contains a set of skills that common AI agent interfaces
will recognize and help speed up code development by providing
concise instructions on how to use the `nvalchemi` API for elementary
tasks.

- `nvalchemi-data-structures`: how to use individual atomic systems as well as batches.
- `nvalchemi-data-storage`: how to write, read, compose, and load atomic data.
- `nvalchemi-zarr-perf`: how to tune Zarr-backed Dataset/DataLoader throughput.
- `nvalchemi-model-wrapping`: how to wrap MLIPs to use arbitrary models within `nvalchemi`.
- `nvalchemi-training-api`: how to configure training strategies, losses,
optimizers, schedulers, validation, and checkpoints.
- `nvalchemi-fine-tuning`: how to fine-tune pretrained or user-provided models
with `nvalchemi`.
- `nvalchemi-reporting`: how to add Rich, TensorBoard, custom scalar,
and dynamics CSV observability to training and dynamics workflows.
- `nvalchemi-dynamics-implementation`: how to implement a simple dynamics class.
- `nvalchemi-dynamics-hooks`: how to implement and use `Hook`s in dynamics.
- `nvalchemi-dynamics-api`: how to find available dynamics classes, configure
them, and scale up and out.
