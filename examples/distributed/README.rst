Distributed Pipeline Examples
==============================

These examples demonstrate multi-GPU distributed simulation pipelines
using :class:`~nvalchemi.dynamics.DistributedPipeline`.  They require
multiple GPUs and must be launched with ``torchrun``.

.. warning::

   These examples are **not executed** during the Sphinx documentation
   build.  To run them, use ``torchrun`` as shown in each example.

Architecture Overview
---------------------

A :class:`~nvalchemi.dynamics.DistributedPipeline` maps GPU ranks to
dynamics stages.  Systems flow between stages via fixed-size NCCL
communication buffers:

.. code-block:: text

    ┌──────────────────┐     ┌────────────────────┐
    │  Rank 0: FIRE    │────→│  Rank 1: Langevin  │
    │  (upstream)      │NCCL │  (downstream + sink)│
    └──────────────────┘     └────────────────────┘

Key concepts:

- **Upstream ranks** (``prior_rank=None``): hold a
  :class:`~nvalchemi.dynamics.SizeAwareSampler` and push graduated
  (converged) systems to the next rank.
- **Downstream ranks** (``next_rank=None``): receive systems from the
  prior rank and write results to a sink.
- **BufferConfig**: must be set to a fixed size on all ranks; NCCL
  requires identical message sizes every communication step.
- ``torchrun --nproc_per_node=N`` launches one process per GPU; each
  process runs only the stage assigned to its rank.

Running the Examples
--------------------

**01 — Parallel FIRE → Langevin** (4 GPUs required):

.. code-block:: bash

   torchrun --nproc_per_node=4 examples/distributed/01_distributed_pipeline.py

   # CPU/debug mode (set backend="gloo" in the script first):
   torchrun --nproc_per_node=4 --master_port=29500 examples/distributed/01_distributed_pipeline.py

**02 — Monitoring with LoggingHook, ProfilerHook, and ZarrData** (4 GPUs required):

.. code-block:: bash

   torchrun --nproc_per_node=4 examples/distributed/02_distributed_monitoring.py

After running example 02, per-rank CSV logs and Zarr trajectory stores are
written to the working directory.  Rank 0 also prints a collated summary.

Example Descriptions
--------------------

**01 — Distributed Pipeline**
   Two independent FIRE → NVTLangevin sub-pipelines running on 4 GPUs.
   Demonstrates DistributedPipeline wiring, BufferConfig, and HostMemory sinks.
   Based on the original ``examples/05_distributed_pipeline_example.py``.

**02 — Distributed Monitoring**
   Same topology as example 01, augmented with per-rank LoggingHook and
   ProfilerHook for observability, and ZarrData sinks for persistent
   trajectory storage.  Shows post-run log collation on rank 0.
