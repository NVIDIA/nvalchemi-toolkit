Distributed Examples
====================

These examples cover the two multi-GPU paths in NVAlchemi:

- **Pipeline parallelism** (examples 01–02) — map ranks to dynamics
  stages with :class:`~nvalchemi.dynamics.DistributedPipeline`.
- **Domain decomposition** (examples 03–05) — shard one system across
  ranks with :class:`~nvalchemi.distributed.DomainParallel`, including
  the "bring your own model" arc.

All require multiple GPUs and must be launched with ``torchrun``.

.. warning::

   These examples are **not executed** during the Sphinx documentation
   build.  To run them, use ``torchrun`` as shown in each example.

Pipeline Architecture Overview
------------------------------

A :class:`~nvalchemi.dynamics.DistributedPipeline` maps GPU ranks to
dynamics stages.  Systems flow between stages via fixed-size NCCL
communication buffers:

.. graphviz::

   digraph distributed_pipeline {
       rankdir=LR;
       node [shape=box, style="rounded,filled", fillcolor="#e8f4fd",
             fontname="Helvetica", fontsize=11];
       edge [fontname="Helvetica", fontsize=10];

       rank0 [label="Rank 0: FIRE\n(upstream)"];
       rank1 [label="Rank 1: Langevin\n(downstream + sink)"];

       rank0 -> rank1 [label="NCCL"];
   }

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

**02 — Distributed Monitoring**
   Same topology as example 01, augmented with per-rank LoggingHook and
   ProfilerHook for observability, and ZarrData sinks for persistent
   trajectory storage.  Shows post-run log collation on rank 0.

Domain-Decomposition Examples
-----------------------------

These shard a single system across ranks with
:class:`~nvalchemi.distributed.DomainParallel` (halo exchange + force
consolidation handled by the framework).

.. code-block:: bash

   # 03 — MACE NVT Langevin MD, trajectory written to xyz from rank 0
   torchrun --nproc_per_node=2 examples/distributed/03_mace_nvt_distributed.py

   # 04 / 05 — bring-your-own model, validated against a single-process reference
   torchrun --nproc_per_node=2 examples/distributed/04_byo_pytorch_mpnn.py
   torchrun --nproc_per_node=2 examples/distributed/05_byo_graph_transformer.py

**03 — MACE NVT Distributed**
   End-to-end distributed MD with a stock
   :class:`~nvalchemi.models.mace.MACEWrapper`: a short
   :class:`~nvalchemi.dynamics.NVTLangevin` trajectory under
   ``DomainParallel``, with per-step neighbour-list rebuild and xyz
   snapshot logging from rank 0.  No distributed-aware code at the user
   layer.

**04 — BYO PyTorch MPNN**
   The full bring-your-own arc for a plain-PyTorch Behler-Parrinello
   potential: architecture → wrapper → run → ``trace_and_validate``
   against a single-process reference → ``MLIPSpec.save``/``load``.  An
   MPNN-halo model whose forward is scatter-aggregations + autograd
   needs no distributed code.

**05 — BYO Graph Transformer (Warp kernel)**
   The same arc when the model embeds a performance-critical Warp
   kernel that is opaque to ShardTensor dispatch.  Shows declaring the
   kernel's distribution semantics once via
   :class:`~nvalchemi.distributed.spec.OpAdapter`.

Benchmarks
----------

Performance + force-equivalence benchmarks for the
domain-decomposition path live in ``benchmark/distributed/`` (two
config-driven runners covering LJ, Ewald, PME, MACE, AIMNet2, and UMA).
See ``benchmark/distributed/README.md``.
