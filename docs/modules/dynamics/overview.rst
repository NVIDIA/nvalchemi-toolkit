.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _dynamics-overview:

=========================
Architecture Overview
=========================

The ``nvalchemi.dynamics`` module provides a composable framework for
running batched molecular-dynamics (MD) and optimization workflows on
GPUs. It is built around four core ideas:

1. **One base class, two extension points** — subclass
   :class:`~nvalchemi.dynamics.BaseDynamics` and override only
   ``pre_update`` / ``post_update``.
2. **A pluggable hook system** — observe *or* modify every stage of
   the integration step via the :class:`~nvalchemi.dynamics.Hook`
   protocol.
3. **Composable convergence criteria** —
   :class:`~nvalchemi.dynamics.ConvergenceHook` lets you combine
   multiple criteria with AND semantics and automatically migrate
   samples between stages.
4. **Operator-based composition** — fuse stages on a single GPU
   with ``+`` or distribute across GPUs with ``|``.

.. code-block:: text

   ┌──────────────────────────────────────────────────────┐
   │                    BaseDynamics                       │
   │                                                       │
   │  ┌─────────┐  ┌──────────┐  ┌────────────┐           │
   │  │pre_update│→ │ compute  │→ │post_update │           │
   │  └─────────┘  └──────────┘  └────────────┘           │
   │       ↑              ↑              ↑                 │
   │   BEFORE/AFTER    BEFORE/AFTER   BEFORE/AFTER         │
   │    _PRE_UPDATE     _COMPUTE      _POST_UPDATE         │
   │                                                       │
   │  ← BEFORE_STEP                   AFTER_STEP →         │
   │                                  ON_CONVERGE →        │
   └──────────────────────────────────────────────────────┘

Inheritance hierarchy
---------------------

.. code-block:: text

   object
   └── _CommunicationMixin          # inter-rank buffers & isend/irecv
       └── BaseDynamics              # step loop, hooks, compute()
           └── FusedStage            # single-GPU multi-stage fusion

   DistributedPipeline              # multi-GPU orchestrator (standalone)

Every :class:`~nvalchemi.dynamics.BaseDynamics` subclass inherits
communication capabilities automatically — no extra mixins required.

Quick-start
-----------

.. code-block:: python

   from nvalchemi.dynamics import DemoDynamics
   from nvalchemi.dynamics.hooks import LoggingHook, NaNDetectorHook

   # 1. Wrap your model
   dynamics = DemoDynamics(
       model=my_model,
       n_steps=10_000,
       dt=0.5,
       hooks=[LoggingHook(frequency=100), NaNDetectorHook()],
   )

   # 2. Run
   dynamics.run(batch)

Single-GPU multi-stage (relax → MD):

.. code-block:: python

   fused = optimizer + md_dynamics          # uses the  +  operator
   fused.run(batch)                         # one forward pass per step

Multi-GPU pipeline (2 ranks):

.. code-block:: python

   pipeline = optimizer | md_dynamics       # uses the  |  operator
   with pipeline:
       pipeline.run()                       # each rank runs its stage

For a detailed walkthrough of how data flows through buffers,
communication channels, and sinks — including the back-pressure
mechanism and sample lifecycle — see :ref:`buffers-data-flow`.


Key concepts
------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Concept
     - Description
   * - ``BaseDynamics``
     - Coordinates a :class:`~nvalchemi.models.base.BaseModelMixin`
       with a numerical integrator. Manages the step loop, hook
       execution, and model forward pass.
   * - ``Hook``
     - A ``runtime_checkable`` :class:`~typing.Protocol` with
       ``frequency``, ``stage``, and ``__call__``.
   * - ``HookStageEnum``
     - Nine insertion points covering every phase of a dynamics step.
   * - ``ConvergenceHook``
     - Composable convergence detector with AND semantics and
       optional ``status`` migration for ``FusedStage``.
   * - ``FusedStage``
     - Composes *N* dynamics on one GPU with a shared forward pass.
       Each sub-stage processes only samples matching its status code.
   * - ``DistributedPipeline``
     - Maps one ``BaseDynamics`` per GPU rank and coordinates
       ``isend`` / ``irecv`` for sample graduation between stages.
   * - ``DataSink``
     - Pluggable storage for graduated / snapshotted samples:
       ``GPUBuffer``, ``HostMemory``, ``ZarrData``.
   * - ``SizeAwareSampler``
     - Bin-packing sampler for *inflight batching*: graduated
       samples are replaced on the fly without rebuilding the batch.
   * - ``BufferConfig``
     - Pre-allocation capacities for inter-rank communication
       buffers. See :ref:`buffers-data-flow`.
