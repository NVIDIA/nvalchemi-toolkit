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

.. graphviz::
   :caption: A single ``BaseDynamics.step()`` and its hook stages.

    digraph step_architecture {
        rankdir=LR
        compound=true
        fontname="Helvetica"
        node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled" fillcolor="#dce6f1"]
        edge [fontname="Helvetica" fontsize=10]

        subgraph cluster_dynamics {
            label="BaseDynamics.step()"
            style=rounded
            color="#4a90d9"
            fontcolor="#4a90d9"
            fontname="Helvetica"
            fontsize=12

            pre  [label="pre_update"]
            comp [label="compute"]
            post [label="post_update"]

            pre -> comp -> post [style=bold]
        }

        before_step [label="BEFORE_STEP" fillcolor="#f9e2ae"]
        after_step  [label="AFTER_STEP" fillcolor="#f9e2ae"]
        on_converge [label="ON_CONVERGE" fillcolor="#f9e2ae"]

        hook_pre  [label="BEFORE / AFTER\n_PRE_UPDATE"  fillcolor="#f9e2ae"]
        hook_comp [label="BEFORE / AFTER\n_COMPUTE"     fillcolor="#f9e2ae"]
        hook_post [label="BEFORE / AFTER\n_POST_UPDATE" fillcolor="#f9e2ae"]

        before_step -> pre [style=dashed color="#999999"]
        post -> after_step [style=dashed color="#999999"]
        after_step -> on_converge [style=dashed color="#999999"]
        hook_pre  -> pre  [style=dotted color="#999999" arrowhead=none]
        hook_comp -> comp [style=dotted color="#999999" arrowhead=none]
        hook_post -> post [style=dotted color="#999999" arrowhead=none]

        {rank=same; hook_pre; hook_comp; hook_post}
    }

Inheritance hierarchy
---------------------

.. graphviz::
   :caption: Inheritance hierarchy of the dynamics module.

   digraph inheritance {
       rankdir=BT
       fontname="Helvetica"
       node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled" fillcolor="#dce6f1"]
       edge [fontname="Helvetica" fontsize=10]

       object        [label="object" fillcolor="#eeeeee"]
       comm          [label="_CommunicationMixin\n(inter-rank buffers)"]
       base          [label="BaseDynamics\n(step loop, hooks, compute)"]
       fused         [label="FusedStage\n(single-GPU multi-stage)"]
       distributed   [label="DistributedPipeline\n(multi-GPU orchestrator)" fillcolor="#f9e2ae"]

       comm  -> object
       base  -> comm
       fused -> base

       note [label="standalone\n(not a BaseDynamics subclass)" shape=plaintext fillcolor=none style=""]
       note -> distributed [style=dotted arrowhead=none color="#999999"]
   }

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
       hooks=[LoggingHook(backend="csv", log_path="log.csv", frequency=100), NaNDetectorHook()],
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
   * - :class:`~nvalchemi.dynamics.BaseDynamics`
     - Coordinates a :class:`~nvalchemi.models.base.BaseModelMixin`
       with a numerical integrator. Manages the step loop, hook
       execution, and model forward pass.
   * - :class:`~nvalchemi.hooks.Hook`
     - A ``runtime_checkable`` :class:`~typing.Protocol` with
       ``frequency``, ``stage``, and ``__call__``.
   * - :class:`~nvalchemi.dynamics.DynamicsStage`
     - Nine insertion points covering every phase of a dynamics step.
   * - :class:`~nvalchemi.dynamics.ConvergenceHook`
     - Composable convergence detector with AND semantics and
       optional ``status`` migration for :class:`~nvalchemi.dynamics.FusedStage`.
   * - :class:`~nvalchemi.dynamics.FusedStage`
     - Composes *N* dynamics on one GPU with a shared forward pass.
       Each sub-stage processes only samples matching its status code.
   * - :class:`~nvalchemi.dynamics.DistributedPipeline`
     - Maps one :class:`~nvalchemi.dynamics.BaseDynamics` per GPU rank and coordinates
       ``isend`` / ``irecv`` for sample graduation between stages.
   * - :class:`~nvalchemi.dynamics.DataSink`
     - Pluggable storage for graduated / snapshotted samples:
       :class:`~nvalchemi.dynamics.GPUBuffer`, :class:`~nvalchemi.dynamics.HostMemory`,
       :class:`~nvalchemi.dynamics.ZarrData`.
   * - :class:`~nvalchemi.dynamics.SizeAwareSampler`
     - Bin-packing sampler for *inflight batching*: graduated
       samples are replaced on the fly without rebuilding the batch.
   * - :class:`~nvalchemi.dynamics.base.BufferConfig`
     - **Required** pre-allocation capacities for inter-rank
       communication buffers. See :ref:`buffers-data-flow`.
