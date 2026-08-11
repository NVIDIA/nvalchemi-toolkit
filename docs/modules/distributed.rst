.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _distributed-api:

===========================================
Distributed (spatial domain decomposition)
===========================================

The :mod:`nvalchemi.distributed` package runs the toolkit's dynamics and models
across multiple GPUs by partitioning atoms in space. This page is organised
around the tasks a developer actually performs — *running* an existing model
distributed, *bringing your own* model under domain decomposition, and *writing*
the wrapper code that makes a model distribution-aware — rather than as a flat
symbol list. For the concepts behind each piece, read the companion guides first:
:doc:`/userguide/distributed` (overview and the two storage strategies),
:doc:`/userguide/distributed_byo` (writing a wrapper + spec), and
:doc:`/userguide/distributed_shardtensor` (the sharded-tensor dispatch layer).

Everything below is importable from the package root, e.g.
``from nvalchemi.distributed import DomainParallel``; the few symbols that live in
a submodule (the ``SPEC_*`` presets, :class:`~nvalchemi.distributed.config.StrategyKind`)
are noted where they appear.

Running a model under domain decomposition
==========================================

The entry point is :class:`~nvalchemi.distributed.DomainParallel`: wrap any
single-GPU :class:`~nvalchemi.dynamics.base.BaseDynamics` integrator or optimizer
with a :class:`~nvalchemi.distributed.DomainConfig`, and it partitions the system,
exchanges halos, runs the model, consolidates outputs, and migrates atoms across
domain boundaries each step. The model wrapper, hooks, and integrator are
unchanged from the single-process API — the only additions at the user layer are
the config and the wrap.

:class:`~nvalchemi.distributed.DomainConfig` carries the three concerns a scope
needs: the interaction ``cutoff``/``skin`` and ``ghost_width`` (halo geometry),
the process-mesh topology (``mesh``/``mesh_dim``/``grid_dims``), and the
``strategy`` — :class:`~nvalchemi.distributed.config.StrategyKind.HALO` (spatial
domain decomposition, the default) or ``GRAPH_PARTITION`` (a node partition for
models that build their own neighbour list). Set ``compile=True`` to let the
framework own a shape-stable compiled forward. Hooks run at a
:class:`~nvalchemi.distributed.HookScope` (``LOCAL`` per-rank, ``GLOBAL`` after an
all-gather, or ``RANK_ZERO``).

Under the hood, :class:`~nvalchemi.distributed.DistributedModel` (or
:class:`~nvalchemi.distributed.DistributedPipelineModel` for a composed pipeline)
is the per-step adapter that owns halo exchange, neighbor rebuild, and output
consolidation; :class:`~nvalchemi.distributed.ShardedBatch` is the partitioned
per-atom state it operates on, produced by
:class:`~nvalchemi.distributed.SpatialPartitioner`. Most users never touch these
directly, but they are the seams a custom runtime hooks into.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DomainParallel
   DomainConfig
   HookScope
   DistributedModel
   DistributedPipelineModel
   ShardedBatch
   SpatialPartitioner

.. currentmodule:: nvalchemi.distributed.config

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   StrategyKind

Bringing your own model: the distribution spec
==============================================

A model tells the framework how to parallelize it by returning a
:class:`~nvalchemi.distributed.MLIPSpec` from
``BaseModelMixin.distribution_spec(strategy)``. The spec is the single source of
truth for how each output is combined, which storage policy applies, and which
opaque kernels need adapters. Declare each output once with an
:class:`~nvalchemi.distributed.OutputSpec` (see `Declaring model outputs`_); the
spec lowers that onto the wire fields and round-trips through
:meth:`~nvalchemi.distributed.MLIPSpec.to_dict` /
:meth:`~nvalchemi.distributed.MLIPSpec.from_dict`. Specs compose with ``|`` so a
model can start from a preset and override a field.

Most models never write a spec by hand — the ``SPEC_*`` presets in
:mod:`nvalchemi.distributed.spec` cover the shipped families
(``SPEC_MPNN_HALO`` for scatter-heavy message-passing nets, ``SPEC_LJ_HALO``,
``SPEC_UMA_HALO``, ``SPEC_EWALD_HALO``, ``SPEC_PME_HALO``, ``SPEC_DFTD3_HALO``,
and ``SPEC_MPNN_GP`` for the graph-partition strategy). Start from the preset that
matches your model's communication pattern and adjust.
:class:`~nvalchemi.distributed.DistributionSpec` is the framework-generic layer
underneath (storage policy + custom-op/third-party-helper tuples);
:class:`~nvalchemi.distributed.CompilePolicy` and
:class:`~nvalchemi.distributed.ForceStrategy` tune the compiled-forward and
force-derivation behaviour.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MLIPSpec
   DistributionSpec
   CompilePolicy
   ForceStrategy

.. currentmodule:: nvalchemi.distributed.spec

.. autosummary::
   :toctree: generated
   :nosignatures:

   SPEC_MPNN_HALO
   SPEC_LJ_HALO
   SPEC_UMA_HALO
   SPEC_EWALD_HALO
   SPEC_PME_HALO
   SPEC_DFTD3_HALO
   SPEC_MPNN_GP

Adapters for opaque kernels
===========================

Domain decomposition works by dispatching per-atom tensor operations through a
:class:`ShardTensor` (see :doc:`/userguide/distributed_shardtensor`). Kernels that
bypass ``__torch_function__`` — Warp/Triton launches, ``@torch.jit.script`` ops,
or a model's internal graph builder — are invisible to that dispatch and must be
declared as *adapters* on the spec's ``custom_ops``. Pick the adapter that matches
how the kernel is invoked: :class:`~nvalchemi.distributed.OpAdapter` (a registered
custom op), :class:`~nvalchemi.distributed.MethodAdapter` (a method on a named
class), :class:`~nvalchemi.distributed.FunctionAdapter` (a module-level function),
:class:`~nvalchemi.distributed.PythonAdapter` (an arbitrary attribute patch), or
:class:`~nvalchemi.distributed.JitAdapter` (a scripted op needing marshalling
across the ShardTensor boundary). :class:`~nvalchemi.distributed.AdapterRegistry`
collects them and :class:`~nvalchemi.distributed.AdapterStatus` reports whether
each applied.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   OpAdapter
   MethodAdapter
   FunctionAdapter
   PythonAdapter
   JitAdapter
   AdapterRegistry
   AdapterStatus

Declaring model outputs
=======================

Consolidation needs to know, for every model output, its *shape* (per-atom vs
per-system) and how each rank's partial value is *combined* into the global
result. Declare this with an :class:`~nvalchemi.distributed.OutputSpec` per output
on the model's :class:`~nvalchemi.distributed.MLIPSpec`::

   outputs={"stress": OutputSpec(kind=OutputKind.PER_GRAPH, reduce=Reduce.ALL_REDUCE)}

:class:`~nvalchemi.distributed.OutputKind` covers the shape axis
(``PER_NODE`` / ``PER_GRAPH``, plus ``GLOBAL`` passthrough and ``UNKNOWN``
fallback) and :class:`~nvalchemi.distributed.Reduce` the combine rule
(``NONE`` per-kind default, ``ALL_REDUCE`` to sum partials across the mesh, or
``OWNED_ONLY`` for values already correct on every rank). Getting these right is
what makes a distributed forward numerically match the single-process result.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   OutputSpec
   OutputKind
   Reduce

Validating a spec
=================

Before trusting a new wrapper, run
:func:`~nvalchemi.distributed.trace_and_validate`: it traces the model under
domain decomposition, checks that every per-atom operation and opaque kernel is
covered, and returns a report with a pass/fail verdict, the applied fixes, and
per-layer diagnostics pinpointing any uncovered op. This is the first thing to
reach for when a distributed forward disagrees with the single-process baseline.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   trace_and_validate

Writing adapter bodies: the intent vocabulary
=============================================

Inside a distributed method or adapter body, express what you need *by intent*
rather than naming the mechanism, and the helper does the right thing under the
halo policy, in single-process, and under ``torch.compile``. Mark a wrapper method
as distribution-aware with :func:`~nvalchemi.distributed.distributed_method`; then
call :func:`~nvalchemi.distributed.refresh_neighbors` to update halo rows after a
neighbor rebuild, :func:`~nvalchemi.distributed.scatter_to_owners` to reverse-sum
halo contributions back to owning ranks, :func:`~nvalchemi.distributed.system_sum`
to reduce a per-system quantity across the mesh, and
:func:`~nvalchemi.distributed.to_local` / :func:`~nvalchemi.distributed.localize`
to drop to owned-only rows. :func:`~nvalchemi.distributed.current_dd_context`
exposes the live context, :func:`~nvalchemi.distributed.autograd_target` returns
the in-graph leaf to differentiate against, and
:class:`~nvalchemi.distributed.Scope` selects owned vs padded extent.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   distributed_method
   refresh_neighbors
   neighbor_refresh_adapters
   scatter_to_owners
   system_sum
   to_local
   localize
   current_dd_context
   autograd_target
   Scope

Graph-parallel padding
=======================

Under ``StrategyKind.GRAPH_PARTITION`` (and any compiled forward), per-rank atom
and edge counts drift as the system evolves, which would force recompilation
every step. The :class:`~nvalchemi.distributed.GraphPadder` family pads counts to
stable caps so the compiled graph is reused: :class:`~nvalchemi.distributed.COOPadder`
for sparse edge indices, :class:`~nvalchemi.distributed.DensePadder` /
:class:`~nvalchemi.distributed.DenseBatchPadder` for dense neighbor matrices, with
:func:`~nvalchemi.distributed.resolve_cap` choosing the padded size.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   GraphPadder
   COOPadder
   DensePadder
   DenseBatchPadder

.. autosummary::
   :toctree: generated
   :nosignatures:

   resolve_cap

Halo and resharding primitives
==============================

Lower-level building blocks for custom distributed flows:
:class:`~nvalchemi.distributed.ParticleHaloConfig` configures the halo exchange,
and :func:`~nvalchemi.distributed.reshard_by_destination` moves atoms to new
owning ranks (the atom-migration primitive ``DomainParallel`` uses each step).

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ParticleHaloConfig

.. autosummary::
   :toctree: generated
   :nosignatures:

   reshard_by_destination

.. seealso::

   The general-purpose process-group manager and rank/world/device resolvers
   (:class:`~nvalchemi.distributed.DistributedManager` and friends) are not
   specific to domain decomposition — they live in
   :doc:`/modules/distributed_runtime`.
