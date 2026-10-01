.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _distributed-runtime:

=============================
Distributed runtime utilities
=============================

General-purpose helpers for running any distributed nvalchemi workflow — DDP
training, multi-GPU inference, or your own multi-process script. They are
independent of the spatial :doc:`domain-decomposition API </modules/distributed>`
and are re-exported from the package root
(``from nvalchemi.distributed import DistributedManager``).

Process-group manager
=====================

:class:`~nvalchemi.distributed.DistributedManager` is the recommended way to
initialise and query the process group. It is the single object to construct
once per process; the rest of the toolkit (for example
:class:`~nvalchemi.training.hooks.DDPHook`) reads rank, world size, and the
rank-local device from it. Use it whenever a workflow needs a coordinated group
of processes, not just for domain decomposition.

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DistributedManager
   PhysicsNeMoUninitializedDistributedManagerWarning

Supplied process groups
=======================

``ProcessGroupContext`` describes one initialized PyTorch process group without
creating it or taking ownership of its lifetime. Its ``rank`` and ``world_size``
refer to that group; ``global_rank(group_rank)`` translates a group-local rank
for APIs that require global ranks.

For an external group, select WORLD explicitly after initialization:

.. code-block:: python

   import torch.distributed as dist
   from nvalchemi.distributed import ProcessGroupContext

   dist.init_process_group(backend="gloo")
   context = ProcessGroupContext(dist.group.WORLD, execution_device="cpu")
   # The caller destroys the group after all work has finished.

For a Manager-created group, every participating process initializes the
Manager and creates the named group in the same order:

.. code-block:: python

   import torch.distributed as dist
   from nvalchemi.distributed import DistributedManager, ProcessGroupContext

   DistributedManager.initialize()
   manager = DistributedManager()
   context = None
   if dist.is_initialized():
       DistributedManager.create_process_subgroup(
           "packing", size=manager.world_size
       )
       context = ProcessGroupContext(
           manager.group("packing"), execution_device=manager.device
       )

``manager.group()`` returns ``None`` for the default group. Pass
``dist.group.WORLD`` or an actual named group to ``ProcessGroupContext``.
Manager initialization in a single-process run may leave PyTorch distributed
uninitialized; in that case, use local execution.

Gloo communication uses CPU. NCCL uses an explicit CUDA execution-device hint
first, then an initialized Manager's CUDA device, then the caller's current
CUDA device. An unindexed CUDA hint resolves against the current device.
A CPU hint uses the NCCL fallback. Invalid explicit CUDA devices raise locally.
``communication_scope()`` selects the resolved NCCL device temporarily and
restores the caller's current device, including when the body raises. It does
not change the scientific execution device or choose a stream.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ProcessGroupContext
   CollectivePhase

Errors and control metadata
===========================

``collective_error_sync`` combines ordinary Python error agreement with one
exchange of bounded control metadata. All group members must enter matching
calls in the same order:

.. code-block:: python

   from nvalchemi.distributed import collective_error_sync

   with collective_error_sync(context, phase="local_setup") as phase:
       phase.metadata = {"rank": context.rank, "ready": True}
   records = phase.records  # Ordered by group-local rank.

The phase label may be updated while local work runs. Metadata is serialized
once before communication, so successful records capture its submitted state.
``None`` is a valid record. Reports, layouts, counts, and CPU zero-capacity Batch
templates belong here; structure tensor payloads use typed transport.
``records`` is available only after successful context-manager exit.

An ordinary body or metadata-serialization exception is reported on every rank.
When several ranks fail, the lowest failing group-local rank supplies the phase,
exception type, and message; only that originating rank chains its original
exception. The default raised type is ``RuntimeError``; callers may select an
``Exception`` subclass that accepts a message. Configuration and communication
setup errors remain local. The helper does not recover communication failures,
catch ``BaseException``, retry, or cancel peers. Process loss, device failure,
and incompatible metadata-deserialization environments remain outside this
ordinary error agreement.

.. autosummary::
   :toctree: generated
   :nosignatures:

   collective_error_sync

Parameter resolvers
===================

Rather than reading environment variables or ``torch.distributed`` state by hand,
use these best-practice resolvers. Each returns a sensible value whether the run
is launched under :class:`~nvalchemi.distributed.DistributedManager`, plain
``torch.distributed``, ``torchrun`` environment variables, or single-process — so
the same code path works in every launch mode.

- :func:`~nvalchemi.distributed.resolve_world_size` — the number of processes.
- :func:`~nvalchemi.distributed.resolve_global_rank` — this process's global rank
  (accepts an explicit override).
- :func:`~nvalchemi.distributed.collective_device` — the device to place tensors
  on for collectives (CPU for the Gloo backend, the rank-local CUDA device for
  NCCL).

.. currentmodule:: nvalchemi.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   resolve_world_size
   resolve_global_rank
   collective_device

.. seealso::

   :doc:`/userguide/distributed_training` walks through using these to scale
   training across GPUs and nodes with :class:`~nvalchemi.training.hooks.DDPHook`.
