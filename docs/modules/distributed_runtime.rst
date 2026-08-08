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
