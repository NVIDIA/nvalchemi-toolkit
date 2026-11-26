.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _mc-api:

Monte Carlo (nvalchemi.mc)
==========================

Batched, GPU-resident Monte Carlo samplers. Each step proposes one move per
active graph, evaluates the trial energies in one model call, and accepts or
restores every graph independently. Samplers are
:class:`~nvalchemi.dynamics.base.BaseDynamics` engines, so they take the same
hooks, run inside a :class:`~nvalchemi.dynamics.base.FusedStage`, and keep
``atomic_masses`` in step with the species. For usage and the UMA settings
that make them fast, see the :ref:`dynamics user guide <dynamics_simulations_guide>`.

.. currentmodule:: nvalchemi.mc

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BaseMonteCarlo
   MonteCarloStats
   SGC
   VCSGC
   Kawasaki
   NonlocalSwap
