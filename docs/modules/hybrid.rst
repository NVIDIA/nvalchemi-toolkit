.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _hybrid-api:

Hybrid MC-MD (nvalchemi.hybrid)
===============================

:class:`~nvalchemi.hybrid.HybridMCMD` alternates blocks of a Monte Carlo
sampler and an MD integrator on the same batch of walkers, refreshing the
energy, forces and stress at each hand-off. See the
:ref:`dynamics user guide <dynamics_simulations_guide>` for usage and UMA settings.

.. currentmodule:: nvalchemi.hybrid

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   HybridMCMD
