.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _scheduling-api:

Simulation scheduling (nvalchemi.scheduling)
============================================

Batch-width planning for independent simulations and dependency-aware
campaigns of restartable runs. See the
:ref:`dynamics user guide <dynamics_simulations_guide>` for usage.

.. currentmodule:: nvalchemi.scheduling

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   SimulationBatchPlanner
   BatchMeasurement
   BatchMemoryEstimate
   RunAssignment
   RunSpec
   CampaignSpec
   CampaignScheduler
   FinalStateStore
