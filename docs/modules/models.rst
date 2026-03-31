.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Models module (composite calculators, potentials, and neighbor builders)
========================================================================

.. currentmodule:: nvalchemi.models

Core orchestration
------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CompositeCalculator
   EnergyDerivativesStep
   Potential
   CalculatorResults

Contracts and metadata
----------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PotentialCard
   MLIPPotentialCard
   NeighborListCard
   NeighborRequirement
   ModelCard
   CheckpointInfo
   PhysicalTerm

Neighbor builders
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   NeighborListBuilder
   NeighborListBuilderConfig
   AdaptiveNeighborListBuilder
   AdaptiveNeighborListConfig

Machine-learned potentials
--------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MACEPotential
   AIMNet2Potential
   DemoPotential

Physical / classical potentials
-------------------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DSFCoulombPotential
   DSFCoulombConfig
   LennardJonesPotential
   LennardJonesConfig
   DFTD3Potential
   DFTD3Config
   EwaldCoulombPotential
   EwaldCoulombConfig
   PMEPotential
   PMEConfig

Registry
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   KnownArtifactEntry
   ResolvedArtifact
   list_known_artifacts
   get_known_artifact
   register_known_artifact
   resolve_known_artifact
