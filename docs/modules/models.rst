.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Models module (BaseModelMixin, ModelConfig, wrappers)
=====================================================

.. currentmodule:: nvalchemi.models.base

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ModelConfig
   ModelCard
   BaseModelMixin

Demo utilities
--------------

.. currentmodule:: nvalchemi.models.demo

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DemoModel
   DemoModelWrapper

Machine-learned potentials
--------------------------

.. currentmodule:: nvalchemi.models.mace

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MACEWrapper

Physical / classical models
---------------------------

.. currentmodule:: nvalchemi.models.lj

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   LennardJonesModelWrapper

.. currentmodule:: nvalchemi.models.dftd3

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DFTD3ModelWrapper

.. currentmodule:: nvalchemi.models.pme

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PMEModelWrapper

.. currentmodule:: nvalchemi.models.ewald

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   EwaldModelWrapper

Composition
-----------

.. currentmodule:: nvalchemi.models.composable

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ComposableModelWrapper
