.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Generative module (AtomisticGenerator, hooks, pipelines)
=========================================================

The generative API drives inference for generative models of any family —
diffusion / flow matching, GANs, VAEs, normalizing flows — through the
abstract :class:`~nvalchemi.gen.generator.AtomisticGenerator` driver: a
condition → generate → materialize pipeline with lifecycle hooks, streaming,
and sequential composition. For orientation and recipes, see the
:doc:`generative models user guide </userguide/generative>`.

.. currentmodule:: nvalchemi.gen.generator

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomisticGenerator

.. currentmodule:: nvalchemi.gen

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   GeneratingFunction
   ConditionFunction
   GenerationStage
   GenerationContext
   GenerationPipeline

Model-side API
--------------

.. currentmodule:: nvalchemi.models.gen

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   GenerativeModelConfig
   GenerativeModelMixin

Demo models
-----------

.. currentmodule:: nvalchemi.models.gen.demo

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DemoGANModel
   DemoDiffusionModel
