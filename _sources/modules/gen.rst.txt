.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Generative module (AtomisticGenerator, hooks, pipelines)
=========================================================

The generative API drives inference for generative models of any family —
diffusion / flow matching, GANs, VAEs, normalizing flows — through the
:class:`~nvalchemi.gen.AtomisticGenerator` driver: an optional condition step
followed by generation, with lifecycle hooks, streaming, and sequential
composition. For orientation and recipes, see the
:doc:`generative models user guide </userguide/generative>`.

.. currentmodule:: nvalchemi.gen

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomisticGenerator
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
