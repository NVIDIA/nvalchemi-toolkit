.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Generative module (AtomGenerator, hooks, pipelines, specs)
==========================================================

The generative API drives inference for generative models of any family —
diffusion / flow matching, GANs, VAEs, normalizing flows — through the
abstract :class:`~nvalchemi.gen.generator.AtomGenerator` interface: a fixed
condition → generate pipeline with lifecycle hooks, streaming,
and sequential composition. For orientation and recipes, see the
:doc:`generative models user guide </userguide/generative>`.

.. currentmodule:: nvalchemi.gen.generator

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomGenerator

.. currentmodule:: nvalchemi.gen

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   GeneratingFunction
   GenerationStage
   GenerationContext
   GenerationPipeline

Helpers
-------

.. currentmodule:: nvalchemi.gen

.. autosummary::
   :toctree: generated
   :nosignatures:

   default_condition


Model-side API
--------------

.. currentmodule:: nvalchemi.models.gen

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   GenerativeModelConfig
   GenerativeModelMixin
