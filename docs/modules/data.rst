.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Data module (AtomicData, Batch, readers/writers)
================================================

.. currentmodule:: nvalchemi.data

Core classes
------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomicData
   Batch
   LevelSchema

Device helpers
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   resolve_device

Transforms
----------

.. currentmodule:: nvalchemi.data.transforms

.. autosummary::
   :toctree: generated
   :nosignatures:

   Compose
   make_supercell

.. data:: DEFAULT_EXTENSIVE_SYSTEM_KEYS
   :type: frozenset[str]

   System-level fields a k-fold supercell carries k times over.

.. data:: DEFAULT_INTENSIVE_SYSTEM_KEYS
   :type: frozenset[str]

   System-level fields a supercell carries unchanged.

.. currentmodule:: nvalchemi.data

I/O and pipelines
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomicDataZarrWriter
   AtomicDataZarrReader
   FieldSchema
   Dataset
   InMemoryDataset
   DataLoader
   Reader

Dataset composition and sampling
--------------------------------

.. currentmodule:: nvalchemi.data.datapipes

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MultiDataset
   MultiDatasetSampler
   MultiDatasetBatchSampler

.. autosummary::
   :toctree: generated
   :nosignatures:

   distributed_shard

Device helpers
--------------

.. currentmodule:: nvalchemi.data.datapipes

.. autosummary::
   :toctree: generated
   :nosignatures:

   dataset_device
   same_device

Write configuration
-------------------

.. currentmodule:: nvalchemi.data.datapipes

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ZarrArrayConfig
   ZarrWriteConfig
