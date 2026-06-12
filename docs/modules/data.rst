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

I/O and pipelines
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   AtomicDataZarrWriter
   AtomicDataZarrReader
   Dataset
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

Write configuration
-------------------

.. currentmodule:: nvalchemi.data.datapipes

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   ZarrArrayConfig
   ZarrWriteConfig
