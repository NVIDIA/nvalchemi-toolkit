.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _csp-api:

Crystal structure prediction API
================================

The CSP API is organized by the tasks in a molecular crystal search: prepare
formula-unit inputs, choose cell volumes and symmetry, generate starting
packings, save and expand their asymmetric-unit (ASU) representations, and
screen atomistic structures for possible duplicates. An ASU representation
stores independent molecule placements and symmetry instead of coordinates
for every atom in the cell. Atomistic optimization does not automatically
preserve the generated space group.

.. seealso::

   :ref:`CSP user guide <csp-guide>` for the application workflow, examples,
   and scientific limits.


Prepare a formula unit
----------------------

``MolecularPackingInput`` holds the molecules, conformers, contact distances,
and starting volume estimate needed by the Packer. The optional RDKit helpers
build it from molecules and generate conformers; the tensor input can also be
constructed directly.

.. currentmodule:: nvalchemi.csp

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MolecularPackingInput

.. currentmodule:: nvalchemi.csp.chem

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   RDKitConformerConfig

.. autosummary::
   :toctree: generated
   :nosignatures:

   generate_conformers_from_smiles
   build_contact_distance_matrix
   build_molecular_packing_input


Choose volume and symmetry
--------------------------

Estimate a starting cell-volume range and select symmetry through
``SpaceGroupPolicy``. Its ``fixed`` and ``sampled`` constructors describe group
selection; ``draw`` samples compatible groups independently of packing.
``PackingConfig`` checks the policy against ``z / z_prime``.

.. currentmodule:: nvalchemi.csp

.. autosummary::
   :toctree: generated
   :nosignatures:

   estimate_formula_unit_volume
   get_default_atomic_volumes

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CrystalSystem
   SpaceGroupPolicy

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_space_group_candidates
   sample_space_groups
   csd_space_group_probabilities
   get_space_group_operations
   get_space_group_operation_count
   get_crystal_system
   is_sohncke_space_group

.. automethod:: SpaceGroupPolicy.fixed

.. automethod:: SpaceGroupPolicy.sampled

.. automethod:: SpaceGroupPolicy.draw


Generate candidates
-------------------

``CrystalPacker`` samples and adjusts rigid molecular placements until their
intermolecular overlaps meet the chosen tolerance or the trial budget ends.
It changes whole-molecule translations and rotations and can transform the
cell while leaving each chosen conformer's internal geometry fixed.
Its result contains accepted ASU structures and the stop reason; an optional
callback receives progress events during the search. Packing does not
minimize a physical crystal energy.

.. currentmodule:: nvalchemi.csp.packer

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   PackingConfig
   CrystalPacker
   PackingResult
   PackingProgress
   PackingStopReason

.. autosummary::
   :nosignatures:

   CrystalPacker.pack


Store and expand selected structures
------------------------------------

``RigidMoleculeASUBatch`` holds ASU representations and expands selected
structures to full-cell Toolkit ``Batch`` objects. The Zarr writer and reader
persist those representations without first expanding every structure.

.. currentmodule:: nvalchemi.csp

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   RigidMoleculeASUBatch
   CSPZarrWriter
   CSPZarrReader

.. autosummary::
   :nosignatures:

   RigidMoleculeASUBatch.to_batch
   CSPZarrWriter.write
   CSPZarrWriter.append
   CSPZarrReader.read
   CSPZarrReader.read_batch


Screen similarity and duplicates
--------------------------------

``RadialComparisonIndex`` screens fully periodic, partially periodic, and
nonperiodic atomistic batches using local atom-neighbor distances. Its scores
and matches are approximate; a match can be a false positive. Topological
atom types can distinguish different connectivity roles of the same element
before a caller's more specific confirmation step.

.. currentmodule:: nvalchemi.csp.comparison

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   RadialComparisonIndex
   DeduplicationResult
   deduplicate_batch
   deduplicate_stream
   iter_matches_stream

.. autosummary::
   :nosignatures:

   RadialComparisonIndex.build
   RadialComparisonIndex.score_pairs
   RadialComparisonIndex.find_matches
   RadialComparisonIndex.iter_matches
   RadialComparisonIndex.deduplicate

.. currentmodule:: nvalchemi.csp.chem

.. autosummary::
   :toctree: generated
   :nosignatures:

   TopologicalAtomTypeMap
   topological_atom_types_from_connectivity
   topological_atom_types_from_mol

.. autosummary::
   :nosignatures:

   TopologicalAtomTypeMap.for_batch
   TopologicalAtomTypeMap.for_connectivity
