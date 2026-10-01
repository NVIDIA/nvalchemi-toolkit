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
and starting volume estimate needed by the rigid packer. The optional RDKit
helpers build it from molecules and generate conformers; the tensor input can
also be constructed directly.

.. currentmodule:: nvalchemi.csp

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   MolecularPackingInput

``MolecularPackingInput.sha256`` recomputes the established exact-input digest,
including tensor dtype, shape, contents, metadata, and formula-unit volume.
It identifies the input representation, not chemical equivalence. Device-resident
reads may copy tensors to CPU and synchronize; use it for identity or startup,
rather than in the packing loop.

.. autoproperty:: MolecularPackingInput.sha256

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
``OverlapReliefConfig`` checks the policy against ``z / z_prime``.

You can save a space-group policy or packing configuration and reload it for
another search. Reloading preserves all supplied weights, including groups
excluded by the current filters.

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

``CSPGenerator`` extends ``AtomisticGenerator`` with packing execution policy,
optional gathering, and native-result callbacks. It accepts any structural
``CrystalPacker``; algorithms retain their own configuration and
scientific loop.
``PackingResult`` contains rigid ASU data or a ``Batch``, plus per-rank reports.
Expansion defaults to true; raw results bypass Batch hooks and dynamics.

When a finite candidate cap has no explicit rank budgets, custom rank targets
receive proportional shares. Shares are rounded down, and leftover trials go
to the largest fractional shares, with lower group-local ranks winning ties.
Explicit budgets take precedence. The global sample target remains positive;
individual rank targets may be zero.

.. currentmodule:: nvalchemi.csp

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CSPGenerator

.. autodata:: CSP_OUTPUT_FIELDS

.. currentmodule:: nvalchemi.csp.packer

``OverlapReliefPacker`` samples and adjusts rigid molecular placements until
their intermolecular overlaps meet the chosen tolerance or
the trial budget ends.
It changes whole-molecule translations and rotations and can transform the
cell while leaving each chosen conformer's internal geometry fixed.
Its result contains accepted ASU structures and the stop reason; an optional
callback receives progress events during the search. Packing does not
minimize a physical crystal energy.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   CrystalPacker
   CandidateBudgetPacker
   PackingContext
   PackingReport
   PackingResult
   PackingStopReason
   OverlapReliefConfig
   OverlapReliefPacker
   OverlapReliefProgress

.. autosummary::
   :nosignatures:

   OverlapReliefPacker.pack
   OverlapReliefPacker.resolve_candidate_budget
   PackingContext.structure_ids


Store and expand selected structures
------------------------------------

``RigidMoleculeASUBatch`` holds ASU representations and expands selected
structures to full-cell Toolkit ``Batch`` objects. The Zarr writer and reader
persist those representations without first expanding every structure.
``check_integrity()`` explicitly checks ASU pointer spans, multiplicities,
bundled operation counts, and formula-molecule conformer-pool membership.
Construction, reading, packing, and expansion do not call it automatically.
Geometry validity remains a caller precondition. CPU checking before expansion
avoids additional GPU readbacks; device-resident checks may synchronize.

.. currentmodule:: nvalchemi.csp

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   RigidMoleculeASUBatch
   RigidMoleculeASUZarrWriter
   RigidMoleculeASUZarrReader

.. autosummary::
   :nosignatures:

   RigidMoleculeASUBatch.check_integrity
   RigidMoleculeASUBatch.to_batch
   RigidMoleculeASUZarrWriter.write
   RigidMoleculeASUZarrWriter.append
   RigidMoleculeASUZarrReader.read
   RigidMoleculeASUZarrReader.read_batch


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


Hooks for custom packers
------------------------

Custom packers, including those based on generative models, may propose repeated
crystal candidates. ``DeduplicateHook`` provides optional filtering of their
generated structures before downstream processing.

The bundled ``radial`` engine compares expanded atomic structures within each
generated Batch using approximate similarity. It keeps no reference set between
calls.

.. currentmodule:: nvalchemi.csp.hooks

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   DeduplicationEngine
   DeduplicateHook

.. autosummary::
   :nosignatures:

   DeduplicationEngine.deduplicate
   DeduplicateHook.radial

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
