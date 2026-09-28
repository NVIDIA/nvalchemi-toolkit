.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-distillation-api:

Distillation API
================

Teacher scoring, offline dataset labeling, and the offline distillation
strategy and loss terms for knowledge-distillation workflows.

.. seealso::

   - **Training strategy API**: :ref:`training-strategy-api`
   - **Fine-tuning API**: :ref:`training-finetuning-api`
   - **Loss API**: :ref:`losses-api`


Scoring
-------

A scorer turns a :class:`~nvalchemi.data.Batch` into named teacher signals,
each a :class:`~nvalchemi.training.distillation.TeacherSignal` mapping one
teacher output to a batch field, a level, and a canonical shape. The built-in
ones — ``energy``, ``forces``, ``stress``, ``atomic_energies``, and
``embeddings`` — are requested by name; any other teacher output is requested
as a spec of its own, passed beside the built-in names:

.. code-block:: python

   from nvalchemi.training.distillation import InProcessTeacherScorer, TeacherSignal

   charges = TeacherSignal("charges", "charges", "teacher_charges", "node")
   scorer = InProcessTeacherScorer(teacher, ["energy", "forces", charges])
   scorer.label_fields  # ('teacher_charges', 'teacher_energy', 'teacher_forces')

The spec reads the teacher's ``charges`` output into the node-level
``teacher_charges`` field; a ``normalize`` callable reshapes a raw output whose
layout differs from the field's, and the scorer refuses the spec at
construction when the teacher does not declare that output.
:class:`~nvalchemi.training.distillation.InProcessTeacherScorer` evaluates a
teacher loaded in the current process and leaves the scored batch exactly as it
found it, including neighbor tensors.

.. currentmodule:: nvalchemi.training.distillation

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherScorer
   InProcessTeacherScorer
   TeacherSignal
   signal_fields
   scorer_fields
   signal_for_field
   SignalLevel
   TeacherLabels
   NeighborListPolicy
   BUILTIN_SIGNALS
   SUPPORTED_SIGNALS

Scorers speak two public type aliases: ``SignalLevel``, the ``"node"`` or
``"system"`` level a signal is attached at, and ``TeacherLabels``, the
``{batch field: (detached tensor, level)}`` mapping
:meth:`~nvalchemi.training.distillation.TeacherScorer.label` returns. The
built-in specs are published as
:data:`~nvalchemi.training.distillation.BUILTIN_SIGNALS`, keyed by name, and
their names as :data:`~nvalchemi.training.distillation.SUPPORTED_SIGNALS`. A
:class:`~nvalchemi.training.distillation.TeacherSignal` names the teacher
output it reads, the ``teacher_*`` field it writes, the level, and an optional
``normalize`` callable shaping the raw output; the namespace and level rules are
enforced when the spec is built, and the in-process scorer refuses a spec
naming an output the teacher does not declare. The scorer publishes its
resolved specs as ``signal_specs`` and the fields they write as
``label_fields``. A custom scorer may publish ``label_fields``, the batch
fields its ``label()`` populates, which consumers resolve through
:func:`~nvalchemi.training.distillation.scorer_fields` rather than reading the
attribute.

Where the teacher's neighbor list comes from is an explicit setting,
``neighbor_list``. The default ``"rebuild"`` builds the teacher's own list for
every call and rolls it back afterwards, whatever list the batch carries; a
composed pipeline keeps its default source's list as an instance attribute and
captures its whole per-source table alongside it, and both are hidden from the
teacher for the duration of scoring, so a teacher scoring a live student batch
never reads the student's neighborhoods. ``"reuse"`` is for the case where the
student has already built the list the teacher needs, in the teacher's format
and at its cutoff: the scorer consumes the batch's list and builds nothing,
checking only what it cannot infer — that the keys the teacher's format reads
are present and that a cutoff stamp, if the batch carries one, equals the
teacher's — and raising a :class:`ValueError` naming the missing key or the
mismatched cutoff otherwise, never falling back to a rebuild. Whether a list
holds each pair once or twice is recorded nowhere on the batch, so a reused
list must match the teacher's ``half_list`` by construction. A teacher
composition that plans more than one neighbor-list source is refused at
construction, because the scorer builds exactly one list per batch; compose it
with ``neighbor_adaptation="always"`` or a ``max_cutoff_ratio`` of at least its
largest-to-smallest cutoff ratio so it adapts that one list per step.

A composed teacher also wires one stage into the next through the batch: an
intermediate such as ``charges`` is written straight onto it, and an autograd
group swaps each of its gradient inputs for a fresh leaf. The scorer records
the batch's fields before the forward pass and afterwards drops the ones that
appeared and puts back the ones that were replaced, so a teacher never leaves
its charges, or a positions tensor cut loose from the student's graph, behind
for a later student forward to read.


Labeling
--------

Offline labeling walks a dataset once, scores it, and writes the source fields
plus the teacher fields to a Zarr store that the ordinary reader and dataset
path consume. Runs are resumable: the first ``len(store)`` samples are skipped,
a store that already covers the dataset is a no-op, and a store holding more
samples than the dataset — one written from a different dataset — is refused.
Every chunk must write the fields, levels, dtypes, and row shapes the store
holds, since the writer would otherwise misalign, cast, or truncate labels
without an error, and a store whose arrays disagree about how many samples it
contains — what an interrupted run leaves behind — is reported rather than
resumed from a misaligned offset. Both checks read the store through the
reader's own description of it:
:meth:`~nvalchemi.data.AtomicDataZarrReader.check_integrity` refuses the torn
store, and :meth:`~nvalchemi.data.AtomicDataZarrReader.schema` supplies the
per-field :class:`~nvalchemi.data.FieldSchema` each chunk is compared to. Each
label is held to the chunk's atom or graph count before it is attached, because
the split into per-graph rows would otherwise drop whatever a scorer returned
beyond it.

The neighbor tensors are dropped by default. The dense ones cannot append into
a fixed-width store array, and a sparse list is dropped because the cutoff it
was built at is a batch attribute the store does not hold, so a reloaded list
is one nothing downstream can check; ``keep_neighbors=True`` stores the sparse
list anyway. Build the student's list from the stored positions with a
:class:`~nvalchemi.hooks.NeighborListHook` at ``BEFORE_FORWARD``. Labels may
be stored in any dtype an ALCHEMI store holds (``dtype`` on the scorer picks
it), but they read back at the reading dataset's ``positions`` dtype, because a
dataset coerces every floating-point field it loads; the stored dtype governs
the store's size, not what training sees.

Labels are written with ``overwrite=True``, so a scorer that reached outside the
``teacher_*`` namespace would replace the reference field of that name and
persist the replacement. A scorer's declared ``label_fields`` is refused before
the first chunk is written, and the fields each chunk actually returns are
refused again per chunk, which is what polices a scorer that declares nothing.

The chunk loop can read one chunk ahead of the scoring and writing of the
previous one, through the dataset's fused-prefetch surface
(``prefetch_fused_batches`` / ``get_fused_batches``). Reading ahead saves up
to one load per chunk when the store is slow to read, such as a network or
object store, or when per-sample validation dominates the load; on a fast
local store the prefetch thread competes with the main thread while the
teacher's kernels are launched, and labeling can run a little slower than the
sequential loop. ``prefetch="auto"`` (the default) therefore measures rather
than assumes: it reads the first two chunks sequentially, reads ahead only
when the second chunk's load took at least half of its scoring and writing,
and falls back to sequential reads if the first chunk read entirely ahead was
not faster per atom than the sequential one. ``True`` always reads ahead (a
dataset without the surface falls back with a warning) and ``False`` keeps
the sequential loop. The per-chunk writes, the resume bookkeeping, and the
store's contents are the same in every mode. A dataset that emits
host-resident chunks, with ``device`` passed to ``label_dataset`` for the
move, keeps the device transfer on the main thread and reads ahead faster
than one that transfers from the prefetch thread.

.. autosummary::
   :toctree: generated
   :nosignatures:

   label_dataset


Strategy
--------

:class:`~nvalchemi.training.distillation.DistillationStrategy` is a
:class:`~nvalchemi.training.TrainingStrategy` over the named models
``"student"`` and ``"teacher"``. The teacher is frozen by omission from
``optimizer_configs``, the teacher signals are derived from the ``teacher_*``
targets the loss reads, and batches that arrive unlabeled are labeled on the fly
unless ``label_missing=False`` skips the teacher and lets the missing target
surface from the loss. ``training_fn`` stays a plain student forward, defaulting
to :func:`~nvalchemi.training.distillation.default_distillation_fn`, whose
``predicted_*`` keys are checked at construction against the outputs the student
actually computes — its ``active_outputs`` intersected with its declared
``outputs`` — so a student whose active set is narrowed is caught before the run
rather than on its first batch. A ``teacher_*`` target that no built-in signal
populates — a field a custom scorer wrote through ``label_dataset`` — is read
from the batch as it arrives: it is neither derived into a signal nor attached
on the fly, so a batch lacking it surfaces as a missing loss target.

A ``validation_config`` carrying its own ``loss_fn`` takes part in both checks:
its ``teacher_*`` targets widen the derived signal set, and its prediction keys
are checked the same way whenever the effective validation function
(``validation_fn`` falling back to ``training_fn``) is the stock one. Neither
re-runs on assignment, so pass ``validation_config`` to the constructor — or,
when rebuilding from a spec, to ``from_spec_dict``, which takes it as a runtime
override because specs exclude it — or name the wider set in
``teacher_signals``. Every resolved signal — derived or
explicit — is a request for its fields on every batch: a batch counts as
labeled only when it carries every resolved field, so adding a validation loss
with a new ``teacher_*`` target puts a training store written before it back on
the teacher, batch after batch, at identical values.

Training and validation batches go through one labeling seam: an internal hook
on ``BEFORE_FORWARD``, a stage both loops dispatch on the device-placed batch.
The teacher runs there with autocast disabled, so mixed-precision training does
not change the targets, and an on-the-fly label matches the offline one exactly
wherever the store returns the label dtype (see Labeling above): over the usual
float32 dataset every student but a float64 one agrees on both paths, while a
float64 student reads float32 back and needs a ``dtype_policy``. Labels are
never cast below single precision, so a ``bfloat16`` or ``float16`` student gets
float32 labels and needs ``dtype_policy="prediction_to_target"`` on its loss
terms; ``label_dtype`` overrides that inference with an explicit floating-point
dtype. The first batch the seam labels raises one ``UserWarning`` naming the
missing fields, since every later batch without them costs a teacher pass too.
Pointing ``validation_config`` at a store written by
:func:`~nvalchemi.training.distillation.label_dataset` still avoids the teacher
pass entirely, and validating an EMA-averaged student against the live teacher
is ``ValidationConfig(use_ema="auto")``, reported as ``model_source="mixed"``;
``use_ema="always"`` currently also demands an inference-slot entry for the
frozen teacher and fails at the first validation pass without one.

The seam's work is callable directly:
:meth:`~nvalchemi.training.distillation.DistillationStrategy.attach_teacher_labels`
attaches the ``teacher_*`` fields a device-placed batch is missing and reports
whether the teacher ran. It is idempotent, so pre-labeling a batch that later
reaches ``run()`` costs one teacher pass rather than two; a batch carrying only
some of the required fields is re-scored in full, since a partial set was
written for a different signal set than the objective reads.

Checkpoints serialize every entry of ``models``, so each write duplicates the
frozen teacher's weights; size the checkpoint interval accordingly with a large
teacher.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationStrategy
   default_distillation_fn


Losses
------

Every teacher signal shaped like a total energy, a force, or a stress is
consumed by a built-in loss term with its ``target_key`` pointed at the teacher
field — ``EnergyMSELoss(target_key="teacher_energy")``, and so on. Signals with
no supervised counterpart get their own term.

:class:`~nvalchemi.training.ComposedLossFunction` renormalizes its weights by
default, so composed weights are relative ratios: ``a + b + 0.2 * c`` runs at
``1/2.2``, ``1/2.2``, and ``0.2/2.2``. Build the composition with
``normalize_weights=False`` for literal coefficients, which also keeps a weight
schedule on one term from rescaling the others as it ramps.

.. autosummary::
   :toctree: generated
   :nosignatures:

   AtomicEnergyMatchingLoss
