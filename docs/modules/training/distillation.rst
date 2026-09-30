.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-distillation-api:

Distillation API
================

Teacher scoring, offline dataset labeling, the offline distillation strategy
and loss terms, and the on-policy generation components for
knowledge-distillation workflows.

.. seealso::

   - **Training strategy API**: :ref:`training-strategy-api`
   - **Fine-tuning API**: :ref:`training-finetuning-api`
   - **Loss API**: :ref:`losses-api`


Scoring
-------

A scorer turns a :class:`~nvalchemi.data.Batch` into named teacher signals,
each a :class:`~nvalchemi.training.distillation.TeacherSignal` mapping one
teacher output to a batch field, a level, and a canonical shape. The built-in
ones — ``energy``, ``forces``, ``stress``, ``atomic_energies``,
``embeddings``, and ``hessian`` — are requested by name; any other teacher
output is requested as a spec of its own, passed beside the built-in names:

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
found it, including neighbor tensors. Label precision is the scorer's decision.
Nothing that calls a scorer opens an autocast region of its own, so a scorer
runs inside whatever autocast region is open at the call site. The in-process
scorer's ``autocast`` setting picks the mode. The default ``False`` disables
autocast for the scoring pass, so a mixed-precision region around the call
never reaches the teacher. ``None`` leaves the caller's region in force. A
floating-point dtype enables autocast at that dtype, whether or not a region is
open. ``True`` enables it at the autocast dtype in force for the device: the
device default when no region is open, the caller's region's dtype when one is.

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
   SignalNormalizer
   NeighborListPolicy
   BUILTIN_SIGNALS
   SUPPORTED_SIGNALS

.. data:: TeacherLabels
   :type: TypeAlias

   Teacher signals for one batch, keyed by the batch field they populate:
   ``dict[str, tuple[torch.Tensor, SignalLevel]]``.

Scorers use three public type aliases. ``SignalLevel`` is the ``"node"`` or
``"system"`` level a signal is attached at. ``TeacherLabels`` is the
``{batch field: (detached tensor, level)}`` mapping that
:meth:`~nvalchemi.training.distillation.TeacherScorer.label` returns.
``SignalNormalizer`` is the type of the callable a spec's ``normalize`` slot
takes. The built-in specs are published as
:data:`~nvalchemi.training.distillation.BUILTIN_SIGNALS`, keyed by name, and
their names as :data:`~nvalchemi.training.distillation.SUPPORTED_SIGNALS`. A
:class:`~nvalchemi.training.distillation.TeacherSignal` names the teacher
output it reads, the ``teacher_*`` field it writes, the level, and an optional
``normalize`` callable shaping the raw output. When a spec declares companion
``extra_fields``, its ``normalize`` returns a mapping over all of its
``fields`` instead of a single tensor, and the spec is refused without a
``normalize``. The namespace and level rules are enforced when the spec is
built, and the in-process scorer refuses a spec naming an output the teacher
does not declare. The scorer publishes its
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

Forward-pass signals share one teacher pass, and ``embeddings`` adds a second.
``hessian`` labels a *Hessian-vector product*: the product of the teacher's
energy Hessian with a random probe direction, computed without forming the
Hessian. It adds an energy-only pass plus the two backward passes that
:func:`~nvalchemi.training.distillation.hessian_vector_product` takes through
it. ``hessian`` is the only built-in signal that writes two fields: the product
in ``teacher_hvp`` and the probe direction in ``teacher_hvp_probe``. The
student is later differentiated along that same probe.
:meth:`~nvalchemi.training.distillation.InProcessTeacherScorer.label_hvp`
computes one product for a probe the caller chooses. ``probe_seed`` pins the
probe direction. Left unset, every labeling draws a fresh direction, which is
what covers the Hessian over a run.
:class:`~nvalchemi.training.distillation.DistillationStrategy` sets it per
validation batch, so that the validation metric is comparable across passes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   hessian_vector_product


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
The strategy's own scorer disables autocast, so mixed-precision training does
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

Checkpoints store the frozen teacher *once per checkpoint root* rather than at
every index, so a periodic write costs only the student's weights. One root
holds one copy. Saving a different teacher into a root that already holds one
raises, rather than repointing the checkpoints already written there at
weights they were not written against. See
:meth:`~nvalchemi.training.distillation.DistillationStrategy.checkpoint_model_references`
and :ref:`distillation_recipes_guide` for how the stored copy is referenced,
fingerprinted, and read back on a restart.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationStrategy
   default_distillation_fn

Two objectives need a prediction that the student's forward pass does not
return, and each ships the training function that produces it. Both functions
are module-level, so a recipe that uses one still survives
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`.
Both are additive: they return the stock ``predicted_*`` outputs plus one key.
:func:`~nvalchemi.training.distillation.embedding_distillation_fn` runs the
student's ``compute_embeddings`` and routes the result through the
``"projector"`` model when one is registered.
:func:`~nvalchemi.training.distillation.hessian_distillation_fn` differentiates
the student's energy twice along the labeled probe. A recipe that needs both
predictions writes one module-level function of its own. Calling both stock
functions runs the student forward pass twice; building the union from
:func:`~nvalchemi.training.distillation.hessian_vector_product` and the
student's ``compute_embeddings`` avoids that.

.. autosummary::
   :toctree: generated
   :nosignatures:

   embedding_distillation_fn
   hessian_distillation_fn


On-policy generation
--------------------

On-policy distillation trains the student on frames it generated itself. The
run is a loop of *segments*. Each segment has a generation phase and a training
phase. In the generation phase, the student's own propagator advances a batch
of structures, and the teacher labels selected frames. The labeled frames
accumulate in a *replay buffer*. In the training phase, each training batch is
a *mixture* of samples from the replay buffer and from a *reference dataset*, a
teacher-labeled store, split at a fixed ratio, ``replay_ratio``.

:class:`~nvalchemi.training.distillation.OnPolicyConfig` describes one segment
loop. It sets which propagator generates, how many steps a segment runs, how
often the teacher labels, and how much of each training batch is replayed. The
propagator is any :class:`~nvalchemi.dynamics.base.BaseDynamics`, so a
relaxation optimizer generates paths exactly as an integrator generates
trajectories. The scalar settings live in
:class:`~nvalchemi.training.distillation.OnPolicySettings`. That class validates
on its own, so a recipe's settings can be checked before a teacher is built.

The *initial structures* are the structures the generated trajectories start
from. They come from any :class:`~nvalchemi.dynamics.StructureSource`, the
protocol that lists the members the loop reads. The protocol is also importable
here under its historical name, ``InitialStructuresSource``.
:class:`~nvalchemi.training.distillation.InitialStructures` is the reference
implementation. The sampler behind it lives in the core package:
:class:`~nvalchemi.dynamics.OrderedStructureSampler` serves a dataset's rows in
order, over the rows one rank owns, from a single position, ``next_row``. The
initial batch and a restart share that position. ``InitialStructures`` adds
only the recipe round-trip. A bare dataset is wrapped in an
``InitialStructures``, and an object that is neither a source nor a dataset is
refused with an error that names the protocol.

:meth:`~nvalchemi.dynamics.OrderedStructureSampler.draw` serves the structures.
It admits each candidate through one :class:`~nvalchemi.dynamics.FitPolicy`, a
predicate over the running atom and edge totals;
:class:`~nvalchemi.dynamics.WithinBudget` bounds those totals. At the first
candidate that does not fit, ``draw`` either stops or skips it. Stopping is how
an initial batch is packed. Skipping lets a backfill fill the room a graduated
structure freed.

Construction checks one row of the initial structures in two ways. First,
:meth:`~nvalchemi.dynamics.BaseDynamics.check_initial_batch` checks the row for
every field the propagator updates in place from its first step. Second, the
propagator's ``compute()`` runs once on the row, with the same isolation the
scorer uses: evaluation mode and ``requires_grad`` flags are restored
afterwards, and the propagator's last outputs are put back. This second check
catches declarations that no longer match the implementation, such as a
``__needs_keys__`` output the student never produces or a field ``compute()``
reads that nothing declared. The error names the offending key, and it is
raised where the config is built rather than on the first step of a long run.
The check costs one student forward pass, which front-loads the kernel and CUDA
initialization the first step pays anyway.

A graph model is probed with the neighbor list its ``neighbor_config``
declares. The list is built on the row and rolled back afterwards, so the probe
needs no hook. A model that plans more than one neighbor-list source is not
probed, and a warning says so: the probe builds exactly one list, and the check
must not refuse a propagator the loop can run. ``probe=False`` skips the
forward pass entirely. Use it for a propagator whose ``compute()`` must not run
outside the loop, or for a recipe check that should not pay for a forward pass.

.. autosummary::
   :toctree: generated
   :nosignatures:

   OnPolicyConfig
   OnPolicySettings
   SpecSerializable
   InitialStructures

The :class:`~nvalchemi.dynamics.ResizableSink` protocol that a capture sink
may satisfy is re-exported from this package and documented with the dynamics
sinks.

Three settings need care when sizing a run.

``label_frequency`` is the throughput setting, because the teacher is the
expensive model. It counts against the propagator's cumulative ``step_count``,
so the labeling cadence does not restart at a segment boundary. Each segment
also labels the frame it ends on, which is the most on-policy frame it
produced. Both labels are keyed on the pre-increment step count: the cadence
fires at ``AFTER_STEP`` before the counter advances, and the forced label of
the last frame is keyed at ``step_count - 1`` once the segment has run. The
next segment's first cadence dispatch lands one step after that forced label
and is therefore skipped rather than paid for twice. With ``generation_steps``
a multiple of ``label_frequency``, each trajectory is labeled
``generation_steps // label_frequency`` times per segment, which is once per
segment only when the two are equal. The first segment pays one more, for the
cadence dispatch at step ``0``.

``replay_capacity`` is enforced by FIFO eviction, which drops whole frames in
arrival order. A segment contributes one frame per trajectory per labeled step.
A capacity that is not a multiple of the trajectory count therefore cuts a
segment's frames partway through a labeled step. The structures at the back of
the batch are then over-represented in every mixture drawn afterwards. Size the
capacity as a multiple of the trajectory count.

``seed`` seeds every segment's mixture sampler, which adds the segment index to
it. Runs with consecutive seeds therefore draw the same mixtures, shifted by
one segment. Replicate runs draw independently only when their seeds are at
least ``num_steps // training_steps_per_segment`` apart.

``weight_sync_frequency`` is reserved at ``1``. The propagator shares the
student module, so an eager run is never out of sync.

The *labeling hook*, :class:`~nvalchemi.training.distillation.TeacherLabelHook`,
labels frames inline during generation. It is an ``AFTER_STEP`` dynamics hook.
It attaches ``teacher_*`` fields to the frame the propagator just resolved, each
at the level its signal declares. It can also copy a stripped version of the
frame into a :class:`~nvalchemi.dynamics.sinks.DataSink`. The hook never
touches the ``energy`` and ``forces`` the student wrote on the live batch,
because they drive the next step. It does strip them from the copy, together
with the neighbor tensors and the dynamics bookkeeping. A stored frame is
therefore a training sample rather than a propagator state, and it never
carries the propagated model's own prediction under a reference target's name.
Do not confuse the labeling hook with the strategy's private
``BEFORE_FORWARD`` labeling seam, which labels batches on their way into a
*training* step.

Labeling is idempotent per propagator step. A scorer that publishes
``label_fields``, or whose signal names are all built-in, is skipped when a
step it already labeled is dispatched again. A scorer that publishes neither is
skipped from its second dispatch on, once the first dispatch has revealed what
it writes. A forced label is never skipped, so the last frame of an
early-exiting segment and the final frame of the run are always labeled.

The segment loop registers the labeling hook itself. The hook stages each
segment's frames in a *capture sink*, which the loop drains into the replay
buffer at the segment boundary. The capture sink is host memory by default, or
the :class:`~nvalchemi.dynamics.sinks.DataSink` passed as
``OnPolicyConfig.capture_sink``. A :class:`~nvalchemi.dynamics.sinks.GPUBuffer`
keeps the staged frames on the generation device and avoids a device-to-host
copy per labeled frame. The loop sizes the capture sink. A segment captures at
most one frame per trajectory per labeled step, including the forced last
frame, so the sink must hold ``(generation_steps + 1)`` frames per trajectory
in the propagated batch. A configured sink with less capacity is grown through
``resize(capacity)`` when it satisfies
:class:`~nvalchemi.dynamics.ResizableSink`, and refused otherwise. A sink that
still holds frames when a segment starts is refused rather than drained as
generated data. Like ``dynamics`` and ``teacher_scorer``, the capture sink is
runtime-only.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherLabelHook
   nonfinite_divergence

Generated frames land in the replay buffer, a
:class:`~nvalchemi.training.distillation.ReplayBuffer`: an in-memory dataset
with a frozen key schema. The schema is frozen because appending a batch keeps
only the keys both sides hold, so one unlabeled frame would otherwise strip
``teacher_*`` from every frame already stored.
:func:`~nvalchemi.training.distillation.build_mixed_loader` then draws each
training batch with an exact reference/replay composition, resolved to whole
samples of the batch size. Rebuild the loader after every segment, because the
batch sampler reads the child dataset lengths once, at construction.

The two sources must agree on their whole batch schema. The schemas are
compared on a probe batch drawn from each side, not on field names, because a
Zarr-backed store and an in-memory buffer report field names differently.
Collation drops a field that only one side holds and zero-fills a whole level
that only one side holds. It also casts the second part of a mixed batch to the
dtype of the first, and either source may lead a chunk. All three kinds of
difference are therefore rejected. The reference dataset must be
teacher-labeled and stored in the replay-frame shape: the structure, the
propagator state, and the ``teacher_*`` labels. A reference dataset that
carries its own reference ``energy`` or ``forces`` is rejected, because mixing
it in would silently lose or fabricate those labels. Supervising one batch from
teacher labels and reference labels at once is not supported yet; it needs
masked loss composition.

The framework owns the schema freeze and the mixture. What enters the buffer
and what leaves it are left to policy. An
:class:`~nvalchemi.training.distillation.AdmissionPolicy` is a predicate over
the incoming frames that returns one boolean per graph. It runs before the
schema check, so the frames it rejects never enter. Examples are the
NaN-labeled frames of a diverged trajectory, or frames that fail a size or
diversity filter. An :class:`~nvalchemi.training.distillation.EvictionPolicy`
runs after the admitted frames are appended. Given the resident batch, ordered
oldest first with the admitted frames last, and the capacity, it returns the
indices to drop. It must name at least as many frames as the buffer is over
capacity. :class:`~nvalchemi.training.distillation.FIFO` is the reference
eviction policy. The string ``"fifo"`` builds it; that string is the only value
``ReplayEviction`` admits, and the one a recipe stores. The segment loop passes
``OnPolicyConfig.replay_admission`` and ``replay_eviction`` to the buffer it
owns. A policy instance on either field is runtime-only, and the declarative
``settings`` record a custom eviction policy as ``"fifo"`` with a warning.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReplayBuffer
   AdmissionPolicy
   EvictionPolicy
   FIFO
   ReplayEviction
   build_mixed_loader

Setting ``on_policy`` on the strategy combines these pieces into a run.
:meth:`~nvalchemi.training.distillation.DistillationStrategy.run` then takes no
dataloader. It seeds a state batch from ``initial_structures`` and repeats
generate-label-train segments until ``num_steps`` optimizer steps are done. The
``1 - replay_ratio`` share of every batch is drawn from ``reference_dataset``.
A reference dataset is required unless the ratio is ``1``, and refused when the
ratio is ``1``, because it would then be checked but never sampled.

The initial batch is stamped with fresh dynamics bookkeeping on the way in.
Structures loaded from a store that an earlier relaxation graduated therefore
do not arrive frozen at ``exit_status``. At construction, the strategy probes
the reference dataset once to check three things: the fields the labeling hook
strips, the device the dataset emits on, and the teacher fields the
propagator's scorer declares. A mismatch in any of them is a guaranteed mixture
failure that would otherwise surface only after a whole generation segment had
been paid for.

One segment is one epoch. ``AFTER_EPOCH`` and epoch-cadence validation
therefore run at segment boundaries, and step-cadence validation runs inside
segments. The run's closing validation is skipped when a cadence already
validated at the final step. The segment is also the restart granularity. A
run resumed from a checkpoint taken mid-segment counts that segment as finished
rather than replaying the batches it had left. The same holds for an offline
run that continues on-policy from a partial epoch. A second call to ``run()``
on the same strategy keeps the replay buffer the first call filled and reseeds
only the trajectory. Installing the rank shard resets the sampler to the front
of its rows, so the second call generates from the same structures again, not
from whatever the first call left over.

Under a ``DDPHook`` the loop runs data-parallel. Each rank propagates its own
shard of the initial structures; see :ref:`distillation-scaling-out`.

Generated frames are drained to host memory and then staged on the reference
dataset's own device, so a GPU-resident reference dataset and the replay buffer
collate on one device. ``replay_device`` overrides that choice and is checked
against the reference dataset at construction. The reference device is the one
the dataset actually emits batches on, read off a batch whenever no declaration
settles it. A :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset`
declares no device, and a store opened without a device declares an index-less
``cuda`` that names whichever device is current.

The student is held in evaluation mode to generate and switched to training
mode for the training phase only. Generated frames therefore cost no
second-order graph and no moving batch-norm statistics. A propagator model that
merely *composes* the student is held in evaluation mode for the whole loop and
moved whole to the generation device. The training phase runs
``models["student"]`` rather than the composition, and only the named models
move with the strategy. The propagator must hold the very module registered as
``models["student"]``, either on its own or composed into a larger model. That
object identity is what makes each segment generate from the weights the
previous segment trained, and it is checked at construction.

Splitting a built-in propagator's run into segments is exact. ``run`` never
resets ``step_count`` or the integrator state, and the Langevin thermostat
draws from a counter-based generator keyed on the cumulative step count. Two
segments of ``K`` steps therefore reproduce one run of ``2K`` steps. A
dynamics hook that is sensitive to being opened and closed, such as
:class:`~nvalchemi.dynamics.hooks.LoggingHook`, is re-entered once per
segment. The progress of a segment that converges and
exits early is read from ``dynamics.step_count`` rather than assumed to be
``generation_steps``. A :class:`~nvalchemi.dynamics.FusedStage` pays its
priming forward pass once per segment, so prefer a bare propagator.

A custom ``teacher_*`` field that the propagator's scorer writes is an ordinary
loss target, exactly as in offline distillation. Generation writes it on every
captured frame, so ``reference_dataset`` must carry it too. The
generation/reference parity check enforces this whenever the scorer declares
``label_fields``. Validation data must also arrive with the field, because the
strategy's own scorer produces built-in signals only and cannot fill it in. At
least one built-in ``teacher_*`` target, or an explicit ``teacher_signals``, is
still required alongside it. A scorer that declares neither ``label_fields``
nor built-in signals writes fields that cannot be known before it scores a
batch. The strategy then warns that the parity check is deferred to the first
segment's loader.

``on_policy`` and ``reference_dataset`` serialize as references rather than as
the objects themselves.
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
therefore carries the whole on-policy run, and a rebuild needs only its models
supplied back. A run whose datasets live in memory, or whose propagator hides
its constructor arguments, leaves the ``on_policy`` block out of the spec with
a warning that names the piece. Either way, every rebuild entry point takes
the live objects as keyword arguments:
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_spec_dict`,
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_checkpoint_dict`,
and
:meth:`~nvalchemi.training.distillation.DistillationStrategy.load_checkpoint`
all accept ``on_policy`` and ``reference_dataset``. The checkpoint entry points
pass them to ``from_spec_dict`` as *runtime overrides*: live objects that a
spec cannot carry whole, which :func:`nvalchemi.training.load_checkpoint`
forwards as extra keyword arguments. A live object handed over that way
outranks the spec's own ``on_policy`` block. The segment loop is tied to the
student it propagates, so pass back the ``models`` the propagator was built
around as well, through the loader's own ``models=`` keyword. The checkpoint's
weights are then restored into those very objects.
Alternatively, calling
:meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint` on a strategy
already constructed with the loop reaches the same result. A Boltzmann term,
which matches the teacher's Boltzmann distribution over the configurations the
student generated (see :ref:`distillation-advanced-objectives`), is defined
only on generated batches. It refuses to rebuild as an offline run, so the
loop must come back with it, either from the spec's ``on_policy`` block or by
re-supply.

Relaxation
----------

A relaxation propagator generates paths that *end*. ``fmax`` tells the segment
loop about that and turns on the *trajectory lifecycle* described below.
``fmax`` is the max-force-norm threshold, a plain number a recipe can hold.
``convergence_hook`` instead takes a
:class:`~nvalchemi.dynamics.base.ConvergenceHook` when the run needs a live hook
rather than a number. ``convergence_criterion`` resolves the two into one
criterion. For the duration of the run, the loop installs that criterion on the
propagator in two roles: as the hook that migrates status, and as the
convergence detector. Graduation and detection therefore cannot disagree. A
detector the propagator was built with is set aside and restored afterwards.

A hook passed as ``convergence_hook`` must migrate status on every step, from
the status ``0`` the run stamps its structures with. A criterion that only
reports convergence would look configured while freezing and graduating
nothing. A criterion that skips steps would graduate a structure late, and both
capture routes (see below) would store that frame. The lifecycle must also be
the only thing that migrates status. A propagator that already carries its own
status-migrating ``ConvergenceHook``, or its own sampler, is therefore refused;
it would otherwise run at two thresholds or refill mid-segment. A
multi-sub-stage :class:`~nvalchemi.dynamics.FusedStage` is refused at
construction, where that shape is fixed: the stage builds a migrator for every
sub-stage except the last, and for the last one too when it declares a
``convergence_hook``.

The construction probe runs the propagator's ``compute()`` on one row. It also
dispatches a copy of the criterion to that row, stamped with the ``status`` the
run gives its structures. A criterion that raises on the propagator's outputs,
or whose firing leaves the status column unchanged where it converged, is
refused before a run is paid for. The probe does not check whether the
structure converges, which depends on the data. It checks that the migration
works. A criterion that reads a key no ``compute()`` produces is not
dispatched, because a hook may write that key during the step. A warning names
the key instead.

The lifecycle keeps the buffer filling with informative frames. A converged
structure freezes in the propagator's step and is stored once, as the minimum
it reached. Every later capture of the segment leaves it out instead of writing
it again. At the segment boundary it *graduates*: it leaves the batch, and the
optimizer's own per-structure state follows the membership change. The initial
structures then *backfill* the room it freed, with at most as many structures
as graduated, within the atoms they held. The backfill goes through
:meth:`~nvalchemi.dynamics.OrderedStructureSampler.draw` with
``on_miss="skip"``, so one oversized row never starves the refills behind it.

A budgeted :class:`~nvalchemi.training.distillation.InitialStructures` packs the
initial batch and leaves the remaining rows, in order, for the backfill. An
unbudgeted one is propagated whole, so its position starts past the last row.
The batch then narrows by one trajectory per graduation, unless
``recycle=True`` wraps the position to the front of the rows this rank owns. A
backfilled structure is restamped with fresh bookkeeping, keeping only the
``system_id`` the source assigned. A store of minima that an earlier relaxation
graduated therefore does not arrive frozen.

A trajectory can also end by diverging. No criterion ever accepts a NaN, so the
``OnPolicyConfig.divergence`` predicate handles that case. By default it is
:func:`~nvalchemi.training.distillation.nonfinite_divergence`, which flags a
graph whose positions or forces are no longer finite. A flagged graph is frozen
at ``exit_status`` on that step and kept out of both capture routes. At the
boundary it is retired and backfilled like a converged one, and one warning per
boundary counts the diverged graphs. A custom predicate takes the live frame
and returns one boolean per graph, the same shape an
:class:`~nvalchemi.training.distillation.AdmissionPolicy` has. It is
runtime-only, and one that returns any other shape is refused on its first
dispatch. When the last trajectory finishes and nothing is left to start
another, the loop warns once and trains its remaining steps on the frames it
has.

Frames reach the buffer by two *capture routes*, and each frame takes exactly
one of them. The *path route* is
:class:`~nvalchemi.training.distillation.TeacherLabelHook`. The lifecycle gives
it the propagator's ``exit_status``, so it stores only the structures still
relaxing. It labels them inline, and it narrows the frame to them before the
teacher runs rather than after, so a mostly frozen batch costs only a small
teacher pass. A run without a lifecycle leaves the hook unnarrowed, so a
propagator that manages its own convergence keeps its final frames. The
*converged route* is a converged-frame hook that stores each minimum once. It
captures the frame, unlabeled, at ``ON_GRADUATE``: the stage every
:class:`~nvalchemi.dynamics.base.BaseDynamics` and
:class:`~nvalchemi.dynamics.FusedStage` propagator dispatches with the graphs
whose status reached ``exit_status`` on the step. A fused stage's own
``ON_CONVERGE`` fires on its sub-stages only; it dispatches ``ON_GRADUATE``
after its step-budget migration. A
:class:`~nvalchemi.distributed.DomainParallel` propagator dispatches no
``ON_GRADUATE``, so ``OnPolicyConfig`` refuses one with a criterion rather
than let its minima go uncaptured. The hook leaves out a graph the divergence
predicate has flagged, and a graph whose final frame the path route stored on
the same step. The converged frames are labeled in a single teacher pass when
the hook's sink is drained, which keeps the teacher's batch size independent of
the propagated one.

The path route stages its frames in ``OnPolicyConfig.capture_sink`` when one
is configured. Every segment needs ``(generation_steps + 1)`` frames per
trajectory in the batch. A sink too small for that is grown through
``resize(capacity)`` when it offers one, and refused up front when it does not.
The sink is never shrunk. A backfill never grows the batch past its initial
size, so a sink that fits the initial batch fits every later one, and the
growth happens at most once, for the first segment. The converged route keeps
its own host-memory sink, one frame per graph.

A custom :class:`~nvalchemi.training.distillation.InitialStructuresSource`
drives the lifecycle too, provided its ``initial_batch`` stamps the ``status``
zeros and ``system_id`` numbers the lifecycle graduates and backfills on.
Distribution-matching objectives are defined on equilibrium ensembles, which a
relaxation path is not. A Boltzmann term is therefore refused at construction
beside a relaxation propagator. When ``on_policy.samples_equilibrium`` is left
``None``, whether the propagator samples an equilibrium ensemble is inferred
from the propagator itself: from the
:attr:`~nvalchemi.dynamics.BaseDynamics.samples_equilibrium` its class
declares, which the relaxation optimizers set to ``False``, and from the
convergence criteria it or the loop carries. ``True`` admits a propagator that
the inference would refuse, and ``False`` refuses one that it would admit.
Pointwise energy,
force, and atomic-energy matching distill a relaxation path exactly as they
distill a trajectory.

.. _distillation-scaling-out:

Scaling out: multi-GPU and multi-node
-------------------------------------

On-policy distillation scales as synchronous data parallelism. The teacher is
frozen and only runs forward passes, so a teacher that fits on one accelerator
is *replicated* onto every rank, while the student is trained data-parallel.
Each rank generates its own trajectories, labels them with its own teacher
replica, and fills its own replay buffer. The only per-step training traffic
between ranks is the student's gradient all-reduce. Setup adds small
collectives, which check the shards and the replay placement, and validation
all-reduces its metrics. The teacher never joins a collective. The script is
the single-process one plus a :class:`~nvalchemi.training.hooks.DDPHook`,
launched with one process per GPU:

.. code-block:: python

   strategy = DistillationStrategy(
       models={"student": student, "teacher": teacher},
       optimizer_configs={
           "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
       },
       loss_fn=(
           EnergyMSELoss(target_key="teacher_energy")
           + ForceMSELoss(target_key="teacher_forces")
       ),
       num_steps=10_000,
       devices=[torch.device("cuda")],
       hooks=[
           DDPHook(),
           CheckpointHook("runs/distill/checkpoints", epoch_interval=1),
       ],
       reference_dataset=labeled_store,
       on_policy=OnPolicyConfig(
           dynamics=propagator,
           teacher_scorer=scorer,
           initial_structures=InitialStructures(structure_store),
           replay_ratio=0.5,
           training_steps_per_segment=32,
       ),
   )
   strategy.run()

.. code-block:: bash

   # One node, one process per GPU.
   torchrun --standalone --nproc_per_node=8 distill.py

   # Four nodes, run on each of them.
   torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d \
       --rdzv_id=distill --rdzv_endpoint=$HOST:29500 distill.py

``DDPHook`` wraps every optimizer-configured model, so it wraps the student and
never the teacher, and it pins each rank to its node-local device. The segment
loop adds the sharding that the generation phase needs. The initial structures
are dealt out strided: rank ``r`` takes every ``world_size``-th row, starting
at row ``r``. They must therefore hold at least one structure per rank, and are
best sized as a whole multiple of the world size. A set that does not divide
evenly triggers a warning. Every rank draws the same number of replay samples
from a replay buffer that holds only its own trajectories, so the frames of a
shorter shard are drawn more often. The deal balances the number of rows, not
the work, so sort the dataset by atom count when structure sizes vary. The rows
a rank owns are public as
:attr:`~nvalchemi.training.distillation.DistillationStrategy.structure_shard`,
and every backfill on that rank draws from them alone.

Each rank offsets the mixture sampler's ``OnPolicyConfig.seed``, and every
``random_seed`` that the propagator and its sub-stages expose, by its global
rank times ``rank_seed_stride``. The propagator side goes through
:meth:`~nvalchemi.dynamics.BaseDynamics.seed_offset`, which walks the fused
sub-stages and reports what it cannot move. Both streams add a step counter to
their seed, so the stride has to stay above every counter the run reaches. The
default, the prime ``1_000_003``, does so for a run whose counters stay below
it. Set a different stride when a replicate launch's seeds would land on another
rank's stride. A stage that holds randomness the offset cannot move, such as a
:class:`torch.Generator` with no integer ``random_seed``, is named in a warning
from every rank, and the caller must give it a rank-distinct seed. This matters
most when the initial structures are replicas of one geometry, because sharding
then separates nothing.

A multi-rank launch whose student nothing wraps is refused. The check is only
that *something* owns ``models["student"]`` after setup, so a wrapper of your
own passes it just as ``DDPHook`` does. A wrapper that works in place, such as
FSDP2's ``fully_shard`` or hook-based gradient synchronization, leaves nothing
for the check to find. For such a wrapper, ``require_wrapped_student=False``
waives the check with a one-time warning, and keeping the ranks' students in
step becomes your responsibility.

The reference dataset is *not* sharded. Every rank draws from all of it with
replacement, so ranks share reference samples, while the generated frames and
the teacher passes that label them are partitioned. Each rank's replay buffer
is staged on the reference dataset's device. Keep that dataset in host memory,
or let it emit lazily: a :class:`~nvalchemi.data.datapipes.dataset.Dataset`
opened with no ``device``, or with an index-less ``"cuda"``, draws its first
batch after ``DDPHook`` has pinned the rank, so the batch lands on that rank's
GPU. Staging the dataset eagerly on a GPU before the pin concentrates the whole
world's replay buffers on that one GPU, and every rank warns about it. Moving
the dataset onto ``ctx.workflow.devices[0]`` in a ``TrainingStage.SETUP`` hook
places it correctly. An index-less ``replay_device`` names the device this rank
has made current.

Multi-node runs take the same code path with a larger world. Sharding keys on
the global rank, and device placement on the node-local rank. The ``c10d``
rendezvous above is what lets one command run on every node. Validation runs on
every rank and all-reduces its metrics, so never guard it behind a rank check.
:class:`~nvalchemi.training.hooks.CheckpointHook` writes from global rank zero
only.

A restart resumes the optimizer state and the counters. Under
``restart="reseed"`` it reseeds every rank's trajectories from that rank's
shard and refills the replay buffer from scratch, so budget the first segments
after a restart as cold. The default ``restart="error"`` refuses a multi-rank
restart instead, because the restart bundle it would drop holds rank zero's
state alone. A restart needs no device bookkeeping:
:meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint` loads onto the
live ``devices``, and ``run()`` re-homes the optimizer state after the hook has
pinned the rank.

Every rank runs the same number of segments, and the same number of batches per
segment, which keeps the ranks arriving at each all-reduce together. An update
orchestrator that vetoes optimizer steps unevenly across ranks would
desynchronize them. A stalled rank blocks its peers for the process group's
default timeout. ``DDPHook`` exposes no timeout setting, so to bound the wait,
initialize the process group yourself with ``timeout=``.

The world *divides* the generation work. A segment's total frame count and
teacher cost equal the single-process run's, with each rank contributing
``1/world_size`` of them. ``generation_steps``, ``label_frequency``, and
``replay_capacity`` are per rank. At a fixed ``replay_capacity``, each rank's
buffer therefore spans ``world_size`` times as many segments, and every mixed
batch grows staler as the world grows. Raise ``generation_steps`` or the
structure count with the world, or lower ``replay_capacity`` by the world size,
but not both.


Recipes and the CLI
-------------------

A whole on-policy run survives
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict` as
references. :meth:`~nvalchemi.training.distillation.OnPolicyConfig.to_spec_dict`
carries every scalar setting verbatim. It carries the propagator as the
``cls_path`` and keyword arguments it rebuilds from, with the student rebound
at build time. It carries the scorer as its signal set, dtype, and probe seed
over the strategy model named ``"teacher"``. It carries ``initial_structures``
as the store it reads under the budgets it was given, but never its position,
which is restart state. ``reference_dataset`` serializes the same way, as the
store it reads.

A ``convergence_hook``, a propagator's hooks, convergence hook, and sinks, and
a dataset holding its samples in memory are the runtime-only parts. The
``convergence_hook`` and the propagator's collaborators are omitted with a
warning that names them. The warning covers a hand-built propagator and one a
recipe built alike, and skips the segment loop's own labeling hook. A dataset
held in memory is refused with the fix in the message. A piece that cannot be
described leaves the whole ``on_policy`` entry out, rather than writing a
recipe that would rebuild into a different run.
:meth:`~nvalchemi.training.distillation.OnPolicyConfig.from_spec_dict` and
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_spec_dict`
rebuild around supplied models, and both take overrides for the runtime-only
pieces.

An interrupted on-policy run also carries a *restart bundle* through the
checkpoint: its live trajectory batch, the propagator's cumulative step count,
the initial structures' position, its replay frames, and the settings it ran
under. A resumed run therefore continues the same trajectory instead of
seeding a fresh one, and backfills from the row the interrupted run reached.
The restored frames replace the buffer's contents rather than being merged
into them. The recorded settings are compared against the resumed loop's, so
a run whose two halves differ says so. A run whose generation ran dry carries
its frames and the exhaustion, and resumes training on the buffer.

The restart bundle is rank-local, because the strategy checkpoint it rides in
is written on rank zero alone. It is consumed only when a single rank wrote it
and a single rank is restoring it. Otherwise ``OnPolicySettings.restart``
decides. ``"error"`` (the default) refuses to start. ``"reseed"`` drops the
bundle with a warning, and each rank reseeds with a cold replay buffer.
``"resume"`` also refuses a restore that carries no bundle. A resumed run
starts at a segment boundary: the interrupted segment is counted as finished,
as above. The fresh segment the run opens begins by generating, so a
checkpoint written part-way through a training phase costs the resumed run
one extra generation phase.

``nvalchemi.training.distillation.cli`` wraps all of that as a ``distill``
group on the ``nvalchemi-training`` entry point. A *recipe* is one JSON file,
validated by
:class:`~nvalchemi.training.distillation.cli.DistillationJobSpec`, that
describes a whole distillation run. The group writes a recipe
(``distill init``), publishes its schema (``distill schema``), validates and
renders it (``distill spec report``), and runs it (``distill spec run``).
``distill spec resume`` picks an interrupted run back up, at the budget the
checkpoint recorded or at the recipe's under ``--budget recipe``.
``distill evaluate`` gates the result, on the EMA average or on the trained
weights as ``--weights`` says.

Pre-flight deserializes the strategy bundle with the same helpers the runtime
uses, and checks an ``on_policy`` block against
:class:`~nvalchemi.training.distillation.OnPolicyConfig`'s own field
constraints. Everything the recipe settles on its own is therefore refused at
``spec report`` rather than after a teacher has reached a GPU: a setting out
of range, a step budget below one, a dataset format no loader builds, a model
source the CLI could never load, a batch too small to hold a whole sample from
each mixture source, or an ``initial_structures`` block that names no store or
carries a budget that is not a positive count. A rule the strategy or the
segment loop owns is not repeated: the optimizer configuration, a recycling
source without a lifecycle, the replay device, or a replay-only mixture is
refused once, at ``spec run``, and reported as a CLI error with the owner's
message.

``init`` scaffolds a :class:`~nvalchemi.training.hooks.CheckpointHook` into
``student.hooks``, so the run leaves a checkpoint to resume from and to
evaluate. :class:`~nvalchemi.training.distillation.cli.EvaluationSpec` accepts
only the accuracy bars ``distill evaluate`` can fill, because a bar with no
measurement behind it fails the student rather than being skipped. A *student
tier* is a size template, never an architecture: a width, a depth, and a
radial basis size for whatever constructor ``student.spec`` names. Each tier is a
:class:`~nvalchemi.training.distillation.cli.StudentTier` in the registry
:data:`~nvalchemi.training.distillation.cli.DEFAULT_STUDENT_TIERS`, which
holds ``small``, ``base``, and ``large`` and grows through
:func:`~nvalchemi.training.distillation.cli.register_student_tier`.
``init --tier`` is checked against the registry when the command runs, and
``--tier-kwargs`` overrides a template's arguments. The public alias
:data:`~nvalchemi.training.distillation.cli.DistillationMode` names the loop a
scaffold is written for: ``offline`` or ``on-policy``.
:ref:`distillation_recipes_guide` walks the lifecycle end to end.

.. currentmodule:: nvalchemi.training.distillation.cli

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationJobSpec
   StudentSpec
   EvaluationSpec
   DistillationMode
   StudentTier
   DEFAULT_STUDENT_TIERS
   register_student_tier

.. currentmodule:: nvalchemi.training.distillation


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


.. _distillation-advanced-objectives:

Embedding, Hessian, and Boltzmann objectives
--------------------------------------------

Three further terms distill quantities that a reference dataset has no column
for. Each needs more from the run than a target field. Each is checked at
construction, in the training loss and in a ``validation_config`` loss alike.

:class:`~nvalchemi.training.distillation.EmbeddingMatchingLoss` is the
*embedding term*: it matches the student's per-atom representation (its node
embeddings) to the teacher's, component by component. Both sides come from
``compute_embeddings`` rather than from a forward pass, so the term needs
:func:`~nvalchemi.training.distillation.embedding_distillation_fn`, and the
student runs twice per batch. Across architectures the two embedding widths
differ, and the learnable
:class:`~nvalchemi.training.distillation.EmbeddingProjector` reconciles them.
Construct it with the student's width and the teacher's width, and register it
as a ``"projector"`` model with an ``optimizer_configs`` entry of its own. The
training function then routes the student's embeddings through it. The
projection is applied to the student and never to the teacher, whose
embeddings stay fixed targets. A learnable map on the target side would
minimize the objective by collapsing the teacher's representation. The
projector is a training-time artifact: the distilled model is the student
alone.

The training function refuses student embeddings that are detached from the
student's trainable parameters, treating them as an accident. A wrapper that
computes them under ``torch.no_grad`` produces such embeddings, and training on
them would leave the projector absorbing the term. A student representation
frozen on purpose, such as frozen trunk layers beside a trainable head, is
declared by registering the projector with ``frozen_student=True``, so that the
projector alone carries the term.

.. code-block:: python

   from nvalchemi.training.distillation import (
       DistillationStrategy,
       EmbeddingMatchingLoss,
       EmbeddingProjector,
       embedding_distillation_fn,
   )

   projector = EmbeddingProjector(student_width, teacher_width)
   strategy = DistillationStrategy(
       models={"student": student, "teacher": teacher, "projector": projector},
       optimizer_configs={
           "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)],
           "projector": [OptimizerConfig(optimizer_cls=torch.optim.Adam)],
       },
       loss_fn=EnergyMSELoss(target_key="teacher_energy")
       + 0.1 * EmbeddingMatchingLoss(),
       training_fn=embedding_distillation_fn,
       num_steps=10_000,
   )

Two representations agree only up to the symmetries of each architecture's
embedding space, such as a channel permutation or a rotation of an equivariant
block. Both are linear maps, so a linear projector absorbs them; without a
projector they are matched component by component and leave a residual floor.
Differences that no linear map closes, such as a feature one architecture
builds and the other does not, leave a floor with or without a projector, which
is why a residual floor on the embedding term is normal. Weight the term as a
regularizer beside the terms that carry the physical targets.

:class:`~nvalchemi.training.distillation.HessianMatchingLoss` matches the
curvature of the teacher's energy surface. Curvature decides vibrational
spectra and integrator stability, and energies and forces do not pin it down.
Neither side forms a Hessian. Both sides compute a Hessian-vector product along
one random probe direction, at two backward passes each. The ``hessian`` signal
materializes the teacher's product and its probe onto the batch, either offline
through :func:`~nvalchemi.training.distillation.label_dataset` or on the fly
through the strategy's labeling seam. The student's product comes from
:func:`~nvalchemi.training.distillation.hessian_distillation_fn`, which takes it
on a second student pass narrowed to the energy alone. The second pass is
needed because a conservative student derives its forces from the very graph
the second derivative needs, and frees that graph outside training mode, so
the stock forward cannot be differentiated again. Every validation pass costs
the same two passes.

One probe constrains one direction, so coverage comes from redrawing the probe.
An on-policy run gets a fresh probe every time it labels a frame, whereas a
store labeled once freezes one direction per structure. Because each probe
component is standard normal, the graph-balanced value is a Hutchinson estimate
of ``||dH||_F^2 / 3V`` in (eV/A^2)^2. For a near-converged student, that value
is one to two orders of magnitude above the force mean-squared error. Start the
term a hundred to ten thousand times lighter than the force term, and treat a
single batch's value as the noisy one-sample estimate it is. For a direct-force
student, the strategy warns that the term supervises the energy head alone.

:class:`~nvalchemi.training.distillation.BoltzmannMatchingLoss` implements
*Boltzmann matching*: it matches the distribution of configurations rather than
each configuration. The term is the relative entropy between the teacher's and
student's Boltzmann distributions at a given temperature. It is blind to a
constant energy offset and to any error that does not change relative
populations. ``beta`` interpolates between the forward direction (``0``,
mass-covering) and the reverse direction (``1``, mode-seeking).

The estimator reads a batch as a sample of the *student's* own canonical
ensemble, which is what makes the student-side weights uniform. The strategy
therefore requires ``on_policy``. It refuses a relaxation propagator and any
convergence criterion: the propagator's own hook, a
:class:`~nvalchemi.dynamics.base.ConvergenceHook` registered on it that
graduates graphs out, or one the segment loop installs from ``fmax`` or
``convergence_hook``. It also warns when ``replay_ratio`` mixes reference
frames that the student never visited into the batch. Reweighting an
off-policy sample back onto the student's distribution is not offered, so an
existing dataset reaches the term as ``reference_dataset``, mixed into
generated frames by ``replay_ratio``.

The batch also has to hold configurations of one system, because energies of
different systems are not comparable. Seed the run with replicas of one
structure, one walker per graph. The one-system check compares atom counts
only: it refuses a batch whose graphs differ in size, and it cannot tell two
systems of the same size apart. Under data parallelism, each rank holds a
shard of one *world batch*, the union of every rank's batch. The check reads
the gathered world batch, so a rank holding one system beside a rank holding
another of a different size is refused too. ``check_one_system=False`` turns
the check off for an ensemble whose composition is meant to vary. A graph
whose teacher or student energy is not finite is dropped from the ensemble
(``ignore_nonfinite``), since one such row would otherwise reach every rank's
softmax through the gather. The temperature cannot be checked: set the term's
temperature and the thermostat's from the same number.

The two directions differ in scale. The forward direction is bounded above by
``log B``, and its gradient vanishes once the softmax saturates, as it does for
a student whose error spreads over more than a few ``k_B T``. ``beta=0`` can
therefore read as converged while the student is far off. Hold ``beta`` at
``0.5`` or above until the student is within a couple of ``k_B T``. Reducing
energies by ``k_B T`` also puts the gradient of either direction at up to
``1/k_B T`` per configuration, about 39 eV^-1 at 300 K, well above what a
pointwise energy term produces. Under a
:class:`~nvalchemi.training.hooks.DDPHook`, the term gathers the reduced
energies across ranks with a differentiable all-gather and normalizes the
softmax over the world batch. Each rank reports the world loss, the averaged
gradient is the world loss's own, and the distribution the softmax sees is
``world_size`` times ``batch_size`` wide.

The recommended recipe is therefore ``replay_ratio=1`` *and* a bounded
``replay_capacity``. The ratio keeps reference rows out of the batch. The
capacity keeps stale generated rows out, because every segment's loader draws
uniformly over the whole replay buffer and an unbounded buffer retires nothing.
Size the capacity to the frames that one segment or a few segments yield.
Validation is the other off-policy path, and the strategy refuses it outright.
A Boltzmann term in the validation loss is refused at construction, and so is
a ``ValidationConfig`` without a ``loss_fn`` of its own, which would reuse a
training loss that holds the term. Give the validation config a pointwise loss
instead.

.. autosummary::
   :toctree: generated
   :nosignatures:

   EmbeddingMatchingLoss
   EmbeddingProjector
   HessianMatchingLoss
   BoltzmannMatchingLoss


.. _distillation-evaluation:

Evaluation and acceptance
-------------------------

``nvalchemi.training.distillation.evaluation`` answers whether a distilled
student is good enough to ship. It is its own subpackage: its names are
imported from ``nvalchemi.training.distillation.evaluation``, not from
``nvalchemi.training.distillation``. A student is accepted when it clears every
*acceptance bar* it is given. An acceptance bar is a limit on
one number from the student's measurements, such as a maximum force MAE or a
minimum throughput.

Accuracy is measured over a held-out set with
:func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy`. It
compares the student against either the dataset's own labels
(``targets="reference"``) or the teacher's labels (``targets="teacher"``).
Teacher labels are written offline by
:func:`~nvalchemi.training.distillation.label_dataset` or scored on the fly by a
``scorer``. The pass runs through :class:`~nvalchemi.training.ValidationLoop`,
so eval mode, device placement, and the autograd policy an autograd-force
student needs behave as they do in training validation. No autocast is
applied, so the student predicts in its own dtype. A bare model passed as the
scorer is wrapped in an in-process scorer whose labels are cast to
``label_dtype``, by default the student's first floating-point parameter dtype
floored at float32. A supplied scorer's labels are not cast. Every residual is
accumulated in float64 as an exact global sum rather than read off the
graph-balanced loss.

The weights scored are exactly the ones passed in. To score a student trained
under an ``EMAHook`` on its averaged weights, pass
``strategy.inference_model``. ``StudentEvaluation.weights`` records which of
the two weight sets was scored. A scorer's labels pass the same ``teacher_*``
namespace guard that every other labeling route applies. A custom scorer that
returns ``energy`` or ``positions`` is therefore refused before it can rewrite
the inputs or reference targets of the batch the student is about to read.

Whenever forces are compared, against either target family, two
force-alignment numbers fill in. ``force_cosine_mean`` weights every atom
equally, so atoms whose force is at or below the student's own error dominate
it. The ``min_force_cosine`` bar therefore reads the
magnitude-weighted ``force_cosine_aggregate`` instead. The compared quantities
form an open table. The built-ins are
:data:`~nvalchemi.training.distillation.evaluation.BUILTIN_ACCURACY_QUANTITIES`,
one :class:`~nvalchemi.training.distillation.evaluation.AccuracyQuantitySpec`
each. A spec names the prediction key, the reference field, the teacher signal,
and the loss term that drives the pass. A spec passed in ``quantities`` scores
a custom head, or a built-in quantity read from another field, and reports it
under ``AccuracyMetrics.errors``.

.. currentmodule:: nvalchemi.training.distillation.evaluation

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate_accuracy
   AccuracyMetrics
   AccuracyQuantity
   AccuracyQuantitySpec

.. data:: BUILTIN_ACCURACY_QUANTITIES
   :type: Mapping[str, AccuracyQuantitySpec]

   Specs of the built-in quantities, keyed by the name a caller requests them
   under.

The *non-conservative residual* is the part of a teacher's force field that no
conservative student can fit.
:func:`~nvalchemi.training.distillation.evaluation.non_conservative_residual`
measures it. A student that differentiates an energy produces a curl-free
field, so it fits only the conservative part of a direct-force teacher. The
probe integrates the teacher's work around closed loops in configuration space.
That work is zero for a conservative field. The probe converts the leftover
work into ``force_floor``: a lower bound on the root-mean-square per-atom force
error that a conservative student must make somewhere on the loop.

The bound holds at the displacement scale that ``amplitude`` probes, so choose
an amplitude on the order of a thermal vibration. The bound loosens as
``1 / sqrt(N)`` with system size, because one randomly oriented loop spans a
``1 / sqrt(3N)`` fraction of the field's curl. Compare floors only between
probes of similar system size. A conservative teacher reports the midpoint
rule's quadrature error, which falls as ``segments`` rises. Below that, it
reports the round-off of the batch's own dtype. In float32 that round-off is
near ``1e-9`` eV/A for an argon-like Lennard-Jones solid, and a hundred times
more for a lattice a hundred times stiffer. A floor below that level needs a
float64 batch and teacher.

.. autosummary::
   :toctree: generated
   :nosignatures:

   non_conservative_residual
   NonConservativeResidual

Stability is what small students actually fail at, so it is measured on a
trajectory the student drives itself. The *stability monitor*,
:class:`~nvalchemi.dynamics.hooks.StabilityMonitor`, is a dynamics hook that
records the total energy and momentum of that trajectory. It reports energy
drift and momentum conservation once the run is over. It is the offline
counterpart of :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook`: it
keeps the whole series instead of comparing one live value against a
threshold. The monitor, its :class:`~nvalchemi.dynamics.hooks.StabilityMetrics`
record, and :func:`~nvalchemi.dynamics.hooks.total_momentum` live in
:mod:`nvalchemi.dynamics.hooks`, because nothing about them is specific to
distillation; the evaluation suite re-exports them.

The endpoint drift and the fitted per-nanosecond rate both include whatever
transient the series starts with. A student seeded from frames that are not
equilibria of its own potential therefore needs a ``warmup_steps`` window that
covers the relaxation. Without one, the transient is reported as drift and can
cancel a genuine drift. Read ``energy_fluctuation_per_atom`` beside the rate: a
drift no larger than the fluctuation is a line through an oscillation rather
than a trend. Momentum is conserved only by an integrator that conserves it, so
set no ``max_momentum_drift`` bar under a stochastic thermostat.

The monitor's ``divergence`` predicate defaults to
:func:`~nvalchemi.dynamics.hooks.nonfinite_graph_mask`, the same predicate the
on-policy loop uses. The first firing that flags a graph ends the
series, and its step is recorded as ``first_divergence_step``. A trajectory
that blew up is therefore scored on the segment before it did, not on
non-finite samples. ``aggregate="mean"`` reports the figures as the mean over
graphs instead of the worst graph. Recording stops with a warning when the
batch composition changes, so a propagator that graduates systems mid-run is
scored on the segment before the first graduation.
``stop_on_composition_change=False`` keeps recording through an inflight refill
that preserves every graph's size.

*Extensivity* is the property that a structure's energy grows in proportion to
its size: a ``k``-fold supercell has ``k`` times the energy of the cell it
replicates.
:func:`~nvalchemi.training.distillation.evaluation.extensivity_error` checks
that the student's energy scales this way across replicated cells. The
*radial distribution* ``g(r)`` is the pair correlation function: how often
pairs of atoms sit at separation ``r``, normalized so an ideal gas gives ``1``.
:func:`~nvalchemi.training.distillation.evaluation.radial_distribution`
accumulates it over a trajectory, reading frames straight out of a
:class:`~nvalchemi.dynamics.sinks.DataSink` filled by
:class:`~nvalchemi.dynamics.hooks.SnapshotHook`.
:func:`~nvalchemi.training.distillation.evaluation.compare_radial_distributions`
scores the structure a trajectory samples against a reference trajectory's
structure with a bounded Jensen-Shannon divergence. By default the comparison
pools every species into one histogram. That histogram cannot see a student
that puts the right distances between the wrong kinds of atom. Pass ``pair``
to resolve one species pair and gate a chemically ordered system on its
partial curves. The extensivity and radial distribution checks read
periodicity from ``pbc`` when a batch carries it. Both therefore refuse a
cluster stored with a box but no periodic axis, like one without a cell.

.. autosummary::
   :toctree: generated
   :nosignatures:

   extensivity_error
   ExtensivityMetrics
   radial_distribution
   RadialDistribution
   compare_radial_distributions
   RDFComparison

:func:`~nvalchemi.dynamics.measure_throughput` times a propagator at steady
state. It runs a discarded warmup window, then a timed window with the device
synchronized at both ends. It reports atoms per second and simulated
nanoseconds per day, computed from the steps the propagator's own counter says
it took. It warns when a relaxer converged inside the timed window. The
measurement lives in :mod:`nvalchemi.dynamics`, since it times any propagator;
the evaluation suite re-exports it with its
:class:`~nvalchemi.dynamics.ThroughputMetrics` record. ``atoms_per_second``
scales with the batch it was measured on, so the column ranks a family of
students only when every student was timed on the same batch.
:func:`~nvalchemi.training.distillation.evaluation.build_acceptance_report`
rejects a family whose throughput measurements used different batches. The
monitor, its record, ``total_momentum``, and the throughput benchmark are
documented on the :doc:`dynamics hooks </modules/dynamics/hooks>` and
:doc:`dynamics API </modules/dynamics/api>` pages.

The verdict is assembled from those measurements. A caller collects one
:class:`~nvalchemi.training.distillation.evaluation.StudentEvaluation` per
candidate student and sets the acceptance bars on an
:class:`~nvalchemi.training.distillation.evaluation.AcceptanceThresholds`.
:func:`~nvalchemi.training.distillation.evaluation.build_acceptance_report`
then returns a report that renders as Rich tables and exports as a plain
dictionary or a flat scalar map. A bar with no measurement behind it fails the
student rather than being skipped. A check whose family was measured but whose
own number was not says which quantity or timestep was missing. A measurement
that is not finite fails its bar with a ``not finite`` detail, because a NaN
would fail every comparison and read as an ordinary miss, ``-inf`` would clear
every maximum, and ``+inf`` every minimum. Such a measurement is also left off
the speed-versus-accuracy Pareto front.

The from-scratch bar, ``max_from_scratch_ratio``, compares the distilled
student against an equal-size student trained from scratch. It takes the ratio
of the distilled error to the from-scratch error on ``energy_per_atom_mae``,
``forces_mae``, and ``stress_mae``, whichever both carry, and keeps the worst
ratio. That ratio has to be at most the bar.
Both students must be scored on the same held-out set, which the bar checks. A
family scored on different held-out sets is rejected outright.

Each measurement a ``StudentEvaluation`` holds, and the evaluation itself, is a
*measurement record*: a frozen result, built on
:class:`~nvalchemi.training.distillation.evaluation.MeasurementRecord`, that
exports with ``to_dict`` and rebuilds from its own export with ``from_dict``. A
sweep that evaluates each student in its own job can therefore persist the
results and assemble one report at the end. A student entry taken out of a
report export also rebuilds, with its verdict dropped. A caller that runs only
part of the suite asks
:func:`~nvalchemi.training.distillation.evaluation.measured_bars` which bars
its measurements can decide. The answer depends on the families the caller
filled and, for accuracy, on the quantities the pass compared.

The bars themselves are a public table.
:data:`~nvalchemi.training.distillation.evaluation.DEFAULT_BARS` is the tuple of
:class:`~nvalchemi.training.distillation.evaluation.AcceptanceBar` entries the
report applies by default. Each entry names the threshold it is set under, the
families it reads, and the field it gates.
:data:`~nvalchemi.training.distillation.evaluation.BAR_FAMILIES` is derived from
it. A measurement outside the typed slots is filed under
``StudentEvaluation.extra`` as a flat map of numbers per family. A bar whose
family is ``"extra:<family>"`` gates it, and the bar's limit is set under the
bar's name in ``AcceptanceThresholds.extra``. Pass the extended table as
``build_acceptance_report(..., bars=...)`` and ``measured_bars(..., bars=...)``.
A limit set for a bar that the table does not carry is refused rather than
skipped.

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_acceptance_report
   AcceptanceReport
   AcceptanceThresholds
   AcceptanceCheck
   StudentEvaluation
   StudentVerdict
   MeasurementRecord
   measured_bars
   MetricFamily
   AcceptanceBar

.. data:: DEFAULT_BARS
   :type: tuple[AcceptanceBar, ...]

   Built-in bars of :class:`AcceptanceThresholds`, in the order they are
   applied.

.. data:: BAR_FAMILIES
   :type: Mapping[str, frozenset[str]]

   Measurement families each built-in bar reads, keyed by threshold field.

.. currentmodule:: nvalchemi.training.distillation
