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

Checkpoints serialize every entry of ``models``, so each write duplicates the
frozen teacher's weights; size the checkpoint interval accordingly with a large
teacher.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationStrategy
   default_distillation_fn


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

The loop is single-process. The rank shard is installed on the structure
sampler, but each rank would build its segment loader from its own replay
buffer and an unsharded reference dataset, so the loop refuses to start on more
than one rank rather than train ranks on generated frames no rank shares and on
the same reference samples. Offline distillation over a labeled store
distributes through ``DDPHook`` as usual.

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

Because ``on_policy`` and ``reference_dataset`` hold live runtime objects,
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
leaves them out and warns. A strategy rebuilt from that spec runs offline until
they are supplied again.

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
relaxation path is not. Pointwise energy, force, and atomic-energy matching
distill a relaxation path exactly as they distill a trajectory.

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
