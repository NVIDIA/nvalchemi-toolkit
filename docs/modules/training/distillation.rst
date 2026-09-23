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


On-policy generation
--------------------

On-policy distillation trains on frames the student itself generated.
:class:`~nvalchemi.training.distillation.OnPolicyConfig` describes one segment
loop: which propagator generates, how many steps a segment runs, how often the
teacher labels, and how much of each training batch is replayed. The propagator
is any :class:`~nvalchemi.dynamics.base.BaseDynamics`, so relaxation optimizers
generate paths exactly as integrators generate trajectories. Its scalar half is
:class:`~nvalchemi.training.distillation.OnPolicySettings`, which validates on its
own so a recipe's settings can be checked before a teacher is built, and its
initial structures are any
:class:`~nvalchemi.training.distillation.InitialStructuresSource` — the
members the loop reads, with
:class:`~nvalchemi.training.distillation.InitialStructures` as the reference
implementation: a cursor over the rows one rank owns, shared by the initial
batch and a restart. A bare dataset is wrapped in one; an object that is
neither is refused naming the protocol. Structures are served
by :meth:`~nvalchemi.training.distillation.InitialStructures.draw`, which admits
each candidate through one :class:`~nvalchemi.training.distillation.FitPolicy`
predicate over the running atom and edge totals —
:class:`~nvalchemi.training.distillation.WithinBudget` bounds them — and either
stops at the first miss, which packs an initial batch, or skips it, which lets a
backfill fill the room a graduation freed.

Construction probes one row of the initial structures twice over. The row is
checked for every field the propagator updates in place from its first step,
and the propagator's ``compute()`` then runs once on it under the scorer's
isolation — evaluation mode restored, ``requires_grad`` flags restored, the
propagator's last outputs put back — so declarations that have drifted from the
implementation are refused where the config is built rather than on the first
step of a long run: a ``__needs_keys__`` output the student never produces, or
a field ``compute()`` reads that nothing declared, each named in the error. The
cost is one student forward, front-loading the kernel and CUDA initialization
the first step pays anyway. A graph model is probed with the neighbor list its
``neighbor_config`` declares, built on the row and rolled back, so no hook is
needed for the probe; a model planning more than one neighbor-list source is
not probed, because that builder makes exactly one list and the check must not
refuse a propagator the loop can run, and a warning says so. ``probe=False``
skips the forward altogether, for a propagator whose ``compute()`` must not run
outside the loop or a recipe check that should not pay for one.

.. autosummary::
   :toctree: generated
   :nosignatures:

   OnPolicyConfig
   OnPolicySettings
   ResizableSink
   InitialStructuresSource
   InitialStructures
   FitPolicy
   WithinBudget

Three settings deserve a sizing note. ``label_frequency`` is the throughput
setting, since the teacher is the expensive model, and it is counted against the
propagator's cumulative ``step_count``, so the cadence does not restart at a
segment boundary. Each segment also labels the frame it ends on, the most
on-policy one it produced; the cadence fires on the pre-increment step count and
the forced frame is one step later, so the cadence dispatch landing right after
a labeled step is passed over rather than paid for twice, and
``generation_steps`` a multiple of ``label_frequency`` labels each trajectory
exactly once per segment. ``replay_capacity`` is spent by FIFO eviction on whole
frames in arrival order, and a segment contributes one frame per trajectory per
labeled step, so a capacity that is not a multiple of the trajectory count cuts
a segment mid-step and over-represents the back of the batch in every mixture
drawn afterwards; size it as a multiple. ``seed`` keys every segment's mixture
sampler, added to the segment index, so consecutive seeds overlap by a shift of
one segment and replicate runs draw independently only with seeds at least
``num_steps // training_steps_per_segment`` apart. ``weight_sync_frequency`` is
reserved at ``1``: the propagator shares the student module, so an eager run is
never out of sync.

:class:`~nvalchemi.training.distillation.TeacherLabelHook` is the inline
labeling route: an ``AFTER_STEP`` dynamics hook that attaches ``teacher_*``
fields to the frame the propagator just resolved, at the level each signal
declares, and optionally mirrors a stripped copy of it into a
:class:`~nvalchemi.dynamics.sinks.DataSink`. It never touches the ``energy``
and ``forces`` the student wrote on the live batch, which drive the next step,
but it does strip them from the copy, along with the neighbor tensors and the
dynamics bookkeeping, so a stored frame is a training sample rather than a
propagator state and carries no self-label under a reference target's name. Do
not confuse it with the strategy's own private ``BEFORE_FORWARD`` labeling
seam, which labels batches on their way into a *training* step. Labeling is
idempotent per propagator step: a scorer publishing ``label_fields``, or one
whose signal names are all built-in, is skipped on a re-dispatch of the step it
already labeled, and a scorer publishing neither is skipped from its second
dispatch on, once the first pass has revealed what it writes. A forced label is
never passed over, which keeps an early-exiting segment and a run's final frame
intact.

The segment loop registers this hook itself and stages each segment's frames
in a sink it drains into the replay buffer at the boundary: host memory by
default, or the :class:`~nvalchemi.dynamics.sinks.DataSink` passed as
``OnPolicyConfig.capture_sink`` — a
:class:`~nvalchemi.dynamics.sinks.GPUBuffer` keeps the staging on the
generation device instead of paying a device-to-host copy per labeled frame.
The loop owns the sizing: a segment captures at most one frame per trajectory
per labeled step, the forced last frame included, so the sink has to hold
``(generation_steps + 1)`` frames per trajectory of the batch being propagated;
a configured sink with less capacity is resized through ``resize(capacity)``
when it satisfies :class:`ResizableSink` and refused otherwise, and one still
holding frames when a
segment starts is refused rather than drained as generated data. Like
``dynamics`` and ``teacher_scorer`` it is runtime-only.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherLabelHook
   nonfinite_divergence

Generated frames land in a
:class:`~nvalchemi.training.distillation.ReplayBuffer`, an in-memory dataset
behind a frozen key schema — appending a batch keeps only the keys both sides
hold, so one unlabeled frame would strip ``teacher_*`` from everything already
stored. :func:`~nvalchemi.training.distillation.build_mixed_loader` then draws
each training batch with an exact reference/replay composition, resolved to
whole samples of the batch size, and must be rebuilt after every segment
because the batch sampler reads the child dataset lengths once, at
construction. The two sources have to agree on their whole batch schema,
compared on a probe batch drawn from each side rather than on the field names a
Zarr-backed store and an in-memory buffer report differently: collation drops a
field only one side holds, zero-fills a whole level only one side holds, and
casts the second part of a mixed batch to the dtype the first carries while
which source leads a chunk is not fixed, so all three differences are rejected.
The reference dataset therefore has to be teacher-labeled, in the replay-frame
shape — structure, propagator state, ``teacher_*`` labels — and one carrying
reference ``energy`` or ``forces`` of its own is rejected rather than mixed
into batches that silently lose or fabricate them. Supervising one batch from
teacher labels and reference labels at once is masked-composition work that
comes later.

The schema freeze and the mixture are framework-owned; what enters the buffer
and what leaves it are policy. An
:class:`~nvalchemi.training.distillation.AdmissionPolicy` is a predicate over
the incoming frames — one boolean per graph — applied before the schema check,
so the NaN-labeled frames of a diverged trajectory, or frames failing a size or
diversity gate, never enter; an
:class:`~nvalchemi.training.distillation.EvictionPolicy` is asked, once the
admitted frames are appended, for the indices to drop out of the resident
batch — oldest first, the admitted frames last — given the capacity, and has to
name at least as many as the buffer is over by.
:class:`~nvalchemi.training.distillation.FIFO` is the reference eviction and
the policy the string ``"fifo"`` — the only spelling ``ReplayEviction`` admits,
and the one a recipe carries — builds. The segment loop wires
``OnPolicyConfig.replay_admission`` and ``replay_eviction`` into the buffer it
owns; a policy instance on either is runtime-only, and the declarative
``settings`` record a custom eviction as ``"fifo"`` with a warning.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReplayBuffer
   AdmissionPolicy
   EvictionPolicy
   FIFO
   ReplayEviction
   build_mixed_loader

Setting ``on_policy`` on the strategy is what turns those pieces into a run.
:meth:`~nvalchemi.training.distillation.DistillationStrategy.run` then takes no
dataloader: it seeds a state batch from ``initial_structures`` and repeats
generate-label-train segments until ``num_steps`` optimizer steps are done,
drawing the ``1 - replay_ratio`` share of every batch from
``reference_dataset``, which is required unless the ratio is ``1`` and refused
when it is, because a ratio of ``1`` would leave the reference dataset policed
but never sampled. The initial batch is restamped with fresh dynamics
bookkeeping on the way in, so structures loaded from a store an earlier
relaxation graduated do not arrive frozen at ``exit_status``, and the reference
dataset is probed once at construction for the fields the labeling hook strips,
for the device it emits on, and for the teacher fields the propagator's scorer
declares — each a guaranteed mixture failure that would otherwise surface only
after a whole generation segment had been paid for. One segment is one epoch, so
``AFTER_EPOCH`` and epoch-cadence validation land at segment boundaries while
step-cadence validation fires inside them, and the run's closing validation is
skipped when a cadence already validated at the final step. The segment is also
the restart granularity: a checkpoint taken mid-segment, or an offline run
graduating from a partial epoch, resumes by counting that segment as finished
rather than replaying the batches it had left. A second call to ``run()`` on one
strategy keeps the replay buffer the first filled and reseeds only the
trajectory: installing the rank shard reopens the cursor at the front of its
rows, so a rerun generates from the same structures again rather than from
whatever remainder the first call left.

The loop is single-process for now: nothing shards its loader or its structure
cursor, so it refuses to start on more than one rank rather than have every rank
regenerate and retrain the same frames, while offline distillation over a
labeled store distributes through ``DDPHook`` as usual. Generated frames are
drained to host memory and staged on the reference dataset's own device, so a
GPU-resident reference dataset and the buffer collate on one device;
``replay_device`` overrides that and is checked against the reference dataset at
construction. That device is the one the reference dataset actually emits on,
read off a batch whenever no declaration settles it — a
:class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` declares none, and
a store opened without a device declares an index-less ``cuda`` that names
whichever device is current. The student is held in evaluation mode to generate
and flipped to training mode for the training phase only, so generated frames
cost no second-order graph and no moving batch-norm statistics; a propagator
model that merely *composes* the student is held in evaluation mode for the
whole loop and moved whole to the generation device, because the training phase
forwards ``models["student"]`` rather than the composition and only the named
models travel with the strategy. The propagator must hold the very module
registered as ``models["student"]``, on its own or composed into a larger model
— that object identity is what makes each segment generate from the weights the
previous one trained, and it is checked at construction. Chunking the built-in
propagators across segments is exact: ``run`` never resets ``step_count`` or the
integrator state, and the Langevin thermostat draws from a counter-based
generator keyed on the cumulative step count, so two segments of ``K`` steps
reproduce one run of ``2K``. An open/close-sensitive dynamics hook such as
:class:`~nvalchemi.dynamics.hooks.LoggingHook` is re-entered once per segment,
a chunk that converges out early is read from ``dynamics.step_count`` rather
than assumed to be ``generation_steps``, and a
:class:`~nvalchemi.dynamics.FusedStage` pays its priming forward pass once per
segment, so prefer a bare propagator.

A custom ``teacher_*`` field the propagator's scorer writes is an ordinary loss
target, exactly as offline: generation writes it onto every captured frame, so
``reference_dataset`` has to carry it too — the generation/reference parity
check enforces that whenever the scorer declares ``label_fields`` — and
validation data has to arrive with it, because the strategy's own scorer
produces built-in signals only and cannot backfill it. At least one built-in
``teacher_*`` target, or an explicit ``teacher_signals``, is still required
alongside it. A scorer declaring no ``label_fields`` and no built-in signals
writes fields nothing can know before it has scored a batch, so the strategy
warns that the parity check is deferred to the first segment's loader.

Because ``on_policy`` and ``reference_dataset`` hold live runtime objects, they
are left out of
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`,
which warns, and a strategy rebuilt from that spec runs offline until they are
supplied again.

Relaxation
----------

A relaxation propagator generates paths that *end*, and ``fmax`` is what
teaches the segment loop about that. It is the max-force-norm threshold a
recipe can hold, with ``convergence_hook`` taking a
:class:`~nvalchemi.dynamics.base.ConvergenceHook` the run needs whole;
``convergence_criterion`` resolves the two, and the loop puts that one criterion
on the propagator as both the status-migrating hook and the convergence detector
for the duration of the run, so graduation and detection cannot disagree; a
detector the propagator was built with is put aside and restored afterwards. A
hook passed whole must migrate status, off the status ``0`` the run stamps its
structures with, on every step: a criterion that merely reports convergence
would look configured while freezing and graduating nothing, and one that skips
steps would let both capture routes store the frame it graduates late. The
lifecycle also has to be the only thing migrating status, so a propagator that
already carries a status-migrating ``ConvergenceHook`` of its own, or a sampler
of its own, is refused rather than run at two thresholds or refilled
mid-segment, and a multi-sub-stage :class:`~nvalchemi.dynamics.FusedStage` —
whose sub-stages each carry a migrator the stage built itself — is refused at
construction, where that shape is fixed. The construction probe that runs the
propagator's ``compute()`` on one row dispatches a copy of the criterion to
that row too, stamped with the ``status`` the run gives its structures: a
criterion that raises on the propagator's outputs, or whose firing leaves the
status column unmoved where it converged, is refused before a run is paid for.
Whether a structure converges is data; that the mechanism works is not. A
criterion reading a key no ``compute()`` produces is not dispatched — a hook may
write it during the step — and a warning names the key instead.

What the lifecycle buys is a buffer that keeps filling with informative frames.
A converged structure freezes in the propagator's step, is stored once as the
minimum it reached, and is left out of every later capture of the segment
instead of being written again on each one; at the segment boundary it
graduates out of the batch, with the optimizer's own per-structure state
following the membership change, and the initial structures are drawn for the
room it freed — as many structures as graduated, within the atoms they held —
through :meth:`~nvalchemi.training.distillation.InitialStructures.draw` with
``on_miss="skip"``, so one oversized row never starves the refills behind it. A
budgeted :class:`~nvalchemi.training.distillation.InitialStructures` packs the
initial batch and leaves the remainder in cursor order for that backfill; an
unbudgeted one is propagated whole, so its cursor opens past the last row and
the batch narrows by one trajectory per graduation unless ``recycle`` restarts
the cursor at the front of the rows this rank owns. A backfilled structure is
restamped with fresh bookkeeping, keeping only the ``system_id`` the source
numbered, so a store of minima an earlier relaxation graduated does not arrive
frozen. A trajectory can also end by diverging: no criterion ever accepts a NaN,
so a graph the ``OnPolicyConfig.divergence`` predicate flags — by default
:func:`~nvalchemi.training.distillation.nonfinite_divergence`, one whose
positions or forces stop being finite — is frozen at ``exit_status`` on that
step, kept out of both capture routes, and retired and backfilled at the
boundary like a converged one, with one warning per boundary counting them. A
custom predicate takes the live frame and returns one boolean per graph, the
shape an :class:`~nvalchemi.training.distillation.AdmissionPolicy` has; it is
runtime-only, and one returning any other shape is refused on its first
dispatch. When the last trajectory finishes and nothing is left to start
one, the loop warns once and trains its remaining steps on the frames it has.

Frames reach the buffer by two routes that partition them:
:class:`~nvalchemi.training.distillation.TeacherLabelHook`, given the
propagator's ``exit_status`` by the lifecycle, stores the structures still
relaxing, labeled inline and narrowed to those before the teacher runs rather
than after, so a mostly-frozen batch costs a mostly-frozen teacher pass — a run
without a lifecycle leaves the hook unnarrowed, so a propagator managing its
own convergence keeps its final frames;
and a converged-frame hook stores each minimum once, captured raw off the status
transition — which every propagator publishes, including a
:class:`~nvalchemi.dynamics.FusedStage`, whose own ``ON_CONVERGE`` fires on its
sub-stages alone — and labeled in a single teacher pass as its sink is drained,
which keeps the teacher's batch size independent of the propagated one. A fused
sub-stage that graduates on an ``n_steps`` budget migrates after the step's
hook dispatch, so the loop captures those frames once the chunk returns. The
path route stages its frames in ``OnPolicyConfig.capture_sink`` when one is
configured, re-sized per segment to ``(generation_steps + 1)`` frames per
trajectory still in the batch — through ``resize(capacity)`` when the sink
offers one, and refused up front when a smaller sink does not, though a sink
that fits the initial batch fits every later one, since a backfill never grows
the batch past it; the converged route keeps a host-memory sink of its own,
one frame per graph. A custom
:class:`~nvalchemi.training.distillation.InitialStructuresSource` drives the
lifecycle too, provided its ``initial_batch`` stamps the ``status`` zeros and
``system_id`` numbers the lifecycle graduates and backfills on.
Distribution-matching objectives are defined on equilibrium ensembles, which a
relaxation path is not; pointwise energy, force, and atomic-energy matching
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
