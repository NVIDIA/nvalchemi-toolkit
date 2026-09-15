.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-distillation-api:

Distillation API
================

Teacher scoring, offline dataset labeling, the offline distillation strategy
and loss terms, and the on-policy generation components for
knowledge-distillation workflows.

.. seealso::

   - **User guide**: :ref:`distillation_guide`
   - **Training strategy API**: :ref:`training-strategy-api`
   - **Fine-tuning API**: :ref:`training-finetuning-api`
   - **Loss API**: :ref:`losses-api`


Scoring
-------

A scorer turns a :class:`~nvalchemi.data.Batch` into named teacher signals —
``energy``, ``forces``, ``stress``, ``node_energies``, ``embeddings``, and
``hessian`` — each mapped to a batch field, a level, and a canonical shape.
:class:`~nvalchemi.training.distillation.InProcessTeacherScorer` evaluates a
teacher loaded in the current process and leaves the scored batch exactly as it
found it, including neighbor tensors.

.. currentmodule:: nvalchemi.training.distillation

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherScorer
   InProcessTeacherScorer
   signal_fields
   scorer_fields
   signal_for_field

Scorers speak two public type aliases: ``SignalLevel``, the ``"node"`` or
``"system"`` level a signal is attached at, and ``TeacherLabels``, the
``{batch field: (detached tensor, level)}`` mapping
:meth:`~nvalchemi.training.distillation.TeacherScorer.label` returns; the
signal names themselves are published as
:data:`~nvalchemi.training.distillation.SUPPORTED_SIGNALS`. A custom scorer may
publish ``label_fields``, the batch fields its ``label()`` populates, which
consumers resolve through
:func:`~nvalchemi.training.distillation.scorer_fields` rather than reading the
attribute.

Signals differ in cost. All the forward-pass ones share a single teacher pass;
``embeddings`` adds a second, because the model contract computes
representations in their own method rather than returning them from the forward
pass; and ``hessian`` adds an energy-only pass plus the two backward passes
:func:`~nvalchemi.training.distillation.hessian_vector_product` takes through
it. The ``hessian`` signal is the only one that writes two fields —
``teacher_hvp`` and the ``teacher_hvp_probe`` direction it was taken along,
which the student has to be differentiated along too for the two to be
comparable, so it is stored and travels with the label.
:meth:`~nvalchemi.training.distillation.InProcessTeacherScorer.label_hvp`
computes one product for a probe the caller chose.

.. autosummary::
   :toctree: generated
   :nosignatures:

   hessian_vector_product


Labeling
--------

Offline labeling walks a dataset once, scores it, and writes the source fields
plus the teacher fields to a Zarr store that the ordinary reader and dataset
path consume. Runs are resumable, and the neighbor tensors are dropped because
a stored list records no cutoff for a consumer to check; ``keep_neighbors=True``
keeps a sparse one. Every chunk must write the schema the store
holds, and a store whose arrays disagree about how many samples it contains —
what an interrupted run leaves behind — is reported rather than resumed from a
misaligned offset.

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
rather than on its first batch.

A ``validation_config`` carrying its own ``loss_fn`` takes part in both checks:
its ``teacher_*`` targets widen the derived signal set, and its prediction keys
are checked the same way whenever the effective validation function
(``validation_fn`` falling back to ``training_fn``) is the stock one. Neither
re-runs on assignment, so pass ``validation_config`` to the constructor or name
the wider set in ``teacher_signals``. Every resolved signal — derived or
explicit — is a request for its fields on every batch: a batch counts as
labeled only when it carries every resolved field, so adding a validation loss
with a new ``teacher_*`` target puts a training store written before it back on
the teacher, batch after batch, at identical values.

Training and validation batches go through one labeling seam: an internal hook
on ``BEFORE_FORWARD``, a stage both loops dispatch on the device-placed batch.
The teacher runs there with autocast disabled, so mixed-precision training does
not change the targets and an on-the-fly label matches the offline one exactly
wherever the store returns the label dtype: a store round-trips every floating
field to the dataset's ``positions`` dtype, so over the usual float32 dataset
every student but a float64 one agrees on both paths, while a float64 student
reads float32 back and needs a ``dtype_policy``. Labels are never cast below single
precision, so a ``bfloat16`` or ``float16`` student gets float32 labels and
needs ``dtype_policy="prediction_to_target"`` on its loss terms. Pointing
``validation_config`` at a store written by
:func:`~nvalchemi.training.distillation.label_dataset` still avoids the teacher
pass entirely, and validating an EMA-averaged student against the live teacher
is ``ValidationConfig(use_ema="auto")``, reported as ``model_source="mixed"``;
``use_ema="always"`` currently also demands an inference-slot entry for the
frozen teacher and fails at the first validation pass without one.

Checkpoints store the frozen teacher *once per checkpoint root* rather than at
every index, so a periodic write costs the student's weights rather than the
student's plus the teacher's. One root holds one copy: saving a different
teacher into a root that already holds one raises rather than repointing the
checkpoints already written there at weights they were not written against.
See
:meth:`~nvalchemi.training.distillation.DistillationStrategy.checkpoint_model_references`
and :ref:`distillation_recipes_guide` for how the stored copy is referenced,
fingerprinted, and read back on a restart.

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationStrategy
   default_distillation_fn

Two objectives need a prediction the student's forward pass does not return, and
each ships the training function that produces it. Both are module-level
functions, so a recipe using one still survives
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`, and
both are additive: they return the stock ``predicted_*`` outputs plus one key.
:func:`~nvalchemi.training.distillation.embedding_distillation_fn` runs the
student's ``compute_embeddings`` and routes the result through the
``"projector"`` model when one is registered;
:func:`~nvalchemi.training.distillation.hessian_distillation_fn` differentiates
the student's energy twice along the labeled probe. A recipe wanting both writes
one module-level function of its own — calling both costs the student forward
pass twice, which building the union out of
:func:`~nvalchemi.training.distillation.hessian_vector_product` and the
student's ``compute_embeddings`` avoids.

.. autosummary::
   :toctree: generated
   :nosignatures:

   embedding_distillation_fn
   hessian_distillation_fn


On-policy generation
--------------------

On-policy distillation trains on frames the student itself generated.
:class:`~nvalchemi.training.distillation.OnPolicyConfig` describes one segment
loop: which propagator generates, how many steps a segment runs, how often the
teacher labels, and how much of each training batch is replayed. The propagator
is any :class:`~nvalchemi.dynamics.base.BaseDynamics`, so relaxation optimizers
generate paths exactly as integrators generate trajectories. Its scalar half is
:class:`~nvalchemi.training.distillation.OnPolicyKnobs`, which validates on its
own so a recipe's knobs can be checked before a teacher is built, and its seed
structures live behind a
:class:`~nvalchemi.training.distillation.SeedSource` — one cursor over the rows
one rank owns, shared by the initial batch, the backfill, and a restart.

.. autosummary::
   :toctree: generated
   :nosignatures:

   OnPolicyConfig
   OnPolicyKnobs
   SeedSource

:class:`~nvalchemi.training.distillation.TeacherLabelHook` is the inline
labeling route: an ``AFTER_STEP`` dynamics hook that attaches ``teacher_*``
fields to the frame the propagator just resolved, at the level each signal
declares, and optionally mirrors a stripped copy of it into a
:class:`~nvalchemi.dynamics.sinks.DataSink`. It never touches the ``energy``
and ``forces`` the student wrote on the live batch, which drive the next step —
but it does strip them from the copy, along with the neighbor tensors and the
dynamics bookkeeping, so a stored frame is a training sample rather than a
propagator state and carries no self-label under a reference target's name. Do
not confuse it with the strategy's own private ``BEFORE_FORWARD`` labeling
seam, which labels batches on their way into a *training* step. Labeling is
idempotent per propagator step: a scorer publishing ``label_fields``, or one
whose signal names are all built-in, is skipped on a re-dispatch of the step it
already labeled, and a scorer publishing neither is skipped from its second
dispatch on, once the first pass has revealed what it writes. A segment also
forces a label on the frame it ends on, whatever the cadence; because the
registry fires on the pre-increment step count, that forced frame and the next
segment's first cadence dispatch would otherwise be one propagator step apart,
so a cadence dispatch landing immediately after a labeled step is passed over.
A forced label never is, which keeps an early-exiting segment and a run's final
frame intact.

.. autosummary::
   :toctree: generated
   :nosignatures:

   TeacherLabelHook

Generated frames land in a
:class:`~nvalchemi.training.distillation.ReplayBuffer`, an in-memory dataset
behind a frozen key schema — appending a batch keeps only the keys both sides
hold, so one unlabeled frame would strip ``teacher_*`` from everything already
stored. :func:`~nvalchemi.training.distillation.build_mixed_loader` then draws
each training batch with an exact reference/replay composition, resolved to
whole samples of the batch size, and must be rebuilt after every segment
because the batch sampler reads the child dataset lengths once, at
construction. What the two sources have to agree on is their whole batch
schema, compared on a probe batch drawn from each side rather than on the field
names a Zarr-backed store and an in-memory buffer report differently. Collation
drops a field only one side holds, zero-fills a whole level only one side
holds, and casts the second part of a mixed batch to the dtype the first
carries while which source leads a chunk is not fixed, so all three differences
are rejected — a field the two sides hold at different dtypes among them: the
anchor has to be a teacher-labeled
dataset in the replay-frame shape — structure, propagator state, ``teacher_*``
labels — and one carrying reference ``energy`` or ``forces`` of its own is
rejected rather than mixed into batches that silently lose or fabricate them.
Supervising one batch from teacher labels and reference labels at once is
masked-composition work that comes later. ``ReplayEviction`` names the policy
retiring frames from a full buffer.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ReplayBuffer
   build_mixed_loader

Setting ``on_policy`` on the strategy is what turns those pieces into a run.
:meth:`~nvalchemi.training.distillation.DistillationStrategy.run` then takes no
dataloader: it seeds a state batch from ``seeds``, the
:class:`~nvalchemi.training.distillation.SeedSource` whose cursor the backfill
and a restart go on reading from, and repeats generate-label-train segments
until ``num_steps`` optimizer steps are done, drawing the ``1 - replay_ratio``
share of every batch from ``reference_dataset``, which is required unless the
ratio is ``1`` and refused when it is, because a ratio of ``1`` draws whole
batches from the buffer and would leave the anchor policed but never sampled.
The seed batch is restamped with fresh dynamics bookkeeping by the source on the
way in, so seeds loaded from a store an earlier relaxation graduated do not
arrive frozen at ``exit_status``, and the anchor is probed once at construction
for the fields the labeling hook strips — a guaranteed mixture failure that
would otherwise surface only after a whole generation segment had been paid for.
One segment is one epoch, so ``AFTER_EPOCH`` and epoch-cadence validation land
at segment boundaries while step-cadence validation fires inside them, and the
run's closing validation is skipped when a cadence already validated at the
final step. The segment is also the restart granularity: a checkpoint taken
mid-segment, or an offline run graduating from a partial epoch, resumes by
counting that segment as finished rather than replaying the batches it had left.
A second call to ``run()`` on one strategy keeps the replay buffer the first
filled and reseeds only the trajectory: installing the rank shard reopens the
source at the front of its rows, so a rerun generates from the same seeds again
rather than from whatever remainder the first call left.
``OnPolicyConfig.seed`` keys the mixture sampler, which is how replicate runs
are made to draw independently.
Generated frames are drained to host memory and staged on the reference
dataset's own device, so a
GPU-resident anchor and the buffer collate on one device; ``replay_device``
overrides that and is checked against the anchor at construction. That device is
the one the anchor actually emits on, read off a batch whenever no declaration
settles it — a :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset`
declares none, and a store opened without a device declares an index-less
``cuda`` that names whichever device is current. The student is
held in evaluation mode to generate and flipped to training mode for the
training phase only, so generated frames cost no second-order graph and no
moving batch-norm statistics; a propagator model that merely *composes* the
student is held in evaluation mode for the whole loop instead, because the
training phase forwards ``models["student"]`` rather than the composition. The
propagator must hold the very module registered as
``models["student"]``, on its own or composed into a larger model — that object
identity is what makes each segment generate from the weights the previous one
trained, and it is checked at construction.

On-policy runs also relax the reserved ``teacher_`` namespace in exactly one
way. A loss target under that prefix normally has to name a built-in signal,
but a propagator scorer that declares the field in ``label_fields`` *supplies*
it: the labeling hook writes it onto every captured frame, ``reference_dataset``
has to carry it too — the generation/anchor parity check is what enforces
that — and validation data has to arrive with it, because the strategy's own
scorer produces built-in signals only and cannot backfill it. At least one
built-in ``teacher_*`` target, or an explicit ``teacher_signals``, is still
required alongside it. A scorer that declares no ``label_fields`` and no
built-in signals supplies nothing: its fields are unknowable until it has
scored a batch, so the strategy warns that it cannot check the anchor parity
yet and refuses a custom target read against it.


``on_policy`` and ``reference_dataset`` serialize as references rather than as
the objects themselves, so
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
carries the whole recipe and a rebuild needs only its models supplied back; a
run whose datasets live in memory, or whose propagator hides its constructor
arguments, leaves the recipe out with a warning naming the piece instead.
Either way the live objects are a keyword argument on every rebuild entry
point:
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_spec_dict`,
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_checkpoint_dict`,
and
:meth:`~nvalchemi.training.distillation.DistillationStrategy.load_checkpoint`
all take ``on_policy`` and ``reference_dataset``, and a live object handed over
that way outranks any recipe the spec carries. The segment loop travels with the
student it propagates, so the ``models`` the propagator was built around go back
in alongside it and the checkpoint's weights are restored into those very
objects; restoring with
:meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint` into a strategy
that was constructed with the loop reaches the same place from the other end.
An objective defined only on generated batches — an ensemble term — makes this
mandatory rather than optional, since it refuses to rebuild offline-shaped at
all.

A relaxation propagator generates paths that *end*, and ``convergence`` is what
teaches the segment loop about that. It is the ``fmax`` threshold a recipe can
hold, with ``convergence_hook`` taking a
:class:`~nvalchemi.dynamics.base.ConvergenceHook` the run needs whole;
``convergence_criterion`` resolves the two, and that criterion is put on the
propagator as both the status-migrating hook and
the convergence detector for the duration of the run — one criterion deciding
when a structure is done, rather than a run whose graduation and detection
disagree — a criterion the propagator was built with is put aside for the run
and restored afterwards. A hook passed whole must already migrate status,
because a criterion that merely reports convergence would look configured while
freezing and graduating nothing; it must migrate off status ``0``, which is
what the run stamps its seeds with; and it must run on every step, because a
structure is captured on the step it converges and has to be frozen on that
same one. The lifecycle also has to be the only thing migrating status, so a
propagator that already carries a status-migrating ``ConvergenceHook`` of its
own is refused rather than run at two thresholds at once, and a multi-sub-stage
:class:`~nvalchemi.dynamics.FusedStage` — whose sub-stages each carry one the
stage built itself — is refused at construction, where that shape is fixed.

What the lifecycle buys is a buffer that keeps filling with informative frames.
A converged structure freezes in the propagator's step, is stored once as the
minimum it reached, and is left out of every later capture of the segment
instead of being written again on each one; at the segment boundary it
graduates out of the batch through
:meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check`, with the
optimizer's own per-structure state following the membership change. What takes
its slot depends on the seed source. An unbudgeted
:class:`~nvalchemi.training.distillation.SeedSource` seeds every row the rank
owns, so its cursor opens past the last structure and the batch simply narrows
by one trajectory per graduation unless ``recycle`` restarts it at the front of
those rows; a budgeted one packs the initial batch and leaves the remainder in
cursor order for the backfill, which is the way to keep a run's occupancy up
without re-relaxing a structure. Either way, when the last trajectory finishes
the loop warns once and trains its remaining steps on the frames it has.

Frames reach the buffer by two routes that partition them:
:class:`~nvalchemi.training.distillation.TeacherLabelHook` stores the
structures still relaxing, labeled inline and narrowed to those before the
teacher runs rather than after, so a mostly-frozen batch costs a mostly-frozen
teacher pass; and a converged-frame hook stores each minimum once, captured raw
off the status transition — which every propagator publishes, including a
:class:`~nvalchemi.dynamics.FusedStage`, whose own ``ON_CONVERGE`` fires
on its sub-stages alone — and labeled in a single teacher pass as its sink is
drained, which is what keeps the teacher's batch size independent of the
propagated one. Nothing is stored twice, and seed structures are checked at
construction against the fields the propagator opens its step with — ``forces``,
``velocities``, and ``atomic_masses`` for FIRE, plus ``stress`` and ``cell``
for a variable-cell one — named from its own ``__needs_keys__`` and
``__provides_keys__`` rather than surfacing from inside a kernel.

Distribution-matching and path objectives are defined on equilibrium ensembles,
which a relaxation path is not; they are refused at construction for
relaxation-only generation. Pointwise energy, force, and per-atom energy
matching distill a relaxation path exactly as they distill a trajectory.


Scaling out: multi-GPU and multi-node
-------------------------------------

On-policy distillation scales as synchronous data parallelism, and the
placement follows from the loop's one asymmetry: the teacher is frozen and only
ever runs a forward pass, while the student is small and trains. So a teacher
that fits on one accelerator is *replicated* onto every rank rather than
sharded — sharding a frozen forward would only add collectives — and the
student is data-parallel across the ranks. Each rank then generates its own
trajectories, labels them with its own teacher replica, and fills its own
replay buffer; the only traffic between ranks is the student's gradient
all-reduce, which is small enough to tolerate a slower interconnect.

The script is the ordinary single-process one plus a
:class:`~nvalchemi.training.hooks.DDPHook`, launched one process per GPU:

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
           seeds=SeedSource(seed_store),
           replay_ratio=0.5,
           steps_per_segment=32,
       ),
   )
   strategy.run()

.. code-block:: bash

   # One node, one process per GPU.
   torchrun --standalone --nproc_per_node=8 distill.py

   # Four nodes, run on each of them.
   torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d \
       --rdzv_id=distill --rdzv_endpoint=$HOST:29500 distill.py

``DDPHook`` wraps every optimizer-configured model, which is the student and
any auxiliary head but never the teacher, and pins each rank to its node-local
device. What the segment loop adds on top is the sharding the generation phase
needs. ``seeds`` is dealt out strided, rank ``r`` taking every
``world_size``-th structure, so it must hold at least one structure per rank and
is best sized as a whole multiple of the world. A seed set the world cannot
deal out evenly warns, because every rank draws the same number of replay
samples per batch from a buffer holding only its own trajectories and the
gradients are averaged rank by rank, so a frame generated on a shard one
structure shorter reaches the optimizer with more weight than one from a longer
shard. That deal balances the structure count rather than the work: it strides
by index and never reads how big a structure is, so a seed set whose sizes vary
with position — every other row a slab, say — can hand one rank many times
another's atom count. The generation phase then sizes to the heaviest shard
while the rest of the world waits for it at the segment's all-reduce, and that
is the rank that runs out of memory first. Sort the seed dataset by atom count
and the strided deal balances by construction. The rows a rank owns are public
as :attr:`~nvalchemi.training.distillation.DistillationStrategy.seed_shard`, and
they are the whole of what it may propagate: anything that refills or backfills
the trajectory batch draws from that tuple alone, counting what it has consumed,
where it wraps, and when it is exhausted against the shard rather than against
the dataset, since a structure served to a rank that does not own it is
propagated and billed to the teacher twice. A restart restores that cursor
separately from the next ``system_id``, which is stamped per rank from zero and
so names no row of the dataset. Both seeded streams the loop owns are moved
onto a per-rank stride of the seed space — the mixture sampler's
``OnPolicyConfig.seed`` and every integer seed the propagator exposes, its
sub-stages included, so a composed relax-then-sample propagator is separated as
a bare thermostat is. The accounting is per stage rather than per composition,
so a propagator mixing seeded and unseeded stages does not pass for moved on the
strength of one seed found somewhere in it: a stage exposing a
:class:`torch.Generator` and no integer seed is named in a warning, from every
rank including rank zero and before the first segment is generated. It stays on
the shared stream and needs a rank-distinct seed from the caller. That matters
most when the seed structures are replicas of one geometry — how a run asks for
one trajectory per rank — because sharding separates nothing there: an unmoved
stage makes every rank generate identical frames for as long as it owns the
batch, and the teacher is billed once per copy. Randomness the loop cannot see
at all — a differently named attribute, the global ``torch`` stream, a closure —
stays on the shared stream without a warning, because nothing tells it apart
from a deterministic stage.

The reference dataset is deliberately *not* sharded: every rank builds its
mixture over the whole anchor, and the sampler draws with replacement, so each
rank's mixture stays exact while its draws are independent rather than
disjoint. Ranks are expected to share anchor samples; only the generated frames
and the teacher passes paying for them are partitioned. The replay buffer
likewise stays rank-local and is not shared or gathered, but it is staged on
the anchor's device. Stage the anchor in host memory — a device-less
:class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`, or one
opened with ``device="cpu"`` — and leave ``replay_device`` unset so the buffer
follows it there. The mixture is then collated on the host and moved to each
rank's device by the training step. An accelerator-resident anchor is a matter
of *when* it is placed rather than whether. A store that emits lazily — the
``labeled_store`` above, a :class:`~nvalchemi.data.datapipes.dataset.Dataset`
opened over a labeled store with no ``device``, or with an index-less
``"cuda"`` — draws its first batch inside the first segment, once ``DDPHook``
has pinned the rank, and so puts each rank's anchor batches and its replay
buffer on that rank's own GPU with nothing reported. Pre-staging a batch
eagerly, before the pin, is what does not survive: ``.to("cuda:0")`` resolves
to GPU 0 in every process and concentrates the whole world's buffers and
mixture collation there, which every rank reports, while ``.to("cuda")`` moves
the tensors to whichever device is current but records the spelling rather than
that device, so once the launcher pins the rank the record resolves elsewhere
and the anchor cannot be drawn from at all — a parent-toolkit defect in how a
storage records an index-less device, tracked separately from this work. Moving
a host-memory anchor once the rank is pinned is the other shape that places
correctly: a ``TrainingStage.SETUP`` hook reassigning
``ctx.workflow.reference_dataset``'s batch to ``ctx.workflow.devices[0]``,
which is indexed by the time that stage runs. ``replay_device`` is read the
same way from the other end — set to an index-less ``"cuda"`` it names the
device this rank has made current, rather than a spelling every rank resolves
anew. A world staging on an indexed device some rank does not train on warns,
from every rank once the ranks have reduced the question between them: the rank
that owns that device is the one the world concentrates onto, and its own
placement cannot tell a shared anchor from a per-rank one. That warning catches
an explicitly indexed device only, and deliberately: an index-less one names
whichever device is current, which is what a rank-local anchor emitting after
the pin looks like. A multi-rank launch that leaves the student unwrapped is
refused rather than run, because nothing would keep the ranks' policies
together and the divergence compounds through the generation phase; the check
is that *something* owns ``models["student"]`` after setup, so a wrapper of
your own clears it as a ``DDPHook`` does.

Multi-node is the same code path with a larger world: nodes self-label, only
student gradients cross the interconnect, and sharding keys on the global rank
while device placement keys on the node-local one. The launch line above uses
the ``c10d`` rendezvous rather than the default static one, which is what lets
the identical command run on every node — the static backend assigns node ranks
from ``--node_rank``, which defaults to zero everywhere. Bookkeeping follows
the ordinary training conventions — validation runs on every rank and
all-reduces its metrics, so it must never be rank-gated, and
:class:`~nvalchemi.training.hooks.CheckpointHook` writes from global rank zero
only. Restarting resumes the optimizer state and the counters, and every rank
reseeds its trajectories from its own shard — no rank propagates the shard the
checkpoint was written from — and refills its replay buffer from scratch, since
the buffer is rank-local runtime state no checkpoint carries. Budget the first
segments after a restart as cold: their mixtures draw the replay half from that
segment's frames alone. A multi-rank restart also has to name this rank's
device in two places. Rank zero writes ``strategy.json`` after ``DDPHook`` has
collapsed ``devices`` to the one GPU it pinned, and that recorded device is the
load location every rank restores against, before its own hook has pinned
anything; ``run()`` then moves the parameters but reuses the resumed optimizer,
and ``Optimizer.load_state_dict`` re-homes the moments to the parameter without
ever moving Adam's ``step``. So construct the restarting strategy with
``devices=[torch.device(f"cuda:{local_rank}")]`` *and* pass
``map_location=f"cuda:{local_rank}"`` to
:meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint`; either alone
still strands a state tensor on the device the checkpoint was written from, and
that surfaces as a hang rather than a traceback — the rank raises inside the
optimizer and then blocks tearing the process group down while its peers wait
on the gradient all-reduce. A single-rank restart is unaffected: there is one
device, and it is the one recorded.

Two things to size deliberately. Every rank runs the same number of segments
and the same number of batches per segment, which is what keeps the ranks
arriving at each all-reduce together, so an update orchestrator that vetoes
optimizer steps unevenly across ranks would desynchronize them. A
desynchronized world does not fail fast: a rank that stalls or raises leaves
its peers blocked in the next all-reduce for the process group's default
timeout — thirty minutes on gloo, ten on the NCCL watchdog — and a rank that
raises while ``DDPHook`` owns the group blocks tearing it down as well, which
is the live-but-stalled job a mis-mapped restart also produces. ``DDPHook``
exposes no process-group timeout of its own, so bound that wait by initializing
the process group yourself with ``timeout=`` before the run: the hook finds
communication already established, leaves it alone, and never destroys it. And
the world *divides* the generation work rather than multiplying it: the seeds
are sharded, so a segment's aggregate frame count — and the teacher bill paying
for it — is whatever the single-process run produced, while each rank
contributes its ``1/world_size`` share. ``segment_steps``, ``label_frequency``,
and ``replay_capacity`` are all per rank, and the sizing consequence runs the
other way from the frame count: at a fixed ``replay_capacity`` each rank's
buffer now spans ``world_size`` times as many segments before FIFO eviction
reaches back, so every mixed batch grows staler as the world grows. One
correction is enough, and which one depends on what you hold fixed. Raise
``segment_steps`` or the seed count alongside the world and the per-rank yield
per segment is unchanged, which restores the history depth along with it. Leave
both fixed and it is ``replay_capacity`` that comes down by the world size
instead. Applying both corrections together is the mistake the arithmetic
invites: the buffer then spans ``1/world_size`` of the history the
single-process run had.


Recipes and the CLI
-------------------

A whole on-policy run survives
:meth:`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict` as
references: :meth:`~nvalchemi.training.distillation.OnPolicyConfig.to_spec_dict`
carries every scalar knob verbatim, the propagator as the ``cls_path`` and
keyword arguments it rebuilds from with the student rebound at build time, the
scorer as its signal set and cast dtype over the strategy model named
``"teacher"``, and ``seeds`` as the store it reads under the budgets it was
given — never its cursor, which is restart state; ``reference_dataset``
serializes the same way. A ``convergence_hook``, a propagator's hooks,
convergence hook, and sinks, and a dataset holding its samples in memory are
the runtime-only parts: the first two are omitted with a warning naming them
(on a hand-built propagator and on one a recipe built alike, the segment loop's
own labeling hook excepted), the third refuses with the fix in the message, and
a piece that cannot be described leaves the whole ``on_policy`` entry out rather
than writing a recipe that would rebuild into a different run.
:meth:`~nvalchemi.training.distillation.OnPolicyConfig.from_spec_dict` and
:meth:`~nvalchemi.training.distillation.DistillationStrategy.from_spec_dict`
rebuild around supplied models, and both take overrides for the runtime-only
pieces.

An interrupted on-policy run additionally carries its live trajectory batch,
the propagator's cumulative step count, its seed cursor, and its replay frames
through the checkpoint, so a resumed run continues the same trajectory instead
of seeding a fresh one and backfills from where the interrupted run left the
cursor; the restored frames replace the buffer's contents rather than being
merged into them, and the knobs the bundle records are compared against the
resumed loop's so a run whose halves differ says so. The bundle is rank-local,
because the strategy checkpoint it rides in is written on rank zero alone: it is
consumed only when a single rank wrote it and a single rank is restoring it, so
any multi-rank restart drops it with a warning and each rank reseeds with a cold
replay buffer. It resumes at a segment boundary — the interrupted segment is
counted as finished, as above, and the fresh segment the run opens begins by
generating, so a checkpoint written part-way through a training phase costs the
resumed run one extra generation phase.

``nvalchemi.training.distillation.cli`` wraps all of that as a ``distill``
group on the ``nvalchemi-training`` entry point, aliased as ``nvalchemi-distill``.
:class:`~nvalchemi.training.distillation.cli.DistillationJobSpec` is the JSON
recipe the group authors (``distill init``), publishes a schema for
(``distill schema``), validates and renders (``distill spec report``), executes
(``distill spec run``), picks back up after an interruption
(``distill spec resume``), and gates (``distill evaluate``). Pre-flight
deserializes the strategy bundle with the same helpers the runtime uses and
puts an ``on_policy`` block through
:class:`~nvalchemi.training.distillation.OnPolicyConfig`'s own field
constraints, so what the recipe settles on its own --- a knob out of range, a
step budget below one, a dataset format no loader builds, a model source the
CLI could never load, a batch mixture leaving one of its two sources out, a
``seeds`` block naming no store or carrying a budget that is not a positive
count --- is refused at ``spec report`` rather than after a teacher has
reached a GPU;
what still needs the models built is reported as a CLI error when they are.
``init`` scaffolds a
:class:`~nvalchemi.training.hooks.CheckpointHook` into ``student.hooks`` so that
sequence has a checkpoint to resume from and to evaluate, and
:class:`~nvalchemi.training.distillation.cli.EvaluationSpec` accepts only the
accuracy bars ``distill evaluate`` can fill, since a bar with no measurement
behind it fails the student rather than being skipped. Student tiers are size
templates only --- a width and a depth for whatever constructor
``student.spec`` names --- never architectures.
:ref:`distillation_recipes_guide` walks the lifecycle end to end.

.. currentmodule:: nvalchemi.training.distillation.cli

.. autosummary::
   :toctree: generated
   :nosignatures:

   DistillationJobSpec
   StudentSpec
   EvaluationSpec

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

   PerAtomEnergyMatchingLoss


Representation, curvature, and ensemble objectives
--------------------------------------------------

Three further terms distill things a reference dataset has no column for. Each
needs more from the run than a target field, and each is checked at
construction.

:class:`~nvalchemi.training.distillation.EmbeddingMatchingLoss` matches the
teacher's per-atom representation. Both sides come from
``compute_embeddings`` rather than from a forward pass, so the objective needs
:func:`~nvalchemi.training.distillation.embedding_distillation_fn`, and the
student is run twice per batch. Across architectures the two widths differ,
which the learnable
:class:`~nvalchemi.training.distillation.EmbeddingProjector` reconciles: give it
the student's width by the teacher's, register it as a ``"projector"`` model
with an ``optimizer_configs`` entry of its own, and the training function routes
the student's embeddings through it. The projection is applied to the student
and never to the teacher, whose embeddings stay fixed targets — a learnable map
on the target side is optimized to be easy to hit, and the pair would minimize
the objective by collapsing the teacher's representation. The projector is a
training-time artifact: the distilled model is the student alone, so nothing
about the student's own outputs depends on it.

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

Two representations agree only up to whatever symmetry each architecture's
embedding space carries — a channel permutation, a rotation of an equivariant
block — which is what the projector absorbs and why a residual floor on this
term is normal. Weight it as a regularizer beside the terms carrying the
physical targets.

:class:`~nvalchemi.training.distillation.HessianMatchingLoss` matches the
curvature of the teacher's energy surface, which decides vibrational spectra and
integrator stability and which energies and forces do not pin down. Neither side
forms a Hessian: both are products with one random probe direction, two backward
passes each. The teacher's product and its probe are materialized onto the batch
by the ``hessian`` signal — offline through
:func:`~nvalchemi.training.distillation.label_dataset` or on the fly through the
strategy's labeling seam — and the student's comes from
:func:`~nvalchemi.training.distillation.hessian_distillation_fn`, which takes it
on a second student pass narrowed to the energy alone: a conservative student
derives its forces from the very graph the second derivative needs, and frees
that graph outside training mode, so the stock forward cannot be differentiated
again and the narrowed pass derives no forces to consume it. The student is
therefore run twice per batch here too, and every validation pass costs the
same. One probe constrains one direction, so coverage comes from
redrawing: an on-policy run gets a fresh probe every time it labels a frame,
while a store labeled once freezes one direction per structure. Because that
probe is standard normal per component, the graph-balanced value is a Hutchinson
estimate of ``||dH||_F^2 / 3V`` in (eV/A^2)^2, which for a near-converged
student runs one to two orders of magnitude above a force mean-squared error on
the same batch. Start the term a hundred to ten thousand times lighter than the
force term rather than at parity, and read a single batch's value as the noisy
one-sample estimate it is.

:class:`~nvalchemi.training.distillation.BoltzmannMatchingLoss` matches the
ensemble rather than the configuration: it is the relative entropy between the
teacher's and student's Boltzmann distributions at a temperature, blind to a
constant energy offset and to any error that does not change relative
populations. ``beta`` interpolates the forward (``0``, mass-covering) and
reverse (``1``, mode-seeking) directions. The estimator reads a batch as a
sample of the *student's* own ensemble, which is what makes the weights uniform
on the student side, so the strategy requires ``on_policy``, rejects a
relaxation propagator and any convergence criterion — the propagator's own hook
or a :class:`~nvalchemi.dynamics.base.ConvergenceHook` registered on it that
graduates graphs out, as well as one the segment loop installs from
``convergence`` or ``convergence_hook``, since none of them samples an
equilibrium ensemble — and warns when ``replay_ratio`` mixes anchor frames the
student never visited into the batch. Reweighting an off-policy sample back
onto the student's ensemble is not offered — the weights this form folds away
as uniform are not recoverable from a batch — so an existing dataset reaches
the term as ``reference_dataset``, mixed into generated frames by
``replay_ratio``. The
batch also has to be one system's configurations, since energies of different
systems are not comparable at all; seed the run with replicas of one structure,
one walker per graph. What cannot be checked is the temperature: set the term's
and the thermostat's from the same number. The two directions are not
interchangeable in scale either: the forward one is bounded above by ``log B``
and its gradient vanishes once the softmax saturates — a student whose error
spreads over more than a few ``k_B T`` — so ``beta=0`` can read as converged
while the student is far off, and ``beta`` is better held at ``0.5`` or above
until it is within a couple of ``k_B T``. Reducing energies by ``k_B T`` also puts the
gradient of either direction at up to ``1/k_B T`` per configuration, about
39 eV^-1 at 300 K, well above what a pointwise energy term produces.

The recommended recipe is therefore ``replay_ratio=1`` *and* a bounded
``replay_capacity``: the ratio keeps anchor rows out of the batch, and the
capacity keeps stale generated ones out, since every segment's loader draws
uniformly over the whole replay buffer and an unbounded one retires nothing —
after ``N`` segments only about one ``N``-th of a batch came from the current
student. Size it to the frames one segment or a few segments yield. Validation
is the other off-policy path, and the strategy refuses it outright: a
``ValidationConfig`` without a ``loss_fn`` of its own reuses the training
objective, ensemble term included, on a held-out set the student never visited,
so give the validation config a pointwise loss instead.

.. autosummary::
   :toctree: generated
   :nosignatures:

   EmbeddingMatchingLoss
   EmbeddingProjector
   HessianMatchingLoss
   BoltzmannMatchingLoss


Evaluation and acceptance
-------------------------

``nvalchemi.training.distillation.evaluation`` answers whether a distilled
student is good enough to ship. It is imported from its own subpackage rather
than the distillation namespace, because an acceptance run pulls in the
dynamics engine and the reporting stack that training itself does not need.

Accuracy is measured over a held-out set with
:func:`~nvalchemi.training.distillation.evaluation.evaluate_accuracy`, against
either the dataset's own labels or the teacher's, on-disk or scored on the fly.
The pass runs through :class:`~nvalchemi.training.ValidationLoop` — so eval
mode, the autograd policy an autograd-force student needs, and device placement
behave exactly as they do in training validation, though the weights scored are
always the live ones, never an averaged copy, and no autocast is applied: the
student predicts in its own dtype, teacher labels are cast to the dtype its own
labels are stored at, and every residual is accumulated in float64 — while the
metrics themselves are accumulated as exact global residual sums rather than
read off the loss, which is graph-balanced for training reasons an evaluation
does not share. Against a teacher, force alignment and per-atom energy residuals
fill in too. Two force-alignment numbers are reported: ``force_cosine_mean``
weights every atom equally and is dominated by atoms whose force is at or below
the student's own error, so it is the magnitude-weighted
``force_cosine_aggregate`` that ``min_force_cosine`` is read off.

.. currentmodule:: nvalchemi.training.distillation.evaluation

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate_accuracy
   AccuracyMetrics

The quantities an evaluation compares are named by the public ``AccuracyQuantity``
alias: ``"energy"``, ``"forces"``, ``"stress"``, and the diagnostic-only
``"atomic_energies"``.

:func:`~nvalchemi.training.distillation.evaluation.nonconservative_residual` is
the diagnostic behind the direct-force teacher story. A student that
differentiates an energy produces a curl-free field and can only fit the
conservative part of its teacher; the probe integrates the teacher's work
around closed loops in configuration space, which a conservative field
integrates to zero, and converts the leftover into the root-mean-square
per-atom force error a conservative student cannot avoid on that loop. The
loop's ``amplitude`` is the per-atom displacement it probes at, so calibrate it
against a thermal vibration. It is a scale-dependent lower bound rather than a
dataset-wide error bar, and one loop through a large cell's configuration space
only spans a fraction of the field's curl, so the bound loosens with system
size — read the estimator's own docstring before quoting the number.

.. autosummary::
   :toctree: generated
   :nosignatures:

   nonconservative_residual
   NonConservativeResidual

Stability is what small students actually fail at, so it is measured on a
trajectory the student drives itself.
:class:`~nvalchemi.training.distillation.evaluation.StabilityMonitor` is a
dynamics hook — the offline counterpart of
:class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook`, keeping the series
instead of comparing one live value against a threshold — and reports energy
drift and momentum conservation once the run is over. Both the endpoint drift
and the fitted rate integrate whatever the series starts with, so a student
seeded from frames that are not equilibria of its own potential needs a
``warmup_steps`` window long enough to cover the relaxation; without one, the
transient is reported as drift and can cancel a genuine one outright.
:func:`~nvalchemi.training.distillation.evaluation.extensivity_error` checks
that energy scales with replicated cells, and the radial-distribution pair
compares the structure a trajectory samples against a reference trajectory's,
reading frames straight out of a
:class:`~nvalchemi.dynamics.sinks.DataSink` filled by
:class:`~nvalchemi.dynamics.hooks.SnapshotHook`. That comparison pools every
species into one histogram by default, which cannot see a student that puts the
right distances between the wrong kinds of atom; pass ``pair`` to resolve one
species pair, and gate a chemically ordered system on the partials rather than
on the total.

.. autosummary::
   :toctree: generated
   :nosignatures:

   StabilityMonitor
   StabilityMetrics
   total_momentum
   extensivity_error
   ExtensivityMetrics
   radial_distribution
   RadialDistribution
   compare_radial_distributions
   RDFComparison

:func:`~nvalchemi.training.distillation.evaluation.measure_throughput` times a
propagator at steady state, discarding a warmup window and synchronizing the
device on both sides of the clock, and reports atoms per second and simulated
nanoseconds per day. The rate is formed from the steps the propagator's own
counter says it took, so a relaxer that converges inside the window is scored
on the window it ran and warns rather than reporting the speed it would have
needed to run the whole one. ``atoms_per_second`` is not size-independent — on
a device the batch does not saturate it climbs steeply with the batch — so
every student of a family has to be timed on the same batch for the column to
rank them, and
:func:`~nvalchemi.training.distillation.evaluation.build_acceptance_report`
rejects a family whose throughput measurements disagree on it.

.. autosummary::
   :toctree: generated
   :nosignatures:

   measure_throughput
   ThroughputMetrics

The verdict is assembled from those measurements. A caller collects one
:class:`~nvalchemi.training.distillation.evaluation.StudentEvaluation` per
candidate, states the bars as
:class:`~nvalchemi.training.distillation.evaluation.AcceptanceThresholds`, and
:func:`~nvalchemi.training.distillation.evaluation.build_acceptance_report`
returns a report that renders as Rich tables and exports as a plain
dictionary or a flat scalar map. A bar with no measurement behind it fails the
student rather than being skipped, and the from-scratch gate — the PRD's own
success criterion — compares the distilled student against an equal-size
student trained from scratch on every accuracy metric the two share, keeping
the worst ratio. Both sides of that ratio have to be one holdout's, which the
gate checks rather than assumes, and a family whose students were scored on
different holdouts is rejected outright for the same reason one timed on
different batches is. A measurement that is not a finite number fails its bar
on a detail of its own: a NaN fails every comparison it is put to and an
infinity clears every maximum, so neither decides a verdict as though it were a
number, and neither is ranked on the speed-versus-accuracy front.
Speculative-MD drafter rows are part of the report's shape and
appear once an evaluation carries
:class:`~nvalchemi.training.distillation.evaluation.DrafterMetrics`; the metric
that fills them ships with the drafter objectives. Its bar is the one exception
to fail-on-missing: drafting is a property of the student rather than a
measurement any student could have run, so ``min_drafter_acceptance_rate`` is
checked against the drafters of a mixed family and skipped for the plain
students — and rejected outright on a family with no drafter in it, so the bar
still cannot be satisfied by silence.

Every measurement rebuilds from its own export with ``from_dict``, the inverse
of the ``to_dict`` each one already had, so a sweep that evaluates each student
in its own job can persist the results and assemble one report at the end —
giving every job the same throughput batch, since that is what makes the speed
column comparable across them. A
student entry taken straight out of a report export rebuilds too; its verdict
is dropped, since verdicts belong to the thresholds of the report being built.

A caller that runs only part of the suite asks
:func:`~nvalchemi.training.distillation.evaluation.measured_bars` which bars its
measurements can decide, rather than restating the mapping: it takes the
measurement families that were filled — plus, for the accuracy family, the
quantities the pass actually compared, since a holdout scored on energy alone
leaves a force bar as unfillable as no holdout at all — and returns the
:class:`~nvalchemi.training.distillation.evaluation.AcceptanceThresholds` fields
that would then be gated on a number rather than on silence. The families each
bar reads are public as
:data:`~nvalchemi.training.distillation.evaluation.BAR_FAMILIES` and are the
same table :func:`~nvalchemi.training.distillation.evaluation.build_acceptance_report`
applies the bars from, so a bar added to the threshold model cannot go missing
from one answer while staying in the other.

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_acceptance_report
   AcceptanceReport
   AcceptanceThresholds
   AcceptanceCheck
   StudentEvaluation
   StudentVerdict
   DrafterMetrics
   measured_bars

.. currentmodule:: nvalchemi.training.distillation
