(distillation_guide)=

# Distilling a Teacher Into a Student

Knowledge distillation trains a small model — the *student* — to reproduce the
predictions of a larger, frozen one — the *teacher*. For interatomic potentials
the motivation is throughput. A foundation teacher may be far too expensive to
drive long molecular dynamics, while a student that reproduces its energies and
forces on the states that matter runs orders of magnitude faster. The teacher
also removes the usual data bottleneck, because it can label any structure,
including ones no reference calculation was ever run on.

{py:class}`~nvalchemi.training.distillation.DistillationStrategy` is the entry
point. It is a {py:class}`~nvalchemi.training.TrainingStrategy` subclass, so
everything in {ref}`training_guide` — optimizers, schedulers, validation, hooks,
checkpoints — applies unchanged. This guide covers what distillation adds: the
offline path, which labels a dataset with the teacher once, and the on-policy
path, which labels the structures the student visits in its own dynamics. A
*recipe* is one JSON file that describes a whole distillation run. Recipes, and
the `distill` CLI that runs them end to end, are in
{ref}`distillation_recipes_guide`; the symbols are in
{ref}`training-distillation-api`.

This guide assumes that you already have:

- a teacher wrapped with {py:class}`~nvalchemi.models.base.BaseModelMixin`;
- a student that is trainable and declares the outputs your objective reads;
- a dataset of structures, with or without reference labels.

For those prerequisites, see {ref}`models_guide`, {ref}`datapipes_guide`, and
{ref}`training_guide`.

## The shape of a distillation run

`DistillationStrategy` takes a named-model mapping holding `"student"` and
`"teacher"`. The teacher is **frozen by omission**: it must not appear in
`optimizer_configs`. That absence puts it in evaluation mode, with gradients
disabled, for the duration of the run. The student, and any auxiliary model
such as a learned projection, must be given an optimizer config. The strategy
raises at construction if either half of that contract is broken.

Teacher knowledge reaches the loss as ordinary batch fields. A *teacher signal*
is one quantity the teacher labels a batch with, such as its energy or its
forces. Each signal populates one `teacher_*` field, and a loss term consumes
it by pointing its `target_key` there:

```python
from nvalchemi.training import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.distillation import AtomicEnergyMatchingLoss

loss_fn = (
    EnergyMSELoss(target_key="teacher_energy")
    + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    + 0.2 * AtomicEnergyMatchingLoss()
)
```

Distillation therefore needs no special loss machinery: any built-in term
distills by naming a teacher field. Offline, where every sample carries its own
reference labels, mixing teacher targets with reference targets in one
objective is ordinary loss composition. An on-policy run cannot mix them, for
reasons covered below. The built-in signals, and the field and shape each one
lands as, are:

| Signal | Batch field | Level | Shape |
| --- | --- | --- | --- |
| `energy` | `teacher_energy` | system | `(B, 1)` |
| `forces` | `teacher_forces` | node | `(V, 3)` |
| `stress` | `teacher_stress` | system | `(B, 3, 3)` |
| `atomic_energies` | `teacher_atomic_energies` | node | `(V,)` |
| `embeddings` | `teacher_node_embeddings` | node | `(V, D)` |
| `hessian` | `teacher_hvp`, with `teacher_hvp_probe` | node | `(V, 3)` |

The first four come from the teacher's forward pass. Each shares its name with
the output the teacher must declare in `ModelConfig.outputs`, and that name is
what the construction check reports as missing. `teacher_node_embeddings` comes
from {py:meth}`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, which
costs a second pass. `teacher_hvp` comes from a Hessian-vector product along a
random probe direction, which the scorer stores beside it in
`teacher_hvp_probe`.

Each row is a {py:class}`~nvalchemi.training.distillation.TeacherSignal`, and
the table is open. To label any other teacher output, pass a spec of your own
beside the built-in names. `TeacherSignal("charges", "charges",
"teacher_charges", "node")` reads the teacher's `charges` output into a
node-level `teacher_charges` field; an optional `normalize=` callable gives the
output the shape the field should land as. Three rules are checked at
construction: the field must start with `teacher_`, the level must be `node` or
`system`, and the scorer refuses a spec naming an output the teacher does not
declare. The built-in specs are published as
{py:data}`~nvalchemi.training.distillation.BUILTIN_SIGNALS`.

`embeddings` and `hessian` are the two signals the stock training function
cannot supervise on its own, because neither has a student-side counterpart in
a plain forward pass. For those, use
{py:func}`~nvalchemi.training.distillation.embedding_distillation_fn` or
{py:func}`~nvalchemi.training.distillation.hessian_distillation_fn` (see
*Objectives beyond pointwise matching*). Using the default training function
with embedding prediction keys raises a construction error directing you to
these functions.

You do not normally declare which signals you want. With
`teacher_signals=None`, the default, the strategy derives the set from the
`teacher_*` targets the losses read: the training loss, plus the validation
loss when `validation_config` carries a `loss_fn` of its own. The objective and
the teacher therefore cannot drift apart. An explicit set must cover the
derived one and may request more. At construction, the strategy checks that the
teacher declares every resolved signal in `outputs`. It also checks that the
student computes every loss prediction key, reading `active_outputs`
intersected with the declared `outputs`, so a pretrained wrapper whose active
set was narrowed fails at construction rather than on its first batch.

These checks do not run on property assignment. Pass `validation_config` to
the constructor, or declare the extra signals in `teacher_signals`, rather than
assigning either afterwards. A spec leaves `validation_config` out, because it
carries a live loader, so a rebuild takes it as a runtime override alongside
the models: `from_spec_dict(spec, models=..., validation_config=...)`, and the
same keyword on `from_checkpoint_dict` and `load_checkpoint`. A restored run
that validates against a `teacher_*` target its training loss does not read
needs `validation_config` passed there rather than assigned afterwards.

Every resolved signal is a request for its fields on every batch. A batch
counts as labeled only when it holds every resolved field. Suppose a
validation loss reads a `teacher_*` target that the training loss does not: a
store labeled without that field then sends every batch back to the teacher. A
store meant to train with no teacher pass at all has to be labeled with the
same signal set the strategy resolves. A `teacher_*` target that names no
built-in signal, such as a field a custom scorer persisted, is an ordinary loss
target. It is read off the batch as it arrives, never derived into a signal,
and never attached on the fly, so a batch lacking it surfaces as a missing loss
target.

The training function stays a plain student forward pass. The default,
{py:func}`~nvalchemi.training.distillation.default_distillation_fn`, calls the
student and prefixes every output with `predicted_`. The teacher is never called
there. That is why the teacher can never enter the student's autograd graph,
and why the recipe survives serialization.

{py:class}`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss` is the one
pointwise term distillation adds. It matches the teacher's per-atom energy
decomposition, a quantity no reference-labeled dataset carries. Per-atom
energies are not physically observable on their own, so treat the term as a
regularizer on the student's internal decomposition, and keep a total-energy
term weighted above it. The term asks the most of both models, because the
decomposition has to exist on each side. The teacher must declare
`atomic_energies` in `ModelConfig.outputs` for the signal to be scorable. The
student must both declare *and* compute it, since the term reads
`prediction_key="predicted_atomic_energies"`. A student that emits only
`energy` and `forces`, including
{py:class}`~nvalchemi.models.demo.DemoModelWrapper`, therefore fails at
construction on the three-term objective above, with an error naming the loss
component and the missing `atomic_energies`.

### Objectives beyond pointwise matching

Three further terms distill what a pointwise target cannot carry, and each asks
the run for something a plain forward pass does not produce. The weighting and
the physics behind each are in {ref}`distillation-advanced-objectives`. This
section covers what the run has to provide.

{py:class}`~nvalchemi.training.distillation.EmbeddingMatchingLoss` matches the
teacher's per-atom representation rather than a prediction. Both sides come
from `compute_embeddings`, so the objective needs
{py:func}`~nvalchemi.training.distillation.embedding_distillation_fn` as the
`training_fn`, and the student runs twice per batch. Widths rarely agree
across architectures, so register an
{py:class}`~nvalchemi.training.distillation.EmbeddingProjector` under the model
name `"projector"` and give it an optimizer config. The training function
routes the student's embeddings through it. At construction, the strategy
checks that the student's width, the projector's `in_features`/`out_features`,
and the teacher's width compose. It refuses a student that publishes no
`node_embeddings` shape. It also refuses a student whose `compute_embeddings`
returns embeddings detached from its trainable parameters, because the
projector would then absorb the whole objective while the student learned
nothing from it. When that is the intent — a trunk frozen on purpose beside a
trainable head, with the projector alone carrying the term — register the
projector with `frozen_student=True`. The strategy refuses that flag when every
student parameter is trainable.

{py:class}`~nvalchemi.training.distillation.BoltzmannMatchingLoss` is the one
term that scores a *batch* rather than a sample. It matches the Boltzmann
weights that the two energy surfaces imply over the batch at a `temperature`,
in Kelvin. `beta` is not an inverse temperature: it interpolates
between the forward (`0`) and the reverse (`1`) relative entropy. The forward
direction is bounded by `log B` across `B` scorable graphs. Its gradient
vanishes when the softmax saturates, which happens once the student's energy
errors exceed a few `k_B T`. Under `beta=0`, a saturated gradient can look like
convergence while the student's errors remain large, so keep `beta >= 0.5`
until the errors drop within a few `k_B T`.

The Boltzmann term reads a batch as a sample of the student's own Boltzmann
distribution. It therefore requires `on_policy`, and it refuses a relaxation
propagator and any convergence criterion. It warns when `replay_ratio` mixes
in samples the student never visited; the recommended shape is
`replay_ratio=1` with a bounded `replay_capacity`. It is refused on the
validation side, because a fixed holdout is not a sample of the student's
distribution. That covers a validation `loss_fn` that holds the term, and a
validation config with no `loss_fn` of its own that would reuse a training
loss holding it. The refusal of a relaxation propagator or a convergence
criterion rests on whether the propagator samples an equilibrium ensemble at
all. The strategy infers that unless `OnPolicySettings.samples_equilibrium`
says otherwise. That setting is the override for a propagator the inference
misreads, and the refusal names it.

Under a {py:class}`~nvalchemi.training.hooks.DDPHook` the softmax runs over the
world batch. The reduced energies are gathered across ranks differentiably, so
every rank reports the world loss. The one-system check reads that gathered
batch, so a rank holding one system beside a rank holding a system of a
different size is refused too. The check compares atom counts alone: it
refuses a batch whose graphs differ in size, and it cannot tell two systems of
the same size apart. The gather and the check are constructor settings.
`world_batch=None`, the default, gathers only when a process group with more
than one rank is initialized, and `True` or `False` forces it.
`check_one_system=False` drops the guard for a batch you know to be comparable.
With `ignore_nonfinite` (the default), a graph whose teacher or student energy
is not finite is dropped from the ensemble rather than reaching every rank's
softmax. A batch of one graph scores exactly `0.0`.

{py:class}`~nvalchemi.training.distillation.HessianMatchingLoss` matches the
teacher's curvature along a random probe direction. The student's product is
taken by {py:func}`~nvalchemi.training.distillation.hessian_distillation_fn` on
a second, energy-only pass that reuses the neighbor list the stock forward just
ran on. Under the default `normalize_by_atom_count=True` the term is
graph-balanced: every graph weighs the same whatever its atom count. The
reduction is the core
{py:func}`~nvalchemi.training.losses.graph_balanced_mean`, which the
atomic-energy, embedding, and Hessian terms share and a term of your own can
call; `normalize_by_atom_count=False` takes one global mean over the valid
components instead. The graph-balanced value carries the square of a force
constant's units.
For a near-converged student it runs one to two orders of magnitude above a
force mean-squared error, so start the term a hundred to ten thousand times
lighter than the force term. Read one batch's value as the one-sample estimate
it is. Set the weight on the composition rather than on the term, since leaves
are weightless and {py:class}`~nvalchemi.training.ComposedLossFunction`
renormalizes by default. A loss pointed at the companion `teacher_hvp_probe`
is refused, since a probe is not a quantity the student is supervised against.
A student whose forces come from a head of their own is warned that the term
supervises its energy head alone.

The Boltzmann term also changes what a restart needs. It is defined on
generated batches, so it refuses to be rebuilt without the on-policy loop
described below. When the checkpoint's spec could not carry that loop whole,
re-supply it at load time, as `load_checkpoint(root, models=...,
on_policy=..., reference_dataset=...)`, or restore into a strategy already
built with them. The Hessian term has no such requirement: it reads only what
the student computes on the batch in hand.

## Offline distillation

The offline path scores the dataset once and trains from the result. It is the
cheaper path by a wide margin whenever the same structures are visited more than
once, and it is where any distillation project should start.

### Label the dataset once

{py:func}`~nvalchemi.training.distillation.label_dataset` walks a dataset in
chunks, scores each chunk with a
{py:class}`~nvalchemi.training.distillation.TeacherScorer`, and writes the
source fields plus the teacher fields into a Zarr store:

```python
from nvalchemi.training.distillation import InProcessTeacherScorer, label_dataset

scorer = InProcessTeacherScorer(teacher, ["energy", "forces", "atomic_energies"])
num_labeled = label_dataset(dataset, scorer, "labeled.zarr", batch_size=64)
```

{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer` handles the
teacher's evaluation contract for the caller. It narrows the teacher's
`active_outputs` to what the requested signals need, builds the teacher's own
neighbor list and rolls it back, picks the grad mode a teacher with autograd
outputs requires, detaches every tensor it returns, and normalizes each signal
to the canonical shape above. It leaves the batch it is handed exactly as it
found it, so the same scorer works mid-training and mid-trajectory. `dtype=`
stores labels at a reduced dtype. `probe_seed=` pins the Hessian probe
direction; leave it unset for training and labeling, where coverage comes from
redrawing the probe.

The `neighbor_list=` setting decides where the teacher's neighbor list comes
from. The default, `"rebuild"`, builds the teacher's own list on every call,
whatever the batch carries. `"reuse"` is for a student that has already built
the list the teacher needs, in the teacher's format and at its cutoff. The
scorer then consumes that list and builds nothing. It checks only what it
cannot infer: that the keys the teacher's format reads are present, and that a
cutoff stamp, if the batch has one, equals the teacher's cutoff. Otherwise it
raises a `ValueError` naming the missing key or the mismatched cutoff; it never
falls back to a rebuild. The batch does not record whether a list holds each
pair once or twice, so a reused list has to match the teacher's `half_list` by
construction.

The scorer builds exactly one list per batch, so it refuses a
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` teacher that plans
more than one neighbor-list source; compose such a teacher with
`neighbor_adaptation="always"` or a wide enough `max_cutoff_ratio` (see
{ref}`models_guide`).

Labels are attached with `overwrite=True`. A scorer reaching outside the
`teacher_*` namespace would therefore replace the reference field of that
name, which is the very label the student is trained against. `label_dataset`
refuses declared `label_fields` outside the namespace before it writes the
first chunk, and it checks the fields each chunk actually returns again. The
same namespace rule holds on every other labeling route.

Labeling is resumable. An existing store is continued from `len(store)`, and
every resumed chunk is checked against the store's fields, levels, dtypes, and
row shapes. Two stores are refused rather than resumed from a misaligned
offset: one longer than the dataset, which was written from a different
dataset, and one that an interrupted append left inconsistent. The neighbor
list is not carried over, in either format. The dense tensors cannot append
into a fixed-width store array, and a sparse list is dropped because the cutoff
it was built at is a batch attribute the store does not hold.
`keep_neighbors=True` stores the sparse list anyway. A store written under one
setting and resumed under the other is refused as field-set drift. Nothing
rebuilds a list when batches are read back out of the store; the next section
adds the hook that does.

### Train from the labeled store

Reading the labeled store back is not distillation-specific. The teacher fields
arrive as ordinary batch attributes at the levels they were written at. The
reader, dataset, and loader are the ones any training run uses, and no teacher
forward pass happens during training at all:

```python
import torch

from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.training import OptimizerConfig
from nvalchemi.training.distillation import DistillationStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DataLoader(
    Dataset(reader=AtomicDataZarrReader("labeled.zarr"), device=device),
    batch_size=32,
)

strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={
        "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    },
    loss_fn=loss_fn,
    num_steps=10_000,
    devices=[device],
)
strategy.run(loader)
```

Open the store on the device the strategy trains on, and name that device on
both sides: a `Dataset` opened without `device=` emits on the current CUDA
device whenever one exists, while `devices` defaults to CPU and takes
`torch.device` objects (see {ref}`training-strategy-api`). `label_dataset`
takes the same `device=` and moves each chunk there before scoring; put the
teacher there yourself with `teacher.to(device)`.

Nothing in the training loop builds a neighbor list. Labeling drops the
neighbor tensors from the store, and a wrapped MLIP raises rather than building
its own list, so a graph student needs a
{py:class}`~nvalchemi.hooks.NeighborListHook` at `TrainingStage.BEFORE_FORWARD`
in `hooks=`, configured from its own neighbor config. *Graph students need a
neighbor list on both engines* below shows it beside the propagator-side hook
the on-policy loop needs as well, and {ref}`data_guide` covers the list formats.
Both examples use neighbor-free demo potentials, so the hook becomes necessary
only when a real MLIP takes the student's place.

A complete, runnable version of this workflow is
{doc}`/examples/intermediate/09_offline_distillation`.

### Consuming the labeled store elsewhere

The store is a plain Zarr hierarchy, so the labels are not locked to this
toolkit. `teacher_energy` and `teacher_forces` are arrays under the `core`
group beside `positions`, with the per-sample offsets of per-atom arrays in
`meta/atoms_ptr` (the layout is in {ref}`zarr_compression_guide`), and the
store's root attributes record whether each field is per-atom or per-system.
Any Zarr client can therefore read a labeled store: a student written in another
framework consumes the same teacher labels without running the teacher again,
and without importing `nvalchemi` at all.

### Labeling on the fly

A batch that arrives without the required `teacher_*` fields is labeled on the
fly instead. The strategy registers an internal hook for this on
`BEFORE_FORWARD`, prepended ahead of your own hooks, so labeling happens before
a neighbor-list hook of yours runs; that order costs nothing, because the scorer
builds and rolls back the teacher's own list whatever is on the batch. Short
runs and interactive sessions
therefore work with no labeling pass at all, and unlabeled validation data
needs no preparation. The first batch labeled this way triggers a one-time
warning that names the teacher fields it lacked, so a long run does not pay an
unplanned teacher pass per step unnoticed. Set `label_missing=False` to turn
on-the-fly labeling off; an unlabeled batch then surfaces as a missing loss
target.

On-the-fly labels are attached to the device-placed batch the strategy trains
on. That batch is a copy of the one you handed over, so the labels do not
persist on your object. A loader that replays the same systems every epoch
therefore pays one teacher pass per epoch, which is why a long run should label
offline first. Label precision is the scorer's:
`InProcessTeacherScorer(autocast=False)` (the default) disables autocast around
the scoring pass whatever region surrounds it, so an on-the-fly label equals
the offline one and the two paths are interchangeable. `autocast=None` runs the
teacher under the caller's region; `True` or a floating dtype enables it
(`True` keeps the ambient region's dtype, a dtype pins the pass).

## On-policy distillation

Offline distillation trains the student on whatever structures the dataset
happens to hold. Those are not the structures the student visits once it drives
dynamics itself, and that gap is what makes a distilled potential drift or blow
up on long trajectories. On-policy distillation closes the gap: the student's
own propagator generates frames, the teacher labels them, and the student
trains on them.

Setting `on_policy` turns
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.run` into a
*segment loop*. A *segment* is one generate-label-train cycle, and the loop
repeats segments until the step budget is spent. `run()` then takes no
dataloader, because each segment builds its own. One segment has three phases:

1. **Generate** — the propagator advances the live batch by
   `generation_steps`. The first segment starts from the *initial
   structures*, the structures the trajectories begin at, passed as
   `initial_structures`.
2. **Label and capture** — the *labeling hook*, a
   {py:class}`~nvalchemi.training.distillation.TeacherLabelHook` on the
   propagator, scores a frame every `label_frequency` steps. It copies each
   labeled frame into the *capture sink*, the
   {py:class}`~nvalchemi.dynamics.sinks.DataSink` that holds a segment's frames
   until the segment boundary. The capture sink is host memory unless
   `capture_sink` names another sink; a
   {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` keeps the frames on the
   device. The segment's final frame is labeled too. The sink is then drained
   into the *replay buffer*, a
   {py:class}`~nvalchemi.training.distillation.ReplayBuffer` that accumulates
   labeled frames across segments.
3. **Train** — a freshly built mixed loader draws `training_steps_per_segment`
   batches, and each batch goes through the ordinary per-batch stages. Every
   batch is a *mixture* of replay-buffer frames and samples from the *reference
   dataset*, a fixed teacher-labeled dataset, split at `replay_ratio` (see
   *The mixture ratio*).

```python
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.training.distillation import InitialStructures, OnPolicyConfig

strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={
        "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    },
    loss_fn=loss_fn,
    num_steps=10_000,
    on_policy=OnPolicyConfig(
        dynamics=NVTLangevin(student, dt=0.5, temperature=300.0, friction=0.01),
        teacher_scorer=scorer,
        initial_structures=InitialStructures(initial_dataset),
        replay_ratio=0.25,
        training_steps_per_segment=32,
        batch_size=16,
        generation_steps=50,
        label_frequency=10,
        replay_capacity=8192,
    ),
    reference_dataset=reference_dataset,
)
strategy.run()
```

The propagator must hold the very module registered as `models["student"]`,
either directly or composed into a larger model. The strategy checks that
object identity at construction. Because the module is shared, every optimizer
update reaches the trajectories the propagator generates next. A propagator
that merely *composes* the student is moved whole to the generation device and
held in evaluation mode for the whole loop, because the training phase forwards
`models["student"]` rather than the composition. The mode contexts are the core
{py:func}`~nvalchemi.training.evaluating` and
{py:func}`~nvalchemi.training.eval_configured_models` (see
{ref}`training-strategy-api`).

The run is sized in optimizer steps rather than epochs, since each segment
builds its own loader. One segment counts as one epoch for hooks and
epoch-cadence validation, while a step-cadence validation fires inside
segments. The run closes with one terminal validation, skipped when a cadence
already validated at the final step, so a metric-driven scheduler is never
stepped twice on one set of metrics.

The loop is data-parallel across ranks. Each rank propagates its own strided
shard of the initial structures, labels those frames with its own teacher
replica, and fills its own replay buffer. The student's gradient all-reduce,
run by a {py:class}`~nvalchemi.training.hooks.DDPHook`, is the only per-step
training traffic between ranks. *Scaling the segment loop out* below covers
what a multi-rank run asks of the settings.

The propagator is any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` — an
integrator generating trajectories, or an optimizer generating relaxation
paths. Nothing downstream of the config reads a velocity or a temperature.

The initial structures have to carry the fields the propagator updates in
place. Building the `OnPolicyConfig` loads one row from `initial_structures`
and hands it to the propagator's
{py:meth}`~nvalchemi.dynamics.BaseDynamics.check_initial_batch`, so a missing
field is a construction error named against the propagator rather than an
`AttributeError` on the first step ({ref}`implementing-dynamics` covers which
keys that method reads). For the built-in propagators the check comes to
`velocities` and `atomic_masses`, which
{py:class}`~nvalchemi.data.AtomicData` fills unless a store they were written
to dropped them. The variable-cell propagators also need a `cell`, because
nothing fills one in for an aperiodic structure.

You do not need to supply the keys listed in `__needs_keys__`: `forces` for
every integrator and optimizer, plus `stress` for NPT, NPH, and the
variable-cell FIRE optimizers. The propagator computes them with one `compute`
before its first step. `OnPolicySettings.probe` gates that construction-time
forward, and the criterion probe a relaxation lifecycle runs on the same row;
`probe=False` skips both and defers a mismatch to the first step. The probe
builds exactly one neighbor list, so it cannot run a propagator whose model
plans more than one neighbor-list source. Such a propagator is named in a
warning, which `probe=False` also silences.

Initial structures therefore need not differ from reference-dataset samples,
which must carry no `energy` or `forces` at all. They may also be the outputs
of earlier relaxation runs. Such a store marks its structures with an exit
status, which the propagator would otherwise read as finished and refuse to
move. `InitialStructures` strips that metadata and re-initializes `status` and
`system_id`, so the propagator can advance the structures without a manual
cleanup pass.

(distillation-initial-structures)=

The initial structures live behind an
{py:class}`~nvalchemi.training.distillation.InitialStructures`: a thin subclass
of the core {py:class}`~nvalchemi.dynamics.OrderedStructureSampler` that adds
only the recipe round trip. The sampler serves the dataset's rows in order from
one position, `next_row`, which the initial batch, any later *backfill* (the
structures a relaxation lifecycle draws to replace the ones that finished), and
a restart all share. A structure is therefore propagated once, and a run
restored from a checkpoint picks up where it stopped rather than at row zero;
{ref}`dynamics-structure-sources` covers the position, the budgets, `draw`, and
sharding. `initial_structures=` takes any
{py:class}`~nvalchemi.dynamics.StructureSource`, the protocol of the members
the loop reads, importable from the distillation package under its historical
name, `InitialStructuresSource`. A bare `BatchDatasetProtocol` dataset, the
common case, is wrapped in an unbudgeted source. See
[Custom components](#custom-components) for a source of your own.

An *unbudgeted* source is propagated whole, as a single batch, so it *is* the
set of systems the run generates from; size it to the device. A budget —
`InitialStructures(dataset, max_atoms=4096)`, or `max_batch_size`, or
`max_edges` — packs the initial batch first-fit in row order instead and
leaves the remainder, in row order, for the backfill (see *Relaxation paths
need a convergence lifecycle*).

(distillation-cadence)=

`label_frequency` is the throughput setting. The teacher is the expensive
model. A segment that labels every tenth frame costs a tenth of the teacher
passes, while it still generates every frame at student speed. The cadence
counts the propagator's cumulative `step_count`, which carries across segments.
The hook reads the count before it is incremented, so a frequency `f` fires at
steps `0, f, 2f, ...`, while a segment's forced last frame lands one step
later. When `generation_steps` is a multiple of `label_frequency`, as with the
defaults (`100` and `100`), the cadence and boundary labels would land on
adjacent steps. To avoid scoring twice when `label_frequency > 1`, the hook
skips the cadence dispatch that immediately follows a labeled frame. Forced
boundary labels are always kept, so an early-exiting segment and the final
frame of the run are still labeled. Under the defaults that leaves one label
per trajectory per segment, on its last frame. The very first segment adds one
more: the frame after the first step, which the cadence lands on at step count
zero. `step_count` never resets, so a later `run()` or a restart does not pay
that extra label again.

Size `replay_capacity` with that arithmetic in hand. A segment contributes one
frame per trajectory per labeled step, and FIFO eviction retires whole frames
in arrival order. A capacity that is not a multiple of the trajectory count
therefore cuts a segment's contribution mid-step. The trajectories at the back
of the batch are then over-represented in every mixture drawn afterwards, so
make the capacity a multiple of the number of initial structures. Two
decisions are left to policy: what enters the buffer and what leaves it.
`replay_admission` takes an
{py:class}`~nvalchemi.training.distillation.AdmissionPolicy`, a predicate over
each segment's frames applied before the schema check. `replay_eviction` takes
the string `"fifo"`, which means the
{py:class}`~nvalchemi.training.distillation.FIFO` reference and is the one
spelling a recipe carries, or an
{py:class}`~nvalchemi.training.distillation.EvictionPolicy` instance that names
the frames a full buffer drops.

```{note}
Request the same signals on `OnPolicyConfig.teacher_scorer` that the loss
reads. With a reference dataset there is no choice. The generation scorer's
teacher fields and the reference dataset's stored ones are compared for
equality at construction, so a scorer narrower than the dataset is rejected
outright. A narrow scorer constructs only against an equally narrow dataset,
and then every generated frame is scored twice: once during generation, and
again on its way into a training step. The strategy warns whenever the
generation scorer is narrower than the loss, with or without a reference
dataset.
```

Any {py:class}`~nvalchemi.training.distillation.TeacherScorer` may drive
generation, not only the in-process one. A custom scorer should declare
`label_fields`, the batch fields its `label()` populates. The strategy reads
that declaration through
{py:func}`~nvalchemi.training.distillation.scorer_fields` to check the targets
up front. Without `label_fields`, the strategy cannot inspect the targets
before scoring. It then warns and defers the parity check to the first
segment's mixed loader, so a mismatch surfaces only after a whole generation
phase has been paid for. A `label_fields` entry outside the `teacher_*`
namespace is refused at construction, because the labeling hook must never
overwrite the `energy` and `forces` that drive the propagator's next step. A
custom `teacher_*` field the scorer writes is an ordinary loss target, as it is
offline. Generation writes it onto every captured frame. The reference dataset
has to carry it too, which the parity check enforces. Validation data has to
arrive already carrying it, since nothing labels it on the fly. At least one
built-in `teacher_*` target, or an explicit `teacher_signals`, is still
required alongside it.

A runnable three-segment loop is
{doc}`/examples/intermediate/10_onpolicy_distillation`.

### Graph students need a neighbor list on both engines

Both examples generate with neighbor-free demo potentials. The configuration
above fails for a student that reads a neighbor list off the batch, for three
reasons. The initial batch comes from a store, and a store holds no neighbor
tensors. The construction check reads the propagator's `__needs_keys__`, which
never includes them. And nothing in a dynamics step builds a list. A wrapped
MLIP therefore raises a `KeyError` naming the missing neighbor tensor on the
first propagator step, before a single frame is generated. The
training side has the same gap. The labeling hook strips the neighbor tensors
from every captured frame, and `label_dataset` drops them from the reference
dataset, so the first mixed batch also reaches the student's forward without a
list.

The remedy is one hook per engine, both configured from the student's own
neighbor config. The propagator takes a
{py:class}`~nvalchemi.hooks.NeighborListHook` at `BEFORE_COMPUTE`, and the
strategy takes one at `BEFORE_FORWARD`:

```python
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import OptimizerConfig, TrainingStage
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InitialStructures,
    OnPolicyConfig,
)

neighbor_config = student.model_config.neighbor_config
propagator = NVTLangevin(student, dt=0.5, temperature=300.0, friction=0.01)
propagator.register_hook(
    NeighborListHook(neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
)
strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={
        "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    },
    loss_fn=loss_fn,
    num_steps=10_000,
    hooks=[NeighborListHook(neighbor_config, stage=TrainingStage.BEFORE_FORWARD)],
    on_policy=OnPolicyConfig(
        dynamics=propagator,
        teacher_scorer=scorer,
        initial_structures=InitialStructures(initial_dataset),
        replay_ratio=0.25,
        training_steps_per_segment=32,
    ),
    reference_dataset=reference_dataset,
)
```

`student.make_neighbor_hooks()` builds the propagator-side hook from the same
config (see {ref}`models_guide`). The teacher needs neither hook: the scorer
builds and rolls back its own list on every batch it labels, so the student's
neighborhoods never reach it. The propagator's hook is a live object that no
recipe describes. A loop rebuilt from a recipe or a checkpoint therefore starts
without it, and fails the same loud way on its first step. Register the hook
again on `strategy.on_policy.dynamics` before `run()`, or restore into a
strategy built with it. A student whose forces are an energy gradient also has
to take them with `create_graph=True` while training, as
{py:class}`~nvalchemi.models.demo.DemoModelWrapper` does; that is a wrapping
matter covered in {ref}`models_guide`.

### Relaxation paths need a convergence lifecycle

A relaxation propagator differs from an integrator in one way that matters
here: its trajectories *end*. Without convergence tracking, the propagator
keeps stepping converged structures and the labeling hook keeps copying them.
The replay buffer then fills with redundant minimum configurations every
`label_frequency` steps, while the run reports plausible losses. Set `fmax` to
turn on the *convergence lifecycle*, which tracks convergence and *graduates*
converged structures, retiring them from the batch:

```python
from nvalchemi.dynamics import FIRE
from nvalchemi.training.distillation import InitialStructures

on_policy = OnPolicyConfig(
    dynamics=FIRE(student, dt=0.1),
    teacher_scorer=scorer,
    initial_structures=InitialStructures(initial_dataset, recycle=True),
    fmax=0.05,
    replay_ratio=0.25,
    training_steps_per_segment=32,
    batch_size=16,
    generation_steps=50,
    label_frequency=10,
)
```

`fmax` is the max-force-norm threshold. It is compared against the student's
forces, the ones the relaxation itself converges on. A criterion the threshold
cannot express is passed whole, as `convergence_hook=`, instead. The two are
one criterion under two spellings, so setting both is refused. A hook passed
whole has to migrate status from the `0` the run stamps its structures with to
at least the propagator's `exit_status` (status migration is covered in
{ref}`hooks_guide`, and graduation in {ref}`dynamics_guide`), and it has to fire
on every step, because a structure is captured at the step it converges. Prefer
the threshold unless the criterion genuinely needs a hook. `convergence_hook` is a
*runtime-only field*: it holds a live object that no recipe can describe, so a
rebuilt run needs it supplied again, while `fmax` travels in the recipe. Either
way, `OnPolicyConfig.convergence_criterion` is the live hook the lifecycle
drives, built once and handed over by identity.

The lifecycle alone manages graduation and backfilling. To prevent conflicting
status transitions, a propagator that already holds a status-migrating
{py:class}`~nvalchemi.dynamics.ConvergenceHook`, a multi-sub-stage
{py:class}`~nvalchemi.dynamics.FusedStage` (constructing one registers a
migrator on every non-last sub-stage), and a
{py:class}`~nvalchemi.distributed.DomainParallel` propagator (its step
dispatches no `ON_GRADUATE`) are refused when the `OnPolicyConfig` is built. A
migrator registered afterwards, or an internal `sampler` on the propagator, is
refused when `run()` starts. The only fused shape the lifecycle accepts is a
single sub-stage with no criterion of its own. Relaxation paths
that genuinely need staged dynamics should use the propagator's own lifecycle
instead, with `fmax` left unset. A propagator managing its own convergence
keeps its final frames, since graduated frames are left out of capture only
under the managed lifecycle.

With `fmax` set, a converged structure freezes where it stopped and graduates on
the step its status reaches `exit_status`. The propagator reports that step at
`DynamicsStage.ON_GRADUATE` with `ctx.graduated_mask`, where a converged-frame
hook stores the structure once, unlabeled, as the minimum it reached; the
segment boundary retires it from the active batch and backfills its place.
Frames then reach the buffer by two routes that partition them. The labeling
hook stores the structures still relaxing, narrowing the frame to them *before*
the teacher pass, so the teacher pass shrinks as the batch converges. The
converged structures are labeled in a single teacher pass as their sink drains
onto the buffer's device.

A trajectory can also end by diverging. No criterion accepts a NaN, so a
structure whose positions or forces stop being finite is frozen at `exit_status`
on that step and kept out of both routes. It is retired and backfilled at the
boundary like a converged one, and each boundary warns once with a count of such
structures. `OnPolicyConfig.divergence` decides what counts as diverging: a
callable over the frame batch that returns one boolean per graph. The default is
{py:func}`~nvalchemi.training.distillation.nonfinite_divergence` (an alias of
the core {py:func}`~nvalchemi.dynamics.hooks.nonfinite_graph_mask`, over
`positions` and `forces`). The predicate is evaluated once per step, and its
verdicts are ORed into a record that the capture hook and the segment boundary
both read. A predicate of your own, such as a bond-length or cell-volume bound,
is a runtime-only field like `replay_admission`. A predicate returning anything
but one `bool` per graph is refused.

The backfill draws from the same position the initial batch was packed from.
It draws as many structures as graduated, within the atoms they held, and it
skips a row that does not fit rather than stalling on it. An unbudgeted source
has nothing left to draw, because it started every row it owns, so the batch
narrows by one trajectory per graduation. `InitialStructures(dataset,
recycle=True)` wraps the position back to the front instead, so the trajectory
count holds and the run relaxes the same structures again. The structures are
reloaded from the dataset as it stores them, not from where the propagator's
last frame left them, so the second pass starts from the same geometries under
a fresher student. One draw reaches every row at most once, so two copies of
one structure never enter a batch together. `recycle` is the source's flag, and
only a run managing a lifecycle ever backfills, so setting `recycle` with
`fmax` unset is refused at construction. A budget is the other way to keep the
batch full: the source packs a first-fit batch and spends the remaining rows on
the backfill without serving a structure twice. When the position reaches the
end with nothing left and no `recycle`, the run warns once and spends its
remaining training steps on the frames it already has.

### The mixture ratio

`replay_ratio`, written λ, is the fraction of every training batch drawn from
the replay buffer; the rest comes from the reference dataset,
`reference_dataset`. It is the setting to reach for first, because it decides
how far the run is allowed to follow its own trajectories:

- **λ = 1** trains on generated data alone and takes no reference dataset. The
  student is pulled entirely toward wherever its own dynamics go, which is also
  the failure mode: if the trajectory drifts into configurations the teacher
  was never meant to describe, nothing pulls it back. Passing a
  `reference_dataset` anyway is rejected rather than ignored, because it would
  be checked for schema and device and then never sampled.
- **0 < λ < 1** keeps a fixed, teacher-labeled distribution in every batch. The
  reference dataset is the pull: it holds the student on data whose coverage
  you chose, while the generated share keeps closing the gap between the
  training distribution and the one the student actually visits.
- **λ = 0** is offline distillation. The loop rejects it rather than running
  generation whose frames it would never train on. Drop `on_policy` and call
  `run(loader)` over the labeled store instead.

The composition is exact per batch rather than an average: with
`replay_ratio=0.25` and `batch_size=16`, every optimizer step sees twelve
reference samples and four generated ones. The achievable granularity is
`1 / batch_size`, so the two settings only mean something together. A ratio
that rounds either source down to zero samples per batch is rejected at
construction, and the error names the smallest batch size that works.

The loop rebuilds the mixed loader every segment, because the batch sampler
reads its child datasets' lengths once and the buffer grows between segments.
If you drive {py:func}`~nvalchemi.training.distillation.build_mixed_loader`
yourself, do the same, or the newest frames are never sampled. Each rebuilt
sampler seeds its generator with `OnPolicyConfig.seed` plus the segment index,
so the reference draw is reproducible across runs without repeating within one.
That setting, not the global `torch` seed, is the mixture's randomness. Because
the generator seed is a sum, two runs whose `seed` values differ by one draw
the same sequence shifted by one segment. Replicates meant to be independent
need `seed` values at least `num_steps // training_steps_per_segment` apart.

(distillation-reference-dataset)=

### The reference dataset must be teacher-labeled

A mixed batch is one collated `Batch`, and collation is not a merge. It keeps
only the fields *both* sources hold and drops the rest, while a whole level
that only one side holds is zero-filled for the other side's samples. Either
behavior would be silent, so both are rejected instead: the reference
dataset's schema is compared against the buffer's, on a probe batch drawn from
each side.

The schema the reference dataset has to match is the replay-frame contract:
the structure, whatever propagator state travels with it, and the `teacher_*`
labels. The contract excludes the `energy`, `forces`, and `stress` the
propagator wrote on the live frame. The labeling hook strips those on the way
into the buffer, so a stored frame never carries the student's own prediction
under a reference target's name.

```{warning}
**A raw DFT-labeled dataset cannot be used as the reference dataset.** Its
`energy` and `forces` are reference labels, not teacher labels, and it is
refused rather than quietly mixed in. Running it through
{py:func}`~nvalchemi.training.distillation.label_dataset` is necessary but not
sufficient: labeling carries every source field over, so the labeled store
holds `teacher_energy` and `teacher_forces` *alongside* the `energy` and
`forces` it started with, and the check refuses it just the same.
```

The remedy is to strip the reference labels on the way in, because
`label_dataset` writes what the dataset hands it and applies no transform of
its own. A per-sample transform on the streaming
{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` is the general tool. A
field it sets to `None` is gone
from the batch `load_batches` re-forms, because
{py:meth}`~nvalchemi.data.Batch.from_data_list` takes its key list from the
sample's non-`None` fields.

```python
from nvalchemi.data import AtomicData
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset


def strip_reference_labels(
    data: AtomicData, metadata: dict
) -> tuple[AtomicData, dict]:
    """Drop the reference labels a generated frame never carries."""
    data.energy = None
    data.forces = None
    data.stress = None
    return data, metadata


unlabeled = Dataset(
    reader=AtomicDataZarrReader("dft.zarr"),
    device="cpu",
    transforms=[strip_reference_labels],
)
label_dataset(unlabeled, scorer, "reference.zarr", batch_size=64)
reference_dataset = Dataset(reader=AtomicDataZarrReader("reference.zarr"), device="cpu")
```

Assigning `None` is the deletion idiom, since `AtomicData` has no
`__delitem__`. The transform has to return the `(data, metadata)` pair it was
handed. Two things defeat it. `skip_validation=True` builds the fused batch
straight from raw tensor dicts and never runs the per-sample pipeline, so leave
it at its default here. And the transform has to strip every sample alike: a
batch whose first sample kept `forces` while later ones dropped them fails
collation on a batch-dimension mismatch.

When the labeled dataset fits in memory, an `InMemoryDataset` built with a
`batch_transforms=` entry that deletes the three keys off the resident `Batch`
does the same. That transform runs once, as the batch is materialized, so every
chunk `label_dataset` reads is already stripped, and `Batch`, unlike
`AtomicData`, supports `del`. {ref}`datapipes_guide` covers both transform
kinds. Choose between the two on memory: the streaming form has no ceiling. The
third option is to label structures that never carried reference labels at all,
which is what `build_systems` does in
{doc}`/examples/intermediate/10_onpolicy_distillation`.

Two checks enforce all of this, at different points. At construction, the
`teacher_*` field sets of the two sources are compared for *equality*: a
reference dataset with no teacher labels, one narrower than the generation
scorer, and one wider all fail alike. The dataset is also probed for any field
a generated frame can never carry. That probe catches a store labeled over an
existing DFT-labeled dataset before a single teacher pass is paid. The full
field, level, and dtype comparison against real frames is left to the first
segment's mixed loader, so it surfaces only once a segment has generated and
labeled. That is a whole segment of forward passes with an expensive teacher,
so it is worth knowing the frame schema up front. A stored frame is whatever
the initial structures carry, minus everything run-local, plus one field per
teacher signal. The run-local fields are `energy`, `forces`, `stress`, the
neighbor tensors, and the dynamics bookkeeping, which is whatever
{py:meth}`~nvalchemi.dynamics.BaseDynamics.bookkeeping_keys` reports across the
propagator's stage tree (`status` and `system_id` among them).
Structures built from plain {py:class}`~nvalchemi.data.AtomicData` with
`energy` and `forces` zero-filled therefore store `positions`,
`atomic_numbers`, `atomic_masses`, `atom_categories`, and `velocities`, plus
`cell` and `pbc` for a periodic system.

To read it off the run rather than off this list, take one throwaway segment
with no reference dataset and compare:

```python
probe = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs=optimizer_configs,
    loss_fn=loss_fn,
    num_steps=1,
    on_policy=OnPolicyConfig(
        dynamics=dynamics,
        teacher_scorer=scorer,
        initial_structures=InitialStructures(initial_dataset),
        replay_ratio=1.0,
        training_steps_per_segment=1,
        batch_size=1,
        generation_steps=1,
        label_frequency=1,
    ),
)
probe.run()
print(sorted(probe.replay_buffer.schema))
```

The names come back as `level.field`, the form a mismatch is reported in.
Strip the level off each name to compare against the reference dataset's bare
`field_names`, or compare against the same `level.field` names read off a
probe batch drawn from the dataset.

Mixed batches require matching dtypes across sources. Batch collation casts
the trailing slice to the lead slice's dtype, and chunk order is not fixed, so
mixing float64 reference data with float32 generated frames would change
precision nondeterministically. The loader therefore rejects dtype mismatches
outright. One scorer alone does not give the two sides the same dtypes. A
store hands every floating field back at the dtype of the dataset's
`positions`, which is float32 for essentially every dataset, while a generated
frame keeps whatever dtype the generation scorer emitted. Build that scorer
with an explicit `dtype` matching what the store returns, and label the
reference dataset with the same scorer. The example needs no cast only because
its teacher already computes at float32.

Supervising one batch against teacher labels and reference labels at once
needs masked loss composition, which is not supported yet. An on-policy run is
therefore supervised by the teacher throughout. Annealing between teacher and
reference targets with a
{py:class}`~nvalchemi.training.losses.base.LossWeightSchedule` is an offline
technique.

Both mixture sources are collated before the strategy moves the batch, so the
reference dataset's device pins the replay buffer to it. Leaving `replay_device`
unset stages generated frames there. Naming a different device is rejected at
construction, rather than discovered mid-run as a cross-device collation
failure. The device that counts is the one the dataset actually *emits* on, read
through the core {py:func}`~nvalchemi.data.datapipes.dataset_device`, with
{py:func}`~nvalchemi.data.datapipes.same_device` deciding whether an index-less
`cuda` and an indexed one point at the same place (both are documented in
{doc}`/modules/data`). To move the mixture, therefore, open the reference
dataset on the device the run trains on. `replay_device` matters only in a run
with no reference dataset, where the frames stay in host memory unless it names
another device.

### Scaling the segment loop out

The runbook — the `DDPHook`, the `torchrun` lines, and the multi-node
rendezvous — is in {ref}`distillation-scaling-out`. This section covers what
the segment loop asks of the settings once the world is larger than one rank.

The initial structures are dealt out *strided*, by
{py:meth}`~nvalchemi.dynamics.OrderedStructureSampler.shard`: rank `r` takes
every `world_size`-th structure from offset `r`. The shards are therefore
disjoint, cover the dataset, and differ by at most one structure. Those rows
are all that the rank may propagate, and they are public as
{py:attr}`~nvalchemi.training.distillation.DistillationStrategy.structure_shard`.
The position counts rows of that shard rather than rows of the dataset, so the
initial batch and every later backfill draw from the rank's own shard alone.
`system_id` is not a row of that shard. The ids number the trajectories a rank
has started, so under `recycle` they keep climbing while the position wraps,
and each rank hands them out from its own base. The deal itself is the core
{py:func}`~nvalchemi.data.datapipes.distributed_shard`, which a source of your
own can call for the same disjoint, strided share.

Two consequences follow from the deal. First, a dataset holding fewer
structures than there are ranks is refused; the verdict is agreed across ranks
through {py:func}`~nvalchemi.training.distributed.all_reduce_flags` before the
first gradient collective, and the error names the ranks whose shard seeded
nothing. Second, a dataset that does not
divide evenly draws a warning rather than a refusal. Every rank draws the same
number of replay samples per batch from a buffer holding only its own
trajectories, and the gradients are averaged rank by rank, so a frame
generated on a shorter shard reaches the optimizer with more weight. Size the
dataset as a whole *multiple* of the world size. Striding by index balances
sample count rather than atom count, so sort the dataset by atom count before
sharding to balance the computational load across ranks.

The world *divides* the generation work rather than multiplying it. A
segment's aggregate frame count, and the teacher bill paying for it, is what
the single-process run produced, with each rank contributing a `1/world_size`
share. `generation_steps`, `label_frequency`, and `replay_capacity` are all
per rank. At a fixed `replay_capacity`, each rank's buffer therefore spans
`world_size` times as many segments before FIFO eviction reaches back, and
every mixed batch grows staler as the world grows. Raise `generation_steps` or
the structure count alongside the world, or lower `replay_capacity` by the
world size. Do not do both, or the buffer spans only `1/world_size` of the
history the single-process run had.

Sharding separates the structures, but it does not separate the randomness on
its own. The loop owns two seeded streams: the mixture sampler's
`OnPolicyConfig.seed`, and every integer `random_seed` in the propagator's
stage tree, such as {py:attr}`~nvalchemi.dynamics.NVTLangevin.random_seed`.
Both are moved onto a per-rank stride of the seed space,
`OnPolicySettings.rank_seed_stride` (default `1_000_003`), which a recipe
records and a restart checks: the propagator side goes through
{py:meth}`~nvalchemi.dynamics.BaseDynamics.seed_offset` with
`rank * rank_seed_stride`, undone with the negated offset when the run exits.
Keep the stride above the run's step count, since both streams add a step
counter to their base seed. `seed_offset` returns the stages it could not move,
such as one exposing a {py:class}`torch.Generator` and no integer seed, and
those stay on the shared stream. Every rank names them in a warning before the
first segment, and you have to give them a rank-distinct seed yourself. That
matters most when the
initial structures are replicas of one geometry, which is how a run asks for
one trajectory per rank. Sharding separates nothing there, and an unmoved
stage makes every rank generate identical frames, billed once per copy.

A multi-rank launch requires a synchronized student. With a `DDPHook`
registered, the check reads
{py:attr}`~nvalchemi.training.hooks.DDPHook.wrapped_keys` and passes when
`"student"` is among the models the hook wrapped; without one, something has to
own `models["student"]` after `SETUP` in the way
{py:func}`~nvalchemi.training.unwrap_model` reads it, as a
gradient-synchronizing wrapper of your own does. A launch that leaves the
bare student registered is refused, since each rank would otherwise train a
private student and only rank zero's would be checkpointed. A wrapper working
in place, such as FSDP2's `fully_shard` or hook-based gradient
synchronization, leaves the registered object as it was, so the student reads
as unwrapped. `OnPolicySettings.require_wrapped_student=False` waives the
refusal with a one-time warning, and keeping the ranks' students in step is
then up to you.

Where the reference dataset sits decides where every rank collates, because
the replay buffer follows it: keep it in host memory or let it emit lazily, and
never stage it eagerly on one GPU before the hook has pinned the rank.
{ref}`distillation-scaling-out` covers the placements, including the `SETUP`
hook that moves a host-memory dataset onto the rank's device, and the
index-less `replay_device` that {py:func}`~nvalchemi.data.resolve_device`
resolves to the device this rank has made current.

A multi-rank **restart** needs no device bookkeeping, but the restart bundle
(see *The segment is the restart granularity* under *Operational notes*) is
rank-local, so *any* multi-rank restart needs `restart="reseed"` in its
settings and a budget for a cold replay buffer. The device rules and the three
`OnPolicySettings.restart` modes are in {ref}`distillation_recipes_guide`
(*Restarting an interrupted run*).

## Non-conservative teachers

Some teachers predict forces from a dedicated head rather than as the negative
gradient of their energy. Such a force field is *non-conservative*: it does not
integrate to a potential energy surface, and its curl need not vanish. That is
a fine trade for a labeling model and a bad one for a model driving long MD, so
distilling a direct-force teacher into a student whose forces *are* an energy
gradient is a core use case here rather than a workaround.

Nothing in the distillation path gates on conservativeness — not the strategy,
not the scorer, not the losses. There is no flag to set. The scorer detaches
every signal it returns, so how the teacher produced a force never reaches the
student's autograd graph; a teacher force is a number in a batch field, exactly
like a DFT force from a dataset.

A conservative student cannot represent the non-conservative part at all. Its
force field is, by construction, minus the gradient of a scalar, and gradient
fields are curl-free. Minimizing a force-matching objective therefore drives the
student toward the curl-free field closest to the teacher's, in the
least-squares sense the loss defines. The non-conservative component is
projected out rather than badly fitted. That is usually what you want, since
it is the component that would have shown up as energy drift in the student's
own dynamics. The force-matching loss is therefore bounded below by the
teacher's non-conservative residual on the training distribution. A nonzero
plateau in force error reflects this irreducible residual rather than poor
optimization.

Two practical consequences follow. Do not expect force-matching error against
a non-conservative teacher to go to zero: the floor is the size of the
projected component, and
{py:func}`~nvalchemi.training.distillation.evaluation.non_conservative_residual`
measures it. And keep a total-energy term in the objective. A force-matching
term only ever sees the gradient of the student's energy, so forces alone fix
the student's energy only up to an additive constant.

## Evaluating the student

Acceptance is a handful of measurements and one verdict formed from them. A
student is accepted when it clears every *acceptance bar* it is given; an
acceptance bar is a limit on one number from the student's measurements, such
as a maximum force MAE. Import the measurements from the `evaluation`
subpackage rather than from the distillation namespace: an acceptance run
pulls in the dynamics engine and the reporting stack that training itself does
not need.

{py:func}`~nvalchemi.training.distillation.evaluation.evaluate_accuracy` scores
a held-out set against the dataset's own labels or against the teacher's,
whether the teacher's labels are on disk or scored on the fly. The quantities
it compares are an open table. Each built-in quantity is an
{py:class}`~nvalchemi.training.distillation.evaluation.AccuracyQuantitySpec` in
`BUILTIN_ACCURACY_QUANTITIES`, which names the prediction and reference keys,
the teacher signal that labels it, and the supervised loss that scores it.
`evaluate_accuracy(quantities=[...])` takes those names, or a spec of your own
for a head the table lacks, as long as one quantity in the set carries a
supervised loss. `label_dtype=` pins the dtype that scored-on-the-fly labels
are cast to. A student behind a data-parallel wrapper is read through
{py:func}`~nvalchemi.training.unwrap_model` when the gradient mode is decided.

Accuracy alone is not what a small student fails at, so stability is measured
on a trajectory the student drives itself. Register a
{py:class}`~nvalchemi.training.distillation.evaluation.StabilityMonitor` on the
propagator, and read `monitor.metrics()` once the run is over. `metrics()` is a
method, not an attribute, and it needs two samples at two different steps. The
monitor has three settings. `divergence` is the predicate that decides which
frames count as diverged (the loop's `nonfinite_divergence` by default), and
the step it first fired on is reported as `first_divergence_step`. `aggregate`
chooses whether the reported drift is the worst graph in the batch (`"max"`,
matching the energy-drift monitor hook) or the mean over graphs (`"mean"`).
`stop_on_composition_change`, on by default, stops the measurement with a
warning when the batch composition changes.

{py:func}`~nvalchemi.training.distillation.evaluation.measure_throughput` times
that same propagator at steady state and reports atoms per second and simulated
nanoseconds per day. Time every student of a family on the same batch, or the
column cannot rank them.
{py:func}`~nvalchemi.training.distillation.evaluation.extensivity_error` checks
that energy still scales with replicated cells, built with the core
{py:func}`~nvalchemi.data.transforms.make_supercell`. Pass `extensive_keys` and
`intensive_keys` through `extensivity_error` for a system field of your own,
since one named in neither is refused rather than guessed at, and `drop_keys=`
for the bookkeeping a sampler-seeded run stamped. Periodicity is read from `pbc`
before the presence of a `cell`. The radial-distribution pair,
`radial_distribution` and `compare_radial_distributions`, compares the structure
a trajectory samples against a reference trajectory's. A student distilled from
a direct-force teacher also needs
{py:func}`~nvalchemi.training.distillation.evaluation.non_conservative_residual`,
which bounds how well any conservative student can fit that teacher (see
*Non-conservative teachers*).

Those measurements go into one
{py:class}`~nvalchemi.training.distillation.evaluation.StudentEvaluation` per
candidate. State the acceptance bars as
{py:class}`~nvalchemi.training.distillation.evaluation.AcceptanceThresholds`
and hand both to
{py:func}`~nvalchemi.training.distillation.evaluation.build_acceptance_report`,
which returns a report that renders as Rich tables, exports as a plain
dictionary, and says whether the student is accepted:

```python
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    measured_bars,
)

evaluation = StudentEvaluation(
    name="small",
    accuracy=evaluate_accuracy(student, holdout, targets="teacher", scorer=teacher),
    stability=monitor.metrics(),
    weights="raw",
)
thresholds = AcceptanceThresholds(
    max_forces_mae=0.05, max_energy_drift_per_atom_per_ns=0.005
)
print(sorted(measured_bars("accuracy", "stability", accuracy_quantities=("energy", "forces"))))
report = build_acceptance_report([evaluation], thresholds)
print(report.accepted)
```

A bar with no measurement behind it **fails** the student rather than being
skipped, so state only the bars the measurements in hand can decide. Ask
{py:func}`~nvalchemi.training.distillation.evaluation.measured_bars` which those
are, instead of restating the mapping in your own script. It takes the
measurement families that were filled, plus the quantities an accuracy pass
actually compared. The from-scratch bar, `max_from_scratch_ratio`, needs a
`baseline_accuracy` scored on the same holdout by an equal-size student trained
from scratch. The ratio is taken on `energy_per_atom_mae`, `forces_mae`, and
`stress_mae`, whichever both carry, and the worst one is kept.

The bars form an open table rather than a closed list. Each is an
{py:class}`~nvalchemi.training.distillation.evaluation.AcceptanceBar` that
names the threshold it is set under, the measurement families it reads, the
check, the accuracy quantities that decide it, and the direction it passes in.
The shipped bars are `DEFAULT_BARS`, and `BAR_FAMILIES` is derived from them.
A bar of your own is a limit on a number you file under
`StudentEvaluation.extra`. Put it in a table handed to
`build_acceptance_report(..., bars=)` and `measured_bars(..., bars=)`, and set
its limit under its name in `AcceptanceThresholds.extra`. Every measurement is
a Pydantic model sharing
{py:class}`~nvalchemi.training.distillation.evaluation.MeasurementRecord`, so
`to_dict` and `from_dict` carry those open slots along with the built-in
fields. `AccuracyMetrics.errors` holds the quantities the built-in fields do
not name.

`weights` is not a measurement. It records which of the student's two weight
sets the numbers came from, `"ema"` or `"raw"`. Nothing downstream can infer
it, so set it here; two exports of the same student then say which artifact
each one gated on. `None` records nothing, which is not the same as `"raw"`.
`evaluate_accuracy` scores exactly the object it is handed, with no EMA swap in
either direction. A student trained under an
{py:class}`~nvalchemi.training.hooks.EMAHook` therefore has to be handed over
as `strategy.inference_model["student"]` and recorded as `"ema"`; passing
`strategy.models["student"]` gates on weights that will not ship. The
`inference_model` slot survives no checkpoint. A reloaded strategy holds
nothing to score there until it dispatches `SETUP`, through the public
{py:meth}`~nvalchemi.training.TrainingStrategy.run_setup_hooks`, which
republishes the averaged copy from the re-attached hook.

The CLI covers the accuracy half of this. `distill evaluate` scores a recipe's
holdout on the weights `--weights` names. `auto`, the default, reads the EMA
average when `student.hooks` carries an `EMAHook` and the trained weights
otherwise; `ema` fails when the checkpoint holds no average; and `raw` scores
the trained weights regardless. The command applies the accuracy bars the
recipe carries, prints which weights it scored, records that same marker as
the entry's `weights`, and exits non-zero on a missed bar. Drift, speed,
extensivity, the RDF, and the from-scratch baseline stay on the Python path
above, because no recipe names a propagator, a supercell builder, or a second
trained model. See {ref}`distillation_recipes_guide`.

Read `StabilityMetrics.energy_fluctuation_per_atom` and
`max_energy_excursion_per_atom` beside a drift number, rather than reading the
drift alone. A drift rate is the slope of a least-squares line. The
fluctuation is the RMS residual about that fit, so a drift no larger than the
fluctuation is a line drawn through an oscillation rather than a trend. The
excursion says how far the series went in the meantime. Neither is a bar: the
stability family gates on `max_energy_drift_per_atom_per_ns`,
`max_energy_drift_per_atom_per_step`, and `max_momentum_drift` alone.

The radial-distribution comparison is continuous in the positions. Pairs are
deposited cloud-in-cell into bins, from a neighbor list built one bin past
`r_max`. A rigid translation of a crystal therefore scores a Jensen-Shannon
divergence at round-off, where a hard-edged histogram reports a few times
`1e-2`. That is what makes the metric usable on relaxed and crystalline
frames. `r_max` may exceed half the shortest cell vector.

(custom-components)=

## Custom components

Six extension points of the distillation loop are protocols rather than base
classes. A component of your own is any object satisfying the protocol; the
{ref}`training-distillation-api` reference documents the distillation ones in
full, and {doc}`/modules/dynamics/api` the core ones.

- {py:class}`~nvalchemi.training.distillation.TeacherScorer` — `signals` and
  `label(batch)` returning `{teacher_field: (detached tensor, level)}`; declare
  `label_fields` so the fields you write are known before the first batch. A
  teacher output the built-in table lacks rarely needs a scorer of its own: a
  {py:class}`~nvalchemi.training.distillation.TeacherSignal` passed to the
  in-process scorer covers it. A scorer travels in a recipe under `scorer_cls`
  when it satisfies
  {py:class}`~nvalchemi.training.distillation.SpecSerializable`: a
  `to_spec_dict()` and a `from_spec_dict()` classmethod. Without them it stays
  runtime-only, and `OnPolicyConfig.to_spec_dict` refuses it with an error
  naming that remedy.
- `initial_structures` — any object satisfying the core
  {py:class}`~nvalchemi.dynamics.StructureSource` protocol (see
  {doc}`/modules/dynamics/api`), importable from the distillation package as
  `InitialStructuresSource`. A source driving a relaxation lifecycle stamps
  `status` zeros and `system_id`s on the batch it hands over, as the core
  {py:class}`~nvalchemi.dynamics.OrderedStructureSampler` does. A recipe names
  a source through `to_spec_dict()` / `from_spec_dict()`; a source without them
  stays runtime-only.
- The `fits=` policy of a draw you drive yourself — any object satisfying the
  core {py:class}`~nvalchemi.dynamics.FitPolicy` protocol;
  {py:class}`~nvalchemi.dynamics.WithinBudget` is the stock one. Both are
  re-exported from the distillation package.
- {py:class}`~nvalchemi.training.distillation.AdmissionPolicy` — a callable
  over a batch of captured frames returning one boolean per graph; frames it
  refuses never enter the replay buffer.
- {py:class}`~nvalchemi.training.distillation.EvictionPolicy` —
  `select(buffer, incoming, capacity)` returning the indices into the resident
  batch (oldest first, the admitted frames last) to drop, at least as many as
  the buffer is over capacity by;
  {py:class}`~nvalchemi.training.distillation.FIFO` is the reference.
- `capture_sink` — any {py:class}`~nvalchemi.dynamics.sinks.DataSink`. The
  loop sizes it to `(generation_steps + 1)` frames per trajectory, growing it
  through `resize(capacity)` when the sink also satisfies the core
  {py:class}`~nvalchemi.dynamics.ResizableSink` protocol (see
  {doc}`/modules/dynamics/api`); a smaller sink that does not is refused.
  {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` is the in-tree
  device-resident one.

Minimal implementations of three of these protocols follow, wired into one loop
with a device-resident capture sink:

```python
import torch

from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.dynamics.sinks import GPUBuffer
from nvalchemi.training.distillation import OnPolicyConfig, TeacherScorer


class TabulatedScorer:
    signals = frozenset({"energy"})
    label_fields = ("teacher_energy",)

    def label(self, batch):
        return {"teacher_energy": (torch.zeros(batch.num_graphs, 1), "system")}


def finite_labels(frames):
    return torch.isfinite(frames.teacher_energy.view(-1))


class DropNewest:
    def select(self, buffer, incoming, capacity):
        return torch.arange(capacity, buffer.num_graphs, device=buffer.device)


assert isinstance(TabulatedScorer(), TeacherScorer)
config = OnPolicyConfig(
    dynamics=NVTLangevin(student, dt=0.5, temperature=300.0),
    teacher_scorer=TabulatedScorer(),
    initial_structures=dataset,
    capture_sink=GPUBuffer(capacity=4096, max_atoms=64, max_edges=0, device="cuda"),
    replay_admission=finite_labels,
    replay_eviction=DropNewest(),
    replay_ratio=1.0,
    training_steps_per_segment=32,
)
```

## Operational notes

**Validation data is labeled on the fly too.** On-the-fly labeling runs on
`BEFORE_FORWARD`, a stage that both the training loop and the validation loop
dispatch on the device-placed batch. Unlabeled validation data therefore needs
no preparation, and a caller-supplied `training_fn` is covered too. Pointing
`validation_config` at a store written by `label_dataset` still avoids the
teacher pass entirely. Wrap that store in a
{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader`, or any iterable of
`Batch`, before handing it to
{py:class}`~nvalchemi.training.ValidationConfig`; a bare `Dataset` iterates
`(AtomicData, metadata)` pairs rather than batches. `every_n_epochs=1`
validates at every segment boundary, and `every_n_steps` fires inside
segments.

**Composed weights are ratios, not coefficients.**
{py:class}`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
default, so the three-term objective above runs at `1/2.2`, `1/2.2`, and
`0.2/2.2`. `normalize_weights=False` keeps them literal, which also stops a
weight schedule on one term from rescaling the others (see {ref}`losses_guide`).

**Label dtype follows the student, floored at single precision, unless
`label_dtype` says otherwise.** By default, teacher labels are cast to the
dtype of the student's first floating-point parameter, so a float64 teacher
feeds a float32 student without a dtype error at the loss. The cast never goes
below float32. A store hands every floating field back at the dtype of the
dataset's `positions`, and a narrower label would disagree with what
`label_dataset` persisted. A `bfloat16` or `float16` student therefore needs
`dtype_policy="prediction_to_target"` on its loss terms. A float64 student
training from a store needs `"target_to_prediction"`, which widens the float32
labels a store returns. The cast is resolved at construction, so a student
whose dtype changes afterwards also needs a `dtype_policy`, or an explicit
`DistillationStrategy(label_dtype=torch.float64)`, which names the dtype
outright. A non-floating `label_dtype` is refused.

**The teacher is stored once per checkpoint root.** The first write under a
root holds the frozen teacher's weights, every later
{py:class}`~nvalchemi.training.hooks.CheckpointHook` write records a reference
to that index plus a fingerprint, and a load verifies the stored copy, so a
periodic checkpoint costs the student's weights alone. One root holds one
teacher; the manifest layout, the fingerprint, and the repair rule are in
{ref}`distillation_recipes_guide` (*Teacher checkpoints*).

**The segment loop round-trips as references.**
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
carries `on_policy` and `reference_dataset` inline, with the propagator, the
scorer, and every dataset named as the references they rebuild from and every
runtime-only field omitted with a warning. The field-by-field table, the
refusals, and the rebuild order in which a live object passed to
`from_spec_dict` or `load_checkpoint` outranks the recipe are in
{ref}`distillation_recipes_guide` (*Serializable versus runtime-only*).

**The segment is the restart granularity.** An interrupted on-policy run
carries a *restart bundle* through the checkpoint: the part of an on-policy
checkpoint that lets a resumed run continue its trajectories. The bundle holds
the
propagator's `dynamics_step_count`, the live `trajectory` batch, the
`replay_frames` it had filled, the `settings` it ran under, the
`initial_structures` position, and whether generation was exhausted. A resumed
run therefore continues the same trajectory rather than starting a fresh one.
The restored frames *replace* the buffer's contents rather than merging into
them, the backfill picks up at the row the interrupted run had reached, and a
setting the resumed loop sets differently is reported. The position is the
next row, its wrap count, the next `system_id`, and the rank shard the three
were counted in. A bundle written for another shard is refused rather than
replayed against the wrong rows. The bundle does not carry RNG state, the
neighbor tensors, or FIRE's adaptive state. Only a counter-based-RNG
integrator therefore reproduces its stream exactly, and a resumed relaxation
re-initializes its optimizer history.

A segment that a checkpoint interrupted part-way is counted as finished on the
way in. Its `AFTER_EPOCH` hooks never fire, the batches it had left are not
replayed, and the run opens a fresh segment, which begins by generating. An
exhausted run resumes exhausted: the bundle carries the frames and the
exhaustion rather than a trajectory, and the resumed run trains on the buffer
without regenerating. The buffer also outlives a run. A second `run()` on one
strategy appends to the frames the first run filled, but it reseeds its own
trajectory, because installing the rank shard rewinds the position. Only a
restart bundle resumes a trajectory.

Resuming an on-policy run has two routes. The first is
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.load_checkpoint`,
which rebuilds the segment loop from the recipe the checkpoint carries. In the
common case it returns an on-policy strategy that runs without a dataloader.
Pass `models=` so the propagator is rebound to the very student the optimizer
updates, and `on_policy=` / `reference_dataset=` to override a piece the recipe
could not name. Only a run whose recipe left out the `on_policy` entry comes
back offline-shaped, and its `run()` then rejects the `None` dataloader. The
second route is to rebuild the strategy with the same propagator, scorer,
reference dataset, and hooks, then restore the counters, weights, optimizer
state, and checkpointable hook state into it in place with
{py:meth}`~nvalchemi.training.TrainingStrategy.restore_checkpoint`. Unlike
`load_checkpoint`, it takes no `hooks` override, because it restores into the
hooks the rebuilt strategy already holds:

```python
from nvalchemi.training.hooks import CheckpointHook

strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs=optimizer_configs,
    loss_fn=loss_fn,
    num_steps=20,
    on_policy=on_policy,
    reference_dataset=reference_dataset,
    hooks=[CheckpointHook("runs/on_policy/checkpoints", step_interval=3)],
)
strategy.restore_checkpoint("runs/on_policy/checkpoints")
strategy.run()
```

Attaching `on_policy` to an already-loaded strategy is not a substitute.
Assignment is not validated, so the propagator would keep a student the
optimizer never updates, and the run would silently stop being on-policy.
`num_steps` is an absolute target rather than a budget for the resumed leg, so
a run that already reached it resumes to nothing until the target is raised.
From the CLI, `spec resume --budget` says whose `num_steps` sizes the
continued run. `checkpoint`, the default, keeps the budget the checkpoint's
stored spec recorded. `recipe` takes the edited recipe's budget, so a recipe
can extend or shorten a run; a budget below what the checkpoint completed, or
one in the other unit, is refused. Earlier versions let the recipe's budget
win by default. `spec run` and `spec resume` also take the training CLI's
loader options, shared through `nvalchemi.training.cli_common`.

```{note}
**Reserved and runtime-only settings.** `replay_eviction` admits one spelling
in a recipe, `"fifo"`; a custom
{py:class}`~nvalchemi.training.distillation.EvictionPolicy` instance rides on
`OnPolicyConfig` alone and is recorded as `"fifo"` with a warning, like the
other runtime-only fields listed in {ref}`distillation_recipes_guide`.
`weight_sync_frequency` must be `1`: the propagator and the trainer share one
module object, so an eager run is never out of sync, and the setting becomes
meaningful only once the propagator holds a compiled or remote copy of the
student.
```

## API reference

See {ref}`training-distillation-api` for the reference documentation of
{py:class}`~nvalchemi.training.distillation.DistillationStrategy`,
{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer`,
{py:func}`~nvalchemi.training.distillation.label_dataset`,
{py:class}`~nvalchemi.training.distillation.OnPolicyConfig`,
{py:class}`~nvalchemi.training.distillation.OnPolicySettings`,
{py:class}`~nvalchemi.training.distillation.SpecSerializable`,
{py:func}`~nvalchemi.training.distillation.nonfinite_divergence` (an alias of
the core {py:func}`~nvalchemi.dynamics.hooks.nonfinite_graph_mask`),
{py:class}`~nvalchemi.training.distillation.InitialStructures`,
{py:class}`~nvalchemi.training.distillation.TeacherLabelHook`,
{py:class}`~nvalchemi.training.distillation.ReplayBuffer`,
{py:class}`~nvalchemi.training.distillation.AdmissionPolicy`,
{py:class}`~nvalchemi.training.distillation.EvictionPolicy`,
{py:class}`~nvalchemi.training.distillation.FIFO`,
{py:class}`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss`,
{py:class}`~nvalchemi.training.distillation.EmbeddingMatchingLoss`,
{py:class}`~nvalchemi.training.distillation.HessianMatchingLoss`, and
{py:class}`~nvalchemi.training.distillation.BoltzmannMatchingLoss`. It also
covers the core {py:class}`~nvalchemi.dynamics.OrderedStructureSampler` and
{py:class}`~nvalchemi.dynamics.StructureSource`, and the core
{py:class}`~nvalchemi.dynamics.FitPolicy`,
{py:class}`~nvalchemi.dynamics.WithinBudget`, and
{py:class}`~nvalchemi.dynamics.ResizableSink` that the distillation package
re-exports.

{ref}`distillation_recipes_guide` covers the JSON recipe and the `distill` CLI
that author, run, resume, and gate the runs this guide describes.
