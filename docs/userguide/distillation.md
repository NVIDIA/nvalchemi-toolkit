(distillation_guide)=

# Distilling a Teacher Into a Student

Knowledge distillation trains a small model — the *student* — to reproduce the
predictions of a larger, frozen one — the *teacher*. For interatomic potentials
the motivation is throughput: a foundation teacher may be far too expensive to
drive long molecular dynamics, while a student that reproduces its energies and
forces on the states that matter runs orders of magnitude faster. The teacher
also removes the usual data bottleneck, because it can label any structure,
including ones no reference calculation was ever run on.

{py:class}`~nvalchemi.training.distillation.DistillationStrategy` is the entry
point. It is a {py:class}`~nvalchemi.training.TrainingStrategy` subclass, so
everything in {ref}`training_guide` — optimizers, schedulers, validation, hooks,
checkpoints — applies unchanged. The segment loop reads a few of those concepts
its own way, and this guide says so where it matters: one segment is one epoch,
and resuming an on-policy run has two routes with different guarantees. The
rest of it covers what distillation adds. The JSON recipe and the `distill`
CLI that run these workflows end to end are in {ref}`distillation_recipes_guide`;
the symbols are in {ref}`training-distillation-api`.

This guide assumes that you already have:

- a teacher wrapped with {py:class}`~nvalchemi.models.base.BaseModelMixin`;
- a student that is trainable and declares the outputs your objective reads;
- a dataset of structures, with or without reference labels.

For those prerequisites, see {ref}`models_guide`, {ref}`datapipes_guide`, and
{ref}`training_guide`.

## The shape of a distillation run

`DistillationStrategy` takes a named-model mapping holding `"student"` and
`"teacher"`. The teacher is **frozen by omission**: it must not appear in
`optimizer_configs`, and that absence is what puts it in evaluation mode with
gradients disabled for the duration of the run. The student — and any auxiliary
model, such as a learned projection — must be given an optimizer config. The
strategy raises at construction if either half of that contract is broken.

Teacher knowledge reaches the loss as ordinary batch fields. Every signal the
teacher produces populates one `teacher_*` field, and a loss term consumes it by
pointing its `target_key` there:

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
distills by naming a teacher field, and mixing teacher targets with reference
targets in one objective is ordinary loss composition — offline, where every
sample carries its own reference labels. An on-policy run cannot, for reasons
covered below. The signals available, and the field and shape each one lands
as, are:

| Signal | Batch field | Level | Shape |
| --- | --- | --- | --- |
| `energy` | `teacher_energy` | system | `(B, 1)` |
| `forces` | `teacher_forces` | node | `(V, 3)` |
| `stress` | `teacher_stress` | system | `(B, 3, 3)` |
| `atomic_energies` | `teacher_atomic_energies` | node | `(V,)` |
| `embeddings` | `teacher_node_embeddings` | node | `(V, D)` |
| `hessian` | `teacher_hvp`, with `teacher_hvp_probe` | node | `(V, 3)` |

The first four come from the teacher's forward pass and share the name of the
output that has to appear in the teacher's `ModelConfig.outputs`, which is the
name the construction check reports as missing. `teacher_node_embeddings` comes
from {py:meth}`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, which
costs a second pass, and `teacher_hvp` from a Hessian-vector product along a
random probe direction the scorer stores beside it in `teacher_hvp_probe`.

`embeddings` and `hessian` are the two signals the stock training function
cannot supervise on its own, because neither has a student-side counterpart in
a plain forward pass. Each has its own training function instead —
{py:func}`~nvalchemi.training.distillation.embedding_distillation_fn` and
{py:func}`~nvalchemi.training.distillation.hessian_distillation_fn`, both
described under *Objectives beyond pointwise matching* — and a loss component
whose prediction key is an embedding is refused at construction under the stock
one, with that instruction rather than with the generic missing-output message.

You do not normally declare which signals you want. `teacher_signals=None`, the
default, derives the set from the `teacher_*` targets the losses read — the
training loss and, when `validation_config` carries a `loss_fn` of its own, the
validation loss too — so objective and teacher cannot drift apart; an explicit
set must cover the derived one and may request more. The resolved set is
checked against the teacher's declared `outputs` at construction, and every
loss component's prediction key is checked against the outputs the student
*actually computes* — its `active_outputs` narrowed to its declared ones — so a
pretrained wrapper whose active set was narrowed is caught here rather than on
its first batch. Neither check re-runs on assignment, so pass
`validation_config` to the constructor, or name the wider set in
`teacher_signals`.

Every resolved signal is a request for its fields on every batch. A batch
counts as labeled only when it holds every resolved field, so a validation loss
reading a `teacher_*` target the training loss does not puts a store labeled
without that field back on the teacher, batch after batch. A store meant to
train with no teacher pass at all has to be labeled with the same signal set
the strategy resolves. A `teacher_*` target that names no built-in signal — a
field a custom scorer persisted — is an ordinary loss target: it is read off
the batch as it arrives, never derived into a signal and never attached on the
fly, so a batch lacking it surfaces as a missing loss target.

The training function stays a plain student forward pass. The default,
{py:func}`~nvalchemi.training.distillation.default_distillation_fn`, calls the
student and prefixes every output with `predicted_`. The teacher is never called
there, which is why the teacher can never enter the student's autograd graph and
why the recipe survives serialization.

{py:class}`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss` is the one
term distillation adds. It matches the teacher's per-atom energy decomposition,
a quantity no reference dataset carries. Per-atom energies are not physically
observable on their own, so treat the term as a regularizer on the student's
internal decomposition and keep a total-energy term weighted above it. It asks
the most of both models, because the decomposition has to exist on each side:
the teacher must declare `atomic_energies` in `ModelConfig.outputs` for the
signal to be scorable, and the student must both declare *and* compute it,
since the term reads `prediction_key="predicted_atomic_energies"`. A student
that emits only `energy` and `forces` — including
{py:class}`~nvalchemi.models.demo.DemoModelWrapper` — therefore fails at
construction on the three-term objective above, naming the loss component and
the missing `atomic_energies`.

### Objectives beyond pointwise matching

Three further terms distill what a pointwise target cannot carry, and each asks
the run for something a plain forward pass does not produce. The weighting and
the physics behind each are in {ref}`distillation-advanced-objectives`; this
section is what the run has to provide.

{py:class}`~nvalchemi.training.distillation.EmbeddingMatchingLoss` matches the
teacher's per-atom representation rather than a prediction. Both sides come
from `compute_embeddings`, so the objective needs
{py:func}`~nvalchemi.training.distillation.embedding_distillation_fn` as the
`training_fn`, and the student is run twice per batch. Widths rarely agree
across architectures, so register an
{py:class}`~nvalchemi.training.distillation.EmbeddingProjector` under the model
name `"projector"` and give it an optimizer config: the training function routes
the student's embeddings through it, and the strategy checks at construction
that the student's width, the projector's `in_features`/`out_features`, and the
teacher's width compose. A student that publishes no `node_embeddings` shape is
refused there, and so is one whose `compute_embeddings` returns embeddings
detached from its trainable parameters — a projector would then absorb the
whole objective while the student learned nothing from it.

The other two score a *batch* rather than a sample.
{py:class}`~nvalchemi.training.distillation.BoltzmannMatchingLoss` matches the
Boltzmann weights the two energy surfaces imply over the batch at a
`temperature`, in Kelvin; `beta` is not an inverse temperature but the
interpolation between the forward (`0`) and reverse (`1`) relative entropy. The
forward direction is bounded by `log B` over the batch's `B` scorable graphs and
its gradient vanishes once the softmax saturates, which a student whose error
spreads over more than a few `k_B T` already does, so `beta=0` can read as
converged while the student is far off: hold `beta` at `0.5` or above until the
student is within a couple of `k_B T`. The term reads a batch as a sample of the
student's own Boltzmann distribution, so it requires `on_policy`, refuses a
relaxation propagator and any convergence criterion, and warns when
`replay_ratio` mixes in reference frames the student never visited; the
recommended shape is `replay_ratio=1` with a bounded `replay_capacity`. It is
refused on the validation side — in a validation `loss_fn`, or in a validation
config with no `loss_fn` of its own reusing a training loss that holds one —
because a fixed holdout is not a sample of the student's distribution. Under a
{py:class}`~nvalchemi.training.hooks.DDPHook` the softmax runs over the world
batch: the reduced energies are gathered across ranks differentiably, so every
rank reports the world loss. A batch of one graph scores exactly `0.0`.

{py:class}`~nvalchemi.training.distillation.HessianMatchingLoss` matches the
teacher's curvature along a random probe direction, with the student's product
taken by {py:func}`~nvalchemi.training.distillation.hessian_distillation_fn` on a
second, energy-only pass that reuses the neighbor list the stock forward just
ran on. The graph-balanced value carries the square of a force constant's units
and runs one to two orders of magnitude above a force mean-squared error for a
near-converged student, so start it a hundred to ten thousand times lighter
than the force term and read one batch's value as the one-sample estimate it
is. Weight it on the composition rather than on the term, since leaves are
weightless and {py:class}`~nvalchemi.training.ComposedLossFunction`
renormalizes by default. Pointing a loss at the companion `teacher_hvp_probe`
is refused, since a probe is not a quantity the student is supervised against,
and a student whose forces come from a head of their own is warned that the
term supervises its energy head alone.

The Boltzmann term also changes what a restart needs. Because it is defined on
generated batches, it refuses to be rebuilt without the segment loop, so a
recipe the spec could not carry whole has to be re-supplied at load time, as
`load_checkpoint(root, models=..., on_policy=..., reference_dataset=...)`, or
restored into a strategy already built with them. The curvature term carries no
such requirement: it reads only what the student computes on the batch in hand.

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

{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer` owns the
teacher's evaluation contract so callers do not have to. It narrows the
teacher's `active_outputs` to what the requested signals need, reuses the
batch's neighbor list only when it is a known full list at the teacher's own
cutoff and format and otherwise builds one and rolls it back, picks the grad
mode a teacher with autograd outputs requires, detaches every tensor it
returns, and normalizes each signal to the canonical shape above. The batch it
is handed is left exactly as it was found, which is what makes the same scorer
usable mid-training and mid-trajectory. `dtype=` stores labels at a reduced
dtype; `probe_seed=` pins the Hessian probe direction, and is best left unset
for training and labeling, where coverage comes from redrawing.

The one teacher it refuses is a composition that plans more than one
neighbor-list source: a
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` whose stages'
cutoffs differ by more than `max_cutoff_ratio` under the default
`neighbor_adaptation="auto"`. The scorer builds exactly one list per batch, so
compose such a teacher with `neighbor_adaptation="always"`, or with a
`max_cutoff_ratio` of at least its largest-to-smallest cutoff ratio, and it
adapts that single list per stage.

Labels are attached with `overwrite=True`, so a scorer reaching outside the
`teacher_*` namespace would replace the reference field of that name — the very
label the student is trained against. A scorer's declared `label_fields` is
therefore refused before the first chunk is written, and the fields each chunk
actually returns are refused again per chunk. The same namespace rule holds on
every other labeling route.

Labeling is resumable: an existing store is continued from `len(store)`, with
every resumed chunk checked against the store's fields, levels, dtypes, and row
shapes, and a store longer than the dataset — one written from a different
dataset — or one an interrupted append left inconsistent is refused rather than
resumed from a misaligned offset. The one thing not carried over is the
neighbor list, in either format: the dense tensors cannot append into a
fixed-width store array, and a sparse list is dropped because the cutoff it was
built at is a batch attribute the store does not hold. `keep_neighbors=True`
stores the sparse list anyway, and a store written under one setting and
resumed under the other is refused as field-set drift. Nothing rebuilds a list
on the way back out of the store; the next section wires that up.

### Train from the labeled store

Nothing about the consumption path is distillation-specific. The teacher fields
arrive as ordinary batch attributes at the levels they were written at, so
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

Open the store on the device the strategy trains on, and say which one that is:
a `Dataset` opened without `device=` emits on the current CUDA device whenever
one exists, while `DistillationStrategy` — like every `TrainingStrategy` —
defaults to `devices=[torch.device("cpu")]`, and `devices` takes
`torch.device` objects, not strings. `label_dataset` takes the same `device=`,
which moves each chunk onto that device before scoring; put the teacher there
yourself with `teacher.to(device)`, or the labeling pass stops on a device
mismatch.

Nothing in the training loop builds a neighbor list, so a graph student needs
one built for it — labeling drops the neighbor tensors from the store, and a
wrapped MLIP raises rather than building its own. Add a
{py:class}`~nvalchemi.hooks.NeighborListHook` at `BEFORE_FORWARD`, configured
from the student's own neighbor config:

```python
from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import TrainingStage

neighbor_hook = NeighborListHook(
    student.model_config.neighbor_config, stage=TrainingStage.BEFORE_FORWARD
)
```

Pass it in `hooks=[neighbor_hook]`. The internal labeling hook is prepended
ahead of your own, so on-the-fly labeling happens first — which costs nothing
here, because the scorer builds and rolls back the teacher's own list regardless
of what is on the batch. Both examples use neighbor-free demo potentials, so
the hook only becomes necessary when a real MLIP takes the student's place; the
on-policy section shows it beside the propagator-side hook a graph student
needs there as well.

A complete, runnable version of this workflow is
{doc}`/examples/intermediate/09_offline_distillation`.

### Consuming the labeled store elsewhere

The store is a plain Zarr hierarchy, so the labels are not locked to this
toolkit. Each field is one array under the store's `core` group —
`teacher_energy` and `teacher_forces` sit next to `positions` and
`atomic_numbers` — per-atom arrays are concatenated across samples with the
per-sample offsets in `meta/atoms_ptr`, and the store's root attributes record
whether each field is per-atom or per-system. Any Zarr client can therefore
read a labeled store, so a student written in another framework consumes the
same teacher labels without running the teacher again, and without importing
`nvalchemi` at all.

### Labeling on the fly

A batch that arrives without the required `teacher_*` fields is labeled on the
fly instead, by an internal hook the strategy registers ahead of your own on
`BEFORE_FORWARD`. That keeps short runs and interactive sessions working with no
labeling pass at all, and it is why unlabeled validation data needs no
preparation. Set `label_missing=False` to turn it off, in which case an
unlabeled batch surfaces as a missing loss target.

On-the-fly labels are attached to the device-placed batch the strategy trains
on, which is a copy of the one you handed over, so they do not persist on your
object. A loader that replays the same systems every epoch therefore pays one
teacher pass per epoch — the reason a long run should label offline first. The
teacher runs with autocast disabled whatever precision context surrounds it, so
an on-the-fly label is bit-for-bit the label `label_dataset` would have
written, which is what makes the two paths interchangeable.

## On-policy distillation

Offline distillation trains the student on whatever structures the dataset
happens to hold. Those are not the structures the student will visit once it is
driving dynamics itself, and the gap between the two is what makes a distilled
potential drift or blow up on long trajectories. On-policy distillation closes
it: the student's own propagator generates frames, the teacher labels them, and
the student trains on them.

Setting `on_policy` turns
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.run` into a
segment loop that takes no dataloader, because each segment builds its own. One
segment is three phases:

1. **Generate** — the propagator advances the live state batch by
   `generation_steps`, started on the first segment from `initial_structures`.
2. **Label and capture** — a
   {py:class}`~nvalchemi.training.distillation.TeacherLabelHook` on the
   propagator scores every `label_frequency` steps and mirrors each labeled
   frame into the capture sink — host memory unless `capture_sink` names
   another {py:class}`~nvalchemi.dynamics.sinks.DataSink`, a
   {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` to stay on the device; the
   segment's final frame is labeled too, then the sink is drained into a
   {py:class}`~nvalchemi.training.distillation.ReplayBuffer`.
3. **Train** — a freshly built mixed loader draws `training_steps_per_segment`
   batches at the configured ratio, each going through the ordinary per-batch
   stages.

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
either directly or composed into a larger model, and that object identity is
checked at construction. It is the whole point of the loop: because the trainer
and the propagator share one module, every optimizer step is immediately
visible to the next generated frame. The run is sized in optimizer steps rather
than epochs, since each segment builds its own loader; one segment counts as
one epoch for hooks and epoch-cadence validation, while a step-cadence
validation fires inside segments. The run closes with one terminal validation,
skipped when a cadence already validated at the final step, so a metric-driven
scheduler is never stepped twice on one set of metrics. A propagator that
merely *composes* the student is moved whole to the generation device and held
in evaluation mode for the whole loop, because the training phase forwards
`models["student"]` rather than the composition.

The loop is data-parallel across ranks: each one propagates its own strided
shard of the initial structures, labels those frames with its own teacher
replica, and fills its own replay buffer, with the student's gradients
all-reduced by a {py:class}`~nvalchemi.training.hooks.DDPHook`. What a
multi-rank run asks of the settings is covered below, under *Scaling the
segment loop out*.

The propagator is any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` — an
integrator generating trajectories, or an optimizer generating relaxation
paths. Nothing downstream of the config reads a velocity or a temperature.

Initial structures have to carry the fields the propagator updates in place.
Building the `OnPolicyConfig` loads one row from `initial_structures` and
checks it against the propagator's `__provides_keys__`, so a missing field is
a construction error named against the propagator rather than an
`AttributeError` on the first step. The model outputs its `__needs_keys__`
names — `forces` for every integrator and optimizer, plus `stress` for NPT,
NPH, and the variable-cell FIRE optimizers — need no supplying: a propagator
primes them with one `compute` before its first step. For the built-ins the
check comes to `velocities` and `atomic_masses`, which
{py:class}`~nvalchemi.data.AtomicData` fills unless a store they were written
to dropped them, plus a `cell` for the variable-cell propagators, because
nothing fills one in for an aperiodic structure.

Initial structures therefore need not differ from reference samples, which
must carry no `energy` or `forces` at all. What they may safely carry is
the *stale* half of a run: a store filled by an earlier relaxation hands back
structures already sitting at their exit status, which the propagator would
read as "already finished" and refuse to move. The source strips that
bookkeeping from the batch it hands over and stamps its own `status` and
`system_id` on it, so starting from a previous run's output is safe without a
cleanup pass.

Initial structures live behind an
{py:class}`~nvalchemi.training.distillation.InitialStructures`, a dataset plus
the one cursor the initial batch, any later backfill, and a restart all read
from — so a structure is propagated once, and a run restored from a checkpoint
picks up where it stopped rather than at row zero. `initial_structures=` takes
any {py:class}`~nvalchemi.training.distillation.InitialStructuresSource` — the
protocol of the members the loop reads, of which `InitialStructures` is the
reference implementation — and a bare dataset handed to it is wrapped in an
unbudgeted source, which is the common case; see
[Custom components](#custom-components) for a source of your own.

An *unbudgeted* source is propagated whole, as a single batch, so it *is* the
set of systems the run generates from — size it to the device. Giving the
source a budget — `InitialStructures(dataset, max_atoms=4096)`, or
`max_batch_size`, or `max_edges` — packs the initial batch first-fit in row
order instead, stopping at the first structure that does not fit and leaving
the remainder in cursor order for the backfill a relaxation lifecycle draws.
Both are one {py:meth}`~nvalchemi.training.distillation.InitialStructures.draw`
call under a {py:class}`~nvalchemi.training.distillation.WithinBudget` policy,
and a policy of your own is any callable of that shape when you drive `draw`
yourself. `max_edges` is honored only
when you set it, because the edges of a live frame are whatever neighbor list
the propagator's hook builds each step, while the count a store reports is
whatever it happened to save.

`label_frequency` is the throughput setting. The teacher is the expensive
model, and a segment that labels every tenth frame costs a tenth of the teacher
passes while still generating every frame at student speed. The cadence is
counted against the propagator's cumulative `step_count`, which carries across
segments, and it is read before the count is incremented, so a frequency `f`
fires at steps `0, f, 2f, ...` while a segment's forced last frame is one step
later. With `generation_steps` a multiple of `label_frequency` — the defaults,
`100` and `100`, among them — the two would land on adjacent frames at every
boundary, so the hook passes over a cadence dispatch on the step right after a
labeled one whenever the frequency is above `1`; a forced label is never passed
over, which keeps an early-exiting segment and the run's final frame labeled.
Under the defaults that leaves one label per trajectory per segment, on its
last frame — plus, in the very first segment, the frame after the first step,
which the cadence lands on at step count zero; `step_count` never resets, so a
later `run()` or a restart does not pay it again.

Size `replay_capacity` with that arithmetic in hand. A segment contributes one
frame per trajectory per labeled step, and FIFO eviction retires whole frames in
arrival order, so a capacity that is not a multiple of the trajectory count
cuts a segment's contribution mid-step and over-represents the trajectories at
the back of the batch in every mixture drawn afterwards. Make it a multiple of
the number of initial structures. What enters the buffer and what leaves it are
the two decisions left to policy: `replay_admission` takes an
{py:class}`~nvalchemi.training.distillation.AdmissionPolicy`, a predicate over
each segment's frames applied before the schema check, and `replay_eviction`
takes the string `"fifo"` — the
{py:class}`~nvalchemi.training.distillation.FIFO` reference, and the one
spelling a recipe carries — or an
{py:class}`~nvalchemi.training.distillation.EvictionPolicy` instance naming the
frames a full buffer drops.

```{note}
Request the same signals on `OnPolicyConfig.teacher_scorer` that the loss
reads. With a reference dataset there is no choice: the generation scorer's
teacher fields and the reference dataset's stored ones are compared for
equality at construction, so a scorer narrower than the dataset is rejected
outright. It constructs only against an equally narrow dataset — and then
every generated frame is scored twice, once during generation and again on its
way into a training step. The strategy warns whenever the generation scorer is
narrower than the loss, reference dataset or not.
```

Any {py:class}`~nvalchemi.training.distillation.TeacherScorer` may drive
generation, not only the in-process one, and a custom scorer is worth declaring
`label_fields` on — the batch fields its `label()` writes. That declaration is
what the checks above read, through
{py:func}`~nvalchemi.training.distillation.scorer_fields`: a scorer with a
signal name of its own and no declaration has fields nothing can know before it
has scored a batch, so the strategy warns that the parity check is deferred to
the first segment's loader, where a mismatch surfaces once a whole generation
phase has been paid for. A `label_fields` entry outside the `teacher_*`
namespace is refused at construction, because the hook must never overwrite the
`energy` and `forces` that drive the propagator's next step. A custom
`teacher_*` field the scorer writes is an ordinary loss target, as offline:
generation writes it onto every captured frame, the reference dataset has to
carry it too — which the parity check enforces — and validation data has to
arrive already carrying it, since nothing labels it on the fly. At least one
built-in `teacher_*` target, or an explicit `teacher_signals`, is still
required alongside it.

A runnable three-segment loop is
{doc}`/examples/intermediate/10_onpolicy_distillation`.

### Graph students need a neighbor list on both engines

Both examples generate with neighbor-free demo potentials, and the recipe above
fails for a student that reads a neighbor list off the batch. The initial batch
comes from a store, and a store holds no neighbor tensors; the construction
check reads the propagator's `__needs_keys__`, which never includes them; and
nothing in a dynamics step builds one. A wrapped MLIP therefore raises on the
first propagator step —
`KeyError: 'neighbor_matrix' required but not found in input data` for a dense
list — before a single frame is generated. The training side has the same gap:
the labeling hook strips the neighbor tensors from every captured frame, and
`label_dataset` drops them from the reference dataset, so the first mixed batch
reaches the student's forward without a list too.

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
config. The teacher needs neither: the scorer builds and rolls back its own
list on every batch it labels, so the student's neighborhoods never reach it.
The propagator's hook is a live collaborator no recipe describes, so a loop
rebuilt from a recipe or a checkpoint starts without it and fails the same loud
way on its first step; register it again on `strategy.on_policy.dynamics`
before `run()`, or restore into a strategy built with it. A student whose
forces are an energy gradient also has to take them with `create_graph=True`
while training, as {py:class}`~nvalchemi.models.demo.DemoModelWrapper` does,
or the training step's backward finds the graph already freed; that is a
wrapping matter rather than a distillation one, see {ref}`models_guide`.

### Relaxation paths need a convergence lifecycle

A relaxation propagator differs from an integrator in one way that matters
here: its trajectories *end*. A structure that reaches its minimum keeps being
propagated by a loop that does not know it has arrived, and the labeling hook
keeps mirroring it, so the replay buffer fills with near-duplicate frames of
the same minimum every `label_frequency` steps. Nothing errors — the run
reports plausible losses over a mixture those duplicates have quietly taken
over. Set `fmax` to give the trajectories an ending:

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

`fmax` is the max-force-norm threshold, compared against the student's forces —
the ones the relaxation itself converges on. A criterion the threshold does not
express is passed whole as `convergence_hook=` instead; the two are one
criterion under two spellings, so setting both is refused. A hook passed whole
has to migrate status — off the `0` the run stamps its structures with, to at
least the propagator's `exit_status` — and fire on every step, because a
structure is captured at the step it converges. Prefer the threshold unless the
criterion genuinely needs a hook: `convergence_hook` is a live object no recipe
describes, while `fmax` travels. Either way
`OnPolicyConfig.convergence_criterion` is the live hook the lifecycle drives,
built once and handed over by identity.

The lifecycle owns graduation and the refill, and it refuses to share either. A
propagator already carrying another status-migrating
{py:class}`~nvalchemi.dynamics.ConvergenceHook` or a `sampler` of its own is
refused at `run()`, and a multi-sub-stage
{py:class}`~nvalchemi.dynamics.FusedStage` is refused when the config is built,
because constructing one registers a migrator on every non-last sub-stage; the
only fused shape the lifecycle accepts is a single sub-stage with no criterion
of its own. Relaxation paths that genuinely need staged dynamics want the
propagator's own lifecycle instead, with `fmax` left unset — a propagator
managing its own convergence keeps its final frames, since graduated frames are
left out of capture only under the managed lifecycle.

With it set, a converged structure freezes where it stopped, is stored once as
the minimum it reached, and graduates out of the active batch at the segment
boundary. Frames then reach the buffer by two routes that partition them: the
labeling hook stores the structures still relaxing, narrowed to those *before*
the teacher pass, so a mostly-converged batch costs a mostly-converged teacher
pass; and the converged ones are labeled in a single teacher pass as their sink
drains onto the buffer's device. A trajectory can also end by diverging: no
criterion accepts a NaN, so a structure whose positions or forces stop being
finite is frozen at `exit_status` on that step, kept out of both routes, and
retired and backfilled at the boundary like a converged one, with one warning
per boundary counting them.

The backfill draws from the same cursor the initial batch was packed from — as
many structures as graduated, within the atoms they held, skipping a row that
does not fit rather than stalling on it. An unbudgeted source has nothing left
to draw, because it started every row it owns, so the batch narrows by one
trajectory per graduation. `InitialStructures(dataset, recycle=True)` wraps the
cursor back to the front instead, so the trajectory count holds and the run
relaxes the same structures again — reloaded from the dataset as it stores
them, not from where the propagator's last frame left them, so the second pass
starts from the same geometries under a fresher student. One draw reaches every
row at most once, so two copies of one structure never enter a batch together.
`recycle` is the source's flag, and only a run managing a lifecycle ever
backfills, so setting it with `fmax` unset is refused at construction. Giving
the source a budget is the other way to keep the batch full: it packs a
first-fit batch and spends the remaining rows on the backfill without serving a
structure twice. When the cursor reaches the end with nothing left and no
`recycle`, the run warns once and spends its remaining training steps on the
frames it already has.

### The mixture ratio

`replay_ratio` — λ — is the fraction of every training batch drawn from
generated frames; the rest comes from `reference_dataset`. It is the setting to
reach for first, because it decides how far the run is allowed to follow its
own trajectory:

- **λ = 1** trains on generated data alone and takes no reference dataset. The
  student is pulled entirely toward wherever its own dynamics go, which is also
  the failure mode: if the trajectory drifts into configurations the teacher
  was never meant to describe, nothing pulls it back. Passing a
  `reference_dataset` anyway is rejected rather than ignored, because it would
  be policed for schema and device and then never sampled.
- **0 < λ < 1** keeps a fixed, teacher-labeled distribution in every batch. The
  reference dataset is the pull: it holds the student on data whose coverage
  you chose, while the generated share keeps closing the gap between the
  training distribution and the one the student actually visits.
- **λ = 0** is offline distillation. The loop rejects it rather than running
  generation whose frames it would never train on — drop `on_policy` and call
  `run(loader)` over the labeled store instead.

The composition is exact per batch rather than an average: with
`replay_ratio=0.25` and `batch_size=16`, every optimizer step sees twelve
reference samples and four generated ones. The achievable granularity is
`1 / batch_size`, so the two settings only mean something together — a ratio
that rounds either source down to zero samples of a batch is rejected at
construction, with the smallest batch size that works named in the error.

The loop rebuilds the mixed loader every segment, because the batch sampler
reads its child dataset lengths once and the buffer grows between segments; if
you drive {py:func}`~nvalchemi.training.distillation.build_mixed_loader`
yourself, do the same, or the newest frames are never sampled. Each rebuilt
sampler keys its generator on `OnPolicyConfig.seed` plus the segment index, so
the reference draw is reproducible across runs without repeating within one.
That setting, not the global `torch` seed, is the mixture's randomness. The
sampler seeds its generator with the sum, so consecutive values overlap by a
shift of one segment, and replicates meant to be independent want values spaced
at least `num_steps // training_steps_per_segment` apart.

### The reference dataset must be teacher-labeled

A mixed batch is one collated `Batch`, and collation is not a merge: it keeps
only the fields *both* sources hold and drops the rest, while a whole level
only one side holds is zero-filled for the other's samples. Either behavior
would be silent, so both are rejected instead — the reference dataset's schema
is compared against the buffer's, on a probe batch drawn from each side.

The schema the reference dataset has to match is the replay-frame contract: the
structure, whatever propagator state travels with it, and the `teacher_*`
labels — with none of the `energy`, `forces`, or `stress` the propagator wrote
on the live frame, which the labeling hook strips on the way into the buffer so
a stored frame never carries the student's self-label under a reference
target's name.

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
{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` is the general lever: it
runs on each sample after device transfer, and a field set to `None` there is
gone from the batch `load_batches` re-forms, because
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

Assigning `None` is the deletion idiom, since `AtomicData` has no `__delitem__`,
and the transform has to return the `(data, metadata)` pair it was handed. Two
settings defeat it: `skip_validation=True` builds the fused batch straight from
raw tensor dicts and never runs the per-sample pipeline, so leave it at its
default here; and the transform has to strip every sample alike, because a
batch whose first sample kept `forces` and whose later ones dropped them fails
collation on a batch-dimension mismatch.

The batch-level equivalent is there when the reference set fits in memory:

```python
from nvalchemi.data import Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, InMemoryDataset


def drop_reference_labels(batch: Batch) -> Batch:
    """Strip the reference labels a generated frame never carries."""
    for key in ("energy", "forces", "stress"):
        if key in batch:
            del batch[key]
    return batch


unlabeled = InMemoryDataset(
    reader=AtomicDataZarrReader("dft.zarr"),
    batch_transforms=[drop_reference_labels],
)
label_dataset(unlabeled, scorer, "reference.zarr", batch_size=64)
reference_dataset = Dataset(reader=AtomicDataZarrReader("reference.zarr"), device="cpu")
```

That one runs once, as the resident batch is materialized, so every chunk
`label_dataset` reads is already stripped; `Batch`, unlike `AtomicData`, does
support `del`. Choose on memory: the streaming form has no ceiling. The third
recipe is to label structures that never carried reference labels at all,
which is what `build_systems` does in
{doc}`/examples/intermediate/10_onpolicy_distillation`.

Two checks enforce all of this, at different seams. At construction, the
`teacher_*` field sets of the two sources are compared for *equality* — a
reference dataset with no teacher labels, one narrower than the generation
scorer, and one wider all fail alike — and the dataset is probed for any field
a generated frame can never carry, which is what catches a store labeled over
an existing reference set before a single teacher pass is paid. What is left
for the first segment's mixed loader is the full field, level, and dtype
comparison against real frames, which surfaces only once a segment has
generated and labeled. That is a whole segment of forward passes with an
expensive teacher, so it is worth knowing the frame schema up front: a stored
frame is whatever the initial structures carry, minus everything run-local —
`energy`, `forces`, `stress`, the neighbor tensors, and the dynamics
bookkeeping (`status`, `system_id`) — plus one field per teacher signal.
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

The names come back as `level.field`, the form the mismatch is reported in, so
strip the level off each one to compare against the reference dataset's bare
`field_names`, or compare them against the same `level.field` names read off a
probe batch drawn from it.

Dtype parity is the part of that comparison that is easy to break by accident.
Collation casts the second part of a mixed batch to the dtype of the first, and
which source leads a chunk is not fixed, so a float64 reference dataset beside
float32 generated frames would change the targets' precision from chunk to
chunk; the loader rejects the pair instead. The trap is that the two sides do
not see the same labels even from one scorer: a store hands every floating
field back at the dtype of the dataset's `positions` — float32 for essentially
every dataset — while a generated frame keeps whatever the generation scorer
emitted. Build that scorer with an explicit `dtype` matching what the store
returns, and label the reference dataset with the same scorer. The example
needs no cast only because its teacher already computes at float32.

Supervising one batch from teacher labels and reference labels at once is
masked-composition work that is not modeled yet, so an on-policy run is
supervised by the teacher throughout; annealing between teacher and reference
targets with a {py:class}`~nvalchemi.training.losses.base.LossWeightSchedule`
is an offline technique.

Both mixture sources are collated before the strategy moves the batch, so the
reference dataset's device pins the buffer to it: leaving `replay_device` unset
stages generated frames there, and naming a different one is rejected at
construction rather than discovered as a cross-device collation failure
mid-run. That device is the one the dataset actually *emits* on — the `device=`
a Zarr-backed {py:class}`~nvalchemi.data.datapipes.dataset.Dataset` was opened
with, the device an `InMemoryDataset` holds its batch on, and otherwise a probe
batch's, since a
{py:class}`~nvalchemi.data.datapipes.multidataset.MultiDataset` declares no
device and a store opened without one declares an index-less `cuda`. The way
to move the mixture is therefore to open the reference dataset on the device
the run trains on. `replay_device` decides anything only in a run with no
reference dataset, where the frames stay in host memory unless it says
otherwise.

### Scaling the segment loop out

The runbook — the `DDPHook`, the `torchrun` lines, and the multi-node
rendezvous — is in {ref}`distillation-scaling-out`; this section is what the
segment loop asks of the settings once the world is larger than one.

Initial structures are dealt out *strided*, by
{py:meth}`~nvalchemi.training.distillation.InitialStructures.shard`: rank `r`
takes every `world_size`-th structure from offset `r`, so the shards are
disjoint, cover the dataset, and differ by at most one structure. Those rows
are the whole of what that rank may propagate — public as
{py:attr}`~nvalchemi.training.distillation.DistillationStrategy.structure_shard`
— and the cursor counts positions in them rather than rows of the dataset, so
the initial batch and every later backfill draw from the rank's own shard
alone. `system_id` is not a position in that shard: ids number the trajectories
a rank has started, so under `recycle` they keep climbing while the cursor
wraps, and each rank hands them out from its own base.

Two consequences follow from the deal. A dataset holding fewer structures than
there are ranks is refused, and one that does not divide evenly is warned about
rather than refused — every rank draws the same number of replay samples per
batch from a buffer holding only its own trajectories, and the gradients are
averaged rank by rank, so a frame generated on a shorter shard reaches the
optimizer with more weight. Size the dataset as a whole *multiple* of the world
size. And because the deal strides by index rather than by size, it balances
the count and not the work: sorting the dataset by atom count makes the strided
deal balance both.

The world *divides* the generation work rather than multiplying it: a segment's
aggregate frame count — and the teacher bill paying for it — is what the
single-process run produced, with each rank contributing its `1/world_size`
share. `generation_steps`, `label_frequency`, and `replay_capacity` are all per
rank, so at a fixed `replay_capacity` each rank's buffer spans `world_size`
times as many segments before FIFO eviction reaches back, and every mixed batch
grows staler as the world grows. Raise `generation_steps` or the structure
count alongside the world, or lower `replay_capacity` by the world size — not
both, or the buffer spans `1/world_size` of the history the single-process run
had.

Sharding separates the structures; it does not separate the randomness on its
own. Both seeded streams the loop owns — the mixture sampler's
`OnPolicyConfig.seed` and every integer seed the propagator exposes, a
composition's sub-stages included — are moved onto a per-rank stride of the
seed space. A stage exposing a {py:class}`torch.Generator` and no integer seed
is named in a warning from every rank before the first segment, stays on the
shared stream, and needs a rank-distinct seed from you. That matters most when
the initial structures are replicas of one geometry — how a run asks for one
trajectory per rank — because sharding separates nothing there, and an unmoved
stage makes every rank generate identical frames billed once per copy.

A multi-rank launch requires a synchronized student: after `SETUP`, something
has to own `models["student"]` the way
{py:func}`~nvalchemi.training.unwrap_model` reads it, which a `DDPHook` does
and a gradient-synchronizing wrapper of your own does too. A launch that leaves
the bare student registered is refused, since each rank would otherwise train a
private student and only rank zero's would be checkpointed.

Where the reference dataset sits decides where every rank collates. A
{py:class}`~nvalchemi.data.datapipes.dataset.Dataset` opened with no `device`
or with an index-less `"cuda"` draws its first batch after the hook has pinned
the rank and lands on that rank's own GPU, which is per-rank correct with no
move. An eager `.to("cuda:0")` before the pin concentrates every rank's
reference batches — and with them the whole world's replay frames — on GPU 0,
and is reported from every rank; moving a host-memory dataset in a `SETUP`
hook onto `ctx.workflow.devices[0]` places it correctly. Host memory is the
safe placement to reach for, and `OnPolicyConfig(replay_device="cuda")`,
spelled index-less, resolves to the device this rank has made current.

A multi-rank **restart** needs no device bookkeeping.
{py:meth}`~nvalchemi.training.TrainingStrategy.restore_checkpoint` loads onto
the strategy's live `devices` and `run()` re-homes the optimizer state after
the hook has pinned the rank; `load_checkpoint(root, map_location=...)` names
the device once, and a multi-rank `spec resume` defaults `--map-location` to
this rank's device. What such a restart forfeits is the generation state: the
restart bundle is rank-local, written by rank zero alone, so *any* multi-rank
restart — a two-rank run resuming on two ranks included — drops it with a
warning, and every rank reseeds its trajectory from its own shard with a cold
replay buffer. Weights, optimizer state, and the step counters come back as
they always do. Budget the first segments after such a restart accordingly.

Finally, a desynchronized world does not fail fast. A rank that stalls or raises
while the `DDPHook` owns the process group leaves a live job that never
advances, and the hook exposes no process-group timeout of its own, so bound
the wait by initializing the process group before the run with an explicit
`timeout=`; the hook then finds communication already established and leaves
it alone.

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

What happens to the non-conservative part decides what a conservative student
can and cannot fit. Such a student cannot represent that part at all: its force
field is, by construction, minus the gradient of a scalar, and gradient fields
are curl-free. Minimizing a force-matching objective therefore drives the
student toward the closest curl-free field to the teacher's, in the
least-squares sense the loss defines; the non-conservative component is
projected out rather than badly fitted, which is usually what you want, since
it is the component that would have shown up as energy drift in the student's
own dynamics. What it leaves behind is a floor rather than a verdict: the
residual is bounded below by how non-conservative the teacher was on the
sampled states, so a force error that stops falling is not by itself evidence
of a bad run.

Two practical consequences. Do not expect force-matching error against a
non-conservative teacher to go to zero — the floor is the size of the projected
component, and {py:func}`~nvalchemi.training.distillation.evaluation.non_conservative_residual`
measures it. And keep a total-energy term in the objective: a force-matching
term only ever sees the gradient of the student's energy, so with forces alone
its energy scale is unconstrained.

## Evaluating the student

Acceptance is a handful of measurements and one verdict formed from them.
Import them from the `evaluation` subpackage rather than from the distillation
namespace: an acceptance run pulls in the dynamics engine and the reporting
stack that training itself does not need.

{py:func}`~nvalchemi.training.distillation.evaluation.evaluate_accuracy` scores
a held-out set, against the dataset's own labels or against the teacher's, on
disk or scored on the fly. Accuracy alone is not what a small student fails at,
so stability is measured on a trajectory the student drives itself: register a
{py:class}`~nvalchemi.training.distillation.evaluation.StabilityMonitor` on the
propagator and read `monitor.metrics()` once the run is over — it is a method,
not an attribute, and it needs two samples at two different steps.
{py:func}`~nvalchemi.training.distillation.evaluation.measure_throughput` times
that same propagator at steady state and reports atoms per second and simulated
nanoseconds per day; every student of a family has to be timed on the same
batch for the column to rank them.
{py:func}`~nvalchemi.training.distillation.evaluation.extensivity_error` checks
that energy still scales with replicated cells, and the radial-distribution
pair compares the structure a trajectory samples against a reference
trajectory's. A student distilled from a direct-force teacher also wants
{py:func}`~nvalchemi.training.distillation.evaluation.non_conservative_residual`,
which bounds how well any conservative student can fit that teacher — the
number the section above is about.

Those measurements go into one
{py:class}`~nvalchemi.training.distillation.evaluation.StudentEvaluation` per
candidate. State the bars as
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
are — it takes the families that were filled, plus the quantities an accuracy
pass actually compared — instead of restating the mapping in your own script.
The from-scratch gate, `max_from_scratch_ratio`, needs a `baseline_accuracy`
scored on the same holdout by an equal-size student trained from scratch.

`weights` is not a measurement but the record of which of the student's two
weight sets the numbers came from, `"ema"` or `"raw"`. Nothing downstream can
infer it, so set it here: two exports of the same student then say which
artifact each one gated on. `None` records nothing, which is not the same as
`"raw"`. `evaluate_accuracy` scores exactly the object it is handed — there is
no EMA swap in either direction — so a student trained under an
{py:class}`~nvalchemi.training.hooks.EMAHook` has to be handed over as
`strategy.inference_model["student"]` and recorded as `"ema"`; passing
`strategy.models["student"]` gates on weights that will not ship. That slot
survives no checkpoint: a reloaded strategy has to dispatch `SETUP`, which
republishes the averaged copy from the re-attached hook, before it holds
anything to score.

The CLI covers the accuracy half of this. `distill evaluate` scores a recipe's
holdout, applies the accuracy bars the recipe carries, prints which weights it
scored and records that same marker as the entry's `weights`, and exits
non-zero on a missed bar; drift, speed, extensivity, the RDF,
and the from-scratch baseline are the Python path above, because no recipe
names a propagator, a supercell builder, or a second trained model. See
{ref}`distillation_recipes_guide`.

Read `StabilityMetrics.energy_fluctuation_per_atom` and
`max_energy_excursion_per_atom` beside a drift number rather than reading the
drift alone. A drift rate is the slope of a least-squares line, so a drift no
larger than the fluctuation — the RMS residual about that fit — is a line drawn
through an oscillation rather than a trend, and the excursion says how far the
series went in the meantime. Neither is a bar: the stability family gates on
`max_energy_drift_per_atom_per_ns`, `max_energy_drift_per_atom_per_step`, and
`max_momentum_drift` alone. The radial-distribution comparison is continuous in
the positions — pairs are deposited cloud-in-cell into bins whose neighbor list
is built one bin past `r_max` — so a rigid translation of a crystal scores a
Jensen-Shannon divergence at round-off where a hard-edged histogram reports a
few times `1e-2`, which is what makes the metric usable on relaxed and
crystalline frames; `r_max` may exceed half the shortest cell vector.

(custom-components)=

## Custom components

Six seams of the distillation loop are protocols rather than base classes, so a
component of your own is any object with the members named here; the
{ref}`training-distillation-api` reference documents each one in full.

- {py:class}`~nvalchemi.training.distillation.TeacherScorer` — `signals` and
  `label(batch)` returning `{teacher_field: (detached tensor, level)}`; declare
  `label_fields` so the fields you write are known before the first batch.
- {py:class}`~nvalchemi.training.distillation.InitialStructuresSource` —
  `probe()`, `initial_batch()`, `shard(rank, world_size)`, `exhausted`,
  `draw(*, limit, fits, on_miss)`, and `state_dict()` / `load_state_dict()`;
  a source driving a relaxation lifecycle stamps `status` zeros and
  `system_id`s on the batch it hands over, as
  {py:class}`~nvalchemi.training.distillation.InitialStructures` does. A
  recipe names a source through `to_spec_dict()` / `from_spec_dict()`; a
  streaming source without them stays runtime-only.
- {py:class}`~nvalchemi.training.distillation.FitPolicy` — a callable over the
  running atom and edge totals of the batch being drawn, returning whether the
  candidate still fits; {py:class}`~nvalchemi.training.distillation.WithinBudget`
  bounds them.
- {py:class}`~nvalchemi.training.distillation.AdmissionPolicy` — a callable
  over a batch of captured frames returning one boolean per graph; frames it
  refuses never enter the replay buffer.
- {py:class}`~nvalchemi.training.distillation.EvictionPolicy` —
  `select(buffer, incoming, capacity)` returning the indices into the resident
  batch (oldest first, the admitted frames last) to drop, at least as many as
  the buffer is over capacity by;
  {py:class}`~nvalchemi.training.distillation.FIFO` is the reference.
- {py:class}`~nvalchemi.dynamics.sinks.DataSink`, through `capture_sink` —
  `write(batch)`, `read()`, `zero()`, `__len__()`, and `capacity`; the loop
  sizes it to `(generation_steps + 1)` frames per trajectory and calls
  `resize(capacity)` when the sink offers one, refusing a smaller sink that does
  not. {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` is the in-tree
  device-resident one.

Minimal implementations of the four callable protocols, wired into one loop
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


class UnderMemory:
    def __init__(self, bytes_per_atom: float, budget: float) -> None:
        self.cost, self.budget = bytes_per_atom, budget

    def __call__(self, num_atoms: int, num_edges: int) -> bool:
        return num_atoms * self.cost <= self.budget


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

A `FitPolicy` such as `UnderMemory` is passed to
{py:meth}`~nvalchemi.training.distillation.InitialStructures.draw` when you
drive the draw yourself; the loop's own draws use
{py:class}`~nvalchemi.training.distillation.WithinBudget`.

## Operational notes

**Validation data goes through the same seam.** The internal labeling hook fires
on `BEFORE_FORWARD`, a stage both the training loop and the validation loop
dispatch on the device-placed batch, so unlabeled validation data needs no
preparation and a caller-supplied `training_fn` is covered too. Pointing
`validation_config` at a store written by `label_dataset` still avoids the
teacher pass entirely. Wrap that store in a
{py:class}`~nvalchemi.data.datapipes.dataloader.DataLoader` — or any iterable of
`Batch` — before handing it to
{py:class}`~nvalchemi.training.ValidationConfig`: a bare `Dataset` iterates
`(AtomicData, metadata)` pairs rather than batches. `every_n_epochs=1`
validates at every segment boundary; `every_n_steps` fires inside segments.

**Composed weights are ratios, not coefficients.**
{py:class}`~nvalchemi.training.ComposedLossFunction` renormalizes weights by
default, so the three-term objective above runs at `1/2.2`, `1/2.2`, and
`0.2/2.2`. Build the composition with `normalize_weights=False` for literal
weights, which also stops a weight schedule on one term from rescaling the
others as it ramps.

**Label dtype follows the student, down to single precision.** Teacher labels
are cast to the student's first floating-point parameter dtype, so a float64
teacher feeds a float32 student without a dtype error at the loss. The cast
never goes below float32, because a store hands every floating field back at
the dtype of the dataset's `positions` and a narrower label would disagree with
what `label_dataset` persisted. A `bfloat16` or `float16` student therefore
needs `dtype_policy="prediction_to_target"` on its loss terms, and a float64
student training from a store needs `"target_to_prediction"`, which widens the
float32 labels a store returns. The cast is resolved at construction, so a
student whose dtype changes afterwards needs a `dtype_policy` as well.

**The teacher is stored once per checkpoint root.** The first write under a root
holds the frozen teacher's weights and every later
{py:class}`~nvalchemi.training.hooks.CheckpointHook` write records a reference
to that index plus a fingerprint, so a periodic checkpoint costs the student's
weights alone and a load verifies the stored copy. One root holds one copy:
saving a *different* teacher into a root that already holds one is refused,
while an identical copy is written again freely, which repairs a root whose
weight file went missing. A trainable-only save keeps the teacher's stored
copy whole, since the checkpoints already written reference it. The
fingerprint and the manifest layout are in {ref}`distillation_recipes_guide`.

**The segment loop round-trips as references.**
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.to_spec_dict`
carries `on_policy` and `reference_dataset` inline: every
{py:class}`~nvalchemi.training.distillation.OnPolicySettings` field verbatim —
`OnPolicyConfig.settings` is that half on its own, validated without a
propagator — the propagator as the constructor reference it rebuilds from with
the student rebound at build time, the scorer as its signals, `dtype`, and
`probe_seed` over the model named `"teacher"`, and each dataset as the store
it reads, a `MultiDataset` as the list of stores it concatenates.
`initial_structures` goes in as its store, the budgets it was *declared* with,
and `recycle`; the cursor does not, because it is state and belongs to a
restart bundle. What stays runtime-only is what no recipe can name — a
propagator's live hooks and sinks, a dataset holding its samples in memory, and
a criterion passed whole as `convergence_hook` rather than as `fmax` — and the
strategy leaves the whole `on_policy` entry out with a warning rather than
writing a recipe that would rebuild into a different run. On every rebuild
entry point a live object outranks the recipe: the keyword passed to
`from_spec_dict` or `load_checkpoint`, then the object the rebuild was offered
for the restore, then the recipe. The full table, including which propagators
introspect, is in {ref}`distillation_recipes_guide`.

**The segment is the restart granularity.** An interrupted on-policy run carries
a restart bundle through the checkpoint — the propagator's
`dynamics_step_count`, the live `trajectory` batch, the `replay_frames` it had
filled, the `settings` it ran under, the `initial_structures` cursor state, and
whether generation was exhausted — so a resumed run continues the same
trajectory rather than starting a fresh one, the restored frames *replace* the
buffer's contents rather than merging into them, the backfill picks up at the
row the interrupted run had reached, and a setting the resumed loop sets
differently is reported. The cursor state is the position, its wrap count, the
next `system_id`, and the rank shard the three were counted in; a bundle
written for another shard is refused rather than replayed against the wrong
rows. Not carried are RNG state, the neighbor tensors, and FIRE's adaptive
state, so only a counter-based-RNG integrator reproduces its stream exactly
and a resumed relaxation re-initializes its optimizer history. A segment a
checkpoint interrupted part-way is counted as finished on the way in — its
`AFTER_EPOCH` hooks never fire, the batches it had left are not replayed, and
the run opens a fresh segment, which begins by generating. An exhausted run
resumes exhausted: the bundle carries the frames and the exhaustion rather
than a trajectory, and the resumed run trains on the buffer without
regenerating. The buffer also outlives a run: a second `run()` on one strategy
appends to the frames the first filled, while reseeding its own trajectory,
because installing the rank shard rewinds the cursor. Only a restart bundle
resumes one.

Resuming an on-policy run has two routes.
{py:meth}`~nvalchemi.training.distillation.DistillationStrategy.load_checkpoint`
rebuilds the segment loop from the recipe the checkpoint carries, so in the
common case it returns an on-policy strategy that runs without a dataloader;
pass `models=` so the propagator is rebound to the very student the optimizer
updates, and `on_policy=` / `reference_dataset=` to override a piece the recipe
could not name. Only a run whose recipe was left out comes back
offline-shaped, and then its `run()` rejects the `None` dataloader. The other
route is to rebuild the strategy with the same propagator, scorer, reference
dataset, and hooks, then restore the counters, weights, optimizer state, and
checkpointable hook state into it in place with
{py:meth}`~nvalchemi.training.TrainingStrategy.restore_checkpoint` — which,
unlike `load_checkpoint`, takes no `hooks` override, because it restores into
the hooks the rebuilt strategy already holds:

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

Attaching `on_policy` to an already-loaded strategy is not a substitute:
assignment is not validated, so the propagator would keep a student the
optimizer never updates and the run would silently stop being on-policy.
`num_steps` is an absolute target rather than a budget for the resumed leg, so a
run that already reached it resumes to nothing until the target is raised.

```{note}
**Reserved and runtime-only settings.** `replay_eviction` admits one spelling
in a recipe, `"fifo"`; a custom
{py:class}`~nvalchemi.training.distillation.EvictionPolicy` instance rides on
`OnPolicyConfig` alone and is recorded as `"fifo"` with a warning, as
`capture_sink` and `replay_admission` are omitted with one. Bound
`replay_capacity` on long runs, as a multiple of the trajectory count.
`weight_sync_frequency` must be `1`: the propagator and the trainer share one
module object, so an eager run is never out of sync, and the setting only
becomes meaningful once the propagator holds a compiled or remote copy of the
student.
```

## API reference

See {ref}`training-distillation-api` for the API reference for
{py:class}`~nvalchemi.training.distillation.DistillationStrategy`,
{py:class}`~nvalchemi.training.distillation.InProcessTeacherScorer`,
{py:func}`~nvalchemi.training.distillation.label_dataset`,
{py:class}`~nvalchemi.training.distillation.OnPolicyConfig`,
{py:class}`~nvalchemi.training.distillation.OnPolicySettings`,
{py:class}`~nvalchemi.training.distillation.InitialStructuresSource`,
{py:class}`~nvalchemi.training.distillation.InitialStructures`,
{py:class}`~nvalchemi.training.distillation.TeacherLabelHook`,
{py:class}`~nvalchemi.training.distillation.ReplayBuffer`,
{py:class}`~nvalchemi.training.distillation.AdmissionPolicy`,
{py:class}`~nvalchemi.training.distillation.EvictionPolicy`,
{py:class}`~nvalchemi.training.distillation.FIFO`,
{py:class}`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss`,
{py:class}`~nvalchemi.training.distillation.EmbeddingMatchingLoss`,
{py:class}`~nvalchemi.training.distillation.HessianMatchingLoss`, and
{py:class}`~nvalchemi.training.distillation.BoltzmannMatchingLoss`.

{ref}`distillation_recipes_guide` covers the JSON recipe and the `distill` CLI
that author, run, resume, and gate the runs this guide describes.
