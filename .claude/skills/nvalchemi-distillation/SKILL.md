---
name: nvalchemi-distillation
description: >-
  How to distill a large teacher MLIP into a small student with
  DistillationStrategy — teacher signals and offline dataset labeling, the
  teacher_* loss targets, the on-policy segment loop (propagator, replay
  buffer, mixed loader), teacher weights stored once per checkpoint root and
  restart, accuracy/stability/throughput evaluation with acceptance
  thresholds, and the JSON recipe CLI. Use when training a small student to
  reproduce a big model's energies, forces, stress, or per-atom energies,
  generating training frames from the student's own trajectories, or gating a
  distilled student against acceptance bars.
---

# nvalchemi Distillation

## Overview

Distillation trains a small **student** to reproduce a large frozen
**teacher**. In `nvalchemi` it is a `TrainingStrategy` subclass over two named
models, so everything from `nvalchemi-training-api` applies: optimizers,
schedulers, validation, hooks, checkpoints, DDP. What distillation adds is a
teacher whose outputs become loss *targets*.

Read `nvalchemi-training-api` and `nvalchemi-loss-api` first. Deeper details
live in `docs/userguide/distillation_recipes.md` and
`docs/modules/training/distillation.rst`.

```python
import torch

from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.training import (
    ComposedLossFunction,
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InProcessTeacherScorer,
    OnPolicyConfig,
    OnPolicyKnobs,
    PerAtomEnergyMatchingLoss,
    ReplayBuffer,
    SeedSource,
    TeacherLabelHook,
    build_mixed_loader,
    default_distillation_fn,
    label_dataset,
)
```

Two loops are available:

- **Offline** — train over a fixed dataset, with the teacher's labels either
  precomputed into a store or produced on the fly. Start here; it distributes
  over ranks like ordinary training.
- **On-policy** — generate frames by running dynamics with the student itself,
  label them with the teacher, and train on a mixture of those and reference
  data. Use it when the student is stable on the reference distribution but
  fails on the states it actually visits.

---

## Minimal Pattern (offline)

```python
strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={
        "student": [
            OptimizerConfig(
                optimizer_cls=torch.optim.AdamW,
                optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
            )
        ]
    },
    loss_fn=EnergyMSELoss(target_key="teacher_energy")
    + 10.0 * ForceMSELoss(target_key="teacher_forces"),
    num_steps=10_000,
)
strategy.run(dataloader)
```

Three rules govern the shape of that call:

- **The teacher is frozen by omission.** Give `optimizer_configs` a `"student"`
  entry and no `"teacher"` entry. Adding one is an error, not a way to train
  both.
- **The model names are fixed.** `"student"` and `"teacher"` are required keys.
- **Teacher signals are derived, not declared.** The strategy reads the
  `teacher_*` `target_key`s out of the loss and asks the teacher for exactly
  those signals, validating them against the teacher's declared outputs at
  construction. Only set `teacher_signals=` to request a signal no loss term
  consumes.

`training_fn` defaults to `default_distillation_fn`, a plain student forward.
Its `predicted_*` keys are checked at construction against the student's
`active_outputs`, so a narrowed student is caught before the run.

---

## Teacher Signals

A scorer turns a `Batch` into named signals, each mapped to a batch field:

| Signal | Batch field | Level |
| --- | --- | --- |
| `energy` | `teacher_energy` | system |
| `forces` | `teacher_forces` | node |
| `stress` | `teacher_stress` | system |
| `node_energies` | `teacher_node_energies` | node |
| `embeddings` | `teacher_node_embeddings` | node |

`InProcessTeacherScorer(teacher, signals, cast_to=None)` evaluates a teacher
loaded in the current process. It narrows the teacher's `active_outputs` to the
requested signals, builds and rolls back the teacher's own neighbor list, and
detaches every output — the scored batch comes back exactly as it went in.
`cast_to` stores labels at a reduced dtype.

---

## Offline Labeling

Labeling once beats labeling every epoch: a labeled store trains with **no
teacher forward pass at all**.

```python
scorer = InProcessTeacherScorer(teacher, ("energy", "forces"))
label_dataset(source_dataset, scorer, "data/labeled.zarr", batch_size=32)
```

`label_dataset` walks the dataset in contiguous chunks, attaches the teacher
fields, and writes the original fields plus the teacher fields to a Zarr store
read back through the ordinary `AtomicDataZarrReader` / `Dataset` path. It is
resumable (`resume=True`), drops the ephemeral neighbor tensors, rejects a
chunk whose schema drifts from the store's, and refuses a store an interrupted
run left inconsistent rather than resuming from a misaligned offset. Point
`validation_config` at a labeled store too.

Batches that arrive **unlabeled** are labeled on the fly by an internal
`BEFORE_FORWARD` hook, with autocast disabled so mixed-precision training
leaves the targets untouched and an on-the-fly label matches an offline one
exactly. Set `label_missing=False` to skip the teacher and let a missing target
surface from the loss instead.

---

## Losses

Any built-in term distills by pointing its `target_key` at a teacher field.
Signals with no supervised counterpart get their own term.

```python
loss_fn = ComposedLossFunction(
    [
        EnergyMSELoss(target_key="teacher_energy"),
        ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        PerAtomEnergyMatchingLoss(),  # teacher_node_energies
    ],
    weights=[1.0, 10.0, 1.0],
    normalize_weights=False,
)
```

`PerAtomEnergyMatchingLoss` matches the teacher's per-atom energy
decomposition — a target no reference dataset carries — against the student's
`predicted_atomic_energies` head.

`ComposedLossFunction` **renormalizes weights by default**, so composed weights
are ratios: `a + b + 0.2 * c` runs at `1/2.2`, `1/2.2`, `0.2/2.2`. Pass
`normalize_weights=False` for literal coefficients, which also stops a
`LossWeightSchedule` on one term from rescaling the others as it ramps.

---

## On-Policy Generation

`OnPolicyConfig` describes one generate-label-train segment. Setting
`on_policy=` on the strategy is what turns the pieces into a run; `run()` then
takes **no** dataloader.

```python
config = OnPolicyConfig(
    dynamics=NVTLangevin(student, dt=0.5, temperature=300.0, friction=0.01),
    teacher_scorer=InProcessTeacherScorer(teacher, ("energy", "forces")),
    seeds=SeedSource(seed_dataset),
    segment_steps=50,        # propagator steps per segment
    label_frequency=10,      # label every Nth generated frame
    steps_per_segment=32,    # optimizer steps per segment
    batch_size=8,
    replay_ratio=0.25,       # share of each batch drawn from generated frames
    replay_capacity=8192,
)
strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={"student": [...]},
    loss_fn=...,
    num_steps=10_000,
    reference_dataset=anchor_dataset,
    on_policy=config,
)
strategy.run()
```

Constraints worth knowing before you write the script:

- **The propagator must hold the very module registered as
  `models["student"]`** — on its own, or composed into a larger model. That
  object identity is what makes each segment generate from the weights the
  previous one trained, and it is checked at construction.
- **`reference_dataset` is the anchor** and is required unless
  `replay_ratio == 1`, which is refused when an anchor is supplied. It must be
  a teacher-labeled dataset in the replay-frame shape; one carrying reference
  `energy` or `forces` of its own is rejected rather than silently mixed in.
- **`seeds` is a `SeedSource`** — one cursor over the seed rows, shared by the
  initial batch, any backfill, and a restart. A bare dataset is wrapped for you.
  An unbudgeted source propagates every row it owns as one batch, so size the
  store to the device; a budgeted one (`max_atoms`, `max_batch_size`) packs the
  batch first-fit and leaves the rest for the backfill.
- **One segment is one epoch.** `AFTER_EPOCH` and epoch-cadence validation land
  at segment boundaries; step-cadence validation fires inside them.
- **Single-process for now.** Nothing shards the loop's loader or seed state,
  so it refuses to start on more than one rank. Distill offline (label the
  store, train it with `DDPHook`) to scale out.
- `OnPolicyConfig.seed` keys the mixture sampler; vary it, not the global torch
  seed, to make replicate runs draw independently.

Lower-level pieces, if you drive generation yourself: `TeacherLabelHook` is the
`AFTER_STEP` dynamics hook that attaches `teacher_*` fields to the live frame
and mirrors a stripped copy into a `DataSink`; `ReplayBuffer` accumulates those
frames behind a frozen key schema; `build_mixed_loader` draws each batch at an
exact reference/replay composition and **must be rebuilt after every segment**
because its batch sampler reads child dataset lengths once.

---

## Checkpoints And Restart

Checkpointing works as it does for any strategy, with two additions.

**The teacher is stored once per checkpoint root.** A frozen teacher whose
weights are written into every periodic checkpoint duplicates a model that
never changed. The first checkpoint under a root writes
`models/teacher/checkpoints/0.pt`; every later one records a `model_references`
entry naming that index plus a cheap fingerprint, and no weight file of its
own. So the checkpoint interval costs the student's weights alone, whatever the
teacher's size — shorten it freely.

The fingerprint carries a tensor count, an element count, and a digest over
each state-dict entry's name, shape, and dtype together with its values, read
at `float64` on the host so the device does not change the digest. A tensor of
at most 4096 values is hashed whole (per-element tables, biases); a larger one
contributes 64 values spanning its whole index range, first and last included.
Loading reads the stored weights back and verifies the fingerprint, so a
replaced or truncated copy raises `ValueError` instead of quietly training
against a different model. Precision is part of the identity: a `bfloat16` copy
is a different model to the fingerprint.

**One root holds one copy.** Saving a *different* copy of the teacher into a
root that already holds one raises `ValueError` at save time, because the
`model_references` entry is root-global and moving it would repoint every
checkpoint already written there. Give a second teacher its own checkpoint
root. Re-storing an *identical* copy is allowed, which is how a root whose
stored weight file went missing is repaired.

The teacher's `checkpoint_spec()` still rebuilds its architecture but is never
trusted for its weights — a teacher loaded from a fine-tune checkpoint
publishes the spec of what it was originally built from.

**An on-policy run resumes its trajectory.** The live trajectory batch, the
propagator's cumulative step count, the seed source's cursor, and the replay
frames travel through the checkpoint, so a resumed run continues the same
trajectory rather than seeding a fresh one, and backfills from where the
interrupted run left the cursor rather than re-serving structures it already
relaxed. Where the caller declared no budget, the cursor carries the size
envelope the seeds established, so a run restored after a graduation refills
under the width it started at rather than under the narrowed batch it resumes; a
source the caller budgeted keeps the budget its recipe names. With the built-in
integrators — whose Langevin noise comes from a counter-based generator keyed on
the step count — that continuation is exact. The bundle also records the knobs
it ran under, so a resumed loop that sets one differently says so with a
`UserWarning`; a bundle written before the cursor was checkpointed still
restores, with its own warning.

Restart lands on a **segment boundary**. The interrupted segment is counted as
finished on the way in: its `AFTER_EPOCH` hooks do not fire, its leftover
training batches are not replayed, and the mixture sampler advances past its
epoch index. The run then opens a fresh segment, which begins by generating —
so a checkpoint written part-way through a training phase costs one extra
generation phase.

Two things to budget for. The bundle is **rank-local**: it rides in a strategy
checkpoint, which `CheckpointHook` writes on rank zero alone. It is consumed
only when a single rank wrote it and a single rank is restoring it, so any
multi-rank restart — matched world sizes included — drops it with a
`UserWarning` and each rank reseeds from its own share with a **cold replay
buffer**, so the first segments after such a restart draw from the reference
dataset alone. And a restore **replaces** the replay frames rather than merging
them (`buffer.clear()` then refill) — merging would skew the weighting toward
stale pre-restart states, double the memory, and reach the eviction horizon a
restart early. It is not a diversity loss; the mixed loader draws with
replacement.

```python
strategy.restore_checkpoint(run_dir / "checkpoints")
strategy.run()
```

From the CLI the same restart is one command against the recipe the run started
from:

```bash
nvalchemi-training distill spec resume runs/onpolicy/checkpoints \
  --spec onpolicy.json
```

---

## Recipe Serialization

`to_spec_dict()` carries a whole on-policy run as references. Round-tripping
needs the models supplied, because a recipe names them by role:

```python
spec = strategy.to_spec_dict()
rebuilt = DistillationStrategy.from_spec_dict(
    spec, models={"student": student, "teacher": teacher}
)
```

What serializes: every scalar knob verbatim; the propagator as `cls_path` plus
kwargs, with the student rebound at build time; the scorer as its signal set,
cast dtype, and the model name `"teacher"`; `seeds` as its store plus the
budgets and `recycle` it was built with, never its cursor, which is restart
state; path-backed datasets as the store they read.

What stays **runtime-only**: `convergence_hook` (pass the `convergence`
threshold instead to keep the criterion in the recipe); a propagator's hooks,
sinks, and convergence hook; and any dataset holding its samples in memory. The first two
are omitted with a warning naming them — read off the *live* propagator, so a
collaborator registered after construction counts and a propagator a recipe
built is checked too, with the segment loop's own `TeacherLabelHook` excluded.
An in-memory dataset raises with the fix in the message (write it with
`label_dataset`, point the recipe at the path). A piece the recipe cannot
describe leaves the whole `on_policy` entry out rather than producing a recipe
that rebuilds into a different run.

A rebuilt propagator loses those collaborators. A missing neighbor-list hook is
loud — the model reads neighbor tensors off the batch and raises `KeyError`
without them. The silent losses are the convergence hook, the sinks, and any
thermostat or logging hook.

`from_spec_dict` takes `on_policy=` and `reference_dataset=` overrides for
exactly those cases, and an explicitly supplied `on_policy` outranks the spec's
own recipe rather than being replaced by it.

---

## Evaluation And Acceptance

Import from the `evaluation` subpackage, not the distillation namespace — an
acceptance run pulls in the dynamics engine and the reporting stack that
training does not need.

```python
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    measure_throughput,
    measured_bars,
    nonconservative_residual,
    StabilityMonitor,
)

accuracy = evaluate_accuracy(student, holdout_loader, targets="teacher",
                             scorer=teacher)
report = build_acceptance_report(
    [StudentEvaluation(name="small", accuracy=accuracy)],
    AcceptanceThresholds(max_forces_mae=0.05),
)
print(report.accepted)
```

- `evaluate_accuracy` runs through `ValidationLoop`, so eval mode, autograd
  policy, autocast, and device placement match training validation. Metrics are
  exact global residual sums, not the (graph-balanced) loss.
- `StabilityMonitor` is a dynamics hook reporting energy drift and momentum
  conservation over a trajectory the student drives. Give it `warmup_steps`
  long enough to cover relaxation, or a transient is reported as drift.
- `nonconservative_residual` bounds how well a conservative student can fit a
  direct-force teacher. It is scale-dependent — read its docstring before
  quoting the number.
- `measure_throughput`, `extensivity_error`, and the radial-distribution pair
  round out the report.
- `StabilityMonitor.metrics()` is a **method**, not an attribute, and needs at
  least two samples recorded at two different steps.
- **A bar with no measurement behind it fails the student**, rather than being
  skipped. Every metric rebuilds from its own `to_dict` export with
  `from_dict`, so a sweep can evaluate each student in its own job and assemble
  one report at the end.
- Never hard-code which bars you may state. `measured_bars(*families,
  accuracy_quantities=...)` answers it from the measurements in hand: naming a
  family is necessary, and for the accuracy bars the compared quantities narrow
  it further, since a pass scored on energy alone fills no force bar.
- That is why a **recipe** may only carry
  `measured_bars("accuracy", accuracy_quantities=evaluation.quantities)`:
  `distill evaluate` scores a holdout and fills nothing else, so any other bar
  in `evaluation.thresholds` is refused at parse time rather than failing the
  student on a number nobody took. Widen `evaluation.quantities` to earn a bar
  the pass skipped; measure drift, throughput, extensivity, RDF, and the
  from-scratch baseline in Python and build the report there.
- `distill evaluate --json-out` writes a non-finite metric as the string
  `"nan"`, `"inf"`, or `"-inf"`, so the export stays parseable by a strict JSON
  reader instead of carrying Python's bare `NaN` token.
- `distill evaluate` scores the **averaged** weights when the recipe's
  `student.hooks` carry an `EMAHook` — the run's own validation reads them, so
  the gate does too — and prints `weights: ema (student.hooks EMAHook)` or
  `weights: raw` above the report, so a number is attributable to weights.

---

## CLI

Use the CLI when the user wants an on-the-rails run from a JSON file; use the
API when they need custom model construction, dynamic data routing, or
non-standard orchestration.

```bash
nvalchemi-training distill init \
  --tier small \
  --teacher-model mace --teacher-id small-0b \
  --dataset data/labeled.zarr \
  --holdout-dataset data/holdout.zarr \
  --output-dir runs/distill \
  --out recipe.json

nvalchemi-training distill spec report recipe.json
nvalchemi-training distill spec run recipe.json
nvalchemi-training distill evaluate recipe.json \
  --student-checkpoint runs/distill/checkpoints --json-out acceptance.json
```

The group is also installed as `nvalchemi-distill`. Commands: `init`,
`schema`, `spec report`, `spec run`, `spec resume`, `evaluate`.
`init --mode on-policy` adds the segment loop and **requires**
`--seed-dataset` — `--dataset` is the anchor the mixture draws its reference
share from, it carries no `forces` for the propagator's first step, and the
strategy rejects an anchor carrying labels of its own as a seed.

`spec run` and `spec resume` take `--distributed/--no-distributed` (auto when
`WORLD_SIZE > 1`) and `--ddp-backend`; a multi-rank `spec resume` defaults
`--map-location` to this rank's device, which a restart has to name in both the
load location and the restored strategy's `devices`.

`init` also writes a `CheckpointHook` into `student.hooks` at
`<output-dir>/checkpoints`, saving every `num_steps // 10` steps (minimum 1),
so `spec resume` has a checkpoint to resume from and `evaluate
--student-checkpoint` has one to score. A cadence saves nothing at training
end, so `spec run` and `spec resume` write a terminal checkpoint at the next
index whenever the run finished on a step the interval missed: `evaluate`
scores the weights the run ended with, and a later `resume` has nothing left to
repeat.

`--tier small|base|large` selects a **size template only** — a width, a depth,
and a radial-basis count written into `student.spec.kwargs` for whatever
constructor `--student-cls-path` names. It never selects an architecture or a
model family.

`evaluate` exits non-zero on a missed bar, so a sweep gates on the command
rather than on parsing its output. Its `--map-location` names the one device
the student, the teacher, the holdout, and the errors all run on, so a
GPU-trained student scores on a host with no GPU; on `spec resume` the same
flag names the device the continued run takes, not only the one its tensors
are read onto.

---

## Caveats

- Give the teacher the outputs the signals need. Requesting `stress` from a
  teacher that does not declare it fails at construction, which is the point.
- Point `target_key` at `teacher_*`, never at the reference field. A term left
  on `energy` silently trains against dataset labels.
- Prefer `normalize_weights=False` when you mean literal loss coefficients.
- Precompute labels with `label_dataset` whenever the dataset is reused; it
  removes the teacher from the training loop entirely.
- On-policy runs need an anchor. A `replay_ratio` near `1.0` drifts off the
  reference distribution with nothing pulling it back.
- The checkpoint interval is not a teacher-size trade-off: the teacher is
  stored once per checkpoint root, so a short interval costs the student's
  weights alone. Loading verifies the stored copy against a sampled
  fingerprint.
- Distributed: offline distillation scales with `DDPHook`; the on-policy loop
  does not, and says so.

---

## Key files

| File | Contents |
| --- | --- |
| `nvalchemi/training/distillation/strategy.py` | `DistillationStrategy`, `default_distillation_fn` |
| `nvalchemi/training/distillation/scoring.py` | `TeacherScorer`, `InProcessTeacherScorer`, signal table |
| `nvalchemi/training/distillation/labeling.py` | `label_dataset` |
| `nvalchemi/training/distillation/config.py` | `OnPolicyKnobs`, `OnPolicyConfig` and its spec round trip |
| `nvalchemi/training/distillation/seeding.py` | `SeedSource`: the seed cursor, its shard, budget and state dict |
| `nvalchemi/training/distillation/replay.py` | `ReplayBuffer`, `build_mixed_loader` |
| `nvalchemi/training/distillation/hooks.py` | `TeacherLabelHook` |
| `nvalchemi/training/distillation/losses/` | `PerAtomEnergyMatchingLoss` |
| `nvalchemi/training/distillation/evaluation/` | accuracy, stability, throughput, acceptance |
| `nvalchemi/training/distillation/cli.py` | `DistillationJobSpec` and the `distill` group |
| `docs/userguide/distillation_recipes.md` | Recipe lifecycle, CLI, objective/literature catalog |
| `examples/intermediate/09_offline_distillation.py` | Runnable offline example |
