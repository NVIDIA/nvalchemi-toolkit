---
name: nvalchemi-distillation
description: >-
  How to distill a large teacher MLIP into a small student with
  DistillationStrategy — teacher signals and offline dataset labeling, the
  teacher_* loss targets, the on-policy segment loop (propagator, replay
  buffer, mixed loader), checkpoints and restart, and accuracy/stability
  evaluation with acceptance thresholds. Use when training a small student to
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

Read `nvalchemi-training-api` and `nvalchemi-loss-api` first.
`docs/userguide/distillation.md` is the concept guide and
`docs/modules/training/distillation.rst` is the symbol-by-symbol reference.

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
    OnPolicySettings,
    AtomicEnergyMatchingLoss,
    ReplayBuffer,
    InitialStructures,
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
| `atomic_energies` | `teacher_atomic_energies` | node |
| `embeddings` | `teacher_node_embeddings` | node |
| `hessian` | `teacher_hvp` + `teacher_hvp_probe` | node |

`InProcessTeacherScorer(teacher, signals, *, dtype=None, probe_seed=None,
neighbor_list="rebuild", autocast=False)` evaluates a teacher loaded in the
current process.
`signals` mixes built-in names with `TeacherSignal(name, model_output, field,
level, normalize=None)` specs for any other teacher output (`field` must start
with `teacher_`). It narrows the teacher's `active_outputs` to the requested
signals, builds and rolls back the teacher's own neighbor list (or, under
`neighbor_list="reuse"`, consumes the batch's list and refuses a missing key or
a mismatched cutoff stamp), and detaches every output — the scored batch comes back
exactly as it went in. `dtype` stores labels at a reduced dtype. `autocast`
decides the scoring pass's precision: `False` disables autocast around it
whatever region surrounds the call, `None` keeps the caller's region, and
`True` or a floating dtype enables it.

Signals differ in cost: the forward-pass ones share a single teacher pass,
`embeddings` adds a second (embeddings come from `compute_embeddings`, not from
the forward pass), and `hessian` adds an energy-only pass plus two backward
passes through it. `hessian` is the one signal that writes two fields, because
a Hessian-vector product is only comparable to a student's along the same
direction, so the probe travels with it. `probe_seed` names the stream that
direction is drawn from; leave it unset for training and offline labeling,
where coverage comes from redrawing, and set it — as the strategy does per
validation batch — for a number compared across passes.

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
`BEFORE_FORWARD` hook. The hook opens no autocast region of its own; the
scorer's `autocast` (default `False`) decides the label precision, so an
on-the-fly label matches an offline one exactly under mixed-precision training.
The first batch labeled this way triggers a one-time warning naming
the teacher fields it lacked. Set `label_missing=False` to skip the teacher and
let a missing target surface from the loss instead. Labels are cast to the
dtype of the student's first floating parameter, floored at `float32`.
`label_dtype=` on the strategy names the dtype outright, and a non-floating
dtype is refused.

---

## Losses

Any built-in term distills by pointing its `target_key` at a teacher field.
Signals with no supervised counterpart get their own term.

```python
loss_fn = ComposedLossFunction(
    [
        EnergyMSELoss(target_key="teacher_energy"),
        ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True),
        AtomicEnergyMatchingLoss(),  # teacher_atomic_energies
    ],
    weights=[1.0, 10.0, 1.0],
    normalize_weights=False,
)
```

`AtomicEnergyMatchingLoss` matches the teacher's per-atom energy
decomposition — a target no reference dataset carries — against the student's
`predicted_atomic_energies` head.

`ComposedLossFunction` **renormalizes weights by default**, so composed weights
are ratios: `a + b + 0.2 * c` runs at `1/2.2`, `1/2.2`, `0.2/2.2`. Pass
`normalize_weights=False` for literal coefficients, which also stops a
`LossWeightSchedule` on one term from rescaling the others as it ramps.

---

## Embedding, Hessian, And Boltzmann Objectives

Three further terms distill what no reference dataset has a column for. The
physics and the weighting are in `docs/userguide/distillation.md` ("Objectives
beyond pointwise matching"); this is what each asks of the run.

```python
from nvalchemi.training.distillation import (
    BoltzmannMatchingLoss,
    EmbeddingMatchingLoss,
    EmbeddingProjector,
    HessianMatchingLoss,
    embedding_distillation_fn,
    hessian_distillation_fn,
)
```

- **`EmbeddingMatchingLoss`** needs `training_fn=embedding_distillation_fn`
  (both sides come from `compute_embeddings`, so the student runs twice per
  batch) and, when widths differ, an `EmbeddingProjector(student_width,
  teacher_width)` registered as a third model `"projector"` with an
  `optimizer_configs` entry of its own. The projector maps the student side
  only and is a training-time artifact; weight the term as a regularizer. A
  student whose embeddings are detached from its parameters is refused unless
  the projector is registered with `frozen_student=True`.
- **`HessianMatchingLoss`** needs `training_fn=hessian_distillation_fn` (a
  second, energy-only pass) and the `hessian` signal, which writes
  `teacher_hvp` and its probe. Its graph-balanced value runs one to two orders
  of magnitude above a force MSE, so start it a hundred to ten thousand times
  lighter than the force term and read one batch's value as a one-sample
  estimate. A loss pointed at `teacher_hvp_probe` is refused.
- **`BoltzmannMatchingLoss`** requires `on_policy`, refuses a relaxation
  propagator and any convergence criterion (the inference reads the
  propagator's `samples_equilibrium`; `OnPolicySettings.samples_equilibrium`
  overrides it), refuses any place in the validation loss, and warns when
  `replay_ratio < 1`. Seed it with replicas of one structure, set its
  `temperature` and the thermostat's from the same number, and hold
  `beta >= 0.5`: at `0` a saturated softmax reads as converged. Under a
  `DDPHook` the softmax runs over the world batch (`world_batch=None` infers
  the gather, `True`/`False` force it; `check_one_system=False` drops the
  one-system guard).

---

## On-Policy Generation

`OnPolicyConfig` describes one generate-label-train segment. Setting
`on_policy=` on the strategy is what turns the pieces into a run; `run()` then
takes **no** dataloader.

```python
config = OnPolicyConfig(
    dynamics=NVTLangevin(student, dt=0.5, temperature=300.0, friction=0.01),
    teacher_scorer=InProcessTeacherScorer(teacher, ("energy", "forces")),
    initial_structures=InitialStructures(initial_dataset),
    generation_steps=50,        # propagator steps per segment
    label_frequency=10,      # label every Nth generated frame
    training_steps_per_segment=32,    # optimizer steps per segment
    batch_size=8,
    replay_ratio=0.25,       # share of each batch drawn from generated frames
    replay_capacity=8192,
)
strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
    optimizer_configs={"student": [...]},
    loss_fn=...,
    num_steps=10_000,
    reference_dataset=reference_dataset,
    on_policy=config,
)
strategy.run()
```

Constraints worth knowing before you write the script:

- **The propagator must hold the very module registered as
  `models["student"]`** — on its own, or composed into a larger model. That
  object identity is what makes each segment generate from the weights the
  previous one trained, and it is checked at construction.
- **`reference_dataset` is the reference share of the mixture** and is
  required unless `replay_ratio == 1`, which is refused when one is supplied. It must be
  a teacher-labeled dataset in the replay-frame shape; one carrying reference
  `energy` or `forces` of its own is rejected rather than silently mixed in.
- **`initial_structures` is any `StructureSource`**, the core
  `nvalchemi.dynamics` protocol, importable here as `InitialStructuresSource`.
  `InitialStructures` is the reference source: a thin subclass of the core
  `nvalchemi.dynamics.OrderedStructureSampler` that adds the
  `to_spec_dict`/`from_spec_dict` round trip. Its one position (`next_row`)
  over its rows is shared by the initial batch and any backfill; the segment
  loop re-shards the sampler at every start, which resets it, so a restart
  reseeds from the front of the shard. A bare dataset is wrapped for you.
  Unbudgeted, it propagates every row it owns as one batch, so size the store
  to the device; budgeted (`max_atoms`, `max_batch_size`), it packs the batch
  first-fit and leaves the rest for the backfill, drawn through `draw` under a
  `FitPolicy`. `WithinBudget` is the stock `FitPolicy`, and both live in
  `nvalchemi.dynamics`. A
  source of your own implements `probe`, `initial_batch`, `shard`, `exhausted`,
  `draw`, `state_dict`, and `load_state_dict`.
- **One segment is one epoch.** `AFTER_EPOCH` and epoch-cadence validation land
  at segment boundaries; step-cadence validation fires inside them.
- **The construction probe is a setting.** `OnPolicySettings.probe` (default
  `True`) runs the propagator's `compute()`, and a relaxation criterion, on
  one initial structure at construction. `probe=False` defers a mismatch to
  the first step. It also silences the warning raised when the probe cannot
  run a propagator whose model plans more than one neighbor-list source.
- **Divergence is a predicate.** `OnPolicyConfig.divergence` (runtime-only,
  like `replay_admission`) returns one `bool` per graph, marking the graphs a
  lifecycle freezes without capturing them. It is evaluated once per step and
  ORed into a record the capture hook and the segment boundary read. The
  default, `nonfinite_divergence`, is an alias of the core
  `nvalchemi.dynamics.hooks.nonfinite_graph_mask` over positions and forces.
- **Graduation is reported at `ON_GRADUATE`.** A structure graduates on the
  step its status reaches `exit_status`; the propagator dispatches
  `DynamicsStage.ON_GRADUATE` with `ctx.graduated_mask`, where the
  converged-frame hook stores it once, unlabeled, and the boundary retires and
  backfills it. A propagator already carrying a status migrator, a
  multi-sub-stage `FusedStage`, or a `DomainParallel` propagator is refused
  when `OnPolicyConfig` is built; a migrator registered later, or a
  propagator-owned sampler, is refused when `run()` starts.
- **The capture sink grows at most once.** `capture_sink` is grown through
  `resize(capacity)`, never shrunk, and only when it satisfies the
  `ResizableSink` protocol from `nvalchemi.dynamics.sinks` (re-exported here).
  A smaller sink that does not satisfy it is refused up front.
- **Multi-rank runs are data-parallel.** Add a `DDPHook` and launch one
  process per GPU. Each rank propagates its own strided shard of the initial
  structures — `DistillationStrategy.structure_shard` — labels it with its own
  teacher replica, and fills its own replay buffer; the student's gradient
  all-reduce is the only per-step training traffic between ranks. Size the
  initial structures to a whole multiple of
  the world and sort them by atom count, since the deal strides by index and
  balances structure counts rather than work. The reference dataset is *not*
  sharded. A multi-rank launch that leaves the student unwrapped is refused:
  with a `DDPHook` registered the check reads `DDPHook.wrapped_keys`, otherwise
  ownership is read through `unwrap_model`; `require_wrapped_student=False`
  waives it, with a one-time warning, for in-place wrappers such as FSDP2
  `fully_shard`. A shard that seeded nothing is refused by a verdict agreed
  across ranks through `nvalchemi.training.distributed.all_reduce_flags`. Ranks
  decorrelate through `rank_seed_stride` (default `1_000_003`): the sampler
  seed moves by `rank * stride`, and every integer `random_seed` in the
  propagator's stage tree through `BaseDynamics.seed_offset`, which returns the
  stages it could not move (each is warned about). An index-less
  `replay_device` resolves through the core `nvalchemi.data.resolve_device`.
- `OnPolicyConfig.seed` keys the mixture sampler; vary it, not the global torch
  seed, to make replicate runs draw independently.

Lower-level pieces, if you drive generation yourself: `TeacherLabelHook` is the
`AFTER_STEP` dynamics hook that attaches `teacher_*` fields to the live frame
and mirrors a stripped copy into a `DataSink` — `OnPolicyConfig.capture_sink`
picks that sink for the loop (a `GPUBuffer` stays on the device); `ReplayBuffer`
accumulates those frames behind a frozen key schema, with an `AdmissionPolicy`
deciding what enters (`OnPolicyConfig.replay_admission`) and an
`EvictionPolicy` what a full buffer drops (`FIFO` is `"fifo"`; a policy
instance goes on `replay_eviction`); `build_mixed_loader` draws each batch at an
exact reference/replay composition and **must be rebuilt after every segment**
because its batch sampler reads child dataset lengths once.

---

## Checkpoints And Restart

Checkpointing works as it does for any strategy. The mechanics are in
`docs/userguide/distillation.md` ("Operational notes"); this is what to budget
for.

**Checkpoints serialize every model, teacher included**, so a periodic
`CheckpointHook` write costs the frozen teacher's weights as well as the
student's. Size the interval for a large teacher.

**The propagator state is not checkpointed.** A checkpoint carries the
counters, the weights, the optimizer state, and the checkpointable hook state.
A resumed on-policy run reseeds its trajectory from the front of its shard with
a cold replay buffer, counts a segment the checkpoint interrupted as finished,
and opens a fresh segment, which begins by generating. A multi-rank restart
needs no device bookkeeping: `restore_checkpoint` loads onto the live
`devices`, and `run()` re-homes the optimizer state once the hook has pinned
the rank.

Two routes. `load_checkpoint` rebuilds the strategy from the spec the checkpoint
carries; pass `models=` so the propagator is rebound to the live student, and
`on_policy=`/`reference_dataset=`, which the spec cannot carry (without
`on_policy` the strategy comes back offline-shaped). `restore_checkpoint`
restores in place into a strategy rebuilt with the same propagator, scorer,
dataset, and hooks, and takes no `hooks` override:

```python
strategy = DistillationStrategy.load_checkpoint(
    run_dir / "checkpoints",
    models={"student": student, "teacher": teacher},
    on_policy=on_policy,
    reference_dataset=ref,
)
strategy.run()

strategy = DistillationStrategy(..., on_policy=on_policy, reference_dataset=ref)
strategy.restore_checkpoint(run_dir / "checkpoints")
strategy.run()
```

---

## Spec Serialization

`to_spec_dict()` carries the declarative settings and names its own class under
`strategy_cls`; rebuilding needs the models supplied because a spec names them
by role:

```python
spec = strategy.to_spec_dict()
rebuilt = DistillationStrategy.from_spec_dict(
    spec, models={"student": student, "teacher": teacher}
)
```

`on_policy` and `reference_dataset` hold a live propagator, scorer, and datasets
that no spec can describe, so `to_spec_dict` omits them with a warning, and a
strategy rebuilt from the spec runs offline over the dataloader passed to
`run()`. `from_spec_dict`, `from_checkpoint_dict`, and `load_checkpoint` take
them back as keyword arguments, beside `validation_config`, through the core
`**runtime_overrides`; a spec naming a `DistillationStrategy` subclass
dispatches to it carrying every override. Inside the loop, the declarative
`OnPolicySettings` are what `OnPolicyConfig.settings` detaches; the rest is
**runtime-only**:

- `convergence_hook` (set `fmax` instead to keep the criterion declarative)
- `capture_sink`, `replay_admission`, and `divergence`
- a policy instance on `replay_eviction` (the detached settings record it as
  `"fifo"`, with a warning unless it is a `FIFO`)

`InitialStructures.to_spec_dict()` names the store it reads, its budgets, and
`recycle`, never its position; a dataset holding its samples in memory is
refused, with the fix in the message (write it with `label_dataset` and point
the source at the path).

---

## Evaluation And Acceptance

Import from the `evaluation` subpackage, not the distillation namespace: an
acceptance run pulls in the dynamics engine and the reporting stack that
training does not need. The measurements are explained in
`docs/userguide/distillation.md` ("Evaluating the student"). `StabilityMonitor`,
`StabilityMetrics`, and `total_momentum` are core (`nvalchemi.dynamics.hooks`)
and re-exported here.

```python
from nvalchemi.training.distillation.evaluation import (
    AcceptanceThresholds,
    StudentEvaluation,
    build_acceptance_report,
    evaluate_accuracy,
    measured_bars,
    non_conservative_residual,
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

- `evaluate_accuracy` runs through a standalone `ValidationLoop`: **no
  autocast**, exact global residual sums rather than the graph-balanced loss,
  and it scores exactly the object handed over. For an EMA-trained student pass
  `strategy.inference_model["student"]` and record `weights="ema"`; on a
  reloaded strategy that slot is republished only once `run()` dispatches
  `SETUP`; before that, read `EMAHook.get_averaged_model()`.
- `evaluate_accuracy(quantities=[...])` takes names from
  `BUILTIN_ACCURACY_QUANTITIES` or `AccuracyQuantitySpec` instances; at least
  one must carry a supervised loss. `label_dtype=` pins the dtype of labels
  scored on the fly.
- `StabilityMonitor.metrics()` is a **method**, not an attribute, and needs two
  samples at two different steps. Give `warmup_steps` room to cover relaxation,
  or a transient is reported as drift. The monitor refuses a run whose
  `ctx.active_graph_mask` leaves a graph out. Set the `no_divergence` bar: the
  drift bars describe the segment before a divergence, which can look conserved.
- `extensivity_error(student, state, drop_keys=propagator.bookkeeping_keys())`
  for a batch a sampler-seeded run returned; a system field named in neither
  `extensive_keys` nor `intensive_keys` is refused rather than guessed at.
- `non_conservative_residual` is scale-dependent; read its docstring before
  quoting the number.
- **A bar with no measurement behind it fails the student.** Never hard-code
  which bars you may state: `measured_bars(*families, accuracy_quantities=...)`
  answers from the measurements in hand. The bars are an open table
  (`AcceptanceBar`, `DEFAULT_BARS`, `BAR_FAMILIES`); a custom bar's limit goes
  in `AcceptanceThresholds.extra` and its number in `StudentEvaluation.extra`.
- Every record is a pydantic `MeasurementRecord` (`to_dict`/`from_dict`), so a
  sweep can score each student in its own job and assemble one report; use
  `model_copy(update=...)`, not `dataclasses.replace`.

---

## Caveats

- Give the teacher the outputs the signals need. Requesting `stress` from a
  teacher that does not declare it fails at construction, which is the point.
- Point `target_key` at `teacher_*`, never at the reference field. A term left
  on `energy` silently trains against dataset labels.
- Prefer `normalize_weights=False` when you mean literal loss coefficients.
- Precompute labels with `label_dataset` whenever the dataset is reused; it
  removes the teacher from the training loop entirely.
- On-policy runs need a reference dataset. A `replay_ratio` near `1.0` drifts off the
  reference distribution with nothing pulling it back.
- The checkpoint interval is a teacher-size trade-off: checkpoints serialize
  every model, teacher included, so a short interval pays for the teacher's
  weights each time.
- Distributed: both loops scale with `DDPHook`. Offline sharding is ordinary
  training; the on-policy loop additionally shards the initial structures per
  rank, keeps the replay buffer rank-local, and leaves the reference dataset
  replicated.
- Core helpers the loop leans on, all public: `nvalchemi.training.evaluating`
  / `eval_configured_models`, `nvalchemi.training.losses.graph_balanced_mean`
  / `per_graph_sum` / `masked_mean`, `nvalchemi.training.unwrap_model`,
  `nvalchemi.training.distributed.all_reduce_flags` / `all_gather_rows` /
  `all_gather_objects`, `DDPHook.wrapped_keys`,
  `HessianOperator.from_energy(...).matvec(create_graph=True)`,
  `nvalchemi.data.transforms.make_supercell`,
  `nvalchemi.data.datapipes.distributed_shard` / `dataset_device` /
  `same_device`, `nvalchemi.data.resolve_device`, `Batch.index_select(drop=)`
  / `Batch.clone(drop=)` / `Batch.without_keys`,
  `DynamicsStage.ON_GRADUATE` with `ctx.graduated_mask`,
  `nvalchemi.dynamics.hooks.nonfinite_graph_mask`,
  `BaseDynamics.active_graph_mask` / `check_initial_batch` / `seed_offset` /
  `bookkeeping_keys` / `samples_equilibrium`, `NVTLangevin.random_seed`,
  `OrderedStructureSampler.rank` / `world_size`,
  `load_checkpoint(models=)`, and the `**runtime_overrides` that
  `load_checkpoint`/`from_checkpoint_dict`/`from_spec_dict` forward to the
  strategy class a spec names.

---

## Key files

| File | Contents |
| --- | --- |
| `nvalchemi/training/distillation/strategy.py` | `DistillationStrategy`, `default_distillation_fn` |
| `nvalchemi/training/distillation/scoring.py` | `TeacherScorer`, `InProcessTeacherScorer`, signal table |
| `nvalchemi/training/distillation/labeling.py` | `label_dataset` |
| `nvalchemi/training/distillation/config.py` | `OnPolicySettings`, `OnPolicyConfig` |
| `nvalchemi/training/distillation/seeding.py` | `InitialStructures`: the spec round trip of the core `OrderedStructureSampler` |
| `nvalchemi/training/distillation/replay.py` | `ReplayBuffer`, `build_mixed_loader` |
| `nvalchemi/training/distillation/hooks.py` | `TeacherLabelHook` |
| `nvalchemi/training/distillation/losses/` | `AtomicEnergyMatchingLoss`, `EmbeddingMatchingLoss` and `EmbeddingProjector`, `HessianMatchingLoss`, `BoltzmannMatchingLoss` |
| `nvalchemi/training/distillation/evaluation/` | accuracy, stability, acceptance |
| `docs/userguide/distillation.md` | Concept guide: signals, both loops, objectives, evaluation |
| `examples/intermediate/09_offline_distillation.py` | Runnable offline example |
| `examples/intermediate/10_onpolicy_distillation.py` | Runnable on-policy example |
