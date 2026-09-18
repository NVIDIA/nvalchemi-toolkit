# Changelog

## Unreleased

### Added

- Add support for PEFT fine-tuning within `FineTuningStrategy`, including
  LoRA workflows with `LoRAConfig`, `load_peft_checkpoint_into_model`,
  and base-model fingerprint checks for PEFT checkpoint loading.
- Add `DynamicsStage.ON_ADMISSION` to `BaseDynamics`, enabling hooks to
  initialize per-system state once when a batch is admitted, before force
  priming and outside compiled fused steps.
- `FusedStage(reprime_on_entry=...)` — status codes whose newly entering
  graphs skip one integrator update so the shared compute and target-stage
  `AFTER_COMPUTE` hooks can refresh forces under the new stage's context
  before it advances them.
- `ReplayBuffer.clear()` drops every stored frame and unfreezes the key
  schema, so the next `extend` freezes it afresh. It is what lets an
  on-policy restart replace a live buffer's contents with the frames a
  checkpoint carries rather than merge the two.

### Distillation

- **Teacher scoring and offline labeling** — new `nvalchemi.training.distillation`
  package. A `TeacherScorer` protocol defines the teacher-signal interface
  (`energy`, `forces`, `stress`, `atomic_energies`, `embeddings`, each mapped to
  a batch field and level; `SUPPORTED_SIGNALS`, `signal_fields`,
  `signal_for_field`, and `scorer_fields` publish the mapping).
  `InProcessTeacherScorer` implements it for a teacher loaded in the current
  process: it narrows `active_outputs` to the requested signals, builds and
  rolls back the teacher's neighbor list while hiding a composed pipeline's
  own lists, holds the teacher in evaluation mode, optionally casts outputs
  (`dtype`), and detaches everything it returns; a composition planning more
  than one neighbor-list source is refused. `label_dataset` walks a dataset
  once and persists the source fields plus the teacher fields to a resumable
  Zarr store, dropping neighbor tensors unless `keep_neighbors=True`, holding
  scorers to the `teacher_*` namespace, and refusing a chunk whose fields,
  levels, dtypes, or row shapes drift from the store's, a store an interrupted
  run left inconsistent, and a store holding more samples than the dataset.
- **Offline distillation strategy** — `DistillationStrategy` trains a student
  against a `"teacher"` frozen by omission from `optimizer_configs`. Teacher
  signals reach the loss as `teacher_*` batch fields, so any built-in term
  distills by pointing its `target_key` at one; the signal set is derived from
  those targets — a `validation_config` loss's included — and checked against
  the teacher's outputs at construction, as are both losses' prediction keys
  against the outputs the student actually computes. Stores from
  `label_dataset` train with no teacher pass, and a custom `teacher_*` field
  such a store carries is an ordinary loss target; unlabeled training and
  validation batches are labeled on the fly by an internal `BEFORE_FORWARD`
  hook that scores with autocast disabled. The serialized spec names its own
  strategy class, which `from_spec_dict` dispatches to. New
  `AtomicEnergyMatchingLoss` matches the teacher's per-atom energy
  decomposition, a signal no reference dataset carries. See the new
  `examples/intermediate/09_offline_distillation.py`.
- **On-policy generation components** — `TeacherLabelHook` is an `AFTER_STEP`
  dynamics hook that attaches `teacher_*` fields to the live frame at the level
  each signal declares, with autocast disabled, leaves the `energy` and `forces`
  driving the propagator alone, and optionally mirrors each labeled frame into a
  `DataSink` stripped of neighbor tensors, dynamics bookkeeping, and the
  propagated model's own predictions, so a stored frame is a training sample
  rather than a propagator state; labeling is idempotent per step, and a
  cadence dispatch landing right after a forced label is passed over.
  `ReplayBuffer` accumulates those frames behind a frozen key schema, with an
  optional staging device and two policy seams: an `AdmissionPolicy` masks the
  frames each `extend` admits before the schema check, and an `EvictionPolicy`
  (`select(buffer, incoming, capacity)`, `FIFO` shipped as the reference and
  the meaning of `"fifo"`) names the frames a full buffer drops;
  `OnPolicyConfig.replay_admission` and a policy instance on `replay_eviction`
  wire them into the loop's buffer as runtime-only objects, while
  `OnPolicySettings.replay_eviction` keeps the string form for recipes.
  `build_mixed_loader` draws each
  training batch with an exact reference/replay composition and requires both
  sources to carry one batch schema, at one dtype per field, on one device.
  `OnPolicyConfig` collects the segment loop's live objects over the
  JSON-native `OnPolicySettings`; its propagator is any `BaseDynamics`, its
  initial structures are any `InitialStructuresSource` — the protocol of the
  members the loop reads (`probe`, `initial_batch`, `shard`, `exhausted`,
  `draw`, `state_dict`/`load_state_dict`), a bare dataset being wrapped —
  with `InitialStructures` as the reference implementation: a cursor that
  shards per rank, restarts from a `state_dict`, round-trips through
  `to_spec_dict`, and serves structures through
  `draw(limit=..., fits=FitPolicy, on_miss="stop" | "skip")` with
  `WithinBudget` as the stock policy, and one row is checked at
  construction against the fields the propagator reads before its first force
  evaluation, then propagated through one `compute()` so a `__needs_keys__`
  output the student never produces, or a field the propagator reads that
  nothing declared, is refused before a run is paid for (a graph model is
  probed with the neighbor list its `neighbor_config` declares).
  `OnPolicyConfig.capture_sink` chooses the `DataSink` each
  segment's labeled frames are staged in before the boundary drains them into
  the replay buffer — host memory by default, a `GPUBuffer` to stay on the
  generation device — sized by the loop to `(generation_steps + 1)` frames per
  trajectory and resized through `resize(capacity)` when the sink offers one.
- **On-policy segment loop** — `DistillationStrategy` accepts `on_policy` and
  `reference_dataset`, and `run()` then drives generate-label-train segments
  until `num_steps`: seed a state batch, generate `generation_steps` frames with
  the student's own propagator, label and capture them, and take
  `training_steps_per_segment` optimizer steps on a freshly mixed reference/replay
  stream whose sampler seeds from `OnPolicyConfig.seed` plus the segment index.
  One segment is one epoch; the segment is also the restart granularity, a
  second `run()` keeps the replay buffer it filled, and the closing validation
  is skipped when a cadence already validated at the final step. The propagator
  must hold the very student module being trained, alone or composed, and is
  held in evaluation mode to generate. The reference dataset is probed at construction for
  fields the labeling hook strips, for the device it emits on, and for the
  teacher fields the propagator's scorer declares. Generated frames are staged
  on the reference dataset's device unless `replay_device` overrides it, and every
  placement blocks on a copy into host memory. `on_policy` and
  `reference_dataset` are omitted from `to_spec_dict`, which warns.
- **Relaxation on-policy generation** — `OnPolicyConfig` gains `fmax`
  and `convergence_hook`, which give a relaxation propagator such as `FIRE` the
  trajectory lifecycle its paths need: converged structures freeze, are stored
  once as the minimum they reached, and graduate out of the batch at the
  segment boundary, where the initial structures are drawn for the room they
  freed through `InitialStructures.draw(..., on_miss="skip")`, so the replay
  buffer keeps filling with informative frames instead of near-duplicates of a
  structure that stopped moving. `fmax` is the max-force-norm threshold a recipe
  can hold and `convergence_hook` the live criterion no recipe describes;
  `OnPolicyConfig.convergence_criterion` resolves the two into the one
  status-migrating, every-step hook the lifecycle drives, which is also the
  propagator's convergence detector for the duration of the run. The lifecycle
  refuses to run beside a second status migrator or a propagator-owned sampler,
  off a status the structures never carry, or under a multi-sub-stage
  `FusedStage`. `InitialStructures` gains `recycle`, which wraps the cursor to
  the front of the rows this rank owns instead of letting the batch narrow, and
  records its wrap count in the restart bundle. Frames are captured by two
  routes that partition them: `TeacherLabelHook` stores the structures still
  relaxing, narrowing to them before the teacher runs, and a converged-frame
  hook stores each minimum once off the status transition, labeled in one
  teacher pass as its sink is drained onto the buffer's own device, which a
  device-less `ReplayBuffer` now pins on its first `extend`; `TeacherLabelHook`
  narrows only when given the propagator's `exit_status`, which the lifecycle
  sets, so a propagator managing its own convergence keeps its final frames. A
  budget-graduated
  fused sub-stage is captured once the chunk returns, a backfilled structure is
  restamped with fresh bookkeeping, and a run whose last trajectory finishes
  warns once and trains its remaining steps on the frames it has. A trajectory
  whose positions or forces stop being finite is frozen uncaptured on that step
  and retired and backfilled at the boundary, with a warning counting them,
  rather than propagated and labeled as NaN into the loss. A reference
  dataset emitting on an accelerator other than `devices[0]` is refused at
  construction. The path route stages each segment in the configured
  `capture_sink`, re-sized to the trajectories still in the batch, and a
  custom `InitialStructuresSource` drives the lifecycle once its
  `initial_batch` stamps the `status` and `system_id` bookkeeping. The
  construction probe dispatches a copy of the criterion to the probed row, so
  one that raises on the propagator's outputs or leaves `status` unmoved where
  it converged is refused up front; a criterion reading a key no `compute()`
  produces warns instead of refusing, since a hook may write it during the step.
- **Multi-GPU and multi-node on-policy distillation** — the segment loop runs
  data-parallel under a `DDPHook` instead of refusing a multi-rank launch. Each
  rank propagates the strided shard of `initial_structures` it is dealt, labels
  those frames with its own teacher replica, and fills its own replay buffer,
  so no generated frame or teacher pass is duplicated; the reference dataset
  stays replicated and every rank draws from all of it. The mixture sampler's
  `seed` and every integer seed the propagator and its sub-stages expose are
  moved onto a per-rank stride so ranks decorrelate, and a stage holding a
  `torch.Generator` and no integer seed is named in a warning. The student's
  gradient all-reduce is the only cross-rank traffic; a multi-rank run whose
  student nothing wraps, or with fewer initial structures than ranks, is
  refused up front, and a structure count the world cannot deal out evenly
  warns, since a shorter shard's frames are drawn more often. A reference
  dataset staged on an indexed accelerator some rank does not train on is
  reported from every rank, because the replay buffer follows it; an
  index-less `replay_device` names this rank's current device. The rows a
  rank owns are public as `DistillationStrategy.structure_shard`.
  `TrainingStrategy` narrows its named-model device check to more than one
  *distinct* device, so a per-model list naming one device repeatedly is
  accepted, and the idiom that reaches past a data-parallel wrapper to the
  module it owns is public as `nvalchemi.training.unwrap_model`.
- **Representation, curvature, and Boltzmann objectives** — three loss terms
  distill what a reference dataset has no column for, each checked at
  construction on the training side and on a `validation_config` loss alike.
  `EmbeddingMatchingLoss` matches the teacher's per-atom representation;
  `embedding_distillation_fn` takes the student's `compute_embeddings` pass
  and routes it through an `EmbeddingProjector` registered as a `"projector"`
  model with an optimizer of its own whenever the two widths differ, and the
  student, projector, and teacher widths are reconciled up front.
  `HessianMatchingLoss` matches Hessian-vector products along one probe: the
  new `hessian` teacher signal writes `teacher_hvp` and the `teacher_hvp_probe`
  it was taken along (`InProcessTeacherScorer.label_hvp` and `probe_seed`, and
  the shared `hessian_vector_product`), `hessian_distillation_fn`
  differentiates the student's energy twice along that probe on a pass
  narrowed to the energy that reuses the stock forward's neighbor list, a
  companion field is refused as a loss target, a direct-force student is
  warned that the term reaches its energy head alone, and
  `DistillationStrategy.validate` pins the probe per validation batch so the
  metric compares across passes. `BoltzmannMatchingLoss` is the
  beta-interpolated relative entropy between the teacher's and student's
  Boltzmann distributions at a temperature over the batch's configurations,
  read as a sample of the student's own ensemble: it requires `on_policy`,
  refuses a relaxation propagator and any convergence criterion — the
  propagator's own, one registered on it, or `fmax`/`convergence_hook` — and
  any place in the validation loss, an explicit validation-side term or a
  `ValidationConfig` that would reuse the training loss, warns about a mixed
  `replay_ratio` or an unbounded replay buffer, and under data parallelism
  gathers the reduced energies across ranks with a differentiable all-gather so
  every rank trains on the world-batch loss. `from_spec_dict`,
  `from_checkpoint_dict`, and `load_checkpoint` take `on_policy` and
  `reference_dataset` (and `load_checkpoint` takes `models`, since the
  propagator holds the live student), so a Boltzmann run's checkpoint restores
  with its loop, and a spec naming a `DistillationStrategy` subclass
  dispatches to it carrying both. `EmbeddingProjector.compute_embeddings`
  writes through the public `add_key` path.
- **Evaluation and acceptance suite** — new
  `nvalchemi.training.distillation.evaluation` subpackage deciding whether a
  distilled student ships. `evaluate_accuracy` measures energy, force, and
  stress MAE/RMSE over a holdout against the dataset's own labels or the
  teacher's (on disk or scored on the fly) through `ValidationLoop`, with no
  autocast, a scorer's labels cast to the dtype the store would hold them at,
  and exact global residual sums accumulated in float64; against a teacher it
  adds force cosine similarity per atom and magnitude-weighted (the aggregate
  is what `min_force_cosine` reads), per-atom energy residuals, and a
  `force_nonfinite_atoms` count, and it refuses a scorer paired with reference
  targets or one returning a label outside `teacher_*`.
  `non_conservative_residual` integrates the teacher's work around
  closed loops in configuration space, laid out around each graph's own
  centroid, and reports the lower bound it places on a conservative student's
  RMS per-atom force error, absolute and relative to each graph's force scale.
  `StabilityMonitor` is a dynamics hook reporting energy drift (per atom, per
  step, and as a fitted per-nanosecond rate), the RMS fluctuation and largest
  excursion about the fit, and momentum conservation over a student-driven
  trajectory, discarding a `warmup_steps` window, naming the field a sample
  lacks, and stopping with a warning when the batch composition changes;
  `extensivity_error` checks energy scaling across replicated cells with every
  field carried into the supercell; `radial_distribution` and
  `compare_radial_distributions` score structural match with a bounded
  Jensen-Shannon divergence, pooled over every species or resolved to one
  pair, each pair apportioned between two bins so the histogram is continuous
  in the positions, and a frame enclosing no volume is refused.
  `measure_throughput` reports atoms/s and ns/day from a warmup-discarded,
  device-synchronized window over the steps the propagator actually took.
  `build_acceptance_report` turns those measurements into per-student verdicts
  against `AcceptanceThresholds`, a speed-versus-accuracy Pareto table, and
  the from-scratch gate (`max_from_scratch_ratio`, the largest accepted ratio
  of the distilled error to an equal-size from-scratch student's), rendering
  as Rich tables and exporting as nested dictionaries or flat scalars: a bar
  with no measurement behind it fails rather than being skipped, a bar whose
  family was measured but whose number was not names the missing quantity or
  timestep, a non-finite
  measurement fails on `not finite` and is left off the Pareto front, a
  baseline of exactly zero is unbeatable, and a family scored on different
  holdouts or timed on different batches is refused. `BAR_FAMILIES` maps each
  bar to the `StudentEvaluation` slots it reads and `measured_bars` answers
  which bars a partial measurement can decide; every measurement rebuilds from
  its export with `from_dict`, and `StudentEvaluation.weights` records whether
  a student was scored on `"ema"` or `"raw"` weights.
- **Reproducible recipes, teacher references, and the `distill` CLI** — a
  distillation run now survives a round trip. Checkpoints store the frozen
  teacher *once per checkpoint root*:
  `DistillationStrategy.checkpoint_model_references` declares it, the first
  write under a root holds its weights, the manifest gains a `model_references`
  entry naming that index plus a fingerprint, later indices contribute no
  teacher weight file, and a load reads the stored copy back and verifies the
  fingerprint, so a replaced copy raises instead of quietly training a student
  against a different model. The fingerprint hashes each state-dict entry's
  name, shape, dtype, and a sample of its values read at `float64` on the host,
  so it identifies a model rather than validating it. One root holds one copy:
  storing a *different* copy of a declared model into a root that already
  holds one is refused, while an identical copy is written again freely, which
  repairs a root whose weight file went missing. The manifest stays at
  `schema_version` 1, so an older nvalchemi still reads it, but only at the
  index holding the teacher's weights; the teacher's `checkpoint_spec()`
  rebuilds its architecture and is never trusted for its weights, and
  `save_trainable_state_only=True` narrows the student alone, so the
  once-stored teacher stays whole.
  `OnPolicyConfig.to_spec_dict`/`from_spec_dict` carry the whole segment loop —
  every `OnPolicySettings` field verbatim, the propagator as the `cls_path` and
  keyword arguments it rebuilds from with the student rebound at build time,
  the scorer as its `signals`, `dtype`, and `probe_seed` over the strategy's own
  `"teacher"`,
  and `initial_structures` as the store it reads under its budgets and
  `recycle`, never its cursor, or, for another `InitialStructuresSource`, as
  its own `to_spec_dict` under its class path (`source_cls`), a source with
  neither `to_spec_dict` nor `from_spec_dict` being refused with the remedy —
  while a `convergence_hook`, `capture_sink`, `replay_admission`, a policy
  instance on `replay_eviction` (recorded as `"fifo"`), a propagator's live
  hooks and sinks, and an in-memory dataset stay runtime-only and are named
  rather than approximated, and a `MultiDataset` travels as the list of stores
  it concatenates. `DistillationStrategy.to_spec_dict` carries
  `on_policy` and `reference_dataset` on the same terms, a spec naming a
  subclass under `strategy_cls` rebuilds that subclass with every runtime
  override handed on, a live object passed to `from_spec_dict`,
  `from_checkpoint_dict`, or `load_checkpoint` outranks the recipe, and the
  stores a recipe names are opened on the rebuilt strategy's own device, so a
  checkpoint restored under another `map_location` reads its data there. An
  interrupted on-policy run resumes its trajectory, propagator counter,
  initial-structure cursor, and replay frames through the checkpoint — exactly
  for the counter-based-RNG integrators, at segment granularity, with the
  labeling cadence resumed so the restart neither pays a second teacher pass
  at the boundary it stopped on nor stores the frame beside it; the restored
  frames replace the buffer's contents rather than merging into them, a
  setting the resumed loop sets differently from the recorded one is reported,
  and a run whose generation ran dry resumes training on its buffer rather than
  regenerating.
  The bundle is rank-local, since it rides in a strategy checkpoint
  `CheckpointHook` writes on rank zero alone, so a world size differing at
  either end of a restart — read off the shard the cursor records — drops it
  with a warning and the rank reseeds from its own share with a cold replay
  buffer. New `nvalchemi-training distill` group (aliased `nvalchemi-distill`)
  authors, validates, runs, and gates a JSON `DistillationJobSpec`: `init`
  scaffolds offline or on-policy recipes at size-only student tiers, with a
  `CheckpointHook` in `student.hooks` and `dataset.batch_size` (`--batch-size`,
  default `8`) recorded, and requires `--initial-structures` in on-policy mode;
  `spec report` renders derived teacher signals, batch composition, and
  acceptance bars, and refuses everything a recipe settles on its own before a
  teacher reaches a device — an `on_policy` block is validated through
  `OnPolicySettings`' and `InitialStructures`' own constraints, so an
  out-of-range setting, a misspelled or non-positive budget, a block naming no
  store, `recycle` under no `fmax`, a step budget below one, a `dataset.format`
  no loader builds, an unloadable teacher or student source, a `replay_ratio`
  or `batch_size` leaving one mixture source without a whole sample, and a
  `replay_device` off the reference dataset's device are all refused at
  `spec report`; `spec run` executes and `spec resume` continues from a
  checkpoint directory and the recipe, both writing a terminal checkpoint at
  the next index whenever the run ended on a step the cadence missed, and both
  taking `--distributed/--no-distributed` (auto when `WORLD_SIZE > 1`) and
  `--ddp-backend`; `evaluate` scores the weights the recipe trained — the EMA
  average when `student.hooks` carries an `EMAHook` — over the recipe's
  holdout, exits non-zero on a missed bar, and writes a non-finite metric to
  `--json-out` as the string `"nan"`, `"inf"`, or `"-inf"`, which every
  metric's `from_dict` decodes back into the float, and records which of the
  two weight sets it scored as `StudentEvaluation.weights`.
  `evaluation.thresholds` is narrowed to the accuracy bars `evaluate` can fill,
  so a stability, throughput, extensivity, RDF, or from-scratch bar, or an
  accuracy bar reading a quantity the recipe never compares, is refused when
  the recipe is parsed. A multi-rank `spec resume` defaults `--map-location`
  to this rank's device, so no rank stages its weights through rank zero's; the
  live strategy's `devices` decide where the restored run continues. See
  `docs/userguide/distillation_recipes.md` and the `nvalchemi-distillation`
  agent skill.
- **Distillation user guide and on-policy example** — new
  `docs/userguide/distillation.md` covers the whole feature from the user's
  side: the teacher signals and how the strategy resolves them, the offline
  path over a teacher-labeled Zarr store, the on-policy segment loop with its
  mixture, cadence, and capacity arithmetic, the neighbor-list hooks a graph
  student needs on the propagator and on the strategy, the convergence
  lifecycle a relaxation propagator needs, the representation, curvature, and
  Boltzmann objectives and what each asks of the run, scaling the loop across
  ranks, the accuracy, stability, throughput, and extensivity measurements and
  the acceptance report that gates the student on them, and the checkpoint and
  restart contract. Two topics get their own treatment: why an on-policy
  reference dataset has to be teacher-labeled and how to reshape an existing
  reference set into one, and distilling a non-conservative direct-force
  teacher into a conservative student. New
  `examples/intermediate/10_onpolicy_distillation.py` runs three
  generate-label-train segments on CPU against a labeled reference dataset.

### Fixed

- **Dynamics hook lifecycle** — fused-level hooks now fire at the
  `BEFORE_PRE_UPDATE`, `AFTER_PRE_UPDATE`, `BEFORE_POST_UPDATE`, and
  `AFTER_POST_UPDATE` boundaries, and sub-stage `BEFORE_COMPUTE` hooks now
  fire. Fused-level `AFTER_COMPUTE` hooks now run after the sub-stage loop
  instead of before it. Existing workarounds that register the same hook at
  both fused and sub-stage levels will therefore invoke it twice at each
  matching boundary and should remove the duplicate registration.
- **`FusedStage` force priming** — a dynamics instance's own adaptive
  optimizer state (e.g. FIRE's per-graph `dt`/`alpha`/step counters in
  `self._state`) is now preserved across masked `pre_update`/`post_update` calls.

### Deprecated

- `FusedStage.register_fused_hook()`. Use the inherited `register_hook()`
  method instead; hooks on a `FusedStage` already observe the complete fused
  batch.

## 0.2.0 — 2026-08-07

### Added

- Domain decomposition for distributed inference and dynamics: a spatial halo
  strategy and a graph-parallel strategy, both driven by a declarative
  `MLIPSpec` a model wrapper publishes as `distribution_spec`. Ewald, PME,
  MACE, AIMNet2 and UMA ship specs; composed pipelines decompose per stage.
  Energy, forces and stress agree with a single-GPU reference to fp32 rounding
  under both strategies, eager and compiled. `nvalchemi.distributed.pin_fp32`
  pins full-precision fp32 for runs that must match a reference, since TF32
  makes distributed and single-process results diverge well beyond fp32 noise.

- MACE training example for end-to-end model training workflows.
- `EMAHook._build_averaged_model` override seam, so a caller that owns
  model sharding can supply a pre-built `AveragedModel` instead of the
  default deepcopy — enabling EMA on `fully_shard` (FSDP2) / DTensor
  models. Default behaviour unchanged.
- Checkpointable training hooks. Hooks such as EMA can now save restart
  state with strategy checkpoints, so resumed training keeps averaged
  weights instead of starting them over.
- Training strategy checkpoint restart support, including a periodic
  checkpoint hook for step- or epoch-based saves and restart loading with
  models, optimizers, schedulers, runtime counters, and restart-safe device
  placement.
- PhysicsNeMo-compatible atomic datapipes with `MultiDataset` composition,
  multidataset-aware sampling policies, and fused batch loading that preserves
  the Zarr reader's coalesced I/O path.
- First-class validation on `TrainingStrategy`. Set a `ValidationConfig`
  on `strategy.validation_config` and validation runs automatically at the
  configured step or epoch cadence, plus one final pass at end-of-training;
  the latest summary is stored on `strategy.last_validation`. Mechanics live
  in a public, context-managed `ValidationLoop` that can also be run
  standalone outside training. An `inference_model` slot lets EMA (or SWA /
  a distillation teacher) publish averaged weights for validation to read.
  A new `AFTER_VALIDATION` hook stage fires immediately after each pass so
  loggers can read the live summary. For per-batch logging, pass a
  `batch_callback` (any object matching the `BatchValidationCallback`
  protocol) on the config; it is invoked once per validation batch with the
  batch, predictions, and per-batch loss.
- Metric-driven learning-rate schedulers. `ReduceLROnPlateau` is now
  supported via `OptimizerConfig.scheduler_metric_adapter` (a summary-dict
  key string or a callable). Time-based schedulers step every optimizer
  step as before; metric-driven schedulers step only at validation
  checkpoints, where the validation summary supplies the metric.

- Python 3.14 support across the core package and the cu12/cu13 CUDA extras,
  including pure-`pip` installs: `requires-python` is now `>=3.11,<3.15`.
  Python 3.15 is not publicly supported yet (upstream wheels missing).
- numpy relaxed to `>=2,<3` — downstream users may use any numpy 2.x.

### Model Wrappers

- **Pipeline neighbor-list adaptation policy** — `PipelineModelWrapper`
  now accepts `neighbor_adaptation` (`"auto"`, `"always"`, `"never"`) and
  `max_cutoff_ratio` (default `1.5`). The default `"auto"` mode only filters
  a source neighbor list for a smaller cutoff when the source cutoff is at most
  `max_cutoff_ratio` times the target cutoff; larger gaps get separate source
  lists. `"always"` builds one max-cutoff source list, while `"never"` builds
  exact cutoff source groups and skips cutoff filtering.

### Core Data Layer

- **Extensible batch levels** - `LevelSchema` and `Batch` now support custom
  uniform, segmented, and ordered product levels. Custom definitions,
  fields, and product-derived cardinalities are preserved through construction,
  reconstruction, selection, append, reusable buffers, point-to-point transport,
  and Zarr persistence. Pointer-only levels remain available in direct `Batch` and
  Zarr storage workflows. Existing atom, edge, and system APIs remain compatible,
  and legacy-only Zarr stores retain their existing layout.
- **In-memory datapipes** - new `InMemoryDataset` stores a fully materialized
  `Batch` in memory and serves graph-indexed `Batch` selections through the
  same `load_batches` / fused-prefetch interface used by `DataLoader`. It can
  be constructed from an existing `Batch` or materialized from a reader in
  chunks, with optional field-level metadata and batch transforms.
- **User-specified transforms** - `Dataset` accepts a `transforms=` kwarg
  (per-sample `(AtomicData, metadata) -> (AtomicData, metadata)`) and
  `DataLoader` accepts a `batch_transforms=` kwarg (per-batch `Batch -> Batch`).
  Both default to `None` (backward compatible). New `nvalchemi.data.transforms`
  subpackage exposes a polymorphic `Compose` utility plus `SampleTransform`
  and `BatchTransform` type aliases, re-exported from `nvalchemi.data`.
  Per-sample transforms run after device transfer on both sync and prefetch
  paths; per-batch transforms run on the consumer thread after `Batch.from_data_list`.
  Transform failures are wrapped in `RuntimeError` with `transform[<i>]`
  breadcrumb and `__cause__` preserved.

### Models

- **UMA (fairchem-core) wrapper** — new `UMAWrapper` exposes UMA
  (Universal Models for Atoms) foundation models (`uma-s-1p1`,
  `uma-s-1p2`, `uma-m-1p1`) through the `BaseModelMixin` interface,
  ready for any dynamics engine or standalone inference. UMA is
  multi-task; the wrapper is pinned to one head at construction (OMol,
  OMat, OC20, ODAC, OMC). Input conversion is tensor-native (no ASE
  round trip); energy is the differentiable primitive with forces and
  (for periodic tasks) stress from autograd. Install via the new `uma`
  optional CUDA variants (`pip install 'nvalchemi-toolkit[uma-cu12]'` or
  `nvalchemi-toolkit[uma-cu13]`), which remain incompatible with `mace` because
  of their `e3nn` pins. `from_checkpoint` forwards fairchem's `inference_settings`,
  including the compiled `"default"` and `"turbo"` presets and the eager
  `"batch"` preset. See the
  `examples/advanced/09_uma_nve.py` NVE/NVT/NPT walkthrough.

### Fixed

- Cap `plotext<6`: plotext 6 removed `clf()`, which hooks/reporting and the
  training CLI call; fresh resolves were silently installing 6.x and breaking
  Rich dashboards and `nvalchemi-training` on all Python versions.

- **UMA CUDA dependency resolution** — add standalone `uma-cu12` and
  `uma-cu13` extras. They select the matching torch build without installing
  PhysicsNeMo's RAPIDS extras, whose numba upper bound conflicts with Fairchem
  2.22.

- **Segment expansion on a non-default GPU** — the Warp expansion kernel behind
  `Batch.index_select` was launched against whichever CUDA device happened to be
  current, so a batch whose storage records a bare `cuda` while its tensors live
  on another GPU read unmapped memory (`an illegal memory access was
  encountered`) on every host where the current device is not `cuda:0`, and the
  launch left that device selected for the rest of the process. The kernel now
  launches on the device the batch pointer lives on and restores the caller's
  current device.
- **Level storages recorded an unresolved device** — `to_device("cuda")` and
  construction with `device="cuda"` stored the bare request while the tensors
  landed on whichever GPU was current, so the storage's `device` disagreed with
  its own data as soon as the current device changed and later pointer builds
  and concatenations raised `Expected all tensors to be on the same device`. A
  bare `cuda` is now resolved to the current device at the moment it is
  recorded. A `Batch` built around a storage takes that storage's device rather
  than resolving the request on its own, so `Batch(storage=..., device="cuda")`
  no longer reports the current GPU while its data sits on another one; an
  indexed device that contradicts the storage raises `ValueError`. A batch that
  allocates its own storage builds it on the requested device too, so
  `Batch(device="cuda:1")` no longer records `cuda:1` while every tensor
  assigned to it lands on CPU.
- **Merging batches held on different devices** — the bulk merge behind
  `MultiLevelStorage.from_batches` and `MultiLevelStorage.concatenate` moved
  every contributed tensor to the merge device except `segment_lengths`, which
  it read where each group already held them, so folding a CPU storage into a
  CUDA one raised `Expected all tensors to be on the same device` out of
  `torch.cat`. The segment lengths are now moved like everything else.
- **Low-precision graph-balanced losses** — `per_graph_sum` accumulated in the
  input dtype, and CUDA scatter atomics round after every add, so a bf16 running
  sum stopped growing at 256 and an fp16 one at 2048. A per-atom-normalized
  force loss over 3000 atoms was wrong by a factor of 3.6 in bf16 and 10% in
  fp16. Sums now accumulate in at least fp32 and are returned in fp32 for
  half-precision inputs — on the padded `(B, V_max, 3)` force layout as well
  as the dense `(V, 3)` one — so a per-graph total past the fp16 ceiling of
  65504 no longer saturates to `inf` before the loss normalizes it, and a
  half-precision force loss returns the same fp32 value whichever layout it is
  given; fp32 and fp64 results are unchanged. The dense graph-balanced path sums
  each atom's three Cartesian components in fp32 too, so a single fp16 residual
  large enough to overflow that inner sum (components near 150) no longer makes
  the dense loss `inf` where the padded loss is finite.
- **Demo model embeddings on a batch** — `DemoModelWrapper.compute_embeddings`
  set `node_embeddings` as a plain attribute, which a `Batch` routes to its
  system group, so the per-atom tensor failed the batch-size check and the call
  raised on every batch. Node embeddings are now written through
  `Batch.add_key(..., level="node")`, which registers the field with the
  storage's attribute map so a later plain `batch.node_embeddings = ...` routes
  back to the atoms group instead of the system group. `MACEWrapper` writes its
  node embeddings through the same path. The graph embeddings the same call
  returns were also pooled with an unexpanded `(N, 1)` scatter index, which
  `scatter_add_` does not broadcast over an `(N, H)` source, so every feature
  but the first came back zero; the index is now expanded and all `H` features
  are summed.
- **Segfault on a cross-device buffer write** — `Batch.put` and the
  `GPUBuffer.write` that calls it took the Warp launch device for their fit-mask
  kernel from the *source* batch, so writing a CPU batch into a CUDA buffer ran
  the kernel on the host against CUDA destination pointers and killed the
  process with a segmentation fault rather than raising. `GPUBuffer.write` now
  moves an incoming batch to the buffer's device, as `HostMemory.write` already
  moves its items to CPU; `Batch.put` raises `ValueError` on a source held
  elsewhere; and the put kernels launch on the destination and refuse a
  mixed-device pair.
- **Attribute writes routed past their own group** — a tensor assigned to a
  `Batch` resolved its level through the attribute map alone, so any key the map
  did not know about went to the system group even when the batch already held
  it at node or edge level, leaving the same name at two levels and breaking the
  next `to_data_list()`. A write now follows the group that already holds the
  key, and `Batch.add_key` registers what it adds.
- **Half-precision totals in the default loss reduction** — the validity-weighted
  mean every loss leaf inherits from `BaseLossFunction.reduce` summed in the
  residual's dtype, so a finite per-graph fp16 residual saturated the moment its
  total passed 65504: an `EnergyMSELoss` over 64 graphs with a 40 eV residual
  returned `inf`. Both sums now accumulate in at least fp32 and a half-precision
  input returns an fp32 loss, as the force terms already did; fp32 and fp64 are
  bit-identical.
- **Validation summaries over half-precision losses** — the validation loss
  accumulator kept its running sums in the loss's own dtype and widened only
  when the summary was built, so a bf16 running sum stopped growing once each
  batch's contribution fell below half an ulp: 500 batches averaging 0.8
  reported 0.512, 36% low (fp16: 0.75% low). Every running sum is now widened to
  float64 as it is taken.
- **`TrainingStrategy.validate()` before `run()`** — models were moved to
  `devices` only by `run()` and the checkpoint restore path, so a standalone
  validation pass on a CUDA strategy fed GPU batches to CPU models and failed
  with `Expected all tensors to be on the same device`. `validate()` now makes
  the same (idempotent) move, and places a published `inference_model` the same
  way, so an EMA slot filled before `devices` changed no longer meets batches on
  a device it was never moved to.
- **Checkpoint resume across devices** — a live restore loaded weights and
  optimizer state onto the device recorded in the checkpoint rather than the one
  the live strategy runs on, and `run()` reused resumed optimizer state without
  following the models it had just moved. Resuming a `cuda:0` checkpoint on
  another GPU, or on a rank a `DDPHook` re-pins, died in the first optimizer
  step with `Expected all tensors to be on the same device`. Live restores now
  target the live device, and the new
  `nvalchemi.training.rehome_optimizer_state` helper (applied automatically
  whenever a resumed optimizer is reused, by `run()` and by `train_batch()`)
  moves resumed state onto its parameters, including tensors a custom optimizer
  nests inside dicts, lists, or tuples. On that path `map_location` only stages
  the load — the live strategy's `devices` still decide where the restored
  objects come to rest — so the returned `strategy_metadata` now reports the
  strategy's devices instead of the raw `map_location`, which could name a
  device none of the restored models were on.
- **Ewald charge gradients and cell derivatives** — the reciprocal term was only
  ever differentiated with respect to positions and charges, so a non-hybrid
  Ewald returned a wrong `dE/dq`, and strain-autograd through the detached
  Green's function gave a wrong stress. Work needing a cell derivative now
  routes to the staged reciprocal.
- **Ewald / PME strain cache** — cached k-vectors were rebuilt from the strained
  cell, so a second stress evaluation reused the first call's autograd graph.

- **Distributed dynamics lifecycle** — keep per-system integrator state aligned
  when pipeline receives and graduates systems, clear reusable communication
  buffers before every send without shrinking their segmented capacity, and run
  distributed stages with explicit per-system step budgets and optional early
  convergence.
- **Zarr dataloader custom fields** — validated `Dataset` batch paths now
  preserve reader field-level metadata so custom atom-, edge-, and
  system-level tensors survive batching like the `skip_validation` path.
- EMA checkpointing now restores averaged tensors to the corresponding live
  model tensor devices, publishes restored EMA weights during SETUP before validation,
  and supports callable reconstruction specs for model wrappers that must
  rebuild from factory methods, including MACE checkpoints with
  cuEquivariance enabled.
- **NVT Nosé-Hoover velocity collapse** (#104) — reset the NHC
  `total_scale` scratch accumulator to the multiplicative identity on
  each chain update, preventing persistent state from zeroing or
  compounding velocity rescaling.
- **MTK NPT barostat runaway** (#89, #90) — four bugs in
  `nvalchemi/dynamics/integrators/npt.py` (with matching fixes in
  `nph.py`) that combined to drive unbounded cell-volume drift in long
  NPT runs. Cross-validated against ASE `MTKNPT`/`IsotropicMTKNPT` and
  TorchSim `npt_nose_hoover_isotropic`. Isotropic users will see their
  barostat mass `W` shrink by 3× (now matches canonical MTK).
- **Ewald / PME energies buffer leak** (#82) — in-place `scatter_add_`
  of gradient-carrying `per_atom_energies` chained each forward's Warp
  backward tape onto `_energies_buf`, causing linear per-step slowdown
  and unbounded GPU memory growth. `detach_()` the buffer after each
  forward.
- **FusedStage graduation not reported** — `FusedStage.step()` returned
  `exit_converged=None` for samples graduated via a sub-stage's `n_steps`
  counter or its `ConvergenceHook`, so consumers of the returned indices
  (e.g. `DistributedPipeline`) silently dropped such samples.

### Deprecated

- `cells_inv` argument on `_cell_kinetic_energy`. Cell kinetic energy
  is computed directly from the strain rate `ε̇` and no longer needs
  the cell inverse. The argument is retained for backwards
  compatibility (a `DeprecationWarning` is emitted when passed) and
  will be removed in a future release.

### Breaking Changes

- The `cu12`/`cu13` extras no longer install the RAPIDS stack (`cuml`, `cupy`,
  `pylibraft`, NVIDIA DALI) or PhysicsNeMo's CUDA extras — they now provide the
  CUDA torch build, `nvalchemi-toolkit-ops`, `cuequivariance-ops-torch`, and
  PhysicsNeMo core. Nothing in `nvalchemi` imports the RAPIDS stack, and this
  removes upstream pins that made `pip install nvalchemi-toolkit[cu12]`
  unresolvable on Python 3.14. Users needing RAPIDS should install it
  directly (`cuml-cuXX`, `cupy-cuda1Xx`).

- `EwaldModelWrapper` and `PMEModelWrapper` now default to `hybrid_forces=False`.
  The analytic direct-output path (`hybrid_forces=True`) does not produce
  consistent gradients and is not supported under domain decomposition, where
  `distribution_spec` rejects it. Forces and stress now come from autograd over
  the energy; pass `hybrid_forces=True` explicitly to keep the old path.

- Dataset-level explicit batch reads now use `load_batches(...)`. The raw
  `read_many(...)` API remains on readers, where storage backends can optimize
  ordered I/O, but `Dataset.read_many(...)` and `Dataset.get_batch(...)` have
  been removed to keep the public Dataset API focused on sample access,
  batch materialization, and prefetching.
- Split hook context state into `HookContext`, `DynamicsContext`, and
  `TrainContext` so each workflow exposes only the fields it owns.
  Dynamics-specific state such as `step_count`, `converged_mask`, and
  `global_rank` now lives on `DynamicsContext`, while training state lives on
  `TrainContext`. Existing hooks that used `HookContext` for dynamics-only
  fields should update their annotations to `DynamicsContext`.
- Standardized public `stress` outputs on tensile-positive Cauchy stress
  (`sigma = -W / V`) while keeping low-level virials defined as negative
  strain derivatives.
- Removed `EvaluateHook` in favor of first-class validation on
  `TrainingStrategy`. Validation is no longer a registered hook. Migrate by
  moving the hook's arguments onto a `ValidationConfig`:

  ```python
  # Before
  strategy.register_hook(
      EvaluateHook(validation_data=val_data, every_n_epochs=1)
  )

  # After
  strategy.validation_config = ValidationConfig(
      validation_data=val_data, every_n_epochs=1
  )
  ```

   Validation then runs automatically during `strategy.run(...)` at the
   configured cadence and once at end-of-training. The `EvaluationSink` /
   `EvaluationZarrSink` output classes were removed; replace summary logging
   with an `AFTER_VALIDATION` hook and per-batch logging with a
   `ValidationConfig(batch_callback=...)`.

## 0.1.0 — 2026-04-16

Initial public-beta release of NVIDIA ALCHEMI Toolkit, a GPU-first Python
framework for AI-driven atomic simulation workflows.

### Core Data Layer

- **AtomicData** — Pydantic-backed graph representation of atomic systems
  (positions, atomic numbers, masses, node/edge properties) with factory
  constructors `from_atoms()` (ASE) and `from_structure()` (pymatgen).
- **Batch** — GPU-resident graph batch with `MultiLevelStorage` backend
  supporting node-, edge-, and system-level tensors. Lazy `batch_idx`/`batch_ptr`,
  `index_select`, `append`, and `from_data_list` for efficient batching.
- **Zarr I/O** — `AtomicDataZarrWriter` and `AtomicDataZarrReader` with
  configurable Zstd compression, chunking, and sharding for high-throughput
  trajectory storage.
- **Dataset & DataLoader** — CUDA-stream prefetching, async I/O, and
  drop-in `DataLoader` replacement yielding `Batch` objects.

### Model Wrappers

All wrappers implement `BaseModelMixin` with a unified `ModelConfig` for
capability declaration and runtime control.

- **DemoModelWrapper** — Lightweight test/demo model (point-cloud energy +
  autograd forces).
- **MACEWrapper** — MACE equivariant neural network; supports foundation
  checkpoints; COO neighbor format; conservative forces via autograd.
- **AIMNet2Wrapper** — AIMNet2 atom-in-molecule network; energy, forces,
  charges, stress; MATRIX neighbor format; NSE auto-detection.
- **LennardJonesModelWrapper** — Warp-accelerated single-species LJ with
  analytical forces and optional virial stress.
- **EwaldModelWrapper** — Real + reciprocal space Ewald summation for
  periodic charged systems; k-vector caching; hybrid analytical forces.
- **PMEModelWrapper** — Particle Mesh Ewald (FFT-based, O(N log N)) for
  large periodic systems.
- **DFTD3ModelWrapper** — DFT-D3(BJ) dispersion correction with
  auto-downloaded reference parameters and cutoff smoothing.
- **PipelineModelWrapper** — Compose multiple models into groups with
  independent derivative strategies (autograd vs. analytical).

### Dynamics Engine

- **BaseDynamics** — Abstract base orchestrating model evaluation, integrator
  updates, hook dispatch, convergence detection, and inflight batching.
- **9 hook insertion points** per step (`DynamicsStage` enum): `BEFORE_STEP`,
  `BEFORE_PRE_UPDATE`, `AFTER_PRE_UPDATE`, `BEFORE_COMPUTE`, `AFTER_COMPUTE`,
  `BEFORE_POST_UPDATE`, `AFTER_POST_UPDATE`, `AFTER_STEP`, `ON_CONVERGE`.
- **ConvergenceHook** — Flexible convergence criteria with `from_fmax()`
  convenience constructor and per-system masking.

#### Integrators

- **NVE** — Velocity Verlet; symplectic, time-reversible, energy-conserving.
- **NVTLangevin** — BAOAB Langevin dynamics with Ornstein-Uhlenbeck
  thermostat for canonical sampling.
- **NVTNoseHoover** — Nosé-Hoover chain thermostat with Yoshida-Suzuki
  factorization; deterministic and ergodic.
- **NPT** — Martyna-Tobias-Klein isothermal-isobaric with dual Nosé-Hoover
  chains (particle + cell DOFs).
- **NPH** — MTK isenthalpic-isobaric without thermostat.

#### Optimizers

- **FIRE** — Fast Inertial Relaxation Engine with adaptive timestep.
- **FIREVariableCell** — FIRE with NPH-like variable-cell propagation.
- **FIRE2** — Improved FIRE (Shuang et al. 2020) with better restart
  conditions and modified velocity mixing.
- **FIRE2VariableCell** — FIRE2 with variable-cell structural relaxation.

### Built-in Hooks

**Dynamics hooks** (`nvalchemi.dynamics.hooks`):

- `LoggingHook` — Per-graph scalar statistics with thread-pooled I/O and
  optional CUDA stream prefetch.
- `NaNDetectorHook` — Immediate NaN/Inf detection in forces and energy.
- `MaxForceClampHook` — Clamps force magnitudes to prevent numerical
  explosions.
- `EnergyDriftMonitorHook` — Cumulative energy drift tracking with
  configurable thresholds (absolute and per-atom-per-step).
- `FreezeAtomsHook` — Freezes selected atoms by category during MD.
- `SnapshotHook` — Periodic full-state snapshots to a `DataSink`.
- `ConvergedSnapshotHook` — Snapshot on convergence.
- `ProfilerHook` — Per-stage wall-clock profiling with NVTX annotations
  and CSV output.
- `AlignCellHook` — Upper-triangular cell alignment for variable-cell
  optimization.

**General hooks** (`nvalchemi.hooks`):

- `NeighborListHook` — On-the-fly neighbor list construction/refresh with
  Verlet skin buffer; MATRIX and COO formats.
- `WrapPeriodicHook` — GPU-accelerated PBC wrapping via Warp kernel.
- `BiasedPotentialHook` — External bias potentials for enhanced sampling
  (umbrella sampling, metadynamics, etc.).

### Multi-stage Pipelines

- **FusedStage** (`+` operator) — Compose dynamics stages on a single GPU
  with shared forward pass and masked updates per sub-stage.
- **DistributedPipeline** (`|` operator) — Distribute stages across GPU
  ranks with blocking inter-rank communication.
- **SizeAwareSampler** — Bin-packing inflight batching that respects
  `max_atoms`, `max_edges`, and `max_batch_size` constraints.
- **Data sinks** — `HostMemory` (CPU), `GPUBuffer` (device), `ZarrData`
  (persistent disk) for capturing pipeline outputs.

### GPU Primitives

All low-level kernels built on
[`nvalchemi-toolkit-ops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops)
via NVIDIA Warp:

- Velocity Verlet position/velocity updates
- BAOAB Langevin half-steps
- Nosé-Hoover chain integration
- MTK barostat (NPT/NPH) cell and position propagation
- FIRE/FIRE2 coordinate and cell steps
- Kinetic energy and velocity initialization
- Neighbor list rebuild with Verlet skin
- Cell alignment to upper-triangular form

### Developer & Agent Experience

- 20 worked examples across four tiers (basic, intermediate, advanced,
  distributed) covering data structures, optimization, MD ensembles,
  Zarr I/O, inflight batching, custom hooks, model composition, Ewald
  electrostatics, and multi-GPU pipelines.
- 7 Claude Code agent skills (`.claude/skills/`) for guided workflows:
  model wrapping, data structures, data storage, dynamics API, dynamics
  hooks, dynamics implementation, and engineering scoping.
- `OptionalDependency` guards for graceful degradation when MACE, AIMNet2,
  ASE, or pymatgen are not installed.

### Requirements

- Python 3.11–3.13
- PyTorch >= 2.8
- `nvalchemi-toolkit-ops[torch]` >= 0.3.1
- Optional: `[mace]`, `[aimnet]`, `[ase]`, `[pymatgen]` extras
