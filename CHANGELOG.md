# Changelog

## Unreleased

### Added

- Add support for PEFT fine-tuning within `FineTuningStrategy`, including
  LoRA workflows with `LoRAConfig`, `load_peft_checkpoint_into_model`,
  and base-model fingerprint checks for PEFT checkpoint loading.
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
- `ReplayBuffer.clear()` drops every stored frame and unfreezes the key
  schema, so the next `extend` freezes it afresh. It is what lets an
  on-policy restart replace a live buffer's contents with the frames a
  checkpoint carries rather than merge the two.
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

### Distillation

- **Teacher scoring and offline labeling** — new `nvalchemi.training.distillation`
  package. A `TeacherScorer` protocol defines the teacher-signal interface
  (`energy`, `forces`, `stress`, `node_energies`, `embeddings`, each mapped to a
  batch field and level), `InProcessTeacherScorer` implements it for a teacher
  loaded in the current process (narrowing `active_outputs` to the requested
  signals, building and rolling back the teacher's neighbor list — including a
  list a composed pipeline keeps as an instance attribute, and hiding for the
  duration of scoring the per-source table such a pipeline captures alongside
  it, while refusing a teacher composition that plans more than one neighbor
  list — and detaching every output), and `label_dataset` walks a dataset once
  to persist the original fields plus the teacher fields to a resumable Zarr
  store, rejecting a chunk whose schema drifts from the store's and a store an
  interrupted run left inconsistent instead of resuming from a misaligned
  offset. The supported signal set (`SUPPORTED_SIGNALS`) and the fields each
  signal populates (`signal_fields`, `signal_for_field`) are public, as is a
  scorer's own `label_fields` declaration, which consumers resolve through
  `scorer_fields`. `label_dataset` drops the source neighbor list by default,
  since a stored list records no cutoff a consumer could check
  (`keep_neighbors=True` keeps a sparse one), and `cast_to` accepts any
  floating-point dtype while `label_dataset` refuses a dtype the store cannot
  hold before writing — though a labeled store reads back at the reading
  dataset's `positions` dtype, whatever it was written at. The teacher is held
  in evaluation mode for the whole of every `label` call, not only at
  construction, and `scorer_fields` refuses a `label_fields` declared as a bare
  string rather than resolving it to its characters.
- **Offline distillation strategy** — `DistillationStrategy` trains a student
  against a `"teacher"` frozen by omission from `optimizer_configs`. Teacher
  signals reach the loss as `teacher_*` batch fields, so any built-in term
  distills by pointing its `target_key` at one; the requested signal set is
  derived from those targets — a `validation_config` loss's included — and
  validated against the teacher's outputs at construction, as are both losses'
  prediction keys against the outputs the student actually computes (its
  `active_outputs`, not just its declared ones), while the serialized spec
  records its own strategy class, which `DistillationStrategy.from_spec_dict`
  builds — dispatching to the named subclass with every runtime override —
  and refuses to rebuild from if it names a foreign strategy.
  Labeled stores from `label_dataset` train with no teacher forward pass, while
  unlabeled training *and* validation batches are labeled on the fly by an
  internal `BEFORE_FORWARD` hook that scores with autocast disabled, so
  mixed-precision training leaves the teacher targets untouched. New
  `PerAtomEnergyMatchingLoss` matches the teacher's per-atom energy
  decomposition, a signal no reference dataset carries. See the new
  `examples/intermediate/09_offline_distillation.py`.
- **On-policy generation components** — building blocks for training a student
  on frames it generated itself. `TeacherLabelHook` is an `AFTER_STEP` dynamics
  hook that attaches `teacher_*` fields to the live frame at the level each
  signal declares, leaves the `energy` and `forces` driving the propagator
  alone on the live batch, and optionally mirrors a copy of each labeled frame
  into a `DataSink` — stripped of neighbor tensors, dynamics bookkeeping, and
  the model's own predictions, so a stored frame is a training sample rather
  than a propagator state and never carries a self-label under a reference
  target's name. `ReplayBuffer` accumulates those frames behind a frozen key
  schema, so an unlabeled frame is rejected instead of silently stripping
  `teacher_*` from everything already stored, with FIFO eviction at capacity
  and an optional staging device. `build_mixed_loader` draws each training
  batch with an exact reference/replay composition, resolved to whole samples
  of the batch size, sizes every path to the requested batch count, and is
  rebuilt per segment; it requires the two sources to carry one batch schema
  and to emit on one device, comparing a probe batch from each side rather than
  the field names a Zarr store and the in-memory buffer never report alike.
  `OnPolicyConfig` collects the segment knobs; its propagator is any
  `BaseDynamics`, so relaxation optimizers generate on-policy paths exactly as
  integrators generate trajectories.
- **On-policy segment loop** — `DistillationStrategy` now accepts `on_policy`
  and `reference_dataset`, and `run()` drives the loop itself when they are
  set: seed a state batch from `seeds`, generate `segment_steps` frames with
  the student's own propagator, label and capture them, then take
  `steps_per_segment` optimizer steps on a freshly mixed reference/replay batch
  stream, until `num_steps` is reached. Each segment advances its sampler's
  epoch, so the mixture keeps drawing fresh reference samples rather than
  replaying one seeded draw. One
  segment is one epoch, so epoch hooks and validation checkpoints keep the
  offline loop's semantics. The student generates in evaluation mode and enters
  training mode for the training phase only. The propagator is checked at
  construction for holding the student it trains — directly or composed into a
  larger model — which is what makes each segment on-policy, as are the
  ratio/batch-size allocation and the teacher fields the two mixture sources
  carry. `on_policy` and `reference_dataset` hold live runtime objects and are
  omitted from `to_spec_dict`, which warns; recipe serialization follows later.
- **On-policy mixing, model modes, and device staging** — the mixture guard now
  compares the two sources' full batch schemas instead of their `teacher_*`
  fields alone, so a periodic anchor mixed with cluster replay frames no longer
  trains on silently de-periodized structures and an anchor carrying its own
  `energy` or `forces` is rejected instead of zero-filling those targets for
  every replay row; the anchor must therefore be teacher-labeled in the
  replay-frame shape. Generated frames are staged on the reference dataset's
  device unless `replay_device` says otherwise, which makes a CUDA run over a
  Zarr anchor work by default, and an explicit `replay_device` that disagrees
  with the anchor is rejected at construction rather than mid-run. A propagator
  model that only *composes* the student is held in evaluation mode for the
  whole loop, so generation stops building second-order graphs and moving the
  batch-norm statistics of the submodules the student does not own. Cross-GPU
  mixtures are compared by device index, the batch size a rejected
  ratio/batch-size pair suggests is now one the allocator accepts, and the
  "scored twice" warning fires whenever the propagator's scorer is narrower
  than the loss, anchor or no anchor.
- **On-policy launch, seeding, and mixture reproducibility** — the segment loop
  refuses to start in a distributed world of more than one rank, because
  nothing shards its loader or its seed state and every rank would otherwise
  regenerate, relabel, and retrain the same frames at N times the teacher cost;
  offline distillation still distributes through `DDPHook` as before. A seed
  batch is restamped with fresh dynamics bookkeeping, so seeds loaded from a
  store a previous relaxation graduated no longer arrive frozen at
  `exit_status` and silently generate nothing. New `OnPolicyConfig.seed` keys
  every segment's mixture sampler, so replicate runs can draw independently
  instead of all sharing the sampler's default. An anchor carrying fields the
  labeling hook strips from every generated frame — the shape `label_dataset`
  leaves an existing reference set in — is now rejected at construction rather
  than after a full generation segment, as is `replay_ratio=1` paired with an
  anchor the mixture would never sample. A propagator composition is restored
  submodule by submodule, so a correction head the caller had frozen alone
  comes back in evaluation mode. `TeacherLabelHook` labels with autocast
  disabled, matching the offline path bit for bit, and stores each step's frame
  once even for a scorer whose signals it cannot map to fields.
- **Generation-supplied teacher targets** — an on-policy loss may now read a
  `teacher_*` target that names no built-in signal, provided the propagator's
  scorer declares it in `label_fields`: generation writes it onto every
  captured frame, so `reference_dataset` and any validation data have to carry
  it too, and at least one built-in teacher target is still required alongside
  it. Offline distillation is unchanged, and a scorer declaring no
  `label_fields` still supplies nothing. `TeacherLabelHook` now resolves its
  idempotency fields through `scorer_fields` and remembers what the first pass
  wrote, so an undeclared custom scorer is no longer re-scored on every
  re-dispatch of the same step — a segment's forced last-frame labeling cost a
  second full teacher pass. Unknown generation fields are treated as unknown
  rather than as none: the strategy warns that the anchor parity and the
  double-pass check are deferred instead of falsely rejecting a custom scorer
  against the anchor, that parity is compared as fields rather than signal
  names, and a `label_fields` entry outside the `teacher_*` namespace is
  refused at hook and strategy construction rather than mid-run.
- **On-policy restart, rerun, and validation bookkeeping** — a run resuming
  with a nonzero `epoch_step_count` (a checkpoint taken mid-segment, or an
  offline run graduating from a partial epoch) now closes that segment on the
  way in, so `BEFORE_EPOCH` fires for the resumed segment, `epoch_step_count`
  stays inside `steps_per_segment`, and the mixture sampler advances instead of
  redrawing the reference samples the interrupted segment already trained on.
  The segment is the restart granularity, and it is documented as such. A
  second `run()` on one strategy — continuing a finished run with a raised
  `num_steps` — now keeps the replay buffer it filled rather than silently
  discarding it and regenerating from scratch. The loop's closing validation is
  skipped when a cadence already validated at the final step, so metric-driven
  LR schedulers are no longer stepped twice on one set of metrics.
- **On-policy labeling cadence and mixture dtype/device parity** — a segment's
  forced last-frame label and the hook's own cadence no longer label adjacent
  frames: with `segment_steps` a multiple of `label_frequency` (the defaults
  are 100 and 100) every segment used to pay two teacher passes one propagator
  step apart and fill the buffer with near-duplicate pairs, and a cadence
  dispatch landing on the step right after a labeled one is now passed over.
  A forced label is never passed over, so an early-exiting segment and a run's
  final frame are unaffected. `build_mixed_loader` now compares the two mixture
  sources per-field *dtypes* as well as their names, because collation casts
  one part to the other's dtype and which source leads a chunk is not fixed —
  a float64 anchor beside float32 generated frames silently changed the
  targets' precision from chunk to chunk. And the device a source emits on is
  measured from a batch when no declaration settles it, so a `MultiDataset`
  anchor (which declares none) and a Zarr store opened without a device (which
  declares an index-less `cuda`) are staged and validated against the device
  they actually collate on instead of dying inside the first segment's loader.
  `OnPolicyConfig` documents that mixture seeds must be spaced by at least the
  segment count, since the sampler adds `seed` to the segment index, and that
  `replay_capacity` should be a multiple of the trajectory count so FIFO
  eviction does not favor the trajectories at the back of the batch. It also
  names the seed contract correctly: a seed carries the batch fields its
  propagator declares in `__needs_keys__` and `__provides_keys__`, which is
  `forces` for every shipped integrator and optimizer, `stress` for the
  variable-cell ones, and the `velocities`, `atomic_masses`, and `cell` they
  update in place.
- **Relaxation on-policy generation** — `OnPolicyConfig` gains `convergence`
  and `convergence_hook`, which give a relaxation propagator such as `FIRE` the
  trajectory lifecycle its paths need: converged structures freeze, are stored
  once as the minimum they reached, and graduate out of the batch through
  `BaseDynamics.refill_check` at the segment boundary, so the replay buffer
  keeps filling with informative frames instead of near-duplicates of a
  structure that stopped moving. `convergence` is the `fmax` threshold a
  recipe can hold and `convergence_hook` the live criterion no recipe
  describes; `OnPolicyConfig.convergence_criterion` resolves the two into the
  one status-migrating, every-step hook the lifecycle drives, which is also the
  propagator's own convergence detector for the duration of the run. The
  lifecycle refuses to run beside a second status migrator, off a status the
  seeds never carry, or under a multi-sub-stage `FusedStage`, each of which
  would graduate structures at the wrong threshold or not at all. The backfill
  is served by `OnPolicyConfig.seeds` under the seeded batch's own size
  envelope; because an unbudgeted source seeds every row it owns, a graduation
  narrows the batch unless `SeedSource(..., recycle=True)` restarts it at the
  front of those rows. A run whose
  last trajectory finishes warns once and trains its remaining steps on the
  frames it already has. Frames are captured by two routes that partition
  them: the labeling hook stores the structures still relaxing, narrowing to
  them before the teacher runs rather than after, so a mostly-converged batch
  no longer spends most of its teacher budget on frozen structures; and a
  converged-frame hook stores each minimum once, reading the status transition
  every propagator publishes rather than the `ON_CONVERGE` stage a `FusedStage`
  fires only on its sub-stages, then labeled in one teacher pass as its sink is
  drained onto the buffer's own device. Seed structures are checked against the
  fields the propagator opens its step with, named from its own
  `__needs_keys__` and `__provides_keys__`.
- **Relaxation lifecycle ownership and backfill bookkeeping** — the segment
  loop now stamps its own bookkeeping over the rows a backfill appended, so a
  seed source that stored `status` alongside its structures — an
  `InMemoryDataset` of minima a `ConvergedSnapshotHook` captured, say — no
  longer backfills frozen structures that are propagated by nothing, stored raw
  as minima they never reached, and graduated again at the next boundary; the
  `system_id` the sampler handed out is the one field kept. The competing
  migrator check reads a `FusedStage` sub-stage by sub-stage, which is where
  the stage puts the migrators it builds itself, so a fused propagator that
  would graduate the batch at its own threshold before the configured criterion
  ever saw it is refused rather than run silently. A propagator carrying a
  `sampler` of its own is refused too, because it would refill mid-segment and
  compact the batch under the capture hook's positional bookkeeping; give
  `OnPolicyConfig.seeds` the same budget instead. And a fused sub-stage that
  graduates on an
  `n_steps` budget rather than on a criterion migrates after the step's hook
  dispatch, so the segment loop captures those frames once the chunk returns —
  previously the whole batch's last frame was lost whenever the budget ended
  the chunk and the labeling cadence had skipped that step. The backfill is
  restricted to the rows one rank owns, so a run that divides its seeds across
  ranks never draws a row another rank is already relaxing: what the cursor has
  consumed, where it wraps, how far one pass reaches, and when it reports
  itself exhausted all count shard positions. That cursor and the `system_id`
  it stamps are tracked separately, because an id numbers a trajectory rather
  than a row — under `SeedSource.recycle` ids climb past the shard's length
  while the cursor wraps back through it, so a restart deriving one from the
  other rewound to the first structure instead of resuming where it stopped.
- **Multi-GPU and multi-node distillation** — the on-policy segment loop now
  runs data-parallel instead of refusing a multi-rank launch. Each rank
  propagates the strided shard of `seeds` it is dealt — every `world_size`-th
  structure from its own offset — labels those frames with its own teacher
  replica, and fills its own replay buffer and mixed loader, so no generated
  frame or teacher pass is duplicated; the anchor stays replicated and each
  rank draws from all of it. Both seeded streams the loop owns — the mixture
  sampler's `OnPolicyConfig.seed` and every integer seed the propagator
  exposes, a composition's sub-stages included — are moved onto a per-rank
  stride so ranks decorrelate, stage by stage rather than tree-wide: a stage
  holding a `torch.Generator` and no integer seed to offset is named in a
  warning even when the stages beside it were moved, from every rank including
  rank zero and before the first segment is generated, and a seed readable only
  through a getter-only property is moved under its writable name instead of
  raising where the offsets are applied. The anchor has to be left in host
  memory or moved onto each rank's own device, because every rank stages its
  replay frames on the anchor's device and one pre-staged on an accelerator
  concentrates the whole world's buffers on a single GPU; where that device is
  indexed the ranks reduce the question between them and every one of them
  reports it, since the rank owning the device the world piles onto cannot tell
  a shared anchor from a per-rank one by its own placement. A seed set the
  world cannot deal out in equal shares warns as well:
  every rank draws the same number of replay samples per batch from a buffer
  holding only its own trajectories and the gradients are averaged rank by rank,
  so a frame from a shard one structure shorter reaches the optimizer with more
  weight. The only cross-rank traffic is the student's gradient all-reduce
  through a `DDPHook`, which leaves the frozen teacher replicated and out of the
  collective; a multi-rank run with an unwrapped student, or with fewer seed
  structures than there are ranks, is refused up front. Multi-node is the same
  code path: sharding keys on the global rank while device placement keys on
  the node-local one. `TrainingStrategy` also narrows its named-model device
  check from "more than one device" to "more than one *distinct* device", so a
  per-model list that names one device repeatedly is accepted — it places every
  model exactly where a single-entry list would — while cross-device named-model
  placement stays rejected. The rows a rank owns are public as
  `DistillationStrategy.seed_shard`, and they bound anything that refills or
  backfills the trajectory batch: a refill cursor counts consumed positions,
  wrapping, and exhaustion against the shard rather than against the dataset,
  since a structure served to a rank that does not own it is propagated and
  billed to the teacher twice. The anchor's staging device is measured rather
  than memoized from validation — once per `run()`, where the buffer's staging
  device is resolved, and once per segment inside `build_mixed_loader` — because
  a launcher pins the process only after the datasets are built and the anchor
  may be moved onto the rank's own device after setup. And the idiom that
  reaches past a data-parallel wrapper to
  the module it owns is public as `nvalchemi.training.runtime.unwrap_model`,
  which reads that module off whatever publishes `.module` rather than off one
  wrapper class.
- **Ensemble checkpoints rebuild with their segment loop re-supplied** —
  `DistillationStrategy.from_spec_dict`, `from_checkpoint_dict`, and
  `load_checkpoint` take `on_policy` and `reference_dataset` (and
  `load_checkpoint` takes `models`, since the propagator holds the live
  student), so a checkpoint of a run whose loss carries a
  `BoltzmannMatchingLoss` — which refuses to rebuild offline-shaped — comes
  back with the loop and the very models its propagator was built around. The
  refusal now names that way back instead of asking a checkpoint holder to
  configure a loop or drop the term. A spec naming a `DistillationStrategy`
  subclass dispatches to it carrying both runtime objects too, so the subclass
  runs the loop the caller handed over rather than one rebuilt from the recipe.
- **Companion fields are not loss targets** — a loss reading
  `teacher_hvp_probe`, the direction `teacher_hvp` was taken along, is refused
  at construction rather than resolving to the `hessian` signal; adopting the
  public signal surface had let it through.
- **Validation-side objectives are checked at construction** — an
  `EmbeddingMatchingLoss` or `HessianMatchingLoss` carried only by
  `validation_config` goes through the same width and energy checks as a
  training-side term, naming the side in the message.
- **The curvature term reuses the student's neighbor list** —
  `hessian_distillation_fn`'s energy-only pass runs on the list the stock
  forward just consumed instead of rebuilding and tearing down its own on every
  step. A direct-force student — one whose forces are a head output rather than
  an energy gradient — is warned that the term supervises its energy head alone,
  while the force head its force loss trains receives no curvature signal.
- **Weighting and `beta` guidance for the advanced objectives** —
  `HessianMatchingLoss` documents that its standard-normal probe makes the
  graph-balanced value a Hutchinson estimate of `||dH||_F^2 / 3V` in
  (eV/A^2)^2, one to two orders above a force mean-squared error for a
  near-converged student and a one-sample estimate whose relative spread is of
  order one, so it wants a weight a hundred to ten thousand times lighter than
  the force term as a starting point. `BoltzmannMatchingLoss` documents that
  reducing energies by `k_B T` puts its gradient at up to `1/k_B T` per
  configuration (about 39 eV^-1 at 300 K), and that the self-normalized forward
  direction is bounded by `log B` with a gradient that vanishes once the softmax
  saturates — a student whose error spreads over more than roughly four `k_B T`
  — so `beta=0` can read as converged while the student is far off; hold `beta`
  at `0.5` or `1.0` until the student is within a couple of `k_B T`.
- **Evaluation and acceptance suite** — new
  `nvalchemi.training.distillation.evaluation` subpackage deciding whether a
  distilled student ships. `evaluate_accuracy` measures energy, force, and
  stress MAE/RMSE over a holdout against either the dataset's own labels or the
  teacher's (on disk or scored on the fly), running the pass through
  `ValidationLoop` for eval-mode, autograd, and device behavior while
  accumulating exact global residual sums rather than reading a graph-balanced
  training loss. The student predicts in its own dtype — no autocast is applied
  — and a scorer's labels are cast to the dtype the store would hold them at,
  so a float64 teacher scores a float32 student and a reduced-precision student
  is measured rather than refused. Against a teacher it also reports force
  cosine similarity, per-atom (over the atoms whose force does not vanish on
  either side, where the angle is undefined) and aggregate; the per-atom mean
  is dominated by atoms whose force sits at or below the student's own error,
  so `min_force_cosine` is read off the magnitude-weighted aggregate. Per-atom
  energy residuals fill in as well.
  `nonconservative_residual` quantifies what no conservative student can fit by
  integrating the teacher's work around closed loops in configuration space —
  zero for a conservative field by construction — and converting the leftover
  into a lower bound on the root-mean-square per-atom force error, probed at a
  per-atom displacement of the caller's chosen amplitude.
  A scorer paired with reference targets is rejected rather than paid for and
  thrown away, since nothing would then be compared against the teacher it
  labels with.
  `StabilityMonitor` is a dynamics hook reporting energy drift (per atom, per
  step, and as a fitted per-nanosecond rate) and momentum conservation over a
  student-driven trajectory, discarding a `warmup_steps` equilibration window
  so the relaxation of a seeded frame is not fitted as drift; `extensivity_error`
  checks energy scaling across replicated cells, and `radial_distribution` with
  `compare_radial_distributions` scores structural match against a reference
  trajectory with a bounded Jensen-Shannon divergence, pooled over every species
  or resolved to one species pair for a chemically ordered system; a frame
  whose cell encloses no volume is rejected rather than normalized by an
  infinite ideal-gas density, which scored any two molecular trajectories as a
  perfect match. `measure_throughput`
  reports atoms/s and ns/day from a warmup-discarded, device-synchronized
  window; the rate scales with the batch it was measured on, so
  `build_acceptance_report` rejects a family whose students were timed on
  different ones. `build_acceptance_report` turns those measurements into per-student
  verdicts against configurable thresholds, a speed-versus-accuracy Pareto
  table, and the from-scratch-baseline gate, rendering as Rich tables and
  exporting as plain dictionaries or flat scalars; a bar with no measurement
  behind it fails rather than being skipped, and every measurement rebuilds
  from its own export with `from_dict`, so a sweep can evaluate each student in
  its own job and assemble one report at the end. Each student evaluation also
  optionally records which of the student's weights the numbers came off —
  `weights="ema"` or `"raw"` — so two exports of the same student say which
  artifact each one gated on; only the caller that swapped averaged weights in,
  by handing `evaluate_accuracy` a `strategy.inference_model` entry, knows, and
  the marker rides the export without becoming a bar or moving a verdict.
  Speculative-MD drafter rows
  are wired as an optional input and omitted until the drafter metric lands;
  their bar is checked against the drafters of a mixed family and skipped for
  the plain students it was never aimed at, and rejected outright on a family
  with no drafter in it.
- **On-policy batches reach the host with a blocking copy** — the segment
  loop placed its seed state and every training batch with
  `Batch.to(device, non_blocking=True)` whatever the direction. Into device
  memory that is the point; into *host* memory it is a race, because ATen
  issues the transfer and returns without synchronizing while the loop reads
  the moved batch's `segment_lengths` and `batch_ptr` on the host right
  after. A CUDA-resident mixture source paired with `devices=[cpu]` could
  therefore train on half-written index tensors, surfacing as `repeats can
  not be negative`, an out-of-range `index_select`, or a hang. Both
  placements now overlap the copy only into device memory.
- **On-policy configuration split, and a seed source with a cursor** —
  `OnPolicyConfig` now inherits its scalar half from a public `OnPolicyKnobs`,
  a JSON-native model with no arbitrary types, so a recipe's knobs validate
  standalone — before a teacher is built — against the very constraints the
  config enforces rather than against a second copy of them. The bounds checks
  the strategy used to run (`replay_ratio=0`, and the ratio-versus-batch-size
  allocation) moved onto the knobs with their messages unchanged. Seed
  structures now live behind a public `SeedSource`: one cursor over the rows a
  rank owns, shared by the initial batch, the refill backfill, and a restart,
  with a strided `shard`, an optional size budget, `recycle`, a
  `state_dict`/`load_state_dict` pair, and a `to_spec_dict` round trip. It
  answers the five members `BaseDynamics.refill_check` reads, so a run that
  graduates converged trajectories backfills from its own shard only. Seed
  structures are checked against what the propagator reads before its first
  force evaluation at construction, so a missing `forces` is a config error
  rather than `'Batch' object has no attribute 'forces'` mid-kernel. The
  pre-`SeedSource` spellings — `seed_dataset`, `sampler`, `recycle_seeds`, and
  a hook-valued `convergence` — are accepted with a `DeprecationWarning` and
  mapped onto the new shape; a run converted from a `sampler` packs its initial
  batch first-fit in row order rather than largest-bin-first, while the budget
  it respects and the source it refills from are unchanged. A `state_dict` also
  carries the envelope an unbudgeted source measured off the rows it seeded, so
  a run restored after a graduation refills under the width it started at: the
  batch a restart resumes has already narrowed away every trajectory it
  graduated, and re-deriving the envelope from that batch ratcheted the
  composition of the generated data down a little further at every restart.
  `record_envelope` is the fallback for a bundle written before the figure was
  checkpointed and no longer overrides one, and a budgeted source writes none,
  so a stale bundle cannot talk a run out of the budget its recipe declares.
  `SeedSource.from_spec_dict` validates the block it is handed instead of
  reading keys off it: a budget that is not a positive count, a `recycle` flag
  nothing reads as a boolean, and a misspelled budget are all refused where the
  recipe is read — the misspelling most of all, since a source is unbudgeted by
  default and a knob that reached no field used to run a whole job silently
  unbudgeted. A `seeds.dataset` block naming no `path` is refused the same way
  rather than raising a bare `KeyError` from inside the rebuild, and so is the
  reference dataset's, which is reopened through the same helper.
- **An index-less `replay_device` names this rank's own device** — set to
  `"cuda"`, `OnPolicyConfig.replay_device` is now resolved to the device the
  process has made current, which under a launcher is the one it pinned this
  rank to. The spelling would otherwise survive into the staged frames: a batch
  moved by `.to("cuda")` records it rather than the device its tensors landed
  on, and an index into those frames is resolved against the record, so every
  rank but the first crashed indexing its own replay buffer and then hung its
  peers. A device the anchor emits on is concrete already and is still left as
  measured.
- **Multi-rank restarts name this rank's device in two places** — rank zero
  writes `strategy.json` after `DDPHook` collapsed `devices` to the GPU it
  pinned, and that recorded device is the load location every rank restores
  against, before its own hook pins anything; `run()` then moves the parameters
  but reuses the resumed optimizer, and `Optimizer.load_state_dict` re-homes the
  moments to the parameter without ever moving Adam's `step`, so one state
  tensor is stranded on the device the checkpoint was written from. The runbook
  now prescribes the shape that works today — an indexed `devices=[...]` on the
  restarting strategy *and* a matching `map_location=` on `restore_checkpoint`,
  either alone being insufficient — and names the symptom, a hang rather than a
  traceback, once. Two in-process `multigpu` tests cover it: the recipe ends
  with every optimizer state tensor on the rank's own device after `run()`, and
  the shape that strands one is a strict `xfail` naming the root, which is
  core's. Single-rank restarts are unaffected.
- **The scale-out runbook, trued against a two-rank launch** — the
  anchor-placement paragraph claimed pre-staging on an accelerator never
  partitions per rank, and that moving the anchor after setup is the only
  accelerator-resident shape that places it correctly. Measured on two ranks, a
  `Dataset` opened over a labeled store with no `device` — or with an index-less
  `"cuda"` — emits lazily, fixes its device on the first draw after `DDPHook`
  has pinned the rank, and lands each rank's anchor batches and replay buffer on
  its own GPU unremarked; an eager `.to("cuda:0")` concentrates the world on GPU
  0 and every rank reports it; an eager `.to("cuda")` cannot be drawn from at
  all once the pin has moved the current device, which is the parent-toolkit
  index-less-recording defect tracked separately. The after-setup move is now
  one option among those, named with its hook stage and target rather than as
  the only one. The paragraph also says what a desynchronized world looks like —
  peers blocked in the next all-reduce for the process group's default timeout,
  and a raising rank blocking the teardown — and that the way to bound the wait
  is to initialize the process group yourself with `timeout=`, since `DDPHook`
  exposes none and leaves an established group alone. `TrainingStrategy`'s
  device-check note narrows its `DDPHook` claim to the NCCL backend, the only
  one that pins per rank.
- **Multi-rank test rigor** — the two-rank gloo run now asserts that the step
  the world takes is the step one process takes over the union of the shards,
  which is what says the all-reduce averaged once over both gradients rather
  than merely agreeing across ranks; an unequal-shard leg covers a seed set the
  world cannot halve, and asserts the warning, the lockstep, and the aggregate
  frame count it still owes; the same unequal deal pins that each rank
  checkpoints the envelope of its own shard rather than of the seed set, and
  that a rank refuses the cursor its peer wrote; and the spawn helper polls its
  children instead of blocking on the result queue, so a rank that dies without
  reporting — taking its peers into a collective that will never complete —
  fails the run in seconds rather than at the timeout.
- **The rank shard is the seed source's own** — `SeedSource.shard` installs the
  strided deal the segment loop used to make by hand, so the cursor a backfill
  and a restart share counts positions in this rank's rows rather than rows of
  the dataset, and
  `DistillationStrategy.seed_shard` reports what the source is narrowed to once
  a run has installed it. The refusal of a `sampler` above one rank is gone
  with it: a budgeted source packs its initial batch from the shard it was
  dealt and leaves the remainder of that shard to the backfill, which is the
  rank view the sampler never had. Two spawned gloo ranks prove the backfill
  disjoint — each serves only the rows it owns, and together they serve the
  dataset once.
- **Ensemble objectives refuse a segment loop configured to converge** — a
  `BoltzmannMatchingLoss` already refused a propagator carrying a
  `ConvergenceHook`, but a criterion set as `OnPolicyConfig.convergence` or
  `convergence_hook` reaches the propagator only once the loop is running, so
  it slipped past that probe and the term matched against a batch its own run
  was graduating graphs out of. It is now refused at construction, beside the
  propagator check.
- **Ensemble objectives refuse a registered convergence hook** — the propagator
  probe of a `BoltzmannMatchingLoss` read `dynamics.convergence_hook` only, so
  the same criterion attached with `dynamics.register_hook(...)` reached the run
  unrefused and froze every graph it converged at its exit status. The hooks
  registered on each propagator in the composition are now scanned too, and one
  migrating graphs to the root's exit status is refused like the attribute; a
  hook handing graphs to another sub-stage, which a `FusedStage` installs
  between its own, keeps them sampling and is still accepted.
- **Acceptance bars declare the measurements they read** — `BAR_FAMILIES` maps
  every `AcceptanceThresholds` field to the `StudentEvaluation` slots its check
  reads, and is the table `build_acceptance_report` now applies the bars from,
  so a bar cannot be added to the model without one. `measured_bars(*families,
  accuracy_quantities=...)` answers which bars a partial measurement can decide
  — every family a bar reads has to be supplied, so the from-scratch gate needs
  both the distilled and the baseline accuracy, and `min_drafter_acceptance_rate`
  needs drafter metrics this package never produces — and narrows the accuracy
  bars by the quantities the holdout pass actually compared, since a student
  scored on energy alone leaves a force bar as unfillable as no holdout at all.
  A caller that measures a subset, such as a CLI holdout pass, reads the bars it
  may accept off it rather than restating the mapping. A bar whose family was
  measured but whose own number was not now says which quantity or timestep was
  missing instead of reporting the measurement absent, and a measurement slot
  holding something other than its metrics class — an accessor left uncalled,
  most often — is rejected where it is filled rather than deep inside the
  report.
- **Energy-only evaluation of an autograd-force student** — `evaluate_accuracy`
  resolves `grad_mode="auto"` from the student's own `model_config` as well as
  the loss, so a student that differentiates its forces inside `forward` can be
  scored on energies alone, and `grad_mode="disabled"` is refused for such a
  student up front instead of failing inside its forward.
- **The non-conservative floor is conditioned per graph** —
  `nonconservative_residual` lays each probe loop out around its own graph's
  centroid instead of the batch's, so a float32 batch mixing frames far apart in
  space no longer reports an inflated floor, and `relative_floor` divides each
  probe by its own graph's force scale (with a new `relative_floor_max`) so a
  batch mixing force scales reports a figure between its graphs' own ratios.
- **`StabilityMonitor` names the field a sample lacks** — a batch carrying no
  `energy`, `velocities`, or `atomic_masses` is refused with a message naming
  the field and the seeding fix, instead of dying with a bare `AttributeError`
  on the first recorded firing; the propagator copies energies only into a
  field the batch already carries.
- **`StabilityMetrics` sizes the fluctuation** — `energy_fluctuation_per_atom`
  (the RMS residual about the fitted drift line) and
  `max_energy_excursion_per_atom` are reported as diagnostics that size a
  bounded oscillation the two drift figures disagree about; both default to
  `None` so exports written before them still load.
- **Every evaluation places its batches up front** — `evaluate_accuracy`
  handed device-resident batches to the validation loop's asynchronous host
  copy whenever no teacher scorer was supplied, so a CPU student over a
  `Dataset` left on its default CUDA device read half-written index tensors;
  the batches now land on the run device before the loop sees them on both
  paths.
- **The RDF comparison is continuous in the positions** —
  `radial_distribution` apportions each pair linearly between the two bins
  whose centres bracket its distance and builds the neighbor list one bin past
  `r_max`, so a coordination shell sitting on a bin edge or on the cutoff is no
  longer split by round-off: a rigid translation of a crystal scores a
  Jensen–Shannon divergence at round-off rather than `5e-2`, and a
  lattice-constant sweep rises smoothly instead of holding exactly `0` until a
  shell crosses an edge and then leaping by `0.4`. The `r_max` docstring no
  longer asks for half the shortest cell vector; the build enumerates every
  periodic image the cutoff needs.
- **Non-finite measurements are first-class in the acceptance gate** — a metric
  that came out `nan` or `inf` now fails its bar with a `not finite` detail
  instead of reading as a measurement nobody took: a `nan` failed every
  comparison and an `inf` cleared every `max_*` bar. The from-scratch gate
  refuses a non-finite operand before taking its worst-of, so a `nan` can no
  longer vanish inside `max`, and the Pareto front ranks only finite
  `(error, speed)` pairs, so a diverged student no longer heads a front nothing
  can dominate it on. `AccuracyMetrics.force_cosine_aggregate` reports `nan`
  when its sums are non-finite and keeps `None` only for a holdout whose forces
  all vanish, and a new `force_nonfinite_atoms` count records the atoms the
  per-atom cosine mean had to drop.
- **The from-scratch gate compares like with like** — a baseline metric of
  exactly `0.0` is unbeatable rather than silently dropped from the comparison,
  `0/0` ties at `1.0`, and a baseline scored on a different number of graphs or
  atoms fails that student's own check with both counts named instead of being
  divided into. `build_acceptance_report` rejects a family whose students were
  scored on different holdouts, as it already did for throughput measured on
  different batches.
- **Reproducible recipes — serialization, CLI, docs** — a distillation run now
  survives a round trip. Checkpoints store the frozen teacher *once per
  checkpoint root*: the first write holds its weights, the manifest gains a
  `model_references` entry naming that index plus a fingerprint, later
  checkpoints contribute no teacher weight file, and loading reads the stored
  copy back and verifies the fingerprint, so a replaced copy raises instead of
  quietly training a student against a different model. The fingerprint hashes
  each state-dict entry's name, shape, and dtype together with its values, read
  at `float64` on the host so the device does not change the digest: a tensor of
  at most 4096 values whole, so a per-element table or a bias leaves no gap for
  a change to hide in, and a larger one at 64 values spanning its whole index
  range with the first and last included. It identifies a model rather than
  validating it. One root holds one copy: saving a *different* copy of a
  declared model into a root that already holds one is refused, because moving
  the reference would repoint every checkpoint already written there at weights
  they were not written against, while an identical copy is written again
  freely, which is what repairs a root whose stored weight file went missing.
  What a root already holds is the copy on disk rather than the manifest entry
  naming it, so a root a non-declaring writer left the teacher in is
  fingerprinted from that file once and then continued under the same rule, and
  a `save_checkpoint(models=...)` carrying the teacher is held to it too rather
  than dropping the reference and orphaning the indices that read it. A
  manifest carrying `model_references` stays at `schema_version` 1 and an older
  nvalchemi still reads it, but only the models it holds a weight file for at
  the index asked — the student at any index, the teacher only at the stored
  one — so a load that includes the teacher elsewhere fails with
  `FileNotFoundError`, remedied by upgrading or by asking for the stored index.
  The teacher's `checkpoint_spec()` still rebuilds its architecture but is
  never trusted for its weights, so a teacher loaded from a fine-tune
  checkpoint restores the weights it trained with.
  `OnPolicyConfig.to_spec_dict`/`from_spec_dict` carry the whole segment loop —
  scalar knobs verbatim, the propagator as the constructor
  reference it rebuilds from with the student rebound at build time, the scorer
  as its signals and cast dtype over the strategy's own teacher, and
  path-backed datasets as the stores they read — while a sampler, a
  propagator's live hooks and sinks, and an in-memory dataset stay runtime-only
  and are named rather than approximated; `DistillationStrategy.to_spec_dict`
  now carries `on_policy` and `reference_dataset` on the same terms. A
  propagator whose class no import reaches — one defined inside a function —
  is named and omitted like any other collaborator a recipe cannot describe
  rather than ending the run at its first checkpoint, and a propagator keyword
  argument JSON cannot carry is refused when the propagator is built rather
  than when a checkpoint is written. A spec naming a `DistillationStrategy`
  subclass under `strategy_cls` rebuilds that subclass, with every runtime
  override handed on to it, and a `strategy_cls` or propagator `cls_path` that
  does not import is reported as a recipe error rather than a leaked
  traceback. An
  interrupted on-policy run resumes its trajectory, propagator counter, and
  replay frames through the checkpoint, exactly for the counter-based-RNG
  integrators and at segment granularity, and the labeling cadence resumes with
  them, so a restart neither pays a second teacher pass at the segment boundary
  it stopped on nor stores the frame beside it; the restored frames replace the
  buffer's contents rather than merging into them, since merging would skew the
  mixture's weighting toward stale pre-restart states, double the buffer memory,
  and reach the eviction horizon a restart early. The bundle is rank-local — it
  rides in a strategy checkpoint, which `CheckpointHook` writes on rank zero
  alone — so a world size differing at either end of a restart drops it with a
  warning and the rank reseeds from its own share with a cold replay buffer.
  New `nvalchemi-training distill` group (aliased `nvalchemi-distill`) authors,
  validates, runs, and gates a JSON `DistillationJobSpec`: `init` scaffolds
  offline or on-policy recipes at generic size-only student tiers — writing a
  `CheckpointHook` into `student.hooks` so a scaffolded run leaves something
  for `spec resume` and `evaluate` to read, and requiring `--seed-dataset` in
  on-policy mode, since the anchor `--dataset` names carries no forces for the
  propagator's first step and is rejected as a seed if it carries labels of its
  own — `spec report` renders derived teacher signals, batch composition, the
  training batch size, and acceptance bars with pre-flight validation
  through the runtime's own helpers — an `on_policy` block goes through
  `OnPolicyConfig`'s own field constraints, so a bad knob is refused before a
  teacher reaches a device, and so is everything else a recipe settles on its
  own: a step budget below one, a `dataset.format` no loader builds, a teacher
  or student source the CLI could never load, a `replay_ratio` at either end of
  its range (a recipe always names an anchor for `dataset` to open, so both
  ends contradict it), a `replay_ratio` and `batch_size` leaving one mixture
  source without a whole sample of every batch, and a `replay_device` that is
  not the device the anchor loads on. The composition row is the allocator's
  own split rather than a second rounding of it, the report names a
  `checkpoint_dir` no `CheckpointHook` writes *into that directory* and a
  checkpoint root that already holds a teacher stored once per root, and `init`
  records `dataset.batch_size` (`--batch-size`, default `8`) so a scaffolded
  run trains on batches rather than one graph at a time — `spec run` executes,
  `spec resume` continues an
  interrupted run from its checkpoint directory and its recipe, and `evaluate`
  scores a trained student over the recipe's holdout and exits non-zero on a
  missed bar, writing a non-finite metric to `--json-out` as the string `"nan"`,
  `"inf"`, or `"-inf"` so the export stays parseable by a strict JSON reader.
  A `CheckpointHook` cadence saves nothing at training end, so both `spec run`
  and `spec resume` write a terminal checkpoint at the next index whenever the
  run finished on a step the interval missed: `evaluate` scores the weights the
  run ended with rather than the ones it had several updates earlier, and a
  later `spec resume` has nothing left to repeat.
  `evaluate` scores the weights the recipe trained: with an `EMAHook` in
  `student.hooks` that is the averaged copy the run's own validation reads,
  revived by rebuilding that hook alone over the strategy checkpoint, and the
  line above the report names whether `ema` or `raw` weights were scored. Its
  `--map-location` names the one device the student, the teacher, the holdout,
  and the errors all run on — as it does on `spec resume`, where it names the
  device the continued run takes and not only the one its tensors are read
  onto — while a device the host does not have, and a quantity the teacher
  cannot produce, are reported as CLI errors rather than leaked tracebacks. A
  recipe's `evaluation.thresholds` is narrowed to
  `measured_bars("accuracy", accuracy_quantities=evaluation.quantities)`: a
  stability, throughput, extensivity, RDF, drafter, or from-scratch bar is
  refused when the recipe is parsed, and so is an accuracy bar reading a
  quantity the recipe never compares — `max_stress_mae` without `"stress"`
  among the quantities — because a bar with no measurement behind it fails the
  student rather than being skipped, so such a recipe could never be accepted.
  Both modes honor
  `dataset.paths` as well as `dataset.path`, and `mode` is the single source of
  truth for which loop runs. `spec run` and `spec resume` take
  `--distributed/--no-distributed` (auto when `WORLD_SIZE > 1`) and
  `--ddp-backend` as `train spec run` does, attaching a `DistributedManager`
  and a `DDPHook` and building the datasets on the rank's own device; a
  multi-rank `spec resume` pins the restart to that device too, defaulting
  `--map-location` to it and refusing one that names another rank's, since
  restoring against the device the checkpoint records leaves part of the
  optimizer state there and hangs the world in the process-group teardown. See
  the new `docs/userguide/distillation_recipes.md` and the
  `nvalchemi-distillation` agent skill.
- **Recipes, restarts and the CLI follow the knob split** — a recipe names its
  seed store under `on_policy.seeds` — the store, its budgets, and `recycle`,
  never the cursor — and `OnPolicyConfig.from_spec_dict` rebuilds a
  `SeedSource` from it, so `DistillationStrategy.from_spec_dict` no longer
  takes a `sampler` override. A `convergence_hook` passed whole is omitted from
  the recipe with a warning naming `convergence` as the spelling that travels,
  the way an in-memory seed dataset already was. The on-policy restart bundle
  gains the seed cursor and the knobs it was written under, so a resumed run
  backfills from where the interrupted one stopped instead of re-serving
  structures it had already relaxed, hands its restored trajectory to
  `SeedSource.record_envelope` so an unbudgeted source refills under the
  envelope it actually holds, and reports any knob the resumed loop sets
  differently. A bundle written before the cursor was checkpointed still
  restores, with a warning. The CLI pre-flight asks `OnPolicyKnobs` for the
  mixture arithmetic instead of keeping a second copy of its rounding, and
  `distill init` scaffolds the `seeds` block. It validates that block the way
  `SeedSource` validates it, so a budget that is not a positive count, a
  misspelled budget, and a block naming no store are all refused at `distill
  spec report` rather than at `distill spec run` once a teacher and a student
  have been built on a device — the misspelling most of all, which used to
  reach no field and run a whole job silently unbudgeted — along with the one
  pairing the block cannot refuse on its own, `recycle` set under no
  convergence criterion. The world a bundle was written on is read off the
  shard its seed cursor records rather than inferred from the ratio of the two
  step counters, which is not invariant across a run whose history spans world
  sizes, and a cursor this rank's shard cannot take drops the bundle with the
  same warning instead of raising out of `run()` once the weights, the
  optimizers and the counters have been restored.
- **Distillation user guide and on-policy example** — new
  `docs/userguide/distillation.md` covers the whole feature from the user's
  side: the teacher signals and how the strategy resolves them, the offline
  path over a teacher-labeled Zarr store and the neighbor-list hook a graph
  student needs to read one back, the on-policy segment loop with its mixture,
  cadence and capacity arithmetic, the convergence lifecycle a relaxation
  propagator needs, the representation, curvature and ensemble objectives and
  what each asks of the run, scaling the loop across ranks, evaluating and
  gating the student, and the checkpoint and restart contract. Two topics get
  their own treatment: why an on-policy anchor has to be teacher-labeled and
  how to reshape an existing reference set into one, and distilling a
  non-conservative direct-force teacher into a conservative student. New
  `examples/intermediate/10_onpolicy_distillation.py` runs three
  generate-label-train segments on CPU against a labeled anchor.

### Model Wrappers

- **Pipeline neighbor-list adaptation policy** — `PipelineModelWrapper`
  now accepts `neighbor_adaptation` (`"auto"`, `"always"`, `"never"`) and
  `max_cutoff_ratio` (default `1.5`). The default `"auto"` mode only filters
  a source neighbor list for a smaller cutoff when the source cutoff is at most
  `max_cutoff_ratio` times the target cutoff; larger gaps get separate source
  lists. `"always"` builds one max-cutoff source list, while `"never"` builds
  exact cutoff source groups and skips cutoff filtering.

### Core Data Layer

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

- **UMA CUDA dependency resolution** — add standalone `uma-cu12` and
  `uma-cu13` extras. They select the matching torch build without installing
  PhysicsNeMo's RAPIDS extras, whose numba upper bound conflicts with Fairchem
  2.22.
- **int32 batch pointers in the Warp segment-expansion kernel** —
  `Batch.index_select` raised from `_expand_segments_warp` on CUDA whenever the
  storage held its `batch_ptr` in int32, which is what the storage constructor
  casts an explicit pointer to and therefore what every `clone()` and device
  move produces once the pointer has been materialized — a path plain dynamics
  reach as well, through the compaction `refill_check` performs on a batch
  moved after its pointer was built. The pointer slices the kernel reads are
  now cast to the launch dtype, so a moved or cloned batch selects on the
  accelerator like any other.
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
