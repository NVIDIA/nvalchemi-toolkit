# Changelog

## Unreleased

### Added

- Add support for PEFT fine-tuning within `FineTuningStrategy`, including
  LoRA workflows with `LoRAConfig`, `load_peft_checkpoint_into_model`,
  and base-model fingerprint checks for PEFT checkpoint loading.
- Toolkit-level generative API (`nvalchemi.gen`): an `AtomisticGenerator`
  driver runs a user-supplied generating function (optionally conditioned)
  and returns a `Batch` on the contract path (hooks, device and field
  checks, pipeline and dynamics composition), or passes through any
  other container untouched. Lifecycle hooks, streaming, session-managed
  CUDA stream, RNG, and compile. Sequential composition via
  `gen_a | gen_b` (`GenerationPipeline`), including dynamics engines
  driven through their own `run()` loop, with per-call options addressed
  per stage via `stage_kwargs`.
- Demo generative models (`nvalchemi.models.gen.demo`): `DemoGANModel` and
  `DemoDiffusionModel` placeholders.

- Add `DynamicsStage.ON_ADMISSION` to `BaseDynamics`, enabling hooks to
  initialize per-system state once when a batch is admitted, before force
  priming and outside compiled fused steps.
- `FusedStage(reprime_on_entry=...)` — status codes whose newly entering
  graphs skip one integrator update so the shared compute and target-stage
  `AFTER_COMPUTE` hooks can refresh forces under the new stage's context
  before it advances them.
- **Data** — `AtomicDataZarrReader.schema()` (one `FieldSchema` per stored
  field), `level_sizes()`, `check_integrity()`, `field_array`, `num_samples`,
  and `store`; `Batch.add_key(..., level="system")`, which creates the system
  group a bare batch lacks; the public level API `Batch.drop_level`,
  `pop_level`, `set_level`, and `level_keys`, which detach and re-attach a
  level's storage group validated against the batch's graph count, device,
  and schema; `Batch.index_select(idx, *, drop=())` and
  `Batch.clone(*, drop=())`; the `Batch.without_keys(*keys)` context manager;
  `Batch.to_raw_dicts(*, drop=())`; `nvalchemi.data.resolve_device`;
  `nvalchemi.data.datapipes.distributed_shard(indices, num_replicas=, rank=,
  drop_last=, pad=)`, `dataset_device`, and `same_device`; and
  `nvalchemi.data.transforms.make_supercell` with
  `DEFAULT_EXTENSIVE_SYSTEM_KEYS` and `DEFAULT_INTENSIVE_SYSTEM_KEYS`.
- **Dynamics** — `DynamicsStage.ON_GRADUATE`, dispatched with
  `ctx.graduated_mask` on the step a system's status reaches `exit_status`
  (`StageTimingHook("all")` excludes it); `nvalchemi.dynamics.hooks.nonfinite_graph_mask`;
  `BaseDynamics.active_graph_mask`, `check_initial_batch`,
  `required_input_keys`, `seed_offset`, `bookkeeping_keys`, and the
  `samples_equilibrium` class variable (`False` on the FIRE optimizers);
  `NVTLangevin.random_seed`; `nvalchemi.dynamics.OrderedStructureSampler`
  (with `recycle`, `rank`, `world_size`, `shard`, and `draw(limit=, fits=,
  on_miss=)`), the `StructureSource` and `FitPolicy` protocols, and
  `WithinBudget`; `nvalchemi.dynamics.sinks.ResizableSink`;
  `nvalchemi.dynamics.hooks.StabilityMonitor`, `StabilityMetrics`, and
  `total_momentum`; `nvalchemi.dynamics.measure_throughput` and
  `ThroughputMetrics`; and the `KB_EV` and `kinetic_energy_per_graph`
  re-exports from `nvalchemi.dynamics.hooks`.
- **Models** — `nvalchemi.models.hessian_vector_product`,
  `BaseModelMixin.narrowed_outputs`, and `BaseModelMixin.requires_autograd`.
- **Training** — `nvalchemi.training.evaluating` and
  `eval_configured_models`; `nvalchemi.training.losses.graph_balanced_mean`,
  `per_graph_sum`, and `masked_mean`; `nvalchemi.training.unwrap_model`;
  `nvalchemi.training.ensure_reiterable_validation_data`;
  `nvalchemi.training.ModelReference` and
  `TrainingStrategy.checkpoint_model_references()` (empty by default);
  `CheckpointHook(save_at_end=)`, which also saves at `AFTER_TRAINING` unless
  the newest save already recorded that step; `TrainingStrategy.run_setup_hooks`;
  `TrainingStrategy.load_checkpoint`, `from_checkpoint_dict`, and
  `from_spec_dict` (with `FineTuningStrategy.from_spec_dict` and the
  module-level `load_checkpoint`) forwarding `**runtime_overrides` to the
  strategy class a spec names, and `load_checkpoint(models=)` handing live
  models to the rebuild; `nvalchemi.training.distributed.all_reduce_flags`,
  `all_gather_rows`, and `all_gather_objects`; `DDPHook.wrapped_keys`;
  `ComposedLossFunction.to_spec()`;
  `nvalchemi.training._spec_utils.dataset_spec_dict`,
  `dataset_from_spec_dict`, `DatasetRef`, and `SpecSerializable`;
  `nvalchemi._serialization.json_safe` and `MeasurementRecord`; and the
  public `nvalchemi.training.cli_common` module (`HookSpec`,
  `RuntimeHookSpec`, `SourceSpec`, `MaceSourceOptions`, `DatasetSpec`,
  `OutputSpec`, `ValidationSpec`, `DataJobSpec`, `ModelSource`,
  `DatasetFormat`, `ResumeBudget`, `console`, `apply_resume_budget`,
  `build_checked_hook`, `build_dataloader`, `build_runtime_hooks`,
  `build_supported_source_model`, `build_validation_config`,
  `dataset_device`, `hook_spec_is`, `path_exists`, `primary_strategy_device`,
  `resolve_distributed_enabled`, `restart_map_location`,
  `setup_distributed_manager`, `validate_pretrained_source`,
  `write_or_print`, and the `common_loader_options`,
  `common_prefetch_options`, and `common_validation_options` decorators),
  with `build_dataloader`, `build_runtime_hooks`, `build_validation_config`,
  `primary_strategy_device`, and `dataset_device` re-exported from
  `nvalchemi.training.cli`.

### Changed

- `TrainingStrategy` compares per-model devices through
  `nvalchemi.data.resolve_device` when CUDA is available, so `[cuda, cuda:0]`
  is one device on the rank whose current device is 0; without a CUDA runtime
  entries are compared as written. The named-model check refuses more than
  one *distinct* device rather than more than one entry.
- **Behavior change:** the distillation CLI's `spec report` no longer repeats
  the four refusals the strategy or the segment loop owns: an optimizer
  configured for the teacher or missing for the student, `recycle` under no
  `fmax`, a `replay_device` off the reference dataset's device, and a
  `replay_ratio` of `1` beside a reference dataset. They surface once, at
  `spec run`, with the owner's message.
- **Behavior change:** the `nvalchemi-distill` console-script alias that
  earlier revisions of the distillation series registered is gone;
  `nvalchemi-training distill` is the one entry point.

### Distillation

- **Teacher scoring and offline labeling** — new `nvalchemi.training.distillation`
  package. A `TeacherScorer` protocol defines the teacher-signal interface.
  Each teacher signal is a public `TeacherSignal(name, model_output, field,
  level, normalize=None)` spec. The built-in signals (`energy`, `forces`,
  `stress`, `atomic_energies`, `embeddings`, `hessian`) are published as
  `BUILTIN_SIGNALS`, with their names as `SUPPORTED_SIGNALS`, and are requested
  by name. Any other teacher output is requested by a spec of its own, held to
  the `teacher_*` namespace and the node/system levels at construction.
  `signal_fields`, `signal_for_field`, and `scorer_fields` publish the mapping.
  `InProcessTeacherScorer` implements the protocol for a teacher loaded in the
  current process. It narrows `active_outputs` to the requested signals,
  refuses a spec naming an output the teacher lacks, hides a composed
  pipeline's own lists, and restores every field a composed teacher writes onto
  the batch to wire one stage into the next. It also holds the teacher in
  evaluation mode, optionally casts outputs (`dtype`), and detaches everything
  it returns. Its `autocast` setting decides the scoring pass's precision:
  `False`, the default, disables autocast around the pass whatever region
  surrounds the call, `None` keeps the caller's region, and `True` or a
  floating-point dtype enables it. A composition planning more than one
  neighbor-list source is refused. The explicit `neighbor_list` setting decides
  where the teacher's neighbor list comes from: `"rebuild"` (default) builds
  the teacher's own list and rolls it back, and `"reuse"` consumes the batch's
  list. `"reuse"` refuses by name a missing key or a cutoff stamp other than
  the teacher's, rather than falling back. `label_dataset` walks a dataset once
  and persists the source fields plus the teacher fields to a resumable Zarr
  store. It drops neighbor tensors unless `keep_neighbors=True`, and it holds
  scorers to the `teacher_*` namespace. It refuses a label that does not hold
  one row per atom or per graph, a chunk whose fields, levels, dtypes, or row
  shapes drift from the store's, a store an interrupted run left inconsistent,
  and a store holding more samples than the dataset. Fields at a
  user-registered custom level are stored, checked, and resumed like the
  built-in ones. `prefetch` (`"auto"` by default) reads one chunk ahead through
  the dataset's fused-prefetch surface when the store is slow to read,
  deciding from the timing of the first chunks; a fast local store keeps the
  sequential loop. The stored result is the same either way. Core additions
  used here: `AtomicDataZarrReader.schema`, `level_sizes`, `check_integrity`,
  `field_array`, `num_samples`; `FieldSchema`; `Batch.add_key(level=)`;
  `nvalchemi.training.evaluating`, `eval_configured_models`;
  `BaseModelMixin.narrowed_outputs`.
- **Offline distillation strategy** — `DistillationStrategy` trains a student
  against a `"teacher"` frozen by omission from `optimizer_configs`. Teacher
  signals reach the loss as `teacher_*` batch fields, so any built-in term
  distills by pointing its `target_key` at one. The signal set is derived from
  those targets, a `validation_config` loss's included, and checked against the
  teacher's outputs at construction. Both losses' prediction keys are checked
  there too, against the outputs the student actually computes. Stores from
  `label_dataset` train with no teacher pass, and a custom `teacher_*` field
  such a store carries is an ordinary loss target. Unlabeled training and
  validation batches are labeled on the fly by an internal `BEFORE_FORWARD`
  hook that opens no autocast region of its own, so the scorer's `autocast`
  decides the label precision and an on-the-fly label equals the offline one.
  The first time the hook labels a batch, it warns once and names the teacher
  fields the batch lacked, so a run does not pay an unplanned teacher pass per
  step unnoticed. `label_dtype` is the dtype those on-the-fly labels are cast
  to. `None`, the default, infers it from the student's first floating
  parameter, floored at `float32`, and a non-floating dtype is refused. The
  serialized spec names its own strategy class, which `from_spec_dict`
  dispatches to. A spec excludes `validation_config`, because it carries a
  live loader. `from_spec_dict`, `from_checkpoint_dict`, `load_checkpoint`,
  and the recipe CLI's `spec run` and `spec resume` therefore take it as a
  runtime override, and they hold a rebuilt strategy to the construction-time
  checks a directly built one runs. New `AtomicEnergyMatchingLoss` matches the
  teacher's per-atom energy decomposition, a signal no reference-labeled
  dataset carries. See the new
  `examples/intermediate/09_offline_distillation.py`. Core additions used
  here: `nvalchemi.training.losses.graph_balanced_mean`.
- **On-policy generation components** — `TeacherLabelHook` is an `AFTER_STEP`
  dynamics hook that attaches `teacher_*` fields to the live frame, at the
  level each signal declares, and opens no autocast region of its own. It
  leaves the `energy` and `forces` driving the propagator alone. It optionally
  copies each labeled frame into a `DataSink`, stripped of neighbor tensors,
  the dynamics bookkeeping that `BaseDynamics.bookkeeping_keys()` reports
  across the stage tree, and the propagated model's own predictions, so a
  stored frame is a training sample rather than a propagator state. Labeling
  is idempotent per step, and a cadence dispatch landing right after a forced
  label is passed over. `ReplayBuffer` accumulates those frames behind a
  frozen key schema, with an optional staging device and two policy extension
  points. An `AdmissionPolicy` masks the frames each `extend` admits, before
  the schema check. An `EvictionPolicy` (`select(buffer, incoming,
  capacity)`, with `FIFO` shipped as the reference and as the meaning of
  `"fifo"`) names the frames a full buffer drops.
  `OnPolicyConfig.replay_admission` and a policy instance on `replay_eviction`
  wire them into the loop's buffer as runtime-only fields, while
  `OnPolicySettings.replay_eviction` keeps the string form for recipes.
  `build_mixed_loader` draws each training batch with an exact
  reference/replay composition, and it requires both sources to carry one
  batch schema, at one dtype per field, on one device. `OnPolicyConfig`
  collects the segment loop's live objects over the JSON-native
  `OnPolicySettings`. Its propagator is any `BaseDynamics`. Its initial
  structures come from any `StructureSource`, the core `nvalchemi.dynamics`
  protocol, importable from the distillation package under its historical
  name, `InitialStructuresSource`. A bare `BatchDatasetProtocol` dataset is
  wrapped, and `InitialStructures` is the reference implementation: a thin
  subclass of the core `OrderedStructureSampler`, which serves a dataset's
  rows in order from a single position, `next_row`, shards them strided per
  rank (publishing `rank` and `world_size`), restarts from a `state_dict` of
  `next_row`, `wraps`, `next_system_id`, `rank`, and `world_size`, and serves
  structures through `draw(limit=..., fits=FitPolicy, on_miss="stop" |
  "skip")`. The subclass adds only the `to_spec_dict`/`from_spec_dict` round
  trip. At construction, one row is checked through
  `BaseDynamics.check_initial_batch` against the `required_input_keys()` the
  propagator updates in place from its first step. The row is then propagated
  through one `compute()`, so a `__needs_keys__` output the student never
  produces, or a field the propagator reads that nothing declared, is refused
  before a run is paid for. A graph model is probed with the neighbor list its
  `neighbor_config` declares. `OnPolicySettings.probe` (default `True`) gates
  that construction-time forward. A propagator the probe skips, because its
  model plans more than one neighbor-list source, is named in a warning that
  `probe=False` silences. `OnPolicyConfig.capture_sink` chooses the capture
  sink: the `DataSink` that stages each segment's labeled frames before the
  boundary drains them into the replay buffer. It is host memory by default,
  or a `GPUBuffer` to stay on the generation device. The loop sizes it to
  `(generation_steps + 1)` frames per trajectory, and grows it through
  `resize(capacity)` when the sink is a `ResizableSink`; a smaller sink that
  is not one is refused up front. Core additions used here:
  `OrderedStructureSampler`, `StructureSource`, `FitPolicy`, `WithinBudget`,
  `ResizableSink`, `BaseDynamics.check_initial_batch`, `required_input_keys`,
  `bookkeeping_keys`, `nvalchemi.data.datapipes.distributed_shard`,
  `AtomicDataZarrReader.store`, `Batch.drop_level`, `pop_level`, `set_level`,
  `level_keys`.
- **On-policy segment loop** — `DistillationStrategy` accepts `on_policy` and
  `reference_dataset`, and `run()` then drives generate-label-train segments
  until `num_steps`. The loop seeds a state batch; each segment then generates
  `generation_steps` frames with the student's own propagator, labels and
  captures them, and takes `training_steps_per_segment` optimizer steps on a
  freshly mixed reference/replay stream. That stream's sampler seeds from
  `OnPolicyConfig.seed` plus the segment index. One segment is one epoch, and
  the segment is also the restart granularity. A second `run()` keeps the
  replay buffer the first one filled, and the closing validation is skipped
  when a cadence already validated at the final step. The propagator must hold
  the very student module being trained, alone or composed, and is held in
  evaluation mode to generate. At construction, the reference dataset is
  probed for fields the labeling hook strips, for the device it emits on, and
  for the teacher fields the propagator's scorer declares. Generated frames
  are staged on the reference dataset's device unless `replay_device`
  overrides it, and every placement blocks on a copy into host memory.
  `on_policy` and `reference_dataset` travel through `to_spec_dict` as
  references, on the terms the recipes entry below describes. Core additions
  used here: `nvalchemi.data.datapipes.dataset_device`, `same_device`.
- **Relaxation on-policy generation** — `OnPolicyConfig` gains `fmax` and
  `convergence_hook`, which give a relaxation propagator such as `FIRE` the
  trajectory lifecycle its paths need. A converged structure freezes and
  graduates on the step its status reaches `exit_status`; it is captured
  there, once, as the minimum it reached, and retired from the batch and
  backfilled at the segment boundary. There, initial structures are drawn into
  the room they freed through `OrderedStructureSampler.draw(...,
  on_miss="skip")`, so the replay buffer keeps filling with informative frames
  instead of near-duplicates of a structure that stopped moving. `fmax` is the
  max-force-norm threshold a recipe can hold, and `convergence_hook` is the
  live criterion no recipe describes. `OnPolicyConfig.convergence_criterion`
  resolves the two into the one status-migrating, every-step hook the
  lifecycle drives, which is also the propagator's convergence detector for
  the duration of the run. A propagator already carrying a status migrator, a
  multi-sub-stage `FusedStage`, and a `DomainParallel` propagator (its step
  dispatches no `ON_GRADUATE`) are refused when the config is built, as is a
  criterion migrating off a status the structures never carry; a migrator
  registered afterwards, or a propagator-owned sampler, is refused when the
  run starts. The core `OrderedStructureSampler`, and `InitialStructures` with
  it, gains `recycle`. It wraps the position to the front of the rows this
  rank owns instead of letting the batch narrow, and it records its wrap count
  in the restart bundle. Frames are captured by two routes that partition
  them. `TeacherLabelHook` stores the structures still relaxing, narrowing the
  frame to them before the teacher runs. A converged-frame hook listens at
  `DynamicsStage.ON_GRADUATE` and stores each minimum once, off
  `ctx.graduated_mask`, through the core `ConvergedSnapshotHook` and
  `Batch.index_select`/`clone(drop=)`; the minima are labeled in one teacher
  pass as its sink is drained onto the buffer's own device, and a device-less
  `ReplayBuffer` now pins that device on its first `extend`.
  `TeacherLabelHook` narrows only when given the propagator's `exit_status`,
  which the lifecycle sets, reading the live status through
  `BaseDynamics.active_graph_mask`, so a propagator managing its own
  convergence keeps its final frames. A budget-graduated fused sub-stage is
  captured on the step it graduates, a backfilled structure is restamped with
  fresh bookkeeping, and a run whose last trajectory finishes warns once and
  trains its remaining steps on the frames it has. A trajectory whose
  positions or forces stop being finite is frozen uncaptured on that step,
  then retired and backfilled at the boundary with a warning counting such
  trajectories, rather than propagated and labeled as NaN into the loss.
  `OnPolicyConfig.divergence` decides what counts as divergent: a runtime-only
  per-graph boolean predicate over the batch, evaluated once per step and
  ORed into a record the capture hook and the boundary read. It defaults to
  the exported `nonfinite_divergence`, an alias of the core
  `nvalchemi.dynamics.hooks.nonfinite_graph_mask`. A predicate returning
  anything but one `bool` per graph is refused. Label precision on every
  route is the scorer's `autocast`. A reference dataset emitting on an
  accelerator other than `devices[0]` is refused at construction. The
  `TeacherLabelHook` route stages each segment in the configured
  `capture_sink`, which is grown at most once, for the first segment, and
  never shrunk, since a backfill never grows the batch past its initial size.
  A custom `StructureSource` drives the lifecycle once its `initial_batch`
  stamps the `status` and `system_id` bookkeeping. The construction probe
  dispatches a copy of the criterion to the probed row. A criterion that
  raises on the propagator's outputs, or leaves `status` unmoved where it
  converged, is refused up front. A criterion reading a key no `compute()`
  produces warns instead of refusing, since a hook may write it during the
  step. This probe follows `OnPolicySettings.probe`, like the propagator's.
  Core additions used here: `DynamicsStage.ON_GRADUATE`,
  `DynamicsContext.graduated_mask`, `nonfinite_graph_mask`,
  `BaseDynamics.active_graph_mask`, `Batch.index_select(drop=)`,
  `Batch.clone(drop=)`, `OrderedStructureSampler.recycle`.
- **Multi-GPU and multi-node on-policy distillation** — the segment loop runs
  data-parallel under a `DDPHook` instead of refusing a multi-rank launch. Each
  rank propagates the strided shard of `initial_structures` it is dealt, labels
  those frames with its own teacher replica, and fills its own replay buffer,
  so no generated frame or teacher pass is duplicated. The reference dataset
  stays replicated, and every rank draws from all of it. The mixture sampler's
  `seed` moves by `rank * rank_seed_stride`, and every integer `random_seed`
  in the propagator's stage tree (`NVTLangevin.random_seed` among them) moves
  by the same offset through `BaseDynamics.seed_offset`, undone with the
  negated offset when the run exits, so ranks decorrelate. The stride is
  `OnPolicySettings.rank_seed_stride` (default `1_000_003`), recorded in the
  recipe and checked on restart. A stage holding randomness `seed_offset`
  cannot move, such as a `torch.Generator` and no integer seed, is named in a
  warning. The student's gradient all-reduce is the only per-step training
  traffic between ranks. A multi-rank run whose student nothing wraps is
  refused up front: with a `DDPHook` registered the check reads
  `DDPHook.wrapped_keys`, and without one it reads ownership through
  `unwrap_model`. A run with fewer initial structures than ranks is refused by
  a verdict agreed across ranks through
  `nvalchemi.training.distributed.all_reduce_flags` before the first gradient
  collective, naming the ranks whose shard seeded nothing. A structure count
  the world cannot deal out evenly warns, since a shorter shard's frames are
  drawn more often. `require_wrapped_student=False` waives the wrapped-student
  refusal with a one-time warning, for wrappers working in place such as
  FSDP2's `fully_shard`, and leaves keeping the ranks' students in step to the
  caller. A reference dataset staged on an indexed accelerator that some rank
  does not train on is reported from every rank, because the replay buffer
  follows it. An index-less `replay_device` names this rank's current device,
  resolved through the core `nvalchemi.data.resolve_device`. The rows a rank
  owns are public as `DistillationStrategy.structure_shard`, and the
  run-local fields a frame sheds are read off `BaseDynamics.bookkeeping_keys()`.
  Core additions used here: `BaseDynamics.seed_offset`, `bookkeeping_keys`,
  `NVTLangevin.random_seed`, `OrderedStructureSampler.rank`, `world_size`,
  `nvalchemi.training.distributed.all_reduce_flags`, `DDPHook.wrapped_keys`,
  `nvalchemi.training.unwrap_model`, `nvalchemi.data.resolve_device`.
- **Embedding, Hessian, and Boltzmann objectives** — three loss terms distill
  what a reference-labeled dataset has no column for. Each is checked at
  construction, on the training side and on a `validation_config` loss alike.
  `EmbeddingMatchingLoss` matches the teacher's per-atom representation.
  `embedding_distillation_fn` takes the student's `compute_embeddings` pass and
  routes it through an `EmbeddingProjector` whenever the two widths differ; the
  projector is registered as a `"projector"` model with an optimizer of its
  own. The student, projector, and teacher widths are reconciled up front. A
  projector registered with `frozen_student=True` lets the term train it alone
  over a deliberately frozen student, whose detached embeddings are otherwise
  refused. `HessianMatchingLoss` matches Hessian-vector products along one
  probe. The new `hessian` teacher signal writes `teacher_hvp` and the
  `teacher_hvp_probe` it was taken along (`InProcessTeacherScorer.label_hvp`,
  `probe_seed`, and `label(batch, probe_seed=)`). `hessian_distillation_fn`
  differentiates the student's energy twice along that probe, on a pass
  narrowed to the energy that reuses the stock forward's neighbor list. A
  companion field is refused as a loss target, a direct-force student is
  warned that the term reaches its energy head alone, and
  `DistillationStrategy.validate` pins the probe per validation batch so the
  metric compares across passes. `BoltzmannMatchingLoss` is the
  beta-interpolated relative entropy between the teacher's and the student's
  Boltzmann distributions at a temperature over the batch's configurations,
  read as a sample of the student's own ensemble. It therefore requires
  `on_policy`. It refuses a relaxation propagator and any convergence
  criterion: the propagator's own, one registered on it, or
  `fmax`/`convergence_hook`. It also refuses any place in the validation loss,
  whether an explicit validation-side term or a `ValidationConfig` that would
  reuse the training loss. It warns about a mixed `replay_ratio` or an
  unbounded replay buffer. Under data parallelism it gathers the reduced
  energies across ranks with a differentiable all-gather, so every rank trains
  on the world-batch loss. It checks the one-system guard on that gathered
  batch by atom counts alone: a rank holding one system beside a rank holding
  another of a different size is refused, while two systems of the same size
  pass. It drops a graph whose teacher or student energy is not finite. The
  gather and the guard are constructor settings: `world_batch` (`None` infers
  the gather from the `nvalchemi.training.distributed` predicates, and
  `True`/`False` force it) and `check_one_system` (default `True`). The
  equilibrium-sampling check the term runs on the propagator reads
  `OnPolicySettings.samples_equilibrium`. `None` infers it from the
  propagator's `samples_equilibrium` class variable and its convergence
  criteria, treating a relaxation optimizer or a criterion as not sampling
  one, and the refusal names the setting as the override. A Boltzmann run's
  checkpoint restores with its loop, because `DistillationStrategy.from_spec_dict`,
  `from_checkpoint_dict`, and `load_checkpoint` take `on_policy`,
  `reference_dataset`, and `validation_config` through the core
  `**runtime_overrides`, with `load_checkpoint(models=)` handing over the live
  student the propagator holds; a spec naming a `DistillationStrategy`
  subclass dispatches to it carrying every override. Core additions used
  here: `nvalchemi.models.hessian_vector_product`,
  `BaseModelMixin.narrowed_outputs`, `Batch.without_keys`,
  `nvalchemi.training.losses.masked_mean`, `graph_balanced_mean`,
  `BaseDynamics.samples_equilibrium`,
  `nvalchemi.training.distributed.all_gather_rows`, `all_gather_objects`,
  `**runtime_overrides`, `load_checkpoint(models=)`.
- **Evaluation and acceptance suite** — new
  `nvalchemi.training.distillation.evaluation` subpackage that decides whether
  a distilled student ships. `evaluate_accuracy` measures energy, force, and
  stress MAE/RMSE over a holdout through `ValidationLoop`, against the
  dataset's own labels or the teacher's (on disk or scored on the fly). It
  runs with no autocast and accumulates exact global residual sums in float64.
  A bare model handed as `scorer=` is wrapped in a scorer whose labels are
  cast to the dtype the store would hold them at; a supplied scorer's labels
  are not cast. Whenever forces are compared, against either target family,
  it adds force cosine similarity, per atom and magnitude-weighted (the
  aggregate is what `min_force_cosine` reads), and a `force_nonfinite_atoms`
  count. It adds per-atom energy residuals when `atomic_energies` is among the
  quantities, and it refuses a scorer paired with reference targets or one
  returning a label outside `teacher_*`. The quantities it measures are an
  open table. Each is an `AccuracyQuantitySpec` naming the prediction and
  reference keys, the teacher signal that labels it, and the supervised loss
  it carries. The built-in quantities are published as
  `BUILTIN_ACCURACY_QUANTITIES`, and `evaluate_accuracy(quantities=...)` takes
  their names or specs of your own, requiring at least one that carries a
  supervised loss. `label_dtype` pins the dtype scored-on-the-fly labels are
  cast to. The gradient mode is decided from the model behind a data-parallel
  wrapper, through `unwrap_model` and `BaseModelMixin.requires_autograd`.
  `non_conservative_residual` integrates the teacher's work around closed
  loops in configuration space, laid out around each graph's own centroid. It
  reports the lower bound this places on a conservative student's RMS
  per-atom force error, absolute and relative to each graph's force scale.
  `StabilityMonitor` is a dynamics hook that reports energy drift (per atom,
  per step, and as a fitted per-nanosecond rate), the RMS fluctuation and
  largest excursion about the fit, and momentum conservation over a
  student-driven trajectory. It discards a `warmup_steps` window, names the
  field a sample lacks, stops with a warning when the batch composition
  changes (`stop_on_composition_change=False` keeps going), and refuses a run
  that passes `ctx.active_graph_mask`. Its `divergence` predicate decides
  which frames count as diverged, defaulting to the loop's
  `nonfinite_divergence`, and the step it first fired on is reported as
  `first_divergence_step`. `aggregate` chooses whether the reported drift is
  the worst graph in the batch (`"max"`) or the mean over graphs (`"mean"`).
  `extensivity_error(..., drop_keys=)` checks energy scaling across
  replicated cells built through the core `make_supercell`, which refuses a
  system field named in neither `extensive_keys` nor `intensive_keys`, and
  drops the bookkeeping a sampler-seeded run stamped. The measurements read
  `pbc` before falling back to the presence of a `cell`, and a cell whose
  `pbc` marks every axis non-periodic is refused by name.
  `radial_distribution` and `compare_radial_distributions` score structural
  match with a bounded Jensen-Shannon divergence, pooled over every species
  or resolved to one pair. Each pair is apportioned between two bins, so the
  histogram is continuous in the positions, and a frame enclosing no volume
  is refused. `measure_throughput` reports atoms/s and ns/day from a
  warmup-discarded, device-synchronized window over the steps the propagator
  actually took. `build_acceptance_report` turns those measurements into
  per-student verdicts against `AcceptanceThresholds`, a speed-versus-accuracy
  Pareto table, and the from-scratch bar. That bar, `max_from_scratch_ratio`,
  is the largest accepted ratio of the distilled error to an equal-size
  from-scratch student's, taken on `energy_per_atom_mae`, `forces_mae`, and
  `stress_mae`, whichever both carry, keeping the worst. The report renders
  as Rich tables and exports as nested dictionaries or flat scalars. An
  acceptance bar with no measurement behind it fails rather than being
  skipped, and a bar whose family was measured but whose number was not names
  the missing quantity or timestep. A non-finite measurement fails on `not
  finite` and is left off the Pareto front, a baseline of exactly zero is
  unbeatable, and a family scored on different holdouts or timed on different
  batches is refused. The bars themselves are a public, extensible table. Each
  is an `AcceptanceBar` naming the threshold it is set under, the measurement
  families and the check it reads, the accuracy quantities that decide it,
  and the direction it passes in. The shipped bars are `DEFAULT_BARS`, and
  `BAR_FAMILIES` (derived from them) maps each bar to the `StudentEvaluation`
  slots it reads. `measured_bars(..., bars=)` answers which bars a partial
  measurement can decide, `build_acceptance_report(..., bars=)` judges against
  a table of your own, and `AcceptanceThresholds.extra` carries the thresholds
  of bars the model has no field for. Every measurement (`AccuracyMetrics`,
  `StabilityMetrics`, `ExtensivityMetrics`, `RDFComparison`,
  `ThroughputMetrics`, `StudentEvaluation`, and the residual) is a Pydantic
  model sharing `MeasurementRecord`, and it rebuilds from its export with
  `from_dict`. `AccuracyMetrics.errors` and `StudentEvaluation.extra` are the
  open slots for quantities and measurements the built-in fields do not name,
  and `StudentEvaluation.weights` records whether a student was scored on
  `"ema"` or `"raw"` weights. `StabilityMonitor`, `StabilityMetrics`,
  `total_momentum`, `measure_throughput`, `ThroughputMetrics`, and
  `MeasurementRecord` live in core and are re-exported from the subpackage.
  Core additions used here: `nvalchemi.dynamics.hooks.StabilityMonitor`,
  `StabilityMetrics`, `total_momentum`, `kinetic_energy_per_graph`, `KB_EV`,
  `nvalchemi.dynamics.measure_throughput`, `ThroughputMetrics`,
  `nvalchemi._serialization.MeasurementRecord`,
  `BaseModelMixin.requires_autograd`, `nvalchemi.training.losses.per_graph_sum`,
  `nvalchemi.training.ensure_reiterable_validation_data`,
  `nvalchemi.data.transforms.make_supercell`, `DEFAULT_EXTENSIVE_SYSTEM_KEYS`,
  `DEFAULT_INTENSIVE_SYSTEM_KEYS`.
- **Reproducible recipes, teacher references, and the `distill` CLI** — a
  distillation run now survives a round trip. Checkpoints store the frozen
  teacher *once per checkpoint root*, and
  `DistillationStrategy.checkpoint_model_references` declares it. Ordinarily
  the first write under a root holds its weights, the manifest gains a
  `model_references` entry naming that index plus a fingerprint, and later
  indices contribute no teacher weight file. A load reads the stored copy back
  and verifies the fingerprint, so a replaced copy raises instead of quietly
  training a student against a different model. The fingerprint hashes each
  state-dict entry's name, shape, dtype, and a sample of its values read at
  `float64` on the host, so it identifies a model rather than validating it.
  One root holds one copy. Storing a *different* copy of a declared model into
  a root that already holds one is refused. An identical copy is reused while
  its file is on disk, and rewritten at the current index when that file went
  missing, which repairs the root. The manifest stays at `schema_version` 1,
  so an older nvalchemi still reads it, but only at the index holding the
  teacher's weights. The teacher's `checkpoint_spec()` rebuilds its
  architecture and is never trusted for its weights, and
  `save_trainable_state_only=True` narrows the student alone, so the
  once-stored teacher stays whole.
  `OnPolicyConfig.to_spec_dict`/`from_spec_dict` carry the whole segment loop:
  every `OnPolicySettings` field verbatim; the propagator as the `cls_path` and
  keyword arguments it rebuilds from, with the student rebound at build time;
  the scorer as its `signals`, `dtype`, `probe_seed`, `neighbor_list`, and
  `autocast` (a dtype by its `torch` name) over the strategy's own
  `"teacher"`; and `initial_structures` as the store it reads under its
  budgets and `recycle`, never its position. Another `StructureSource`
  travels as its own `to_spec_dict` under its class path (`source_cls`), and a
  source with neither `to_spec_dict` nor `from_spec_dict` is refused with the
  remedy. A custom `TeacherScorer` travels the same way under `scorer_cls`
  when it satisfies the `SpecSerializable` protocol
  (`to_spec_dict`/`from_spec_dict`), and is refused with the remedy otherwise.
  The runtime-only fields (a `convergence_hook`, `capture_sink`,
  `replay_admission`, and `divergence`) are omitted with a warning that names
  them; a rebuilt loop stages frames in host memory, admits every frame, and
  flags non-finite positions or forces. A policy instance on `replay_eviction`
  is recorded as `"fifo"`, with a warning unless it is a `FIFO`. A
  propagator's live hooks and sinks are omitted with a warning, and an
  in-memory dataset is refused rather than approximated. A `MultiDataset`
  travels as the list of stores it concatenates, and a dataset is named by its
  store through the core `dataset_spec_dict`/`DatasetRef`.
  `DistillationStrategy.to_spec_dict` carries `on_policy` and
  `reference_dataset` on the same terms, except that it catches those
  refusals and leaves the whole `on_policy` entry out with a warning, so the
  spec rebuilds a strategy that runs offline over the dataloader passed to
  `run()`. A spec naming a subclass under
  `strategy_cls` rebuilds that subclass with every runtime override handed on.
  A live object passed to `from_spec_dict`, `from_checkpoint_dict`, or
  `load_checkpoint` outranks the recipe. The stores a recipe names are opened
  on the rebuilt strategy's own device, so a checkpoint restored under another
  `map_location` reads its data there. An interrupted on-policy run resumes
  its trajectory, propagator counter, initial-structure position, and replay
  frames through the checkpoint, exactly for the counter-based-RNG
  integrators, at segment granularity. The labeling cadence resumes too, so
  the restart neither pays a second teacher pass at the boundary it stopped on
  nor stores the frame beside it. The restored frames replace the buffer's
  contents rather than merging into them: `ReplayBuffer.clear()` drops every
  stored frame and unfreezes the key schema, so the next `extend` freezes it
  afresh. A setting the resumed loop sets differently from the recorded one
  is reported, and a run whose generation ran dry resumes training on its
  buffer rather than regenerating. The restart bundle packs the trajectory
  batch through the core `Batch.to_raw_dicts(drop=)`. The bundle is
  rank-local, since it rides in a strategy checkpoint that `CheckpointHook`
  writes on rank zero alone. A world size that differs at either end of a
  restart, read off the shard the position records, therefore leaves a bundle
  the run cannot consume, and `OnPolicySettings.restart` says what happens.
  `"error"` (the default) refuses to start. `"reseed"` drops the bundle with a
  warning, and every rank reseeds from its own share with a cold replay
  buffer. `"resume"` additionally refuses a restore carrying no bundle.
  **Behavior change:** a multi-rank restart used to drop the bundle silently;
  it now raises unless the recipe sets `restart: "reseed"`. New
  `nvalchemi-training distill` group authors, validates, runs, and gates a
  JSON `DistillationJobSpec`. `init` scaffolds offline or on-policy recipes at
  a student tier from a registry. `StudentTier(name, kwargs)` is a size
  template of width, depth, and radial basis size (`hidden_dim`,
  `num_layers`, `num_radial`) and nothing else; the shipped tiers are
  `DEFAULT_STUDENT_TIERS` (`small`, `base`, `large`), and
  `register_student_tier(name, **kwargs)` adds one. `--tier` is validated when
  the command runs, and `--tier-kwargs KEY=VALUE` overrides a tier's
  constructor arguments. `init` records a `CheckpointHook` in `student.hooks`
  with `save_at_end: true`, so a run that ends on a step the cadence missed
  still writes a terminal checkpoint while a hand-written hook runs as
  declared, records `dataset.batch_size` (`--batch-size`, default `8`), and
  requires `--initial-structures` in on-policy mode. `spec report` renders
  derived teacher signals, batch composition, and acceptance bars. It refuses
  what a recipe settles on its own before a teacher reaches a device. An
  `on_policy` block is validated through `OnPolicySettings`' and
  `InitialStructures`' own constraints, so `spec report` refuses an
  out-of-range setting, a misspelled or non-positive budget, a block naming no
  store, a step budget below one, a `dataset.format` no loader builds, an
  unloadable teacher or student source, a `replay_ratio` or `batch_size`
  leaving one mixture source without a whole sample, and a `teacher_scorer`
  block whose `neighbor_list` or `autocast` is spelled in a way the scorer
  does not accept. A rule the strategy or the segment loop owns is not
  repeated by the report; it surfaces once, at `spec run`, with the owner's
  message. A recipe whose `teacher_signals` fail to cover the losses is
  refused when it is read, through the public
  `DistillationStrategy.resolve_teacher_signals(loss_fn, validation_config=,
  teacher_signals=)` classmethod the CLI shares with the strategy. `spec run`
  executes a recipe. `spec resume` continues from a checkpoint directory and
  the recipe, at the budget `--budget` names: `checkpoint` (the default) keeps
  the `num_steps`/`num_epochs` the checkpoint's stored spec recorded, and
  `recipe` takes the recipe's, so an edited recipe extends or shortens the
  run. A recipe budget below what the checkpoint completed, or in the other
  unit, is refused instead of silently overwriting the run. **Behavior
  change:** the recipe's budget used to win by default; pass `--budget recipe`
  to keep that. Both commands take `--distributed/--no-distributed` (auto when
  `WORLD_SIZE > 1`), `--ddp-backend`, and the loader options the core `train
  spec run`/`resume` take (`--batch-size`, `--drop-last`, the validation
  dataset and cadence options, and the rest of
  `cli_common.common_loader_options`). `evaluate` scores the student's weights
  over the recipe's holdout as `--weights` says: `auto` (the default) reads the
  EMA average when `student.hooks` carries an `EMAHook` and the trained
  weights otherwise, `ema` fails when the checkpoint holds no average, and
  `raw` scores the trained weights regardless. It takes the core prefetch
  options and exits non-zero on a missed bar. It writes a non-finite metric to
  `--json-out` as the string `"nan"`, `"inf"`, or `"-inf"` through
  `nvalchemi._serialization.json_safe`, which every metric's `from_dict`
  decodes back into the float, and it records which of the two weight sets it
  scored as `StudentEvaluation.weights`. Recipe hooks are recognized by class,
  so a `CheckpointHook` or `EMAHook` subclass counts as one, and the dataset
  formats and model sources a recipe admits are the core CLI's own literals.
  `evaluation.thresholds` is narrowed to the accuracy bars `evaluate` can
  fill. A stability, throughput, extensivity, RDF, or from-scratch bar, or an
  accuracy bar reading a quantity the recipe never compares, is therefore
  refused when the recipe is parsed. A multi-rank `spec resume` defaults
  `--map-location` to this rank's device, so no rank stages its weights
  through rank zero's; the live strategy's `devices` decide where the restored
  run continues. See `docs/userguide/distillation_recipes.md` and the
  `nvalchemi-distillation` agent skill. Core additions used here:
  `nvalchemi.training.ModelReference`,
  `TrainingStrategy.checkpoint_model_references`, `CheckpointHook(save_at_end=)`,
  `TrainingStrategy.run_setup_hooks`, `ComposedLossFunction.to_spec`,
  `nvalchemi.training._spec_utils.dataset_spec_dict`,
  `dataset_from_spec_dict`, `DatasetRef`, `SpecSerializable`,
  `Batch.to_raw_dicts`, `nvalchemi._serialization.json_safe`,
  `nvalchemi.training.cli_common` and its `nvalchemi.training.cli`
  re-exports.
- **Distillation user guide and on-policy example** — new
  `docs/userguide/distillation.md` covers the whole feature from the user's
  side: the teacher signals and how the strategy resolves them; the offline
  path over a teacher-labeled Zarr store; the on-policy segment loop with its
  mixture, cadence, and capacity arithmetic; the neighbor-list hooks a graph
  student needs on the propagator and on the strategy; the convergence
  lifecycle a relaxation propagator needs; the embedding, Hessian, and
  Boltzmann objectives and what each asks of the run; scaling the loop across
  ranks; the accuracy, stability, throughput, and extensivity measurements and
  the acceptance report that gates the student on them; and pointers into the
  recipes guide for the checkpoint and restart contract. Two topics get their
  own sections. One explains why an on-policy reference dataset has to be
  teacher-labeled and how to reshape an existing DFT-labeled dataset into one;
  the other covers distilling a non-conservative direct-force teacher into a
  conservative student. The core pieces the guide leans on are documented
  where they live: `docs/userguide/dynamics.md` gains a "Structure sources"
  section on `OrderedStructureSampler`, and
  `docs/userguide/distributed_training.md` the process-group timeout note. New
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
