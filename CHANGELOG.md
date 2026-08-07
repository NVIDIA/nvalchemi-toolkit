# Changelog

## 0.2.0 — 2026-08-07

Second public-beta release. The headline additions are a training and
fine-tuning framework, domain-decomposed multi-GPU inference and dynamics, a
UMA model wrapper, and a reporting layer for live progress and TensorBoard
output.

### Training And Fine-Tuning

- **`nvalchemi.training`** — a Pydantic-driven supervised training framework.
  `TrainingStrategy` owns a run: the models to train, the dataloader, losses,
  optimizers, schedulers, hooks, and the step or epoch budget.
  `FineTuningStrategy` extends it with pretrained-checkpoint initialization,
  module patches, trainable-parameter filters, and opt-in reuse of the source
  checkpoint's loss and optimizer configuration.
- **Loss library** (`nvalchemi.training.losses`) — energy (`EnergyMSELoss`,
  `EnergyMAELoss`, `EnergyHuberLoss`), force (`ForceMSELoss`, `ForceHuberLoss`,
  `ForceL2NormLoss`), and stress (`StressMSELoss`, `StressHuberLoss`) terms
  built on a common `BaseLossFunction` template with per-atom normalization,
  masking, and graph-balanced reductions. `ComposedLossFunction` combines terms
  under static weights or a `LossWeightSchedule` (`ConstantWeight`,
  `LinearWeight`, `CosineWeight`, `PiecewiseWeight`).
- **Training hooks** (`nvalchemi.training.hooks`) — `CheckpointHook` for step-
  or epoch-based saves, `EMAHook` for exponential moving averages, `DDPHook`
  for data-parallel wrapping, `MixedPrecisionHook`, and `ModulePatchHook` /
  `TrainableParameterHook` for fine-tuning control.
- **Restartable checkpoints** — strategy checkpoints save and restore models,
  optimizers, schedulers, runtime counters, and hook state, with restart-safe
  device placement. Hooks implementing `CheckpointableHook`, such as EMA, carry
  their own restart state, so a resumed run keeps its averaged weights.
- **First-class validation on `TrainingStrategy`** — set a `ValidationConfig`
  on `strategy.validation_config` and validation runs at the configured step or
  epoch cadence, plus one final pass at end-of-training; the latest summary is
  stored on `strategy.last_validation`. Mechanics live in a public,
  context-managed `ValidationLoop` that also runs standalone outside training.
  An `inference_model` slot lets EMA (or SWA, or a distillation teacher)
  publish averaged weights for validation to read. The `AFTER_VALIDATION` hook
  stage fires immediately after each pass so loggers can read the live summary.
  For per-batch logging, pass a `batch_callback` (any object matching the
  `BatchValidationCallback` protocol) on the config; it is invoked once per
  validation batch with the batch, predictions, and per-batch loss.
- **Metric-driven learning-rate schedulers** — `ReduceLROnPlateau` is supported
  via `OptimizerConfig.scheduler_metric_adapter` (a summary-dict key string or
  a callable). Time-based schedulers step every optimizer step as before;
  metric-driven schedulers step at validation checkpoints, where the validation
  summary supplies the metric.
- **`EMAHook._build_averaged_model` override point** — a caller that owns model
  sharding can supply a pre-built `AveragedModel` in place of the default
  deepcopy, which enables EMA on `fully_shard` (FSDP2) / DTensor models.
  Default behavior is unchanged.
- **`nvalchemi-training` CLI** — scaffolds, validates, reports on, and runs
  training specifications. `train init` scaffolds a from-scratch spec and
  `finetune init mace|aimnet2|checkpoint|custom` scaffolds a fine-tuning spec
  per source; `schema dump` / `schema template` emit JSON schema and templates
  for offline authoring; `spec report` validates and renders a spec, and
  `spec run` / `spec resume` build the runtime components and execute it.
- MACE training example (`examples/advanced/10_mace_training.py`) covering an
  end-to-end training workflow.

### Distributed Inference And Dynamics

- **Domain decomposition** — a spatial halo strategy and a graph-parallel
  strategy, both driven by a declarative `MLIPSpec` a model wrapper publishes
  as `distribution_spec`. Ewald, PME, MACE, AIMNet2 and UMA ship specs;
  composed pipelines decompose per stage. Energy, forces and stress agree with
  a single-GPU reference to fp32 rounding under both strategies, eager and
  compiled. `nvalchemi.distributed.pin_fp32` pins full-precision fp32 for runs
  that must match a reference, since TF32 makes distributed and single-process
  results diverge well beyond fp32 noise.
- `nvalchemi/distributed.py` is now a package. `DistributedManager`,
  `resolve_world_size`, `resolve_global_rank`, and `collective_device` moved to
  `nvalchemi/distributed/_runtime.py` and stay importable from
  `nvalchemi.distributed`.

### Model Wrappers

- **UMA (fairchem-core) wrapper** — new `UMAWrapper` exposes UMA
  (Universal Models for Atoms) foundation models (`uma-s-1p1`,
  `uma-s-1p2`, `uma-m-1p1`) through the `BaseModelMixin` interface,
  ready for any dynamics engine or standalone inference. UMA is
  multi-task; the wrapper is pinned to one head at construction (OMol,
  OMat, OC20, ODAC, OMC). Input conversion is tensor-native (no ASE
  round trip); energy is the differentiable primitive with forces and
  (for periodic tasks) stress from autograd. Install via the new `uma`
  optional extra (`pip install 'nvalchemi-toolkit[uma]'`), which is
  declared conflicting with the `mace` and `cu12`/`cu13` extras
  (incompatible `e3nn` / `torch` pins) and resolves into its own
  environment. `from_checkpoint` forwards fairchem's `inference_settings`
  (including `"turbo"` for `torch.compile`). See the
  `examples/advanced/09_uma_nve.py` NVE/NVT/NPT walkthrough.
- **Slab correction for electrostatics** — `EwaldModelWrapper` and
  `PMEModelWrapper` accept `slab_correction` (default `False`) to apply the
  two-dimensional slab correction. Per-system `pbc` flags with one `False`
  entry mark slab systems, for example `[True, True, False]`, and mixed slab
  and three-dimensional periodic batches are supported.
- **Trainable model wrappers** — `MACEWrapper` forwards its own `self.training`
  flag to the MACE model, and `PipelineModelWrapper` retains the autograd graph
  when it is in training mode with gradients enabled, so both can sit inside a
  `TrainingStrategy`.
- **Pipeline neighbor-list adaptation policy** — `PipelineModelWrapper`
  now accepts `neighbor_adaptation` (`"auto"`, `"always"`, `"never"`) and
  `max_cutoff_ratio` (default `1.5`). The default `"auto"` mode only filters
  a source neighbor list for a smaller cutoff when the source cutoff is at most
  `max_cutoff_ratio` times the target cutoff; larger gaps get separate source
  lists. `"always"` builds one max-cutoff source list, while `"never"` builds
  exact cutoff source groups and skips cutoff filtering.

### Core Data Layer

- **Explicit batch loading** — `Dataset.load_batches(...)` is the canonical
  explicit batch API. It issues fused `read_many` requests through the reader,
  preserving the Zarr backend's coalesced I/O path, and returns one `Batch` per
  requested batch-index list. `Dataset.read_many(...)` and
  `Dataset.get_batch(...)` cover single-request sample and batch reads.
- **Dataset composition and sampling** — `MultiDataset` wraps several `Dataset`
  instances behind one global index space and routes `load_batches` requests to
  the owning child datasets, restoring the requested global sample order.
  `MultiDatasetSampler` and `MultiDatasetBatchSampler` add multidataset-aware
  sampling policies over that index space.
- **In-memory datapipes** — new `InMemoryDataset` stores a fully materialized
  `Batch` in memory and serves graph-indexed `Batch` selections through the
  same `load_batches` / fused-prefetch interface used by `DataLoader`. It can
  be constructed from an existing `Batch` or materialized from a reader in
  chunks, with optional field-level metadata and batch transforms.
- **User-specified transforms** — `Dataset` accepts a `transforms=` kwarg
  (per-sample `(AtomicData, metadata) -> (AtomicData, metadata)`) and
  `DataLoader` accepts a `batch_transforms=` kwarg (per-batch `Batch -> Batch`).
  Both default to `None` (backward compatible). New `nvalchemi.data.transforms`
  subpackage exposes a polymorphic `Compose` utility plus `SampleTransform`
  and `BatchTransform` type aliases, re-exported from `nvalchemi.data`.
  Per-sample transforms run after device transfer on both sync and prefetch
  paths; per-batch transforms run on the consumer thread after `Batch.from_data_list`.
  Transform failures are wrapped in `RuntimeError` with `transform[<i>]`
  breadcrumb and `__cause__` preserved.

### Reporting And Hooks

- **Reporting subsystem** (`nvalchemi.hooks.reporting`) —
  `ReportingOrchestrator` fans workflow events out to registered reporters.
  `RichReporter` renders a live terminal dashboard through `DynamicsRichLayout`
  or `TrainingRichLayout`, and `TensorBoardReporter` writes scalar summaries
  through the `tensorboard` extra. Scalar extraction helpers
  (`collect_scalars`, `extract_dynamics_scalars`, `extract_loss_scalars`,
  `extract_optimizer_lr_scalars`) pull values out of dynamics and training
  state, and reporting is rank-safe under distributed runs.
- **`StageTimingHook`** — per-stage wall-clock timing for any hook-enabled
  workflow, for both dynamics and training.
- **`TorchProfilerHook`** — captures PyTorch profiler traces through
  PhysicsNeMo's profiler wrapper.
- **`CheckpointableHook`** protocol — hooks that own restart state declare it
  and participate in strategy checkpoint save and load.

### Fixed

- **Ewald charge gradients and cell derivatives** — the reciprocal term was only
  ever differentiated with respect to positions and charges, so a non-hybrid
  Ewald returned a wrong `dE/dq`, and strain-autograd through the detached
  Green's function gave a wrong stress. Work needing a cell derivative now
  routes to the staged reciprocal.
- **Ewald / PME strain cache** — cached k-vectors were rebuilt from the strained
  cell, so a second stress evaluation reused the first call's autograd graph.
- **Merged force and stress autograd** (#88) — `PipelineModelWrapper` and
  `AIMNet2Wrapper` now take forces and stress from a single merged autograd
  call when both are requested, and detach autograd tensors at the pipeline
  boundary so graph-attached batch tensors are released on both normal returns
  and exceptions.
- **Asymmetric strain displacement** (#111) — `prepare_strain` applies only the
  symmetric part of the displacement tensor when deforming positions and cells,
  so an asymmetric energy yields a symmetric stress.
- **Double particle thermostat coupling in NPT** (#96) — high-level NPT
  velocity half-steps route through the no-drag NPH primitive, since particle
  Nosé-Hoover scaling is applied separately in the toolkit's operator split.
- **Inflight refill system IDs** (#113) — systems retained across an inflight
  refill keep their IDs and status bookkeeping; only replaced slots take new
  sampler IDs.
- **Inactive cells during masked updates** (#116) — a masked post-update
  restores the cells of systems outside the mask instead of overwriting them.
- **MACE cuEquivariance device context** (#114) — checkpoint target devices are
  normalized before load and final placement, and cuEq conversion runs under
  the requested CUDA device context. Non-CUDA targets warn and skip conversion.
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

- `cells_inv` argument on `_cell_kinetic_energy` in
  `nvalchemi/dynamics/integrators/npt.py`. Cell kinetic energy
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

### Requirements

- Python 3.11–3.13
- PyTorch >= 2.8
- `nvalchemi-toolkit-ops` >= 0.4.1; the CUDA build now comes from the `cu12` /
  `cu13` extras
- `nvidia-physicsnemo` >= 2.0.0, `click`, and `plotext` join the required
  dependencies; `numpy` is pinned to `<2.4`
- Optional extras: `aimnet`, `ase`, `cu12`, `cu13`, `mace`, `pymatgen`,
  `tensorboard`, `uma`. `cu12` and `cu13` select the matching CUDA build of
  `nvalchemi-toolkit-ops`, `nvidia-physicsnemo`, `cuml`, and
  `cuequivariance-ops-torch`, so pick exactly one. `uma` conflicts with `mace`
  and with both CUDA extras and resolves into its own environment.

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
