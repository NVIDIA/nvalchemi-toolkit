# Changelog

## Unreleased

### Added

- `EnhancedSampling` runner for biased dynamics, plus the first built-in
  biases. The runner installs one internal hook on an existing `BaseDynamics`
  and owns what a bias cannot: walker identity stamping (`walker_id`,
  `thermodynamic_state_id`, `sampling_step`, `exchange_segment`,
  `sampling_epoch`), the force-step ordering, exactly-once `update()`
  delivery, and force priming. Every bias is evaluated against the same
  unmodified model output and the contributions summed once, so no bias can
  observe another's forces and the total is independent of registration
  order. Built-ins: `HarmonicUmbrellaBias` (per-window centers and stiffness
  selected by `thermodynamic_state_id`, validated symmetric
  positive-semidefinite), `UpperWall`, `LowerWall`, and
  `FlatBottomRestraint`. `AdaptivePotentialMixin` supplies the
  `update`/`commit_epoch`/state-version battery for biases whose state
  evolves during sampling; it must precede `nn.Module` in the base list, and
  raises `TypeError` otherwise rather than letting `nn.Module.state_dict`
  shadow it and drop bias history from checkpoints. `periodic_difference`
  wraps CV differences onto a circle. `warm_start()` gives approximate
  continuation from prior frames; for exact resumption see the checkpoint
  entry below.

- Synchronous replica exchange. `ReplicaExchange` and `ThermodynamicState`
  advance a ladder of states as one batch and periodically swap which walker
  holds which rung, with an even/odd pair schedule so every pair in a segment
  is disjoint and decidable simultaneously. Exchange permutes
  `thermodynamic_state_id`; coordinates never move between rows. The
  acceptance rule is inferred from the ladder rather than declared — varying
  temperatures select the Metropolis temperature rule, equal ones the
  umbrella rule — because a declared rule that disagreed with the ladder
  would break detailed balance silently. An accepted swap is indivisible:
  label, integrator target temperature, velocity rescaling, and forces move
  together, and an integrator that cannot rebind (`NVE`) is rejected at
  construction rather than sampling the state it just left. Acceptance draws
  derive from `random_seed + exchange_id`, so a restored run reproduces the
  same decisions and the checkpoint stores two integers rather than an RNG
  blob; exchange state lives under `sampling/exchange/`. The manifest records
  the ladder (mode, acceptance rule, interval, temperatures) and `restore()`
  refuses a mismatch, including exchange-versus-none in either direction —
  the ladder decides what a swap means, so restoring into a different one
  would keep the assignment and counters while silently changing the
  acceptance exponent. Per-pair acceptance
  rates are reported for ladder tuning. Asynchronous exchange and force-only
  (ABF-style) biases are rejected explicitly, as is the unimplemented
  combined temperature-plus-window rule: a bias declaring
  `state_dependent_for_exchange` is refused at construction, and the runner
  additionally probes every bias empirically at prime time by evaluating it
  under a permuted assignment, which catches a user bias that declares
  nothing. A single-window `HarmonicUmbrellaBias` now ignores
  `thermodynamic_state_id` rather than indexing it, so one shared restraint
  can run alongside a multi-rung temperature ladder.

- Metadynamics, in two flavours. `WellTemperedMetaDynamicsBias` deposits
  Gaussian hills along any differentiable CV, one per walker per deposition,
  with the well-tempered height damping
  `h_t = h_0 exp(-V(s_t) / (k_B T (gamma - 1)))` that makes the sum converge;
  `bias_factor=None` gives standard metadynamics. `free_energy()` returns
  `-(gamma / (gamma - 1)) V(s)`. Three storage policies, chosen rather than
  defaulted: `preallocated` keeps tensor shapes fixed for the whole run and
  **raises** when capacity is exhausted, because silently dropping hills would
  change the physics of a converging run with nothing to show for it; `grow`
  allocates another chunk and recompiles; `fifo` bounds memory by discarding
  the oldest hill, which is a scientific choice and not a cache policy — the
  well-tempered convergence argument no longer applies, so `free_energy()`
  refuses under it rather than returning a plausible number. Three history
  modes: `shared` (the multiple-walker scheme — `B` walkers fill a basin
  roughly `B` times faster in one batched force evaluation), `walker` (`B`
  independent runs in one batch), and `state` (per-rung history for a
  replica-exchange ladder, which declares `state_dependent_for_exchange`).
  `periods` wraps the hill difference onto a circle so a hill near a branch
  cut repels from both sides. `sigma` and `periods` are checked against the
  CV on first evaluation rather than broadcast against it: a mismatched
  length would widen the hill table and silently change the Gaussian
  exponent. `sigma` may be a scalar shared across components or one entry
  per component; `periods` must be per-component, since a `0` entry is what
  marks a component non-periodic and one value cannot carry that
  distinction. The hill table takes its width from the CV, so a scalar
  `sigma` works with a multi-component CV.

  `RMSDMetaDynamicsBias` is the xTB/CREST-style variant, whose history is a
  set of retained structures rather than CV values, and which therefore needs
  no collective variable at all. Optimal translation/rotation alignment is
  solved by the quaternion characteristic-polynomial route rather than an SVD
  Kabsch: the proper-rotation constraint is built in instead of needing a
  non-differentiable `det` correction, and only the largest eigenvalue is
  taken, which stays well-conditioned for symmetric-top and linear molecules
  where singular-vector gradients blow up. The squared RMSD is used
  throughout — `sqrt` has infinite derivative at zero, and a reference is
  visited at RMSD zero every time one is deposited. Consequences: the energy
  is invariant to rigid motion and the bias forces sum to exactly zero.
  Non-periodic systems only — a batch with a non-zero cell is rejected,
  because an atom crossing a cell face is physically unmoved but
  Cartesian-displaced by a lattice vector, which would inject a large spurious
  force. Atom correspondence is fixed, `atom_indices` selects a per-graph
  subset, warm-start references seed the history, and there is deliberately no
  `free_energy()` — it is a structure generator, not an estimator.

  Both deposit at `AFTER_STEP`, so a hill marks the configuration the walker
  reached; both bump the state version so the runner re-primes forces and the
  new hill is felt on the next step rather than one late; and neither deposits
  during `prime_forces()`.

- Exact checkpoint and restore for enhanced sampling.
  `EnhancedSampling.checkpoint()` writes a transactional Zarr store that
  extends the existing `AtomicData` layout with a `sampling/` group holding
  integrator, bias, and runner state; `restore()` reads it back and returns a
  force-primed batch that reproduces the identical trajectory. The manifest is
  written last and is the commit marker, so an interrupted write has none and
  is refused rather than half-restored. Integrity cover is total: each
  `sampling/` component is SHA-256 checksummed, and a separate
  `batch_checksum` covers `meta/`, `core/`, and `custom/` — the positions,
  velocities, pointer arrays, and walker identity that `AtomicDataZarrWriter`
  writes outside the component path. All are verified on read, and cover is
  mandatory: a manifest with a gap — a declared component lacking a checksum,
  a checksum naming no component, or no batch checksum — is rejected as
  invalid rather than read unverified. State is Zarr
  arrays and JSON attributes with **no pickle payloads** — an unsupported
  value type raises rather than falling back, so loading a checkpoint cannot
  execute code. Checkpoints are permitted
  only at a consistency-epoch boundary, the one point with no pending
  `update()` or in-flight epoch commit; the error names the next valid step.
  `checkpoint()` also drains the completed epoch's `commit_epoch()` before
  collecting state, since that normally fires lazily on the next step — so a
  shared-history bias is saved merged rather than mid-merge. The drain is
  tracked per epoch index and cannot double-count.
  `BaseDynamics` gains `state_dict()`, `load_state_dict()`,
  `redistribute_state()`, and `apply_thermodynamic_state()`, the last
  implemented for `NVTLangevin` (velocity rescaling) and `NVTNoseHoover`
  (chain masses and velocities transformed with kT, leaving the chain kinetic
  energy invariant). Model weights are never restored from a checkpoint; the
  manifest records model, dynamics, and bias classes and `restore()` refuses a
  mismatch.

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
  optional extra (`pip install 'nvalchemi-toolkit[uma]'`), which is
  declared conflicting with the `mace` and `cu12`/`cu13` extras
  (incompatible `e3nn` / `torch` pins) and resolves into its own
  environment. `from_checkpoint` forwards fairchem's `inference_settings`
  (including `"turbo"` for `torch.compile`). See the
  `examples/advanced/09_uma_nve.py` NVE/NVT/NPT walkthrough.

### Fixed

- **Zero-dimensional tensors in enhanced-sampling checkpoints** — Zarr reads a
  0-d array back as shape `(1,)`, so a component holding a scalar buffer (a
  step counter, a deposition count — the kind of state a compile-safe bias
  keeps as a tensor rather than a Python int) no longer matched the digest
  taken when it was written, and `restore()` failed the component's own
  checksum. The true shape is now recorded alongside the dtype and reapplied
  on decode; checkpoints written before this are read exactly as before.

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

- `BiasedPotentialHook`, superseded by the `nvalchemi.enhanced_sampling`
  subpackage. Its `bias_fn(batch) -> (energy, forces)` contract has no slot
  for a cell response, so a bias applied through the hook contributes no
  stress and is invisible to the NPT/NPH barostat — the cell evolves as if
  the bias were absent, with no error raised. It also requires bias forces
  to be written by hand (nothing checks they are `-dE/dr`), and composes
  several biases by sequential in-place mutation of `batch.forces` rather
  than summing them against the unmodified model output. Constructing the
  hook now emits a `DeprecationWarning`. It remains functional so existing
  code keeps working, and no removal date is set; `EnhancedSampling` (also in
  this release) covers everything it does. No adapter is provided:
  bridging a `BiasPotential` onto `bias_fn` would have to discard
  `BiasResult.stress`, reintroducing the exact failure the new API removes.

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
