# Changelog

## Unreleased

### Added

- `validate_contribution` and `aggregate_contributions` now refuse a
  contribution carrying an output no consumer can apply, and
  `APPLIED_OUTPUT_KEYS` is the single definition of that set.
  `aggregate_contributions` sums only `energy`, `forces`, `stress` and
  `virial`, so a bias returning `hessian` or `dipole` — both documented
  `ModelOutputs` keys for a full forward pass — had its contribution dropped
  between the producer and the buffer, with the run continuing as though it
  had been applied and nothing raised. The applied set is closed by design
  (each member needs a destination buffer, a reshape rule and a combination
  rule); it is now closed by enforcement too, with the error naming the
  offending key, the producer or contribution index, and both ways out —
  report it as `diagnostics/<key>`, or add the output to the framework.

  Checked in both places rather than only at the producer: the aggregation is
  public and its docstring promises that nothing is dropped silently, so it
  cannot rest that promise on a caller having validated first. Both read the
  same constant through one predicate, so they cannot drift.
  `sum_outputs` is deliberately unchanged — generic model composition may
  legitimately carry `hessian` or `dipole`, and a contribution is narrower
  than a forward pass precisely because it gets added into a buffer.

- `PairSwapHook` and `BaseDynamics.apply_per_system_params()` — pairwise swaps
  of per-system state, with the physics taken out. Strip the thermodynamics
  from replica exchange and the mechanism is: propose pairs, evaluate an
  acceptance rule, and permute per-system parameters for the pairs that pass,
  rebinding whatever integrator state travels with them. That is also basin
  hopping with swaps, population or evolutionary structure search, and any
  annealing ladder, so it lives in `nvalchemi/dynamics/hooks/swap.py` with the
  rule supplied as `accept_fn(batch, i, j) -> Bool[P]` and the parameter
  mapping as `params_fn(slots)`. `even_odd_pairs` and `apply_pair_swaps` are
  exposed alongside it.

  `apply_thermodynamic_state(state_ids, temperatures)` and
  `rescale_velocities_for_state(batch)` are replaced by one
  `apply_per_system_params(params, batch)`. Generalising the name is what makes
  the adapter available to methods that are not replica exchange; merging the
  two calls is what makes the change indivisible, since the velocity rescale
  needs the batch and a caller who forgot the second call got exactly the
  silent failure the adapter exists to prevent. An integrator now raises on a
  parameter it cannot rebind rather than ignoring it.

  `ReplicaExchange` keeps what is genuinely enhanced-sampling physics — the
  Sugita-Okamoto acceptance rules, the ladder, the bijection check, the
  counter-based acceptance RNG and the per-pair tallies — and gains
  `swap_hook()`, which configures the generic hook with them.
  `ReplicaExchangeHook` is gone from `enhanced_sampling/`. The acceptance rule
  now has one implementation reached by both callers: the live hook and
  `decide()`, which still exposes the same decision without a batch so the
  formula can be checked against hand-computed numbers.

- `nvalchemi/_checkpoint.py` — transactional, pickle-free Zarr checkpoints for
  any set of stateful objects. `save_checkpoint(path, components, batch=...)`
  and `load_checkpoint(path, components, ...)` take a mapping of name to
  `Stateful` — the `state_dict` / `load_state_dict` pair that `nn.Module`,
  `BaseDynamics`, `ReplicaExchange`, every `CheckpointableHook` and every
  adaptive bias already satisfy — so a checkpoint is a mapping of name to
  *something that knows its own state* rather than a per-workflow schema.
  Names may nest (`"biases/umbrella"`, `"hooks/epoch"`), so a store is
  inspectable group by group; `manifest` is reserved, and an empty path
  segment is refused rather than silently flattened. The manifest is written
  last and is the commit marker; every component and the batch are SHA-256
  checksummed and verified on read, and cover is mandatory rather than
  best-effort. A `compatibility` fingerprint and a `metadata` blob are stored
  verbatim, and `load_checkpoint` takes a `validate` callback invoked after
  integrity checks and before any state is applied — deciding what a
  fingerprint mismatch *means* is domain knowledge the layer does not have.

  This replaces the copy that shipped inside `enhanced_sampling/_checkpoint.py`.
  Writing transactionally and refusing a torn store is wanted by MD restart and
  NEB restart too, and a second implementation under one subpackage is what
  guarantees a third. `training/_checkpoint.py` is **not** refactored onto it:
  that layer has a different storage format — a directory of `.pt` files with
  per-component specs, indices, and model/optimizer/scheduler associations —
  and migrating it is a format change with its own compatibility story.

- `DynamicsStrategy` — a declarative recipe that configures a `BaseDynamics`
  it does not own. Construction validates the whole configuration, `build()`
  turns it into a live engine, `to_spec_dict()` serialises the knobs, and
  `run()` delegates to `BaseDynamics.run`. It mirrors `TrainingStrategy`, with
  one deliberate difference: `TrainingStrategy` owns its loop because nothing
  below it does, whereas a dynamics strategy has an engine that already owns
  one. A workflow built on top of dynamics — enhanced sampling, NEB, a
  relaxation schedule, an equation-of-state scan — differs from plain dynamics
  only in *what it configures*, so it becomes a subclass overriding
  `build_hooks()` rather than a second runner with a second stepping loop that
  every future workflow would have to choose between.

- `EnhancedSampling`, the first `DynamicsStrategy`, plus the first built-in
  biases. It contributes four hooks to the engine it builds and owns what a
  bias cannot: `WalkerIdentityHook` stamps identity and counters (`walker_id`,
  `thermodynamic_state_id`, `sampling_step`, `exchange_segment`,
  `sampling_epoch`); `BiasHook` owns the force-step ordering, exactly-once
  `update()` delivery and force priming; `EpochCommitHook` fires `commit()` on
  a `steps_per_epoch` cadence; `ReplicaExchangeHook` attempts swaps on an
  `attempt_interval` one. The last two carry their cadence as
  `Hook.frequency`, so the hook registry gates them and nothing re-implements
  "has the boundary been crossed". Every bias is evaluated against the same
  unmodified model output and the contributions summed once — which is why
  they share **one** `BiasHook` rather than one hook each — so no bias can
  observe another's forces and the total is independent of registration
  order. Built-ins: `HarmonicUmbrellaBias` (per-window centers and stiffness
  selected by `thermodynamic_state_id`, validated symmetric
  positive-semidefinite), `UpperWall`, `LowerWall`, and
  `FlatBottomRestraint`.

  A bias is an ordinary **additive potential**: a `BaseModelMixin` whose
  `forward` returns `ModelOutputs`, the same shape as `DFTD3ModelWrapper` and
  `LennardJonesModelWrapper`. There is no bias-specific protocol and no
  bias-specific result type — "produces `ModelOutputs` from a `Batch`" is what
  `BaseModelMixin` already means, and a second name for it would only be a
  second thing to keep in sync. `ConservativeBias` adds the autograd
  derivation on top of that and nothing else; the isolation it used to own
  now lives in `nvalchemi.models._utils.isolated_energy_derivatives`, where an
  NEB spring term or a hand-written wall can reach it without depending on the
  enhanced-sampling package.

  `AdaptivePotentialMixin` supplies the `StatefulHook` half for biases whose
  state evolves during sampling — `frequency`, `stage`, `read_only`, `update`,
  `commit`, and state versioning — reusing the hook vocabulary rather than
  inventing a parallel lifecycle. The context it receives, `BiasContext`,
  lives in `nvalchemi/hooks/_context.py` beside `DynamicsContext` and
  `TrainContext`, matching how training's own context is placed. The mixin
  must precede `nn.Module` in the base list, and raises `TypeError` otherwise
  rather than letting
  `nn.Module.state_dict` shadow it and drop bias history from checkpoints.
  `update` is spelled `update(ctx, stage)` rather than `__call__(ctx, stage)`
  because a bias is also an `nn.Module`, whose `__call__` is the model forward
  that `BaseDynamics` and `PipelineModelWrapper` invoke; `BiasHook` owns
  protocol compliance on its behalf, exactly as `TrainingUpdateOrchestrator`
  does for `TrainingUpdateHook`.
  `periodic_difference` wraps CV differences onto a circle. `warm_start()`
  gives approximate continuation from prior frames; for exact resumption see
  the checkpoint entry below.

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
  `state_dependent_for_exchange` is refused at construction, and the strategy
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
  The latter two require `walker_id` / `thermodynamic_state_id` on the batch
  and raise if it is missing or the wrong length, rather than defaulting to a
  single owner — that fallback filed every hill under one key and produced
  energies numerically identical to `history="shared"`, silently delivering
  the opposite of what was configured. `WalkerIdentityHook` stamps both fields
  on every step, so only a directly evaluated bias has to supply them.
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
  Non-periodic systems only — a periodic batch is rejected, because an atom
  crossing a cell face is physically unmoved but Cartesian-displaced by a
  lattice vector, which would inject a large spurious force. Periodicity is
  read from `batch.pbc` rather than from the presence of a cell, matching
  `pair_distance` and the rest of the toolkit: a bounding-box cell with `pbc`
  all-False is the non-periodic case this bias is for and is accepted, a slab
  is rejected, and a cell carrying no `pbc` flags is refused as undeclared.
  Atom correspondence is fixed, `atom_indices` selects a per-graph
  subset, warm-start references seed the history, and there is deliberately no
  `free_energy()` — it is a structure generator, not an estimator.

  Both deposit at `AFTER_STEP`, so a hill marks the configuration the walker
  reached; both bump the state version so `BiasHook` re-primes forces and the
  new hill is felt on the next step rather than one late; and neither deposits
  during `prime_forces()`.

- `AdaptiveBiasingForce` — measures the mean force along a pair distance and
  applies its negative, so a well-sampled bin leaves no residual force along
  the CV and the walker diffuses across it. What it accumulates already *is*
  the free-energy gradient, so `free_energy()` integrates it directly with no
  hills to deconvolve and no histogram to reweight.

  The estimator is
  `dA/dr = <-(F_j - F_i).u / 2 - 2 kB T / r>`, and the second term — the
  metric correction — is not a refinement. A naive Cartesian projection gives
  the mean force in the *constrained* ensemble; the unconstrained free energy
  differs by the Jacobian of the coordinate change, which for a distance in
  three dimensions is `-2 kB T / r`. Omitting it produces a smoothly wrong
  answer rather than noise: two non-interacting particles come out with a flat
  PMF when the true one is the purely entropic `-2 kB T ln r`. This is why the
  class takes an atom **pair** rather than a general `cv` callable, unlike
  every other bias in the subpackage — the correction belongs to this
  coordinate, and accepting an arbitrary CV would apply a distance-shaped
  correction to something that is not a distance.

  At or below `min_samples` a bin applies nothing, and between `min_samples`
  and `full_samples` the applied fraction ramps linearly to one, so no bin
  switches on with a jump. `full_samples == min_samples` asks for a step
  instead — nothing until the threshold, full force at it — which is the
  classic hard-threshold form and what `min_samples=0` gives by default, so it
  is handled as its own case rather than falling through the linear formula
  and arriving a sample late; `max_force` optionally caps what a bin visited once
  at an awkward geometry can do. Its `stage` is `AFTER_COMPUTE`, where
  `batch.forces` still holds the unbiased physical force — an estimator shown
  its own output converges to whatever it had already decided. An update
  landing in a bin still below its threshold does not bump the state version,
  since the applied force has not changed.

  `forward()` returns `forces` and no `energy`: there is no potential to
  report, which is what non-conservative means here. It is still a
  `BaseModelMixin` declaring `outputs={"forces"}` — non-conservative does not
  mean non-model, which is what lets `BiasHook` treat every bias alike. `supplies_exchange_energy`
  is `False`, so `ReplicaExchange` refuses the combination at construction
  rather than dropping the bias from the acceptance exponent. `mean_force()`
  and `free_energy()` report unvisited bins as `nan` rather than zero, and
  `free_energy()` raises on an interior gap because integrating across a hole
  would leave every value beyond it wrong by an unknown constant. Per-step
  diagnostics are `cv`, `bin`, `applied_gradient`, `samples`, `ramp`, and
  `in_range`; a walker outside `cv_range` reports `bin = -1` and zero for
  every per-bin quantity, so the diagnostics agree with the zero force it
  receives rather than reporting the nearest edge bin's statistics.

- Adaptive biases now record a configuration fingerprint in their
  `state_dict()` and reject a mismatched restore. The manifest records each
  bias's class, but a class name says nothing about the settings its saved
  state depends on: an ABF histogram restored under a different `cv_range` is
  shape-compatible and silently relabels every bin, `bias_factor` changes the
  ratio `free_energy()` applies to hills deposited under a different gamma,
  and `k_push`/`alpha` change what stored RMSD references do. Worse for
  settings held as buffers — `sigma`, `atom_indices` —
  `nn.Module.load_state_dict` *overwrote* the caller's value with the
  checkpoint's rather than leaving it unvalidated. `AdaptivePotentialMixin`
  gains `config_fingerprint()` (empty by default, so other biases are
  unaffected) and checks it before delegating, so the guard covers a bias
  restored directly as well as through the strategy. Capacity is deliberately
  excluded, since `storage="grow"` legitimately reaches a size the
  constructor never had.

- `isolated_energy_derivatives`, `validate_contribution` and
  `aggregate_contributions` in `nvalchemi/models/_utils.py`, alongside the
  existing `autograd_*` helpers and `sum_outputs`. The last is the strict
  counterpart to `sum_outputs`: the same element-wise sum, but a key collision
  that would silently drop a producer's value is an error rather than
  last-write-wins, `stress`/`virial` may not be mixed, and `state_version` is
  dropped because an aggregate over several producers has no single revision.
  All four contribution helpers now sit together rather than one of them
  living in `enhanced_sampling/`.
  The first evaluates an energy function against a *detached view* of a live
  `Batch` and returns detached forces and tensile-positive Cauchy stress,
  restoring every field it substituted in a `finally` block so the caller's
  batch never carries a `grad_fn` afterwards. That isolation is what anything
  differentiating against a batch it does not own needs — an enhanced-sampling
  bias, an NEB spring term, a hand-written wall potential — so it is a shared
  helper rather than a method on one class. The second checks a `ModelOutputs`
  mapping meant as an additive *contribution*: detachment, shapes,
  stress/virial mutual exclusion, batch-size consistency, and finiteness, with
  a `source=` label naming the producer in the error.

- `ModelOutputs` gains two general conventions beyond the physical keys.
  `state_version` (`[B]`, integer) identifies which revision of a producer's
  internal state generated the rest of the mapping, for components whose state
  evolves during a run. Keys under `diagnostics/` are arbitrary reported
  tensors with no shape contract, never summed and never applied — for
  quantities a consumer displays or records rather than acts on. Both are
  documented on the type alias and checked by `validate_contribution`.

- `StatefulHook` in `nvalchemi/hooks/_protocol.py`, composing the existing
  `Hook` and `CheckpointableHook` and adding only `read_only` and `commit()`.
  `Hook` already says "every N steps, at this stage" and `CheckpointableHook`
  already carries `state_dict`/`load_state_dict`; what neither said is whether
  a dispatch may *change* anything and when accumulated changes become visible
  to other workers. The pattern recurs well beyond any one workflow — an
  adaptive bias depositing hills, NEB promoting its climbing image, an
  adaptive thermostat retuning its coupling, a neighbour list widening its
  skin — and without a shared protocol each invents its own vocabulary for the
  same three ideas.

- `pair_displacement` — the vector form of `pair_distance`, exposed for
  methods that work with the CV gradient rather than its value.
  `pair_distance` is now its norm, so the two cannot drift apart in their
  validation, device handling, or minimum-image convention.

- Exact checkpoint and restore for enhanced sampling, through the shared
  checkpoint layer above. `EnhancedSampling.checkpoint()` names every object
  whose state matters — the engine, each hook, each bias, the ladder — and
  `restore()` reads it back and returns a force-primed batch that reproduces
  the identical trajectory. `WalkerIdentityHook`, `EpochCommitHook` and
  `ReplicaExchangeHook` gained `state_dict`/`load_state_dict` so each owns its
  own cursor rather than being collected into one opaque `runner` blob:
  walker-id allocation, last committed epoch, last attempted segment.
  `"dynamics"` is the one component applied by hand, because the integrator's
  per-system arrays have to be allocated against the restored batch before
  they can be written into. Checkpoints are permitted
  only at a consistency-epoch boundary, the one point with no pending
  `update()` or in-flight epoch commit; the error names the next valid step.
  `checkpoint()` also drains the completed epoch's `commit()` before
  collecting state, since that normally fires lazily on the next step — so a
  shared-history bias is saved merged rather than mid-merge. The drain is
  tracked per epoch index and cannot double-count.
  `BaseDynamics` gains `state_dict()`, `load_state_dict()`,
  `redistribute_state()`, and `apply_per_system_params()`, the last
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

- **A save killed mid-replacement hid the checkpoint** — `_move_into_place`
  renames the old store aside before renaming the new one in, and between
  those two renames nothing is at the documented path. An exception there is
  caught and undone; a `SIGKILL` or a power loss cannot be. The previous,
  perfectly good checkpoint was left under `<name>.superseded-<pid>` and
  `load_checkpoint(path)` raised `FileNotFoundError`, so a run could not
  restart from the path it was told to use even though its restart point had
  survived intact beside it.

  `load_checkpoint` now recovers it: when nothing is at *path* and exactly one
  sibling `.superseded-*` store carries a committed manifest, it is moved back
  and loaded, with a warning saying it is the generation before the
  interrupted save. It never promotes a store without a manifest, never acts
  when *path* exists — so it cannot shadow a newer checkpoint — and refuses
  rather than guesses when several superseded stores are present. Closing the
  window itself would need an atomic directory exchange, which POSIX does not
  offer.

- **Walkers sampled the constructor temperature, not their rung** — a
  thermodynamic state is a temperature, but only `PairSwapHook` ever told the
  integrator so. `WalkerIdentityHook` wrote the assignment onto the batch and
  left the thermostat alone, so the two moments where the assignment changes
  without a swap sampled the wrong ensemble:

  - **The first stamp.** The documented recipe builds the engine with one
    scalar `temperature` and lets the ladder say the rest. On a four-rung
    ladder from 300 K, every walker targeted 300 K from step zero while its
    label claimed 300/360/432/518 K; three of the four sampled the wrong
    ensemble until a swap happened to touch them. Measured over 20 steps with
    exchange disabled, per-walker kinetic energy was `[0.215, 0.138, 0.148,
    0.140]` — flat, where the ladder calls for `[0.215, 0.154, 0.191, 0.204]`.
  - **A refill.** `_sync_state_to_batch` appends *default* integrator state
    for an admitted walker, so a replacement given a vacant rung by
    `_assign_state_ids` ran at the constructor temperature until an accepted
    swap corrected it.

  The hook now rebinds through `apply_per_system_params` whenever it writes an
  assignment, forcing the integrator's lazy state allocation so the binding
  lands before the first step rather than after it. **This changes the
  trajectory of every existing replica-exchange run** that did not pass a
  per-walker temperature tensor to the engine — the previous trajectories were
  sampling the wrong ensembles, so the change is the fix, not a regression.

- **A FIFO reference ring smaller than one deposition dropped references
  silently** — `RMSDMetaDynamicsBias` and `WellTemperedMetaDynamicsBias` each
  carried their own copy of the same ring-buffer subsystem, and the copies had
  drifted: the guard refusing a `"fifo"` ring narrower than the walker count
  was added to the hill table only. With `max_references=2` and four walkers,
  one deposition allocated slots `(0, 1, 0, 1)`, so walkers 2 and 3 overwrote
  walkers 0 and 1 *within the same deposition*. That is not discarding the
  oldest reference, which is what `storage='fifo'` means; `references_written`
  counted structures the table never held, and half the walkers were pushed
  away from geometries they had never actually visited.

  Both biases now share one implementation, `DepositHistoryMixin` in
  `nvalchemi/enhanced_sampling/_history.py`, which owns `HISTORY_MODES`,
  `STORAGE_POLICIES`, option validation, `capacity`, `_grow`, `_next_slots`
  and `_owner_key`. Buffer names stay with each bias — `hill_owner` and
  `reference_owner` are the names in every checkpoint written so far — and the
  mixin is told which attributes to use. No behaviour changes for the hill
  table.

- **A missing energy buffer accepted every swap** — temperature acceptance
  substituted zeros when the batch carried no `energy`, where `decide()`
  raises for the same omission. Zero is not a neutral stand-in:
  `log a = (beta_i - beta_j)(U_i - U_j)` is zero for equal energies, so
  *every* swap was accepted — a random relabelling with no energetic
  criterion — and the per-pair acceptance rate a ladder is tuned on read
  1.00, which looks like rungs that are too close rather than a missing
  field. It is reachable through the documented bias-free REMD recipe, since
  priming requires only `forces` and a batch without `energy` is otherwise
  runnable. It now raises, and names the buffer to allocate. A non-finite
  energy is refused too, in the one place both entry points meet, since that
  exponent is `nan` and `nan` compares false — the silent reject rather than
  the silent accept.

- **A non-finite bias energy silently rejected every swap** — `bias_energy`,
  the evaluation umbrella acceptance needs, did not check what the biases
  returned. It is the only place a bias is scored under an assignment the run
  is *not* in, which is exactly where a window bias overflows: finite with
  every walker at home, infinite the moment one is scored against another's
  window. The force path therefore never sees it. Nor does anything raise —
  the acceptance exponent becomes `nan`, `nan` compares false, and the swap
  is rejected, so the run reads as a ladder with poor overlap. Contributions
  are now checked here too, naming the bias and the fact that the assignment
  was a proposed one. The `isfinite` sync this costs is per exchange attempt,
  not per step, so the reason `ConservativeBias.forward` skips the check does
  not apply.

- **A FIFO hill ring smaller than one deposition overwrote itself** — one
  deposition writes one hill per walker, so with more walkers than
  `max_hills` the ring indices repeat *within* a single call —
  `(0, 1, 2, 0, 1)` for five walkers and three slots. The later walkers
  overwrote the earlier ones at the same instant, which is not what
  `storage="fifo"` promises: it discards the *oldest* hill, and here the
  hills lost were the same age as the ones kept. `hills_written` counted all
  five, so the ring reported history it had never stored.
  `storage="preallocated"` already raised in this situation and
  `storage="grow"` already resized; only the ring had no check. It now
  refuses a capacity below the walker count, and still accepts a deposition
  that exactly fills the ring.

- **A non-positive or infinite temperature corrupted the integrator** —
  `apply_per_system_params` is a public rebinding API that a custom swap or
  an annealing schedule supplies values to, and it checked only that the
  parameter *names* were ones it could rebind, never the values. None of the
  bad ones fail on their own: a negative target makes the velocity rescale
  `sqrt(T_new / T_old)` imaginary, so velocities come back `nan`; zero scales
  a Nosé-Hoover chain's masses `Q ∝ kT` to zero and `eta_dot ∝ 1/sqrt(kT)` to
  infinity; infinity reaches velocities and chain masses directly. The run
  continued, and surfaced as `nan` coordinates somewhere else entirely.

  Worse for Nosé-Hoover, the state was *partially* mutated — target copied,
  masses scaled — before the first `nan` appeared, so a caller catching a
  later error was already left with a corrupt thermostat, which also broke
  the "a refused rebinding changes nothing" guarantee `PairSwapHook` builds
  its atomicity on.

  Temperature must now be positive and finite, checked before anything is
  touched. The check and the unknown-key check both moved to one
  `BaseDynamics._validated_temperature`, since duplicated validation across
  two integrators is validation that drifts.

- **Overlapping pairs corrupted the assignment** — `pairing` is an extension
  point, so what it returns is input, and nothing checked it. Two pairs
  sharing a slot write over each other: `[(0,1), (1,2)]` on `[0,1,2]` yields
  `[1,0,1]`, duplicating one label and losing another. The run then rebound
  parameters from that — two walkers on the same rung, one rung held by
  nobody — and only reported it on the *next* segment, after a round had
  already been taken at temperatures nobody asked for.

  The schedule is now validated before use, with the error naming the
  pairing: a slot outside the ladder (which `row_of_slot` would answer with a
  bare `KeyError`) and a slot in more than one pair are both refused.
  `apply_pair_swaps` checks the same property on the rows it is about to
  write, since it is public and its result is supposed to be a permutation —
  on the *accepted* pairs only, so a schedule whose overlap acceptance
  happens to resolve is still allowed.

- **A refused rebinding left the labels swapped** — `PairSwapHook` wrote the
  new slot assignment onto the batch and *then* asked the integrator to
  rebind its parameters, so a `params_fn` naming something the integrator
  cannot rebind left the batch saying a walker had moved rung while the
  integrator still targeted the old one — the state the assignment says it
  has left, which is precisely the indivisibility the hook exists to provide.
  The segment was also marked attempted before the application ran, so the
  swap could not be retried.

  Indivisibility is now enforced by ordering rather than rollback: the
  parameters are built and the engine given its chance to refuse before
  anything is written, the label write itself cannot fail, and `on_swap` runs
  last because it repairs quantities derived from a swap that has by then
  definitely happened. `attempted_segment` advances only on success, with a
  re-entrancy guard in place of the old mark-first idempotence.

- **Saving a checkpoint over an existing one** — the store was written in
  place, with two consequences. Saving twice to the same path *failed*
  outright once any component held a tensor, because an array cannot be
  created where one already exists — and checkpointing every epoch to one
  path is the ordinary workflow. And where the write did proceed, an
  interruption before the new manifest landed left the **old** manifest
  attesting to component data that had already been replaced, so it failed
  its own checksum: an interrupted save destroyed a restart point that was
  valid a moment earlier, which is the opposite of what writing the manifest
  last is for.

  The store is now built beside the destination and moved into place only
  once complete, via two renames so the old store is removed only after the
  new one has landed. A failure at any point leaves either the previous
  checkpoint or the new one, never a mixture, and no staging directory
  behind. Saving over an existing checkpoint replaces it rather than merging
  into it, so a stale row from a previous generation cannot survive into the
  restored batch.

- **A reused strategy evaluated the wrong potential** — `DynamicsStrategy`
  caches its engine so consecutive `run()` calls continue one trajectory, but
  `dynamics(model)` returned that cache without looking at the argument. A
  strategy reused with a second potential therefore ran
  `run(batch, other_model)` against the *first* one and produced a trajectory
  for a model nobody asked for. It now refuses a changed model and names both
  ways forward: `build(model)` for an independent engine, or a second
  strategy. `build()` is unaffected and still returns a fresh engine every
  call.

- **`to_spec_dict()` did not produce JSON** — it copied `engine_kwargs`
  verbatim, and `NVTLangevin` annotates `temperature` as
  `float | torch.Tensor`, so `json.dumps(strategy.to_spec_dict())` failed for
  a configuration the engine itself accepts. Values are now converted:
  tensors to nested lists, dtypes, devices and paths to their string form,
  containers element-wise. A value with no JSON form raises and names the key
  rather than being coerced to a `repr` nothing can read back.

- **Replacement walkers reused graduated identities** — `walker_id` was
  registered as a bookkeeping key with an `arange(n)` factory, so after
  `refill_check` dropped graduated graphs and appended replacements, each
  replacement received the id of whichever walker had just graduated out of
  that slot. With `WellTemperedMetaDynamicsBias(history="walker")` the new
  configuration then inherited the departed walker's hills, and
  `WalkerIdentityHook` left the values alone because the field was present.
  Measured over three refills of a three-walker batch, ids `1` and `2` were
  each handed to four different physical walkers.

  `walker_id` now registers a `-1` sentinel, the way `system_id` already did,
  and the stamp allocates a fresh identity for every sentinel row while
  leaving assigned rows untouched. The counter also clears past any ids a
  caller supplied, so a batch arriving with its own identities does not
  collide with the first refill.

  `thermodynamic_state_id` had the same defect with a sharper consequence: it
  registered a `zeros` factory, so every replacement claimed rung 0 while the
  rungs the graduates vacated were left held by nobody — not a ladder replica
  exchange can pair on. It is now a `-1` sentinel too, and a sentinel row
  takes a **vacant** rung, so the replacement entering a slot inherits the
  rung that slot was holding and the bijection survives the refill. A refill
  that would add more walkers than the ladder has rungs is refused by name,
  and the assignment is re-validated after a refill rather than only on
  arrival — a change of batch membership is the one event that can break a
  bijection that was valid a step earlier.

- **A bias contribution could broadcast onto the whole batch** —
  `validate_contribution` checked that `forces` had shape `[?, 3]` but never
  compared the row count with the batch, so a bias returning `[1, 3]` passed
  and `batch.forces.add_` then applied that one vector to every atom —
  a force the bias never computed, with nothing raised. The same held for
  `energy` and `stress` across a multi-graph batch. It now takes optional
  `num_atoms` / `num_graphs`, which `BiasHook` supplies from the live batch,
  and `_apply` reshapes `forces` the way it already reshaped `energy` and
  `stress`, so the broadcast is no longer expressible at the addition.

  Checked per contribution rather than on the aggregate, because summing
  absorbs it: `[4, 3] + [1, 3]` is `[4, 3]`, so a bad bias standing beside a
  correct one produces a correctly shaped total carrying wrong forces, and
  the producer can no longer be named. The error names the bias.

- **An order-dependent RMSD test, and the global it was blamed on** —
  `TestSquaredRMSD::test_identical_structures_give_zero` asserted that the
  squared RMSD of a structure with itself was below `1e-18`. For a float64
  computation at that scale one ulp is `7.9e-16`, so the threshold was about
  1/750th of the representable resolution: it was asserting that the
  cancellation `(g_x + g_y - 2 lambda_max)` rounded to *exactly* zero, not
  that it was correct. Which side of zero the residue lands on depends on the
  LAPACK path `eigvalsh` takes, which varies with BLAS threading and warm-up,
  so the test failed or passed according to what had run before it. The bound
  is now derived from the scale of the cancellation (`64 eps x` mean square
  radius) and still catches a `1e-9` relative error in the formula with five
  orders of magnitude to spare.

  Separately, `test/models/test_pipeline.py` set
  `torch._dynamo.config.suppress_errors = True` by assignment in two tests,
  leaking it into every test that ran afterwards in the same session. Both
  now use `torch._dynamo.config.patch(...)`, which restores it.

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
  bridging a bias onto `bias_fn` would have to discard its `stress`,
  reintroducing the exact failure the new API removes.

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
