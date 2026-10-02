# Enhanced Sampling

Molecular dynamics follows the natural motion of atoms, which means it spends
almost all of its time in free-energy minima. Barrier crossings — diffusion
events, reactions, nucleation, conformational change — are rare on MD
timescales. Enhanced sampling adds a bias potential that pushes the system
into regions it would not visit on its own, so that a fixed budget of model
evaluations buys more of the physics you actually care about.

`nvalchemi.enhanced_sampling` provides the bias abstractions, a set of
built-in biases, and the `EnhancedSampling` strategy that configures a
`BaseDynamics` engine to run them.

```{contents}
:local:
:depth: 2
```

## Quick start

```python
import torch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.enhanced_sampling import (
    EnhancedSampling, HarmonicUmbrellaBias, pair_distance,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

pair = torch.tensor([0, 5], device=device)
umbrella = HarmonicUmbrellaBias(
    cv=lambda batch: pair_distance(batch, pair),
    centers=torch.tensor([[2.0], [2.5], [3.0]]),   # three windows
    stiffness=10.0,                                 # eV/A^2
    name="umbrella",
)

sampling = EnhancedSampling(
    engine=NVTLangevin,                             # the class, not an instance
    engine_kwargs={"dt": 0.5, "temperature": 300.0, "friction": 0.05},
    biases={"umbrella": umbrella},
)

# One window per graph. batch already carries forces/energy buffers — see
# "Batch requirements" below. `model` is any BaseModelMixin.
batch["thermodynamic_state_id"] = torch.tensor([0, 1, 2], device=device)
batch = sampling.run(batch, model, n_steps=10_000)
```

Every window is a row of one batch, so all three are advanced by a single
batched force evaluation per step rather than three separate simulations.

`EnhancedSampling` is a {class}`~nvalchemi.dynamics.DynamicsStrategy`: it holds
a *recipe* for the engine — the class and its constructor arguments — rather
than a live engine, and `run()` builds one on first use and drives it.
`sampling.dynamics(model)` returns that engine if you need it directly; it is
cached, so consecutive `run()` calls continue one trajectory.

## Collective variables

A CV is **any differentiable callable** `cv(batch) -> Tensor[B, D]`. There is
no base class, no registration, and nothing to subclass:

```python
pair = torch.tensor([0, 5], device=device)

def bond(batch):
    return pair_distance(batch, pair)
```

`atom_indices` is moved to the batch's device for you, so a CV closure built
before the batch reaches the GPU still works — but hoisting the tensor out of
the closure and placing it explicitly avoids reallocating it on every call.
The same applies to a bias: `ConservativeBias` moves its buffers to the
batch's device on first evaluation, so `HarmonicUmbrellaBias(...)` built on
CPU evaluates correctly against a CUDA batch without an explicit `.to()`.

`pair_distance` is the built-in geometric CV. It handles non-periodic systems
and the minimum-image convention for **Minkowski-reduced** cells.

:::{warning}
`pair_distance` is not a general triclinic MIC. The 27-image search it uses is
correct only for reduced cells; it raises `ValueError` in eager mode when the
cell violates the reduction condition. Under `torch.compile` that check is
skipped and supplying a reduced cell is the caller's responsibility — pre-reduce
with `atoms.get_cell().niggli_reduce()` or equivalent.
:::

For a CV that lives on a circle (a dihedral, say), use `periodic_difference`
so a restraint at `+3.0 rad` does not pull a configuration at `-3.0 rad` the
long way round:

```python
from nvalchemi.enhanced_sampling import periodic_difference

delta = periodic_difference(values, centers, periods=torch.tensor([2 * math.pi]))
```

## Built-in biases

| Bias | Energy | Use for |
|------|--------|---------|
| `HarmonicUmbrellaBias` | `0.5 * delta^T K delta` | Umbrella sampling, restrained MD |
| `UpperWall` | `(k/p) * max(s - s0, 0)^p` | Stop a CV rising past a bound |
| `LowerWall` | `(k/p) * max(s0 - s, 0)^p` | Stop a CV falling below a bound |
| `FlatBottomRestraint` | both of the above | Confine a CV to an interval |
| `WellTemperedMetaDynamicsBias` | `sum_i h_i exp(-(s - c_i)^2 / 2 sigma^2)` | Free energy along a CV you chose |
| `RMSDMetaDynamicsBias` | `sum_r k_push exp(-alpha RMSD(x, x_r)^2)` | Structure search with no CV at all |
| `AdaptiveBiasingForce` | *(no energy — force only)* | Free-energy profile along a pair distance |

The first four are static: their energy depends only on the current
configuration. The last three are **history-dependent** — they accumulate
state as sampling proceeds, and so mix in `AdaptivePotentialMixin`.
`AdaptiveBiasingForce` is further the only one that is **non-conservative**:
it applies a measured force that is not the gradient of anything it holds.

Walls contribute **exactly zero** energy and force inside the allowed region,
and their default quadratic form has zero force at the boundary, so switching
one on does not deliver an impulse.

### Per-window parameters

`HarmonicUmbrellaBias` accepts `centers` of shape `[D]` (shared) or `[S, D]`
(one row per thermodynamic state). Each graph selects its row via
`batch.thermodynamic_state_id`; without that field every graph uses state `0`.
`stiffness` accepts a scalar, `[D]`, `[D, D]`, or `[S, D, D]`, and is validated
to be symmetric positive-semidefinite — a negative eigenvalue would turn the
restraint into an unbounded repulsion.

## Metadynamics

Umbrella sampling needs you to name the windows in advance. Metadynamics does
not: it deposits a Gaussian hill wherever the system currently is, so the
accumulated bias pushes it towards wherever it has not yet been.

### Well-tempered metadynamics

`WellTemperedMetaDynamicsBias` deposits one hill per walker at its current CV
value, every `frequency` steps. In the well-tempered scheme each hill
is damped by the bias already standing at that point,

```text
h_t = h_0 * exp(-V(s_t) / (k_B T (gamma - 1)))
```

so the sum converges rather than filling forever, and the converged bias is a
free-energy estimate:

```python
from nvalchemi.enhanced_sampling import WellTemperedMetaDynamicsBias

metad = WellTemperedMetaDynamicsBias(
    cv=bond_distance,
    height=0.005,          # h_0, eV
    sigma=0.25,            # hill width, CV units
    temperature=300.0,
    bias_factor=8.0,       # gamma; None gives standard metadynamics
    frequency=500,
    storage="preallocated",
    max_hills=2000,
    name="metad",
)
...
profile = metad.free_energy(grid)   # -(gamma / (gamma - 1)) * V(s)
```

`bias_factor=None` is the `gamma -> infinity` limit: every hill keeps height
`h_0` and `F(s) = -V(s)`. That is standard metadynamics, which does not
converge — the bias keeps growing and oscillates about the true profile.

Pass `periods` for a CV that lives on a circle. A hill at `+3.10 rad` must
repel a configuration at `-3.10 rad`, which is `0.083 rad` away the short way
and `6.20 rad` the long way; without a period the bias sees the second number
and does nothing.

### Storage policies

The hill table has to be bounded somehow, and the three ways of bounding it
are not interchangeable.

| `storage` | On reaching `max_hills` | Compile | `free_energy()` |
|-----------|-------------------------|---------|-----------------|
| `preallocated` | **raises** | Shapes fixed for the whole run | Valid |
| `grow` | Allocates another chunk | Recompiles on each resize | Valid |
| `fifo` | Overwrites the oldest hill | Shapes fixed | **Raises** |

`preallocated` raises rather than evicting because silently dropping hills
would change the physics of a converging run with nothing to show for it. The
error names the ways out.

`grow` carries a limit that is easy to miss. Under torch's default of static
parameter shapes, each resize changes the hill-tensor shape and a compiled
`energy()` retraces — and Dynamo caps retraces per code object at
`torch._dynamo.config.recompile_limit`, 8 by default. The next growth
therefore **hard-fails mid-run**, once the trajectory is already underway.

Three ways out, in rough order of preference:

1. Use `preallocated`, which holds a single trace for the whole run.
2. Size `max_hills` so the number of growths stays under the limit.
3. Set `force_parameter_static_shapes = False` on `torch._dynamo.config`,
   which makes Dynamo trace the hill-table dimension symbolically so growth
   stops triggering a retrace at all.

Both settings are process-global, and other code in this toolkit changes them:
`DistributedModel` raises the limit to 64 *and* disables static parameter
shapes, so a domain-decomposed run will not see this failure. Read the live
config rather than assuming the defaults.

`fifo` is not merely a cache policy: once hills are
discarded the accumulated bias is no longer the integral of everything
deposited, the well-tempered convergence argument no longer applies, and
`free_energy()` refuses rather than returning a number that looks fine.

### Multi-walker history

`history` decides which hills a given walker feels.

- `"shared"` (default) — every walker feels every hill. This is the
  multiple-walker scheme: `B` walkers fill a basin roughly `B` times faster,
  at the cost of one batched force evaluation per step rather than `B`
  separate runs.
- `"walker"` — each walker feels only its own hills, so one batch runs `B`
  genuinely independent metadynamics simulations.
- `"state"` — hills belong to the `thermodynamic_state_id` that deposited
  them, which is what a replica-exchange ladder needs.

`"state"` sets `state_dependent_for_exchange = True`, so combining it with a
temperature ladder is rejected rather than run under an acceptance rule that
does not cover it; see [Acceptance](#acceptance).

`"state"` and `"walker"` read `thermodynamic_state_id` and `walker_id` off the
batch, and **raise** if the field is absent or the wrong length. The
identity hook
stamps both on every step, so this only affects a bias you evaluate directly.
The alternative — defaulting a missing field to zero — would file every hill
under one key and silently collapse the per-owner histories into a single
shared one, which is the opposite of what was asked for and produces energies
numerically identical to `history="shared"`.

### xTB-style RMSD metadynamics

`RMSDMetaDynamicsBias` drops the collective variable entirely. Its history is
a set of retained *structures*, and it pushes away from all of them at once:

```text
V(x, t) = sum_r f_r(t) * k_push * exp(-alpha * RMSD(x, x_r)^2)
```

RMSD is measured after optimal translation and rotation, so the bias is
invariant to rigid motion and its forces sum to exactly zero. This is the
scheme xTB/CREST uses for conformer and isomer searching, and it is the right
tool when you cannot say in advance which coordinate matters.

```python
from nvalchemi.enhanced_sampling import RMSDMetaDynamicsBias

explorer = RMSDMetaDynamicsBias(
    k_push=0.08,                        # eV
    alpha=10.0,                         # A^-2
    frequency=500,
    max_references=64,                  # FIFO by default
    atom_indices=torch.tensor([0, 4, 7]),   # heavy atoms only
    name="explorer",
)
```

Choosing `alpha` is the main decision, and it must match the RMSD scale the
system actually explores: the kernel only has usable gradient where
`alpha * RMSD^2` is of order one. Set it far too small and
`exp(-alpha * RMSD^2)` sits at ~1 for every structure, leaving the bias nearly
constant and nearly forceless. A rigid cluster moving 0.3 A wants `alpha`
around 10; a floppy molecule sampling 1 A wants `alpha` around 1.

Three constraints are worth knowing before reaching for it:

- **Non-periodic systems only.** Cartesian RMSD against a stored reference is
  not defined under periodic boundary conditions — an atom crossing a cell face
  is physically unmoved but Cartesian-displaced by a lattice vector, which would
  inject a large spurious force. Bias a periodic-aware CV with
  `WellTemperedMetaDynamicsBias` instead.

  Periodicity is read from `batch.pbc`, not from the presence of a cell. A cell
  is a box; only the flags say whether atoms wrap through its faces. So a
  molecular batch carrying a **bounding box** with `pbc` all-False is accepted
  — the common case for a boxed or solvated molecule — while a slab
  (`pbc=[True, True, False]`) is rejected, since wrapping along any one axis is
  enough to break the metric. A non-zero cell with no `pbc` flags at all is
  refused as undeclared rather than assumed harmless.
- **Fixed atom correspondence.** Atom `i` is always compared with atom `i` of
  the reference; there is no permutation search, so two structures identical
  up to relabelling of equivalent atoms count as distinct.
- **No free energy.** This is a structure generator, not an estimator, and
  there is deliberately no `free_energy()` method. What it produces is a set
  of structures worth optimising or re-scoring at a higher level of theory.

`atom_indices` are **per-graph local** indices. Restricting to heavy atoms is
the usual choice: methyl hydrogens spinning freely generate RMSD that says
nothing about the conformer.

### Deposition timing

Both biases deposit at `AFTER_STEP`, so a hill marks the configuration the
walker actually reached rather than the one it started from. A deposition
bumps the bias state version, and the bias hook re-primes forces in response, so
a new hill is felt on the very next step rather than one step late.

Neither deposits during `prime_forces()`: priming evaluates forces, it does
not advance the trajectory, and depositing there would double-count the
starting configuration.

## Adaptive biasing force

Umbrella sampling restrains, metadynamics fills. ABF does neither: it
*measures* the mean force along the CV in each bin and applies its negative,
so once a bin is well sampled the residual force along the CV averages to
zero and the walker diffuses across the coordinate.

The accumulated quantity already **is** the free-energy gradient, so there is
nothing to deconvolve or reweight at the end — `free_energy()` integrates it.

```python
from nvalchemi.enhanced_sampling import AdaptiveBiasingForce

abf = AdaptiveBiasingForce(
    atom_indices=torch.tensor([0, 1]),
    temperature=300.0,          # must match the thermostat
    cv_range=(2.0, 6.0),        # angstrom
    n_bins=40,
    min_samples=200,
    full_samples=400,
    name="abf",
)
...
profile = abf.free_energy()      # PMF at bin_centers, in eV
```

### The metric correction

For a pair distance the estimator is

```text
dA/dr = < -(F_j - F_i).u / 2  -  2 kB T / r >
```

That second term is not a refinement. Projecting Cartesian forces onto the CV
gradient and averaging gives the mean force in the *constrained* ensemble; the
free energy of the unconstrained one differs by the Jacobian of the coordinate
change. For a distance in three dimensions the number of configurations at
separation `r` grows as the sphere surface `4 pi r^2`, contributing
`-2 kB T / r`.

Omitting it does not add noise — it produces a smoothly wrong answer. Two
non-interacting particles have zero Cartesian force, so a naive projection
reports a **flat** PMF when the true one is `-2 kB T ln r`, a purely entropic
profile that drives the pair apart.

This is why `AdaptiveBiasingForce` takes an atom **pair** rather than a
general `cv` callable, unlike every other bias here. The correction is
specific to this coordinate, and accepting an arbitrary CV would mean applying
a distance-shaped correction to something that is not a distance. Other CVs
need their own correction and are P1.

### Sample threshold and ramp

A mean force estimated from a handful of samples is noise, and applying it
would drive the walker on the strength of that noise. At or below
`min_samples` a bin applies nothing; between `min_samples` and `full_samples`
the applied fraction ramps linearly to one, so no bin switches on with a jump.

Setting `full_samples == min_samples` asks for a step instead — nothing until
the threshold, full force at it. That is the classic hard-threshold form, and
it is what `min_samples=0` gives by default, so it is supported rather than
rejected. A bin with no samples reports a ramp of zero whatever the thresholds
are, since it has no estimate to apply.

`max_force` optionally caps `|dA/dr|`, bounding what a bin visited once at an
awkward geometry can do while its average settles.

### Observation ordering

`stage` is `AFTER_COMPUTE`, where `batch.forces` still holds the
**unbiased** physical force. This is load-bearing: an estimator shown its own
output converges to whatever it had already decided, and the resulting profile
looks perfectly smooth. The bias hook captures the frame before applying any bias
contribution, so this holds even with several biases registered.

For the same reason, an `update()` that only lands in a bin still below its
threshold does **not** bump the state version — the applied force has not
changed, so re-priming would be pure cost.

### No energy, and therefore no replica exchange

`forward()` returns `forces` and no `energy`. There is genuinely no
potential to report, which is what makes ABF non-conservative. The Metropolis
acceptance rule needs each bias's energy evaluated under both states being
swapped, so `supplies_exchange_energy` is `False` and `ReplicaExchange`
refuses the combination at construction rather than dropping the bias from the
exponent and breaking detailed balance silently.

### Reading the profile

`mean_force()` returns the per-bin estimate and `free_energy()` its integral.
The per-step diagnostic is named `bias/<name>/applied_gradient` rather than
`mean_force`, because it is the ramped and capped value actually used — a
threshold-suppressed zero there is not a measured zero mean force.

The per-step diagnostics are `cv`, `bin`, `applied_gradient`, `samples`,
`ramp`, and `in_range`. A walker outside `cv_range` reports `bin = -1` and
zero for every per-bin quantity, matching the zero force it receives. Only
`cv` is still reported, since the coordinate is genuinely measured wherever
the walker is.
Bins never visited come back as `nan` rather than zero — a bin with no samples
has no estimate, and zero is a perfectly plausible value that would hide that.
`free_energy()` **raises** on an interior gap: integration carries the profile
across a hole, so every value beyond it would be wrong by an unknown constant.

## Writing your own bias

### A bias is an additive potential

There is no bias protocol and no bias result type. A bias is a
`BaseModelMixin` that maps a `Batch` to `ModelOutputs` — the same shape as
`DFTD3ModelWrapper` and `LennardJonesModelWrapper`, which are also pure
additive potentials with no learned parameters:

```python
class MyBias(nn.Module, BaseModelMixin):
    def __init__(self):
        super().__init__()
        self.name = "my_bias"
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
        )

    def forward(self, data, **kwargs):
        return OrderedDict(energy=..., forces=...)
```

In practice you subclass `ConservativeBias`, which is that plus the autograd
derivation — but nothing forces you to.

#### Diagnostics and state versions

`ModelOutputs` is an open mapping, and two general conventions ride in it:

- **`diagnostics/<key>`** — arbitrary reported tensors, no shape contract,
  never summed and never applied. The bias hook surfaces them as
  `bias/<name>/<key>`. A per-atom energy decomposition belongs here: it is a
  diagnostic, not a contribution to `batch.energy`.
- **`state_version`** — integer revision IDs, shape `[B]`, saying which
  revision of the producer's state generated the rest of the mapping.
  `AdaptivePotentialMixin` stamps it for you.

Neither is enhanced-sampling-specific; both are documented on `ModelOutputs`
and checked by `validate_contribution`.

#### What the bias hook will actually apply

Only `energy`, `forces`, and `stress` are added into the batch. For each one
the bias hook has to know the destination buffer, whether it is per-graph or
per-atom, and how it combines across biases. An unrecognised *applied* key has
none of that, so it is refused — in two places, for two different failures:

| Failure | Raised by | When |
|---------|-----------|------|
| The key is not one the framework can apply (`hessian`, `dipole`, anything novel) | `validate_contribution`, and `aggregate_contributions` independently | at the producer, and again before summing |
| The key is applicable but the batch has no buffer for it | `_check_destinations` | after aggregation, before applying |

The first matters because aggregation keeps only the applied keys, so an
unrecognised physical output would otherwise vanish between the bias and the
buffer with the run carrying on as though it had been applied. It is checked
in both places on purpose: `aggregate_contributions` is public and promises
that nothing is dropped silently, so it cannot rely on its caller having
validated first. `APPLIED_OUTPUT_KEYS` is the single definition of the set and
both checks read it, so they cannot disagree about what they refuse.

`sum_outputs`, the generic composition helper, is deliberately *not* affected:
a full model forward pass may legitimately report `hessian` or `dipole`, and
composing two models keeps them. A contribution is narrower than a forward
pass precisely because it gets added into a buffer.

The cost is real: a method producing a genuinely new applied output cannot
express it without editing the framework. That is accepted, because the
alternative is a contribution the bias hook silently drops. Anything you only
want to *see* goes under `diagnostics/`, which is unconstrained — and
`diagnostics/hessian_trace` is fine where `hessian` is not.

### The batteries: mixins

Capability is opt-in per bias, supplied as composable mixins:

```python
class MyRestraint(ConservativeBias): ...                     # energy -> forces + stress
class MyMetaD(AdaptivePotentialMixin, ConservativeBias): ... # ...and evolving state
class MyABF(AdaptivePotentialMixin, nn.Module, BaseModelMixin): ...  # adaptive, no energy
```

A non-conservative bias has no energy to differentiate, so it does not inherit
`ConservativeBias` — but it is still a `BaseModelMixin`. `AdaptiveBiasingForce`
declares `outputs={"forces"}` and returns no `"energy"` key. *Non-conservative
does not mean non-model*, and that is what lets the bias hook type-check, apply,
and distribute every bias the same way.

:::{important}
`AdaptivePotentialMixin` must come **first** in the base list. `ConservativeBias`
inherits `nn.Module`, whose `state_dict` would otherwise shadow the mixin's and
silently drop the bias history from every checkpoint. Getting the order wrong
raises `TypeError` at class-creation time.
:::

### `ConservativeBias`

Override `energy()` and get forces and stress by autograd:

```python
class MyRestraint(ConservativeBias):
    def __init__(self, k):
        super().__init__(name="my_restraint")   # required
        self.k = k

    def energy(self, current):
        return 0.5 * self.k * my_cv(current) ** 2      # [B, 1]
```

Notes:

- **Stress, not virial.** `ConservativeBias` emits tensile-positive Cauchy
  stress, matching every model wrapper in the toolkit, so bias output sums
  directly with model output. A `"virial"` key exists for hand-written biases
  that produce a virial directly, but the bias hook will reject it — convert with
  `sigma = -W/V` first.
- **Partial dependence is fine.** An energy that depends only on the cell (a
  volume restraint) yields zero forces and real stress; one that returns a
  constant on some branch yields zeros for both. Neither is an error.
- **The isolation is shared, not private.** `forward()` delegates to
  `nvalchemi.models._utils.isolated_energy_derivatives`, which evaluates an
  energy function against a detached view of a live `Batch`, restores every
  field it touched in a `finally` block, and returns derivatives with no grad
  graph attached. Anything that differentiates against a batch it does not own
  — an NEB spring term, a hand-written wall — can call it directly without
  depending on this package.
- **`torch.compile` boundary is `energy()`, not `forward()`.**
  `forward()` reaches `requires_grad_()`, which `torch.compile` cannot trace.
  `EnhancedSampling(compile_biases=True)` compiles each bias's `energy()`.

:::{important}
Because `compile_biases=True` hands `energy()` to `torch.compile`, **keep
data-dependent Python branches out of it**. A `bool(tensor.any())` there —
a bounds check, a "did anything violate this" guard — breaks `fullgraph=True`
outright with a "Could not guard on data-dependent expression" error.

Put such validation in an override of `forward()` instead, which is eager by
construction, then call `super().forward(data)`. `HarmonicUmbrellaBias`
validates `thermodynamic_state_id` this way. That placement is strictly better
than an eager-only `torch.compiler.is_compiling()` guard: the check still runs
when `energy()` is compiled, rather than being skipped exactly when a mistake
is hardest to diagnose.
:::

### Adaptive biases

`AdaptivePotentialMixin` separates read-only evaluation from state mutation.
It is the shared `StatefulHook` lifecycle from `nvalchemi.hooks`, filled in —
`frequency`, `stage`, `read_only`, and `commit` mean here exactly what they
mean for any other hook:

```python
class MyMetaD(AdaptivePotentialMixin, ConservativeBias):
    frequency = 100                          # Hook: every N steps
    stage = DynamicsStage.AFTER_STEP         # Hook: where in the step
    read_only = False                        # StatefulHook: update() mutates

    def energy(self, current): ...           # read-only, compile-friendly

    def update(self, ctx, stage):            # called once per due step
        self.deposit_hill(ctx.batch)
        self.bump_state_version()            # forces are now stale

    def commit(self):                        # StatefulHook sync boundary
        ...                                  # publish shared history, if any
```

"Read-only while forces are computed, mutate after the step, synchronise
occasionally" is not specific to sampling — NEB's climbing-image promotion, an
adaptive thermostat, and an adaptive neighbour skin have the same shape — so
the vocabulary lives in `nvalchemi.hooks.StatefulHook` and this mixin only
fills it in. `StatefulHook` itself composes `Hook` (`frequency`, `stage`) and
`CheckpointableHook` (`state_dict`, `load_state_dict`), adding only `read_only`
and `commit`.

`stage` decides which frame `update` receives:

- `AFTER_STEP` — post-step coordinates. What metadynamics wants.
- `AFTER_COMPUTE` — captured while `batch.forces` still holds the **unbiased**
  physical forces. What ABF requires; an estimator fed its own output diverges.

`ctx` is a `BiasContext` — a `DynamicsContext` with one extra field,
`ctx.contribution`, holding what this bias returned during the force
evaluation that preceded the capture. It lives in `nvalchemi.hooks` beside
`DynamicsContext` and `TrainContext`, not in this subpackage, for the same
reason the lifecycle does.

:::{note}
`update` is not spelled `__call__` even though a `Hook` is dispatched that way.
A bias is also an `nn.Module`, whose `__call__` is the model forward that
`BaseDynamics` and `PipelineModelWrapper` both invoke; one name cannot be both.
This is the resolution `TrainingUpdateHook` already uses — a domain hook family
keeps the signature its semantics need, and an orchestrator owns protocol
compliance on its behalf. Here that orchestrator is the `BiasHook`
`EnhancedSampling` installs, which reads `frequency` and `stage` and dispatches
`update` and `commit`.
:::

## The strategy

`EnhancedSampling` contributes hooks to the engine it builds and otherwise
leaves it alone — the model, integrator, thermostat, and every caller-supplied
hook behave exactly as they would unbiased. `BaseDynamics` owns the stepping
loop; there is no second one.

### The hooks it installs

`build_hooks()` returns four, in this order, ahead of any `extra_hooks` you
pass:

| Hook | Stage | Cadence | Does |
|------|-------|---------|------|
| `WalkerIdentityHook` | `BEFORE_STEP` | every step | stamps the five identity fields |
| `BiasHook` | `AFTER_COMPUTE`, `AFTER_STEP` | every step | evaluates and applies biases, delivers `update()` |
| `PairSwapHook` | `BEFORE_STEP` | `attempt_interval` | attempts the completed segment's swaps |
| `EpochCommitHook` | `BEFORE_STEP` | `steps_per_epoch` | fires `commit()` on the completed epoch |

`PairSwapHook` is not an enhanced-sampling class. Propose pairs, evaluate an
acceptance rule, permute per-system parameters for the pairs that pass — that
is also basin hopping with swaps, population search, and any annealing ladder,
so it lives in {mod}`nvalchemi.dynamics.hooks` and
`ReplicaExchange.swap_hook()` supplies the two pieces that are physics: the
Sugita-Okamoto acceptance rule and the temperature table.

The last two carry their cadence as `Hook.frequency`, so the hook registry
gates them: dispatched at step *kN*, each acts on boundary `step // N - 1`,
the one that has just completed. Exchange precedes commit because a commit
publishes shared history and doing it before the swap would publish under
labels that are about to change.

All the biases share **one** `BiasHook` rather than one hook each. Registered
separately, the second bias would evaluate against a batch already carrying
the first one's forces, and hook registration order is user-owned, so the
total would depend on it.

### What it guarantees

1. **Every bias sees the same unmodified physical output.** Contributions are
   summed once and applied together, so no bias can observe another's forces
   and the total does not depend on registration order.
2. **`update()` is delivered exactly once per due step**, after integration.
3. **Diagnostics are namespaced** `bias/<name>/<key>`, so two biases of the
   same type cannot collide.
4. **Forces are primed** before the first step. A velocity-Verlet-style
   integrator reads `batch.forces` in its first half-step, before any model
   call; without priming, step 0 would be the one step that ignores the bias.

The strategy's hooks come **before** `extra_hooks`, so a safety hook such as
`MaxForceClampHook` passed as an extra clamps the *total* force rather than
the model force alone.

### Diagnostics

```python
sampling.last_outputs["physical/forces"]       # model only, before any bias
sampling.last_outputs["bias/umbrella/energy"]  # one bias's contribution
sampling.last_outputs["bias_total/forces"]     # sum across all biases
sampling.last_outputs["total/forces"]          # physical + bias
```

`total/*` is read back from the batch after the bias is applied, so
`total == physical + bias_total`. Note that this is the state as the *bias
hook* leaves it, not necessarily what the integrator consumed: that hook runs
ahead of any `extra_hooks` at `AFTER_COMPUTE` (so a force clamp acts on the
total rather than the model force alone), which means a later hook can still
modify `batch.forces`.
Read the batch directly if you need the exact value the integrator used.

For WHAM or MBAR you want `physical/energy` and the per-bias energies
separately, not `total/energy` — the reweighting needs the unbiased potential.
Free-energy reconstruction is deliberately not built in; use `pymbar` or an
equivalent.

### Batch requirements

Dynamics writes model outputs back **in place**, so the buffers must exist:

```python
AtomicData(
    positions=..., atomic_numbers=..., atomic_masses=...,
    forces=torch.zeros(n_atoms, 3),
    energy=torch.zeros(1, 1),
    stress=torch.zeros(1, 3, 3),   # required whenever a bias produces stress
)
```

The bias hook raises a named `ValueError` naming the field, the biases that
produced it, and how to allocate the buffer — rather than skipping the field
and letting the contribution vanish. Because `run()` primes before the first
step, this surfaces at setup, not part-way through a trajectory.

:::{warning}
`stress` is the one that matters most. `ConservativeBias` produces stress
whenever the batch has a cell and at least one periodic dimension, so a
periodic run needs the buffer even under NVT. A stress contribution dropped on
the floor is invisible to an NPT/NPH barostat — the cell evolves as if the
bias were absent, with nothing to indicate it. If a run genuinely has no use
for a cell response, pass `compute_stress=False` to the bias, which drops
`"stress"` from its `active_outputs` and skips the strain leaf entirely. That
is a deliberate choice; a missing buffer is not.
:::

### Walker identity

`WalkerIdentityHook` stamps five graph-level fields each step. Batch *position* is not
an identity — selection and refill can move a walker to a different row — so
anything that must follow a physical configuration is carried as data:

| Field | Meaning |
|-------|---------|
| `walker_id` | Immutable identity, assigned once |
| `thermodynamic_state_id` | Window / temperature / energy-function state |
| `sampling_step` | Dynamics force-evaluation step |
| `exchange_segment` | Exchange segment, `step // attempt_interval` (see [Replica exchange](#replica-exchange)) |
| `sampling_epoch` | Consistency epoch, `step // steps_per_epoch` |

A `thermodynamic_state_id` you set yourself is preserved, never overwritten.
Without replica exchange, `exchange_segment` falls back to the epoch length,
since there are no exchange segments to count.

## Checkpoint and restore

```python
sampling.checkpoint("run.zarr")          # only at an epoch boundary

# ... later, in a fresh process ...
sampling2 = EnhancedSampling(
    engine=NVTLangevin,
    engine_kwargs={"dt": 0.5, "temperature": 300.0, "friction": 0.05},
    biases={"umbrella": umbrella},
)
batch = sampling2.restore("run.zarr", model)   # returns a force-primed batch
batch = sampling2.run(batch, model, n_steps=10_000, prime=False)
```

Resuming reproduces the **identical trajectory**. `NVTLangevin` derives its
noise from `random_seed + step_count` rather than a stateful generator, so
restoring those two integers restores the noise sequence exactly — there is no
RNG object to serialise.

### Only at an epoch boundary

`checkpoint()` raises unless `step_count % steps_per_epoch == 0`, and the error
names the next valid step. This is not bookkeeping fussiness: an epoch boundary
is the only point with no pending `update()` and no in-flight epoch commit, so
anywhere else risks capturing a bias mid-mutation.

Being *at* a boundary is not the same as being quiescent, though. Both the
epoch commit and the replica exchange fire **lazily** — the first step of the
next epoch or segment is what notices the boundary was crossed — so
immediately after `run(..., n_steps=N)` neither has happened.

`checkpoint()` therefore drains both itself before collecting any state, in
the same order the runtime uses: **exchange first, then commit**, because the
commit publishes shared history and doing it before the swap would publish
under labels that are about to change. A shared-history bias is recorded with
its deposits merged rather than still pending, and the labels are post-swap.

Both drains are idempotent — tracked per epoch index and per segment index —
so the lazy path on the next step sees them as already done and cannot
double-count.

:::{note}
`checkpoint()` is therefore **not a passive snapshot**: it can advance the
exchange assignment as part of reaching a quiescent point. Read
`batch.thermodynamic_state_id` *after* checkpointing if you want the value
that was saved.
:::

### Transactional by construction

Writing state transactionally and refusing a torn store is not specific to
enhanced sampling, so it lives in the shared checkpoint layer
(`nvalchemi._checkpoint`) rather than here: `save_checkpoint` and
`load_checkpoint` take a mapping of name to anything with
`state_dict`/`load_state_dict`, which is what hooks, integrators, biases and
the ladder already are. This section describes what an enhanced-sampling run
puts in it.

The store is written batch → components → **manifest last**. The manifest is
the commit marker:

- No manifest ⇒ the write was interrupted ⇒ `load_checkpoint` refuses it.
- **Everything** is checksummed (SHA-256) and verified on read, so damage
  *after* the manifest landed is caught too: each component individually,
  plus a `batch_checksum` covering `meta/`, `core/`, and `custom/`. Cover is
  mandatory, not best-effort — a manifest that declares a component without a
  checksum, carries a checksum for no component, or omits the batch checksum
  is rejected as invalid — otherwise deleting one key from the manifest would
  be enough to leave that component free to modify.
- The batch checksum is the one most easily forgotten: the batch is written by
  `AtomicDataZarrWriter`, outside the component path, so covering only the
  component state would attest to the bias and integrator while restoring
  corrupted positions or a scrambled walker identity in silence.
- **No pickle payloads.** State is Zarr arrays and JSON attributes, so a
  checkpoint is readable by anything that reads Zarr and loading one cannot
  execute code. An unsupported value type raises rather than falling back.

```text
run.zarr/
  meta/, core/, custom/    walker batch (custom/ carries walker identity)
  checkpoint/
    manifest               written last — the commit; holds every checksum
    dynamics/              step counter, RNG seed, per-system integrator state
    biases/<name>/         each bias's state_dict()
    hooks/identity/        walker-id allocation
    hooks/bias/            per-bias update() delivery record
    hooks/epoch/           last committed epoch
    hooks/exchange/        last attempted segment
    exchange/              ladder counters and acceptance-RNG position
```

Each of those groups is one object's `state_dict()`, so a checkpoint can be
inspected piece by piece rather than as one blob. The manifest also carries a
free-form `compatibility` fingerprint — model, dynamics and bias classes, and
the ladder — which `restore()` checks before applying anything.

### Bias configuration is validated, not just bias class

The manifest records each bias's *class*, but a class name says nothing about
the settings its saved state depends on. An ABF histogram is state; the
`cv_range` that decides what its bins mean is configuration. Restoring the
first without the second leaves the counts shape-compatible and silently
relabels every bin — bin 5 stops meaning `r = 1.55` and starts meaning
`r = 3.1`, carrying its accumulated mean force with it.

Every adaptive bias therefore records a `config_fingerprint()` inside its own
`state_dict()`, checked on load:

| Bias | Checked |
|------|---------|
| `AdaptiveBiasingForce` | `atom_indices`, `cv_range`, `n_bins`, `temperature`, `min_samples`, `full_samples`, `max_force` |
| `WellTemperedMetaDynamicsBias` | `height`, `sigma`, `temperature`, `bias_factor`, `storage`, `history`, `ramp_depositions`, `periods` |
| `RMSDMetaDynamicsBias` | `k_push`, `alpha`, `storage`, `history`, `ramp_depositions`, `atom_indices` |

Because the check lives in `load_state_dict` rather than in the manifest, it
covers a bias restored directly as well as one restored through the strategy.

Capacity (`max_hills`, `max_references`) is deliberately **not** checked:
`storage="grow"` legitimately reaches a size the constructor never had, and
`load_state_dict` already resizes to match.

The check runs *before* delegating to `nn.Module.load_state_dict`, which
matters for configuration held as a buffer — `sigma`, `atom_indices`. Loading
overwrites buffers, so a check afterwards would come too late to stop the
caller's value being replaced by the checkpoint's, which is the opposite of
what asking for it meant.

### Model weights are not restored

Reconstruct the model — including loading its weights through its own API —
before calling `restore()`. The manifest records the model class, dynamics
class, and bias set, and `restore()` refuses a mismatch; but that proves the
*architecture* agrees, not the weights.

### `warm_start` vs `restore`

| | `warm_start(frames)` | `restore(path)` |
|---|---|---|
| Bias history | replayed, approximately | exact |
| Velocities, RNG, integrator state | not restored | exact |
| Use when | continuing from a trajectory snapshot | resuming a run exactly |

They are mutually exclusive: `warm_start()` after `restore()` raises, because
replaying history the restored state already contains would corrupt it.

## Replica exchange

A ladder of thermodynamic states, all advanced as one batch, with periodic
swaps of which walker sits on which rung:

```python
from nvalchemi.enhanced_sampling import ReplicaExchange, ThermodynamicState

states = [
    ThermodynamicState(state_id=i, temperature=300.0 * 1.15 ** i)
    for i in range(4)
]
exchange = ReplicaExchange(
    states=states,
    initial_state_ids=torch.arange(4),   # must be a permutation
    attempt_interval=100,                # steps per exchange segment
    random_seed=2024,
)
sampling = EnhancedSampling(
    engine=NVTLangevin,
    engine_kwargs={"dt": 0.5, "temperature": 300.0, "friction": 0.05},
    biases={},
    replica_exchange=exchange,
)
batch = sampling.run(batch, model, n_steps=100_000)
```

### One walker per rung

Exchange presumes a bijection: every walker holds exactly one state and every
state exactly one walker, because pairing looks up "which walker holds state
*k*". `WalkerIdentityHook` validates that on the first stamp, whether the
assignment came from `initial_state_ids` or was already on the batch:

```text
ReplicaExchange: the ladder has 4 state(s) but the batch has 2 walker(s).
ReplicaExchange: batch.thermodynamic_state_id must be a permutation of 0..3,
                 got [0, 0, 1, 2].
```

Both are configuration errors that would otherwise surface much later and much
less clearly — a size mismatch as `Length mismatch: 4 vs 2` from inside the
batch storage, a duplicate as a bare `KeyError` from the pair lookup.

### Labels move, coordinates do not

An accepted swap permutes `thermodynamic_state_id`. The walker keeps its row,
its velocities, and its integrator arrays; the temperature assigned to it
changes. Nothing is copied between rows, which is what makes the move viable
inside a batched GPU step.

The swap is **indivisible**: the label, the integrator's target temperature,
the velocity rescaling, and the forces all move together. A walker labelled
one rung while its thermostat targets another samples the wrong ensemble with
no symptom, so the strategy refuses at construction any engine class that
cannot rebind:

```text
TypeError: replica exchange needs NVE to implement
apply_per_system_params(), so an accepted swap can rebind temperature,
velocities, and thermostat state together.
```

`apply_per_system_params(params, batch)` is the generic adapter — the batch is
a parameter rather than a follow-up call precisely so the velocity rescale
cannot be forgotten. An integrator that rebinds a parameter it does not
understand raises rather than ignoring it.

`NVTLangevin` rescales velocities by `sqrt(T_new / T_old)`. `NVTNoseHoover`
additionally transforms its chain: `Q` scales with `kT` and `eta_dot` with
`1/sqrt(kT)`, which leaves the chain kinetic energy invariant — injecting
thermostat energy on a swap is exactly what breaks detailed balance.

### Acceptance

The rule is **inferred from the ladder**, never declared, because a declared
rule that disagreed with the ladder would be silent and wrong acceptance
breaks detailed balance without any symptom a run would show.

| Ladder | Rule | Formula |
|--------|------|---------|
| Temperatures differ | temperature | `log a = min(0, (β_i − β_j)(U_i − U_j))` |
| Temperatures equal | umbrella | `log a = min(0, u_i(x_i) + u_j(x_j) − u_i(x_j) − u_j(x_i))` |

Umbrella acceptance needs the bias evaluated under swapped labels, which
costs one extra bias evaluation per attempt.

A ladder that varies temperature *and* bias window at once needs a combined
rule that is not implemented. The temperature rule alone omits the cross-state
bias terms, so running it anyway breaks detailed balance with no symptom —
it is therefore **rejected**, twice over:

- A bias that sets `state_dependent_for_exchange` is refused at construction.
  `HarmonicUmbrellaBias` sets it whenever it has more than one window.
- At prime time the strategy **probes** every bias empirically: it evaluates
  each one under the current assignment and under a rotated one, at identical
  coordinates. A bias whose energy is independent of the assignment returns
  the same number twice; one that reads `thermodynamic_state_id` does not.
  That catches a user-written bias which declares nothing.

Vary one or the other. A **single-window** `HarmonicUmbrellaBias` is fine
alongside a temperature ladder — it ignores `thermodynamic_state_id` rather
than treating it as a window index, so the ids are free to address the
ladder.

### Pair schedule

Segments alternate even and odd offsets: `(0,1),(2,3)` then `(1,2),(3,4)`.
No state appears twice in one segment, which is what lets every pair be
decided simultaneously; two segments cover every neighbouring pair.

A segment's pairs are attempted when it **completes** — entering segment *s*
decides segment *s−1*, the same way entering epoch *e* commits epoch *e−1*.
So the first swap lands at `attempt_interval`, using segment 0's pairs.

### Tuning the ladder

```python
exchange.acceptance_rate            # overall
exchange.pair_acceptance_rates()    # per neighbouring pair
```

Per-pair rates are what a ladder is tuned on. A pair far below the others is
a gap the walkers cannot cross and needs another rung; uniformly high rates
mean the rungs are closer than they need to be.

### Restoring an exchange run

The manifest records the ladder — mode, acceptance rule, `attempt_interval`,
and temperatures — and `restore()` refuses a mismatch. That includes both
directions of exchange-versus-none:

```text
EnhancedSampling.restore: the checkpoint was written by a different configuration:
  exchange temperatures: checkpoint has [300.0, 350.0, 400.0],
                         this strategy has [100.0, 200.0, 900.0]
```

This is not pedantry. The ladder decides what a swap *means*: restoring into
different temperatures would keep the walker assignment and the acceptance
counters while silently changing the exponent every future swap is decided
on. `initial_state_ids` is deliberately *not* checked — it seeds the
assignment only when the batch does not already carry one, and a restored
batch always does.

### Reproducibility

Acceptance draws come from `random_seed + exchange_id` rather than a
long-lived generator — the same counter-based scheme `NVTLangevin` uses for
its noise. A checkpoint therefore stores two integers instead of an RNG blob,
and a restored run reproduces the same accept/reject decisions. Exchange
state lives under `sampling/exchange/`.

### Not supported

Asynchronous exchange (pair-local rendezvous, non-blocking workers) is not
implemented; `mode="asynchronous"` raises. A force-only bias such as adaptive
biasing force cannot participate — the acceptance rule needs a cross-state
bias energy — and is rejected rather than silently excluded.

## Relationship to `BiasedPotentialHook`

{class}`~nvalchemi.hooks.BiasedPotentialHook` covers similar ground and is
**deprecated**. Its `bias_fn(batch) -> (energy, forces)` contract has no slot
for a cell response, so a bias applied through it contributes no stress and is
invisible to an NPT/NPH barostat — the cell evolves as if the bias were absent,
with no error raised. It also cannot check that the returned forces are
`-dE/dr`, and composes several biases by sequential in-place mutation.

With `EnhancedSampling` now available, the migration path is complete —
anything the hook does, this subpackage does. Existing hook-based code is
correct under NVE and NVT, where nothing reads the stress, so it can be
migrated when convenient rather than urgently. The hook remains functional and
no removal date is set.

No adapter is provided: bridging a bias onto `bias_fn` would have
to discard its `stress`, reintroducing the exact failure the new API
removes.

## Not yet implemented

- Adaptive biasing force over any CV other than a pair distance; each
  coordinate needs its own metric correction.
- Asynchronous replica exchange (pair-local rendezvous, non-blocking
  workers). `mode="asynchronous"` raises; synchronous exchange is available.
- The combined temperature-plus-window acceptance rule; see
  [Acceptance](#acceptance) for what is rejected and why.
- General triclinic MIC for unreduced cells.
- Domain decomposition. `ConservativeBias.distribution_spec()` returns `None`,
  which makes `DistributedModel` raise rather than shard a bias whose
  cross-rank semantics are undefined. A bias that genuinely is local can
  override it; a CV like `pair_distance` across the cell is not.

## See also

- {doc}`Conventions <about/conventions>` — virial, stress, and pressure signs.
- {doc}`Hooks <hooks>` — the hook protocol the strategy builds on.
- {doc}`Dynamics <dynamics>` — integrators and the step sequence.
