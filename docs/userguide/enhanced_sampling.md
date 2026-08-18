# Enhanced Sampling

Molecular dynamics follows the natural motion of atoms, which means it spends
almost all of its time in free-energy minima. Barrier crossings — diffusion
events, reactions, nucleation, conformational change — are rare on MD
timescales. Enhanced sampling adds a bias potential that pushes the system
into regions it would not visit on its own, so that a fixed budget of model
evaluations buys more of the physics you actually care about.

`nvalchemi.enhanced_sampling` provides the bias abstractions, a set of
built-in biases, and the `EnhancedSampling` runner that wires them into an
existing dynamics object.

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

pair = torch.tensor([0, 5])
umbrella = HarmonicUmbrellaBias(
    cv=lambda batch: pair_distance(batch, pair),
    centers=torch.tensor([[2.0], [2.5], [3.0]]),   # three windows
    stiffness=10.0,                                 # eV/A^2
    name="umbrella",
)

dynamics = NVTLangevin(model=model, dt=0.5, temperature=300.0, friction=0.05)
sampling = EnhancedSampling(dynamics, {"umbrella": umbrella})

batch["thermodynamic_state_id"] = torch.tensor([0, 1, 2])  # window per graph
batch = sampling.run(batch, n_steps=10_000)
```

Every window is a row of one batch, so all three are advanced by a single
batched force evaluation per step rather than three separate simulations.

## Collective variables

A CV is **any differentiable callable** `cv(batch) -> Tensor[B, D]`. There is
no base class, no registration, and nothing to subclass:

```python
def bond(batch):
    return pair_distance(batch, torch.tensor([0, 5]))
```

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

## Writing your own bias

### The boundary: `BiasPotential`

`BiasPotential` is a `@runtime_checkable` Protocol that **inherits nothing**.
A bias needs a `name` and an `evaluate`:

```python
class MyBias:
    name = "my_bias"

    def evaluate(self, current):
        return BiasResult(energy=..., forces=...)
```

That class satisfies the protocol with no base class at all.

### The batteries: mixins

Capability is opt-in per bias, supplied as composable mixins:

```python
class MyRestraint(ConservativeBias): ...                        # energy -> forces + stress
class MyMetaD(AdaptivePotentialMixin, ConservativeBias): ...    # ...and evolving state
class MyABF(AdaptivePotentialMixin): ...                        # adaptive, no energy
```

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
  directly with model output. `BiasResult.virial` exists for hand-written
  biases that produce a virial directly, but the runner will reject it —
  convert with `sigma = -W/V` first.
- **Partial dependence is fine.** An energy that depends only on the cell (a
  volume restraint) yields zero forces and real stress; one that returns a
  constant on some branch yields zeros for both. Neither is an error.
- **`torch.compile` boundary is `energy()`, not `evaluate()`.**
  `evaluate()` calls `requires_grad_()`, which `torch.compile` cannot trace.
  `EnhancedSampling(compile_biases=True)` compiles each bias's `energy()`.

### Adaptive biases

`AdaptivePotentialMixin` separates read-only evaluation from state mutation:

```python
class MyMetaD(AdaptivePotentialMixin, ConservativeBias):
    update_frequency = 100
    observation_stage = DynamicsStage.AFTER_STEP

    def energy(self, current): ...          # read-only, compile-friendly

    def update(self, frames, result):       # called once per due step
        self.deposit_hill(frames)
        self.bump_state_version()           # tells the runner forces are stale
```

`observation_stage` decides which frame `update` receives:

- `AFTER_STEP` — post-step coordinates. What metadynamics wants.
- `AFTER_COMPUTE` — captured while `batch.forces` still holds the **unbiased**
  physical forces. What ABF requires; an estimator fed its own output diverges.

## The runner

`EnhancedSampling` installs one internal hook on the dynamics and otherwise
leaves it alone — the model, integrator, thermostat, and every other hook
behave exactly as they would unbiased.

### What it guarantees

1. **Every bias sees the same unmodified physical output.** Contributions are
   summed once and applied together, so no bias can observe another's forces
   and the total does not depend on registration order.
2. **`update()` is delivered exactly once per due step**, after integration.
3. **Observables are namespaced** `bias/<name>/<key>`, so two biases of the
   same type cannot collide.
4. **Forces are primed** before the first step. A velocity-Verlet-style
   integrator reads `batch.forces` in its first half-step, before any model
   call; without priming, step 0 would be the one step that ignores the bias.

The bias hook is inserted at the **front** of the hook list, so a safety hook
such as `MaxForceClampHook` clamps the *total* force rather than the model
force alone.

### Diagnostics

```python
sampling.last_outputs["physical/forces"]      # model only
sampling.last_outputs["bias/umbrella/energy"] # one bias
sampling.last_outputs["total/forces"]         # what the integrator used
```

These are the tensors a WHAM or MBAR post-processing step needs. Free-energy
reconstruction is deliberately not built in — use `pymbar` or an equivalent.

### Batch requirements

Dynamics writes model outputs back **in place**, so the buffers must exist:

```python
AtomicData(
    positions=..., atomic_numbers=..., atomic_masses=...,
    forces=torch.zeros(n_atoms, 3),
    energy=torch.zeros(1, 1),
    stress=torch.zeros(1, 3, 3),   # periodic runs only
)
```

`prime_forces` raises a named `ValueError` if `forces` is missing rather than
letting the model output be silently discarded.

### Walker identity

The runner stamps five graph-level fields each step. Batch *position* is not
an identity — selection and refill can move a walker to a different row — so
anything that must follow a physical configuration is carried as data:

| Field | Meaning |
|-------|---------|
| `walker_id` | Immutable identity, assigned once |
| `thermodynamic_state_id` | Window / temperature / energy-function state |
| `sampling_step` | Dynamics force-evaluation step |
| `exchange_segment` | Exchange segment number (always 0 until replica exchange lands) |
| `sampling_epoch` | Consistency epoch, `step // steps_per_epoch` |

A `thermodynamic_state_id` you set yourself is preserved, never overwritten.

## Relationship to `BiasedPotentialHook`

{class}`~nvalchemi.hooks.BiasedPotentialHook` covers similar ground and is
**deprecated**. Its `bias_fn(batch) -> (energy, forces)` contract has no slot
for a cell response, so a bias applied through it contributes no stress and is
invisible to an NPT/NPH barostat — the cell evolves as if the bias were absent,
with no error raised. It also cannot check that the returned forces are
`-dE/dr`, and composes several biases by sequential in-place mutation.

It remains functional and is not scheduled for removal. No adapter is provided:
bridging a `BiasPotential` onto `bias_fn` would have to discard
`BiasResult.stress`, reintroducing the exact failure the new API removes.

## Not yet implemented

- Metadynamics (well-tempered and xTB-style RMSD) and adaptive biasing force.
- `ThermodynamicState` and `ReplicaExchange`.
- Zarr checkpointing: `EnhancedSampling.checkpoint()` and `restore()` raise
  `NotImplementedError`. Individual biases expose `state_dict()` /
  `load_state_dict()` now, and `warm_start()` gives approximate continuation.
- General triclinic MIC for unreduced cells.
- Domain decomposition. `ConservativeBias.distribution_spec()` returns `None`,
  which makes `DistributedModel` raise rather than shard a bias whose
  cross-rank semantics are undefined. A bias that genuinely is local can
  override it; a CV like `pair_distance` across the cell is not.

## See also

- {doc}`Conventions <about/conventions>` — virial, stress, and pressure signs.
- {doc}`Hooks <hooks>` — the hook protocol the runner builds on.
- {doc}`Dynamics <dynamics>` — integrators and the step sequence.
