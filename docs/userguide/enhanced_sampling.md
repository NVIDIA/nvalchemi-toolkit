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

device = "cuda" if torch.cuda.is_available() else "cpu"

pair = torch.tensor([0, 5], device=device)
umbrella = HarmonicUmbrellaBias(
    cv=lambda batch: pair_distance(batch, pair),
    centers=torch.tensor([[2.0], [2.5], [3.0]]),   # three windows
    stiffness=10.0,                                 # eV/A^2
    name="umbrella",
)

dynamics = NVTLangevin(model=model, dt=0.5, temperature=300.0, friction=0.05)
sampling = EnhancedSampling(dynamics, {"umbrella": umbrella})

# One window per graph. batch already carries forces/energy buffers — see
# "Batch requirements" below.
batch["thermodynamic_state_id"] = torch.tensor([0, 1, 2], device=device)
batch = sampling.run(batch, n_steps=10_000)
```

Every window is a row of one batch, so all three are advanced by a single
batched force evaluation per step rather than three separate simulations.

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

The first four are static: their energy depends only on the current
configuration. The last two are **history-dependent** — they accumulate state
as sampling proceeds, and so mix in `AdaptivePotentialMixin`.

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
value, every `update_frequency` steps. In the well-tempered scheme each hill
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
    update_frequency=500,
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
error names the ways out. `fifo` is not merely a cache policy: once hills are
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
    update_frequency=500,
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

- **Non-periodic systems only.** A batch with a non-zero cell is rejected.
  Cartesian RMSD against a stored reference is not defined under periodic
  boundary conditions — an atom crossing a cell face is physically unmoved but
  Cartesian-displaced by a lattice vector, which would inject a large spurious
  force. Bias a periodic-aware CV with `WellTemperedMetaDynamicsBias` instead.
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
bumps the bias state version, and the runner re-primes forces in response, so
a new hill is felt on the very next step rather than one step late.

Neither deposits during `prime_forces()`: priming evaluates forces, it does
not advance the trajectory, and depositing there would double-count the
starting configuration.

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

:::{important}
Because `compile_biases=True` hands `energy()` to `torch.compile`, **keep
data-dependent Python branches out of it**. A `bool(tensor.any())` there —
a bounds check, a "did anything violate this" guard — breaks `fullgraph=True`
outright with a "Could not guard on data-dependent expression" error.

Put such validation in an override of `evaluate()` instead, which is eager by
construction, then call `super().evaluate(current)`. `HarmonicUmbrellaBias`
validates `thermodynamic_state_id` this way. That placement is strictly better
than an eager-only `torch.compiler.is_compiling()` guard: the check still runs
when `energy()` is compiled, rather than being skipped exactly when a mistake
is hardest to diagnose.
:::

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
sampling.last_outputs["physical/forces"]       # model only, before any bias
sampling.last_outputs["bias/umbrella/energy"]  # one bias's contribution
sampling.last_outputs["bias_total/forces"]     # sum across all biases
sampling.last_outputs["total/forces"]          # physical + bias
```

`total/*` is read back from the batch after the bias is applied, so
`total == physical + bias_total`. Note that this is the state as the *runner*
leaves it, not necessarily what the integrator consumed: the runner's hook runs
first at `AFTER_COMPUTE` (so a force clamp acts on the total rather than the
model force alone), which means a later hook can still modify `batch.forces`.
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

The runner raises a named `ValueError` naming the field, the biases that
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

The runner stamps five graph-level fields each step. Batch *position* is not
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
sampling2 = EnhancedSampling(dynamics, {"umbrella": umbrella})
batch = sampling2.restore("run.zarr")    # returns a force-primed batch
batch = sampling2.run(batch, n_steps=10_000, prime=False)
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

The store is written walker batch → components → **manifest last**. The
manifest is the commit marker:

- No manifest ⇒ the write was interrupted ⇒ `read_checkpoint` refuses it.
- **Everything** is checksummed (SHA-256) and verified on read, so damage
  *after* the manifest landed is caught too: each `sampling/` component
  individually, plus a `batch_checksum` covering `meta/`, `core/`, and
  `custom/`. Cover is mandatory, not best-effort — a manifest that declares a
  component without a checksum, carries a checksum for no component, or omits
  the batch checksum is rejected as invalid — otherwise deleting one key from
  the manifest would be enough to leave that component free to modify.
- The batch checksum is the one most easily forgotten: the walker batch is
  written by `AtomicDataZarrWriter`, outside the component path, so covering
  only the sampling state would attest to the bias and integrator while
  restoring corrupted positions or a scrambled walker identity in silence.
- **No pickle payloads.** State is Zarr arrays and JSON attributes, so a
  checkpoint is readable by anything that reads Zarr and loading one cannot
  execute code. An unsupported value type raises rather than falling back.

```text
run.zarr/
  meta/, core/, custom/    walker batch (custom/ carries walker identity)
  sampling/
    manifest               written last — the commit; holds every checksum
    dynamics/              step counter, RNG seed, per-system integrator state
    biases/<name>/         each bias's state_dict()
    runner/                walker-id allocation, epoch counters
```

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
sampling = EnhancedSampling(dynamics, biases={}, replica_exchange=exchange)
batch = sampling.run(batch, n_steps=100_000)
```

### One walker per rung

Exchange presumes a bijection: every walker holds exactly one state and every
state exactly one walker, because pairing looks up "which walker holds state
*k*". The runner validates that on the first step, whether the assignment came
from `initial_state_ids` or was already on the batch:

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
no symptom, so the runner refuses at construction any integrator that cannot
rebind:

```text
TypeError: replica exchange needs NVE to implement
apply_thermodynamic_state(), so an accepted swap can rebind temperature,
velocities, and thermostat state together.
```

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
- At prime time the runner **probes** every bias empirically: it evaluates
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
                         this runner has [100.0, 200.0, 900.0]
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

No adapter is provided: bridging a `BiasPotential` onto `bias_fn` would have
to discard `BiasResult.stress`, reintroducing the exact failure the new API
removes.

## Not yet implemented

- Adaptive biasing force.
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
- {doc}`Hooks <hooks>` — the hook protocol the runner builds on.
- {doc}`Dynamics <dynamics>` — integrators and the step sequence.
