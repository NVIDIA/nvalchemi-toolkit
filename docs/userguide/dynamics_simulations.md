<!-- markdownlint-disable MD014 -->

(dynamics_simulations_guide)=

# Optimization and Integrators

This page covers the concrete simulation types provided by the dynamics module.
All of them follow the [execution loop](dynamics_guide) described in the dynamics
overview --- they generally differ only in what `pre_update` and `post_update` do.

## Geometry optimization

Geometry optimization finds the nearest local energy minimum by iteratively moving
atoms downhill on the potential energy surface. The toolkit provides the **FIRE2**
(Fast Inertial Relaxation Engine, improved variant) algorithm and the quasi-Newton
**L-BFGS**, each with fixed- and variable-cell variants.

### Fixed-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire2.FIRE2` optimizes atomic positions
while keeping the simulation cell fixed:

```python
from nvalchemi.dynamics import FIRE2, ConvergenceHook

with FIRE2(
    model=model,
    dt=0.1,           # initial timestep (femtoseconds)
    n_steps=500,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
) as opt:
    relaxed = opt.run(batch)
```

FIRE2 uses an adaptive timestep and velocity mixing: when the system is moving
downhill (forces aligned with velocities), the timestep grows and velocities are
biased toward the force direction. When the system overshoots, the timestep shrinks
and velocities are zeroed. This makes it robust across a wide range of systems
without manual tuning.

### Variable-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire2.FIRE2VariableCell` extends FIRE2 to
simultaneously optimize both atomic positions and the simulation cell. This is
useful for finding equilibrium crystal structures where the lattice parameters are
not known a priori:

```python
from nvalchemi.dynamics.optimizers.fire2 import FIRE2VariableCell
from nvalchemi.dynamics import ConvergenceHook

with FIRE2VariableCell(
    model=model,
    dt=0.1,
    n_steps=500,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
) as opt:
    relaxed = opt.run(batch)
```

The cell degrees of freedom are propagated using an NPH-like scheme at zero target
pressure. The model must return tensile-positive `stress` in addition
to `forces`.

### L-BFGS

{py:class}`~nvalchemi.dynamics.optimizers.lbfgs.LBFGS` and
{py:class}`~nvalchemi.dynamics.optimizers.lbfgs.LBFGSVariableCell` are drop-in
alternatives to FIRE2: one force evaluation per step, usually far fewer steps.

Each L-BFGS step also launches noticeably more Warp kernels than a FIRE2 step
(the two-loop recursion iterates over `history_size` in Python), so its
per-step overhead is higher. With an expensive model forward pass this is
negligible next to fewer total steps, but with a cheap model — like the LJ
model in the examples — the per-step overhead can outweigh the reduction in
step count, making L-BFGS slower in wall-clock time despite converging in
fewer steps. Prefer FIRE2 when the model is cheap and steps are many;
L-BFGS wins when the model forward pass dominates step cost.

```python
from nvalchemi.dynamics import ConvergenceHook
from nvalchemi.dynamics.optimizers import LBFGS

with LBFGS(
    model=model,
    history_size=6,     # stored curvature pairs
    maxstep=0.2,        # largest displacement per step (angstroms)
    n_steps=500,
    convergence_hook=ConvergenceHook.from_fmax(0.05),
) as opt:
    relaxed = opt.run(batch)
```

- Fixed-cell `LBFGS(by_group=True)` updates each graph group as one unit, sharing
  curvature history and step scaling. Set groups with
  `batch.set_group_layout(group_idx)` and use `by_group=True` on any convergence
  hook. Grouped sampling/refill and variable-cell grouping are unsupported.

- `LBFGSVariableCell` needs tensile-positive `stress` and aligned cells: a cell
  is "aligned" when it is lower-triangular (`a` along x, `b` in the xy-plane; see
  {py:class}`~nvalchemi.dynamics.hooks.AlignCellHook`). On admission (and again
  on refill, for a system newly added to an inflight batch), the optimizer
  validates the incoming cell and snapshots it as that system's reference
  chart — raising a `ValueError` right there if it isn't lower-triangular.
  This check runs only at admission/refill, not on every step: once a system
  is running, nothing re-validates its cell before `pre_update`, so a cell
  that drifts out of alignment afterward (no `AlignCellHook`, or one with
  `frequency` other than 1) is silently used against a now-stale reference
  chart instead of raising. Install `AlignCellHook()` (`frequency=1`), as for
  `FIRE2VariableCell`, on the optimizer or its `FusedStage` so every step's
  cell is re-aligned *before* it would otherwise drift — this is what keeps
  the contract true in practice, not a runtime guard that enforces it.
- Do not edit positions between steps (e.g. `WrapPeriodicHook`); L-BFGS builds
  its quasi-Newton direction from `s = x_k - x_{k-1}`, so any out-of-band edit
  (such as wrapping coordinates back into the cell) introduces a spurious jump
  that is not the optimizer's own displacement and corrupts the stored
  curvature pairs. `FreezeAtomsHook` is supported because it restores frozen
  atoms to the same position every step, so their contribution to `s` is
  always zero rather than a fictitious jump.
- The first step after admission moves the largest-force atom by at most `maxstep`.
- The cell reference is captured when a system is admitted. Under `FusedStage`, a
  system entering the stage later keeps it; this stays correct but can take more
  steps if another stage has since changed the cell shape.

Both variable-cell optimizers accept `cell_force_scale` (default 1.0; larger moves
the cell less). `FIRE2VariableCell` reads it every step; `LBFGSVariableCell` fixes
it, like `history_size`, when state is allocated.

### Choosing between fixed and variable cell

Use fixed-cell FIRE2 when the cell is known (e.g. a bulk crystal at experimental
lattice parameters, or a molecule in vacuum where the cell is just a bounding box).
Use variable-cell FIRE2 when the equilibrium cell shape or volume is unknown, such as
when screening candidate crystal structures or computing equations of state.

## Molecular dynamics

Molecular dynamics (MD) propagates the equations of motion forward in time, sampling
the trajectory of a system at finite temperature. The toolkit provides integrators
for three standard ensembles. All MD integrators require `by_group=False`.
Grouped updates (`by_group=True`) are supported by fixed-cell FIRE2 and LBFGS
only. Legacy FIRE and all variable-cell optimizers also reject grouped mode
at construction.

### NVE: energy conservation

{py:class}`~nvalchemi.dynamics.integrators.nve.NVE` uses the Velocity Verlet
algorithm --- a symplectic integrator that conserves total energy in the
microcanonical ensemble:

```python
from nvalchemi.dynamics import NVE

with NVE(model=model, dt=1.0, n_steps=1000) as md:
    trajectory = md.run(batch)
```

NVE is the natural choice for verifying that a model's energy surface is smooth
enough for stable dynamics: if the total energy drifts significantly, the force
field is likely too noisy for the chosen timestep.

### NVT: constant temperature

{py:class}`~nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin` implements the
BAOAB Langevin splitting scheme, which samples the canonical (NVT) ensemble exactly
--- the thermostat does not introduce systematic bias:

```python
from nvalchemi.dynamics import NVTLangevin

with NVTLangevin(
    model=model,
    dt=1.0,              # femtoseconds
    temperature=300.0,    # Kelvin
    friction=0.01,        # collision frequency (1/fs)
    n_steps=10000,
) as md:
    trajectory = md.run(batch)
```

The `friction` parameter controls how strongly the thermostat couples to the
system. A low value gives longer correlation times (closer to NVE); a high value
thermalises quickly but damps real dynamics.

### NPT: constant pressure

{py:class}`~nvalchemi.dynamics.integrators.npt.NPT` uses the
Martyna--Tobias--Klein (MTK) barostat with Nose--Hoover chains to sample the
isothermal-isobaric ensemble. Both the atomic positions and the simulation cell
evolve:

```python
from nvalchemi.dynamics import NPT

with NPT(
    model=model,
    dt=1.0,
    temperature=300.0,
    pressure=1.0,            # target pressure (eV/Å^3; positive = compression)
    barostat_time=100.0,     # barostat coupling time (fs)
    thermostat_time=100.0,   # thermostat coupling time (fs)
    n_steps=10000,
) as md:
    trajectory = md.run(batch)
```

The model must return `stress` for NPT to propagate the cell degrees of freedom.

## Monte Carlo

The independent `nvalchemi.mc` package provides batched Monte Carlo samplers.
`SGC` proposes single-site species transmutations and samples the
semi-grand-canonical potential `E - sum(mu_i N_i)`:

```python
from nvalchemi.mc import SGC

with SGC(
  model=model,
  temperature=1000.0,
  species=[1, 2],
  chemical_potentials={1: 0.0, 2: 0.2},
  n_steps=10000,
) as mc:
  result = mc.run(batch)

```

One proposal is attempted independently for every active graph per step.
Accepted moves are available as the graph-level boolean `batch.mc_accepted`.

### Variance-constrained SGC

`VCSGC` (variance-constrained SGC, binary systems) uses the same transmutation
move but replaces the linear reservoir with a quadratic constraint on the
concentration `c` of one species, sampling `E + N (phi c + kappa c^2)`. Plain
SGC cannot hold a composition inside a miscibility gap, because the
concentration jumps to one side; `VCSGC` can, and the slope of the free energy
at the sampled mean `cbar` is `mu_B - mu_A = reference - phi - 2 kappa cbar`.
Integrating that slope over a grid of target compositions gives `F(c)` for a
common-tangent construction:

```python
from nvalchemi.mc import VCSGC

mc = VCSGC(
  model=model,
  temperature=700.0,
  species=[79, 78],            # c is the fraction of species[1] (Pt) by default
  kappa=1.0,                   # eV, intensive
  target_concentration=0.4,    # sets phi = -2 * kappa * c0
  reference_exchange_potential=-2.87,  # calibrated mu_Pt - mu_Au at this T, eV
)
mc.run(batch, n_steps=2000)                  # equilibrate
c_sum, n_samples = 0.0, 100
for _ in range(n_samples):
  mc.run(batch, n_steps=100)
  c_sum = c_sum + mc.concentration(batch)    # per-graph instantaneous c
cbar = c_sum / n_samples                     # the relation needs the ensemble mean
delta_mu = mc.exchange_chemical_potential(cbar, batch)
```

Choose `kappa > -min f''(c) / 2`, where `f` is the free energy per atom:
below that the composition distribution turns bimodal while `cbar` can still
look right, so check that the sampled concentration is unimodal before using
it. `kappa = 0` recovers `SGC` with `mu_B - mu_A = reference - phi`.

Always pass `reference_exchange_potential` for a machine-learned potential.
Per-element energy offsets put the system's own `mu_B - mu_A` electronvolts
away from zero, while the constraint can supply at most `2 kappa`, so without
the calibrated reference (the same one an SGC scan is centred on) the walker
runs to the favoured end member instead of sampling near the target.

### Kawasaki and NonlocalSwap: fixed composition

Two canonical (fixed-composition) samplers swap species instead of
transmuting them; neither needs a chemical potential or a reference.

`Kawasaki` swaps the species of one nearest-neighbour pair per graph (by
default only unlike pairs, so no move is wasted), so solutes move one bond at a
time. The pairs come from a short-range proposal graph,
`Kawasaki(model=model, temperature=1000.0, cutoff=3.4)`, built independently of
the model's own neighbour list; call `synchronize(batch)` after positions change
outside the sampler (e.g. after an MD block) so the graph is rebuilt.

`NonlocalSwap` exchanges two unlike atoms anywhere in the graph, with no
neighbour graph, so one accepted move can take a solute from a nanoparticle's
core to its surface. Its draw (a uniform atom, then a uniform atom of another
species) is symmetric at fixed composition, so plain Metropolis is exact:

```python
from nvalchemi.mc import NonlocalSwap

mc = NonlocalSwap(model=model, temperature=400.0, random_seed=1)
```

### Species, masses and fused stages

Every sampler updates `atomic_masses` for each atom whose species an accepted
move changes, so MD run afterwards always integrates with the current species'
mass. MC samplers also compose with MD in a `FusedStage` (`mc + md`), which
runs MC on some graphs and MD on others, selected by `status`, sharing one
forward pass per step. MC takes its acceptance baseline from the energy on the
batch, so a graph that migrates into the MC stage needs no extra call. A graph
migrating out of it after an accepted trial carries that trial's energy and
forces, which describe its current state, into MD. After a rejected trial the
carried forces describe the undone configuration, so the graph sits out one MD
update while the shared forward pass re-primes it (no extra model call).

### UMA settings for Monte Carlo

Monte Carlo reads energies only, so set
`model.set_config("active_outputs", {"energy"})`: `UMAWrapper` then skips the
forces and stress backward pass. For runs where composition changes (SGC,
VC-SGC) or where systems of different composition share a batch, none of the
named presets fits: `"default"` and `"turbo"` merge the mixture-of-experts
weights for one composition, and `"batch"` turns on activation checkpointing.
Pass a `key=value` spec instead:

```python
from nvalchemi.models.uma import UMAWrapper

model = UMAWrapper.from_checkpoint(
    "uma-s-1p2",
    task_name="omat",
    inference_settings="compile=false,merge_mole=false,tf32=true,activation_checkpointing=false",
)
model.set_config("active_outputs", {"energy"})
```

On Au-Pt, 500 atoms, 4 systems per batch on an A100-SXM4-80GB, this ran SGC at
24.5 s per 100-step block and 5.0 GiB peak, against 67.3 s and 9.2 GiB for
`"batch"` (2.75x), with the identical sampled chain. Leave `compile` off for
composition-changing MC unless you have measured it on your hardware.

When composition cannot change (`Kawasaki`, `NonlocalSwap`, MD with one
composition per batch), `merge_mole=True` folds the experts into one plain
model and is the largest single speed-up, and `"turbo"` adds compilation for
long runs. A merged model is valid for one composition only: every later
structure, and every system in a batch, must have the composition it was merged
on.

## Writing your own dynamics

All integrators and optimizers inherit from
{py:class}`~nvalchemi.dynamics.base.BaseDynamics`. To implement a custom one, you
subclass it and override `pre_update` and `post_update` --- the two methods that
define how the batch state evolves within a single step.

### The minimal contract

Your subclass must provide:

1. **`__needs_keys__`** --- a set of strings naming the model outputs your dynamics
   reads (e.g. `{"forces"}`, or `{"forces", "stress"}` for cell-aware schemes).
2. **`__provides_keys__`** --- a set of strings naming the batch keys your dynamics
   writes (e.g. `{"positions", "velocities"}`).
3. **`pre_update(batch)`** --- called *before* the model forward pass. Typically
   updates positions using current velocities and/or forces.
4. **`post_update(batch)`** --- called *after* the model forward pass. Typically
   completes the velocity update with the newly computed forces.

Both methods receive the {py:class}`~nvalchemi.data.Batch` and modify it
**in-place**. Return value is `None`.

### Example: a Velocity Verlet integrator

The `DemoDynamics` class in `nvalchemi.dynamics.demo` is a complete, minimal
Velocity Verlet implementation that is useful as a template:

```python
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook

class MyVerlet(BaseDynamics):
    __needs_keys__ = {"forces"}
    __provides_keys__ = {"positions", "velocities"}

    def __init__(self, model, n_steps, dt=1.0, hooks=None, convergence_hook=None, **kwargs):
        super().__init__(
            model=model, hooks=hooks, convergence_hook=convergence_hook,
            n_steps=n_steps, **kwargs,
        )
        self.dt = dt
        self._prev_accelerations = None

    def pre_update(self, batch: Batch) -> None:
        """Position half-step: x(t+dt) = x(t) + v*dt + 0.5*a*dt^2."""
        import torch
        positions = batch.positions
        velocities = batch.velocities
        forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)

        with torch.no_grad():
            if forces is not None and not torch.all(forces == 0):
                acc = forces / masses
                self._prev_accelerations = acc.clone()
                positions.add_(velocities * self.dt + 0.5 * acc * self.dt**2)
            else:
                positions.add_(velocities * self.dt)

    def post_update(self, batch: Batch) -> None:
        """Velocity half-step: v(t+dt) = v(t) + 0.5*(a_old + a_new)*dt."""
        import torch
        velocities = batch.velocities
        forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)

        with torch.no_grad():
            new_acc = forces / masses
            if self._prev_accelerations is not None:
                velocities.add_(0.5 * (self._prev_accelerations + new_acc) * self.dt)
            else:
                velocities.add_(new_acc * self.dt)
```

```{important}
The demo ``DemoDynamics`` class is intended for debugging and pedagogy
only. Do not use this class for production runs, and instead, see the
{py:class}`~nvalchemi.dynamics.integrators.nve.NVE` class instead.
```

### Data flow through a step

Understanding what the batch contains at each point is key to writing correct
updates:

| Point in step | What just happened | What the batch contains |
|---------------|--------------------|-------------------------|
| `pre_update` entry | Hooks ran | Positions and velocities from the *previous* step; forces may be from the previous `compute` (or absent on step 0) |
| `pre_update` exit | You updated positions | New positions; velocities partially updated (or unchanged) |
| After `compute` | Model ran | Fresh `forces` (and `energy`, `stress`, etc.) for the new positions |
| `post_update` entry | Forces are fresh | Complete the velocity update with new forces |
| `post_update` exit | Step is done | Consistent positions, velocities, and forces for the current timestep |

### Gotchas and tips

- **Use `torch.no_grad()`**: Wrap in-place updates in `torch.no_grad()` to avoid
  conflicts with autograd. When `forces_via_autograd=True`, `compute()` sets
  `requires_grad_(True)` on positions to compute forces via backprop.
- **In-place operations**: Modify batch tensors in-place (`positions.add_(...)`)
  rather than reassigning. The batch's storage model expects tensors to be updated
  in place.
- **First-step fallback**: On the first call to `pre_update`, forces may be `None`
  or zero (no model evaluation has happened yet). Guard against this and fall back
  to an Euler step.
- **Per-system state**: If your integrator needs auxiliary state (e.g. thermostat
  variables, previous accelerations), store it as instance attributes. The
  `_prev_accelerations` pattern above is typical.
- **`__needs_keys__` matters**: `BaseDynamics` uses this set to verify the model
  produces the required outputs before the simulation starts. If your dynamics needs
  stress, declare `{"forces", "stress"}`.
- **FusedStage compatibility**: When your dynamics runs inside a
  {py:class}`~nvalchemi.dynamics.base.FusedStage`, a save-and-restore mask is
  applied around `pre_update` and `post_update` so that only systems belonging to
  your stage are modified. You do not need to handle masking yourself.
- **Running under domain decomposition**: A per-atom integrator works under
  {py:class}`~nvalchemi.distributed.DomainParallel` unchanged, but any *global*
  reduction (kinetic energy, temperature, a convergence dot-product) needs
  cross-rank handling. See {doc}`distributed` → *Distributed dynamics* for the
  contract and the `HookScope.GLOBAL` recipe.

## See also

- **Overview**: The [Dynamics overview](dynamics_guide) describes the shared execution
  loop and multi-stage pipelines.
- **Hooks**: The [Hooks guide](hooks_guide) covers convergence criteria,
  logging, and snapshots.
- **Reaction paths**: [Reaction Paths and NEB](dynamics_mep_guide) runs batched
  nudged elastic band calculations with a configurable optimizer.
- **Examples**: ``basic/02_geometry_optimization.py`` demonstrates a complete relaxation
  workflow.
