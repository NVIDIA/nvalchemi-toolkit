<!-- markdownlint-disable MD014 -->

(dynamics_simulations_guide)=

# Optimization and Integrators

This page covers the concrete simulation types provided by the dynamics module.
All of them follow the [execution loop](dynamics_guide) described in the dynamics
overview --- they generally differ only in what `pre_update` and `post_update` do.

## Geometry optimization

Geometry optimization finds the nearest local energy minimum by iteratively moving
atoms downhill on the potential energy surface. The toolkit provides the **FIRE**
(Fast Inertial Relaxation Engine) algorithm in two variants.

### Fixed-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire.FIRE` optimizes atomic positions
while keeping the simulation cell fixed:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook

with FIRE(
    model=model,
    dt=0.1,           # initial timestep (femtoseconds)
    n_steps=500,
    hooks=[ConvergenceHook(fmax=0.05)],
) as opt:
    relaxed = opt.run(batch)
```

FIRE uses an adaptive timestep and velocity mixing: when the system is moving
downhill (forces aligned with velocities), the timestep grows and velocities are
biased toward the force direction. When the system overshoots, the timestep shrinks
and velocities are zeroed. This makes it robust across a wide range of systems
without manual tuning.

### Variable-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire.FIREVariableCell` extends FIRE to
simultaneously optimize both atomic positions and the simulation cell. This is
useful for finding equilibrium crystal structures where the lattice parameters are
not known a priori:

```python
from nvalchemi.dynamics.optimizers.fire import FIREVariableCell
from nvalchemi.dynamics import ConvergenceHook

with FIREVariableCell(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[ConvergenceHook(fmax=0.05)],
) as opt:
    relaxed = opt.run(batch)
```

The cell degrees of freedom are propagated using an NPH-like scheme at zero target
pressure. The model must return `stresses` (or `virials`) in addition to `forces`.

### Choosing between fixed and variable cell

Use fixed-cell FIRE when the cell is known (e.g. a bulk crystal at experimental
lattice parameters, or a molecule in vacuum where the cell is just a bounding box).
Use variable-cell FIRE when the equilibrium cell shape or volume is unknown, such as
when screening candidate crystal structures or computing equations of state.

## Molecular dynamics

Molecular dynamics (MD) propagates the equations of motion forward in time, sampling
the trajectory of a system at finite temperature. The toolkit provides integrators
for three standard ensembles.

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
    pressure=1.0,            # target pressure (bar)
    n_steps=10000,
) as md:
    trajectory = md.run(batch)
```

The model must return `stresses` for NPT to propagate the cell degrees of freedom.

## Writing your own dynamics

All integrators and optimizers inherit from
{py:class}`~nvalchemi.dynamics.base.BaseDynamics`. To implement a custom one, you
subclass it and override `pre_update` and `post_update` --- the two methods that
define how the batch state evolves within a single step.

### The minimal contract

Your subclass must provide:

1. **`__needs_keys__`** --- a set of strings naming the model outputs your dynamics
   reads (e.g. `{"forces"}`, or `{"forces", "stresses"}` for cell-aware schemes).
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
The demo ``VelocityVerlet`` class is intended for debugging and pedagogy
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
| After `compute` | Model ran | Fresh `forces` (and `energies`, `stresses`, etc.) for the new positions |
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
  stresses, declare `{"forces", "stresses"}`.
- **FusedStage compatibility**: When your dynamics runs inside a
  {py:class}`~nvalchemi.dynamics.base.FusedStage`, a save-and-restore mask is
  applied around `pre_update` and `post_update` so that only systems belonging to
  your stage are modified. You do not need to handle masking yourself.

## See also

- **Overview**: The [Dynamics overview](dynamics_guide) describes the shared execution
  loop and multi-stage pipelines.
- **Hooks**: The [Hooks guide](dynamics_hooks_guide) covers convergence criteria,
  logging, and snapshots.
- **Examples**: ``02_dynamics_example.py`` demonstrates a complete relaxation and MD
  workflow.
