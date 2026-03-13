<!-- markdownlint-disable MD014 -->

(dynamics_guide)=

# Dynamics: Optimization and Molecular Dynamics

The dynamics module provides a unified framework for running geometry optimizations
and molecular dynamics simulations on GPU. All simulation types share a common
execution loop --- hooks, model evaluation, convergence checking --- so you learn the
pattern once and apply it to any integrator.

.. tip::

  It is important to keep in mind that ``nvalchemi`` follows a batch-first principle:
  users should think and reason about dynamics workflows with multiple structures
  simultaneously, as opposed to individual structures being processed sequentially.

## The execution loop

Every simulation is driven by {py:class}`~nvalchemi.dynamics.base.BaseDynamics`,
which defines a single `step()` that all integrators and optimizers follow:

1. **BEFORE_STEP** hooks run (logging, snapshots, ...).
2. `pre_update(batch)` --- the integrator's first half-step (e.g. update velocities
   by half a timestep).
3. `compute(batch)` --- the wrapped ML model evaluates forces (and stresses, if
   needed).
4. `post_update(batch)` --- the integrator's second half-step (e.g. complete the
   velocity update with the new forces).
5. **AFTER_STEP** hooks run (convergence checks, more logging, ...).
6. Convergence is evaluated: any system that satisfies its convergence criterion is
   marked done and (in multi-stage pipelines) migrates to the next stage.

`run(batch, n_steps)` simply calls `step()` in a loop until all systems converge or
`n_steps` is reached.

## Geometry optimization

Geometry optimization finds the nearest local energy minimum by iteratively moving
atoms downhill on the potential energy surface. The toolkit provides the **FIRE**
(Fast Inertial Relaxation Engine) algorithm in two variants.

### Fixed-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire.FIRE` optimizes atomic positions
while keeping the simulation cell fixed:

```python
from nvalchemi.dynamics import FIRE

opt = FIRE(
    model=model,
    dt=0.1,           # initial timestep (femtoseconds)
    n_steps=500,
)
relaxed = opt.run(batch)
```

FIRE uses an adaptive timestep and velocity mixing: when the system is moving
downhill (forces aligned with velocities), the timestep grows and velocities are
biased toward the force direction. When the system overshoots, the timestep shrinks
and velocities are zeroed. This makes it robust across a wide range of systems
without manual tuning.

Convergence is typically controlled by a
{py:class}`~nvalchemi.dynamics.hooks.ConvergenceHook` that checks the maximum
force magnitude:

```python
from nvalchemi.dynamics.hooks import ConvergenceHook

opt = FIRE(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[ConvergenceHook(fmax=0.05)],
)
```

### Variable-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire.FIREVariableCell` extends FIRE to
simultaneously optimize both atomic positions and the simulation cell. This is
useful for finding equilibrium crystal structures where the lattice parameters are
not known a priori:

```python
from nvalchemi.dynamics.optimizers.fire import FIREVariableCell

opt = FIREVariableCell(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[ConvergenceHook(fmax=0.05)],
)
relaxed = opt.run(batch)
```

The cell degrees of freedom are propagated using an NPH-like scheme at zero target
pressure. The model must return `stresses` (or `virials`) in addition to `forces`.

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

md = NVE(model=model, dt=1.0, n_steps=1000)
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

md = NVTLangevin(
    model=model,
    dt=1.0,              # femtoseconds
    temperature=300.0,    # Kelvin
    friction=0.01,        # collision frequency (1/fs)
    n_steps=10000,
)
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

md = NPT(
    model=model,
    dt=1.0,
    temperature=300.0,
    pressure=1.0,            # target pressure (bar)
    n_steps=10000,
)
trajectory = md.run(batch)
```

The model must return `stresses` for NPT to propagate the cell degrees of freedom.

## Hooks

Hooks observe or modify the batch at specific points in the simulation loop, without
touching the integrator code. The toolkit ships several built-in hooks:

| Hook | Purpose |
|------|---------|
| {py:class}`~nvalchemi.dynamics.hooks.ConvergenceHook` | Marks systems as converged when a criterion is met (e.g. `fmax < threshold`) |
| {py:class}`~nvalchemi.dynamics.hooks.LoggingHook` | Records scalar observables (energy, temperature, fmax) per step |
| {py:class}`~nvalchemi.dynamics.hooks.SnapshotHook` | Saves the full batch state to a data sink at a given interval |
| {py:class}`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` | Saves only systems that just converged in the current step |

Hooks are passed as a list at construction time:

```python
from nvalchemi.dynamics.hooks import ConvergenceHook, LoggingHook, SnapshotHook
from nvalchemi.dynamics.sinks import ZarrData

opt = FIRE(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[
        ConvergenceHook(fmax=0.05),
        LoggingHook(interval=10),
        SnapshotHook(sink=ZarrData("/tmp/traj.zarr"), interval=50),
    ],
)
```

## Data sinks

Snapshot hooks need somewhere to write data. A **sink** is a pluggable storage
backend:

- {py:class}`~nvalchemi.dynamics.sinks.GPUBuffer` --- keeps snapshots in GPU memory
  for maximum throughput (useful for short trajectories or inter-stage communication).
- {py:class}`~nvalchemi.dynamics.sinks.HostMemory` --- stages snapshots in host RAM.
- {py:class}`~nvalchemi.dynamics.sinks.ZarrData` --- writes snapshots to a
  persistent Zarr store on disk (recommended for long trajectories and
  post-processing).

## Multi-stage pipelines with FusedStage

Real workflows often chain multiple simulation phases: relax a structure, then run
MD at increasing temperatures, then relax again. The
{py:class}`~nvalchemi.dynamics.base.FusedStage` abstraction lets you compose stages
with the `+` operator:

```python
from nvalchemi.dynamics import FIRE, NVTLangevin

relax = FIRE(model=model, dt=0.1, n_steps=200, hooks=[ConvergenceHook(fmax=0.05)])
md = NVTLangevin(model=model, dt=1.0, temperature=300.0, n_steps=5000)

pipeline = relax + md
pipeline.run(batch)
```

Systems start in the first stage (relaxation). As each system converges, it
automatically migrates to the next stage (MD). Different systems can be in different
stages simultaneously --- the batch is partitioned internally, and a single model
forward pass is shared across all active systems regardless of which stage they
belong to.

## See also

- **Examples**: ``02_dynamics_example.py`` demonstrates a complete relaxation and MD
  workflow.
- **API**: See the {py:mod}`nvalchemi.dynamics` module for the full reference,
  including the hook protocol and distributed pipeline documentation.
- **Data guide**: The [AtomicData and Batch](data_guide) guide covers the input data
  structures consumed by dynamics.
