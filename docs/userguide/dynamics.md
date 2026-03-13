<!-- markdownlint-disable MD014 -->

(dynamics_guide)=

# Dynamics: Optimization and Molecular Dynamics

The dynamics module provides a unified framework for running geometry optimizations
and molecular dynamics simulations on GPU. All simulation types share a common
execution loop --- hooks, model evaluation, convergence checking --- so you learn the
pattern once and apply it to any integrator.

```{tip}
It is important to keep in mind that ``nvalchemi`` follows a batch-first principle:
users should think and reason about dynamics workflows with multiple structures
simultaneously, as opposed to individual structures being processed sequentially.
```

## The execution loop

Every simulation is driven by {py:class}`~nvalchemi.dynamics.base.BaseDynamics`,
which defines a single `step()` that all integrators and optimizers follow. The
loop is broken into discrete stages, enumerated by
{py:class}`~nvalchemi.dynamics.base.HookStageEnum`:

| Stage | When it fires |
|-------|---------------|
| `BEFORE_STEP` | At the very beginning of a step, before any operations |
| `BEFORE_PRE_UPDATE` | Just before the integrator's first half-step |
| `AFTER_PRE_UPDATE` | After the first half-step completes |
| `BEFORE_COMPUTE` | Just before the model forward pass |
| `AFTER_COMPUTE` | After the model forward pass completes |
| `BEFORE_POST_UPDATE` | Just before the integrator's second half-step |
| `AFTER_POST_UPDATE` | After the second half-step completes |
| `AFTER_STEP` | At the very end of a step, after all operations |
| `ON_CONVERGE` | When a convergence criterion is met |

A single call to `step()` proceeds through these stages in order:

1. **BEFORE_STEP** hooks fire.
2. `pre_update(batch)` --- the integrator's first half-step (e.g. update velocities
   by half a timestep), bracketed by BEFORE/AFTER_PRE_UPDATE hooks.
3. `compute(batch)` --- the wrapped ML model evaluates forces (and stresses, if
   needed), bracketed by BEFORE/AFTER_COMPUTE hooks.
4. `post_update(batch)` --- the integrator's second half-step (e.g. complete the
   velocity update with the new forces), bracketed by BEFORE/AFTER_POST_UPDATE hooks.
5. **AFTER_STEP** hooks fire (convergence checks, logging, ...).
6. Convergence is evaluated: converged systems fire **ON_CONVERGE** hooks and (in
   multi-stage pipelines) migrate to the next stage.

`run(batch, n_steps)` calls `step()` in a loop until all systems converge or
`n_steps` is reached. Every hook declares which
{py:class}`~nvalchemi.dynamics.base.HookStageEnum` stage it should fire at and at
what frequency, so you have fine-grained control over when callbacks execute.

## Using dynamics as a context manager

All dynamics objects (optimizers, integrators, fused stages) support Python's
context manager protocol. The `with` block manages a dedicated
`torch.cuda.Stream` for the simulation and ensures hooks are properly opened and
closed:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook

with FIRE(model=model, dt=0.1, n_steps=500, hooks=[ConvergenceHook(fmax=0.05)]) as opt:
    relaxed = opt.run(batch)
```

When you call `run()` without a `with` block, hook setup and teardown happen
automatically inside `run()`. The context manager form is useful when you need to
call `step()` manually or interleave dynamics with other operations while keeping
hook state (e.g. open log files) alive.

## Multi-stage pipelines with FusedStage

Real workflows often chain multiple simulation phases: relax a structure, then run
MD at increasing temperatures, then relax again. The
{py:class}`~nvalchemi.dynamics.base.FusedStage` abstraction lets you compose stages
with the `+` operator:

```python
from nvalchemi.dynamics import FIRE, NVTLangevin, ConvergenceHook

relax = FIRE(model=model, dt=0.1, n_steps=200, hooks=[ConvergenceHook(fmax=0.05)])
md = NVTLangevin(model=model, dt=1.0, temperature=300.0, n_steps=5000)

pipeline = relax + md
with pipeline:
  pipeline.run(batch)
```

Systems start in the first stage (relaxation). As each system converges, it
automatically migrates to the next stage (MD). Different systems can be in different
stages simultaneously --- the batch is partitioned internally, and a single model
forward pass is shared across all active systems regardless of which stage they
belong to.

## What's next

```{toctree}
:maxdepth: 1

dynamics_simulations
dynamics_hooks
dynamics_sinks
```

- [Optimization and Integrators](dynamics_simulations) --- FIRE, NVE, NVT, NPT and
  their configuration.
- [Hooks](dynamics_hooks) --- the hook protocol, built-in hooks, and writing custom
  hooks.
- [Data Sinks](dynamics_sinks) --- recording trajectories and simulation results.

## See also

- **Examples**: ``02_dynamics_example.py`` demonstrates a complete relaxation and MD
  workflow.
- **API**: See the {py:mod}`nvalchemi.dynamics` module for the full reference,
  including the hook protocol and distributed pipeline documentation.
- **Data guide**: The [AtomicData and Batch](data_guide) guide covers the input data
  structures consumed by dynamics.
