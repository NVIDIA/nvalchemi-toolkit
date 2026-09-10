<!-- markdownlint-disable MD014 -->

(dynamics_paths_guide)=

# Reaction Paths and NEB

The `nvalchemi.dynamics.paths` subpackage runs batched Nudged Elastic Band
(NEB) calculations to find minimum-energy reaction paths between two
endpoint structures. Like every other simulation type, it follows the
[execution loop](dynamics_guide): a path is just a batch where each graph is
one image, and images belonging to the same path share a `group_layout`
group.

There are two ways to use it:

- The **{py:class}`~nvalchemi.dynamics.paths.NEB` strategy** --- a
  declarative, serializable wrapper that builds a correctly-ordered engine
  for you. Use this for standard regular or climbing-image NEB runs.
- **An optimizer (or `FusedStage`) with the NEB hooks attached directly**
  --- for full control over hook ordering, custom convergence criteria, or
  non-standard staging.

## Building the initial path

Whether you use the high-level `NEB` strategy or the low-level `FusedStage`
approach, the input is the same: a `Batch` of images with `group_layout`
set, one group per path, at least three images each, matching atoms/cell
across images (see {py:func}`~nvalchemi.dynamics.paths.validate_paths`). You
can build this batch yourself, or use the two basic interpolation utilities
below to go from just reactant/product endpoints:

```python
from nvalchemi.dynamics.paths import interpolate_paths, IDPPModel, prepare_idpp_targets, NEB

paths = interpolate_paths(initial, final, num_images=7)  # linear interpolation

# optional: refine with IDPP before switching to the real model
prepare_idpp_targets(paths)
paths = NEB(model=IDPPModel(), fmax=0.1, n_steps=200).run(paths)
```

`interpolate_paths` linearly interpolates positions between index-matched
endpoint graphs and drops stale fields (`forces`, `energy`, `velocities`,
...) --- reattach `velocities` before optimizing.
{py:func}`~nvalchemi.dynamics.paths.prepare_idpp_targets` +
{py:class}`~nvalchemi.dynamics.paths.IDPPModel` relax the path against target
pairwise distances (IDPP,
[Smidstrup et al. 2014](https://doi.org/10.1063/1.4878664)) to avoid poor
linear-interpolation guesses where atoms pass through each other.
`IDPPModel` is just a `BaseModelMixin` like any other, so running it through
`NEB` (as above) is the normal way to do this refinement --- it is a regular
NEB optimization whose "physical" model happens to be the analytic IDPP
potential instead of your real MLIP. See
``examples/advanced/16_batched_neb.py`` for the full IDPP-then-MLIP workflow.

## The high-level `NEB` strategy

```python
from nvalchemi.dynamics.paths import NEB, ClimbingImageConfig

neb = NEB(
    model=model,
    spring=0.1,                  # constant spring force (eV/Å); or a SpringConfig
    method="improved_tangent",   # tangent + spring-force formulation
    climbing=ClimbingImageConfig(mode="after_regular", regular_fmax=0.1),
    fmax=0.05,                   # force convergence threshold
    n_steps=500,
    diagnostics_log_path="neb_diagnostics.csv",  # per-path fmax/barrier/length CSV
)
relaxed_paths = neb.run(paths)
```

Key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `spring` | `0.1` | Constant spring force, or a custom {py:class}`~nvalchemi.dynamics.paths.SpringConfig` |
| `method` | `"improved_tangent"` | Named or custom `NEBMethod` (tangent, effective force, climbing force) |
| `climbing` | `None` | `None` runs regular NEB only; set a `ClimbingImageConfig` to enable climbing-image NEB |
| `optimizer` / `optimizer_kwargs` | `FIRE2` | Optimizer driving each image; kwargs forwarded to it |
| `fmax` | `0.05` | Force threshold for the final (or only) stage |
| `endpoint_mode` | `"fixed"` | `"fixed"` keeps endpoints static; `"relaxed"` lets them move |
| `fixed_atom_indices` | `None` | Per-path, image-local atom indices held fixed in every image |
| `diagnostics_log_path` | `None` | CSV path for per-path `fmax`, `energy_barrier`, `highest_interior_image_idx`, `path_length` |
| `diagnostics_frequency` | `1` | Step frequency for computing and logging diagnostics |
| `neighbor_hooks` | `None` | Override `model.make_neighbor_hooks()` with a custom neighbor-list hook list |
| `compile` | `False` | Compile the fused NEB step with `torch.compile` |

`endpoint_mode="fixed"` and `fixed_atom_indices` are both enforced by a
{py:class}`~nvalchemi.dynamics.hooks.FreezeAtomsHook` that `NEB` appends
automatically --- `NEBForceHook` only computes *which* atoms are fixed (as a
`neb_fixed_node_mask` batch field); `FreezeAtomsHook` is what actually zeroes
their forces/velocities and restores their positions across integrator
stages.

`ClimbingImageConfig.mode="after_regular"` runs regular NEB to
`regular_fmax` (or `fmax` if unset) before promoting the highest-energy
interior image to a climbing image and continuing to `fmax`;
`mode="immediate"` climbs from the first step.
`selection="fixed"` locks in the initial climbing image;
`selection="dynamic"` reselects it every evaluation.

`neb.run(paths)` builds a fresh engine via `neb.build_engine()` and calls
`engine.run(paths)`. `NEB` is a Pydantic model, so it also serializes with
`to_spec_dict()` / `from_spec_dict()` alongside other
{py:class}`~nvalchemi.dynamics.strategy.DynamicsStrategy` subclasses used for
training and fine-tuning specs.

### The `method` equations

`method="improved_tangent"` (the default and only named, serializable method)
implements the Henkelman--Jónsson improved tangent
([*J. Chem. Phys.* 113, 9978 (2000)](https://doi.org/10.1063/1.1323224)).
For an interior image with neighbor energies $E_-, E, E_+$ and
displacement vectors $\mathbf{d}_\pm$ to the adjacent images:

- **Tangent**: $\boldsymbol{\tau} = w_+\mathbf{d}_+ + w_-\mathbf{d}_-$
  (then normalized), where the weights $w_\pm$ favor the higher-energy
  neighbor on monotonic sections of the path and blend both links at an
  extremum.
- **Effective (regular) force**:
  $\mathbf{F}^{\mathrm{NEB}} = \mathbf{F} - (\mathbf{F}\cdot\hat{\boldsymbol{\tau}})\hat{\boldsymbol{\tau}} + F^{\mathrm{s}}_{\parallel}\hat{\boldsymbol{\tau}}$,
  i.e. the physical force projected perpendicular to the tangent, plus a
  spring force along it, $F^{\mathrm{s}}_{\parallel} = k_+\lVert\mathbf{d}_+\rVert - k_-\lVert\mathbf{d}_-\rVert$.
- **Climbing-image force**:
  $\mathbf{F}^{\mathrm{climb}} = \mathbf{F} - 2(\mathbf{F}\cdot\hat{\boldsymbol{\tau}})\hat{\boldsymbol{\tau}}$
  --- the spring force is dropped and the parallel physical-force component
  is reversed, driving the image uphill along the path toward the saddle
  point.

To use a different formulation, pass a custom
{py:class}`~nvalchemi.dynamics.paths.NEBMethod` to `method`. It bundles three
`warp.func`-decorated device functions, any of which can be overridden
independently (unset ones fall back to the improved-tangent equations above):

| Field | Inputs available | Returns |
|-------|-------------------|---------|
| `tangent_weights_fn` | `energy_prev`, `energy_curr`, `energy_next` (scalars) | `(weight_plus, weight_minus)` link weights for the tangent |
| `effective_force_fn` | `physical_force`, `tangent`, the forward/backward link vectors and norms (`d_plus`, `d_minus`, `norm_d_plus`, `norm_d_minus`), their dot products with the force and each other, `force_squared_norm`, spring constants `k_plus`/`k_minus`, the three neighbor energies, and `path_energy_ref`/`path_energy_max` | effective force vector for a regular (non-climbing) interior image |
| `climbing_force_fn` | `physical_force`, `tangent`, `force_dot_tangent` | effective force vector for the climbing image |

```python
import warp as wp
from nvalchemi.dynamics.paths import NEB, NEBMethod

@wp.func
def central_tangent_weights(energy_prev: float, energy_curr: float, energy_next: float):
    return 1.0, 1.0  # plain central-difference tangent instead of improved-tangent weighting

neb = NEB(model=model, method=NEBMethod(tangent_weights_fn=central_tangent_weights), fmax=0.05)
```

The full Gram-statistics contract exists so `effective_force_fn` can go
beyond the stored-tangent projection above --- for example, to implement
doubly-nudged elastic band (DNEB,
[Trygubenko & Wales 2004](https://doi.org/10.1063/1.1636455)).

A custom `NEBMethod` without a `name` is runtime-only (it cannot round-trip
through `to_spec_dict()`); giving it a stable, registered `name` is required
for serialization and for use under `torch.compile`/CUDA graph capture.
`spring` follows the same pattern: pass a plain `float` for a constant spring
constant, or a custom {py:class}`~nvalchemi.dynamics.paths.SpringConfig`
(`resolve(context)` returning one spring constant per link) for e.g.
energy-dependent springs.

## Building NEB manually with hooks

`NEB` is a factory around a fixed hook order. Assembling it by hand gives you
control over convergence criteria, staging, or optimizer choice beyond what
the strategy exposes. For regular (non-climbing) NEB there is only one stage,
so any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` subclass that
accepts `by_group=True` and a `hooks` list is enough on its own --- no
{py:class}`~nvalchemi.dynamics.FusedStage` wrapper required. `FusedStage` is
only needed once you have more than one stage, e.g. climbing-image NEB (see
below). The example uses {py:class}`~nvalchemi.dynamics.optimizers.fire2.FIRE2`.

```python
from nvalchemi.dynamics import ConvergenceHook
from nvalchemi.dynamics.hooks import FreezeAtomsHook, LoggingHook
from nvalchemi.dynamics.optimizers.fire2 import FIRE2
from nvalchemi.dynamics.paths.hooks import PathEnergyStatsHook, PathDiagnosticsHook
from nvalchemi.dynamics.paths.neb.hooks import NEBForceHook, ClimbingImageSelectionHook

energy_stats = PathEnergyStatsHook()
force_hook = NEBForceHook(
    energy_stats_hook=energy_stats,
    spring=0.1,
    method="improved_tangent",
    endpoint_mode="fixed",
)
freeze = FreezeAtomsHook(mask_key="neb_fixed_node_mask")
diagnostics = PathDiagnosticsHook(energy_stats_hook=energy_stats)
logger = LoggingHook(
    backend="csv",
    log_path="neb_diagnostics.csv",
    by_group=True,
    custom_scalars={
        "fmax": lambda _ctx: diagnostics.get_diagnostics().fmax,
        "energy_barrier": lambda _ctx: diagnostics.get_diagnostics().energy_barrier,
        "path_length": lambda _ctx: diagnostics.get_diagnostics().path_length,
    },
)

optimizer = FIRE2(
    model=model,
    dt=0.01,
    n_steps=500,
    by_group=True,
    convergence_hook=ConvergenceHook.from_fmax(threshold=0.05, by_group=True),
    hooks=[
        *model.make_neighbor_hooks(),
        energy_stats,
        force_hook,
        freeze,
        diagnostics,
        logger,
    ],
)
relaxed_paths = optimizer.run(paths)
```

The NEB path hooks, in the order they must be registered:

1. **{py:class}`~nvalchemi.dynamics.paths.hooks.PathEnergyStatsHook`** ---
   always first. Computes per-path endpoint reference energy and the
   highest-energy interior image; every other path hook reads its results
   via `get_stats()`.
2. **{py:class}`~nvalchemi.dynamics.paths.neb.hooks.ClimbingImageSelectionHook`**
   (optional) --- flips the selected image's `force_mode` to climbing before
   `NEBForceHook` runs. Needed only for climbing-image NEB.
3. **{py:class}`~nvalchemi.dynamics.paths.neb.hooks.NEBForceHook`** --- the
   core hook: computes the local tangent and replaces `batch.forces` with the
   spring + perpendicular-physical-force NEB update, saving the raw model
   force to `batch.physical_forces`. When `endpoint_mode="fixed"` or
   `fixed_atom_indices` is set, it also writes a `neb_fixed_node_mask` batch
   field marking which atoms should stay put --- it does not enforce that
   itself.
4. **{py:class}`~nvalchemi.dynamics.hooks.FreezeAtomsHook`** (optional, needed
   whenever step 3 produces a fixed-atom mask) --- a general-purpose,
   non-NEB-specific hook (`nvalchemi.dynamics.hooks`) that zeroes forces and
   velocities and restores positions for masked atoms across every integrator
   stage. Pass `mask_key="neb_fixed_node_mask"` to use the mask from
   `NEBForceHook` instead of the default `atom_categories`-based freezing.
5. **{py:class}`~nvalchemi.dynamics.paths.hooks.PathDiagnosticsHook`**
   (optional) --- exposes per-path `fmax`, `energy_barrier`, and
   `path_length` via `get_diagnostics()`. Accepts a `frequency` to throttle
   recomputation on long runs.
6. **{py:class}`~nvalchemi.dynamics.hooks.LoggingHook`** (optional, pairs with
   step 5) --- reads `diagnostics.get_diagnostics()` through `custom_scalars`
   callbacks and writes per-path rows to CSV. This is exactly how `NEB` wires
   `diagnostics_log_path`: both this hook and `PathDiagnosticsHook` should
   share the same `frequency`/`diagnostics_frequency` so the CSV rows line up
   with the diagnostics they log.

All of these hooks require the enclosing workflow to be constructed with
`by_group=True` (each `group_layout` group is one path) and raise
`ValueError` at registration otherwise. Convergence uses the standard
{py:class}`~nvalchemi.dynamics.ConvergenceHook`, evaluated per path with
`by_group=True` --- there is no NEB-specific convergence criterion.

Because you assemble the `hooks` list yourself, you are not limited to the
built-in hooks above: any of them can be replaced with a custom
implementation (e.g. a different `PathDiagnosticsHook`-like hook that logs
extra scalars, or your own freezing hook instead of `FreezeAtomsHook`), and
you can insert additional hooks anywhere in the stack --- as long as you
preserve the dependency order (`PathEnergyStatsHook` before anything that
calls `get_stats()`, `NEBForceHook` before anything that reads
`neb_fixed_node_mask` or the NEB force decomposition). See the
[Hooks guide](hooks_guide) for the hook protocol used to write one.

Climbing-image NEB needs two stages (regular, then climbing) and is
therefore built on `FusedStage`, not a single optimizer. Rather than
duplicating that construction here, read `NEB.build_engine()` in
`nvalchemi/dynamics/paths/neb/neb.py` --- it is the reference
implementation for a two-substage, `status`-gated `FusedStage` with
`ClimbingImageSelectionHook(status_code=...)` scoped to the climbing stage.

## See also

- **Example**: ``examples/advanced/16_batched_neb.py`` runs a full
  climbing-image NEB workflow (IDPP initialization, AIMNet2-rxn, diagnostics
  CSV, and a comparison against DFT reference paths) on eight batched
  reactions.
- **Optimization**: [Optimization and Integrators](dynamics_simulations) ---
  the `FIRE2` optimizer driving each NEB image.
- **Hooks**: The [Hooks guide](hooks_guide) covers the hook protocol and
  `ConvergenceHook`.
- **Overview**: The [Dynamics overview](dynamics_guide) describes the shared
  execution loop and `FusedStage` composition.
