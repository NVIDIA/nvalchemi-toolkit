(dynamics_mep_guide)=

# Reaction Paths and NEB

The `nvalchemi.dynamics.mep` subpackage provides tools for computing
minimum-energy paths (MEPs), and therefore transition-state estimates, between
reactant and product structures.
Each path is represented as an ordered band of structures called images, and
many paths can be optimized together: every image is one graph in a `Batch`,
and the images of a path share a `group_layout` group.

The subpackage currently provides batched nudged elastic band (NEB) methods.
There are two ways to run NEB:

- The **{py:class}`~nvalchemi.dynamics.mep.NEB` strategy**: a
  declarative, serializable wrapper that builds a correctly-ordered engine
  for you. Use this for standard regular or climbing-image NEB runs.
- **An optimizer (or `FusedStage`) with the NEB hooks attached directly**:
  for full control over hook ordering, custom convergence criteria, or
  non-standard staging.

## Building the initial path

Whether you use the high-level `NEB` strategy or the low-level `FusedStage`
approach, the input is the same: a `Batch` of images with `group_layout`
set, one group per path, at least three images each, matching atoms/cell
across images (see {py:func}`~nvalchemi.dynamics.mep.validate_paths`). You
can build this batch yourself, or use the two basic interpolation utilities
below to go from just reactant/product endpoints:

```python
import torch

from nvalchemi.dynamics.mep import (
    IDPPModel,
    NEB,
    interpolate_paths,
    prepare_idpp_targets,
)

paths = interpolate_paths(initial, final, num_images=7)  # linear interpolation

# optionally remove rigid endpoint displacement before interpolation
paths = interpolate_paths(
    initial,
    final,
    num_images=7,
    remove_translation_and_rotation=True,
    fit_mask=initial.atomic_numbers > 1,  # fit on corresponding heavy atoms
)
paths.velocities = torch.zeros_like(paths.positions)  # dropped by interpolation

# optional: refine with IDPP before switching to the real model
prepare_idpp_targets(paths)
paths = NEB(model=IDPPModel(), fmax=0.1, n_steps=200).run(paths)
```

`interpolate_paths` linearly interpolates positions between index-matched
endpoint graphs and drops stale fields (`forces`, `energy`, `velocities`,
...), so reattach `velocities` before optimizing. Set
`remove_translation_and_rotation=True` to align each final structure to its
paired initial structure first. For periodic paths, this reconciles
minimum-image positions and removes translation without rotating the cell. An
optional Boolean `fit_mask`, following the flattened node layout of `initial`,
restricts the corresponding atom pairs used to fit each transform without
restricting which atoms are transformed. Every path must select at least one
atom; for example, `initial.atomic_numbers > 1` fits molecular paths using only
heavy atoms.

For alignment without interpolation, call
{py:func}`~nvalchemi.dynamics.mep.align_batch_positions` directly on two
`Batch`es with matching atom order (optionally passing `fit_mask`):

```python
from nvalchemi.dynamics.mep import align_batch_positions

alignment = align_batch_positions(
    initial.positions,
    final.positions,
    initial.batch_idx,
    initial.num_nodes_per_graph,
)
aligned_final = alignment.positions
```

The returned {py:class}`~nvalchemi.dynamics.mep.PositionAlignment` also holds
the rotation and translation for each graph. Pass `cell` and `pbc` for periodic
graphs; the function then uses minimum-image positions and translation without
rotating the cell.

{py:func}`~nvalchemi.dynamics.mep.prepare_idpp_targets` +
{py:class}`~nvalchemi.dynamics.mep.IDPPModel` relax the path against target
pairwise distances (IDPP,
[Smidstrup et al. 2014](https://doi.org/10.1063/1.4878664)) to avoid poor
linear-interpolation guesses where atoms pass through each other.
`IDPPModel` is just a `BaseModelMixin` like any other, so running it through
`NEB` (as above) is the normal way to do this refinement. It is a regular
NEB optimization whose "physical" model happens to be the analytic IDPP
potential instead of your real MLIP. See
``examples/advanced/11_batched_neb.py`` for the full IDPP-then-MLIP workflow.

## The high-level `NEB` strategy

```python
from nvalchemi.dynamics.mep import NEB, ClimbingImageConfig

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
| `spring` | `0.1` | Constant spring force, or a custom {py:class}`~nvalchemi.dynamics.mep.SpringConfig` |
| `method` | `"improved_tangent"` | Named method, custom Warp `NEBMethod`, or callable `TorchNEBMethod` |
| `climbing` | `None` | `None` runs regular NEB only; set a `ClimbingImageConfig` to enable climbing-image NEB |
| `optimizer` | `FIRE2` | `BaseDynamics` subclass supporting fixed-cell, group-aware updates |
| `optimizer_kwargs` | `{}` | Arguments forwarded to each optimizer stage; unspecified parameters use NEB-tuned defaults for `FIRE2`, or the custom optimizer's own defaults |
| `fmax` | `0.05` | Force threshold for the final (or only) stage |
| `n_steps` | `None` | Limit on the total number of optimization steps; `None` for no fixed limit |
| `convergence_hook` | `None` | Custom `ConvergenceHook` for the final (or only) stage; takes precedence over `fmax` |
| `regular_convergence_hook` | `None` | Custom `ConvergenceHook` for the regular stage of an `"after_regular"` run; takes precedence over `regular_fmax` |
| `endpoint_mode` | `"fixed"` | `"fixed"` keeps endpoints static; `"relaxed"` lets them move |
| `fixed_atom_indices` | `None` | Per-path, image-local atom indices held fixed in every image |
| `diagnostics_log_path` | `None` | CSV path for per-path `fmax`, `energy_barrier`, `highest_interior_image_idx`, `path_length` |
| `diagnostics_frequency` | `1` | Step frequency for computing and logging diagnostics |
| `neighbor_hooks` | `None` | Override `model.make_neighbor_hooks()` with a custom neighbor-list hook list |
| `compile` | `False` | Compile the fused NEB step with `torch.compile` |
| `compile_kwargs` | `{}` | Keyword arguments forwarded to `torch.compile` when compilation is enabled |

`optimizer_kwargs` is forwarded to every internal optimizer stage, for
settings such as `dt` or `maxstep`. `NEB` sets `model`, `hooks`, `by_group`,
`convergence_hook`, and `n_steps` itself for each stage, so passing any of
them in `optimizer_kwargs` raises `ValueError`. Use the corresponding `NEB`
fields instead.

`optimizer` accepts an optimizer class derived from
{py:class}`~nvalchemi.dynamics.BaseDynamics` that supports fixed-cell,
group-aware updates through {py:class}`~nvalchemi.dynamics.FusedStage`.
Each path is one update unit, and `pre_update` / `post_update` consume the
projected NEB forces in `batch.forces`. Regular and climbing stages use
separate optimizer instances and history.

FIRE2 is the default. Its NEB-tuned parameters are applied to
`optimizer_kwargs` during validation, with explicit values taking precedence.
Other optimizer classes use their own defaults.
Inspect `neb.optimizer_kwargs` to see the arguments forwarded to the optimizer.

`NEBForceHook` always publishes `neb_fixed_node_mask`, a Boolean node mask
marking fixed endpoints and any `fixed_atom_indices`. It zeroes the physical
and effective forces at those nodes. When `endpoint_mode="fixed"` or a nonempty
`fixed_atom_indices` mapping is provided, `NEB` also appends a
{py:class}`~nvalchemi.dynamics.hooks.FreezeAtomsHook` to clear velocities and
restore positions across optimizer stages.

`ClimbingImageConfig.mode="after_regular"` runs regular NEB to
`regular_fmax` (or `fmax` if unset) before promoting the highest-energy
interior image to a climbing image and continuing to `fmax`;
`mode="immediate"` climbs from the first step.
`selection="fixed"` locks in the initial climbing image;
`selection="dynamic"` reselects it every evaluation.
`max_regular_steps` and `max_climbing_steps` optionally limit each path's steps
in the respective stage. A path advances when its stage limit is reached even
if it has not met the force threshold. `max_regular_steps` is only valid with
`mode="after_regular"`.

`neb.run(paths)` builds a fresh engine via `neb.build_engine()` and calls
`engine.run(paths)`. `NEB` is a Pydantic model, so it also serializes with
`to_spec_dict()` / `from_spec_dict()` alongside other
{py:class}`~nvalchemi.dynamics.strategy.DynamicsStrategy` subclasses used for
training and fine-tuning specs.

### NEB methods

`method="improved_tangent"` (the default and only named method)
implements the Henkelman--Jónsson improved tangent
([*J. Chem. Phys.* 113, 9978 (2000)](https://doi.org/10.1063/1.1323224)).
It corresponds to the default `NEBMethod()` without overrides.
For an interior image with neighbor energies $E^-, E, E^+$ and
displacement vectors $\mathbf{d}^\pm$ to the adjacent images:

- **Tangent**: $\vec{\tau} = w^+\mathbf{d}^+ + w^-\mathbf{d}^-$
  (then normalized), where the weights $w^\pm$ favor the higher-energy
  neighbor on monotonic sections of the path and blend both links at an
  extremum.
- **Effective (regular) force**:

  $$
  \mathbf{F}^{\mathrm{NEB}} = \mathbf{F}
  - (\mathbf{F}\cdot\hat{\tau})\hat{\tau}
  {}+ F^{\mathrm{s},\parallel}\hat{\tau}
  $$

  That is the physical force projected perpendicular to the tangent, plus the
  parallel spring contribution

  $$
  F^{\mathrm{s},\parallel}
  = k^+\lVert\mathbf{d}^+\rVert - k^-\lVert\mathbf{d}^-\rVert.
  $$

- **Climbing-image force**:

  $$
  \mathbf{F}^{\mathrm{climb}} = \mathbf{F}
  - 2(\mathbf{F}\cdot\hat{\tau})\hat{\tau}
  $$

  The spring force is dropped and the parallel physical-force component is
  reversed, driving the image uphill along the path toward the saddle point.

To use a different formulation, pass a custom
{py:class}`~nvalchemi.dynamics.mep.NEBMethod` to `method`. It bundles three
`warp.func`-decorated device functions, any of which can be overridden
independently (unset ones fall back to the improved-tangent equations above):

| Field | Inputs available | Returns |
|-------|-------------------|---------|
| `tangent_weights_fn` | `energy_prev`, `energy_curr`, `energy_next` (scalars) | `(weight_plus, weight_minus)` link weights for the tangent |
| `effective_force_fn` | One of the two ordered signatures below | effective force vector for a regular (non-climbing) interior image |
| `climbing_force_fn` | `physical_force`, `tangent`, `force_dot_tangent` | effective force vector for the climbing image |

The `effective_force_fn` parameter names and order must match one of these
signatures exactly; they select the corresponding Warp kernel:

```text
Stored tangent:
(physical_force, tangent, force_dot_tangent, k_plus, k_minus,
 norm_d_plus, norm_d_minus, energy_prev, energy_curr, energy_next,
 path_energy_ref, path_energy_max)

Gram statistics:
(physical_force, tangent, d_plus, d_minus, force_dot_tangent,
 dplus_dot_tangent, dminus_dot_tangent, norm_d_plus, norm_d_minus,
 dplus_dot_dminus, force_dot_dplus, force_dot_dminus, force_squared_norm,
 k_plus, k_minus, energy_prev, energy_curr, energy_next,
 path_energy_ref, path_energy_max)
```

`physical_force`, `tangent`, `d_plus`, and `d_minus` are vectors for the
current atom. The dot products and link norms are scalars reduced over the
current image; `k_plus` and `k_minus` are the adjacent link spring constants.
`energy_prev`, `energy_curr`, and `energy_next` are image energies, while
`path_energy_ref` and `path_energy_max` are path-wide energy statistics.

```python
import warp as wp
from nvalchemi.dynamics.mep import NEB, NEBMethod

@wp.func
def central_tangent_weights(energy_prev: float, energy_curr: float, energy_next: float):
    return 1.0, 1.0  # plain central-difference tangent instead of improved-tangent weighting

neb = NEB(model=model, method=NEBMethod(tangent_weights_fn=central_tangent_weights), fmax=0.05)
```

Use the Gram-statistics signature when the force needs more than the tangent
projection, for example the link vectors `d_plus` and `d_minus` and their dot
products. The doubly nudged elastic band (DNEB,
[Trygubenko & Wales 2004](https://doi.org/10.1063/1.1636455)) uses them to add
a perpendicular spring correction. `examples/advanced/11_batched_neb.py` uses
the improved-tangent method by default; set `method = "dneb"` there to run the
`dneb_effective_force` equation from `_dneb_method.py`.

Custom `NEBMethod` objects round-trip through `to_spec_dict()` and
`from_spec_dict()`. The spec stores a stable method key with the kernel kind and
dotted import paths for all three Warp equations. Restoring the spec imports
those equations and rebuilds the `NEBMethod`; each equation must therefore be
defined at module level in an importable module. The same key selects the kernel
for compilation and CUDA graph capture.

If you prefer to implement an NEB method entirely in PyTorch, pass a callable
matching {py:class}`~nvalchemi.dynamics.mep.TorchNEBMethod` as `method`. It
receives the current batch, link spring constants, path energy statistics, and
prepared minimum-image geometry. It returns effective forces for every atom
and one minimum-image length per forward image link. Importable classes whose
constructor arguments are stored as attributes also round-trip through
`to_spec_dict()` and `from_spec_dict()`. Other callables, such as lambdas or
locally defined functions, still work with `NEB.run()`, but `to_spec_dict()`
raises `ValueError` for them.

```python
from nvalchemi.dynamics.mep import NEB
from nvalchemi.dynamics.mep._geometry import minimum_image_displacement

class MyTorchNEBMethod:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale  # stored as an attribute, so the spec round-trips

    def __call__(self, batch, *, spring_constants, path_energy_ref, path_energy_max, mic):
        forces = batch.physical_forces.clone()  # don't modify the batch in place
        # ... apply your equations to regular, climbing, and endpoint images,
        # using batch.positions, batch.energy, and batch.force_mode
        # use minimum_image_displacement(..., prepared=mic) for link vectors
        link_lengths = ...  # one minimum-image length per forward link
        return forces, link_lengths

neb = NEB(model=model, method=MyTorchNEBMethod(scale=0.5), fmax=0.05)
```

`spring` follows the same pattern: pass a plain `float` for a constant spring
constant, or a custom {py:class}`~nvalchemi.dynamics.mep.SpringConfig`
(`resolve(context)` returning one spring constant per link) for e.g.
energy-dependent springs. Custom spring policies round-trip through
`NEB.to_spec_dict()` when their class is importable and their constructor
arguments are stored as attributes (or they provide a `checkpoint_spec()`
constructor spec). Other custom spring policies can be used at runtime.
Numeric springs and {py:class}`~nvalchemi.dynamics.mep.ConstantSpringConfig`
retain their existing spec representation.

## Building NEB manually with hooks

`NEB` is a factory around a fixed hook order. Assembling it by hand gives you
control over convergence criteria, staging, or optimizer choice beyond what
the strategy exposes. For regular (non-climbing) NEB there is only one stage,
so any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` subclass that
accepts `by_group=True` and a `hooks` list is enough on its own; no
{py:class}`~nvalchemi.dynamics.FusedStage` wrapper is required. `FusedStage` is
only needed once you have more than one stage, e.g. climbing-image NEB (see
below). The example uses {py:class}`~nvalchemi.dynamics.optimizers.fire2.FIRE2`.

```python
from nvalchemi.dynamics import ConvergenceHook
from nvalchemi.dynamics.hooks import FreezeAtomsHook, LoggingHook
from nvalchemi.dynamics.optimizers.fire2 import FIRE2
from nvalchemi.dynamics.mep.hooks import (
    ClimbingImageSelectionHook,
    NEBForceHook,
    PathDiagnosticsHook,
    PathEnergyStatsHook,
)

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

`ClimbingImageSelectionHook`, `NEBForceHook`, and `PathDiagnosticsHook` must
receive the same `PathEnergyStatsHook` instance registered in the hooks list.
That hook refreshes its statistics first at `AFTER_COMPUTE`, before any of the
three consumers reads them. `NEB.build_engine()` wires this shared instance and
order automatically.

The NEB path hooks, in the order they must be registered:

1. **{py:class}`~nvalchemi.dynamics.mep.hooks.PathEnergyStatsHook`**:
   always first. Computes per-path endpoint reference energy and the
   highest-energy interior image; the selection, force, and diagnostics hooks
   read its results via `get_stats()`.
2. **{py:class}`~nvalchemi.dynamics.mep.hooks.ClimbingImageSelectionHook`**
   (optional): flips the selected image's `force_mode` to climbing before
   `NEBForceHook` runs. Needed only for climbing-image NEB.
3. **{py:class}`~nvalchemi.dynamics.mep.hooks.NEBForceHook`**: the
   core hook. It computes the local tangent and replaces `batch.forces` with the
   spring + perpendicular-physical-force NEB update, saving the raw model
   force to `batch.physical_forces`. When `endpoint_mode="fixed"` or
   `fixed_atom_indices` is set, it populates `neb_fixed_node_mask` (present
   on every batch) with the constrained atoms and zeroes their forces. The mask
   alone does not keep their positions fixed during optimizer updates.
4. **{py:class}`~nvalchemi.dynamics.hooks.FreezeAtomsHook`** (optional, needed
   whenever step 3 produces a fixed-atom mask): a general-purpose,
   non-NEB-specific hook (`nvalchemi.dynamics.hooks`) that zeroes forces and
   velocities and restores positions for masked atoms across every integrator
   stage. Pass `mask_key="neb_fixed_node_mask"` to use the mask from
   `NEBForceHook` instead of the default `atom_categories`-based freezing.
5. **{py:class}`~nvalchemi.dynamics.mep.hooks.PathDiagnosticsHook`**
   (optional): exposes per-path `fmax`, `energy_barrier`,
   `highest_interior_image_idx`, and `path_length` via `get_diagnostics()`.
   Accepts a `frequency` to throttle recomputation on long runs.
6. **{py:class}`~nvalchemi.dynamics.hooks.LoggingHook`** (optional, pairs with
   step 5): reads `diagnostics.get_diagnostics()` through `custom_scalars`
   callbacks and writes per-path rows to CSV. This is exactly how `NEB` wires
   `diagnostics_log_path`: both this hook and `PathDiagnosticsHook` should
   share the same `frequency`/`diagnostics_frequency` so the CSV rows line up
   with the diagnostics they log.

All of these hooks require the enclosing workflow to be constructed with
`by_group=True` (each `group_layout` group is one path) and raise
`ValueError` at registration otherwise. Convergence uses the standard
{py:class}`~nvalchemi.dynamics.ConvergenceHook`, evaluated per path with
`by_group=True`; there is no NEB-specific convergence criterion.

Because you assemble the `hooks` list yourself, you are not limited to the
built-in hooks above: any of them can be replaced with a custom
implementation (e.g. a different `PathDiagnosticsHook`-like hook that logs
extra scalars, or your own freezing hook instead of `FreezeAtomsHook`), and
you can insert additional hooks anywhere in the stack, as long as you
preserve the dependency order (`PathEnergyStatsHook` before anything that
calls `get_stats()`, `NEBForceHook` before anything that reads
`neb_fixed_node_mask` or the NEB force decomposition). See the
[Hooks guide](hooks_guide) for the hook protocol used to write one.

Climbing-image NEB needs two stages (regular, then climbing) and is
therefore built on `FusedStage`, not a single optimizer. Rather than
duplicating that construction here, read `NEB.build_engine()` in
`nvalchemi/dynamics/mep/neb.py`. It is the reference
implementation for a two-substage, `status`-gated `FusedStage` with
`ClimbingImageSelectionHook(status_code=...)` scoped to the climbing stage.
With `mode="after_regular"` it sets `reprime_on_entry={1}`, so a path entering
the climbing stage skips one optimizer update while the shared model evaluation
and the climbing sub-stage's `AFTER_COMPUTE` hooks refresh its forces. If you
build the engine yourself, register the path hooks on the `FusedStage` and
scope stage-specific ones with `status_code`, as `NEB.build_engine()` does. At
`AFTER_COMPUTE`, sub-stage hooks run before fused-stage hooks, so a sub-stage
hook would read stale `PathEnergyStatsHook` results.

## Periodic paths

Interpolation, IDPP, and NEB automatically apply the minimum-image convention
(MIC): each displacement uses the nearest periodic copy of an atom. This works
with orthogonal or skewed cells and with periodicity in only some directions.
Each `pbc` entry enables the corresponding row of `cell` (a lattice vector),
which need not align with the Cartesian x, y, or z axis. The enabled cell
vectors must be linearly independent. Nonperiodic paths are left unchanged.

Wrapped endpoints do not record whether a path is intended to cross one or more
whole cells. For such winding paths, provide appropriately unwrapped images.

## See also

- **Example**: ``examples/advanced/11_batched_neb.py`` runs a full
  climbing-image NEB workflow (IDPP initialization, AIMNet2-rxn, diagnostics
  CSV, and a comparison against DFT reference paths) on eight batched
  reactions.
- **Optimization**: [Optimization and Integrators](dynamics_simulations)
  covers the `FIRE2` optimizer driving each NEB image.
- **Hooks**: The [Hooks guide](hooks_guide) covers the hook protocol and
  `ConvergenceHook`.
- **Overview**: The [Dynamics overview](dynamics_guide) describes the shared
  execution loop and `FusedStage` composition.
