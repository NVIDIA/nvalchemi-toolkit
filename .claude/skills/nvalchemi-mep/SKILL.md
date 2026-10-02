---
name: nvalchemi-mep
description: >-
  How to compute minimum-energy paths (MEPs) and transition-state estimates
  with nvalchemi.dynamics.mep: building initial paths (interpolation,
  alignment, IDPP) and running batched nudged elastic band (NEB) with the NEB
  strategy or by building it manually with hooks. Use when setting up reaction paths between
  reactant and product structures, running regular or climbing-image NEB,
  reading NEB diagnostics, or writing a custom NEB force method or spring
  policy.
---

# nvalchemi Minimum-Energy Paths (MEP)

## Overview

`nvalchemi.dynamics.mep` provides tools for computing minimum-energy paths
between reactant and product structures. A path is an ordered band of
structures called images. Many paths are optimized together in one `Batch`:
every image is one graph, and the images of a path share a `group_layout`
group. NEB is currently the only path-optimization method.

Details (force equations, method signatures, a full manual-hooks example)
live in `docs/userguide/dynamics_mep.md`. The end-to-end workflow (IDPP,
climbing-image NEB with AIMNet2-rxn, diagnostics CSV) is
`examples/advanced/11_batched_neb.py`. Upstream skills:
`nvalchemi-data-structures` (`Batch`, `group_layout`),
`nvalchemi-dynamics-api` (`FIRE2`, `FusedStage`), and
`nvalchemi-dynamics-hooks` (hook stages and ordering).

```python
from nvalchemi.dynamics.mep import (
    NEB,
    ClimbingImageConfig,
    IDPPModel,
    interpolate_paths,
    prepare_idpp_targets,
    validate_paths,
)
```

## Build the initial paths

Endpoints are two `Batch`es: graph `i` of `initial` and graph `i` of `final`
define path `i`. Atoms must match by index (same atomic numbers, same order).

```python
import torch

from nvalchemi.data import Batch

initial = Batch.from_data_list(reactants, device=device)
final = Batch.from_data_list(products, device=device)

paths = interpolate_paths(
    initial,
    final,
    num_images=10,  # total images per path, endpoints included; or one per path
    remove_translation_and_rotation=True,  # optional rigid alignment first
    fit_mask=initial.atomic_numbers > 1,  # optional: fit on heavy atoms only
)
validate_paths(paths)  # >= 3 images per path, matching atoms/cell/pbc
```

`interpolate_paths` drops fields that no longer apply (`forces`, `energy`,
`velocities`, `stress`, embeddings, ...). Reinitialize them before any
optimizer run, as the example's `initialize_dynamics_fields` does:

```python
def initialize_dynamics_fields(batch: Batch) -> None:
    batch.energy = torch.zeros(
        batch.num_graphs, 1, dtype=batch.positions.dtype, device=batch.device
    )
    batch.forces = torch.zeros_like(batch.positions)
    batch.velocities = torch.zeros_like(batch.positions)
```

To align without interpolating, use `align_batch_positions` (see the guide).
Periodic paths use the minimum-image convention automatically. For paths that
cross whole cells, supply unwrapped images.

### Refine with IDPP

Linear interpolation can push atoms through each other. IDPP relaxes the band
toward interpolated pairwise distances using an analytic potential, with no
MLIP evaluation. `IDPPModel` is a regular `BaseModelMixin`, so run it through
`NEB`:

```python
idpp_band = prepare_idpp_targets(paths.clone())  # replaces edges with all pairs
initialize_dynamics_fields(idpp_band)
idpp = NEB(model=IDPPModel(), fmax=0.1, n_steps=100)
idpp_band = idpp.run(idpp_band)
if not bool(torch.all(idpp_band.status == 1)):  # 1 = converged (one stage)
    raise RuntimeError("IDPP did not converge")
```

Before switching to the real model, copy IDPP positions into a fresh band so
the IDPP pair list and dynamics state are not carried over:

```python
neb_band = paths.clone()
neb_band.positions.copy_(idpp_band.positions)
initialize_dynamics_fields(neb_band)
```

## Run NEB with the strategy

```python
neb = NEB(
    model=model,
    spring=0.1,  # eV/Å, same constant for every link; or a SpringConfig
    method="improved_tangent",  # default; or NEBMethod / TorchNEBMethod
    climbing=ClimbingImageConfig(regular_fmax=0.5),  # None = regular NEB only
    fmax=0.05,
    n_steps=500,
    diagnostics_log_path="neb.csv",
)
neb_band = neb.run(neb_band)
```

Other fields: `endpoint_mode` (`"fixed"` default, or `"relaxed"`),
`fixed_atom_indices` (`{path_index: (image-local atom indices, ...)}`),
`diagnostics_frequency`, `optimizer_kwargs`, `convergence_hook`,
`regular_convergence_hook`, `neighbor_hooks`, `compile`, `compile_kwargs`.

- `optimizer` defaults to `FIRE2` and accepts `BaseDynamics` subclasses
  supporting fixed-cell, group-aware `FusedStage` updates. FIRE2's NEB defaults
  are filled in `optimizer_kwargs` during validation, preserving explicit
  values. Other optimizers use their own defaults.
- `optimizer_kwargs` is forwarded to every internal optimizer stage (for
  example `dt`, `maxstep`). It must not contain `model`, `hooks`, `by_group`,
  `convergence_hook`, or `n_steps`; `NEB` sets these per stage and raises
  `ValueError`.
- `NEB` is a Pydantic `DynamicsStrategy`: `to_spec_dict()` round-trips it and
  `NEB.from_spec_dict(spec, model=model)` rebuilds it with a live model.

### ClimbingImageConfig

- `mode="after_regular"` (default): regular NEB until `regular_fmax` (or
  `fmax`), then the highest-energy interior image climbs until `fmax`.
- `mode="immediate"`: climbing from the first step (single stage).
- `selection="fixed"` keeps the first climbing image; `"dynamic"` reselects
  every evaluation.
- `max_regular_steps` / `max_climbing_steps`: per-path stage budgets. A path
  advances when its budget runs out even if not converged.
  `max_regular_steps` requires `mode="after_regular"`.

### Read the results

`batch.status` (also a CSV column) gives each path's stage. For
`after_regular` climbing: `0` regular, `1` climbing, `2` finished. For a
single-stage run (regular only or `immediate`): `0` running, `1` finished.
With stage budgets set, "finished" does not imply converged; check `fmax`.

The diagnostics CSV has one row per path per logged step with `fmax`,
`energy_barrier`, `highest_interior_image_idx`, `path_length`, and `status`.
Diagnostics reuse the already-computed path state, so they cost no extra model
evaluation. For equal-size paths, reshape per path:

```python
path_energies = neb_band.energy.reshape(n_paths, num_images)
path_positions = neb_band.positions.reshape(n_paths, num_images, n_atoms, 3)
```

## Custom NEB methods and springs

- **Warp equations**: `NEBMethod(tangent_weights_fn=..., effective_force_fn=...,
  climbing_force_fn=...)`, each a module-level `@wp.func`; unset ones use the
  improved tangent. `effective_force_fn` must exactly match one of the two
  signatures in the guide. `dneb_effective_force` in
  `examples/advanced/_dneb_method.py` is a working example.
- **PyTorch**: pass a callable matching `TorchNEBMethod`:
  `__call__(batch, *, spring_constants, path_energy_ref, path_energy_max, mic)`
  returning `(forces, link_lengths)`. Do not modify `batch` in place; handle
  regular, climbing, and endpoint images via `batch.force_mode`.
- **Springs**: a `SpringConfig` implements `resolve(context: SpringContext)`
  returning one constant per link, plus `refresh` (`DynamicsStage.ON_ADMISSION`
  or `DynamicsStage.AFTER_COMPUTE`).
- Custom methods and springs serialize only if importable and their
  constructor arguments are stored as attributes; other callables still run
  but `to_spec_dict()` raises `ValueError`.

## Build NEB manually with hooks

Use this for custom convergence, staging, or extra hooks. Regular NEB needs
only one optimizer with `by_group=True`; climbing-image NEB needs a
`FusedStage` (copy `NEB.build_engine()` in `nvalchemi/dynamics/mep/neb.py`).

Hooks from `nvalchemi.dynamics.mep.hooks`, in required order:

1. `PathEnergyStatsHook()`: always first; others read `get_stats()`.
2. `ClimbingImageSelectionHook(energy_stats_hook=..., status_code=...)`:
   climbing only.
3. `NEBForceHook(energy_stats_hook=..., spring=..., method=...,
   endpoint_mode=...)`: saves `batch.physical_forces`, writes NEB forces to
   `batch.forces`, publishes `neb_fixed_node_mask`.
4. `FreezeAtomsHook(mask_key="neb_fixed_node_mask")` from
   `nvalchemi.dynamics.hooks`: needed whenever atoms are fixed; the mask alone
   does not hold positions.
5. `PathDiagnosticsHook(energy_stats_hook=..., frequency=...)`: optional.
6. `LoggingHook(..., by_group=True, custom_scalars=...)`: optional CSV; use
   the same frequency as step 5.

Rules:

- Pass the same `PathEnergyStatsHook` instance to hooks 2, 3, and 5, and
  register it in the list.
- Every workflow must use `by_group=True`; the hooks raise `ValueError`
  otherwise. Convergence is a standard
  `ConvergenceHook.from_fmax(threshold=..., by_group=True)`.
- In a `FusedStage`, register the path hooks on the fused stage and scope
  stage-specific ones with `status_code`. At `AFTER_COMPUTE`, sub-stage hooks
  run before fused-stage hooks, so a sub-stage hook would read stale
  `PathEnergyStatsHook` results.

## Key files

- `nvalchemi/dynamics/mep/neb.py`: `NEB`, `ClimbingImageConfig`,
  `build_engine()`.
- `nvalchemi/dynamics/mep/neb_configs.py`: `NEBMethod`, `TorchNEBMethod`,
  `SpringConfig`, `SpringContext`, `ConstantSpringConfig`.
- `nvalchemi/dynamics/mep/hooks/`: path hooks.
- `nvalchemi/dynamics/mep/interpolate.py`, `idpp.py`, `validate.py`,
  `_alignment.py`: path construction.
