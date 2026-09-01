# Batched NEB with group-aware `Batch` and `BaseDynamics`

2026-08-18, YS Teh

References:
- BatchedNEB-v3.md
- neb-counter-proposal.md

## 1. General Idea

Expose batched NEB through the existing `Batch` and `BaseDynamics` APIs:

```python
relaxed: Batch = optimizer.run(bands)
```

The same `Batch` is used by the model, optimizer, hooks, convergence, refill,
`FusedStage`, sinks, and `DistributedPipeline`. Every graph is one image. Images
belonging to the same path are coupled by a workload-neutral `GroupLayout`.

The design separates three concerns:

```text
Batch                       one graph per image
GroupLayout                 images -> complete optimization/lifecycle groups
hooks                       NEB force, climbing policy, and diagnostics
```

There is no public or private `PathBatch`, optimizer view, or overloaded `run()` target inside `BaseDynamics`. 

## 2. Public API

### 2.1 Explicit engine and hooks

The lower-level API uses the existing `BaseDynamics.run(batch) -> Batch` entry
point:

```python
bands: Batch = interpolate_endpoints(
    initial,
    final,
    n_images=[8, 12, 8, 11],
    method="idpp",
)

optimizer = FIRE2(
    model=model,
    dt=0.01,
    n_steps=1100,
    group_key="path_idx", # new
    hooks=[
        NeighborListHook(config=neighbor_config, skin=0.3),
        ClimbingImageHook(
            max_regular_steps=500,
            max_climbing_steps=500,
        ), # new
        NEBForceHook( # new
            spring=ConstantSpringConfig(0.1), # or EnergyWeightedSpringConfig(...)
            method="improved_tangent", # or "doubly_nudged" or ...
            ),
        NaNDetectorHook(),
        SnapshotHook(sink=ZarrData("bands.zarr"), frequency=10),
    ],
    convergence_hook=GroupedConvergenceHook.from_fmax( # new
        0.05,
        group_key="path_idx",
    ),
)

relaxed: Batch = optimizer.run(bands)
```

`bands` contains one graph per image. The new graph-level field `path_idx`
describes complete paths; graph order within each group defines image order.
No alternative batch view is constructed.

Hook registration order defines the force pipeline:

```text
model physical outputs
    -> neighbor-list preparation
    -> model physical outputs
    -> climbing-image selection
    -> NEBForceHook
        copies batch.forces -> batch.physical_forces
        computes NEB effective forces
        writes effective forces -> batch.forces
    -> effective-force safety and diagnostics
    -> optimizer uses batch.forces to perform update
```

`NEBForceHook` saves physical forces to a separate preallocated field before it
overwrites `batch.forces` with optimizer-facing effective forces.

### 2.2 `NEB.run()` high-level API

The higher-level API builds the same ordinary engine and hook stack (with the exact arguments of NEB to be decided):

```python
neb = NEB(
    spring=0.1,
    fmax=0.05,
    climbing=True,
    climbing_activation="after_regular",
    climbing_reselection="locked",
    max_regular_steps=500,
    max_climbing_steps=500,
    n_steps=1100,
    optimizer=FIRE2,
    optimizer_kwargs={"dt": 0.01},
)

relaxed: Batch = neb.run(bands, model=model)
```

Its implementation calls the public optimizer API:

```python

class DynamicsStrategy(BaseModel):
    n_steps: int | None = None

    def build_hooks(self) -> list[Hook]: ...
    def build_convergence(self) -> ConvergenceHook | None: ...
    def build_engine(self, model: BaseModelMixin) -> BaseDynamics:
        raise NotImplementedError
    def run(self, batch, model, n_steps=None) -> Batch: ...
    def to_spec_dict(self) -> dict; from_spec_dict(spec, extra_hooks=None)

class NEB(DynamicsStrategy):
    # Declarative fields omitted.

    def run(
        self,
        bands: Batch,
        model: BaseModelMixin,
        n_steps: int | None = None,
    ) -> Batch:
        optimizer = self.build_engine(model)
        return optimizer.run(
            bands,
            n_steps=self.n_steps if n_steps is None else n_steps,
        )
```

The call chain is:

```text
neb.run(bands, model)
    -> NEB.build(model)
    -> FIRE2(group_key="path_idx", hooks=[..., NEBForceHook(...)])
    -> FIRE2.run(bands)
    -> return bands
```

Path construction remains independent of the NEB recipe:

```python
bands = interpolate_endpoints(initial, final, n_images=8, method="idpp")
bands = concat_bands(band_a, band_b, band_c)
```

`validate_bands(bands)` remains available as an optional explicit preflight.
The authoritative validation occurs in `NEBForceHook.prepare_batch()` when the
batch enters the engine and after membership changes.

Generated paths and paths created by other packages use `Batch` as the
interchange type.

Restrict optimizer support to FIRE2 for now within NEB.

## 3. Group-aware `Batch`

### 3.1 Minimal `Batch` additions

Every image remains one ordinary graph. Grouping is optional metadata on an
ordinary `Batch`. A grouped workload carries one dynamic system-level integer
tensor:

```text
group_idx   [B]  dense local group index for each graph
```

For NEB:

```text
graph       = image
group       = band representing one path
graph order = image order within the band
```

The initial design added only `group_idx`, but letting `Batch` also own the
derived `GroupLayout` cache makes `BaseDynamics` simpler.

`set_group_layout()` stores the grouping tensor and prepares its cached layout:

```python
def set_group_layout(
    self,
    group_idx: torch.Tensor,
    group_key: str = "group_idx",
) -> None:
    group_idx = _normalize_and_validate_group_idx(
        group_idx,
        num_graphs=self.num_graphs,
        device=self.device,
    )
    self.add_key(
        group_key,
        list(group_idx.unbind()),
        level="system",
        overwrite=group_key in self,
    )

    # Cache the layout derived from the stored system tensor.
    self._group_layout_key = group_key
    self._group_layout = GroupLayout.from_batch(self, key=group_key)
```

Retrieval returns the cached object, rebuilding it after deserialization or
when a different key is selected:

```python
def get_group_layout(
    self,
    group_key: str = "group_idx",
) -> GroupLayout:
    if self._group_layout is None or self._group_layout_key != group_key:
        self._group_layout_key = group_key
        self._group_layout = GroupLayout.from_batch(self, key=group_key)
    return self._group_layout
```

Example:

```python
bands.set_group_layout(
    torch.tensor([4, 4, 4, 1, 1, 1, 1, 2, 2, 2]), group_key="path_idx",
)

bands.path_idx
# tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

layout = bands.get_group_layout("path_idx")
```

The single cache is private and non-serialized; only the system-level grouping
tensor is stored. The first `get_group_layout()` after loading rebuilds it.
Membership changes (e.g., `batch.append`) invalidate it and call `set_group_layout()` after rebasing
the grouping tensor.

### 3.2 `GroupLayout`

`GroupLayout` is the derived, workload-neutral view of grouping metadata.
`BaseDynamics` and `FusedStage` use it to become group-aware; users normally do
not use it directly.

```python
# Toolkit shape notation:
# B = graphs/images, V = nodes/atoms, E = edges, G = groups/bands.
# nvalchemi/data/group_layout.py
@dataclass(frozen=True)
class GroupLayout:
    system_group_ids: torch.Tensor | None  # [G], compact stable group IDs
    graph_to_group: torch.Tensor           # [B], graph -> group
    graph_rank: torch.Tensor               # [B], position within group
    node_to_group: torch.Tensor            # [V], node -> group
    group_ptr: torch.Tensor                # [G+1], group -> graph range
    num_graphs_per_group: torch.Tensor     # [G]

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        key: str = "group_idx",
    ) -> GroupLayout: ...

    @property
    def num_groups(self) -> int: ...

    def reduce_all(self, graph_mask: torch.Tensor) -> torch.Tensor: ...
    def reduce_any(self, graph_mask: torch.Tensor) -> torch.Tensor: ...
    def broadcast(self, group_values: torch.Tensor) -> torch.Tensor: ...
    def graph_mask(self, group_mask: torch.Tensor) -> torch.Tensor: ...
    def selected_group_idx(self, group_mask: torch.Tensor) -> torch.Tensor: ...
```

`from_batch()` validates that the group field has integral shape `[B]`, groups
are dense and contiguous, and any `system_group_id` is constant within its
group. It derives rank from graph order:

```python
num_graphs_per_group = torch.bincount(graph_to_group)
group_ptr = torch.cat(
    [
        torch.zeros(1, dtype=torch.long, device=batch.device),
        num_graphs_per_group.cumsum(0),
    ]
)
graph_rank = (
    torch.arange(batch.num_graphs, device=batch.device)
    - group_ptr[graph_to_group]
)
```

For example:

```python
batch.group_idx
# tensor([0, 0, 0, 0, 1, 1, 1])

layout = GroupLayout.from_batch(batch, key="group_idx")

layout.group_ptr
# tensor([0, 4, 7])

layout.graph_rank
# tensor([0, 1, 2, 3, 0, 1, 2])
```

It builds the other derived tensors when batch membership changes.
`system_group_ids` compacts the optional per-graph `system_group_id` field from
`[B]` to `[G]`.

The operations convert between graph and group cardinality:

```text
reduce_all / reduce_any   [B]      -> [G]
broadcast                 [G, ...] -> [B, ...]
graph_mask                [G]      -> [B]
selected_group_idx        [G] mask -> dense [B_selected]
```

Example:

```text
graph_to_group   = [0, 0, 0, 1, 1, 1]
graph_mask       = [T, T, T, T, F, T]

reduce_all       = [T, F]
reduce_any       = [T, T]
broadcast([T,F]) = [T, T, T, F, F, F]
```

One unconverged image therefore prevents its complete path from graduating:

```python
path_converged = layout.reduce_all(image_converged)
completed_images = layout.broadcast(path_converged)
```

### 3.3 Group-atomic runtime helpers

These private helpers can be added when a group-aware pipeline changes batch
membership. A fixed-membership `NEB.run()` calls neither helper.

`_select_groups()` is called when convergence graduation separates complete
bands from bands that remain active:

```python
completed = _select_groups(batch, layout, group_converged)
active = _select_groups(batch, layout, ~group_converged)
```

It broadcasts the `[G]` group mask to a `[B]` graph mask, selects complete
bands, and rebases their local `group_idx`:

```python
def _select_groups(
    batch: Batch,
    layout: GroupLayout,
    group_mask: torch.Tensor,
) -> Batch:
    graph_mask = layout.graph_mask(group_mask)
    selected = batch.index_select(graph_mask)
    selected.set_group_layout(
        group_idx=layout.selected_group_idx(group_mask),
    )
    return selected
```

`_put_groups()` is called during capacity-limited refill or distributed handoff:

```python
copied_groups = _put_groups(
    send_buffer,
    active,
    layout,
    requested_groups,
)
remaining = _select_groups(active, layout, ~copied_groups)
```

It admits a band only when all of its graphs, atoms, and edges fit, calls the
existing `Batch.put()` for the admitted graphs, and returns a `[G]` mask of the
groups copied.

### 3.4 Path construction

Provide:

```python
def interpolate_endpoints(
    initial: Batch,
    final: Batch,
    n_images: int | Sequence[int],
    *,
    method: Literal["linear", "idpp"] = "idpp",
) -> Batch: ...


def concat_bands(*bands: Batch) -> Batch: ...
    # Rebase group_idx and preserve optional stable system_group_id.

def validate_bands(bands: Batch) -> None:
    # Validate GroupLayout, >=3 images per band,
    # Validate matching atoms, species, cell/PBC.
```

## 4. Group-aware dynamics

### 4.1 GroupLayout's role

`GroupLayout` adapts graph-oriented `Batch` data to group-oriented dynamics.
`Batch` owns one cached layout and records the key used to prepare it:

```text
Batch               caches one GroupLayout and its group key
BaseDynamics        selects that key or uses ungrouped behavior
FIRE2          retrieve it for grouped state and reductions
Hooks/refill        retrieve it from the same Batch
```

When `group_key` is `None`, no layout is retrieved and existing ungrouped
behavior is unchanged.

### 4.2 BaseDynamics group lifecycle

Add optional grouping without changing the public execution signatures:

```python
class BaseDynamics:
    def __init__(
        self,
        ...,
        group_key: str | None = None, # new
    ) -> None:
        self.group_key = group_key
        self._evaluate_on_admission = False  # new; preserves existing behavior

    # The following key methods will be called the same way as before
    def run(self, batch: Batch, n_steps: int | None = None) -> Batch: ...
    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]: ...
    def compute(self, batch: Batch) -> ModelOutputs: ...
```

`_prepare_masks()` runs on every admission and writes each mask directly onto
`batch`, like `status` already is (`sampler.py:365`). `fixed_node_mask` is
always allocated; when `group_key` is set, it additionally adds four group
masks:

- `group_active_mask`: which groups belong to this stage, e.g. `[True, False]`.
- `group_active_graph_mask`: those groups' images, group-consistent.
- `group_update_graph_mask`: which active images the optimizer may move -- starts
  equal to `group_active_graph_mask`, narrowed by DyNEB or endpoint exclusion
  (Section 5.1). `False` doesn't remove the image from the batch (it still
  gets a forward pass every step); only graduation (Section 4.5) does that.
- `group_reset_mask`: which groups reset optimizer state once, e.g. for a
  regular-to-CI transition or a growing-string topology change.
- `fixed_node_mask`: which individual nodes are permanently pinned regardless
  of image activity, e.g. a fixed substrate atom in an otherwise-updating
  image. All-`False` initially; a hook such as `NEBForceHook.prepare_batch()`
  writes into it. Don't add endpoint atoms here: `group_update_graph_mask`
  already excludes them, and `resolve_frozen_node_mask()` (below) broadcasts that
  exclusion to every atom of the image, so marking them again in
  `fixed_node_mask` is redundant.

```python
def _prepare_masks(self, batch: Batch) -> None:  # new
    batch.fixed_node_mask = torch.zeros(
        batch.num_nodes, dtype=torch.bool, device=batch.device,
    )  # new; populated by hooks, e.g. NEBForceHook.prepare_batch()

    if self.group_key is None:
        return  # no group_idx, so no group masks to allocate

    layout = batch.get_group_layout(self.group_key)
    batch.group_active_mask = torch.ones(
        layout.num_groups,
        dtype=torch.bool,
        device=batch.device,
    )
    batch.group_active_graph_mask = layout.graph_mask(batch.group_active_mask)
    batch.group_update_graph_mask = batch.group_active_graph_mask.clone()
    batch.group_reset_mask = torch.zeros_like(batch.group_active_mask)
```

`group_active_mask`, `group_active_graph_mask`, `group_update_graph_mask`, and
`group_reset_mask` are engine bookkeeping: allocated by `BaseDynamics`, written
by hooks, each at its own cardinality and for its own reason (Section 4.2
summary above). An optimizer's `pre_update()` should not read any of them
directly. Instead, `BaseDynamics.resolve_frozen_node_mask()` is the one method
optimizers call -- it folds all image- and node-level exclusions down to the
single `[V]` answer to "may this atom move right now":

```python
def resolve_frozen_node_mask(self, batch: Batch) -> torch.Tensor:  # new
    if self.group_key is None:
        return batch.fixed_node_mask
    frozen_graph_mask = batch.group_active_graph_mask & ~batch.group_update_graph_mask
    return frozen_graph_mask[batch.batch_idx] | batch.fixed_node_mask
```

Computed fresh from the source masks on every call rather than cached, so it
can never go stale relative to whatever a hook last wrote into
`group_update_graph_mask` or `fixed_node_mask`.

`_build_context()` needs no new fields: hooks already receive `ctx.batch`,
which now carries every mask directly.

Initial admission and batch changes use the same eager path:

```python
def _admit_batch(  # new
    self,
    batch: Batch,
    *,
    retained_groups: torch.Tensor | None = None,
    n_new_groups: int = 0,
) -> None:
    self._prepare_masks(batch)  # new

    if retained_groups is None:
        self._ensure_state_initialized(batch)  # existing
    else:
        self._sync_state_to_batch(  # existing, now passed group rows
            retained_groups,
            n_new_groups,
            batch,
        )

    ctx = self._build_context(batch)  # existing
    self._prepare_hooks(ctx)  # new - will be mentioned in Section 5

    if self._evaluate_on_admission:
        # Existing FusedStage initial-force sequence, moved here.
        self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        self.compute(batch)
        self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

        self._apply_group_reset_requests(batch)  # new
```

### 4.3 Reset requests

A retained group may change optimization regime or topology:

```text
regular NEB -> climbing-image NEB  # Effective force changes.
growing string -> insert an image  # Group topology changes.
```

That group's FIRE2 history -- `dt`, `alpha`, and positive-step count -- is no
longer valid. Only that group should reset; unaffected groups continue.

`batch` carries one reusable `[G]` request mask. Hooks request a partial
reset without directly mutating optimizer state:

```python
ctx.batch.group_reset_mask.logical_or_(transitioning_groups)
```

The engine applies the mask after `AFTER_COMPUTE`:

```python
self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
self._apply_group_reset_requests(batch)

def _apply_group_reset_requests(self, batch: Batch) -> None:
    # Unconditional in compiled execution; an all-false mask is a no-op.
    self._reset_group_state(batch, batch.group_reset_mask)
    batch.group_reset_mask.zero_()
```

FIRE2 implements the requested partial reset and retrieves the layout from
`Batch`:

```python
def _reset_group_state(
    self,
    batch: Batch,
    group_mask: torch.Tensor,  # [G]
) -> None: ...
```

The public reset starts an entirely new optimizer run. It is included for
completeness and is not currently needed for NEB.

```python
def reset_state(self, batch: Batch) -> None:
    self._prepare_masks(batch)
    self._init_state(batch)
    self.step_count = 0
    self._reset_convergence_state()
```

`_reset_group_state()` preserves unaffected groups; `reset_state()` resets all
groups and run-level counters.

### 4.4 Grouped convergence and status

Instead of extending the existing `ConvergenceHook`, a `GroupedConvergenceHook(ConvergenceHook)` is created to emphasize that grouping is used. It is possible to use data with ``group_idx`` and still use the regular convergence hook.

```python
convergence = GroupedConvergenceHook.from_fmax(
    0.05,
    group_key="group_idx",
)
```

`__call__` and `_check_convergence()` both delegate to `self.evaluate(batch)`
(`base.py:1747`, `2746`), so `GroupedConvergenceHook` overrides `evaluate()`
rather than adding separate logic:

```python
class GroupedConvergenceHook(ConvergenceHook):
    def evaluate(self, batch: Batch) -> torch.Tensor | None:
        layout = batch.get_group_layout(self.group_key)

        # Exclude fixed atoms, same as FIRE2 (Section 4.6); restore
        # immediately, since nothing later re-derives batch.forces before a
        # caller could see it.
        original_forces, batch.forces = (
            batch.forces,
            batch.forces.masked_fill(batch.fixed_node_mask[:, None], 0),
        )
        # _per_graph_converged() is ConvergenceHook's existing per-criterion
        # AND-reduce (base.py:2709-2722), factored out for reuse here.
        image_converged = self._per_graph_converged(batch)  # [B]
        batch.forces = original_forces

        # Excluded images (endpoints, via group_update_graph_mask) count as
        # trivially converged, so reduce_all only requires images actually
        # being optimized.
        frozen_graph_mask = batch.group_active_graph_mask & ~batch.group_update_graph_mask
        image_converged = image_converged | frozen_graph_mask

        group_converged = layout.reduce_all(image_converged)  # [G]
        image_converged = layout.broadcast(group_converged)   # [B]

        if not image_converged.any():
            return None
        return torch.where(image_converged)[0]
```

If registered as an `AFTER_STEP` hook -- needed for Section 4.5's refill -- `__call__` also syncs `status`,
group-consistent for `step()`'s existing `exit_status` freeze:

```python
group_status = batch.status[layout.group_ptr[:-1]]
next_group_status = update_status(current=group_status, converged=group_converged)
batch.status.copy_(layout.broadcast(next_group_status))
```

TODO: validation

### 4.5 Group-atomic refill

This extends sampler-driven `FusedStage` refill; fixed-membership `NEB.run()`
does not use it. The grouped branch is:

```python
layout = batch.get_group_layout(self.group_key)
status = batch.status.view(-1)
graduated_groups = status[layout.group_ptr[:-1]] >= exit_status
retained = _select_groups(batch, layout, ~graduated_groups)
admitted = sampler.request_replacement_groups_budget(...)
updated = concat_groups(retained, admitted)
updated.set_group_layout(rebased_group_idx, group_key=self.group_key)
self._admit_batch(updated, retained_groups=..., n_new_groups=len(admitted))
```

The implementation must:

- consume path-level status;
- sink, retain, and admit complete groups within atom, edge, and graph budgets;
- rebase and cache the replacement layout, preserve retained state by stable
  identity when available, and admit the batch before its next update.

See Appendix A for detailed pseudocode.

### 4.6 FIRE2 grouped implementation

Masking via `resolve_frozen_node_mask()` applies whether or not `group_key` is
set -- it already falls back to plain `fixed_node_mask` when ungrouped (Section
4.2). Grouping changes only the state size and the reduction index used by
`fire2_step_coord` itself:

```python
if self.group_key is not None:
    layout = batch.get_group_layout(self.group_key)
    n_state = layout.num_groups
    system_idx = layout.node_to_group.int()
else:
    n_state = batch.num_graphs
    system_idx = batch.batch_idx.int()
```

Setting `n_state = layout.num_groups` gives each path one shared set of FIRE2
state rows (`dt`, `alpha`, positive-step counter, reduction scratch).

FIRE2 reads no mask directly. It calls `self.resolve_frozen_node_mask(batch)`
(Section 4.2) and zeros force and velocity for whatever it returns. A true
per-atom exclusion inside `fire2_step_coord` itself would need a kernel change
in `nvalchemi-toolkit-ops`; out of scope here.

```python
def pre_update(self, batch: Batch) -> None:
    # State initialization and grouped system_idx resolution happen first.
    frozen_node_mask = self.resolve_frozen_node_mask(batch)
    # Transient: NEBForceHook rewrites batch.forces from the physical force every
    # step, so this zeroing never reaches the forces/physical_forces reported
    # back to the user.
    batch.forces.masked_fill_(frozen_node_mask[:, None], 0)
    batch.velocities.masked_fill_(frozen_node_mask[:, None], 0)

    fire2_step_coord(
        batch.positions.detach(),
        batch.velocities,
        batch.forces,
        system_idx,
        ...
    )
```

- The hook chooses which images are frozen; `FIRE2.pre_update()` owns the zeroing
  because it is the common optimizer boundary for both direct and fused execution.
- With force and velocity both zeroed for a frozen atom, the unchanged kernel
  leaves that atom stationary and its zero-valued terms drop out of its path's
  reductions.
- No atom- or group-level masking inside `fire2_step_coord` is needed.

The masking mutates `batch.forces` and `batch.velocities` in place rather than a
copy:

- By the time `pre_update` runs, `batch.forces` already holds the NEB *effective*
  force, and `NEBForceHook` has already copied the *physical* force into the
  preallocated `batch.physical_forces` before overwriting it (Section 2.1) -- so
  no data is lost by zeroing it here.
- In-place `masked_fill_` keeps this step compatible with compiled,
  CUDA-graph-captured execution, where a fresh per-step allocation (e.g.
  `.clone()`) would be unsafe.
- An all-false mask is simply a no-op write, the same "unconditional in compiled
  execution" pattern used for group resets (Section 4.3).

A climbing-image transition resets only the selected paths:

```python
def _reset_group_state(self, batch: Batch, group_mask: torch.Tensor) -> None:
    dt_init = _to_per_system(
        self._dt_init,
        self._state.num_graphs,
        batch.device,
        batch.positions.dtype,
    )
    self._state.dt.copy_(torch.where(group_mask, dt_init, self._state.dt))
    self._state.alpha.masked_fill_(group_mask, self.alpha0)
    self._state.nsteps_inc.masked_fill_(group_mask, 0)

    for key in ("vf", "v_sumsq", "f_sumsq"):
        getattr(self._state, key).masked_fill_(group_mask, 0)

    layout = batch.get_group_layout(self.group_key)
    node_mask = layout.node_mask(group_mask)
    batch.velocities.masked_fill_(node_mask[:, None], 0)
```

Refill uses the existing synchronizer with group indices:

```python
self._sync_state_to_batch(retained_group_idx, n_new_groups, updated_batch)
```

## 5. NEB through hooks

### 5.1 `NEBForceHook` arguments and preparation

```python
class NEBForceHook:
    def __init__(
        self,
        *,
        spring: float | SpringConfig = 0.1,
method: Literal["improved_tangent", "doubly_nudged"] = "improved_tangent",
        endpoint_mode: Literal["fixed", "relaxed"] = "fixed",
        fixed_atom_key: str | None = None,
    ) -> None: ...
```

- `spring`: a constant or an energy-dependent spring configuration.
- `method`: a named kernel in `NEB_METHODS`.
- `endpoint_mode`: fixed endpoints by default; relaxed endpoints use physical
  force perpendicular to the path.
- `fixed_atom_key`: optional node-level boolean Batch field.

The hook does not take `group_key`, CI policy, or final `fmax`; those
belong to the optimizer, `ClimbingImageHook`, and convergence hook.

`_admit_batch()` calls `_prepare_hooks(ctx)`, which calls
`prepare_batch()` on hooks that provide it:

```python
def _prepare_hooks(self, ctx: DynamicsContext) -> None:  # new
    for hook in self._iter_hooks():
        if prepare := getattr(hook, "prepare_batch", None):
            prepare(ctx)
```

`NEBForceHook.prepare_batch()` resolves those arguments against the admitted
batch:

```python
def prepare_batch(self, ctx: DynamicsContext) -> None:
    batch = ctx.batch
    layout = batch.get_group_layout(ctx.workflow.group_key)

    validate_complete_paths(batch, layout)
    fixed_atoms = resolve_node_mask(batch, self.fixed_atom_key)
    batch.fixed_node_mask.logical_or_(fixed_atoms)  # new; shares the engine-owned mask
    initialize_neb_fields(
        batch,
        endpoint_mode=self.endpoint_mode,
    )
    if self.endpoint_mode == "fixed":
        # Whole endpoint images, not individual atoms -- graph-level exclusion,
        # not fixed_node_mask. Permanent for this admission, unlike DyNEB's
        # per-step toggling of interior images (Section 10.2). Endpoints are
        # rank 0 and rank (count - 1) within each group (Section 3.2).
        last_rank = (layout.num_graphs_per_group - 1)[layout.graph_to_group]
        is_endpoint = (layout.graph_rank == 0) | (layout.graph_rank == last_rank)
        batch.group_update_graph_mask &= ~is_endpoint

    self._workspace = _NEBWorkspace.from_batch(
        batch,
        group_layout=layout,
        spring=self.spring,
        method=self.method,  # selects required scratch
        fixed_atom_mask=fixed_atoms,
    )
```

This initializes endpoint/interior `force_mode` and replaceable kernel
buffers.

`resolve_node_mask()` still does the one real lookup -- turning `fixed_atom_key`
into a tensor -- but only `NEBForceHook` calls it. FIRE2 never resolves a key
itself; it only reads the engine-owned `fixed_node_mask` that this hook writes
into (Section 4.6).

### 5.2 `NEBForceHook` after compute

After each model evaluation, the hook preserves physical force, refreshes any
energy-dependent spring constants, and publishes effective NEB force:

```python
def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
    batch = ctx.batch
    layout = batch.get_group_layout(ctx.workflow.group_key)

    batch.physical_forces.copy_(batch.forces)

    # Constant spring only needs to be prepared once.
    # Energy-dependent springs will need to be refresh every step
    update_spring_constants(
        self._workspace,
        energies=batch.energy,
    )
    
    # neb_forces will be initially in toolkit, but will be moved to toolkit-ops in the future
    # `method` resolves a registered static NEB specification. On first use for a
    # device, its tangent and effective-force functions are specialized into a
    # Warp kernel and cached. Subsequent calls launch the cached kernel.
    neb_forces(  
        method=self.method,
        positions=batch.positions,
        physical_forces=batch.physical_forces,
        image_energies=batch.energy.view(-1),
        image_ptr=batch.batch_ptr,
        path_ptr=layout.group_ptr,
        image_path_idx=layout.graph_to_group,
        spring_constants=self._workspace.spring_constants,
        fixed_atom_mask=self._workspace.fixed_atom_mask,
        image_force_mode=batch.force_mode,  # fixed / relaxed / regular / climbing
        cell=self._workspace.cell,
        inv_cell=self._workspace.inv_cell,
        pbc=self._workspace.pbc,
        tangent_buffer=self._workspace.tangent_buffer, # method dependent
        effective_forces=self._workspace.effective_forces,
        link_lengths=self._workspace.link_lengths,
    )

    active_nodes = layout.node_mask(batch.group_active_mask)
    batch.forces[active_nodes] = self._workspace.effective_forces[active_nodes]
```

A constant spring may be prepared once; only energy-dependent springs need the
refresh. The same kernel handles regular and climbing force according to
`force_mode`.

### 5.3 Climbing image

`ClimbingImageHook` selects the climbing image, tracks when climbing should start,
and requests the optimizer-state reset that follows:

```python
class ClimbingImageHook:
    def __init__(
        self,
        *,
        activation: Literal["after_regular", "immediate"] = "after_regular",
        reselection: Literal["locked", "every_evaluation"] = "locked",
        max_regular_steps: int | None = None,
        max_climbing_steps: int | None = None,
    ) -> None: ...
```

The default runs regular NEB to convergence, selects the highest-energy
interior image, resets only that group's optimizer state, and continues with a
locked climber. `max_regular_steps` can force that transition;
`max_climbing_steps` limits the CI stage. `activation="immediate"` selects on
the initial evaluation, cannot be combined with `max_regular_steps`, and does
not reset fresh optimizer state. `reselection="every_evaluation"` updates the
climber from current physical energies without resetting optimizer state when
the selected image changes. The path stays active in either phase.

### 5.4 Diagnostics

`NEBDiagnosticsHook` runs after `NEBForceHook` and publishes values used by
convergence and reporting:

```python
batch.neb_fmax = reduce_path_fmax(batch.forces, layout)
batch.neb_barrier = reduce_path_barrier(batch.energy, layout)
batch.neb_path_length = reduce_path_length(batch.positions, layout)
batch.neb_saddle_idx = reduce_path_argmax(batch.energy, layout)
```

### 5.5 Ordering and context

```text
admission -> prepare_batch
model
-> ClimbingImageHook      # set CI mode and request reset
-> NEBForceHook           # physical force -> effective CI force
-> NEBDiagnosticsHook     # convergence/reporting fields
-> engine resets requested FIRE2 group rows following request from CI
-> next FIRE2 update
```

The hook sets bits in `ctx.batch.group_reset_mask`. After all `AFTER_COMPUTE`
hooks, the engine applies and clears the request:

```python
self._reset_group_state(batch, batch.group_reset_mask)
batch.group_reset_mask.zero_()
```

This clears stale velocity/controller history before those groups use climbing
force. The hook does not access FIRE2 state directly.

No `HookContext` or `DynamicsContext` change is required. Dynamics hooks
already receive `ctx.batch`, which now carries every mask (Section 4.2)
directly:

```python
@dataclass(kw_only=True)
class DynamicsContext(HookContext):
    step_count: int = 0
    converged_mask: torch.Tensor | None = None
```

## 6. Serialization and results

The `NEB` class serializes named configuration: spring, `fmax`, climbing
policy, optimizer type and scalar arguments, group key, step limits, and
tangent/projection method.

Optimization returns the same canonical image `Batch`, updated with:

```text
energy             final physical per-image energy
physical_forces    final physical per-atom force
forces             final effective NEB optimizer force
force_mode         endpoint / regular / climber
neb_phase          regular / climbing / complete
```

The batch retains `group_idx`, optional `system_group_id` and `system_id`, and
group-consistent `status`. Image order is given by `batch._group_layout.graph_rank`.

Per-image effective and physical maximum force norms can be derived from `forces`
and `physical_forces`, respectively. See v3 proposal for additional fields for results / diagnostics.

## 7. `FusedStage` changes


## 8. `DistributedPipeline` changes


## 9. Compilation and performance



## 10. Future method support

### 10.1 Growing string

Growing string changes path topology and remains an eager lifecycle operation:

- `StringGrowthHook` selects paths and constructs replacement images but only
  queues the replacement during hook dispatch.
- After the step, `BaseDynamics` admits the replacement, rebuilds
  `GroupLayout` and hook scratch, preserves unchanged group state, and resets
  groups whose topology changed.
- `BaseDynamics` force-primes the admitted batch before its next optimizer
  update without rerunning the growth policy.
- Priming adds one model evaluation but is not an optimizer iteration. The
  fixed-topology steady-state step remains compilable.

### 10.2 DyNEB

DyNEB is compatible with the proposed architecture because it changes only
per-image optimizer activity.

`DyNEBActivityHook` runs after `NEBForceHook` and reuses the regular
`ConvergenceHook` criterion through a new `evaluate_mask(batch) -> [B]`
method. It writes `batch.group_update_graph_mask` for the next optimizer
step without changing `group_active_graph_mask` or `status`.

- Unconverged interior images in active paths remain enabled; the
  saddle/climbing image also remains enabled and endpoints remain fixed.
- The hook reevaluates every image after each force evaluation, allowing a
  temporarily frozen image to reactivate. FIRE2 zeroes the effective forces and
  velocities of images that remain frozen immediately before its update.
- The same per-image convergence mask is reduced with `GroupLayout.reduce_all()`
  to decide complete-path convergence.
- No FIRE2 algorithm or kernel changes are needed; its wrapper only applies the
  image mask by zeroing force and velocity before calling the existing kernel.
- A custom criterion may implement distance-scaled thresholds.

## Appendix A. Group-atomic refill pseudocode

The existing graph-wise path remains unchanged when `group_key is None`.
The grouped branch is:

```python
def refill_check(
    self,
    batch: Batch,
    exit_status: int,
) -> Batch | None:
    if self.group_key is None:
        ...  # Existing graph-wise refill is unchanged.

    # The admitted batch owns the layout for its current membership.
    layout = batch.get_group_layout(self.group_key)

    # Grouped convergence keeps status constant within each group.
    # Per-image DyNEB activity uses a separate mask and is ignored here.
    status = batch.status.view(-1)
    group_status = status[layout.group_ptr[:-1]]
    graduated_groups = group_status >= exit_status
    if not graduated_groups.any():
        return batch

    graduated_graphs = layout.graph_mask(graduated_groups)
    self._overflow_to_sinks(batch, mask=graduated_graphs)

    retained_group_idx = torch.where(~graduated_groups)[0]
    retained = _select_groups(batch, layout, ~graduated_groups)

    admitted = self.sampler.request_replacement_groups_budget(
        atom_budget=remaining_atoms(retained),
        edge_budget=remaining_edges(retained),
        graph_budget=remaining_graphs(retained),
    )

    updated = concat_groups(retained, admitted)
    updated.set_group_layout(
        rebase_group_idx(getattr(updated, self.group_key)),
        group_key=self.group_key,
    )

    self._admit_batch(
        updated,
        retained_groups=retained_group_idx,
        n_new_groups=len(admitted),
    )
    return updated
```

When `system_group_id` is present, retained state is matched by stable identity
rather than the rebased local group index.
