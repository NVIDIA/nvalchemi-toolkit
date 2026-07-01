<!-- markdownlint-disable MD014 -->

(distributed_byo_guide)=

# Bring Your Own Model: Authoring a Distribution Spec

This guide walks through deploying a new
{py:class}`~nvalchemi.models.base.BaseModelMixin` wrapper under domain
decomposition. The arc is the same regardless of the model's internals:

1. **Wrap the model** with `BaseModelMixin` — the standard
   single-process pattern documented in {doc}`models`.
2. **Declare or derive a spec.** Most models drop into one of the
   shipped {py:class}`~nvalchemi.distributed.spec.MLIPSpec` presets;
   models with custom kernels need an
   {py:class}`~nvalchemi.distributed.spec.OpAdapter` declaration.
3. **Validate.** Use
   {py:func}`~nvalchemi.distributed.validate.trace_and_validate` to
   confirm the multi-rank forward matches single-process to
   tolerance — and, when it doesn't, get a diagnostic that points at
   the root cause.
4. **Persist the spec.** {py:meth}`MLIPSpec.save` writes a JSON
   artefact alongside your checkpoint;
   {py:meth}`MLIPSpec.load` reads it back. Production wrappers ship
   the saved spec so distributed deployment is a one-line
   construction.

The two runnable walkthroughs in `examples/distributed/`:

| Example | Path | Demonstrates |
|---|---|---|
| Pure-PyTorch model | `04_byo_pytorch_mpnn.py` | Behler-Parrinello descriptor, `SPEC_MPNN_HALO` preset, autograd forces |
| Model with a Warp kernel | `05_byo_graph_transformer.py` | Custom op + `OpAdapter` declaration, energy-only validation |

This guide stitches the patterns those examples illustrate into a
reference workflow.

## Step 1: Wrap your model

The wrapper inherits from {py:class}`torch.nn.Module` and
{py:class}`~nvalchemi.models.base.BaseModelMixin`. It declares a
{py:class}`~nvalchemi.models.base.ModelConfig`, translates a
{py:class}`~nvalchemi.data.Batch` into the inner model's kwargs in
`adapt_input`, and translates the inner model's return into a
{py:class}`~nvalchemi._typing.ModelOutputs` ordered dict in
`adapt_output`.

```{tip}
The wrapper is **single-process** in design. Domain decomposition is
opt-in via the spec; the same wrapper runs unchanged whether you
call it directly or under
{py:class}`~nvalchemi.distributed.DomainParallel`. Resist the
temptation to add halo / mesh awareness to the wrapper body —
declare it on the spec instead.
```

The minimal pattern:

```python
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin, ModelConfig, NeighborConfig, NeighborListFormat,
)

class MyWrapper(nn.Module, BaseModelMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            neighbor_config=NeighborConfig(
                cutoff=model.cutoff, format=NeighborListFormat.COO
            ),
        )

    def adapt_input(self, data, **kw):
        # Build the inner model's kwargs from the Batch.
        return {"positions": data.positions, ...}

    def adapt_output(self, raw, data):
        # Reorder the inner model's return into a ModelOutputs.
        return OrderedDict(raw)

    def forward(self, data):
        return self.adapt_output(self.model(**self.adapt_input(data)), data)
```

See {doc}`models` for the full pattern.

## Step 2a: Pick a spec preset

If your model fits one of the patterns these presets cover, the
spec is a single import:

```python
from nvalchemi.distributed.spec import SPEC_MPNN_HALO

class MyWrapper(nn.Module, BaseModelMixin):
    @property
    def distribution_spec(self):
        return SPEC_MPNN_HALO
```

The shipped presets cover:

- **`SPEC_MPNN_HALO`**: scatter-heavy MPNNs (MACE, NequIP, Allegro,
  ORB, generic message-passing models with autograd forces).
- **`SPEC_LJ_HALO`**: pair potentials with kernel-direct forces
  (Lennard-Jones, Buckingham, Morse).
- **`SPEC_UMA_GATHER`**: UMA-style eSCN backbones — halo storage but
  with `scatter="local"` because the backbone isn't halo-aware (it
  computes its own internal full-graph edge index).
- **`SPEC_AIMNET2_GATHER`**: charge-equilibration networks with
  global per-system reductions; sharded storage.
- **`SPEC_EWALD_HALO`** / **`SPEC_PME_HALO`**: long-range
  electrostatics with reciprocal-space dispatch via `OpAdapter`s.

If your model is structurally identical to one of these — that's the
pedagogical point of declaring presets — you're done with spec
authoring. Skip to step 3.

## Step 2b: Author a custom spec

Models with non-PyTorch kernels (Warp / Triton / fused CUDA ops) need
an explicit
{py:class}`~nvalchemi.distributed.spec.OpAdapter` declaration on the
spec. The adapter tells the framework:

- Which op handle to install a handler on
  (`torch.ops.<namespace>.<op>.default`).
- What pre-processing each input position needs
  (`arg_transforms`).
- What post-processing each output position needs
  (`output_transforms`).

```python
from nvalchemi.distributed.ops import HaloStoragePolicy, ScatterOutputs
from nvalchemi.distributed.spec import (
    DistributionSpec, MLIPSpec, OpAdapter, OutputKind,
)

class MyWrapper(nn.Module, BaseModelMixin):
    @property
    def distribution_spec(self):
        return MLIPSpec(
            distribution=DistributionSpec(
                # The per-field storage layout. HaloStoragePolicy() keeps
                # ``[owned | halo]`` rows of every per-atom tensor; its
                # scatter_mode/gather_mode default to "halo_correction"/
                # "halo_read" (the right choice for an MPNN).
                policy=HaloStoragePolicy(),
                # One OpAdapter per opaque kernel the framework can't see
                # into (Warp / Triton / fused CUDA).
                custom_ops=(
                    OpAdapter(
                        op=torch.ops.mymodel.fused_message.default,
                        arg_transforms={},  # inputs already halo-padded
                        output_transforms={0: ScatterOutputs()},
                    ),
                ),
            ),
            output_kinds={
                "energy": OutputKind.PER_GRAPH,
                "forces": OutputKind.PER_NODE,
            },
        )
```

The available transforms (importable from
{py:mod}`nvalchemi.distributed.ops`):

| Transform | Pre/post-kernel action | Use case |
|---|---|---|
| `GatherInputs` | halo-pad owned input to `(n_padded, *F)` | Kernel needs neighbour rows but receives an owned-only tensor |
| `GatherInputsFull` | full-gather sharded input to `(n_global + 1, *F)` | Sharded-storage analogue (AIMNet2 conv) |
| `SliceOwned` | slice halo-padded input to `(n_owned, *F)` | Kernel must integrate over each owned atom exactly once (Ewald stage 1) |
| `ScatterOutputs` | `halo_reverse + halo_forward` on per-atom output | Per-receiver scatter from a kernel; halo rows carry partial sums |
| `AllReduceSum` | cross-mesh `SUM` all-reduce | Per-rank partial that needs cross-rank summation |
| `SliceOutputsOwned` | slice `(n_global + 1, *F)` to `(n_owned + 1, *F)` | Sharded-storage analogue of slicing back to per-rank |

See `examples/distributed/05_byo_graph_transformer.py` for a complete
walkthrough authoring an `OpAdapter` for a Warp kernel.

### `output_kinds`: explicit shape classification

Every spec should declare `output_kinds` for the keys in
`active_outputs`:

| Kind | Meaning |
|---|---|
| `PER_NODE` | per-atom, shape `(n_atoms, *F)` (or `(n_padded, *F)` under halo) |
| `PER_GRAPH` | per-system, shape `(n_systems, *F)` |
| `GLOBAL` | already globally-correct on every rank; passthrough |
| `UNKNOWN` | fall back to the legacy `shape[0] == n_padded` heuristic + warn |

Phase 4 of the distributed refactor introduced this enum to replace
the shape heuristic. Always declare; the heuristic exists for back-compat
only.

## Step 3: Validate with `trace_and_validate`

```python
from nvalchemi.distributed.validate import trace_and_validate

def model_factory():
    """Fresh wrapper instance — called once in the launcher and once per
    spawned worker. Workers re-import the launcher's module, so the
    factory must live at module scope (not inside a function or
    closure)."""
    return MyWrapper(MyModel(...)).cuda()

report = trace_and_validate(
    model_factory=model_factory,
    sample_batch=sample_batch,    # a small representative Batch
    world_size=2,                 # virtual ranks on the same GPU
    device="cuda:0",
    atol=1e-4,
    rtol=1e-3,
)

if report.ok:
    print(f"PASSED in {len(report.attempts)} attempt(s).")
    if report.fix_applied:
        print(f"auto-fix: {report.fix_applied!r}")
else:
    print(f"FAIL — {report.next_action.splitlines()[0]}")
```

What the validator does:

1. **Reference run** — single-process forward with
   `dispatch_trace` + `helper_trace` + per-atom NL summary captured.
2. **Inferred initial spec** — reads
   `wrapper.distribution_spec`. If `None`, falls back to a halo-storage
   default.
3. **Multi-rank validation** — spawns `world_size` workers on the
   same GPU, runs each through
   {py:class}`DistributedModel(spec=spec)`, compares per-output
   tensors against the reference using a partition-invariant diff
   metric (`min(elementwise, sum, max-magnitude)`).
4. **Auto-fix** — when the diff exceeds tolerance, tries a small
   corpus of rule-based mutations:

   - `halo_correction → local`: when a halo-correction handler fired
     and outputs still diverge, try disabling it.
   - `drop_extra_all_reduce`: when an `all_reduce_outputs` key shows
     a `× world_size` blow-up, drop it from the set.

   The first rule whose result clears tolerance wins. The returned
   `report.spec` is the working spec.

The {py:class}`~nvalchemi.distributed.validate.types.TraceReport`
returned by the validator carries:

- `ok`: bool — passed or not.
- `spec`: the working spec (best variant on failure).
- `attempts`: list of every spec tried, with diff metrics +
  helper diagnostics + halo-completeness verdict.
- `next_action`: a one-line guidance string.
- `fix_applied`: rule name when auto-fix engaged, else `None`.

### When validation fails

The report includes two diagnostic fields that point at common
failure modes:

- **`attempts[-1].halo_completeness`**: cross-references each rank's
  halo-padded NL against single-process's NL. A mismatch (`matches:
  False`) means the partition is dropping edges — the most common
  cause of "model output diverges by a few percent under partial
  halo coverage." Spec-level fixes can't recover the missing edges;
  the halo construction or test batch needs changing.
- **`attempts[-1].helper_diagnostics`**: surfaces helpers that look
  like distribution gaps (per-system reductions whose per-rank
  outputs sum to the reference output but aren't declared in
  `spec.distribution.third_party_helpers`). This is the AIMNet2 `mol_sum`
  pattern — the diagnostic flags it explicitly.

Failure modes the auto-fix doesn't cover that you'll see in the
report:

- **Stress-style residual on energy / forces.** Halo-row partial
  scatter sums weren't reverse-exchanged. Fix: declare
  `output_transforms={0: ScatterOutputs()}` on the relevant
  `OpAdapter`.
- **Features collapse / NaN.** Likely an unnecessary
  `arg_transforms={0: GatherInputs()}` on an already-halo-padded
  input — double-padding produces unexpected shapes. Fix: drop the
  transform.
- **Per-system output is `world_size × ref`**: the model's per-system
  scatter is firing per-system-reduce (which all-reduces) AND the
  spec lists the key in `all_reduce_outputs` (which all-reduces
  again). Fix: drop from `all_reduce_outputs`.

## Step 4: Persist + load

```python
from nvalchemi.distributed.spec import MLIPSpec
from pathlib import Path

# After validation passes:
report.spec.save(Path("my_model_spec.json"))

# Later, in production:
spec = MLIPSpec.load(Path("my_model_spec.json"))
wrapper = MyWrapper(MyModel.from_checkpoint(...))
domain_cfg = DomainConfig(cutoff=wrapper.cutoff, mesh=mesh)
dist_model = DistributedModel(wrapper, domain_cfg, spec=spec)
# … integrate dist_model under DomainParallel + an integrator.
```

The JSON format is versioned (`"version": 2`) and stable across
nvalchemi-toolkit releases for the same major version. Op handles
serialise as schema-qualified strings (`"<namespace>::<op>"`) so the
loader resolves them via {py:func}`torch.ops` at load time —
the registering module must already be imported (typically a
side-effect of constructing the wrapper).

```{tip}
Spec JSON files commit cleanly into a model checkpoint repository.
The convention used by the shipped wrappers (MACE, AIMNet2, UMA) is
to ship the spec alongside the checkpoint and load it in the
wrapper's ``distribution_spec`` property — so the spec is a
deployment artefact, not a developer artefact.
```

## Common patterns by model class

### Pure-PyTorch MPNN with autograd forces

This is the easy path. Use `SPEC_MPNN_HALO`. The framework promotes
`data.positions` to a ShardTensor view; `torch.zeros_like(positions)`
in the model preserves the subclass; per-layer `scatter_add_` calls
fire halo-correction. The wrapper's only distributed-aware code is
the autograd-leaf walk in `adapt_input` (because
:func:`torch.autograd.grad` needs the underlying leaf, not the
ShardTensor alias).

See `examples/distributed/04_byo_pytorch_mpnn.py`.

### Model with a custom op (Warp / Triton)

Declare an `OpAdapter` on the spec. For each output position the
kernel writes that the framework needs to reduce, declare the
transform — typically `ScatterOutputs()` for per-atom outputs and
`AllReduceSum()` for per-system partials. Inputs usually need no
transform (the wrapper passes already-halo-padded `positions`).

If the kernel computes forces internally via a fused gradient,
register an autograd formula via
{py:func}`torch.library.register_autograd` so PyTorch's autograd can
backprop through it.

See `examples/distributed/05_byo_graph_transformer.py`.

### Model with a third-party Python helper (e.g. AIMNet2 `mol_sum`)

Some models call into third-party Python helpers that aren't aware
of distribution — e.g. a `mol_sum` that reads `mol_idx[-1] + 1` for
its output size, which is wrong under partition. Declare a
{py:class}`~nvalchemi.distributed.PythonAdapter` on
the spec's `third_party_helpers`. The framework's
{py:class}`~nvalchemi.distributed.AdapterRegistry`
swaps in your distribution-aware replacement on scope-entry and
restores the original on scope-exit.

```python
from nvalchemi.distributed import PythonAdapter

PythonAdapter(
    module_path="aimnet.nbops",
    attr_name="mol_sum",
    replacement=_my_distributed_mol_sum,
)
```

When the replacement closes over runtime metadata (halo config /
gather meta), build it inside the wrapper's `distributed_setup(ctx)`
hook and install via the adapter's `install()` method directly — see
`AIMNet2Wrapper.distributed_setup` for the canonical pattern.

## Reference

All of these import from the public `nvalchemi.distributed` surface (or
its `spec` / `ops` / `validate` submodules) — a BYO model never reaches
into a private `_core` module.

| Symbol | Import from | Notes |
|---|---|---|
| `MLIPSpec` | {py:mod}`nvalchemi.distributed.spec` | Top-level spec: distribution, output_kinds, owned_only / all_reduce sets |
| `DistributionSpec` | {py:mod}`nvalchemi.distributed.spec` | policy + custom_ops + third_party_helpers, no chemistry vocabulary |
| `HaloStoragePolicy` / `PlainShard` | {py:mod}`nvalchemi.distributed.ops` | Per-field storage layout |
| `OpAdapter` | {py:mod}`nvalchemi.distributed` | One custom-op handler declaration |
| `JitAdapter` / `PythonAdapter` | {py:mod}`nvalchemi.distributed` | Third-party-helper replacements |
| transforms (`ScatterOutputs`, `GatherInputs`, …) | {py:mod}`nvalchemi.distributed.ops` | Per-arg / per-output kernel transforms |
| `OutputKind` | {py:mod}`nvalchemi.distributed` | Per-output shape classification |
| `trace_and_validate` | {py:mod}`nvalchemi.distributed.validate` | Validator entry point |

For the architecture behind this workflow — storage policies, the
ShardTensor dispatch model, and how the framework owns halo exchange,
caps, and compile — see {doc}`distributed_design`.

## Next steps

- The runnable walkthroughs:
  `examples/distributed/04_byo_pytorch_mpnn.py` and
  `examples/distributed/05_byo_graph_transformer.py`.
- The {doc}`distributed user guide <distributed>` for the full
  per-step architecture.
- The {doc}`ShardTensor walkthrough <distributed_shardtensor>` for
  the dispatch mechanics behind the spec.
