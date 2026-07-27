<!-- markdownlint-disable MD014 -->

(serialization_guide)=

# Serialization and Reproducibility

Reproducibility in `nvalchemi` rests on one idea: **an object is described by a
recipe, not by a memory dump**. A recipe records _how to rebuild_ something — an
importable path plus the keyword arguments it was constructed with — as plain
JSON. Rebuilding imports the target and calls it again.

This is a deliberate rejection of `pickle`. Pickle stores bytecode-level state,
which makes it both a security liability (unpickling executes arbitrary code)
and a reproducibility liability (a pickle silently rots when the class it
describes changes). A JSON recipe is inspectable, diffable, reviewable, and
survives refactors loudly rather than silently.

The machinery lives in {py:mod}`nvalchemi._serialization` and
{py:class}`~nvalchemi.training.BaseSpec`, and is used across the toolkit —
model wrappers, optimizers, schedulers, loss terms and weight schedules, hooks,
and the training CLI all describe themselves the same way. This page explains
the mechanics once; {doc}`training` and {doc}`/modules/training/checkpoints`
cover how training applies them.

```{tip}
The short version: **make every constructor argument JSON-representable, and
store it on `self` under the same name.** Everything below is the reasoning
behind that one rule.
```

## Two artifacts: the recipe and the checkpoint

It helps to keep these separate, because they answer different questions.

| Artifact | Contains | Answers |
|---|---|---|
| **Spec / recipe** (JSON) | `cls_path` + constructor kwargs | "How do I rebuild this object?" |
| **Checkpoint** (JSON + tensor bundles) | Recipe _plus_ weights, optimizer state, counters, hook state | "How do I resume this exact run?" |

A checkpoint is a superset: it embeds recipes for everything reconstructible and
stores learned state alongside them. Tensor bundles are written and reloaded
with `torch.load(..., weights_only=True)` — the only pickle-free weight path
PyTorch offers.

{py:class}`~nvalchemi.training.TrainingStrategy` exposes the recipe layer on its
own via `to_spec_dict()` / `from_spec_dict()`, which round-trip the training
configuration without any tensors — useful for describing, diffing, or
version-controlling an experiment before you run it.

## Anatomy of a spec

{py:class}`~nvalchemi.training.BaseSpec` is a Pydantic model with two metadata
fields — `cls_path` (validated at construction to be importable) and
`timestamp` — plus **one field per constructor keyword argument**. Concrete spec
classes are built dynamically by
{py:func}`~nvalchemi.training.create_model_spec`, which inspects the target's
signature and annotates each field accordingly.

```python
from nvalchemi.training import create_model_spec

spec = create_model_spec(MyModel, hidden_size=128, num_layers=4, cutoff=6.0)
spec.model_dump_json()
```

```json
{
  "cls_path": "my_package.models.MyModel",
  "timestamp": "2026-07-27T18:04:11.921043+00:00",
  "hidden_size": 128,
  "num_layers": 4,
  "cutoff": 6.0
}
```

Rebuilding is the mirror image — {py:func}`~nvalchemi.training.create_model_spec_from_json`
rehydrates the spec object, and `build()` imports `cls_path` and calls it:

```python
from nvalchemi.training import create_model_spec_from_json

model = create_model_spec_from_json(spec_dict).build()
```

`build()` also accepts positional and keyword arguments for values that
_cannot_ be serialized because they only exist at runtime — an optimizer needs
live `model.parameters()`, a scheduler needs a live optimizer:

```python
optimizer = optimizer_spec.build(model.parameters())
scheduler = scheduler_spec.build(optimizer)
```

If the target's signature has drifted since the spec was written, `build()`
raises `TypeError` naming the `cls_path` **and the spec's timestamp** — the
failure is explicit rather than a silent misconstruction.

## What a spec can hold

A field value must be representable in JSON, either natively or through the
type-serializer registry.

**Natively:** strings, numbers, booleans, `None`, lists, and dicts of the same.

**Through the registry**, four types are pre-registered:

| Type | JSON form |
|---|---|
| {py:class}`torch.dtype` | its string name (rehydrated behind an `isinstance` guard, so a hostile string cannot smuggle arbitrary `torch.*` attributes through `getattr`) |
| {py:class}`torch.device` | its string form |
| {py:class}`torch.Tensor` | `{dtype, shape, data}` — a data structure, not a bytecode payload |
| `type` (a class object) | its dotted import path |

**Nested specs** are supported: a spec field may itself hold a `BaseSpec`, and
`build()` constructs it recursively before passing it to the outer constructor.
Non-empty lists/tuples of specs are built item-wise, preserving non-spec items
and the container type.

```{warning}
Two limits are worth committing to memory:

- **Nested collections are not traversed.** `list[list[BaseSpec]]` will not be
  rebuilt element-wise. Flatten the collection, or wrap it in an object that
  has its own spec.
- **Positional-only parameters are rejected.** `create_model_spec` raises
  `TypeError` for targets that declare them, because a spec addresses every
  argument by name.
```

## How a model spec is created

When a checkpoint is written, each model is asked for a spec in a fixed order of
precedence:

```{graphviz}
:caption: Spec resolution for a model at checkpoint time.
:alt: Spec resolution order

digraph spec_resolution {
    rankdir=TB
    node [shape=box style="rounded,filled" fontsize=11]

    start [label="model to serialize"]
    explicit [label="does it define\ncheckpoint_spec()?"]
    use_explicit [label="use the returned BaseSpec\n(trusted, no rebuild check)" fillcolor="#26351d"]
    introspect [label="fall back to attribute\nintrospection", fillcolor="#183449"]
    validate [label="rebuild from the spec\nand verify the result", fillcolor="#4a3315"]
    ok [label="spec stored in checkpoint" fillcolor="#26351d"]
    omit [label="UserWarning:\n'Omitting model spec'\nweights saved, recipe lost" fillcolor="#4a1515"]

    start -> explicit
    explicit -> use_explicit [label="yes"]
    explicit -> introspect [label="no"]
    use_explicit -> ok
    introspect -> validate
    validate -> ok [label="rebuild succeeds"]
    validate -> omit [label="raises"]
}
```

**1. An explicit `checkpoint_spec()`.** If the model defines a callable
`checkpoint_spec()` returning a `BaseSpec` (or `None` to decline), that spec is
used as-is. Returning anything else raises `TypeError`. This is the escape hatch
for models whose constructor arguments are transformed before being stored —
a wrapper that converts a checkpoint path into a live module, for example.

**2. Attribute introspection.** Otherwise the framework reads the target's
`__init__` signature and, for each parameter (skipping `self`, `*args`, and
`**kwargs`), looks for **an attribute of the same name on the instance**. A
parameter with no matching attribute is silently skipped. Submodule values are
recursed into, producing nested specs.

That second path is why the golden rule matters. Given:

```python
class MyModel(BaseModelMixin):
    def __init__(self, hidden_size: int, cutoff: float):
        super().__init__()
        self.hidden_size = hidden_size      # discoverable
        self.r_cut = cutoff                 # NOT discoverable — name differs
```

`hidden_size` is recovered and `cutoff` is not, so the rebuilt model silently
falls back to the constructor default for `cutoff`. Storing it as `self.cutoff`
fixes it.

Because introspection is a heuristic, the framework **validates it at save
time**: the spec is rebuilt and the result checked. If that fails, the spec is
dropped with a `UserWarning` (`Omitting model spec for '<name>'`) — the weights
are still saved, but the checkpoint can no longer reconstruct the architecture
on its own. Treat that warning as an error in any run you intend to resume.

## Designing serializable code

Six rules, in rough order of how often they bite:

1. **Store every constructor argument on `self` under the same name.** This is
   what makes the introspection fallback work.
2. **Keep constructor arguments JSON-representable** — natives, registered
   types, or nested objects that have their own specs. Push non-serializable
   inputs (open file handles, live modules, sessions) behind a factory that
   takes serializable arguments instead.
3. **Avoid positional-only parameters** in anything you expect to be
   reconstructed.
4. **Make callables importable.** Functions referenced by a recipe must be
   module-level: lambdas, closures, locally-defined functions, and bound methods
   are rejected with an explicit error, because a dotted path cannot address
   them.
5. **Register custom types** you want to appear in constructor arguments (next
   section).
6. **Implement `checkpoint_spec()`** when your constructor arguments are
   transformed rather than stored — that is, whenever rule 1 is impossible.

## Registering a custom type

{py:func}`~nvalchemi.training.register_type_serializer` teaches the spec layer a
new type. Supply the type and a symmetric pair of functions:

```python
from nvalchemi.training import register_type_serializer

register_type_serializer(
    MyUnitSystem,
    serialize=lambda u: u.name,          # -> JSON-safe value
    deserialize=MyUnitSystem.from_name,  # <- back to the instance
)
```

Both directions should be total and side-effect free, and `deserialize` should
validate its input rather than trusting it — the value may come from a file
written elsewhere.

## What is deliberately never serialized

Some things are excluded by design, and no amount of configuration will include
them:

- **`training_fn` and `loss_target_assembler`** are recorded only as importable
  dotted paths, never as code. There is no way to guarantee a serialized
  callable is safe to execute or that it has not been swapped in flight. Supply
  them again at load time, and keep them versioned alongside your checkpoints.
- **Hooks** are runtime objects: you reconstruct them in your script and pass
  them at load time. Only hooks implementing
  {py:class}`~nvalchemi.hooks.CheckpointableHook` have their _state_ saved and
  restored into the live objects you supply.
- **Anything reducible to neither an importable reference nor serializable
  arguments** — arbitrary Python objects, open handles, live modules.

The practical consequence: your training script is part of the reproducible
artifact. The checkpoint stores data and references; the script supplies the
code.

## Reproducibility checklist

Before trusting a long run to be resumable:

- [ ] The run produced **no** `Omitting model spec` warnings.
- [ ] `training_fn` and `loss_target_assembler` are importable module-level
      functions, versioned with the checkpoint.
- [ ] Custom loss weight schedules implement `to_spec()`.
- [ ] Hooks owning restart-critical state implement
      {py:class}`~nvalchemi.hooks.CheckpointableHook`.
- [ ] Custom types in constructor arguments are registered.
- [ ] A load-and-resume has actually been exercised, not merely assumed.

```{note}
`nvalchemi` does not seed RNGs on your behalf. Bitwise-identical reruns also
require seeding `torch`, and the usual determinism caveats for
non-deterministic CUDA kernels still apply. Serialization guarantees the
_recipe_ is faithfully reproduced, not that floating-point results are
bit-identical across hardware.
```

## See also

- {doc}`training` — reproducibility in the training lifecycle.
- {doc}`/modules/training/checkpoints` — checkpoint layout, restart semantics,
  and the full save/load API.
- {doc}`/modules/training/index` — `BaseSpec`, `create_model_spec`, and
  `register_type_serializer` reference.
