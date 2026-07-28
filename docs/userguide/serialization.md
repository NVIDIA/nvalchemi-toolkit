<!-- markdownlint-disable MD014 -->

(serialization_guide)=

# Serialization and Reproducibility

Reproducibility in `nvalchemi` rests on one idea: **an object is described by a
recipe, not by a memory dump** (i.e. not `pickle`!).
A recipe records _how to rebuild_ something — an
importable path plus the keyword arguments it was constructed with — as plain
JSON and the rebuilding process simply imports the target and calls it again;
for model weights and checkpoint loading, this means we rebuild the object
before adding state back in. While this might _seem_ like more work, decoupling
the process means better security (i.e. you aren't running arbitrary code),
less redundancy (one spec is defined for all checkpoints), and better
reproducibility and version control.

```{tip}
This documentation is primarily intended for developers to ensure their
code is round-trip serializable. For users of things like training, it
can be helpful to understand the internals but the intended workflow
should follow existing API. For example,
{py:class}`~nvalchemi.training.TrainingStrategy` provides methods like
`to_spec_dict()` / `from_spec_dict()` for this purpose, or at a higher
level, use checkpoints.
```

## What a recipe looks like

The recipe itself is a {py:class}`~nvalchemi.training.BaseSpec`: a `pydantic`
model holding the dotted import path of whatever it describes (`cls_path`,
validated the moment it is created, so a typo or a moved class fails there
rather than months later at load time), a `timestamp`, and one field per
constructor keyword argument. You never write one of these classes by hand —
{py:func}`~nvalchemi.training.create_model_spec` reads the target's signature
and generates the spec class on the fly, annotating each field to match.

```python
from nvalchemi.training import create_model_spec

spec = create_model_spec(MyModel, hidden_size=128, num_layers=4, cutoff=6.0)
spec.model_dump_json()
```

What comes out is plain JSON, which is rather the point — it can be read,
diffed, reviewed, and version-controlled like any other configuration:

```json
{
  "cls_path": "my_package.models.MyModel",
  "timestamp": "2026-07-27T18:04:11.921043+00:00",
  "hidden_size": 128,
  "num_layers": 4,
  "cutoff": 6.0
}
```

Going the other way is the same path in reverse:
{py:func}`~nvalchemi.training.create_model_spec_from_json` turns the JSON back
into a spec object, and `build()` imports `cls_path` and calls it with the
stored arguments.

```python
from nvalchemi.training import create_model_spec_from_json

model = create_model_spec_from_json(spec_dict).build()
```

Some arguments genuinely cannot be written down (an optimizer needs live
`model.parameters()`; a scheduler needs an actual optimizer instance), so
`build()` accepts positional and keyword arguments that are injected at
rebuild time rather than stored:

```python
optimizer = optimizer_spec.build(model.parameters())
scheduler = scheduler_spec.build(optimizer)
```

If the target's signature has drifted since the spec was written — a renamed
argument, say — `build()` raises `TypeError` naming both the `cls_path` and the
spec's timestamp, so the mismatch is reported rather than quietly papered over.

```{note}
Not to be confused with {py:class}`~nvalchemi.distributed.MLIPSpec`, which
shares the word "spec" and little else. A `BaseSpec` records _how to rebuild an
object_; an `MLIPSpec` records _how a model parallelizes_ — storage policy,
custom-op adapters, and output classification — and carries its own versioned
`to_dict()` / `from_dict()` format. The two are independent, and domain
decomposition does not use the recipe layer described here at all. See
{doc}`distributed_byo` for authoring the latter.
```

(recipe-custom-types)=

## What a recipe can carry

Since the whole thing ends up as JSON, every field value has to be
representable there. Strings, numbers, booleans, `None`, and lists or dicts of
those work directly. Beyond that, a small registry handles the types that come
up constantly in this domain, and four are registered out of the box:

| Type | JSON form |
|---|---|
| {py:class}`torch.dtype` | its string name, rehydrated behind an `isinstance` guard (i.e. a hostile string cannot smuggle arbitrary `torch.*` attributes through `getattr`) |
| {py:class}`torch.device` | its string form |
| {py:class}`torch.Tensor` | `{dtype, shape, data}` — a data structure, not a bytecode payload |
| `type` (a class object) | its dotted import path |

That registry is not closed, though — if your constructor takes a type of your
own, you can teach the recipe layer how to write it down with
{py:func}`~nvalchemi.training.register_type_serializer`. Say a model takes a
small configuration object rather than a pile of loose floats:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RadialBasis:
    kind: str
    num_basis: int
    cutoff: float


class MyModel(BaseModelMixin):
    def __init__(self, basis: RadialBasis, hidden_size: int = 64):
        super().__init__()
        self.basis = basis            # same name as the argument
        self.hidden_size = hidden_size
```

Registering it is a matter of saying how it collapses to something JSON-safe
and how it comes back. Do this once, at import time of the module that defines
the type, so the pair is in place before any recipe is written or rebuilt:

```python
from nvalchemi.training import register_type_serializer

register_type_serializer(
    RadialBasis,
    serialize=lambda rb: {
        "kind": rb.kind,
        "num_basis": rb.num_basis,
        "cutoff": rb.cutoff,
    },
    deserialize=lambda d: RadialBasis(**d),
)
```

`RadialBasis` is now a first-class citizen: it can be passed as a constructor
argument, and it survives the round trip as itself rather than as a dict.

```python
spec = create_model_spec(MyModel, basis=RadialBasis("bessel", 8, 6.0), hidden_size=128)
rebuilt = create_model_spec_from_json(spec.model_dump()).build()

assert isinstance(rebuilt.basis, RadialBasis)   # RadialBasis(kind='bessel', ...)
```

The nested object is stored inline, and stays just as readable as everything
else:

```json
{
  "cls_path": "my_package.models.MyModel",
  "timestamp": "2026-07-27T23:56:28.740318+00:00",
  "basis": {
    "kind": "bessel",
    "num_basis": 8,
    "cutoff": 6.0
  },
  "hidden_size": 128
}
```

Both directions should be total and free of side effects, and `deserialize`
should validate what it is handed rather than trusting it — the value may well
have come from a file somebody else wrote. It is also worth making
`deserialize` tolerant of already-typed input (i.e. returning the value
unchanged when it is _already_ a `RadialBasis`), which is what the built-in
{py:class}`torch.device` handler does; it costs a line and saves you worrying
about which path called it.

```{tip}
The `RadialBasis` example is pedagogical, rather than the best pattern.
In this example, all of the data being tracked are built-in types, and
so a {py:class}`pydantic.BaseModel` is probably a better fit.
```

Specs also nest: a field may hold another `BaseSpec`, and `build()` constructs
the inner one first before handing it to the outer constructor. Non-empty lists
and tuples of specs are built item-wise, leaving any non-spec items alone and
preserving the container type.

```{warning}
Two limits are worth knowing before you design around them:

- **Nested collections are not traversed.** Something like
  `list[list[BaseSpec]]` will not be rebuilt element-wise. Flatten it, or wrap
  the inner collection in an object that has a spec of its own.
- **Positional-only parameters are rejected.** `create_model_spec` raises
  `TypeError` for targets that declare them, since a recipe addresses every
  argument by name.
```

## How your class gets a recipe

This applies whenever a {py:class}`~nvalchemi.training.TrainingStrategy`
produces its recipe — `to_spec_dict()`, and therefore also the native
strategy checkpoint written by `save_checkpoint()` and
{py:class}`~nvalchemi.training.hooks.CheckpointHook`, since both build on it.
(A bare `torch.save(model.state_dict())` or a PyTorch Distributed Checkpoint
has no recipe layer at all — they move tensors, and reconstructing the
architecture is left to you.)

In that flow each model is asked for a spec in a fixed order of precedence, and
which branch you land on determines how much care you need to take:

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
    omit [label="UserWarning:\n'Omitting model spec'\n-> save_checkpoint raises" fillcolor="#4a1515"]

    start -> explicit
    explicit -> use_explicit [label="yes"]
    explicit -> introspect [label="no"]
    use_explicit -> ok
    introspect -> validate
    validate -> ok [label="rebuild succeeds"]
    validate -> omit [label="raises"]
}
```

The first branch is the explicit one. If your class defines a callable
`checkpoint_spec()` that returns a `BaseSpec`, that spec is taken at face value
(returning `None` declines and falls through to the second branch; returning
anything else raises `TypeError`). This is the escape hatch for classes whose
constructor arguments are transformed rather than kept — a wrapper that takes a
checkpoint path and turns it into a live module, for instance, has nothing
useful to introspect.

Otherwise the framework works it out by introspection: it reads your `__init__`
signature and, for each parameter (skipping `self`, `*args`, and `**kwargs`),
looks for an attribute of the same name on the instance. Submodules found this
way are recursed into, producing nested specs. The catch is that a parameter
with no matching attribute is _silently skipped_, which is exactly how a class
ends up half-described:

```python
class MyModel(BaseModelMixin):
    def __init__(self, hidden_size: int, cutoff: float):
        super().__init__()
        self.hidden_size = hidden_size      # discoverable
        self.r_cut = cutoff                 # NOT discoverable — name differs
```

Here `hidden_size` round-trips and `cutoff` does not, so the rebuilt model
quietly falls back to whatever default the constructor declares. Storing it as
`self.cutoff` is the entire fix.

Because that path is a heuristic, it is checked rather than trusted: the spec
is rebuilt on the spot and the result verified. If that fails, the spec is
dropped with a `UserWarning` (`Omitting model spec for '<name>'`), and what
happens next depends on what you asked for. `to_spec_dict()` simply leaves that
model out of the recipe. `save_checkpoint()` refuses outright, raising
`ValueError: Cannot save strategy checkpoint because model spec generation
failed for model(s) [...]`.

That second behaviour is deliberate, and worth knowing before you see it in a
traceback at 3am: an unserializable model does not quietly degrade a
checkpoint, it prevents one from being written at all. Nothing — not even the
weights — lands on disk. Better a loud failure on the first checkpoint than a
directory full of files that cannot be restored.

## Making your code round-trip

Most of this is handled for you, and in practice a class becomes serializable
by following one habit: **store each constructor argument on `self` under the
same name**. Everything else is a variation on that theme, roughly in the order
the problems tend to show up:

1. Keep constructor arguments representable — natives, one of the registered
   types, or nested objects that have specs of their own. Anything that cannot
   be written down (open file handles, live modules, sessions) belongs behind a
   factory that _does_ take serializable arguments, with the spec pointing at
   the factory instead.
2. Avoid positional-only parameters in anything you expect to be rebuilt.
3. Keep referenced callables importable. A dotted path can only address
   module-level functions, so lambdas, closures, locally-defined functions, and
   bound methods are rejected outright (with an error saying so).
4. Register any custom types you want to appear in constructor arguments, as
   with the {ref}`RadialBasis example <recipe-custom-types>`.
5. Implement `checkpoint_spec()` when your constructor genuinely transforms its
   arguments.

## What is never written down

A few things are excluded deliberately, and no amount of configuration will
include them.

Callables supplied to a workflow — `training_fn` and `loss_target_assembler`
being the obvious ones — are recorded only as importable dotted paths, never as
code. There is no way to guarantee that a serialized callable is safe to run,
or that it has not been swapped out in flight, so they are passed in again at
load time. Hooks are runtime objects for the same reason: you reconstruct them
in your script, and only those implementing
{py:class}`~nvalchemi.hooks.CheckpointableHook` have their _state_ restored
into the instances you provide. Anything that reduces to neither an importable
reference nor serializable arguments falls in the same bucket.

The practical consequence is worth stating plainly: your script is part of the
reproducible artifact. The checkpoint holds data and references; the code they
point at is yours to version.

## Checking that it actually round-trips

The cheapest way to be sure is to do it, rather than to reason about it:

```python
from nvalchemi.training import create_model_spec_from_json

spec = create_model_spec(MyModel, hidden_size=128, cutoff=6.0)
rebuilt = create_model_spec_from_json(spec.model_dump()).build()
```

If that returns an equivalent object, the class is serializable; if it raises,
the error names the argument or path at fault. Beyond that, a component is in
good shape when it emits no `Omitting model spec` warnings, keeps its
referenced callables importable at module level, registers any custom argument
types, implements `to_spec()` on custom loss weight schedules, and implements
{py:class}`~nvalchemi.hooks.CheckpointableHook` on any hook that owns
restart-critical state.

## See also

- {doc}`training` — reproducibility in the training lifecycle.
- {doc}`/modules/training/checkpoints` — checkpoint layout, restart semantics,
  and the full save/load API.
- {doc}`/modules/training/index` — `BaseSpec`, `create_model_spec`, and
  `register_type_serializer` reference.
