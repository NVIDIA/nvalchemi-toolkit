<!-- markdownlint-disable MD014 -->

(models_guide)=

# Models: Wrapping ML Interatomic Potentials

The ALCHEMI Toolkit uses a standardized interface ---
{py:class}`~nvalchemi.models.base.BaseModelMixin` --- that sits between your
PyTorch model and the rest of the framework (dynamics, data loading, active
learning). Any machine-learning interatomic potential (MLIP) can be used with
the toolkit as long as it is wrapped with this interface.

This guide covers:

1. What models are currently supported out of the box.
2. The two building blocks: {py:class}`~nvalchemi.models.base.ModelConfig`
   and {py:class}`~nvalchemi.models.base.BaseModelMixin`.
3. How to wrap your own model, using
   {py:class}`~nvalchemi.models.demo.DemoModelWrapper` as a worked example.
4. How to compose multiple models using the `+` operator or the explicit
   {py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` API.

## Supported models

The {py:mod}`nvalchemi.models` package ships wrappers for the following
potentials:

| Wrapper class | Underlying model | Notes |
|---|---|---|
| {py:class}`~nvalchemi.models.demo.DemoModelWrapper` | {py:class}`~nvalchemi.models.demo.DemoModel` | Non-invariant demo; useful for testing and tutorials |
| `AIMNet2Wrapper` | `AIMNet2` | Requires the `aimnet2` optional dependency |
| {py:class}`~nvalchemi.models.mace.MACEWrapper` | Any MACE variant | Requires the `mace-torch` optional dependency |

`AIMNet2Wrapper` and `MACEWrapper` are lazily imported --- they only
load when accessed, so missing dependencies will not break other imports.

## Architecture overview

A wrapped model uses **multiple inheritance**: your existing `nn.Module`
subclass provides the forward pass, while `BaseModelMixin` adds the
standardized interface.

```{graphviz}
:caption: Multiple-inheritance pattern for model wrapping.

digraph model_inheritance {
    rankdir=BT
    compound=true
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=11 shape=box style="filled,rounded"]
    edge [fontname="Helvetica" fontsize=10]

    YourModel [
        label="YourModel(nn.Module)\l- forward()\l- your layers\l"
        fillcolor="#E8F4FD"
        color="#4A90D9"
    ]
    BaseModelMixin [
        label="BaseModelMixin\l- model_config\l- adapt_input()\l- adapt_output()\l"
        fillcolor="#E8F4FD"
        color="#4A90D9"
    ]
    YourModelWrapper [
        label="YourModelWrapper\l(YourModel, BaseModelMixin)\l"
        fillcolor="#D5E8D4"
        color="#82B366"
    ]

    YourModelWrapper -> YourModel
    YourModelWrapper -> BaseModelMixin
}
```

The wrapper's `forward` method follows a three-step pipeline:

1. **adapt_input** --- convert {py:class}`~nvalchemi.data.AtomicData` /
   {py:class}`~nvalchemi.data.Batch` into the keyword arguments your model
   expects.
2. **super().forward** --- call the underlying model unchanged.
3. **adapt_output** --- map raw model outputs to the framework's
   `ModelOutputs` ordered dictionary.

## ModelConfig: capability declaration and runtime control

{py:class}`~nvalchemi.models.base.ModelConfig` is a single Pydantic model
that serves two purposes:

1. **Capability fields** (frozen at construction) describe what the model
   checkpoint can do.  These use ``frozenset`` to signal immutability.
2. **Runtime fields** (mutable) control what the model computes on each
   forward pass.  These can be changed freely at any time.

Every wrapper sets ``self.model_config`` in its ``__init__``.  The config
uses free-form strings for outputs and inputs, so new properties
(e.g. ``"magnetic_moment"``, ``"charges"``) can be added without modifying
the schema.

### Capability fields (frozen)

| Field | Default | Meaning |
|---|---|---|
| `outputs` | `frozenset({"energy"})` | All property names the model can produce. Well-known keys: `energy`, `forces`, `stress`, `hessian`, `dipole`, `charges`. |
| `autograd_outputs` | `frozenset()` | Subset of `outputs` computed via autograd (e.g. `{"forces"}` for conservative MLIP forces). Empty for analytical-force models. |
| `autograd_inputs` | `frozenset({"positions"})` | Input keys that need `requires_grad_(True)` when any autograd output is requested. |
| `required_inputs` | `frozenset()` | Extra inputs beyond `{positions, atomic_numbers}` that the model **requires** (error if missing). Neighbor-list keys are auto-derived from `neighbor_config`. |
| `optional_inputs` | `frozenset()` | Extra inputs the model can **optionally use** if present, silently skipped if absent. |
| `supports_pbc` | `False` | Model handles periodic boundary conditions. |
| `needs_pbc` | `False` | Model requires `pbc` and `cell` in its input. |
| `neighbor_config` | `None` | {py:class}`~nvalchemi.models.base.NeighborConfig` describing neighbor list requirements, or `None` if the model does not use a neighbor list. |

### Runtime fields (mutable)

| Field | Default | Meaning |
|---|---|---|
| `active_outputs` | `None` (defaults to `outputs`) | Set of property names to compute this run.  Change this to narrow or expand what the model computes. |
| `gradient_keys` | `set()` | Additional tensor keys that need `requires_grad_(True)` beyond those implied by `autograd_inputs`. |

The method {py:meth}`~nvalchemi.models.base.BaseModelMixin.output_data`
intersects ``active_outputs`` with ``outputs`` and warns if any requested
keys are unsupported.

```python
from nvalchemi.models.base import ModelConfig, NeighborConfig

# An autograd-forces MLIP with PBC support
cfg = ModelConfig(
    outputs={"energy", "forces", "stress"},
    autograd_outputs={"forces", "stress"},
    supports_pbc=True,
    needs_pbc=False,
    neighbor_config=NeighborConfig(cutoff=5.0, format="coo"),
)

# An analytical-forces model (e.g. Lennard-Jones)
cfg = ModelConfig(
    outputs={"energy", "forces", "stress"},
    autograd_outputs=set(),   # forces computed by kernel, not autograd
    supports_pbc=True,
    needs_pbc=False,
    neighbor_config=NeighborConfig(cutoff=8.5, format="matrix"),
)

# A model that requires charges as input (e.g. Ewald)
cfg = ModelConfig(
    outputs={"energy", "forces", "stress"},
    required_inputs={"charges"},
    needs_pbc=True,
    supports_pbc=True,
    neighbor_config=NeighborConfig(cutoff=10.0, format="matrix"),
)

# A model with optional inputs (e.g. AIMNet2 — works with or without PBC)
cfg = ModelConfig(
    outputs={"energy", "forces", "charges"},
    autograd_outputs={"forces"},
    required_inputs={"charge"},       # system charge is required
    optional_inputs={"cell", "mult"}, # PBC cell and multiplicity are optional
)
```

### Changing active outputs at runtime

The ``active_outputs`` field is the primary lever for controlling what a
model computes on each forward pass.  It defaults to ``outputs`` (i.e.
compute everything the model supports), but you can narrow or expand it
at any time:

```python
# Start with full computation
model = MyWrapper()
out = model(batch)  # computes energies + forces (the defaults)

# Switch to energy-only evaluation (faster — skips force computation)
model.model_config.active_outputs = {"energy"}
out = model(batch)  # only energies

# Enable stress computation for NPT dynamics
model.model_config.active_outputs = {"energy", "forces", "stress"}
out = model(batch)  # energies + forces + stresses

# Restore defaults (compute everything the model supports)
model.model_config.active_outputs = set(model.model_config.outputs)
```

This is particularly useful in multi-stage workflows: use energy-only
evaluation during screening, then switch to forces + stresses for
production dynamics.

## Wrapping your own model: step by step

This section walks through every method you need to implement, using
{py:class}`~nvalchemi.models.demo.DemoModelWrapper` as the running example.

### Required interface checklist

Your wrapper class must provide the following.  Methods marked **abstract**
will raise ``TypeError`` at instantiation if missing:

| Method / Property | Abstract? | Classical potential stub |
|---|---|---|
| `model_config` attribute | — (enforced by post-init check) | Set `self.model_config = ModelConfig(...)` in `__init__` |
| `embedding_shapes` (property) | **Yes** | `return {}` |
| `compute_embeddings()` | **Yes** | `raise NotImplementedError` |
| `adapt_input()` | No (has default) | Override to collect model-specific inputs |
| `adapt_output()` | No (has default) | Override to map raw outputs |
| `forward()` | No (inherit from nn.Module) | Implement the three-step pipeline |
| `export_model()` | No (has default) | Override if needed |

For classical potentials with no learned embeddings, stub both embedding
methods:

```python
@property
def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
    return {}

def compute_embeddings(self, data, **kwargs):
    raise NotImplementedError("No embeddings for this model.")
```

### Step 1 --- Create the wrapper class

Subclass ``nn.Module`` and mix in ``BaseModelMixin``, then hold the
underlying model as ``self.model``:

```python
from torch import nn
from nvalchemi.models.base import BaseModelMixin, ModelConfig

class DemoModelWrapper(nn.Module, BaseModelMixin):
    def __init__(self, model: DemoModel) -> None:
        super().__init__()
        self.model: DemoModel = model
        ...
```

### Step 2 --- Set `model_config` in `__init__`

Create a {py:class}`~nvalchemi.models.base.ModelConfig` describing your model's
capabilities and set it as ``self.model_config`` in ``__init__``:

```python
def __init__(self, model: DemoModel) -> None:
    super().__init__()
    self.model = model
    self.model_config = ModelConfig(
        outputs={"energy", "forces"},
        autograd_outputs={"forces"},
        needs_pbc=False,
    )
```

```{important}
Always set ``model_config`` as an **instance attribute** in ``__init__``.
There is intentionally no class-level default — a shared class attribute
would cause mutations in one wrapper to silently affect all others.
```

### Step 3 --- Implement `embedding_shapes`

Return a dictionary mapping embedding names to their trailing shapes.
This is used by downstream consumers (e.g. active learning) to know what
representations the model can provide:

```python
@property
def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
    return {
        "node_embeddings": (self.hidden_dim,),
        "graph_embedding": (self.hidden_dim,),
    }
```

### Step 4 --- Implement `adapt_input`

Convert framework data to the keyword arguments your underlying model's
`forward()` expects. **Always call `super().adapt_input()` first** --- the
base implementation enables gradients on the required tensors (using
`autograd_inputs` and `autograd_outputs` from the model card) and validates
that all required input keys are present:

```python
def adapt_input(self, data: AtomicData | Batch, **kwargs) -> dict[str, Any]:
    model_inputs = super().adapt_input(data, **kwargs)

    # Extract tensors in the format your model expects
    model_inputs["atomic_numbers"] = data.atomic_numbers
    model_inputs["positions"] = data.positions.to(self.dtype)

    # Handle batched vs. single input
    if isinstance(data, Batch):
        model_inputs["batch_indices"] = data.batch_idx
    else:
        model_inputs["batch_indices"] = None

    # Pass config flags to control model behavior
    model_inputs["compute_forces"] = "forces" in self.model_config.active_outputs
    return model_inputs
```

### Step 5 --- Implement `adapt_output`

Map the model's raw output dictionary to `ModelOutputs`, an
`OrderedDict[str, Tensor | None]` with standardized keys. **Always call
`super().adapt_output()` first** --- it creates the OrderedDict pre-filled
with expected keys (derived from the intersection of
``model_config.active_outputs`` and ``model_config.outputs``) and auto-maps
any keys whose names already match:

```python
def adapt_output(self, model_output, data: AtomicData | Batch) -> ModelOutputs:
    output = super().adapt_output(model_output, data)

    energy = model_output["energy"]
    if isinstance(data, AtomicData) and energy.ndim == 1:
        energy = energy.unsqueeze(-1)  # must be [B, 1]
    output["energy"] = energy

    if "forces" in self.model_config.active_outputs:
        output["forces"] = model_output["forces"]

    # Validate: no expected key should be None
    for key, value in output.items():
        if value is None:
            raise KeyError(
                f"Key '{key}' not found in model output "
                "but is supported and requested."
            )
    return output
```

The standard output shapes are:

| Key | Shape | Description |
|---|---|---|
| `energy` | `[B, 1]` | Per-graph total energy |
| `forces` | `[V, 3]` | Per-atom forces |
| `stress` | `[B, 3, 3]` | Per-graph stress tensor |
| `hessians` | `[V, 3, 3]` | Per-atom Hessian |
| `dipole` | `[B, 3]` | Per-graph dipole moment |
| `charges` | `[V]` | Per-atom partial charges |

### Step 6 --- Implement `compute_embeddings`

This method is **abstract** — you must implement it even if your model has
no learned embeddings.  For classical potentials, a one-line stub suffices:

```python
def compute_embeddings(self, data, **kwargs):
    raise NotImplementedError("No embeddings for this model.")
```

For learned models, extract intermediate representations and write them to
the data structure **in-place**. This is used by active learning and other
downstream consumers:

```python
def compute_embeddings(self, data: AtomicData | Batch, **kwargs) -> AtomicData | Batch:
    model_inputs = self.adapt_input(data, **kwargs)

    # Run the model's internal layers
    atom_z = self.embedding(model_inputs["atomic_numbers"])
    coord_z = self.coord_embedding(model_inputs["positions"])
    embedding = self.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))
    embedding = embedding + atom_z + coord_z

    # Aggregate to graph level via scatter
    if isinstance(data, Batch):
        batch_indices = data.batch_idx
        num_graphs = data.batch_size
    else:
        batch_indices = torch.zeros_like(model_inputs["atomic_numbers"])
        num_graphs = 1

    graph_shape = self.embedding_shapes["graph_embedding"]
    graph_embedding = torch.zeros(
        (num_graphs, *graph_shape),
        device=embedding.device,
        dtype=embedding.dtype,
    )
    graph_embedding.scatter_add_(0, batch_indices.unsqueeze(-1), embedding)

    # Write in-place
    data.node_embeddings = embedding
    data.graph_embeddings = graph_embedding
    return data
```

### Step 7 --- Implement `forward`

Wire the three-step pipeline together:

```python
def forward(self, data: AtomicData | Batch, **kwargs) -> ModelOutputs:
    model_inputs = self.adapt_input(data, **kwargs)
    model_outputs = self.model(**model_inputs)
    return self.adapt_output(model_outputs, data)
```

`self.model(**model_inputs)` calls the underlying `DemoModel.forward`
with the unpacked keyword arguments --- your original model is never modified.
For additional flair, the ``@beartype.beartype`` decorator can be applied to
the ``forward`` method, which will provide runtime type checking on the
inputs *and* outputs, as well as shape checking.

### Step 8 (optional) --- Implement `export_model`

Export the model **without** the `BaseModelMixin` interface, for use with
external tools (e.g. ASE calculators):

```python
def export_model(self, path: Path, as_state_dict: bool = False) -> None:
    base_cls = self.__class__.__mro__[1]  # the original nn.Module
    base_model = base_cls()
    for name, module in self.named_children():
        setattr(base_model, name, module)
    if as_state_dict:
        torch.save(base_model.state_dict(), path)
    else:
        torch.save(base_model, path)
```

## Putting it all together

A complete minimal wrapper for a custom potential:

```python
import torch
from torch import nn
from typing import Any
from pathlib import Path

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi._typing import ModelOutputs


class MyPotential(nn.Module):
    """Your existing PyTorch MLIP."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(3, hidden_dim)
        self.energy_head = nn.Linear(hidden_dim, 1)

    def forward(self, positions, batch_indices=None, **kwargs):
        h = self.encoder(positions)
        node_energy = self.energy_head(h)
        if batch_indices is not None:
            num_graphs = batch_indices.max() + 1
            energy = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
            energy.scatter_add_(0, batch_indices.unsqueeze(-1), node_energy)
        else:
            energy = node_energy.sum(dim=0, keepdim=True)
        return {"energy": energy}


class MyPotentialWrapper(MyPotential, BaseModelMixin):
    """Wrapped version for use in nvalchemi."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__(hidden_dim=hidden_dim)
        self.model_config = ModelConfig(
            outputs={"energy", "forces"},
            autograd_outputs={"forces"},
            needs_pbc=False,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"node_embeddings": (self.hidden_dim,)}

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        model_inputs = super().adapt_input(data, **kwargs)
        model_inputs["positions"] = data.positions
        model_inputs["batch_indices"] = data.batch_idx if isinstance(data, Batch) else None
        return model_inputs

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        output = super().adapt_output(model_output, data)
        output["energy"] = model_output["energy"]
        if "forces" in self.model_config.active_outputs:
            output["forces"] = -torch.autograd.grad(
                model_output["energy"],
                data.positions,
                grad_outputs=torch.ones_like(model_output["energy"]),
                create_graph=self.training,
            )[0]
        return output

    def compute_embeddings(self, data: AtomicData | Batch, **kwargs) -> AtomicData | Batch:
        model_inputs = self.adapt_input(data, **kwargs)
        data.node_embeddings = self.encoder(model_inputs["positions"])
        return data

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        model_inputs = self.adapt_input(data, **kwargs)
        model_outputs = super().forward(**model_inputs)
        return self.adapt_output(model_outputs, data)
```

Usage:

```python
model = MyPotentialWrapper(hidden_dim=128)

data = AtomicData(
    positions=torch.randn(5, 3),
    atomic_numbers=torch.tensor([6, 6, 8, 1, 1], dtype=torch.long),
)
batch = Batch.from_data_list([data])
outputs = model(batch)
# outputs["energy"] shape: [1, 1]
# outputs["forces"] shape: [5, 3]
```

## Composing multiple models

nvalchemi provides three tiers of model composition, from simplest to most
powerful.  Choose the simplest tier that fits your use case.

### Tier 1: The `+` operator (independent additive sum)

The `+` operator is the simplest way to combine models whose outputs should
be summed element-wise.  Each model computes its own forces independently
(analytically or via its own internal autograd) and the pipeline sums
energies, forces, and stresses across all models:

```python
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.models.ewald import EwaldModelWrapper

lj = LennardJonesModelWrapper(epsilon=0.05, sigma=2.5, cutoff=8.0)
ewald = EwaldModelWrapper(cutoff=8.0)

combined = lj + ewald            # sums energies, forces, stresses
# With more models:
# combined = model_a + model_b + model_c  # chains naturally (3 groups)
```

The result is a
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` where each model
occupies its own group with ``use_autograd=False``.  Use this when:

* Each model computes its outputs independently (no data flows between them).
* Each model handles its own force computation (analytical kernels or
  self-contained autograd).
* You just want to sum energies, forces, and stresses.

The `+` operator does **not** support:

* Wiring one model's output into another's input (e.g. charges -> electrostatics).
* Shared autograd groups (differentiating the summed energy of multiple models).

For those cases, use the explicit pipeline API (Tier 2).

### Tier 2: Explicit PipelineModelWrapper (dependent pipelines & shared autograd)

{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` gives full
control over derivative strategy, inter-model data wiring, and autograd scope.
Models are organized into **groups**, where each group is a mini-pipeline
with its own derivative computation strategy:

* **`use_autograd=False`** (default) --- each model computes its own
  outputs.  The group sums them.
* **`use_autograd=True`** --- the group sums all model energies, then
  computes derivatives (forces, stresses, etc.) from the summed energy.
  This is required when one model's output feeds into another's energy
  computation and forces must backpropagate through the full chain.

```python
from nvalchemi.models.pipeline import (
    PipelineModelWrapper, PipelineGroup, PipelineStep,
)

# AIMNet2 predicts charges + energy; Ewald uses those charges.
# Both use the key "charges" — auto-wired, no explicit mapping needed.
# Forces must backpropagate through both → shared autograd.
pipe = PipelineModelWrapper(groups=[
    PipelineGroup(
        steps=[aimnet2, ewald],
        use_autograd=True,
    ),
    PipelineGroup(steps=[dftd3]),
])
```

Key concepts:

* **`PipelineStep(model, wire={...})`** --- wraps a model with an output
  rename mapping.  Only needed when a model's output key doesn't match the
  downstream input key.  For models that don't need renaming (like AIMNet2 +
  Ewald above where both use `"charges"`), pass the bare model directly.
* **`PipelineGroup(steps=[...], use_autograd=True|False)`** --- a group
  of steps with a shared derivative strategy.
* **Auto-wiring** --- if an upstream model's output key matches a
  downstream model's input key, the pipeline connects them automatically.
* **Cross-group data flow** --- a model in group 2 can read outputs
  produced by group 1 (the forward context accumulates across groups).
* **`active_outputs` drives derivatives** --- the pipeline's default
  ``model_config.active_outputs`` is synthesized as the union of all
  sub-model active output sets.  Add ``"stress"`` to request stress
  computation.

### Tier 3: Fully custom composition (utility functions)

For total control, write a custom `nn.Module, BaseModelMixin` subclass and
use the utility functions in {py:mod}`nvalchemi.models._utils`:

```python
from nvalchemi.models._utils import autograd_forces, autograd_stresses, prepare_strain, sum_outputs
```

* `autograd_forces(energy, positions)` --- compute forces as `-dE/dr`.
* `autograd_stresses(energy, displacement, cell, num_graphs)` --- compute
  stresses as `-1/V * dE/d(strain)`.
* `prepare_strain(positions, cell, batch_idx)` --- set up the affine strain
  trick for autograd stress computation (see below).
* `sum_outputs(*outputs)` --- element-wise sum on additive keys (energies,
  forces, stresses), last-write-wins for everything else.

## Autograd derivatives: forces, stresses, and beyond

This section explains how autograd-based derivatives work in nvalchemi ---
both for single-model wrapper authors and for pipeline composition.

### Single-model wrappers: you own the derivatives

When writing a model wrapper, **you decide how to compute derivatives**.
The framework imposes no constraints.  If your model computes forces
analytically (like LJ or Ewald via Warp kernels), declare
``autograd_outputs=set()`` in your ``ModelConfig`` and compute forces
directly in your ``forward()`` method.  If your model uses autograd
(like MACE), declare ``autograd_outputs={"forces"}`` and call
``torch.autograd.grad`` in your ``forward()``.

The key expectation is that your ``forward()`` returns a ``ModelOutputs``
dict with whatever keys your ``model_config.active_outputs`` requests,
however you choose to compute them.

#### Example: autograd forces in a wrapper

```python
def forward(self, data, **kwargs):
    model_inputs = self.adapt_input(data, **kwargs)
    raw = self.model(**model_inputs)  # returns {"energy": tensor}

    energy = raw["energy"]
    result = {"energy": energy.unsqueeze(-1)}

    if "forces" in self.model_config.active_outputs:
        result["forces"] = -torch.autograd.grad(
            energy, data.positions,
            grad_outputs=torch.ones_like(energy),
            create_graph=False,  # set True for training
        )[0]

    return self.adapt_output(result, data)
```

#### Example: autograd stresses using `prepare_strain`

Computing stresses via autograd requires the "affine strain trick" --- a
non-trivial setup step that scales positions and cell through a
displacement tensor.  The
{py:func}`~nvalchemi.models._utils.prepare_strain` helper handles this:

```python
from nvalchemi.models._utils import prepare_strain

def forward(self, data, **kwargs):
    compute_stresses = "stress" in self.model_config.active_outputs

    if compute_stresses:
        scaled_pos, scaled_cell, displacement = prepare_strain(
            data.positions, data.cell, data.batch_idx
        )
        # Run model on scaled tensors
        energy = self.model(scaled_pos, scaled_cell, ...)
    else:
        energy = self.model(data.positions, data.cell, ...)

    result = {"energy": energy.unsqueeze(-1)}

    if "forces" in self.model_config.active_outputs:
        positions_for_grad = scaled_pos if compute_stresses else data.positions
        result["forces"] = -torch.autograd.grad(
            energy, positions_for_grad,
            grad_outputs=torch.ones_like(energy),
            retain_graph=compute_stresses,
        )[0]

    if compute_stresses:
        grad = torch.autograd.grad(
            energy, displacement,
            grad_outputs=torch.ones_like(energy),
        )[0]
        volume = torch.det(data.cell).abs().view(-1, 1, 1)
        result["stress"] = -grad.view(data.num_graphs, 3, 3) / volume

    return self.adapt_output(result, data)
```

You don't *have* to use ``prepare_strain`` --- it's a convenience.  MACE
uses its own internal displacement trick via ``compute_displacement=True``.
The only requirement is that your ``forward()`` returns the requested
outputs.

#### Example: Hessians and Jacobians

These are standard ``torch.autograd`` operations --- nvalchemi does not
wrap them:

```python
# Hessian (second derivative of energy w.r.t. positions)
# Models expect a Batch, not raw positions — define a closure.
def energy_fn(pos):
    data.positions = pos
    return model(data)["energy"].sum()

hessian = torch.autograd.functional.hessian(energy_fn, data.positions)

# Born effective charges (Jacobian of dipoles w.r.t. positions)
dipoles = model(data)["dipole"]  # [B, 3]
Z_star = torch.autograd.functional.jacobian(
    lambda pos: model_dipoles(pos), data.positions
)
```

### Pipeline autograd groups: default and custom derivatives

When models are composed in a
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` with
``use_autograd=True``, the pipeline sums sub-model energies and computes
derivatives from the total.  What gets computed is driven by
``model_config.active_outputs``:

```python
pipe = PipelineModelWrapper(groups=[
    PipelineGroup(steps=[aimnet2, ewald], use_autograd=True),
])

# Default: pipeline inherits sub-model active output sets
# (typically {"energy", "forces"}).  Forces computed via autograd.
out = pipe(batch)

# Request stresses: pipeline uses affine strain trick automatically.
pipe.model_config.active_outputs = {"energy", "forces", "stress"}
out = pipe(batch)  # now includes stresses
```

The pipeline's default ``model_config.active_outputs`` is the **union of
all sub-model active output sets** at construction time.  If sub-models
default to ``{"energy", "forces"}``, the pipeline does too.  You can
expand it (add ``"stress"``) or narrow it (remove ``"forces"``).

**Default behavior:** The pipeline's built-in derivative function computes
forces as ``-dE/dr`` and stresses via the affine strain trick.  This
covers the vast majority of inference use cases.

**Custom `derivative_fn`:** For anything beyond forces and stresses,
provide a custom function that receives the summed energy, the batch, and
the set of requested keys.  You write whatever ``torch.autograd.grad``
calls you want --- the same power as a single-model wrapper's
``forward()``:

```python
def my_derivatives(energy, data, requested):
    """Custom derivative function for a pipeline autograd group.

    Parameters
    ----------
    energy : torch.Tensor
        Summed energy from all models in the group.  On the autograd
        graph --- ready for torch.autograd.grad.
    data : Batch
        The batch.  data.positions has requires_grad=True.
    requested : set[str]
        Output keys still needed (e.g. {"forces", "hessian"}).

    Returns
    -------
    dict[str, torch.Tensor]
        Computed derivatives.
    """
    result = {}
    if "forces" in requested:
        result["forces"] = -torch.autograd.grad(
            energy, data.positions,
            grad_outputs=torch.ones_like(energy),
            retain_graph="hessian" in requested,
        )[0]
    if "hessian" in requested:
        # Your custom Hessian implementation
        result["hessian"] = compute_chunked_hessian(energy, data.positions)
    return result

pipe = PipelineModelWrapper(groups=[
    PipelineGroup(
        steps=[aimnet2, ewald],
        use_autograd=True,
        derivative_fn=my_derivatives,
    ),
])
pipe.model_config.active_outputs = {"energy", "forces", "hessian"}
out = pipe(batch)  # forces + hessian via your function
```

When ``derivative_fn`` is provided, the pipeline does **not** apply the
strain trick or compute forces automatically --- your function has full
control.  If you want stresses, use
{py:func}`~nvalchemi.models._utils.prepare_strain` inside your function.

### Neighbor list handling and `make_neighbor_hooks()`

All composition tiers handle neighbor lists transparently:

1. The pipeline (or `+` result) synthesizes a single
   {py:class}`~nvalchemi.models.base.NeighborConfig` at the **maximum
   cutoff** across all sub-models, using MATRIX format if any sub-model
   needs it.
2. `make_neighbor_hooks()` returns **one**
   {py:class}`~nvalchemi.hooks.NeighborListHook` at that max
   cutoff.
3. Each sub-model's `adapt_input()` calls `prepare_neighbors_for_model()`
   which filters the max-cutoff neighbor list down to the model's own
   cutoff and converts formats as needed.

**Choosing a registration pattern:**

`make_neighbor_hooks()` works for both single models and composed models.
For a single model it is equivalent to constructing a
{py:class}`~nvalchemi.hooks.NeighborListHook` manually from the model's
{py:class}`~nvalchemi.models.base.NeighborConfig`:

```python
# These two are equivalent for a single model:

# (a) make_neighbor_hooks — recommended
for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)

# (b) manual construction — use when you need extra control (e.g. skin distance)
from nvalchemi.hooks import NeighborListHook
dynamics.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.5),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
```

For **composed models** (pipeline), `composed.make_neighbor_hooks()`
reads the already-synthesized maximum cutoff from the pipeline's
`model_config.neighbor_config`.  Manual construction from
`composed.model_config.neighbor_config` is equally valid.

Hooks returned by `make_neighbor_hooks()` have `stage=None` by default —
provide the stage when registering.

## How models integrate with dynamics

Once wrapped, a model plugs directly into the dynamics framework. The
dynamics integrator calls the wrapper's `forward` method internally via
`BaseDynamics.compute()`, and the resulting forces and energy are written
back to the batch:

```python
from nvalchemi.dynamics import DemoDynamics

model = MyPotentialWrapper(hidden_dim=128)
dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5)
# DemoDynamics expects forces to exist on the batch.
batch.forces = torch.zeros_like(batch.positions)
dynamics.run(batch)
```

The `__needs_keys__` set on the dynamics class (e.g. `{"forces"}`) is
validated against the model's output after every `compute()` call, so
mismatches between the model's declared capabilities and the integrator's
requirements are caught immediately at runtime.

## See also

* **Examples**: The gallery includes dynamics examples that demonstrate model
  usage in context.

* **API**: {py:mod}`nvalchemi.models` for the full reference of
  {py:class}`~nvalchemi.models.base.BaseModelMixin` and
  {py:class}`~nvalchemi.models.base.ModelConfig`.

* **Dynamics guide**: {ref}`dynamics <dynamics_guide>` for how models are used
  inside optimization and MD workflows.
