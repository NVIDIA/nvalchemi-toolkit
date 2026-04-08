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
2. The three building blocks: {py:class}`~nvalchemi.models.base.ModelCard`,
   {py:class}`~nvalchemi.models.base.ModelConfig`, and
   {py:class}`~nvalchemi.models.base.BaseModelMixin`.
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
        label="BaseModelMixin\l- model_card\l- adapt_input()\l- adapt_output()\l"
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

## ModelCard: declaring capabilities

{py:class}`~nvalchemi.models.base.ModelCard` is an **immutable** Pydantic
model that describes what a model can compute and what inputs it requires.
Every wrapper must return a `ModelCard` from its
{py:attr}`~nvalchemi.models.base.BaseModelMixin.model_card` property.

`ModelCard` uses **string sets** for outputs, inputs, and autograd
declarations.  This means new properties (e.g. `"magnetic_moment"`,
`"charges"`) can be added without modifying the `ModelCard` schema.

### Fields

| Field | Default | Meaning |
|---|---|---|
| `outputs` | `{"energies"}` | Set of property names the model can produce. Well-known keys: `energies`, `forces`, `stresses`, `hessians`, `dipoles`, `charges`. |
| `autograd_outputs` | `set()` | Subset of `outputs` computed via autograd (e.g. `{"forces"}` for conservative MLIP forces). Empty for analytical-force models. |
| `autograd_inputs` | `{"positions"}` | Input keys that need `requires_grad_(True)` when any autograd output is requested. Override for models needing grad on other inputs (e.g. displacement for stresses). |
| `inputs` | `set()` | Extra inputs beyond `{positions, atomic_numbers}`. Neighbor-list keys are auto-derived from `neighbor_config`. |
| `supports_pbc` | `False` | Model handles periodic boundary conditions |
| `needs_pbc` | `False` | Model requires `pbc` and `cell` in its input |
| `neighbor_config` | `None` | {py:class}`~nvalchemi.models.base.NeighborConfig` describing neighbor list requirements, or `None` if the model does not use a neighbor list |

The card is frozen (immutable after construction) and serializable via
Pydantic, so it can be saved alongside the checkpoint.

```python
from nvalchemi.models.base import ModelCard, NeighborConfig

# An autograd-forces MLIP with PBC support
card = ModelCard(
    outputs={"energies", "forces", "stresses"},
    autograd_outputs={"forces", "stresses"},
    supports_pbc=True,
    needs_pbc=False,
    neighbor_config=NeighborConfig(cutoff=5.0, format="coo"),
)

# An analytical-forces model (e.g. Lennard-Jones)
card = ModelCard(
    outputs={"energies", "forces", "stresses"},
    autograd_outputs=set(),   # forces computed by kernel, not autograd
    supports_pbc=True,
    needs_pbc=False,
    neighbor_config=NeighborConfig(cutoff=8.5, format="matrix", max_neighbors=128),
)

# A model that needs charges as input (e.g. Ewald)
card = ModelCard(
    outputs={"energies", "forces", "stresses"},
    inputs={"node_charges"},
    needs_pbc=True,
    supports_pbc=True,
    neighbor_config=NeighborConfig(cutoff=10.0, format="matrix", max_neighbors=256),
)
```

## ModelConfig: runtime computation control

{py:class}`~nvalchemi.models.base.ModelConfig` controls **what to compute** on
each forward pass. It lives as the `model_config` attribute on every
`BaseModelMixin` instance and can be changed at any time.

| Field | Default | Meaning |
|---|---|---|
| `compute` | `{"energies", "forces"}` | Set of property names to compute this run. |
| `gradient_keys` | `set()` | Additional tensor keys that need `requires_grad_(True)` beyond those implied by `autograd_inputs`. |

The method {py:meth}`~nvalchemi.models.base.BaseModelMixin.output_data`
intersects `compute` with `outputs` and warns if any requested keys are
unsupported.

```python
from nvalchemi.models.base import ModelConfig

# Default: energies + forces
model.model_config = ModelConfig()

# Enable stress computation for NPT
model.model_config = ModelConfig(compute={"energies", "forces", "stresses"})

# Energy-only evaluation
model.model_config = ModelConfig(compute={"energies"})
```

## Wrapping your own model: step by step

This section walks through every method you need to implement, using
{py:class}`~nvalchemi.models.demo.DemoModelWrapper` as the running example.

### Step 1 --- Create the wrapper class

Use multiple inheritance with your model first and `BaseModelMixin` second:

```python
from nvalchemi.models.base import BaseModelMixin, ModelCard

class DemoModelWrapper(DemoModel, BaseModelMixin):
    ...
```

### Step 2 --- Implement `model_card`

Return a {py:class}`~nvalchemi.models.base.ModelCard` describing your model's
capabilities. This is a `@property`:

```python
@property
def model_card(self) -> ModelCard:
    return ModelCard(
        outputs={"energies", "forces"},
        autograd_outputs={"forces"},
        needs_pbc=False,
    )
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
        model_inputs["batch_indices"] = data.batch
    else:
        model_inputs["batch_indices"] = None

    # Pass config flags to control model behavior
    model_inputs["compute_forces"] = "forces" in self.model_config.compute
    return model_inputs
```

### Step 5 --- Implement `adapt_output`

Map the model's raw output dictionary to `ModelOutputs`, an
`OrderedDict[str, Tensor | None]` with standardized keys. **Always call
`super().adapt_output()` first** --- it creates the OrderedDict pre-filled
with expected keys (derived from the intersection of `model_config.compute`
and `model_card.outputs`) and auto-maps any keys whose names already match:

```python
def adapt_output(self, model_output, data: AtomicData | Batch) -> ModelOutputs:
    output = super().adapt_output(model_output, data)

    energies = model_output["energies"]
    if isinstance(data, AtomicData) and energies.ndim == 1:
        energies = energies.unsqueeze(-1)  # must be [B, 1]
    output["energies"] = energies

    if "forces" in self.model_config.compute:
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
| `energies` | `[B, 1]` | Per-graph total energy |
| `forces` | `[V, 3]` | Per-atom forces |
| `stresses` | `[B, 3, 3]` | Per-graph stress tensor |
| `hessians` | `[V, 3, 3]` | Per-atom Hessian |
| `dipoles` | `[B, 3]` | Per-graph dipole moment |
| `charges` | `[V, 1]` | Per-atom partial charges |

### Step 6 (optional) --- Implement `compute_embeddings`

Extract intermediate representations and write them to the data structure
**in-place**. This is used by active learning and other downstream consumers:

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
        batch_indices = data.batch
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
    model_outputs = super().forward(**model_inputs)
    return self.adapt_output(model_outputs, data)
```

`super().forward(**model_inputs)` calls the underlying `DemoModel.forward`
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
from nvalchemi.models.base import BaseModelMixin, ModelCard, ModelConfig
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
            energies = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
            energies.scatter_add_(0, batch_indices.unsqueeze(-1), node_energy)
        else:
            energies = node_energy.sum(dim=0, keepdim=True)
        return {"energies": energies}


class MyPotentialWrapper(MyPotential, BaseModelMixin):
    """Wrapped version for use in nvalchemi."""

    @property
    def model_card(self) -> ModelCard:
        return ModelCard(
            outputs={"energies", "forces"},
            autograd_outputs={"forces"},
            needs_pbc=False,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"node_embeddings": (self.hidden_dim,)}

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        model_inputs = super().adapt_input(data, **kwargs)
        model_inputs["positions"] = data.positions
        model_inputs["batch_indices"] = data.batch if isinstance(data, Batch) else None
        return model_inputs

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        output = super().adapt_output(model_output, data)
        output["energies"] = model_output["energies"]
        if "forces" in self.model_config.compute:
            output["forces"] = -torch.autograd.grad(
                model_output["energies"],
                data.positions,
                grad_outputs=torch.ones_like(model_output["energies"]),
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
model.model_config = ModelConfig(compute={"energies", "forces"})

data = AtomicData(
    positions=torch.randn(5, 3),
    atomic_numbers=torch.tensor([6, 6, 8, 1, 1], dtype=torch.long),
)
batch = Batch.from_data_list([data])
outputs = model(batch)
# outputs["energies"] shape: [1, 1]
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
combined = mace + dftd3 + ewald  # chains naturally (3 groups)
```

The result is a
{py:class}`~nvalchemi.models.pipeline.PipelineModelWrapper` where each model
occupies its own `"direct"`-force group.  Use this when:

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
control over force strategy, inter-model data wiring, and autograd scope.
Models are organized into **groups**, where each group is a mini-pipeline
with its own force computation strategy:

* **`forces="direct"`** --- each model computes its own forces.  The group
  sums them.
* **`forces="autograd"`** --- the group sums all model energies, then
  computes forces as `-dE_total/dr` via a single autograd pass.  This is
  required when one model's output feeds into another's energy computation
  and forces must backpropagate through the full chain.

```python
from nvalchemi.models.pipeline import (
    PipelineModelWrapper, PipelineGroup, PipelineStep,
)

# AIMNet2 predicts charges + energy; Ewald uses those charges.
# Forces must backpropagate through both → shared autograd.
pipe = PipelineModelWrapper(groups=[
    PipelineGroup(
        steps=[
            PipelineStep(aimnet2, wire={"charges": "node_charges"}),
            ewald,
        ],
        forces="autograd",
    ),
    PipelineGroup(steps=[dftd3], forces="direct"),
])
```

Key concepts:

* **`PipelineStep(model, wire={...})`** --- wraps a model with an output
  rename mapping.  Only needed when a model's output key doesn't match the
  downstream input key (e.g. model outputs `"charges"`, downstream expects
  `"node_charges"`).  For models that don't need renaming, pass the bare
  model directly.
* **`PipelineGroup(steps=[...], forces="direct"|"autograd")`** --- a group
  of steps with a shared force strategy.
* **Auto-wiring** --- if an upstream model's output key matches a
  downstream model's input key, the pipeline connects them automatically.
* **Cross-group data flow** --- a model in group 2 can read outputs
  produced by group 1 (the forward context accumulates across groups).

### Tier 3: Fully custom composition (utility functions)

For total control, write a custom `nn.Module, BaseModelMixin` subclass and
use the utility functions in {py:mod}`nvalchemi.models._utils`:

```python
from nvalchemi.models._utils import autograd_forces, autograd_stresses, sum_outputs
```

* `autograd_forces(energy, positions)` --- compute forces as `-dE/dr`.
* `autograd_stresses(energy, displacement, cell, num_graphs)` --- compute
  stresses as `-1/V * dE/d(strain)`.
* `sum_outputs(*outputs)` --- element-wise sum on additive keys (energies,
  forces, stresses), last-write-wins for everything else.

### Neighbor list handling in composed models

All composition tiers handle neighbor lists transparently:

1. The pipeline (or `+` result) synthesizes a single
   {py:class}`~nvalchemi.models.base.NeighborConfig` at the **maximum
   cutoff** across all sub-models, using MATRIX format if any sub-model
   needs it.
2. `make_neighbor_hooks()` returns **one**
   {py:class}`~nvalchemi.dynamics.hooks.NeighborListHook` at that max
   cutoff.
3. Each sub-model's `adapt_input()` calls `prepare_neighbors_for_model()`
   which filters the max-cutoff neighbor list down to the model's own
   cutoff and converts formats as needed.

## How models integrate with dynamics

Once wrapped, a model plugs directly into the dynamics framework. The
dynamics integrator calls the wrapper's `forward` method internally via
`BaseDynamics.compute()`, and the resulting forces and energies are written
back to the batch:

```python
from nvalchemi.dynamics import DemoDynamics

model = MyPotentialWrapper(hidden_dim=128)
dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5)
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
  {py:class}`~nvalchemi.models.base.BaseModelMixin`,
  {py:class}`~nvalchemi.models.base.ModelCard`, and
  {py:class}`~nvalchemi.models.base.ModelConfig`.

* **Dynamics guide**: {ref}`dynamics <dynamics_guide>` for how models are used
  inside optimization and MD workflows.
