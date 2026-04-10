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

### Capability fields

| Field | Default | Meaning |
|---|---|---|
| `forces_via_autograd` | *(required)* | `True` if forces come from autograd of the energy |
| `supports_energies` | `True` | Model can predict energies |
| `supports_forces` | `False` | Model can predict forces |
| `supports_stresses` | `False` | Model can predict stress tensors |
| `supports_hessians` | `False` | Model can predict Hessians |
| `supports_dipoles` | `False` | Model can predict dipole moments |
| `supports_pbc` | `False` | Model handles periodic boundary conditions |
| `supports_non_batch` | `False` | Model accepts single `AtomicData` (not just `Batch`) |
| `supports_node_embeddings` | `False` | Model can expose per-atom embeddings |
| `supports_edge_embeddings` | `False` | Model can expose per-edge embeddings |
| `supports_graph_embeddings` | `False` | Model can expose per-graph embeddings |

### Requirement fields

| Field | Default | Meaning |
|---|---|---|
| `needs_pbc` | *(required)* | Model expects `pbc` and `cell` in its input |
| `needs_neighborlist` | `False` | Model expects `neighbor_list` in its input |
| `needs_node_charges` | `False` | Model expects partial charges per atom |
| `needs_system_charges` | `False` | Model expects total system charge |

`ModelCard` uses `ConfigDict(extra="allow")`, so you can attach additional
metadata (e.g. `model_name`) without modifying the schema.

```python
from nvalchemi.models.base import ModelCard

card = ModelCard(
    forces_via_autograd=True,
    supports_energies=True,
    supports_forces=True,
    needs_pbc=False,
    needs_neighborlist=False,
    model_name="MyPotential",  # extra metadata
)
```

## ModelConfig: runtime computation control

{py:class}`~nvalchemi.models.base.ModelConfig` controls **what to compute** on
each forward pass. It lives as the `model_config` attribute on every
`BaseModelMixin` instance and can be changed at any time.

| Field | Default | Meaning |
|---|---|---|
| `compute_energies` | `True` | Compute energies |
| `compute_forces` | `True` | Compute forces |
| `compute_stresses` | `False` | Compute stresses |
| `compute_hessians` | `False` | Compute Hessians |
| `compute_dipoles` | `False` | Compute dipoles |
| `compute_charges` | `False` | Compute partial charges |
| `compute_embeddings` | `False` | Compute intermediate embeddings |
| `gradient_keys` | `set()` | Tensor keys that need `requires_grad_(True)` |

`gradient_keys` is populated automatically --- when `compute_forces` is
`True`, `"positions"` is added so that autograd-based force computation works.

```python
from nvalchemi.models.base import ModelConfig

model.model_config = ModelConfig(
    compute_forces=True,
    compute_stresses=True,  # enable stress computation
)
```

The helper {py:meth}`~nvalchemi.models.base.BaseModelMixin._verify_request`
checks whether a requested computation is both enabled in `ModelConfig` and
supported by `ModelCard`. If it is requested but not supported, a
`UserWarning` is issued.

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
        forces_via_autograd=True,
        supports_energies=True,
        supports_forces=True,
        supports_non_batch=True,
        needs_pbc=False,
        needs_neighborlist=False,
        model_name=self.__class__.__name__,
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
base implementation enables gradients on the required tensors and validates
that all required input keys (from `model_card`) are present:

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
    model_inputs["compute_forces"] = self.model_config.compute_forces
    return model_inputs
```

### Step 5 --- Implement `adapt_output`

Map the model's raw output dictionary to `ModelOutputs`, an
`OrderedDict[str, Tensor | None]` with standardized keys. **Always call
`super().adapt_output()` first** --- it creates the OrderedDict pre-filled
with expected keys (derived from `model_config` + `model_card`) and
auto-maps any keys whose names already match:

```python
def adapt_output(self, model_output, data: AtomicData | Batch) -> ModelOutputs:
    output = super().adapt_output(model_output, data)

    energy = model_output["energy"]
    if isinstance(data, AtomicData) and energy.ndim == 1:
        energy = energy.unsqueeze(-1)  # must be [B, 1]
    output["energy"] = energy

    if self.model_config.compute_forces:
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
            energy = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
            energy.scatter_add_(0, batch_indices.unsqueeze(-1), node_energy)
        else:
            energy = node_energy.sum(dim=0, keepdim=True)
        return {"energy": energy}


class MyPotentialWrapper(MyPotential, BaseModelMixin):
    """Wrapped version for use in nvalchemi."""

    @property
    def model_card(self) -> ModelCard:
        return ModelCard(
            forces_via_autograd=True,
            supports_energies=True,
            supports_forces=True,
            supports_non_batch=True,
            needs_neighborlist=False,
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
        if self.model_config.compute_forces:
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
model.model_config = ModelConfig(compute_forces=True)

data = AtomicData(
    positions=torch.randn(5, 3),
    atomic_numbers=torch.tensor([6, 6, 8, 1, 1], dtype=torch.long),
)
batch = Batch.from_data_list([data])
outputs = model(batch)
# outputs["energy"] shape: [1, 1]
# outputs["forces"] shape: [5, 3]
```

## How models integrate with dynamics

Once wrapped, a model plugs directly into the dynamics framework. The
dynamics integrator calls the wrapper's `forward` method internally via
`BaseDynamics.compute()`, and the resulting forces and energy are written
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

- **Examples**: The gallery includes dynamics examples that demonstrate model
  usage in context.
- **API**: {py:mod}`nvalchemi.models` for the full reference of
  {py:class}`~nvalchemi.models.base.BaseModelMixin`,
  {py:class}`~nvalchemi.models.base.ModelCard`, and
  {py:class}`~nvalchemi.models.base.ModelConfig`.
- **Dynamics guide**: {ref}`dynamics <dynamics_guide>` for how models are used
  inside optimization and MD workflows.
