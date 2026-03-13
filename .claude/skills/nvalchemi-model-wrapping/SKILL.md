---
name: nvalchemi-model-wrapping
description: How to wrap an arbitrary MLIP (Machine Learning Interatomic Potential) using the BaseModelMixin interface to standardize inputs, outputs, and embeddings.
---

# nvalchemi Model Wrapping

## Overview

To use an arbitrary MLIP (Machine Learning Interatomic Potential) within `nvalchemi`,
wrap it using the `BaseModelMixin` interface. This standardizes how models receive
`AtomicData`/`Batch` inputs and produce `ModelOutputs`.

```python
from nvalchemi.models.base import BaseModelMixin, ModelCard, ModelConfig
from nvalchemi.data import AtomicData, Batch
```

---

## Architecture

A wrapped model uses **multiple inheritance**: your PyTorch model class + `BaseModelMixin`.

```
┌──────────────────────┐    ┌──────────────────┐
│  YourModel(nn.Module)│    │  BaseModelMixin   │
│  - forward()         │    │  - model_card     │
│  - your layers       │    │  - adapt_input()  │
└──────┬───────────────┘    │  - adapt_output() │
       │                    └────────┬─────────┘
       └──────────┬─────────────────┘
                  │
       ┌──────────▼──────────┐
       │  YourModelWrapper   │
       │  (YourModel,        │
       │   BaseModelMixin)   │
       └─────────────────────┘
```

---

## Step-by-step guide

### 1. Define ModelCard (capabilities & requirements)

`ModelCard` declares what your model can compute and what inputs it needs.

```python
@property
def model_card(self) -> ModelCard:
    return ModelCard(
        # Capabilities
        forces_via_autograd=True,   # forces via autograd (not direct prediction)
        supports_energies=True,
        supports_forces=True,
        supports_stresses=False,
        supports_hessians=False,
        supports_dipoles=False,
        supports_non_batch=True,        # handles single AtomicData (not just Batch)
        supports_pbc=False,             # handles periodic boundary conditions
        supports_node_embeddings=False,
        supports_edge_embeddings=False,
        supports_graph_embeddings=False,
        # Requirements
        needs_neighborlist=False,       # expects edge_index in input
        needs_pbc=False,                # requires cell/pbc in input
        needs_node_charges=False,       # requires node_charges
        needs_system_charges=False,     # requires graph_charges
    )
```

### 2. Define embedding_shapes

```python
@property
def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
    return {
        "node_embeddings": (self.hidden_dim,),
        "graph_embedding": (self.hidden_dim,),
    }
```

### 3. Implement adapt_input

Converts `AtomicData`/`Batch` to a dict of keyword arguments for the underlying model's `forward()`.

**Always call `super().adapt_input()` first** — it enables gradients on required tensors
(e.g. `positions` when computing forces) and validates that required input keys are present.

```python
def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
    model_inputs = super().adapt_input(data, **kwargs)

    # Extract tensors in the format your model expects
    model_inputs["atomic_numbers"] = data.atomic_numbers
    model_inputs["positions"] = data.positions.to(self.dtype)

    # Handle batched vs single input
    if isinstance(data, Batch):
        model_inputs["batch_indices"] = data.batch
    else:
        model_inputs["batch_indices"] = None

    # Pass config flags to control model behavior
    model_inputs["compute_forces"] = self.model_config.compute_forces
    return model_inputs
```

### 4. Implement adapt_output

Converts the model's raw output to `ModelOutputs` (an `OrderedDict[str, Tensor | None]`).

**Always call `super().adapt_output()` first** — it creates an OrderedDict pre-filled with
expected keys (set to `None`) and auto-maps matching key names.

```python
def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
    output = super().adapt_output(model_output, data)

    # Map model outputs to standardized keys
    energies = model_output["energies"]
    if isinstance(data, AtomicData) and energies.ndim == 1:
        energies = energies.unsqueeze(-1)   # must be [B, 1]
    output["energies"] = energies

    if self.model_config.compute_forces:
        output["forces"] = model_output["forces"]

    return output
```

**Standard output keys and shapes:**

| Key          | Shape        | Notes                    |
|--------------|-------------|--------------------------|
| `energies`   | `[B, 1]`   | Per-graph energy (eV)    |
| `forces`     | `[V, 3]`   | Per-node forces          |
| `stresses`   | `[B, 3, 3]`| Per-graph stress tensor  |
| `hessians`   | `[V, 3, 3]`| Energy Hessian           |
| `dipoles`    | `[B, 3]`   | Dipole moment            |
| `charges`    | `[V, 1]`   | Partial charges          |

### 5. Implement compute_embeddings

Extract intermediate representations from the model. Writes embeddings to the data
structure in-place.

```python
def compute_embeddings(self, data: AtomicData | Batch, **kwargs: Any) -> AtomicData | Batch:
    model_inputs = self.adapt_input(data, **kwargs)

    # Run model layers to get intermediate representations
    atom_z = self.embedding(model_inputs["atomic_numbers"])
    coord_z = self.coord_embedding(model_inputs["positions"])
    embedding = self.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))

    # Aggregate to graph level
    if isinstance(data, Batch):
        batch_indices = data.batch
        num_graphs = data.batch_size
    else:
        batch_indices = torch.zeros_like(model_inputs["atomic_numbers"])
        num_graphs = 1

    graph_embedding = torch.zeros(
        (num_graphs, *self.embedding_shapes["graph_embedding"]),
        device=embedding.device, dtype=embedding.dtype,
    )
    graph_embedding.scatter_add_(0, batch_indices.unsqueeze(-1), embedding)

    # Write to data structure in-place
    data.node_embeddings = embedding
    data.graph_embeddings = graph_embedding
    return data
```

### 6. Implement forward

The main entry point. Adapts input, calls the underlying model, adapts output.

```python
def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
    model_inputs = self.adapt_input(data, **kwargs)
    model_outputs = super().forward(**model_inputs)   # calls YourModel.forward()
    return self.adapt_output(model_outputs, data)
```

### 7. (Optional) Implement export_model

Export the model without the `BaseModelMixin` interface (e.g. for use with ASE calculators).

```python
def export_model(self, path: Path, as_state_dict: bool = False) -> None:
    base_cls = self.__class__.__mro__[1]  # get the original model class
    base_model = base_cls()
    for name, module in self.named_children():
        setattr(base_model, name, module)
    if as_state_dict:
        torch.save(base_model.state_dict(), path)
    else:
        torch.save(base_model, path)
```

---

## ModelConfig (runtime computation control)

`ModelConfig` controls what to compute on each forward pass. It is set as the
`model_config` attribute on the wrapper instance.

```python
from nvalchemi.models.base import ModelConfig

model = MyModelWrapper()
model.model_config = ModelConfig(
    compute_energies=True,      # default: True
    compute_forces=True,        # default: True
    compute_stresses=False,     # default: False
    compute_hessians=False,     # default: False
    compute_dipoles=False,      # default: False
    compute_charges=False,      # default: False
    compute_embeddings=False,   # default: False
    gradient_keys=set(),        # auto-populated (e.g. "positions" for forces)
)
```

Use `_verify_request()` to check if a computation is both requested and supported:

```python
if self._verify_request(self.model_config, self.model_card, "stresses"):
    output["stresses"] = compute_stress(...)
```

---

## Helper methods

| Method | Returns | Description |
|--------|---------|-------------|
| `input_data()` | `set[str]` | Required input keys based on `model_card` |
| `output_data()` | `set[str]` | Expected output keys based on `model_config` & `model_card` |
| `_verify_request(config, card, key)` | `bool` | True if computation is requested AND supported |
| `add_output_head(prefix)` | `None` | Add an MLP output head (override for custom models) |

---

## Complete example

```python
import torch
from torch import nn
from pathlib import Path
from typing import Any
from collections import OrderedDict

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelCard, ModelConfig
from nvalchemi._typing import ModelOutputs


class MyPotential(nn.Module):
    """Your existing PyTorch MLIP model."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(3, hidden_dim)
        self.energy_head = nn.Linear(hidden_dim, 1)

    def forward(self, positions, batch_indices=None):
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
        if isinstance(data, Batch):
            model_inputs["batch_indices"] = data.batch
        else:
            model_inputs["batch_indices"] = None
        return model_inputs

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        output = super().adapt_output(model_output, data)
        output["energies"] = model_output["energies"]
        if self.model_config.compute_forces:
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


# Usage
model = MyPotentialWrapper(hidden_dim=128)
model.model_config = ModelConfig(compute_forces=True)

data = AtomicData(
    positions=torch.randn(5, 3),
    atomic_numbers=torch.tensor([6, 6, 8, 1, 1], dtype=torch.long),
)
batch = Batch.from_data_list([data])
outputs = model(batch)
# outputs["energies"] shape: [1, 1]
# outputs["forces"] shape: [5, 3]
```
