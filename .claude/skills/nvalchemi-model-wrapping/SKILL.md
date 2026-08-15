---
name: nvalchemi-model-wrapping
description: >-
  How to wrap an arbitrary MLIP (Machine Learning Interatomic Potential) using
  the BaseModelMixin interface to standardize inputs, outputs, and embeddings.
  Use when integrating a model such as MACE or AIMNet2 (e.g. MACEWrapper,
  loading pretrained checkpoints) so dynamics, training, or fine-tuning stages
  can call it, or when exposing energies, forces, or embeddings from a custom
  PyTorch model.
---

# nvalchemi Model Wrapping

## Overview

To use an arbitrary MLIP (Machine Learning Interatomic Potential) within `nvalchemi`,
pair it with the `BaseModelMixin` interface. This standardizes how models receive
`AtomicData`/`Batch` inputs and produce `ModelOutputs`.

```python
from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.data import AtomicData, Batch
from nvalchemi._typing import ModelOutputs
```

---

## Architecture

A wrapper subclasses `nn.Module` **and** `BaseModelMixin`, and holds the
underlying model by **composition** (`self.model = ...`). This is the pattern
used by every built-in wrapper (`DemoModelWrapper`, `MACEWrapper`,
`AIMNet2Wrapper`, `LennardJonesModelWrapper`).

```text
┌──────────────────────┐    ┌──────────────────┐
│  YourModel(nn.Module)│    │  BaseModelMixin   │
│  - forward()         │    │  - model_config   │
│  - your layers       │    │  - adapt_input()  │
└──────────────────────┘    │  - adapt_output() │
        held via            └────────┬─────────┘
      composition                    │
             ┌──────────▼───────────────────────┐
             │  YourModelWrapper                 │
             │  (nn.Module, BaseModelMixin)      │
             │  self.model = YourModel(...)      │
             │  self.model_config = ModelConfig(…)│
             └───────────────────────────────────┘
```

`nn.Module` must come first in the bases so PyTorch initializes correctly.

---

## Step-by-step guide

### 1. Set `model_config` in `__init__` (capabilities & runtime control)

`ModelConfig` unifies two kinds of fields:

- **Capability fields** (frozen `frozenset`/bool at construction) describe what
  the checkpoint can do: `outputs`, `autograd_outputs`, `autograd_inputs`,
  `required_inputs`, `optional_inputs`, `supports_pbc`, `needs_pbc`,
  `neighbor_config`.
- **Runtime fields** (mutable) control what to compute each pass:
  `active_outputs` (defaults to `outputs`) and `gradient_keys`.

`BaseModelMixin` enforces that every wrapper sets `self.model_config` in
`__init__` (a missing one raises `TypeError` at construction).

```python
def __init__(self, model: nn.Module) -> None:
    super().__init__()
    self.model = model
    self.model_config = ModelConfig(
        outputs=frozenset({"energy", "forces"}),    # everything the model CAN produce
        autograd_outputs=frozenset({"forces"}),     # subset computed via autograd
        autograd_inputs=frozenset({"positions"}),   # inputs needing requires_grad
        required_inputs=frozenset(),                # extra required beyond positions/atomic_numbers
        optional_inputs=frozenset(),                # used if present, skipped if absent
        supports_pbc=False,
        needs_pbc=False,
        neighbor_config=None,                       # NeighborConfig(...) if needed
    )
```

Well-known output keys: `energy`, `forces`, `stress`, `hessians`, `dipoles`,
`charges`, `embeddings`. `outputs`/`required_inputs` are free-form strings, so
new properties can be added without changing `ModelConfig`.

### 2. Define `embedding_shapes`

```python
@property
def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
    return {
        "node_embeddings": (self.model.hidden_dim,),
        "graph_embedding": (self.model.hidden_dim,),
    }
```

### 3. Implement `adapt_input`

Converts `AtomicData`/`Batch` to a dict of keyword arguments for the underlying model's `forward()`.

**Always call `super().adapt_input()` first** — it enables `requires_grad` on
`autograd_inputs` (when an autograd output is active) plus any `gradient_keys`,
and collects the keys declared by `input_data()`.

```python
def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
    model_inputs = super().adapt_input(data, **kwargs)

    # Extract tensors in the format your model expects
    model_inputs["atomic_numbers"] = data.atomic_numbers
    model_inputs["positions"] = data.positions.to(self.dtype)

    # Handle batched vs single input
    if isinstance(data, Batch):
        model_inputs["batch_indices"] = data.batch_idx
    else:
        model_inputs["batch_indices"] = None

    # Gate behavior on the active outputs, not a compute_* flag
    model_inputs["compute_forces"] = "forces" in self.model_config.active_outputs
    return model_inputs
```

### 4. Implement `adapt_output`

Converts the model's raw output to `ModelOutputs` (an `OrderedDict[str, Tensor | None]`).

**Always call `super().adapt_output()` first** — it returns an `OrderedDict`
pre-filled with the `output_data()` keys (set to `None`) and auto-maps matching
key names (unsqueezing a 1-D `energy` to `[B, 1]`).

```python
def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
    output = super().adapt_output(model_output, data)

    # Map model outputs to standardized keys
    energy = model_output["energy"]
    if isinstance(data, AtomicData) and energy.ndim == 1:
        energy = energy.unsqueeze(-1)   # must be [B, 1]
    output["energy"] = energy

    if "forces" in self.model_config.active_outputs:
        output["forces"] = model_output["forces"]

    return output
```

**Standard output keys and shapes:**

| Key          | Shape        | Notes                    |
|--------------|-------------|--------------------------|
| `energy`     | `[B, 1]`   | Per-graph energy (eV)    |
| `forces`     | `[V, 3]`   | Per-node forces          |
| `stress`     | `[B, 3, 3]`| Per-graph stress tensor  |
| `hessian`    | `[V, 3, 3]`| Energy Hessian           |
| `dipole`     | `[B, 3]`   | Dipole moment            |
| `charges`    | `[V]`      | Partial charges          |

### 5. Implement `compute_embeddings`

`embedding_shapes` and `compute_embeddings` are abstract on `BaseModelMixin`, so
every wrapper must define them (raise `NotImplementedError` if the model has no
embeddings). `compute_embeddings` writes embeddings to the data structure
in-place and returns it.

```python
def compute_embeddings(self, data: AtomicData | Batch, **kwargs: Any) -> AtomicData | Batch:
    model_inputs = self.adapt_input(data, **kwargs)

    # Run model layers to get intermediate representations
    atom_z = self.model.embedding(model_inputs["atomic_numbers"])
    coord_z = self.model.coord_embedding(model_inputs["positions"])
    embedding = self.model.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))

    # Aggregate to graph level
    if isinstance(data, Batch):
        batch_indices = data.batch_idx
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

### 6. Implement `forward`

The main entry point. Adapts input, calls the underlying model, adapts output.

```python
def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
    model_inputs = self.adapt_input(data, **kwargs)
    model_outputs = self.model(**model_inputs)   # call the composed model
    return self.adapt_output(model_outputs, data)
```

### 7. (Optional) Override `export_model` / `add_output_head`

`BaseModelMixin.export_model` and `add_output_head` default to raising
`NotImplementedError`. Override them if your model needs to be exported without
the mixin (e.g. for ASE calculators) or supports extra output heads.

```python
def export_model(self, path: Path, as_state_dict: bool = False) -> None:
    if as_state_dict:
        torch.save(self.model.state_dict(), path)
    else:
        torch.save(self.model, path)
```

---

## Runtime control: `active_outputs`

`active_outputs` selects what to compute on each forward pass. Change it with
`set_config`, which validates that the field exists and is mutable:

```python
model = MyModelWrapper(MyPotential())

# Enable stress (e.g. for NPT/NPH) — must already be in outputs
model.set_config("active_outputs", {"energy", "forces", "stress"})

# Add extra gradient inputs beyond those implied by autograd_inputs
model.set_config("gradient_keys", {"positions"})
```

`set_config(key, value)` is equivalent to `model.model_config.<key> = value`.
`output_data()` returns `active_outputs & outputs` and warns if you request a
key the model does not support.

---

## Helper methods

| Method | Returns | Description |
|--------|---------|-------------|
| `input_data()` | `set[str]` | Required input keys from `model_config` (`positions`, `atomic_numbers`, neighbor-list keys, `pbc`, `required_inputs`) |
| `output_data()` | `set[str]` | `active_outputs & outputs` (warns on unsupported requests) |
| `set_config(key, value)` | `None` | Set a mutable `ModelConfig` field with validation |
| `direct_derivative_keys()` | `set[str]` | Outputs computed analytically alongside an autograd energy (pipeline autograd); default empty |
| `add_output_head(prefix)` | `None` | Override to add an MLP output head; default raises `NotImplementedError` |
| `export_model(path, as_state_dict=False)` | `None` | Override to export the raw model; default raises `NotImplementedError` |

---

## Complete example

```python
import torch
from torch import nn
from pathlib import Path
from typing import Any

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi._typing import ModelOutputs


class MyPotential(nn.Module):
    """Your existing PyTorch MLIP model."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Linear(3, hidden_dim)
        self.energy_head = nn.Linear(hidden_dim, 1)

    def forward(self, positions, atomic_numbers=None, batch_indices=None):
        h = self.encoder(positions)
        node_energy = self.energy_head(h)
        if batch_indices is not None:
            num_graphs = int(batch_indices.max()) + 1
            energy = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
            energy.scatter_add_(0, batch_indices.unsqueeze(-1), node_energy)
        else:
            energy = node_energy.sum(dim=0, keepdim=True)
        return {"energy": energy}


class MyPotentialWrapper(nn.Module, BaseModelMixin):
    """Wrapped version for use in nvalchemi."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.model = MyPotential(hidden_dim)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            supports_pbc=False,
            needs_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"node_embeddings": (self.model.hidden_dim,)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        model_inputs = self.adapt_input(data, **kwargs)
        data.node_embeddings = self.model.encoder(model_inputs["positions"])
        return data

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        model_inputs = super().adapt_input(data, **kwargs)
        model_inputs["positions"] = data.positions
        if isinstance(data, Batch):
            model_inputs["batch_indices"] = data.batch_idx
        else:
            model_inputs["batch_indices"] = None
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

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        model_inputs = self.adapt_input(data, **kwargs)
        model_outputs = self.model(**model_inputs)
        return self.adapt_output(model_outputs, data)


# Usage
model = MyPotentialWrapper(hidden_dim=128)
model.set_config("active_outputs", {"energy", "forces"})

data = AtomicData(
    positions=torch.randn(5, 3),
    atomic_numbers=torch.tensor([6, 6, 8, 1, 1], dtype=torch.long),
)
batch = Batch.from_data_list([data])
outputs = model(batch)
# outputs["energy"] shape: [1, 1]
# outputs["forces"] shape: [5, 3]
```
