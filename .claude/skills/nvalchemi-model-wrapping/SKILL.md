---
name: nvalchemi-model-wrapping
description: How to use and extend the current `nvalchemi.models` composite calculator API with built-in steps, custom `Potential` classes, neighbor builders, and the artifact registry.
---

# nvalchemi Model Wrapping

## Overview

The current `nvalchemi.models` API is a composite calculator system. Instead of
wrapping one model behind one large interface, you assemble a pipeline from
explicit steps:

- neighbor builders
- ML potentials such as `MACEPotential` and `AIMNet2Potential`
- direct physical terms such as `DFTD3Potential`, `DSFCoulombPotential`,
  `EwaldCoulombPotential`, `PMEPotential`, and `LennardJonesPotential`
- derivative steps such as `EnergyDerivativesStep`

The central model flow is naturally expressed as a sequence of steps, for
example:

`NL -> MLIP -> NL -> Coulomb -> AutoGrad`

Use this architecture when you need to:

- compose short-range ML and direct physical terms explicitly
- control where autograd-derived forces and stresses are taken
- add a custom calculation step without rebuilding a monolithic wrapper
- register stable artifact names for checkpoints or processed parameter files

## Architecture

The main public symbols are exported from `nvalchemi.models`:

```python
from nvalchemi.models import (
    AIMNet2Potential,
    CalculatorResults,
    CompositeCalculator,
    DFTD3Potential,
    DSFCoulombPotential,
    EnergyDerivativesStep,
    EwaldCoulombPotential,
    MACEPotential,
    NeighborListBuilder,
    PMEPotential,
    Potential,
    PotentialCard,
)
```

Important concepts:

- `CompositeCalculator`
  Runs an ordered sequence of steps and merges named outputs.
- `Potential`
  Base class for calculation steps that consume declared inputs and return
  `CalculatorResults`.
- `NeighborListBuilder`
  Produces reusable external neighbor data for later steps.
- `EnergyDerivativesStep`
  Differentiates accumulated `energies` into `forces` and `stresses`.
- `card`
  Class-level declaration of a step's inputs, outputs, defaults, and neighbor
  contract.
- `profile`
  Resolved runtime contract for a configured step instance.
- `model_card`
  Metadata about model family, provided terms, and checkpoint provenance.

Practical rules:

- MLIPs usually contribute `energies`.
- `EnergyDerivativesStep()` turns accumulated `energies` into `forces` and
  `stresses`.
- Direct physical terms may emit direct `forces` and `stresses`.
- External-neighbor potentials advertise a `neighbor_requirement`.
- `neighbor_list_builder_config(...)` is the preferred user-facing bridge from
  a potential's neighbor contract to a concrete `NeighborListBuilderConfig`.
- For charge-coupled PME or Ewald, use `derivative_mode="autograd"`.
- DSF is the built-in hybrid Coulomb option.
- Stages exchange named values by key. Shape and dtype compatibility is not
  enforced automatically between stages.

## Step-by-step guide

### 1. Build a batch

```python
import torch

from nvalchemi.data import AtomicData, Batch

data = AtomicData(
    positions=torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
    ),
    atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
)

batch = Batch.from_data_list([data])
```

Cutoffs use the same length units as `batch.positions`.

### 2. Use built-in steps

Short-range MLIP plus direct dispersion:

```python
from nvalchemi.models import (
    CompositeCalculator,
    DFTD3Potential,
    EnergyDerivativesStep,
    MACEPotential,
    NeighborListBuilder,
)

mace = MACEPotential(
    model="mace-mp-0b3-medium",
    neighbor_list_name="short_range",
)
short_range_nl = NeighborListBuilder(mace.neighbor_list_builder_config())

d3 = DFTD3Potential(functional="pbe", neighbor_list_name="dispersion")
dispersion_nl = NeighborListBuilder(d3.neighbor_list_builder_config())

calculator = CompositeCalculator(
    short_range_nl,
    mace,
    EnergyDerivativesStep(),
    dispersion_nl,
    d3,
    outputs={"energies", "forces"},
)
```

Charge-aware pipeline:

```python
from nvalchemi.models import (
    AIMNet2Potential,
    CompositeCalculator,
    DFTD3Potential,
    EnergyDerivativesStep,
    NeighborListBuilder,
    PMEPotential,
)

aimnet2 = AIMNet2Potential(model="aimnet2")

pme = PMEPotential(
    cutoff=12.0,
    neighbor_list_name="long_range",
    derivative_mode="autograd",
    reuse_if_available=True,
)

d3 = DFTD3Potential(functional="pbe", neighbor_list_name="long_range")
long_range_nl = NeighborListBuilder(
    d3.neighbor_list_builder_config(reuse_if_available=True)
)

calculator = CompositeCalculator(
    aimnet2,
    long_range_nl,
    pme,
    EnergyDerivativesStep(),
    d3,
    outputs={"energies", "forces", "stresses", "node_charges"},
)
```

### 3. Understand neighbor contracts

External-neighbor potentials expose a `neighbor_requirement` through their
resolved `profile`.

The current contract is:

- neighbor-list `name` must match
- `format` must match
- provided cutoff must be greater than or equal to the advertised cutoff

The preferred user path is:

```python
builder = NeighborListBuilder(potential.neighbor_list_builder_config())
```

You may override the emitted builder config if the override keeps the contract
valid, for example by increasing the cutoff.

Internal-neighbor models such as `AIMNet2Potential` do not require an upstream
neighbor builder, so `neighbor_list_builder_config()` returns `None`.

### 4. Write a custom potential

Most custom steps only need:

1. a module-level `PotentialCard`
2. a subclass of `Potential`
3. a `compute(batch, ctx) -> CalculatorResults` implementation

`Potential.__init__()` resolves the profile from the class `card`, so custom
steps do not need to call `card.to_profile(...)` manually in the common path.

```python
import torch

from nvalchemi.data import Batch
from nvalchemi.models import CalculatorResults, Potential, PotentialCard
from nvalchemi.models.base import ForwardContext

QuadraticBiasCard = PotentialCard(
    required_inputs=frozenset({"positions"}),
    result_keys=frozenset({"energies"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies"}),
)


class QuadraticBiasPotential(Potential):
    card = QuadraticBiasCard

    def __init__(self, strength: float = 0.1, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.strength = strength

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        positions = self.require_input(batch, "positions", ctx)
        per_atom = self.strength * positions.pow(2).sum(dim=-1)
        energies = torch.zeros(
            batch.num_graphs,
            1,
            device=positions.device,
            dtype=positions.dtype,
        )
        energies.index_add_(0, batch.batch, per_atom.unsqueeze(-1))
        return self.build_results(ctx, energies=energies)
```

Use profile overrides only when the instance changes its public contract, for
example when changing:

- neighbor-list name
- required neighbor format
- declared required inputs

### 5. Use the registry for named artifacts

The registry is the runtime artifact-resolution layer for named checkpoints and
processed assets.

Built-in examples:

```python
from nvalchemi.models import AIMNet2Potential, MACEPotential

aimnet2 = AIMNet2Potential(model="aimnet2")
mace = MACEPotential(model="mace-mp-0b3-medium")
```

Useful helpers:

- `register_known_artifact(...)`
- `resolve_known_artifact(...)`
- `list_known_artifacts(...)`

Custom artifact example:

```python
from nvalchemi.models import (
    AIMNet2Potential,
    KnownArtifactEntry,
    register_known_artifact,
)

register_known_artifact(
    KnownArtifactEntry(
        name="aimnet2_2025",
        family="aimnet2",
        url="https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_2025_b973c_d3_0.pt",
        cache_subdir="aimnet2",
        filename="aimnet2_2025_b973c_d3_0.pt",
        metadata={
            "model_name": "aimnet2_2025",
            "reference_xc_functional": "b97-3c",
        },
    )
)

model = AIMNet2Potential(model="aimnet2_2025")
```

## Helper methods

Useful current helper methods and behaviors:

| Name | What it does |
|---|---|
| `potential.neighbor_list_builder_config(...)` | Build a `NeighborListBuilderConfig` from an external-neighbor contract |
| `step.active_outputs(...)` | Resolve or validate requested output keys for a step |
| `step.required_inputs(...)` | Return required inputs for the requested outputs |
| `step.optional_inputs(...)` | Return optional inputs for the requested outputs |
| `step.build_results(...)` | Return a `CalculatorResults` containing only active requested keys |
| `list_known_artifacts(...)` | List canonical registry names |
| `resolve_known_artifact(...)` | Resolve a named artifact to a local cached path |

## Complete example

```python
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    CompositeCalculator,
    DFTD3Potential,
    EnergyDerivativesStep,
    MACEPotential,
    NeighborListBuilder,
)

batch = Batch.from_data_list(
    [
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
                dtype=torch.float32,
            ),
            atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        )
    ]
)

mace = MACEPotential(
    model="mace-mp-0b3-medium",
    neighbor_list_name="short_range",
)
short_range_nl = NeighborListBuilder(mace.neighbor_list_builder_config())

d3 = DFTD3Potential(functional="pbe", neighbor_list_name="dispersion")
dispersion_nl = NeighborListBuilder(d3.neighbor_list_builder_config())

calculator = CompositeCalculator(
    short_range_nl,
    mace,
    EnergyDerivativesStep(),
    dispersion_nl,
    d3,
    outputs={"energies", "forces"},
)

results = calculator(batch)
```

## Current limitations

- Stage interfaces are key-based. The pipeline does not currently enforce shape
  or dtype compatibility between stages.
- The public composite API standardizes energies, forces, stresses, charges,
  and neighbor/coulomb plumbing used by the built-in steps. It does not define
  one general public contract for richer ML outputs such as embeddings.

## What not to use for new work

Do not write new code or new guidance around the previous wrapper-oriented
model API, including:

- `BaseModelMixin`
- `ModelConfig`
- `adapt_input(...)`
- `adapt_output(...)`
- `ComposableModelWrapper`

The repository still contains an `EmbeddingModel` typing protocol in
`nvalchemi._typing`, but that is separate from the public composite calculator
contract described in this skill.
