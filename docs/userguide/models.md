<!-- markdownlint-disable MD013 MD014 -->

(models_guide)=

# Models: Building Composite Calculators

`nvalchemi.models` provides a composite calculator API for atomistic models.
Instead of wrapping one model behind one large interface, you assemble a
pipeline from small explicit steps:

- neighbor builders
- ML potentials such as MACE or AIMNet2
- direct physical terms such as DFT-D3, DSF, Ewald, or PME
- optional derivative steps such as
  {py:class}`~nvalchemi.models.EnergyDerivativesStep`

This guide focuses on using the current API:

1. Understand the composite mental model.
2. Build a few practical pipelines.
3. Add your own potential when the built-in steps are not enough.
4. Use the registry when you want stable model names such as `"aimnet2"` or your own custom names.


## Main Concepts

| Concept | What it does | Typical user action |
|---|---|---|
| {py:class}`~nvalchemi.models.CompositeCalculator` | Runs an ordered sequence of steps and merges their outputs | Build one calculator from the steps you want |
| Neighbor builder | Produces reusable neighbor data for later steps | Add one when a potential requires external neighbors |
| Potential | Computes energy and possibly direct forces/stresses or auxiliary outputs | Instantiate built-in wrappers such as MACE, DFT-D3, or PME |
| {py:class}`~nvalchemi.models.EnergyDerivativesStep` | Differentiates the current total energy into forces/stresses | Insert it after the energy terms that should participate in autograd |

## How a Composite Works

A composite calculator is an ordered pipeline:

1. Early steps may build helper data such as neighbor lists.
2. Potentials contribute energies and sometimes direct derivative outputs.
3. `EnergyDerivativesStep()` differentiates the current accumulated energy.
4. Later direct-output terms can still add more forces or stresses afterward.

Outputs are named. If an early step produces a value, a later step can reuse it.

```{note}
Practical rule:

- MLIPs usually contribute `energies`
- `EnergyDerivativesStep()` turns those energies into `forces` and `stresses`
- direct physical terms may instead return `forces` and `stresses` directly
- direct outputs are added to the running total
```

## Quick Start

The package root exports the main user-facing API:

- {py:class}`~nvalchemi.models.CompositeCalculator`
- {py:class}`~nvalchemi.models.EnergyDerivativesStep`
- built-in potentials such as `MACEPotential`, `AIMNet2Potential`,
  `DFTD3Potential`, `DSFCoulombPotential`, `EwaldCoulombPotential`,
  and `PMEPotential`
- neighbor builders such as `NeighborListBuilder`
- cards, metadata, and registry helpers

```{note}
Cutoffs use the same length units as `batch.positions`. The examples below
assume Angstrom-based coordinates, so a cutoff like `6.0` means `6.0 A`.
```

```{tip}
Most config-based potentials and neighbor builders accept either a config
object or flat keyword arguments. The flat-kwargs form is recommended for most
users. Pass a config object when you need to serialize or share configuration
separately.
```

### Example 1: `NL -> MACE -> Grad -> NL -> DFT-D3`

This is the simplest good pattern for a short-range MLIP plus a direct
dispersion correction:

- MACE provides the learned short-range energy
- `EnergyDerivativesStep()` produces MACE forces and stresses
- DFT-D3 adds direct dispersion outputs afterward

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

short_range_nl = NeighborListBuilder(
    mace.neighbor_list_builder_config()
)

d3 = DFTD3Potential(functional="pbe", neighbor_list_name="dispersion")

dispersion_nl = NeighborListBuilder(
    d3.neighbor_list_builder_config()
)

calculator = CompositeCalculator(
    short_range_nl,
    mace,
    EnergyDerivativesStep(),
    dispersion_nl,
    d3,
    outputs={"energies", "forces", "stresses"},
)

results = calculator(batch)
print(results["energies"].shape)
print(results["forces"].shape)
```

This ordering works because:

- the MACE neighbor list is created directly from MACE's advertised
  external-neighbor contract
- `EnergyDerivativesStep()` sees MACE energy before the dispersion term runs
- DFT-D3 adds direct outputs after autograd has already handled the MLIP term

If you want a larger neighbor cutoff than the advertised minimum, override it
explicitly:

```python
short_range_nl = NeighborListBuilder(
    mace.neighbor_list_builder_config(cutoff=6.5, reuse_if_available=True)
)
```

Increasing the cutoff is valid. Decreasing it below the advertised minimum is
not.

### Example 2: `AIMNet2 -> NL -> PME -> Grad -> DFT-D3`

This is an advanced charge-aware composition:

- AIMNet2 handles its own short-range neighbors internally
- the explicit neighbor builder is for PME and DFT-D3
- PME participates through energy and `EnergyDerivativesStep()`
- DFT-D3 still adds direct outputs afterward

```python
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    AIMNet2Potential,
    CompositeCalculator,
    DFTD3Potential,
    EnergyDerivativesStep,
    NeighborListBuilder,
    PMEPotential,
)

batch = Batch.from_data_list(
    [
        AtomicData(
            positions=torch.tensor(
                [[0.1, 0.2, 0.0], [1.1, 0.2, 0.0]],
                dtype=torch.float32,
            ),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
            cell=(8.0 * torch.eye(3, dtype=torch.float32)).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
        )
    ]
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

results = calculator(batch)
```

```{warning}
For charge-coupled pipelines, use `PMEPotential(..., derivative_mode="autograd")`
or `EwaldCoulombPotential(..., derivative_mode="autograd")`. Direct-mode PME
and Ewald are for fixed-charge use. DSF is the built-in hybrid Coulomb option.
```

In this example:

- AIMNet2 does not need an upstream neighbor builder
- PME and DFT-D3 both consume external matrix neighbors
- the shared long-range builder is created from DFT-D3's advertised contract
- `EnergyDerivativesStep()` differentiates the total AIMNet2 + PME energy once

## External Neighbor Lists

Some potentials require an upstream neighbor builder. When they do, the
contract is:

- the neighbor-list `name` must match
- the `format` must match
- the provided cutoff must be greater than or equal to the advertised cutoff

The easiest current path is:

```python
config = potential.neighbor_list_builder_config()
neighbor_builder = NeighborListBuilder(config)
```

The emitted config is seeded from the potential's resolved external-neighbor
contract. You may override it when the override keeps the contract valid.

```python
neighbor_builder = NeighborListBuilder(
    potential.neighbor_list_builder_config(cutoff=8.0)
)
```

Internal-neighbor potentials such as AIMNet2 return `None` from
`neighbor_list_builder_config()`, because they do not require an upstream
neighbor builder.

## Writing a Custom Potential

Most custom wrappers only need:

1. a module-level `PotentialCard`
2. a subclass of `Potential`
3. a `compute()` method that returns `CalculatorResults`

`Potential.__init__()` resolves the profile from the class `card`, so wrapper
authors normally do not need to call `card.to_profile(...)` directly.

### Minimal Energy-Only Example

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

Only reach for profile overrides when the instance really changes its public
contract, for example:

- changing the neighbor-list name it consumes
- changing neighbor format
- changing the declared required inputs

That is wrapper-author work, not normal end-user work.

## Known Model Registry

The registry is a runtime artifact-resolution layer. Use it when you want a
stable public name such as `"my_lab_model_v1"` to resolve to a cached local
artifact.

Built-in wrappers already use this idea for names such as:

- `AIMNet2Potential(model="aimnet2")`
- `MACEPotential(model="mace-mp-0b3-medium")`

### Register a Custom AIMNet Name

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

aimnet2_2025 = AIMNet2Potential(model="aimnet2_2025")
```

This pattern is also how you would add your own lab-specific checkpoint names.

### Resolve a Name in a Custom Wrapper

```python
from pathlib import Path

from nvalchemi.models import resolve_known_artifact


def resolve_model_path(model: str | Path) -> Path:
    model_path = Path(model)
    if model_path.exists():
        return model_path
    artifact = resolve_known_artifact(str(model), family="my_potential")
    return artifact.local_path
```

You can inspect built-in registry entries with
`list_known_artifacts(...)` before constructing a wrapper.

```{note}
Registry resolves artifacts only. It does not describe pipeline behavior,
neighbor contracts, or derivative semantics.
```

## Cards and Modules You May Care About

| Name | Why it matters |
|---|---|
| {py:class}`~nvalchemi.models.CompositeCalculator` | Main entry point for running a composite model |
| {py:class}`~nvalchemi.models.EnergyDerivativesStep` | Turns the current total energy into forces and stresses |
| {py:class}`~nvalchemi.models.PotentialCard` | Describes what a potential can consume and produce |
| {py:class}`~nvalchemi.models.MLIPPotentialCard` | Convenience card for MLIPs that participate in autograd derivatives |
| {py:class}`~nvalchemi.models.NeighborListCard` | Contract type for neighbor builders |
| {py:class}`~nvalchemi.models.ModelCard` | Metadata about model physics and checkpoint provenance |
| Registry helpers | Provide stable names and artifact download/cache resolution |

Most users work mainly with:

- concrete wrappers
- `CompositeCalculator`
- `EnergyDerivativesStep`
- `NeighborListBuilder`

Most users do not need to edit cards directly unless they are writing a new
wrapper.

## Practical Rules of Thumb

- Start with one MLIP, then add `EnergyDerivativesStep()`, then add direct terms.
- For external-neighbor potentials, use `neighbor_list_builder_config(...)`
  instead of guessing `name`, `format`, and minimum cutoff manually.
- Put `EnergyDerivativesStep()` after the energy terms that should participate
  in autograd.
- Put direct-output terms after `EnergyDerivativesStep()` if they should add
  their own forces or stresses directly.
- For charge-aware Coulomb with geometry-dependent charges, use `PMEPotential`
  or `EwaldCoulombPotential` in `autograd` mode.
- `DSFCoulombPotential` is the built-in hybrid Coulomb option.
- Cached `k_vectors` and `k_squared` are cell-dependent; they are not reused
  for stress or cell-derivative paths.

## Recommendations to Improve User Experience

The current API is usable, but a few additions would make it easier to adopt:

- Export a smaller, more opinionated top-level surface in docs and examples.
- Add helper functions for common recipes such as:
  - short-range MLIP + `EnergyDerivativesStep()`
  - MLIP + DSF
  - AIMNet2 + PME + `EnergyDerivativesStep()` + DFT-D3
- Add high-level neighbor presets so users do not have to choose `coo` versus
  `matrix` so early.
- Add a short decision table: "Which composite should I use?"
- Surface registry discovery more prominently in package-level examples; the
  low-level listing helper already exists, but it is still easy to miss.
- Reduce repeated configuration for shared long-range neighbor lists beyond the
  current per-potential `neighbor_list_builder_config(...)` helper.

The guiding principle should stay simple:

- avoid mechanical setup work for the user
- keep the important physical choices explicit
- hide framework plumbing unless the user is writing a new wrapper
