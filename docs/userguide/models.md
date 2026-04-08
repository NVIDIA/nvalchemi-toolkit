<!-- markdownlint-disable MD014 -->

(models_guide)=

# Models: Wrappers, Composition, and Inspection

The ALCHEMI Toolkit exposes model wrappers through
{py:class}`~nvalchemi.models.base.BaseModelMixin`. Every wrapper declares a
static execution contract with
{py:class}`~nvalchemi.models.base.ModelConfig`, and composed models are built
with {py:class}`~nvalchemi.models.composable.ComposableModelWrapper`.

This guide covers:

1. The wrapper classes available out of the box.
2. The current wrapper contract: `spec`, canonical input/output keys, and
   neighbor requirements.
3. The two composition paths:
   - simplified UX for canonical models
   - explicit UX for renamed or ambiguous data flow
4. How to inspect composed model.

## Supported wrappers

The {py:mod}`nvalchemi.models` package currently ships wrappers for the
following model families:

| Wrapper class | Role | Notes |
| :--- | :--- | :--- |
| {py:class}`~nvalchemi.models.demo.DemoModelWrapper` | Small deterministic demo model | Useful for tests and tutorials |
| {py:class}`~nvalchemi.models.aimnet2.AIMNet2Wrapper` | Learned energy + charge model | Requires the optional `aimnet` dependency |
| {py:class}`~nvalchemi.models.mace.MACEWrapper` | Learned short-range MLIP | Requires the optional `mace-torch` dependency |
| {py:class}`~nvalchemi.models.dftd3.DFTD3ModelWrapper` | Direct dispersion correction | Uses bundled DFT-D3 parameter caching |
| {py:class}`~nvalchemi.models.lj.LennardJonesModelWrapper` | Direct short-range pair potential | Uses external matrix neighbor lists |
| {py:class}`~nvalchemi.models.dsf.DSFModelWrapper` | Coulomb interaction with hybrid forces | Consumes external COO neighbor lists |
| {py:class}`~nvalchemi.models.ewald.EwaldModelWrapper` | Long-range Coulomb via Ewald | Periodic systems only |
| {py:class}`~nvalchemi.models.pme.PMEModelWrapper` | Long-range Coulomb via PME | Periodic systems only |

`AIMNet2Wrapper` and `MACEWrapper` are lazily imported. Missing optional
dependencies do not break unrelated model imports.

## Wrapper contract

The current wrapper boundary is small and explicit:

- subclass `nn.Module` and `BaseModelMixin`
- provide a class-level or instance-level `spec: ModelConfig`
- implement `forward(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`
- use canonical Toolkit keys for public inputs and outputs

The most important pieces of `ModelConfig` are:

| Field | Meaning |
| :--- | :--- |
| `required_inputs` | Keys that must be present for the wrapper to run |
| `optional_inputs` | Keys the wrapper can consume when available |
| `outputs` | Keys always produced by the wrapper |
| `optional_outputs` | Outputs gated by optional inputs such as `cell` / `pbc` |
| `additive_outputs` | Outputs that should be summed across composed models |
| `use_autograd` | Whether the wrapper belongs to the autograd-connected region |
| `autograd_outputs` | Non-additive outputs that stay available inside the autograd region |
| `neighbor_config` | Whether neighbors are internal, external, or not needed |

For example, a charge-producing wrapper should expose canonical
`node_charges`, not a backend-specific name such as `charges`.

```python
from __future__ import annotations

import torch
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig


class ChargePredictorWrapper(nn.Module, BaseModelMixin):
    spec = ModelConfig(
        required_inputs=frozenset({"atomic_numbers", "positions"}),
        optional_inputs=frozenset({"cell", "pbc"}),
        outputs=frozenset({"energies", "node_charges"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
        autograd_outputs=frozenset({"node_charges"}),
        pbc_mode="any",
    )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        energies = positions.pow(2).sum(dim=-1, keepdim=True).sum(dim=0, keepdim=True)
        node_charges = positions.sum(dim=-1, keepdim=True)
        return {"energies": energies, "node_charges": node_charges}
```

The default {py:meth}`~nvalchemi.models.base.BaseModelMixin.adapt_input` and
{py:meth}`~nvalchemi.models.base.BaseModelMixin.adapt_output` helpers are still
available, but most wrappers work directly with canonical batch/context
keys and a regular `forward(data)` method.

## Simplified composition UX

The recommended common case is the short path:

```python
from nvalchemi.models import DFTD3ModelWrapper, MACEWrapper

calc = MACEWrapper(model="medium-0b2") + DFTD3ModelWrapper(functional="pbe")
```

That works when:

- the models use canonical names
- the data flow is unambiguous
- autograd-connected models appear before trailing direct corrections

The same simplified path also covers dependent composition when producer and
consumer agree on the canonical key names:

```python
from nvalchemi.models import AIMNet2Wrapper, PMEModelWrapper

calc = AIMNet2Wrapper(model="aimnet2") + PMEModelWrapper(cutoff=12.0)
outputs = calc(batch, compute={"energies", "forces", "node_charges"})
```

In that example:

- `AIMNet2Wrapper` publishes `node_charges`
- `PMEModelWrapper` consumes `node_charges`
- the composed model can backpropagate forces through the full energy sum

No explicit wiring is required in the canonical case.

## Explicit composition UX

Use explicit wiring only when the producer and consumer do not already agree on
the public key names, or when multiple producers make the dataflow ambiguous.

```python
from nvalchemi.models import ComposableModelWrapper, PMEModelWrapper

source = ChargeNetWrapper()
target = PMEModelWrapper(cutoff=12.0)

calc = ComposableModelWrapper(source, target)
calc.wire_output(source, target, {"node_charges": "charges"})
```

`wire_output(source, target, mapping)` interprets `mapping` as:

```python
{target_input_key: source_output_key}
```

So the mapping above means:

- take `source.charges`
- provide it to `target` as `node_charges`

The explicit path is the escape hatch. The simplified path is the recommended
default for wrappers that already use canonical keys.

## Reading `repr(calc)`

`repr(calc)` shows the effective runtime plan for a
{py:class}`~nvalchemi.models.composable.ComposableModelWrapper`.

```python
from nvalchemi.models import AIMNet2Wrapper, PMEModelWrapper

calc = AIMNet2Wrapper(model="aimnet2") + PMEModelWrapper(cutoff=12.0)
print(calc)
```

Representative output:

```text
ComposableModelWrapper(
  inputs: atomic_numbers, positions, cell?, pbc?
  outputs: energies, forces, node_charges, stresses?

  -----
  grad enabled: cell*, positions*

  [0] AIMNet2Wrapper(model='aimnet2')
      inputs: atomic_numbers, positions*, cell*?, pbc?
      outputs: energies*, node_charges*

  [1] NeighborListBuilder(cutoff=12.0, format='matrix')
      inputs: positions, cell?, pbc?
      outputs: neighbor_matrix, num_neighbors, neighbor_shifts?

  [2] PMEModelWrapper(cutoff=12.0)
      inputs: neighbor_matrix, node_charges*, positions*, cell*?, neighbor_shifts?, num_neighbors?, pbc?
      outputs: energies*, stresses*?

  [3] DerivativeStep(forces=True, stresses=True)
      inputs: energies*, cell?, pbc?
      grad targets: cell, positions
      outputs: forces, stresses?
)
```

Read it as follows:

- steps are zero-based: `[0]`, `[1]`, `[2]`, ...
- `NeighborListBuilder(...)` steps are synthesized only for wrappers that
  declare external neighbors
- internal-neighbor wrappers, such as AIMNet2, stay on their own model step
- `*` marks values that are graph-connected inside the active autograd region
- `?` marks inputs or outputs that are conditional
- `wires:` appears only when explicit `wire_output(...)` mappings are present

That makes `repr(calc)` useful for answering three concrete questions:

1. What inputs does the composite really expect?
2. Where are neighbor lists synthesized?
3. Which values stay connected through the autograd region before derivatives
   are taken?

## Worked example: wrapping a small model

`DemoModelWrapper` is the smallest fully documented reference wrapper in the
repository. It follows the same current pattern that custom wrappers should
use:

- declare a stable `spec`
- read canonical keys from `data`
- return canonical outputs from `forward`

```python
from nvalchemi.models.demo import DemoModelWrapper

model = DemoModelWrapper(hidden_dim=32, name="demo")
print(model)
```

When your model has no special input reshaping, the default
`BaseModelMixin.adapt_input()` and `adapt_output()` implementations are usually
sufficient. The wrapper work is then mostly about declaring the right
`ModelConfig` and returning the correct canonical keys.

## See also

- `examples/advanced/07_composable_model_composition.py`
- `examples/advanced/08_composable_model_dynamics.py`
- {py:class}`~nvalchemi.models.composable.ComposableModelWrapper`
- {py:class}`~nvalchemi.models.base.ModelConfig`
- {py:class}`~nvalchemi.models.base.NeighborConfig`
