# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Composable model examples with the current `nvalchemi.models` API.

This example shows three current composition patterns:

1. A simplified additive composition with the ``+`` operator.
2. A simplified dependent chain that relies on canonical ``node_charges``.
3. An explicit wire for a non-canonical producer that emits ``charges``.

The additive example executes end to end. The dependent and explicit-wire
sections focus on construction and ``repr(calc)`` so the walkthrough stays
lightweight and centered on the current composition UX.

Two tiny local helpers are defined below only to make that contrast concrete:

* ``ExampleChargeModelCanonical`` publishes canonical ``node_charges``, so the
  dependent chain works without any explicit wiring.
* ``ExampleChargeModel`` publishes legacy ``charges``, so the explicit
  ``wire_output(...)`` path can be demonstrated with the same toy model shape.

The example also prints ``repr(calc)`` for each composed model. The repr is an
effective-pipeline summary:

* steps are numbered from ``[0]``
* synthesized external neighbor builders appear explicitly
* internal-neighbor models stay on their own step
* ``*`` marks graph-connected values in the autograd region
* ``wires:`` appears only when an explicit ``wire_output(...)`` mapping exists
"""

from __future__ import annotations

import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    ComposableModelWrapper,
    DemoModelWrapper,
    PMEModelWrapper,
)
from nvalchemi.models.base import BaseModelMixin, ModelConfig


def _sum_per_graph(values: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """Reduce one node-level tensor to graph-level sums."""

    num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 1
    out = torch.zeros(
        (num_graphs, values.shape[-1]),
        device=values.device,
        dtype=values.dtype,
    )
    out.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(values), values)
    return out


class ExampleChargeModelCanonical(nn.Module, BaseModelMixin):
    """Small canonical charge-producing model used in the simplified example.

    This helper intentionally publishes ``node_charges`` as its public output.
    It exists to demonstrate the preferred composition path: when wrappers use
    canonical names consistently, dependent composition does not need an
    explicit ``wire_output(...)`` call.
    """

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers"}),
        optional_inputs=frozenset({"batch", "cell", "pbc"}),
        outputs=frozenset({"energies", "node_charges"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
        autograd_outputs=frozenset({"node_charges"}),
        pbc_mode="any",
    )

    def __init__(self, *, name: str = "charge_net") -> None:
        super().__init__()
        self._name = name

    def __repr__(self) -> str:
        """Return a stable display name for the composite repr."""

        return "ExampleChargeModelCanonical()"

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute simple canonical node charges and graph energies."""

        positions = data["positions"]
        batch_idx = data.get(
            "batch",
            torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        ).long()
        node_charges = positions.sum(dim=-1, keepdim=True)
        per_node_energy = 0.25 * positions.pow(2).sum(dim=-1, keepdim=True)
        energies = _sum_per_graph(per_node_energy, batch_idx)
        return {"energies": energies, "node_charges": node_charges}


class ExampleChargeModel(nn.Module, BaseModelMixin):
    """Small non-canonical charge-producing model used for the wire example.

    This helper computes the same toy energies and per-node values as
    :class:`ExampleChargeModelCanonical`, but it intentionally exposes the
    legacy output key ``charges`` instead of the canonical ``node_charges``.
    That one difference is enough to require an explicit ``wire_output(...)``
    mapping in the composed example below.
    """

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers"}),
        optional_inputs=frozenset({"batch", "cell", "pbc"}),
        outputs=frozenset({"energies", "charges"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
        autograd_outputs=frozenset({"charges"}),
        pbc_mode="any",
    )

    def __init__(self, *, name: str = "charge_net") -> None:
        super().__init__()
        self._name = name

    def __repr__(self) -> str:
        """Return a stable display name for the composite repr."""

        return "ExampleChargeModel()"

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute simple non-canonical charges and graph energies."""

        positions = data["positions"]
        batch_idx = data.get(
            "batch",
            torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        ).long()
        charges = positions.sum(dim=-1, keepdim=True)
        per_node_energy = 0.25 * positions.pow(2).sum(dim=-1, keepdim=True)
        energies = _sum_per_graph(per_node_energy, batch_idx)
        return {"energies": energies, "charges": charges}


def _make_nonperiodic_batch() -> Batch:
    """Build one small non-periodic batch for additive examples."""

    data = AtomicData(
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.8, 0.1, 0.0],
                [0.2, 1.7, 0.0],
                [1.7, 1.8, 0.0],
            ],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([6, 6, 8, 1], dtype=torch.long),
    )
    return Batch.from_data_list([data])


def _make_periodic_batch() -> Batch:
    """Build one small periodic batch for charge-plus-PME examples."""

    cell = torch.tensor(
        [[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]],
        dtype=torch.float32,
    )
    data = AtomicData(
        positions=torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [3.0, 1.5, 1.2],
                [2.0, 3.5, 1.4],
                [5.0, 4.5, 3.8],
            ],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([8, 1, 1, 6], dtype=torch.long),
        cell=cell,
        pbc=torch.tensor([[True, True, True]]),
    )
    return Batch.from_data_list([data])


def _print_section(title: str) -> None:
    """Print one section header."""

    print()
    print(title)
    print("=" * len(title))


def _print_snippet(snippet: str) -> None:
    """Print one short construction snippet."""

    print("constructed as:")
    for line in snippet.strip().splitlines():
        print(f"  {line}")


def _print_output_shapes(outputs: dict[str, torch.Tensor]) -> None:
    """Print one compact output summary."""

    print("outputs:")
    for key, value in outputs.items():
        print(f"  - {key}: shape={tuple(value.shape)}")


def _show_composite(
    *,
    title: str,
    snippet: str,
    calc: ComposableModelWrapper,
    batch: Batch | None = None,
    compute: set[str] | None = None,
    notes: tuple[str, ...],
) -> None:
    """Print one composite construction, repr, and runtime output summary."""

    _print_section(title)
    _print_snippet(snippet)
    print()
    print("repr(calc):")
    print(calc)
    print()
    print("what to look for:")
    for note in notes:
        print(f"  - {note}")
    print()
    if batch is not None and compute is not None:
        outputs = calc(batch, compute=compute)
        _print_output_shapes(outputs)
    else:
        print("runtime call:")
        print(
            "  - skipped in this example; the focus here is the constructed composite and repr(calc)."
        )


def main() -> None:
    """Run the composition walkthrough."""

    additive_batch = _make_nonperiodic_batch()

    additive_calc = DemoModelWrapper(name="demo_a") + DemoModelWrapper(
        hidden_dim=32,
        name="demo_b",
    )
    _show_composite(
        title="Simplified additive composition",
        snippet="""
demo_a = DemoModelWrapper(name="demo_a")
demo_b = DemoModelWrapper(hidden_dim=32, name="demo_b")
calc = demo_a + demo_b
""",
        calc=additive_calc,
        batch=additive_batch,
        compute={"energies", "forces"},
        notes=(
            "Both models contribute additive outputs.",
            "No explicit wire is needed because the models are independent.",
            "The same + syntax also applies to real stacks such as MACEWrapper + DFTD3ModelWrapper.",
        ),
    )

    # Preferred path: the example model already publishes canonical
    # `node_charges`, so the dependent composition needs no explicit wire.
    canonical_charge_calc = ExampleChargeModelCanonical() + PMEModelWrapper(cutoff=8.0)
    _show_composite(
        title="Simplified dependent composition",
        snippet="""
charge_model = ExampleChargeModelCanonical()
pme = PMEModelWrapper(cutoff=8.0)
calc = charge_model + pme
""",
        calc=canonical_charge_calc,
        notes=(
            "The canonical example model publishes node_charges directly.",
            "PME consumes node_charges without any explicit wiring.",
            "The starred values in repr(calc) stay graph-connected before the derivative step.",
        ),
    )

    # Escape hatch: this otherwise-equivalent example model publishes the
    # legacy key `charges`, so the rename has to be stated explicitly.
    renamed_source = ExampleChargeModel()
    explicit_wire_calc = ComposableModelWrapper(
        renamed_source,
        PMEModelWrapper(cutoff=8.0),
    )
    explicit_wire_calc.wire_output(
        renamed_source,
        explicit_wire_calc.models[1],
        {"node_charges": "charges"},
    )
    _show_composite(
        title="Explicit rename wire",
        snippet="""
source = ExampleChargeModel()
target = PMEModelWrapper(cutoff=8.0)
calc = ComposableModelWrapper(source, target)
calc.wire_output(source, target, {"node_charges": "charges"})
""",
        calc=explicit_wire_calc,
        notes=(
            "This example reuses the same toy charge model idea, but with the legacy output key charges.",
            "Use wire_output only when a producer and consumer do not already agree on key names.",
            "The wires: line in repr(calc) shows the explicit rename.",
            "The rest of the runtime plan remains inferred automatically.",
        ),
    )


if __name__ == "__main__":
    main()
