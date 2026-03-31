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
from __future__ import annotations

from collections.abc import Iterable

import torch

from nvalchemi.data import Batch
from nvalchemi.models.base import (
    ForwardContext,
    RuntimeState,
    _CalculationStep,
)
from nvalchemi.models.contracts import StepCard, StepProfile
from nvalchemi.models.results import CalculatorResults


class EnergyDerivativesStep(_CalculationStep):
    """Step that differentiates accumulated energy to produce forces and stresses.

    This is a regular pipeline step placed explicitly by the user.
    It reads the total accumulated ``energies`` from pipeline results
    and differentiates w.r.t. positions (forces) and cell scaling
    (stresses) using ``torch.autograd.grad``.

    Gradient tracking is set up by participating potentials during the
    ``prepare()`` phase via the shared :func:`ensure_derivative_targets`
    helper.  This step only declares what derivative targets it needs
    through :meth:`requested_derivative_targets` and consumes the
    prepared targets during :meth:`compute`.
    """

    card = StepCard(
        required_inputs=frozenset({"energies"}),
        result_keys=frozenset({"forces", "stresses"}),
        additive_result_keys=frozenset({"forces", "stresses"}),
        parameterized_by=frozenset(),
    )

    profile: StepProfile

    _OUTPUT_TARGETS: dict[str, frozenset[str]] = {
        "forces": frozenset({"positions"}),
        "stresses": frozenset({"cell_scaling"}),
    }

    def __init__(self, *, name: str | None = None) -> None:
        """Initialise an energy-derivatives step.

        Parameters
        ----------
        name
            Human-readable step name.  Defaults to
            ``"energy_derivatives"``.
        """

        super().__init__(type(self).card.to_profile(), name=name or "energy_derivatives")

    def required_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return dynamic required inputs based on requested outputs.

        Forces need ``energies``.  Stresses additionally need ``cell``
        and ``pbc`` for the affine stress scaling setup.
        """

        active = self.active_outputs(outputs)
        base = frozenset({"energies"})
        if "stresses" in active:
            return base | frozenset({"cell", "pbc"})
        return base

    def requested_derivative_targets(
        self,
        outputs: frozenset[str],
    ) -> frozenset[str]:
        """Return derivative targets needed for the given outputs.

        Parameters
        ----------
        outputs
            The resolved set of outputs requested from this step.

        Returns
        -------
        frozenset[str]
            ``{"positions"}`` for forces, ``{"cell_scaling"}`` for stresses,
            or both when both are requested.
        """

        return self._resolve_derivative_targets(outputs)

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Differentiate accumulated energy w.r.t. derivative targets."""

        if ctx.runtime_state is None:
            raise RuntimeError("EnergyDerivativesStep requires runtime derivative state.")

        energy = self.require_input(batch, "energies", ctx)
        return self._differentiate_energy(
            energy,
            runtime_state=ctx.runtime_state,
            outputs=ctx.outputs,
        )

    # -- internal helpers -----------------------------------------------------

    @classmethod
    def _resolve_derivative_targets(
        cls,
        outputs: frozenset[str],
    ) -> frozenset[str]:
        """Resolve requested public outputs to internal derivative targets."""

        targets: set[str] = set()
        for output_name in outputs:
            targets |= cls._OUTPUT_TARGETS.get(output_name, frozenset())
        return frozenset(targets)

    @classmethod
    def _differentiate_energy(
        cls,
        energy: torch.Tensor,
        *,
        runtime_state: RuntimeState,
        outputs: frozenset[str] = frozenset(),
    ) -> CalculatorResults:
        """Differentiate energy with respect to internal named targets."""

        target_names = cls._resolve_derivative_targets(outputs)
        if not target_names:
            return CalculatorResults()

        missing_targets = [
            name for name in target_names if name not in runtime_state.derivative_targets
        ]
        if missing_targets:
            raise RuntimeError(
                "Missing derivative targets in internal runtime state: "
                f"{missing_targets}. "
                "Ensure at least one potential in the pipeline declares "
                "gradient_setup_targets for these targets."
            )

        target_tensors = [runtime_state.derivative_targets[name] for name in target_names]
        gradients = torch.autograd.grad(
            energy.sum(),
            target_tensors,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        gradient_by_target = dict(zip(target_names, gradients, strict=True))
        return cls._gradients_to_results(
            gradient_by_target,
            runtime_state=runtime_state,
            outputs=outputs,
        )

    @staticmethod
    def _gradients_to_results(
        gradients: dict[str, torch.Tensor],
        *,
        runtime_state: RuntimeState,
        outputs: frozenset[str],
    ) -> CalculatorResults:
        """Convert target gradients into physical result outputs."""

        result = CalculatorResults()
        if "forces" in outputs:
            result["forces"] = -gradients["positions"]
        if "stresses" in outputs:
            cell = runtime_state.input_overrides.get("cell")
            if cell is None:
                raise ValueError("Stress computation requires a scaled cell override.")
            stress_grad = gradients["cell_scaling"]
            volume = torch.abs(torch.linalg.det(cell))
            if volume.ndim == 0:
                result["stresses"] = stress_grad / volume
            else:
                result["stresses"] = stress_grad / volume.unsqueeze(-1).unsqueeze(-1)
        return result
