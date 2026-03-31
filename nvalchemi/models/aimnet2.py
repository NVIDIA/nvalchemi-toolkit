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

from pathlib import Path

import torch
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.models.base import (
    ForwardContext,
    Potential,
)
from nvalchemi.models.contracts import (
    MLIPPotentialCard,
    NeighborRequirement,
)
from nvalchemi.models.metadata import (
    ATOMIC_CHARGES,
    DISPERSION,
    ELECTROSTATICS,
    PAIRWISE,
    SHORT_RANGE,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.registry import ResolvedArtifact
from nvalchemi.models.results import CalculatorResults

__all__ = ["AIMNet2Potential"]

AIMNet2PotentialCard = MLIPPotentialCard(
    required_inputs=frozenset({"positions", "atomic_numbers"}),
    optional_inputs=frozenset({"cell", "pbc", "graph_charges", "graph_spins"}),
    result_keys=frozenset({"energies", "node_charges"}),
    boundary_modes=frozenset({"non_pbc", "pbc"}),
    neighbor_requirement=NeighborRequirement(source="internal"),
    parameterized_by=frozenset(),
)


class AIMNet2Potential(Potential):
    """AIMNet2 potential backed by the canonical AIMNetCentral flat path.

    Attributes
    ----------
    card : MLIPPotentialCard
        Class-level contract card describing required/optional inputs,
        result keys, and neighbor requirements.
    model_family : str
        Registry family identifier (``"aimnet2"``).
    model_card : ModelCard or None
        Checkpoint-level scientific and provenance metadata.

    Examples
    --------
    Instantiate from a registry name:

    >>> from nvalchemi.models.aimnet2 import AIMNet2Potential
    >>> potential = AIMNet2Potential("aimnet2")

    Load from a local checkpoint without ``torch.compile``:

    >>> potential = AIMNet2Potential("/path/to/checkpoint.pt", compile_model=False)
    """

    card = AIMNet2PotentialCard
    model_family = "aimnet2"

    def __init__(
        self,
        model: str | Path | nn.Module,
        *,
        device: str | torch.device | None = "cuda",
        compile_model: bool = True,
        model_card: ModelCard | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise an AIMNet2 potential wrapper.

        Parameters
        ----------
        model
            An AIMNet2 model specified as a named registry artifact,
            a local checkpoint path, or a pre-instantiated ``nn.Module``.
        device
            Target execution device.  Defaults to ``"cuda"``.
        compile_model
            Flag to apply ``torch.compile`` to the loaded model.
            Default ``True``.
        model_card
            Explicit model metadata.  When ``None`` a default card is
            inferred from the checkpoint provenance.  Default ``None``.
        name
            Human-readable step name.  Default ``None``.

        Raises
        ------
        ImportError
            If the ``aimnet`` package is not installed.
        """

        try:
            from aimnet.calculators import AIMNet2Calculator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                f"aimnet is required to use {type(self).__name__}. "
                "Install AIMNetCentral or make it importable on PYTHONPATH."
            ) from exc

        resolved_artifact = self._resolve_known_model(model)
        effective_model: str | Path | nn.Module = (
            resolved_artifact.local_path if resolved_artifact is not None else model
        )

        super().__init__(name=name)
        self.model_card = model_card or self._default_model_card(
            model,
            resolved_artifact=resolved_artifact,
        )

        model_arg = (
            str(effective_model)
            if isinstance(effective_model, Path)
            else effective_model
        )
        self._calculator = AIMNet2Calculator(
            model=model_arg,
            device=str(device) if device is not None else None,
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=compile_model,
            train=False,
        )
        self._device = torch.device(self._calculator.device)
        self._model = self._calculator.model
        raw_model = self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        self._is_nse = getattr(raw_model, "num_charge_channels", 1) == 2

    @classmethod
    def _default_model_card(
        cls,
        model: str | Path | nn.Module,
        *,
        resolved_artifact: ResolvedArtifact | None = None,
    ) -> ModelCard:
        """Return default checkpoint metadata for the current AIMNet2 input.

        Extends the base card with AIMNet2-specific physics terms and,
        when available, the reference XC functional from registry metadata.

        Parameters
        ----------
        model
            Model identifier passed to the constructor (name, path, or module).
        resolved_artifact
            Registry resolution result, or ``None`` when the model was not
            resolved through the artifact registry.

        Returns
        -------
        ModelCard
            Populated card with AIMNet2-specific ``provided_terms``,
            ``required_external_terms``, and ``optional_external_terms``.
        """

        base = super()._default_model_card(model, resolved_artifact=resolved_artifact)
        xc_functional: str | None = (
            str(resolved_artifact.entry.metadata["reference_xc_functional"])
            if resolved_artifact is not None
            and resolved_artifact.entry.metadata.get("reference_xc_functional")
            is not None
            else None
        )

        return base.model_copy(
            update={
                "reference_xc_functional": xc_functional,
                "provided_terms": (PhysicalTerm(kind=SHORT_RANGE),),
                "required_external_terms": (
                    PhysicalTerm(kind=ELECTROSTATICS, variant=ATOMIC_CHARGES),
                ),
                "optional_external_terms": (
                    PhysicalTerm(kind=DISPERSION, variant=PAIRWISE),
                ),
            },
        )

    def _build_flat_input(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> dict[str, torch.Tensor]:
        """Build the canonical flat AIMNet calculator input from the toolkit batch.

        Parameters
        ----------
        batch
            Current batched atomic graph.
        ctx
            Forward context carrying resolved outputs, results, and
            runtime state.

        Returns
        -------
        dict[str, torch.Tensor]
            Flat dictionary accepted by the AIMNet calculator
            (keys: ``"coord"``, ``"numbers"``, ``"mol_idx"``,
            ``"charge"``, and optionally ``"cell"`` / ``"mult"``).

        Raises
        ------
        ValueError
            If the ``cell`` tensor does not have one entry per graph.
        """

        positions = torch.as_tensor(
            self.require_input(batch, "positions", ctx),
            device=self._calculator.device,
            dtype=torch.float32,
        )
        atomic_numbers = torch.as_tensor(
            self.require_input(batch, "atomic_numbers", ctx),
            device=self._calculator.device,
            dtype=torch.long,
        )

        model_input: dict[str, torch.Tensor] = {
            "coord": positions,
            "numbers": atomic_numbers,
            "mol_idx": batch.batch.to(device=self._calculator.device, dtype=torch.long),
            "charge": self.normalize_graph_scalar(
                batch,
                "graph_charges",
                ctx,
                default=0.0,
                dtype=torch.float32,
                device=self._calculator.device,
            ),
        }

        cell, _pbc = self.resolve_periodic_inputs(batch, ctx)
        if cell is not None:
            cell_tensor = torch.as_tensor(
                cell,
                device=self._calculator.device,
                dtype=torch.float32,
            )
            if cell_tensor.ndim == 2:
                cell_tensor = cell_tensor.unsqueeze(0)
            if cell_tensor.shape[0] != batch.num_graphs:
                raise ValueError(
                    f"'cell' must have one entry per graph. Expected {batch.num_graphs}, "
                    f"got {tuple(cell_tensor.shape)}."
                )
            model_input["cell"] = cell_tensor

        if self._is_nse:
            mult = self.normalize_graph_scalar(
                batch,
                "graph_spins",
                ctx,
                default=0.0,
                dtype=torch.float32,
                device=self._calculator.device,
            )
            model_input["mult"] = mult + 1.0

        return model_input

    def _prepare_model_input(
        self,
        calc_input: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Run the canonical flat AIMNet preprocessing path without detaching tensors.

        Parameters
        ----------
        calc_input
            Flat dictionary produced by :meth:`_build_flat_input`.

        Returns
        -------
        tuple[dict[str, torch.Tensor], bool]
            The preprocessed model input and a flag indicating whether
            flat padding was applied (needed to call ``unpad_output``
            after inference).
        """

        data = self._calculator.mol_flatten(calc_input)
        used_flat_padding = False
        if data["coord"].ndim == 2:
            if "nbmat" not in data:
                data = self._calculator.make_nbmat(data)
            data = self._calculator.pad_input(data)
            used_flat_padding = True
        return data, used_flat_padding

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Run AIMNet2 on the current batch and return requested outputs.

        Parameters
        ----------
        batch
            Current batched atomic graph.
        ctx
            Forward context carrying resolved outputs, results, and
            runtime state.

        Returns
        -------
        CalculatorResults
            Result container with ``energies`` and, when available,
            ``node_charges``.
        """

        calc_input = self._build_flat_input(batch, ctx)
        model_input, used_flat_padding = self._prepare_model_input(calc_input)
        raw_output = self._model(model_input)
        if used_flat_padding:
            raw_output = self._calculator.unpad_output(raw_output)

        energies = self.normalize_system_energies(
            raw_output["energy"],
            num_graphs=batch.num_graphs,
            source_name="AIMNet",
        )

        node_charges = None
        if "charges" in raw_output:
            node_charges = raw_output["charges"]
            if node_charges.ndim == 1:
                node_charges = node_charges.unsqueeze(-1)

        return self.build_results(ctx, energies=energies, node_charges=node_charges)
