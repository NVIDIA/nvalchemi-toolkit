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
"""AIMNet2 model wrapper.

Wraps an AIMNet2 ``nn.Module`` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model, ready for
use in any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine or standalone
inference.

Usage
-----
Load from a checkpoint (downloads if needed)::

    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    wrapper = AIMNet2Wrapper.from_checkpoint("aimnet2", device="cuda")

Or wrap an already-loaded ``nn.Module``::

    raw_model = torch.load("aimnet2.pt", weights_only=False)
    wrapper = AIMNet2Wrapper(raw_model)

Notes
-----
* Energy is the primitive differentiable output. Forces and stresses are
  derived via autograd (``autograd_outputs={"forces", "stress"}``).
* AIMNet2 also predicts partial charges, which are available as a direct
  output (``"charges" in model_config.outputs``).
* Coulomb and D3 dispersion contributions are **disabled** inside the
  calculator — use :class:`~nvalchemi.models.pipeline.PipelineModelWrapper`
  to compose with :class:`~nvalchemi.models.ewald.EwaldModelWrapper` or
  :class:`~nvalchemi.models.dftd3.DFTD3ModelWrapper` for long-range
  interactions.
* AIMNet2 runs in **float32 only**. The wrapper enforces this.
* NSE (Neutral Spin Equilibrated) models are auto-detected at construction
  time. When detected, ``spin_charges`` is added to the output set.
* The wrapper uses an **external neighbor list** (MATRIX format) provided
  by :class:`~nvalchemi.dynamics.hooks.NeighborListHook`.  The neighbor
  matrix is converted to AIMNet2's internal ``nbmat`` format (with a
  padding row) before the model forward pass.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import prepare_strain
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

__all__ = ["AIMNet2Wrapper"]


@OptionalDependency.AIMNET.require
class AIMNet2Wrapper(nn.Module, BaseModelMixin):
    """Wrapper for AIMNet2 interatomic potentials.

    Energy is always computed as the primitive differentiable output via
    the raw AIMNet2 model. Forces and stresses are derived from energy
    via autograd. Partial charges and node embeddings (AIM features) are
    taken directly from the model outputs.

    The wrapper declares an **external** MATRIX-format neighbor list
    requirement at the model's AEV cutoff. The
    :class:`~nvalchemi.dynamics.hooks.NeighborListHook` (or the pipeline's
    synthesized hook) populates ``neighbor_matrix`` on the batch before
    each forward pass.  The wrapper converts this to AIMNet2's internal
    ``nbmat`` format (with a padding row for the padding atom).

    Coulomb and D3 dispersion are disabled.  Use
    :class:`~nvalchemi.models.pipeline.PipelineModelWrapper` to compose
    AIMNet2 with electrostatics or dispersion models.

    Parameters
    ----------
    model : nn.Module
        An AIMNet2 model (loaded from checkpoint or instantiated
        directly).  Use :meth:`from_checkpoint` for the common
        construction path.

    Attributes
    ----------
    model_config : ModelConfig
        Configuration with capability and runtime fields.
    model : nn.Module
        The underlying AIMNet2 model. If you want your model
        to be compiled, wrap with torch.compile(model, **kwargs)
        before passing here.
    """

    model: nn.Module

    def __init__(self, model: nn.Module) -> None:
        from aimnet.calculators import AIMNet2Calculator

        super().__init__()
        self.model = model

        # Build a calculator for its pad_input / unpad_output utilities.
        # We no longer use it for neighbor list construction.
        self._calculator = AIMNet2Calculator(
            model=model,
            device=str(next(model.parameters()).device),
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=False,
            train=False,
        )

        # Detect NSE (Neutral Spin Equilibrated) models.
        raw_model = model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        self._is_nse = getattr(raw_model, "num_charge_channels", 1) == 2
        if self._is_nse:
            if "spin_charges" not in self._calculator.keys_out:
                self._calculator.keys_out = [*self._calculator.keys_out, "spin_charges"]

        # Extract cutoff from the loaded model.
        self._cutoff = self._extract_cutoff(raw_model)

        # Build the model config with external neighbor list.
        outputs = {"energy", "forces", "stress", "charges"}
        if self._is_nse:
            outputs.add("spin_charges")

        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charge"}),
            optional_inputs=frozenset({"cell", "mult"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self._cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=False,
                # max_neighbors left as None — NeighborListHook will
                # auto-estimate via estimate_max_neighbors(cutoff).
            ),
            active_outputs={"energy", "forces", "charges"},
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
        compile_model: bool = False,
        **compile_kwargs: Any,
    ) -> "AIMNet2Wrapper":
        """Load an AIMNet2 model and return a wrapped instance.

        Uses ``AIMNet2Calculator`` to resolve and load the checkpoint,
        then extracts the raw ``nn.Module`` and wraps it.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to an AIMNet2 checkpoint file, or a model alias
            recognized by ``AIMNet2Calculator`` (e.g. ``"aimnet2"``).
        device : torch.device | str, optional
            Target device. Defaults to ``"cpu"``.
        compile_model: bool, optional
            Apply ``torch.compile``.  Sets eval mode and freezes parameters;
            the model is **inference-only** after this step.
        **compile_kwargs
            Forwarded to ``torch.compile``.
        Returns
        -------
        AIMNet2Wrapper
        """
        from aimnet.calculators import AIMNet2Calculator

        calc = AIMNet2Calculator(
            model=str(checkpoint_path),
            device=str(device),
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=compile_model,
            compile_kwargs=compile_kwargs,
            train=False,
        )
        raw_model = calc.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        return cls(raw_model)

    @staticmethod
    def _extract_cutoff(raw_model: nn.Module) -> float:
        """Extract the AEV interaction cutoff from the loaded model."""
        aev = getattr(raw_model, "aev", None)
        if aev is None:
            return 5.0  # default AIMNet2 cutoff
        rc_s = getattr(aev, "rc_s", None)
        rc_v = getattr(aev, "rc_v", None)
        values = [float(v) for v in (rc_s, rc_v) if v is not None]
        return max(values) if values else 5.0

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return AIMNet2 AIM feature embedding shapes."""
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        aim_dim = 256
        aev = getattr(raw_model, "aev", None)
        if aev is not None:
            output_size = getattr(aev, "output_size", None)
            if output_size is not None:
                aim_dim = int(output_size)
        return {"node_embeddings": (aim_dim,)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute AIMNet2 AIM feature embeddings and attach to data."""
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        model_input = self.adapt_input(data, **kwargs)
        n_real = data.num_nodes

        with torch.no_grad():
            raw_output = self._calculator.model(model_input)

        if "aim" in raw_output:
            data.add_key("node_embeddings", [raw_output["aim"][:n_real]], level="node")
        return data

    # ------------------------------------------------------------------
    # adapt_input / adapt_output
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Build the flat-padded input dict expected by the AIMNet2 model.

        Handles:

        1. ``AtomicData`` → ``Batch`` promotion.
        2. Gradient enabling on positions when autograd outputs are active.
        3. Collecting positions, numbers, charges, cell from the batch.
        4. Converting the batch's ``neighbor_matrix`` (from
           :class:`NeighborListHook`) to AIMNet2's internal ``nbmat``
           format by appending a padding row.
        5. Running ``mol_flatten`` and ``pad_input`` to produce the
           flat-padded layout the model architecture expects.

        .. note::

            This method does **not** call ``super().adapt_input()``
            because AIMNet2 uses its own input key conventions
            (``coord``, ``numbers``, ``nbmat``) rather than the
            framework's standard keys.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch with positions, atomic_numbers, charge, and
            neighbor_matrix / num_neighbors (from NeighborListHook).

        Returns
        -------
        dict[str, Any]
            Flat-padded dict ready for ``self._calculator.model()``.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        # Enable grad on positions if any autograd output is active.
        if self.model_config.autograd_outputs & self.model_config.active_outputs:
            data.positions.requires_grad_(True)

        N = data.num_nodes
        device = data.positions.device

        # -- Core inputs --
        charge = getattr(data, "charge", None)
        if charge is None:
            charge = torch.zeros(data.num_graphs, dtype=torch.float32, device=device)
        if charge.ndim == 0:
            charge = charge.unsqueeze(0)
        elif charge.ndim > 1:
            charge = charge.squeeze(-1)

        result: dict[str, torch.Tensor] = {
            "coord": data.positions.to(torch.float32),
            "numbers": data.atomic_numbers.to(torch.long),
            "mol_idx": data.batch_idx.to(torch.long),
            "charge": charge.to(torch.float32),
        }

        # -- PBC cell --
        cell = getattr(data, "cell", None)
        if cell is not None:
            result["cell"] = cell.to(torch.float32)

        # -- NSE multiplicity --
        if self._is_nse:
            mult = getattr(data, "mult", None)
            if mult is not None:
                result["mult"] = mult

        # -- Neighbor matrix → AIMNet2 nbmat (with padding row) --
        neighbor_matrix = getattr(data, "neighbor_matrix", None)
        if neighbor_matrix is not None:
            nbmat = neighbor_matrix.to(torch.long)
            # AIMNet2 expects a padding row of fill_value=N appended.
            K = nbmat.shape[1]
            padding_row = torch.full((1, K), N, dtype=torch.long, device=device)
            result["nbmat"] = torch.cat([nbmat, padding_row], dim=0)

            # PBC shifts
            neighbor_matrix_shifts = getattr(data, "neighbor_matrix_shifts", None)
            if neighbor_matrix_shifts is not None:
                shifts_padding = torch.zeros(
                    1, K, 3, dtype=neighbor_matrix_shifts.dtype, device=device
                )
                result["shifts"] = torch.cat(
                    [neighbor_matrix_shifts.to(torch.float32), shifts_padding], dim=0
                )

        # -- mol_flatten (sets _max_mol_size, may reshape 3D→2D) --
        result = self._calculator.mol_flatten(result)

        # -- make_nbmat only if we don't already have external nbmat --
        if result["coord"].ndim == 2:
            if "nbmat" not in result:
                result = self._calculator.make_nbmat(result)
            # pad_input adds padding atom to coord/numbers/mol_idx
            result = self._calculator.pad_input(result)

        return result

    def _strip_padding(
        self,
        raw_output: dict[str, torch.Tensor],
        n_real: int,
    ) -> dict[str, torch.Tensor]:
        """Strip the padding atom from AIMNet2 outputs."""
        for key in self._calculator.atom_feature_keys:
            if key in raw_output and raw_output[key].shape[0] > n_real:
                raw_output[key] = raw_output[key][:n_real]
        for key in ("aim", "spin_charges"):
            if key in raw_output and raw_output[key].shape[0] > n_real:
                raw_output[key] = raw_output[key][:n_real]
        return raw_output

    def adapt_output(
        self, model_output: dict[str, Any], data: AtomicData | Batch
    ) -> ModelOutputs:
        """Map AIMNet2 outputs to nvalchemi standard keys."""
        output: ModelOutputs = OrderedDict()

        energy = model_output.get("energy")
        if energy is not None:
            output["energy"] = energy.unsqueeze(-1) if energy.ndim == 1 else energy

        if "forces" in self.model_config.active_outputs and "forces" in model_output:
            output["forces"] = model_output["forces"]
        if "stress" in self.model_config.active_outputs and "stress" in model_output:
            output["stress"] = model_output["stress"]
        if "charges" in self.model_config.active_outputs:
            output["charges"] = model_output.get("charges")
        if "spin_charges" in self.model_config.active_outputs:
            output["spin_charges"] = model_output.get("spin_charges")

        return output

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the AIMNet2 model and return outputs.

        Energy is always computed as the primitive differentiable output
        via the raw model.  Forces and stresses are derived from energy
        via autograd when requested.

        For stresses, the affine strain trick is applied before the
        forward pass using :func:`~nvalchemi.models._utils.prepare_strain`.
        This scales positions and cell through a displacement tensor so
        that ``dE/d(displacement)`` gives the strain derivative.

        In a pipeline with ``use_autograd=True``, the pipeline handles
        derivative computation externally — it strips forces/stresses
        from ``active_outputs`` so this method only computes energy.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch with positions, atomic_numbers, charge, and
            neighbor_matrix (from NeighborListHook).

        Returns
        -------
        ModelOutputs
            OrderedDict with requested output keys.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        compute_forces = "forces" in (
            self.model_config.active_outputs & self.model_config.outputs
        )
        compute_stresses = "stress" in (
            self.model_config.active_outputs & self.model_config.outputs
        )

        # Set up affine strain BEFORE adapt_input so the scaled positions
        # flow through the model forward pass.
        displacement = None
        orig_positions = None
        orig_cell = None
        if compute_stresses and hasattr(data, "cell") and data.cell is not None:
            scaled_pos, scaled_cell, displacement = prepare_strain(
                data.positions, data.cell, data.batch_idx
            )
            orig_positions = data.positions
            orig_cell = data.cell
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        n_real = data.num_nodes
        model_input = self.adapt_input(data, **kwargs)
        raw_output = self._calculator.model(model_input)
        raw_output = self._strip_padding(raw_output, n_real)

        # Collect results.
        result: dict[str, Any] = {"energy": raw_output["energy"]}

        if "charges" in self.model_config.active_outputs:
            result["charges"] = raw_output.get("charges")
        if "spin_charges" in self.model_config.active_outputs:
            result["spin_charges"] = raw_output.get("spin_charges")

        # Autograd-derived forces.
        if compute_forces:
            energy = result["energy"]
            forces = -torch.autograd.grad(
                energy,
                data.positions,
                grad_outputs=torch.ones_like(energy),
                create_graph=False,
                retain_graph=compute_stresses,
            )[0]
            result["forces"] = forces

        # Autograd-derived stresses via the affine strain trick.
        if compute_stresses and displacement is not None:
            from nvalchemi.models._utils import autograd_stresses

            result["stress"] = autograd_stresses(
                result["energy"],
                displacement,
                orig_cell,
                data.num_graphs,
            )

        # Restore original positions/cell if strain was applied.
        if orig_positions is not None:
            data["positions"] = orig_positions
            data["cell"] = orig_cell

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export the raw AIMNet2 model."""
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if as_state_dict:
            torch.save(raw_model.state_dict(), path)
        else:
            torch.save(raw_model, path)
