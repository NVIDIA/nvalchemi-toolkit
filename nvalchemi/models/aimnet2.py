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
inference.  An ``AIMNet2Calculator`` is constructed internally for its
preprocessing utilities (neighbor list construction, padding, etc.).

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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
)

__all__ = ["AIMNet2Wrapper"]


def __getattr__(name: str):
    """Lazy re-export of AIMNet2Calculator as AIMNet2."""
    if name == "AIMNet2":
        from aimnet.calculators import AIMNet2Calculator

        return AIMNet2Calculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@OptionalDependency.AIMNET.require
class AIMNet2Wrapper(nn.Module, BaseModelMixin):
    """Wrapper for AIMNet2 interatomic potentials.

    Energy is always computed as the primitive differentiable output via
    the raw AIMNet2 model. Forces and stresses are derived from energy
    via autograd. Partial charges and node embeddings (AIM features) are
    taken directly from the model outputs.

    Coulomb and D3 dispersion are disabled inside the calculator. Use
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
        Mutable configuration controlling which outputs are computed.
    model : nn.Module
        The underlying AIMNet2 model.
    calculator : AIMNet2Calculator
        Calculator wrapping the model, used internally for preprocessing
        (neighbor lists, padding, flattening).
    """

    model: nn.Module

    def __init__(self, model: nn.Module) -> None:
        from aimnet.calculators import AIMNet2Calculator

        super().__init__()
        self.model = model

        # Build a calculator around the model for its preprocessing
        # utilities (mol_flatten, make_nbmat, pad_input, unpad_output).
        self.calculator = AIMNet2Calculator(
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
            if "spin_charges" not in self.calculator.keys_out:
                self.calculator.keys_out = [*self.calculator.keys_out, "spin_charges"]

        # Extract cutoff from the loaded model.
        self._cutoff = self._extract_cutoff(raw_model)

        # Build the model config with capability fields.
        outputs = {"energy", "forces", "stress", "charges"}
        if self._is_nse:
            outputs.add("spin_charges")

        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charge"}),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=None,  # AIMNet2 manages its own neighbor list
            active_outputs=outputs,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
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

        Returns
        -------
        AIMNet2Wrapper

        Raises
        ------
        ImportError
            If the ``aimnet`` package is not installed.
        """
        from aimnet.calculators import AIMNet2Calculator

        # Use the calculator to resolve aliases and download checkpoints,
        # then extract the raw model.
        calc = AIMNet2Calculator(
            model=str(checkpoint_path),
            device=str(device),
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=False,
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
        # AIMNet2 AIM features are typically 256-dimensional.
        aim_dim = 256
        aev = getattr(raw_model, "aev", None)
        if aev is not None:
            # Try to get the actual dimension from the model.
            output_size = getattr(aev, "output_size", None)
            if output_size is not None:
                aim_dim = int(output_size)
        return {"node_embeddings": (aim_dim,)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute AIMNet2 AIM feature embeddings and attach to data.

        Writes ``node_embeddings`` into *data* in-place and returns it.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data.

        Returns
        -------
        AtomicData | Batch
            Data with ``node_embeddings`` attached.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        calc_input = self._build_calculator_input(data)
        model_input, used_flat_padding = self._prepare_model_input(calc_input)
        with torch.no_grad():
            raw_output = self.calculator.model(model_input)
        if used_flat_padding:
            raw_output = self._unpad_outputs(raw_output, model_input)
        if "aim" in raw_output:
            data.node_embeddings = raw_output["aim"]
        return data

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def _build_calculator_input(self, data: Batch) -> dict[str, torch.Tensor]:
        """Build a flat dict in AIMNet2 key conventions from a Batch.

        Parameters
        ----------
        data : Batch
            Input batch with positions, atomic_numbers, and charge.

        Returns
        -------
        dict[str, torch.Tensor]
            Flat tensors keyed by AIMNet2 names.
        """
        # AIMNet2 expects 'charge' as a system-level tensor.
        charge = getattr(data, "charge", None)
        if charge is None:
            # Default to neutral system (charge=0 for each graph).
            charge = torch.zeros(
                data.num_graphs, dtype=torch.float32, device=data.positions.device
            )
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

        # Optional PBC inputs.
        cell = getattr(data, "cell", None)
        if cell is not None:
            result["cell"] = cell.to(torch.float32)

        # NSE models may use multiplicity.
        if self._is_nse:
            mult = getattr(data, "mult", None)
            if mult is not None:
                result["mult"] = mult

        return result

    def _prepare_model_input(
        self, calc_input: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Run the calculator's flat preprocessing pipeline.

        Calls ``mol_flatten`` -> ``make_nbmat`` -> ``pad_input`` to
        produce the flat-padded format with neighbor lists that the raw
        AIMNet2 model expects. Does **not** call ``to_input_tensors``
        (which detaches) so the autograd graph is preserved.

        Returns
        -------
        tuple[dict[str, torch.Tensor], bool]
            ``(model_input, used_flat_padding)``
        """
        data = self.calculator.mol_flatten(calc_input)
        used_flat_padding = False
        if data["coord"].ndim == 2:
            if "nbmat" not in data:
                data = self.calculator.make_nbmat(data)
            data = self.calculator.pad_input(data)
            used_flat_padding = True
        return data, used_flat_padding

    def _unpad_outputs(
        self,
        raw_output: dict[str, torch.Tensor],
        model_input: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Strip trailing padding from flat-padded AIMNet outputs."""
        raw_output = self.calculator.unpad_output(raw_output)
        n_atoms = model_input["nbmat"].shape[0] - 1
        for key in ("aim", "spin_charges"):
            if key in raw_output and raw_output[key].shape[0] > n_atoms:
                raw_output[key] = raw_output[key][:n_atoms]
        return raw_output

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Prepare inputs for AIMNet2 with gradient setup from base class.

        Calls ``super().adapt_input()`` to enable gradients on required
        tensors and validate required inputs, then wraps the batch for
        downstream processing by ``forward()``.

        AIMNet2 manages its own neighbor list internally, so
        ``neighbor_config`` is ``None`` and no neighbor-list keys are
        collected by the base class.

        Parameters
        ----------
        data : AtomicData | Batch
            Input atomic structure.

        Returns
        -------
        dict[str, Any]
            Dict with ``"data"`` key containing the preprocessed Batch.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        # Base class enables requires_grad on autograd_inputs and
        # validates that required_inputs (e.g. "charge") are present.
        super().adapt_input(data, **kwargs)

        return {"data": data}

    def adapt_output(
        self, model_output: dict[str, Any], data: AtomicData | Batch
    ) -> ModelOutputs:
        """Map AIMNet2 outputs to nvalchemi standard keys.

        Calls ``super().adapt_output()`` to create the initial
        ``ModelOutputs`` OrderedDict, then populates it with
        AIMNet2-specific output mapping.

        Parameters
        ----------
        model_output : dict[str, Any]
            Raw output dict from the AIMNet2 model and autograd.
        data : AtomicData | Batch
            Original input data.

        Returns
        -------
        ModelOutputs
            Standardized output dict.
        """
        output = super().adapt_output(model_output, data)

        # Energy (always present, base auto-maps if key matches).
        energy = model_output.get("energy")
        if energy is not None:
            if energy.ndim == 1:
                energy = energy.unsqueeze(-1)
            output["energy"] = energy

        # Forces and stresses (autograd-derived, set in forward()).
        if "forces" in model_output and "forces" in output:
            output["forces"] = model_output["forces"]
        if "stress" in model_output and "stress" in output:
            output["stress"] = model_output["stress"]

        # Charges (direct model output).
        if "charges" in output:
            output["charges"] = model_output.get("charges")

        # Spin charges (NSE models only).
        if "spin_charges" in output:
            output["spin_charges"] = model_output.get("spin_charges")

        return output

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the AIMNet2 model and return outputs.

        Energy is always computed as the primitive differentiable output
        via the raw model. Forces and stresses are derived from energy
        via autograd. Charges and embeddings are taken directly from the
        model.

        .. note::

            This wrapper is currently **inference-only**.  Autograd forces
            use ``create_graph=False``, so higher-order gradients needed
            for training are not available.  Training support requires
            adapting AIMNet2's internal preprocessing to preserve the
            computation graph and is planned for a future release.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch with positions, atomic_numbers, and optionally
            charge, cell, pbc.

        Returns
        -------
        ModelOutputs
            OrderedDict with requested output keys.
        """
        inp = self.adapt_input(data, **kwargs)
        batch = inp["data"]

        # Build calculator input and run model.
        calc_input = self._build_calculator_input(batch)
        model_input, used_flat_padding = self._prepare_model_input(calc_input)
        raw_output = self.calculator.model(model_input)
        if used_flat_padding:
            raw_output = self._unpad_outputs(raw_output, model_input)

        # Collect results.
        result: dict[str, Any] = {"energy": raw_output["energy"]}

        # Charges (direct output).
        if "charges" in self.model_config.active_outputs:
            result["charges"] = raw_output.get("charges")

        # Spin charges (NSE only).
        if "spin_charges" in self.model_config.active_outputs:
            result["spin_charges"] = raw_output.get("spin_charges")

        # Autograd-derived forces.
        compute_forces = "forces" in (
            self.model_config.active_outputs & self.model_config.outputs
        )
        compute_stresses = "stress" in (
            self.model_config.active_outputs & self.model_config.outputs
        )

        if compute_forces:
            energy = result["energy"]
            forces = -torch.autograd.grad(
                energy,
                batch.positions,
                grad_outputs=torch.ones_like(energy),
                create_graph=False,
                retain_graph=compute_stresses,
            )[0]
            result["forces"] = forces

        if compute_stresses:
            # Stresses require a displacement tensor — not yet implemented
            # for standalone AIMNet2. Available via PipelineModelWrapper
            # with autograd groups.
            pass

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export the raw AIMNet2 model.

        Parameters
        ----------
        path : Path
            Destination path.
        as_state_dict : bool, optional
            If ``True``, save only the ``state_dict``.
        """
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if as_state_dict:
            torch.save(raw_model.state_dict(), path)
        else:
            torch.save(raw_model, path)
