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

Wraps AIMNetCentral's :class:`AIMNet2Calculator` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model wrapper,
ready for use in :class:`~nvalchemi.models.composable.ComposableModelWrapper`
or direct single-model composable execution.

Usage
-----
Load a named AIMNetCentral model directly::

    model = AIMNet2Wrapper("aimnet2")

Or wrap a local checkpoint path::

    model = AIMNet2Wrapper("/path/to/aimnet2.ckpt")

Notes
-----
* Named-string resolution is delegated directly to AIMNetCentral.
* The wrapper publishes ``energies`` and atomic ``node_charges``.
* Open-shell AIMNet-NSE models automatically extend the optional input
  contract to include ``graph_spins``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.utils import (
    build_model_repr,
    collect_nondefault_repr_kwargs,
    initialize_model_repr,
    mapping_get,
)

__all__ = ["AIMNet2Wrapper"]

AIMNet2ModelConfig = ModelConfig(
    required_inputs=frozenset({"positions", "atomic_numbers"}),
    optional_inputs=frozenset({"cell", "pbc", "graph_charges", "batch"}),
    outputs=frozenset({"energies", "node_charges"}),
    additive_outputs=frozenset({"energies"}),
    use_autograd=True,
    autograd_outputs=frozenset({"node_charges"}),
    pbc_mode="any",
)

_AIMNET_OPTIONAL_INPUTS_WITH_SPIN = frozenset(
    {"cell", "pbc", "graph_charges", "graph_spins", "batch"}
)


class AIMNet2Wrapper(nn.Module, BaseModelMixin):
    """Wrapper for AIMNet2 models loaded through AIMNetCentral.

    Parameters
    ----------
    model : str | Path | nn.Module
        Upstream AIMNetCentral model name, local checkpoint path, or an
        already-instantiated AIMNet module.
    device : str | torch.device | None
        Execution device passed through to AIMNetCentral.
    compile_model : bool
        Whether AIMNetCentral should compile the backend model.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract.  For NSE (open-shell) models the optional
        inputs are extended to include ``graph_spins``.
    """

    spec = AIMNet2ModelConfig

    def __init__(
        self,
        model: str | Path | nn.Module,
        *,
        device: str | torch.device | None = "cuda",
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        model_label: str | None = None
        if isinstance(model, Path):
            model_label = str(model)
        elif isinstance(model, str):
            model_label = model
        try:
            from aimnet.calculators import AIMNet2Calculator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "aimnet is required to use AIMNet2Wrapper. "
                "Install AIMNetCentral or make it importable on PYTHONPATH."
            ) from exc

        model_arg: str | nn.Module
        if isinstance(model, Path):
            model_arg = str(model)
        elif isinstance(model, str):
            path = Path(model)
            model_arg = str(path) if path.exists() else model
        else:
            model_arg = model

        self._calculator = AIMNet2Calculator(
            model=model_arg,
            device=str(device) if device is not None else None,
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=compile_model,
            train=False,
        )
        self._model = self._calculator.model
        self._device = torch.device(self._calculator.device)
        raw_model = (
            self._model._orig_mod if hasattr(self._model, "_orig_mod") else self._model
        )
        self._is_nse = getattr(raw_model, "num_charge_channels", 1) == 2
        if self._is_nse:
            self.spec = ModelConfig(
                required_inputs=AIMNet2ModelConfig.required_inputs,
                optional_inputs=_AIMNET_OPTIONAL_INPUTS_WITH_SPIN,
                outputs=AIMNet2ModelConfig.outputs,
                additive_outputs=AIMNet2ModelConfig.additive_outputs,
                use_autograd=AIMNet2ModelConfig.use_autograd,
                autograd_outputs=AIMNet2ModelConfig.autograd_outputs,
                pbc_mode=AIMNet2ModelConfig.pbc_mode,
                neighbor_config=AIMNet2ModelConfig.neighbor_config,
            )
        static_kwargs: dict[str, object] = {}
        if model_label is not None:
            static_kwargs["model"] = model_label
        static_kwargs.update(
            collect_nondefault_repr_kwargs(
                explicit_values={
                    "device": device,
                    "compile_model": compile_model,
                },
                defaults={
                    "device": "cuda",
                    "compile_model": True,
                },
                order=("device", "compile_model"),
            )
        )
        initialize_model_repr(
            self,
            static_kwargs=static_kwargs,
            kwarg_order=("model", "device", "compile_model"),
        )

    @property
    def device(self) -> torch.device:
        """Return the current execution device.

        Returns
        -------
        torch.device
            Device used by the wrapped AIMNet model.
        """

        return self._device

    def __repr__(self) -> str:
        """Return a compact constructor-style representation."""

        return build_model_repr(self)

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        """Move the wrapper and keep calculator-side device metadata aligned."""

        result = super()._apply(fn)
        reference = None
        for parameter in self.parameters():
            reference = parameter
            break
        if reference is None:
            for buffer in self.buffers():
                reference = buffer
                break
        if reference is None:
            reference = fn(torch.empty(0, device=self._device))
        self._device = reference.device
        self._calculator.device = str(reference.device)
        self._calculator.model = self._model
        return result

    def _build_flat_input(
        self,
        data: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build the canonical flat AIMNet input from pipeline data."""

        positions = torch.as_tensor(
            data["positions"],
            device=self.device,
            dtype=torch.float32,
        )
        atomic_numbers = torch.as_tensor(
            data["atomic_numbers"],
            device=self.device,
            dtype=torch.long,
        )
        batch_idx = torch.as_tensor(
            mapping_get(
                data,
                "batch",
                torch.zeros(
                    positions.shape[0], dtype=torch.long, device=positions.device
                ),
            ),
            device=self.device,
            dtype=torch.long,
        )

        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 1
        model_input: dict[str, torch.Tensor] = {
            "coord": positions,
            "numbers": atomic_numbers,
            "mol_idx": batch_idx,
            "charge": torch.as_tensor(
                mapping_get(
                    data,
                    "graph_charges",
                    torch.zeros(
                        num_graphs, device=positions.device, dtype=torch.float32
                    ),
                ),
                device=self.device,
                dtype=torch.float32,
            ).reshape(num_graphs),
        }

        if self._is_nse:
            model_input["mult"] = torch.as_tensor(
                mapping_get(
                    data,
                    "graph_spins",
                    torch.ones(
                        num_graphs, device=positions.device, dtype=torch.float32
                    ),
                ),
                device=self.device,
                dtype=torch.float32,
            ).reshape(num_graphs)

        cell = mapping_get(data, "cell")
        if cell is not None:
            model_input["cell"] = torch.as_tensor(
                cell, device=self.device, dtype=torch.float32
            )

        return model_input

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run AIMNet2 on one prepared composable input mapping.

        Parameters
        ----------
        data
            Prepared input mapping resolved by the composable runtime.

        Returns
        -------
        dict[str, torch.Tensor]
            Output mapping containing ``energies`` and ``node_charges``.

        Raises
        ------
        ValueError
            If the AIMNet backend does not provide the expected output keys.
        """

        model_input = self._build_flat_input(data)
        model_input = self._calculator.mol_flatten(model_input)
        model_input = self._calculator.make_nbmat(model_input)
        model_input = self._calculator.pad_input(model_input)
        raw_output = self._model(model_input)
        raw_output = self._calculator.unpad_output(raw_output)

        if "charges" not in raw_output:
            raise ValueError("AIMNet2Wrapper expected 'charges' in backend output.")

        energy = raw_output.get("energy")
        if energy is None:
            raise ValueError("AIMNet2Wrapper expected 'energy' in backend output.")
        energies = energy.unsqueeze(-1) if energy.ndim == 1 else energy
        node_charges = raw_output["charges"]
        if node_charges.ndim == 1:
            node_charges = node_charges.unsqueeze(-1)
        return {"energies": energies, "node_charges": node_charges}
