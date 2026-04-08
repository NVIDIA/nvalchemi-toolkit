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
"""MACE model wrapper.

Wraps any MACE model (``MACE``, ``ScaleShiftMACE``, cuEq-converted models,
``torch.compile``-d models, etc.) as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible wrapper for use in
composable model execution.

Usage
-----
Load a named foundation-model checkpoint::

    model = MACEWrapper("medium-0b2")

Or wrap a local checkpoint path::

    model = MACEWrapper("/path/to/mace.pt")

Or wrap an already-instantiated model::

    import torch

    mace_model = torch.load("my_mace.pt", weights_only=False)
    model = MACEWrapper(mace_model)

Notes
-----
* Forces are computed **conservatively** via autograd
  inside the composable runtime, so ``spec.use_autograd`` is ``True``.
* ``node_attrs`` (one-hot atomic-number encodings) are computed via a
  pre-built GPU lookup table — no CPU round-trips per step.
* The wrapper expects externally prepared COO neighbor data
  (``edge_index`` and ``unit_shifts``).
* **dtype conversion**: when ``dtype`` is specified, model parameters are
  cast to the requested precision.  Atomic energies are preserved in
  ``float64`` to improve numeric stability of simulations.
* **cuEquivariance**: when ``enable_cueq=True`` (default), the wrapper
  attempts to convert the model to a cuEquivariance-accelerated form
  using ``mace.cli.convert_e3nn_cueq``.  Conversion is silently skipped
  when the ``cuequivariance`` package is not installed.
* **torch.compile**: when ``compile_model=True``, the backend model is
  compiled with ``torch.compile(model)`` after all other
  preparation steps.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.models.utils import (
    _UNSET,
    build_model_repr,
    collect_nondefault_repr_kwargs,
    initialize_model_repr,
    mapping_get,
    replace_model_spec,
)

__all__ = ["MACEWrapper"]


def _patch_e3nn_irrep_len_for_compile() -> None:
    """Patch ``e3nn.o3.Irrep.__len__`` for ``torch.compile`` compatibility."""

    from e3nn.o3 import Irrep

    if Irrep.__len__ is not tuple.__len__:
        Irrep.__len__ = tuple.__len__


_MACEModelConfig = ModelConfig(
    required_inputs=frozenset(
        {"positions", "atomic_numbers", "edge_index", "unit_shifts"}
    ),
    optional_inputs=frozenset({"cell", "pbc", "batch"}),
    outputs=frozenset({"energies"}),
    additive_outputs=frozenset({"energies"}),
    use_autograd=True,
    pbc_mode="any",
    neighbor_config=NeighborConfig(
        source="external",
        cutoff=6.0,
        format="coo",
        half_list=False,
    ),
)


class MACEWrapper(nn.Module, BaseModelMixin):
    """Wrapper for any MACE model implementing the :class:`BaseModelMixin` interface.

    Accepts any MACE model variant (``MACE``, ``ScaleShiftMACE``, cuEq-converted
    models, ``torch.compile``-d models, etc.).  The wrapper handles:

    * One-hot ``node_attrs`` encoding via a pre-built GPU lookup table
      (no CPU round-trip per step).
    * COO-format neighbor data (``edge_index``, ``unit_shifts``) expected from
      the composable runtime's :class:`NeighborListBuilder`.

    Parameters
    ----------
    model : str | Path | nn.Module
        Upstream MACE model name (e.g. ``"medium-0b2"``), local checkpoint
        path, or an already-instantiated MACE module.
    device : str | torch.device | None
        Execution device for the wrapped model.
    dtype : torch.dtype | None
        Optional compute dtype override.
    enable_cueq : bool
        Whether to enable cuEquivariance conversion when supported.
    compile_model : bool
        Whether to compile the backend model with ``torch.compile``.
    cutoff : float | None
        Optional neighbor cutoff override.  When omitted the cutoff is read
        from ``model.r_max``.
    pbc_mode : str | None
        Optional override for the periodic-boundary support contract.
    name : str | None
        Optional stable display name used by the composable runtime.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract describing inputs, outputs, autograd participation,
        and neighbor requirements.
    """

    spec = _MACEModelConfig

    def __init__(
        self,
        model: str | Path | nn.Module,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        enable_cueq: bool = True,
        compile_model: bool = False,
        cutoff: float = _UNSET,
        pbc_mode: str | None = _UNSET,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        model_label: str | None = None
        if isinstance(model, Path):
            model_label = str(model)
        elif isinstance(model, str):
            model_label = model

        prepared = self._prepare_model(
            model,
            device=device,
            dtype=dtype,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )
        self._model = prepared["model"]
        self._device = prepared["device"]
        self.compute_dtype = prepared["compute_dtype"]

        resolved_cutoff = prepared["cutoff"]
        if cutoff is not _UNSET:
            resolved_cutoff = float(cutoff)
        self.spec = replace_model_spec(
            _MACEModelConfig,
            pbc_mode=_MACEModelConfig.pbc_mode if pbc_mode is _UNSET else pbc_mode,
            neighbor_cutoff=resolved_cutoff,
        )

        node_emb = self._build_node_embedding_table(prepared["atomic_numbers"]).to(
            device=self._device,
            dtype=self.compute_dtype,
        )
        self.register_buffer(
            "_node_emb",
            node_emb,
            persistent=False,
        )
        static_kwargs: dict[str, object] = {}
        if model_label is not None:
            static_kwargs["model"] = model_label
        static_kwargs.update(
            collect_nondefault_repr_kwargs(
                explicit_values={
                    key: value
                    for key, value in {
                        "device": device,
                        "dtype": dtype,
                        "enable_cueq": enable_cueq,
                        "compile_model": compile_model,
                        "cutoff": cutoff,
                        "pbc_mode": pbc_mode,
                        "name": name,
                    }.items()
                    if value is not _UNSET
                },
                defaults={
                    "device": None,
                    "dtype": None,
                    "enable_cueq": True,
                    "compile_model": False,
                    "name": None,
                },
                order=(
                    "device",
                    "dtype",
                    "enable_cueq",
                    "compile_model",
                    "cutoff",
                    "pbc_mode",
                    "name",
                ),
            )
        )
        initialize_model_repr(
            self,
            static_kwargs=static_kwargs,
            kwarg_order=(
                "model",
                "device",
                "dtype",
                "enable_cueq",
                "compile_model",
                "cutoff",
                "pbc_mode",
                "name",
            ),
        )

    @property
    def device(self) -> torch.device:
        """Return the current execution device."""

        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the current compute dtype."""

        return self.compute_dtype

    def __repr__(self) -> str:
        """Return a compact constructor-style representation."""

        return build_model_repr(self)

    @classmethod
    def load_checkpoint(cls, path: Path) -> nn.Module:
        """Load a serialized MACE checkpoint from a local path.

        Parameters
        ----------
        path
            Path to a ``torch.save``-produced MACE checkpoint.

        Returns
        -------
        nn.Module
            Loaded MACE module.

        Raises
        ------
        TypeError
            If the checkpoint does not deserialize to an ``nn.Module``.
        """

        loaded = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(loaded, nn.Module):
            raise TypeError(
                "MACEWrapper expected a serialized nn.Module. "
                "State-dict-only checkpoints are not supported by this wrapper."
            )
        return loaded

    @staticmethod
    def _unwrap_loaded_model(model: object) -> nn.Module:
        """Unwrap calculator-style results and validate the loaded model."""

        if isinstance(model, nn.Module):
            return model
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, nn.Module):
            return inner_model
        raise TypeError(
            "MACEWrapper expected a loaded nn.Module or calculator-style object with "
            "a '.model' nn.Module."
        )

    @staticmethod
    def _capture_atomic_numbers(model: nn.Module) -> list[int]:
        """Return the atomic numbers supported by one loaded MACE model."""

        atomic_numbers = getattr(model, "atomic_numbers", None)
        if atomic_numbers is None:
            raise ValueError("MACE model must expose an 'atomic_numbers' attribute.")
        return torch.as_tensor(atomic_numbers, dtype=torch.long).tolist()

    @staticmethod
    def _capture_atomic_energies(model: nn.Module) -> torch.Tensor | None:
        """Return float64 atomic energies from one loaded MACE model, if present."""

        atomic_energies_fn = getattr(model, "atomic_energies_fn", None)
        if atomic_energies_fn is None or not hasattr(
            atomic_energies_fn, "atomic_energies"
        ):
            return None
        return torch.as_tensor(
            atomic_energies_fn.atomic_energies,
            dtype=torch.float64,
        ).clone()

    @staticmethod
    def _capture_cutoff(model: nn.Module) -> float:
        """Return the interaction cutoff declared by one loaded MACE model."""

        cutoff = getattr(model, "r_max", None)
        if cutoff is None:
            raise ValueError("MACE model does not expose an 'r_max' cutoff attribute.")
        return float(torch.as_tensor(cutoff).item())

    @staticmethod
    def _restore_atomic_energies(
        model: nn.Module,
        atomic_energies: torch.Tensor | None,
    ) -> None:
        """Restore float64 atomic energies after dtype/device conversion."""

        if atomic_energies is None:
            return
        atomic_energies_fn = getattr(model, "atomic_energies_fn", None)
        if atomic_energies_fn is None or not hasattr(
            atomic_energies_fn, "atomic_energies"
        ):
            return
        atomic_energies_fn.atomic_energies = atomic_energies.to(
            device=atomic_energies_fn.atomic_energies.device,
            dtype=torch.float64,
        )

    @staticmethod
    def _is_cueq_model(model: nn.Module) -> bool:
        """Return whether the model appears to use cuEquivariance blocks."""

        for group_name in ("interactions", "products"):
            group = getattr(model, group_name, None)
            if group is None:
                continue
            for block in group:
                config = getattr(block, "cueq_config", None)
                if getattr(config, "enabled", False):
                    return True
        return False

    @staticmethod
    def _convert_to_cueq(model: nn.Module) -> nn.Module:
        """Convert one MACE model to cuEquivariance when the backend is available."""

        try:
            from mace.cli.convert_e3nn_cueq import run
        except ImportError:  # pragma: no cover - optional dependency
            warnings.warn(
                "cuEquivariance conversion requested but mace.cli.convert_e3nn_cueq "
                "is unavailable; using the original MACE model.",
                UserWarning,
                stacklevel=2,
            )
            return model

        try:
            return run(model)
        except Exception as exc:  # pragma: no cover - backend-specific
            warnings.warn(
                f"Failed to convert the MACE model to cuEquivariance: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return model

    @classmethod
    def _prepare_loaded_model(
        cls,
        model: nn.Module,
        *,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        enable_cueq: bool,
        compile_model: bool,
    ) -> dict[str, object]:
        """Normalize one already loaded MACE model for execution."""

        atomic_numbers = cls._capture_atomic_numbers(model)
        atomic_energies = cls._capture_atomic_energies(model)
        try:
            cutoff = cls._capture_cutoff(model)
        except ValueError as exc:
            warnings.warn(str(exc), UserWarning, stacklevel=2)
            cutoff = _MACEModelConfig.neighbor_config.cutoff

        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        model = model.to(device=target_device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        cls._restore_atomic_energies(model, atomic_energies)

        if (
            enable_cueq
            and target_device.type == "cuda"
            and not cls._is_cueq_model(model)
        ):
            model = cls._convert_to_cueq(model)
            if not cls._is_cueq_model(model):
                warnings.warn(
                    "cuEquivariance conversion returned a model without cueq "
                    "acceleration enabled; continuing with the returned model.",
                    UserWarning,
                    stacklevel=2,
                )

        if compile_model:
            _patch_e3nn_irrep_len_for_compile()
            model = torch.compile(model)

        compute_dtype = next(
            (
                parameter.dtype
                for parameter in model.parameters()
                if parameter.dtype.is_floating_point
            ),
            dtype or torch.float32,
        )
        return {
            "model": model,
            "cutoff": cutoff,
            "atomic_numbers": atomic_numbers,
            "compute_dtype": compute_dtype,
            "device": target_device,
        }

    @classmethod
    def _prepare_model(
        cls,
        model: str | Path | nn.Module,
        *,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        enable_cueq: bool,
        compile_model: bool,
    ) -> dict[str, object]:
        """Load and normalize a MACE model from a path, name, or module.

        Parameters
        ----------
        model
            Upstream model name, local checkpoint path, or instantiated MACE
            module.
        device
            Target execution device.
        dtype
            Optional floating-point dtype override.
        enable_cueq
            Whether cuEquivariance conversion should be attempted.
        compile_model
            Whether the loaded model should be passed through
            :func:`torch.compile`.

        Returns
        -------
        dict[str, object]
            Normalized model payload with the loaded module, cutoff, supported
            atomic numbers, device, and compute dtype.
        """

        if isinstance(model, nn.Module):
            loaded = model
        elif isinstance(model, Path) or Path(model).exists():
            loaded = cls.load_checkpoint(Path(model))
        else:
            try:
                from mace.calculators import mace_mp
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "mace is required to load named MACE models. "
                    "Install mace-torch or pass a local checkpoint path / nn.Module."
                ) from exc
            loaded = mace_mp(
                model=str(model), device=str(device) if device is not None else "cpu"
            )

        return cls._prepare_loaded_model(
            cls._unwrap_loaded_model(loaded),
            device=device,
            dtype=dtype,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )

    @staticmethod
    def _build_node_embedding_table(atomic_numbers: list[int]) -> torch.Tensor:
        """Build one simple one-hot table indexed by atomic number."""

        max_z = max(atomic_numbers) if atomic_numbers else 0
        table = torch.zeros(max_z + 1, len(atomic_numbers), dtype=torch.float32)
        for idx, atomic_number in enumerate(atomic_numbers):
            table[atomic_number, idx] = 1.0
        return table

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        """Move wrapper metadata together with module parameters and buffers."""

        result = super()._apply(fn)
        reference = None
        for parameter in self.parameters():
            reference = parameter
            break
        if reference is None:
            for buffer in self.buffers():
                reference = buffer
                break
        if reference is not None:
            self._device = reference.device
            if reference.dtype.is_floating_point:
                self.compute_dtype = reference.dtype
        return result

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the wrapped MACE model and return per-system energies.

        Parameters
        ----------
        data
            Prepared input mapping resolved by the composable runtime. Required
            keys are ``positions``, ``atomic_numbers``, ``edge_index``, and
            ``unit_shifts``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping with one ``"energies"`` tensor of shape ``(B, 1)``.

        Raises
        ------
        ValueError
            If the backend model does not publish an energy-like output.
        """

        positions = data["positions"].to(device=self.device, dtype=self.compute_dtype)
        atomic_numbers = data["atomic_numbers"].long().to(device=self.device)
        edge_index = data["edge_index"].to(device=self.device)
        if edge_index.ndim == 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        unit_shifts = data["unit_shifts"].to(
            device=self.device, dtype=self.compute_dtype
        )
        batch_idx = (
            mapping_get(
                data,
                "batch",
                torch.zeros(positions.shape[0], dtype=torch.long, device=self.device),
            )
            .long()
            .to(device=self.device)
        )
        ptr = mapping_get(data, "ptr")
        if ptr is None:
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 1
            counts = torch.bincount(batch_idx, minlength=num_graphs)
            ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    counts.cumsum(0),
                ]
            )
        else:
            ptr = ptr.long().to(device=self.device)

        model_input = {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "node_attrs": self._node_emb.index_select(0, atomic_numbers),
            "batch": batch_idx,
            "ptr": ptr,
            "edge_index": edge_index.long(),
            "unit_shifts": unit_shifts,
        }
        if "cell" in data:
            model_input["cell"] = data["cell"].to(
                device=self.device, dtype=self.compute_dtype
            )

        raw = self._model(model_input)
        if "energy" in raw:
            energies = raw["energy"]
        elif "energies" in raw:
            energies = raw["energies"]
        elif "node_energy" in raw:
            node_energy = raw["node_energy"]
            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 1
            energies = torch.zeros(
                num_graphs,
                dtype=node_energy.dtype,
                device=node_energy.device,
            )
            energies.scatter_add_(0, batch_idx, node_energy.reshape(-1))
        else:
            raise ValueError(
                "MACEWrapper expected backend output with 'energy', 'energies', or "
                "'node_energy'."
            )
        if energies.ndim == 1:
            energies = energies.unsqueeze(-1)
        return {"energies": energies}
