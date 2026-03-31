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
from typing import Any

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
    SHORT_RANGE,
    CheckpointInfo,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import neighbor_result_key
from nvalchemi.models.registry import ResolvedArtifact, resolve_known_artifact
from nvalchemi.models.results import CalculatorResults

__all__ = ["MACEPotential"]


def _patch_e3nn_irrep_len_for_compile() -> None:
    """Patch ``e3nn.o3.Irrep.__len__`` for ``torch.compile`` compatibility."""

    from e3nn.o3 import Irrep

    if Irrep.__len__ is not tuple.__len__:
        Irrep.__len__ = tuple.__len__


MACEPotentialCard = MLIPPotentialCard(
    required_inputs=frozenset(
        {
            "positions",
            "atomic_numbers",
            neighbor_result_key("default", "neighbor_list"),
            neighbor_result_key("default", "unit_shifts"),
        }
    ),
    optional_inputs=frozenset({"cell", "pbc"}),
    result_keys=frozenset({"energies"}),
    boundary_modes=frozenset({"non_pbc", "pbc"}),
    neighbor_requirement=NeighborRequirement(
        source="external",
        format="coo",
        name="default",
    ),
    parameterized_by=frozenset({"neighbor_list_name"}),
)


class MACEPotential(Potential):
    """MACE potential using explicit rewrite COO neighbors and composite autograd."""

    card = MACEPotentialCard

    def __init__(
        self,
        model: str | Path | nn.Module,
        *,
        neighbor_list_name: str = "default",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        enable_cueq: bool = True,
        compile_model: bool = False,
        model_card: ModelCard | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise a MACE potential wrapper.

        Parameters
        ----------
        model
            A MACE model specified as a named registry artifact
            (e.g. ``"mace-mp-0b3-medium"``), a local checkpoint path,
            or a pre-instantiated ``nn.Module``.
        neighbor_list_name
            Logical neighbor-list name for result-key namespacing.
        device
            Target execution device.  ``None`` means CPU.
        dtype
            Compute dtype.  ``None`` means use the checkpoint's dtype.
        enable_cueq
            Convert to cuEquivariance format on CUDA devices.
        compile_model
            Apply ``torch.compile`` to the loaded model.
        model_card
            Explicit model metadata.  When ``None`` a default card is
            inferred from the checkpoint provenance.
        name
            Human-readable step name.
        """

        resolved_artifact = self._resolve_known_model(model)
        effective_model: str | Path | nn.Module = (
            resolved_artifact.local_path if resolved_artifact is not None else model
        )
        prepared = self._prepare_model(
            effective_model,
            device=device,
            dtype=dtype,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )
        super().__init__(
            name=name,
            device=prepared["device"],
            required_inputs=frozenset(
                {
                    "positions",
                    "atomic_numbers",
                    neighbor_result_key(neighbor_list_name, "neighbor_list"),
                    neighbor_result_key(neighbor_list_name, "unit_shifts"),
                }
            ),
            neighbor_requirement=NeighborRequirement(
                source="external",
                cutoff=prepared["cutoff"],
                format="coo",
                name=neighbor_list_name,
            ),
        )
        self.neighbor_list_name = neighbor_list_name
        self.model_card = model_card or self._default_model_card(
            model,
            resolved_artifact=resolved_artifact,
        )
        self._model = prepared["model"]
        self.compute_dtype = prepared["compute_dtype"]

        self.register_buffer(
            "_node_emb",
            self._build_node_embedding_table(prepared["atomic_numbers"]),
            persistent=False,
        )

    @staticmethod
    def _resolve_known_model(
        model: str | Path | nn.Module,
    ) -> ResolvedArtifact | None:
        """Resolve one known MACE artifact name when possible."""

        if isinstance(model, nn.Module):
            return None
        model_path = Path(model)
        if model_path.exists():
            return None
        try:
            return resolve_known_artifact(str(model), family="mace")
        except KeyError:
            return None

    @staticmethod
    def _capture_atomic_numbers(model: nn.Module) -> list[int]:
        """Return the model atomic numbers as a Python list."""

        atomic_numbers = getattr(model, "atomic_numbers", None)
        if atomic_numbers is None:
            raise ValueError("MACE model must expose an 'atomic_numbers' attribute.")
        return torch.as_tensor(atomic_numbers, dtype=torch.long).tolist()

    @staticmethod
    def _capture_atomic_energies(model: nn.Module) -> torch.Tensor | None:
        """Return the in-model atomic energies as a float64 tensor."""

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
        """Return the interaction cutoff declared by one MACE model."""

        cutoff = getattr(model, "r_max", None)
        if cutoff is None:
            raise ValueError("MACE model must expose an 'r_max' cutoff attribute.")
        return float(torch.as_tensor(cutoff).item())

    @staticmethod
    def _restore_atomic_energies(
        model: nn.Module,
        atomic_energies: torch.Tensor | None,
    ) -> None:
        """Restore float64 atomic energies into a MACE model."""

        if atomic_energies is None:
            return
        atomic_energies_fn = getattr(model, "atomic_energies_fn", None)
        if atomic_energies_fn is None or not hasattr(
            atomic_energies_fn, "atomic_energies"
        ):
            return
        target_device = atomic_energies_fn.atomic_energies.device
        atomic_energies_fn.atomic_energies = atomic_energies.to(
            device=target_device,
            dtype=torch.float64,
        )

    @staticmethod
    def _convert_to_cueq(model: nn.Module) -> nn.Module:
        """Convert a MACE model to cuEquivariance format."""

        from mace.cli.convert_e3nn_cueq import run as convert_cueq

        return convert_cueq(model)

    @staticmethod
    def _is_cueq_model(model: nn.Module) -> bool:
        """Return whether a MACE model has already been converted to cueq."""

        products = getattr(model, "products", None)
        if not products:
            return False
        symmetric = getattr(products[0], "symmetric_contractions", None)
        return symmetric is not None and hasattr(symmetric, "projection")

    @classmethod
    def _infer_module_dtype(cls, module: nn.Module) -> torch.dtype:
        """Return the dtype of the first parameter in a MACE model."""

        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return torch.float32

    @classmethod
    def _prepare_model(
        cls,
        model: str | Path | nn.Module,
        *,
        device: str | torch.device | None,
        dtype: torch.dtype | None,
        enable_cueq: bool,
        compile_model: bool,
    ) -> dict[str, Any]:
        """Prepare the wrapped model for rewrite inference."""

        if isinstance(model, nn.Module):
            resolved_device = next(model.parameters(), torch.empty(0)).device
            return {
                "model": model,
                "atomic_numbers": cls._capture_atomic_numbers(model),
                "compute_dtype": cls._infer_module_dtype(model),
                "cutoff": cls._capture_cutoff(model),
                "device": resolved_device,
            }

        prepared_model = cls._load_checkpoint_model(model)
        atomic_numbers = cls._capture_atomic_numbers(prepared_model)
        atomic_energies = cls._capture_atomic_energies(prepared_model)
        cutoff = cls._capture_cutoff(prepared_model)

        compute_dtype = dtype or cls._infer_module_dtype(prepared_model)
        if dtype is not None:
            prepared_model = prepared_model.to(dtype=compute_dtype)
        cls._restore_atomic_energies(prepared_model, atomic_energies)

        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        prepared_model = prepared_model.to(target_device)
        cls._restore_atomic_energies(prepared_model, atomic_energies)

        if (
            enable_cueq
            and target_device.type == "cuda"
            and not cls._is_cueq_model(prepared_model)
        ):
            prepared_model = cls._convert_to_cueq(prepared_model)
            cls._restore_atomic_energies(prepared_model, atomic_energies)

        if compile_model:
            _patch_e3nn_irrep_len_for_compile()
            prepared_model = torch.compile(prepared_model)

        prepared_model.eval()
        cls.freeze_parameters(prepared_model)
        return {
            "model": prepared_model,
            "atomic_numbers": atomic_numbers,
            "compute_dtype": compute_dtype,
            "cutoff": cutoff,
            "device": target_device,
        }

    @staticmethod
    def _default_model_card(
        model: str | Path | nn.Module,
        *,
        resolved_artifact: ResolvedArtifact | None = None,
    ) -> ModelCard:
        """Return default metadata for one configured MACE model."""

        checkpoint: CheckpointInfo | None = None
        if isinstance(model, nn.Module):
            model_name = type(model).__name__
        elif resolved_artifact is not None:
            model_name = str(
                resolved_artifact.entry.metadata.get(
                    "model_name",
                    resolved_artifact.entry.name,
                )
            )
            checkpoint = resolved_artifact.checkpoint
        else:
            model_name = str(model)
            checkpoint = CheckpointInfo(
                identifier=model_name,
                source="local_path" if Path(model).exists() else "registry",
            )

        return ModelCard(
            model_family="mace",
            model_name=model_name,
            provided_terms=(PhysicalTerm(kind=SHORT_RANGE),),
            checkpoint=checkpoint,
        )

    @staticmethod
    def _load_checkpoint_model(model: str | Path) -> nn.Module:
        """Load a MACE model from a local path or named foundation checkpoint."""

        model_path = Path(model)
        if model_path.exists():
            loaded = torch.load(model_path, weights_only=False, map_location="cpu")
        else:
            try:
                from mace.calculators.foundations_models import (
                    download_mace_mp_checkpoint,
                )
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "mace-torch is required to load named MACE checkpoints. "
                    "Install it or pass an instantiated nn.Module or local checkpoint path."
                ) from exc

            cached_path = download_mace_mp_checkpoint(str(model))
            loaded = torch.load(cached_path, weights_only=False, map_location="cpu")

        if not isinstance(loaded, nn.Module):
            raise TypeError(
                "MACEPotential expected a serialized nn.Module. "
                "State-dict-only checkpoints are not supported by this wrapper."
            )
        return loaded

    def _build_node_embedding_table(self, atomic_numbers: list[int]) -> torch.Tensor:
        """Build the one-hot atomic-number table expected by MACE."""

        node_emb = torch.zeros(
            max(atomic_numbers) + 1,
            len(atomic_numbers),
            dtype=self.compute_dtype,
            device=self.device,
        )
        for index, atomic_number in enumerate(atomic_numbers):
            node_emb[atomic_number, index] = 1.0
        return node_emb

    def _resolve_cell(
        self,
        batch: Batch,
        ctx: ForwardContext,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the cell tensor expected by MACE."""

        cell, _pbc = self.resolve_periodic_inputs(batch, ctx)
        if cell is None:
            return (
                torch.eye(3, dtype=positions.dtype, device=positions.device)
                .unsqueeze(0)
                .expand(batch.num_graphs, 3, 3)
                .contiguous()
            )

        cell_tensor = torch.as_tensor(
            cell, dtype=positions.dtype, device=positions.device
        )
        if cell_tensor.ndim == 2:
            cell_tensor = cell_tensor.unsqueeze(0)
        if cell_tensor.shape[0] != batch.num_graphs:
            raise ValueError(
                "'cell' must provide one 3x3 matrix per graph. "
                f"Expected {batch.num_graphs}, got {tuple(cell_tensor.shape)}."
            )
        return cell_tensor

    def _resolve_neighbor_inputs(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the named COO neighbor tensors required by MACE."""

        edge_index = torch.as_tensor(
            self.require_input(
                batch,
                neighbor_result_key(self.neighbor_list_name, "neighbor_list"),
                ctx,
            ),
            device=self.device,
            dtype=torch.long,
        )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "MACE requires COO neighbors in shape [2, E]. "
                f"Got {tuple(edge_index.shape)}."
            )

        unit_shifts = torch.as_tensor(
            self.require_input(
                batch,
                neighbor_result_key(self.neighbor_list_name, "unit_shifts"),
                ctx,
            ),
            device=self.device,
            dtype=self.compute_dtype,
        )
        if unit_shifts.ndim != 2 or unit_shifts.shape[1] != 3:
            raise ValueError(
                "MACE requires unit shifts in shape [E, 3]. "
                f"Got {tuple(unit_shifts.shape)}."
            )
        if unit_shifts.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "MACE neighbor inputs must describe the same number of edges. "
                f"Got {edge_index.shape[1]} edges and {unit_shifts.shape[0]} shifts."
            )
        return edge_index, unit_shifts

    def _build_model_inputs(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> dict[str, Any]:
        """Build the native MACE input dictionary from rewrite inputs."""

        model_device = self.device
        positions = torch.as_tensor(
            self.require_input(batch, "positions", ctx),
            device=model_device,
            dtype=self.compute_dtype,
        )
        atomic_numbers = torch.as_tensor(
            self.require_input(batch, "atomic_numbers", ctx),
            device=model_device,
            dtype=torch.long,
        )
        edge_index, unit_shifts = self._resolve_neighbor_inputs(batch, ctx)
        cell = self._resolve_cell(batch, ctx, positions)

        batch_idx = batch.batch.to(device=model_device, dtype=torch.long)
        batch_per_edge = batch_idx.index_select(0, edge_index[0])
        shifts = torch.einsum(
            "eb,ebc->ec", unit_shifts, cell.index_select(0, batch_per_edge)
        )

        return {
            "positions": positions,
            "node_attrs": self._node_emb.index_select(0, atomic_numbers),
            "batch": batch_idx,
            "ptr": batch.ptr.to(device=model_device, dtype=torch.long),
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "cell": cell,
        }

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Evaluate the wrapped MACE model for direct rewrite outputs."""

        model_inputs = self._build_model_inputs(batch, ctx)
        raw_output = self._model.forward(
            model_inputs,
            compute_force=False,
            compute_stress=False,
            compute_displacement=False,
            training=False,
        )

        energies = self.normalize_system_energies(
            torch.as_tensor(raw_output["energy"], device=self.device),
            num_graphs=batch.num_graphs,
            source_name="MACE",
        )
        return self.build_results(ctx, energies=energies)
