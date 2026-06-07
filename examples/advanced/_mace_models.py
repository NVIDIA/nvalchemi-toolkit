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
"""MACE model builders for training.

``build_vanilla_mace_model`` wraps ScaleShiftMACE for energy, forces, and stress.

``ChargedMACE`` adds charge and weight readout heads on final node features and
equilibrates charges to each graph's ``total_charge``. ``ChargedMACEWrapper``
adapts inputs/outputs for the nvalchemi pipeline.

``build_charged_mace_model`` combines ChargedMACE with nvalchemi's Ewald model,
and optionally subtracts a switched short-range point-charge Coulomb term in an
autograd pipeline so electrostatic energy contributes to forces and stress.

``build_training_mace_model`` is the checkpoint-reconstructable factory used by
the advanced MACE training recipe.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

torch.serialization.add_safe_globals([slice])

from e3nn import o3
from e3nn.util.jit import compile_mode
from mace.modules import (
    RealAgnosticResidualInteractionBlock,
    ScaleShiftMACE,
)
from mace.modules.blocks import NonLinearReadoutBlock
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scatter import scatter_sum
from omegaconf import DictConfig, OmegaConf

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.ewald import EwaldModelWrapper
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.models.pipeline import (
    PipelineGroup,
    PipelineModelWrapper,
)
from nvalchemi.training import create_model_spec


def build_cueq_config(model_config: DictConfig) -> CuEquivarianceConfig | None:
    """Build the optional cuEquivariance config for MACE modules."""
    cueq_cfg = model_config.get("cueq", None)
    if cueq_cfg is None:
        return None
    if not bool(cueq_cfg.get("enabled", False)):
        return None
    kwargs = {
        "enabled": True,
        "layout": str(cueq_cfg.get("layout", "mul_ir")),
        "group": str(cueq_cfg.get("group", "O3")),
        "optimize_all": bool(cueq_cfg.get("optimize_all", False)),
        "optimize_linear": bool(cueq_cfg.get("optimize_linear", False)),
        "optimize_channelwise": bool(cueq_cfg.get("optimize_channelwise", False)),
        "optimize_symmetric": bool(cueq_cfg.get("optimize_symmetric", False)),
        "optimize_fctp": bool(cueq_cfg.get("optimize_fctp", False)),
        "conv_fusion": bool(cueq_cfg.get("conv_fusion", False)),
    }
    return CuEquivarianceConfig(**kwargs)


def _attach_checkpoint_spec(
    model: torch.nn.Module,
    *,
    model_type: str,
    atomic_numbers: Sequence[int],
    atomic_energies: Sequence[float],
    r_max: float,
    avg_num_neighbors: float,
    model_config: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
    active_outputs: Sequence[str],
) -> torch.nn.Module:
    """Attach an explicit training-recipe reconstruction spec to ``model``."""
    checkpoint_spec = create_model_spec(
        build_training_mace_model,
        model_type=model_type,
        atomic_numbers=list(atomic_numbers),
        atomic_energies=[float(value) for value in atomic_energies],
        r_max=float(r_max),
        avg_num_neighbors=float(avg_num_neighbors),
        model_config=model_config,
        dtype=dtype,
        device=device,
        active_outputs=sorted(active_outputs),
    )

    def _checkpoint_spec() -> Any:
        return checkpoint_spec

    model.checkpoint_spec = _checkpoint_spec  # type: ignore[attr-defined]
    return model


def get_scale_shift_config(model_config: DictConfig) -> dict[str, Any]:
    """Return ScaleShiftMACE kwargs using MACE's force-RMS scaling convention."""
    return {
        "atomic_inter_scale": model_config.get(
            "atomic_inter_scale",
            model_config.get(
                "std",
                model_config.get(
                    "forces_rms",
                    model_config.get("force_rms", model_config.get("rms_forces", 1.0)),
                ),
            ),
        ),
        "atomic_inter_shift": model_config.get(
            "atomic_inter_shift",
            model_config.get("mean", 0.0),
        ),
    }


def _exp_charge_fractions(
    raw_weights: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Return numerically stable equivalent of exp-normalized charge redistribution fractions per graph."""
    batch_idx = batch.reshape(-1)
    logits = raw_weights.reshape(-1)
    max_per_graph = torch.full(
        (num_graphs,),
        float("-inf"),
        device=logits.device,
        dtype=logits.dtype,
    )
    max_per_graph.scatter_reduce_(0, batch_idx, logits, reduce="amax")
    exp_shifted = (logits - max_per_graph[batch_idx]).exp()
    denom = scatter_sum(
        exp_shifted.unsqueeze(-1),
        index=batch,
        dim=0,
        dim_size=num_graphs,
    )
    return (
        exp_shifted.reshape_as(raw_weights)
        / denom.index_select(0, batch_idx).reshape_as(raw_weights).clamp(min=1e-12)
    )


@compile_mode("script")
class ChargedMACE(ScaleShiftMACE):
    """MACE with an additional charge-equilibration prediction head.

    The inherited MACE readouts predict energy contributions. This class adds:

    - ``raw_charges``: unconstrained per-atom raw charges, shape ``[n_atoms, 1]``.
    - ``charge_weights``: normalized per-atom redistribution fractions (sum to 1 per graph).
    - ``charges``: graph-charge-conserved per-atom charges.
    - ``total_charge``: graph-level sum of conserved charges.

    Parameters
    ----------
    charge_mlp_irreps : o3.Irreps
        Irreps for the MLP in the charge head.
    """

    def __init__(
        self,
        *args: Any,
        charge_mlp_irreps: o3.Irreps = o3.Irreps("16x0e"),
        **kwargs: Any,
    ) -> None:
        cueq_config = kwargs.get("cueq_config", None)
        super().__init__(*args, **kwargs)

        # Readouts for raw charges and redistribution weights on final-block node features.
        final_node_irreps = self.products[-1].linear.irreps_out
        self.charge_feature_dim = final_node_irreps.dim

        self.charge_head = NonLinearReadoutBlock(
            irreps_in=final_node_irreps,
            MLP_irreps=charge_mlp_irreps,
            gate=torch.nn.functional.silu,
            irrep_out=o3.Irreps("1x0e"),
            num_heads=1,
            cueq_config=cueq_config,
        )
        self.charge_weight_head = NonLinearReadoutBlock(
            irreps_in=final_node_irreps,
            MLP_irreps=charge_mlp_irreps,  # same as charge head for simplicity
            gate=torch.nn.functional.silu,
            irrep_out=o3.Irreps("1x0e"),
            num_heads=1,
            cueq_config=cueq_config,
        )

    def equilibrate_charges(
        self,
        *,
        raw_charges: torch.Tensor,
        raw_weights: torch.Tensor,
        target_total_charge: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project raw charges so they sum to ``target_total_charge`` within each graph.
        Implement charge equilibration as in AIMNet2 (https://doi.org/10.1039/D4SC08572H).
        charges = raw_charges + fraction * charge_residual
        """
        raw_total_charge = scatter_sum(
            src=raw_charges,
            index=batch,
            dim=0,
            dim_size=num_graphs,
        ) # shape: [num_graphs, 1]
        charge_residual = target_total_charge - raw_total_charge
        charge_residual = charge_residual.index_select(0, batch)
        weight_fractions = _exp_charge_fractions(raw_weights, batch, num_graphs)
        charges = raw_charges + weight_fractions * charge_residual
        return charges, weight_fractions

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        output = super().forward(
            data=data,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            compute_atomic_stresses=compute_atomic_stresses,
            lammps_mliap=lammps_mliap,
        )

        # Select node features from the final block
        node_feats = output["node_feats"]
        if node_feats is None:
            raise RuntimeError("ChargedMACE requires MACE to return 'node_feats'.")
        final_node_feats = node_feats[:, -self.charge_feature_dim :]

        # Compute raw charges and weights
        raw_charges = self.charge_head(final_node_feats)
        raw_weights = self.charge_weight_head(final_node_feats)

        # Perform charge equilibration (``total_charge`` set by ``ChargedMACEWrapper``)
        batch = data["batch"]
        num_graphs = int(data["ptr"].numel() - 1)
        target_total_charge = data["total_charge"]
        charges, charge_weights = self.equilibrate_charges(
            raw_charges=raw_charges,
            raw_weights=raw_weights,
            target_total_charge=target_total_charge,
            batch=batch,
            num_graphs=num_graphs,
        )

        output["raw_charges"] = raw_charges
        output["charge_weights"] = charge_weights
        output["charges"] = charges
        output["target_total_charge"] = target_total_charge
        return output


class ChargedMACEWrapper(MACEWrapper):
    """MACE wrapper with charge-related outputs."""

    def __init__(self, model: ChargedMACE) -> None:
        super().__init__(model)
        # Add charge-related outputs to the model config
        self.model_config.outputs = self.model_config.outputs | frozenset(
            {
                "charges",
                "raw_charges",
                "charge_weights",
                "target_total_charge",
            }
        )
        self.model_config.active_outputs.add("charges")

    @staticmethod
    def _set_total_charge_input(
        data: AtomicData | Batch,
        model_inputs: dict[str, Any],
    ) -> None:
        """Set ``model_inputs["total_charge"]`` with shape ``[num_graphs, 1]``.

        Reads ``total_charge`` from *data* when present and shape-compatible;
        otherwise uses zero total charge per graph.
        """
        num_graphs = int(model_inputs["ptr"].numel() - 1)
        device = model_inputs["positions"].device
        dtype = model_inputs["positions"].dtype
        value = getattr(data, "total_charge", None)
        if isinstance(value, torch.Tensor):
            target = value.to(device=device, dtype=dtype)
            if target.shape[0] == num_graphs:
                model_inputs["total_charge"] = target.reshape(num_graphs, -1)[:, :1]
                return
        model_inputs["total_charge"] = torch.zeros(
            num_graphs, 1, device=device, dtype=dtype
        )

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        model_inputs = super().adapt_input(data, **kwargs)
        self._set_total_charge_input(data, model_inputs)
        return model_inputs

    def adapt_output(self, raw_output: dict[str, Any], data: Any) -> Any:
        mapped = super().adapt_output(raw_output, data)
        for key in (
            "charges",
            "raw_charges",
            "charge_weights",
            "target_total_charge",
        ):
            if key in raw_output:
                mapped[key] = raw_output.get(key)
        return mapped


class ShortRangeCoulombCorrection(torch.nn.Module, BaseModelMixin):
    """Subtract a switched short-range point-charge Coulomb term.

    The correction is evaluated over a neighbor matrix and returns a graph-level
    energy in eV:

    ``E = -0.5 * lambda_sub * k_e * sum_ij q_i q_j s(r_ij) / r_ij``.

    Here ``k_e`` has units eV A / e^2, charges are in elementary charge units,
    distances are in Angstrom, ``s(r)`` is a cosine switching function, and the
    resulting energy is eV.

    Parameters
    ----------
    inner_radius : float
        Radius in Angstrom below which the correction is fully applied.
    outer_radius : float
        Radius in Angstrom at which the correction is tapered to zero.
    lambda_sub : float, optional
        Dimensionless scale factor for the subtraction.
    coulomb_constant : float, optional
        Coulomb prefactor in eV A / e^2.
    """

    def __init__(
        self,
        *,
        inner_radius: float,
        outer_radius: float,
        lambda_sub: float = 1.0,
        coulomb_constant: float = 14.3996,
    ) -> None:
        super().__init__()
        if inner_radius < 0.0:
            raise ValueError("inner_radius must be non-negative.")
        if outer_radius <= 0.0:
            raise ValueError("outer_radius must be positive.")
        if inner_radius >= outer_radius:
            raise ValueError("inner_radius must be smaller than outer_radius.")
        if lambda_sub < 0.0:
            raise ValueError("lambda_sub must be non-negative.")
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.lambda_sub = float(lambda_sub)
        self.coulomb_constant = float(coulomb_constant)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            active_outputs={"energy"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charges"}),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=True,
            neighbor_config=NeighborConfig(
                cutoff=self.outer_radius,
                format=NeighborListFormat.MATRIX,
                half_list=False,
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute embeddings is not meaningful for this correction."""
        raise NotImplementedError(
            "ShortRangeCoulombCorrection does not produce embeddings."
        )

    def input_data(self) -> set[str]:
        """Return required input keys."""
        return {"positions", "charges", "neighbor_matrix", "num_neighbors"}

    def output_data(self) -> set[str]:
        """Return output keys currently produced by the correction."""
        return {"energy"}

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect tensor inputs from a batch."""
        if not isinstance(data, Batch):
            raise TypeError(
                "ShortRangeCoulombCorrection requires a Batch input; "
                "got AtomicData. Use Batch.from_data_list([data]) to wrap it."
            )
        cell = getattr(data, "cell", None)
        return {
            "positions": data.positions,
            "charges": data.charges,
            "neighbor_matrix": data.neighbor_matrix,
            "neighbor_matrix_shifts": getattr(data, "neighbor_matrix_shifts", None),
            "batch_idx": data.batch_idx,
            "cell": cell,
            "fill_value": data.num_nodes,
            "num_graphs": data.num_graphs,
        }

    def adapt_output(
        self,
        model_output: dict[str, torch.Tensor],
        data: AtomicData | Batch,
    ) -> OrderedDict[str, torch.Tensor]:
        """Adapt the model output to the framework output format."""
        output: OrderedDict[str, torch.Tensor] = OrderedDict()
        output["energy"] = model_output["energy"]
        return output

    def _cutoff_switch(self, distances: torch.Tensor) -> torch.Tensor:
        """Return a smooth switch that tapers from inner to outer radius."""
        switch = torch.ones_like(distances)
        width = self.outer_radius - self.inner_radius
        taper = (distances - self.inner_radius) / width
        taper = taper.clamp(min=0.0, max=1.0)
        switch = torch.where(
            distances > self.inner_radius,
            0.5 * (1.0 + torch.cos(torch.pi * taper)),
            switch,
        )
        return torch.where(
            distances < self.outer_radius,
            switch,
            torch.zeros_like(switch),
        )

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> OrderedDict[str, torch.Tensor]:
        """Compute the graph-level short-range Coulomb correction energy."""
        inp = self.adapt_input(data, **kwargs)
        positions = inp["positions"]
        charges = inp["charges"].reshape(-1)
        neighbor_matrix = inp["neighbor_matrix"].long()
        neighbor_matrix_shifts = inp["neighbor_matrix_shifts"]
        batch_idx = inp["batch_idx"].long()
        cell = inp["cell"]
        fill_value = int(inp["fill_value"])
        num_graphs = int(inp["num_graphs"])

        valid_neighbor = neighbor_matrix != fill_value
        safe_neighbor = torch.where(
            valid_neighbor,
            neighbor_matrix,
            torch.zeros_like(neighbor_matrix),
        )

        neighbor_positions = positions.index_select(0, safe_neighbor.reshape(-1))
        neighbor_positions = neighbor_positions.reshape(*safe_neighbor.shape, 3)
        displacements = neighbor_positions - positions.unsqueeze(1)

        if neighbor_matrix_shifts is not None:
            if cell is None:
                raise ValueError(
                    "neighbor_matrix_shifts were provided but data.cell is missing."
                )
            shifts = neighbor_matrix_shifts.to(dtype=positions.dtype)
            cell_per_atom = cell.index_select(0, batch_idx)
            shift_vectors = torch.einsum("nka,nab->nkb", shifts, cell_per_atom)
            displacements = displacements + shift_vectors

        distances = displacements.norm(dim=-1)
        valid_pair = valid_neighbor & (distances > 1.0e-12)
        safe_distances = distances.clamp(min=1.0e-12)
        q_i = charges.unsqueeze(-1)
        q_j = charges.index_select(0, safe_neighbor.reshape(-1)).reshape_as(distances)

        pair_energy = (
            -0.5
            * self.lambda_sub
            * self.coulomb_constant
            * q_i
            * q_j
            * self._cutoff_switch(safe_distances)
            / safe_distances
        )
        pair_energy = torch.where(
            valid_pair,
            pair_energy,
            torch.zeros_like(pair_energy),
        )

        per_atom_energy = pair_energy.sum(dim=1)
        energy = torch.zeros(
            num_graphs,
            dtype=positions.dtype,
            device=positions.device,
        )
        energy.scatter_add_(0, batch_idx, per_atom_energy)
        return self.adapt_output({"energy": energy.unsqueeze(-1)}, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for this correction."""
        raise NotImplementedError


def build_charged_mace_model(
    *,
    atomic_numbers: list[int],
    atomic_energies: np.ndarray,
    r_max: float,
    avg_num_neighbors: float,
    model_config: DictConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> PipelineModelWrapper:
    """Construct a ChargedMACE + Ewald electrostatics pipeline.

    Parameters
    ----------
    atomic_numbers : list[int]
        Atomic numbers expected in the training set.
    atomic_energies : np.ndarray
        Atomic reference energies ordered like ``atomic_numbers``.
    r_max : float
        Neighbor cutoff in Angstrom.
    avg_num_neighbors : float
        Dataset-level average number of neighbor edges per atom.
    model_config : DictConfig
        MACE architecture configuration. May include ``charge_mlp_irreps`` for
        the charge head; otherwise ``mlp_irreps`` is used.
    dtype : torch.dtype
        Floating-point dtype for model parameters.
    device : torch.device
        Device for model parameters.

    Returns
    -------
    PipelineModelWrapper
        Pipeline that sums ChargedMACE and Ewald energies before differentiating.
    """
    charge_mlp_irreps = getattr(model_config, "charge_mlp_irreps", None)
    cueq_config = build_cueq_config(model_config)
    mace_model = ChargedMACE(
        **get_scale_shift_config(model_config),
        r_max=r_max,
        num_bessel=model_config.num_bessel,
        num_polynomial_cutoff=model_config.num_polynomial_cutoff,
        max_ell=model_config.max_ell,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        distance_transform="Agnesi",
        num_interactions=model_config.num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps(model_config.hidden_irreps),
        MLP_irreps=o3.Irreps(model_config.mlp_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=model_config.correlation,
        gate=torch.nn.functional.silu,
        pair_repulsion=False,
        heads=list(model_config.heads),
        charge_mlp_irreps=(
            o3.Irreps(charge_mlp_irreps)
            if charge_mlp_irreps is not None
            else o3.Irreps(model_config.mlp_irreps)
        ),
        cueq_config=cueq_config,
    )
    mace_model = mace_model.to(device=device, dtype=dtype)
    charged_mace = ChargedMACEWrapper(mace_model)
    charged_mace.model_config.active_outputs = {
        "energy",
        "forces",
        "stress",
        "charges",
    }

    ewald_config = model_config.get("ewald", {})
    ewald_accuracy = float(ewald_config.get("accuracy", 1e-6))
    coulomb_constant = float(ewald_config.get("coulomb_constant", 14.3996))
    ewald = EwaldModelWrapper(
        cutoff=float(ewald_config.get("cutoff", r_max)),
        accuracy=ewald_accuracy,
        coulomb_constant=coulomb_constant,
        hybrid_forces=False,
    )

    steps: list[BaseModelMixin] = [charged_mace, ewald]
    correction_config = model_config.get("charge_correction", {})
    if bool(correction_config.get("enabled", False)):
        outer_radius = float(correction_config.get("outer_radius", r_max))
        inner_radius = float(correction_config.get("inner_radius", 0.0))
        steps.append(
            ShortRangeCoulombCorrection(
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                lambda_sub=float(correction_config.get("lambda_sub", 1.0)),
                coulomb_constant=float(
                    correction_config.get("coulomb_constant", coulomb_constant)
                ),
            )
        )

    model = PipelineModelWrapper(
        groups=[
            PipelineGroup(
                steps=steps,
                use_autograd=True,
            ),
        ]
    )
    model.model_config.active_outputs = {"energy", "forces", "stress", "charges"}
    model.train()
    return model


def build_vanilla_mace_model(
    *,
    atomic_numbers: list[int],
    atomic_energies: np.ndarray,
    r_max: float,
    avg_num_neighbors: float,
    model_config: DictConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> MACEWrapper:
    """Construct a vanilla ScaleShiftMACE model.

    Parameters
    ----------
    atomic_numbers : list[int]
        Atomic numbers expected in the training set.
    atomic_energies : np.ndarray
        Atomic reference energies ordered like ``atomic_numbers``.
    r_max : float
        Neighbor cutoff in Angstrom.
    avg_num_neighbors : float
        Dataset-level average number of neighbor edges per atom.
    model_config : DictConfig
        MACE architecture configuration.
    dtype : torch.dtype
        Floating-point dtype for model parameters.
    device : torch.device
        Device for model parameters.

    Returns
    -------
    MACEWrapper
        Random-initialized wrapped MACE model.
    """
    cueq_config = build_cueq_config(model_config)
    mace_model = ScaleShiftMACE(
        **get_scale_shift_config(model_config),
        r_max=r_max,
        num_bessel=model_config.num_bessel,
        num_polynomial_cutoff=model_config.num_polynomial_cutoff,
        max_ell=model_config.max_ell,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        distance_transform="Agnesi",
        num_interactions=model_config.num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=o3.Irreps(model_config.hidden_irreps),
        MLP_irreps=o3.Irreps(model_config.mlp_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=model_config.correlation,
        gate=torch.nn.functional.silu,
        pair_repulsion=False,
        heads=list(model_config.heads),
        cueq_config=cueq_config,
    )
    mace_model = mace_model.to(device=device, dtype=dtype)
    model = MACEWrapper(mace_model)
    model.model_config.active_outputs = {"energy", "forces", "stress"}
    model.train()
    return model


def build_training_mace_model(
    *,
    model_type: str,
    atomic_numbers: Sequence[int],
    atomic_energies: Sequence[float],
    r_max: float,
    avg_num_neighbors: float,
    model_config: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
    active_outputs: Sequence[str],
) -> torch.nn.Module:
    """Construct the MACE model used by the advanced training recipe.

    This factory is intentionally plain and importable so
    :class:`~nvalchemi.training.TrainingStrategy` checkpoints can rebuild the
    same MACE architecture before loading saved weights.
    """
    cfg = OmegaConf.create(model_config)
    atomic_numbers_list = [int(value) for value in atomic_numbers]
    atomic_energies_array = np.asarray(
        [float(value) for value in atomic_energies],
        dtype=float,
    )
    if model_type == "charged_mace":
        builder = build_charged_mace_model
    elif model_type == "mace":
        builder = build_vanilla_mace_model
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    model = builder(
        atomic_numbers=atomic_numbers_list,
        atomic_energies=atomic_energies_array,
        r_max=float(r_max),
        avg_num_neighbors=float(avg_num_neighbors),
        model_config=cfg,
        dtype=dtype,
        device=device,
    )
    model.model_config.active_outputs = set(active_outputs)
    return _attach_checkpoint_spec(
        model,
        model_type=model_type,
        atomic_numbers=atomic_numbers_list,
        atomic_energies=atomic_energies_array.tolist(),
        r_max=float(r_max),
        avg_num_neighbors=float(avg_num_neighbors),
        model_config=dict(model_config),
        dtype=dtype,
        device=device,
        active_outputs=active_outputs,
    )
