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
"""MACE model builders for the training example (user guide §2).

``build_vanilla_mace_model`` wraps ScaleShiftMACE for energy, forces, and stress.

``build_training_mace_model`` constructs ScaleShiftMACE with dataset-derived
metadata and wraps it in :class:`~nvalchemi.models.mace.MACEWrapper` for
:class:`~nvalchemi.training.TrainingStrategy`. It is the checkpoint-reconstructable
factory used by ``examples/advanced/10_mace_training.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

torch.serialization.add_safe_globals([slice])

from e3nn import o3
from mace.modules import (
    RealAgnosticDensityInteractionBlock,
    RealAgnosticDensityResidualInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    ScaleShiftMACE,
)
from mace.modules.wrapper_ops import CuEquivarianceConfig
from omegaconf import DictConfig, OmegaConf

from nvalchemi.models.mace import MACEWrapper
from nvalchemi.training import create_model_spec

INTERACTION_BLOCKS: dict[str, Any] = {
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock": RealAgnosticDensityResidualInteractionBlock,
}


def build_cueq_config(model_config: DictConfig) -> CuEquivarianceConfig | None:
    """Build the optional cuEquivariance config for MACE modules."""
    cueq_cfg = model_config.get("cueq", None)
    if cueq_cfg is None:
        return None
    if not bool(cueq_cfg.get("enabled", False)):
        return None
    kwargs = {
        "enabled": True,
        "layout": str(cueq_cfg.get("layout", "ir_mul")),
        "group": str(cueq_cfg.get("group", "O3_e3nn")),
        "optimize_all": bool(cueq_cfg.get("optimize_all", False)),
        "optimize_linear": bool(cueq_cfg.get("optimize_linear", False)),
        "optimize_channelwise": bool(cueq_cfg.get("optimize_channelwise", False)),
        "optimize_symmetric": bool(cueq_cfg.get("optimize_symmetric", False)),
        "optimize_fctp": bool(cueq_cfg.get("optimize_fctp", False)),
        "conv_fusion": bool(cueq_cfg.get("conv_fusion", False)),
    }
    return CuEquivarianceConfig(**kwargs)


def _get_interaction_block(name: str) -> Any:
    try:
        return INTERACTION_BLOCKS[name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(INTERACTION_BLOCKS))
        raise ValueError(
            f"Unknown MACE interaction block {name!r}. Expected one of: {valid_names}.",
        ) from exc


def get_e0s(model_cfg: DictConfig) -> tuple[list[int], np.ndarray]:
    """Return atomic numbers and reference per-element energies from the config.

    Parameters
    ----------
    model_cfg : DictConfig
        Hydra model config containing an ``E0s`` mapping from atomic number to
        reference energy.

    Returns
    -------
    tuple[list[int], np.ndarray]
        Sorted atomic numbers and matching reference energies.

    Raises
    ------
    ValueError
        If ``E0s`` is not a mapping from atomic number to energy.
    """
    e0s = OmegaConf.to_container(model_cfg.E0s, resolve=True)
    if not isinstance(e0s, dict):
        raise ValueError("cfg.model.E0s must be a mapping from atomic number to E0.")
    e0_by_z = {int(z): float(e0) for z, e0 in e0s.items()}
    atomic_numbers = sorted(e0_by_z)
    atomic_energies = np.asarray([e0_by_z[z] for z in atomic_numbers])
    return atomic_numbers, atomic_energies


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
        interaction_cls=_get_interaction_block(
            str(
                model_config.get("interaction", "RealAgnosticResidualInteractionBlock")
            ),
        ),
        interaction_cls_first=_get_interaction_block(
            str(
                model_config.get(
                    "interaction_first",
                    model_config.get(
                        "interaction",
                        "RealAgnosticResidualInteractionBlock",
                    ),
                ),
            ),
        ),
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
        heads=["default"],
        use_reduced_cg=bool(model_config.get("use_reduced_cg", True)),
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
    """Construct the MACE model used by the training recipe.

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
    if model_type == "mace":
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

    # Create a checkpointable model specification.
    checkpoint_spec = create_model_spec(
        build_training_mace_model,
        model_type=model_type,
        atomic_numbers=atomic_numbers_list,
        atomic_energies=atomic_energies_array.tolist(),
        r_max=float(r_max),
        avg_num_neighbors=float(avg_num_neighbors),
        model_config=dict(model_config),
        dtype=dtype,
        device=device,
        active_outputs=sorted(active_outputs),
    )
    model.checkpoint_spec = lambda: checkpoint_spec  # type: ignore[attr-defined]

    return model
