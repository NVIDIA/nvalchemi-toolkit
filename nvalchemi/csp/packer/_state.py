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

"""Mutable candidate state used only while one CrystalPacker call runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nvalchemi.csp.packer._kernels.relaxation import relax_step as _relax_step
from nvalchemi.csp.packer._kernels.sampling import initialize_rows


@dataclass
class WorkingState:
    """Mutable geometry and sampling state for a fixed candidate row capacity.

    ``centers`` are fractional row vectors in ``[0, 1)``. ``rotations`` map
    conformer Cartesian coordinates into Cartesian cell coordinates. ``cells``
    stores lower-triangular row vectors in Å, and ``reference_volumes`` stores
    volumes in Å³. ``symmetry_ops`` contains indices into the symmetry-operation
    table.
    """

    conformer_ids: Tensor
    centers: Tensor
    rotations: Tensor
    cells: Tensor
    inverse_cells: Tensor
    reference_volumes: Tensor
    space_groups: Tensor
    symmetry_ops: Tensor
    steps: Tensor
    active: Tensor

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        molecule_count: int,
        symmetry_operation_count: int,
        device: torch.device,
    ) -> WorkingState:
        """Allocate candidate tensors with identity cells and rotations on
        the requested device.
        """
        identity_cells = (
            torch.eye(3, dtype=torch.float32, device=device)
            .expand(batch_size, 3, 3)
            .clone()
        )
        return cls(
            conformer_ids=torch.zeros(
                (batch_size, molecule_count), dtype=torch.int32, device=device
            ),
            centers=torch.zeros(
                (batch_size, molecule_count, 3), dtype=torch.float32, device=device
            ),
            rotations=torch.eye(3, dtype=torch.float32, device=device)
            .expand(batch_size, molecule_count, 3, 3)
            .clone(),
            cells=identity_cells,
            inverse_cells=identity_cells.clone(),
            reference_volumes=torch.zeros(
                (batch_size,), dtype=torch.float32, device=device
            ),
            space_groups=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            symmetry_ops=torch.zeros(
                (batch_size, symmetry_operation_count), dtype=torch.int32, device=device
            ),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            active=torch.zeros((batch_size,), dtype=torch.bool, device=device),
        )

    def initialize(
        self,
        *,
        rows: Tensor,
        sampled_cells: Tensor,
        sampled_groups: Tensor,
        op_indices: Tensor,
        op_ptr: Tensor,
        conformer_starts: Tensor,
        conformer_stops: Tensor,
        seed: int,
    ) -> None:
        """Initialize the selected candidate rows in the supplied order."""
        if rows.numel() == 0:
            return
        initialize_rows(
            rows=rows.to(dtype=torch.int32).contiguous(),
            sampled_cells=sampled_cells.contiguous(),
            sampled_groups=sampled_groups.contiguous(),
            op_indices=op_indices,
            op_ptr=op_ptr,
            conformer_starts=conformer_starts,
            conformer_stops=conformer_stops,
            seed=seed,
            conformer_ids=self.conformer_ids,
            centers=self.centers,
            rotations=self.rotations,
            cells=self.cells,
            inverse_cells=self.inverse_cells,
            reference_volumes=self.reference_volumes,
            groups=self.space_groups,
            selected_ops=self.symmetry_ops,
            steps=self.steps,
        )
        self.active[rows] = True


def relax_step(
    *,
    state: WorkingState,
    conformer_positions: Tensor,
    conformer_ptr: Tensor,
    molecule_atom_ptr: Tensor,
    forces: Tensor,
    torques: Tensor,
    virial: Tensor,
    max_overlap: Tensor,
    expanded_atom_count: int,
    step_scale: float,
    max_step: float,
    cell_step_scale: float,
    max_cell_strain: float,
    volume_compression_scale: float,
    active: Tensor | None = None,
) -> None:
    """Apply a batched rigid-body and crystal-system-preserving cell update."""
    _relax_step(
        active=state.active if active is None else active,
        conformer_positions=conformer_positions,
        conformer_ptr=conformer_ptr,
        conformer_ids=state.conformer_ids,
        molecule_atom_ptr=molecule_atom_ptr,
        forces=forces,
        torques=torques,
        max_overlap=max_overlap,
        virial=virial,
        cells=state.cells,
        inverse_cells=state.inverse_cells,
        centers=state.centers,
        rotations=state.rotations,
        space_groups=state.space_groups,
        reference_volumes=state.reference_volumes,
        steps=state.steps,
        expanded_atom_count=expanded_atom_count,
        step_scale=step_scale,
        max_step=max_step,
        cell_step_scale=cell_step_scale,
        max_cell_strain=max_cell_strain,
        volume_compression_scale=volume_compression_scale,
    )
