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

"""Periodic, symmetry-expanded contact forces for Packer candidate rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp
from nvalchemiops.math import wpdivmod
from nvalchemiops.torch.neighbors.batch_cell_list import (
    batch_build_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import allocate_cell_list
from torch import Tensor

from nvalchemi.csp.packer._interop import as_warp, launch_device, scoped_warp_stream


@dataclass(frozen=True)
class ContactMaps:
    """Static formula-atom indexing for the symmetry-expanded P1 cell."""

    molecule: Tensor
    symmetry: Tensor
    asu_atom: Tensor
    local_atom: Tensor
    expanded_meta: Tensor
    num_molecules: int
    num_symmetry_operations: int
    num_asu_atoms: int


@dataclass(frozen=True)
class ContactEvaluation:
    """Contact outputs backed by a reusable workspace; copy tensors before
    the next evaluation if they must be retained.

    ``virial`` is a numerical overlap tensor normalized by expanded atom count
    for cell updates; it is not physical stress.
    """

    forces: Tensor
    torques: Tensor
    total_overlap: Tensor
    max_overlap: Tensor
    virial: Tensor


@dataclass
class ContactWorkspace:
    """Reusable expanded-atom and cell-list storage for one fixed batch shape.

    Cell-list capacity is re-estimated from the current candidate cells before
    each build. The atom buffers and force outputs retain their fixed shapes;
    cell arrays grow only when the conservative estimate exceeds capacity.
    """

    batch_size: int
    num_molecules: int
    expanded_atom_count: int
    max_cell_count: int
    positions: Tensor
    image_rel_asu: Tensor
    image_offsets: Tensor
    batch_idx: Tensor
    pbc: Tensor
    forces: Tensor
    torques: Tensor
    total_overlap: Tensor
    max_overlap: Tensor
    virial: Tensor
    cells_per_dimension: Tensor | None = None
    neighbor_search_radius: Tensor | None = None
    atom_periodic_shifts: Tensor | None = None
    atom_to_cell_mapping: Tensor | None = None
    atoms_per_cell_count: Tensor | None = None
    cell_atom_start_indices: Tensor | None = None
    cell_atom_list: Tensor | None = None
    cell_offsets: Tensor | None = None

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_molecules: int,
        expanded_atom_count: int,
        device: torch.device,
    ) -> ContactWorkspace:
        """Allocate fixed-shape contact buffers on the requested device."""
        total = batch_size * expanded_atom_count
        batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
        batch_idx = batch_idx.repeat_interleave(expanded_atom_count)
        return cls(
            batch_size=batch_size,
            num_molecules=num_molecules,
            expanded_atom_count=expanded_atom_count,
            max_cell_count=0,
            positions=torch.empty((total, 3), dtype=torch.float32, device=device),
            image_rel_asu=torch.empty((total, 3), dtype=torch.float32, device=device),
            image_offsets=torch.empty((total, 3), dtype=torch.int32, device=device),
            batch_idx=batch_idx,
            pbc=torch.ones((batch_size, 3), dtype=torch.bool, device=device),
            forces=torch.empty(
                (batch_size, num_molecules, 3), dtype=torch.float32, device=device
            ),
            torques=torch.empty(
                (batch_size, num_molecules, 3), dtype=torch.float32, device=device
            ),
            total_overlap=torch.empty(
                (batch_size,), dtype=torch.float32, device=device
            ),
            max_overlap=torch.empty((batch_size,), dtype=torch.float32, device=device),
            virial=torch.empty((batch_size, 3, 3), dtype=torch.float32, device=device),
        )

    def ensure_cell_capacity(self, cells: Tensor, cutoff: float) -> None:
        """Grow the cell-list buffers when the current cells require more capacity."""
        max_cells, radius = estimate_batch_cell_list_sizes(
            cells.contiguous(), self.pbc, cutoff, min_cells_per_dimension=1
        )
        if max_cells > self.max_cell_count or self.cells_per_dimension is None:
            (
                self.cells_per_dimension,
                _,
                self.atom_periodic_shifts,
                self.atom_to_cell_mapping,
                self.atoms_per_cell_count,
                self.cell_atom_start_indices,
                self.cell_atom_list,
            ) = allocate_cell_list(
                self.batch_size * self.expanded_atom_count,
                max_cells,
                radius,
                cells.device,
            )
            self.max_cell_count = max_cells
        self.neighbor_search_radius = radius

    def build_cell_list(self, cells: Tensor, cutoff: float) -> None:
        """Build cell-list indices for the current expanded atom positions."""
        self.ensure_cell_capacity(cells, cutoff)
        cells_per_dimension = self.cells_per_dimension
        neighbor_search_radius = self.neighbor_search_radius
        atom_periodic_shifts = self.atom_periodic_shifts
        atom_to_cell_mapping = self.atom_to_cell_mapping
        atoms_per_cell_count = self.atoms_per_cell_count
        cell_atom_start_indices = self.cell_atom_start_indices
        cell_atom_list = self.cell_atom_list
        if any(
            value is None
            for value in (
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )
        ):
            raise RuntimeError("contact cell-list workspace was not allocated")
        batch_build_cell_list(
            self.positions,
            cutoff,
            cells.contiguous(),
            self.pbc,
            self.batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            min_cells_per_dimension=1,
        )
        cells_per_system = cells_per_dimension.to(torch.int64).prod(dim=1)
        self.cell_offsets = torch.cat(
            (
                torch.zeros((1,), dtype=torch.int32, device=cells.device),
                cells_per_system[:-1].cumsum(dim=0).to(torch.int32),
            )
        )


def make_contact_maps(
    molecule_atom_ptr: Tensor, num_symmetry_operations: int
) -> ContactMaps:
    """Build expanded atom maps in molecule/symmetry/local-atom order."""
    pointers = molecule_atom_ptr.detach().to(device="cpu").tolist()
    molecule: list[int] = []
    symmetry: list[int] = []
    asu_atom: list[int] = []
    local_atom: list[int] = []
    for molecule_index, (start, stop) in enumerate(zip(pointers, pointers[1:])):
        for symmetry_index in range(num_symmetry_operations):
            for atom_index in range(start, stop):
                molecule.append(molecule_index)
                symmetry.append(symmetry_index)
                asu_atom.append(atom_index)
                local_atom.append(atom_index - start)
    device = molecule_atom_ptr.device
    molecule_t = torch.tensor(molecule, dtype=torch.int32, device=device)
    symmetry_t = torch.tensor(symmetry, dtype=torch.int32, device=device)
    asu_atom_t = torch.tensor(asu_atom, dtype=torch.int32, device=device)
    local_atom_t = torch.tensor(local_atom, dtype=torch.int32, device=device)
    image = molecule_t.to(torch.int64) * int(num_symmetry_operations) + symmetry_t.to(
        torch.int64
    )
    meta = torch.stack(
        (
            molecule_t.to(torch.int32),
            symmetry_t.to(torch.int32),
            image.to(torch.int32),
            asu_atom_t,
        ),
        dim=1,
    ).contiguous()
    return ContactMaps(
        molecule=molecule_t,
        symmetry=symmetry_t,
        asu_atom=asu_atom_t,
        local_atom=local_atom_t,
        expanded_meta=meta,
        num_molecules=len(pointers) - 1,
        num_symmetry_operations=num_symmetry_operations,
        num_asu_atoms=pointers[-1],
    )


@wp.func
def _load_geometry_matrix(
    values: wp.array(dtype=wp.float32, ndim=3), row: int
) -> wp.mat33f:
    """Load one 3 by 3 matrix from a row-major geometry tensor."""
    return wp.mat33f(
        values[row, 0, 0],
        values[row, 0, 1],
        values[row, 0, 2],
        values[row, 1, 0],
        values[row, 1, 1],
        values[row, 1, 2],
        values[row, 2, 0],
        values[row, 2, 1],
        values[row, 2, 2],
    )


@wp.func
def _load_geometry_rotation(
    values: wp.array(dtype=wp.float32, ndim=4), row: int, molecule: int
) -> wp.mat33f:
    """Load one molecule rotation matrix from a batched tensor."""
    return wp.mat33f(
        values[row, molecule, 0, 0],
        values[row, molecule, 0, 1],
        values[row, molecule, 0, 2],
        values[row, molecule, 1, 0],
        values[row, molecule, 1, 1],
        values[row, molecule, 1, 2],
        values[row, molecule, 2, 0],
        values[row, molecule, 2, 1],
        values[row, molecule, 2, 2],
    )


@wp.func
def _geometry_vector(
    values: wp.array(dtype=wp.float32, ndim=3), row: int, molecule: int
) -> wp.vec3f:
    """Load one three-component vector from a batched geometry tensor."""
    return wp.vec3f(
        values[row, molecule, 0], values[row, molecule, 1], values[row, molecule, 2]
    )


@wp.kernel(enable_backward=False)
def _materialize_expanded_geometry(
    conformer_positions: wp.array(dtype=wp.float32, ndim=2),
    conformer_ptr: wp.array(dtype=wp.int32),
    conformer_ids: wp.array(dtype=wp.int32, ndim=2),
    centers: wp.array(dtype=wp.float32, ndim=3),
    rotations: wp.array(dtype=wp.float32, ndim=4),
    cells: wp.array(dtype=wp.float32, ndim=3),
    inverse_cells: wp.array(dtype=wp.float32, ndim=3),
    symmetry_table: wp.array(dtype=wp.float32, ndim=2),
    selected_symmetry_ops: wp.array(dtype=wp.int32, ndim=2),
    molecule_map: wp.array(dtype=wp.int32),
    symmetry_map: wp.array(dtype=wp.int32),
    local_atom_map: wp.array(dtype=wp.int32),
    expanded_atom_count: wp.int32,
    positions: wp.array(dtype=wp.float32, ndim=2),
    asu_arms: wp.array(dtype=wp.float32, ndim=2),
    image_offsets: wp.array(dtype=wp.int32, ndim=2),
) -> None:
    """Write symmetry-expanded Cartesian atom positions and image metadata."""
    flat_atom = wp.tid()
    row = flat_atom / expanded_atom_count
    atom = flat_atom - row * expanded_atom_count
    molecule = wp.int32(molecule_map[atom])
    symmetry = wp.int32(symmetry_map[atom])
    local_atom = wp.int32(local_atom_map[atom])
    conformer = conformer_ids[row, molecule]
    conformer_atom = conformer_ptr[conformer] + local_atom
    local = wp.vec3f(
        conformer_positions[conformer_atom, 0],
        conformer_positions[conformer_atom, 1],
        conformer_positions[conformer_atom, 2],
    )
    asu_displacement = _load_geometry_rotation(rotations, row, molecule) * local
    frac_displacement = (
        wp.transpose(_load_geometry_matrix(inverse_cells, row)) * asu_displacement
    )

    operation_id = selected_symmetry_ops[row, symmetry]
    operation = wp.mat33f(
        symmetry_table[operation_id, 0],
        symmetry_table[operation_id, 1],
        symmetry_table[operation_id, 2],
        symmetry_table[operation_id, 3],
        symmetry_table[operation_id, 4],
        symmetry_table[operation_id, 5],
        symmetry_table[operation_id, 6],
        symmetry_table[operation_id, 7],
        symmetry_table[operation_id, 8],
    )
    translation = wp.vec3f(
        symmetry_table[operation_id, 9],
        symmetry_table[operation_id, 10],
        symmetry_table[operation_id, 11],
    )
    center = _geometry_vector(centers, row, molecule)
    sym_center = operation * center + translation
    sym_center = sym_center - wp.vec3f(
        wp.floor(sym_center[0]), wp.floor(sym_center[1]), wp.floor(sym_center[2])
    )
    unwrapped = sym_center + operation * frac_displacement
    offset = wp.vec3i(
        wp.int32(wp.floor(unwrapped[0])),
        wp.int32(wp.floor(unwrapped[1])),
        wp.int32(wp.floor(unwrapped[2])),
    )
    wrapped = unwrapped - wp.vec3f(
        wp.float32(offset[0]), wp.float32(offset[1]), wp.float32(offset[2])
    )
    cartesian = wp.transpose(_load_geometry_matrix(cells, row)) * wrapped
    positions[flat_atom, 0] = cartesian[0]
    positions[flat_atom, 1] = cartesian[1]
    positions[flat_atom, 2] = cartesian[2]
    asu_arms[flat_atom, 0] = asu_displacement[0]
    asu_arms[flat_atom, 1] = asu_displacement[1]
    asu_arms[flat_atom, 2] = asu_displacement[2]
    image_offsets[flat_atom, 0] = offset[0]
    image_offsets[flat_atom, 1] = offset[1]
    image_offsets[flat_atom, 2] = offset[2]


def _expanded_geometry(
    *,
    conformer_positions: Tensor,
    conformer_ptr: Tensor,
    conformer_ids: Tensor,
    centers: Tensor,
    rotations: Tensor,
    cells: Tensor,
    inverse_cells: Tensor,
    symmetry_table: Tensor,
    selected_symmetry_ops: Tensor,
    maps: ContactMaps,
    positions: Tensor,
    arms: Tensor,
    image_offsets: Tensor,
) -> Tensor:
    """Materialize expanded coordinates and return force-pull matrices.

    The returned matrices transform expanded atom forces into independent
    molecules' Cartesian frames.
    """
    batch = conformer_ids.shape[0]
    operation_rotations = symmetry_table[selected_symmetry_ops][..., :9].reshape(
        batch, maps.num_symmetry_operations, 3, 3
    )
    pull = torch.matmul(
        torch.matmul(
            inverse_cells[:, None, :, :],
            operation_rotations.transpose(-1, -2),
        ),
        cells[:, None, :, :],
    ).contiguous()
    with scoped_warp_stream(cells):
        wp.launch(
            _materialize_expanded_geometry,
            dim=batch * maps.num_asu_atoms * maps.num_symmetry_operations,
            inputs=[
                as_warp(conformer_positions, wp.float32),
                as_warp(conformer_ptr, wp.int32),
                as_warp(conformer_ids, wp.int32),
                as_warp(centers, wp.float32),
                as_warp(rotations, wp.float32),
                as_warp(cells, wp.float32),
                as_warp(inverse_cells, wp.float32),
                as_warp(symmetry_table, wp.float32),
                as_warp(selected_symmetry_ops, wp.int32),
                as_warp(maps.molecule, wp.int32),
                as_warp(maps.symmetry, wp.int32),
                as_warp(maps.local_atom, wp.int32),
                wp.int32(maps.num_asu_atoms * maps.num_symmetry_operations),
                as_warp(positions, wp.float32),
                as_warp(arms, wp.float32),
                as_warp(image_offsets, wp.int32),
            ],
            device=launch_device(cells),
        )
    return pull


@wp.func
def _lower_tri_cell_shift(
    shift: wp.vec3i,
    l00: wp.float32,
    l10: wp.float32,
    l11: wp.float32,
    l20: wp.float32,
    l21: wp.float32,
    l22: wp.float32,
) -> wp.vec3f:
    """Convert an integer lattice shift with the lower-triangular cell matrix."""
    return wp.vec3f(
        l00 * wp.float32(shift[0])
        + l10 * wp.float32(shift[1])
        + l20 * wp.float32(shift[2]),
        l11 * wp.float32(shift[1]) + l21 * wp.float32(shift[2]),
        l22 * wp.float32(shift[2]),
    )


@wp.func
def _pair_force(
    vector: wp.vec3f, distance: wp.float32, overlap: wp.float32, asu_i: int, asu_j: int
) -> wp.vec3f:
    """Return the overlap force along a pair direction, with a deterministic zero-distance axis."""
    unit = wp.vec3f(1.0, 0.0, 0.0)
    if distance > 1.0e-8:
        unit = vector / distance
    else:
        key = asu_i * 31 + asu_j * 17 + 13
        axis = key % 3
        sign = 1.0
        if (key & 1) != 0:
            sign = -1.0
        if axis == 0:
            unit = wp.vec3f(sign, 0.0, 0.0)
        elif axis == 1:
            unit = wp.vec3f(0.0, sign, 0.0)
        else:
            unit = wp.vec3f(0.0, 0.0, sign)
    return overlap * unit


@wp.kernel(enable_backward=False)
def _accumulate_cell_contacts(
    positions: wp.array(dtype=wp.vec3f),
    image_rel_asu: wp.array(dtype=wp.vec3f),
    image_offsets: wp.array(dtype=wp.vec3i),
    expanded_meta: wp.array(dtype=wp.vec4i),
    cells: wp.array(dtype=wp.mat33f),
    pull_matrices: wp.array(dtype=wp.mat33f, ndim=2),
    contact_distances: wp.array(dtype=wp.float32, ndim=2),
    active: wp.array(dtype=wp.bool),
    cells_per_dimension: wp.array(dtype=wp.vec3i),
    neighbor_search_radius: wp.array(dtype=wp.vec3i),
    atom_periodic_shifts: wp.array(dtype=wp.vec3i),
    atom_to_cell_mapping: wp.array(dtype=wp.vec3i),
    atoms_per_cell_count: wp.array(dtype=wp.int32),
    cell_atom_start_indices: wp.array(dtype=wp.int32),
    cell_atom_list: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    expanded_atom_count: wp.int32,
    forces: wp.array(dtype=wp.vec3f, ndim=2),
    torques: wp.array(dtype=wp.vec3f, ndim=2),
    total_overlap: wp.array(dtype=wp.float32),
    max_overlap: wp.array(dtype=wp.float32),
    virial: wp.array(dtype=wp.mat33f),
) -> None:
    """Accumulate periodic pair forces, torques, overlaps, and virial per candidate.

    Requires lower-triangular cell matrices.
    """
    atom_i = wp.tid()
    row = atom_i / expanded_atom_count
    if not active[row]:
        return
    local_i = atom_i - row * expanded_atom_count
    pos_i = positions[atom_i]
    pos_i_rel = image_rel_asu[atom_i]
    cell_i = atom_to_cell_mapping[atom_i]
    dims = cells_per_dimension[row]
    radius = neighbor_search_radius[row]
    shift_i = atom_periodic_shifts[atom_i]
    offset_i = image_offsets[atom_i]
    meta_i = expanded_meta[local_i]
    molecule_i = meta_i[0]
    symmetry_i = meta_i[1]
    image_i = meta_i[2]
    asu_i = meta_i[3]
    cell_row = cells[row]
    l00 = cell_row[0, 0]
    l10 = cell_row[1, 0]
    l11 = cell_row[1, 1]
    l20 = cell_row[2, 0]
    l21 = cell_row[2, 1]
    l22 = cell_row[2, 2]
    inv_n = 1.0 / wp.float32(expanded_atom_count)
    force_i_sum = wp.vec3f(0.0, 0.0, 0.0)
    torque_i_sum = wp.vec3f(0.0, 0.0, 0.0)
    virial_diagonal = wp.vec3f(0.0, 0.0, 0.0)
    virial_upper = wp.vec3f(0.0, 0.0, 0.0)
    total_sum = float(0.0)
    max_sum = float(0.0)

    for dz in range(-radius[2], radius[2] + 1):
        for dy in range(-radius[1], radius[1] + 1):
            for dx in range(radius[0] + 1):
                if not (
                    dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz >= 0)
                ):
                    continue
                target_x = cell_i[0] + dx
                target_y = cell_i[1] + dy
                target_z = cell_i[2] + dz
                cs_x, wc_x = wpdivmod(target_x, dims[0])
                cs_y, wc_y = wpdivmod(target_y, dims[1])
                cs_z, wc_z = wpdivmod(target_z, dims[2])
                cell_index = (
                    cell_offsets[row] + wc_x + dims[0] * (wc_y + dims[1] * wc_z)
                )
                start = cell_atom_start_indices[cell_index]
                count = atoms_per_cell_count[cell_index]
                for cell_atom in range(count):
                    atom_j = cell_atom_list[start + cell_atom]
                    if dx == 0 and dy == 0 and dz == 0 and atom_j <= atom_i:
                        continue
                    local_j = atom_j - row * expanded_atom_count
                    shift_j = atom_periodic_shifts[atom_j]
                    shift = wp.vec3i(
                        cs_x + shift_i[0] - shift_j[0],
                        cs_y + shift_i[1] - shift_j[1],
                        cs_z + shift_i[2] - shift_j[2],
                    )
                    meta_j = expanded_meta[local_j]
                    molecule_j = meta_j[0]
                    symmetry_j = meta_j[1]
                    image_j = meta_j[2]
                    asu_j = meta_j[3]
                    offset_j = image_offsets[atom_j]
                    if image_i == image_j:
                        if (
                            shift[0] - offset_j[0] + offset_i[0] == 0
                            and shift[1] - offset_j[1] + offset_i[1] == 0
                            and shift[2] - offset_j[2] + offset_i[2] == 0
                        ):
                            continue
                        if (
                            shift[0] == 0
                            and shift[1] == 0
                            and shift[2] == 0
                            and asu_i >= asu_j
                        ):
                            continue
                    cutoff = contact_distances[asu_i, asu_j]
                    shift_cart = _lower_tri_cell_shift(
                        shift, l00, l10, l11, l20, l21, l22
                    )
                    vector = positions[atom_j] - pos_i + shift_cart
                    distance_sq = wp.dot(vector, vector)
                    if distance_sq >= cutoff * cutoff:
                        continue
                    distance = wp.sqrt(distance_sq)
                    overlap = cutoff - distance
                    if overlap <= 0.0:
                        continue
                    pair_force = _pair_force(vector, distance, overlap, asu_i, asu_j)
                    force_i = pull_matrices[row, symmetry_i] * (-pair_force)
                    force_j = pull_matrices[row, symmetry_j] * pair_force
                    force_i_sum = force_i_sum + force_i
                    torque_i_sum = torque_i_sum + wp.cross(pos_i_rel, force_i)
                    wp.atomic_add(forces, row, molecule_j, force_j)
                    wp.atomic_add(
                        torques,
                        row,
                        molecule_j,
                        wp.cross(image_rel_asu[atom_j], force_j),
                    )
                    virial_diagonal = virial_diagonal + wp.vec3f(
                        vector[0] * pair_force[0],
                        vector[1] * pair_force[1],
                        vector[2] * pair_force[2],
                    )
                    virial_upper = virial_upper + wp.vec3f(
                        vector[0] * pair_force[1],
                        vector[0] * pair_force[2],
                        vector[1] * pair_force[2],
                    )
                    total_sum = total_sum + overlap
                    max_sum = wp.max(max_sum, overlap)

    if total_sum > 0.0:
        wp.atomic_add(forces, row, molecule_i, force_i_sum)
        wp.atomic_add(torques, row, molecule_i, torque_i_sum)
        wp.atomic_add(total_overlap, row, total_sum)
        wp.atomic_max(max_overlap, row, max_sum)
    wp.atomic_add(
        virial,
        row,
        wp.mat33f(
            virial_diagonal[0] * inv_n,
            virial_upper[0] * inv_n,
            virial_upper[1] * inv_n,
            virial_upper[0] * inv_n,
            virial_diagonal[1] * inv_n,
            virial_upper[2] * inv_n,
            virial_upper[1] * inv_n,
            virial_upper[2] * inv_n,
            virial_diagonal[2] * inv_n,
        ),
    )


def contact_forces(
    *,
    conformer_positions: Tensor,
    conformer_ptr: Tensor,
    conformer_ids: Tensor,
    centers: Tensor,
    rotations: Tensor,
    cells: Tensor,
    inverse_cells: Tensor,
    symmetry_table: Tensor,
    selected_symmetry_ops: Tensor,
    contact_distances: Tensor,
    max_contact_distance: float,
    maps: ContactMaps,
    workspace: ContactWorkspace,
    active_mask: Tensor | None = None,
) -> ContactEvaluation:
    """Build a capacity-checked periodic cell list and accumulate its pairs.

    Requires lower-triangular cell matrices.
    """
    batch, num_molecules = conformer_ids.shape
    if batch != workspace.batch_size or num_molecules != workspace.num_molecules:
        raise ValueError("contact workspace does not match candidate state shape")
    workspace.forces.zero_()
    workspace.torques.zero_()
    workspace.total_overlap.zero_()
    workspace.max_overlap.zero_()
    workspace.virial.zero_()
    active = (
        torch.ones((batch,), dtype=torch.bool, device=cells.device)
        if active_mask is None
        else active_mask.to(dtype=torch.bool, device=cells.device).contiguous()
    )
    pull_matrices = _expanded_geometry(
        conformer_positions=conformer_positions,
        conformer_ptr=conformer_ptr,
        conformer_ids=conformer_ids,
        centers=centers,
        rotations=rotations,
        cells=cells,
        inverse_cells=inverse_cells,
        symmetry_table=symmetry_table,
        selected_symmetry_ops=selected_symmetry_ops,
        maps=maps,
        positions=workspace.positions,
        arms=workspace.image_rel_asu,
        image_offsets=workspace.image_offsets,
    )
    with scoped_warp_stream(cells):
        workspace.build_cell_list(cells, max_contact_distance)
    cells_per_dimension = workspace.cells_per_dimension
    neighbor_search_radius = workspace.neighbor_search_radius
    atom_periodic_shifts = workspace.atom_periodic_shifts
    atom_to_cell_mapping = workspace.atom_to_cell_mapping
    atoms_per_cell_count = workspace.atoms_per_cell_count
    cell_atom_start_indices = workspace.cell_atom_start_indices
    cell_atom_list = workspace.cell_atom_list
    cell_offsets = workspace.cell_offsets
    expanded_count = workspace.expanded_atom_count
    with scoped_warp_stream(cells):
        wp.launch(
            _accumulate_cell_contacts,
            dim=batch * expanded_count,
            inputs=[
                as_warp(workspace.positions, wp.vec3f),
                as_warp(workspace.image_rel_asu, wp.vec3f),
                as_warp(workspace.image_offsets, wp.vec3i),
                as_warp(maps.expanded_meta, wp.vec4i),
                as_warp(cells.contiguous(), wp.mat33f),
                as_warp(pull_matrices, wp.mat33f),
                as_warp(contact_distances.contiguous(), wp.float32),
                as_warp(active, wp.bool),
                as_warp(cells_per_dimension, wp.vec3i),
                as_warp(neighbor_search_radius, wp.vec3i),
                as_warp(atom_periodic_shifts, wp.vec3i),
                as_warp(atom_to_cell_mapping, wp.vec3i),
                as_warp(atoms_per_cell_count, wp.int32),
                as_warp(cell_atom_start_indices, wp.int32),
                as_warp(cell_atom_list, wp.int32),
                as_warp(cell_offsets, wp.int32),
                wp.int32(expanded_count),
                as_warp(workspace.forces, wp.vec3f),
                as_warp(workspace.torques, wp.vec3f),
                as_warp(workspace.total_overlap, wp.float32),
                as_warp(workspace.max_overlap, wp.float32),
                as_warp(workspace.virial, wp.mat33f),
            ],
            device=launch_device(cells),
        )
    return ContactEvaluation(
        forces=workspace.forces,
        torques=workspace.torques,
        total_overlap=workspace.total_overlap,
        max_overlap=workspace.max_overlap,
        virial=workspace.virial,
    )
