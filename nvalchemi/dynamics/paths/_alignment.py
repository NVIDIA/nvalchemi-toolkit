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
"""Rigid-position alignment shared by reaction-path implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

from nvalchemi.dynamics.paths._geometry import minimum_image_displacement

if TYPE_CHECKING:
    from nvalchemi.dynamics.paths._geometry import PreparedMIC


class PositionAlignment(NamedTuple):
    """Result of aligning one or more sets of positions to references.

    Attributes
    ----------
    positions : Tensor
        Aligned mobile positions. For non-periodic systems, these are obtained
        as ``mobile_positions @ rotation.T + translation``. For periodic
        systems, minimum-image positions are translated and ``rotation`` is
        simply identity. Shape ``(n_atoms, 3)``.
    rotation : Tensor
        Proper rotation applied to row-vector positions as
        ``positions @ rotation.T``, shape ``(3, 3)`` or
        ``(num_graphs, 3, 3)`` for a batch. The same transform can be applied
        to vector-valued state such as velocities.
    translation : Tensor
        Common translation applied after rotation, or after periodic images
        have been reconciled, shape ``(3,)`` or ``(num_graphs, 3)`` for a batch.
    """

    positions: Tensor
    rotation: Tensor
    translation: Tensor


def _quaternion_rotation(covariance: Tensor) -> Tensor:
    """Return least-squares proper rotations using quaternion alignment [1]_.

    For each position cross-covariance, this constructs the symmetric ``4 x 4``
    quaternion matrix, selects the eigenvector associated with its largest
    eigenvalue as the optimal quaternion, and converts that quaternion to a
    ``3 x 3`` rotation matrix. A zero covariance is assigned the identity
    rotation.

    The cross-covariance for each graph is ``sum_i (mobile_i - mobile_centroid)
    (reference_i - reference_centroid)^T``, where ``i`` indexes atoms and each
    parenthesized term is a three-dimensional vector.

    Parameters
    ----------
    covariance : Tensor
        Position cross-covariance matrices with shape ``(..., 3, 3)``.

    References
    ----------
    .. [1] M. Melander, K. Laasonen, and H. Jónsson, "Removing External
       Degrees of Freedom from Transition-State Search Methods Using
       Quaternions," *Journal of Chemical Theory and Computation*, 11(3),
       1055-1062 (2015). https://doi.org/10.1021/ct501155k
    """
    # Each component has shape ``covariance.shape[:-2]``, for example
    # ``(num_graphs,)`` for a batch of covariance matrices.
    r11, r12, r13 = covariance[..., 0, :].unbind(dim=-1)
    r21, r22, r23 = covariance[..., 1, :].unbind(dim=-1)
    r31, r32, r33 = covariance[..., 2, :].unbind(dim=-1)
    # Each inner stack has shape ``(..., 4)``; the outer stack is ``(..., 4, 4)``.
    quaternion_matrix = torch.stack(
        (
            torch.stack((r11 + r22 + r33, r23 - r32, r31 - r13, r12 - r21), dim=-1),
            torch.stack((r23 - r32, r11 - r22 - r33, r12 + r21, r13 + r31), dim=-1),
            torch.stack((r31 - r13, r12 + r21, -r11 + r22 - r33, r23 + r32), dim=-1),
            torch.stack((r12 - r21, r13 + r31, r23 + r32, -r11 - r22 + r33), dim=-1),
        ),
        dim=-2,
    )
    # torch.linalg.eigh synchronizes CUDA with the CPU.
    # This batched 4x4 solve therefore introduces a host synchronization
    # on each alignment call.
    quaternion = torch.linalg.eigh(quaternion_matrix).eigenvectors[..., :, -1]
    q0, q1, q2, q3 = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        (
            torch.stack(
                (
                    q0.square() + q1.square() - q2.square() - q3.square(),
                    2 * (q1 * q2 - q0 * q3),
                    2 * (q1 * q3 + q0 * q2),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    2 * (q1 * q2 + q0 * q3),
                    q0.square() - q1.square() + q2.square() - q3.square(),
                    2 * (q2 * q3 - q0 * q1),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    2 * (q1 * q3 - q0 * q2),
                    2 * (q2 * q3 + q0 * q1),
                    q0.square() - q1.square() - q2.square() + q3.square(),
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    return torch.where(
        (covariance.abs().amax(dim=(-2, -1)) == 0)[..., None, None],
        identity,
        rotation,
    )


def _validate_align_batch_positions_inputs(
    reference_positions: Tensor,
    mobile_positions: Tensor,
    batch_idx: Tensor,
    num_atoms_per_graph: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
) -> None:
    """Validate structural inputs not guaranteed by the alignment calculation."""
    # Validate the paired position tensors.
    if reference_positions.ndim != 2 or reference_positions.shape[-1] != 3:
        raise ValueError("reference_positions must have shape (total_atoms, 3)")
    if mobile_positions.shape != reference_positions.shape:
        raise ValueError(
            "mobile_positions must have the same shape as reference_positions"
        )
    if reference_positions.shape[0] == 0:
        raise ValueError("position tensors must contain at least one atom")
    if (
        not reference_positions.is_floating_point()
        or not mobile_positions.is_floating_point()
    ):
        raise TypeError("position tensors must use a floating-point dtype")
    if mobile_positions.dtype != reference_positions.dtype:
        raise ValueError("position tensors must use the same dtype")
    if mobile_positions.device != reference_positions.device:
        raise ValueError("position tensors must be on the same device")

    # Validate packed graph-batch metadata.
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if batch_idx.ndim != 1 or batch_idx.shape[0] != reference_positions.shape[0]:
        raise ValueError("batch_idx must have shape (total_atoms,)")
    if batch_idx.dtype not in integer_dtypes:
        raise TypeError("batch_idx must use an integer dtype")
    if batch_idx.device != reference_positions.device:
        raise ValueError("batch_idx and positions must be on the same device")
    if num_atoms_per_graph.ndim != 1 or num_atoms_per_graph.shape[0] == 0:
        raise ValueError("num_atoms_per_graph must have shape (num_graphs,)")
    if num_atoms_per_graph.dtype not in integer_dtypes:
        raise TypeError("num_atoms_per_graph must use an integer dtype")
    if num_atoms_per_graph.device != reference_positions.device:
        raise ValueError("num_atoms_per_graph and positions must be on the same device")
    if torch.any(num_atoms_per_graph <= 0):
        raise ValueError("every graph must contain at least one atom")

    # Confirm atom memberships match the declared graph sizes.
    num_graphs = num_atoms_per_graph.shape[0]
    if torch.any(batch_idx < 0) or torch.any(batch_idx >= num_graphs):
        raise ValueError("batch_idx values must be in [0, num_graphs)")
    observed_counts = torch.bincount(batch_idx.to(torch.long), minlength=num_graphs)
    if not torch.equal(observed_counts, num_atoms_per_graph.to(torch.long)):
        raise ValueError("num_atoms_per_graph must agree with batch_idx")

    # Validate periodicity flags and cells used by periodic graphs.
    if pbc is not None:
        if pbc.shape != (num_graphs, 3):
            raise ValueError("pbc must have shape (num_graphs, 3)")
        if pbc.dtype != torch.bool:
            raise TypeError("pbc must use the bool dtype")
        if pbc.device != reference_positions.device:
            raise ValueError("pbc and positions must be on the same device")
    if cell is not None:
        if cell.shape != (num_graphs, 3, 3):
            raise ValueError("cell must have shape (num_graphs, 3, 3)")
        if cell.dtype != reference_positions.dtype:
            raise ValueError("cell and positions must use the same dtype")
        if cell.device != reference_positions.device:
            raise ValueError("cell and positions must be on the same device")
    if pbc is not None and torch.any(pbc) and cell is None:
        raise ValueError("A cell is required when any PBC axis is active")


def align_batch_positions(
    reference_positions: Tensor,
    mobile_positions: Tensor,
    batch_idx: Tensor,
    num_atoms_per_graph: Tensor,
    cell: Tensor | None = None,
    pbc: Tensor | None = None,
    fit_mask: Tensor | None = None,
    *,
    prepared_mic: PreparedMIC | None = None,
    skip_extra_checks: bool = False,
) -> PositionAlignment:
    """Align each mobile structure in a batch to its paired reference.

    Both position tensors contain one or more graphs in the same flattened node
    layout. ``batch_idx`` assigns each atom to a graph, and every mobile graph
    is aligned independently to the reference graph with the same index.

    Fully non-periodic graphs are independently centered and
    quaternion-rotated to minimize their unweighted, atom-wise Cartesian RMSD.
    An optional node mask can restrict the atoms used to fit each rigid
    transform, which is still applied to every atom. For graphs with any active
    periodic axis, rotation is disabled: per-atom displacements are mapped to
    their minimum images and the mean displacement of the selected atoms is
    removed. Neither ``reference_positions`` nor ``mobile_positions`` is modified.

    Parameters
    ----------
    reference_positions : Tensor
        Flattened fixed target positions for all graphs, shape
        ``(total_atoms, 3)``.
    mobile_positions : Tensor
        Flattened positions to align using the same graph and atom ordering as
        ``reference_positions``, shape ``(total_atoms, 3)``.
    batch_idx : Tensor
        Integer graph membership for every atom, shaped ``(total_atoms,)``.
    num_atoms_per_graph : Tensor
        Number of atoms in each graph, shaped ``(num_graphs,)``. Every graph
        must contain at least one atom and the counts must agree with
        ``batch_idx``.
    cell : Tensor or None, optional
        Per-graph cell matrices shaped ``(num_graphs, 3, 3)``. Required for
        every graph with an active periodic axis. Selected periodic cell
        vectors must be linearly independent; unused rows may be singular.
    pbc : Tensor or None, optional
        Boolean periodicity flags shaped ``(num_graphs, 3)``. ``None`` means
        every graph is fully non-periodic.
    fit_mask : Tensor or None, optional
        Boolean node mask shaped ``(total_atoms,)``. Selected corresponding atom
        pairs determine the fitted transform for each graph, while that
        transform is applied to every atom. Every graph must select at least one
        atom. ``None`` selects every atom.
    prepared_mic : PreparedMIC or None, optional
        Reusable MIC data prepared for ``cell`` and ``pbc``. Supplying it
        avoids repeating cell-dependent lattice analysis.
    skip_extra_checks : bool, optional
        Skip structural checks for prevalidated batch inputs. The fitting mask
        is always validated. Default is false.

    Returns
    -------
    PositionAlignment
        Aligned flat positions, per-graph rotations shaped
        ``(num_graphs, 3, 3)``, and per-graph translations shaped
        ``(num_graphs, 3)``. Periodic graphs have identity rotations.

    Raises
    ------
    TypeError
        If positions are not floating point, graph metadata is not integer,
        or periodicity flags or ``fit_mask`` are not boolean.
    ValueError
        If shapes, graph membership, dtypes, devices, or periodic inputs are
        incompatible.
    """
    if not skip_extra_checks:
        _validate_align_batch_positions_inputs(
            reference_positions,
            mobile_positions,
            batch_idx,
            num_atoms_per_graph,
            cell,
            pbc,
        )

    num_graphs = num_atoms_per_graph.shape[0]
    graph_idx = batch_idx.to(torch.long)

    # Validate the optional fitting selection.
    if fit_mask is not None:
        if fit_mask.ndim != 1 or fit_mask.shape[0] != reference_positions.shape[0]:
            raise ValueError("fit_mask must have shape (total_atoms,)")
        if fit_mask.dtype != torch.bool:
            raise TypeError("fit_mask must use the bool dtype")
        if fit_mask.device != reference_positions.device:
            raise ValueError("fit_mask and positions must be on the same device")
        fit_counts = torch.zeros(
            num_graphs, dtype=torch.long, device=reference_positions.device
        ).index_add(0, graph_idx, fit_mask.to(torch.long))
        if torch.any(fit_counts == 0):
            raise ValueError("fit_mask must select at least one atom in every graph")

    # Fit an independent rigid transform (translation and rotation) for each graph.
    if fit_mask is None:
        fit_weights = reference_positions.new_ones(reference_positions.shape[0])
        fit_counts = num_atoms_per_graph.to(reference_positions.dtype).unsqueeze(-1)
    else:
        fit_weights = fit_mask.to(reference_positions.dtype)
        fit_counts = reference_positions.new_zeros((num_graphs, 1)).index_add(
            0, graph_idx, fit_weights.unsqueeze(-1)
        )
    # Both centroid tensors have shape ``(num_graphs, 3)``.
    reference_centroid = (
        reference_positions.new_zeros((num_graphs, 3)).index_add(
            0, graph_idx, reference_positions * fit_weights.unsqueeze(-1)
        )
        / fit_counts
    )
    mobile_centroid = (
        mobile_positions.new_zeros((num_graphs, 3)).index_add(
            0, graph_idx, mobile_positions * fit_weights.unsqueeze(-1)
        )
        / fit_counts
    )
    centered_reference = reference_positions - reference_centroid[graph_idx]
    centered_mobile = mobile_positions - mobile_centroid[graph_idx]
    # Per-atom outer products have shape ``(total_atoms, 3, 3)``.
    atom_covariance = (
        centered_mobile.unsqueeze(-1)
        * centered_reference.unsqueeze(-2)
        * fit_weights[:, None, None]
    )
    # Sum the contributions by graph to obtain shape ``(num_graphs, 3, 3)``.
    covariance = reference_positions.new_zeros((num_graphs, 3, 3)).index_add(
        0, graph_idx, atom_covariance
    )
    rotation = _quaternion_rotation(covariance)
    rotated_mobile_centroid = torch.bmm(
        rotation, mobile_centroid.unsqueeze(-1)
    ).squeeze(-1)
    translation = reference_centroid - rotated_mobile_centroid
    # Apply each graph’s fitted rotation and translation to all of its mobile atoms.
    aligned = (
        torch.bmm(rotation[graph_idx], mobile_positions.unsqueeze(-1)).squeeze(-1)
        + translation[graph_idx]
    )

    # Skip periodic image reconciliation when periodic inputs are absent.
    if cell is None or pbc is None:
        return PositionAlignment(aligned, rotation, translation)

    # Reconcile periodic images without rotating periodic graphs.
    periodic_graph = pbc.any(dim=-1)
    identity = torch.eye(
        3, dtype=reference_positions.dtype, device=reference_positions.device
    )
    displacement = minimum_image_displacement(
        mobile_positions - reference_positions,
        graph_idx,
        cell,
        pbc,
        prepared=prepared_mic,
    )
    mean_displacement = (
        reference_positions.new_zeros((num_graphs, 3)).index_add(
            0, graph_idx, displacement * fit_weights.unsqueeze(-1)
        )
        / fit_counts
    )
    periodic_aligned = reference_positions + displacement - mean_displacement[graph_idx]

    aligned = torch.where(
        periodic_graph[graph_idx].unsqueeze(-1), periodic_aligned, aligned
    )
    rotation = torch.where(periodic_graph[:, None, None], identity, rotation)
    translation = torch.where(periodic_graph[:, None], -mean_displacement, translation)
    return PositionAlignment(aligned, rotation, translation)


__all__ = ["PositionAlignment", "align_batch_positions"]
