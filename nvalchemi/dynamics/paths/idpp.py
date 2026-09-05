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
"""Image-dependent pair-potential inputs and model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.level_storage import SegmentedLevelStorage
from nvalchemi.dynamics.paths._geometry import minimum_image_displacement
from nvalchemi.dynamics.paths.validate import validate_paths
from nvalchemi.models.base import BaseModelMixin, ModelConfig

_TARGET_DISTANCE_KEY = "idpp_target_distances"


def _pair_displacements(
    positions: Tensor,
    pair_indices: Tensor,
    pair_batch: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
) -> Tensor:
    """Return pair displacements with batched minimum-image wrapping.

    Parameters
    ----------
    positions : Tensor
        Cartesian atom positions with shape ``(N, 3)``.
    pair_indices : Tensor
        Global atom indices for each pair with shape ``(E, 2)``. Each row
        contains the source and destination indices ``(i, j)``.
    pair_batch : Tensor
        Image index for each pair with shape ``(E,)``.
    cell : Tensor or None
        Cell vectors for every image with shape ``(B, 3, 3)``. ``None``
        denotes nonperiodic geometry.
    pbc : Tensor or None
        Periodic-boundary flags for every image with shape ``(B, 3)``.
        ``None`` denotes nonperiodic geometry.

    Returns
    -------
    Tensor
        Minimum-image Cartesian displacements from atom ``i`` to atom ``j`` with
        shape ``(E, 3)``.
    """
    displacements = positions[pair_indices[:, 1]] - positions[pair_indices[:, 0]]
    return minimum_image_displacement(displacements, pair_batch, cell, pbc)


def prepare_idpp_targets(paths: Batch) -> Batch:
    """Build full pairwise IDPP target distances for a batch of reaction paths.

    IDPP (image dependent pair potential) initialization needs a target
    interatomic distance for every atom pair in every image. This function
    replaces whatever edge/neighbor list ``paths`` currently has with a complete
    upper-triangle pair list per image, and assigns each pair a target distance
    linearly interpolated between that pair's distance in the path's first image
    and its distance in the path's last image.

    ``paths`` is a :class:`Batch` of one or more reaction paths, where each
    path is a group of images (``group_layout``) and each image is one
    graph in that group. All images within a path must share the same
    atoms and ordering (enforced by :func:`validate_paths`) so that pair
    ``(i, j)`` refers to the same atoms across every image of the path.

    Parameters
    ----------
    paths : Batch
        Grouped reaction paths, where each group is a path and each graph
        within a group is one image along that path, with corresponding
        atoms in every image.

    Returns
    -------
    Batch
        ``paths``, mutated in place: its edge storage now holds the full
        pair ``neighbor_list`` and the interpolated ``idpp_target_distances``
        for every image.
    """
    validate_paths(paths)

    layout = paths.group_layout
    node_ptr = paths.batch_ptr.long()  # [B + 1]
    node_image = paths.batch_idx.long()  # [N]
    num_atoms = paths.num_nodes_per_graph.long()  # [B]
    nodes = torch.arange(paths.num_nodes, device=paths.device)  # [N]
    node_rank = nodes - node_ptr[node_image]  # [N]

    # Enumerate each image's complete upper-triangle pair list.
    # For one three-atom image with global nodes [0, 1, 2]:
    #   repeats     = [2, 1, 0]  -> pair_i      = [0, 0, 1]
    #   pair_offset = [0, 1, 0]  -> pair_j      = [1, 2, 2]
    #   pair_indices = [[0, 1], [0, 2], [1, 2]]
    # Later images produce the same local pairs shifted to their global node indices.
    repeats = num_atoms[node_image] - node_rank - 1  # [N]
    pair_i = torch.repeat_interleave(nodes, repeats)  # [E]
    pair_i_starts = repeats.cumsum(0) - repeats  # [N]
    pair_offset = torch.arange(pair_i.numel(), device=paths.device) - (
        torch.repeat_interleave(pair_i_starts, repeats)
    )  # [E]
    pair_j = pair_i + pair_offset + 1  # [E]
    pair_indices = torch.stack((pair_i, pair_j), dim=-1)  # [E, 2]

    # Prepare quantities needed to locate each pair's path and image, and each path's
    # last-image rank - used below to select the two endpoint images
    pair_counts = num_atoms * (num_atoms - 1) // 2  # [B]
    pair_ptr = torch.cat((pair_counts.new_zeros(1), pair_counts.cumsum(0)))  # [B + 1]
    pair_image = node_image[pair_i]  # [E]
    pair_rank = (
        torch.arange(pair_i.numel(), device=paths.device) - pair_ptr[pair_image]
    )  # [E]
    pair_path = layout.group_idx[pair_image]  # [E]
    image_rank = layout.graph_rank[pair_image]  # [E]
    last_image_rank = layout.num_graphs_per_group[pair_path] - 1  # [E]

    # Measure pair distances in the initial and final path images.
    # E_init = E_final = number of edges in the initial or the final image.
    initial_pair_mask = image_rank == 0  # [E]
    initial_endpoint_pairs = pair_indices[initial_pair_mask]  # [E_init, 2]
    initial_endpoint_pair_image = pair_image[initial_pair_mask]  # [E_init]
    initial_endpoint_displacements = _pair_displacements(
        paths.positions,
        initial_endpoint_pairs,
        initial_endpoint_pair_image,
        paths.cell if "cell" in paths else None,
        paths.pbc if "pbc" in paths else None,
    )  # [E_init, 3]
    initial_endpoint_distances = torch.linalg.vector_norm(
        initial_endpoint_displacements, dim=-1
    )  # [E_init]

    final_pair_mask = image_rank == last_image_rank  # [E]
    final_endpoint_pairs = pair_indices[final_pair_mask]  # [E_final, 2]
    final_endpoint_pair_image = pair_image[final_pair_mask]  # [E_final]
    final_endpoint_displacements = _pair_displacements(
        paths.positions,
        final_endpoint_pairs,
        final_endpoint_pair_image,
        paths.cell if "cell" in paths else None,
        paths.pbc if "pbc" in paths else None,
    )  # [E_final, 3]
    final_endpoint_distances = torch.linalg.vector_norm(
        final_endpoint_displacements, dim=-1
    )  # [E_final]

    # Map each per-image atom pair to the corresponding atom pair at both endpoints.
    pairs_per_path = pair_counts[layout.group_ptr[:-1]]  # [P]
    path_pair_ptr = torch.cat(
        (pairs_per_path.new_zeros(1), pairs_per_path.cumsum(0))
    )  # [P + 1]
    pair_endpoint_index = path_pair_ptr[pair_path] + pair_rank  # [E]
    initial_distances = initial_endpoint_distances[pair_endpoint_index]  # [E]
    final_distances = final_endpoint_distances[pair_endpoint_index]  # [E]

    # Linearly interpolate each target according to the image's position in its path.
    fractions = image_rank.to(paths.positions.dtype) / last_image_rank.to(
        paths.positions.dtype
    )
    target_distances = torch.lerp(initial_distances, final_distances, fractions)

    segment_lengths = pair_counts.to(torch.int32)
    paths._storage.groups["edges"] = SegmentedLevelStorage(
        data={
            "neighbor_list": pair_indices.to(torch.int32),
            _TARGET_DISTANCE_KEY: target_distances,
        },
        device=paths.device,
        segment_lengths=segment_lengths,
        validate=False,
    )
    if paths.keys is not None:
        paths.keys["edge"] = {"neighbor_list", _TARGET_DISTANCE_KEY}
    # Full pair edges do not use a neighbor-list cutoff.
    if hasattr(paths, "_neighbor_list_cutoff"):
        object.__delattr__(paths, "_neighbor_list_cutoff")
    return paths


class IDPPModel(nn.Module, BaseModelMixin):
    r"""Evaluate the image-dependent pair potential for reaction-path images.

    Linear Cartesian interpolation can place atoms unnaturally close together
    or produce uneven changes in interatomic distances. IDPP provides a better
    initial guess for reaction-path calculations, such as NEB, by relaxing the
    interior images toward smoothly varying pair distances before evaluating
    them with a physical potential [1]_.

    For image fraction :math:`s \in [0, 1]`, the target distance for atoms
    :math:`i` and :math:`j` is interpolated between the endpoint distances:

    .. math::

        d_{ij}^{\mathrm{target}}(s)
        = (1-s)d_{ij}^{(0)} + s d_{ij}^{(1)}.

    The artificial IDPP energy for an image is

    .. math::

        E_{\mathrm{IDPP}}
        = \sum_{i<j}
          \frac{\left(d_{ij} - d_{ij}^{\mathrm{target}}\right)^2}{d_{ij}^4},

    which weights close atom pairs more strongly. Forces are the negative
    coordinate gradient, :math:`\mathbf{F}_i = -\partial E_{\mathrm{IDPP}} /
    \partial \mathbf{r}_i`.

    Call :func:`prepare_idpp_targets` on each grouped path batch before
    evaluation. Pair displacements use minimum-image wrapping for periodic images.
    IDPP energies are optimization objectives, not physical potential energies.

    References
    ----------
    .. [1] S. Smidstrup, A. Pedersen, K. Stokbro, and H. Jónsson,
       "Improved initial guess for minimum energy path calculations,"
       *The Journal of Chemical Physics*, 140, 214106 (2014).
       `doi:10.1063/1.4878664 <https://doi.org/10.1063/1.4878664>`_.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"neighbor_list", _TARGET_DISTANCE_KEY}),
            optional_inputs=frozenset({"cell", "pbc"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes because IDPP has no learned features."""
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Raise because the analytic IDPP model has no embeddings."""
        raise NotImplementedError("IDPPModel does not provide embeddings")

    def direct_derivative_keys(self) -> set[str]:
        """Return outputs computed analytically by the IDPP model.

        Returns
        -------
        set[str]
            The analytic force output.
        """
        return {"forces"}

    def forward(self, data: Batch, **kwargs: Any) -> ModelOutputs:
        """Compute active IDPP outputs for every image.

        Parameters
        ----------
        data : Batch
            Complete grouped paths prepared by :func:`prepare_idpp_targets`.

        Returns
        -------
        ModelOutputs
            Active per-image energies and per-atom forces.

        Raises
        ------
        TypeError
            If ``data`` is not a :class:`Batch`.
        KeyError
            If ``data`` has not been prepared with IDPP target distances.
        RuntimeError
            If any distinct atom pair is coincident within machine precision.
        """
        if _TARGET_DISTANCE_KEY not in data:
            raise KeyError("IDPPModel requires paths prepared by prepare_idpp_targets")

        inputs = self.adapt_input(data, **kwargs)
        pair_indices = inputs["neighbor_list"].long()
        target_distances = inputs[_TARGET_DISTANCE_KEY]

        positions = inputs["positions"]
        num_graphs = data.num_graphs
        pair_batch = data.batch_idx.index_select(0, pair_indices[:, 0]).long()

        with torch.no_grad():
            displacements = _pair_displacements(
                positions,
                pair_indices,
                pair_batch,
                inputs.get("cell"),
                inputs.get("pbc"),
            )
            distances = torch.linalg.vector_norm(displacements, dim=-1)
            minimum_distance = torch.finfo(positions.dtype).eps
            # This guards against overlaps introduced unexpectedly during optimization
            # without synchronizing every evaluation.
            torch._assert_async(
                torch.all(distances > minimum_distance),
                "IDPP cannot evaluate coincident atom pairs; perturb overlapping "
                "atoms before optimization",
            )
            safe_distances = distances.clamp_min(minimum_distance)
            distance_errors = distances - target_distances
            inverse_distances = safe_distances.reciprocal()
            model_output: dict[str, Tensor] = {}

            if "energy" in self.model_config.active_outputs:
                pair_energies = distance_errors.square() * inverse_distances.pow(4)
                accumulation_dtype = (
                    torch.float64
                    if pair_energies.dtype == torch.float32
                    else pair_energies.dtype
                )
                energy = torch.zeros(
                    num_graphs,
                    dtype=accumulation_dtype,
                    device=positions.device,
                )
                energy.scatter_add_(0, pair_batch, pair_energies.to(accumulation_dtype))
                model_output["energy"] = energy.to(positions.dtype).unsqueeze(-1)

            if "forces" in self.model_config.active_outputs:
                pair_force_scale = (
                    2
                    * distance_errors
                    * (2 * target_distances - distances)
                    * inverse_distances.pow(6)
                )
                pair_forces = pair_force_scale.unsqueeze(-1) * displacements
                forces = torch.zeros_like(positions)
                forces.index_add_(0, pair_indices[:, 0], pair_forces)
                forces.index_add_(0, pair_indices[:, 1], -pair_forces)
                model_output["forces"] = forces

        return self.adapt_output(model_output, data)


__all__ = ["IDPPModel", "prepare_idpp_targets"]
