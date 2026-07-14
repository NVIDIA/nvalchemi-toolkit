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

"""Spatial partitioner for domain decomposition."""

from __future__ import annotations

import math
import warnings

import torch

from nvalchemi.distributed.config import DomainConfig


class SpatialPartitioner:
    """Assigns atoms to spatial sub-domains on a Cartesian grid.

    The partitioner divides the simulation cell into axis-aligned blocks
    and maps each atom to the rank that owns its block.

    Parameters
    ----------
    config : DomainConfig
        Domain decomposition configuration.
    cell_matrix : torch.Tensor
        Cell / box matrix describing the simulation domain.  Accepts
        either the Batch convention ``(1, 3, 3)`` or the raw ``(3, 3)``
        shape; leading batch dimensions are squeezed internally.
    pbc : torch.Tensor
        Periodic boundary conditions per axis.  Accepts either
        ``(1, 3)`` (Batch convention) or ``(3,)``; leading batch
        dimensions are squeezed internally.
    """

    def __init__(
        self,
        config: DomainConfig,
        cell_matrix: torch.Tensor,
        pbc: torch.Tensor,
    ) -> None:
        self.config = config
        # Normalize to (3, 3) and (3,) regardless of whether the caller
        # passed Batch-convention shapes (1, 3, 3) / (1, 3).
        self.cell_matrix = (
            cell_matrix.squeeze(0) if cell_matrix.ndim == 3 else cell_matrix
        )
        self.pbc = pbc.squeeze(0) if pbc.ndim == 2 else pbc

        # Determine world size from mesh or default to 1.
        if config.mesh is not None:
            self.world_size: int = config.mesh.size()
        else:
            self.world_size = 1

        # Compute cells_per_dimension from cell geometry and cutoff.
        if config.grid_dims is not None:
            self.cells_per_dim: tuple[int, int, int] = config.grid_dims
        else:
            self.cells_per_dim = self._compute_cells_per_dim(cell_matrix, config.cutoff)

        # Refine the cell grid if there are fewer cells than ranks.
        total_cells = (
            self.cells_per_dim[0] * self.cells_per_dim[1] * self.cells_per_dim[2]
        )
        if total_cells < self.world_size:
            self.cells_per_dim = SpatialPartitioner.refine_grid_for_ranks(
                self.cells_per_dim, self.world_size
            )

        # Compute the rank grid (Px, Py, Pz).
        self.rank_grid: tuple[int, int, int] = SpatialPartitioner.compute_rank_grid(
            self.cells_per_dim, self.world_size
        )

        # Balance the partition: round each axis's cell count to a multiple of
        # its rank-grid factor so the block assignment gives every rank an
        # equal-width domain. Only for auto-computed grids — an explicit
        # ``grid_dims`` is the user's deliberate choice and is left untouched.
        if config.grid_dims is None:
            self.cells_per_dim = SpatialPartitioner.balance_cells_for_ranks(
                self.cells_per_dim, self.rank_grid
            )

        # Precompute neighbor ranks for every rank.
        self._neighbor_ranks: dict[int, list[int]] = self._compute_all_neighbor_ranks()

        # Precompute the cell-matrix inverse used by ``assign_atoms_to_ranks``.
        # The cell is fixed at construction (NVT/NVE), so caching avoids a
        # per-step 3x3 inversion in the hot path. The per-call
        # ``.to(device=, dtype=)`` is a no-op when device+dtype already match.
        self._inv_cell: torch.Tensor = torch.linalg.inv(self.cell_matrix)

    def update_cell(self, cell_matrix: torch.Tensor) -> None:
        """Refresh the physical cell when a barostat (NPT/NPH) deforms the box.

        Recomputes ``cell_matrix`` and its cached inverse; the fractional cell
        grid (``cells_per_dim``) and rank layout are intentionally kept fixed so
        rank assignment stays consistent as the box scales — only the physical
        size of each grid cell changes. Halo regions (``rank_to_cell_bounds`` →
        cartesian via ``cell_matrix``) and the fractional ghost width scale with
        the updated cell automatically. Without this, the partitioner keeps the
        partition-time box: as the cell grows, wrapped positions fall outside it
        (fractional coords ≥ 1) and ``assign_atoms_to_ranks`` misroutes atoms.

        Note: this handles cell *scaling* (the fix for barostat expansion). Large
        *contraction* additionally needs the cell grid / neighbor-rank set to
        adapt (the ghost region spans more grid cells as the box shrinks); that
        adaptive-grid work is not covered here.
        """
        cm = cell_matrix.squeeze(0) if cell_matrix.ndim == 3 else cell_matrix
        self.cell_matrix = cm.detach()
        self._inv_cell = torch.linalg.inv(self.cell_matrix)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cells_per_dim(
        cell_matrix: torch.Tensor, cutoff: float
    ) -> tuple[int, int, int]:
        """Compute cells per dimension from cell geometry and cutoff.

        Uses the nvalchemiops formula:
            face_distance = 1.0 / norm(inverse_cell_T[dim])
            cells = max(floor(face_distance / cutoff), 1)
        """
        # Normalise to a strict 2D ``(3, 3)`` so torch's ``x.T``
        # deprecation warning (fires on any non-2D tensor) doesn't
        # trip — callers may pass ``(1, 3, 3)`` (Batch convention).
        if cell_matrix.ndim > 2:
            cell_matrix = cell_matrix.squeeze(0)
        inv_cell = torch.linalg.inv(cell_matrix)
        inv_cell_T = inv_cell.mT  # (3, 3) — same as .T for 2D, deprecation-free

        dims: list[int] = []
        for dim in range(3):
            face_distance = 1.0 / torch.linalg.norm(inv_cell_T[dim]).item()
            dims.append(max(int(math.floor(face_distance / cutoff)), 1))
        return (dims[0], dims[1], dims[2])

    @staticmethod
    def compute_rank_grid(
        cells_per_dim: tuple[int, int, int], world_size: int
    ) -> tuple[int, int, int]:
        """Compute ``(Px, Py, Pz)`` rank grid minimizing surface area.

        Enumerates all 3-factor factorizations of *world_size* and picks the
        one that minimizes the surface-area proxy
        ``2 * (dx*dy + dy*dz + dx*dz)`` where ``dx = Nx/Px``, etc.
        """
        Nx, Ny, Nz = cells_per_dim

        best_grid: tuple[int, int, int] | None = None
        best_surface = float("inf")

        for Px in range(1, world_size + 1):
            if world_size % Px != 0:
                continue
            remainder = world_size // Px
            for Py in range(1, remainder + 1):
                if remainder % Py != 0:
                    continue
                Pz = remainder // Py

                dx = Nx / Px
                dy = Ny / Py
                dz = Nz / Pz
                surface = 2.0 * (dx * dy + dy * dz + dx * dz)

                if surface < best_surface:
                    best_surface = surface
                    best_grid = (Px, Py, Pz)

        if best_grid is None:
            raise ValueError("No valid factorization found")
        return best_grid

    @staticmethod
    def refine_grid_for_ranks(
        cells_per_dim: tuple[int, int, int], world_size: int
    ) -> tuple[int, int, int]:
        """Subdivide cells until there are at least *world_size* cells.

        Doubles the smallest dimension iteratively. Warns if total cells
        remain less than *world_size* after 64 iterations (safety cap).
        """
        Nx, Ny, Nz = cells_per_dim
        max_iters = 64
        for _ in range(max_iters):
            if Nx * Ny * Nz >= world_size:
                break
            # Double the smallest dimension.
            min_val = min(Nx, Ny, Nz)
            if Nx == min_val:
                Nx *= 2
            elif Ny == min_val:
                Ny *= 2
            else:
                Nz *= 2

        if Nx * Ny * Nz < world_size:
            warnings.warn(
                f"Could not refine cell grid to {world_size} cells; "
                f"got {Nx * Ny * Nz} cells with grid ({Nx}, {Ny}, {Nz}).",
                stacklevel=2,
            )
        return (Nx, Ny, Nz)

    @staticmethod
    def balance_cells_for_ranks(
        cells_per_dim: tuple[int, int, int], rank_grid: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """Round each axis's cell count to a multiple of its rank-grid factor.

        The cell->rank block assignment (``cx = ceil(Nx / Px)``) gives the first
        ranks ``cx`` cells and the last rank the remainder, so when ``Nx`` is
        not a multiple of ``Px`` the domains are unequal — e.g. 3 cells across
        2 ranks splits 2:1. Making ``Nx`` a multiple of ``Px`` gives every rank
        ``Nx / Px`` equal-width cells.

        Rounds DOWN to the nearest multiple of ``Pi`` (floored at ``Pi`` so each
        rank keeps at least one cell). Rounding down — never up — keeps each
        cell at least as wide as the cutoff-derived size, so cells stay
        ``>= cutoff``. Axes with a single rank (``Pi == 1``) are unchanged.
        """
        out: list[int] = []
        for n_i, p_i in zip(cells_per_dim, rank_grid):
            out.append(n_i if p_i <= 1 else max(p_i, (n_i // p_i) * p_i))
        return (out[0], out[1], out[2])

    # ------------------------------------------------------------------
    # Cell ↔ rank mapping
    # ------------------------------------------------------------------

    def cell_to_rank(
        self, ix: int | torch.Tensor, iy: int | torch.Tensor, iz: int | torch.Tensor
    ) -> int | torch.Tensor:
        """Map cell indices ``(ix, iy, iz)`` to the owning rank.

        Works with both scalar ints and batched :class:`torch.Tensor` inputs.
        Uses ceiling-division block assignment.
        """
        Nx, Ny, Nz = self.cells_per_dim
        Px, Py, Pz = self.rank_grid

        cx = math.ceil(Nx / Px)
        cy = math.ceil(Ny / Py)
        cz = math.ceil(Nz / Pz)

        if isinstance(ix, torch.Tensor):
            rx = torch.clamp(ix // cx, max=Px - 1)
            ry = torch.clamp(iy // cy, max=Py - 1)
            rz = torch.clamp(iz // cz, max=Pz - 1)
            return rx + Px * (ry + Py * rz)
        else:
            rx = min(ix // cx, Px - 1)
            ry = min(iy // cy, Py - 1)
            rz = min(iz // cz, Pz - 1)
            return rx + Px * (ry + Py * rz)

    def keeps_owner(
        self,
        positions: "torch.Tensor",
        owner_rank: int,
        hysteresis: float,
    ) -> "torch.Tensor":
        """Hysteresis-aware ownership test.

        An atom keeps its owner until it drifts more than ``hysteresis``
        (Cartesian Angstrom) past the owner's domain boundary — i.e. it stays
        while inside the owner's domain expanded by ``hysteresis`` on every axis
        (PBC-wrapped on periodic axes). This stops the per-step migration
        thrashing of atoms that merely vibrate across a domain plane.

        Parameters
        ----------
        positions : torch.Tensor
            ``[N, 3]`` atom positions in Cartesian coordinates.
        owner_rank : int
            The rank whose ownership is being tested.
        hysteresis : float
            Cartesian margin (Angstrom) an atom must exceed past the boundary
            before it loses its owner.

        Returns
        -------
        torch.Tensor
            ``[N]`` bool, True where an atom currently owned by ``owner_rank``
            should keep that owner.

        Notes
        -----
        Correctness relies on ``hysteresis <= skin / 2`` (enforced by
        ``DomainConfig``): a deferred atom plus inter-rebuild drift stays within
        the owner's halo (``ghost_width = cutoff + skin``), so the owner still
        has all the atom's neighbors and the neighbor still ghosts the atom.
        Uses the same fractional-bounds + reciprocal-norm geometry as the halo
        ghost region it must stay inside.
        """
        import torch  # noqa: PLC0415

        device, dtype = positions.device, positions.dtype
        inv = self._inv_cell.to(device=device, dtype=dtype)
        frac = positions @ inv  # (N, 3) fractional coords (row-vector convention)
        cells = torch.tensor(self.cells_per_dim, device=device, dtype=dtype)
        lo_cell, hi_cell = self.rank_to_cell_bounds(owner_rank)
        frac_lo = torch.tensor(lo_cell, device=device, dtype=dtype) / cells  # (3,)
        frac_hi = torch.tensor(hi_cell, device=device, dtype=dtype) / cells  # (3,)
        # Cartesian hysteresis -> fractional per axis: |reciprocal vector| are the
        # rows of inv(cell).T (matches _ghost_width_fractional).
        norms = torch.linalg.norm(inv.T, dim=1)  # (3,)
        h_frac = float(hysteresis) * norms  # (3,)
        a = frac_lo - h_frac  # expanded-domain lower bound per axis
        b = frac_hi + h_frac  # expanded-domain upper bound per axis
        pbc = self.pbc.to(device=device)
        keep = torch.ones(frac.shape[0], dtype=torch.bool, device=device)
        for d in range(3):
            fd = frac[:, d]
            if bool(pbc[d]):
                # Membership of f (mod 1) in [a, b] on the unit circle; the band
                # width (hi-lo + 2*hysteresis) is < 1, so test f, f-1, f+1.
                fm = fd - torch.floor(fd)
                in_d = (
                    ((fm >= a[d]) & (fm <= b[d]))
                    | ((fm - 1.0 >= a[d]) & (fm - 1.0 <= b[d]))
                    | ((fm + 1.0 >= a[d]) & (fm + 1.0 <= b[d]))
                )
            else:
                in_d = (fd >= a[d]) & (fd <= b[d])
            keep = keep & in_d
        return keep

    def rank_to_cell_bounds(
        self, rank: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return cell index bounds ``(lo, hi)`` owned by *rank*.

        ``lo`` is inclusive, ``hi`` is exclusive.
        """
        Nx, Ny, Nz = self.cells_per_dim
        Px, Py, Pz = self.rank_grid

        rx, ry, rz = self.rank_to_grid_coords(rank)

        cx = math.ceil(Nx / Px)
        cy = math.ceil(Ny / Py)
        cz = math.ceil(Nz / Pz)

        lo = (rx * cx, ry * cy, rz * cz)
        hi = (min((rx + 1) * cx, Nx), min((ry + 1) * cy, Ny), min((rz + 1) * cz, Nz))
        return lo, hi

    def rank_to_grid_coords(self, rank: int) -> tuple[int, int, int]:
        """Decompose a linear rank index into ``(rx, ry, rz)`` grid coords."""
        Px, Py, _Pz = self.rank_grid
        rx = rank % Px
        ry = (rank // Px) % Py
        rz = rank // (Px * Py)
        return (rx, ry, rz)

    # ------------------------------------------------------------------
    # Neighbor ranks
    # ------------------------------------------------------------------

    def _compute_all_neighbor_ranks(self) -> dict[int, list[int]]:
        """Precompute the set of neighbor ranks for every rank."""
        Px, Py, Pz = self.rank_grid
        total_ranks = Px * Py * Pz
        neighbor_map: dict[int, list[int]] = {}
        for rank in range(total_ranks):
            neighbor_map[rank] = self._compute_neighbor_ranks_for(rank)
        return neighbor_map

    def _compute_neighbor_ranks_for(self, rank: int) -> list[int]:
        """Return up to 26 spatial neighbor ranks for *rank*.

        For PBC dimensions, wrap around. For non-PBC, skip out-of-bounds.
        """
        Px, Py, Pz = self.rank_grid
        rx, ry, rz = self.rank_to_grid_coords(rank)
        pbc_x = bool(self.pbc[0])
        pbc_y = bool(self.pbc[1])
        pbc_z = bool(self.pbc[2])

        neighbors: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx = rx + dx
                    ny = ry + dy
                    nz = rz + dz

                    # Check bounds / wrap for each dimension.
                    if not self._in_bounds_or_wrap(nx, Px, pbc_x):
                        continue
                    if not self._in_bounds_or_wrap(ny, Py, pbc_y):
                        continue
                    if not self._in_bounds_or_wrap(nz, Pz, pbc_z):
                        continue

                    nx = nx % Px
                    ny = ny % Py
                    nz = nz % Pz

                    neighbor_rank = nx + Px * (ny + Py * nz)
                    # Exclude self (can happen when PBC wraps a dimension
                    # that has only 1 rank, e.g. rank_grid (1, 1, 2) with
                    # full PBC — dx=±1 along Px=1 wraps back to self).
                    if neighbor_rank != rank and neighbor_rank not in neighbors:
                        neighbors.append(neighbor_rank)

        return neighbors

    @staticmethod
    def _in_bounds_or_wrap(coord: int, size: int, periodic: bool) -> bool:
        """Check if a neighbor coordinate is valid, considering PBC."""
        if 0 <= coord < size:
            return True
        if periodic:
            return True
        return False

    def get_neighbor_ranks(self, rank: int) -> list[int]:
        """Return precomputed neighbor ranks for *rank*."""
        return self._neighbor_ranks[rank]

    # ------------------------------------------------------------------
    # Atom assignment (vectorized)
    # ------------------------------------------------------------------

    def assign_atoms_to_ranks(self, positions: torch.Tensor) -> torch.Tensor:
        """Assign each atom to a rank based on its position.

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` atom positions in Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            ``(N,)`` integer tensor of rank assignments.
        """
        device = positions.device
        dtype = positions.dtype

        # Fractional coordinates. ``cart = frac @ cell_matrix`` (rows of
        # cell_matrix = lattice vectors), so ``frac = cart @ inv(cell_matrix)``
        # — not ``inv(cell).T``, which gives wrong fractional coords on skew
        # cells (hex / triclinic) and mis-assigns boundary atoms.
        inv_cell = self._inv_cell.to(device=device, dtype=dtype)
        frac = positions @ inv_cell  # (N, 3)

        cells_per_dim_t = torch.tensor(self.cells_per_dim, device=device, dtype=dtype)

        # Cell coordinates.
        cell_coords = torch.floor(frac * cells_per_dim_t).to(torch.int64)

        # PBC wrap for periodic dimensions; clamp for non-periodic.
        cells_per_dim_int = torch.tensor(
            self.cells_per_dim, device=device, dtype=torch.int64
        )
        pbc_mask = self.pbc.to(device=device)

        # Wrap periodic dims via modulo.
        wrapped = cell_coords % cells_per_dim_int
        # Clamp non-periodic dims.
        clamped = torch.clamp(
            cell_coords,
            min=torch.zeros_like(cells_per_dim_int),
            max=cells_per_dim_int - 1,
        )
        # Select based on pbc mask.
        cell_coords = torch.where(pbc_mask.unsqueeze(0), wrapped, clamped)

        # Vectorized cell_to_rank.
        Nx, Ny, Nz = self.cells_per_dim
        Px, Py, Pz = self.rank_grid

        cx = math.ceil(Nx / Px)
        cy = math.ceil(Ny / Py)
        cz = math.ceil(Nz / Pz)

        rx = torch.clamp(cell_coords[:, 0] // cx, max=Px - 1)
        ry = torch.clamp(cell_coords[:, 1] // cy, max=Py - 1)
        rz = torch.clamp(cell_coords[:, 2] // cz, max=Pz - 1)

        ranks = rx + Px * (ry + Py * rz)
        return ranks


class IndexPartitioner:
    """Assigns atoms to ranks by contiguous, count-balanced index ranges.

    A geometry-free alternative to :class:`SpatialPartitioner`: atom ``i`` is
    owned by the rank holding its slice of ``arange(N)``, split into ``W``
    contiguous chunks with the remainder spread over the low ranks. Every rank
    neighbors every other (no spatial locality), so a decomposition built on this
    partitioner exchanges across the whole mesh rather than a boundary shell.
    """

    def __init__(self, config: DomainConfig) -> None:
        self.config = config
        self.world_size: int = config.mesh.size() if config.mesh is not None else 1

    def get_neighbor_ranks(self, rank: int) -> list[int]:
        return [r for r in range(self.world_size) if r != rank]

    def assign_atoms_to_ranks(self, positions: torch.Tensor) -> torch.Tensor:
        n = positions.shape[0]
        counts = self._owned_counts(n)
        return torch.repeat_interleave(
            torch.arange(self.world_size, device=positions.device),
            torch.tensor(counts, device=positions.device),
        )

    def _owned_counts(self, n: int) -> list[int]:
        w = self.world_size
        base, rem = divmod(n, w)
        return [base + (1 if r < rem else 0) for r in range(w)]
