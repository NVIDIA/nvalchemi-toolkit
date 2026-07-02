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

"""Fixed-shape graph padding for compiled domain-decomposed MD.

Under ``torch.compile``, a domain-decomposed model's graph must keep a stable
shape across MD steps, or every step that changes the owned+ghost atom / edge
count triggers a recompile. The fix is to pad each step's graph to fixed
per-rank capacities with inert dead atoms / dead edges that contribute nothing.

This module owns two pieces of that mechanism:

* :class:`GraphPadder` — the protocol a model declares (via
  ``CompilePolicy(graph_padder=...)``) for *how* its graph representation is
  padded to a capacity and stripped back. The framework owns *when* to pad; the
  padder owns the representation-specific ``pad`` / ``unpad``. Built-ins cover
  common representations (COO ``edge_index``, dense ``(N, K)`` neighbor matrix)
  so most models declare nothing.
* :func:`resolve_cap` — the shared grow-only capacity policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = ["COOPadder", "DenseBatchPadder", "GraphPadder", "resolve_cap"]


@runtime_checkable
class GraphPadder(Protocol):
    """How a model's graph representation is padded to a fixed capacity.

    Declared on ``CompilePolicy.graph_padder``; the default is :class:`COOPadder`.
    :meth:`pad` runs before the compiled forward and :meth:`unpad` on the raw
    output, so the wrapper stays distribution-agnostic and the compiled graph
    shape stays constant.

    The padder owns capacity resolution: it is handed a mutable ``cap_state``
    dict (persistent across MD steps) and sizes its own caps with
    :func:`resolve_cap`. This matters because *when* a capacity becomes knowable
    is representation-specific — a COO graph knows its atom/edge counts up front,
    but a model that rebuilds its graph inside :meth:`pad` only learns its edge
    count partway through.

    Padding must be inert: dead atoms carry no contribution (e.g. ``Z=0``, parked
    beyond any cutoff) and dead edges have a zero envelope (e.g. a self-loop
    longer than the cutoff, or a non-degenerate image so spherical harmonics
    don't ``NaN``). The owned-only output consolidation drops the dead rows
    regardless, but they must not perturb the real atoms' values.
    """

    def pad(self, data: Any, cap_state: dict[str, int], cap_atoms: bool = True) -> Any:
        """Return ``data`` padded to fixed per-rank capacities.

        Resolve the capacities with :func:`resolve_cap` against ``cap_state``
        (grow-only, persistent across steps), then pad ``data`` and return it.

        ``cap_atoms`` (strategy-supplied) selects whether the atom dim is capped
        too or only edges: halo caps both (owned+ghost fluctuate); a graph-parallel
        node partition caps edges only (fixed atom set). Padders whose layout only
        ever caps atoms may ignore it.
        """
        ...

    def unpad(self, output: Any, n_real: int | None = None) -> Any:
        """Drop the dead-atom / dead-edge rows from a raw model output.

        ``n_real`` overrides the real row count to strip to; pass it on paths
        where no :meth:`pad` ran (e.g. eager / sharded), else leave it ``None``
        to use the count the matching :meth:`pad` recorded.
        """
        ...


def resolve_cap(
    state: dict[str, int],
    key: str,
    real: int,
    *,
    initial_factor: float,
    grow_factor: float = 1.30,
    stride: int = 16,
    extra: int = 0,
    strict_gt: bool = True,
) -> int:
    """Grow-only fixed-shape capacity for ``key``, ``>= real + extra``.

    A cap is sized on first sight with ``initial_factor`` headroom, regrows with
    ``grow_factor`` only when the real count would overflow, and is always
    rounded up to a multiple of ``stride`` so small MD-step fluctuation lands in
    the same bucket — keeping the compiled graph from recompiling. The cap only
    ever grows, so a hot path reuses one compiled graph.

    Parameters
    ----------
    state
        Mutable dict holding the persistent caps across forwards (the caller
        owns its lifetime).
    key
        Which capacity (e.g. ``"atoms"`` / ``"edges"`` / ``"max_send"``).
    real
        The real count needed this step (before padding).
    initial_factor
        Headroom multiplier applied the first time ``key`` is sized, set to cover
        the equilibrated peak from the first compile (e.g. edges climb ~25%
        through equilibration, atoms barely move).
    grow_factor
        Headroom multiplier applied on a later overflow.
    stride
        Bucket size; the cap is rounded up to a multiple of this (16 for
        kernel-friendly shapes; coarser counts that swing more use a larger one).
    extra
        Slots reserved beyond ``real`` (e.g. UMA reserves 2 for the dead-edge
        anchor pair).
    strict_gt
        Overflow test: ``real + extra > cap`` when True (the default; edges /
        send), or ``>= cap`` when False (atoms, which need a strictly-larger cap
        because the dead row sits at ``cap - 1``).

    Returns
    -------
    int
        The (possibly grown) capacity for ``key``, recorded in ``state``.
    """
    need = real + extra

    def _bucket(x: int, factor: float) -> int:
        return ((int(x * factor) + 1 + stride - 1) // stride) * stride

    cap = state.get(key)
    if cap is None:
        cap = _bucket(need, initial_factor)
        state[key] = cap
        return cap
    overflow = need > cap if strict_gt else need >= cap
    if overflow:
        cap = max(stride, _bucket(need, grow_factor))
        state[key] = cap
    return cap


class COOPadder:
    """Built-in :class:`GraphPadder` for COO ``edge_index`` graphs.

    The inferred default: a model whose halo-padded graph is an ordinary
    :class:`~nvalchemi.data.Batch` with per-atom fields and a per-edge
    ``neighbor_list`` (COO endpoints) declares nothing. It works on the abstract
    ``Batch`` storage groups, so it is model-agnostic.

    Atom / edge counts are knowable up front, so :meth:`pad` resolves both caps
    from ``cap_state`` before padding: ``"atoms"`` (1.15 initial headroom,
    ``strict_gt=False`` since the dead node sits at the last slot) and ``"edges"``
    (1.35 initial headroom — edge count climbs ~25% through equilibration). Both
    regrow x1.30 on overflow, stride 16.

    Layout:

    * Per-atom fields -> ``n_cap``: appended rows carry zeros and join the last
      graph (their node outputs are dropped by the owned-only consolidation).
    * Per-edge fields -> ``e_cap``: invalid edges (sentinel rows with endpoint
      ``>= n_real``) and the fill are routed to an isolated dead node (the last
      row, ``n_cap - 1``) as self-loops. ``neighbor_list_shifts`` for those rows
      uses the ``[1, 0, 0]`` image so the edge vector is non-degenerate (a zero
      vector ``NaN``\\ s through spherical harmonics). The dead node is referenced
      by no real edge and masked out of every owned output.

    :meth:`unpad` is a no-op: the owned-only output consolidation already drops
    the dead / ghost rows. The framework restores the transient padded storage
    separately (the padded ``Batch`` is reused in place across MD steps).
    """

    def pad(
        self, data: "Batch", cap_state: dict[str, int], cap_atoms: bool = True
    ) -> "Batch":  # noqa: ARG002  (halo padder always caps atoms)
        """Resolve atom / edge caps from ``cap_state`` and pad the halo-padded
        ``Batch`` to them. Mutates ``data`` in place and returns it; ``None`` is a
        safe no-op."""
        if data is None:
            return data
        n_cap = resolve_cap(
            cap_state, "atoms", data.num_nodes,
            initial_factor=1.15, grow_factor=1.30, stride=16, strict_gt=False,
        )
        e_cap = resolve_cap(
            cap_state, "edges", data.num_edges,
            initial_factor=1.35, grow_factor=1.30, stride=16,
        )
        return _pad_coo_to_caps(data, n_cap, e_cap)

    def unpad(self, output: Any, n_real: int | None = None) -> Any:
        """No-op: the owned-only output consolidation drops the dead rows."""
        return output


def _pad_coo_to_caps(data: "Batch", n_cap: int, e_cap: int) -> "Batch":
    """Pad a halo-padded COO ``Batch`` to ``n_cap`` atoms / ``e_cap`` edges.

    Shared by :meth:`COOPadder.pad` (resolves the caps first) and
    ``ShardedBatch.pad_padded_view_to_caps`` (handed explicit caps). Mutates
    ``data`` in place and returns it; raises on cap overflow.
    """
    import torch  # noqa: PLC0415

    from nvalchemi.data.level_storage import (  # noqa: PLC0415
        SegmentedLevelStorage,
    )

    pb = data
    if pb is None:
        return data
    n_real = pb.num_nodes
    e_real = pb.num_edges
    if n_real >= n_cap:
        raise RuntimeError(
            f"atom pad cap overflow: n_padded={n_real} >= n_cap={n_cap}"
        )
    if e_real > e_cap:
        raise RuntimeError(f"edge pad cap overflow: E={e_real} > e_cap={e_cap}")
    dead = n_cap - 1
    pad_n = n_cap - n_real

    # per-atom fields -> n_cap (zero pad rows, joined to last graph)
    atoms = pb._atoms_group
    new_atom_data = {
        k: torch.cat(
            [atoms[k], atoms[k].new_zeros((pad_n,) + tuple(atoms[k].shape[1:]))],
            dim=0,
        )
        for k in atoms.keys()
    }
    n_graphs = len(atoms)
    sl = atoms.segment_lengths[:n_graphs].clone()
    sl[-1] = sl[-1] + pad_n
    pb._storage.groups["atoms"] = SegmentedLevelStorage(
        data=new_atom_data,
        device=atoms.device,
        attr_map=atoms.attr_map,
        segment_lengths=sl,
    )

    # per-edge fields -> e_cap (invalid/pad -> isolated dead self-loop)
    edges = pb._edges_group
    if edges is None or edges.num_elements() == 0:
        return data
    nl = edges["neighbor_list"]  # [E, 2]; sentinel endpoints == n_real
    invalid = (nl >= n_real).any(dim=1)
    pad_e = e_cap - e_real
    new_edge_data: dict[str, Any] = {}
    for k in edges.keys():
        t = edges[k]
        trailing = tuple(t.shape[1:])
        if k == "neighbor_list":
            routed = torch.where(invalid.unsqueeze(1), torch.full_like(t, dead), t)
            fill = t.new_full((pad_e,) + trailing, dead)
            new_edge_data[k] = torch.cat([routed, fill], dim=0)
        elif k == "neighbor_list_shifts":
            # nonzero image for dead/invalid edges (zero vector -> NaN).
            unit = t.new_zeros(trailing).reshape(-1)
            if unit.numel():
                unit[0] = 1
            unit = unit.reshape(trailing)
            routed = torch.where(invalid.unsqueeze(1), unit.expand_as(t), t)
            fill = unit.unsqueeze(0).expand((pad_e,) + trailing)
            new_edge_data[k] = torch.cat([routed, fill], dim=0)
        else:
            fill = t.new_zeros((pad_e,) + trailing)
            new_edge_data[k] = torch.cat([t, fill], dim=0)
    n_edge_seg = len(edges)
    esl = edges.segment_lengths[:n_edge_seg].clone()
    esl[-1] = esl[-1] + pad_e
    pb._storage.groups["edges"] = SegmentedLevelStorage(
        data=new_edge_data,
        device=edges.device,
        attr_map=edges.attr_map,
        segment_lengths=esl,
    )
    return data


# Sentinel: pad this per-system label with the last system's index (value
# depends on the system count, not a constant).
_LAST_SYSTEM = object()


class DensePadder:
    """Built-in :class:`GraphPadder` for dense ``(N, K)`` neighbor-matrix graphs.

    The dense counterpart of :class:`COOPadder`: the graph is an ``(N, K)``
    neighbor matrix that rides the atom dimension (no separate edge dim). Pads the
    per-atom row fields to a fixed atom capacity and repoints the neighbor
    matrix's padding sentinel to an isolated dead atom (the last row); ``unpad``
    slices the dead rows off the per-atom outputs.

    Parametrized by the model's field names: ``count_key`` (field whose row count
    is the atom count), ``nbmat_key`` (the neighbor matrix), ``row_pads``
    (per-atom field -> pad fill value; pass :data:`LAST_SYSTEM` to pad a
    per-system label with the last system's index), and ``atom_output_keys``
    (per-atom outputs that get dead rows stripped in :meth:`unpad`).

    Layout assumption: the input's last pre-pad row is the model's own
    padding/sentinel atom, so the real atom count is ``n_rows - 1`` and neighbor
    entries ``>= n_rows - 1`` are the sentinel — both get repointed to the dead
    row.
    """

    LAST_SYSTEM = _LAST_SYSTEM

    def __init__(
        self,
        *,
        count_key: str,
        nbmat_key: str,
        row_pads: dict[str, Any],
        atom_output_keys: tuple[str, ...] = (),
        n_systems_key: str | None = None,
        cap_key: str = "atoms",
        initial_factor: float = 1.15,
        grow_factor: float = 1.15,
        stride: int = 16,
    ) -> None:
        self.count_key = count_key
        self.nbmat_key = nbmat_key
        self.row_pads = dict(row_pads)
        self.atom_output_keys = tuple(atom_output_keys)
        self.n_systems_key = n_systems_key
        self.cap_key = cap_key
        self.initial_factor = initial_factor
        self.grow_factor = grow_factor
        self.stride = stride
        # Owned+ghost atom count of the last padded graph (== n_rows - 1, the
        # model's sentinel/pad index), stashed in pad() for unpad().
        self._n_real: int | None = None

    def pad(
        self, data: dict[str, Any], cap_state: dict[str, int], cap_atoms: bool = True
    ) -> dict[str, Any]:  # noqa: ARG002  (halo padder always caps atoms)
        """Resolve the atom cap from ``cap_state`` and pad the dense fields.

        ``data`` is the model's plain-tensor input dict; returns a shallow copy
        with the row fields + neighbor matrix padded to the atom cap.
        """
        import torch  # noqa: PLC0415

        n_cur = int(data[self.count_key].shape[0])
        n_cap = resolve_cap(
            cap_state, self.cap_key, n_cur,
            initial_factor=self.initial_factor, grow_factor=self.grow_factor,
            stride=self.stride, strict_gt=False,
        )
        dead = n_cap - 1
        sent_old = n_cur - 1
        self._n_real = sent_old
        n_sys = int(data.get(self.n_systems_key, 1)) if self.n_systems_key else 1

        def _pad_rows(t: Any, fill: Any) -> Any:
            if t is None or not hasattr(t, "shape"):
                return t
            p = n_cap - int(t.shape[0])
            if p <= 0:
                return t
            return torch.cat(
                [t, t.new_full((p,) + tuple(t.shape[1:]), fill)], dim=0
            )

        out = dict(data)
        for key, fill in self.row_pads.items():
            if key not in out:
                continue
            fill_val = (n_sys - 1) if fill is _LAST_SYSTEM else fill
            out[key] = _pad_rows(out[key], fill_val)

        nb = out.get(self.nbmat_key)
        if nb is not None:
            # Repoint the old sentinel (entries >= sent_old) to the dead row,
            # then fill pad rows with dead self-refs; masking (slot == dead)
            # drops them from every owned output.
            nb = torch.where(nb >= sent_old, torch.full_like(nb, dead), nb)
            p = n_cap - int(nb.shape[0])
            if p > 0:
                nb = torch.cat(
                    [nb, nb.new_full((p,) + tuple(nb.shape[1:]), dead)], dim=0
                )
            out[self.nbmat_key] = nb
        return out

    def unpad(
        self, output: dict[str, Any], n_real: int | None = None
    ) -> dict[str, Any]:
        """Slice the dead-atom rows off the per-atom outputs.

        Strips to ``n_real`` when given (eager / sharded paths, where no
        :meth:`pad` ran), else to the count the matching :meth:`pad` stashed.
        """
        n = n_real if n_real is not None else self._n_real
        if n is None:
            return output
        for key in self.atom_output_keys:
            t = output.get(key)
            if t is not None and hasattr(t, "shape") and t.shape[0] > n:
                output[key] = t[:n]
        return output


class DenseBatchPadder:
    """Built-in :class:`GraphPadder` for dense ``(N, K)`` neighbor-matrix
    :class:`~nvalchemi.data.Batch`\\ es (AIMNet2).

    The batch-level counterpart of :class:`DensePadder`. The framework compiles
    the whole ``wrapper.forward``, so the fixed-shape padding must land on the
    halo-padded ``Batch`` *before* ``adapt_input`` runs — the same seam
    :class:`COOPadder` uses. This padder pads the atom-level storage group to a
    fixed atom capacity with inert dead atoms (zeros, ``Z=0``, joined to the last
    graph) and repoints the ``neighbor_matrix`` sentinel so no real atom ever sees
    a dead atom as a neighbor.

    ``adapt_input`` then appends its own padding atom on top of this fixed-shape
    batch, so the compiled model input keeps a constant ``(n_cap + 1, …)`` shape
    across MD steps. The sentinel is repointed to ``n_cap`` — the index of that
    appended pad atom — so aimnet's ``calc_masks`` masks every dead / sentinel
    neighbor slot to zero.

    :meth:`unpad` is a no-op: the owned-only ``mol_sum`` (masked by ``n_owned``)
    drops the dead rows from the energy, and the per-atom force output is sliced
    by the framework's output consolidation. The framework restores the transient
    padded storage separately (the padded ``Batch`` is reused in place across MD
    steps).

    Parameters
    ----------
    nbmat_key : str, default ``"neighbor_matrix"``
        The dense neighbor-matrix node field whose padding sentinel (unused slots,
        set to the pre-pad node count) must be repointed to the appended pad-atom
        index.
    initial_factor, grow_factor, stride
        Forwarded to :func:`resolve_cap` for the ``"atoms"`` capacity.
    """

    def __init__(
        self,
        *,
        nbmat_key: str = "neighbor_matrix",
        initial_factor: float = 1.15,
        grow_factor: float = 1.30,
        stride: int = 16,
    ) -> None:
        self.nbmat_key = nbmat_key
        self.initial_factor = initial_factor
        self.grow_factor = grow_factor
        self.stride = stride

    def pad(
        self, data: "Batch", cap_state: dict[str, int], cap_atoms: bool = True
    ) -> "Batch":  # noqa: ARG002  (halo padder always caps atoms)
        """Resolve the atom cap from ``cap_state`` and pad the halo-padded
        ``Batch`` to it. Mutates ``data`` in place and returns it; ``None`` is a
        safe no-op."""
        if data is None:
            return data
        n_cap = resolve_cap(
            cap_state, "atoms", data.num_nodes,
            initial_factor=self.initial_factor, grow_factor=self.grow_factor,
            stride=self.stride, strict_gt=False,
        )
        return _pad_dense_batch_to_cap(data, n_cap, self.nbmat_key)

    def unpad(self, output: Any, n_real: int | None = None) -> Any:
        """No-op: the owned-only mol_sum + output consolidation drop dead rows."""
        return output


def _pad_dense_batch_to_cap(data: "Batch", n_cap: int, nbmat_key: str) -> "Batch":
    """Pad a halo-padded dense-nbmat ``Batch`` to ``n_cap`` atoms.

    Pads every atom-level node field to ``n_cap`` (zero pad rows, joined to the
    last graph) and repoints the ``neighbor_matrix`` sentinel (entries
    ``>= n_real``) to ``n_cap`` — the index of the pad atom ``adapt_input``
    appends — so dead / unused neighbor slots are masked by aimnet's
    ``calc_masks``. Dead rows self-reference ``n_cap`` too. Mutates ``data`` in
    place and returns it; raises on cap overflow.
    """
    import torch  # noqa: PLC0415

    from nvalchemi.data.level_storage import (  # noqa: PLC0415
        SegmentedLevelStorage,
    )

    pb = data
    n_real = pb.num_nodes
    if n_real >= n_cap:
        raise RuntimeError(
            f"atom pad cap overflow: n_padded={n_real} >= n_cap={n_cap}"
        )
    pad_n = n_cap - n_real
    sentinel = n_cap  # the pad atom adapt_input appends sits at index n_cap

    atoms = pb._atoms_group
    new_atom_data: dict[str, Any] = {}
    for k in atoms.keys():
        t = atoms[k]
        trailing = tuple(t.shape[1:])
        if k == nbmat_key:
            # Repoint unused/sentinel slots (>= n_real) to the future pad atom,
            # then fill dead rows with pad-atom self-refs. calc_masks masks them
            # all to zero, so no real atom sees a dead atom as a neighbor.
            repointed = torch.where(t >= n_real, torch.full_like(t, sentinel), t)
            fill = t.new_full((pad_n,) + trailing, sentinel)
            new_atom_data[k] = torch.cat([repointed, fill], dim=0)
        else:
            new_atom_data[k] = torch.cat(
                [t, t.new_zeros((pad_n,) + trailing)], dim=0
            )
    n_graphs = len(atoms)
    sl = atoms.segment_lengths[:n_graphs].clone()
    sl[-1] = sl[-1] + pad_n
    pb._storage.groups["atoms"] = SegmentedLevelStorage(
        data=new_atom_data,
        device=atoms.device,
        attr_map=atoms.attr_map,
        segment_lengths=sl,
    )
    return data
