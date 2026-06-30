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

"""UMA (fairchem-core) model wrapper.

Wraps a fairchem ``MLIPPredictUnit`` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model.

UMA is multi-task — a single backbone checkpoint (``uma-s-1p1``,
``uma-s-1p2``, ``uma-m-1p1``) ships with heads for OMol25 (molecules),
OMat24 (crystals), OC20 (catalysis), ODAC23 (direct air capture), and
OMC (molecular crystals). Task selection happens at checkpoint load
time via ``task_name=`` and is baked into the wrapper — matching the
"one wrapper, one model" pattern used by :class:`~nvalchemi.models.mace.MACEWrapper`.

Usage
-----
::

    from nvalchemi.models.uma import UMAWrapper

    # OMol molecular potential
    mol_wrapper = UMAWrapper.from_checkpoint(
        "uma-s-1p1", task_name="omol", device="cuda"
    )

    # OMat bulk-crystal potential (same checkpoint, different head)
    mat_wrapper = UMAWrapper.from_checkpoint(
        "uma-s-1p1", task_name="omat", device="cuda"
    )

Notes
-----
* Energy is the primitive differentiable output; forces and (for
  periodic tasks) stress are derived via autograd.
* OMol requires a total-charge field; the wrapper reads ``charge`` off
  the input ``AtomicData`` / ``Batch`` and defaults to 0 if absent.
  Spin multiplicity (``spin`` on the batch) defaults to 1 for OMol and
  0 for periodic tasks.
* ``active_outputs`` is task-aware: ``{"energy", "forces"}`` for
  molecular tasks, ``{"energy", "forces", "stress"}`` for periodic.

``torch.compile``
-----------------
Unlike :class:`~nvalchemi.models.mace.MACEWrapper` /
:class:`~nvalchemi.models.aimnet2.AIMNet2Wrapper`, the UMA wrapper does
**not** expose a ``compile_model`` flag. fairchem owns compilation
internally: it is a field on ``fairchem.core.units.mlip_unit.api.inference.InferenceSettings``
(``compile: bool``), not a ``torch.compile(model)`` call. Reach it
through :meth:`from_checkpoint`'s ``inference_settings`` argument:

* ``inference_settings="turbo"`` — fairchem's optimized preset, which
  sets ``compile=True`` **and** ``tf32=True`` / ``merge_mole=True`` /
  ``activation_checkpointing=False``. Best for long simulations with
  fixed atomic composition; it changes numerics relative to ``"default"``.
* For a pure compile toggle, pass an ``InferenceSettings`` instance with
  ``compile=True`` and the other fields left at their ``"default"``
  values::

      from fairchem.core.units.mlip_unit.api.inference import (
          InferenceSettings,
      )

      wrapper = UMAWrapper.from_checkpoint(
          "uma-s-1p1",
          task_name="omat",
          device="cuda",
          inference_settings=InferenceSettings(compile=True),
      )
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed._core.context import current_dd_context
from nvalchemi.distributed._core.enums import Scope
from nvalchemi.distributed.compile_bridge import force_compile_static
from nvalchemi.distributed.helpers import (
    distributed_method,
    refresh_neighbors,
    scatter_to_owners,
    system_sum,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit import MLIPPredictUnit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

__all__ = ["UMATask", "UMAWrapper"]


# UMA task heads. The ``Literal`` is the single source of truth for valid task
# names; the membership set is derived from it.
UMATask = Literal["omol", "omat", "oc20", "odac", "omc"]
_UMA_TASKS: frozenset[str] = frozenset(get_args(UMATask))

# Tasks that declare PBC (stress supported). ``omol`` is molecular — no stress.
_PBC_TASKS: frozenset[str] = frozenset({"omat", "oc20", "odac", "omc"})


# Fixed-shape caps for compiled MD. fairchem's compiled graph needs static
# shapes, but per-rank atom/edge counts drift across an MD trajectory, forcing
# repeated recompiles. We pad inputs to fixed per-rank capacities: the atom dim
# grows to ``n_cap`` with inert "dead" atoms and the edge dim to ``e_cap`` with
# dead edges longer than the cutoff (zero contribution). Caps grow on overflow
# and persist, so there is one recompile per grow, then steady state.
_CAP_GROWTH = 1.15  # fractional headroom over the first / overflowing real count
_DEAD_COORD = 1.0e4  # base coordinate for inert dead atoms (well outside any box)
# Round capacities up to a stride so step-to-step count fluctuation lands in the
# same bucket; edges swing more than atoms, so they get a coarser stride.
_CAP_STRIDE: dict[str, int] = {"n_cap": 64, "e_cap": 1024}


def _pad_capped_graph(
    backbone: Any, fc_data: Any, n_real: int, cap_state: dict[str, int]
) -> Any:
    """Pad ``fc_data`` to fixed per-rank atom/edge capacities for compiled MD.

    Builds the real edge list with fairchem's ``generate_graph``, then pads atoms
    to ``n_cap`` and edges to ``e_cap`` with inert dead atoms / dead edges and
    writes the fixed-shape ``edge_index`` / ``cell_offsets`` / ``nedges``. Mutates
    and returns ``fc_data``; the caller sets the backbone's ``otf_graph`` so it
    consumes these precomputed edges.
    """
    import torch as _t  # noqa: PLC0415

    from fairchem.core.graph.compute import generate_graph  # noqa: PLC0415
    from nvalchemi.distributed.graph_padder import resolve_cap  # noqa: PLC0415

    device = fc_data.pos.device
    dtype = fc_data.pos.dtype

    # Real edges via fairchem's own graph generation.
    pbc = fc_data.pbc
    pbc2d = _t.atleast_2d(pbc)
    gd = generate_graph(
        fc_data,
        cutoff=backbone.cutoff,
        max_neighbors=backbone.max_neighbors,
        enforce_max_neighbors_strictly=backbone.enforce_max_neighbors_strictly,
        radius_pbc_version=backbone.radius_pbc_version,
        pbc=pbc2d,
    )
    edge_index = gd["edge_index"]  # (2, E)
    cell_offsets = gd["cell_offsets"].to(dtype)  # (E, 3)
    e_real = int(edge_index.shape[1])

    # Persistent grow-only per-rank caps (>= 2 dead-atom slots for the dead edge).
    n_cap = resolve_cap(
        cap_state, "n_cap", n_real, initial_factor=_CAP_GROWTH,
        grow_factor=_CAP_GROWTH, stride=_CAP_STRIDE["n_cap"], extra=2,
    )
    e_cap = resolve_cap(
        cap_state, "e_cap", e_real, initial_factor=_CAP_GROWTH,
        grow_factor=_CAP_GROWTH, stride=_CAP_STRIDE["e_cap"],
    )
    n_dead = n_cap - n_real
    e_dead = e_cap - e_real

    # Pad atoms with inert dead atoms (Z=0, batch=0, no edges reference them).
    # Two anchors sit far apart so the dead-edge length exceeds the cutoff.
    if n_dead > 0:
        dead_pos = _t.zeros(n_dead, 3, dtype=dtype, device=device)
        dead_pos[:, 0] = _DEAD_COORD
        if n_dead >= 2:
            dead_pos[1, 0] = _DEAD_COORD + 2.0 * float(backbone.cutoff)
        fc_data.pos = _t.cat([fc_data.pos, dead_pos], dim=0)
        fc_data.atomic_numbers = _t.cat(
            [fc_data.atomic_numbers,
             _t.zeros(n_dead, dtype=fc_data.atomic_numbers.dtype, device=device)],
            dim=0,
        )
        fc_data.batch = _t.cat(
            [fc_data.batch,
             _t.zeros(n_dead, dtype=fc_data.batch.dtype, device=device)],
            dim=0,
        )
        # Keep natoms summing to n_cap by charging the dead atoms to system 0.
        fc_data.natoms = fc_data.natoms.clone()
        fc_data.natoms[0] = fc_data.natoms[0] + n_dead
        fixed = getattr(fc_data, "fixed", None)
        if fixed is not None:
            fc_data.fixed = _t.cat(
                [fixed, _t.zeros(n_dead, dtype=fixed.dtype, device=device)], dim=0
            )
        tags = getattr(fc_data, "tags", None)
        if tags is not None:
            fc_data.tags = _t.cat(
                [tags, _t.zeros(n_dead, dtype=tags.dtype, device=device)], dim=0
            )

    # Pad edges between the two dead anchors at indices [n_real, n_real+1]: their
    # 2*cutoff separation gives a zero envelope, hence zero contribution.
    if e_dead > 0:
        a = n_real
        b = n_real + 1 if n_dead >= 2 else n_real
        dead_ei = _t.tensor(
            [[a] * e_dead, [b] * e_dead], dtype=edge_index.dtype, device=device
        )
        dead_co = _t.zeros(e_dead, 3, dtype=dtype, device=device)
        edge_index = _t.cat([edge_index, dead_ei], dim=1)
        cell_offsets = _t.cat([cell_offsets, dead_co], dim=0)

    fc_data.edge_index = edge_index
    fc_data.cell_offsets = cell_offsets
    fc_data.nedges = _t.tensor(
        [edge_index.shape[1]], dtype=_t.long, device=device
    )
    return fc_data


class _UMAGraphPadder:
    """UMA's :class:`~nvalchemi.distributed.graph_padder.GraphPadder`.

    UMA rebuilds its edge list inside the padder via fairchem's ``generate_graph``,
    so the edge count is known only partway through :meth:`pad`. The padder owns
    the whole caps lifecycle: graph rebuild + dead-atom/dead-edge padding
    (:meth:`pad`), the backbone ``otf_graph`` save/restore (:meth:`restore`), and
    the per-atom dead-row strip (:meth:`unpad`).
    """

    def __init__(self, backbone: Any) -> None:
        self._backbone = backbone
        # Real owned+ghost atom count of the last padded graph, saved by pad() so
        # unpad() knows where the dead rows begin. One forward at a time.
        self._n_real: int | None = None
        # Backbone ``otf_graph`` flag saved while the fixed-shape graph is in use;
        # ``None`` means not currently padded.
        self._orig_otf_graph: bool | None = None

    def pad(self, data: Any, cap_state: dict[str, int]) -> Any:
        """Rebuild and pad the fairchem graph to fixed per-rank caps.

        Parameters
        ----------
        data : Any
            The fairchem ``AtomicData`` graph to pad in place.
        cap_state : dict[str, int]
            Persistent per-rank capacity state (``n_cap`` / ``e_cap``).

        Returns
        -------
        Any
            ``data``, padded to fixed shapes. Also switches the backbone to
            ``otf_graph=False`` so it consumes the precomputed edges;
            :meth:`restore` puts the flag back.
        """
        self._n_real = int(data.pos.shape[0])
        out = _pad_capped_graph(self._backbone, data, self._n_real, cap_state)
        self._orig_otf_graph = self._backbone.otf_graph
        self._backbone.otf_graph = False
        return out

    def unpad(self, output: dict[str, Any]) -> dict[str, Any]:
        """Strip the inert dead-atom rows from per-atom outputs.

        Parameters
        ----------
        output : dict[str, Any]
            The raw fairchem output dict (per-atom ``forces`` are padded).

        Returns
        -------
        dict[str, Any]
            ``output`` with ``forces`` sliced back to the real atom count.
        """
        if self._n_real is not None and "forces" in output:
            output["forces"] = output["forces"][: self._n_real]
        return output

    def restore(self) -> None:
        """Restore the backbone's ``otf_graph`` flag.

        Idempotent: a no-op when no pad happened. Called in a ``finally`` so a
        forward error never leaves the backbone stuck on the fixed-shape path.

        Returns
        -------
        None
        """
        if self._orig_otf_graph is not None:
            self._backbone.otf_graph = self._orig_otf_graph
            self._orig_otf_graph = None


@torch._dynamo.disable  # type: ignore[misc]
def _eager_block_refresh(x: "torch.Tensor") -> "torch.Tensor":
    """Eager ghost-row refresh of the per-node features ``x``.

    Delegates to :func:`~nvalchemi.distributed.helpers.refresh_neighbors`, which
    halo-exchanges this rank's owned rows into its ghost rows. The
    ``@torch._dynamo.disable`` is load-bearing: it re-reads the per-step halo
    routing eagerly instead of baking the first step's into the compiled graph,
    keeping the collective in lockstep across ranks. The backward routes ghost-row
    gradients to owners; single-process is the identity.
    """
    return refresh_neighbors(x)


@distributed_method
def _distributed_escn_block_forward(
    ctx: Any,
    original: Any,
    block_self: Any,
    x: "torch.Tensor",
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Refresh this rank's neighbor rows of the block input under DD.

    A message-passing block reads each node's neighbors, so the input's
    neighbor rows must be current. ``refresh_neighbors`` makes them so per the
    active policy: the ghost-row exchange under :class:`RefreshOnlyHaloPolicy`
    (UMA's halo), the identity under :class:`GraphReplicatePolicy` (nodes are
    already full). The cross-rank recombine of the *message* lives on the
    aggregation op (:func:`_distributed_edgewise_fold` on ``Edgewise.forward``),
    not here — the block output is residual + a node-wise nonlinearity, so
    folding it would double-count the residual.

    ``@distributed_method`` runs the stock block off any distributed path.

    Parameters
    ----------
    ctx : DistributedContext
        The live context (supplied by :func:`distributed_method`; unused here).
    original : callable
        The unpatched ``eSCNMD_Block.forward``.
    block_self : eSCNMD_Block
        The block instance.
    x : torch.Tensor
        Per-node features over this rank's node rows.
    *args, **kwargs
        Remaining block-forward arguments, passed through unchanged.

    Returns
    -------
    Any
        The block's output, computed on neighbor-refreshed input features.
    """
    return original(block_self, _eager_block_refresh(x), *args, **kwargs)


@distributed_method
def _distributed_edgewise_fold(
    ctx: Any, original: Any, edgewise_self: Any, *args: Any, **kwargs: Any
) -> Any:
    """Recombine an ``Edgewise`` block's edge→node aggregation across ranks.

    ``Edgewise.forward`` returns the pure per-node aggregated message (the
    edge→node sum, before the block's residual add and node-wise MLP). That is
    the one quantity whose cross-rank parts must be summed under domain
    decomposition, so :func:`scatter_to_owners` folds it per the active policy:
    the identity under :class:`RefreshOnlyHaloPolicy` (UMA's aggregation is
    owned-complete from the ghost shell), the all-reduce under
    :class:`GraphReplicatePolicy` (each rank holds only its edge slice, so the
    message is partial). Folding here — not on the block output — keeps the
    residual and the node-wise nonlinearity off the collective.
    """
    return scatter_to_owners(original(edgewise_self, *args, **kwargs))


@distributed_method
def _distributed_edge_degree_fold(
    ctx: Any, original: Any, ed_self: Any, x: "torch.Tensor", *args: Any, **kwargs: Any
) -> Any:
    """Recombine the pre-block ``EdgeDegreeEmbedding`` aggregation across ranks.

    ``EdgeDegreeEmbedding.forward`` returns ``x + scatter(edge_contributions)``:
    the initial node embedding plus an edge→node sum that seeds message passing.
    Like the per-block :func:`_distributed_edgewise_fold`, only the edge sum may
    cross ranks — the residual ``x`` is already replicated. So fold the
    aggregation delta (``out - x``) per the active policy and re-add ``x``: the
    identity under :class:`RefreshOnlyHaloPolicy` (owned-complete from the ghost
    shell), the all-reduce under :class:`GraphReplicatePolicy` (each rank holds
    only its edge slice). Without this the seed embedding stays partial under
    graph-replicate and every downstream block inherits the error.
    """
    out = original(ed_self, x, *args, **kwargs)
    return x + scatter_to_owners(out - x)


@torch._dynamo.disable  # type: ignore[misc]
@distributed_method
def _distributed_slice_edges(
    ctx: Any, original: Any, backbone_self: Any, *args: Any, **kwargs: Any
) -> Any:
    """Shard the model-built graph to this rank's edge slice (graph-replicate).

    ``@torch._dynamo.disable`` is load-bearing under ``torch.compile``: it forces
    the ``is_distributed`` gate + the edge slice to run eagerly per call, reading
    the live context, instead of baking the first-traced branch (which on a
    shared compiled model can be a non-distributed reference forward → no slice →
    full edges). fairchem's own GP collectives use the same compiler-disable
    pattern; the model's heavy convs around it stay compiled.

    UMA builds its full neighbor list internally (``otf_graph``); under the
    node-replicate graph-parallel policy each rank must run message passing over
    only a disjoint slice of those edges, so the per-block folds recombine the
    partial messages into the full sum. Every rank builds the same full graph
    from the replicated node set, so a contiguous ``[lo:hi]`` slice is a clean
    cross-rank partition. Registered on the graph-parallel spec only — the halo
    policy keeps the owned+ghost graph, so this never runs there.
    """
    gd = original(backbone_self, *args, **kwargs)
    n_edges = int(gd["edge_index"].shape[1])
    world = ctx.world_size
    lo = (n_edges * ctx.rank) // world
    hi = (n_edges * (ctx.rank + 1)) // world
    gd["edge_index"] = gd["edge_index"][:, lo:hi].contiguous()
    for key, val in gd.items():
        if (
            key != "edge_index"
            and isinstance(val, torch.Tensor)
            and val.dim() >= 1
            and val.shape[0] == n_edges
        ):
            gd[key] = val[lo:hi].contiguous()
    return gd


def _is_node_partition(ctx: Any) -> bool:
    """True when the active policy is node-partition graph-parallel.

    Node-partition runs the backbone on this rank's owned atom block (the
    backbone's node-wise inputs are sliced to it), so its per-system reductions
    are LOCAL owned partials with ``owned_offset == 0``; the halo / node-replicate
    paths run over the owned+ghost or full node set and use the OWNED scope."""
    from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
        GraphParallelPolicy,
    )

    return isinstance(ctx.policy, GraphParallelPolicy)


def _node_partition_bounds(ctx: Any) -> tuple[int, int]:
    """This rank's ``(offset, count)`` block in the rank-ordered full node set.

    The node-partition policy assigns each rank a contiguous balanced block of
    the global atoms; the all-gathered node tensor is in rank order, so the owned
    block is ``[offset : offset + count]``."""
    meta = ctx.gather_meta
    rank, world = ctx.rank, ctx.world_size
    counts = [int((meta.owner_rank == r).sum()) for r in range(world)]
    return sum(counts[:rank]), counts[rank]


@torch._dynamo.disable  # type: ignore[misc]
@distributed_method
def _distributed_partition_graph(
    ctx: Any, original: Any, backbone_self: Any, data_dict: Any, *args: Any, **kwargs: Any
) -> Any:
    """Restrict the backbone's node-wise work to this rank's owned atom block.

    The node-partition graph-parallel path replicates the full geometry, so eSCN
    builds the full neighbor graph; this mirrors eSCN's native ``gp_utils``
    partition but driven by our context. Keep only the edges whose receiver this
    rank owns (a contiguous block of the rank-ordered node set), slice the
    per-node inputs (atomic numbers, system index) to that block, and set
    ``gp_node_offset`` so the per-layer edge→node scatter lands on the owned rows.
    ``atomic_numbers_full`` — stashed by ``forward`` before this runs — stays full
    so the edge source/target embeddings index global senders; the per-layer
    sender all-gather is injected at ``Edgewise`` (:func:`_distributed_edgewise_gather`).

    ``@torch._dynamo.disable`` keeps the gate + slice eager per call under compile
    (the live partition is read each step, never baked into the traced graph).
    """
    gd = original(backbone_self, data_dict, *args, **kwargs)
    nlo, n_owned = _node_partition_bounds(ctx)
    ei = gd["edge_index"]
    n_edges = int(ei.shape[1])
    keep = (ei[1] >= nlo) & (ei[1] < nlo + n_owned)
    gd["edge_index"] = ei[:, keep].contiguous()
    for key, val in gd.items():
        if (
            key != "edge_index"
            and isinstance(val, torch.Tensor)
            and val.dim() >= 1
            and val.shape[0] == n_edges
        ):
            gd[key] = val[keep].contiguous()
    data_dict["atomic_numbers"] = data_dict["atomic_numbers"][nlo : nlo + n_owned]
    data_dict["batch"] = data_dict["batch"][nlo : nlo + n_owned]
    data_dict["gp_node_offset"] = nlo
    return gd


@distributed_method
def _distributed_edgewise_gather(
    ctx: Any,
    original: Any,
    edgewise_self: Any,
    x: "torch.Tensor",
    x_edge: "torch.Tensor",
    edge_index: "torch.Tensor",
    wigner: "torch.Tensor",
    wigner_inv_envelope: "torch.Tensor",
    *args: Any,
    **kwargs: Any,
) -> Any:
    """All-gather owned node features to the full set for the conv (node-partition).

    The node-partition policy runs the block on this rank's owned features; the
    convolution reads source features for globally-indexed edges, so all-gather
    the owned rows to the full replicated tensor (``refresh_neighbors`` → the
    policy's owned→full all-gather, reduce-scatter on the backward) and run
    ``Edgewise.forward_chunk`` with the *owned* row count as the scatter target
    and ``node_offset`` shifting global receivers into the owned-local range.
    Replaces — not wraps — ``Edgewise.forward``, bypassing the stock ``gp_utils``
    gather branch and activation-checkpoint chunking (graph parallel disables AC).

    ``@distributed_method`` falls back to the stock forward off any distributed
    path (single process).
    """
    node_offset = kwargs.get("node_offset", 0)
    n_owned = x.shape[0]
    x_full = refresh_neighbors(x)
    return edgewise_self.forward_chunk(
        x_full,
        n_owned,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        node_offset,
    )


def _distributed_reduce_node_to_system(
    node_values: "torch.Tensor",
    batch: "torch.Tensor",
    num_systems: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Halo-aware replacement for fairchem ``reduce_node_to_system``.

    Under domain decomposition ``node_values`` arrives over this rank's
    ``owned + ghost`` atoms, so the stock per-system reduce would leave a
    rank-local value. Per-node energy (1-D) reduces owned rows only and all-reduces
    in fp64 to the global per-system energy; the per-atom virial (multi-D) reduces
    all local rows here, deferring the cross-rank sum to consolidation.
    ``energy_part`` equals ``reduced`` (fairchem derives forces/stress from it via
    autograd). Single-process falls back to the stock reduce.

    Parameters
    ----------
    node_values : torch.Tensor
        Per-node values over this rank's owned+ghost atoms — 1-D for energy,
        multi-D for the virial.
    batch : torch.Tensor
        Per-node system index, ``(n_local,)``.
    num_systems : int
        Number of systems in the (global) batch.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(reduced, energy_part)`` — both the per-system value fairchem expects;
        equal here.
    """
    import torch as _t  # noqa: PLC0415

    if current_dd_context().is_distributed:
        if node_values.dim() == 1:
            # Per-node energy. Node-partition: a LOCAL owned partial (the backbone
            # ran on this rank's owned atoms, offset 0; the framework SUM-reduces
            # across ranks). Halo / node-replicate: owned-only scatter + all-reduce
            # → global per-system energy, replicated.
            scope = (
                Scope.LOCAL if _is_node_partition(current_dd_context()) else Scope.OWNED
            )
            reduced = system_sum(node_values, batch, num_systems, scope=scope)
            return reduced, reduced

        # Per-atom virial: scatter all local rows with no all-reduce here; the
        # cross-rank sum is deferred to consolidation (all-reducing now would
        # double-count fairchem's un-reduced per-rank cell-virial term).
        out_shape = (num_systems,) + tuple(node_values.shape[1:])
        sysv = _t.zeros(out_shape, device=node_values.device, dtype=node_values.dtype)
        flat = node_values.reshape(node_values.shape[0], -1)
        sysv = (
            sysv.reshape(num_systems, -1)
            .index_add(0, batch, flat)
            .reshape(out_shape)
        )
        return sysv, sysv

    # Stock (non-distributed) path: nothing to all-reduce, so
    # ``reduced == system_values``.
    out_shape = (num_systems,) + tuple(node_values.shape[1:])
    system_values = _t.zeros(out_shape, device=node_values.device, dtype=_t.float64)
    if node_values.dim() == 1:
        system_values = system_values.index_add(
            0, batch, node_values.to(_t.float64)
        )
    else:
        flat_node = node_values.reshape(node_values.shape[0], -1)
        system_values = (
            system_values.reshape(num_systems, -1)
            .index_add(0, batch, flat_node.to(_t.float64))
            .reshape(out_shape)
        )
    return system_values, system_values


@distributed_method
def _distributed_undo_refs(
    ctx: Any, original: Any, refs_self: Any, batch: Any, tensor: "torch.Tensor"
) -> "torch.Tensor":
    """Halo-aware replacement for ``ElementReferences.undo_refs``.

    fairchem adds the element-reference energy back by reducing ``ref[Z]`` over
    this rank's owned + ghost atoms, but the model energy was reduced owned-only
    (see :func:`_distributed_reduce_node_to_system`), so including ghost references
    over-counts. The fix sums references over owned atoms only, then all-reduces
    across the mesh. ``@distributed_method`` falls back to stock off the halo path.

    Parameters
    ----------
    ctx : DistributedContext
        The live context, used for ``n_owned``.
    original : callable
        The unpatched ``ElementReferences.undo_refs``.
    refs_self : ElementReferences
        The element-references module.
    batch : Any
        fairchem batch carrying ``atomic_numbers_full`` / ``batch_full``.
    tensor : torch.Tensor
        Per-system energies ``(num_systems, …)`` to add references onto.

    Returns
    -------
    torch.Tensor
        ``tensor`` plus the global owned-only reference sum, replicated per rank.
    """
    num_systems = int(tensor.shape[0])
    elem_refs = refs_self.element_references
    # Node-partition reduced its energy over this rank's owned atoms (offset 0),
    # so add references over the owned-sliced atoms as a LOCAL partial (the
    # framework SUM-reduces across ranks). Halo / node-replicate reduce the full
    # owned+ghost (or full) node set with the offset-aware OWNED scope, so the
    # ``_full`` rows are passed and ``system_sum`` slices this rank's owned rows.
    if _is_node_partition(current_dd_context()):
        z_src, batch_src, scope = batch.atomic_numbers, batch.batch, Scope.LOCAL
    else:
        z_src, batch_src, scope = (
            batch.atomic_numbers_full,
            batch.batch_full,
            Scope.OWNED,
        )
    z = z_src.to(dtype=torch.long, device=elem_refs.device)
    per_atom = elem_refs[z].to(dtype=tensor.dtype)
    ref_sum = system_sum(per_atom, batch_src, num_systems, scope=scope)
    return tensor + ref_sum.view(tensor.shape)


@distributed_method
def _distributed_get_composition_info(
    ctx: Any, original: Any, backbone_self: Any, data: Any
) -> Any:
    """Caps-aware replacement for ``eSCNMDBackbone._get_composition_info``.

    Under fixed-shape caps the input carries inert dead atoms (``Z=0``) appended
    beyond ``n_padded`` (see :func:`_pad_capped_graph`). Their count varies step to
    step, so histogramming them would drift the composition and trip fairchem's
    MoLE consistency check; the histogram covers the real ``owned+ghost`` atoms
    only. ``@distributed_method`` falls back to stock off the halo path.

    Parameters
    ----------
    ctx : DistributedContext
        The live context, used for ``n_padded`` (the real owned+ghost count).
    original : callable
        The unpatched ``eSCNMDBackbone._get_composition_info``.
    backbone_self : eSCNMDBackbone
        The backbone instance.
    data : Any
        fairchem batch (its ``atomic_numbers`` includes the dead atoms).

    Returns
    -------
    Any
        ``(composition, charge, spin, dataset)`` with the composition histogram
        over the real atoms only.
    """
    n = ctx.n_padded
    an = data.atomic_numbers[:n].to(torch.int)
    composition = data.atomic_numbers.new_zeros(
        backbone_self.max_num_elements, dtype=torch.int
    ).index_add(0, an, torch.ones(n, dtype=torch.int, device=an.device))
    return (
        composition,
        getattr(data, "charge", None),
        getattr(data, "spin", None),
        getattr(data, "dataset", [None]),
    )


@distributed_method
def _distributed_set_mole_coefficients(
    ctx: Any,
    original: Any,
    backbone_self: Any,
    atomic_numbers_full: "torch.Tensor",
    batch_full: "torch.Tensor",
    csd_mixed_emb: "torch.Tensor",
) -> Any:
    """Caps-aware replacement for ``eSCNMDMoeBackbone.set_MOLE_coefficients``.

    The MoLE expert-mixing coefficients depend on a per-system mean of the
    composition embedding. Under domain decomposition the input carries this rank's
    ``owned + ghost`` atoms plus inert dead atoms (``Z=0``) from the caps padder
    (see :func:`_pad_capped_graph`); both pollute the stock mean (dead rows add a
    ``Z=0`` embedding, and a per-rank mean differs from the global one), shifting
    every MoLE-linear weight by a small per-system amount.

    The fix mirrors the energy reduction: average the composition embedding over
    this rank's **owned** real atoms only (``Scope.OWNED`` drops ghost and dead
    rows) and all-reduce across the mesh, yielding the global per-system mean the
    single-process model computes. The remaining routing (``routing_mlp`` +
    coefficient norm) is unchanged. ``@distributed_method`` falls back to stock off
    the halo path. The merged path (compiled / ``merge_mole``) runs on CPU where
    the mesh collective is unavailable and is already exact, so it too uses stock.

    Parameters
    ----------
    ctx : DistributedContext
        The live context, used for ``n_owned`` via :func:`system_sum`.
    original : callable
        The unpatched ``set_MOLE_coefficients``.
    backbone_self : eSCNMDMoeBackbone
        The MoLE backbone instance.
    atomic_numbers_full : torch.Tensor
        Per-atom numbers over ``owned + ghost + dead`` rows.
    batch_full : torch.Tensor
        Per-atom system index over the same rows.
    csd_mixed_emb : torch.Tensor
        The charge/spin/dataset embedding, ``(num_systems, sphere_channels)``.

    Returns
    -------
    Any
        ``None`` — the coefficients are written onto
        ``backbone_self.global_mole_tensors`` in place, matching fairchem.
    """
    import numpy as _np  # noqa: PLC0415

    # No experts, no composition gating, or the CPU merge prep where the mesh
    # collective can't run (and the merged path is already exact): use stock.
    if (
        backbone_self.num_experts == 0
        or not getattr(backbone_self, "use_composition_embedding", False)
        or not atomic_numbers_full.is_cuda
    ):
        return original(
            backbone_self, atomic_numbers_full, batch_full, csd_mixed_emb
        )

    nsys = int(csd_mixed_emb.shape[0])
    with torch.autocast(
        device_type=atomic_numbers_full.device.type, enabled=False
    ):
        # ``system_sum`` slices this rank's owned rows (offset-aware), dropping
        # ghost and dead rows, so the full rows are passed here.
        comp_by_atom = backbone_self.composition_embedding(atomic_numbers_full)
        # Global per-system sum + owned count, both all-reduced across the mesh.
        comp_sum = system_sum(comp_by_atom, batch_full, nsys, scope=Scope.OWNED)
        ones = comp_by_atom.new_ones(comp_by_atom.shape[0], 1)
        count = system_sum(ones, batch_full, nsys, scope=Scope.OWNED)
        # fairchem's index_reduce(mean, include_self) seeds an extra zero row on
        # model_version 1.0; match it so the denominator is identical.
        include_self = 1.0 if _np.isclose(backbone_self.model_version, 1.0).item() else 0.0
        composition = comp_sum / (count + include_self).clamp_min(1.0)

        embeddings = [composition.unsqueeze(0), csd_mixed_emb[None]]
        pre_norm = backbone_self.routing_mlp(
            torch.vstack(embeddings).transpose(0, 1).reshape(nsys, -1)
        )
        backbone_self.global_mole_tensors.expert_mixing_coefficients = (
            backbone_self.mole_expert_coefficient_norm(
                backbone_self.mole_dropout(pre_norm)
            )
        )
    return None


@OptionalDependency.UMA.require
class UMAWrapper(nn.Module, BaseModelMixin):
    """Wrapper for fairchem's UMA (Universal Models for Atoms).

    Wraps a :class:`fairchem.core.units.mlip_unit.MLIPPredictUnit` — the
    level at which energy / forces / stress are computed by fairchem
    (the raw backbone only produces node embeddings). Task is fixed at
    construction; ``active_outputs`` reflects what that task supports.

    Parameters
    ----------
    predict_unit : fairchem.core.units.mlip_unit.MLIPPredictUnit
        Pre-loaded UMA prediction unit. Use :meth:`from_checkpoint` for
        the typical construction path that resolves a registered
        checkpoint name and downloads via HuggingFace Hub.
    task_name : str
        UMA task: one of ``_UMA_TASKS``. Determines which per-task head
        in the multi-task model is used and which inputs (charge, spin)
        must be populated.
    train : bool
        ``False`` (default) freezes all weights for inference (lossless for
        autograd forces); ``True`` keeps fairchem's trainable/frozen split
        so weights stay exposed for fine-tuning.

    Attributes
    ----------
    model_config : ModelConfig
        Task-dependent outputs + autograd + neighbor config.
    task_name : str
        The UMA task this wrapper is pinned to.
    """

    def __init__(
        self,
        predict_unit: "MLIPPredictUnit",
        task_name: UMATask = "omol",
        train: bool = False,
    ) -> None:
        super().__init__()
        if task_name not in _UMA_TASKS:
            raise ValueError(
                f"UMAWrapper task_name {task_name!r} must be one of {get_args(UMATask)}"
            )

        # Validate that the checkpoint actually supports this task —
        # surface the error at construction, not on first forward.
        available = list(predict_unit.dataset_to_tasks.keys())
        if task_name not in available:
            raise ValueError(
                f"Checkpoint does not ship a '{task_name}' head. "
                f"Available: {available}. Load a different checkpoint "
                f"or pick one of the available tasks."
            )

        self.predict_unit = predict_unit
        self.task_name = task_name
        self._is_pbc_task = task_name in _PBC_TASKS
        self._cutoff = self._extract_cutoff()

        # Task-dependent output set. Energy + forces are universal;
        # stress only makes sense for periodic tasks.
        outputs: set[str] = {"energy", "forces"}
        autograd_outputs: set[str] = {"forces"}
        active_outputs: set[str] = {"energy", "forces"}
        if self._is_pbc_task:
            outputs.add("stress")
            autograd_outputs.add("stress")
            active_outputs.add("stress")

        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            autograd_outputs=frozenset(autograd_outputs),
            autograd_inputs=frozenset({"positions"}),
            # All optional (not required for OMol) to keep one config shape
            # across tasks — the adapter fills charge/spin defaults.
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell", "charge", "spin", "tags"}),
            supports_pbc=True,
            needs_pbc=self._is_pbc_task,
            neighbor_config=NeighborConfig(
                cutoff=self._cutoff,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
            active_outputs=active_outputs,
        )

        # Inference (train=False): freeze all weights — conservative forces
        # come from autograd w.r.t. positions, so this is lossless and avoids
        # building a weight-grad graph each forward. Training: leave fairchem's
        # loaded trainable/frozen split intact so weights stay exposed.
        self._train = train
        if not train:
            for p in self.predict_unit.model.parameters():
                p.requires_grad_(False)
        self.train(train)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        name_or_path: str | Path,
        task_name: UMATask = "omol",
        device: str | torch.device = "cpu",
        inference_settings: "InferenceSettings | str" = "default",
        overrides: dict | None = None,
        train: bool = False,
    ) -> "UMAWrapper":
        """Resolve and load a UMA checkpoint.

        Accepts either a registered model name
        (``"uma-s-1p1"`` / ``"uma-s-1p2"`` / ``"uma-m-1p1"``) or a
        local filesystem path to a ``.pt`` file. The multi-task
        checkpoints ship all five task heads; ``task_name`` picks which
        one the wrapper exposes via :attr:`model_config.active_outputs`.

        Parameters
        ----------
        name_or_path : str | Path
            Registered model name (see
            ``fairchem.core.calculate.pretrained_mlip.available_models``)
            or a local file path.
        task_name : str
            One of ``omol``, ``omat``, ``oc20``, ``odac``, ``omc``.
            Defaults to ``omol`` (molecular chemistry) — the most
            common entry point; override explicitly for crystals /
            catalysis.
        device : str | torch.device
            Target device for inference. Defaults to ``"cpu"``.
        inference_settings : InferenceSettings | str
            fairchem inference configuration. Either a preset name
            (``"default"`` or ``"turbo"``) or a
            ``fairchem.core.units.mlip_unit.api.inference.InferenceSettings``
            instance. ``torch.compile`` is reached through this argument
            — see the module docstring's *torch.compile* section.
            Defaults to ``"default"``.
        overrides : dict | None
            Optional overrides forwarded to fairchem's inference-settings
            builder.
        train : bool
            If ``False`` (default), freeze all weights for inference —
            lossless, since conservative forces come from autograd on
            positions. If ``True``, keep fairchem's loaded trainable/frozen
            split so weights remain exposed for fine-tuning. Note: the
            ``forward`` path goes through fairchem's inference ``predict``
            (eval mode, detached forces); gradient-based training requires a
            separate path through the raw model.

        Returns
        -------
        UMAWrapper
            A wrapper pinned to ``task_name`` over the loaded predict unit.

        Raises
        ------
        ValueError
            If ``name_or_path`` is neither a registered model name nor a local
            file path.
        """
        import os as _os

        from fairchem.core.calculate import pretrained_mlip  # noqa: PLC0415
        from fairchem.core.units.mlip_unit import load_predict_unit  # noqa: PLC0415

        if isinstance(device, torch.device):
            device = device.type

        name_str = str(name_or_path)
        if name_str in pretrained_mlip.available_models:
            predict_unit = pretrained_mlip.get_predict_unit(
                name_str,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        elif _os.path.isfile(name_str):
            predict_unit = load_predict_unit(
                name_str,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        else:
            raise ValueError(
                f"{name_str!r} is neither a registered model name nor a "
                f"local file path. Known names: "
                f"{sorted(pretrained_mlip.available_models)}"
            )
        return cls(predict_unit, task_name=task_name, train=train)

    def _extract_cutoff(self) -> float:
        """Pull the radial cutoff from the loaded backbone.

        UMA's ASE calculator uses a 6.0 Å radius when external graph
        generation is enabled; the inference-settings default is 6.0.
        Prefer the backbone's own ``r_max`` attribute when present.
        """
        backbone = getattr(self.predict_unit.model.module, "backbone", None)
        if backbone is not None:
            r_max = getattr(backbone, "r_max", None)
            if r_max is not None:
                return float(r_max.item() if hasattr(r_max, "item") else r_max)
            cutoff = getattr(backbone, "cutoff", None)
            if cutoff is not None:
                return float(cutoff)
        return 6.0

    # ------------------------------------------------------------------
    # BaseModelMixin contract
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Radial cutoff (Å) for neighbor-list construction."""
        return self._cutoff

    @property
    def distribution_spec(self) -> Any:
        """MLIPSpec for UMA under domain decomposition.

        Each rank computes a full forward over its ``owned + ghost`` atoms on plain
        tensors; DD happens only at the boundaries: per-block ghost-row feature
        refresh, owned-only + all-reduce per-system energy reduction, and
        forces/stress through fairchem's autograd (ghost contributions routed to
        owners in consolidation). Nothing is sharded (``shard_fields`` is empty);
        the spec carries the fixed-shape-caps :class:`_UMAGraphPadder`.

        Returns
        -------
        MLIPSpec
            The memoized halo spec: boundary adapters, empty ``shard_fields``, and
            a :class:`CompilePolicy` carrying the graph padder.
        """
        import os  # noqa: PLC0415

        # Opt-in graph-parallel strategy. ``=1`` → node-replicate (full nodes,
        # edge shard, per-layer all-reduce; same boundary adapters as halo).
        # ``=partition`` → node-partition (owned node slice, per-layer node-feature
        # all-gather; its own minimal adapter set). Each cached separately.
        gp_mode = os.environ.get("NVALCHEMI_UMA_GP")
        gp = gp_mode == "1"
        partition = gp_mode == "partition"
        cache_attr = (
            "_dist_spec_gp_part_cache"
            if partition
            else "_dist_spec_gp_cache"
            if gp
            else "_dist_spec_cache"
        )
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached

        import dataclasses  # noqa: PLC0415

        from fairchem.core.models.uma.escn_md import (  # noqa: PLC0415
            eSCNMDBackbone,
        )
        from fairchem.core.models.uma.escn_md_block import (  # noqa: PLC0415
            Edgewise,
            eSCNMD_Block,
        )
        from fairchem.core.models.uma.escn_moe import (  # noqa: PLC0415
            eSCNMDMoeBackbone,
        )
        from fairchem.core.models.uma.nn.embedding import (  # noqa: PLC0415
            EdgeDegreeEmbedding,
        )
        from fairchem.core.modules.normalization.element_references import (  # noqa: PLC0415
            ElementReferences,
        )
        from nvalchemi.distributed.spec import (  # noqa: PLC0415
            SPEC_UMA_HALO,
            MethodAdapter,
            PythonAdapter,
        )

        helpers = (
            # Per-block neighbor-row refresh of the input node features.
            MethodAdapter(eSCNMD_Block, "forward", _distributed_escn_block_forward),
            # Edge→node aggregation recombine: identity under the refresh-only
            # halo policy (owned-complete), all-reduce under graph-replicate.
            MethodAdapter(Edgewise, "forward", _distributed_edgewise_fold),
            # Pre-block edge-degree seed embedding: same recombine (its output is
            # ``x + edge_sum``, so fold only the edge sum, re-add the residual).
            MethodAdapter(
                EdgeDegreeEmbedding, "forward", _distributed_edge_degree_fold
            ),
            # Owned-only + all_reduce per-system energy reduction.
            # reduce_node_to_system is re-exported under two modules (the
            # ``escn_md`` binding is used by the stress heads), so patch both.
            PythonAdapter(
                module_path="fairchem.core.models.uma.outputs",
                attr_name="reduce_node_to_system",
                replacement=_distributed_reduce_node_to_system,
            ),
            PythonAdapter(
                module_path="fairchem.core.models.uma.escn_md",
                attr_name="reduce_node_to_system",
                replacement=_distributed_reduce_node_to_system,
            ),
            # Element-reference undo: sum refs over owned atoms only + all_reduce
            # (the owned-only model energy would otherwise be over-counted).
            MethodAdapter(ElementReferences, "undo_refs", _distributed_undo_refs),
            # MoLE composition check: histogram the real atoms only so dead atoms
            # don't drift the composition.
            MethodAdapter(
                eSCNMDBackbone, "_get_composition_info", _distributed_get_composition_info
            ),
            # MoLE expert-coefficient gating: average the composition embedding
            # over global owned real atoms (drop ghost + dead caps rows) so the
            # mixing coefficients match the single-process model.
            MethodAdapter(
                eSCNMDMoeBackbone,
                "set_MOLE_coefficients",
                _distributed_set_mole_coefficients,
            ),
        )
        # Halo preset + boundary helpers only (no custom_ops). Stress is a
        # per-rank partial virial summed across the mesh in consolidation.
        from nvalchemi.distributed.spec import (  # noqa: PLC0415
            CompilePolicy,
            OutputKind,
            OutputSpec,
            Reduce,
        )

        # The fixed-shape-caps padder rides CompilePolicy.graph_padder; it pads the
        # fairchem graph built inside adapt_input.
        backbone = self.predict_unit.model.module.backbone
        from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
            RefreshOnlyHaloPolicy,
        )

        spec = dataclasses.replace(
            SPEC_UMA_HALO,
            distribution=dataclasses.replace(
                SPEC_UMA_HALO.distribution,
                # UMA's per-block aggregation is owned-complete (ghost shell):
                # refresh-only, no per-layer fold. The refresh-only halo policy
                # makes ``scatter_to_owners`` the identity, so the block adapter's
                # policy-agnostic sandwich reduces to a pure input refresh here,
                # and swaps to an all-reduce under the graph-parallel policy.
                policy=RefreshOnlyHaloPolicy(scatter_mode="local"),
                adapters=helpers,
                shard_fields=(),
            ),
            outputs={
                "stress": OutputSpec(
                    kind=OutputKind.PER_GRAPH, reduce=Reduce.ALL_REDUCE
                )
            },
            compile=CompilePolicy(graph_padder=_UMAGraphPadder(backbone)),
        )
        if gp:
            # Node-replicate: same policy-agnostic adapters, swap the storage
            # policy. Forces join stress in all-reduce — each is an autograd
            # output over this rank's edge slice, so consolidation does
            # ``/world_size`` + cross-rank sum to recover the global value.
            from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
                GraphReplicatePolicy,
            )

            # The graph-parallel-only edge shard: slice the model-built graph to
            # this rank's edge portion so message passing is partitioned (the
            # folds recombine). Halo keeps the owned+ghost graph, so this rides
            # the GP spec only. The shared ``helpers`` are already lowered onto
            # ``third_party_helpers`` by the halo spec's ``__post_init__``; pass
            # ONLY the new slice adapter here — passing ``helpers`` again would
            # re-lower them, double-installing every fold (doubled all-reduce).
            spec = dataclasses.replace(
                spec,
                distribution=dataclasses.replace(
                    spec.distribution,
                    policy=GraphReplicatePolicy(),
                    adapters=(
                        MethodAdapter(
                            eSCNMDBackbone,
                            "_generate_graph",
                            _distributed_slice_edges,
                        ),
                    ),
                ),
                all_reduce_outputs=spec.all_reduce_outputs | frozenset({"forces"}),
            )
        elif partition:
            # Node-partition: each rank runs the backbone on its owned atom block.
            # This needs its OWN minimal adapter set, NOT the halo helpers — the
            # block-input refresh would all-gather (un-partitioning the node-wise
            # work) and the per-layer folds would double-reduce. Drop them by
            # clearing ``custom_ops`` / ``third_party_helpers`` (the halo spec
            # already lowered ``helpers`` onto them) and lower only the partition
            # adapters: slice the node-wise work + owned-receiver edges
            # (``_generate_graph``), all-gather node features for the conv
            # (``Edgewise``; reduce-scatter adjoint), and reduce energy/refs as
            # LOCAL owned partials. MoLE stays stock — the replicated full atomic
            # numbers are the true global composition. Forces/energy are
            # consolidated by the framework's node-partition internal path
            # (owned-only, cross-rank SUM, no ``/world``).
            from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
                GraphParallelPolicy,
            )

            partition_helpers = (
                MethodAdapter(
                    eSCNMDBackbone, "_generate_graph", _distributed_partition_graph
                ),
                MethodAdapter(Edgewise, "forward", _distributed_edgewise_gather),
                PythonAdapter(
                    module_path="fairchem.core.models.uma.outputs",
                    attr_name="reduce_node_to_system",
                    replacement=_distributed_reduce_node_to_system,
                ),
                PythonAdapter(
                    module_path="fairchem.core.models.uma.escn_md",
                    attr_name="reduce_node_to_system",
                    replacement=_distributed_reduce_node_to_system,
                ),
                MethodAdapter(
                    ElementReferences, "undo_refs", _distributed_undo_refs
                ),
            )
            spec = dataclasses.replace(
                spec,
                distribution=dataclasses.replace(
                    spec.distribution,
                    policy=GraphParallelPolicy(),
                    custom_ops=(),
                    third_party_helpers=(),
                    adapters=partition_helpers,
                ),
            )
        setattr(self, cache_attr, spec)
        return spec

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Shape of the per-node backbone embedding.

        eSCN-MD (and eSEN) backbones produce ``[N, (lmax+1)^2, sphere_channels]``,
        read off the backbone's ``sph_feature_size`` / ``sphere_channels`` attrs.

        Returns
        -------
        dict[str, tuple[int, ...]]
            ``{"node_embeddings": (sph_feature_size, sphere_channels)}``.

        Raises
        ------
        RuntimeError
            If the predict unit's module exposes no ``backbone``.
        """
        backbone = getattr(self.predict_unit.model.module, "backbone", None)
        if backbone is None:
            raise RuntimeError(
                "predict_unit.model.module has no .backbone attribute — "
                "embedding_shapes cannot be inferred."
            )
        sph = int(getattr(backbone, "sph_feature_size"))
        ch = int(getattr(backbone, "sphere_channels"))
        return {"node_embeddings": (sph, ch)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Run the backbone only and attach node embeddings.

        UMA/eSEN backbones return ``{"embedding": [N, sph, ch], "batch": [N]}``;
        the embedding is attached as a node property so pipelines can consume it
        without re-running the heads.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; an ``AtomicData`` is promoted to a ``Batch``.
        **kwargs
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        AtomicData | Batch
            *data*, with ``node_embeddings`` ``[N, sph, ch]`` attached when the
            backbone returns an embedding.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        fc_data = self.adapt_input(data, **kwargs)
        fc_data = fc_data.to(next(self.predict_unit.model.parameters()).device)
        backbone = self.predict_unit.model.module.backbone
        with torch.no_grad():
            out = backbone(fc_data)
        if "embedding" in out:
            data.add_key("node_embeddings", [out["embedding"]], level="node")
        return data

    # ------------------------------------------------------------------
    # adapt_input / adapt_output
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> Any:
        """Convert an nvalchemi ``AtomicData`` / ``Batch`` to a fairchem graph.

        Tensor-native (no ASE round trip): tensors stay on
        ``data.positions.device``, preserving GPU residency and autograd.
        ``edge_index`` is left empty ``(2, 0)`` so fairchem's ``MLIPPredictUnit``
        rebuilds the graph internally, matching the default ``FAIRChemCalculator``
        path (``r_edges=False``), so outputs are equivalent. Charge/spin default
        per the ASE-calculator convention (per-system LongTensors; spin defaults
        to the closed-shell singlet for OMol, 0 for periodic tasks) unless the
        caller provides them on the batch.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; an ``AtomicData`` is promoted to a single-graph
            ``Batch``.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        fairchem.core.datasets.atomic_data.AtomicData
            The fairchem graph: ``pos`` ``[N, 3]``, ``atomic_numbers`` ``[N]``,
            ``cell`` ``[B, 3, 3]``, ``pbc`` ``[B, 3]``, per-system
            ``charge`` / ``spin`` ``[B]``, and empty ``edge_index`` ``[2, 0]``.
        """
        from fairchem.core.datasets.atomic_data import (  # noqa: PLC0415
            AtomicData as FCAtomicData,
        )

        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        device = data.positions.device
        target_dtype = self.predict_unit.inference_settings.base_precision_dtype

        # Nothing is sharded: under DD every input arrives plain (owned+ghost) and
        # fairchem's GNN never sees a ShardTensor.
        pos = data.positions.to(target_dtype)
        atomic_numbers = data.atomic_numbers.to(torch.long)
        batch_idx = data.batch_idx.to(torch.long)
        n_systems = int(data.num_graphs)
        total_atoms = pos.shape[0]

        # Per-system atom counts — precomputed on the batch.
        natoms = data.num_nodes_per_graph.to(torch.long)

        # Batch already shapes cell/pbc as (B, 3, 3) / (B, 3) — just cast.
        cell = getattr(data, "cell", None)
        if cell is None:
            cell = torch.zeros(n_systems, 3, 3, dtype=target_dtype, device=device)
        else:
            cell = cell.to(target_dtype)

        pbc = getattr(data, "pbc", None)
        if pbc is None:
            pbc = torch.full(
                (n_systems, 3), self._is_pbc_task, dtype=torch.bool, device=device
            )
        else:
            pbc = pbc.to(torch.bool)

        # charge/spin: the typed AtomicData fields are float (B, 1); fairchem
        # wants per-system long (B,), so flatten + cast (also handles raw (B,)).
        charge = getattr(data, "charge", None)
        if charge is None:
            charge = torch.zeros(n_systems, dtype=torch.long, device=device)
        else:
            charge = charge.to(torch.long).reshape(n_systems)

        spin = getattr(data, "spin", None)
        if spin is None:
            # spin is the multiplicity (only used by the OMol head); default to
            # the closed-shell singlet (1), matching FAIRChemCalculator. Open-
            # shell systems must set it explicitly. Periodic heads ignore spin.
            spin_default = 0 if self._is_pbc_task else 1
            spin = torch.full(
                (n_systems,), spin_default, dtype=torch.long, device=device
            )
        else:
            spin = spin.to(torch.long).reshape(n_systems)

        # Empty edges — predict_unit rebuilds the graph internally.
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        cell_offsets = torch.empty((0, 3), dtype=target_dtype, device=device)
        nedges = torch.zeros(n_systems, dtype=torch.long, device=device)

        fixed = torch.zeros(total_atoms, dtype=torch.long, device=device)

        # fairchem tags = nvalchemi atom_categories (sub-surface/surface/
        # adsorbate for OC20/ODAC); defaults to zeros when unset.
        cats = getattr(data, "atom_categories", None)
        if cats is None:
            tags = torch.zeros(total_atoms, dtype=torch.long, device=device)
        else:
            tags = cats.to(torch.long).reshape(total_atoms)

        return FCAtomicData(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            batch=batch_idx,
            sid=[""] * n_systems,
            dataset=[self.task_name] * n_systems,
        )

    def adapt_output(
        self, raw: dict, data: AtomicData | Batch | None = None
    ) -> ModelOutputs:
        """Map fairchem's prediction dict to nvalchemi's output keys.

        Parameters
        ----------
        raw : dict
            fairchem's prediction dict: ``"energy"`` (per-system), ``"forces"``
            (per-atom ``[N, 3]``), and optionally ``"stress"`` (per-system).
        data : AtomicData | Batch | None, optional
            The input system the outputs were computed for. Unused here.

        Returns
        -------
        ModelOutputs
            The active subset of ``energy`` ``[B, 1]``, ``forces`` ``[N, 3]``, and
            ``stress`` ``[B, 3, 3]``.
        """
        out: dict[str, torch.Tensor] = {}
        active = self.model_config.active_outputs
        target_dtype = self.predict_unit.inference_settings.base_precision_dtype

        if "energy" in active:
            energy = raw["energy"]
            # fairchem returns energy in fp64; cast to base precision so all
            # outputs share one dtype.
            energy = energy.to(target_dtype)
            # Ensure per-system 2D shape (n_graphs, 1).
            if energy.dim() == 1:
                energy = energy.unsqueeze(-1)
            out["energy"] = energy
        if "forces" in active:
            out["forces"] = raw["forces"]
        if "stress" in active and "stress" in raw:
            stress = raw["stress"]
            if stress.dim() == 2 and stress.shape[-1] == 9:
                # fairchem sometimes flattens stress; reshape to (n, 3, 3).
                stress = stress.reshape(-1, 3, 3)
            out["stress"] = stress

        return out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the UMA predict unit on ``data``.

        Pipeline: ``adapt_input`` -> ``MLIPPredictUnit.predict`` -> ``adapt_output``.
        The single distribution touchpoint is ``ctx.maybe_pad_graph``, which under
        compiled domain decomposition pads the fairchem graph to stable per-rank
        shapes (a no-op single-process). The two blocks below handle fairchem's own
        compile requirements: CPU routing for the first-call MoLE merge, and
        forcing static shapes.

        Parameters
        ----------
        data : AtomicData | Batch
            Input structure(s); promoted to a ``Batch`` by ``adapt_input``.

        Returns
        -------
        ModelOutputs
            ``energy`` (per system) plus ``forces`` / ``stress`` per
            ``model_config.active_outputs``.
        """
        fc_data = current_dd_context().maybe_pad_graph(self.adapt_input(data, **kwargs))

        settings = getattr(self.predict_unit, "inference_settings", None)
        first_call = not getattr(self.predict_unit, "lazy_model_intialized", True)
        compiling = settings is not None and getattr(settings, "compile", False)
        if first_call and settings is not None and (
            getattr(settings, "merge_mole", False) or compiling
        ):
            fc_data = fc_data.to(torch.device("cpu"))
        static_cm = (
            force_compile_static()
            if (first_call and compiling)
            else contextlib.nullcontext()
        )
        with static_cm:
            raw = self.predict_unit.predict(fc_data, undo_element_references=True)
        return self.adapt_output(raw, data=data)
