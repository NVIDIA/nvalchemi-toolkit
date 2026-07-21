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

"""Post-processing for a halo-distributed model's raw outputs.

A model wrapper running on a padded ``(owned + halo)`` batch returns
tensors shaped either ``(n_padded, *F)`` (per-atom) or ``(n_systems, *F)``
(per-system). Each output is classified and given a matching reduction:

=========================================  =====================================
classification                             reduction
=========================================  =====================================
per-atom AND ``owned_only_outputs``        slice to ``[:n_owned]``
per-atom AND autograd (not owned-only)     halo_reverse_exchange + /world_size
per-atom AND not autograd                  slice to ``[:n_owned]``
per-system AND autograd                    /world_size
per-system AND not autograd                passthrough (already replicated)
=========================================  =====================================

The autograd flag comes from :attr:`ModelConfig.autograd_outputs`; the
owned-only set from
:attr:`~nvalchemi.distributed.spec.MLIPSpec.owned_only_outputs`. Per-atom
vs per-system is inferred from ``shape[0]`` (``n_padded`` vs ``n_systems``).

Two distinct sources of per-atom output:

* **Autograd-derived partials (MACE / UMA / AIMNet2).** Halo rows carry
  partial gradients at this rank's halo copies of other ranks' atoms.
  ``halo_reverse_exchange`` routes those partials to the owners. The
  forward all-reduce replicates the energy across ranks, so the backward
  over-counts the gradient by ``world_size``; ``/world_size`` corrects it.
* **Kernel-direct global-state duplicates (Ewald / PME reciprocal
  forces).** The kernel sees the full replicated state and writes the
  correct force on every padded atom, so halo rows are exact duplicates of
  the owner's value. Halo reverse would over-count them; these models
  declare the affected keys in ``MLIPSpec.owned_only_outputs`` to slice
  instead.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as _dist

# Outputs already warned about this process; avoids per-call log spam.
_UNDECLARED_OUTPUT_WARNED: set[str] = set()


def _warn_undeclared_output_kind(
    key: str, shape: tuple[int, ...], n_padded: int
) -> None:
    """Warn once when consolidation falls back to the shape heuristic for an
    output the spec did not classify."""
    if key in _UNDECLARED_OUTPUT_WARNED:
        return
    _UNDECLARED_OUTPUT_WARNED.add(key)
    import warnings  # noqa: PLC0415

    warnings.warn(
        f"output_consolidation: output {key!r} has no spec.output_kinds "
        f"declaration; falling back to shape heuristic "
        f"(shape={shape}, n_padded={n_padded}). Declare on "
        f"MLIPSpec.output_kinds to silence this warning.",
        stacklevel=2,
    )


def _consolidation_debug_enabled() -> bool:
    """``NVALCHEMI_CONSOLIDATE_DEBUG=1`` dumps the raw, post-halo-reverse,
    and final values for every per-atom autograd key. Diagnostic only."""
    return os.environ.get("NVALCHEMI_CONSOLIDATE_DEBUG", "0") != "0"


if TYPE_CHECKING:
    from nvalchemi.distributed._core.particle_halo import (
        ParticleHaloConfig,
        ParticleHaloMetadata,
    )
    from nvalchemi.models.base import ModelConfig


__all__ = ["consolidate_padded_outputs", "consolidate_sharded_outputs"]


def consolidate_sharded_outputs(
    output: dict[str, Any],
    model_config: "ModelConfig",
    world_size: int,
    owned_only_outputs: "frozenset[str] | None" = None,
    all_reduce_outputs: "frozenset[str] | None" = None,
    halo_config: "ParticleHaloConfig | None" = None,
) -> dict[str, Any]:
    """Reduce a sharded-storage model's raw output dict to per-rank values.

    Sharded-storage wrappers (AIMNet2, UMA) run forward on each rank's
    owned rows, routing cross-rank reads by global id. Per output key,
    the first matching branch wins:

    * ``owned_only_outputs``    — passthrough (already globally correct).
    * ``all_reduce_outputs``    — cross-rank ``SUM`` all-reduce. Used for
      rank-local partials from models whose internal aggregation strips
      the ``ShardTensor`` subclass, so the in-forward reduction never fires.
    * ``autograd_outputs``      — divide by ``world_size`` to undo the
      replicated-energy over-count (forward is already replicated).
    * default                   — passthrough.

    Parameters
    ----------
    output
        Dict returned by ``wrapper(local_batch)``.
    model_config
        The inner wrapper's :class:`~nvalchemi.models.base.ModelConfig`.
        Its ``autograd_outputs`` field flags the outputs that carry the
        ``/world_size`` factor.
    world_size
        Number of ranks.
    owned_only_outputs
        Keys (from
        :attr:`~nvalchemi.distributed.spec.MLIPSpec.owned_only_outputs`)
        already globally correct per-rank; skip ``/world_size``.
    all_reduce_outputs
        Keys (from
        :attr:`~nvalchemi.distributed.spec.MLIPSpec.all_reduce_outputs`)
        needing a cross-rank sum-all-reduce.
    halo_config
        Required when ``all_reduce_outputs`` is non-empty — carries the
        process group to reduce over. Unused otherwise; may be ``None``.

    Returns
    -------
    dict[str, Any]
        Output dict with per-rank-correct tensors. Keys are emitted in sorted
        (rank-independent) order so the per-key collective schedule is identical
        on every rank.
    """
    autograd_outputs = model_config.autograd_outputs
    if owned_only_outputs is None:
        owned_only_outputs = frozenset()
    if all_reduce_outputs is None:
        all_reduce_outputs = frozenset()

    if all_reduce_outputs and halo_config is None:
        raise ValueError(
            "consolidate_sharded_outputs: all_reduce_outputs requires a "
            "halo_config to supply the mesh process group; got None."
        )

    debug = _consolidation_debug_enabled() or os.environ.get("NVALCHEMI_REDUCE_DEBUG")

    reduced: OrderedDict[str, Any] = OrderedDict()
    # Iterate in a rank-independent (sorted) key order:
    for key, value in sorted(output.items()):
        if not isinstance(value, torch.Tensor):
            reduced[key] = value
            continue
        pre_sum = value.detach().to(torch.float64).sum().item() if debug else None
        if key in owned_only_outputs:
            reduced[key] = value
            branch = "owned_only (passthrough)"
        elif key in all_reduce_outputs:
            from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
                distributed_all_reduce,
            )

            # Autograd outputs need ``/world_size`` first (undo the
            # over-counted replicated-energy gradient), then summed across
            # ranks to recover the global value.
            if key in autograd_outputs:
                value = value / world_size
            reduced[key] = distributed_all_reduce(value, halo_config)
            branch = "all_reduce (SUM)"
        elif key in autograd_outputs:
            # Forward replicates the energy; backward over-counts the
            # gradient by world_size. Divide to compensate.
            reduced[key] = value / world_size
            branch = f"autograd (/{world_size})"
        else:
            reduced[key] = value
            branch = "default (passthrough)"
        if debug:
            rank = _dist.get_rank() if _dist.is_initialized() else 0
            post_sum = reduced[key].detach().to(torch.float64).sum().item()
            print(
                f"[reduce-debug rank {rank}] consolidate_sharded key={key!r} "
                f"shape={tuple(value.shape)} branch={branch}  "
                f"pre_sum={pre_sum:+.6e} post_sum={post_sum:+.6e}",
                flush=True,
            )
    return reduced


def consolidate_padded_outputs(
    output: dict[str, Any],
    model_config: "ModelConfig",
    meta: "ParticleHaloMetadata",
    halo_config: "ParticleHaloConfig",
    world_size: int,
    owned_only_outputs: "frozenset[str] | None" = None,
    all_reduce_outputs: "frozenset[str] | None" = None,
    output_kinds: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """Reduce a halo-distributed model's raw output dict to per-rank values.

    Parameters
    ----------
    output
        Dict returned by ``wrapper(padded_batch)``. Tensors may have
        ``shape[0] == n_padded`` (per-atom) or ``shape[0] == n_systems``
        (per-graph). Non-tensor values pass through unchanged.
    model_config
        The inner wrapper's :class:`~nvalchemi.models.base.ModelConfig`.
        Its ``autograd_outputs`` field flags the outputs that carry the
        ``/world_size`` factor.
    meta
        Halo metadata for the owned/padded split on this rank.
    halo_config
        Config for the collective used by
        :func:`halo_reverse_exchange`.
    world_size
        Number of ranks; divides autograd-derived outputs by the
        forward replication factor.
    owned_only_outputs
        Keys (from
        :attr:`~nvalchemi.distributed.spec.MLIPSpec.owned_only_outputs`)
        whose per-atom values are already globally-correct duplicates and
        should be sliced rather than halo-reverse-summed. ``None`` or empty
        applies ``halo_reverse + /world_size`` to every per-atom autograd
        output.
    all_reduce_outputs
        Keys (from
        :attr:`~nvalchemi.distributed.spec.MLIPSpec.all_reduce_outputs`)
        whose per-rank value is a partial that must be summed across the
        mesh. Halo-storage models rarely need this (they reduce in-wrapper);
        present for symmetry with the sharded path and the rare case where a
        model's internal aggregation strips ShardTensor.

    Returns
    -------
    dict[str, Any]
        Output dict with per-rank-correct tensors. Key order preserved.
    """
    from nvalchemi.distributed._core.particle_halo import halo_reverse_exchange
    from nvalchemi.distributed.output_kinds import OutputKind  # noqa: PLC0415

    n_padded = meta.n_padded
    n_owned = meta.n_owned
    autograd_outputs = model_config.autograd_outputs
    if owned_only_outputs is None:
        owned_only_outputs = frozenset()
    if all_reduce_outputs is None:
        all_reduce_outputs = frozenset()
    if output_kinds is None:
        output_kinds = {}

    reduced: OrderedDict[str, Any] = OrderedDict()
    # Rank-independent (sorted) key order
    for key, value in sorted(output.items()):
        if not isinstance(value, torch.Tensor):
            reduced[key] = value
            continue

        # An output may itself be a ShardTensor (e.g. forces from
        # autograd.grad against ShardTensor positions). Consolidation and the
        # user-facing result need plain tensors, so drop to the local view.
        if type(value).__name__ == "ShardTensor":
            value = value.to_local()

        is_autograd = key in autograd_outputs
        # The spec-declared kind decides per-atom vs per-system; falls back to
        # the shape heuristic (with a one-shot warning) when undeclared.
        # GLOBAL short-circuits to passthrough.
        kind = output_kinds.get(key, OutputKind.UNKNOWN)
        if kind is OutputKind.GLOBAL:
            reduced[key] = value
            continue
        if kind is OutputKind.PER_NODE:
            is_per_atom = True
        elif kind is OutputKind.PER_GRAPH:
            is_per_atom = False
        else:
            is_per_atom = value.shape[0] == n_padded
            _warn_undeclared_output_kind(key, value.shape, n_padded)
        is_owned_only = key in owned_only_outputs

        if key in all_reduce_outputs:
            # Non-autograd: each rank's value is a forward partial; sum them.
            # Autograd (e.g. stress): backward inflated the per-rank gradient
            # by world_size, so /world_size first, then sum across ranks.
            from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
                distributed_all_reduce,
            )

            value_to_reduce = value / world_size if is_autograd else value
            reduced[key] = distributed_all_reduce(value_to_reduce, halo_config)
            continue

        if is_per_atom and is_owned_only:
            # Per-atom output from globally-replicated state (Ewald/PME
            # reciprocal forces). Halo rows are exact duplicates of the
            # owner's value, so just slice — halo_reverse and /world_size
            # would both over-count.
            reduced[key] = value[:n_owned] if value.shape[0] > n_owned else value
            continue
        if is_per_atom and is_autograd:
            # Halo rows hold this rank's partial gradient at its halo copy of
            # another rank's atom. Route them to owners, then undo the
            # replicated-energy over-count.
            owned = halo_reverse_exchange(value, meta, halo_config)
            final = owned / world_size

            if _consolidation_debug_enabled():
                rank = _dist.get_rank() if _dist.is_initialized() else 0
                sample = value[:n_owned].detach()
                owned_sample = owned.detach()
                final_sample = final.detach()
                print(
                    f"[consolidate rank {rank}] key={key!r} n_owned={n_owned} "
                    f"n_padded={n_padded} world_size={world_size}\n"
                    f"  pre-halo-reverse (padded[:n_owned], first 2 rows):\n"
                    f"    {sample[:2].cpu().tolist()}\n"
                    f"  post-halo-reverse (owned sum, first 2 rows):\n"
                    f"    {owned_sample[:2].cpu().tolist()}\n"
                    f"  post-/world_size (final, first 2 rows):\n"
                    f"    {final_sample[:2].cpu().tolist()}",
                    flush=True,
                )

            reduced[key] = final
        elif is_per_atom and not is_autograd:
            # Halo rows duplicate owner-rank values (synced during forward);
            # slice to keep only the owned rows.
            reduced[key] = value[:n_owned] if value.shape[0] > n_owned else value
        elif not is_per_atom and is_autograd:
            # Per-system autograd output (e.g. stress). Forward is replicated;
            # backward over-counts the gradient by world_size.
            reduced[key] = value / world_size
        else:
            # Per-system non-autograd output (energy), already replicated
            # across ranks by the forward all-reduce.
            reduced[key] = value
    return reduced
