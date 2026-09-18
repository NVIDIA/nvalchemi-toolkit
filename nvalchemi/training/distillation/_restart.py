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
"""Restart state of an interrupted on-policy segment loop.

A strategy checkpoint carries weights, optimizer state, and counters, none of
which say where the propagator had got to. This module packs the missing half
— the live trajectory batch, the propagator's step count, the initial
structures' cursor, and the replay frames — as the flat tensor bundle a
:class:`~nvalchemi.hooks.CheckpointableHook` carries through the existing
hook-state file, and unpacks it on the way back in. A bundle is checked as it
is packed: the write half is the only one still next to a run that could be
relaunched, so an inconsistency is raised there rather than hours later at the
restore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data.batch import _INDEX_KEYS, Batch
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distillation.scoring import _NEIGHBOR_KEYS

if TYPE_CHECKING:
    from nvalchemi.hooks._context import TrainContext

_LEVEL_BY_GROUP = {"atoms": "atom", "edges": "edge", "system": "system"}
"""Batch storage group name mapped to the level a rebuild classifies fields by."""

_FLAT_SEPARATOR = ":"
"""Separator joining a field's level and name into one flat state-dict key."""


def _checked_counts(counts: list[int], name: str) -> list[int]:
    """Return *counts*, refusing the negative segment lengths a corrupt batch carries."""
    if any(count < 0 for count in counts):
        raise RuntimeError(
            f"The on-policy restart bundle's {name!r} holds a negative segment "
            f"length; got {counts!r}. The trajectory batch it describes can be "
            "neither packed nor rebuilt."
        )
    return counts


def _batch_state(batch: Batch, *, drop: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Return *batch* as a flat ``{level:field -> tensor}`` bundle.

    Every field is truncated to the rows its level's counts describe, since a
    storage that outlives its graphs — one compacted by
    :meth:`~nvalchemi.data.Batch.defrag`, or built by
    :meth:`~nvalchemi.data.Batch.empty` and filled part way — keeps rows
    behind ``num_nodes_list`` that no rebuild could split.

    Parameters
    ----------
    batch : Batch
        Batch to pack. It is read, never modified.
    drop : frozenset[str], optional
        Field names to leave out. Default an empty set.

    Returns
    -------
    dict[str, Any]
        Segment lengths plus one CPU tensor per stored field, keyed by level
        and name. Suitable for :func:`torch.save` under ``weights_only``.

    Raises
    ------
    RuntimeError
        If a segment length is negative, if a field holds fewer rows than its
        level's counts describe, or if a field carrying batch-global node
        indices is packed rather than dropped.
    """
    node_counts = _checked_counts(batch.num_nodes_list, "num_nodes_list")
    edge_counts = _checked_counts(batch.num_edges_list, "num_edges_list")
    limits = {
        "atom": sum(node_counts),
        "edge": sum(edge_counts),
        "system": batch.num_graphs,
    }
    state: dict[str, Any] = {
        "num_nodes_list": torch.tensor(node_counts, dtype=torch.long),
        "num_edges_list": torch.tensor(edge_counts, dtype=torch.long),
    }
    for group_name, group in batch._storage.groups.items():
        level = _LEVEL_BY_GROUP[group_name]
        limit = limits[level]
        for key, tensor in group.items():
            if key in drop:
                continue
            if key in _INDEX_KEYS:
                raise RuntimeError(
                    f"The on-policy restart bundle cannot carry {key!r}: it "
                    "holds batch-global node indices that Batch.from_raw_dicts "
                    "offsets a second time on the way back in, so the rebuilt "
                    "batch would index past its own nodes. Drop the neighbor "
                    "tensors, which are ephemeral and rebuilt from the "
                    "positions the bundle does carry."
                )
            if tensor.shape[0] < limit:
                raise RuntimeError(
                    f"The on-policy restart bundle's {level}{_FLAT_SEPARATOR}"
                    f"{key} holds {tensor.shape[0]} rows against the {limit} "
                    "its segment lengths describe; the trajectory batch is "
                    "internally inconsistent and cannot be packed."
                )
            state[f"{level}{_FLAT_SEPARATOR}{key}"] = tensor[:limit].detach().cpu()
    return state


def _batch_from_state(state: dict[str, Any]) -> Batch:
    """Rebuild the batch :func:`_batch_state` packed.

    Fields are split back into per-graph slices and handed to
    :meth:`~nvalchemi.data.Batch.from_raw_dicts` with the level of every one of
    them named explicitly, so a field the default key sets do not know — a
    ``teacher_*`` label, a dynamics counter — lands at the level it was stored
    at rather than at the raw-dict fallback.

    Parameters
    ----------
    state : dict[str, Any]
        Bundle produced by :func:`_batch_state`.

    Returns
    -------
    Batch
        Batch on the host, equal to the packed one field for field.

    Raises
    ------
    RuntimeError
        If a segment length is negative, or if a field's rows do not sum to
        the counts its level declares — the shape a bundle written before
        :func:`_batch_state` truncated to the kept graphs has.
    """
    node_counts = _checked_counts(
        [int(count) for count in state["num_nodes_list"]], "num_nodes_list"
    )
    edge_counts = _checked_counts(
        [int(count) for count in state["num_edges_list"]], "num_edges_list"
    )
    counts_by_level = {
        "atom": node_counts,
        "edge": edge_counts,
        "system": [1] * len(node_counts),
    }
    samples: list[dict[str, torch.Tensor]] = [{} for _ in node_counts]
    field_levels: dict[str, str] = {}
    for flat_key, tensor in state.items():
        level, separator, key = flat_key.partition(_FLAT_SEPARATOR)
        if not separator:
            continue
        field_levels[key] = level
        counts = counts_by_level[level]
        if tensor.shape[0] != sum(counts):
            raise RuntimeError(
                f"The on-policy restart bundle's {flat_key} holds "
                f"{tensor.shape[0]} rows against the {sum(counts)} its segment "
                "lengths describe; the checkpoint it came from cannot be "
                "resumed."
            )
        for sample, chunk in zip(samples, torch.split(tensor, counts), strict=True):
            sample[key] = chunk
    return Batch.from_raw_dicts(samples, field_levels=field_levels)


class _OnPolicyRestartHook:
    """Carry the segment loop's propagator and replay state through a checkpoint.

    The hook owns no state of its own: it reads the live trajectory batch, the
    propagator's step count, the structure cursor, and the replay buffer off
    the strategy it is bound to when a checkpoint is written, and holds the
    restored bundle until the segment loop consumes it on the way back in. A
    strategy checkpointed outside a run, or before its first segment,
    contributes an empty bundle and restarts by seeding afresh; one whose
    generation has run dry contributes its frames and the exhaustion, so the
    restart keeps training on the buffer rather than reseeding.
    """

    frequency = 1
    stage = TrainingStage.SETUP

    def __init__(self) -> None:
        """Start unbound, with nothing restored."""
        self._strategy: Any = None
        self._restored: dict[str, Any] | None = None

    def prepare_strategy(self, strategy: Any) -> None:
        """Bind the strategy whose on-policy state this hook checkpoints."""
        self._strategy = strategy

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Bind the running strategy, for a run that reached setup by another path."""
        self._strategy = ctx.workflow

    def take(self) -> dict[str, Any] | None:
        """Return the restored bundle once, clearing it.

        Returns
        -------
        dict[str, Any] | None
            Bundle written by :meth:`load_state_dict`, or ``None`` when the run
            starts fresh. Cleared by the call, so a strategy that runs twice
            resumes once and then seeds normally.
        """
        restored, self._restored = self._restored, None
        return restored

    def state_dict(self) -> dict[str, Any]:
        """Return the live trajectory, propagator counter, structure cursor, and frames.

        Neither the trajectory's neighbor tensors nor the replay frames' are
        stored: they are ephemeral, rebuilt from the positions that are, and an
        edge index carried across a rebuild would be offset twice. The cursor
        travels because the trajectory does — a backfill after the restart has
        to continue the rows too, or the run is handed structures it already
        relaxed — and the settings travel as the description a resumed run
        compares its own against.
        """
        strategy = self._strategy
        if strategy is None:
            return {}
        state = strategy._on_policy_state
        exhausted = bool(strategy._generation_exhausted)
        if state is None and not exhausted:
            return {}
        buffer = strategy.replay_buffer
        config = strategy.on_policy
        bundle: dict[str, Any] = {
            "dynamics_step_count": torch.tensor(
                config.dynamics.step_count, dtype=torch.long
            ),
            "initial_structures": config.initial_structures.state_dict(),
            "settings": config.settings.model_dump(mode="json"),
            "generation_exhausted": exhausted,
        }
        if state is not None:
            bundle["trajectory"] = _batch_state(state, drop=_NEIGHBOR_KEYS)
        if buffer is not None and len(buffer) > 0:
            bundle["replay_frames"] = _batch_state(
                buffer.dataset.in_memory_batch, drop=_NEIGHBOR_KEYS
            )
        return bundle

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Hold a checkpoint's on-policy bundle for the next :meth:`take`."""
        self._restored = dict(state) or None
