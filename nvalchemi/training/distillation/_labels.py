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
"""Attachment of teacher signals to a batch as ordinary batch fields.

Shared by the offline path in
:mod:`nvalchemi.training.distillation.labeling`, which attaches labels before
persisting a chunk, and the online path in
:mod:`nvalchemi.training.distillation.strategy`, which attaches them to a live
training batch. The teacher namespace both paths write into, and the storage
pruning both do after dropping fields, live here for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from nvalchemi.data.level_storage import UniformLevelStorage

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.training.distillation.scoring import SignalLevel, TeacherLabels

_TEACHER_FIELD_PREFIX = "teacher_"
"""Namespace every teacher field lives in, clear of a propagator's own state."""


def _reject_foreign_fields(fields: Iterable[str], subject: str) -> None:
    """Refuse batch fields that fall outside the teacher namespace.

    Carries the one message every place the namespace is policed raises: a
    scorer's declared ``label_fields`` when a
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` is built, when a
    :class:`~nvalchemi.training.distillation.DistillationStrategy` validates
    its propagator, and when
    :func:`~nvalchemi.training.distillation.label_dataset` starts; and the
    fields a scorer actually returns at labeling time, which is what polices a
    scorer declaring nothing.

    Parameters
    ----------
    fields : Iterable[str]
        Batch field names to check.
    subject : str
        What the names came from, opening the message.

    Raises
    ------
    ValueError
        If any name falls outside the ``teacher_*`` namespace.
    """
    foreign = sorted(
        field for field in fields if not field.startswith(_TEACHER_FIELD_PREFIX)
    )
    if foreign:
        raise ValueError(
            f"{subject} must populate the 'teacher_*' namespace so the "
            "propagator's own energy and forces survive the step; got "
            f"{foreign!r}. Rename each into the namespace, or stop the scorer "
            "writing it."
        )


def _prune_empty_edges(batch: Batch) -> None:
    """Drop an edge group that dropping the neighbor list left with no fields.

    A store — or a stored frame — that keeps the group records edge pointers
    that no array backs, which a reader then has to reconcile against an edge
    count of zero.
    """
    edges = batch._storage.groups.get("edges")
    if edges is not None and next(edges.keys(), None) is None:
        batch._storage.groups.pop("edges")


def _ensure_system_group(batch: Batch) -> None:
    """Give *batch* a system group so system-level fields can be attached.

    A batch built from samples that carry no system-level field at all — bare
    positions and atomic numbers, say — has no system group, and
    :meth:`~nvalchemi.data.Batch.add_key` cannot create one. The group is
    materialized empty but sized, the form
    :class:`~nvalchemi.data.level_storage.BaseLevelStorage` accepts for exactly
    this case.
    """
    if "system" in batch._storage.groups:
        return
    batch._storage.groups["system"] = UniformLevelStorage(
        data=TensorDict({}, batch_size=[batch.num_graphs], device=batch.device),
        device=batch.device,
        attr_map=batch._storage.attr_map,
        validate=False,
    )


def _split_per_graph(
    batch: Batch, values: torch.Tensor, level: SignalLevel
) -> list[torch.Tensor]:
    """Split a concatenated teacher tensor into one entry per graph."""
    if level == "node":
        return list(torch.split(values, batch.num_nodes_list, dim=0))
    return [values[index : index + 1] for index in range(batch.num_graphs)]


def _attach_teacher_labels(batch: Batch, labels: TeacherLabels) -> None:
    """Attach every teacher label to *batch* at the level its signal declares.

    Existing fields of the same name are overwritten, so re-labeling a batch is
    idempotent.

    Parameters
    ----------
    batch : Batch
        Batch to attach the labels to; mutated in place.
    labels : TeacherLabels
        Mapping from batch field name to ``(tensor, level)`` as returned by
        :meth:`~nvalchemi.training.distillation.TeacherScorer.label`.
    """
    for field, (values, level) in labels.items():
        if level == "system":
            _ensure_system_group(batch)
        batch.add_key(
            field,
            _split_per_graph(batch, values, level),
            level=level,
            overwrite=True,
        )
