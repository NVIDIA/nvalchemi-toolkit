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
"""Attach teacher labels to a batch as ordinary fields at their signal levels.

Shared by the offline path in
:mod:`nvalchemi.training.distillation.labeling`, which attaches labels before
persisting a chunk, and the online path in
:mod:`nvalchemi.training.distillation.strategy`, which attaches them to a live
training batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.training.distillation.scoring import _reject_foreign_fields

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.training.distillation.scoring import SignalLevel, TeacherLabels


def _split_per_graph(
    batch: Batch, field: str, values: torch.Tensor, level: SignalLevel
) -> list[torch.Tensor]:
    """Split a concatenated teacher tensor into one entry per graph.

    The row count is checked against the level first, so a node-sized tensor a
    scorer mislabels as a system-level signal is refused rather than silently
    cut down to its first ``num_graphs`` rows.

    Raises
    ------
    ValueError
        If *values* does not hold one row per atom or per graph. The split
        would otherwise drop the surplus rows before
        :meth:`~nvalchemi.data.Batch.add_key` or the store's schema checks
        could see them.
    """
    expected = batch.num_nodes if level == "node" else batch.num_graphs
    if values.ndim == 0 or values.shape[0] != expected:
        raise ValueError(
            f"Teacher label {field!r} at level {level!r} has shape "
            f"{tuple(values.shape)!r}; expected {expected!r} rows, one per "
            f"{'atom' if level == 'node' else 'graph'}."
        )
    if level == "node":
        return list(torch.split(values, batch.num_nodes_list, dim=0))
    return [values[index : index + 1] for index in range(batch.num_graphs)]


def _attach_teacher_labels(batch: Batch, labels: TeacherLabels) -> None:
    """Attach every teacher label to *batch* at the level its signal declares.

    Every labeling route ends here with the scorer's output verbatim, so this is
    where the :class:`~nvalchemi.training.distillation.TeacherScorer` contract
    is enforced: fields stay in the ``teacher_*`` namespace, each is attached at
    ``"node"`` or ``"system"`` level, and each tensor is detached and moved to
    the batch device before it lands, so a scorer's autograd graph never reaches
    stored or training data and a device mismatch is not left for collation to
    report. Existing fields of the same name are overwritten, so re-labeling a
    batch is idempotent.

    Parameters
    ----------
    batch : Batch
        Batch to attach the labels to; mutated in place.
    labels : TeacherLabels
        Mapping from batch field name to ``(tensor, level)`` as returned by
        :meth:`~nvalchemi.training.distillation.TeacherScorer.label`.

    Raises
    ------
    ValueError
        If a field falls outside the ``teacher_*`` namespace, declares a level
        other than ``"node"`` or ``"system"``, or has a row count other than
        one per atom or one per graph for its level.
    """
    _reject_foreign_fields(labels.keys(), "Teacher labels")
    for field, (values, level) in labels.items():
        if level not in ("node", "system"):
            raise ValueError(
                f"Teacher label {field!r} declares unknown level {level!r}; a scorer "
                "attaches each field at 'node' or 'system'."
            )
        values = values.detach().to(batch.device)
        batch.add_key(
            field,
            _split_per_graph(batch, field, values, level),
            level=level,
            overwrite=True,
        )
