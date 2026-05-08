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
r"""Graph-aware reduction primitives for loss functions.

Scatter reductions (``V ... → B ...``)
--------------------------------------

:func:`per_graph_sum` and :func:`per_graph_mean` take a flat per-node
tensor with a ``batch_idx`` mapping each node to its graph and reduce
the leading node dim into a per-graph output, preserving trailing dims
verbatim.

These helpers only produce per-graph tensors. They do not choose the
final scalar weighting across graphs. For per-graph values :math:`x_i`,
a graph-balanced scalar is :math:`B^{-1} \sum_i x_i`, while an
atom-weighted scalar is :math:`(\sum_i N_i x_i) / (\sum_i N_i)`.

Matrix reductions (``B ... m n → B ...``)
-----------------------------------------

:func:`frobenius_mse` is *not* a scatter reduction: it operates on an
already-per-graph tensor and averages the squared residual over the
trailing two matrix dims. It takes neither ``batch_idx`` nor
``num_graphs``.

Common parameters for scatter reductions
----------------------------------------

- ``values``: per-node tensor whose leading dim indexes nodes; trailing
  dims are preserved.
- ``batch_idx``: 1-D ``BatchIndices`` mapping each node to its graph.
- ``num_graphs`` (optional): when supplied, trusted without scanning
  ``batch_idx`` — the recommended hot-path convention (avoids a GPU→CPU
  sync). When omitted, inferred as ``batch_idx.max().item() + 1``.
  Empty ``batch_idx`` always requires ``num_graphs``.

Scatter reductions raise :class:`ValueError` on shape mismatch, on
non-positive ``num_graphs``, or on inability to infer ``num_graphs``.

Migrating from ``per_graph_mse``
--------------------------------

The former ``per_graph_mse`` helper has been removed. The direct
replacement composes :func:`per_graph_mean` with a pointwise squared
error::

    per_graph_mean((pred - target).pow(2), batch_idx, num_graphs)

Hot-path callers that want a scalar-per-graph MSE should reduce
trailing dims before scattering::

    per_graph_mean(
        (pred - target).pow(2).mean(dim=tuple(range(1, pred.ndim))),
        batch_idx,
        num_graphs,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

from nvalchemi._typing import BatchIndices

if TYPE_CHECKING:
    from jaxtyping import Float, Num

_NumGraphs: TypeAlias = int | torch.Tensor


def _resolve_batch_indices(
    batch_idx: BatchIndices,
    num_graphs: int | None,
    device: torch.device,
) -> tuple[BatchIndices, _NumGraphs]:
    """Return ``batch_idx`` on ``device`` and the graph count it implies.

    When ``num_graphs`` is supplied, trust it and skip any scan of
    ``batch_idx`` — this is the hot path. When omitted, infer via
    ``batch_idx.max()``; under ``torch.compile`` the max stays a tensor,
    otherwise it is materialized on the host (forcing a device sync).
    """
    batch_idx = batch_idx.to(device=device, dtype=torch.long)
    if batch_idx.ndim != 1:
        raise ValueError(f"batch_idx must be 1D, got shape {tuple(batch_idx.shape)}")
    if num_graphs is not None:
        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")
        return batch_idx, num_graphs
    if batch_idx.numel() == 0:
        raise ValueError(
            "Cannot infer num_graphs from empty batch_idx; "
            "pass num_graphs explicitly when reducing an empty batch."
        )
    if torch.compiler.is_compiling():
        return batch_idx, batch_idx.max() + 1
    return batch_idx, int(batch_idx.max().item()) + 1


def _check_leading_dim(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    *,
    name: str,
) -> None:
    """Validate that ``values`` and ``batch_idx`` have matching leading dims."""
    if values.shape[0] != batch_idx.shape[0]:
        raise ValueError(
            f"{name} leading dim ({values.shape[0]}) must match "
            f"batch_idx length ({batch_idx.shape[0]})"
        )


def _prep_reduction(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None,
    *,
    name: str,
) -> tuple[BatchIndices, _NumGraphs]:
    """Validate leading dim and resolve ``(batch_idx, num_graphs)`` on values' device."""
    _check_leading_dim(values, batch_idx, name=name)
    return _resolve_batch_indices(batch_idx, num_graphs, values.device)


def _per_graph_sum_resolved(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: _NumGraphs,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    """Sum per-node values after ``batch_idx`` and ``num_graphs`` are resolved."""
    out_shape = (num_graphs, *values.shape[1:])
    out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
    idx_shape = [1] * (values.ndim - 1)
    index = batch_idx.view(-1, *idx_shape).expand_as(values)
    # TODO: refactor to use warp kernels when backwards ready
    out.scatter_add_(0, index, values)
    return out


def _num_nodes_per_graph(
    batch_idx: BatchIndices,
    num_graphs: _NumGraphs,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Num[torch.Tensor, "B"]:  # noqa: F722
    """Count nodes per graph via :func:`torch.bincount` (single kernel, no scratch)."""
    minlength = int(num_graphs) if isinstance(num_graphs, int) else num_graphs
    counts = torch.bincount(batch_idx, minlength=minlength)
    return counts.to(device=device, dtype=dtype)


def per_graph_sum(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    """Sum per-node values into per-graph values via ``scatter_add_``.

    See the module docstring for ``batch_idx`` / ``num_graphs``
    semantics and error conditions.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph sums of shape ``(num_graphs, *values.shape[1:])``.
    """
    batch_idx, resolved = _prep_reduction(values, batch_idx, num_graphs, name="values")
    return _per_graph_sum_resolved(values, batch_idx, resolved)


def per_graph_mean(
    values: Float[torch.Tensor, "V ..."],  # noqa: F722
    batch_idx: BatchIndices,
    num_graphs: int | None = None,
) -> Float[torch.Tensor, "B ..."]:  # noqa: F722
    r"""Mean of per-node values across each graph.

    This divides each graph's sum by that graph's node count and returns
    one value per graph. It does not choose how those graph values are
    weighted in a later scalar reduction. For per-graph values
    :math:`x_i`, the graph-balanced scalar is :math:`B^{-1} \sum_i x_i`;
    the atom-weighted scalar is
    :math:`(\sum_i N_i x_i) / (\sum_i N_i)`.

    Empty graphs (zero nodes) are safe: their sum is zero and their
    count is clamped to ``1`` before the division, so they yield zero.
    See the module docstring for shared parameter / error semantics.

    Returns
    -------
    Float[torch.Tensor, "B ..."]
        Per-graph means.
    """
    batch_idx, resolved = _prep_reduction(values, batch_idx, num_graphs, name="values")
    totals = _per_graph_sum_resolved(values, batch_idx, resolved)
    counts = _num_nodes_per_graph(
        batch_idx,
        resolved,
        dtype=totals.dtype,
        device=totals.device,
    ).clamp_min(1.0)
    # Broadcast counts across trailing dims of totals.
    count_shape = [1] * (totals.ndim - 1)
    counts = counts.view(-1, *count_shape)
    return totals / counts


def frobenius_mse(
    pred: Float[torch.Tensor, "B 3 3"],  # noqa: F722
    target: Float[torch.Tensor, "B 3 3"],  # noqa: F722
) -> Float[torch.Tensor, "B"]:  # noqa: F722
    """Per-graph Frobenius MSE over the trailing two matrix dims.

    Returns ``((pred - target) ** 2).mean(dim=(-2, -1))`` — the squared
    Frobenius norm of the residual matrix, averaged over its entries.
    Canonical use is on stress tensors of shape ``(B, 3, 3)``.

    Parameters
    ----------
    pred, target
        Same-shape per-graph matrix tensors.

    Returns
    -------
    Float[torch.Tensor, "B"]
        Per-graph Frobenius MSE.

    Raises
    ------
    ValueError
        If shapes differ or input is not at least a batched matrix
        tensor (``ndim >= 3``).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal target shape "
            f"{tuple(target.shape)}"
        )
    if pred.ndim < 3:
        raise ValueError(
            f"frobenius_mse expects at least 3 dims (B, ..., M1, M2); "
            f"got shape {tuple(pred.shape)}"
        )
    return (pred - target).pow(2).mean(dim=(-2, -1))
