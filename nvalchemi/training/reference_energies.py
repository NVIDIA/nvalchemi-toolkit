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
"""Utilities for computing reference energies prior to training or fine-tuning
some models.

TODO: Revisit this module's location and API.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Iterable

import torch

from nvalchemi.data import Batch

__all__ = [
    "AtomicReferenceEnergyFit",
    "fit_atomic_reference_energies",
]


@dataclasses.dataclass(frozen=True, slots=True)
class AtomicReferenceEnergyFit:
    """Least-squares fit of per-element reference energies.

    Parameters
    ----------
    reference_energies : dict[int, float]
        Fitted per-element reference energies keyed by atomic number.
    intercept : float
        Fitted per-structure intercept. This is zero when ``fit_intercept=False``.
    atomic_numbers : tuple[int, ...]
        Element ordering used in the solve.
    rank : int
        Numerical rank of the accumulated design matrix.
    num_structures : int
        Number of graph-level energy observations used in the fit.
    residual_sum_squares : float
        Sum of squared residuals under the fitted compositional model.
    """

    reference_energies: dict[int, float]
    intercept: float
    atomic_numbers: tuple[int, ...]
    rank: int
    num_structures: int
    residual_sum_squares: float


BaselineEnergyFn = Callable[[Batch], torch.Tensor]


def fit_atomic_reference_energies(
    batches: Iterable[Batch],
    *,
    atomic_numbers: Iterable[int] | None = None,
    energy_key: str = "energy",
    baseline_energy_fn: BaselineEnergyFn | None = None,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
    fit_intercept: bool = False,
) -> AtomicReferenceEnergyFit:
    """Fit per-element reference energies from graph-level energies.

    The fitted model is

    ``energy ~= sum_Z count_Z * reference_energy_Z + intercept``.

    When ``baseline_energy_fn`` is provided, it is solving for energies relative
    to the baseline, i.e. ``target_energy - baseline_energy_fn(batch)``.

    Parameters
    ----------
    batches : Iterable[Batch]
        Iterable yielding :class:`~nvalchemi.data.Batch` objects.
    atomic_numbers : Iterable[int] | None, optional
        Element subset/order to fit. If omitted, elements are inferred from the data
        before constructing the least-squares matrix.
    energy_key : str, optional
        Batch attribute containing one graph-level target energy per structure.
    baseline_energy_fn : Callable[[Batch], torch.Tensor] | None, optional
        Optional function returning one baseline energy per graph. The fit uses
        target minus this baseline.
    device : torch.device | str | None, optional
        Device used for optional baseline-energy evaluation. Batches are moved
        to this device before calling ``baseline_energy_fn``; the least-squares
        fit itself is performed on CPU.
    max_batches : int | None, optional
        Maximum number of batches to consume.
    fit_intercept : bool, optional
        If ``True``, also fit a global per-structure intercept.

    Returns
    -------
    AtomicReferenceEnergyFit
        Fit coefficients and diagnostics.
    """
    elements = _normalize_atomic_numbers(atomic_numbers)
    target_device = torch.device(device) if device is not None else None
    count_inputs: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    target_batches: list[torch.Tensor] = []

    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        if not isinstance(batch, Batch):
            raise TypeError(
                "Reference-energy fitting expects Batch items, "
                f"got {type(batch).__name__}."
            )
        count_inputs.append(_batch_count_inputs(batch, elements))
        if baseline_energy_fn is not None and target_device is not None:
            batch = batch.to(target_device, non_blocking=True)

        target = _batch_energy_vector(batch, energy_key)
        if baseline_energy_fn is not None:
            with torch.no_grad():
                baseline = baseline_energy_fn(batch)
            target = target - _validate_baseline_energy(baseline, batch)
        target_batches.append(target)

    if not count_inputs:
        raise ValueError("Cannot fit reference energies from an empty iterable.")

    if elements is None:
        elements = sorted(
            {
                int(atomic_number)
                for atomic_numbers_batch, _, _ in count_inputs
                for atomic_number in atomic_numbers_batch.unique().tolist()
            }
        )
    if not elements:
        raise ValueError("Cannot fit reference energies without atomic numbers.")

    x = torch.cat(
        [
            _batch_count_matrix(
                atomic_numbers_batch, graph_indices, num_graphs, elements
            )
            for atomic_numbers_batch, graph_indices, num_graphs in count_inputs
        ],
        dim=0,
    )
    if fit_intercept:
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
        x = torch.cat((x, ones), dim=1)
    y = torch.cat(target_batches, dim=0)

    rank = int(torch.linalg.matrix_rank(x).item())
    parameter_count = x.shape[1]
    if rank < parameter_count:
        warnings.warn(
            "Reference-energy least-squares matrix is rank deficient: "
            f"rank={rank}/{parameter_count}. "
            "The returned coefficients are the minimum-norm solution; "
            "some values may be underdetermined.",
            RuntimeWarning,
            stacklevel=2,
        )

    solution = torch.linalg.lstsq(x, y, driver="gelsd").solution
    residual = x @ solution - y
    element_solution = solution[: len(elements)]
    intercept = float(solution[-1].item()) if fit_intercept else 0.0
    return AtomicReferenceEnergyFit(
        reference_energies={
            atomic_number: float(energy)
            for atomic_number, energy in zip(elements, element_solution.tolist())
        },
        intercept=intercept,
        atomic_numbers=tuple(elements),
        rank=rank,
        num_structures=int(y.numel()),
        residual_sum_squares=float((residual @ residual).item()),
    )


def _normalize_atomic_numbers(
    atomic_numbers: Iterable[int] | None,
) -> list[int] | None:
    """Return sorted unique atomic numbers, or ``None`` for inference."""
    if atomic_numbers is None:
        return None
    normalized = sorted({int(atomic_number) for atomic_number in atomic_numbers})
    if not normalized:
        raise ValueError("atomic_numbers must not be empty.")
    return normalized


def _batch_energy_vector(batch: Batch, energy_key: str) -> torch.Tensor:
    """Return graph energies as a CPU float64 vector."""
    if not hasattr(batch, energy_key):
        raise KeyError(
            f"Batches used for reference-energy fitting must provide {energy_key!r}."
        )
    energy = getattr(batch, energy_key).detach().cpu().reshape(-1).to(torch.float64)
    if energy.numel() != int(batch.num_graphs):
        raise ValueError(
            f"Batch {energy_key!r} must contain one value per graph: "
            f"{energy.numel()} != {int(batch.num_graphs)}."
        )
    return energy


def _validate_baseline_energy(baseline: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Return validated baseline energies as a CPU float64 vector."""
    baseline = baseline.detach().cpu().reshape(-1).to(torch.float64)
    if baseline.numel() != int(batch.num_graphs):
        raise ValueError(
            "baseline_energy_fn must return one value per graph: "
            f"{baseline.numel()} != {int(batch.num_graphs)}."
        )
    return baseline


def _batch_count_inputs(
    batch: Batch,
    atomic_numbers: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return CPU tensors needed to build graph-by-element counts."""
    numbers = batch.atomic_numbers.detach().cpu().to(torch.long).reshape(-1)
    graph_indices = batch.batch_idx.detach().cpu().to(torch.long).reshape(-1)
    if numbers.numel() != graph_indices.numel():
        raise ValueError(
            "Batch atomic_numbers and batch_idx must have the same length: "
            f"{numbers.numel()} != {graph_indices.numel()}."
        )
    if atomic_numbers is not None:
        allowed = torch.tensor(atomic_numbers, dtype=torch.long)
        unknown = numbers[~torch.isin(numbers, allowed)].unique()
        if unknown.numel() > 0:
            raise ValueError(
                "Batch contains atomic numbers outside the fitted subset: "
                f"{unknown.tolist()}."
            )
    return numbers, graph_indices, int(batch.num_graphs)


def _batch_count_matrix(
    numbers: torch.Tensor,
    graph_indices: torch.Tensor,
    num_graphs: int,
    atomic_numbers: list[int],
) -> torch.Tensor:
    """Build a graph-by-element count matrix with tensor operations."""
    element_tensor = torch.tensor(atomic_numbers, dtype=torch.long)
    matches = numbers[:, None] == element_tensor[None, :]
    if not torch.all(matches.any(dim=1)):
        unknown = numbers[~matches.any(dim=1)].unique()
        raise ValueError(
            f"Batch contains unexpected atomic numbers: {unknown.tolist()}."
        )

    element_indices = matches.to(torch.long).argmax(dim=1)
    flat_indices = graph_indices * len(atomic_numbers) + element_indices
    counts = torch.bincount(
        flat_indices,
        minlength=num_graphs * len(atomic_numbers),
    )
    return counts.reshape(num_graphs, len(atomic_numbers)).to(torch.float64)
