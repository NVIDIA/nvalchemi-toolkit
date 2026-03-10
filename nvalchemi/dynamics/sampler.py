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
"""Size-aware sampler for inflight batching in dynamics simulations.

This module provides :class:`SizeAwareSampler`, which manages dataset access,
capacity budgets, and bin-packing logic for efficient GPU utilization during
dynamics simulations.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Sampler

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch


class SizeAwareSampler(Sampler[int]):
    """Size-aware sampler for inflight batching.

    Manages dataset access, capacity budgets, and bin-packing logic for
    efficient GPU utilization during dynamics simulations. Ensures every
    replacement sample fits within the memory envelope of the graduated
    sample it replaces.

    When CUDA is available, the sampler uses a heuristic to estimate the
    maximum number of atoms that fit in GPU memory. This estimate is combined
    with user-specified ``max_atoms`` — the more restrictive constraint wins.
    The GPU memory heuristic is **best-effort** and **conservative**; users
    who need tighter control should set ``max_atoms`` explicitly.

    Parameters
    ----------
    dataset : Any
        Dataset with ``__len__``, ``__getitem__``, and ``get_metadata(idx)``
        methods. ``get_metadata`` must return ``(num_atoms, num_edges)``.
    max_atoms : int | None
        Maximum total atoms across all samples in a batch. ``None`` disables
        the atom count constraint (GPU memory estimate may still apply).
    max_edges : int | None
        Maximum total edges across all samples in a batch. ``None`` disables
        the edge count constraint.
    max_batch_size : int
        Maximum number of samples (graphs) in a batch.
    bin_width : int
        Atom-count bin width for grouping samples. Default 1.
    shuffle : bool
        Whether to shuffle within bins. Default False.
    max_gpu_memory_fraction : float
        Fraction of GPU memory to use when estimating atom capacity. Default
        0.8 (80%), leaving 20% headroom for model parameters and CUDA context.
        Only used when CUDA is available.

    Raises
    ------
    RuntimeError
        If any sample in the dataset has ``num_atoms > max_atoms`` or
        ``num_edges > max_edges`` — such samples can never be placed into
        any batch and indicate a configuration error.
    ValueError
        If ``max_batch_size < 1``, ``bin_width < 1``, or
        ``max_gpu_memory_fraction`` is not in ``(0.0, 1.0]``.

    Examples
    --------
    >>> sampler = SizeAwareSampler(dataset, max_atoms=100, max_edges=500, max_batch_size=10)
    >>> batch = sampler.build_initial_batch()
    >>> replacement = sampler.request_replacement(num_atoms=5, num_edges=20)
    """

    def __init__(
        self,
        dataset: Any,
        max_atoms: int | None,
        max_edges: int | None,
        max_batch_size: int,
        bin_width: int = 1,
        shuffle: bool = False,
        max_gpu_memory_fraction: float = 0.8,
    ) -> None:
        """Initialize the size-aware sampler.

        Parameters
        ----------
        dataset : Any
            Dataset with ``__len__``, ``__getitem__``, and ``get_metadata(idx)``
            methods. ``get_metadata`` must return ``(num_atoms, num_edges)``.
        max_atoms : int | None
            Maximum total atoms across all samples in a batch. ``None`` disables
            the atom count constraint (GPU memory estimate may still apply).
        max_edges : int | None
            Maximum total edges across all samples in a batch. ``None`` disables
            the edge count constraint.
        max_batch_size : int
            Maximum number of samples (graphs) in a batch.
        bin_width : int
            Atom-count bin width for grouping samples. Default 1.
        shuffle : bool
            Whether to shuffle within bins. Default False.
        max_gpu_memory_fraction : float
            Fraction of GPU memory to use when estimating atom capacity. Default
            0.8 (80%), leaving 20% headroom for model parameters and CUDA context.
            Only used when CUDA is available.

        Raises
        ------
        RuntimeError
            If any sample exceeds ``max_atoms`` or ``max_edges`` constraints.
        ValueError
            If ``max_batch_size < 1``, ``bin_width < 1``, or
            ``max_gpu_memory_fraction`` is not in ``(0.0, 1.0]``.
        TypeError
            If dataset does not implement required interface.
        """
        # Validate parameters
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if bin_width < 1:
            raise ValueError(f"bin_width must be >= 1, got {bin_width}")
        if not 0.0 < max_gpu_memory_fraction <= 1.0:
            raise ValueError(
                f"max_gpu_memory_fraction must be in (0.0, 1.0], got {max_gpu_memory_fraction}"
            )

        # Runtime validation of dataset interface
        if not hasattr(dataset, "__len__"):
            raise TypeError("dataset must implement __len__")
        if not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must implement __getitem__")
        if not hasattr(dataset, "get_metadata"):
            raise TypeError(
                "dataset must implement get_metadata(idx) -> (num_atoms, num_edges)"
            )

        self._dataset = dataset
        self._max_atoms = max_atoms
        self._max_edges = max_edges
        self._max_batch_size = max_batch_size
        self._bin_width = bin_width
        self._shuffle = shuffle
        self._max_gpu_memory_fraction = max_gpu_memory_fraction

        # Pre-scan dataset and build bins
        self._sample_meta: list[tuple[int, int]] = []  # (num_atoms, num_edges) per idx
        self._bins: dict[int, list[int]] = {}  # bin_key -> list of unconsumed indices
        self._consumed: set[int] = set()

        self._prescan_dataset()

    def _prescan_dataset(self) -> None:
        """Pre-scan all samples to extract metadata and organize into bins.

        Raises
        ------
        RuntimeError
            If any sample exceeds atom or edge constraints.
        """
        for idx in range(len(self._dataset)):
            num_atoms, num_edges = self._dataset.get_metadata(idx)

            # Validate sample fits within constraints
            if self._max_atoms is not None and num_atoms > self._max_atoms:
                raise RuntimeError(
                    f"Sample {idx} has {num_atoms} atoms, exceeding max_atoms={self._max_atoms}. "
                    "This sample can never fit in any batch."
                )
            if self._max_edges is not None and num_edges > self._max_edges:
                raise RuntimeError(
                    f"Sample {idx} has {num_edges} edges, exceeding max_edges={self._max_edges}. "
                    "This sample can never fit in any batch."
                )

            self._sample_meta.append((num_atoms, num_edges))

            # Assign to bin based on atom count
            bin_key = num_atoms // self._bin_width
            if bin_key not in self._bins:
                self._bins[bin_key] = []
            self._bins[bin_key].append(idx)

        # Optionally shuffle within bins
        if self._shuffle:
            for bin_indices in self._bins.values():
                random.shuffle(bin_indices)

    def _estimate_max_atoms_from_gpu(self) -> int | None:
        """Estimate the maximum number of atoms that fit in GPU memory.

        Uses ``torch.cuda.get_device_properties`` to query total GPU memory
        and estimates the per-atom memory footprint. Returns ``None`` if CUDA
        is not available.

        This is a **heuristic** and **best-effort** estimate. The per-atom
        memory footprint (~300 bytes) is conservative for typical MLIP
        workloads. Users who need tighter control should set ``max_atoms``
        explicitly.

        Returns
        -------
        int | None
            Estimated max atoms, or ``None`` if CUDA is unavailable.
        """
        if not torch.cuda.is_available():
            return None

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_mem = props.total_memory
        usable_mem = int(total_mem * self._max_gpu_memory_fraction)

        # Estimate per-atom memory: each atom needs storage for
        # positions (3 * 4 bytes float32), atomic_numbers (8 bytes long),
        # forces (3 * 4 bytes), velocities (3 * 4 bytes),
        # atomic_masses (4 bytes), batch index (8 bytes),
        # plus model hidden states (estimate ~256 bytes per atom for embeddings)
        # Conservative estimate: ~300 bytes per atom
        bytes_per_atom = 300

        # Also account for model parameters and CUDA overhead (~20% of memory)
        model_overhead = int(total_mem * 0.2)
        available_for_data = max(usable_mem - model_overhead, 0)

        return max(available_for_data // bytes_per_atom, 1)

    def build_initial_batch(self) -> Batch:
        """Build an initial batch using greedy bin packing.

        Iterates bins in ascending order (smallest atom counts first) and adds
        samples that fit within all capacity constraints until no more samples
        can be added. When CUDA is available, the effective ``max_atoms`` is
        the minimum of the user-specified value and the GPU memory estimate.

        Returns
        -------
        Batch
            A batch with ``status`` and ``fmax`` attributes initialized.

        Raises
        ------
        RuntimeError
            If no samples can be added to the batch (e.g., all consumed or
            constraints too tight).
        """
        # Determine effective max_atoms from user constraint and GPU estimate
        gpu_max_atoms = self._estimate_max_atoms_from_gpu()
        effective_max_atoms = self._max_atoms
        if gpu_max_atoms is not None:
            if effective_max_atoms is not None:
                effective_max_atoms = min(effective_max_atoms, gpu_max_atoms)
            else:
                effective_max_atoms = gpu_max_atoms

        data_list: list[AtomicData] = []
        total_atoms = 0
        total_edges = 0

        # Iterate bins in ascending order (smallest first)
        sorted_bin_keys = sorted(self._bins.keys())

        for bin_key in sorted_bin_keys:
            if bin_key not in self._bins:
                continue

            # Iterate through samples in this bin
            indices_to_remove: list[int] = []
            for idx in self._bins[bin_key]:
                if idx in self._consumed:
                    continue

                num_atoms, num_edges = self._sample_meta[idx]

                # Check capacity constraints
                if len(data_list) >= self._max_batch_size:
                    break
                if (
                    effective_max_atoms is not None
                    and total_atoms + num_atoms > effective_max_atoms
                ):
                    continue
                if (
                    self._max_edges is not None
                    and total_edges + num_edges > self._max_edges
                ):
                    continue

                # Sample fits, load it
                data, _ = self._dataset[idx]
                data_list.append(data)
                total_atoms += num_atoms
                total_edges += num_edges
                self._consumed.add(idx)
                indices_to_remove.append(idx)

            # Remove consumed indices from bin
            for idx in indices_to_remove:
                self._bins[bin_key].remove(idx)

            # Stop if batch is full
            if len(data_list) >= self._max_batch_size:
                break

        if not data_list:
            raise RuntimeError(
                "Cannot build initial batch: no samples available or constraints too tight."
            )

        # Create batch
        batch = Batch.from_data_list(data_list, device=data_list[0].device)

        # Initialize status and fmax attributes
        batch.__dict__["status"] = torch.zeros(
            batch.num_graphs, 1, dtype=torch.long, device=batch.device
        )
        batch.__dict__["fmax"] = torch.full(
            (batch.num_graphs, 1),
            float("inf"),
            dtype=torch.float32,
            device=batch.device,
        )

        return batch

    def request_replacement(self, num_atoms: int, num_edges: int) -> AtomicData | None:
        """Request a replacement sample that fits within the given constraints.

        Searches for an unconsumed sample with at most ``num_atoms`` atoms and
        ``num_edges`` edges, starting from the target bin and progressively
        searching smaller bins.

        Parameters
        ----------
        num_atoms : int
            Maximum number of atoms the replacement can have.
        num_edges : int
            Maximum number of edges the replacement can have.

        Returns
        -------
        AtomicData | None
            A replacement sample if found, or ``None`` if no suitable sample
            is available.
        """
        target_bin = num_atoms // self._bin_width

        # Search from target bin downward (smaller sizes)
        for bin_key in range(target_bin, -1, -1):
            if bin_key not in self._bins:
                continue

            for idx in list(self._bins[bin_key]):  # Copy list to allow modification
                if idx in self._consumed:
                    continue

                cand_atoms, cand_edges = self._sample_meta[idx]

                # Check if candidate fits in the slot
                if cand_atoms <= num_atoms and cand_edges <= num_edges:
                    # Found a match, load and mark consumed
                    data, _ = self._dataset[idx]
                    self._consumed.add(idx)
                    self._bins[bin_key].remove(idx)
                    return data

        return None

    @property
    def exhausted(self) -> bool:
        """Check if all samples have been consumed.

        Returns
        -------
        bool
            ``True`` if all bins are empty, ``False`` otherwise.
        """
        return all(len(indices) == 0 for indices in self._bins.values())

    def __iter__(self) -> Iterator[int]:
        """Yield all remaining unconsumed indices in size-grouped order.

        Yields
        ------
        int
            Dataset indices in ascending bin order.
        """
        sorted_bin_keys = sorted(self._bins.keys())
        for bin_key in sorted_bin_keys:
            if bin_key not in self._bins:
                continue
            for idx in self._bins[bin_key]:
                if idx not in self._consumed:
                    yield idx

    def __len__(self) -> int:
        """Return the number of unconsumed samples.

        Returns
        -------
        int
            Number of samples remaining in the sampler.
        """
        return len(self._dataset) - len(self._consumed)
