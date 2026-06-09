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
"""Quick Zarr I/O benchmark for measuring write/read throughput and compression.

Run with::

    nvalchemi-io-test --help
    nvalchemi-io-test --num-systems 1000 5000 --codec zstd --chunk-size 10000

Readback uses the dataloader fused-prefetch path by default. To compare
against one-sample-at-a-time reads::

    nvalchemi-io-test -n 1000 --read-mode both --batch-size 64 --prefetch-factor 8
    nvalchemi-io-test -n 1000 --read-mode single
    nvalchemi-io-test -n 1000 --read-order shuffle
    nvalchemi-io-test -n 1000 --read-order block-shuffle --read-order-block-size 8192

Edge-specific chunking (useful for large graphs)::

    nvalchemi-io-test -n 100 --codec zstd --chunk-size 10000 --edge-chunk-size 5000
    nvalchemi-io-test -n 100 --codec zstd --shard-size 500 --edge-shard-size 500

"""

from __future__ import annotations

import random
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import click
import torch
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from nvalchemi.data.atomic_data import AtomicData

console = Console(stderr=True)

ReadMode: TypeAlias = Literal["batch", "single"]
ReadOrder: TypeAlias = Literal["sequential", "shuffle", "block-shuffle"]
DEFAULT_BATCH_SIZE = 64
DEFAULT_PREFETCH_FACTOR = 16
DEFAULT_READ_ORDER_BLOCK_SIZE = 8192


def _make_atomic_data(num_atoms: int, num_edges: int) -> AtomicData:
    """Create a minimal AtomicData with random data.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the structure.
    num_edges : int
        Number of edges in the structure.

    Returns
    -------
    AtomicData
        Random AtomicData instance.
    """
    from nvalchemi.data.atomic_data import AtomicData

    return AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
        forces=torch.randn(num_atoms, 3),
        energy=torch.randn(1, 1),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        neighbor_list=torch.stack(
            [
                torch.randint(0, max(num_atoms, 1), (num_edges,)),
                torch.randint(0, max(num_atoms, 1), (num_edges,)),
            ],
            dim=1,
        ),
        shifts=torch.randn(num_edges, 3),
    )


def _plan_data(
    num_systems: int,
    min_atoms: int,
    max_atoms: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Pre-compute atom/edge counts for a batch of structures.

    Parameters
    ----------
    num_systems : int
        Number of structures to plan.
    min_atoms : int
        Minimum atom count per structure.
    max_atoms : int
        Maximum atom count per structure.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[int, int]]
        List of (num_atoms, num_edges) pairs.
    """
    rng = random.Random(seed)
    plan = []
    for _ in range(num_systems):
        n_atoms = rng.randint(min_atoms, max_atoms)
        n_edges = rng.randint(1, n_atoms * 4)
        plan.append((n_atoms, n_edges))
    return plan


def _generate_from_plan(plan: list[tuple[int, int]]) -> list[AtomicData]:
    """Create AtomicData list from a pre-computed plan.

    Parameters
    ----------
    plan : list[tuple[int, int]]
        List of (num_atoms, num_edges) pairs.

    Returns
    -------
    list[AtomicData]
        Generated structures.
    """
    return [_make_atomic_data(n_atoms, n_edges) for n_atoms, n_edges in plan]


def _estimate_uncompressed_size(
    plan: list[tuple[int, int]],
) -> int:
    """Estimate uncompressed size in bytes using actual tensor footprints.

    Creates one representative sample from the plan, measures
    ``Tensor.nbytes`` for each field, uses ``_get_field_level`` to classify
    fields as atom / edge / system level, and scales to the full plan.

    Parameters
    ----------
    plan : list[tuple[int, int]]
        Pre-computed ``(num_atoms, num_edges)`` pairs.

    Returns
    -------
    int
        Estimated bytes.
    """
    if not plan:
        return 0

    from nvalchemi.data.datapipes.backends.zarr import _get_field_level

    num_systems = len(plan)
    total_atoms = sum(n for n, _ in plan)
    total_edges = sum(e for _, e in plan)

    # Create one sample to introspect field shapes, dtypes, and nbytes.
    ref_atoms, ref_edges = plan[0]
    ref = _make_atomic_data(ref_atoms, ref_edges)

    total = 0
    for key in ref.model_fields_set:
        val = getattr(ref, key, None)
        if not isinstance(val, torch.Tensor):
            continue

        level = _get_field_level(key)
        # Compute bytes per unit (atom, edge, or system) from nbytes.
        if level == "atom":
            total += (val.nbytes // max(ref_atoms, 1)) * total_atoms
        elif level == "edge":
            total += (val.nbytes // max(ref_edges, 1)) * total_edges
        else:
            # system-level: one row per structure
            total += val.nbytes * num_systems

    # Meta overhead: ptrs (2 × 8 × (B+1)) + masks (B + V + E)
    total += 2 * 8 * (num_systems + 1) + num_systems + total_atoms + total_edges

    return total


def _build_config(
    codec: str | None,
    level: int,
    chunk_size: int | None,
    shard_size: int | None,
    edge_chunk_size: int | None,
    edge_shard_size: int | None,
) -> dict | None:
    """Build a ZarrWriteConfig dict from CLI flags.

    Parameters
    ----------
    codec : str | None
        Codec name: "zstd", "lz4", "blosc-zstd", or None.
    level : int
        Compression level.
    chunk_size : int | None
        Chunk size along variable axis for node/system arrays.
    shard_size : int | None
        Shard size along variable axis for node/system arrays.
    edge_chunk_size : int | None
        Chunk size for edge-level arrays (neighbor_list, shifts).
    edge_shard_size : int | None
        Shard size for edge-level arrays (neighbor_list, shifts).

    Returns
    -------
    dict | None
        Config dict for ZarrWriteConfig, or None for defaults.
    """
    has_any = any(
        x is not None
        for x in (codec, chunk_size, shard_size, edge_chunk_size, edge_shard_size)
    )
    if not has_any:
        return None

    core_cfg: dict = {}
    compressor = None
    if codec is not None:
        from zarr.codecs import BloscCodec, ZstdCodec

        codec_map = {
            "zstd": lambda: ZstdCodec(level=level),
            "lz4": lambda: BloscCodec(cname="lz4", clevel=level),
            "blosc-zstd": lambda: BloscCodec(cname="zstd", clevel=level),
        }
        if codec not in codec_map:
            msg = f"Unknown codec: {codec!r}"
            raise click.BadParameter(msg)
        compressor = codec_map[codec]()
        core_cfg["compressors"] = (compressor,)

    if chunk_size is not None:
        core_cfg["chunk_size"] = chunk_size
    if shard_size is not None:
        core_cfg["shard_size"] = shard_size

    config: dict = {"core": core_cfg} if core_cfg else {}

    # Build edge-specific field overrides
    if edge_chunk_size is not None or edge_shard_size is not None:
        edge_cfg: dict = {}
        if edge_chunk_size is not None:
            edge_cfg["chunk_size"] = edge_chunk_size
        if edge_shard_size is not None:
            edge_cfg["shard_size"] = edge_shard_size
        if compressor is not None:
            edge_cfg["compressors"] = (compressor,)
        config["field_overrides"] = {
            "neighbor_list": edge_cfg,
            "shifts": edge_cfg,
        }

    return config if config else None


def _dir_size(path: Path) -> int:
    """Recursively compute total file size in bytes.

    Parameters
    ----------
    path : Path
        Directory to measure.

    Returns
    -------
    int
        Total bytes on disk.
    """
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _file_count(path: Path) -> int:
    """Count files in a directory tree (excluding directories).

    Parameters
    ----------
    path : Path
        Directory to count.

    Returns
    -------
    int
        Number of files.
    """
    return sum(1 for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string.

    Parameters
    ----------
    n : int
        Number of bytes.

    Returns
    -------
    str
        Formatted string.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _tensor_bytes(data: AtomicData | dict[str, torch.Tensor]) -> int:
    """Return the total tensor payload size in bytes.

    Parameters
    ----------
    data : AtomicData | dict[str, torch.Tensor]
        AtomicData object or raw tensor dictionary.

    Returns
    -------
    int
        Total tensor bytes.
    """
    if isinstance(data, dict):
        return sum(val.nelement() * val.element_size() for val in data.values())

    total = 0
    for key in data.model_fields_set:
        val = getattr(data, key, None)
        if isinstance(val, torch.Tensor):
            total += val.nelement() * val.element_size()
    return total


def _expand_read_modes(read_mode_options: tuple[str, ...]) -> tuple[ReadMode, ...]:
    """Expand CLI read-mode options into concrete benchmark modes.

    Parameters
    ----------
    read_mode_options : tuple[str, ...]
        CLI options. ``"both"`` expands to ``("batch", "single")``.

    Returns
    -------
    tuple[ReadMode, ...]
        Concrete readback modes in benchmark order.

    Raises
    ------
    click.BadParameter
        If an unknown mode is provided.
    """
    modes: list[ReadMode] = []
    for option in read_mode_options:
        normalized = option.lower()
        if normalized == "both":
            modes.extend(("batch", "single"))
        elif normalized in ("batch", "single"):
            modes.append(cast(ReadMode, normalized))
        else:
            msg = f"Unknown read mode: {option!r}"
            raise click.BadParameter(msg)

    return tuple(modes) if modes else ("batch",)


def _build_read_indices(
    expected_num_systems: int,
    read_order: ReadOrder,
    seed: int,
    read_order_block_size: int,
) -> list[int]:
    """Build the logical sample order used for readback benchmarking.

    Parameters
    ----------
    expected_num_systems : int
        Number of readable samples.
    read_order : {"sequential", "shuffle", "block-shuffle"}
        Logical index order to benchmark.
    seed : int
        Seed for randomized read orders.
    read_order_block_size : int
        Number of contiguous samples per shuffled block in block-shuffle mode.

    Returns
    -------
    list[int]
        Logical sample indices in readback order.

    Raises
    ------
    ValueError
        If *read_order_block_size* is less than 1 or *read_order* is unknown.
    """
    if read_order_block_size < 1:
        raise ValueError(
            f"read_order_block_size must be >= 1, got {read_order_block_size}"
        )

    indices = list(range(expected_num_systems))
    rng = random.Random(seed)

    if read_order == "sequential":
        return indices
    if read_order == "shuffle":
        rng.shuffle(indices)
        return indices
    if read_order == "block-shuffle":
        blocks = [
            indices[start : start + read_order_block_size]
            for start in range(0, expected_num_systems, read_order_block_size)
        ]
        rng.shuffle(blocks)
        return [index for block in blocks for index in block]

    msg = f"Unknown read order: {read_order!r}"
    raise ValueError(msg)


class _FixedOrderSampler(Sampler[int]):
    """Sampler that yields a precomputed logical read order."""

    def __init__(self, indices: Sequence[int]) -> None:
        self._indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        """Yield indices in the configured order."""
        return iter(self._indices)

    def __len__(self) -> int:
        """Return the number of configured sample indices."""
        return len(self._indices)


def _read_back_store(
    store_path: Path,
    expected_num_systems: int,
    read_mode: ReadMode = "batch",
    batch_size: int = DEFAULT_BATCH_SIZE,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    read_order: ReadOrder = "sequential",
    read_seed: int = 0,
    read_order_block_size: int = DEFAULT_READ_ORDER_BLOCK_SIZE,
    pin_memory: bool = False,
) -> tuple[float, int]:
    """Read every sample from a Zarr store and return timing and payload bytes.

    Parameters
    ----------
    store_path : Path
        Zarr store to read.
    expected_num_systems : int
        Expected number of readable samples.
    read_mode : {"batch", "single"}, default="batch"
        Readback path to benchmark. ``"batch"`` uses the public
        :class:`~nvalchemi.data.datapipes.DataLoader` path with fused
        prefetching; ``"single"`` uses one ``reader.read`` call per sample.
    batch_size : int, default=64
        Number of samples per emitted dataloader batch in batch mode.
    prefetch_factor : int, default=16
        Number of emitted dataloader batches to fuse into each backend read in
        batch mode. The effective read window is
        ``batch_size * prefetch_factor``.
    read_order : {"sequential", "shuffle", "block-shuffle"}, default="sequential"
        Logical sample order used for readback. ``"shuffle"`` models fully
        shuffled dataloading. ``"block-shuffle"`` shuffles contiguous index
        blocks while preserving locality inside each block.
    read_seed : int, default=0
        Seed for randomized read orders.
    read_order_block_size : int, default=8192
        Number of contiguous samples per shuffled block in block-shuffle mode.
    pin_memory : bool, default=False
        Request pinned CPU tensors from readers that support pinned-memory reads.

    Returns
    -------
    tuple[float, int]
        Read time in seconds and total tensor payload bytes read.

    Raises
    ------
    ValueError
        If *batch_size* is less than 1, if *prefetch_factor* is negative, if
        *read_order_block_size* is less than 1, or if *read_mode* /
        *read_order* is unknown.
    RuntimeError
        If the store does not expose the expected number of samples.
    """
    from nvalchemi.data.datapipes import DataLoader, Dataset
    from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if prefetch_factor < 0:
        raise ValueError(f"prefetch_factor must be >= 0, got {prefetch_factor}")

    read_indices = _build_read_indices(
        expected_num_systems,
        read_order,
        read_seed,
        read_order_block_size,
    )

    read_bytes = 0
    t0 = time.perf_counter()
    with AtomicDataZarrReader(store_path) as reader:
        if len(reader) != expected_num_systems:
            msg = (
                f"Expected {expected_num_systems} readable samples, "
                f"found {len(reader)}."
            )
            raise RuntimeError(msg)
        if read_mode == "batch":
            dataset = Dataset(reader, device="cpu", skip_validation=True)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=_FixedOrderSampler(read_indices),
                prefetch_factor=prefetch_factor,
                use_streams=False,
                pin_memory=pin_memory,
            )
            for batch in loader:
                read_bytes += sum(
                    value.nelement() * value.element_size()
                    for _key, value in batch
                    if isinstance(value, torch.Tensor)
                )
        elif read_mode == "single":
            for index in read_indices:
                data_dict, _metadata = reader.read(index)
                read_bytes += _tensor_bytes(data_dict)
        else:
            msg = f"Unknown read mode: {read_mode!r}"
            raise ValueError(msg)
    read_time = time.perf_counter() - t0
    return read_time, read_bytes


def _run_benchmark(
    num_systems_list: list[int],
    min_atoms: int,
    max_atoms: int,
    seed: int,
    config: dict | None,
    store_dir: Path,
    read_modes: tuple[ReadMode, ...] = ("batch",),
    batch_size: int = DEFAULT_BATCH_SIZE,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    read_order: ReadOrder = "sequential",
    read_seed: int = 0,
    read_order_block_size: int = DEFAULT_READ_ORDER_BLOCK_SIZE,
    pin_memory: bool = False,
) -> list[dict]:
    """Run the write/read benchmark for each system count.

    Parameters
    ----------
    num_systems_list : list[int]
        System counts to benchmark.
    min_atoms : int
        Minimum atoms per structure.
    max_atoms : int
        Maximum atoms per structure.
    seed : int
        Random seed.
    config : dict | None
        ZarrWriteConfig dict.
    store_dir : Path
        Temporary directory for Zarr stores.
    read_modes : tuple[ReadMode, ...], default=("batch",)
        Readback modes to benchmark for each written store.
    batch_size : int, default=64
        Number of samples per emitted dataloader batch in ``"batch"`` mode.
    prefetch_factor : int, default=16
        Number of emitted batches to fuse into each backend read in ``"batch"``
        mode.
    read_order : {"sequential", "shuffle", "block-shuffle"}, default="sequential"
        Logical sample order used during readback.
    read_seed : int, default=0
        Seed for randomized read orders.
    read_order_block_size : int, default=8192
        Number of contiguous samples per shuffled block in block-shuffle mode.
    pin_memory : bool, default=False
        Request pinned CPU tensors from readers that support pinned-memory reads.

    Returns
    -------
    list[dict]
        One result dict per system count.
    """
    from nvalchemi.data.datapipes.backends.zarr import (
        AtomicDataZarrWriter,
        ZarrWriteConfig,
    )

    write_config = (
        ZarrWriteConfig.model_validate(config) if config else ZarrWriteConfig()
    )
    if not read_modes:
        raise ValueError("At least one read mode must be provided.")

    # Pre-compute plans for all system counts
    max_systems = max(num_systems_list)
    full_plan = _plan_data(max_systems, min_atoms, max_atoms, seed)
    total_atoms_max = sum(n for n, _ in full_plan)
    total_edges_max = sum(e for _, e in full_plan)
    avg_atoms = total_atoms_max / max_systems
    avg_edges = total_edges_max / max_systems
    estimated_size = _estimate_uncompressed_size(full_plan)

    console.print(
        f"Pre-computed: [cyan]{max_systems:,}[/] systems, "
        f"[green]{total_atoms_max:,}[/] total atoms (avg {avg_atoms:.1f}), "
        f"[green]{total_edges_max:,}[/] total edges (avg {avg_edges:.1f})"
    )
    console.print(f"Estimated uncompressed: [yellow]{_fmt_bytes(estimated_size)}[/]")
    console.print()

    results = []
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        for num_systems in num_systems_list:
            task = progress.add_task(
                f"[cyan]{num_systems:>10,} systems", total=3 + len(read_modes)
            )

            # Step 1: generate data from pre-computed plan
            progress.update(task, description=f"[cyan]{num_systems:>10,} gen")
            plan = full_plan[:num_systems]
            data_list = _generate_from_plan(plan)
            total_atoms = sum(n for n, _ in plan)
            total_edges = sum(e for _, e in plan)
            progress.advance(task)

            # Step 2: write
            store_path = store_dir / f"bench_{num_systems}.zarr"
            progress.update(task, description=f"[cyan]{num_systems:>10,} write")
            writer = AtomicDataZarrWriter(store_path, config=write_config)
            t0 = time.perf_counter()
            writer.write(data_list)
            write_time = time.perf_counter() - t0
            progress.advance(task)

            # Step 3: read back through each requested path.
            read_results: list[tuple[ReadMode, float, int]] = []
            for read_mode in read_modes:
                progress.update(
                    task,
                    description=f"[cyan]{num_systems:>10,} read-{read_mode}",
                )
                read_time, read_bytes = _read_back_store(
                    store_path,
                    num_systems,
                    read_mode=read_mode,
                    batch_size=batch_size,
                    prefetch_factor=prefetch_factor,
                    read_order=read_order,
                    read_seed=read_seed,
                    read_order_block_size=read_order_block_size,
                    pin_memory=pin_memory,
                )
                read_results.append((read_mode, read_time, read_bytes))
                progress.advance(task)

            # Final step: measure
            progress.update(task, description=f"[cyan]{num_systems:>10,} measure")
            disk_bytes = _dir_size(store_path)
            num_files = _file_count(store_path)

            # compute uncompressed size from numpy arrays
            raw_bytes = sum(_tensor_bytes(d) for d in data_list)
            progress.advance(task)

            progress.update(
                task,
                description=f"[green]{num_systems:>10,} done",
            )

            avg_atoms_run = total_atoms / num_systems
            avg_edges_run = total_edges / num_systems
            ratio = raw_bytes / disk_bytes if disk_bytes > 0 else float("inf")

            for read_mode, read_time, read_bytes in read_results:
                profile_time = write_time + read_time
                results.append(
                    {
                        "num_systems": num_systems,
                        "read_mode": read_mode,
                        "read_order": read_order,
                        "read_order_block_size": (
                            read_order_block_size
                            if read_order == "block-shuffle"
                            else None
                        ),
                        "batch_size": batch_size if read_mode == "batch" else 1,
                        "prefetch_factor": (
                            prefetch_factor if read_mode == "batch" else 0
                        ),
                        "effective_read_window": (
                            batch_size * max(prefetch_factor, 1)
                            if read_mode == "batch"
                            else 1
                        ),
                        "total_atoms": total_atoms,
                        "total_edges": total_edges,
                        "avg_atoms": avg_atoms_run,
                        "avg_edges": avg_edges_run,
                        "raw_bytes": raw_bytes,
                        "disk_bytes": disk_bytes,
                        "read_bytes": read_bytes,
                        "num_files": num_files,
                        "ratio": ratio,
                        "write_time": write_time,
                        "read_time": read_time,
                        "profile_time": profile_time,
                        "write_throughput": (
                            num_systems / write_time if write_time > 0 else 0
                        ),
                        "read_throughput": (
                            num_systems / read_time if read_time > 0 else 0
                        ),
                        "profile_throughput": (
                            num_systems / profile_time if profile_time > 0 else 0
                        ),
                    }
                )

    return results


def _print_results(results: list[dict], config_desc: str) -> None:
    """Print benchmark results as a Rich table.

    Parameters
    ----------
    results : list[dict]
        Benchmark results.
    config_desc : str
        Description of the configuration used.
    """
    table = Table(
        title=f"Zarr I/O Roundtrip Benchmark — {config_desc}",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("Systems", justify="right", style="cyan", no_wrap=True)
    table.add_column("Read path", justify="left", no_wrap=True)
    table.add_column("Read order", justify="left", no_wrap=True)
    table.add_column("Batch", justify="right", no_wrap=True)
    table.add_column("Prefetch", justify="right", no_wrap=True)
    table.add_column("Read window", justify="right", no_wrap=True)
    table.add_column("Atoms", justify="right", no_wrap=True)
    table.add_column("Edges", justify="right", no_wrap=True)
    table.add_column("Raw", justify="right", no_wrap=True)
    table.add_column("Disk", justify="right", style="green", no_wrap=True)
    table.add_column("Ratio", justify="right", style="yellow", no_wrap=True)
    table.add_column("Write", justify="right", no_wrap=True)
    table.add_column("Read", justify="right", no_wrap=True)
    table.add_column("I/O/s", justify="right", style="bold", no_wrap=True)

    for r in results:
        table.add_row(
            f"{r['num_systems']:,}",
            r["read_mode"],
            r["read_order"],
            f"{r['batch_size']:,}",
            f"{r['prefetch_factor']:,}",
            f"{r['effective_read_window']:,}",
            f"{r['avg_atoms']:.0f}",
            f"{r['avg_edges']:.0f}",
            _fmt_bytes(r["raw_bytes"]),
            _fmt_bytes(r["disk_bytes"]),
            f"{r['ratio']:.2f}x",
            f"{r['write_time']:.2f}s",
            f"{r['read_time']:.2f}s",
            f"{r['profile_throughput']:,.0f}",
        )

    console.print()
    console.print(table)


def _run_read_benchmark(
    store_path: Path,
    read_modes: tuple[ReadMode, ...] = ("batch",),
    batch_size: int = DEFAULT_BATCH_SIZE,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    read_order: ReadOrder = "sequential",
    read_seed: int = 0,
    read_order_block_size: int = DEFAULT_READ_ORDER_BLOCK_SIZE,
    pin_memory: bool = False,
) -> list[dict]:
    """Benchmark read performance against an existing Zarr store.

    Parameters
    ----------
    store_path : Path
        Path to an existing Zarr store written by ``AtomicDataZarrWriter``.
    read_modes : tuple[ReadMode, ...], default=("batch",)
        Readback modes to benchmark.
    batch_size : int, default=64
        Number of samples per emitted dataloader batch in batch mode.
    prefetch_factor : int, default=16
        Number of emitted batches to fuse into each backend read in batch mode.
    read_order : {"sequential", "shuffle", "block-shuffle"}, default="sequential"
        Logical sample order for readback.
    read_seed : int, default=0
        Seed for randomized read orders.
    read_order_block_size : int, default=8192
        Block size for block-shuffle mode.
    pin_memory : bool, default=False
        Request pinned CPU tensors from readers that support pinned-memory reads.

    Returns
    -------
    list[dict]
        One result dict per read mode.
    """
    from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader

    if not read_modes:
        raise ValueError("At least one read mode must be provided.")

    with AtomicDataZarrReader(store_path) as reader:
        num_systems = len(reader)

    results = []
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        for read_mode in read_modes:
            task = progress.add_task(
                f"[cyan]read-{read_mode} ({read_order})",
                total=1,
            )
            read_time, read_bytes = _read_back_store(
                store_path,
                num_systems,
                read_mode=read_mode,
                batch_size=batch_size,
                prefetch_factor=prefetch_factor,
                read_order=read_order,
                read_seed=read_seed,
                read_order_block_size=read_order_block_size,
                pin_memory=pin_memory,
            )
            progress.advance(task)
            progress.update(task, description=f"[green]read-{read_mode} done")

            results.append(
                {
                    "store_path": str(store_path),
                    "num_systems": num_systems,
                    "read_mode": read_mode,
                    "read_order": read_order,
                    "read_order_block_size": (
                        read_order_block_size if read_order == "block-shuffle" else None
                    ),
                    "batch_size": batch_size if read_mode == "batch" else 1,
                    "prefetch_factor": prefetch_factor if read_mode == "batch" else 0,
                    "effective_read_window": (
                        batch_size * max(prefetch_factor, 1)
                        if read_mode == "batch"
                        else 1
                    ),
                    "read_time": read_time,
                    "read_bytes": read_bytes,
                    "read_throughput": (
                        num_systems / read_time if read_time > 0 else 0
                    ),
                }
            )

    return results


def _print_read_results(results: list[dict]) -> None:
    """Print read-only benchmark results as a Rich table.

    Parameters
    ----------
    results : list[dict]
        Read benchmark results from ``_run_read_benchmark``.
    """
    if not results:
        return

    store_path = results[0].get("store_path", "?")
    table = Table(
        title=f"Zarr Read Benchmark — {store_path}",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("Samples", justify="right", style="cyan", no_wrap=True)
    table.add_column("Read path", justify="left", no_wrap=True)
    table.add_column("Read order", justify="left", no_wrap=True)
    table.add_column("Batch", justify="right", no_wrap=True)
    table.add_column("Prefetch", justify="right", no_wrap=True)
    table.add_column("Read window", justify="right", no_wrap=True)
    table.add_column("Read time", justify="right", no_wrap=True)
    table.add_column("Samples/s", justify="right", style="bold", no_wrap=True)
    table.add_column("Data read", justify="right", style="green", no_wrap=True)

    for r in results:
        order_desc = r["read_order"]
        if r["read_order_block_size"] is not None:
            order_desc += f" (blk={r['read_order_block_size']:,})"
        table.add_row(
            f"{r['num_systems']:,}",
            r["read_mode"],
            order_desc,
            f"{r['batch_size']:,}",
            f"{r['prefetch_factor']:,}",
            f"{r['effective_read_window']:,}",
            f"{r['read_time']:.2f}s",
            f"{r['read_throughput']:,.0f}",
            _fmt_bytes(r["read_bytes"]),
        )

    console.print()
    console.print(table)


class _DefaultRoundtripGroup(click.Group):
    """Click group that falls back to ``roundtrip`` for unrecognised args.

    When users invoke ``nvalchemi-io-test --num-systems 1000`` (the pre-
    group signature), Click would normally fail because ``--num-systems``
    is not a group-level option.  This subclass detects that the first
    argument is not a known subcommand and transparently inserts
    ``roundtrip`` so the old invocation style keeps working.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Insert ``roundtrip`` when the first arg is not a subcommand."""
        if args and args[0] not in self.commands and not args[0].startswith("--help"):
            args = ["roundtrip", *args]
        return super().parse_args(ctx, args)


@click.group(
    "nvalchemi-io-test", cls=_DefaultRoundtripGroup, invoke_without_command=True
)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Zarr I/O benchmarks for nvalchemi atomic data.

    Run without a subcommand to see available benchmarks, or use
    ``roundtrip`` / ``read`` directly.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command("roundtrip")
@click.option(
    "--num-systems",
    "-n",
    type=int,
    multiple=True,
    default=[1_000, 10_000, 100_000],
    show_default=True,
    help="Number of systems to benchmark (repeat for multiple).",
)
@click.option(
    "--min-atoms",
    type=int,
    default=10,
    show_default=True,
    help="Minimum atoms per structure.",
)
@click.option(
    "--max-atoms",
    type=int,
    default=100,
    show_default=True,
    help="Maximum atoms per structure.",
)
@click.option(
    "--codec",
    type=click.Choice(["zstd", "lz4", "blosc-zstd"], case_sensitive=False),
    default=None,
    help="Compression codec (omit for no compression).",
)
@click.option(
    "--level",
    type=int,
    default=3,
    show_default=True,
    help="Compression level.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Chunk size along dim 0 (omit for Zarr default).",
)
@click.option(
    "--shard-size",
    type=int,
    default=None,
    help="Shard size along variable axis (omit for no sharding).",
)
@click.option(
    "--edge-chunk-size",
    type=int,
    default=None,
    help="Chunk size for edge arrays: neighbor_list, shifts (omit to use --chunk-size).",
)
@click.option(
    "--edge-shard-size",
    type=int,
    default=None,
    help="Shard size for edge arrays: neighbor_list, shifts (omit to use --shard-size).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for Zarr stores (default: tempdir, cleaned up).",
)
@click.option(
    "--read-mode",
    type=click.Choice(["batch", "single", "both"], case_sensitive=False),
    multiple=True,
    default=("batch",),
    show_default=True,
    help=(
        "Readback path to benchmark. 'batch' uses DataLoader fused prefetch; "
        "'single' uses reader.read per sample; repeat to control order."
    ),
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Number of samples per emitted DataLoader batch for --read-mode=batch.",
)
@click.option(
    "--prefetch-factor",
    type=click.IntRange(min=0),
    default=DEFAULT_PREFETCH_FACTOR,
    show_default=True,
    help="Number of DataLoader batches to fuse into each backend read.",
)
@click.option(
    "--read-order",
    type=click.Choice(["sequential", "shuffle", "block-shuffle"], case_sensitive=False),
    default="sequential",
    show_default=True,
    help=(
        "Logical sample order used for readback. 'shuffle' models full random "
        "dataloader reads; 'block-shuffle' shuffles contiguous index blocks."
    ),
)
@click.option(
    "--read-seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for --read-order=shuffle and --read-order=block-shuffle.",
)
@click.option(
    "--read-order-block-size",
    type=click.IntRange(min=1),
    default=DEFAULT_READ_ORDER_BLOCK_SIZE,
    show_default=True,
    help="Contiguous block size for --read-order=block-shuffle.",
)
@click.option(
    "--pin-memory/--no-pin-memory",
    default=False,
    show_default=True,
    help="Request pinned CPU tensors from readers that support it.",
)
def roundtrip(
    num_systems: tuple[int, ...],
    min_atoms: int,
    max_atoms: int,
    codec: str | None,
    level: int,
    chunk_size: int | None,
    shard_size: int | None,
    edge_chunk_size: int | None,
    edge_shard_size: int | None,
    seed: int,
    output_dir: Path | None,
    read_mode: tuple[str, ...],
    batch_size: int,
    prefetch_factor: int,
    read_order: str,
    read_seed: int,
    read_order_block_size: int,
    pin_memory: bool,
) -> None:
    """Write+read roundtrip benchmark.

    Generates random AtomicData structures with uniform atom counts
    between --min-atoms and --max-atoms, writes them to a Zarr store
    with the specified configuration, reads them back, and reports timing
    and size.
    """
    # Build config description for table title
    parts = []
    if codec is not None:
        parts.append(f"{codec} L{level}")
    if chunk_size is not None:
        parts.append(f"chunk={chunk_size:,}")
    if shard_size is not None:
        parts.append(f"shard={shard_size:,}")
    if edge_chunk_size is not None:
        parts.append(f"edge_chunk={edge_chunk_size:,}")
    if edge_shard_size is not None:
        parts.append(f"edge_shard={edge_shard_size:,}")
    read_modes = _expand_read_modes(read_mode)
    read_desc = ", ".join(read_modes)
    read_order = cast(ReadOrder, read_order.lower())
    config_desc = ", ".join(parts) if parts else "no compression"

    console.print(
        f"[bold]nvalchemi Zarr I/O roundtrip benchmark[/bold]  "
        f"atoms={min_atoms}-{max_atoms}  config={config_desc}  "
        f"read={read_desc}  read_order={read_order}  "
        f"batch={batch_size:,}  prefetch={prefetch_factor:,}  "
        f"read_window={batch_size * max(prefetch_factor, 1):,}"
    )

    config = _build_config(
        codec, level, chunk_size, shard_size, edge_chunk_size, edge_shard_size
    )

    use_temp = output_dir is None
    store_dir = (
        Path(tempfile.mkdtemp(prefix="nvalchemi_bench_")) if use_temp else output_dir
    )
    store_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

    try:
        results = _run_benchmark(
            num_systems_list=sorted(num_systems),
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            seed=seed,
            config=config,
            store_dir=store_dir,
            read_modes=read_modes,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            read_order=read_order,
            read_seed=read_seed,
            read_order_block_size=read_order_block_size,
            pin_memory=pin_memory,
        )
        _print_results(results, config_desc)
    finally:
        if use_temp:
            shutil.rmtree(store_dir, ignore_errors=True)


@main.command("read")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--read-mode",
    type=click.Choice(["batch", "single", "both"], case_sensitive=False),
    multiple=True,
    default=("batch",),
    show_default=True,
    help=(
        "Readback path to benchmark. 'batch' uses DataLoader fused prefetch; "
        "'single' uses reader.read per sample; repeat to control order."
    ),
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Number of samples per emitted DataLoader batch for --read-mode=batch.",
)
@click.option(
    "--prefetch-factor",
    type=click.IntRange(min=0),
    default=DEFAULT_PREFETCH_FACTOR,
    show_default=True,
    help="Number of DataLoader batches to fuse into each backend read.",
)
@click.option(
    "--read-order",
    type=click.Choice(["sequential", "shuffle", "block-shuffle"], case_sensitive=False),
    default="sequential",
    show_default=True,
    help=(
        "Logical sample order used for readback. 'shuffle' models full random "
        "dataloader reads; 'block-shuffle' shuffles contiguous index blocks."
    ),
)
@click.option(
    "--read-seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed for --read-order=shuffle and --read-order=block-shuffle.",
)
@click.option(
    "--read-order-block-size",
    type=click.IntRange(min=1),
    default=DEFAULT_READ_ORDER_BLOCK_SIZE,
    show_default=True,
    help="Contiguous block size for --read-order=block-shuffle.",
)
@click.option(
    "--pin-memory/--no-pin-memory",
    default=False,
    show_default=True,
    help="Request pinned CPU tensors from readers that support it.",
)
def read_cmd(
    path: Path,
    read_mode: tuple[str, ...],
    batch_size: int,
    prefetch_factor: int,
    read_order: str,
    read_seed: int,
    read_order_block_size: int,
    pin_memory: bool,
) -> None:
    """Benchmark read throughput against an existing Zarr store.

    Reads all samples from PATH using the specified access pattern and
    reports timing and throughput. Useful for profiling read performance
    in isolation, or comparing sequential vs. shuffled access.
    """
    read_modes = _expand_read_modes(read_mode)
    read_order_typed = cast(ReadOrder, read_order.lower())

    console.print(
        f"[bold]nvalchemi Zarr read benchmark[/bold]  "
        f"store={path}  read={', '.join(read_modes)}  "
        f"order={read_order_typed}  batch={batch_size:,}  "
        f"prefetch={prefetch_factor:,}  "
        f"read_window={batch_size * max(prefetch_factor, 1):,}"
    )

    results = _run_read_benchmark(
        store_path=path,
        read_modes=read_modes,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        read_order=read_order_typed,
        read_seed=read_seed,
        read_order_block_size=read_order_block_size,
        pin_memory=pin_memory,
    )
    _print_read_results(results)


if __name__ == "__main__":
    main()
