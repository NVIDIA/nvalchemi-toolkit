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
"""Quick Zarr I/O benchmark for measuring write throughput and compression.

Run with::

    nvalchemi-io-test --help
    nvalchemi-io-test --num-systems 1000 5000 --codec zstd --chunk-size 10000

Edge-specific chunking (useful for large graphs)::

    nvalchemi-io-test -n 100 --codec zstd --chunk-size 10000 --edge-chunk-size 5000
    nvalchemi-io-test -n 100 --codec zstd --shard-size 500 --edge-shard-size 500

"""

from __future__ import annotations

import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from nvalchemi.data.atomic_data import AtomicData

console = Console(stderr=True)


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
        energies=torch.randn(1, 1),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        edge_index=torch.stack(
            [
                torch.randint(0, max(num_atoms, 1), (num_edges,)),
                torch.randint(0, max(num_atoms, 1), (num_edges,)),
            ]
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
        Chunk size for edge-level arrays (edge_index, shifts).
    edge_shard_size : int | None
        Shard size for edge-level arrays (edge_index, shifts).

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
            "edge_index": edge_cfg,
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


def _run_benchmark(
    num_systems_list: list[int],
    min_atoms: int,
    max_atoms: int,
    seed: int,
    config: dict | None,
    store_dir: Path,
) -> list[dict]:
    """Run the write benchmark for each system count.

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
            task = progress.add_task(f"[cyan]{num_systems:>10,} systems", total=3)

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

            # Step 3: measure
            progress.update(task, description=f"[cyan]{num_systems:>10,} measure")
            disk_bytes = _dir_size(store_path)
            num_files = _file_count(store_path)

            # compute uncompressed size from numpy arrays
            raw_bytes = 0
            for d in data_list:
                for key in d.model_fields_set:
                    val = getattr(d, key, None)
                    if isinstance(val, torch.Tensor):
                        raw_bytes += val.nelement() * val.element_size()
            progress.advance(task)

            progress.update(
                task,
                description=f"[green]{num_systems:>10,} done",
            )

            avg_atoms_run = total_atoms / num_systems
            avg_edges_run = total_edges / num_systems
            ratio = raw_bytes / disk_bytes if disk_bytes > 0 else float("inf")

            results.append(
                {
                    "num_systems": num_systems,
                    "total_atoms": total_atoms,
                    "total_edges": total_edges,
                    "avg_atoms": avg_atoms_run,
                    "avg_edges": avg_edges_run,
                    "raw_bytes": raw_bytes,
                    "disk_bytes": disk_bytes,
                    "num_files": num_files,
                    "ratio": ratio,
                    "write_time": write_time,
                    "throughput": num_systems / write_time if write_time > 0 else 0,
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
        title=f"Zarr I/O Benchmark — {config_desc}",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("Systems", justify="right", style="cyan")
    table.add_column("Avg atoms", justify="right")
    table.add_column("Avg edges", justify="right")
    table.add_column("Raw size", justify="right")
    table.add_column("Disk size", justify="right", style="green")
    table.add_column("Ratio", justify="right", style="yellow")
    table.add_column("Files", justify="right")
    table.add_column("Write time", justify="right")
    table.add_column("Systems/s", justify="right", style="bold")

    for r in results:
        table.add_row(
            f"{r['num_systems']:,}",
            f"{r['avg_atoms']:.0f}",
            f"{r['avg_edges']:.0f}",
            _fmt_bytes(r["raw_bytes"]),
            _fmt_bytes(r["disk_bytes"]),
            f"{r['ratio']:.2f}x",
            f"{r['num_files']:,}",
            f"{r['write_time']:.2f}s",
            f"{r['throughput']:,.0f}",
        )

    console.print()
    console.print(table)


@click.command("nvalchemi-io-test")
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
    help="Chunk size for edge arrays: edge_index, shifts (omit to use --chunk-size).",
)
@click.option(
    "--edge-shard-size",
    type=int,
    default=None,
    help="Shard size for edge arrays: edge_index, shifts (omit to use --shard-size).",
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
def main(
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
) -> None:
    """Run quick Zarr write benchmarks for nvalchemi data.

    Generates random AtomicData structures with uniform atom counts
    between --min-atoms and --max-atoms, writes them to a Zarr store
    with the specified configuration, and reports timing and size.
    """
    # Build config description for table title
    parts = []
    if codec:
        parts.append(f"{codec} L{level}")
    if chunk_size:
        parts.append(f"chunk={chunk_size:,}")
    if shard_size:
        parts.append(f"shard={shard_size:,}")
    if edge_chunk_size:
        parts.append(f"edge_chunk={edge_chunk_size:,}")
    if edge_shard_size:
        parts.append(f"edge_shard={edge_shard_size:,}")
    config_desc = ", ".join(parts) if parts else "no compression"

    console.print(
        f"[bold]nvalchemi Zarr I/O benchmark[/bold]  "
        f"atoms={min_atoms}-{max_atoms}  config={config_desc}"
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
        )
        _print_results(results, config_desc)
    finally:
        if use_temp:
            shutil.rmtree(store_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
