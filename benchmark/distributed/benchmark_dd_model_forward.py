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
"""Distributed forward benchmark — single-GPU vs multi-GPU + force equivalence.

The model is selected with ``--config <model>.yaml`` (see ``configs/``).
Each timed step is the model forward + autograd for forces (plus the halo
exchange on the distributed path); an extra untimed step gathers per-rank
owned forces to rank 0 and checks them against the single-rank reference.

Usage
-----
::

    python benchmark/distributed/benchmark_dd_model_forward.py \
        --config benchmark/distributed/configs/lj.yaml --sizes 1000 4000 --single-only

    torchrun --nproc_per_node=2 \
        benchmark/distributed/benchmark_dd_model_forward.py \
        --config benchmark/distributed/configs/mace.yaml --sizes 1000 4000

Sweep a config knob without a new file via ``--set`` (dotted keys), e.g.
``--set loader.enable_cueq=true --set loader.compile=true`` for MACE.
"""

from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path
from typing import Any, Callable

import torch

_sys.path.insert(0, str(Path(__file__).parent))
from _benchmark_common import (  # noqa: E402
    BenchConfig,
    add_common_args,
    build_loader,
    build_system,
    load_config,
    resolve_attr,
    resolve_dtype,
    run_main,
)


def make_forward_harness(cfg: BenchConfig, *, dd_compile: bool) -> Callable[..., Any]:
    """Build the ``build_harness`` callback for a forward benchmark config."""
    fwd = cfg.forward
    has_stress = fwd.get("has_stress", False)
    upfront_halo = fwd.get("upfront_halo_exchange", True)
    dd_compile_capable = fwd.get("dd_compile_capable", False)
    cutoff_attr = fwd.get("cutoff_attr", "cutoff")

    def build_harness(
        wrapper: Any,
        n_atoms: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        distributed: bool,
        rank: int = 0,
        world_size: int = 1,
        mesh: Any = None,
    ) -> tuple[Callable, int, torch.Tensor, torch.Tensor, torch.Tensor, Any | None]:
        from nvalchemi.data import AtomicData, Batch

        system = build_system(cfg, n_atoms, dtype)
        n_actual = system.positions.shape[0]
        cell_b = system.cell.unsqueeze(0)
        pbc = system.pbc
        pbc_b = pbc.reshape(1, 3)

        def _data(dev: torch.device) -> AtomicData:
            fields: dict[str, Any] = dict(
                atomic_numbers=system.atomic_numbers.to(dev),
                positions=system.positions.to(dev).clone(),
                atomic_masses=system.masses.to(dev),
                cell=cell_b.to(dev),
                pbc=pbc_b.to(dev),
            )
            if system.charges is not None:
                fields["charges"] = system.charges.to(dev)
            if fwd.get("charge_zero"):
                fields["charge"] = torch.zeros(1, 1, dtype=dtype, device=dev)
            if fwd.get("seed_force_energy"):
                fields["forces"] = torch.zeros(n_actual, 3, dtype=dtype, device=dev)
                fields["energy"] = torch.zeros(1, 1, dtype=dtype, device=dev)
            return AtomicData(**fields)

        def _result(out: dict) -> tuple:
            if has_stress:
                return out["energy"].sum(), out["forces"], out["stress"]
            return out["energy"].sum(), out["forces"]

        if not distributed:
            batch = Batch.from_data_list([_data(device)], device=device)
            if cfg.system.compute_neighbors:
                from nvalchemi.neighbors import compute_neighbors

                compute_neighbors(batch, config=wrapper.model_config.neighbor_config)

            def step_single() -> tuple:
                return _result(wrapper(batch))

            return step_single, n_actual, system.positions, system.cell, pbc, None

        from _benchmark_common import resolve_strategy

        from nvalchemi.distributed.config import DomainConfig
        from nvalchemi.distributed.distributed_model import DistributedModel
        from nvalchemi.distributed.particle_halo import halo_exchange
        from nvalchemi.distributed.sharded_batch import ShardedBatch

        cutoff = float(resolve_attr(wrapper, cutoff_attr))
        # Config-driven strategy: the framework reads DomainConfig.strategy to
        # select the model's per-strategy spec (halo vs graph parallel). The
        # ShardedBatch partition layout is kept consistent with it.
        strategy_kind, default_part_mode = resolve_strategy(cfg.system.strategy)
        domain_config = DomainConfig(cutoff=cutoff, mesh=mesh, strategy=strategy_kind)
        # DD-compile (whole-forward compile over an eager model) is owned by
        # DistributedModel and only offered by the compile-capable models.
        dm_kwargs = {"compile": dd_compile} if dd_compile_capable else {}
        dist_model = DistributedModel(wrapper, domain_config, **dm_kwargs)

        full_batch = (
            Batch.from_data_list([_data(device)], device=device) if rank == 0 else None
        )
        sharded = ShardedBatch.from_batch(
            full_batch,
            mesh=mesh,
            config=domain_config,
            partition_mode=cfg.system.partition_mode or default_part_mode,
        )
        dist_model(sharded)  # warm lazy global-NL metadata
        halo_cfg = dist_model._halo_config
        needs_forces = dist_model._needs_forces()

        def step_dist() -> tuple:
            # Halo-storage models that don't refresh internally need an upfront
            # halo exchange to pull halo rows from owners before the forward.
            # Graph-parallel strategies have no halo config (halo_cfg is None) —
            # the upfront exchange is halo-only, so skip it there.
            if upfront_halo and halo_cfg is not None:
                halo_exchange(sharded, halo_cfg, compute_forces=needs_forces)
            return _result(dist_model(sharded))

        return (
            step_dist,
            n_actual,
            system.positions,
            system.cell,
            pbc,
            domain_config,
        )

    return build_harness


def main() -> None:
    """Entry point: run the distributed forward benchmark for one config."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to a model config YAML.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override a config value (dotted key), e.g. --set loader.compile=true.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if not args.sizes:
        args.sizes = cfg.default_sizes
    dtype = resolve_dtype(cfg)

    # One ``compile`` intent drives both the inner-model compile and the DD
    # whole-forward compile: under torchrun the model stays eager (DD owns the
    # compile) so we only inner-compile in --single-only mode to avoid nesting.
    compile_intent = bool(cfg.loader.get("compile", False))
    load_wrapper = build_loader(cfg, compile_model=compile_intent and args.single_only)
    build_harness = make_forward_harness(cfg, dd_compile=compile_intent)
    run_main(cfg.model, load_wrapper, build_harness, args, dtype=dtype)


if __name__ == "__main__":
    main()
