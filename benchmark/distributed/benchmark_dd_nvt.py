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
"""Distributed NVT end-to-end benchmark — world ∈ {0, 1, 2}.

The model is selected with ``--config <model>.yaml`` (see ``configs/``).
Times a full :class:`~nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin`
run under :class:`~nvalchemi.distributed.domain_parallel.DomainParallel`
for a sweep of system sizes — integrator half-kicks, neighbour-list
rebuild, halo exchange, model forward + autograd, force consolidation,
and atom migration are all inside the timed region.

Three modes — each run as a SEPARATE job (one launch per mode keeps
allocator pools clean between modes):

* world=0: ``python benchmark_dd_nvt.py --config ...``
  Raw ``NVTLangevin.run`` — no DD wrapper.
* world=1: ``torchrun --nproc_per_node=1 benchmark_dd_nvt.py --config ...``
  ``DomainParallel.run`` at single rank — exposes DD-wrapper overhead.
* world=2: ``torchrun --nproc_per_node=2 benchmark_dd_nvt.py --config ...``
  Full DD with halo + NCCL + force consolidation.

NVT runs in float32 (required for cueq / AIMNet2 warp kernels, fine for
the rest). Sweep a config knob via ``--set`` (dotted keys), e.g.
``--set loader.enable_cueq=true`` for MACE.
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys
from pathlib import Path
from typing import Any, Callable

# Per-rank torch.compile cache dirs — set BEFORE importing torch so inductor
# resolves them from the start. Under multi-rank DD a shared cache dir lets one
# rank load another's guarded graph → KeyError → NCCL deadlock; isolating per rank
# keeps the caches on (fast) and collision-free. Keyed on torchrun's LOCAL_RANK.
_rank_id = _os.environ.get("LOCAL_RANK") or _os.environ.get("RANK") or "0"
import tempfile as _tf  # noqa: E402

_cache_root = _os.path.join(_tf.gettempdir(), "nvalchemi_dd_compile_cache")
# Force per-rank (overrides any shared value inherited from the container/env) so
# the inductor FxGraphCache + AOTAutograd cache (a subdir of TORCHINDUCTOR_CACHE_DIR)
# never collide across ranks.
_os.environ["TORCHINDUCTOR_CACHE_DIR"] = _os.path.join(
    _cache_root, f"inductor_rank{_rank_id}"
)
_os.environ["TRITON_CACHE_DIR"] = _os.path.join(_cache_root, f"triton_rank{_rank_id}")
_sys_stderr = __import__("sys").stderr
print(
    f"[dd-cache] rank {_rank_id} TORCHINDUCTOR_CACHE_DIR="
    f"{_os.environ['TORCHINDUCTOR_CACHE_DIR']}",
    file=_sys_stderr,
    flush=True,
)

import torch

_sys.path.insert(0, str(Path(__file__).parent))
from _benchmark_common import (  # noqa: E402
    BenchConfig,
    add_common_args,
    build_loader,
    build_system,
    launched_by_torchrun,
    load_config,
    resolve_attr,
    run_nvt_main,
)


def make_nvt_harness(cfg: BenchConfig, *, dd_compile: bool = False) -> Callable[..., Any]:
    """Build the ``build_nvt_harness`` callback for an NVT benchmark config.

    ``dd_compile`` requests the framework-owned compiled DD forward (fixed-shape
    caps) on the DomainParallel path.
    """
    ncfg = cfg.nvt or {}
    cutoff_attr = ncfg.get("cutoff_attr", "cutoff")
    max_neighbors = ncfg.get("max_neighbors")
    skin = ncfg.get("skin", 0.5)
    dt = ncfg.get("dt", 0.5)
    temperature = ncfg.get("temperature", 300.0)
    friction = ncfg.get("friction", 0.01)

    def build_nvt_harness(
        wrapper: Any,
        n_atoms: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        distributed: bool,
        rank: int = 0,
        world_size: int = 1,
        mesh: Any = None,
    ) -> tuple[Callable[[int], None], Any, Any | None, int]:
        from nvalchemi.data import AtomicData, Batch
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

        system = build_system(cfg, n_atoms, dtype)
        n_actual = system.positions.shape[0]
        cell_b = system.cell.unsqueeze(0)
        pbc_b = system.pbc.reshape(1, 3)

        # A max_neighbors of null means the model builds its own NL (UMA).
        hooks: list[Any] = []
        if max_neighbors is not None:
            from nvalchemi.dynamics.base import DynamicsStage
            from nvalchemi.hooks.neighbor_list import NeighborListHook

            hooks = [
                NeighborListHook(
                    wrapper.model_config.neighbor_config,
                    skin=skin,
                    max_neighbors=max_neighbors,
                    stage=DynamicsStage.BEFORE_COMPUTE,
                )
            ]
        nvt = NVTLangevin(
            model=wrapper,
            dt=dt,
            temperature=temperature,
            friction=friction,
            hooks=hooks,
        )

        def _data() -> AtomicData:
            data = AtomicData(
                atomic_numbers=system.atomic_numbers.to(device),
                positions=system.positions.to(device).clone(),
                atomic_masses=system.masses.to(device),
                cell=cell_b.to(device),
                pbc=pbc_b,
                forces=torch.zeros(n_actual, 3, dtype=dtype, device=device),
                energy=torch.zeros(1, 1, dtype=dtype, device=device),
            )
            data.add_node_property("velocities", system.velocities.to(device))
            return data

        if not distributed:
            batch = Batch.from_data_list([_data()], device=device)
            state = {"batch": batch}

            def runner_world0(n_steps: int) -> None:
                state["batch"] = nvt.run(state["batch"], n_steps=n_steps)

            return runner_world0, nvt, None, n_actual

        from _benchmark_common import resolve_strategy

        from nvalchemi.distributed.config import DomainConfig
        from nvalchemi.distributed.domain_parallel import DomainParallel

        cutoff = float(resolve_attr(wrapper, cutoff_attr))
        # Config-driven strategy: DomainParallel reads DomainConfig.strategy to
        # pick both the scatter layout and the model's per-strategy spec.
        strategy_kind, _ = resolve_strategy(cfg.system.strategy)
        dd_config = DomainConfig(
            cutoff=cutoff, skin=skin, mesh=mesh, mesh_dim="domain",
            strategy=strategy_kind, compile=dd_compile,
        )
        dd = DomainParallel(nvt, config=dd_config)

        full_batch = (
            Batch.from_data_list([_data()], device=device) if rank == 0 else None
        )
        local = dd.partition(full_batch)
        state = {"batch": local}

        def runner_dist(n_steps: int) -> None:
            state["batch"] = dd.run(state["batch"], n_steps=n_steps)

        return runner_dist, nvt, dd, n_actual

    return build_nvt_harness


def main() -> None:
    """Entry point: run the distributed NVT benchmark for one config."""
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
        help="Override a config value (dotted key), e.g. --set loader.enable_cueq=true.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if cfg.nvt is None:
        parser.error(f"config for {cfg.model!r} has no 'nvt' section")
    if not args.sizes:
        args.sizes = cfg.default_sizes

    # ``loader.compile`` = compile the DD forward. On the multi-rank (DD) path the
    # FRAMEWORK owns the compiled forward via DomainConfig.compile: it pads the
    # per-rank atom/edge counts to fixed shapes so the compiled graph is reused
    # across MD steps (a loader-compiled model instead recompiles every step as the
    # owned+ghost count drifts). So on the DD path we do NOT also loader-compile
    # (that would nest compiles); world=0 keeps loader-compile for the reference.
    want_compile = bool(cfg.loader.get("compile", False))
    is_dd = launched_by_torchrun()
    dd_compile = want_compile and is_dd
    compile_model = want_compile and not is_dd
    load_wrapper = build_loader(cfg, compile_model=compile_model)
    build_nvt_harness = make_nvt_harness(cfg, dd_compile=dd_compile)
    run_nvt_main(
        cfg.model, load_wrapper, build_nvt_harness, args, dtype=torch.float32
    )


if __name__ == "__main__":
    main()
