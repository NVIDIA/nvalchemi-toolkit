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

"""Per-rank entry point for the validator's ``mp.spawn`` workers.

Each worker:
1. Initialises the process group (NCCL or Gloo backend).
2. Patches physicsnemo's ``all_to_all_v`` for Gloo when applicable.
3. Constructs the wrapper, evaluates ``distribution_spec`` (so lazy
   imports + custom-op registration fire), reconstructs the spec from
   the dict shipped from the parent.
4. Runs forward inside ``DistributedModel`` and captures dispatch
   trace + helper trace + halo-completeness summary.
5. Ships the results back to the parent over a queue, surviving
   crashes by sending whatever helper records were collected before
   the failure.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.distributed as dist

from nvalchemi.distributed.spec import MLIPSpec
from nvalchemi.distributed.validate.halo_diagnostics import (
    _capture_halo_summary,
)
from nvalchemi.distributed.validate.payloads import (
    _extract_cutoff,
    _payload_to_batch,
)
from nvalchemi.distributed.validate.reference import _ensure_neighbors

__all__ = ["_worker_main", "_patch_physicsnemo_all_to_all_for_gloo"]


def _worker_main(
    rank: int,
    world_size: int,
    backend: str,
    device_str: str,
    model_factory: Callable[[], Any],
    sample_payload: dict[str, Any],
    spec_dict: dict[str, Any],
    queue: Any,
    watched_helper_packages: tuple[str, ...] = (),
    helper_sample_every: int = 8,
    layer_diagnostic: bool = False,
) -> None:
    """Per-rank entry point for ``mp.spawn``. Initialises the process
    group, builds the wrapper + DistributedModel from the spec dict,
    runs forward, ships outputs back.

    Always pins to the same CUDA device — both NCCL (when world_size
    fits in device_count) and Gloo (when ranks share a device) drive
    real GPU kernels, so the kernel/precision path matches what
    production sees.

    Any exception inside the run is captured and shipped back to the
    parent via the queue rather than crashing silently — the validator
    surfaces the traceback in ``Attempt.error``.
    """
    import traceback  # noqa: PLC0415

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29504")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Warp's per-rank kernel cache: each worker compiles+writes
    # its own compiled artifacts. Default ``~/.cache/warp/`` may be
    # read-only in sandboxed environments; route to a writable temp
    # location, partitioned by rank to avoid concurrent-write races
    # when multiple workers JIT the same kernel.
    if "WARP_CACHE_PATH" not in os.environ:
        import tempfile  # noqa: PLC0415

        os.environ["WARP_CACHE_PATH"] = os.path.join(
            tempfile.gettempdir(), f"nvalchemi-validate-warp-cache-rank{rank}"
        )

    # Mirror the launcher's warning-suppression set-up. Spawned workers
    # don't inherit Python's warning filters or warp's ``warnings_seen``
    # dedupe set — both have to be re-applied on the worker side.
    import warnings as _warnings  # noqa: PLC0415

    _warnings.filterwarnings(
        "ignore",
        message=".*warp\\.context.*",
        category=DeprecationWarning,
    )
    _warnings.filterwarnings(
        "ignore",
        message=".*\\.grad attribute of a Tensor that is not a leaf.*",
        category=UserWarning,
    )
    try:
        import warp._src.utils as _wu  # noqa: PLC0415

        _warnings_seen = getattr(_wu, "warnings_seen", None)
        if _warnings_seen is not None:
            for _msg in (
                "The namespace `warp.context` will soon be removed from the "
                "public API. It can still be accessed from `warp._src.context` "
                "but might be changed or removed without notice.",
                "The symbol `warp.context.Device` will soon be removed from "
                "the public API. Use `warp.Device` instead.",
            ):
                _warnings_seen.add((DeprecationWarning, _msg))
    except ImportError:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_str)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except Exception:
        queue.put((rank, "INIT_ERROR", traceback.format_exc()))
        return

    # Patch physicsnemo's indexed_all_to_all_v with isend/irecv when
    # cuda-tensor collectives are gloo-backed — gloo lacks
    # ``all_to_all_v`` but the halo path routes through this physicsnemo
    # helper. Same shim ``test_topology.py`` and the dispatch-trace gloo
    # harness use. Triggered for both the single-GPU ``cpu:gloo,cuda:gloo``
    # fallback and the bare ``"gloo"`` request (kept for forward compat
    # if a caller passes it explicitly).
    if "gloo" in backend:
        _patch_physicsnemo_all_to_all_for_gloo()

    # Helper records are accumulated into the list yielded by
    # ``helper_trace``. Bind it at the outer-try scope so the
    # ``except`` branch can include records collected before the crash
    # — this is what makes the diagnostic useful when the worker dies
    # before forward completes (which is the typical failure mode for
    # an unwrapped third-party helper: the validator's distributed
    # primitives explode on out-of-range indices).
    helper_calls_list: list = []
    # Likewise for dispatch records — bind at outer scope so any
    # handler firings *before* the crash get shipped back, giving the
    # launcher's diagnostic a fingerprint of which dispatch paths the
    # wrapper actually exercised before failing.
    dispatch_records_list: list = []

    try:
        from torch.distributed.device_mesh import DeviceMesh  # noqa: PLC0415

        from nvalchemi.distributed._core.dispatch_trace import (
            dispatch_trace,  # noqa: PLC0415
        )
        from nvalchemi.distributed._core.helper_trace import (
            helper_trace,  # noqa: PLC0415
        )
        from nvalchemi.distributed.config import DomainConfig  # noqa: PLC0415
        from nvalchemi.distributed.distributed_model import (  # noqa: PLC0415
            DistributedModel,
        )
        from nvalchemi.distributed.sharded_batch import ShardedBatch  # noqa: PLC0415

        # Wrap the entire downstream block — wrapper construction +
        # forward — in ``helper_trace`` so any helpers bound at
        # construction time are also intercepted. AIMNet2 binds at
        # call time; this is forward-compat for models that don't.
        with helper_trace(
            watched_helper_packages, sample_every=helper_sample_every
        ) as h_records:
            # Mirror records into the outer-scope list as they
            # accumulate, so a crash mid-forward still ships partial
            # data. Both names point to the same list — ``h_records``
            # is what the proxies append into; ``helper_calls_list``
            # is what the except / success branches read from.
            helper_calls_list = h_records
            # Force ``distribution_spec`` to evaluate before
            # ``MLIPSpec.from_dict`` runs — Ewald/PME-style
            # wrappers register their ``@torch.library.custom_op`` ops
            # inside the property, and ``from_dict`` needs those op
            # qualnames already in ``torch.ops``.
            wrapper = model_factory().to(device_str)
            _ds = getattr(wrapper, "distribution_spec", None)
            if callable(_ds):
                _ds()

            spec = MLIPSpec.from_dict(spec_dict)
            sample_batch = _payload_to_batch(sample_payload, device=device_str)

            # Reconstruct neighbours on the worker side using the wrapper's
            # neighbor_config — ship list deliberately omits NL artefacts
            # (smaller wire payload, no AtomicData-vs-Batch attachment
            # dance for derived state).
            _ensure_neighbors(sample_batch, wrapper)

            # CUDA-typed mesh required: ``ShardTensor.from_local`` calls
            # ``local_tensor.to(mesh.device_type)``, so a cpu-typed mesh
            # silently moves owned-shard tensors to cpu. Direct
            # ``DeviceMesh(...)`` (not ``init_device_mesh``) doesn't
            # spawn an NCCL sub-group, so cuda mesh + gloo backend
            # coexist when all ranks share device 0.
            mesh = DeviceMesh(
                "cuda", list(range(world_size)), mesh_dim_names=("domain",)
            )
            cfg = DomainConfig(
                cutoff=_extract_cutoff(wrapper),
                mesh=mesh,
            )

            # Halo is the only storage policy: spatial partition (owned/ghost
            # split by geometry).
            sharded = ShardedBatch.from_batch(
                sample_batch, mesh=mesh, config=cfg, partition_mode="spatial"
            )

            from nvalchemi.distributed.validate.layer_diagnostics import (  # noqa: PLC0415
                attach_layer_hooks,
                finalize_layer_records,
            )

            layer_records: list = []
            layer_handles: list = []
            if layer_diagnostic and isinstance(wrapper, torch.nn.Module):
                # See ``_reference_run`` for why we hook the wrapper rather
                # than ``wrapper.model``: nested model attribute paths
                # (UMA's ``predict_unit.model.module`` etc.) are reached
                # automatically through Module.named_modules().
                layer_handles = attach_layer_hooks(wrapper, layer_records)
            try:
                with dispatch_trace() as records:
                    # Mirror records into the outer-scope list so the
                    # ``except`` branch sees handler firings up to the
                    # crash. ``records`` and ``dispatch_records_list`` then
                    # alias the same backing list for the rest of the run.
                    dispatch_records_list = records
                    with DistributedModel(wrapper, cfg, spec=spec) as dist_model:
                        outputs = dist_model(sharded)
            finally:
                for h in layer_handles:
                    h.remove()

            # Bulk-materialize on-device per-module sums to floats. One
            # sync replaces N per-hook syncs — critical for big models
            # (UMA: 218 submodules x many MP iterations).
            if layer_records:
                finalize_layer_records(layer_records)

            # Dump per-rank collective counter (no-op unless
            # ``NVALCHEMI_COUNT_COLLECTIVES=1``). Lives here so rank 0
            # and rank 1 each emit their own line; the launcher's
            # post-process collates.
            from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
                dump_collective_counts,
            )

            dump_collective_counts(label="post-forward")

            # Halo-completeness summary: for halo-storage models, the
            # per-rank padded NL has been built and attached to
            # ``sharded.padded_batch`` by ``_call_halo_storage``. Capture
            # per-owned-atom valid-neighbor counts plus the owned
            # positions so the launcher can map back to global IDs and
            # verify each rank's halo covers its owned atoms the same
            # way single-process did.
            halo_summary = _capture_halo_summary(spec, sharded)

        # Convert tensors to numpy + pickle to bytes, ship raw bytes
        # through the queue. ``torch.multiprocessing.reductions`` would
        # otherwise route even ``.cpu().clone()`` tensors through a
        # shared-FD reduction that breaks when the worker exits and
        # tears down its CUDA context — surfacing as the parent's
        # ``FileNotFoundError`` on a vanished resource-sharer socket.
        # Bytes have no shared-memory baggage; deserialization happens
        # in the launcher's clean context.
        import pickle  # noqa: PLC0415

        def _to_numpy(v: torch.Tensor) -> Any:
            # ``.numpy()`` rejects tensor subclasses; a ShardTensor can reach
            # here (e.g. autograd-derived forces, or halo-summary positions
            # read off the promoted ShardTensor batch) — drop to its plain
            # local view first.
            if type(v).__name__ == "ShardTensor":
                v = v.to_local()
            return v.detach().to("cpu").numpy().copy()

        outputs_for_pickle = {
            k: _to_numpy(v) for k, v in outputs.items() if isinstance(v, torch.Tensor)
        }
        handler_counts: dict[str, int] = {}
        for r in records:
            handler_counts[r["handler"]] = handler_counts.get(r["handler"], 0) + 1
        # Convert any tensors in halo_summary to numpy — the dict travels
        # through pickle.dumps below, and tensors with lingering pybind
        # references (e.g. from external C-extension models) fail.
        halo_summary_for_pickle: dict[str, Any] = {}
        for k, v in halo_summary.items():
            if isinstance(v, torch.Tensor):
                halo_summary_for_pickle[k] = _to_numpy(v)
            else:
                halo_summary_for_pickle[k] = v
        queue.put(
            (
                rank,
                pickle.dumps(
                    (
                        outputs_for_pickle,
                        handler_counts,
                        helper_calls_list,
                        halo_summary_for_pickle,
                        layer_records,
                    )
                ),
            )
        )
    except Exception:
        # Ship partial helper records + dispatch counts so the launcher
        # can still diagnose. ``list(...)`` detaches from the trace's
        # internal storage in case the context manager hasn't unwound.
        import pickle  # noqa: PLC0415

        # Aggregate dispatch records → handler counts so the launcher
        # can tell a missed-rebind crash (handlers fired) from a
        # halo / spec mismatch (no firings).
        partial_handler_counts: dict[str, int] = {}
        for r in dispatch_records_list:
            try:
                name = r["handler"]
            except (KeyError, TypeError):
                continue
            partial_handler_counts[name] = partial_handler_counts.get(name, 0) + 1

        queue.put(
            (
                rank,
                "RUN_ERROR",
                traceback.format_exc(),
                pickle.dumps(list(helper_calls_list)),
                pickle.dumps(partial_handler_counts),
            )
        )
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: S110, BLE001
            # Worker is already exiting; cleanup failure should not
            # mask whatever error sent us here.
            pass


def _patch_physicsnemo_all_to_all_for_gloo() -> None:
    """Replace physicsnemo's all_to_all_v with isend/irecv for Gloo.

    Gloo lacks ``all_to_all_v``; the halo path routes through
    ``physicsnemo.distributed.utils.indexed_all_to_all_v_wrapper``.
    Same shim ``test_topology.py`` uses. physicsnemo is a hard dep
    of the broader distributed stack — if it's missing here, the
    overall env is misconfigured and we want to fail loudly.
    """
    import physicsnemo.distributed.utils as pn_utils  # noqa: PLC0415

    def _indexed_all_to_all_v_gloo(tensor, indices, sizes, dim=0, group=None):
        # Gloo's TCP transport rejects raw cuda-tensor isend/irecv
        # (``Bad address`` from writev) — collectives that go through
        # gloo's higher-level ops (all_gather/all_reduce) auto-stage via
        # cpu, but isend/irecv expose the raw transport. Stage cuda→cpu
        # before send and cpu→cuda after recv so the wire path is cpu
        # and the caller sees a cuda result on cuda input.
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        out_device = tensor.device
        on_cuda = out_device.type == "cuda"
        x_send_cpu = [
            (tensor[idx].contiguous().cpu() if on_cuda else tensor[idx].contiguous())
            for idx in indices
        ]
        x_recv_cpu = []
        tensor_shape = list(tensor.shape)
        cpu_dev = torch.device("cpu")
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r][rank]
            x_recv_cpu.append(
                torch.empty(tensor_shape, dtype=tensor.dtype, device=cpu_dev)
            )
        ops = []
        for r in range(comm_size):
            if r == rank:
                x_recv_cpu[r].copy_(x_send_cpu[r])
            else:
                ops.append(dist.isend(x_send_cpu[r], dst=r, group=group))
                ops.append(dist.irecv(x_recv_cpu[r], src=r, group=group))
        for op in ops:
            op.wait()
        joined = torch.cat(x_recv_cpu, dim=dim)
        return joined.to(out_device) if on_cuda else joined

    pn_utils.indexed_all_to_all_v_wrapper = _indexed_all_to_all_v_gloo
