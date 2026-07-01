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

"""Multi-rank inference + initial spec inference.

* :func:`_validate_spec` spawns ``world_size`` workers, gathers their
  outputs / handler counts / helper traces / halo summaries, and
  produces the per-output diff dict against the single-process
  reference.
* :func:`_infer_spec_from_trace` builds the candidate spec for the
  first attempt — uses the wrapper's ``distribution_spec`` as a strong
  prior, falling back to a conservative halo default for wrappers that
  haven't declared one.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
import torch.multiprocessing as mp

from nvalchemi.distributed.spec import MLIPSpec
from nvalchemi.distributed.validate.halo_diagnostics import (
    _check_halo_completeness,
    _partition_health,
)
from nvalchemi.distributed.validate.payloads import (
    _batch_to_payload,
    _diff_outputs,
)
from nvalchemi.distributed.validate.worker import _worker_main

__all__ = ["_validate_spec", "_infer_spec_from_trace"]


def _infer_spec_from_trace(
    wrapper: Any, trace: list[dict[str, Any]]
) -> MLIPSpec:
    """Build a candidate spec.

    Strategy: take the wrapper's existing ``distribution_spec`` (if
    any) as the strong prior — the wrapper author has already declared
    what they think is right. The trace is used to *flag* potential
    issues for auto-fix later, not to override the prior.

    For wrappers that *don't* declare a spec, we fall back to a
    conservative default and rely on auto-fix to refine.
    """
    _ds = getattr(wrapper, "distribution_spec", None)
    spec = _ds() if callable(_ds) else _ds
    if spec is not None:
        return spec
    # Conservative default for wrappers that haven't declared one.
    # Auto-fix will refine.
    from nvalchemi.distributed._core.spec import DistributionSpec  # noqa: PLC0415
    from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
        HaloStoragePolicy,
    )
    from nvalchemi.distributed.output_kinds import (  # noqa: PLC0415
        OutputKind,
        OutputSpec,
    )

    # Seed output_kinds with the canonical MLIP names so auto-fix
    # rules that gate on ``OutputKind.PER_GRAPH`` / ``PER_NODE`` can
    # fire on undeclared wrappers. Wrappers that emit non-standard
    # output keys (or override these classifications) declare them
    # via :attr:`distribution_spec` and skip this branch.
    return MLIPSpec(
        distribution=DistributionSpec(
            policy=HaloStoragePolicy(
                scatter_mode="halo_correction",
                gather_mode="halo_read",
            )
        ),
        outputs={
            "energy": OutputSpec(OutputKind.PER_GRAPH),
            "forces": OutputSpec(OutputKind.PER_NODE),
            "stress": OutputSpec(OutputKind.PER_GRAPH),
            "atomic_energies": OutputSpec(OutputKind.PER_NODE),
        },
    )


def _await_worker_results(
    procs: list,
    queue: Any,
    timeout_sec: float,
    *,
    poll_interval: float = 0.1,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list, str | None]:
    """Wait for the spawned workers, draining their result queue.

    Polls all workers together instead of joining one-by-one. A worker that
    hits an exception ships an error payload, tears down its process group, and
    exits cleanly — but a peer already blocked in a collective never gets its
    partner and hangs. A sequential join would then block the whole timeout on
    that survivor and report a bare "timed out", discarding the crashed rank's
    traceback. Here, the moment any worker exits after emitting an error
    payload, we stop and let the caller's error-payload branch surface the real
    cause. Returns ``(received_payloads, error)`` where ``error`` is set only
    for a hard crash (non-zero exit, no payload) or a genuine symmetric hang.

    ``monotonic`` / ``sleep`` are injectable for testing.
    """
    received: list = []
    error: str | None = None
    deadline = monotonic() + timeout_sec
    while True:
        while not queue.empty():
            received.append(queue.get_nowait())
        if all(not p.is_alive() for p in procs):
            break
        got_error_payload = any(
            len(m) >= 2 and isinstance(m[1], str) for m in received
        )
        if got_error_payload and any(not p.is_alive() for p in procs):
            # Asymmetric crash: a rank failed and left a survivor deadlocked.
            # Leave ``error`` unset — the payload carries the traceback.
            break
        hard_crash = next(
            (p for p in procs if not p.is_alive() and p.exitcode not in (0, None)),
            None,
        )
        if hard_crash is not None:
            error = (
                f"worker pid={hard_crash.pid} exited with code {hard_crash.exitcode}"
            )
            break
        if monotonic() >= deadline:
            hung = next(p for p in procs if p.is_alive())
            error = f"worker pid={hung.pid} timed out after {timeout_sec}s"
            break
        sleep(poll_interval)
    while not queue.empty():
        received.append(queue.get_nowait())
    return received, error


def _validate_spec(
    model_factory: Callable[[], Any],
    sample_batch: Any,
    *,
    spec: MLIPSpec,
    world_size: int,
    device: str | torch.device,
    backend: str,
    timeout_sec: float,
    ref_outputs: dict[str, torch.Tensor],
    atol: float,
    rtol: float,
    watched_helper_packages: tuple[str, ...] = (),
    helper_sample_every: int = 8,
    ref_helper_calls: list | None = None,
    ref_nl_summary: dict[str, Any] | None = None,
    ref_layer_records: list | None = None,
    layer_diagnostic: bool = False,
) -> dict[str, Any]:
    """Spawn ``world_size`` workers on the same GPU device, run each
    through ``DistributedModel(wrapper, cfg, spec=spec)``, return the
    per-output diff dict + handler count dict + helper diagnostics +
    halo completeness verdict."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []

    spec_dict = spec.to_dict()
    sample_payload = _batch_to_payload(sample_batch)
    device_str = str(device)

    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_main,
            args=(
                rank,
                world_size,
                backend,
                device_str,
                model_factory,
                sample_payload,
                spec_dict,
                queue,
                watched_helper_packages,
                helper_sample_every,
                layer_diagnostic,
            ),
        )
        p.start()
        procs.append(p)

    try:
        received, error = _await_worker_results(procs, queue, timeout_sec)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

    if error is not None:
        return {
            "passed": False,
            "max_abs_diff": {},
            "max_rel_diff": {},
            "handler_counts": {},
            "error": error,
            "helper_diagnostics": [],
            "halo_completeness": None,
            "layer_divergence": None,
            "partition_health": None,
        }

    # Worker emits one of:
    #   (rank, pickle_bytes)               — success.
    #   (rank, "INIT_ERROR", tb)           — pre-init crash, no helper data.
    #   (rank, "RUN_ERROR", tb,
    #          helper_calls, handler_counts)  — forward crash with
    #                                           partial trace data.
    error_payloads = [p for p in received if len(p) >= 3 and isinstance(p[1], str)]
    if error_payloads:
        rank, kind, tb = error_payloads[0][:3]
        tb_short = tb if len(tb) < 4000 else tb[:2000] + "\n...\n" + tb[-1500:]
        # Best-effort: classify whatever helper records the workers
        # managed to emit before crashing. Some workers may have
        # crashed during INIT (no records); RUN_ERROR ones ship a
        # partial pickled list as their 4th tuple element.
        partial_helper_diags: list = []
        partial_handler_counts: dict[str, int] = {}
        try:
            import pickle  # noqa: PLC0415

            per_rank_helper_calls: dict[int, list] = {}
            for payload in error_payloads:
                if len(payload) >= 4 and isinstance(payload[3], (bytes, bytearray)):
                    # noqa: S301 — payload comes from our own spawned
                    # worker (this process pickled it), not untrusted
                    # input.
                    per_rank_helper_calls[payload[0]] = pickle.loads(  # noqa: S301
                        payload[3]
                    )
                # 5-tuple ships partial handler counts. Take rank 0's view
                # if present; if rank 0 didn't send (e.g. it timed out
                # while rank 1 raised), fall back to whichever rank did.
                if len(payload) >= 5 and isinstance(payload[4], (bytes, bytearray)):
                    counts = pickle.loads(payload[4])  # noqa: S301
                    if payload[0] == 0 or not partial_handler_counts:
                        partial_handler_counts = counts
            if per_rank_helper_calls and ref_helper_calls:
                from nvalchemi.distributed._core.helper_diagnosis import (  # noqa: PLC0415
                    classify,
                )

                already_wrapped_fns = {
                    (h.module_path, h.attr_name)
                    for h in spec.distribution.third_party_helpers
                    if hasattr(h, "attr_name")  # module-level helpers only (not MethodAdapter)
                }
                partial_helper_diags = classify(
                    ref_helper_calls,
                    per_rank_helper_calls,
                    already_wrapped_fns=already_wrapped_fns,
                )
        except Exception:  # noqa: BLE001
            # Diagnostic is best-effort; never let its failure mask the
            # original crash.
            partial_helper_diags = []
        return {
            "passed": False,
            "max_abs_diff": {},
            "max_rel_diff": {},
            "handler_counts": partial_handler_counts,
            "error": f"rank {rank} {kind}:\n{tb_short}",
            "helper_diagnostics": partial_helper_diags,
            "halo_completeness": None,
            "layer_divergence": None,
            "partition_health": None,
        }

    success_payloads = [p for p in received if len(p) == 2]
    if not success_payloads:
        return {
            "passed": False,
            "max_abs_diff": {},
            "max_rel_diff": {},
            "handler_counts": {},
            "error": "no successful payloads received",
            "helper_diagnostics": [],
            "halo_completeness": None,
            "layer_divergence": None,
            "partition_health": None,
        }

    import pickle  # noqa: PLC0415

    # Deserialize and merge all rank payloads. Per-atom outputs (forces,
    # per-atom-energies, per-atom-charges) come back as each rank's
    # *owned slice* — concatenating in rank order rebuilds the global
    # tensor with shape ``(n_global, ...)`` so the diff metric's
    # partition-invariant aggregates (sum, max-magnitude) compare
    # apples to apples against the single-process reference. Per-system
    # outputs (energy, stress) are already all-reduced inside
    # ``DistributedModel`` and are identical across ranks — take rank 0.
    all_outputs: dict[int, dict[str, torch.Tensor]] = {}
    handler_counts: dict[str, int] = {}
    per_rank_helper_calls: dict[int, list] = {}
    per_rank_halo_summaries: dict[int, dict[str, Any]] = {}
    per_rank_layer_records: dict[int, list] = {}
    for rank_id, payload_bytes in sorted(success_payloads, key=lambda p: p[0]):
        # noqa: S301 — bytes were pickled by our own spawned worker, not
        # external input.
        deserialized = pickle.loads(payload_bytes)  # noqa: S301
        # Wire format evolution:
        # 2-tuple — pre-helper-trace.
        # 3-tuple — adds helper_calls (helper-trace).
        # 4-tuple — adds halo_summary (halo-completeness diagnostic).
        # 5-tuple — adds layer_records (per-module divergence diagnostic).
        # Tolerate older forms by length so a worker built against an
        # older validator still ships back something usable.
        rank_layer_records: list = []
        if len(deserialized) == 5:
            (
                outputs_np,
                rank_counts,
                rank_helper_calls,
                rank_halo_summary,
                rank_layer_records,
            ) = deserialized
        elif len(deserialized) == 4:
            outputs_np, rank_counts, rank_helper_calls, rank_halo_summary = deserialized
        elif len(deserialized) == 3:
            outputs_np, rank_counts, rank_helper_calls = deserialized
            rank_halo_summary = {}
        else:
            outputs_np, rank_counts = deserialized
            rank_helper_calls = []
            rank_halo_summary = {}
        all_outputs[rank_id] = {k: torch.from_numpy(v) for k, v in outputs_np.items()}
        if rank_id == 0:
            handler_counts = rank_counts
        per_rank_helper_calls[rank_id] = rank_helper_calls
        per_rank_halo_summaries[rank_id] = rank_halo_summary
        per_rank_layer_records[rank_id] = rank_layer_records

    rank_ids_sorted = sorted(all_outputs.keys())
    if 0 not in all_outputs:
        return {
            "passed": False,
            "max_abs_diff": {},
            "max_rel_diff": {},
            "handler_counts": handler_counts,
            "error": (f"no payload from rank 0 (got ranks {rank_ids_sorted})"),
            "helper_diagnostics": [],
            "halo_completeness": None,
            "layer_divergence": None,
            "partition_health": None,
        }

    multi_outputs: dict[str, torch.Tensor] = {}
    rank0_outputs = all_outputs[0]
    for k, ref_v in ref_outputs.items():
        rank0_v = rank0_outputs.get(k)
        if rank0_v is None:
            continue
        if rank0_v.shape == ref_v.shape:
            # Per-system / already-global output (energy after
            # all_reduce, scalar stress, etc.). Identical across ranks.
            multi_outputs[k] = rank0_v
            continue
        # Per-atom output. Reassemble the global tensor by concatenating
        # each rank's owned slice in rank order. Order may differ from
        # the ref's atom order — the diff metric handles that by
        # reverting to partition-invariant aggregates.
        slices = [all_outputs[r].get(k) for r in rank_ids_sorted]
        if any(s is None for s in slices):
            multi_outputs[k] = rank0_v  # incomplete — let metric flag it
            continue
        try:
            multi_outputs[k] = torch.cat(slices, dim=0)
        except RuntimeError:
            multi_outputs[k] = rank0_v  # incompatible — let metric flag it

    abs_diff, rel_diff = _diff_outputs(ref_outputs, multi_outputs)
    # Per-output: pass if absolute or relative diff is within tolerance.
    # Mirrors :func:`torch.testing.assert_close` so extensive quantities
    # (energy ~ N) aren't rejected purely because their absolute diff
    # grew with system size while the relative diff stayed at fp32 noise.
    passed = all(
        abs_diff.get(k, float("inf")) <= atol or rel_diff.get(k, float("inf")) <= rtol
        for k in ref_outputs
    )

    # Classify watched-helper calls. Pass-through to the diagnosis
    # module — empty input lists yield an empty diagnoses list, which
    # is what callers want when ``watched_helper_packages`` was
    # disabled.
    from nvalchemi.distributed._core.helper_diagnosis import classify  # noqa: PLC0415

    already_wrapped_fns = {
        (h.module_path, h.attr_name)
        for h in spec.distribution.third_party_helpers
        if hasattr(h, "attr_name")  # module-level helpers only (not MethodAdapter)
    }
    helper_diagnostics = classify(
        ref_helper_calls or [],
        per_rank_helper_calls,
        already_wrapped_fns=already_wrapped_fns,
    )

    halo_completeness = _check_halo_completeness(
        ref_nl_summary or {}, per_rank_halo_summaries
    )
    partition_health = _partition_health(ref_nl_summary or {}, per_rank_halo_summaries)

    # The layer-by-layer diagnostic compares ref vs the SUM of per-rank
    # owned-row module outputs — a combine rule that only holds for
    # halo storage (owned rows partition the atoms, so the per-rank sums
    # add up to the global sum). Sharded specs gather features to global
    # internally, so each rank's intermediate is global-replicated and
    # the sum double-counts (a false "divergent module" at rel_diff≈0.5).
    # Restrict the diagnostic to halo storage; for sharded the final
    # output diff + helper diagnostics carry the signal instead.
    from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
        HaloStoragePolicy,
    )

    layer_divergence: dict[str, Any] | None = None
    is_halo = isinstance(getattr(spec.distribution, "policy", None), HaloStoragePolicy)
    if layer_diagnostic and ref_layer_records is not None and is_halo:
        from nvalchemi.distributed.validate.layer_diagnostics import (  # noqa: PLC0415
            diff_layer_records,
        )

        layer_divergence = diff_layer_records(ref_layer_records, per_rank_layer_records)

    return {
        "passed": passed,
        "max_abs_diff": abs_diff,
        "max_rel_diff": rel_diff,
        "handler_counts": handler_counts,
        "error": None,
        "helper_diagnostics": helper_diagnostics,
        "halo_completeness": halo_completeness,
        "layer_divergence": layer_divergence,
        "partition_health": partition_health,
    }
