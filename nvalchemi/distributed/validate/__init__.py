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

"""Single-call distributed-spec inference + validation.

`trace_and_validate(model_factory, sample_batch, ...)` does:

1. Reference run — single-process forward on the sample batch, captured
   under :func:`~nvalchemi.distributed._core.dispatch_trace.dispatch_trace` so
   we observe which custom ops fire and what shapes they produce. This
   is also the ground-truth output the multi-rank run is compared
   against.
2. Spec inference — translate the trace's observed firings into a
   candidate :class:`MLIPSpec`. The wrapper's existing
   ``distribution_spec`` (if defined) is treated as a strong prior:
   missing fields are inferred; provided fields are kept verbatim.
3. Validation — spawn ``world_size`` processes on the same GPU device,
   each running the wrapper through :class:`DistributedModel` with the
   inferred spec. Compare per-output tensors against the reference and
   produce a per-output diff.
4. Auto-fix — if the validation diff exceeds tolerance, run a small
   rule engine that proposes spec mutations from a corpus of patterns
   we've encountered (UMA halo-correction double-count,
   under/over-reduction). Each rule is tried in turn; the first that
   clears tolerance wins. The returned ``spec`` is the working one.

The returned :class:`TraceReport` is *actionable*: it gives a single
``next_action`` string and a serializable spec, so the user can either
paste the spec into their wrapper or save it to disk for cache reuse.

Module layout
-------------

``trace_and_validate`` (the public entry) lives here. The internals
are split across siblings:

* :mod:`.types` — :class:`Attempt`, :class:`TraceReport` dataclasses.
* :mod:`.payloads` — Batch ↔ tensor-dict wire format + diff metric.
* :mod:`.reference` — single-process reference run + NL summary.
* :mod:`.halo_diagnostics` — halo-completeness check.
* :mod:`.worker` — per-rank ``mp.spawn`` target.
* :mod:`.inference` — spawn orchestration + spec inference.
* :mod:`.autofix` — rule engine + spec signatures.

Each sibling is independently testable; ``trace_and_validate`` is the
glue that wires them in the documented order.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import torch

from nvalchemi.distributed.spec import MLIPSpec
from nvalchemi.distributed.validate.autofix import (
    _next_fix_candidate,
    _suspect_op_summary,
)
from nvalchemi.distributed.validate.inference import (
    _infer_spec_from_trace,
    _validate_spec,
)
from nvalchemi.distributed.validate.reference import _reference_run
from nvalchemi.distributed.validate.scripted_diagnostics import (
    ScriptedOpReport,
    detect_scripted_ops,
)
from nvalchemi.distributed.validate.types import Attempt, TraceReport

__all__ = [
    "Attempt",
    "ScriptedOpReport",
    "TraceReport",
    "detect_scripted_ops",
    "trace_and_validate",
]


_DEFAULT_WATCHED_HELPER_PACKAGES: tuple[str, ...] = ("aimnet.nbops",)


# ----- Worker-error translators ------------------------------------
#
# Generic torch / autograd errors are cryptic when the underlying cause
# is a framework-specific dispatch interaction. We pattern-match on the
# error string + the partial dispatch trace shipped from the worker and
# rewrite the ``next_action`` to point at the actual root cause. New
# patterns get appended here as we encounter them in the wild.


def _detect_dropped_inplace_dispatch_return(
    error_str: str, handler_counts: dict[str, int]
) -> str | None:
    """Detect the "I dropped my ``scatter_add_`` return" footgun.

    Symptom (worker-side): ``torch.autograd.grad`` raises
    ``RuntimeError: One of the differentiated Tensors appears to not
    have been used in the graph``.

    Mechanism: the wrapper called ``t.scatter_add_(0, idx, src)``
    (or ``index_add_`` / ``index_copy_``) with a ShardTensor source.
    Under domain decomposition the dispatch handler
    (:func:`_halo_scatter_correction`) computes the cross-rank-corrected
    output as a *new* tensor — there is no in-place primitive that also
    does halo_reverse + halo_forward. The handler returns the new
    tensor; the caller's ``t.scatter_add_(...)`` (no rebind) drops it.
    ``t`` stays at its pre-scatter zero, the model output detaches from
    its inputs, and ``autograd.grad`` finds no path back to
    ``positions``.

    Single-process: ``scatter_add_`` returns ``self``, the rebind is a
    no-op, the bug is silent until the wrapper meets the validator.

    Detection: marker is "appears to not have been used in the graph"
    AND ``halo_scatter_correction`` (or its index_add / index_copy
    siblings) fired at least once before the crash.

    Returns the suggested-fix string, or ``None`` if the pattern
    doesn't match.
    """
    if "appears to not have been used in the graph" not in error_str:
        return None
    inplace_handlers = (
        "halo_scatter_correction[scatter_add_]",
        "halo_scatter_correction[index_add_]",
        "halo_scatter_correction[index_copy_]",
        "halo_scatter_correction",
    )
    fired = [
        (h, n)
        for h, n in handler_counts.items()
        if any(h.startswith(p) for p in inplace_handlers) and n > 0
    ]
    if not fired:
        return None
    fired_summary = ", ".join(f"{h}×{n}" for h, n in fired)
    return (
        "Likely cause: an in-place ``scatter_add_`` / ``index_add_`` / "
        "``index_copy_`` inside the wrapper or model returned a *new* "
        "tensor (cross-rank halo correction can't preserve in-place "
        "semantics) and the caller dropped the return. Search the "
        "model code for ``t.scatter_add_(...)`` / ``t.index_add_(...)`` "
        "/ ``t.index_copy_(...)`` patterns where the return value is "
        "discarded, and rebind:\n"
        "    G = G.scatter_add_(0, idx, src)\n"
        "Single-process: the rebind is a no-op (``self`` is returned). "
        "Distributed: the rebind is mandatory for the autograd graph "
        f"to thread through the corrected accumulator. (Observed "
        f"firings: {fired_summary}.)"
    )


def _detect_severed_autograd_graph(
    error_str: str, handler_counts: dict[str, int]
) -> str | None:
    """Detect "the distributed forward severed the positions→energy graph".

    Symptom (worker-side): ``torch.autograd.grad`` raises
    ``RuntimeError: element 0 of tensors does not require grad and does not
    have a grad_fn`` — element 0 being the *output* (energy), which means the
    forward graph from ``positions`` to ``energy`` is broken.

    Mechanism: many MLIPs compute conservative forces *internally* via
    ``autograd.grad(energy, positions)`` (MACE's ``compute_forces``,
    AIMNet2's force head). Under domain decomposition the energy must stay
    connected to the input positions through every ShardTensor op. If a
    dispatch handler or a wrap-back returns a tensor detached from autograd,
    the energy loses its ``grad_fn`` and the internal ``autograd.grad`` finds
    no path back — even though single-process works (the graph is intact
    there).

    Returns the suggested-fix string, or ``None`` if the pattern doesn't
    match.
    """
    markers = (
        "does not require grad and does not have a grad_fn",
        "does not have a grad_fn",
    )
    if not any(m in error_str for m in markers):
        return None
    fired = ", ".join(f"{h}×{n}" for h, n in sorted(handler_counts.items())) or "none"
    return (
        "Likely cause: the model computes conservative forces *internally* "
        "via ``torch.autograd.grad(energy, positions)``, but the distributed "
        "forward severed the autograd graph between ``positions`` and "
        "``energy`` — the energy output reached ``autograd.grad`` without a "
        "``grad_fn``. Single-process works because the graph is intact there; "
        "under distribution one of the ShardTensor dispatch handlers or a "
        "wrap-back returned a tensor detached from autograd. Check that every "
        "op on the energy path preserves autograd (the per-system reduction / "
        "halo-correction handlers are autograd.Functions; a plain re-wrap that "
        "drops ``grad_fn`` is the usual culprit). "
        f"(Dispatch handlers that fired before the crash: {fired}.)"
    )


def _detect_scripted_op_shardtensor_ima(
    error_str: str, handler_counts: dict[str, int]
) -> str | None:
    """Detect the ``@torch.jit.script`` + ShardTensor CUDA illegal-memory-access.

    Symptom (worker-side): a ``RuntimeError`` from "the TorchScript interpreter"
    whose payload is an "illegal memory access" / "CUDA driver error" — often
    surfacing as a Warp ``wp_free_device_async`` fault on the next allocation.

    Mechanism: a scripted op (e.g. e3nn's ``_spherical_harmonics``) received a
    requires-grad ShardTensor on the halo path. TorchScript bypasses
    ``__torch_function__``, so the storage-less wrapper enters the JIT graph
    raw; its TensorExpr-fused kernel reads the near-null ``data_ptr`` → IMA.
    Marshalling the op across the boundary (Route C) fixes it.
    """
    markers = ("illegal memory access", "cuda driver error", "wp_free_device_async")
    lowered = error_str.lower()
    if not any(m in lowered for m in markers):
        return None
    scripted_context = (
        "torchscript" in lowered
        or "jit interpreter" in lowered
        or "wp_free_device_async" in lowered
        or "spherical_harmonics" in lowered
    )
    if not scripted_context:
        return None
    return (
        "Likely cause: a ``@torch.jit.script`` op received a requires-grad "
        "ShardTensor on the distributed halo path. TorchScript bypasses "
        "``__torch_function__``, so the storage-less ShardTensor enters the JIT "
        "graph raw and its TensorExpr-fused CUDA kernel reads a near-null "
        "``data_ptr`` → illegal memory access (the Warp ``wp_free_device_async`` "
        "fault is usually the *next* allocation tripping over the corrupted "
        "context, not the real site). Fix by MARSHALLING the scripted op across "
        "the boundary (Route C): unwrap ShardTensor→local, run the still-scripted "
        "op, re-wrap. Scripted *submodules* are auto-marshalled by the default "
        "``DomainConfig.scripted_marshal=\"auto\"``; a module-level scripted "
        "*function* (the usual culprit, e.g. ``e3nn.o3._spherical_harmonics``) "
        "must be DECLARED — add ``JitAdapter(module_path, attr, mode=\"marshal\")`` "
        "to the spec's ``distribution.third_party_helpers``. Run the pre-flight "
        "``detect_scripted_ops(model, spec)`` to list undeclared scripted "
        "functions and get a paste-able delta."
    )


def _translate_worker_error(
    error_str: str, handler_counts: dict[str, int]
) -> str | None:
    """Run all worker-error translators in priority order; return the
    first match, or ``None`` if no translator fires."""
    for translator in (
        _detect_scripted_op_shardtensor_ima,
        _detect_dropped_inplace_dispatch_return,
        _detect_severed_autograd_graph,
    ):
        hint = translator(error_str, handler_counts)
        if hint is not None:
            return hint
    return None


def _format_partition_health(
    ph: dict[str, Any] | None, *, degenerate_only: bool = False
) -> str:
    """Render the partition-health verdict for ``next_action``.

    ``degenerate_only`` (used on the success path) returns text ONLY when the
    partition is degenerate — a clean pass on a healthy partition needs no
    note. On the failure path it always appends the per-rank composition as
    diagnostic context.
    """
    if not ph:
        return ""
    if degenerate_only and not ph.get("degenerate"):
        return ""
    lines: list[str] = []
    if ph.get("degenerate"):
        lines.append(
            "Partition is DEGENERATE — this run did not meaningfully exercise "
            "domain decomposition, so the result is not evidence the spec is "
            "correct. Pick a larger system / different world_size so every rank "
            "has non-trivial owned + halo + remote atoms:"
        )
        lines.extend("  - " + m for m in ph["degenerate"])
        lines.append(
            "  Rule of thumb: a partitioned axis only develops remote atoms once "
            "its per-rank domain exceeds two ghost widths, i.e. "
            "box_axis / ranks_on_axis > 2 * ghost_width  (ghost_width ~= "
            "cutoff + skin). Below that every rank ghosts its neighbour's entire "
            "domain (remote == 0). E.g. ghost_width 6 Ang with a 2-way split "
            "needs box > 24 Ang on that axis."
        )
    comp = ", ".join(
        f"rank{r}(owned={d['owned']}, halo={d['halo']}, remote={d['remote']})"
        for r, d in sorted(ph.get("per_rank", {}).items())
    )
    if comp:
        lines.append(f"Partition composition: {comp}.")
    return "\n".join(lines)


def trace_and_validate(
    model_factory: Callable[[], Any],
    sample_batch: Any,
    *,
    world_size: int = 2,
    device: str | torch.device = "cuda:0",
    atol: float = 1e-5,
    rtol: float = 1e-4,
    auto_fix: bool = True,
    max_fix_attempts: int = 8,
    backend: str = "auto",
    timeout_sec: float = 120.0,
    watched_helper_packages: Sequence[str] | None = None,
    helper_sample_every: int = 8,
    layer_diagnostic: bool = True,
) -> TraceReport:
    """Infer a distribution spec, validate it on a single-GPU multi-process
    run, and (optionally) auto-fix when validation fails.

    Parameters
    ----------
    model_factory
        Callable returning a freshly-constructed wrapper. Called once
        in the launcher process for the reference run, and once per
        rank in each spawned worker. Pristine state every time —
        no shared module graph between processes.
    sample_batch
        A :class:`~nvalchemi.data.Batch` (or compatible) carrying
        positions / cell / pbc on the target ``device``. Small enough
        that ``world_size`` copies fit in memory at once.
    world_size
        Virtual ranks to spawn on the same GPU. The default (2) is
        sufficient to flush the dispatch logic; larger values catch
        partition-dependent bugs but cost spawn overhead linearly.
    device
        CUDA device all ranks bind to. Default ``"cuda:0"``. CPU
        validation is *not* supported by this entry point — CPU/GPU
        numerical drift makes it unreliable; if you need it, call
        the harness in ``test_dispatch_trace_gloo.py`` directly.
    atol
        Per-output absolute tolerance. Pass criterion (per output) is
        ``abs_diff <= atol OR rel_diff <= rtol`` — same convention
        :func:`torch.testing.assert_close` uses, so extensive
        quantities (energy scales linearly with atom count) compare
        correctly across system sizes.
    rtol
        Per-output relative tolerance. Default ``1e-4`` covers fp32
        round-off accumulation across collective reductions on the
        ``cpu:gloo,cuda:gloo`` backend; tighten to e.g. ``1e-5`` when
        running NCCL or fp64.
    auto_fix
        When the initial inferred spec fails validation, try
        rule-based mutations. Disable to get a single-attempt report.
    max_fix_attempts
        Cap on the number of distinct specs auto-fix will try.
    backend
        ``"nccl"``, ``"gloo"``, or ``"auto"`` (NCCL when CUDA is
        available, else Gloo). Both correctly route over CUDA tensors;
        NCCL is faster.
    timeout_sec
        Per-spawn join timeout.
    watched_helper_packages
        Fully-qualified module paths whose top-level Python helpers
        get instrumented during the reference and per-rank runs. The
        :mod:`~nvalchemi.distributed._core.helper_trace` proxy records each
        call's input / output shapes + sums; the
        :mod:`~nvalchemi.distributed._core.helper_diagnosis` classifier then
        flags helpers that look like distribution gaps (per-system
        reductions whose per-rank outputs sum to the reference output
        but aren't declared in ``spec.distribution.third_party_helpers``).
        Defaults to ``("aimnet.nbops",)``. Pass an explicit empty
        tuple to disable. Unimportable packages are skipped silently.
    helper_sample_every
        Record every Nth call after the first call per
        ``(module, function)``. Default 8 keeps overhead bounded for
        hot helpers (``mol_sum`` runs multiple times per layer); set
        to 1 for exhaustive recording (debug only).

    Returns
    -------
    TraceReport
        Carries the working (or best-guess) spec, every attempt's
        diff/handler-counts, and a one-line ``next_action``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "trace_and_validate requires CUDA — single-GPU multi-process "
            "spawn is the validation primitive (CPU validation is "
            "explicitly out of scope)."
        )

    # Warp kernel cache: ensure both the launcher's reference run and
    # spawned workers can write JIT artefacts. Default
    # ``~/.cache/warp/`` may be read-only (sandboxed dev envs); route
    # to a writable temp location. Set at the launcher level so spawned
    # children inherit it before any nvalchemiops/warp import runs.
    if "WARP_CACHE_PATH" not in os.environ:
        import tempfile  # noqa: PLC0415

        os.environ["WARP_CACHE_PATH"] = os.path.join(
            tempfile.gettempdir(), "nvalchemi-validate-warp-cache"
        )

    # Suppress noisy external warnings during the validator run:
    # Gloo connection messages, Warp's ``warp.context`` deprecation,
    # and the ``.grad attribute of a non-leaf`` warning fired inside
    # ``warp/_src/torch.py``. None are actionable from user code.
    import warnings as _warnings  # noqa: PLC0415

    os.environ.setdefault("GLOO_LOG_LEVEL", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    # Silences PyTorch's C++-side ``ProcessGroup`` teardown warnings
    # (``No backend of type 0 found``). Gloo's C++ ``Pair::connect``
    # rank-connect log line still leaks through — those come from
    # ``transport/tcp/pair.cc`` which doesn't honour any of these env
    # vars; they're cosmetic-only and harmless.
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
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
    # Warp's deprecation warnings bypass the Python ``warnings`` machinery
    # (they call ``sys.stdout.write`` directly via ``warp_showwarning``)
    # but DO consult the package-internal ``warnings_seen`` dedupe set
    # — pre-populating that set is the only reliable way to keep the
    # noise out of the validator's output. Both the launcher and each
    # spawned worker pre-populate independently (the workers inherit
    # ``warnings_seen`` clean since it's process-local state).
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

    # The validator spawns every virtual rank on the SAME device, so NCCL is
    # off the table (it rejects multiple ranks sharing one device, "Duplicate
    # GPU detected"). A plain ``backend="gloo"`` still routes cuda-tensor
    # collectives through NCCL (PyTorch's default ``cpu:gloo,cuda:nccl`` map), so
    # pin BOTH device classes to gloo. Real multi-GPU NCCL runs go through the
    # benchmark scripts' torchrun launchers, not this debugging harness.
    if backend == "auto":
        backend_resolved = "cpu:gloo,cuda:gloo"
    else:
        backend_resolved = backend

    if watched_helper_packages is None:
        watched_helper_packages = _DEFAULT_WATCHED_HELPER_PACKAGES
    watched_helper_packages = tuple(watched_helper_packages)

    # Reference run + initial inference.
    (
        ref_outputs,
        initial_trace,
        ref_helper_calls,
        ref_nl_summary,
        ref_layer_records,
    ) = _reference_run(
        model_factory,
        sample_batch,
        watched_helper_packages=watched_helper_packages,
        helper_sample_every=helper_sample_every,
        layer_diagnostic=layer_diagnostic,
    )
    initial_spec = _infer_spec_from_trace(model_factory(), initial_trace)

    # Scripted-op pre-flight (static; no GPU). Flag module-level
    # ``@torch.jit.script`` functions auto-discovery can't wrap — the
    # @torch.jit.script + ShardTensor illegal-memory-access vector — and, when
    # auto-fixing, declare a marshalling JitAdapter for each so the spawn run
    # doesn't IMA.
    from nvalchemi.distributed.validate.scripted_diagnostics import (  # noqa: PLC0415
        apply_marshal_adapters,
        detect_scripted_ops,
    )

    scripted_report = detect_scripted_ops(model_factory(), initial_spec)
    preflight_hint = scripted_report.format_hint()
    rationale = "initial inference from single-rank trace"
    if scripted_report.has_risk and auto_fix:
        initial_spec = apply_marshal_adapters(
            initial_spec, scripted_report.undeclared_functions
        )
        _injected = ", ".join(
            f"{mp}.{attr}" for mp, attr in scripted_report.undeclared_functions
        )
        rationale = (
            "initial inference + auto-marshalled undeclared scripted "
            f"function(s): {_injected}"
        )

    attempts: list[Attempt] = []
    spec = initial_spec

    for attempt_idx in range(max_fix_attempts):
        result = _validate_spec(
            model_factory,
            sample_batch,
            spec=spec,
            world_size=world_size,
            device=device,
            backend=backend_resolved,
            timeout_sec=timeout_sec,
            ref_outputs=ref_outputs,
            atol=atol,
            rtol=rtol,
            watched_helper_packages=watched_helper_packages,
            helper_sample_every=helper_sample_every,
            ref_helper_calls=ref_helper_calls,
            ref_nl_summary=ref_nl_summary,
            ref_layer_records=ref_layer_records,
            layer_diagnostic=layer_diagnostic,
        )
        attempts.append(Attempt(spec=spec, rationale=rationale, **result))

        if attempts[-1].passed:
            ok_action = (
                f"OK — paste spec from report.spec (passed at attempt "
                f"#{attempt_idx + 1})"
            )
            if preflight_hint:
                # The run passed; if undeclared scripted functions were
                # auto-marshalled, tell the user to make it permanent.
                ok_action += (
                    "\n\nScripted-op pre-flight (auto-marshalled for this run "
                    "— declare these on your wrapper's spec to make it "
                    "permanent):\n" + preflight_hint
                )
            # A green result on a degenerate partition is a trap — surface it.
            degen = _format_partition_health(
                attempts[-1].partition_health, degenerate_only=True
            )
            if degen:
                ok_action += "\nWARNING: " + degen
            return TraceReport(
                ok=True,
                spec=spec,
                attempts=attempts,
                next_action=ok_action,
            )

        if not auto_fix:
            break

        # Pick the next rule to try. Returns ``None`` when no rule's
        # predicate matches — we've exhausted what the engine knows.
        candidate = _next_fix_candidate(spec, attempts)
        if candidate is None:
            break
        spec, rationale = candidate

    # Failed. Best guess + actionable next step.
    last = attempts[-1]
    if last.error is not None:
        next_action = f"FAIL — worker raised before completing forward.\n{last.error}"
        # Translate generic torch errors into framework-specific hints
        # using the partial dispatch trace shipped from the worker.
        # When this fires it usually pinpoints the root cause directly,
        # so it goes BEFORE the helper-gap branch.
        translated = _translate_worker_error(last.error, last.handler_counts or {})
        if translated is not None:
            next_action += "\n\nDiagnosis: " + translated
        if last.handler_counts:
            counts_summary = ", ".join(
                f"{h}×{n}" for h, n in sorted(last.handler_counts.items())
            )
            next_action += (
                f"\n\nDispatch trace before the crash: {counts_summary}. "
                "An empty trace usually means the wrapper crashed in "
                "construction or before any ShardTensor reached a "
                "registered op; a non-empty trace tells you which "
                "dispatch paths the wrapper exercised."
            )
        helper_gaps = [
            d for d in last.helper_diagnostics if d.suspected_gap is not None
        ]
        if helper_gaps:
            next_action += (
                "\nPartial helper-trace before the crash flagged "
                f"{len(helper_gaps)} suspected gap(s): "
                + ", ".join(f"{d.module}.{d.function}" for d in helper_gaps)
                + ". An unwrapped third-party helper that should have "
                "been distributed is the typical root cause of this "
                "failure mode. Inspect "
                "``report.attempts[-1].helper_diagnostics`` for details."
            )
    else:
        suspect = _suspect_op_summary(last.handler_counts)
        next_action = (
            f"FAIL — auto-fix exhausted after {len(attempts)} attempts. "
            f"Closest variant in report.spec; "
            f"ΔE_max={max(last.max_abs_diff.values(), default=0.0):.3e}. "
            f"{suspect}"
        )
        helper_gaps = [
            d for d in last.helper_diagnostics if d.suspected_gap is not None
        ]
        if helper_gaps:
            next_action += (
                "\nSuspected third-party helper gaps "
                f"({len(helper_gaps)}): "
                + ", ".join(f"{d.module}.{d.function}" for d in helper_gaps)
                + ". Inspect ``report.attempts[-1].helper_diagnostics`` "
                "for per-helper details (pattern, consistency check, "
                "suggested remedy template)."
            )
        # Surface free-form divergence notes too — these fire even
        # when the formal classifier can't reach a verdict, giving the
        # user a starting point ("rank values agree but disagree with
        # ref by 12% — likely local-edge-graph computation"). Filter
        # to helpers without a formal gap so we don't double-report.
        helper_notes: list[str] = []
        for d in last.helper_diagnostics:
            if d.suspected_gap is not None:
                continue  # already mentioned above
            helper_notes.extend(
                f"  - {d.module}.{d.function}: {note}" for note in d.divergence_notes
            )
        if helper_notes:
            next_action += (
                "\nWatched-helper divergences (no formal classifier "
                f"verdict; informational, {len(helper_notes)} total):\n"
                + "\n".join(helper_notes)
            )
        # Partition health is the FIRST thing to rule out: a degenerate
        # partition (no halo / no remote / empty shard) makes every
        # downstream verdict suspect, and a too-small system is a common
        # cause of "halo coverage incomplete".
        ph_note = _format_partition_health(last.partition_health)
        if ph_note:
            next_action += "\n" + ph_note
        # Halo-completeness verdict comes BEFORE helper-trace
        # interpretation in causal order — if halo is missing edges,
        # downstream output divergences trace to that, not to the
        # combine rule. Surface it prominently so the reader sees the
        # root-cause line first; surface the *positive* case too so
        # readers know halo has been ruled out as a cause.
        hc = last.halo_completeness
        if hc:
            if not hc.get("matches", True):
                next_action += "\nHalo coverage check: " + hc.get(
                    "interpretation", "halo coverage incomplete"
                )
            else:
                ref_total = hc.get("ref_total_valid_edges", "?")
                next_action += (
                    f"\nHalo coverage check: VERIFIED — every owned atom "
                    f"on every rank sees the same neighbor count as "
                    f"single-process ({ref_total} edges total, owned "
                    f"sums match per-rank). Halo construction is "
                    f"correct; output divergences originate elsewhere "
                    f"(combine rule, autograd graph topology, or "
                    f"non-decomposable computation)."
                )
        ld = last.layer_divergence
        if ld is not None:
            fd = ld.get("first_divergent")
            checked = ld.get("checked", 0)
            if fd is not None:
                next_action += (
                    f"\nLayer-by-layer diagnostic: first divergent module "
                    f"is ``{fd['module']}`` at rel_diff="
                    f"{fd['rel_diff']:.2e} (sum-of-ranks "
                    f"{fd['ranks_sum']:.4e} vs ref {fd['ref_sum']:.4e}). "
                    f"Look for a missing distribution wrapper here, or "
                    f"upstream input plumbing — checked {checked} modules "
                    f"in execution order."
                )
            elif checked > 0:
                next_action += (
                    f"\nLayer-by-layer diagnostic: every module's "
                    f"sum-of-ranks matched ref to within tolerance "
                    f"({checked} modules checked, max rel_diff="
                    f"{ld.get('max_rel_diff', 0.0):.2e}). The divergence "
                    f"is in a non-Module computation (autograd-derived "
                    f"output like forces/stress, post-model consolidation, "
                    f"or a kernel that bypasses sub-module hooks)."
                )
    if preflight_hint:
        next_action += "\n\nScripted-op pre-flight: " + preflight_hint
    return TraceReport(
        ok=False,
        spec=attempts[-1].spec,
        attempts=attempts,
        next_action=next_action,
    )
