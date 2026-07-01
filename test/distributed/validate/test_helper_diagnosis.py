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

"""Spoofs that prove the helper-trace diagnostic flags real gaps.

The validator's helper-trace + diagnosis pipeline turns "validator
failed mysteriously" into "here's the third-party helper you forgot
to wrap." These tests cover two layers:

* **Classifier in isolation** (CPU, no spawn): feed synthetic
  :class:`HelperCall` records that mimic an unwrapped per-system
  reduction into :func:`classify` and assert the per-system-reduction
  pattern is detected with the right consistency check, gap text, and
  remedy template.

* **End-to-end through trace_and_validate** (CUDA, spawned): use a
  partial-wrap AIMNet2 spoof whose forward *completes* but produces
  wrong numbers (calc_masks wrapped so dispatch indices stay in range,
  mol_sum left unwrapped so each rank's per-system sum is rank-local).
  Assert the diagnostic flags ``aimnet.nbops.mol_sum`` and stays
  quiet on the fully-correct AIMNet2 wrapper.

Why not test the *no-wraps* case end-to-end? Without ``calc_masks``
wrapped, the dispatch handler hits OOB indices on the cross-rank
neighbor matrix and the worker dies on a CUDA device-side assert. The
validator's timeout still terminates it and the partial helper-trace
records are still shipped (the validator's RUN_ERROR branch handles
this), but the OS-level kill takes long enough that turning that case
into a fast unit test isn't worth it. The classifier-in-isolation test
covers what we'd verify if it ran cleanly.
"""

from __future__ import annotations

import os as _os
import tempfile as _tempfile

# Mirror test_validate_cuda's WARP cache setup so warp.init doesn't
# trip on read-only ``~/.cache/warp/`` in sandboxed dev envs.
_os.environ.setdefault(
    "WARP_CACHE_PATH",
    _os.path.join(_tempfile.gettempdir(), "nvalchemi-validate-warp-cache"),
)

import pytest  # noqa: E402
import torch  # noqa: E402

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required: trace_and_validate uses single-GPU multi-process spawn",
)


# ----------------------------------------------------------------------
# Classifier-in-isolation tests — exercise pattern detection +
# consistency check + remedy text without the heavy spawn path.
# ----------------------------------------------------------------------


def _make_per_system_reduction_records(
    *,
    n_local: int,
    n_systems: int,
    rank_partial_sums: list[float],
    ref_sum: float,
):
    """Build :class:`HelperCall` records that look like a per-system
    reduction. Reference run: shape ``(n_local * world_size, F)``
    input, ``(n_systems, F)`` output. Per-rank: shape
    ``(n_local, F)`` input, ``(n_systems, F)`` output (each rank
    holding a partial sum that aggregates to the reference)."""
    from nvalchemi.distributed._core.helper_trace import HelperCall

    world_size = len(rank_partial_sums)
    n_global = n_local * world_size
    F = 32

    def _summary(shape, dtype, sum_v=None, max_v=None):
        return {
            "shape": shape,
            "dtype": dtype,
            "sum": sum_v,
            "max_abs": max_v,
        }

    ref_input = _summary((n_global, F), "torch.float32", sum_v=1.5, max_v=0.7)
    ref_output = _summary((n_systems, F), "torch.float32", sum_v=ref_sum, max_v=0.5)
    ref = HelperCall(
        module="thirdparty.helpers",
        function="my_per_system_reduce",
        rank=-1,
        call_index=0,
        input_summary={"arg0": ref_input},
        output_summary=ref_output,
    )

    per_rank: dict[int, list[HelperCall]] = {}
    for r, partial in enumerate(rank_partial_sums):
        rank_in = _summary((n_local, F), "torch.float32", sum_v=0.5, max_v=0.7)
        rank_out = _summary((n_systems, F), "torch.float32", sum_v=partial, max_v=0.5)
        per_rank[r] = [
            HelperCall(
                module="thirdparty.helpers",
                function="my_per_system_reduce",
                rank=r,
                call_index=0,
                input_summary={"arg0": rank_in},
                output_summary=rank_out,
            )
        ]
    return [ref], per_rank


def test_classifier_flags_unwrapped_per_system_reduction():
    """Synthetic records: per-rank outputs sum to ref output, helper
    isn't in ``already_wrapped_fns``. Diagnostic should flag it,
    pattern=per_system_reduction, with a remedy mentioning
    ``per_system_reduce`` and ``PythonAdapter``."""
    from nvalchemi.distributed._core.helper_diagnosis import classify

    ref_calls, per_rank_calls = _make_per_system_reduction_records(
        n_local=4,
        n_systems=1,
        rank_partial_sums=[0.6, 0.4],  # sum to 1.0
        ref_sum=1.0,
    )
    diags = classify(ref_calls, per_rank_calls, already_wrapped_fns=set())
    assert len(diags) == 1
    d = diags[0]
    assert (d.module, d.function) == ("thirdparty.helpers", "my_per_system_reduce")
    assert d.pattern == "per_system_reduction"
    assert d.consistency_passed
    assert not d.already_wrapped
    assert d.suspected_gap is not None
    assert "per-system reduction" in d.suspected_gap
    assert d.likely_remedy is not None
    assert "per_system_reduce" in d.likely_remedy
    assert "PythonAdapter" in d.likely_remedy


def test_classifier_silent_when_already_wrapped():
    """Same per-rank-sums-to-ref pattern, but the spec declares this
    helper wrapped. Diagnostic should observe the pattern (still set
    ``pattern`` and ``consistency_passed``) but NOT emit
    ``suspected_gap`` — we trust the spec."""
    from nvalchemi.distributed._core.helper_diagnosis import classify

    ref_calls, per_rank_calls = _make_per_system_reduction_records(
        n_local=4,
        n_systems=1,
        rank_partial_sums=[0.6, 0.4],
        ref_sum=1.0,
    )
    diags = classify(
        ref_calls,
        per_rank_calls,
        already_wrapped_fns={("thirdparty.helpers", "my_per_system_reduce")},
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.already_wrapped
    assert d.suspected_gap is None
    assert d.likely_remedy is None
    # Still classified — the trace data informed the verdict, we just
    # don't emit a remedy for an already-wrapped helper.
    assert d.pattern == "per_system_reduction"


def test_classifier_silent_when_per_rank_sums_disagree():
    """Per-rank outputs DON'T sum to ref — could mean the helper is a
    different pattern, or the wrap is broken in a way that's not the
    "missing all_reduce" gap. Diagnostic should NOT flag a per-system
    reduction (consistency check fails) and so emit no remedy."""
    from nvalchemi.distributed._core.helper_diagnosis import classify

    ref_calls, per_rank_calls = _make_per_system_reduction_records(
        n_local=4,
        n_systems=1,
        rank_partial_sums=[0.6, 0.4],
        ref_sum=0.99,  # off by 0.01 — well above 1e-3 rel
    )
    diags = classify(ref_calls, per_rank_calls, already_wrapped_fns=set())
    d = diags[0]
    # Pattern is still detected by shape, but consistency_passed=False
    # → no gap is asserted.
    assert d.pattern == "per_system_reduction"
    assert not d.consistency_passed
    assert d.suspected_gap is None


# ----------------------------------------------------------------------
# End-to-end test: a partial-wrap AIMNet2 spoof that completes its
# forward (calc_masks IS wrapped, mol_sum is NOT). The validator's
# distributed_index_select stays in range, the forward returns wrong
# numbers, and the diagnostic flags mol_sum.
# ----------------------------------------------------------------------


def _make_octane_chain(n_atoms: int = 8):
    from nvalchemi.data import AtomicData, Batch

    dtype = torch.float32
    positions = torch.stack(
        [
            0.25 + torch.arange(n_atoms, dtype=dtype) * 1.5,
            torch.zeros(n_atoms, dtype=dtype),
            torch.zeros(n_atoms, dtype=dtype),
        ],
        dim=1,
    ).contiguous()
    atomic_numbers = torch.full((n_atoms,), 6, dtype=torch.long)
    # Size the domain box (the partitioner uses the cell) to the chain extent
    # along x so a 2-rank bisection assigns owned atoms to BOTH ranks. A fixed
    # 100 Å cube leaves the short chain bunched in one corner, so the split
    # gives one rank 0 owned atoms — a degenerate partition the framework
    # rejects (masking the helper-diagnosis behaviour under test).
    # Size the domain box (the partitioner uses the cell) so a 2-rank split
    # falls along x — the only axis the chain extends along. The partitioner
    # builds a cell grid of floor(dim / cutoff) cells per axis and splits along
    # the axis with the most cells; off-axis dims >= cutoff create competing
    # cells that can win the split, dropping every (y=z=0) atom onto one rank
    # (the other gets 0 owned — a degenerate partition the framework rejects).
    # Keep x at the chain extent and the off-axis dims below the cutoff (one
    # cell each) so the split is forced onto x and both ranks own atoms.
    x_extent = 0.5 + n_atoms * 1.5
    cell = torch.diag(torch.tensor([x_extent, 3.0, 3.0], dtype=dtype))
    pbc = torch.zeros(3, dtype=torch.bool)
    data = AtomicData(
        positions=positions.cuda(),
        atomic_numbers=atomic_numbers.cuda(),
        cell=cell.unsqueeze(0).cuda(),
        pbc=pbc.unsqueeze(0).cuda(),
    )
    return Batch.from_data_list([data], device="cuda")


def _make_real_aimnet2():
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    w = AIMNet2Wrapper.from_checkpoint("aimnet2", device="cuda")
    w.eval()
    w.model_config.active_outputs = {"energy", "forces"}
    return w


class _ProxyWrapper(torch.nn.Module):
    """Module-level proxy over a real AIMNet2Wrapper with overridden
    ``distribution_spec`` and (optionally) ``distributed_setup`` /
    ``distributed_teardown``.

    Module-level (not nested) so ``mp.spawn`` can pickle it. The runtime
    metadata API is a single :class:`DistributedContext` reference, so the
    proxy doesn't forward per-attr writes to the inner — everything routes
    through ``inner._dist_ctx`` once :meth:`distributed_setup` records it.
    """

    _PROXY_OWN = frozenset({"_inner", "_spec", "_setup_fn", "_teardown_fn"})

    def __init__(self, inner, spec, *, setup_fn=None, teardown_fn=None):
        super().__init__()
        # Use __dict__ directly to avoid recursion with our __setattr__.
        self.__dict__["_inner"] = inner
        self.__dict__["_spec"] = spec
        self.__dict__["_setup_fn"] = setup_fn
        self.__dict__["_teardown_fn"] = teardown_fn

    def __setattr__(self, name, value):
        if name in self._PROXY_OWN:
            self.__dict__[name] = value
            return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        # ``__getattr__`` only fires for missing attributes, so the
        # proxy's own state still resolves normally. Delegate
        # everything else to inner so callers see the real
        # AIMNet2Wrapper's surface (e.g. ``cutoff`` property).
        return getattr(self._inner, name)

    @property
    def distribution_spec(self):
        return self._spec

    @property
    def model_config(self):
        return self._inner.model_config

    @property
    def model(self):
        return self._inner.model

    def to(self, *args, **kwargs):
        self.__dict__["_inner"] = self._inner.to(*args, **kwargs)
        return self

    def parameters(self, *args, **kwargs):
        return self._inner.parameters(*args, **kwargs)

    def eval(self):
        self._inner.eval()
        return self

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def distributed_setup(self, ctx):
        if self._setup_fn is not None:
            return self._setup_fn(self._inner, ctx)
        return self._inner.distributed_setup(ctx)

    def distributed_teardown(self):
        if self._teardown_fn is not None:
            return self._teardown_fn(self._inner)
        return self._inner.distributed_teardown()


def _setup_only_calc_masks(inner, ctx):
    """``distributed_setup`` that wraps ``calc_masks`` but skips
    ``mol_sum``. Forward will run cleanly (dispatch indices stay in
    range) but mol_sum returns rank-local sums instead of global —
    exactly the gap the diagnostic should catch."""
    from nvalchemi.distributed._core.adapter import PythonAdapter
    from nvalchemi.models.aimnet2 import _distributed_calc_masks

    inner._dist_ctx = ctx

    adapter = PythonAdapter(
        module_path="aimnet.nbops",
        attr_name="calc_masks",
        replacement=_distributed_calc_masks,
    )
    inner._python_helper_adapters = [adapter]
    inner._python_helper_mementos = [adapter.install()]


def _teardown_only_calc_masks(inner):
    adapters = getattr(inner, "_python_helper_adapters", [])
    mementos = getattr(inner, "_python_helper_mementos", [])
    for a, m in zip(adapters, mementos):
        a.restore(m)
    inner._python_helper_adapters = []
    inner._python_helper_mementos = []
    inner._dist_ctx = None


def _make_aimnet2_only_calc_masks():
    """Spoof factory: wraps ``calc_masks`` but not ``mol_sum``. Spec
    declares calc_masks wrapped, so the diagnostic correctly trusts
    that one and only flags mol_sum."""
    import dataclasses

    from nvalchemi.distributed._core.adapter import PythonAdapter
    from nvalchemi.distributed.spec import MLIPSpec

    inner = _make_real_aimnet2()
    base_spec = inner.distribution_spec()
    spoof_core = dataclasses.replace(
        base_spec.distribution,
        third_party_helpers=(
            PythonAdapter(module_path="aimnet.nbops", attr_name="calc_masks"),
        ),
    )
    spoof_spec = MLIPSpec(
        distribution=spoof_core,
        owned_only_outputs=base_spec.owned_only_outputs,
        all_reduce_outputs=base_spec.all_reduce_outputs,
    )
    return _ProxyWrapper(
        inner,
        spoof_spec,
        setup_fn=_setup_only_calc_masks,
        teardown_fn=_teardown_only_calc_masks,
    )


def _make_aimnet2_correct():
    """Standard AIMNet2 wrapping — negative control."""
    return _make_real_aimnet2()


@cuda_required
def test_e2e_diagnostic_flags_unwrapped_mol_sum():
    """Partial-wrap spoof: validator fails AND ``aimnet.nbops.mol_sum``
    appears in the diagnostic with ``pattern=per_system_reduction``,
    ``consistency_passed=True``, and a non-empty remedy."""
    pytest.importorskip("aimnet")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_aimnet2_only_calc_masks,
        _make_octane_chain(n_atoms=24),
        world_size=2,
        device="cuda:0",
        atol=1e-5,
        rtol=1e-4,
        auto_fix=False,
    )

    assert not report.ok, (
        "expected validator to fail when mol_sum is unwrapped; "
        f"got next_action={report.next_action!r}"
    )

    diags = report.attempts[-1].helper_diagnostics
    assert diags, (
        "expected at least one helper_diagnostic; the trace wasn't wired through"
    )

    flagged = {(d.module, d.function): d for d in diags if d.suspected_gap}
    assert ("aimnet.nbops", "mol_sum") in flagged, (
        f"expected mol_sum to be flagged; flagged={list(flagged)}"
    )

    diag = flagged[("aimnet.nbops", "mol_sum")]
    assert diag.pattern == "per_system_reduction"
    assert diag.consistency_passed, (
        f"per-rank outputs should sum to ref; got {diag.consistency_check}"
    )
    assert not diag.already_wrapped
    assert "per_system_reduce" in diag.likely_remedy
    assert "PythonAdapter" in diag.likely_remedy

    # next_action should point the user at the diagnostic.
    assert "mol_sum" in report.next_action

    # The wrapped helper should be marked as already_wrapped — the
    # diagnostic ran on it but trusted the spec.
    by_fn = {(d.module, d.function): d for d in diags}
    if ("aimnet.nbops", "calc_masks") in by_fn:
        assert by_fn[("aimnet.nbops", "calc_masks")].already_wrapped
        assert by_fn[("aimnet.nbops", "calc_masks")].suspected_gap is None


@cuda_required
def test_e2e_diagnostic_silent_on_correctly_wrapped():
    """Real AIMNet2 wrapper: nothing flagged. Confirms no false
    positives on already-wrapped helpers in the end-to-end path."""
    pytest.importorskip("aimnet")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        _make_aimnet2_correct,
        _make_octane_chain(n_atoms=24),
        world_size=2,
        device="cuda:0",
        atol=1e-5,
        rtol=1e-4,
    )
    assert report.ok, report.next_action

    diags = report.attempts[-1].helper_diagnostics
    flagged = [d for d in diags if d.suspected_gap]
    assert flagged == [], (
        f"expected no flagged helpers on correctly-wrapped AIMNet2; "
        f"got {[(d.module, d.function) for d in flagged]}"
    )
