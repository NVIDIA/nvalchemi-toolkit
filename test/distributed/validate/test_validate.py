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

"""Unit tests for ``nvalchemi.distributed.validate`` machinery.

These exercise the framework pieces that don't need a multi-process
spawn: the rule engine on synthetic ``Attempt`` records, the spec
serialization round-trip, and the ``DistributedModel(wrapper, spec=...)``
backward-compat surface. The full end-to-end ``trace_and_validate``
run requires CUDA + spawn and lives in
``test_validate_cuda.py`` (CUDA-gated, runs on the user's GPU box).
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.distributed._core.spec import DistributionSpec
from nvalchemi.distributed._core.storage_policy import (
    HaloStoragePolicy,
    PlainShard,
)
from nvalchemi.distributed.spec import SPEC_MPNN_HALO, MLIPSpec
from nvalchemi.distributed.validate import Attempt
from nvalchemi.distributed.validate.autofix import (
    _next_fix_candidate,
    _rule_drop_extra_all_reduce,
    _rule_halo_to_local,
    _spec_signature,
)
from nvalchemi.distributed.validate.halo_diagnostics import (
    _check_halo_completeness,
    _partition_health,
)
from nvalchemi.distributed.validate.layer_diagnostics import attach_layer_hooks


# ======================================================================
# merged from test_validate.py
# ======================================================================
def _spec(
    *,
    storage: str = "halo",
    scatter: str = "halo_correction",
    gather: str = "halo_read",
    system_reductions: bool = False,
    owned_only_outputs=frozenset(),
    all_reduce_outputs=frozenset(),
    custom_ops=(),
) -> MLIPSpec:
    """Test-private helper: build an MLIPSpec from convenient
    ``storage``/``scatter``/``gather`` literal args. Avoids spelling out the
    full ``DistributionSpec(policy=HaloStoragePolicy/PlainShard(...))`` nesting
    in every test case. Production code uses the canonical form
    ``MLIPSpec(distribution=DistributionSpec(policy=HaloStoragePolicy()), ...)``.
    """
    if storage == "halo":
        policy = HaloStoragePolicy(scatter_mode=scatter, gather_mode=gather)
    elif storage == "sharded":
        policy = PlainShard()
    else:
        policy = None
    return MLIPSpec(
        distribution=DistributionSpec(
            policy=policy,
            custom_ops=custom_ops,
        ),
        system_reductions=system_reductions,
        owned_only_outputs=frozenset(owned_only_outputs),
        all_reduce_outputs=frozenset(all_reduce_outputs),
    )
def _attempt(
    spec: MLIPSpec,
    *,
    passed: bool = False,
    abs_diff: dict[str, float] | None = None,
    rel_diff: dict[str, float] | None = None,
    handler_counts: dict[str, int] | None = None,
    rationale: str = "test",
) -> Attempt:
    return Attempt(
        spec=spec,
        rationale=rationale,
        passed=passed,
        max_abs_diff=abs_diff or {},
        max_rel_diff=rel_diff or {},
        handler_counts=handler_counts or {},
    )
class TestSpecSerialization:
    def test_preset_roundtrip(self):
        d = SPEC_MPNN_HALO.to_dict()
        restored = MLIPSpec.from_dict(d)
        assert restored == SPEC_MPNN_HALO

    def test_save_load_via_disk(self, tmp_path):
        path = tmp_path / "mpnn.json"
        SPEC_MPNN_HALO.save(path)
        restored = MLIPSpec.load(path)
        assert restored == SPEC_MPNN_HALO

    def test_owned_only_outputs_roundtrip(self):
        spec = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
            owned_only_outputs=frozenset({"forces", "stress"}),
        )
        restored = MLIPSpec.from_dict(spec.to_dict())
        assert restored.owned_only_outputs == spec.owned_only_outputs

    def test_all_reduce_outputs_roundtrip(self):
        spec = _spec(
            storage="sharded",
            scatter="distributed",
            gather="distributed",
            all_reduce_outputs=frozenset({"energy"}),
        )
        restored = MLIPSpec.from_dict(spec.to_dict())
        assert restored.all_reduce_outputs == spec.all_reduce_outputs

    def test_real_wrapper_with_custom_ops_roundtrip(self):
        """A wrapper with real ``custom_ops`` (PME has 4) should
        serialize through op-qualname encoding and reload to the same
        op handles."""
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=5.0)
        spec = w.distribution_spec()
        d = spec.to_dict()
        spec2 = MLIPSpec.from_dict(d)

        assert len(spec2.distribution.custom_ops) == len(spec.distribution.custom_ops)
        for o1, o2 in zip(spec.distribution.custom_ops, spec2.distribution.custom_ops):
            assert o1.op is o2.op  # identical op handle
            assert o1.gather_inputs == o2.gather_inputs
            assert o1.scatter_outputs == o2.scatter_outputs
            assert o1.owned_slice_inputs == o2.owned_slice_inputs
            assert o1.all_reduce_outputs == o2.all_reduce_outputs
class TestRuleHaloToLocal:
    """``halo_correction`` → ``local`` is the UMA fix's signature: a
    halo-unaware backbone whose edge_index covers the full graph
    causes halo_reverse to double-count cross-rank contributions."""

    def test_proposes_local_when_halo_correction_with_diff(self):
        spec = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
        )
        last = _attempt(
            spec,
            abs_diff={"energy": 422.0},
            handler_counts={"halo_scatter_correction": 4},
        )
        candidate = _rule_halo_to_local(spec, last)
        assert candidate is not None
        assert candidate.distribution.policy.scatter_mode == "local"
        # Other fields preserved
        assert isinstance(candidate.distribution.policy, HaloStoragePolicy)
        assert candidate.distribution.policy.gather_mode == "halo_read"

    def test_skips_when_already_local(self):
        spec = _spec(storage="halo", scatter="local", gather="halo_read")
        last = _attempt(
            spec,
            abs_diff={"energy": 422.0},
            handler_counts={"halo_scatter_correction": 0},
        )
        assert _rule_halo_to_local(spec, last) is None

    def test_skips_when_no_halo_correction_fired(self):
        """If halo_scatter_correction never fired, the rule's
        diagnostic premise doesn't hold — don't propose a fix."""
        spec = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
        )
        last = _attempt(
            spec,
            abs_diff={"energy": 422.0},
            handler_counts={"per_system_reduce": 2},  # nothing about halo
        )
        assert _rule_halo_to_local(spec, last) is None

    def test_skips_within_noise(self):
        """Below-noise diffs aren't a divergence; rule shouldn't fire."""
        spec = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
        )
        last = _attempt(
            spec,
            abs_diff={"energy": 1e-9},
            handler_counts={"halo_scatter_correction": 4},
        )
        assert _rule_halo_to_local(spec, last) is None
class TestRuleDropExtraAllReduce:
    """If a key declared in ``all_reduce_outputs`` is being reduced
    twice (wrapper internals already replicate it), the rule drops
    the key."""

    def test_drops_key_with_huge_relative_diff(self):
        spec = _spec(
            storage="sharded",
            scatter="distributed",
            gather="distributed",
            all_reduce_outputs=frozenset({"energy"}),
        )
        last = _attempt(
            spec,
            abs_diff={"energy": 1057.78},
            rel_diff={"energy": 1.0},  # ~100% off → likely 2× duplicated
        )
        candidate = _rule_drop_extra_all_reduce(spec, last)
        assert candidate is not None
        assert candidate.all_reduce_outputs == frozenset()

    def test_skips_when_no_all_reduce_outputs(self):
        spec = _spec(storage="halo", scatter="halo_correction", gather="halo_read")
        last = _attempt(spec, abs_diff={"energy": 100.0}, rel_diff={"energy": 1.0})
        assert _rule_drop_extra_all_reduce(spec, last) is None

    def test_keeps_keys_with_small_relative_diff(self):
        """Tiny rel_diff doesn't look like a duplicated reduction."""
        spec = _spec(
            storage="sharded",
            scatter="distributed",
            gather="distributed",
            all_reduce_outputs=frozenset({"energy"}),
        )
        last = _attempt(spec, abs_diff={"energy": 1e-3}, rel_diff={"energy": 1e-6})
        assert _rule_drop_extra_all_reduce(spec, last) is None
class TestNextFixCandidate:
    """Top-level dispatcher: tries rules in order, dedups against
    already-attempted specs."""

    def test_returns_first_matching_rule(self):
        spec = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
        )
        attempts = [
            _attempt(
                spec,
                abs_diff={"energy": 422.0},
                handler_counts={"halo_scatter_correction": 4},
            )
        ]
        result = _next_fix_candidate(spec, attempts)
        assert result is not None
        candidate, rationale = result
        assert candidate.distribution.policy.scatter_mode == "local"
        assert "halo_correction → local" in rationale

    def test_returns_none_when_nothing_matches(self):
        spec = _spec(storage="halo", scatter="local", gather="halo_read")
        # No halo_correction firing, no all_reduce_outputs → no rule applies.
        attempts = [_attempt(spec, abs_diff={"energy": 1.0})]
        assert _next_fix_candidate(spec, attempts) is None

    def test_dedups_already_attempted_specs(self):
        """If the only matching rule produces a spec we've already
        tried, return None (don't loop)."""
        original = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
        )
        already_tried = _spec(storage="halo", scatter="local", gather="halo_read")
        # Pretend we tried both halo_correction and local.
        attempts = [
            _attempt(
                original,
                abs_diff={"energy": 422.0},
                handler_counts={"halo_scatter_correction": 4},
            ),
            _attempt(already_tried, abs_diff={"energy": 100.0}),
        ]
        # Asking for next from `original` would propose `local` again —
        # but that signature is in attempts, so dedup returns None.
        assert _next_fix_candidate(original, attempts) is None
class TestSpecSignature:
    def test_same_spec_same_signature(self):
        a = _spec(storage="halo", scatter="local", gather="halo_read")
        b = _spec(storage="halo", scatter="local", gather="halo_read")
        assert _spec_signature(a) == _spec_signature(b)

    def test_different_scatter_different_signature(self):
        a = _spec(storage="halo", scatter="halo_correction", gather="halo_read")
        b = _spec(storage="halo", scatter="local", gather="halo_read")
        assert _spec_signature(a) != _spec_signature(b)

    def test_owned_only_outputs_in_signature(self):
        a = _spec(
            storage="halo",
            scatter="halo_correction",
            gather="halo_read",
            owned_only_outputs=frozenset({"forces"}),
        )
        b = _spec(storage="halo", scatter="halo_correction", gather="halo_read")
        assert _spec_signature(a) != _spec_signature(b)
class TestDistributedModelSpecArg:
    """Verify the new ``spec=`` kwarg works alongside the wrapper-property
    fallback."""

    def test_explicit_spec_takes_precedence(self):
        from nvalchemi.distributed.config import DomainConfig
        from nvalchemi.distributed.distributed_model import DistributedModel
        from nvalchemi.models.pme import PMEModelWrapper

        wrapper = PMEModelWrapper(cutoff=5.0)
        custom_spec = _spec(
            storage="halo",
            scatter="local",  # deliberately differs from wrapper's default
            gather="halo_read",
        )
        cfg = DomainConfig(cutoff=5.0)

        dm = DistributedModel(wrapper, cfg, spec=custom_spec)
        assert dm._spec is custom_spec
        assert dm._spec.distribution.policy.scatter_mode == "local"
        # Wrapper's own property is unchanged.
        assert wrapper.distribution_spec().distribution.policy.scatter_mode == "halo_correction"

    def test_falls_back_to_wrapper_property_when_no_spec(self):
        from nvalchemi.distributed.config import DomainConfig
        from nvalchemi.distributed.distributed_model import DistributedModel
        from nvalchemi.models.pme import PMEModelWrapper

        wrapper = PMEModelWrapper(cutoff=5.0)
        cfg = DomainConfig(cutoff=5.0)

        dm = DistributedModel(wrapper, cfg)  # no spec kwarg
        assert dm._spec is not None
        assert isinstance(dm._spec.distribution.policy, HaloStoragePolicy)
        assert dm._spec.distribution.policy.scatter_mode == "halo_correction"
class TestImportSurface:
    def test_can_import_trace_and_validate_without_cuda(self):
        """The validator's import must not actually need CUDA — the
        CUDA check happens at call time, not import time, so users on
        CPU-only boxes can still import the module (e.g. for
        type-checking or to inspect the report dataclasses)."""
        from nvalchemi.distributed.validate import TraceReport, trace_and_validate

        assert callable(trace_and_validate)
        # TraceReport carries the documented fields.
        fields = TraceReport.__dataclass_fields__
        assert {"ok", "spec", "attempts", "next_action"}.issubset(fields)

    def test_trace_and_validate_raises_on_no_cuda(self):
        """When CUDA isn't available, the function fails fast with a
        clear message rather than silently falling back to CPU."""
        if torch.cuda.is_available():
            pytest.skip("requires CUDA-less env to test the no-CUDA error path")

        from nvalchemi.distributed.validate import trace_and_validate

        with pytest.raises(RuntimeError, match="CUDA"):
            trace_and_validate(lambda: None, None)

# ======================================================================
# merged from test_validate_diagnostics.py
# ======================================================================
class TestHaloCompletenessGating:
    def _ref(self):
        return {
            "positions": torch.zeros(4, 3),
            "per_atom_count": torch.tensor([3, 3, 3, 3]),
            "total_valid": 12,
        }

    def test_all_empty_summaries_returns_none(self):
        """Sharded-storage specs capture no halo summary (no padded NL). The
        check must say "not applicable" rather than inventing a 0-vs-12
        INCOMPLETE mismatch (the false positive seen on AIMNet2)."""
        verdict = _check_halo_completeness(self._ref(), {0: {}, 1: {}})
        assert verdict is None

    def test_no_summaries_at_all_returns_none(self):
        assert _check_halo_completeness(self._ref(), {}) is None

    def test_empty_ref_returns_none(self):
        assert _check_halo_completeness({}, {0: {"total_owned_valid": 5}}) is None
class TestPartitionHealth:
    def _ref(self, n_global):
        return {"per_atom_count": torch.ones(n_global)}

    def test_healthy_partition_non_degenerate(self):
        # n_global=10; each rank owns 5, borrows 2 halo, leaves 3 remote.
        summaries = {
            0: {"n_owned": 5, "n_padded": 7},
            1: {"n_owned": 5, "n_padded": 7},
        }
        v = _partition_health(self._ref(10), summaries)
        assert v["healthy"] is True
        assert v["degenerate"] == []
        assert v["per_rank"][0] == {"owned": 5, "halo": 2, "remote": 3}

    def test_zero_halo_flagged(self):
        # n_padded == n_owned → no halo atoms → no cross-rank dependency.
        v = _partition_health(self._ref(10), {0: {"n_owned": 5, "n_padded": 5}})
        assert v["healthy"] is False
        assert any("0 halo atoms" in m for m in v["degenerate"])

    def test_zero_remote_flagged(self):
        # n_padded == n_global → rank sees every atom → trivial geometry.
        v = _partition_health(self._ref(10), {0: {"n_owned": 5, "n_padded": 10}})
        assert v["healthy"] is False
        assert any("0 remote atoms" in m for m in v["degenerate"])

    def test_none_when_no_summaries_or_ref(self):
        assert _partition_health(self._ref(10), {}) is None
        assert _partition_health({}, {0: {"n_owned": 5, "n_padded": 7}}) is None
class TestLayerHookScriptModuleRobustness:
    def test_scripted_submodule_is_skipped_not_fatal(self):
        """A model with a TorchScript submodule (MACE's blocks are
        ``RecursiveScriptModule``) must not abort hook registration —
        ``register_forward_hook`` raises on ScriptModules."""

        @torch.jit.script
        def _scripted_add(x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

        class Scripted(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1.0

        class Parent(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.plain = torch.nn.Linear(3, 3)
                self.scripted = torch.jit.script(Scripted())

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.scripted(self.plain(x))

        model = Parent()
        records: list = []
        # Must not raise even though ``scripted`` rejects forward hooks.
        handles = attach_layer_hooks(model, records)
        # The plain Linear got hooked; the scripted block was skipped.
        assert len(handles) >= 1
        model(torch.randn(2, 3))
        hooked_names = {r[0] for r in records}
        assert any("plain" in n for n in hooked_names)
        for h in handles:
            h.remove()
class TestWorkerErrorTranslators:
    def test_severed_autograd_graph_diagnosis(self):
        from nvalchemi.distributed.validate import _translate_worker_error

        err = (
            "RuntimeError: element 0 of tensors does not require grad and "
            "does not have a grad_fn"
        )
        hint = _translate_worker_error(
            err, {"per_system_reduce": 2, "halo_scatter_correction": 2}
        )
        assert hint is not None
        assert "severed the autograd graph" in hint
        assert "per_system_reduce" in hint  # echoes the firings

    def test_no_false_translation_on_unrelated_error(self):
        from nvalchemi.distributed.validate import _translate_worker_error

        assert _translate_worker_error("CUDA out of memory", {}) is None
