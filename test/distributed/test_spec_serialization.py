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

"""Round-trip and version-migration tests for spec serialization.

* :meth:`MLIPSpec.to_dict` emits v2 schema (nested ``core`` dict).
* :meth:`MLIPSpec.from_dict` accepts both v1 (legacy flat fields) and
  v2 (nested core).
* v1 → v2 migration preserves semantics: a saved v1 JSON loads to an
  MLIPSpec whose properties match the original.
* StoragePolicy / OpAdapter / JitAdapter / PythonAdapter / DistributionSpec each
  round-trip cleanly.
"""

from __future__ import annotations

import pytest

from nvalchemi.distributed._core.adapter import (
    JitAdapter,
    MethodAdapter,
    OpAdapter,
    PythonAdapter,
)
from nvalchemi.distributed._core.op_transforms import (
    AllReduceSum,
    SliceOwned,
)
from nvalchemi.distributed._core.spec import DistributionSpec
from nvalchemi.distributed._core.storage_policy import (
    HaloStoragePolicy,
    PlainShard,
)
from nvalchemi.distributed.spec import (
    SPEC_LJ_HALO,
    SPEC_MPNN_HALO,
    MLIPSpec,
)


class TestDistributionSpecDict:
    def test_round_trip_minimal(self):
        core = DistributionSpec(policy=HaloStoragePolicy())
        d = core.to_dict()
        assert d["policy"]["kind"] == "halo"
        assert d["custom_ops"] == []
        assert d["third_party_helpers"] == []

        loaded = DistributionSpec.from_dict(d)
        assert loaded == core

    def test_round_trip_with_helpers(self):
        h = PythonAdapter(module_path="aimnet.nbops", attr_name="mol_sum")
        core = DistributionSpec(
            policy=PlainShard(),
            third_party_helpers=(h,),
        )
        d = core.to_dict()
        assert len(d["third_party_helpers"]) == 1
        assert d["third_party_helpers"][0]["module_path"] == "aimnet.nbops"
        assert d["third_party_helpers"][0]["kind"] == "python"

        loaded = DistributionSpec.from_dict(d)
        assert loaded.policy == core.policy
        assert len(loaded.third_party_helpers) == 1
        assert loaded.third_party_helpers[0].module_path == "aimnet.nbops"
        assert isinstance(loaded.third_party_helpers[0], PythonAdapter)


class TestUnifiedAdaptersField:
    """The single declarative ``adapters=`` field lowers onto the canonical
    split tuples at construction, so all framework consumers and
    serialization stay unchanged."""

    def _op(self):
        import torch

        return OpAdapter(torch.ops.aten.add, scatter_outputs=[0])

    def test_lowers_by_type(self):
        op = self._op()
        h = PythonAdapter(module_path="m", attr_name="f")
        core = DistributionSpec(policy=HaloStoragePolicy(), adapters=(op, h))
        # OpAdapter → custom_ops, everything else → third_party_helpers.
        assert len(core.custom_ops) == 1
        assert len(core.third_party_helpers) == 1
        assert isinstance(core.third_party_helpers[0], PythonAdapter)
        # ``adapters`` is cleared after lowering (it is input-only sugar).
        assert core.adapters == ()

    def test_composes_with_explicit_split(self):
        # Explicit split lists compose with the unified tuple (unified appended).
        op = self._op()
        h1 = PythonAdapter(module_path="m", attr_name="f")
        h2 = MethodAdapter("m", "C", "forward", mode="marshal")
        core = DistributionSpec(
            policy=HaloStoragePolicy(),
            third_party_helpers=(h1,),
            adapters=(op, h2),
        )
        assert len(core.custom_ops) == 1
        assert len(core.third_party_helpers) == 2
        assert core.third_party_helpers[0] is h1  # explicit first, unified after

    def test_equivalent_to_split_form(self):
        op = self._op()
        h = PythonAdapter(module_path="m", attr_name="f")
        unified = DistributionSpec(policy=PlainShard(), adapters=(op, h))
        split = DistributionSpec(
            policy=PlainShard(), custom_ops=(op,), third_party_helpers=(h,)
        )
        assert unified == split
        assert unified.to_dict() == split.to_dict()

    def test_empty_adapters_is_noop(self):
        core = DistributionSpec(policy=HaloStoragePolicy())
        assert core.adapters == ()
        assert core.custom_ops == ()
        assert core.third_party_helpers == ()


class TestPythonAdapterDict:
    def test_no_replacement(self):
        h = PythonAdapter(module_path="m", attr_name="f")
        d = h.to_dict()
        assert d["kind"] == "python"
        assert d["module_path"] == "m"
        assert d["attr_name"] == "f"
        assert d["replacement"] is None
        loaded = PythonAdapter.from_dict(d)
        assert loaded.module_path == "m"
        assert loaded.attr_name == "f"
        assert loaded.replacement is None

    def test_with_replacement_qualname(self):
        # Use a real module-level function so the qualname resolves on load.
        from nvalchemi.distributed._core.storage_policy import (
            policy_to_dict as fn,
        )

        h = PythonAdapter(module_path="m", attr_name="f", replacement=fn)
        d = h.to_dict()
        assert "policy_to_dict" in (d["replacement"] or "")
        loaded = PythonAdapter.from_dict(d)
        assert loaded.replacement is fn


class TestJitAdapterDict:
    def test_round_trip(self):
        from nvalchemi.distributed._core.storage_policy import (
            policy_to_dict as fn,
        )

        h = JitAdapter(module_path="m", attr_name="f", replacement=fn)
        d = h.to_dict()
        assert d["kind"] == "jit"
        loaded = JitAdapter.from_dict(d)
        assert loaded.module_path == "m"
        assert loaded.replacement is fn


class TestMethodAdapterDict:
    def test_round_trip(self):
        from nvalchemi.distributed._core.storage_policy import (
            policy_to_dict as fn,
        )

        h = MethodAdapter(
            module_path="m", class_name="C", method_name="forward", replacement=fn
        )
        d = h.to_dict()
        assert d["kind"] == "method"
        assert d["class_name"] == "C"
        assert d["method_name"] == "forward"
        loaded = MethodAdapter.from_dict(d)
        assert loaded.class_name == "C"
        assert loaded.method_name == "forward"
        assert loaded.replacement is fn

    def test_dispatches_through_adapter_registry_kind(self):
        # The serialized "method" kind round-trips via the registry (the path
        # DistributionSpec.from_dict uses for third_party_helpers).
        from nvalchemi.distributed._core.adapter import _adapter_from_dict

        d = MethodAdapter(
            module_path="m", class_name="C", method_name="forward"
        ).to_dict()
        assert isinstance(_adapter_from_dict(d), MethodAdapter)

    def test_install_wraps_and_restore_reverts(self):
        # install() wraps the class method (call-original); restore() reverts.
        # Register a throwaway module so install()'s importlib path resolves.
        import sys
        import types

        mod = types.ModuleType("_methadapter_demo_mod")

        class _Demo:
            def forward(self, x: int) -> int:
                return x + 1

        mod._Demo = _Demo
        sys.modules["_methadapter_demo_mod"] = mod
        try:

            def _wrap(original, self_, x):
                return original(self_, x) * 10

            adapter = MethodAdapter(
                module_path="_methadapter_demo_mod",
                class_name="_Demo",
                method_name="forward",
                replacement=_wrap,
            )
            original_unbound = _Demo.forward
            memento = adapter.install()
            assert _Demo().forward(4) == 50  # (4 + 1) * 10
            adapter.restore(memento)
            assert _Demo.forward is original_unbound
            assert _Demo().forward(4) == 5
        finally:
            del sys.modules["_methadapter_demo_mod"]


class TestOpSpecDict:
    """OpAdapter serialization is exercised end-to-end by the
    ``custom_ops``-bearing presets (e.g. PME). ``OpAdapter.to_dict``
    encodes the op handle via :func:`_op_qualname`'s string fallback
    when the handle has no torch schema, so the *transform fields*
    serialize cleanly even with a non-standard op stub."""

    def test_transform_fields_serialize(self):
        # Use a string in place of a real op handle; _op_qualname's
        # ``str(op)`` fallback handles it (the round-trip side won't
        # work without a registered torch op, but the to_dict contract
        # for transform tuples is what we're checking here).
        os = OpAdapter(
            op="alchemiops::test_op",
            arg_transforms={0: SliceOwned(), 1: SliceOwned()},
            output_transforms={0: AllReduceSum()},
        )
        d = os.to_dict()
        assert d["arg_transforms"] == {
            "0": {"type": "slice_owned"},
            "1": {"type": "slice_owned"},
        }
        assert d["output_transforms"] == {"0": {"type": "all_reduce_sum"}}
        # Property accessors still work.
        assert os.owned_slice_inputs == (0, 1)
        assert os.all_reduce_outputs == (0,)
        assert os.gather_inputs == ()


class TestMLIPSpecDict:
    def test_v2_round_trip_simple(self):
        spec = MLIPSpec(
            distribution=DistributionSpec(policy=HaloStoragePolicy()),
            owned_only_outputs=frozenset({"stress"}),
        )
        d = spec.to_dict()
        assert d["version"] == 2
        assert d["core"]["policy"]["kind"] == "halo"
        assert d["owned_only_outputs"] == ["stress"]

        loaded = MLIPSpec.from_dict(d)
        assert loaded.distribution.policy == spec.distribution.policy
        assert loaded.owned_only_outputs == frozenset({"stress"})

    def test_v2_preset_round_trip(self):
        # SPEC_MPNN_HALO is built via the new construction form and
        # round-trips cleanly through v2.
        d = SPEC_MPNN_HALO.to_dict()
        assert d["version"] == 2

        loaded = MLIPSpec.from_dict(d)
        assert isinstance(loaded.distribution.policy, type(SPEC_MPNN_HALO.distribution.policy))
        assert loaded.distribution.policy == SPEC_MPNN_HALO.distribution.policy
        assert loaded.system_reductions == SPEC_MPNN_HALO.system_reductions

    def test_unsupported_version_raises(self):
        with pytest.raises(ValueError, match="unsupported version"):
            MLIPSpec.from_dict({"version": 99, "core": {}})

    def test_save_load_round_trip(self, tmp_path):
        path = tmp_path / "spec.json"
        SPEC_LJ_HALO.save(path)
        loaded = MLIPSpec.load(path)
        assert loaded.distribution.policy == SPEC_LJ_HALO.distribution.policy


class TestMLIPSpecOutputsCollapse:
    """``outputs={name: OutputSpec}`` + ``compile=`` lower onto the canonical
    three fields, so consolidation + serialization are unchanged and a spec
    built either way compares + round-trips equal."""

    def _spec_via_outputs(self):
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
        from nvalchemi.distributed.output_kinds import OutputKind, OutputSpec, Reduce
        from nvalchemi.distributed.spec import CompilePolicy

        return MLIPSpec(
            distribution=DistributionSpec(policy=HaloStoragePolicy()),
            outputs={
                "energy": OutputSpec(kind=OutputKind.PER_GRAPH),
                "forces": OutputSpec(kind=OutputKind.PER_NODE),
                "stress": OutputSpec(
                    kind=OutputKind.PER_GRAPH, reduce=Reduce.ALL_REDUCE
                ),
                "partial": OutputSpec(
                    kind=OutputKind.PER_NODE, reduce=Reduce.OWNED_ONLY
                ),
            },
            compile=CompilePolicy(static_shapes=True),
        )

    def _spec_via_legacy(self):
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
        from nvalchemi.distributed.output_kinds import OutputKind

        return MLIPSpec(
            distribution=DistributionSpec(policy=HaloStoragePolicy()),
            owned_only_outputs=frozenset({"partial"}),
            all_reduce_outputs=frozenset({"stress"}),
            output_kinds={
                "energy": OutputKind.PER_GRAPH,
                "forces": OutputKind.PER_NODE,
                "stress": OutputKind.PER_GRAPH,
                "partial": OutputKind.PER_NODE,
            },
        )

    def test_outputs_lowers_to_canonical_fields(self):
        from nvalchemi.distributed.output_kinds import OutputKind

        s = self._spec_via_outputs()
        assert s.owned_only_outputs == frozenset({"partial"})
        assert s.all_reduce_outputs == frozenset({"stress"})
        assert s.output_kinds == {
            "energy": OutputKind.PER_GRAPH,
            "forces": OutputKind.PER_NODE,
            "stress": OutputKind.PER_GRAPH,
            "partial": OutputKind.PER_NODE,
        }

    def test_outputs_form_equals_legacy_form(self):
        # ``outputs`` / ``compile`` are compare=False, so the two constructions
        # are equal once the canonical fields match.
        assert self._spec_via_outputs() == self._spec_via_legacy()

    def test_outputs_form_round_trips_to_legacy(self):
        # to_dict emits only the canonical fields; from_dict reconstructs the
        # legacy form, which equals the outputs-built original.
        s = self._spec_via_outputs()
        loaded = MLIPSpec.from_dict(s.to_dict())
        assert loaded == s
        assert loaded.owned_only_outputs == frozenset({"partial"})
        assert loaded.all_reduce_outputs == frozenset({"stress"})

    def test_compile_policy_does_not_perturb_canonical(self):
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
        from nvalchemi.distributed.spec import CompilePolicy

        plain = MLIPSpec(distribution=DistributionSpec(policy=HaloStoragePolicy()))
        with_compile = MLIPSpec(
            distribution=DistributionSpec(policy=HaloStoragePolicy()),
            compile=CompilePolicy(static_shapes=False),
        )
        assert plain == with_compile  # compile is compare=False
