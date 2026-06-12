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
"""Tests for :mod:`nvalchemi.training._spec`."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from nvalchemi.training._spec import (
    _TYPE_SERIALIZERS,
    BaseSpec,
    _check_no_positional_only,
    _dtype_deserialize,
    _import_cls,
    create_model_spec,
    create_model_spec_from_json,
    register_type_serializer,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures used across test classes
# ---------------------------------------------------------------------------


class _KwOnly:
    """Class with ``**kwargs`` in its signature so unknown kwargs are allowed."""

    def __init__(self, a: int = 1, **kwargs: Any) -> None:
        self.a = a
        self.kwargs = kwargs


class _AnnotatedDtype:
    """Class whose ``dtype`` parameter is annotated, exercising the registry path."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype


class _WrapsModule(nn.Module):
    """Class whose ``child`` param is annotated as ``nn.Module`` for nested-build."""

    def __init__(self, child: nn.Module, scale: float = 1.0) -> None:
        super().__init__()
        self.child = child
        self.scale = scale


class _WithDevice:
    """Class whose ``device`` parameter is annotated."""

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device


class _WithTensor:
    """Class holding a single ``torch.Tensor`` parameter."""

    def __init__(self, weights: torch.Tensor) -> None:
        self.weights = weights


class _WithTensorLinspace:
    """Class holding a ``torch.Tensor`` named ``buf`` (distinct from _WithTensor)."""

    def __init__(self, buf: torch.Tensor) -> None:
        self.buf = buf


class _WithStringPath:
    """Class holding a string that may look like an importable class path."""

    def __init__(self, label: str) -> None:
        self.label = label


class _WithUnannotatedStringPath:
    """Class holding an unannotated string-like parameter."""

    def __init__(self, label="") -> None:
        self.label = label


class _WithUnannotatedClassField:
    """Class holding an unannotated class-valued parameter."""

    def __init__(self, plugin_cls=None) -> None:
        self.plugin_cls = plugin_cls


class _WithClassFields:
    """Class holding parametrized and optional class annotations."""

    def __init__(
        self,
        optimizer_cls: type[torch.optim.Optimizer],
        scheduler_cls: type | None = None,
    ) -> None:
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls


class _FactoryBuilt:
    """Object reconstructed through an importable classmethod factory."""

    def __init__(self, value: int, device: torch.device, dtype: torch.dtype) -> None:
        self.value = value
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_config(
        cls,
        value: int,
        *,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype | None = None,
    ) -> "_FactoryBuilt":
        return cls(value=value, device=device, dtype=dtype or torch.float32)


class _TupleWrapsModules(nn.Module):
    """Class whose tuple field is populated from nested specs."""

    def __init__(self, children: tuple[nn.Module, ...]) -> None:
        super().__init__()
        self.children_tuple = children


def _make_positional_only_cls() -> type:
    """Return a class with a positional-only ``__init__`` parameter.

    Using ``exec`` keeps the ``/`` syntax out of the top-level file body
    while still producing a real class with positional-only params.
    """
    ns: dict[str, Any] = {}
    src = (
        "class _PosOnly:\n"
        "    def __init__(self, x, /, y=0):\n"
        "        self.x = x\n"
        "        self.y = y\n"
    )
    exec(src, ns)  # noqa: S102 — deliberately constructs positional-only signature
    return ns["_PosOnly"]


class Outer:
    """Module-level host class for nested-class qualname resolution tests."""

    class Inner:
        """Nested class used to verify ``_import_cls`` handles nested qualnames."""


# Prototype `main()` fixtures — used by TestFullPrototypeScenario. Defined at
# module level (not inside a test class) so their __qualname__ stays clean.


class MyBlock(nn.Module):
    """Residual MLP block mirroring the example_spec.py prototype."""

    def __init__(
        self,
        hidden_dim: int,
        projection_dims: list[int],
        dropout_p: float,
        activation: nn.Module,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)
        dims = [hidden_dim, *projection_dims, hidden_dim]
        self.projections = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout+activation+projection residually."""
        residual = x
        h = x
        for proj in self.projections:
            h = self.dropout(self.activation(proj(h)))
        return h + residual


class MyMLIP(nn.Module):
    """Toy MLIP with a residual block, feature-scale buffer, and output head."""

    def __init__(
        self,
        hidden_dim: int,
        cutoff: float,
        block: MyBlock,
        input_activation: nn.Module,
        feature_scale: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if feature_scale.ndim != 1 or feature_scale.shape[0] != hidden_dim:
            raise ValueError(
                f"feature_scale must be 1D of length {hidden_dim}, "
                f"got shape {tuple(feature_scale.shape)}"
            )
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.input_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.input_activation = input_activation
        self.block = block
        self.output_proj = nn.Linear(hidden_dim, 1, dtype=dtype)
        self.register_buffer("feature_scale", feature_scale.to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the prototype forward graph."""
        h = self.input_activation(self.input_proj(x))
        h = h * self.feature_scale
        h = self.block(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestClsPathResolution:
    """Resolution of dotted ``cls_path`` strings back to class objects."""

    def test_resolves_simple(self) -> None:
        assert _import_cls("torch.nn.Linear") is nn.Linear

    def test_resolves_deep_module_path(self) -> None:
        # Multi-level module path: greedy prefix matching must import
        # the longest valid module and walk the remainder as attributes.
        assert (
            _import_cls("torch.nn.modules.activation.SiLU")
            is nn.modules.activation.SiLU
        )

    def test_resolves_nested_qualname(self) -> None:
        """Resolve a nested-class qualname via greedy module-prefix matching."""
        # ``Outer`` is defined at module scope in this test file, so the
        # importable prefix is the test module and ``Outer.Inner`` is an
        # attribute chain on it.
        cls = _import_cls(f"{Outer.__module__}.Outer.Inner")
        assert cls is Outer.Inner

    def test_raises_on_invalid_module(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            _import_cls("definitely_not_a_real_module.SomeCls")

    def test_raises_on_invalid_attr(self) -> None:
        with pytest.raises(AttributeError):
            _import_cls("torch.nn.ThisAttrDoesNotExist")

    def test_raises_on_non_class_target(self) -> None:
        # torch.tensor is a function, not a class.
        with pytest.raises(TypeError, match="non-class"):
            _import_cls("torch.tensor")


class TestTypeSerializerRegistry:
    """Round-trip behavior and security of the type-serializer registry."""

    def test_register_replaces_existing(self) -> None:
        original = _TYPE_SERIALIZERS[torch.dtype]
        sentinel_ser = lambda d: "sentinel"  # noqa: E731
        sentinel_deser = lambda s: torch.float32  # noqa: E731
        try:
            register_type_serializer(torch.dtype, sentinel_ser, sentinel_deser)
            ser, deser = _TYPE_SERIALIZERS[torch.dtype]
            assert ser is sentinel_ser
            assert deser is sentinel_deser
        finally:
            register_type_serializer(torch.dtype, original[0], original[1])
        # Restored:
        assert _TYPE_SERIALIZERS[torch.dtype] == original

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
    def test_dtype_roundtrip(self, dtype: torch.dtype) -> None:
        spec = create_model_spec(_AnnotatedDtype, dtype=dtype)
        dumped = json.loads(spec.model_dump_json())
        rebuilt = create_model_spec_from_json(dumped)
        assert rebuilt.dtype is dtype

    def test_dtype_raises_on_non_dtype_attr(self) -> None:
        # torch.nn exists but is a module, not a dtype. The isinstance guard
        # in _dtype_deserialize must reject it to block attr-smuggling.
        with pytest.raises(ValueError, match="does not resolve to a torch.dtype"):
            _dtype_deserialize("nn")

    def test_dtype_raises_on_non_string(self) -> None:
        with pytest.raises(TypeError, match="expected str"):
            _dtype_deserialize(42)

    def test_device_roundtrip(self) -> None:
        spec = create_model_spec(_WithDevice, device=torch.device("cpu"))
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert rebuilt.device == torch.device("cpu")

    def test_tensor_roundtrip(self) -> None:
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        spec = create_model_spec(_WithTensor, weights=t)
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert rebuilt.weights.shape == t.shape
        assert rebuilt.weights.dtype == t.dtype
        assert torch.equal(rebuilt.weights, t)

    def test_class_annotations_roundtrip(self) -> None:
        spec = create_model_spec(
            _WithClassFields,
            optimizer_cls=torch.optim.Adam,
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
        )
        dumped = json.loads(spec.model_dump_json())
        assert dumped["optimizer_cls"] == "torch.optim.adam.Adam"
        assert dumped["scheduler_cls"] == "torch.optim.lr_scheduler.StepLR"

        rebuilt = create_model_spec_from_json(dumped)
        assert rebuilt.optimizer_cls is torch.optim.Adam
        assert rebuilt.scheduler_cls is torch.optim.lr_scheduler.StepLR

    def test_optional_class_annotation_roundtrip_none(self) -> None:
        spec = create_model_spec(
            _WithClassFields,
            optimizer_cls=torch.optim.SGD,
            scheduler_cls=None,
        )
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert rebuilt.optimizer_cls is torch.optim.SGD
        assert rebuilt.scheduler_cls is None

    def test_unannotated_class_field_roundtrip(self) -> None:
        spec = create_model_spec(
            _WithUnannotatedClassField,
            plugin_cls=torch.optim.Adam,
        )
        dumped = json.loads(spec.model_dump_json())
        assert dumped["plugin_cls"] == {"__type__": "torch.optim.adam.Adam"}

        rebuilt = create_model_spec_from_json(dumped)
        assert rebuilt.plugin_cls is torch.optim.Adam
        assert rebuilt.build().plugin_cls is torch.optim.Adam


class TestSignatureIntrospection:
    """Signature-level validation helpers."""

    def test_positional_only_rejected(self) -> None:
        cls_ = _make_positional_only_cls()
        with pytest.raises(TypeError, match="positional-only"):
            _check_no_positional_only(cls_)


class TestCreateModelSpec:
    """Construction of a :class:`BaseSpec` via :func:`create_model_spec`."""

    def test_creates_spec_with_cls_path_and_timestamp(self) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        assert spec.cls_path == "torch.nn.modules.linear.Linear"
        # timestamp: ISO-8601, parses
        from datetime import datetime

        parsed = datetime.fromisoformat(spec.timestamp)
        assert parsed.tzinfo is not None

    def test_rejects_unknown_kwarg(self) -> None:
        with pytest.raises(TypeError, match="Unknown kwargs"):
            create_model_spec(nn.Linear, in_features=4, out_features=2, bogus=1)

    def test_accepts_arbitrary_kwargs_with_var_keyword(self) -> None:
        # _KwOnly has **kwargs, so `extra_foo` should be accepted.
        spec = create_model_spec(_KwOnly, a=2, extra_foo="hello")
        assert spec.a == 2
        assert spec.extra_foo == "hello"

    def test_nested_spec_composition(self) -> None:
        act_spec = create_model_spec(nn.SiLU)
        spec = create_model_spec(_WrapsModule, child=act_spec, scale=0.5)
        # The nested field should be a BaseSpec.
        assert isinstance(spec.child, BaseSpec)
        assert spec.child.cls_path.endswith(".SiLU")

    def test_tensor_field(self) -> None:
        t = torch.linspace(0.0, 1.0, 5)
        spec = create_model_spec(_WithTensorLinspace, buf=t)
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert torch.equal(rebuilt.buf, t)
        assert rebuilt.buf.dtype == t.dtype
        assert tuple(rebuilt.buf.shape) == tuple(t.shape)

    def test_importable_classmethod_factory(self) -> None:
        spec = create_model_spec(
            _FactoryBuilt.from_config,
            value=7,
            dtype=torch.float64,
        )

        assert spec.cls_path.endswith("._FactoryBuilt.from_config")
        assert spec.accepts_kwarg("device")
        assert not spec.accepts_kwarg("bogus")

        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        built = rebuilt.build(device=torch.device("cuda:0"))

        assert isinstance(built, _FactoryBuilt)
        assert built.value == 7
        assert built.device == torch.device("cuda:0")
        assert built.dtype == torch.float64


class TestCreateModelSpecFromJson:
    """JSON-dict rehydration via :func:`create_model_spec_from_json`."""

    def test_roundtrip_preserves_timestamp(self) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert rebuilt.timestamp == spec.timestamp

    def test_recursive_nested_spec_rehydrated(self) -> None:
        act_spec = create_model_spec(nn.SiLU)
        spec = create_model_spec(_WrapsModule, child=act_spec, scale=0.5)
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert isinstance(rebuilt.child, BaseSpec)
        assert rebuilt.child.cls_path.endswith(".SiLU")
        assert rebuilt.child.timestamp == act_spec.timestamp

    def test_tuple_nested_spec_sequence_rehydrated(self) -> None:
        specs = (
            create_model_spec(nn.SiLU),
            create_model_spec(nn.GELU),
        )
        spec = create_model_spec(_TupleWrapsModules, children=specs)
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))

        assert isinstance(rebuilt.children, tuple)
        assert [child.cls_path for child in rebuilt.children] == [
            "torch.nn.modules.activation.SiLU",
            "torch.nn.modules.activation.GELU",
        ]
        built = rebuilt.build()
        assert isinstance(built.children_tuple, tuple)
        assert [type(child) for child in built.children_tuple] == [nn.SiLU, nn.GELU]

    @pytest.mark.parametrize("missing", ["cls_path", "timestamp"])
    def test_missing_required_field_raises_valueerror(self, missing: str) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        dumped = json.loads(spec.model_dump_json())
        dumped.pop(missing)
        with pytest.raises(ValueError, match=f"missing required field '{missing}'"):
            create_model_spec_from_json(dumped)

    def test_bad_cls_path_raises_valueerror(self) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        dumped = json.loads(spec.model_dump_json())
        dumped["cls_path"] = "definitely.not.a.real.Class"
        with pytest.raises(ValueError, match="Could not resolve cls_path"):
            create_model_spec_from_json(dumped)

    def test_string_field_preserves_importable_class_path(self) -> None:
        spec = create_model_spec(_WithStringPath, label="torch.nn.Linear")
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert rebuilt.label == "torch.nn.Linear"
        assert rebuilt.build().label == "torch.nn.Linear"

    def test_unannotated_string_field_preserves_importable_class_path(self) -> None:
        spec = create_model_spec(_WithUnannotatedStringPath, label="torch.nn.Linear")
        rebuilt = create_model_spec_from_json(json.loads(spec.model_dump_json()))
        assert rebuilt.label == "torch.nn.Linear"
        assert rebuilt.build().label == "torch.nn.Linear"

    def test_unannotated_param_dtype_rehydrates_via_deserializer_probe(
        self,
    ) -> None:
        # nn.Linear.__init__'s dtype parameter is unannotated. Without the
        # eager deserializer probe in create_model_spec_from_json, the str
        # "torch.float32" would pass through untyped and build() would hand
        # a str to torch.empty. The probe rehydrates the string into a
        # torch.dtype before create_model_spec sees it.
        spec = create_model_spec(
            nn.Linear, in_features=4, out_features=2, dtype=torch.float32
        )
        dumped = json.loads(spec.model_dump_json())
        rebuilt = create_model_spec_from_json(dumped)
        assert rebuilt.dtype == torch.float32  # type: ignore[attr-defined]
        model = rebuilt.build()
        assert model.weight.dtype == torch.float32


class TestBaseSpecBuild:
    """Behavior of :meth:`BaseSpec.build`."""

    def test_build_basic_nn_linear(self) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        m = spec.build()
        assert isinstance(m, nn.Linear)
        out = m(torch.randn(3, 4))
        assert out.shape == (3, 2)

    def test_build_strict_is_noop(self) -> None:
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        m = spec.build(strict=True)
        assert isinstance(m, nn.Linear)

    def test_build_accepts_runtime_args_and_kwargs(self) -> None:
        spec = create_model_spec(nn.Linear, in_features=4)
        m = spec.build(out_features=8)
        assert m.in_features == 4
        assert m.out_features == 8

    def test_build_nested_spec_composition(self) -> None:
        # Prototype-style: child module built recursively.
        act_spec = create_model_spec(nn.SiLU)
        spec = create_model_spec(_WrapsModule, child=act_spec, scale=0.5)
        obj = spec.build()
        assert isinstance(obj, _WrapsModule)
        assert isinstance(obj.child, nn.SiLU)
        assert obj.scale == 0.5


class TestFullPrototypeScenario:
    """End-to-end scenario port of ``example_spec.py::main``."""

    def test_prototype_main_roundtrip(self, tmp_path: Path) -> None:
        from nvalchemi.training._checkpoint import load_checkpoint, save_checkpoint

        hidden_dim = 16  # smaller than prototype for test speed
        feature_scale = torch.linspace(0.5, 1.5, hidden_dim)

        spec = create_model_spec(
            MyMLIP,
            hidden_dim=hidden_dim,
            cutoff=5.0,
            block=create_model_spec(
                MyBlock,
                hidden_dim=hidden_dim,
                projection_dims=[24, 24],
                dropout_p=0.1,
                activation=create_model_spec(nn.SiLU),
                dtype=torch.float32,
            ),
            input_activation=create_model_spec(nn.GELU),
            feature_scale=feature_scale,
            dtype=torch.float32,
        )

        torch.manual_seed(0)
        model = spec.build()
        assert isinstance(model, MyMLIP)
        assert isinstance(model.block, MyBlock)
        assert isinstance(model.block.activation, nn.SiLU)
        assert isinstance(model.input_activation, nn.GELU)
        assert torch.equal(model.feature_scale, feature_scale)

        # --- Save + reload via checkpoint I/O -----------------------------
        save_checkpoint(tmp_path, models={"main": (model, spec)})
        save_checkpoint(tmp_path, models={"main": (model, spec)})
        model_dir = tmp_path / "models" / "main"
        assert (model_dir / "spec.json").is_file()
        assert (model_dir / "checkpoints" / "0.pt").is_file()
        assert (model_dir / "checkpoints" / "1.pt").is_file()

        torch.manual_seed(999)  # different seed to prove weights came from ckpt
        result = load_checkpoint(tmp_path)
        reloaded_model, reloaded_spec = result.models["main"]

        sd_orig = model.state_dict()
        sd_new = reloaded_model.state_dict()
        assert sd_orig.keys() == sd_new.keys()
        for k in sd_orig:
            assert torch.equal(sd_orig[k], sd_new[k])
        assert reloaded_spec.timestamp == spec.timestamp
        assert torch.equal(reloaded_spec.feature_scale, feature_scale)

        # Forward pass under eval() must be bit-identical.
        model.eval()
        reloaded_model.eval()
        x = torch.randn(3, hidden_dim)
        with torch.no_grad():
            y_ref = model(x)
            y_new = reloaded_model(x)
        assert torch.equal(y_ref, y_new)

        # --- Optimizer + scheduler round-trip -----------------------------
        opt_spec = create_model_spec(
            torch.optim.AdamW,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=1e-4,
            eps=1e-8,
        )
        sched_spec = create_model_spec(
            torch.optim.lr_scheduler.CosineAnnealingLR,
            T_max=100,
            eta_min=0.0,
        )
        optimizer = opt_spec.build(model.parameters())
        scheduler = sched_spec.build(optimizer)

        opt_reloaded_spec = create_model_spec_from_json(
            json.loads(opt_spec.model_dump_json())
        )
        sched_reloaded_spec = create_model_spec_from_json(
            json.loads(sched_spec.model_dump_json())
        )
        reloaded_optimizer = opt_reloaded_spec.build(reloaded_model.parameters())
        reloaded_scheduler = sched_reloaded_spec.build(reloaded_optimizer)

        for k in ("lr", "betas", "weight_decay", "eps"):
            assert optimizer.param_groups[0][k] == reloaded_optimizer.param_groups[0][k]
        assert scheduler.T_max == reloaded_scheduler.T_max == 100
        assert scheduler.eta_min == reloaded_scheduler.eta_min == 0.0

        # LR trajectory equivalence over 10 steps.
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        for p in reloaded_model.parameters():
            p.grad = torch.zeros_like(p)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.step()
            reloaded_optimizer.step()
            traj_orig, traj_new = [], []
            for _ in range(10):
                scheduler.step()
                reloaded_scheduler.step()
                traj_orig.append(scheduler.get_last_lr()[0])
                traj_new.append(reloaded_scheduler.get_last_lr()[0])
        assert traj_orig == traj_new


class TestSecurityNoPickle:
    """AST-level security invariants for no-pickle serialization modules."""

    _TARGETS = (
        Path(__file__).resolve().parents[2] / "nvalchemi" / "_serialization.py",
        Path(__file__).resolve().parents[2] / "nvalchemi" / "training" / "_spec.py",
        Path(__file__).resolve().parents[2]
        / "nvalchemi"
        / "training"
        / "_checkpoint.py",
    )
    _FORBIDDEN_MODULES = frozenset({"pickle", "cloudpickle", "dill", "marshal"})

    def _tree(self, path: Path) -> ast.AST:
        return ast.parse(path.read_text())

    def test_no_pickle_imports(self) -> None:
        for path in self._TARGETS:
            tree = self._tree(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        assert root not in self._FORBIDDEN_MODULES, (
                            f"{path.name}:{node.lineno} imports forbidden "
                            f"module {alias.name!r}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        continue
                    root = node.module.split(".")[0]
                    assert root not in self._FORBIDDEN_MODULES, (
                        f"{path.name}:{node.lineno} imports from forbidden "
                        f"module {node.module!r}"
                    )

    def test_torch_load_always_weights_only_true(self) -> None:
        for path in self._TARGETS:
            tree = self._tree(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "load"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                ):
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
                assert "weights_only" in kw, (
                    f"{path.name}:{node.lineno} torch.load() missing "
                    f"weights_only= kwarg"
                )
                val = kw["weights_only"]
                assert isinstance(val, ast.Constant) and val.value is True, (
                    f"{path.name}:{node.lineno} torch.load(weights_only=...) "
                    f"must be literal True, got {ast.dump(val)}"
                )

    def test_torch_save_always_state_dict(self) -> None:
        for path in self._TARGETS:
            tree = self._tree(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "save"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "torch"
                ):
                    continue
                assert node.args, (
                    f"{path.name}:{node.lineno} torch.save() called with no args"
                )
                first = node.args[0]
                # Must be either an x.state_dict() call or a variable
                # named "state_dict" (accepted when the call site already
                # resolved the state_dict, e.g. in _save_component).
                is_state_dict_call = (
                    isinstance(first, ast.Call)
                    and isinstance(first.func, ast.Attribute)
                    and first.func.attr == "state_dict"
                )
                is_state_dict_var = (
                    isinstance(first, ast.Name) and first.id == "state_dict"
                )
                assert is_state_dict_call or is_state_dict_var, (
                    f"{path.name}:{node.lineno} torch.save() first arg must be "
                    f"an x.state_dict() call or a 'state_dict' variable, "
                    f"got {ast.dump(first)}"
                )
