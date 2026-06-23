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

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from nvalchemi.training import (
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
    ConstantWeight,
    EnergyHuberLoss,
    EnergyMAELoss,
    EnergyMSELoss,
    ForceHuberLoss,
    ForceL2NormLoss,
    ForceMSELoss,
    LinearWeight,
    StressHuberLoss,
    StressMSELoss,
    loss_component_to_spec,
)
from nvalchemi.training._spec import create_model_spec, create_model_spec_from_json
from nvalchemi.training.losses import (
    assert_same_shape,
    frobenius_mse,
    per_graph_mean,
    per_graph_sum,
)
from nvalchemi.training.losses.terms import _huber_loss


class _ToyLoss(BaseLossFunction):
    # Concrete subclass returning a constant tensor — used in composition tests.

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)
        self.prediction_key = "prediction"
        self.target_key = "target"

    def forward(
        self,
        pred: torch.Tensor,  # noqa: ARG002
        target: torch.Tensor,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        return torch.tensor(self.value)

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return pred - target


class _PositionsLoss(BaseLossFunction):
    # Toy loss whose ``forward`` sums ``pred`` (gradient-bearing).

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = float(scale)
        self.prediction_key = "positions"
        self.target_key = "positions"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> torch.Tensor:
        return self.scale * pred.sum()

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return pred - target


class _ReturnSchedule:
    # Schedule whose ``__call__`` returns a configurable value.

    per_epoch: bool = False

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, step: int, epoch: int) -> Any:  # noqa: ARG002
        return self.value

    def to_spec(self) -> Any:
        """Return a rebuild spec so this helper satisfies LossWeightSchedule."""
        return create_model_spec(type(self), value=self.value)


def _dummy_loss_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(0.0), torch.tensor(0.0)


def _dummy_loss_mappings() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pred, target = _dummy_loss_tensors()
    return {"prediction": pred}, {"target": target}


def _full_loss_batch() -> SimpleNamespace:
    # Standard 3-graph layout covering energy + forces + stress.
    num_graphs = 3
    num_nodes = 6
    return SimpleNamespace(
        batch_idx=torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32),
        num_graphs=num_graphs,
        num_nodes_per_graph=torch.tensor([2, 3, 1], dtype=torch.long),
        energy=torch.tensor([[1.0], [2.0], [3.0]]),
        predicted_energy=torch.tensor([[1.5], [2.5], [3.5]]),
        forces=torch.zeros(num_nodes, 3),
        predicted_forces=torch.ones(num_nodes, 3),
        stress=torch.zeros(num_graphs, 3, 3),
        predicted_stress=torch.ones(num_graphs, 3, 3),
    )


def _loss_metadata(batch: SimpleNamespace) -> dict[str, Any]:
    return {
        name: getattr(batch, name)
        for name in ("batch_idx", "num_graphs", "num_nodes_per_graph")
        if hasattr(batch, name)
    }


def _tensor_mapping(batch: SimpleNamespace) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in vars(batch).items()
        if isinstance(value, torch.Tensor)
    }


def _call_from_batch(
    loss: BaseLossFunction | ComposedLossFunction,
    batch: SimpleNamespace,
    **metadata: Any,
) -> ComposedLossOutput:
    composed = (
        loss
        if isinstance(loss, ComposedLossFunction)
        else ComposedLossFunction(components=(loss,))
    )
    tensors = _tensor_mapping(batch)
    return composed(tensors, tensors, **(_loss_metadata(batch) | metadata))


class TestReductions:
    def setup_method(self) -> None:
        # 3 graphs with 2, 3, 1 atoms respectively.
        self.batch_idx = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)

    def test_per_graph_sum_matches_manual(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = per_graph_sum(vals, self.batch_idx)
        assert torch.allclose(got, torch.tensor([3.0, 12.0, 6.0]))

    def test_per_graph_sum_preserves_shape(self) -> None:
        vals = torch.randn(6, 3, requires_grad=True)
        got = per_graph_sum(vals, self.batch_idx)
        assert got.shape == (3, 3)
        assert got.grad_fn is not None

    def test_per_graph_sum_explicit_num_graphs_keeps_trailing_empty(self) -> None:
        vals = torch.tensor([1.0, 2.0])
        batch_idx = torch.tensor([0, 0], dtype=torch.int32)
        got = per_graph_sum(vals, batch_idx, num_graphs=3)
        assert torch.allclose(got, torch.tensor([3.0, 0.0, 0.0]))

    def test_per_graph_mean_matches_manual(self) -> None:
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        got = per_graph_mean(vals, self.batch_idx)
        assert torch.allclose(got, torch.tensor([1.5, 4.0, 6.0]))

    def test_frobenius_mse_matches_manual(self) -> None:
        pred = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            ]
        )
        target = torch.zeros(2, 3, 3)
        # Identity * k contributes 3*k^2 nonzero entries; mean over 9.
        got = frobenius_mse(pred, target)
        expected = torch.tensor([3.0 / 9.0, 12.0 / 9.0])
        assert torch.allclose(got, expected)

    def test_frobenius_mse_preserves_grad(self) -> None:
        pred = torch.randn(2, 3, 3, requires_grad=True)
        target = torch.randn(2, 3, 3)
        got = frobenius_mse(pred, target)
        assert got.grad_fn is not None
        got.sum().backward()
        assert pred.grad is not None

    def test_per_graph_sum_bad_num_graphs(self) -> None:
        with pytest.raises(ValueError, match="num_graphs must be positive"):
            per_graph_sum(torch.zeros(3), torch.zeros(3, dtype=torch.int32), 0)


class TestReductionsCompile:
    @staticmethod
    def _compile_kwargs(device: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"fullgraph": True}
        if device == "cuda":
            kwargs["backend"] = "cudagraphs"
        return kwargs

    @staticmethod
    def _batch_idx(device: str) -> torch.Tensor:
        return torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32, device=device)

    @pytest.mark.parametrize(
        ("fn", "args_factory"),
        [
            pytest.param(
                per_graph_sum,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3),
                    batch_idx,
                    3,
                ),
                id="per_graph_sum",
            ),
            pytest.param(
                per_graph_mean,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(6, 3),
                    batch_idx,
                    3,
                ),
                id="per_graph_mean",
            ),
            pytest.param(
                frobenius_mse,
                lambda device, batch_idx: (
                    torch.arange(18, dtype=torch.float32, device=device).reshape(
                        2, 3, 3
                    ),
                    torch.arange(18, dtype=torch.float32, device=device)
                    .reshape(2, 3, 3)
                    .flip(0),
                ),
                id="frobenius_mse",
            ),
        ],
    )
    def test_reduction_compiles(
        self,
        fn: Any,
        args_factory: Any,
        device: str,
    ) -> None:
        batch_idx = self._batch_idx(device)
        args = args_factory(device, batch_idx)
        compiled = torch.compile(fn, **self._compile_kwargs(device))

        got = compiled(*args)
        expected = fn(*args)

        assert torch.allclose(got, expected)


class TestConcreteLossesCompile:
    @staticmethod
    def _compile_kwargs(device: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"fullgraph": True}
        if device == "cuda":
            kwargs["backend"] = "cudagraphs"
        return kwargs

    def test_energy_mae_loss_compiles_fullgraph(self, device: str) -> None:
        loss = EnergyMAELoss()
        pred = torch.tensor([[6.0], [15.0], [8.0]], device=device)
        target = torch.tensor([[3.0], [10.0], [4.0]], device=device)
        counts = torch.tensor([3, 5, 2], dtype=torch.long, device=device)

        def fn(
            pred: torch.Tensor, target: torch.Tensor, counts: torch.Tensor
        ) -> torch.Tensor:
            return loss(pred, target, num_nodes_per_graph=counts)

        compiled = torch.compile(fn, **self._compile_kwargs(device))
        torch.testing.assert_close(
            compiled(pred, target, counts), fn(pred, target, counts)
        )

    def test_force_l2_norm_loss_dense_compiles_fullgraph(self, device: str) -> None:
        loss = ForceL2NormLoss()
        pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            device=device,
        )
        target = torch.zeros_like(pred)
        batch_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=device)

        def fn(
            pred: torch.Tensor, target: torch.Tensor, batch_idx: torch.Tensor
        ) -> torch.Tensor:
            return loss(pred, target, batch_idx=batch_idx, num_graphs=2)

        compiled = torch.compile(fn, **self._compile_kwargs(device))
        torch.testing.assert_close(
            compiled(pred, target, batch_idx), fn(pred, target, batch_idx)
        )

    def test_force_l2_norm_loss_padded_compiles_fullgraph(self, device: str) -> None:
        loss = ForceL2NormLoss()
        pred = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                [[1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
            ],
            device=device,
        )
        target = torch.zeros_like(pred)
        counts = torch.tensor([3, 2], dtype=torch.long, device=device)

        def fn(
            pred: torch.Tensor, target: torch.Tensor, counts: torch.Tensor
        ) -> torch.Tensor:
            return loss(pred, target, num_nodes_per_graph=counts)

        compiled = torch.compile(fn, **self._compile_kwargs(device))
        torch.testing.assert_close(
            compiled(pred, target, counts), fn(pred, target, counts)
        )


class TestBaseLossFunction:
    # ``compute_residual(pred, target, valid)`` is the sole abstract method.
    # ``forward`` is a concrete template orchestrating validate → normalize →
    # mask → compute_residual → reduce.  Weighting lives on
    # :class:`ComposedLossFunction`.

    def test_baseloss_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseLossFunction()

    def test_baseloss_is_nn_module(self) -> None:
        loss = _ToyLoss(value=1.0)
        assert isinstance(loss, nn.Module)

    def test_baseloss_has_no_weight_attribute(self) -> None:
        loss = _ToyLoss(value=3.0)
        assert not hasattr(loss, "weight")

    def test_baseloss_forward_returns_raw_unweighted_tensor(self) -> None:
        # Calling the module must return exactly what ``forward`` returns,
        # with no weighting applied at the leaf.
        loss = _ToyLoss(value=2.5)
        assert torch.allclose(loss(*_dummy_loss_tensors()), torch.tensor(2.5))

    def test_baseloss_forward_accepts_extra_kwargs(self) -> None:
        # Leaves accept arbitrary kwargs so composed losses can forward
        # shared graph metadata without forcing every component to consume it.
        loss = _ToyLoss(value=4.0)
        assert torch.allclose(
            loss(*_dummy_loss_tensors(), unused_metadata=7), torch.tensor(4.0)
        )

    def test_baseloss_to_device_smoke(self) -> None:
        # Stateless loss still supports ``.to()`` via nn.Module.
        loss = EnergyMSELoss()
        moved = loss.to("meta")
        assert isinstance(moved, nn.Module)
        assert moved is loss  # .to() is in-place for nn.Module

    def test_baseloss_state_dict_empty(self) -> None:
        loss = EnergyMSELoss()
        assert len(loss.state_dict()) == 0
        assert list(loss.parameters()) == []
        assert list(loss.buffers()) == []

    def test_baseloss_dtype_policy_defaults_to_strict(self) -> None:
        loss = EnergyMSELoss()
        pred = torch.zeros(2, 1, dtype=torch.float32)
        target = torch.zeros(2, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="dtype mismatch"):
            loss(pred, target)

    def test_baseloss_dtype_policy_casts_prediction_to_target(self) -> None:
        loss = EnergyMSELoss(dtype_policy="prediction_to_target")
        pred = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        target = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        out = loss(pred, target)

        assert out.dtype == torch.float64
        torch.testing.assert_close(out, torch.tensor(2.5, dtype=torch.float64))

    def test_baseloss_dtype_policy_casts_target_to_prediction(self) -> None:
        loss = EnergyMSELoss(dtype_policy="target_to_prediction")
        pred = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        target = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        out = loss(pred, target)

        assert out.dtype == torch.float32
        torch.testing.assert_close(out, torch.tensor(2.5, dtype=torch.float32))

    def test_baseloss_dtype_policy_rejects_unknown_policy(self) -> None:
        with pytest.raises(ValueError, match="dtype_policy must be one of"):
            EnergyMSELoss(dtype_policy="match_target")  # type: ignore[arg-type]


class TestLossRepr:
    @pytest.mark.parametrize(
        ("loss_factory", "class_name", "substrings"),
        [
            pytest.param(
                lambda: EnergyMSELoss(per_atom=True),
                "EnergyMSELoss",
                (
                    "target_key='energy'",
                    "prediction_key='predicted_energy'",
                    "per_atom=True",
                ),
                id="energy",
            ),
            pytest.param(
                lambda: EnergyHuberLoss(delta=0.5),
                "EnergyHuberLoss",
                ("target_key='energy'", "per_atom=True", "delta=0.5"),
                id="energy_huber",
            ),
            pytest.param(
                lambda: ForceMSELoss(normalize_by_atom_count=False),
                "ForceMSELoss",
                ("normalize_by_atom_count=False",),
                id="force",
            ),
            pytest.param(
                lambda: ForceHuberLoss(delta=0.5),
                "ForceHuberLoss",
                (
                    "normalize_by_atom_count=False",
                    "delta=0.5",
                ),
                id="force_huber",
            ),
            pytest.param(
                lambda: StressMSELoss(ignore_nonfinite=True),
                "StressMSELoss",
                ("target_key='stress'", "ignore_nonfinite=True"),
                id="stress",
            ),
            pytest.param(
                lambda: StressHuberLoss(delta=0.5),
                "StressHuberLoss",
                ("target_key='stress'", "ignore_nonfinite=True", "delta=0.5"),
                id="stress_huber",
            ),
        ],
    )
    def test_concrete_loss_repr_contains_hyperparameters(
        self,
        loss_factory: Any,
        class_name: str,
        substrings: tuple[str, ...],
    ) -> None:
        text = repr(loss_factory())
        assert class_name in text
        for substring in substrings:
            assert substring in text, (substring, text)

    def test_concrete_loss_repr_has_no_weight_attribute(self) -> None:
        # Weight lives on the composition, not on leaves.
        for text in (
            repr(EnergyMSELoss()),
            repr(EnergyHuberLoss()),
            repr(ForceMSELoss()),
            repr(ForceHuberLoss()),
            repr(StressMSELoss()),
            repr(StressHuberLoss()),
        ):
            assert "weight" not in text

    def test_composed_repr_shows_nested_components(self) -> None:
        composed = EnergyMSELoss() + ForceMSELoss()
        text = repr(composed)
        assert "ComposedLossFunction" in text
        assert "EnergyMSELoss" in text
        assert "ForceMSELoss" in text
        # nn.ModuleList numbers its children; "(0):" is the first entry.
        assert "(0)" in text

    def test_composed_repr_includes_normalize_weights_flag(self) -> None:
        text = repr(EnergyMSELoss() + ForceMSELoss())
        assert "normalize_weights=True" in text
        text_off = repr(
            ComposedLossFunction(
                (EnergyMSELoss(), ForceMSELoss()), normalize_weights=False
            )
        )
        assert "normalize_weights=False" in text_off

    def test_extra_repr_non_empty_on_concrete(self) -> None:
        for loss in (EnergyMSELoss(), ForceMSELoss(), StressMSELoss()):
            assert loss.extra_repr() != ""


class TestComposedLossFunction:
    def setup_method(self) -> None:
        self.loss_a = _ToyLoss(value=1.0)
        self.loss_b = _ToyLoss(value=1.0)
        self.loss_c = _ToyLoss(value=1.0)

    def test_add_two_losses(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed, ComposedLossFunction)
        assert tuple(composed.components) == (self.loss_a, self.loss_b)

    def test_composed_defaults_to_normalize_weights_true(self) -> None:
        composed = ComposedLossFunction((EnergyMSELoss(), ForceMSELoss()))
        assert composed.normalize_weights is True
        # Defaults to all-1.0 weights → normalized to 1/N each.
        assert composed.current_weight() == [0.5, 0.5]

    def test_composed_default_weights_are_all_one(self) -> None:
        composed = ComposedLossFunction(
            (EnergyMSELoss(), ForceMSELoss()), normalize_weights=False
        )
        assert composed._weights == [1.0, 1.0]
        assert composed.current_weight() == [1.0, 1.0]

    def test_composed_dtype_policy_casts_strict_leaf_at_call_time(self) -> None:
        leaf = EnergyMSELoss()
        composed = ComposedLossFunction(
            (leaf,),
            dtype_policy="prediction_to_target",
        )
        pred = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        target = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        out = composed({"predicted_energy": pred}, {"energy": target})

        assert leaf.dtype_policy == "strict"
        assert out["total_loss"].dtype == torch.float64
        torch.testing.assert_close(
            out["total_loss"], torch.tensor(2.5, dtype=torch.float64)
        )

    def test_composed_dtype_policy_does_not_override_explicit_leaf_policy(self) -> None:
        leaf = EnergyMSELoss(dtype_policy="target_to_prediction")
        composed = ComposedLossFunction(
            (leaf,),
            dtype_policy="prediction_to_target",
        )
        pred = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        target = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        out = composed({"predicted_energy": pred}, {"energy": target})

        assert out["total_loss"].dtype == torch.float32
        torch.testing.assert_close(
            out["total_loss"], torch.tensor(2.5, dtype=torch.float32)
        )

    def test_composed_dtype_policy_rejects_unknown_policy(self) -> None:
        with pytest.raises(ValueError, match="dtype_policy must be one of"):
            ComposedLossFunction(
                (EnergyMSELoss(),),
                dtype_policy="match_target",  # type: ignore[arg-type]
            )

    def test_composed_dtype_policy_can_be_set_after_operator_sugar(self) -> None:
        loss = EnergyMSELoss() + EnergyMSELoss(
            prediction_key="predicted_aux",
            target_key="aux",
        )
        loss.dtype_policy = "prediction_to_target"
        pred = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
        target = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

        out = loss(
            {"predicted_energy": pred, "predicted_aux": pred},
            {"energy": target, "aux": target},
        )

        assert all(component.dtype_policy == "strict" for component in loss.components)
        assert out["total_loss"].dtype == torch.float64

    def test_composed_dtype_policy_assignment_is_validated(self) -> None:
        loss = EnergyMSELoss() + ForceMSELoss()
        with pytest.raises(ValueError, match="dtype_policy must be one of"):
            loss.dtype_policy = "match_target"  # type: ignore[assignment]

    def test_composed_dtype_policy_survives_more_operator_sugar(self) -> None:
        loss = EnergyMSELoss() + ForceMSELoss()
        loss.dtype_policy = "prediction_to_target"

        scaled = 0.5 * loss
        extended = loss + StressMSELoss()

        assert scaled.dtype_policy == "prediction_to_target"
        assert extended.dtype_policy == "prediction_to_target"

    def test_composed_is_nn_module(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed, nn.Module)
        modules = list(composed.modules())
        assert self.loss_a in modules
        assert self.loss_b in modules

    def test_composed_components_stored_as_module_list(self) -> None:
        composed = self.loss_a + self.loss_b
        assert isinstance(composed.components, nn.ModuleList)

    @pytest.mark.parametrize(
        "build",
        [
            lambda a, b, c: (a + b) + c,
            lambda a, b, c: a + (b + c),
        ],
        ids=["left_assoc", "right_assoc"],
    )
    def test_nested_addition_flattens(self, build: Any) -> None:
        composed = build(self.loss_a, self.loss_b, self.loss_c)
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3
        assert all(not isinstance(c, ComposedLossFunction) for c in composed.components)

    def test_sum_over_list(self) -> None:
        # sum() seeds with 0 → exercises __radd__.
        composed = sum([self.loss_a, self.loss_b, self.loss_c])
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3

    def test_weights_length_must_match_components(self) -> None:
        with pytest.raises(ValueError, match="weights has length"):
            ComposedLossFunction((self.loss_a, self.loss_b), weights=[1.0, 2.0, 3.0])

    def test_weights_reject_non_numeric(self) -> None:
        with pytest.raises(TypeError, match="weights\\[0\\] must be"):
            ComposedLossFunction(
                (self.loss_a,),
                weights=["not-a-weight"],  # type: ignore[list-item]
            )

    def test_weights_none_entry_coerced_to_one(self) -> None:
        composed = ComposedLossFunction(
            (self.loss_a, self.loss_b),
            weights=[None, 3.0],
            normalize_weights=False,
        )
        assert composed._weights == [1.0, 3.0]

    def test_normalize_weights_zero_sum_raises(self) -> None:
        composed = ComposedLossFunction((self.loss_a, self.loss_b), weights=[0.0, 0.0])
        with pytest.raises(
            ValueError,
            match=(
                r"sum is not strictly positive \(sum=0\.0\)\. "
                r"Resolved weights at step=0, epoch=None: "
                r"\{'_ToyLoss_0': 0\.0, '_ToyLoss_1': 0\.0\}"
            ),
        ):
            composed.current_weight()

    def test_normalize_weights_nonzero_but_zero_sum_raises(self) -> None:
        # [1, -1] sums to zero; the error message must reflect that the
        # individual resolved weights were non-zero.
        composed = ComposedLossFunction((self.loss_a, self.loss_b), weights=[1.0, -1.0])
        with pytest.raises(
            ValueError,
            match=(
                r"sum is not strictly positive \(sum=0\.0\)\. "
                r"Resolved weights at step=7, epoch=2: "
                r"\{'_ToyLoss_0': 1\.0, '_ToyLoss_1': -1\.0\}"
            ),
        ):
            composed.current_weight(step=7, epoch=2)

    def test_normalize_weights_negative_sum_raises(self) -> None:
        # A negative raw sum would flip every effective weight's sign
        # after normalization; reject it with the same "not strictly
        # positive" error used for zero sums.
        composed = ComposedLossFunction((self.loss_a, self.loss_b), weights=[1.0, -3.0])
        with pytest.raises(
            ValueError, match=r"sum is not strictly positive \(sum=-2\.0\)"
        ):
            composed.current_weight()

    def test_normalize_weights_false_returns_raw(self) -> None:
        composed = ComposedLossFunction(
            (self.loss_a, self.loss_b),
            weights=[3.0, 2.0],
            normalize_weights=False,
        )
        assert composed.current_weight() == [3.0, 2.0]

    def test_weighted_sum_unnormalized_is_pure_weighted_sum(self) -> None:
        a, b, v1, v2 = 2.0, 3.0, 5.0, 7.0
        comp1 = _ToyLoss(value=v1)
        comp2 = _ToyLoss(value=v2)
        composed = ComposedLossFunction(
            (comp1, comp2), weights=[a, b], normalize_weights=False
        )
        out = composed(*_dummy_loss_mappings(), step=0, epoch=0)
        expected = a * v1 + b * v2
        assert torch.allclose(out["total_loss"], torch.tensor(expected), atol=1e-6)

    def test_per_component_unweighted_and_weight_populated(self) -> None:
        comp1 = _ToyLoss(value=2.0)
        comp2 = _ToyLoss(value=4.0)
        composed = ComposedLossFunction(
            (comp1, comp2), weights=[3.0, 2.0], normalize_weights=False
        )
        out = composed(*_dummy_loss_mappings())
        assert set(out) == {
            "total_loss",
            "per_component_unweighted",
            "per_component_weight",
            "per_component_raw_weight",
            "per_component_sample",
        }
        assert torch.allclose(
            out["per_component_unweighted"]["_ToyLoss_0"], torch.tensor(2.0)
        )
        assert torch.allclose(
            out["per_component_unweighted"]["_ToyLoss_1"], torch.tensor(4.0)
        )
        assert out["per_component_weight"] == {
            "_ToyLoss_0": 3.0,
            "_ToyLoss_1": 2.0,
        }
        # Without normalization raw and effective weights match.
        assert out["per_component_raw_weight"] == out["per_component_weight"]
        assert torch.allclose(out["total_loss"], torch.tensor(14.0))

    def test_per_component_weight_reflects_normalization(self) -> None:
        composed = ComposedLossFunction(
            (_ToyLoss(value=1.0), _ToyLoss(value=1.0)),
            weights=[3.0, 2.0],
        )
        out = composed(*_dummy_loss_mappings())
        assert out["per_component_weight"] == {
            "_ToyLoss_0": 0.6,
            "_ToyLoss_1": 0.4,
        }
        # Raw weights expose the pre-normalization values so a user
        # logging a scheduled loss can observe the underlying ramp.
        assert out["per_component_raw_weight"] == {
            "_ToyLoss_0": 3.0,
            "_ToyLoss_1": 2.0,
        }
        assert torch.allclose(
            out["per_component_unweighted"]["_ToyLoss_0"], torch.tensor(1.0)
        )
        assert torch.allclose(
            out["per_component_unweighted"]["_ToyLoss_1"], torch.tensor(1.0)
        )

    def test_per_component_raw_weight_tracks_schedule_on_single_leaf(self) -> None:
        # Single-component normalized composition: effective weight is
        # always 1.0, so raw_weight is the only way to observe the
        # underlying schedule ramp.
        schedule = LinearWeight(start=0.0, end=1.0, num_steps=10)
        composed = schedule * _ToyLoss(value=1.0)
        out_mid = composed(*_dummy_loss_mappings(), step=5)
        assert out_mid["per_component_weight"] == {"_ToyLoss": 1.0}
        assert out_mid["per_component_raw_weight"] == {"_ToyLoss": 0.5}
        out_end = composed(*_dummy_loss_mappings(), step=10)
        assert out_end["per_component_raw_weight"] == {"_ToyLoss": 1.0}

    def test_empty_components_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ComposedLossFunction(components=())

    def test_non_loss_component_rejected(self) -> None:
        with pytest.raises(
            TypeError,
            match="components\\[0\\] must be a BaseLossFunction or ComposedLossFunction",
        ):
            ComposedLossFunction(
                components=("not-a-loss",),  # type: ignore[arg-type]
            )

    def test_requires_eval_grad_true_when_any_component_requires(self) -> None:
        class _GradLoss(_ToyLoss):
            requires_eval_grad = True

        class _NoGradLoss(_ToyLoss):
            requires_eval_grad = False

        composed = ComposedLossFunction((_NoGradLoss(), _GradLoss()))
        assert composed.requires_eval_grad() is True

    def test_requires_eval_grad_false_when_all_components_disclaim(self) -> None:
        class _NoGradLoss(_ToyLoss):
            requires_eval_grad = False

        composed = ComposedLossFunction((_NoGradLoss(), _NoGradLoss()))
        assert composed.requires_eval_grad() is False

    def test_requires_eval_grad_raises_on_undeclared_component(self) -> None:
        class _UndeclaredLoss(_ToyLoss):
            requires_eval_grad = None

        composed = ComposedLossFunction((_UndeclaredLoss(),))
        with pytest.raises(ValueError, match="infer whether"):
            composed.requires_eval_grad()

    def test_gradient_flows_through_all_components(self) -> None:
        positions = torch.randn(4, 3, requires_grad=True)
        loss_a = _PositionsLoss(scale=2.0)
        loss_b = _PositionsLoss(scale=3.0)
        composed = ComposedLossFunction(
            (loss_a, loss_b), weights=[1.0, 1.0], normalize_weights=False
        )
        out = composed(
            {"positions": positions},
            {"positions": torch.zeros_like(positions)},
            step=0,
            epoch=0,
        )
        out["total_loss"].backward()
        # d/dx sum(x) = 1 per element; composed multiplier = 2 + 3 = 5.
        expected_grad = torch.full_like(positions, 5.0)
        assert positions.grad is not None
        assert torch.allclose(positions.grad, expected_grad, atol=1e-6)

    def test_schedule_applied_inside_composition(self) -> None:
        # A schedule attached to a component's slot in the composition is
        # resolved once per call.
        leaf = _ToyLoss(value=4.0)
        composed = ComposedLossFunction(
            (leaf,), weights=[ConstantWeight(value=2.5)], normalize_weights=False
        )
        out = composed(*_dummy_loss_mappings(), step=0, epoch=0)
        # 2.5 (schedule) * 4.0 (forward) = 10.0
        assert torch.allclose(out["total_loss"], torch.tensor(10.0), atol=1e-6)

    def test_schedule_counters_are_not_forwarded_to_leaf(self) -> None:
        class _StrictLoss(BaseLossFunction):
            prediction_key = "prediction"
            target_key = "target"

            def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return pred + target

            def compute_residual(
                self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
            ) -> torch.Tensor:
                return pred - target

        composed = ComposedLossFunction(
            (_StrictLoss(),),
            weights=[LinearWeight(start=0.0, end=1.0, num_steps=10)],
            normalize_weights=False,
        )
        out = composed(
            {"prediction": torch.tensor(2.0)},
            {"target": torch.tensor(3.0)},
            step=5,
            epoch=7,
        )
        assert torch.allclose(out["total_loss"], torch.tensor(2.5), atol=1e-6)

    def test_linear_schedule_on_component_in_composition(self) -> None:
        leaf = _ToyLoss(value=1.0)
        composed = ComposedLossFunction(
            (leaf,),
            weights=[LinearWeight(start=0.0, end=1.0, num_steps=10)],
            normalize_weights=False,
        )
        assert torch.allclose(
            composed(*_dummy_loss_mappings(), step=0)["total_loss"],
            torch.tensor(0.0),
            atol=1e-6,
        )
        assert torch.allclose(
            composed(*_dummy_loss_mappings(), step=10)["total_loss"],
            torch.tensor(1.0),
            atol=1e-6,
        )
        assert torch.allclose(
            composed(*_dummy_loss_mappings(), step=5)["total_loss"],
            torch.tensor(0.5),
            atol=1e-6,
        )

    def test_per_epoch_schedule_with_none_epoch_raises_in_composition(self) -> None:
        leaf = _ToyLoss(value=1.0)
        composed = ComposedLossFunction(
            (leaf,),
            weights=[LinearWeight(start=0.0, end=1.0, num_steps=10, per_epoch=True)],
        )
        with pytest.raises(ValueError, match="per_epoch=True"):
            composed(*_dummy_loss_mappings(), step=3, epoch=None)

    @pytest.mark.parametrize(
        ("bad_value", "match"),
        [
            pytest.param(float("nan"), r"non-finite weight nan", id="nan"),
            pytest.param(float("inf"), r"non-finite weight inf", id="inf"),
            pytest.param(float("-inf"), r"non-finite weight -inf", id="neg_inf"),
        ],
    )
    def test_schedule_non_finite_weight_raises(
        self, bad_value: float, match: str
    ) -> None:
        composed = ComposedLossFunction(
            (_ToyLoss(value=1.0),),
            weights=[_ReturnSchedule(bad_value)],
            normalize_weights=False,
        )
        with pytest.raises(ValueError, match=match):
            composed.current_weight(step=0, epoch=0)

    def test_schedule_non_numeric_weight_raises(self) -> None:
        composed = ComposedLossFunction(
            (_ToyLoss(value=1.0),),
            weights=[_ReturnSchedule("oops")],
            normalize_weights=False,
        )
        with pytest.raises(
            TypeError,
            match=r"_ReturnSchedule returned str; "
            r"LossWeightSchedule\.__call__ must return float",
        ):
            composed.current_weight(step=0, epoch=0)

    def test_nested_composition_applies_each_weight_exactly_once(self) -> None:
        # Nesting must not cause duplicate weight application.
        leaf = _ToyLoss(value=2.0)
        inner = ComposedLossFunction((leaf,), weights=[3.0], normalize_weights=False)
        outer = ComposedLossFunction((inner,), weights=[1.0], normalize_weights=False)
        out = outer(*_dummy_loss_mappings(), step=0, epoch=0)
        # 1 * 3 * 2 = 6
        assert torch.allclose(out["total_loss"], torch.tensor(6.0), atol=1e-6)
        assert torch.allclose(
            out["per_component_unweighted"]["_ToyLoss"], torch.tensor(2.0)
        )

    def test_nested_composition_multiplies_weights_elementwise(self) -> None:
        leaf1 = _ToyLoss(value=1.0)
        leaf2 = _ToyLoss(value=1.0)
        inner = ComposedLossFunction(
            (leaf1, leaf2), weights=[3.0, 2.0], normalize_weights=False
        )
        # Outer wraps the inner composition with weight 5.0.
        outer = ComposedLossFunction((inner,), weights=[5.0], normalize_weights=False)
        # After flattening the effective per-leaf weights are 5*3=15, 5*2=10.
        assert outer.current_weight() == [15.0, 10.0]

    @pytest.mark.parametrize("op", ["add"], ids=["add"])
    def test_not_implemented_for_bad_type(self, op: str) -> None:
        if op == "add":
            with pytest.raises(TypeError):
                _ = self.loss_a + "hello"  # type: ignore[operator]


class TestOperatorSugar:
    """Tests for ``scalar * loss``, ``schedule * loss``, and operator composition."""

    @pytest.mark.parametrize(
        ("side", "weight_kind"),
        [
            ("left", "float"),
            ("right", "float"),
            ("left", "schedule"),
            ("right", "schedule"),
        ],
    )
    def test_multiplication_wraps_leaf_in_composition(
        self, side: str, weight_kind: str
    ) -> None:
        leaf = _ToyLoss(value=1.0)
        weight: float | ConstantWeight = (
            3.0 if weight_kind == "float" else ConstantWeight(value=2.5)
        )
        composed = weight * leaf if side == "left" else leaf * weight
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 1
        assert composed._weights == [weight]

    def test_scaled_leaf_plus_scaled_leaf_flattens_and_normalizes(self) -> None:
        composed = 3.0 * EnergyMSELoss() + 2.0 * ForceMSELoss()
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 2
        # Raw weights preserved on construction; normalization is applied
        # only at call time.
        assert composed._weights == [3.0, 2.0]
        # Default normalize_weights=True: 3/(3+2), 2/(3+2).
        assert composed.current_weight() == [0.6, 0.4]

    def test_single_scaled_leaf_normalizes_to_one(self) -> None:
        composed = 3.0 * _ToyLoss(value=5.0)
        # One component → sum(raw)=3 → effective 1.0.
        assert composed.current_weight() == [1.0]
        out = composed(*_dummy_loss_mappings())
        assert torch.allclose(out["total_loss"], torch.tensor(5.0), atol=1e-6)

    def test_schedule_times_leaf_participates_in_current_weight(self) -> None:
        # Operator-attached schedule is stored on the composition and
        # resolved at call time. Step-interpolation detail is covered by
        # ``test_linear_schedule_on_component_in_composition``.
        schedule = LinearWeight(start=0.0, end=1.0, num_steps=10)
        composed = schedule * _ToyLoss(value=1.0) + _ToyLoss(value=1.0)
        assert composed._weights[0] is schedule
        assert composed.current_weight(step=10) == [0.5, 0.5]

    def test_float_mul_on_composition_scales_every_weight(self) -> None:
        base = ComposedLossFunction(
            (_ToyLoss(value=1.0), _ToyLoss(value=1.0)),
            weights=[3.0, 2.0],
            normalize_weights=False,
        )
        scaled = 4.0 * base
        assert scaled._weights == [12.0, 8.0]
        # Normalization flag is inherited.
        assert scaled.normalize_weights is False

    def test_schedule_mul_on_composition_raises(self) -> None:
        base = ComposedLossFunction((_ToyLoss(value=1.0),))
        schedule = ConstantWeight(value=2.0)
        with pytest.raises(TypeError, match="LossWeightSchedule"):
            _ = schedule * base

    def test_add_mismatched_normalize_raises(self) -> None:
        normalized = ComposedLossFunction(
            (_ToyLoss(value=1.0),), normalize_weights=True
        )
        unnormalized = ComposedLossFunction(
            (_ToyLoss(value=1.0),), normalize_weights=False
        )
        with pytest.raises(
            ValueError,
            match=r"mismatched normalize_weights \(self=True, other=False\)",
        ):
            _ = normalized + unnormalized
        with pytest.raises(
            ValueError,
            match=r"mismatched normalize_weights \(self=False, other=True\)",
        ):
            _ = unnormalized + normalized

    def test_bool_multiplication_rejected(self) -> None:
        # ``True * loss`` could silently mean "1.0 * loss", which hides
        # user bugs. Reject bools explicitly.
        with pytest.raises(TypeError):
            _ = True * _ToyLoss(value=1.0)  # type: ignore[operator]

    def test_radd_bare_leaf_plus_composition(self) -> None:
        composition = 2.0 * _ToyLoss(value=1.0)
        result = _ToyLoss(value=1.0) + composition
        assert isinstance(result, ComposedLossFunction)
        assert len(result.components) == 2
        assert result._weights == [1.0, 2.0]


class TestWeightFactors:
    @pytest.mark.parametrize(
        ("factory", "expected"),
        [
            pytest.param(
                lambda: ComposedLossFunction(
                    (EnergyMSELoss(), ForceMSELoss()),
                    normalize_weights=False,
                ),
                {"EnergyMSELoss": 1.0, "ForceMSELoss": 1.0},
                id="default_weights_unnormalized",
            ),
            pytest.param(
                lambda: ComposedLossFunction(
                    (EnergyMSELoss(), ForceMSELoss()),
                    weights=[ConstantWeight(value=2.0), ConstantWeight(value=3.0)],
                    normalize_weights=False,
                ),
                {"EnergyMSELoss": 2.0, "ForceMSELoss": 3.0},
                id="schedule_weights_unnormalized",
            ),
            pytest.param(
                lambda: ComposedLossFunction(
                    (EnergyMSELoss(), ForceMSELoss()),
                    weights=[3.0, 2.0],
                ),
                {"EnergyMSELoss": 0.6, "ForceMSELoss": 0.4},
                id="float_weights_normalized",
            ),
        ],
    )
    def test_weight_factors_simple_cases(
        self, factory: Any, expected: dict[str, float]
    ) -> None:
        assert factory().weight_factors(step=0, epoch=0) == expected

    def test_weight_factors_no_args_smoke(self) -> None:
        # ``weight_factors`` takes default ``step=0, epoch=None`` so
        # introspection helpers don't demand args.
        composed = ComposedLossFunction(
            (EnergyMSELoss(),), weights=[ConstantWeight(value=2.0)]
        )
        # Single component + normalization → effective weight is 1.0.
        assert composed.weight_factors() == {"EnergyMSELoss": 1.0}

    def test_weight_factors_class_name_collision_gets_indexed_suffix(self) -> None:
        composed = ComposedLossFunction(
            components=(StressMSELoss(), StressMSELoss()),
        )
        got = composed.weight_factors(step=0, epoch=0)
        assert set(got) == {"StressMSELoss_0", "StressMSELoss_1"}
        # Normalized to 0.5 each.
        assert got["StressMSELoss_0"] == 0.5
        assert got["StressMSELoss_1"] == 0.5

    def test_weight_factors_three_way_collision_across_nested_composition(self) -> None:
        # Inner composition contains two ``StressMSELoss`` instances; wrapping in
        # an outer composition with another ``StressMSELoss`` must collapse to
        # three collision-suffixed keys — NOT to a mix like
        # ``{"StressMSELoss_0", "StressMSELoss_1", "StressMSELoss"}`` from per-level
        # suffixing.
        inner = ComposedLossFunction(components=(StressMSELoss(), StressMSELoss()))
        outer = ComposedLossFunction(
            components=(inner, StressMSELoss()), normalize_weights=False
        )
        got = outer.weight_factors(step=0, epoch=0)
        assert set(got) == {"StressMSELoss_0", "StressMSELoss_1", "StressMSELoss_2"}
        assert all(v == 1.0 for v in got.values())

    def test_weight_factors_nested_composition_flattens(self) -> None:
        inner = ComposedLossFunction(
            (EnergyMSELoss(),),
            weights=[ConstantWeight(value=0.5)],
            normalize_weights=False,
        )
        outer = ComposedLossFunction(
            (inner, ForceMSELoss()),
            weights=[1.0, ConstantWeight(value=4.0)],
            normalize_weights=False,
        )
        assert outer.weight_factors(step=0, epoch=0) == {
            "EnergyMSELoss": 0.5,
            "ForceMSELoss": 4.0,
        }


class TestConcreteLosses:
    def setup_method(self) -> None:
        # Mixed-size batch: 3 graphs with 3, 5, 2 atoms respectively.
        self.nodes_per_graph = [3, 5, 2]
        self.num_graphs = 3
        self.num_nodes = sum(self.nodes_per_graph)
        self.batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=torch.int32)
        self.num_nodes_per_graph = torch.tensor(self.nodes_per_graph, dtype=torch.long)

    def _batch(self, **extra: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
            num_nodes_per_graph=self.num_nodes_per_graph,
            **extra,
        )

    @staticmethod
    def _force_l2_dense_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        target = torch.zeros(5, 3)
        pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
            ]
        )
        return pred, target, batch_idx

    @staticmethod
    def _force_l2_padded_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = torch.zeros(2, 3, 3)
        pred = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 3.0],
                ],
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 0.0, 0.0],
                    [99.0, 99.0, 99.0],
                ],
            ]
        )
        counts = torch.tensor([3, 2], dtype=torch.long)
        return pred, target, counts

    @pytest.mark.parametrize(
        ("loss", "pred", "target"),
        [
            pytest.param(
                EnergyMSELoss(dtype_policy="prediction_to_target"),
                torch.tensor([[1.0], [3.0]], dtype=torch.float32),
                torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                id="energy_mse",
            ),
            pytest.param(
                EnergyMAELoss(per_atom=False, dtype_policy="prediction_to_target"),
                torch.tensor([[1.0], [3.0]], dtype=torch.float32),
                torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                id="energy_mae",
            ),
            pytest.param(
                EnergyHuberLoss(per_atom=False, dtype_policy="prediction_to_target"),
                torch.tensor([[1.0], [3.0]], dtype=torch.float32),
                torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                id="energy_huber",
            ),
            pytest.param(
                ForceMSELoss(
                    normalize_by_atom_count=False,
                    dtype_policy="prediction_to_target",
                ),
                torch.tensor(
                    [[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ),
                torch.zeros(2, 3, dtype=torch.float64),
                id="force_mse",
            ),
            pytest.param(
                ForceHuberLoss(
                    normalize_by_atom_count=False,
                    dtype_policy="prediction_to_target",
                ),
                torch.tensor(
                    [[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ),
                torch.zeros(2, 3, dtype=torch.float64),
                id="force_huber",
            ),
            pytest.param(
                ForceL2NormLoss(
                    normalize_by_atom_count=False,
                    dtype_policy="prediction_to_target",
                ),
                torch.tensor(
                    [[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]],
                    dtype=torch.float32,
                ),
                torch.zeros(2, 3, dtype=torch.float64),
                id="force_l2_norm",
            ),
            pytest.param(
                StressMSELoss(dtype_policy="prediction_to_target"),
                torch.ones(2, 3, 3, dtype=torch.float32),
                torch.zeros(2, 3, 3, dtype=torch.float64),
                id="stress_mse",
            ),
            pytest.param(
                StressHuberLoss(dtype_policy="prediction_to_target"),
                torch.ones(2, 3, 3, dtype=torch.float32),
                torch.zeros(2, 3, 3, dtype=torch.float64),
                id="stress_huber",
            ),
        ],
    )
    def test_concrete_losses_can_cast_prediction_to_target_dtype(
        self,
        loss: BaseLossFunction,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        out = loss(pred, target)

        assert out.dtype == target.dtype
        assert torch.isfinite(out)

    def test_energy_loss_gradient_matches_analytic(
        self, fixed_torch_seed: None
    ) -> None:
        target = torch.randn(self.num_graphs, 1)
        pred = (target + torch.randn_like(target) * 0.1).detach().requires_grad_()
        EnergyMSELoss()(pred, target).backward()
        # MSE over (B, 1): d/d pred = 2*(pred - target) / B.
        expected_grad = 2.0 * (pred.detach() - target) / self.num_graphs
        assert pred.grad is not None
        assert torch.allclose(pred.grad, expected_grad, atol=1e-6)

    def test_energy_loss_per_atom_weights_by_atom_count(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])  # per-graph energies
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        got = EnergyMSELoss(per_atom=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        # Per-atom pred: [2, 3, 4]; target: [1, 2, 2]; diffs: [1, 1, 2].
        # Atom-count weighted MSE: (3*1 + 5*1 + 2*4) / (3 + 5 + 2) = 1.6.
        assert torch.allclose(got, torch.tensor(1.6), atol=1e-6)

    def test_energy_loss_per_atom_accepts_padded_node_mask(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])  # per-graph energies
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        node_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, False, False, False],
            ]
        )
        got = EnergyMSELoss(per_atom=True)(pred, target, num_nodes_per_graph=node_mask)
        # The padded mask has row counts [3, 5, 2], matching the dense-count test.
        assert torch.allclose(got, torch.tensor(1.6), atol=1e-6)

    def test_energy_loss_per_atom_accepts_cpu_counts_on_cuda(
        self, gpu_device: str
    ) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]], device=gpu_device)
        pred = torch.tensor([[6.0], [15.0], [8.0]], device=gpu_device)
        got = EnergyMSELoss(per_atom=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )

        assert got.device.type == "cuda"
        assert torch.allclose(got, torch.tensor(1.6, device=gpu_device), atol=1e-6)

    def test_energy_mae_loss_matches_manual_per_atom_mean(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        got = EnergyMAELoss()(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        counts = self.num_nodes_per_graph.to(pred).unsqueeze(-1)
        abs_residual = (pred / counts - target / counts).abs()
        expected = (abs_residual * counts).sum() / counts.sum()
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_mae_loss_ignores_nan_and_inf_targets(self) -> None:
        target = torch.tensor([[3.0], [float("nan")], [float("inf")], [8.0]])
        pred = torch.tensor([[6.0], [20.0], [30.0], [4.0]])
        counts = torch.tensor([3, 5, 2, 2], dtype=torch.long)
        got = EnergyMAELoss()(pred, target, num_nodes_per_graph=counts)
        # Valid entries: index 0 (count=3) and index 3 (count=2).
        # Per-atom abs residuals: |6/3 - 3/3| = 1.0, |4/2 - 8/2| = 2.0
        # Atom-count weighted: (3*1.0 + 2*2.0) / (3+2) = 7/5 = 1.4
        expected = torch.tensor(7.0 / 5.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_mae_loss_gradient_flows(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]], requires_grad=True)
        EnergyMAELoss()(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        ).backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_energy_huber_loss_matches_mace_style_graph_mean(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        got = EnergyHuberLoss(delta=1.5)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        counts = self.num_nodes_per_graph.to(pred).unsqueeze(-1)
        per_graph = _huber_loss(pred / counts - target / counts, 1.5).reshape(-1)
        expected = per_graph.mean()
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_huber_loss_ignores_nan_and_inf_targets(self) -> None:
        target = torch.tensor([[3.0], [float("nan")], [float("inf")], [8.0]])
        pred = torch.tensor([[6.0], [20.0], [30.0], [4.0]])
        counts = torch.tensor([3, 5, 2, 2], dtype=torch.long)
        got = EnergyHuberLoss(delta=1.5)(pred, target, num_nodes_per_graph=counts)
        expected = _huber_loss(torch.tensor([1.0, -2.0]), 1.5).mean()
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_mae_loss_accepts_vector_shape(self) -> None:
        target = torch.tensor([3.0, 10.0, 4.0])
        pred = torch.tensor([6.0, 15.0, 8.0])
        got = EnergyMAELoss()(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        counts = self.num_nodes_per_graph.to(pred)
        abs_residual = (pred / counts - target / counts).abs()
        expected = (abs_residual * counts).sum() / counts.sum()
        assert torch.allclose(got, expected, atol=1e-6)

    @pytest.mark.parametrize(
        "bad_counts",
        [
            torch.tensor([3, 5], dtype=torch.long),
            torch.ones(2, 5, dtype=torch.bool),
        ],
        ids=["count_length", "mask_batch"],
    )
    def test_energy_mae_loss_rejects_count_batch_mismatch(
        self, bad_counts: torch.Tensor
    ) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        with pytest.raises(ValueError, match="must match energy batch size"):
            EnergyMAELoss()(pred, target, num_nodes_per_graph=bad_counts)

    def test_force_loss_matches_hand_computed(self) -> None:
        # 2 graphs with 3 and 2 atoms for a small hand-traceable case.
        batch_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
        target = torch.zeros(5, 3)
        pred = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # graph 0 atom 0: |f|^2 = 1
                [0.0, 2.0, 0.0],  # graph 0 atom 1: |f|^2 = 4
                [0.0, 0.0, 3.0],  # graph 0 atom 2: |f|^2 = 9
                [1.0, 1.0, 1.0],  # graph 1 atom 0: |f|^2 = 3
                [2.0, 0.0, 0.0],  # graph 1 atom 1: |f|^2 = 4
            ]
        )
        # normalize_by_atom_count=True: per-graph mean of |f|^2 then mean
        # over graphs, then / 3 for per-component.
        # graph 0 mean |f|^2 = (1+4+9)/3 = 14/3
        # graph 1 mean |f|^2 = (3+4)/2 = 7/2
        # mean over graphs = (14/3 + 7/2) / 2 = (28/6 + 21/6) / 2 = 49/12
        # divided by 3 components = 49/36
        got_norm = ForceMSELoss(normalize_by_atom_count=True)(
            pred,
            target,
            batch_idx=batch_idx,
            num_graphs=2,
        )
        assert torch.allclose(got_norm, torch.tensor(49.0 / 36.0), atol=1e-6)

        # normalize=False: elementwise mean over the (V, 3) tensor.
        # sum of squares = 1+4+9+3+4 = 21 across 5*3 = 15 entries -> 21/15 = 1.4.
        got_global = ForceMSELoss(normalize_by_atom_count=False)(pred, target)
        assert torch.allclose(got_global, torch.tensor(21.0 / 15.0), atol=1e-6)

    def test_force_huber_loss_matches_componentwise_huber(self) -> None:
        target = torch.zeros(2, 3)
        pred = torch.tensor(
            [
                [0.5, 2.0, -3.0],
                [1.0, -0.25, 0.0],
            ]
        )
        got = ForceHuberLoss(normalize_by_atom_count=False, delta=1.0)(pred, target)
        expected = _huber_loss(pred - target, 1.0).mean()
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_l2_norm_loss_dense_matches_manual_per_graph_reduction(self) -> None:
        pred, target, batch_idx = self._force_l2_dense_case()
        got = ForceL2NormLoss()(pred, target, batch_idx=batch_idx, num_graphs=2)
        per_atom = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
        graph0 = per_atom[:3].mean()
        graph1 = per_atom[3:].mean()
        expected = torch.stack((graph0, graph1)).mean()
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_l2_norm_loss_dense_global_mean_when_not_normalized(self) -> None:
        pred, target, _ = self._force_l2_dense_case()
        got = ForceL2NormLoss(normalize_by_atom_count=False)(pred, target)
        expected = torch.linalg.vector_norm(pred - target, ord=2, dim=-1).mean()
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_padded_layout_matches_flat_hand_computed(self) -> None:
        target = torch.zeros(2, 3, 3)
        pred = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],  # graph 0 atom 0: |f|^2 = 1
                    [0.0, 2.0, 0.0],  # graph 0 atom 1: |f|^2 = 4
                    [0.0, 0.0, 3.0],  # graph 0 atom 2: |f|^2 = 9
                ],
                [
                    [1.0, 1.0, 1.0],  # graph 1 atom 0: |f|^2 = 3
                    [2.0, 0.0, 0.0],  # graph 1 atom 1: |f|^2 = 4
                    [99.0, 99.0, 99.0],  # padding; must be ignored
                ],
            ]
        )
        target[1, 2] = float("nan")
        num_nodes_per_graph = torch.tensor([3, 2], dtype=torch.long)

        got_norm = ForceMSELoss(normalize_by_atom_count=True)(
            pred, target, num_nodes_per_graph=num_nodes_per_graph
        )
        assert torch.allclose(got_norm, torch.tensor(49.0 / 36.0), atol=1e-6)

        got_global = ForceMSELoss(normalize_by_atom_count=False)(
            pred, target, num_nodes_per_graph=num_nodes_per_graph
        )
        assert torch.allclose(got_global, torch.tensor(21.0 / 15.0), atol=1e-6)

    def test_force_huber_loss_padded_layout_ignores_padding(self) -> None:
        target = torch.zeros(2, 3, 3)
        pred = torch.ones(2, 3, 3)
        pred[1, 2] = 100.0
        target[1, 2] = float("nan")
        counts = torch.tensor([3, 2], dtype=torch.long)

        got = ForceHuberLoss(
            normalize_by_atom_count=False,
            delta=1.0,
        )(pred, target, num_nodes_per_graph=counts)

        assert torch.allclose(got, torch.tensor(0.5), atol=1e-6)

    def test_force_l2_norm_loss_padded_ignores_padding_and_nonfinite_targets(
        self,
    ) -> None:
        pred, target, num_nodes_per_graph = self._force_l2_padded_case()
        target[0, 2, 0] = float("inf")
        target[1, 2] = float("nan")

        got = ForceL2NormLoss()(pred, target, num_nodes_per_graph=num_nodes_per_graph)
        expected = torch.stack(
            (
                torch.tensor([1.0, 2.0]).mean(),
                torch.tensor([3.0**0.5, 2.0]).mean(),
            )
        ).mean()
        assert torch.allclose(got, expected, atol=1e-6)

        got_global = ForceL2NormLoss(normalize_by_atom_count=False)(
            pred, target, num_nodes_per_graph=num_nodes_per_graph
        )
        expected_global = torch.tensor([1.0, 2.0, 3.0**0.5, 2.0]).mean()
        assert torch.allclose(got_global, expected_global, atol=1e-6)

    @pytest.mark.parametrize(
        "bad_counts",
        [
            torch.tensor([3], dtype=torch.long),
            torch.ones(1, 3, dtype=torch.bool),
        ],
        ids=["count_length", "mask_batch"],
    )
    def test_force_l2_norm_loss_padded_rejects_count_batch_mismatch(
        self, bad_counts: torch.Tensor
    ) -> None:
        pred, target, _ = self._force_l2_padded_case()
        with pytest.raises(ValueError, match="must match force batch size"):
            ForceL2NormLoss()(pred, target, num_nodes_per_graph=bad_counts)

    def test_force_loss_padded_layout_accepts_node_mask(self) -> None:
        target = torch.zeros(2, 3, 3)
        pred = torch.ones(2, 3, 3)
        pred[1, 2] = 100.0
        target[1, 2] = float("nan")
        node_mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
            ]
        )

        got = ForceMSELoss(normalize_by_atom_count=True)(
            pred, target, num_nodes_per_graph=node_mask
        )

        assert torch.allclose(got, torch.tensor(1.0), atol=1e-6)

    def test_force_loss_gradient_flows(self) -> None:
        pred = torch.randn(self.num_nodes, 3, requires_grad=True)
        target = torch.randn(self.num_nodes, 3)
        ForceMSELoss()(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        ).backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_force_l2_norm_loss_gradient_flows(self) -> None:
        pred = torch.randn(self.num_nodes, 3, requires_grad=True)
        target = torch.randn(self.num_nodes, 3)
        ForceL2NormLoss()(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        ).backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_stress_loss_matches_elementwise_mse(self, fixed_torch_seed: None) -> None:
        pred = torch.randn(self.num_graphs, 3, 3, requires_grad=True)
        target = torch.randn(self.num_graphs, 3, 3)
        got = StressMSELoss()(pred, target)
        # Frobenius MSE averaged over graphs == elementwise MSE.
        expected = torch.nn.functional.mse_loss(pred, target)
        assert torch.allclose(got, expected, atol=1e-6)
        got.backward()
        assert pred.grad is not None

    def test_stress_huber_loss_matches_componentwise_huber(self) -> None:
        pred = torch.tensor(
            [
                [[1.0, 2.0, 0.0], [0.5, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            requires_grad=True,
        )
        target = torch.zeros_like(pred)
        got = StressHuberLoss(delta=1.0)(pred, target)
        expected = _huber_loss(pred - target, 1.0).reshape(2, -1).mean(dim=-1).mean()
        assert torch.allclose(got, expected, atol=1e-6)
        got.backward()
        assert pred.grad is not None

    @pytest.mark.parametrize(
        ("loss_factory", "batch_kwargs", "missing_attr"),
        [
            pytest.param(
                lambda: EnergyMSELoss(),
                {"energy": torch.zeros(3, 1)},  # predicted_energy omitted
                "predicted_energy",
                id="energy_missing_prediction",
            ),
            pytest.param(
                lambda: ForceMSELoss(),
                {"predicted_forces": torch.zeros(10, 3)},  # forces omitted
                "forces",
                id="force_missing_target",
            ),
            pytest.param(
                lambda: StressMSELoss(),
                {"stress": torch.zeros(3, 3, 3)},  # predicted_stress omitted
                "predicted_stress",
                id="stress_missing_prediction",
            ),
        ],
    )
    def test_missing_mapping_key_raises_key_error(
        self,
        loss_factory: Any,
        batch_kwargs: dict[str, torch.Tensor],
        missing_attr: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**batch_kwargs)
        with pytest.raises(KeyError, match=missing_attr):
            _call_from_batch(loss, batch)

    def test_mapping_key_resolving_to_none_raises_type_error(self) -> None:
        loss = ComposedLossFunction(components=(EnergyMSELoss(),))
        predictions = {"predicted_energy": None}  # type: ignore[dict-item]
        targets = {"energy": torch.zeros(3, 1)}
        with pytest.raises(TypeError, match="predicted_energy"):
            loss(predictions, targets)

    @pytest.mark.parametrize(
        ("loss_factory", "batch_kwargs", "loss_name"),
        [
            pytest.param(
                lambda: EnergyMSELoss(),
                {
                    "energy": torch.zeros(3, 2),  # unequal trailing dim (strict)
                    "predicted_energy": torch.zeros(3, 3),
                },
                "EnergyMSELoss",
                id="energy_trailing_mismatch",
            ),
            pytest.param(
                lambda: ForceMSELoss(),
                {
                    "forces": torch.zeros(10, 2),  # unequal trailing dim (strict)
                    "predicted_forces": torch.zeros(10, 3),
                },
                "ForceMSELoss",
                id="force_component_mismatch",
            ),
            pytest.param(
                lambda: StressMSELoss(),
                {
                    "stress": torch.zeros(3, 2),  # rank/shape mismatch (strict)
                    "predicted_stress": torch.zeros(3, 3, 3),
                },
                "StressMSELoss",
                id="stress_rank_mismatch",
            ),
        ],
    )
    def test_prediction_target_shape_mismatch_raises(
        self,
        loss_factory: Any,
        batch_kwargs: dict[str, torch.Tensor],
        loss_name: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**batch_kwargs)
        with pytest.raises(
            ValueError,
            match=rf"{loss_name}: prediction and target shape must match exactly",
        ):
            _call_from_batch(loss, batch)

    @pytest.mark.parametrize(
        ("loss_factory", "tensor_kwargs", "missing_attr"),
        [
            pytest.param(
                lambda: EnergyMSELoss(per_atom=True),
                {
                    "energy": torch.zeros(3, 1),
                    "predicted_energy": torch.zeros(3, 1),
                },
                "num_nodes_per_graph",
                id="energy_per_atom_missing_num_nodes",
            ),
            pytest.param(
                lambda: ForceMSELoss(),
                {
                    "forces": torch.zeros(10, 3),
                    "predicted_forces": torch.zeros(10, 3),
                },
                "batch_idx",
                id="force_missing_batch_idx",
            ),
            pytest.param(
                lambda: ForceMSELoss(),
                {
                    "forces": torch.zeros(10, 3),
                    "predicted_forces": torch.zeros(10, 3),
                },
                "num_graphs",
                id="force_missing_num_graphs",
            ),
            pytest.param(
                lambda: ForceMSELoss(),
                {
                    "forces": torch.zeros(3, 5, 3),
                    "predicted_forces": torch.zeros(3, 5, 3),
                },
                "num_nodes_per_graph",
                id="force_missing_num_nodes",
            ),
        ],
    )
    def test_missing_loss_metadata_raises_value_error(
        self,
        loss_factory: Any,
        tensor_kwargs: dict[str, torch.Tensor],
        missing_attr: str,
    ) -> None:
        loss = loss_factory()
        batch = self._batch(**tensor_kwargs)
        # _batch sets all graph metadata fields; drop the one under test
        # to exercise the tensor-first metadata requirement path.
        delattr(batch, missing_attr)
        with pytest.raises(ValueError, match=missing_attr):
            _call_from_batch(loss, batch)

    def test_composed_losses_backprop_to_all_inputs(self) -> None:
        batch = _full_loss_batch()
        for name in ("energy", "forces", "stress"):
            setattr(batch, name, torch.randn_like(getattr(batch, name)))
        for name in ("predicted_energy", "predicted_forces", "predicted_stress"):
            setattr(
                batch, name, torch.randn_like(getattr(batch, name)).requires_grad_()
            )

        composed = (
            EnergyMSELoss()
            + ConstantWeight(value=10.0) * ForceMSELoss()
            + ConstantWeight(value=0.1) * StressMSELoss()
        )
        assert isinstance(composed, ComposedLossFunction)
        assert len(composed.components) == 3
        out = _call_from_batch(composed, batch)
        assert set(out) == {
            "total_loss",
            "per_component_unweighted",
            "per_component_weight",
            "per_component_raw_weight",
            "per_component_sample",
        }
        assert set(out["per_component_unweighted"]) == {
            "EnergyMSELoss",
            "ForceMSELoss",
            "StressMSELoss",
        }
        out["total_loss"].backward()
        for grad in (
            batch.predicted_energy.grad,
            batch.predicted_forces.grad,
            batch.predicted_stress.grad,
        ):
            assert grad is not None
            assert not torch.all(grad == 0)

    def test_force_loss_reads_from_configured_prediction_key(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        renamed_pred = torch.ones(self.num_nodes, 3)
        predictions = {"my_model_forces": renamed_pred}
        targets = {"forces": target}
        loss = ComposedLossFunction(
            components=(ForceMSELoss(prediction_key="my_model_forces"),)
        )
        got = loss(
            predictions,
            targets,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        # |pred - target|^2 sum over 3 components = 3 per atom.
        # per-graph mean = 3; mean over graphs = 3; / 3 = 1.0.
        # Single-component composition normalizes effective weight to 1.0.
        assert torch.allclose(got["total_loss"], torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(
            got["per_component_unweighted"]["ForceMSELoss"], torch.tensor(1.0)
        )

    def test_force_loss_resolves_from_batch_dense(self) -> None:
        pred = torch.randn(self.num_nodes, 3)
        target = torch.randn(self.num_nodes, 3)
        mini_batch = self._batch()

        got_batch = ForceMSELoss()(pred, target, batch=mini_batch)
        got_explicit = ForceMSELoss()(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        assert torch.allclose(got_batch, got_explicit, atol=1e-6)

    def test_energy_loss_per_atom_resolves_from_batch(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        mini_batch = self._batch()

        got_batch = EnergyMSELoss(per_atom=True)(pred, target, batch=mini_batch)
        got_explicit = EnergyMSELoss(per_atom=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        assert torch.allclose(got_batch, got_explicit, atol=1e-6)

    def test_force_loss_explicit_kwarg_overrides_batch(self) -> None:
        pred = torch.randn(self.num_nodes, 3)
        target = torch.randn(self.num_nodes, 3)
        mini_batch = self._batch()

        # Collapse to a single graph so the override path produces a
        # measurably different loss from the batch-derived grouping.
        override_batch_idx = torch.zeros(self.num_nodes, dtype=torch.int32)
        override_num_graphs = 1

        got_override = ForceMSELoss()(
            pred,
            target,
            batch=mini_batch,
            batch_idx=override_batch_idx,
            num_graphs=override_num_graphs,
        )
        got_direct = ForceMSELoss()(
            pred,
            target,
            batch_idx=override_batch_idx,
            num_graphs=override_num_graphs,
        )
        got_batch_only = ForceMSELoss()(pred, target, batch=mini_batch)

        assert torch.allclose(got_override, got_direct, atol=1e-6)
        assert not torch.allclose(got_override, got_batch_only, atol=1e-6)

    def test_energy_loss_per_atom_explicit_override_wins(self) -> None:
        target = torch.tensor([[3.0], [10.0], [4.0]])
        pred = torch.tensor([[6.0], [15.0], [8.0]])
        mini_batch = self._batch()

        # Flat 1-atom counts produce a different per-atom scale than the
        # batch-derived [3, 5, 2] counts, making the override observable.
        override_counts = torch.tensor([1, 1, 1], dtype=torch.long)

        got_override = EnergyMSELoss(per_atom=True)(
            pred, target, batch=mini_batch, num_nodes_per_graph=override_counts
        )
        got_direct = EnergyMSELoss(per_atom=True)(
            pred, target, num_nodes_per_graph=override_counts
        )
        got_batch_only = EnergyMSELoss(per_atom=True)(pred, target, batch=mini_batch)

        assert torch.allclose(got_override, got_direct, atol=1e-6)
        assert not torch.allclose(got_override, got_batch_only, atol=1e-6)


class TestPerSampleLoss:
    # ---- Shared helpers ----------------------------------------------

    @staticmethod
    def _assert_per_sample(
        loss: BaseLossFunction, expected_shape: tuple[int, ...]
    ) -> torch.Tensor:
        ps = loss.per_sample_loss
        assert ps is not None
        assert ps.shape == expected_shape
        assert ps.requires_grad is False
        return ps

    @pytest.mark.parametrize(
        ("kwargs", "extra", "expected_fn"),
        [
            pytest.param(
                {},
                {},
                lambda pred, target, counts: (pred - target).pow(2).squeeze(-1),
                id="default",
            ),
            pytest.param(
                {"per_atom": True},
                {"num_nodes_per_graph": torch.tensor([2, 3, 1], dtype=torch.long)},
                lambda pred, target, counts: (
                    ((pred - target) / counts.to(pred).unsqueeze(-1)).pow(2).squeeze(-1)
                ),
                id="per_atom_residuals",
            ),
        ],
    )
    def test_energy_loss_per_sample_populated_detached_shape_and_value(
        self,
        kwargs: dict[str, Any],
        extra: dict[str, torch.Tensor],
        expected_fn: Any,
    ) -> None:
        torch.manual_seed(0)
        b = 3
        pred = torch.randn(b, 1, requires_grad=True)
        target = torch.randn(b, 1)
        counts = extra.get("num_nodes_per_graph")
        loss = EnergyMSELoss(**kwargs)
        scalar = loss(pred, target, **extra)
        ps = self._assert_per_sample(loss, (b,))
        torch.testing.assert_close(ps, expected_fn(pred, target, counts))
        if counts is None:
            # Default energy MSE is graph-balanced.
            torch.testing.assert_close(ps.mean(), scalar)
        else:
            # ``per_atom=True`` stores per-graph squared per-atom residuals
            # for diagnostics, while the scalar is atom-count weighted.
            weights = counts.to(ps)
            torch.testing.assert_close(ps.mul(weights).sum() / weights.sum(), scalar)

    def test_energy_loss_per_sample_ignore_nonfinite_populates(self) -> None:
        """``ignore_nonfinite`` populates ``(B,)`` with zero on all-NaN rows.

        Kept as a distinct case: ``per_sample_loss.mean()`` does NOT equal
        the scalar return here because the scalar divides by the global
        valid-entry count while the per-sample view is per-row residual.
        """
        pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        target = torch.tensor([[0.0], [float("nan")], [2.5], [float("nan")]])
        loss = EnergyMSELoss(ignore_nonfinite=True)
        loss(pred, target)
        ps = self._assert_per_sample(loss, (4,))
        assert ps[1].item() == 0.0
        assert ps[3].item() == 0.0
        torch.testing.assert_close(ps[0], torch.tensor(1.0))
        torch.testing.assert_close(ps[2], torch.tensor(0.25))

    @pytest.mark.parametrize(
        "ignore_nonfinite", [False, True], ids=["default", "ignore_nonfinite"]
    )
    def test_stress_loss_per_sample_populated_detached_shape_and_mean(
        self, ignore_nonfinite: bool
    ) -> None:
        torch.manual_seed(0)
        b = 3
        pred = torch.randn(b, 3, 3, requires_grad=True)
        target = torch.randn(b, 3, 3)
        loss = StressMSELoss(ignore_nonfinite=ignore_nonfinite)
        scalar = loss(pred, target)
        ps = self._assert_per_sample(loss, (b,))
        expected = (pred - target).pow(2).mean(dim=(-2, -1))
        torch.testing.assert_close(ps, expected)
        torch.testing.assert_close(ps.mean(), scalar)

    def test_stress_loss_ignore_nonfinite_all_nan_row_is_zero(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(3, 3, 3)
        target = torch.randn(3, 3, 3)
        target[1] = float("nan")
        loss = StressMSELoss(ignore_nonfinite=True)
        loss(pred, target)
        ps = self._assert_per_sample(loss, (3,))
        assert ps[1].item() == 0.0
        for g in (0, 2):
            expected = (pred[g] - target[g]).pow(2).mean()
            torch.testing.assert_close(ps[g], expected)

    @pytest.mark.parametrize(
        ("normalize", "layout"),
        [
            pytest.param(True, "dense", id="dense_normalize"),
            pytest.param(True, "padded", id="padded_normalize"),
            pytest.param(False, "padded", id="padded_no_normalize"),
        ],
    )
    def test_force_loss_per_sample_populated_detached_shape_and_value(
        self, normalize: bool, layout: str
    ) -> None:
        torch.manual_seed(0)
        loss = ForceMSELoss(normalize_by_atom_count=normalize)
        if layout == "dense":
            v = 5
            batch_idx = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int32)
            num_graphs = 3
            pred = torch.randn(v, 3, requires_grad=True)
            target = torch.randn(v, 3)
            loss(pred, target, batch_idx=batch_idx, num_graphs=num_graphs)
            ps = self._assert_per_sample(loss, (num_graphs,))
            per_atom_se = (pred - target).pow(2).sum(dim=-1)
            per_atom_valid = torch.ones(v, dtype=pred.dtype) * 3
            per_graph_num = per_graph_sum(per_atom_se, batch_idx, num_graphs=num_graphs)
            per_graph_den = per_graph_sum(
                per_atom_valid, batch_idx, num_graphs=num_graphs
            ).clamp_min(1.0)
            expected = per_graph_num / per_graph_den
            torch.testing.assert_close(ps, expected)
            return
        # padded layout shared by both normalize=True and normalize=False.
        b = 3
        v_max = 4
        pred = torch.randn(b, v_max, 3, requires_grad=True)
        target = torch.randn(b, v_max, 3)
        num_nodes_per_graph = torch.tensor([2, 1, 4], dtype=torch.long)
        scalar = loss(pred, target, num_nodes_per_graph=num_nodes_per_graph)
        ps = self._assert_per_sample(loss, (b,))
        node_mask = torch.arange(v_max).unsqueeze(0) < num_nodes_per_graph.unsqueeze(-1)
        valid = node_mask.unsqueeze(-1).expand_as(pred).to(dtype=pred.dtype)
        squared_error = ((pred - target) * valid).pow(2)
        per_graph_num = squared_error.sum(dim=(-2, -1))
        per_graph_den = valid.sum(dim=(-2, -1)).clamp_min(1.0)
        expected = per_graph_num / per_graph_den
        torch.testing.assert_close(ps, expected)
        if normalize:
            # Normalized path: per-sample mean equals the scalar return.
            torch.testing.assert_close(ps.mean(), scalar)

    def test_force_loss_dense_no_normalize_per_sample_is_none(self) -> None:
        torch.manual_seed(0)
        pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        loss = ForceMSELoss(normalize_by_atom_count=False)
        loss(pred, target)
        assert loss.per_sample_loss is None

    def test_per_sample_loss_cleared_on_each_forward_call(self) -> None:
        torch.manual_seed(0)
        loss = ForceMSELoss(normalize_by_atom_count=False)
        padded_pred = torch.randn(3, 4, 3)
        padded_target = torch.randn(3, 4, 3)
        num_nodes_per_graph = torch.tensor([2, 1, 4], dtype=torch.long)
        loss(padded_pred, padded_target, num_nodes_per_graph=num_nodes_per_graph)
        assert loss.per_sample_loss is not None
        loss(torch.randn(5, 3), torch.randn(5, 3))
        assert loss.per_sample_loss is None

    def test_per_sample_loss_cleared_on_exception(self) -> None:
        torch.manual_seed(0)
        loss = EnergyMSELoss()
        loss(torch.randn(3, 1), torch.randn(3, 1))
        assert loss.per_sample_loss is not None
        pred = torch.randn(3, 1, dtype=torch.float32)
        target = torch.randn(3, 1, dtype=torch.float64)
        with pytest.raises(ValueError):
            loss(pred, target)
        assert loss.per_sample_loss is None

    @staticmethod
    def _energy_stress_inputs(
        b: int, requires_grad: bool = False
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        predictions = {
            "predicted_energy": torch.randn(b, 1, requires_grad=requires_grad),
            "predicted_stress": torch.randn(b, 3, 3, requires_grad=requires_grad),
        }
        targets = {
            "energy": torch.randn(b, 1),
            "stress": torch.randn(b, 3, 3),
        }
        return predictions, targets

    def test_composed_output_has_per_component_sample_field(self) -> None:
        torch.manual_seed(0)
        b = 3
        composed = EnergyMSELoss() + StressMSELoss()
        out = composed(*self._energy_stress_inputs(b))
        assert set(out["per_component_sample"]) == set(out["per_component_unweighted"])
        for value in out["per_component_sample"].values():
            assert value.shape == (b,)
            assert value.requires_grad is False

    def test_composed_per_component_sample_is_weighted_by_effective_weight(
        self,
    ) -> None:
        torch.manual_seed(0)
        b = 3
        energy = EnergyMSELoss()
        stress = StressMSELoss()
        composed = ComposedLossFunction(
            (energy, stress), weights=[3.0, 1.0], normalize_weights=False
        )
        pred_e = torch.randn(b, 1)
        tgt_e = torch.randn(b, 1)
        pred_s = torch.randn(b, 3, 3)
        tgt_s = torch.randn(b, 3, 3)
        out = composed(
            {"predicted_energy": pred_e, "predicted_stress": pred_s},
            {"energy": tgt_e, "stress": tgt_s},
        )
        assert energy.per_sample_loss is not None
        expected_energy = 3.0 * energy.per_sample_loss
        torch.testing.assert_close(
            out["per_component_sample"]["EnergyMSELoss"], expected_energy
        )

    def test_composed_component_without_per_sample_is_absent(self) -> None:
        torch.manual_seed(0)
        v = 5
        b = 3
        composed = ComposedLossFunction(
            (EnergyMSELoss(), ForceMSELoss(normalize_by_atom_count=False))
        )
        predictions = {
            "predicted_energy": torch.randn(b, 1),
            "predicted_forces": torch.randn(v, 3),
        }
        targets = {
            "energy": torch.randn(b, 1),
            "forces": torch.randn(v, 3),
        }
        out = composed(predictions, targets)
        assert "EnergyMSELoss" in out["per_component_sample"]
        assert "ForceMSELoss" not in out["per_component_sample"]

    def test_composed_per_component_sample_sum_matches_total_loss(self) -> None:
        torch.manual_seed(0)
        b = 4
        composed = EnergyMSELoss() + StressMSELoss()
        predictions, targets = self._energy_stress_inputs(b)
        out = composed(predictions, targets)
        per_sample_sum = sum(out["per_component_sample"].values())
        torch.testing.assert_close(per_sample_sum.mean(), out["total_loss"])

    @pytest.mark.parametrize(
        ("bad_value", "expected_exc", "expected_msg_fragment"),
        [
            (1.0, TypeError, "must be a torch.Tensor or None"),
            (torch.zeros(2, 3), ValueError, "must be a 1-D tensor"),
        ],
        ids=["non_tensor_raises_type_error", "non_1d_tensor_raises_value_error"],
    )
    def test_composed_rejects_invalid_custom_per_sample_loss(
        self,
        bad_value: Any,
        expected_exc: type[BaseException],
        expected_msg_fragment: str,
    ) -> None:
        class _BadPerSampleLoss(_ToyLoss):
            def __init__(self, value: Any) -> None:
                super().__init__()
                self._value = value

            def forward(
                self,
                pred: torch.Tensor,
                target: torch.Tensor,
                **kwargs: Any,
            ) -> torch.Tensor:
                out = super().forward(pred, target, **kwargs)
                self.per_sample_loss = self._value
                return out

        composed = ComposedLossFunction((_BadPerSampleLoss(bad_value),))
        with pytest.raises(expected_exc, match=expected_msg_fragment):
            composed({"prediction": torch.tensor(0.0)}, {"target": torch.tensor(0.0)})


class TestIgnoreNaN:
    """Tests for the opt-in ``ignore_nonfinite`` masking in concrete losses.

    Targets with ``NaN`` represent missing labels and must not contribute
    to loss value or gradient. Predictions are assumed finite. The
    implementation uses branch-free tensor ops, so behavior is the same
    as the eager path these tests assert.
    """

    def setup_method(self) -> None:
        # Reuse the 3,5,2-atom layout from ``TestConcreteLosses`` so per-atom
        # normalization and multi-graph masking paths are all exercised.
        self.nodes_per_graph = [3, 5, 2]
        self.num_graphs = 3
        self.num_nodes = sum(self.nodes_per_graph)
        self.batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2], dtype=torch.int32)
        self.num_nodes_per_graph = torch.tensor(self.nodes_per_graph, dtype=torch.long)

    def _batch(self, **extra: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
            num_nodes_per_graph=self.num_nodes_per_graph,
            **extra,
        )

    # ---- EnergyMSELoss ---------------------------------------------------

    def test_energy_loss_default_propagates_nan(self) -> None:
        target = torch.tensor([[1.0], [float("nan")], [3.0]])
        pred = torch.tensor([[1.5], [2.5], [3.5]])
        got = EnergyMSELoss()(pred, target)
        assert torch.isnan(got)

    def test_energy_loss_ignore_nonfinite_masks_missing_targets(self) -> None:
        target = torch.tensor([[1.0], [float("nan")], [3.0]])
        pred = torch.tensor([[1.5], [2.5], [3.5]])
        got = EnergyMSELoss(ignore_nonfinite=True)(pred, target)
        # Valid entries contribute (0.5)^2 and (0.5)^2; two valid entries.
        expected = torch.tensor((0.25 + 0.25) / 2.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_loss_ignore_nonfinite_zero_gradient_at_nan_positions(self) -> None:
        target = torch.tensor([[1.0], [float("nan")], [3.0]])
        pred = torch.tensor([[1.5], [10.0], [3.5]], requires_grad=True)
        EnergyMSELoss(ignore_nonfinite=True)(pred, target).backward()
        assert pred.grad is not None
        # The NaN-target entry must receive exactly zero gradient.
        assert pred.grad[1].item() == 0.0
        # Other entries must receive finite, non-zero gradient.
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[0].item() != 0.0
        assert pred.grad[2].item() != 0.0

    def test_energy_loss_ignore_nonfinite_all_nan_gives_zero(self) -> None:
        target = torch.full((self.num_graphs, 1), float("nan"))
        pred = torch.randn(self.num_graphs, 1, requires_grad=True)
        got = EnergyMSELoss(ignore_nonfinite=True)(pred, target)
        assert torch.allclose(got, torch.tensor(0.0))
        got.backward()
        assert pred.grad is not None
        assert torch.all(pred.grad == 0.0)

    def test_energy_loss_ignore_nonfinite_per_atom_weights_valid_counts(self) -> None:
        # Per-atom normalization must be applied before masking so the
        # valid-entry MSE is computed on per-atom values, not raw energies.
        target = torch.tensor([[3.0], [float("nan")], [4.0]])  # per-atom: 1, -, 2
        pred = torch.tensor([[6.0], [15.0], [8.0]])  # per-atom: 2, 3, 4
        got = EnergyMSELoss(per_atom=True, ignore_nonfinite=True)(
            pred, target, num_nodes_per_graph=self.num_nodes_per_graph
        )
        # Valid per-atom diffs: (2-1)=1 and (4-2)=2. Only valid graph
        # counts enter the denominator: (3*1 + 2*4) / (3 + 2).
        expected = torch.tensor(11.0 / 5.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_energy_loss_ignore_nonfinite_off_matches_baseline(self) -> None:
        target = torch.randn(self.num_graphs, 1)
        pred = torch.randn(self.num_graphs, 1)
        baseline = EnergyMSELoss()(pred, target)
        opt_in = EnergyMSELoss(ignore_nonfinite=True)(pred, target)
        assert torch.allclose(baseline, opt_in, atol=1e-6)

    # ---- ForceMSELoss ----------------------------------------------------

    def test_force_loss_default_propagates_nan(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        target[4, 1] = float("nan")
        pred = torch.ones(self.num_nodes, 3)
        assert torch.isnan(
            ForceMSELoss(normalize_by_atom_count=True)(
                pred,
                target,
                batch_idx=self.batch_idx,
                num_graphs=self.num_graphs,
            )
        )
        assert torch.isnan(ForceMSELoss(normalize_by_atom_count=False)(pred, target))

    def test_force_loss_ignore_nonfinite_global_masks_missing_components(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        target[4, 1] = float("nan")  # one component missing
        pred = torch.ones(self.num_nodes, 3)
        got = ForceMSELoss(normalize_by_atom_count=False, ignore_nonfinite=True)(
            pred, target
        )
        # V*3 - 1 = 29 valid entries, each contributing (1 - 0)^2 = 1.
        expected = torch.tensor(29.0 / 29.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_ignore_nonfinite_per_graph_all_nan_graph_zero_contribution(
        self,
    ) -> None:
        # Graph 1 (atoms 3..7) has fully-NaN force labels; graphs 0 and 2
        # are fully labeled. The all-NaN graph must contribute zero to the
        # mean over graphs.
        target = torch.zeros(self.num_nodes, 3)
        target[3:8] = float("nan")
        pred = torch.ones(self.num_nodes, 3)
        got = ForceMSELoss(normalize_by_atom_count=True, ignore_nonfinite=True)(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        # Graph 0: 3 atoms * 3 components all valid, each (1-0)^2 = 1,
        # per-graph loss = 9/9 = 1. Graph 2: 2 atoms * 3 components all
        # valid, per-graph loss = 6/6 = 1. Graph 1: all NaN, loss = 0.
        # Mean over 3 graphs = (1 + 0 + 1) / 3.
        expected = torch.tensor(2.0 / 3.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_ignore_nonfinite_per_graph_partial_mask(self) -> None:
        # Single component missing on one atom; check the per-graph
        # denominator reflects 3*n_atoms - missing, not 3*n_atoms.
        target = torch.zeros(self.num_nodes, 3)
        target[0, 0] = float("nan")  # graph 0, atom 0, x component
        pred = torch.ones(self.num_nodes, 3)
        got = ForceMSELoss(normalize_by_atom_count=True, ignore_nonfinite=True)(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        )
        # Graph 0: 8 valid components (out of 9), each contributes 1; loss = 8/8 = 1.
        # Graph 1: 15/15 = 1. Graph 2: 6/6 = 1. Mean = 1.0.
        expected = torch.tensor(1.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_force_loss_ignore_nonfinite_zero_gradient_at_nan_positions(self) -> None:
        target = torch.zeros(self.num_nodes, 3)
        target[0, 0] = float("nan")
        pred = torch.randn(self.num_nodes, 3, requires_grad=True)
        ForceMSELoss(normalize_by_atom_count=True, ignore_nonfinite=True)(
            pred,
            target,
            batch_idx=self.batch_idx,
            num_graphs=self.num_graphs,
        ).backward()
        assert pred.grad is not None
        assert pred.grad[0, 0].item() == 0.0
        # At least one other component receives non-zero gradient.
        assert torch.isfinite(pred.grad).all()
        assert (pred.grad != 0.0).any()

    def test_force_loss_ignore_nonfinite_off_matches_baseline(
        self, fixed_torch_seed: None
    ) -> None:
        target = torch.randn(self.num_nodes, 3)
        pred = torch.randn(self.num_nodes, 3)
        for norm in (True, False):
            metadata = (
                {"batch_idx": self.batch_idx, "num_graphs": self.num_graphs}
                if norm
                else {}
            )
            baseline = ForceMSELoss(normalize_by_atom_count=norm)(
                pred, target, **metadata
            )
            opt_in = ForceMSELoss(normalize_by_atom_count=norm, ignore_nonfinite=True)(
                pred, target, **metadata
            )
            assert torch.allclose(baseline, opt_in, atol=1e-6)

    # ---- StressMSELoss ---------------------------------------------------

    def test_stress_loss_default_propagates_nan(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[1, 2, 2] = float("nan")
        pred = torch.ones(self.num_graphs, 3, 3)
        assert torch.isnan(StressMSELoss()(pred, target))

    def test_stress_loss_ignore_nonfinite_all_nan_graph_zero_contribution(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[1] = float("nan")  # full graph 1 unlabeled
        pred = torch.ones(self.num_graphs, 3, 3)
        got = StressMSELoss(ignore_nonfinite=True)(pred, target)
        # Graph 0: 9 valid entries each (1-0)^2 = 1, per-graph loss = 9/9 = 1.
        # Graph 1: all NaN, loss = 0. Graph 2: loss = 1.
        # Mean = (1 + 0 + 1) / 3.
        expected = torch.tensor(2.0 / 3.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_stress_loss_ignore_nonfinite_partial_mask(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[0, 0, 0] = float("nan")  # one entry missing in graph 0
        pred = torch.ones(self.num_graphs, 3, 3)
        got = StressMSELoss(ignore_nonfinite=True)(pred, target)
        # Graph 0: 8 valid entries of (1-0)^2 = 1 -> loss = 8/8 = 1.
        # Graphs 1, 2: loss = 1. Mean = 1.0.
        expected = torch.tensor(1.0)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_stress_loss_ignore_nonfinite_zero_gradient_at_nan_positions(self) -> None:
        target = torch.zeros(self.num_graphs, 3, 3)
        target[0, 0, 0] = float("nan")
        pred = torch.randn(self.num_graphs, 3, 3, requires_grad=True)
        StressMSELoss(ignore_nonfinite=True)(pred, target).backward()
        assert pred.grad is not None
        assert pred.grad[0, 0, 0].item() == 0.0
        assert torch.isfinite(pred.grad).all()

    def test_stress_loss_ignore_nonfinite_off_matches_baseline(
        self, fixed_torch_seed: None
    ) -> None:
        target = torch.randn(self.num_graphs, 3, 3)
        pred = torch.randn(self.num_graphs, 3, 3)
        baseline = StressMSELoss()(pred, target)
        opt_in = StressMSELoss(ignore_nonfinite=True)(pred, target)
        assert torch.allclose(baseline, opt_in, atol=1e-6)

    # ---- Repr ---------------------------------------------------------

    def test_ignore_nonfinite_appears_in_extra_repr(self) -> None:
        for loss in (
            EnergyMSELoss(ignore_nonfinite=True),
            ForceMSELoss(ignore_nonfinite=True),
            StressMSELoss(ignore_nonfinite=True),
        ):
            assert "ignore_nonfinite=True" in repr(loss)


class TestLossModelSpec:
    """Tests for :func:`create_model_spec` round-trip on concrete losses.

    Since leaf losses no longer carry a ``weight`` kwarg (weighting lives
    on :class:`ComposedLossFunction`), these tests exercise the generic
    spec workflow for plain concrete-loss kwargs only:
    ``create_model_spec(cls, **kwargs)`` → ``model_dump_json`` →
    ``json.loads`` → :func:`create_model_spec_from_json` →
    ``spec.build()``. The rebuilt instance must preserve ``__init__``
    kwargs and stay functionally equivalent on tensor inputs.
    """

    def _roundtrip(self, spec: Any) -> Any:
        dumped = json.loads(spec.model_dump_json())
        return create_model_spec_from_json(dumped)

    @pytest.mark.parametrize(
        ("cls", "kwargs"),
        [
            pytest.param(EnergyMSELoss, {}, id="energy_defaults"),
            pytest.param(
                EnergyMSELoss,
                {"per_atom": True, "ignore_nonfinite": True},
                id="energy_per_atom_ignore_nonfinite",
            ),
            pytest.param(
                EnergyMSELoss,
                {"target_key": "u_ref", "prediction_key": "u_hat"},
                id="energy_renamed_keys",
            ),
            pytest.param(
                EnergyHuberLoss,
                {"per_atom": True, "delta": 0.5, "ignore_nonfinite": True},
                id="energy_huber",
            ),
            pytest.param(ForceMSELoss, {}, id="force_defaults"),
            pytest.param(
                ForceMSELoss,
                {"normalize_by_atom_count": False, "ignore_nonfinite": True},
                id="force_global_ignore_nonfinite",
            ),
            pytest.param(
                ForceHuberLoss,
                {
                    "normalize_by_atom_count": False,
                    "delta": 0.5,
                    "ignore_nonfinite": True,
                },
                id="force_huber",
            ),
            pytest.param(StressMSELoss, {}, id="stress_defaults"),
            pytest.param(
                StressMSELoss, {"ignore_nonfinite": True}, id="stress_ignore_nonfinite"
            ),
            pytest.param(
                StressHuberLoss,
                {"delta": 0.5, "ignore_nonfinite": True},
                id="stress_huber",
            ),
        ],
    )
    def test_loss_basespec_roundtrip(
        self, cls: type[BaseLossFunction], kwargs: dict[str, Any]
    ) -> None:
        """JSON round-trip rebuilds a loss with matching kwargs."""
        spec = create_model_spec(cls, **kwargs)
        rebuilt = self._roundtrip(spec)
        built = rebuilt.build()
        assert isinstance(built, cls)
        for k, v in kwargs.items():
            assert getattr(built, k) == v
        # Leaves no longer carry a ``weight`` attribute — weighting lives
        # on :class:`ComposedLossFunction`.
        assert not hasattr(built, "weight")

    def test_loss_spec_preserves_timestamp(self) -> None:
        """Rehydrated spec keeps the original timestamp byte-for-byte."""
        spec = create_model_spec(EnergyMSELoss, per_atom=True)
        rebuilt = self._roundtrip(spec)
        assert rebuilt.timestamp == spec.timestamp

    def test_rebuilt_loss_is_functionally_equivalent(self) -> None:
        """A round-tripped loss produces the same value as the original."""
        pred = torch.randn(3, 1)
        target = torch.randn(3, 1)
        original = EnergyMSELoss(ignore_nonfinite=True)
        spec = create_model_spec(EnergyMSELoss, ignore_nonfinite=True)
        rebuilt = self._roundtrip(spec).build()

        assert torch.allclose(original(pred, target), rebuilt(pred, target), atol=1e-6)

    def test_loss_component_to_spec_roundtrip(self) -> None:
        """Public loss component spec helper round-trips leaf loss config."""
        spec = loss_component_to_spec(
            EnergyMSELoss(per_atom=True, ignore_nonfinite=True)
        )
        rebuilt = self._roundtrip(spec).build()
        assert isinstance(rebuilt, EnergyMSELoss)
        assert rebuilt.per_atom is True
        assert rebuilt.ignore_nonfinite is True

    def test_huber_loss_component_to_spec_roundtrip(self) -> None:
        """Public loss component spec helper round-trips Huber loss config."""
        spec = loss_component_to_spec(ForceHuberLoss(delta=0.5))
        rebuilt = self._roundtrip(spec).build()
        assert isinstance(rebuilt, ForceHuberLoss)
        assert rebuilt.delta == 0.5

    def test_loss_component_to_spec_rejects_composed_loss(self) -> None:
        """Public loss component spec helper rejects non-leaf compositions."""
        with pytest.raises(
            TypeError,
            match="use ComposedLossFunction spec serialization for composed losses",
        ):
            loss_component_to_spec(ComposedLossFunction([EnergyMSELoss()]))

    def test_loss_component_to_spec_rejects_non_loss(self) -> None:
        """Public loss component spec helper rejects non-loss objects clearly."""
        with pytest.raises(
            TypeError,
            match="loss_component_to_spec accepts only leaf BaseLossFunction objects",
        ):
            loss_component_to_spec(object())


class TestShapeValidationOptIn:
    def test_bare_subclass_does_not_shape_check(self) -> None:
        loss = _ToyLoss(value=1.0)
        pred = torch.randn(3, 1)
        target = torch.randn(4, 5, 7)  # deliberately mismatched
        got = loss(pred, target)
        assert torch.allclose(got, torch.tensor(1.0))

    def test_energy_loss_raises_on_shape_mismatch(self) -> None:
        loss = EnergyMSELoss()
        pred = torch.zeros(3, 2)
        target = torch.zeros(3, 3)  # unequal trailing dim (strict)
        with pytest.raises(
            ValueError,
            match="EnergyMSELoss: prediction and target shape must match exactly",
        ):
            loss(pred, target)

    def test_assert_same_shape_public_helper(self) -> None:
        pred = torch.zeros(3, 2)
        target = torch.zeros(3, 2)
        assert_same_shape(
            pred,
            target,
            name="MyLoss",
            prediction_key="p",
            target_key="t",
        )
        with pytest.raises(
            ValueError,
            match=r"MyLoss: prediction and target shape mismatch; "
            r"prediction_key='p' has shape \(3, 2\), "
            r"target_key='t' has shape \(3, 3\)",
        ):
            assert_same_shape(
                pred,
                torch.zeros(3, 3),
                name="MyLoss",
                prediction_key="p",
                target_key="t",
            )

    def test_assert_same_shape_omits_key_fragments_when_none(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"MyLoss: prediction and target shape mismatch; "
            r"prediction has shape \(3, 2\), target has shape \(3, 3\)",
        ):
            assert_same_shape(
                torch.zeros(3, 2),
                torch.zeros(3, 3),
                name="MyLoss",
            )

    def test_assert_same_shape_accepts_broadcastable(self) -> None:
        # (B, 1) vs (B, 3) is broadcast-compatible; must not raise.
        assert_same_shape(
            torch.zeros(4, 1),
            torch.zeros(4, 3),
            name="MyLoss",
            prediction_key="p",
            target_key="t",
        )

    def test_assert_same_shape_rejects_dtype_mismatch(self) -> None:
        pred = torch.zeros(3, 2, dtype=torch.float32)
        target = torch.zeros(3, 2, dtype=torch.float64)
        with pytest.raises(
            ValueError,
            match=r"MyLoss: prediction and target dtype mismatch; "
            r"prediction_key='p' has dtype torch\.float32, "
            r"target_key='t' has dtype torch\.float64",
        ):
            assert_same_shape(
                pred,
                target,
                name="MyLoss",
                prediction_key="p",
                target_key="t",
            )

    def test_assert_same_shape_dtype_check_runs_before_shape_check(self) -> None:
        # Both dtype AND shape mismatch — must surface dtype error, not shape.
        pred = torch.zeros(3, 2, dtype=torch.float32)
        target = torch.zeros(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="dtype mismatch"):
            assert_same_shape(
                pred,
                target,
                name="MyLoss",
            )


class TestLeafShapeEqualityGuard:
    # The module-level ``assert_same_shape`` helper has two shape
    # policies: its default broadcast-compatible mode accepts shapes
    # like ``(B, 1)`` vs ``(B, 3)``, and ``strict=True`` requires exact
    # equality. All built-in leaf losses opt into ``strict=True`` so
    # the broadcast trap cannot silently corrupt their elementwise
    # arithmetic. These tests lock that in.

    def test_energy_loss_rejects_broadcast_trap(self) -> None:
        # ``(B, 1)`` vs ``(B, 3)`` is broadcast-compatible but broadcasts
        # into a ``(B, 3)`` residual — silently triples the loss.
        loss = EnergyMSELoss()
        pred = torch.zeros(4, 1)
        target = torch.zeros(4, 3)
        with pytest.raises(
            ValueError,
            match=(
                r"EnergyMSELoss: prediction and target shape must match exactly "
                r"for elementwise loss; prediction_key='predicted_energy' has "
                r"shape \(4, 1\), target_key='energy' has shape \(4, 3\)"
            ),
        ):
            loss(pred, target)

    def test_energy_loss_rejects_squeezed_vs_unsqueezed(self) -> None:
        # ``(B, 1)`` vs ``(B,)`` broadcasts to a ``(B, B)`` outer product.
        loss = EnergyMSELoss()
        pred = torch.zeros(4, 1)
        target = torch.zeros(4)
        with pytest.raises(ValueError, match="shape must match exactly"):
            loss(pred, target)

    def test_energy_loss_happy_path(self) -> None:
        # Regression: canonical ``(B, 1)`` vs ``(B, 1)`` still works.
        loss = EnergyMSELoss()
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.5], [2.5], [3.5]])
        scalar = loss(pred, target)
        # Each residual squared is 0.25; mean is 0.25.
        assert torch.allclose(scalar, torch.tensor(0.25))

    def test_force_loss_dense_rejects_component_broadcast(self) -> None:
        # ``(V, 1)`` vs ``(V, 3)`` is broadcast-compatible but semantically
        # wrong — a per-atom scalar compared against a 3-component force.
        loss = ForceMSELoss(normalize_by_atom_count=False)
        pred = torch.zeros(5, 1)
        target = torch.zeros(5, 3)
        with pytest.raises(
            ValueError,
            match=(
                r"ForceMSELoss: prediction and target shape must match exactly "
                r"for elementwise loss; prediction_key='predicted_forces' has "
                r"shape \(5, 1\), target_key='forces' has shape \(5, 3\)"
            ),
        ):
            loss(pred, target)

    def test_force_loss_padded_rejects_component_broadcast(self) -> None:
        # Padded layout ``(B, V_max, 1)`` vs ``(B, V_max, 3)``.
        loss = ForceMSELoss(normalize_by_atom_count=False)
        pred = torch.zeros(2, 4, 1)
        target = torch.zeros(2, 4, 3)
        with pytest.raises(ValueError, match="shape must match exactly"):
            loss(
                pred,
                target,
                num_nodes_per_graph=torch.tensor([4, 4]),
            )

    def test_force_loss_dense_happy_path(self) -> None:
        # Regression: canonical dense ``(V, 3)`` vs ``(V, 3)`` still works.
        loss = ForceMSELoss(normalize_by_atom_count=False)
        pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        scalar = loss(pred, target)
        # Sum of squares 1 + 4 = 5 over 6 components → 5/6.
        assert torch.allclose(scalar, torch.tensor(5.0 / 6.0))

    def test_stress_loss_rejects_component_broadcast(self) -> None:
        # ``(B, 1, 3)`` vs ``(B, 3, 3)`` is broadcast-compatible.
        loss = StressMSELoss()
        pred = torch.zeros(2, 1, 3)
        target = torch.zeros(2, 3, 3)
        with pytest.raises(
            ValueError,
            match=(
                r"StressMSELoss: prediction and target shape must match exactly "
                r"for elementwise loss; prediction_key='predicted_stress' has "
                r"shape \(2, 1, 3\), target_key='stress' has shape \(2, 3, 3\)"
            ),
        ):
            loss(pred, target)

    def test_stress_loss_happy_path(self) -> None:
        # Regression: canonical ``(B, 3, 3)`` vs ``(B, 3, 3)`` still works.
        loss = StressMSELoss()
        pred = torch.zeros(2, 3, 3)
        target = torch.zeros(2, 3, 3)
        scalar = loss(pred, target)
        assert torch.allclose(scalar, torch.tensor(0.0))

    def test_assert_same_shape_strict_rejects_broadcast_compatible(self) -> None:
        # Direct test of the public helper's strict policy: shapes that
        # pass the default broadcast policy must be rejected.
        with pytest.raises(
            ValueError,
            match=(
                r"MyLoss: prediction and target shape must match exactly "
                r"for elementwise loss; prediction_key='p' has shape "
                r"\(4, 1\), target_key='t' has shape \(4, 3\)"
            ),
        ):
            assert_same_shape(
                torch.zeros(4, 1),
                torch.zeros(4, 3),
                name="MyLoss",
                prediction_key="p",
                target_key="t",
                strict=True,
            )

    def test_assert_same_shape_strict_accepts_equal(self) -> None:
        # Strict policy must accept exact-equal shapes.
        assert_same_shape(
            torch.zeros(4, 3),
            torch.zeros(4, 3),
            name="MyLoss",
            strict=True,
        )

    def test_assert_same_shape_strict_still_checks_dtype_first(self) -> None:
        # Strict policy shares the dtype-before-shape ordering.
        with pytest.raises(ValueError, match="dtype mismatch"):
            assert_same_shape(
                torch.zeros(4, 1, dtype=torch.float32),
                torch.zeros(4, 3, dtype=torch.float64),
                name="MyLoss",
                strict=True,
            )
