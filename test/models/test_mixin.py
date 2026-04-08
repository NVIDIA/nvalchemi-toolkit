# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseModelMixin wrapper contract and PipelineContext.resolve_optional."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig, PipelineContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeBatch:
    """Minimal batch-like object for testing resolution."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DefaultsOnlyModel(BaseModelMixin, nn.Module):
    """Wrapper that relies entirely on inherited adapt_input / adapt_output."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers"}),
        optional_inputs=frozenset({"cell"}),
        outputs=frozenset({"energies"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
    )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"energies": data["positions"].sum().unsqueeze(0).unsqueeze(0)}


class _ExtendedModel(BaseModelMixin, nn.Module):
    """Wrapper that extends adapt_input via super()."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers"}),
        optional_inputs=frozenset({"cell"}),
        outputs=frozenset({"energies", "extra"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
    )

    def adapt_input(
        self,
        batch: Any,
        ctx: PipelineContext,
        *,
        compute: set[str] | None = None,
    ) -> dict[str, Any]:
        data = super().adapt_input(batch, ctx, compute=compute)
        data["doubled_positions"] = data["positions"] * 2
        return data

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            "energies": data["positions"].sum().unsqueeze(0).unsqueeze(0),
            "extra": data["doubled_positions"].sum().unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# PipelineContext.resolve_optional
# ---------------------------------------------------------------------------


class TestResolveOptional:
    """Tests for PipelineContext.resolve_optional."""

    def test_returns_from_context_first(self) -> None:
        ctx = PipelineContext()
        ctx["positions"] = torch.ones(3, 3)
        batch = _FakeBatch(positions=torch.zeros(3, 3))
        result = ctx.resolve_optional("positions", batch)
        assert torch.equal(result, torch.ones(3, 3))

    def test_falls_back_to_batch(self) -> None:
        ctx = PipelineContext()
        batch = _FakeBatch(positions=torch.zeros(3, 3))
        result = ctx.resolve_optional("positions", batch)
        assert torch.equal(result, torch.zeros(3, 3))

    def test_returns_default_when_missing(self) -> None:
        ctx = PipelineContext()
        batch = _FakeBatch()
        result = ctx.resolve_optional("missing_key", batch)
        assert result is None

    def test_returns_custom_default(self) -> None:
        ctx = PipelineContext()
        batch = _FakeBatch()
        sentinel = object()
        result = ctx.resolve_optional("missing_key", batch, default=sentinel)
        assert result is sentinel


# ---------------------------------------------------------------------------
# BaseModelMixin.adapt_input (defaults)
# ---------------------------------------------------------------------------


class TestAdaptInputDefaults:
    """Tests for the default adapt_input implementation."""

    def setup_method(self) -> None:
        self.model = _DefaultsOnlyModel()
        self.ctx = PipelineContext()

    def test_resolves_required_from_batch(self) -> None:
        batch = _FakeBatch(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([6, 1, 1, 1]),
        )
        data = self.model.adapt_input(batch, self.ctx)
        assert "positions" in data
        assert "atomic_numbers" in data
        assert torch.equal(data["positions"], batch.positions)

    def test_resolves_required_from_context_over_batch(self) -> None:
        ctx_positions = torch.randn(4, 3)
        self.ctx["positions"] = ctx_positions
        batch = _FakeBatch(
            positions=torch.zeros(4, 3),
            atomic_numbers=torch.tensor([6, 1, 1, 1]),
        )
        data = self.model.adapt_input(batch, self.ctx)
        assert torch.equal(data["positions"], ctx_positions)

    def test_resolves_optional_when_present(self) -> None:
        batch = _FakeBatch(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([6, 1, 1, 1]),
            cell=torch.eye(3),
        )
        data = self.model.adapt_input(batch, self.ctx)
        assert "cell" in data
        assert torch.equal(data["cell"], batch.cell)

    def test_skips_optional_when_absent(self) -> None:
        batch = _FakeBatch(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([6, 1, 1, 1]),
        )
        data = self.model.adapt_input(batch, self.ctx)
        assert "cell" not in data

    def test_raises_on_missing_required(self) -> None:
        batch = _FakeBatch(positions=torch.randn(4, 3))
        with pytest.raises(KeyError, match="atomic_numbers"):
            self.model.adapt_input(batch, self.ctx)


# ---------------------------------------------------------------------------
# BaseModelMixin.adapt_output (defaults)
# ---------------------------------------------------------------------------


class TestAdaptOutputDefaults:
    """Tests for the default adapt_output implementation."""

    def setup_method(self) -> None:
        self.model = _DefaultsOnlyModel()

    def test_passthrough_when_compute_none(self) -> None:
        raw = {"energies": torch.tensor([1.0]), "undeclared": torch.tensor([2.0])}
        result = self.model.adapt_output(raw)
        assert result is raw

    def test_filters_by_compute(self) -> None:
        raw = {"energies": torch.tensor([1.0])}
        result = self.model.adapt_output(raw, compute={"energies"})
        assert "energies" in result

    def test_filters_out_unrequested(self) -> None:
        raw = {"energies": torch.tensor([1.0]), "extra": torch.tensor([2.0])}
        result = self.model.adapt_output(raw, compute={"energies"})
        assert "energies" in result
        assert "extra" not in result

    def test_empty_compute_returns_empty(self) -> None:
        raw = {"energies": torch.tensor([1.0])}
        result = self.model.adapt_output(raw, compute=set())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Extended wrapper using super()
# ---------------------------------------------------------------------------


class TestExtendedAdaptInput:
    """Tests for a wrapper that overrides adapt_input and calls super()."""

    def test_super_plus_extension(self) -> None:
        model = _ExtendedModel()
        ctx = PipelineContext()
        batch = _FakeBatch(
            positions=torch.ones(2, 3),
            atomic_numbers=torch.tensor([1, 1]),
        )
        data = model.adapt_input(batch, ctx)
        assert "positions" in data
        assert "doubled_positions" in data
        assert torch.equal(data["doubled_positions"], torch.ones(2, 3) * 2)


# ---------------------------------------------------------------------------
# Autograd override transparency
# ---------------------------------------------------------------------------


class TestAutogradOverrideTransparency:
    """Tests that resolve/resolve_optional transparently return autograd overrides."""

    def test_resolve_returns_autograd_positions(self) -> None:
        ctx = PipelineContext()
        batch = _FakeBatch(
            positions=torch.zeros(3, 3),
            atomic_numbers=torch.tensor([1, 1, 1]),
        )
        ctx.activate_autograd(batch, frozenset({"positions"}))
        resolved = ctx.resolve("positions", batch)
        assert resolved.requires_grad

    def test_resolve_optional_returns_autograd_positions(self) -> None:
        ctx = PipelineContext()
        batch = _FakeBatch(
            positions=torch.zeros(3, 3),
            atomic_numbers=torch.tensor([1, 1, 1]),
        )
        ctx.activate_autograd(batch, frozenset({"positions"}))
        resolved = ctx.resolve_optional("positions", batch)
        assert resolved.requires_grad

    def test_adapt_input_gets_grad_tracked_positions(self) -> None:
        model = _DefaultsOnlyModel()
        ctx = PipelineContext()
        batch = _FakeBatch(
            positions=torch.zeros(4, 3),
            atomic_numbers=torch.tensor([6, 1, 1, 1]),
        )
        ctx.activate_autograd(batch, frozenset({"positions"}))
        data = model.adapt_input(batch, ctx)
        assert data["positions"].requires_grad


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


class TestModelConfigValidation:
    """Tests for the tightened ModelConfig validation rules."""

    def test_disjoint_inputs_ok(self) -> None:
        ModelConfig(
            required_inputs=frozenset({"positions"}),
            optional_inputs=frozenset({"cell"}),
        )

    def test_overlapping_inputs_raises(self) -> None:
        with pytest.raises(
            ValueError, match="required_inputs and optional_inputs overlap"
        ):
            ModelConfig(
                required_inputs=frozenset({"positions", "cell"}),
                optional_inputs=frozenset({"cell"}),
            )

    def test_autograd_additive_overlap_raises(self) -> None:
        with pytest.raises(
            ValueError, match="autograd_outputs and additive_outputs overlap"
        ):
            ModelConfig(
                required_inputs=frozenset({"positions"}),
                outputs=frozenset({"energies"}),
                autograd_outputs=frozenset({"energies"}),
                additive_outputs=frozenset({"energies"}),
            )

    def test_autograd_additive_disjoint_ok(self) -> None:
        ModelConfig(
            required_inputs=frozenset({"positions"}),
            outputs=frozenset({"energies", "special"}),
            autograd_outputs=frozenset({"special"}),
            additive_outputs=frozenset({"energies"}),
        )
