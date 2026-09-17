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
"""Structural tests for the generative model config and mixin.

Covers:

* :class:`~nvalchemi.models.gen.base.GenerativeModelConfig` construction and
  validation of its four capability fields (``supports_variable_atoms``,
  ``consumes_fields``, ``produces_fields``, ``prediction_outputs``),
  ``extra="forbid"`` rejection, subclassing, and config round-trip
  (serialize -> deserialize -> equality).
* :class:`~nvalchemi.models.gen.base.GenerativeModelMixin` contract via a tiny
  demo subclass: ``model_config`` enforcement (including subclassed configs),
  ``forward``/``adapt_output`` (emitting ``{"flow": velocity}``),
  ``condition`` replication by ``num_samples``, and the optional
  ``to_batch``/``prior_template`` hooks.

These tests are CPU-only, GPU-free and import no optional deps.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict
from torch import Tensor, nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.gen.base import (
    GenerativeModelConfig,
    GenerativeModelMixin,
)


def _make_atomic_data(num_atoms: int = 3) -> AtomicData:
    """Build a minimal :class:`AtomicData` for tests.

    Parameters
    ----------
    num_atoms
        Number of atoms in the dummy structure.

    Returns
    -------
    AtomicData
        A small structure with random positions and carbon atomic numbers.
    """
    return AtomicData(
        positions=torch.randn(num_atoms, 3),
        atomic_numbers=torch.full((num_atoms,), 6, dtype=torch.long),
    )


def _make_batch(num_graphs: int = 2) -> Batch:
    """Build a small :class:`Batch` for tests.

    Parameters
    ----------
    num_graphs
        Number of graphs to batch.

    Returns
    -------
    Batch
        A batch of dummy structures.
    """
    return Batch.from_data_list([_make_atomic_data() for _ in range(num_graphs)])


def _build_cfg(**overrides) -> GenerativeModelConfig:
    """Build a valid config, with overrides.

    Parameters
    ----------
    **overrides
        Field overrides.

    Returns
    -------
    GenerativeModelConfig
        The config.
    """
    fields = {
        "supports_variable_atoms": True,
        "consumes_fields": frozenset({"positions", "atomic_numbers"}),
        "produces_fields": frozenset({"positions", "atomic_numbers"}),
    }
    fields.update(overrides)
    return GenerativeModelConfig(**fields)


class _DemoGenerativeModel(nn.Module, GenerativeModelMixin):
    """Tiny generative model for contract tests.

    A constant-velocity flow model: the state is one 3-vector per graph
    (shape ``(B, 1, 3)``), and the predicted velocity drives the state toward
    a fixed target. It implements the full required surface plus the optional
    ``to_batch`` and ``prior_template`` hooks.
    """

    def __init__(self, target: Tensor | None = None) -> None:
        super().__init__()
        self.target = target if target is not None else torch.tensor([[1.0, 0.0, 0.0]])
        self.model_config = _build_cfg()

    def forward(
        self,
        data: Batch,
        *,
        x: Tensor,
        t: Tensor | float,
        xsc: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Return a raw velocity toward ``self.target``.

        Parameters
        ----------
        data
            Conditioning batch (unused by this toy model).
        x
            Current flow state ``(B, 1, 3)``.
        t
            Flow timestep (unused).
        xsc
            Self-conditioning state (unused).
        **kwargs
            Forwarded arguments (unused).

        Returns
        -------
        Tensor
            Velocity ``target - x`` broadcast over the batch.
        """
        del data, t, xsc, kwargs
        target = self.target.to(x.device, dtype=x.dtype).expand_as(x)
        return target - x

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None = None) -> Batch:
        """Reconstruct by returning the conditioning batch unchanged.

        Parameters
        ----------
        sample
            Sample TensorDict (unused by this toy model).
        cond_batch
            Conditioning batch, if any.

        Returns
        -------
        Batch
            ``cond_batch`` unchanged.
        """
        del sample
        return cond_batch

    def prior_template(self, cond_batch: Batch) -> Tensor:
        """Return a zero template state ``(B, 1, 3)``.

        Parameters
        ----------
        cond_batch
            Conditioning batch.

        Returns
        -------
        Tensor
            Zeros shaped ``(num_graphs, 1, 3)``.
        """
        return torch.zeros(cond_batch.num_graphs, 1, 3)


class _ExtendedConfig(GenerativeModelConfig):
    """A model-specific config subclass (the documented extension pattern)."""

    temperature: float = 1.0


class TestGenerativeModelConfig:
    """``GenerativeModelConfig``: the four-field capability surface."""

    def test_construction(self) -> None:
        """The four fields validate; ``prediction_outputs`` defaults to None."""
        cfg = _build_cfg()
        assert cfg.supports_variable_atoms is True
        assert cfg.consumes_fields == frozenset({"positions", "atomic_numbers"})
        assert cfg.produces_fields == frozenset({"positions", "atomic_numbers"})
        assert cfg.prediction_outputs is None

    def test_unknown_kwarg_rejected(self) -> None:
        """``extra="forbid"``: retired and unknown fields alike raise."""
        with pytest.raises(ValidationError):
            _build_cfg(intents={"create"})
        with pytest.raises(ValidationError):
            _build_cfg(bogus_field=1)

    def test_field_declarations_required(self) -> None:
        """Omitting ``consumes_fields``/``produces_fields`` raises."""
        with pytest.raises(ValidationError, match="consumes_fields"):
            GenerativeModelConfig(supports_variable_atoms=True)
        with pytest.raises(ValidationError, match="produces_fields"):
            GenerativeModelConfig(
                supports_variable_atoms=True, consumes_fields=frozenset()
            )

    def test_config_round_trip(self) -> None:
        """Serialize -> deserialize -> equality."""
        cfg = _build_cfg()
        restored = GenerativeModelConfig.model_validate(cfg.model_dump())
        assert restored == cfg

    def test_prediction_outputs_round_trip(self) -> None:
        """A non-default ``prediction_outputs`` survives a round-trip."""
        cfg = _build_cfg(
            supports_variable_atoms=False,
            consumes_fields=frozenset(),
            produces_fields=frozenset({"positions"}),
            prediction_outputs={"flow"},
        )
        restored = GenerativeModelConfig.model_validate(cfg.model_dump())
        assert restored == cfg
        assert restored.prediction_outputs == {"flow"}

    def test_subclassed_config(self) -> None:
        """A subclass carrying model-specific fields validates as a config."""
        cfg = _ExtendedConfig(
            supports_variable_atoms=False,
            consumes_fields=frozenset(),
            produces_fields=frozenset({"positions"}),
            temperature=2.5,
        )
        assert isinstance(cfg, GenerativeModelConfig)
        assert cfg.temperature == 2.5


class TestGenerativeModelMixin:
    """Tests for :class:`GenerativeModelMixin` via the demo subclass."""

    def test_model_config_enforced_at_construction(self) -> None:
        """A subclass that forgets ``model_config`` raises :class:`TypeError`."""

        class _BadModel(nn.Module, GenerativeModelMixin):
            def __init__(self) -> None:
                super().__init__()
                # intentionally no self.model_config

            def forward(self, data, *, x, t, xsc=None, **kwargs):  # noqa: ANN001
                del data, x, t, xsc, kwargs
                return x

        with pytest.raises(TypeError, match="must set"):
            _BadModel()

    def test_subclassed_config_passes_enforcement(self) -> None:
        """The ``isinstance`` enforcement accepts config subclasses."""

        class _ExtendedModel(nn.Module, GenerativeModelMixin):
            def __init__(self) -> None:
                super().__init__()
                self.model_config = _ExtendedConfig(
                    supports_variable_atoms=True,
                    consumes_fields=frozenset(),
                    produces_fields=frozenset({"positions"}),
                )

            def forward(self, data, *, x, t, xsc=None, **kwargs):  # noqa: ANN001
                del data, x, t, xsc, kwargs
                return x

        model = _ExtendedModel()
        assert model.model_config.temperature == 1.0

    def test_forward_returns_raw_velocity(self) -> None:
        """``forward`` returns a raw tensor (not ``ModelOutputs``)."""
        model = _DemoGenerativeModel()
        batch = _make_batch(num_graphs=2)
        x = torch.zeros(2, 1, 3)
        raw = model.forward(batch, x=x, t=0.5)
        assert isinstance(raw, Tensor)
        assert raw.shape == (2, 1, 3)

    def test_adapt_output_emits_flow_key(self) -> None:
        """``adapt_output`` structures raw output under the ``"flow"`` key."""
        model = _DemoGenerativeModel()
        batch = _make_batch(num_graphs=2)
        raw = torch.randn(2, 1, 3)
        out = model.adapt_output(raw, batch)
        assert isinstance(out, OrderedDict)
        assert "flow" in out
        assert out["flow"] is raw

    def test_adapt_output_uses_prediction_outputs(self) -> None:
        """``prediction_outputs`` drives the ``adapt_output`` key set."""
        model = _DemoGenerativeModel()
        model.model_config = _build_cfg(prediction_outputs={"flow", "score"})
        batch = _make_batch(num_graphs=2)
        raw = {"flow": torch.randn(2, 1, 3), "score": torch.randn(2, 1, 3)}
        out = model.adapt_output(raw, batch)
        assert set(out) == {"flow", "score"}
        assert out["flow"] is raw["flow"]
        assert out["score"] is raw["score"]

    def test_to_batch_and_prior_template_hooks(self) -> None:
        """Optional ``to_batch`` and ``prior_template`` hooks work as documented."""
        model = _DemoGenerativeModel()
        batch = _make_batch(num_graphs=3)
        template = model.prior_template(batch)
        assert template.shape == (3, 1, 3)
        sample = TensorDict({"x1": torch.randn(3, 1, 3)}, batch_size=[3])
        recon = model.to_batch(sample, batch)
        assert recon is batch

    def test_extra_repr_uses_config(self) -> None:
        """``extra_repr`` summarizes the declared field contracts."""
        model = _DemoGenerativeModel()
        rep = model.extra_repr()
        assert "consumes_fields" in rep
        assert "positions" in rep
        assert "produces_fields" in rep

    def test_extra_repr_with_empty_declarations(self) -> None:
        """``extra_repr`` renders empty declarations cleanly."""
        model = _DemoGenerativeModel()
        model.model_config = _build_cfg(
            consumes_fields=frozenset(), produces_fields=frozenset({"positions"})
        )
        rep = model.extra_repr()
        assert "consumes_fields={}" in rep
        assert "positions" in rep
