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

* :class:`~nvalchemi.models.gen.base.GenerativeModelConfig` construction,
  validation (intents-in-map), ``input_modalities``/``output_modalities``
  properties, and config round-trip (serialize -> deserialize -> equality).
* :class:`~nvalchemi.models.gen.base.GenerativeModelMixin` contract via a tiny
  demo subclass: ``model_config`` enforcement, ``forward``/
  ``adapt_output`` (emitting ``{"flow": velocity}``), ``condition`` replication
  by ``num_samples``, and the optional ``to_batch``/``prior_template`` hooks.

These tests are CPU-only, GPU-free and import no optional deps.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.enums import GenerativeIntent, Modality
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
        self.model_config = GenerativeModelConfig(
            intents={GenerativeIntent.CREATE, GenerativeIntent.SAMPLE},
            supports_variable_atoms=True,
            output_artifact=Modality.CRYSTAL,
            intent_modality_map={
                GenerativeIntent.CREATE: frozenset({Modality.CRYSTAL}),
                GenerativeIntent.SAMPLE: frozenset({Modality.CRYSTAL}),
            },
            consumes_fields=frozenset({"positions", "atomic_numbers"}),
            produces_fields=frozenset({"positions", "atomic_numbers"}),
        )

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

    def to_batch(self, sample: TensorDict, cond_batch: Batch) -> Batch:
        """Reconstruct by returning the conditioning batch unchanged.

        Parameters
        ----------
        sample
            Sample TensorDict (unused by this toy model).
        cond_batch
            Conditioning batch.

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


class TestGenerativeModelConfig:
    """Tests for :class:`GenerativeModelConfig`."""

    @staticmethod
    def _build_cfg() -> GenerativeModelConfig:
        """Build a representative config used across the tests.

        Returns
        -------
        GenerativeModelConfig
            A config with create/sample (output) and condition (input) intents.
        """
        return GenerativeModelConfig(
            intents={
                GenerativeIntent.CREATE,
                GenerativeIntent.SAMPLE,
                GenerativeIntent.CONDITION,
            },
            supports_variable_atoms=True,
            output_artifact=Modality.CRYSTAL,
            intent_modality_map={
                GenerativeIntent.CREATE: frozenset({Modality.CRYSTAL}),
                GenerativeIntent.SAMPLE: frozenset({Modality.CRYSTAL}),
                GenerativeIntent.CONDITION: frozenset({Modality.TEXT}),
            },
            consumes_fields=frozenset({"positions", "atomic_numbers"}),
            produces_fields=frozenset({"positions", "atomic_numbers"}),
        )

    def test_construction(self) -> None:
        """Config accepts the documented fields and defaults."""
        cfg = self._build_cfg()
        assert cfg.supports_variable_atoms is True
        assert cfg.output_artifact is Modality.CRYSTAL
        assert cfg.active_prediction_outputs is None

    def test_intents_in_map_validator_rejects_mismatch(self) -> None:
        """A missing intent in the map raises :class:`ValueError`."""
        with pytest.raises(ValueError, match="missing from intent_modality_map"):
            GenerativeModelConfig(
                intents={GenerativeIntent.CREATE, GenerativeIntent.PROPOSE},
                supports_variable_atoms=True,
                output_artifact=Modality.CRYSTAL,
                intent_modality_map={
                    GenerativeIntent.CREATE: frozenset({Modality.CRYSTAL}),
                },
                consumes_fields=frozenset(),
                produces_fields=frozenset({"positions"}),
            )

    def test_input_and_output_modalities(self) -> None:
        """``output_modalities`` covers output intents + artifact; input covers the rest."""
        cfg = self._build_cfg()
        assert cfg.output_modalities == frozenset({Modality.CRYSTAL})
        assert cfg.input_modalities == frozenset({Modality.TEXT})

    def test_config_round_trip(self) -> None:
        """Serialize -> deserialize -> equality ."""
        cfg = self._build_cfg()
        restored = GenerativeModelConfig.model_validate(cfg.model_dump())
        assert restored == cfg
        assert restored.input_modalities == cfg.input_modalities
        assert restored.output_modalities == cfg.output_modalities

    def test_active_prediction_outputs_round_trip(self) -> None:
        """A non-default ``active_prediction_outputs`` survives a round-trip."""
        cfg = GenerativeModelConfig(
            intents={GenerativeIntent.CREATE},
            supports_variable_atoms=False,
            output_artifact=Modality.POINT_CLOUD,
            intent_modality_map={
                GenerativeIntent.CREATE: frozenset({Modality.POINT_CLOUD}),
            },
            consumes_fields=frozenset(),
            produces_fields=frozenset({"positions"}),
            active_prediction_outputs={"flow"},
        )
        restored = GenerativeModelConfig.model_validate(cfg.model_dump())
        assert restored == cfg
        assert restored.active_prediction_outputs == {"flow"}

    def test_field_declarations_required(self) -> None:
        """Omitting ``consumes_fields``/``produces_fields`` raises."""
        with pytest.raises(ValueError, match="consumes_fields"):
            GenerativeModelConfig(
                intents={GenerativeIntent.CREATE},
                supports_variable_atoms=True,
                output_artifact=Modality.CRYSTAL,
                intent_modality_map={
                    GenerativeIntent.CREATE: frozenset({Modality.CRYSTAL}),
                },
            )


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

    def test_condition_replicates_by_num_samples(self) -> None:
        """``condition`` tiles each conditioning graph ``num_samples`` times."""
        model = _DemoGenerativeModel()
        batch = _make_batch(num_graphs=2)
        cond = model.condition(batch, num_samples=3)
        assert isinstance(cond, Batch)
        assert cond.num_graphs == 6

    def test_condition_single_atomic_data(self) -> None:
        """``condition`` on an :class:`AtomicData` builds a replicated batch."""
        model = _DemoGenerativeModel()
        ad = _make_atomic_data()
        cond = model.condition(ad, num_samples=4)
        assert cond is not None
        assert cond.num_graphs == 4

    def test_condition_passes_through_tensor_containers(self) -> None:
        """The default condition passes a TensorDict through unchanged."""
        model = _DemoGenerativeModel()
        container = TensorDict({"emb": torch.randn(4, 8)}, batch_size=[4])
        assert model.condition(container, num_samples=3) is container

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
        """``extra_repr`` summarizes the generative config."""
        model = _DemoGenerativeModel()
        rep = model.extra_repr()
        assert "create" in rep
        assert "crystal" in rep
