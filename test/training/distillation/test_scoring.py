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
"""Tests for :mod:`nvalchemi.training.distillation.scoring`."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.models.pipeline import (
    _PIPELINE_NEIGHBOR_SOURCES_ATTR,
    PipelineGroup,
    PipelineModelWrapper,
)
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.distillation import scoring
from nvalchemi.training.distillation.scoring import (
    _PIPELINE_SOURCES_ATTR,
    _SIGNAL_SPECS,
    SUPPORTED_SIGNALS,
    InProcessTeacherScorer,
    TeacherLabels,
    TeacherScorer,
    _SignalSpec,
    scorer_fields,
    signal_fields,
    signal_for_field,
)
from test.training.distillation.conftest import (
    _LJ_CUTOFF,
    _build_lj_teacher,
    _build_periodic_batch,
    _DirectForceTeacher,
    _PairPotentialTeacher,
)

_COO_CUTOFF = 4.0
"""Cutoff of the COO stub teacher used in the neighbor-isolation tests."""

_SHADOW_CUTOFF = 3.9
"""Cutoff of the shadowed list a composed pipeline leaves on a live batch."""

_STUDENT_CUTOFFS = (3.5, 4.0)
"""Composed-student cutoffs, both inside the lattice batch's first pair shell."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spread_batch(n_systems: int = 2, n_atoms: int = 6) -> Batch:
    """Return a non-periodic batch whose atoms are spread over a few angstroms."""
    data_list = []
    for index in range(n_systems):
        generator = torch.Generator().manual_seed(index)
        data_list.append(
            AtomicData(
                positions=torch.rand(n_atoms, 3, generator=generator) * 6.0,
                atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
                atomic_masses=torch.ones(n_atoms),
                energy=torch.zeros(1, 1),
            )
        )
    return Batch.from_data_list(data_list)


def _make_lattice_batch(n_systems: int = 2, spacing: float = 3.0) -> Batch:
    """Return a jittered cubic-lattice batch whose neighbor shells straddle a cutoff.

    Pairs sit at roughly one, one-and-a-half, and one-and-three-quarter lattice
    spacings, so a list built at :data:`_SHADOW_CUTOFF` holds strictly fewer
    pairs than one built at the teachers' cutoffs — and none sits close enough
    to put the Lennard-Jones teacher on its singularity.
    """
    grid = torch.tensor(
        list(itertools.product(range(2), repeat=3)), dtype=torch.float32
    )
    data_list = []
    for index in range(n_systems):
        generator = torch.Generator().manual_seed(index)
        positions = grid * spacing + 0.1 * torch.rand(grid.shape, generator=generator)
        data_list.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(positions.shape[0], dtype=torch.long),
                atomic_masses=torch.ones(positions.shape[0]),
            )
        )
    return Batch.from_data_list(data_list)


def _shadow_dense_neighbors(batch: Batch, cutoff: float) -> dict[str, Any]:
    """Shadow a dense list built at *cutoff* onto *batch*, the way a pipeline does."""
    source = batch.clone()
    compute_neighbors(source, cutoff=cutoff, format=NeighborListFormat.MATRIX)
    shadows: dict[str, Any] = {
        "neighbor_matrix": source.neighbor_matrix,
        "num_neighbors": source.num_neighbors,
        "_neighbor_list_cutoff": cutoff,
    }
    batch.__dict__.update(shadows)
    return shadows


def _build_declared_neighbors(
    batch: Batch,
    cutoff: float,
    neighbor_format: NeighborListFormat,
    half_list: bool = False,
) -> None:
    """Build a neighbor list on *batch* and declare its half-list provenance."""
    compute_neighbors(batch, cutoff=cutoff, format=neighbor_format, half_list=half_list)
    batch._neighbor_list_half = half_list


def _storage_snapshot(batch: Batch) -> dict[str, torch.Tensor]:
    """Return a value copy of every tensor currently stored on *batch*."""
    return {key: value.clone() for key, value in batch}


def _tracked_snapshot(batch: Batch) -> dict[str, set[str]]:
    """Return a copy of the batch's per-level tracked key sets."""
    return {level: set(names) for level, names in batch.keys.items()}


@contextmanager
def _spy_on_neighbor_builds() -> Iterator[Mock]:
    """Yield a spy counting the neighbor-list builds the scorer performs."""
    with patch.object(
        scoring, "compute_neighbors", wraps=scoring.compute_neighbors
    ) as spy:
        yield spy


def _composed_teacher(*cutoffs: float, **kwargs: Any) -> PipelineModelWrapper:
    """Return a Lennard-Jones composition, one step per cutoff in *cutoffs*."""
    return PipelineModelWrapper(
        [
            PipelineGroup(
                steps=[_build_lj_teacher(cutoff=cutoff) for cutoff in cutoffs],
                use_autograd=False,
            )
        ],
        **kwargs,
    )


def _pipeline_prepared_batch(model: PipelineModelWrapper, batch: Batch) -> Batch:
    """Run *model*'s neighbor hooks over *batch* the way a dynamics step does."""
    ctx = SimpleNamespace(batch=batch)
    for hook in model.make_neighbor_hooks():
        hook(ctx, DynamicsStage.BEFORE_COMPUTE)
    return batch


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class _GradModeRecorder:
    """Delegating callable that records the ambient grad mode at each call."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.grad_enabled: list[bool] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record whether grad is enabled, then delegate to the wrapped callable."""
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.inner(*args, **kwargs)


class _TrainingModeRecorder:
    """Delegating callable that records a module's training flag at each call."""

    def __init__(self, module: torch.nn.Module, inner: Any) -> None:
        self.module = module
        self.inner = inner
        self.training: list[bool] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record whether the module is in training mode, then delegate."""
        self.training.append(self.module.training)
        return self.inner(*args, **kwargs)


class _CooNeighborTeacher(torch.nn.Module, BaseModelMixin):
    """Teacher requiring a sparse (COO) neighbor list, returning an edge-count energy."""

    def __init__(self, cutoff: float = _COO_CUTOFF) -> None:
        super().__init__()
        self.seen_storage_keys: set[str] = set()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=NeighborListFormat.COO
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Return a per-graph energy proportional to the edge count."""
        self.seen_storage_keys = {key for key, _ in data}
        edges = data.neighbor_list
        energy = torch.full(
            (data.num_graphs, 1), float(edges.shape[0]), dtype=data.positions.dtype
        )
        return OrderedDict([("energy", energy)])


class _AddKeyEmbeddingTeacher(BaseModelMixin):
    """Embedding-only teacher attaching both embedding levels via ``Batch.add_key``.

    Mirrors the in-repo wrappers that publish embeddings through ``add_key``,
    which registers the key in ``batch.keys`` and rejects an existing key. It
    is deliberately not an ``nn.Module``, so it also covers a teacher that has
    no ``eval()``.
    """

    hidden_dim = 4

    def __init__(self) -> None:
        self.model_config = ModelConfig(outputs=frozenset({"energy"}))

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the per-node embedding shape published by this teacher."""
        return {"node_embeddings": (self.hidden_dim,)}

    def compute_embeddings(self, data: Batch, **kwargs: Any) -> Batch:  # noqa: ARG002
        """Attach node and graph embeddings derived from the atomic numbers."""
        node = (
            data.atomic_numbers.reshape(-1, 1)
            .to(torch.float32)
            .repeat(1, self.hidden_dim)
        )
        data.add_key(
            "node_embeddings",
            list(torch.split(node, data.num_nodes_list)),
            level="node",
        )
        graph = torch.zeros(data.num_graphs, self.hidden_dim)
        graph.scatter_add_(0, data.batch_idx.unsqueeze(-1).expand_as(node), node)
        data.add_key(
            "graph_embeddings",
            [graph[index : index + 1] for index in range(data.num_graphs)],
            level="system",
        )
        return data


class _NonMutatingEmbeddingTeacher(BaseModelMixin):
    """Teacher whose ``compute_embeddings`` returns a copy instead of mutating input."""

    def __init__(self) -> None:
        self.model_config = ModelConfig(outputs=frozenset({"energy"}))

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the per-node embedding shape published by this teacher."""
        return {"node_embeddings": (2,)}

    def compute_embeddings(self, data: Batch, **kwargs: Any) -> Batch:  # noqa: ARG002
        """Return a clone carrying embeddings, leaving *data* untouched."""
        clone = data.clone()
        clone._atoms_group["node_embeddings"] = torch.zeros(clone.num_nodes, 2)
        return clone


class _EmptyOutputTeacher(torch.nn.Module, BaseModelMixin):
    """Teacher that declares an energy output but returns ``None`` for it."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}), autograd_inputs=frozenset()
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Return an output dict whose only entry is missing."""
        return OrderedDict([("energy", None)])


class _RaisingTeacher(torch.nn.Module, BaseModelMixin):
    """Neighbor-consuming teacher whose forward always raises."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=NeighborConfig(
                cutoff=_LJ_CUTOFF, format=NeighborListFormat.MATRIX
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Raise to exercise the scorer's rollback paths."""
        raise RuntimeError("teacher forward failed")


class _DeclaredFieldsScorer:
    """Scorer publishing ``label_fields`` its signal names alone would not imply."""

    signals = frozenset({"energy"})
    label_fields = ("teacher_energy", "teacher_energy_variance")

    def label(self, batch: Batch) -> TeacherLabels:  # noqa: ARG002
        """Return no labels, since only the declaration matters here."""
        return {}


class _UnknownSignalScorer:
    """Scorer over a signal name of its own, declaring only ``signals`` and ``label``."""

    signals = frozenset({"gaussian_energy"})

    def label(self, batch: Batch) -> TeacherLabels:  # noqa: ARG002
        """Return no labels, since only the signal set matters here."""
        return {}


class _FieldlessScorer:
    """Scorer over a custom signal that declares it populates no field at all."""

    signals = frozenset({"gaussian_energy"})
    label_fields = ()

    def label(self, batch: Batch) -> TeacherLabels:  # noqa: ARG002
        """Return no labels, matching the empty declaration."""
        return {}


class _StringFieldsScorer:
    """Scorer declaring its one field as a bare string rather than a sequence."""

    signals = frozenset({"energy"})
    label_fields = "teacher_energy"

    def label(self, batch: Batch) -> TeacherLabels:  # noqa: ARG002
        """Return no labels, since only the declaration matters here."""
        return {}


_REUSE_CASES = [
    (
        lambda batch: _build_declared_neighbors(
            batch, _LJ_CUTOFF, NeighborListFormat.MATRIX
        ),
        _build_lj_teacher,
        0,
    ),
    (
        lambda batch: compute_neighbors(
            batch, cutoff=_LJ_CUTOFF, format=NeighborListFormat.MATRIX
        ),
        _build_lj_teacher,
        1,
    ),
    (
        lambda batch: _build_declared_neighbors(batch, 8.0, NeighborListFormat.MATRIX),
        _build_lj_teacher,
        1,
    ),
    (
        lambda batch: _build_declared_neighbors(
            batch, _COO_CUTOFF, NeighborListFormat.COO
        ),
        _CooNeighborTeacher,
        0,
    ),
]
"""``(list builder, teacher builder, expected rebuild count)`` reuse cases.

The two format-mismatch cases are separate tests because they also assert which
neighbor data survives the cross-format rebuild.
"""

_REUSE_IDS = [
    "matching-matrix",
    "unknown-provenance",
    "wrong-cutoff",
    "matching-coo",
]
"""Readable ids for :data:`_REUSE_CASES`."""


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestSupportedSignals:
    """The public set of signal names a scorer accepts."""

    def test_supported_signals_are_exactly_the_names_the_scorer_accepts(
        self, demo_teacher: Any
    ) -> None:
        """Every published name is accepted, and only an unpublished one is reported."""
        with pytest.raises(ValueError, match=r"got unsupported \['bogus'\]"):
            InProcessTeacherScorer(demo_teacher, [*SUPPORTED_SIGNALS, "bogus"])


class TestSignalFields:
    """Resolution of signal names to the batch fields they populate."""

    def test_every_supported_signal_maps_to_a_teacher_prefixed_field(self) -> None:
        """Each supported signal populates at least one ``teacher_``-prefixed field."""
        for signal in SUPPORTED_SIGNALS:
            fields = signal_fields([signal])
            assert fields
            assert all(field.startswith("teacher_") for field in fields)

    def test_fields_are_sorted_and_deduplicated(self) -> None:
        """A repeated signal contributes its field once, and the result is sorted."""
        assert signal_fields(["forces", "energy", "forces"]) == (
            "teacher_energy",
            "teacher_forces",
        )

    def test_a_signal_populating_extra_fields_contributes_all_of_them(self) -> None:
        """A signal writing a companion field reports that field too."""
        spec = _SignalSpec(
            "energy", "teacher_energy", "system", ("teacher_energy_variance",)
        )
        with patch.dict(_SIGNAL_SPECS, {"energy": spec}):
            assert signal_fields(["energy"]) == (
                "teacher_energy",
                "teacher_energy_variance",
            )

    def test_an_unknown_signal_raises_listing_the_supported_names(self) -> None:
        """An unknown signal name raises, and the message lists the supported set."""
        with pytest.raises(KeyError, match="node_energies"):
            signal_fields(["energy", "bogus"])


class TestScorerFields:
    """Resolution of a scorer to the batch fields it populates."""

    def test_in_process_scorer_publishes_the_fields_its_signals_populate(
        self, demo_teacher: Any
    ) -> None:
        """The built-in scorer declares the fields its requested signals write."""
        scorer = InProcessTeacherScorer(demo_teacher, ["energy", "forces"])
        assert scorer.label_fields == ("teacher_energy", "teacher_forces")
        assert scorer_fields(scorer) == ("teacher_energy", "teacher_forces")

    def test_a_scorer_declaring_label_fields_is_taken_at_its_word(self) -> None:
        """A declaration wins over the fields the scorer's signals would imply."""
        assert scorer_fields(_DeclaredFieldsScorer()) == (
            "teacher_energy",
            "teacher_energy_variance",
        )

    def test_a_scorer_with_only_built_in_signals_falls_back_to_its_signal_fields(
        self, demo_teacher: Any
    ) -> None:
        """A scorer declaring nothing is resolved through its supported signals."""
        scorer = InProcessTeacherScorer(demo_teacher, ["energy", "forces"])
        del scorer.label_fields
        assert scorer_fields(scorer) == ("teacher_energy", "teacher_forces")

    def test_a_scorer_with_an_unknown_signal_and_no_declaration_is_unknown(
        self,
    ) -> None:
        """A custom signal name maps to fields only the scorer itself knows."""
        assert scorer_fields(_UnknownSignalScorer()) is None

    def test_a_scorer_that_labels_nothing_is_distinguishable_from_an_unknown_one(
        self,
    ) -> None:
        """An empty declaration means no fields, which is not the same as unknown."""
        assert scorer_fields(_FieldlessScorer()) == ()
        assert scorer_fields(_FieldlessScorer()) is not None

    def test_a_string_declaration_is_refused_rather_than_split_into_characters(
        self,
    ) -> None:
        """One field named as a bare string raises instead of resolving silently."""
        scorer = _StringFieldsScorer()
        with pytest.raises(TypeError, match="not a single string"):
            scorer_fields(scorer)


class TestTeacherScorerProtocol:
    """Structural membership of the :class:`TeacherScorer` protocol."""

    def test_a_scorer_declaring_only_signals_and_label_satisfies_the_protocol(
        self,
    ) -> None:
        """The protocol requires nothing beyond ``signals`` and ``label``."""
        assert isinstance(_UnknownSignalScorer(), TeacherScorer)


class TestSignalForField:
    """Reverse lookup from a batch field to the signal that populates it."""

    def test_every_field_including_extras_maps_back_to_its_signal(self) -> None:
        """A signal's field and its companion fields both resolve to that signal."""
        spec = _SignalSpec(
            "energy", "teacher_energy", "system", ("teacher_energy_variance",)
        )
        with patch.dict(_SIGNAL_SPECS, {"energy": spec}):
            for signal in SUPPORTED_SIGNALS:
                assert all(
                    signal_for_field(field) == signal
                    for field in signal_fields([signal])
                )

    def test_an_unknown_field_has_no_signal(self) -> None:
        """A field no signal populates resolves to ``None``."""
        assert signal_for_field("positions") is None


class TestInProcessTeacherScorerValidation:
    """Construction-time validation of the requested signal set."""

    def test_unknown_signal_name_raises(self, demo_teacher: Any) -> None:
        """An unrecognized signal name is rejected and named in the message."""
        with pytest.raises(ValueError, match="bogus"):
            InProcessTeacherScorer(demo_teacher, ["energy", "bogus"])

    def test_signal_the_teacher_cannot_produce_raises(self, demo_teacher: Any) -> None:
        """Requesting stress from an energy/forces teacher names the missing output."""
        with pytest.raises(ValueError, match="stress"):
            InProcessTeacherScorer(demo_teacher, ["energy", "stress"])

    def test_empty_signal_selection_raises(self, demo_teacher: Any) -> None:
        """A scorer with no signals is rejected."""
        with pytest.raises(ValueError, match="At least one teacher signal"):
            InProcessTeacherScorer(demo_teacher, [])

    def test_embeddings_from_teacher_without_embeddings_raises(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A teacher publishing no node-embedding shape cannot serve embeddings."""
        with pytest.raises(ValueError, match="node_embeddings"):
            InProcessTeacherScorer(lj_teacher, ["embeddings"])

    def test_scorer_satisfies_the_teacher_scorer_protocol(
        self, demo_teacher: Any
    ) -> None:
        """:class:`InProcessTeacherScorer` is a structural :class:`TeacherScorer`."""
        scorer = InProcessTeacherScorer(demo_teacher, ["energy"])
        assert isinstance(scorer, TeacherScorer)

    def test_teacher_planning_a_single_list_or_none_is_accepted(
        self, demo_teacher: Any, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A plain teacher, with one neighbor list or none at all, is not refused."""
        assert InProcessTeacherScorer(demo_teacher, ["energy"]).teacher is demo_teacher
        assert InProcessTeacherScorer(lj_teacher, ["energy"]).teacher is lj_teacher

    def test_cast_to_a_non_floating_dtype_raises(self, demo_teacher: Any) -> None:
        """An integer ``cast_to`` is rejected, since only float signals are cast."""
        with pytest.raises(ValueError, match="cast_to"):
            InProcessTeacherScorer(demo_teacher, ["energy"], cast_to=torch.int64)


class TestInProcessTeacherScorerLabeling:
    """Signal shapes, levels, and detachment produced by ``label()``."""

    def test_energy_and_forces_have_canonical_shapes_and_levels(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Energy is ``(B, 1)`` system-level and forces are ``(V, 3)`` node-level."""
        labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert labels["teacher_energy"][0].shape == (small_batch.num_graphs, 1)
        assert labels["teacher_energy"][1] == "system"
        assert labels["teacher_forces"][0].shape == (small_batch.num_nodes, 3)
        assert labels["teacher_forces"][1] == "node"

    def test_returned_tensors_are_detached(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """No returned tensor carries an autograd graph."""
        labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert all(value.requires_grad is False for value, _ in labels.values())

    def test_label_leaves_the_batch_unmodified(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Scoring changes no stored tensor, key set, or ``requires_grad`` flag."""
        before = _storage_snapshot(small_batch)
        flags = {key: value.requires_grad for key, value in small_batch}
        tracked = _tracked_snapshot(small_batch)
        InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(small_batch)
        after = {key: value for key, value in small_batch}
        assert set(after) == set(before)
        assert all(torch.equal(after[key], before[key]) for key in before)
        assert {key: value.requires_grad for key, value in after.items()} == flags
        assert _tracked_snapshot(small_batch) == tracked

    def test_labels_match_a_direct_teacher_forward(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Scored values equal the teacher's own forward outputs."""
        labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
            small_batch
        )
        expected = demo_teacher(small_batch)
        torch.testing.assert_close(labels["teacher_energy"][0], expected["energy"])
        torch.testing.assert_close(labels["teacher_forces"][0], expected["forces"])

    def test_direct_force_teacher_labels_energy_and_forces(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """A teacher with no autograd outputs still yields detached forces."""
        labels = InProcessTeacherScorer(
            direct_force_teacher, ["energy", "forces"]
        ).label(small_batch)
        forces, level = labels["teacher_forces"]
        assert forces.shape == (small_batch.num_nodes, 3)
        assert level == "node"
        assert forces.requires_grad is False

    def test_direct_force_teacher_leaves_positions_grad_free(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """A non-autograd teacher never enables gradients on positions."""
        InProcessTeacherScorer(direct_force_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert small_batch.positions.requires_grad is False

    def test_autograd_teacher_restores_cleared_positions_grad_flag(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """A flag the teacher enabled for autograd forces is cleared again."""
        InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(small_batch)
        assert small_batch.positions.requires_grad is False

    def test_incoming_positions_grad_flag_survives_a_non_autograd_teacher(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """A live student graph on the batch is intact after labeling."""
        small_batch.positions.requires_grad_(True)
        student_energy = (small_batch.positions**2).sum()
        InProcessTeacherScorer(direct_force_teacher, ["energy", "forces"]).label(
            small_batch
        )
        assert small_batch.positions.requires_grad is True
        gradient = torch.autograd.grad(student_energy, small_batch.positions)[0]
        torch.testing.assert_close(gradient, 2.0 * small_batch.positions.detach())

    def test_incoming_positions_grad_flag_survives_an_autograd_teacher(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An autograd-force teacher also hands back the caller's grad flag."""
        small_batch.positions.requires_grad_(True)
        InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(small_batch)
        assert small_batch.positions.requires_grad is True

    def test_active_outputs_are_restored_after_label(
        self, lj_teacher: LennardJonesModelWrapper, periodic_batch: Batch
    ) -> None:
        """A teacher whose active set is narrower than its outputs keeps that set."""
        before = set(lj_teacher.model_config.active_outputs)
        assert before != set(lj_teacher.model_config.outputs)
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(periodic_batch)
        assert lj_teacher.model_config.active_outputs == before

    def test_active_outputs_are_restored_when_the_teacher_raises(self) -> None:
        """A failing forward still restores the config and rolls back neighbors."""
        teacher = _RaisingTeacher()
        batch = _make_spread_batch()
        before = set(teacher.model_config.active_outputs)
        with pytest.raises(RuntimeError, match="teacher forward failed"):
            InProcessTeacherScorer(teacher, ["energy"]).label(batch)
        assert teacher.model_config.active_outputs == before
        assert "neighbor_matrix" not in batch
        assert not hasattr(batch, "_neighbor_list_cutoff")

    def test_energy_only_scorer_does_not_request_forces(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An energy-only scorer narrows the teacher so no force pass runs."""
        scorer = InProcessTeacherScorer(demo_teacher, ["energy"])
        with patch.object(
            demo_teacher.model, "forward", wraps=demo_teacher.model.forward
        ) as spy:
            labels = scorer.label(small_batch)
        assert spy.call_args.kwargs["compute_forces"] is False
        assert set(labels) == {"teacher_energy"}

    def test_node_energies_are_flattened_to_one_dimension(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """Per-atom energies are normalized to ``(V,)``."""
        labels = InProcessTeacherScorer(direct_force_teacher, ["node_energies"]).label(
            small_batch
        )
        values, level = labels["teacher_node_energies"]
        assert values.shape == (small_batch.num_nodes,)
        assert level == "node"

    def test_stress_is_labeled_as_a_system_level_matrix(
        self, lj_teacher: LennardJonesModelWrapper, periodic_batch: Batch
    ) -> None:
        """Stress is normalized to ``(B, 3, 3)`` and matches a direct forward."""
        reference = _build_periodic_batch()
        compute_neighbors(reference, config=lj_teacher.model_config.neighbor_config)
        lj_teacher.set_config("active_outputs", {"stress"})
        expected = lj_teacher(reference)["stress"]
        lj_teacher.set_config("active_outputs", {"energy", "forces"})
        labels = InProcessTeacherScorer(lj_teacher, ["stress"]).label(periodic_batch)
        values, level = labels["teacher_stress"]
        assert values.shape == (periodic_batch.num_graphs, 3, 3)
        assert level == "system"
        torch.testing.assert_close(values, expected)

    def test_cast_to_casts_floating_point_outputs(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """``cast_to`` changes the dtype of every floating-point signal."""
        labels = InProcessTeacherScorer(
            demo_teacher, ["energy", "forces"], cast_to=torch.float64
        ).label(small_batch)
        assert all(value.dtype is torch.float64 for value, _ in labels.values())

    def test_cast_to_bfloat16_yields_bfloat16_labels(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """A dtype no store can hold is still a valid scoring precision."""
        labels = InProcessTeacherScorer(
            demo_teacher, ["energy", "forces"], cast_to=torch.bfloat16
        ).label(small_batch)
        assert all(value.dtype is torch.bfloat16 for value, _ in labels.values())

    def test_missing_teacher_output_raises(self, small_batch: Batch) -> None:
        """A declared output the teacher does not return is reported by name."""
        scorer = InProcessTeacherScorer(_EmptyOutputTeacher(), ["energy"])
        with pytest.raises(RuntimeError, match="'energy'"):
            scorer.label(small_batch)


class TestInProcessTeacherScorerGradMode:
    """Grad mode selected for the teacher forward pass."""

    def test_autograd_teacher_runs_with_grad_enabled(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An autograd-force teacher gets a grad-enabled forward even under no_grad."""
        recorder = _GradModeRecorder(demo_teacher.model.forward)
        scorer = InProcessTeacherScorer(demo_teacher, ["energy", "forces"])
        with patch.object(demo_teacher.model, "forward", recorder), torch.no_grad():
            scorer.label(small_batch)
        assert recorder.grad_enabled == [True]

    def test_non_autograd_scorer_runs_under_no_grad(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """An energy-only scorer forwards with grad disabled."""
        recorder = _GradModeRecorder(demo_teacher.model.forward)
        scorer = InProcessTeacherScorer(demo_teacher, ["energy"])
        with patch.object(demo_teacher.model, "forward", recorder):
            scorer.label(small_batch)
        assert recorder.grad_enabled == [False]

    def test_labeling_inside_no_grad_still_yields_autograd_forces(
        self, demo_teacher: Any, small_batch: Batch
    ) -> None:
        """Forces scored inside ``torch.no_grad()`` equal an ordinary forward."""
        expected = demo_teacher(small_batch.clone())["forces"].detach()
        with torch.no_grad():
            labels = InProcessTeacherScorer(demo_teacher, ["energy", "forces"]).label(
                small_batch
            )
        torch.testing.assert_close(labels["teacher_forces"][0], expected)


class TestInProcessTeacherScorerTrainingMode:
    """Evaluation mode enforced around the teacher forward pass."""

    def test_construction_places_the_teacher_in_evaluation_mode(
        self, direct_force_teacher: _DirectForceTeacher
    ) -> None:
        """A teacher handed over in training mode is evaluated from the start."""
        direct_force_teacher.train()
        InProcessTeacherScorer(direct_force_teacher, ["energy"])
        assert not direct_force_teacher.training

    def test_a_teacher_put_back_in_training_mode_still_scores_evaluating(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """The scorer re-enforces evaluation mode for the forward pass itself."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["energy"])
        direct_force_teacher.train()
        recorder = _TrainingModeRecorder(
            direct_force_teacher, direct_force_teacher.model.forward
        )
        with patch.object(direct_force_teacher.model, "forward", recorder):
            scorer.label(small_batch)
        assert recorder.training == [False]

    def test_the_mode_the_teacher_arrived_in_is_restored(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """Scoring leaves the teacher in the mode the caller had set."""
        scorer = InProcessTeacherScorer(direct_force_teacher, ["energy"])
        direct_force_teacher.train()
        scorer.label(small_batch)
        assert direct_force_teacher.training
        direct_force_teacher.eval()
        scorer.label(small_batch)
        assert not direct_force_teacher.training

    def test_a_teacher_that_is_not_a_module_is_scored_unchanged(
        self, small_batch: Batch
    ) -> None:
        """A teacher with no training mode is left alone rather than probed."""
        teacher = _AddKeyEmbeddingTeacher()
        scorer = InProcessTeacherScorer(teacher, ["embeddings"])
        labels = scorer.label(small_batch)
        assert set(labels) == {"teacher_node_embeddings"}


class TestInProcessTeacherScorerEmbeddings:
    """Embedding signal extraction and batch restoration."""

    def test_embeddings_signal_returns_node_embeddings(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """The embeddings signal is a node-level ``(V, D)`` tensor."""
        labels = InProcessTeacherScorer(direct_force_teacher, ["embeddings"]).label(
            small_batch
        )
        values, level = labels["teacher_node_embeddings"]
        hidden_dim = direct_force_teacher.embedding_shapes["node_embeddings"][0]
        assert values.shape == (small_batch.num_nodes, hidden_dim)
        assert level == "node"
        assert values.requires_grad is False

    def test_batch_keeps_no_embedding_keys_afterwards(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """``compute_embeddings`` output is removed from the batch again."""
        InProcessTeacherScorer(direct_force_teacher, ["embeddings"]).label(small_batch)
        assert "node_embeddings" not in small_batch
        assert "graph_embeddings" not in small_batch

    def test_pre_existing_node_embeddings_are_restored(
        self, direct_force_teacher: _DirectForceTeacher, small_batch: Batch
    ) -> None:
        """Embeddings already on the batch survive scoring unchanged."""
        existing = torch.arange(small_batch.num_nodes * 4, dtype=torch.float32).reshape(
            small_batch.num_nodes, 4
        )
        small_batch._atoms_group["node_embeddings"] = existing
        InProcessTeacherScorer(direct_force_teacher, ["embeddings"]).label(small_batch)
        torch.testing.assert_close(small_batch.node_embeddings, existing)

    def test_add_key_teacher_leaves_both_embedding_levels_clean(
        self, small_batch: Batch
    ) -> None:
        """A teacher writing both levels via ``add_key`` leaves neither behind."""
        before = _storage_snapshot(small_batch)
        tracked = _tracked_snapshot(small_batch)
        labels = InProcessTeacherScorer(
            _AddKeyEmbeddingTeacher(), ["embeddings"]
        ).label(small_batch)
        assert labels["teacher_node_embeddings"][0].shape == (small_batch.num_nodes, 4)
        assert "node_embeddings" not in small_batch
        assert "graph_embeddings" not in small_batch
        assert {key for key, _ in small_batch} == set(before)
        assert _tracked_snapshot(small_batch) == tracked

    def test_registered_embeddings_stay_registered_and_unchanged(
        self, small_batch: Batch
    ) -> None:
        """Embeddings registered in ``batch.keys`` come back with the same values."""
        existing = torch.full((small_batch.num_nodes, 4), 7.0)
        small_batch.add_key(
            "node_embeddings",
            list(torch.split(existing, small_batch.num_nodes_list)),
            level="node",
        )
        InProcessTeacherScorer(_AddKeyEmbeddingTeacher(), ["embeddings"]).label(
            small_batch
        )
        assert "node_embeddings" in small_batch.keys["node"]
        torch.testing.assert_close(small_batch.node_embeddings, existing)

    def test_teacher_that_does_not_mutate_the_batch_raises(
        self, small_batch: Batch
    ) -> None:
        """Embeddings returned on a copy instead of written in place are an error."""
        scorer = InProcessTeacherScorer(_NonMutatingEmbeddingTeacher(), ["embeddings"])
        with pytest.raises(RuntimeError, match="node_embeddings"):
            scorer.label(small_batch)
        assert "node_embeddings" not in small_batch


class TestInProcessTeacherScorerNeighborIsolation:
    """Neighbor-list construction and rollback around the teacher forward."""

    def test_prebuilt_matrix_neighbors_are_restored_exactly(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A list built at another cutoff is rebuilt for the teacher, then restored."""
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=8.0, format=NeighborListFormat.MATRIX)
        expected_matrix = batch.neighbor_matrix.clone()
        expected_counts = batch.num_neighbors.clone()
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(batch)
        assert batch._neighbor_list_cutoff == 8.0
        assert torch.equal(batch.neighbor_matrix, expected_matrix)
        assert torch.equal(batch.num_neighbors, expected_counts)

    def test_batch_without_neighbors_has_none_afterwards(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """Neighbor tensors built for the teacher do not leak onto a bare batch."""
        batch = _make_spread_batch()
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(batch)
        assert "neighbor_matrix" not in batch
        assert "num_neighbors" not in batch
        assert not hasattr(batch, "_neighbor_list_cutoff")
        assert not hasattr(batch, "_neighbor_list_half")

    @pytest.mark.parametrize(
        ("build_list", "build_teacher", "expected_builds"),
        _REUSE_CASES,
        ids=_REUSE_IDS,
    )
    def test_reuse_requires_a_declared_matching_list(
        self,
        build_list: Callable[[Batch], None],
        build_teacher: Callable[[], Any],
        expected_builds: int,
    ) -> None:
        """Only a declared full list at the teacher's cutoff and format is reused."""
        batch = _make_spread_batch()
        build_list(batch)
        scorer = InProcessTeacherScorer(build_teacher(), ["energy"])
        with _spy_on_neighbor_builds() as spy:
            scorer.label(batch)
        assert spy.call_count == expected_builds

    def test_matrix_list_is_rebuilt_for_a_coo_teacher(self) -> None:
        """A dense list at the right cutoff is rebuilt sparse, then rolled back."""
        batch = _make_spread_batch()
        _build_declared_neighbors(batch, _COO_CUTOFF, NeighborListFormat.MATRIX)
        scorer = InProcessTeacherScorer(_CooNeighborTeacher(), ["energy"])
        with _spy_on_neighbor_builds() as spy:
            labels = scorer.label(batch)
        assert spy.call_count == 1
        assert labels["teacher_energy"][0].shape == (batch.num_graphs, 1)
        assert "neighbor_list" not in batch
        assert "neighbor_matrix" in batch

    def test_coo_list_is_rebuilt_for_a_matrix_teacher(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A sparse list at the right cutoff is rebuilt dense, then rolled back."""
        batch = _make_spread_batch()
        _build_declared_neighbors(batch, _LJ_CUTOFF, NeighborListFormat.COO)
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        with _spy_on_neighbor_builds() as spy:
            scorer.label(batch)
        assert spy.call_count == 1
        assert "neighbor_matrix" not in batch
        assert "neighbor_list" in batch

    def test_prebuilt_coo_neighbors_are_restored_exactly(self) -> None:
        """A pre-built edge group survives a COO teacher's rebuild untouched."""
        teacher = _CooNeighborTeacher()
        batch = _make_spread_batch()
        compute_neighbors(batch, cutoff=8.0, format=NeighborListFormat.COO)
        expected_edges = batch.neighbor_list.clone()
        expected_counts = list(batch.num_edges_list)
        scorer = InProcessTeacherScorer(teacher, ["energy"])
        with _spy_on_neighbor_builds() as spy:
            scorer.label(batch)
        assert spy.call_count == 1
        assert batch._neighbor_list_cutoff == 8.0
        assert torch.equal(batch.neighbor_list, expected_edges)
        assert batch.num_edges_list == expected_counts

    def test_half_list_teacher_never_reuses_a_full_list(self) -> None:
        """A half-list teacher rebuilds, so a full list cannot double-count pairs."""
        teacher = _build_lj_teacher(half_list=True)
        scorer = InProcessTeacherScorer(teacher, ["energy"])
        prebuilt = _make_spread_batch()
        _build_declared_neighbors(prebuilt, _LJ_CUTOFF, NeighborListFormat.MATRIX)
        with _spy_on_neighbor_builds() as spy:
            reused = scorer.label(prebuilt)["teacher_energy"][0]
        fresh = scorer.label(_make_spread_batch())["teacher_energy"][0]
        assert spy.call_count == 1
        torch.testing.assert_close(reused, fresh)

    def test_full_list_teacher_never_reuses_a_half_list(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A full-list teacher rebuilds, so a half list cannot halve the energy."""
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        prebuilt = _make_spread_batch()
        _build_declared_neighbors(
            prebuilt, _LJ_CUTOFF, NeighborListFormat.MATRIX, half_list=True
        )
        with _spy_on_neighbor_builds() as spy:
            reused = scorer.label(prebuilt)["teacher_energy"][0]
        fresh = scorer.label(_make_spread_batch())["teacher_energy"][0]
        assert spy.call_count == 1
        torch.testing.assert_close(reused, fresh)

    def test_periodic_neighbor_shifts_are_rolled_back(
        self, lj_teacher: LennardJonesModelWrapper, periodic_batch: Batch
    ) -> None:
        """Shift tensors built for a periodic teacher are removed again."""
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(periodic_batch)
        assert "neighbor_matrix_shifts" not in periodic_batch
        assert "neighbor_matrix" not in periodic_batch

    def test_shadowed_neighbor_list_never_reaches_the_teacher(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A list shadowed in the instance dict cannot poison the teacher's labels."""
        scorer = InProcessTeacherScorer(lj_teacher, ["energy", "forces"])
        expected = scorer.label(_make_lattice_batch())
        batch = _make_lattice_batch()
        _shadow_dense_neighbors(batch, _SHADOW_CUTOFF)
        labels = scorer.label(batch)
        for field, (values, _) in expected.items():
            torch.testing.assert_close(labels[field][0], values)

    def test_shadowed_neighbor_attributes_are_restored_verbatim(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """Shadowed neighbor state comes back in the instance dict, not in storage."""
        batch = _make_lattice_batch()
        shadows = _shadow_dense_neighbors(batch, _SHADOW_CUTOFF)
        InProcessTeacherScorer(lj_teacher, ["energy"]).label(batch)
        assert batch.__dict__["_neighbor_list_cutoff"] == _SHADOW_CUTOFF
        assert batch.neighbor_matrix is shadows["neighbor_matrix"]
        assert batch.num_neighbors is shadows["num_neighbors"]
        assert "neighbor_matrix" not in batch

    def test_shadowed_list_at_the_teacher_cutoff_is_reused(
        self, lj_teacher: LennardJonesModelWrapper
    ) -> None:
        """A declared shadowed list the teacher can consume is reused, not rebuilt."""
        scorer = InProcessTeacherScorer(lj_teacher, ["energy"])
        expected = scorer.label(_make_lattice_batch())["teacher_energy"][0]
        batch = _make_lattice_batch()
        _shadow_dense_neighbors(batch, _LJ_CUTOFF)
        batch._neighbor_list_half = False
        with _spy_on_neighbor_builds() as spy:
            labels = scorer.label(batch)
        assert spy.call_count == 0
        torch.testing.assert_close(labels["teacher_energy"][0], expected)

    def test_stale_dense_list_is_hidden_from_a_sparse_teacher(self) -> None:
        """A dense list from another build cannot re-enter a COO teacher's forward."""
        teacher = _CooNeighborTeacher()
        batch = _make_spread_batch()
        _build_declared_neighbors(batch, 8.0, NeighborListFormat.MATRIX)
        InProcessTeacherScorer(teacher, ["energy"]).label(batch)
        assert "neighbor_matrix" not in teacher.seen_storage_keys
        assert "neighbor_list" in teacher.seen_storage_keys
        assert "neighbor_matrix" in batch


class TestComposedTeacherNeighbors:
    """Scoring a composed teacher against a batch a composed student prepared."""

    def test_composed_teacher_ignores_captured_pipeline_sources(self) -> None:
        """A composed teacher labels a live pipeline batch exactly as it labels a clean one."""
        scorer = InProcessTeacherScorer(_composed_teacher(9.0), ["energy", "forces"])
        expected = scorer.label(_make_lattice_batch())
        student = _composed_teacher(*_STUDENT_CUTOFFS, neighbor_adaptation="never")
        batch = _pipeline_prepared_batch(student, _make_lattice_batch())
        labels = scorer.label(batch)
        for field, (values, _) in expected.items():
            torch.testing.assert_close(labels[field][0], values)

    def test_captured_pipeline_sources_are_restored_verbatim(self) -> None:
        """The captured source table comes back untouched, and the student's forward with it."""
        student = _composed_teacher(*_STUDENT_CUTOFFS, neighbor_adaptation="never")
        batch = _pipeline_prepared_batch(student, _make_lattice_batch())
        captured = batch.__dict__[_PIPELINE_SOURCES_ATTR]
        with torch.no_grad():
            before = student(batch)["energy"].clone()
        InProcessTeacherScorer(_composed_teacher(9.0), ["energy"]).label(batch)
        assert batch.__dict__[_PIPELINE_SOURCES_ATTR] is captured
        with torch.no_grad():
            torch.testing.assert_close(student(batch)["energy"], before)

    def test_reused_list_does_not_expose_captured_sources(self) -> None:
        """A reused list is scored on its own terms, not through the captured table."""
        scorer = InProcessTeacherScorer(
            _composed_teacher(_LJ_CUTOFF), ["energy", "forces"]
        )
        expected = scorer.label(_make_lattice_batch())
        student = _composed_teacher(*_STUDENT_CUTOFFS, neighbor_adaptation="never")
        batch = _pipeline_prepared_batch(student, _make_lattice_batch())
        _shadow_dense_neighbors(batch, _LJ_CUTOFF)
        batch._neighbor_list_half = False
        with _spy_on_neighbor_builds() as spy:
            labels = scorer.label(batch)
        assert spy.call_count == 0
        for field, (values, _) in expected.items():
            torch.testing.assert_close(labels[field][0], values)

    def test_multi_source_composed_teacher_is_refused(self) -> None:
        """A composition planning two neighbor lists is rejected at construction."""
        with pytest.raises(ValueError, match="neighbor-list sources"):
            InProcessTeacherScorer(_composed_teacher(5.0, 15.0), ["energy"])

    def test_single_list_composition_labels_a_plain_batch(self) -> None:
        """The always-adapt escape plans one list, which the scorer builds itself."""
        teacher = _composed_teacher(5.0, 15.0, neighbor_adaptation="always")
        prepared = _pipeline_prepared_batch(teacher, _make_lattice_batch())
        with torch.no_grad():
            expected = teacher(prepared)["energy"]
        labels = InProcessTeacherScorer(teacher, ["energy"]).label(
            _make_lattice_batch()
        )
        torch.testing.assert_close(labels["teacher_energy"][0], expected)

    def test_pipeline_sources_attribute_matches_the_core(self) -> None:
        """The mirrored attribute name tracks the one the pipeline hook writes."""
        assert _PIPELINE_SOURCES_ATTR == _PIPELINE_NEIGHBOR_SOURCES_ATTR


class TestInProcessTeacherScorerAutogradNeighborTeacher:
    """Scoring a teacher that differentiates through a list the scorer builds."""

    def test_labels_match_a_forward_on_a_prebuilt_batch(
        self, pair_potential_teacher: _PairPotentialTeacher
    ) -> None:
        """A rebuilt list yields the same energy and forces as a pre-built one."""
        reference = _make_lattice_batch()
        compute_neighbors(
            reference, config=pair_potential_teacher.model_config.neighbor_config
        )
        expected = pair_potential_teacher(reference)
        batch = _make_lattice_batch()
        with _spy_on_neighbor_builds() as spy:
            labels = InProcessTeacherScorer(
                pair_potential_teacher, ["energy", "forces"]
            ).label(batch)
        assert spy.call_count == 1
        torch.testing.assert_close(labels["teacher_energy"][0], expected["energy"])
        torch.testing.assert_close(
            labels["teacher_forces"][0], expected["forces"].detach()
        )
        assert batch.positions.requires_grad is False
        assert "neighbor_matrix" not in batch

    def test_prebuilt_list_at_another_cutoff_is_rebuilt_and_restored(
        self, pair_potential_teacher: _PairPotentialTeacher
    ) -> None:
        """The caller's list survives a rebuild that an autograd forward runs on."""
        batch = _make_lattice_batch()
        _build_declared_neighbors(batch, 8.0, NeighborListFormat.MATRIX)
        expected_matrix = batch.neighbor_matrix.clone()
        scorer = InProcessTeacherScorer(pair_potential_teacher, ["energy", "forces"])
        clean = scorer.label(_make_lattice_batch())["teacher_forces"][0]
        with _spy_on_neighbor_builds() as spy:
            labels = scorer.label(batch)
        assert spy.call_count == 1
        torch.testing.assert_close(labels["teacher_forces"][0], clean)
        assert torch.equal(batch.neighbor_matrix, expected_matrix)
        assert batch._neighbor_list_cutoff == 8.0

    def test_forces_scored_inside_no_grad_match_an_ordinary_forward(
        self, pair_potential_teacher: _PairPotentialTeacher
    ) -> None:
        """Building a list under ``no_grad`` still leaves the forward differentiable."""
        scorer = InProcessTeacherScorer(pair_potential_teacher, ["energy", "forces"])
        expected = scorer.label(_make_lattice_batch())["teacher_forces"][0]
        with torch.no_grad():
            labels = scorer.label(_make_lattice_batch())
        torch.testing.assert_close(labels["teacher_forces"][0], expected)
        assert labels["teacher_forces"][0].requires_grad is False
