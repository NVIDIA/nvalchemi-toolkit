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
"""Caller-owned CSP materialization through generation and dynamics."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.csp import CSPGenerator
from nvalchemi.csp.comparison import RadialComparisonIndex
from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer import (
    OverlapReliefConfig,
    OverlapReliefPacker,
    PackingReport,
    PackingResult,
    PackingStopReason,
)
from nvalchemi.csp.storage import RigidMoleculeASUZarrReader, RigidMoleculeASUZarrWriter
from nvalchemi.csp.symmetry import SpaceGroupPolicy
from nvalchemi.data import Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from test.csp.fixtures.contact_oracle import contact_observables
from test.csp.test_batch import make_skew_compact

ATOM_SOURCE_FIELDS = (
    "csp_source_asu_atom_index",
    "csp_source_molecule_index",
    "csp_source_component_index",
    "csp_source_conformer_index",
    "csp_source_symmetry_operation_index",
)
SYSTEM_SOURCE_FIELDS = (
    "csp_source_space_group",
    "csp_source_z",
    "csp_source_z_prime",
    "csp_source_structure_id",
)


def _two_atom_formula() -> MolecularPackingInput:
    return MolecularPackingInput(
        conformer_positions=torch.zeros((2, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        atomic_numbers=torch.tensor([6, 8], dtype=torch.int64),
        contact_distances=torch.full((2, 2), 3.25, dtype=torch.float32),
        component_index=torch.tensor([0, 1], dtype=torch.int32),
        formula_unit_volume=125.0,
    )


class _PipelineFakePacker:
    """Small deterministic protocol fake for CSP pipeline integration."""

    device = torch.device("cpu")

    def __init__(self, accepted: int = 1) -> None:
        self.accepted = accepted

    def pack(self, inputs, *, num_samples, rng, context, **options):
        del rng, options
        count = min(num_samples, self.accepted)
        structures = RigidMoleculeASUBatch(
            packing_input=inputs,
            structure_molecule_ptr=torch.arange(count + 1, dtype=torch.int32) * 2,
            conformer_indices=torch.zeros((count * 2,), dtype=torch.int32),
            rotations=torch.eye(3).expand(count * 2, 3, 3).clone(),
            fractional_centers=torch.full((count * 2, 3), 0.5),
            cells=torch.eye(3).expand(count, 3, 3).clone() * 5,
            space_groups=torch.ones((count,), dtype=torch.int32),
            z=torch.ones((count,), dtype=torch.int32),
            z_prime=torch.ones((count,), dtype=torch.int32),
            structure_ids=context.structure_ids(count, device="cpu"),
        )
        reason = "target_reached" if count == num_samples else "fake_shortfall"
        return PackingResult(
            structures=structures,
            run_id=context.run_id,
            reports=(
                PackingReport(
                    rank=context.rank,
                    requested_count=num_samples,
                    accepted_count=count,
                    generated_count=count,
                    stop_reason=reason,
                ),
            ),
        )


class _PipelineEventHook:
    """Capture the ordinary generator hook position in the pipeline fold."""

    stage = GenerationStage.AFTER_GENERATE
    frequency = 1

    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.events = events

    def __call__(self, context, stage) -> None:
        self.events.append(("after_generate", context.sample))


class _ObserveAfterGenerate:
    """Capture the Batch exposed at the public post-generation hook."""

    stage = GenerationStage.AFTER_GENERATE
    frequency = 1

    def __init__(self) -> None:
        self.sample: Batch | None = None
        self.batch: Batch | None = None
        self.source_fields: dict[str, torch.Tensor] = {}

    def __call__(self, ctx, stage) -> None:
        self.sample = ctx.sample
        self.batch = ctx.batch
        self.source_fields = {
            name: ctx.batch[name].clone()
            for name in (*ATOM_SOURCE_FIELDS, *SYSTEM_SOURCE_FIELDS)
        }


def test_caller_wrapper_observes_compact_then_returns_batch_to_pipeline() -> None:
    compact = make_skew_compact()
    events: list[tuple[str, object]] = []

    def compact_result_callback(result: RigidMoleculeASUBatch) -> None:
        """Observe the full accepted compact result before conversion."""
        events.append(("compact_callback", result))
        assert result is compact
        assert result.num_structures == 2

    def produce_compact(inputs=None, *, num_samples=1, rng=None, **kwargs):
        """Produce compact structures using the generator call convention."""
        del inputs, num_samples, rng, kwargs
        return compact

    def generate_batch(inputs=None, *, num_samples=1, rng=None, **kwargs):
        """Own callback, row selection, and P1 expansion before return."""
        accepted = produce_compact(inputs, num_samples=num_samples, rng=rng, **kwargs)
        compact_result_callback(accepted)
        batch = accepted.to_batch(indices=torch.tensor([1], dtype=torch.int32))
        events.append(("batch_return", batch))
        return batch

    class _OrderedObserver(_ObserveAfterGenerate):
        def __call__(self, ctx, stage) -> None:
            events.append(("after_generate", ctx.sample))
            super().__call__(ctx, stage)

    observer = _OrderedObserver()

    generator = AtomisticGenerator(
        generator_func=generate_batch,
        required_inputs=frozenset(),
        outputs=frozenset(
            {
                "positions",
                "atomic_numbers",
                "cell",
                "pbc",
                *ATOM_SOURCE_FIELDS,
                *SYSTEM_SOURCE_FIELDS,
            }
        ),
        hooks=[observer],
        device="cpu",
        dedicated_stream=False,
    )
    fire = FIRE2(model=DemoModelWrapper(DemoModel()), dt=0.05, n_steps=1)
    pipeline = GenerationPipeline(stages=[generator, fire])

    with pipeline:
        result = pipeline()

    assert observer.sample is observer.batch
    assert observer.batch is not None
    assert result is not None
    assert events[0][0] == "compact_callback"
    assert events[0][1] is compact
    assert events[1][0] == "batch_return"
    assert events[2][0] == "after_generate"
    assert events[2][1] is events[1][1]
    assert observer.batch is events[1][1]
    assert fire.step_count == 1
    assert torch.isfinite(result.positions).all()
    assert torch.isfinite(result.cell).all()
    assert result.pbc.shape == (1, 3)
    assert bool(result.pbc.all())
    torch.testing.assert_close(result.cell, compact.cells[1:2])
    for name in (*ATOM_SOURCE_FIELDS, *SYSTEM_SOURCE_FIELDS):
        assert torch.equal(result[name], observer.source_fields[name])


def test_csp_generator_pipeline_orders_callback_hooks_and_dynamics() -> None:
    events: list[tuple[str, object]] = []

    def on_result(result: PackingResult) -> None:
        events.append(("callback", result))

    generator = CSPGenerator(
        _PipelineFakePacker(),
        on_result=on_result,
        hooks=[_PipelineEventHook(events)],
        dedicated_stream=False,
    )
    fire = FIRE2(model=DemoModelWrapper(DemoModel()), dt=0.05, n_steps=1)
    pipeline = GenerationPipeline(stages=[generator, fire])

    with pipeline:
        result = pipeline(
            _two_atom_formula(),
            stage_kwargs=[{"num_samples": 1, "run_id": 800}, {}],
        )

    assert isinstance(events[0][1], PackingResult)
    assert events[0][0] == "callback"
    assert events[1][0] == "after_generate"
    assert isinstance(events[1][1], Batch)
    assert result is not None and result.num_graphs == 1
    assert fire.step_count == 1


def test_empty_csp_batch_short_circuits_pipeline_dynamics() -> None:
    events: list[tuple[str, object]] = []
    callbacks: list[PackingResult] = []
    generator = CSPGenerator(
        _PipelineFakePacker(accepted=0),
        on_result=callbacks.append,
        hooks=[_PipelineEventHook(events)],
        dedicated_stream=False,
    )
    fire = FIRE2(model=DemoModelWrapper(DemoModel()), dt=0.05, n_steps=1)
    pipeline = GenerationPipeline(stages=[generator, fire])

    with pipeline:
        result = pipeline(
            _two_atom_formula(),
            stage_kwargs=[{"num_samples": 1, "run_id": 801}, {}],
        )

    assert len(callbacks) == 1 and callbacks[0].accepted_count == 0
    assert isinstance(result, Batch) and result.num_graphs == 0
    assert events == [("after_generate", result)]
    assert fire.step_count == 0


def test_raw_csp_result_before_dynamics_raises_and_pipeline_compile_rejects() -> None:
    generator = CSPGenerator(
        _PipelineFakePacker(), expand=False, dedicated_stream=False
    )
    fire = FIRE2(model=DemoModelWrapper(DemoModel()), dt=0.05, n_steps=1)
    pipeline = GenerationPipeline(stages=[generator, fire])

    with pytest.raises(
        NotImplementedError, match="CSPGenerator does not support compile"
    ):
        pipeline.compile()
    with pipeline, pytest.raises(TypeError, match="requires a Batch input"):
        pipeline(
            _two_atom_formula(),
            stage_kwargs=[{"num_samples": 1, "run_id": 802}, {}],
        )


def test_compact_callback_write_survives_later_optimization_failure(tmp_path) -> None:
    compact = make_skew_compact()
    packing_result = PackingResult(
        structures=compact,
        run_id=int(compact.structure_ids[0, 0]),
        reports=(
            PackingReport(
                rank=0,
                requested_count=compact.num_structures,
                accepted_count=compact.num_structures,
                generated_count=compact.num_structures,
                stop_reason=PackingStopReason.TARGET_REACHED,
            ),
        ),
    )
    store = tmp_path / "csp.zarr"

    class _FailingDemoModel(DemoModel):
        def forward(self, *args, **kwargs):
            raise RuntimeError("intentional optimization failure")

    with RigidMoleculeASUZarrWriter(store) as writer:

        def write_compact_result(result: PackingResult) -> None:
            writer.write(result.structures)

        def generate_batch(inputs=None, *, num_samples=1, rng=None, **kwargs):
            del inputs, num_samples, rng, kwargs
            write_compact_result(packing_result)
            selected = packing_result.structures.select(
                torch.tensor([1], dtype=torch.int32)
            )
            return selected.to_batch()

        generator = AtomisticGenerator(
            generator_func=generate_batch,
            required_inputs=frozenset(),
            outputs=frozenset(
                {
                    "positions",
                    "atomic_numbers",
                    "cell",
                    "pbc",
                    *ATOM_SOURCE_FIELDS,
                    *SYSTEM_SOURCE_FIELDS,
                }
            ),
            device="cpu",
            dedicated_stream=False,
        )
        fire = FIRE2(model=DemoModelWrapper(_FailingDemoModel()), dt=0.05, n_steps=1)
        pipeline = GenerationPipeline(stages=[generator, fire])
        with pytest.raises(RuntimeError, match="intentional optimization failure"):
            with pipeline:
                pipeline()

    with RigidMoleculeASUZarrReader(store) as reader:
        assert len(reader) == len(packing_result)
        restored = reader.read(torch.tensor([0, 1], dtype=torch.int64))
        expected = packing_result.structures
        for name in (
            "structure_molecule_ptr",
            "conformer_indices",
            "rotations",
            "fractional_centers",
            "cells",
            "space_groups",
            "z",
            "z_prime",
            "structure_ids",
        ):
            torch.testing.assert_close(getattr(restored, name), getattr(expected, name))
        assert set(restored.properties) == set(expected.properties)
        for name, value in expected.properties.items():
            torch.testing.assert_close(restored.properties[name], value)

        expected_input = expected.packing_input.state_dict()
        restored_input = restored.packing_input.state_dict()
        assert set(restored_input) == set(expected_input)
        for name, value in expected_input.items():
            actual = restored_input[name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value)
            else:
                assert actual == value


def test_cpu_packing_round_trips_complete_compact_result_through_zarr(
    tmp_path,
) -> None:
    packing_input = _two_atom_formula()
    result = OverlapReliefPacker(
        OverlapReliefConfig(
            z=1,
            z_prime=1,
            batch_size=2,
            max_candidates=2,
            cell_volume_range=(125.0, 125.0),
            space_groups=SpaceGroupPolicy.fixed(1),
            overlap_tolerance=3.25,
        ),
        device="cpu",
    ).pack(
        packing_input,
        num_samples=2,
        rng=torch.Generator(device="cpu").manual_seed(91),
        run_id=77,
    )
    assert result is not None
    assert len(result) == result.generated_count == 2
    assert result.stop_reason is PackingStopReason.TARGET_REACHED
    compact = result.structures
    assert compact.structure_ids.tolist() == [[77, 0], [77, 1]]
    assert torch.equal(compact.properties["steps"], torch.zeros(2, dtype=torch.int32))
    assert torch.all(compact.properties["total_overlap"] > 0)
    assert torch.all(compact.properties["max_overlap"] > 0)
    assert torch.all(compact.properties["max_overlap"] <= 3.25)

    store = tmp_path / "complete-csp-result.zarr"
    with RigidMoleculeASUZarrWriter(store) as writer:
        writer.write(compact)
    selected_indices = torch.tensor([1, 0, 1], dtype=torch.int64)
    direct = compact.to_batch(indices=selected_indices)
    with RigidMoleculeASUZarrReader(store) as reader:
        assert len(reader) == 2
        stored_compact = reader.read(selected_indices)
        stored = reader.read_batch(selected_indices)

    assert stored_compact.structure_ids.tolist() == [[77, 1], [77, 0], [77, 1]]
    for name in (
        "structure_molecule_ptr",
        "conformer_indices",
        "rotations",
        "fractional_centers",
        "cells",
        "space_groups",
        "z",
        "z_prime",
        "structure_ids",
    ):
        torch.testing.assert_close(
            getattr(stored_compact, name),
            getattr(compact.select(selected_indices), name),
        )
    assert set(stored_compact.properties) == set(compact.properties)
    for name, expected in compact.select(selected_indices).properties.items():
        torch.testing.assert_close(stored_compact.properties[name], expected)
    assert direct.num_nodes_list == stored.num_nodes_list
    assert direct.num_graphs == stored.num_graphs == 3
    for name in ("positions", "atomic_numbers", "cell", "pbc"):
        torch.testing.assert_close(direct[name], stored[name])
    for name in (*ATOM_SOURCE_FIELDS, *SYSTEM_SOURCE_FIELDS):
        assert torch.equal(direct[name], stored[name])
    assert stored.csp_source_structure_id.tolist() == [[77, 1], [77, 0], [77, 1]]

    for graph_index in range(direct.num_graphs):
        atom_start = int(direct.batch_ptr[graph_index])
        atom_stop = int(direct.batch_ptr[graph_index + 1])
        count, _, _, total, maximum = contact_observables(
            direct.positions[atom_start:atom_stop],
            direct.cell[graph_index],
            direct.csp_source_molecule_index[atom_start:atom_stop],
            packing_input.contact_distances,
        )
        assert count > 0
        assert total == pytest.approx(
            float(stored_compact.properties["total_overlap"][graph_index]), abs=5e-5
        )
        assert maximum == pytest.approx(
            float(stored_compact.properties["max_overlap"][graph_index]), abs=5e-5
        )
        assert maximum <= 3.25

    index_direct = RadialComparisonIndex.build(direct, cutoff=4.0)
    index_stored = RadialComparisonIndex.build(stored, cutoff=4.0)
    pairs = torch.tensor([[0, 1], [0, 2]], dtype=torch.int64)
    direct_scores = index_direct.score_pairs(pairs)
    stored_scores = index_stored.score_pairs(pairs)
    torch.testing.assert_close(direct_scores, stored_scores, atol=1e-6, rtol=0)
    assert direct_scores[0] > 0
    assert direct_scores[1] == 0
