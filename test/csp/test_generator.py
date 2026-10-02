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
"""Public CSPGenerator lifecycle and payload contracts."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from nvalchemi.csp import CSP_OUTPUT_FIELDS, CSPGenerator
from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch
from nvalchemi.csp.packer import (
    PackingContext,
    PackingReport,
    PackingResult,
    PackingStopReason,
)
from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage


def _formula(
    *, marker: str = "same", component_charge: int = 0
) -> MolecularPackingInput:
    """Build a one-carbon formula-unit input."""
    return MolecularPackingInput(
        conformer_positions=torch.zeros((1, 3), dtype=torch.float32),
        conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
        molecule_atom_ptr=torch.tensor([0, 1], dtype=torch.int32),
        atomic_numbers=torch.tensor([6], dtype=torch.int64),
        contact_distances=torch.ones((1, 1), dtype=torch.float32),
        component_index=torch.tensor([0], dtype=torch.int32),
        component_charge=torch.tensor([component_charge], dtype=torch.int32),
        formula_unit_volume=1000.0,
        metadata={"marker": marker},
    )


def _compact(
    inputs: MolecularPackingInput,
    context: PackingContext,
    count: int,
) -> RigidMoleculeASUBatch:
    """Build a simple rigid ASU payload with context-owned row IDs."""
    device = torch.device("cpu")
    return RigidMoleculeASUBatch(
        packing_input=inputs,
        structure_molecule_ptr=torch.arange(count + 1, dtype=torch.int32),
        conformer_indices=torch.zeros((count,), dtype=torch.int32),
        rotations=torch.eye(3, dtype=torch.float32).expand(count, 3, 3).clone(),
        fractional_centers=torch.zeros((count, 3), dtype=torch.float32),
        cells=torch.eye(3, dtype=torch.float32).expand(count, 3, 3).clone() * 10,
        space_groups=torch.ones((count,), dtype=torch.int32),
        z=torch.ones((count,), dtype=torch.int32),
        z_prime=torch.ones((count,), dtype=torch.int32),
        structure_ids=context.structure_ids(count, device=device),
    )


def _native_batch(
    context: PackingContext,
    count: int,
    *,
    charge: float = 0.0,
    include_charge: bool = True,
) -> Batch:
    """Build a minimal native P1 Batch using the public data constructors."""
    rows = []
    for row in range(count):
        data = AtomicData(
            positions=torch.tensor([[float(row), 0.0, 0.0]]),
            atomic_numbers=torch.tensor([6], dtype=torch.int64),
            cell=torch.eye(3).unsqueeze(0) * 10,
            pbc=torch.ones((1, 3), dtype=torch.bool),
        )
        data.add_system_property(
            "csp_source_structure_id",
            context.structure_ids(count, device="cpu")[row : row + 1],
        )
        if include_charge:
            data.add_system_property(
                "charge", torch.full((1, 1), charge, dtype=torch.float32)
            )
        rows.append(data)
    if rows:
        return Batch.from_data_list(rows, device="cpu")

    template = _native_batch(context, 1, charge=charge, include_charge=include_charge)
    return Batch.empty(
        num_systems=0,
        num_nodes=0,
        num_edges=0,
        template=template,
        device="cpu",
    )


class _FakePacker:
    """Deterministic protocol implementation for generation-contract tests."""

    device = torch.device("cpu")

    def __init__(
        self,
        *,
        budget: int | None = None,
        accepted: int | None = None,
        generated: int | None = None,
        native_batch: bool = False,
        omit_native_charge: bool = False,
        native_charge: float = 0.0,
    ) -> None:
        self.budget = budget
        self.accepted = accepted
        self.generated = generated
        self.native_batch = native_batch
        self.omit_native_charge = omit_native_charge
        self.native_charge = native_charge
        self.calls: list[dict[str, Any]] = []
        self.budget_calls: list[dict[str, Any]] = []
        self.input_override: MolecularPackingInput | None = None

    def resolve_candidate_budget(
        self, *, num_samples: int, **pack_options: Any
    ) -> int | None:
        self.budget_calls.append({"num_samples": num_samples, **pack_options})
        return self.budget

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int,
        rng: torch.Generator | None,
        context: PackingContext,
        **options: Any,
    ) -> PackingResult:
        self.calls.append(
            {
                "inputs": inputs,
                "num_samples": num_samples,
                "rng": rng,
                "context": context,
                **options,
            }
        )
        accepted = (
            min(num_samples, self.accepted)
            if self.accepted is not None
            else num_samples
        )
        packed_input = self.input_override or inputs
        if self.native_batch:
            structures = _native_batch(
                context,
                accepted,
                charge=self.native_charge,
                include_charge=not self.omit_native_charge,
            )
            packed_result: RigidMoleculeASUBatch | Batch = structures
        else:
            packed_result = _compact(packed_input, context, accepted)
        generated = accepted if self.generated is None else self.generated
        reason = (
            PackingStopReason.TARGET_REACHED
            if accepted == num_samples
            else "fake_stopped"
        )
        report = PackingReport(
            rank=context.rank,
            requested_count=num_samples,
            accepted_count=accepted,
            generated_count=generated,
            stop_reason=reason,
        )
        return PackingResult(
            structures=packed_result,
            run_id=context.run_id,
            reports=(report,),
        )


class _AfterGenerate:
    """Record the ordinary Batch hook boundary."""

    stage = GenerationStage.AFTER_GENERATE
    frequency = 1

    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = events

    def __call__(self, context: Any, stage: Any) -> None:
        self.events.append(("after_generate", context.sample))


def test_public_export_local_conversion_and_callback_order() -> None:
    assert CSP_OUTPUT_FIELDS == frozenset(
        {
            "positions",
            "atomic_numbers",
            "cell",
            "pbc",
            "csp_source_structure_id",
        }
    )
    packer = _FakePacker()
    events: list[tuple[str, Any]] = []
    hook = _AfterGenerate(events)

    def condition(inputs, *, num_samples, rng):
        events.append(("condition", (num_samples, rng)))
        return inputs

    def on_result(result: PackingResult) -> None:
        events.append(("result", result))
        assert isinstance(result.structures, RigidMoleculeASUBatch)

    generator = CSPGenerator(
        packer,
        condition_func=condition,
        hooks=[hook],
        on_result=on_result,
        dedicated_stream=False,
    )
    rng = torch.Generator().manual_seed(5)
    result = generator.sample(_formula(), num_samples=2, rng=rng, option_marker=7)

    assert isinstance(result, Batch)
    assert generator.required_inputs == frozenset()
    assert generator.outputs == frozenset(
        {
            "positions",
            "atomic_numbers",
            "cell",
            "pbc",
            "csp_source_structure_id",
        }
    )
    assert generator.generator_func.outputs == generator.outputs
    assert [event[0] for event in events] == [
        "condition",
        "result",
        "after_generate",
    ]
    assert events[0][1] == (2, rng)
    assert packer.calls[0]["num_samples"] == 2
    assert packer.calls[0]["rng"] is rng
    assert packer.calls[0]["option_marker"] == 7
    assert packer.budget_calls == [{"num_samples": 2, "option_marker": 7}]
    assert result.num_graphs == 2
    assert result["charge"].dtype == torch.float32
    assert result["charge"].tolist() == [[0.0], [0.0]]
    assert result["csp_source_structure_id"].tolist() == [
        [packer.calls[0]["context"].run_id, 0],
        [packer.calls[0]["context"].run_id, 1],
    ]
    assert events[2][1] is result
    assert generator.step_count == 1


def test_known_system_charge_reaches_after_generate_hook() -> None:
    events: list[tuple[str, Any]] = []
    generator = CSPGenerator(
        _FakePacker(), hooks=[_AfterGenerate(events)], dedicated_stream=False
    )
    with pytest.warns(UserWarning, match="Qformula=2"):
        inputs = _formula(component_charge=2)
    result = generator.sample(inputs, num_samples=1, run_id=88)

    assert result.charge.shape == (1, 1)
    assert result.charge.dtype == torch.float32
    assert result.charge.tolist() == [[2.0]]
    assert events[0][1].charge.tolist() == [[2.0]]


def test_native_batch_may_omit_neutral_charge_through_callback_and_hooks() -> None:
    events: list[tuple[str, Any]] = []
    callbacks: list[PackingResult] = []
    generator = CSPGenerator(
        _FakePacker(native_batch=True, omit_native_charge=True),
        hooks=[_AfterGenerate(events)],
        on_result=callbacks.append,
        dedicated_stream=False,
    )
    result = generator.sample(_formula(), num_samples=1, run_id=89)

    assert isinstance(result, Batch)
    assert len(callbacks) == 1
    assert callbacks[0].structures.level_keys["system"] == {
        "cell",
        "pbc",
        "csp_source_structure_id",
    }
    assert result is callbacks[0].structures
    assert "charge" not in result.level_keys["system"]
    assert events == [("after_generate", result)]


def test_neutral_native_batch_pipeline_skips_charge_contract() -> None:
    observed: list[Batch] = []

    def neutral_stage(inputs=None, *, num_samples=1, rng=None):
        del num_samples, rng
        assert isinstance(inputs, Batch)
        assert "charge" not in inputs.level_keys["system"]
        observed.append(inputs)
        return inputs

    producer = CSPGenerator(
        _FakePacker(native_batch=True, omit_native_charge=True),
        dedicated_stream=False,
    )
    neutral_consumer = AtomisticGenerator(
        generator_func=neutral_stage,
        required_inputs=frozenset(),
        outputs=frozenset(),
        dedicated_stream=False,
    )
    pipeline = GenerationPipeline(stages=[producer, neutral_consumer])
    result = pipeline(_formula(), stage_kwargs=[{"num_samples": 1, "run_id": 89}, None])

    assert isinstance(result, Batch)
    assert len(observed) == 1
    assert result is observed[0]

    charged_consumer = AtomisticGenerator(
        generator_func=neutral_stage,
        required_inputs=frozenset({"charge"}),
        outputs=frozenset(),
        dedicated_stream=False,
    )
    with pytest.raises(ValueError, match=r"consumes fields \['charge'\]"):
        GenerationPipeline(
            stages=[
                CSPGenerator(
                    _FakePacker(native_batch=True, omit_native_charge=True),
                    dedicated_stream=False,
                ),
                charged_consumer,
            ]
        )


@pytest.mark.parametrize(("component_charge", "expected_charge"), [(0, 0.0), (2, 2.0)])
def test_explicit_outputs_declare_asu_charge_for_pipeline(
    component_charge: int, expected_charge: float
) -> None:
    observed: list[torch.Tensor] = []

    def consume_charge(inputs=None, *, num_samples=1, rng=None):
        del num_samples, rng
        assert isinstance(inputs, Batch)
        observed.append(inputs["charge"].clone())
        return inputs

    producer = CSPGenerator(
        _FakePacker(), outputs=CSP_OUTPUT_FIELDS | {"charge"}, dedicated_stream=False
    )
    consumer = AtomisticGenerator(
        generator_func=consume_charge,
        required_inputs=frozenset({"charge"}),
        outputs=frozenset(),
        dedicated_stream=False,
    )
    pipeline = GenerationPipeline(stages=[producer, consumer])
    if component_charge:
        with pytest.warns(UserWarning, match="Qformula=2"):
            inputs = _formula(component_charge=component_charge)
    else:
        inputs = _formula()

    result = pipeline(inputs, stage_kwargs=[{"num_samples": 1, "run_id": 90}, None])

    assert isinstance(result, Batch)
    assert producer.outputs == CSP_OUTPUT_FIELDS | {"charge"}
    assert producer.generator_func.outputs == producer.outputs
    assert len(observed) == 1
    assert observed[0].tolist() == [[expected_charge]]
    assert result["charge"].tolist() == [[expected_charge]]


@pytest.mark.parametrize("total_charge", [0.0, 2.0])
def test_native_batch_preserves_explicit_total_charge(total_charge: float) -> None:
    events: list[tuple[str, Any]] = []
    callbacks: list[PackingResult] = []
    declared_outputs = list(CSP_OUTPUT_FIELDS | {"charge"})
    generator = CSPGenerator(
        _FakePacker(native_batch=True, native_charge=total_charge),
        outputs=declared_outputs,
        hooks=[_AfterGenerate(events)],
        on_result=callbacks.append,
        dedicated_stream=False,
    )
    result = generator.sample(_formula(), num_samples=1, run_id=90)

    assert isinstance(result, Batch)
    assert generator.outputs == frozenset(declared_outputs)
    assert generator.generator_func.outputs == generator.outputs
    assert len(callbacks) == 1
    assert callbacks[0].structures is result
    assert result["charge"].dtype == torch.float32
    assert result["charge"].shape == (1, 1)
    assert result["charge"].tolist() == [[total_charge]]
    assert callbacks[0].structures["charge"].tolist() == [[total_charge]]
    assert events == [("after_generate", result)]
    assert events[0][1]["charge"].tolist() == [[total_charge]]


def test_native_charge_declaration_checks_nonempty_results() -> None:
    generator = CSPGenerator(
        _FakePacker(native_batch=True, omit_native_charge=True),
        outputs=CSP_OUTPUT_FIELDS | {"charge"},
        dedicated_stream=False,
    )

    with pytest.raises(ValueError, match=r"returned batch lacks \['charge'\]"):
        generator.sample(_formula(), num_samples=1, run_id=92)


def test_native_zero_accepted_batch_keeps_charge_omitted() -> None:
    empty_accepted = CSPGenerator(
        _FakePacker(native_batch=True, accepted=0, omit_native_charge=True),
        dedicated_stream=False,
    ).sample(_formula(), num_samples=1, run_id=91)

    assert isinstance(empty_accepted, Batch)
    assert empty_accepted.num_graphs == 0
    assert "charge" not in empty_accepted.level_keys["system"]


def test_compact_mode_returns_packing_result_without_batch_hooks() -> None:
    events: list[tuple[str, Any]] = []
    packer = _FakePacker()
    generator = CSPGenerator(
        packer,
        expand=False,
        outputs=[],
        hooks=[_AfterGenerate(events)],
        dedicated_stream=False,
    )

    result = generator.sample(_formula(), num_samples=1, run_id=19)

    assert isinstance(result, PackingResult)
    assert result.run_id == 19
    assert result.structures.structure_ids.tolist() == [[19, 0]]
    assert generator.outputs == frozenset()
    assert generator.generator_func.outputs == generator.outputs
    assert events == []


def test_compact_mode_rejects_nonempty_output_declarations() -> None:
    with pytest.raises(ValueError, match="raw-result mode.*empty outputs declaration"):
        CSPGenerator(_FakePacker(), expand=False, outputs={"charge"})


def test_empty_result_uses_typed_batch_and_calls_owner_once() -> None:
    packer = _FakePacker(accepted=0)
    callbacks: list[PackingResult] = []
    generator = CSPGenerator(packer, on_result=callbacks.append, dedicated_stream=False)

    result = generator.sample(_formula(), num_samples=1, run_id=20)

    assert len(callbacks) == 1
    assert callbacks[0].accepted_count == 0
    assert isinstance(result, Batch)
    assert result.num_graphs == 0
    assert {"positions", "atomic_numbers"} <= result.level_keys["atoms"]
    assert {
        "cell",
        "pbc",
        "charge",
        "csp_source_structure_id",
    } <= result.level_keys["system"]


def test_native_batch_is_passed_through_and_keeps_custom_stop_reason() -> None:
    packer = _FakePacker(native_batch=True, accepted=1)
    callbacks: list[PackingResult] = []
    generator = CSPGenerator(packer, on_result=callbacks.append, dedicated_stream=False)

    result = generator.sample(_formula(), num_samples=2, run_id=21)

    assert isinstance(result, Batch)
    assert result is callbacks[0].structures
    assert callbacks[0].reports[0].stop_reason == "fake_stopped"
    assert result["csp_source_structure_id"].tolist() == [[21, 0]]


def test_unknown_generated_count_and_budget_cap_contract() -> None:
    class _UnbudgetedFake:
        device = torch.device("cpu")

        def pack(self, inputs, *, num_samples, rng, context, **options):
            del rng, options
            structures = _compact(inputs, context, 1)
            return PackingResult(
                structures=structures,
                run_id=context.run_id,
                reports=(
                    PackingReport(
                        rank=context.rank,
                        requested_count=num_samples,
                        accepted_count=1,
                        stop_reason="custom_stop",
                        generated_count=None,
                    ),
                ),
            )

    result = CSPGenerator(
        _UnbudgetedFake(), expand=False, dedicated_stream=False
    ).sample(_formula(), num_samples=2, run_id=22)
    assert isinstance(result, PackingResult)
    assert result.generated_count is None
    assert result.reports[0].stop_reason == "custom_stop"

    finite = _FakePacker(budget=3, generated=4)
    with pytest.raises(ValueError, match="exceeds the local candidate budget"):
        CSPGenerator(finite, expand=False, dedicated_stream=False).sample(
            _formula(), num_samples=1, run_id=23
        )
    assert finite.calls[0]["candidate_budget"] == 3


def test_zero_candidate_budget_rejects_accepts_and_preserves_empty_result() -> None:
    nonempty = _FakePacker(budget=0, accepted=1, generated=0)
    generator = CSPGenerator(nonempty, expand=False, dedicated_stream=False)
    with pytest.raises(ValueError, match="zero candidate budget requires an empty"):
        generator.sample(_formula(), num_samples=1, run_id=24)
    assert nonempty.calls[0]["candidate_budget"] == 0

    empty = _FakePacker(budget=0, accepted=0, generated=0)
    result = CSPGenerator(empty, expand=False, dedicated_stream=False).sample(
        _formula(), num_samples=1, run_id=25
    )
    assert isinstance(result, PackingResult)
    assert result.accepted_count == result.generated_count == 0
    assert result.stop_reason == PackingStopReason.SHORTFALL
    assert empty.calls[0]["candidate_budget"] == 0


def test_formula_mismatch_and_injected_driver_options_are_rejected() -> None:
    inputs = _formula()
    changed_coordinates = inputs.conformer_positions.clone()
    changed_coordinates[0, 0] = 1
    altered = inputs.model_copy(update={"conformer_positions": changed_coordinates})
    for accepted in (1, 0):
        packer = _FakePacker(accepted=accepted)
        packer.input_override = altered
        generator = CSPGenerator(packer, expand=False, dedicated_stream=False)
        with pytest.raises(
            ValueError, match="differs from the conditioned formula input"
        ):
            generator.sample(inputs, num_samples=1)
    with pytest.raises(TypeError, match="context and candidate_budget"):
        CSPGenerator(_FakePacker(), dedicated_stream=False).sample(
            _formula(), num_samples=1, candidate_budget=2
        )
    with pytest.raises(TypeError, match="cannot be overridden per call"):
        CSPGenerator(_FakePacker(), dedicated_stream=False).sample(
            _formula(), num_samples=1, expand=False
        )


def test_compile_is_rejected_and_num_samples_remains_positive() -> None:
    with pytest.raises(NotImplementedError, match="compile_generate"):
        CSPGenerator(_FakePacker(), compile_generate=True)
    generator = CSPGenerator(_FakePacker(), dedicated_stream=False)
    with pytest.raises(NotImplementedError, match="compile"):
        generator.compile()
    with pytest.raises(ValueError, match="num_samples must be positive"):
        generator.sample(_formula(), num_samples=0)
    assert generator.step_count == 1


def test_local_startup_preserves_type_and_range_errors() -> None:
    generator = CSPGenerator(_FakePacker(), expand=False, dedicated_stream=False)

    with pytest.raises(TypeError, match="inputs must be a MolecularPackingInput"):
        generator.sample("not a formula input", num_samples=1)
    with pytest.raises(ValueError, match="num_samples must be positive"):
        generator.sample(_formula(), num_samples=0)


@pytest.mark.parametrize("device_value", ["cuda", torch.device("cuda")])
def test_constructor_pins_bare_cuda_string_or_device(monkeypatch, device_value) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    packer = _FakePacker()
    packer.device = device_value
    generator = CSPGenerator(packer, dedicated_stream=False)

    assert generator.device == torch.device("cuda:1")


@pytest.mark.parametrize("device_value", ["cuda:0", torch.device("cuda:0")])
def test_constructor_preserves_indexed_cuda_string_or_device(
    monkeypatch, device_value
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("an explicit CUDA index must be retained"),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    packer = _FakePacker()
    packer.device = device_value
    generator = CSPGenerator(packer, dedicated_stream=False)

    assert generator.device == torch.device("cuda:0")


def test_constructor_rejects_unavailable_bare_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail(
            "current device must not be queried when CUDA is unavailable"
        ),
    )

    packer = _FakePacker()
    packer.device = torch.device("cuda")
    with pytest.raises(ValueError, match=r"torch.cuda.is_available\(\) is False"):
        CSPGenerator(packer, dedicated_stream=False)


def test_constructor_rejects_out_of_range_cuda_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    packer = _FakePacker()
    packer.device = "cuda:1"
    with pytest.raises(
        ValueError, match="device=cuda:1 is out of range: 1 CUDA device"
    ):
        CSPGenerator(packer, dedicated_stream=False)


def test_constructor_preserves_device_type_and_parse_errors() -> None:
    packer = _FakePacker()
    packer.device = 1
    with pytest.raises(
        TypeError, match="packer.device must be a torch.device or device string"
    ):
        CSPGenerator(packer, dedicated_stream=False)

    packer.device = "not-a-device"
    with pytest.raises(ValueError, match="Invalid device string"):
        CSPGenerator(packer, dedicated_stream=False)
