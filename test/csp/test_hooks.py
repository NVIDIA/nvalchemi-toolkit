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
"""Generation-hook filtering through the public Batch and pipeline APIs."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.csp.hooks import DeduplicateHook
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.hooks import GenerationContext
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch(
    structures: list[torch.Tensor],
    *,
    atomic_numbers: list[torch.Tensor] | None = None,
    run_id: int = 71,
    device: str = "cpu",
) -> Batch:
    data = []
    for index, positions in enumerate(structures):
        numbers = (
            atomic_numbers[index]
            if atomic_numbers is not None
            else torch.ones(len(positions), dtype=torch.int64)
        )
        item = AtomicData(positions=positions.to(torch.float32), atomic_numbers=numbers)
        item.add_node_property(
            "hook_test_atom_id",
            torch.arange(len(positions), dtype=torch.int64) + 10 * index,
        )
        item.add_system_property(
            "csp_source_structure_id",
            torch.tensor([[run_id, index]], dtype=torch.int64),
        )
        data.append(item)
    return Batch.from_data_list(data).to(device)


def _generator(batch: Batch, *hooks) -> AtomisticGenerator:
    def generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
        del inputs, num_samples, rng, kwargs
        return batch

    return AtomisticGenerator(
        generator_func=generate,
        required_inputs=frozenset(),
        outputs=frozenset({"positions", "atomic_numbers", "csp_source_structure_id"}),
        hooks=list(hooks),
        device=batch.device,
        dedicated_stream=False,
    )


class _MaskEngine:
    def __init__(self, result):
        self.result = result
        self.calls: list[Batch] = []

    def deduplicate(self, batch: Batch):
        self.calls.append(batch)
        return self.result(batch) if callable(self.result) else self.result


class _CaptureContext:
    stage = GenerationStage.AFTER_GENERATE
    frequency = 1

    def __init__(self) -> None:
        self.context: GenerationContext | None = None

    def __call__(self, ctx: GenerationContext, stage: GenerationStage) -> None:
        self.context = ctx


class _KeepInputRows:
    stage = GenerationStage.AFTER_GENERATE
    frequency = 1

    def __init__(self, keep: torch.Tensor) -> None:
        self.keep = keep

    def __call__(self, ctx: GenerationContext, stage: GenerationStage) -> None:
        ctx.batch = ctx.batch[self.keep]
        ctx.accepted_mask = self.keep


def _square() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )


def test_custom_engine_filters_current_input_and_preserves_alignment() -> None:
    original = _batch(
        [torch.arange(12, dtype=torch.float32).reshape(4, 3) + 3 * i for i in range(4)]
    )
    engine = _MaskEngine(torch.tensor([False, True, True]))
    capture = _CaptureContext()
    generator = _generator(
        original,
        _KeepInputRows(torch.tensor([False, True, True, True])),
        DeduplicateHook(engine),
        capture,
    )

    result = generator.sample()

    assert result.csp_source_structure_id.tolist() == [[71, 2], [71, 3]]
    assert result.hook_test_atom_id.tolist() == [20, 21, 22, 23, 30, 31, 32, 33]
    assert len(engine.calls) == 1
    assert engine.calls[0].csp_source_structure_id.tolist() == [
        [71, 1],
        [71, 2],
        [71, 3],
    ]
    assert capture.context is not None
    assert capture.context.sample is original
    assert capture.context.batch is result
    assert capture.context.accepted_mask.tolist() == [False, True, True]
    assert capture.context.accepted_mask.dtype is torch.bool
    assert capture.context.accepted_mask.device == original.device


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        ([True, False], TypeError, "torch.Tensor"),
        (torch.tensor([0, 1]), TypeError, "torch.bool"),
        (torch.tensor([[True, False]]), ValueError, "one-dimensional"),
        (torch.tensor([True]), ValueError, "length"),
        (torch.empty((2,), dtype=torch.bool, device="meta"), ValueError, "device"),
    ],
)
def test_invalid_engine_masks_leave_context_unchanged(result, error, message) -> None:
    batch = _batch([torch.zeros((2, 3)), torch.ones((2, 3))])
    previous_mask = torch.tensor([True, False])
    context = GenerationContext(batch=batch, sample=batch, accepted_mask=previous_mask)
    hook = DeduplicateHook(_MaskEngine(result))

    with pytest.raises(error, match=message):
        hook(context, GenerationStage.AFTER_GENERATE)

    assert context.batch is batch
    assert context.sample is batch
    assert context.accepted_mask is previous_mask


def test_integer_retained_indices_are_not_an_engine_mask() -> None:
    batch = _batch([torch.zeros((2, 3)), torch.ones((2, 3))])
    with pytest.raises(TypeError, match="torch.bool"):
        DeduplicateHook(_MaskEngine(torch.tensor([0, 1])))(
            GenerationContext(batch=batch), GenerationStage.AFTER_GENERATE
        )


def test_engine_and_batch_validation_happen_before_context_replacement() -> None:
    batch = _batch([torch.zeros((2, 3)), torch.ones((2, 3))])
    previous_mask = torch.tensor([True, False])
    context = GenerationContext(batch=batch, sample=batch, accepted_mask=previous_mask)

    class _FailingEngine:
        def deduplicate(self, batch: Batch):
            raise RuntimeError("engine failed")

    with pytest.raises(RuntimeError, match="engine failed"):
        DeduplicateHook(_FailingEngine())(context, GenerationStage.AFTER_GENERATE)
    assert context.batch is batch
    assert context.accepted_mask is previous_mask

    context.batch = None
    with pytest.raises(TypeError, match="ctx.batch to be a Batch"):
        DeduplicateHook(_MaskEngine(torch.ones(2, dtype=torch.bool)))(
            context, GenerationStage.AFTER_GENERATE
        )
    assert context.batch is None
    assert context.accepted_mask is previous_mask


def test_other_stage_does_no_work_and_empty_batch_skips_engine() -> None:
    template = _batch([torch.zeros((2, 3)), torch.ones((2, 3))])
    empty = Batch.empty_like(template)
    engine = _MaskEngine(torch.ones(2, dtype=torch.bool))
    hook = DeduplicateHook(engine)
    context = GenerationContext(batch=None)
    hook(context, GenerationStage.BEFORE_CONDITION)
    assert context.batch is None
    assert engine.calls == []

    context = GenerationContext(batch=empty, sample=empty)
    hook(context, GenerationStage.AFTER_GENERATE)
    assert context.batch is empty
    assert context.accepted_mask.shape == (0,)
    assert context.accepted_mask.dtype is torch.bool
    assert context.accepted_mask.device == empty.device
    assert engine.calls == []


def test_frequency_gating_and_raw_generation_boundary() -> None:
    batch = _batch([torch.zeros((2, 3)), torch.ones((2, 3))])
    engine = _MaskEngine(torch.tensor([True, False]))
    generator = _generator(batch, DeduplicateHook(engine, frequency=2))

    first = generator.sample()
    second = generator.sample()

    assert first.num_graphs == 1
    assert second.num_graphs == 2
    assert len(engine.calls) == 1

    raw = {"sample": torch.tensor([1])}
    raw_generator = AtomisticGenerator(
        generator_func=lambda *args, **kwargs: raw,
        required_inputs=frozenset(),
        outputs=frozenset(),
        hooks=[DeduplicateHook(engine)],
        device="cpu",
        dedicated_stream=False,
    )
    assert raw_generator.sample() is raw
    assert len(engine.calls) == 1


@pytest.mark.parametrize("frequency", [True, 1.5, "2"])
def test_frequency_type_is_validated(frequency) -> None:
    with pytest.raises(TypeError, match="frequency"):
        DeduplicateHook(
            _MaskEngine(torch.empty(0, dtype=torch.bool)), frequency=frequency
        )


@pytest.mark.parametrize("frequency", [0, -1])
def test_frequency_must_be_positive(frequency) -> None:
    with pytest.raises(ValueError, match="frequency"):
        DeduplicateHook(
            _MaskEngine(torch.empty(0, dtype=torch.bool)), frequency=frequency
        )


def test_engine_must_provide_callable_deduplicate() -> None:
    with pytest.raises(TypeError, match="callable deduplicate"):
        DeduplicateHook(object())


@pytest.mark.parametrize(
    ("cutoff", "threshold", "confirm", "error"),
    [
        (0.0, 0.0, None, ValueError),
        (float("inf"), 0.0, None, ValueError),
        (2.0, -0.1, None, ValueError),
        (2.0, float("nan"), None, ValueError),
        (2.0, 0.0, 1, TypeError),
    ],
)
def test_radial_factory_validates_options(cutoff, threshold, confirm, error) -> None:
    with pytest.raises(error):
        DeduplicateHook.radial(cutoff=cutoff, threshold=threshold, confirm=confirm)


@pytest.mark.parametrize("threshold", [True, "0.1"])
def test_radial_factory_keeps_wrong_type_threshold_error(threshold) -> None:
    with pytest.raises(ValueError, match="threshold must be finite and nonnegative"):
        DeduplicateHook.radial(cutoff=2.0, threshold=threshold)


def test_radial_engine_keeps_first_duplicate_and_distinct_geometry() -> None:
    square = _square()
    rectangle = torch.tensor([[0.0, 0, 0], [2.0, 0, 0], [2.0, 1.0, 0], [0.0, 1.0, 0]])
    batch = _batch([square, square + 4, rectangle])
    hook = DeduplicateHook.radial(cutoff=2.5, threshold=0.0)
    result = _generator(batch, hook).sample()

    assert result.csp_source_structure_id.tolist() == [[71, 0], [71, 2]]


def test_radial_engine_uses_typed_neighbors_for_element_sensitive_screening() -> None:
    square = _square()
    numbers_a = torch.tensor([1, 1, 2, 2], dtype=torch.int64)
    numbers_b = torch.tensor([1, 2, 1, 2], dtype=torch.int64)
    batch = _batch([square, square + 3], atomic_numbers=[numbers_a, numbers_b])
    result = _generator(
        batch,
        DeduplicateHook.radial(cutoff=2.0, threshold=0.1),
    ).sample()

    assert result.csp_source_structure_id.tolist() == [[71, 0], [71, 1]]


def test_radial_engine_uses_center_types_for_homogeneous_elements() -> None:
    square = _square()
    batch = _batch(
        [square, square + 3],
        atomic_numbers=[torch.ones(4, dtype=torch.int64), torch.full((4,), 2)],
    )
    result = _generator(
        batch,
        DeduplicateHook.radial(cutoff=2.0, threshold=0.0),
    ).sample()

    assert result.csp_source_structure_id.tolist() == [[71, 0], [71, 1]]


def test_confirmation_can_reject_radial_proposals() -> None:
    square = _square()
    batch = _batch([square, square + 3])
    proposals: list[torch.Tensor] = []

    def reject_all(pairs: torch.Tensor) -> torch.Tensor:
        proposals.append(pairs.clone())
        return pairs[:0]

    result = _generator(
        batch,
        DeduplicateHook.radial(cutoff=2.0, threshold=0.0, confirm=reject_all),
    ).sample()

    assert result.csp_source_structure_id.tolist() == [[71, 0], [71, 1]]
    assert proposals
    assert all(pairs.dtype is torch.int32 for pairs in proposals)
    assert all(pairs.device == batch.device for pairs in proposals)
    assert all(pairs.shape[1:] == (2,) for pairs in proposals)
    assert all(torch.all(pairs[:, 0] > pairs[:, 1]) for pairs in proposals)


def test_radial_engine_has_no_cross_call_pool() -> None:
    square = _square()
    batches = [_batch([square]), _batch([square + 5], run_id=72)]

    def generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
        del inputs, num_samples, rng, kwargs
        return batches.pop(0)

    generator = AtomisticGenerator(
        generator_func=generate,
        required_inputs=frozenset(),
        outputs=frozenset({"positions", "atomic_numbers", "csp_source_structure_id"}),
        hooks=[DeduplicateHook.radial(cutoff=2.0, threshold=0.0)],
        device="cpu",
        dedicated_stream=False,
    )
    assert generator.sample().csp_source_structure_id.tolist() == [[71, 0]]
    assert generator.sample().csp_source_structure_id.tolist() == [[72, 0]]


def test_pipeline_short_circuits_after_empty_filter_without_mutating_input() -> None:
    batch = _batch([_square(), _square() + 4])
    snapshot = {name: value.clone() for name, value in batch}
    engine = _MaskEngine(torch.zeros(2, dtype=torch.bool))
    capture = _CaptureContext()
    generator = _generator(batch, DeduplicateHook(engine), capture)
    dynamics = FIRE2(model=DemoModelWrapper(DemoModel()), dt=0.05, n_steps=1)
    pipeline = GenerationPipeline(stages=[generator, dynamics])

    result = pipeline()

    assert result.num_graphs == 0
    assert result.device == batch.device
    assert result.system_capacity == batch.system_capacity
    assert {name for name, _ in result} == set(snapshot)
    assert result.csp_source_structure_id.shape == (result.system_capacity, 2)
    assert result.csp_source_structure_id.dtype is torch.int64
    assert result.hook_test_atom_id.shape == (batch.num_nodes,)
    assert capture.context is not None
    assert capture.context.sample is batch
    assert capture.context.batch is result
    assert capture.context.accepted_mask.tolist() == [False, False]
    assert dynamics.step_count == 0
    assert batch.num_graphs == 2
    for name, value in snapshot.items():
        torch.testing.assert_close(batch[name], value)


def test_nonempty_pipeline_passes_filtered_batch_to_dynamics() -> None:
    square = _square()
    batch = _batch([square, square + 4, square + 8])
    engine = _MaskEngine(torch.tensor([True, False, True]))
    generator = _generator(batch, DeduplicateHook(engine))
    dynamics = FIRE2(model=DemoModelWrapper(DemoModel()), dt=0.05, n_steps=1)
    pipeline = GenerationPipeline(stages=[generator, dynamics])

    result = pipeline()

    assert result.num_graphs == 2
    assert result.csp_source_structure_id.tolist() == [[71, 0], [71, 2]]
    assert result.hook_test_atom_id.tolist() == [0, 1, 2, 3, 20, 21, 22, 23]
    assert len(engine.calls) == 1
    assert engine.calls[0].num_graphs == 3
    assert dynamics.step_count == 1


def test_cuda_radial_duplicate_mask_and_source_ids() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda:0")
    square = _square()
    batch = _batch([square, square + 5, square + 10], device=str(device))
    engine = DeduplicateHook.radial(cutoff=2.0, threshold=0.0)
    capture = _CaptureContext()

    result = _generator(batch, engine, capture).sample()

    assert result.device == device
    assert result.csp_source_structure_id.tolist() == [[71, 0]]
    assert capture.context is not None
    assert capture.context.accepted_mask.device == device
    assert capture.context.accepted_mask.tolist() == [True, False, False]


def test_cuda_custom_mask_preserves_atom_and_system_alignment() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda:0")
    batch = _batch(
        [torch.zeros((2, 3)), torch.ones((3, 3)), torch.full((4, 3), 2.0)],
        device=str(device),
    )
    capture = _CaptureContext()
    result = _generator(
        batch,
        DeduplicateHook(_MaskEngine(torch.tensor([False, True, False], device=device))),
        capture,
    ).sample()

    assert result.device == device
    assert result.csp_source_structure_id.tolist() == [[71, 1]]
    assert result.atomic_numbers.tolist() == [1, 1, 1]
    assert result.hook_test_atom_id.tolist() == [10, 11, 12]
    assert capture.context is not None
    assert capture.context.accepted_mask.device == device
    assert capture.context.accepted_mask.tolist() == [False, True, False]


def test_cuda_wrong_device_mask_leaves_context_unchanged() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    batch = _batch([torch.zeros((2, 3)), torch.ones((2, 3))], device="cuda:0")
    previous_mask = torch.tensor([True, False], device="cuda:0")
    context = GenerationContext(batch=batch, sample=batch, accepted_mask=previous_mask)
    hook = DeduplicateHook(_MaskEngine(torch.ones(2, dtype=torch.bool)))

    with pytest.raises(ValueError, match="input Batch device"):
        hook(context, GenerationStage.AFTER_GENERATE)

    assert context.batch is batch
    assert context.sample is batch
    assert context.accepted_mask is previous_mask
