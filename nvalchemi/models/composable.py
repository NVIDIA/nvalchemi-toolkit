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
"""Composable model composition.

:class:`ComposableModelWrapper` combines one or more
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible models under a
single runtime boundary.  The composite resolves inputs from the batch and
intermediate context, builds shared neighbor lists when needed, executes one
optional autograd-connected region, and accumulates additive outputs such as
energies, forces, and stresses.

Usage
-----
Typical usage via the ``+`` operator::

    combined = model_a + model_b

Explicit wiring is also supported::

    combined = ComposableModelWrapper(model_a, model_b)
    combined.wire_output(model_a, model_b, {"target_name": "source_name"})

Notes
-----
* A single wrapped model is valid.  This is the standard way to run one
  model under composable-managed neighbor-list planning and derivative
  handling.
* An explicit :class:`~nvalchemi.models.derivatives.DerivativeStep` may be
  inserted anywhere in the node order to request custom derivatives.
* Input resolution order is: explicit wiring, previously published context
  values, then original batch fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from nvalchemi.data import Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    PipelineContext,
)
from nvalchemi.models.derivatives import DerivativeStep
from nvalchemi.models.neighbors import (
    NeighborList,
    NeighborListBuilder,
    unify_neighbor_requirements,
)

__all__ = ["ComposableModelWrapper"]

_DEFAULT_OUTPUTS = frozenset({"energies", "forces"})
_ALL_DERIVATIVE_OUTPUTS = frozenset({"energies", "forces", "stresses"})
_OutputPublishPlan = tuple[str, bool, bool, str | None]
_INTERNAL_NEIGHBOR_KEYS = frozenset(
    {
        "edge_index",
        "neighbor_ptr",
        "unit_shifts",
        "neighbor_matrix",
        "num_neighbors",
        "neighbor_shifts",
    }
)


@runtime_checkable
class _ComposableModel(Protocol):
    """Typing protocol for one composable executable model."""

    spec: ModelConfig

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


class _ComposableContract(BaseModel):
    """Internal composite I/O contract derived from ordered nodes."""

    required_inputs: frozenset[str] = Field(
        default=frozenset(),
        description="Input keys that must be present.",
    )
    optional_inputs: frozenset[str] = Field(
        default=frozenset(),
        description="Input keys that may be supplied.",
    )
    outputs: frozenset[str] = Field(
        default=frozenset(),
        description="Output keys always produced.",
    )
    optional_outputs: dict[str, frozenset[str]] = Field(
        default_factory=dict,
        description="Conditional outputs with input dependencies.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_contract(self) -> _ComposableContract:
        overlap = self.outputs & frozenset(self.optional_outputs)
        if overlap:
            raise ValueError(
                f"Keys {overlap} appear in both outputs and optional_outputs"
            )
        all_inputs = self.required_inputs | self.optional_inputs
        for output_name, deps in self.optional_outputs.items():
            if not deps <= all_inputs:
                extra = deps - all_inputs
                raise ValueError(
                    f"optional_outputs['{output_name}'] deps {extra} "
                    f"not in required_inputs | optional_inputs"
                )
        return self


@dataclass(frozen=True, slots=True)
class _WiringEdge:
    """One directed wiring edge between two member models."""

    source: nn.Module
    target: nn.Module
    mapping: dict[str, str]


@dataclass
class _NodePlan:
    """Precomputed execution metadata for one composite node."""

    node: _ComposableModel | NeighborListBuilder | DerivativeStep
    name: str | None = None
    map_inputs: dict[str, str] = field(default_factory=dict)
    map_outputs: dict[str, str] = field(default_factory=dict)
    validate_runtime_outputs: bool = False

    has_adapt_input: bool = field(init=False, default=False)
    has_adapt_output: bool = field(init=False, default=False)
    needs_external_neighbors: bool = field(init=False, default=False)
    neighbor_share_key: tuple[float, bool] | None = field(init=False, default=None)
    output_publish_plan: dict[str, _OutputPublishPlan] = field(
        init=False,
        default_factory=dict,
    )
    autograd_required_bindings: tuple[tuple[str, str], ...] = field(
        init=False,
        default=(),
    )
    autograd_optional_bindings: tuple[tuple[str, str], ...] = field(
        init=False,
        default=(),
    )
    required_bindings: tuple[tuple[str, str], ...] = field(init=False, default=())
    optional_bindings: tuple[tuple[str, str], ...] = field(init=False, default=())
    spec: ModelConfig | None = field(init=False, default=None)
    runtime_validator: Callable[[dict[str, object], set[str]], None] | None = field(
        init=False,
        default=None,
    )

    def __post_init__(self) -> None:
        self._validate_node()
        self._compile()

    def _validate_node(self) -> None:
        node = self.node
        if isinstance(node, (DerivativeStep, NeighborListBuilder)):
            return
        if not isinstance(node, nn.Module):
            raise TypeError(f"Model node must be nn.Module, got {type(node)}")
        if not hasattr(node, "spec"):
            raise TypeError("Model node must have 'spec' attribute")
        if not isinstance(node.spec, ModelConfig):
            raise TypeError(f"node.spec must be ModelConfig, got {type(node.spec)}")

    def _compile(self) -> None:
        node = self.node
        if isinstance(node, NeighborListBuilder):
            self.neighbor_share_key = (node.config.cutoff, node.config.half_list)
            return
        if isinstance(node, DerivativeStep):
            return

        spec = node.spec
        self.spec = spec
        self.has_adapt_input = hasattr(node, "adapt_input")
        self.has_adapt_output = hasattr(node, "adapt_output")

        reverse_inputs = {value: key for key, value in self.map_inputs.items()}
        self.autograd_required_bindings = tuple(
            (key, reverse_inputs.get(key, key))
            for key in spec.required_inputs
            if key in spec.autograd_inputs
        )
        self.autograd_optional_bindings = tuple(
            (key, reverse_inputs.get(key, key))
            for key in spec.optional_inputs
            if key in spec.autograd_inputs
        )
        self.required_bindings = tuple(
            (key, reverse_inputs.get(key, key))
            for key in spec.required_inputs
            if key not in spec.autograd_inputs
        )
        self.optional_bindings = tuple(
            (key, reverse_inputs.get(key, key))
            for key in spec.optional_inputs
            if key not in spec.autograd_inputs
        )
        self.needs_external_neighbors = spec.neighbor_config.source == "external"
        if self.needs_external_neighbors:
            req = spec.neighbor_config
            self.neighbor_share_key = (
                req.cutoff,
                req.half_list if req.half_list is not None else False,
            )

        qualified_prefix = self.name + "." if self.name else None
        for key in spec.outputs | spec.optional_outputs.keys():
            mapped_key = self.map_outputs.get(key, key)
            is_additive = key in spec.additive_outputs
            qualified_key = (
                qualified_prefix + mapped_key
                if qualified_prefix and not is_additive
                else None
            )
            self.output_publish_plan[key] = (
                mapped_key,
                is_additive,
                key in spec.autograd_outputs,
                qualified_key,
            )
        if self.validate_runtime_outputs:
            self.runtime_validator = _make_runtime_validator(spec)


def _make_runtime_validator(
    spec: ModelConfig,
) -> Callable[[dict[str, object], set[str]], None]:
    """Compile one runtime output validator for a fixed model config."""

    required_outputs = spec.outputs
    optional_output_deps = tuple(spec.optional_outputs.items())

    def _validate(result: dict[str, object], available_inputs: set[str]) -> None:
        missing = required_outputs - result.keys()
        if missing:
            raise RuntimeError(f"Model did not produce required outputs: {missing}")
        for name, deps in optional_output_deps:
            if deps <= available_inputs and name not in result:
                raise RuntimeError(
                    f"Optional output '{name}' deps satisfied but not produced"
                )
        expected = required_outputs | frozenset(
            name for name, deps in optional_output_deps if deps <= available_inputs
        )
        unexpected = result.keys() - expected
        if unexpected:
            raise RuntimeError(f"Model produced unexpected outputs: {unexpected}")

    return _validate


def _find_derivative_step(node_plans: list[_NodePlan]) -> int | None:
    for index, plan in enumerate(node_plans):
        if isinstance(plan.node, DerivativeStep):
            return index
    return None


def _precompute_derivative_outputs(
    node_plans: list[_NodePlan],
    derivative_step_index: int | None,
) -> frozenset[str]:
    if derivative_step_index is None:
        return frozenset()
    node = node_plans[derivative_step_index].node
    if not isinstance(node, DerivativeStep):
        return frozenset()
    return frozenset(node.outputs)


def _compute_neighbor_lifecycle(
    node_plans: list[_NodePlan],
) -> dict[tuple[float, bool], int]:
    lifecycle: dict[tuple[float, bool], int] = {}
    for index, plan in enumerate(node_plans):
        if plan.neighbor_share_key is not None and plan.needs_external_neighbors:
            lifecycle[plan.neighbor_share_key] = index
    return lifecycle


def _compute_ambiguous_bare_keys(node_plans: list[_NodePlan]) -> None:
    bare_key_producers: dict[str, list[int]] = {}
    for index, plan in enumerate(node_plans):
        for published in plan.output_publish_plan.values():
            mapped_key, is_additive, _is_autograd, qualified_key = published
            if is_additive or qualified_key is None:
                continue
            bare_key_producers.setdefault(mapped_key, []).append(index)

    ambiguous_bare_keys = {
        bare_key
        for bare_key, producers in bare_key_producers.items()
        if len(producers) > 1
    }
    if not ambiguous_bare_keys:
        return

    for plan in node_plans:
        updated: dict[str, _OutputPublishPlan] = {}
        for key, published in plan.output_publish_plan.items():
            mapped_key, is_additive, is_autograd, qualified_key = published
            if mapped_key in ambiguous_bare_keys and qualified_key is not None:
                updated[key] = (qualified_key, is_additive, is_autograd, None)
            else:
                updated[key] = published
        plan.output_publish_plan = updated


def _validate_steps(node_plans: list[_NodePlan]) -> None:
    derivative_indices = [
        index
        for index, plan in enumerate(node_plans)
        if isinstance(plan.node, DerivativeStep)
    ]
    if len(derivative_indices) > 1:
        raise ValueError(
            "ComposableModelWrapper may contain at most one "
            f"DerivativeStep, found {len(derivative_indices)}"
        )


def _check_requested_outputs(
    outputs: frozenset[str] | None,
    pipeline_contract: _ComposableContract,
) -> None:
    if outputs is None:
        return
    producible = pipeline_contract.outputs | frozenset(
        pipeline_contract.optional_outputs
    )
    missing = outputs - producible
    if missing:
        raise ValueError(
            f"Requested outputs {missing} not produced by any step in the composite"
        )


def _check_duplicate_producers(node_plans: list[_NodePlan]) -> None:
    producers: dict[str, list[str]] = {}
    all_additive: set[str] = set()
    for plan in node_plans:
        node = plan.node
        if not hasattr(node, "spec"):
            continue
        all_additive |= node.spec.additive_outputs
        step_name = plan.name or type(node).__name__
        for published in plan.output_publish_plan.values():
            mapped_key = published[0]
            producers.setdefault(mapped_key, []).append(step_name)
    for key, produced_by in producers.items():
        if len(produced_by) > 1 and key not in all_additive:
            raise ValueError(
                f"Output '{key}' produced by {produced_by} but not declared as additive. "
                f"Use map_outputs to disambiguate or declare it as additive."
            )


def _derive_node_contract(
    plan: _NodePlan,
) -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    dict[str, frozenset[str]],
]:
    node = plan.node
    reverse_inputs = {value: key for key, value in plan.map_inputs.items()}

    if hasattr(node, "spec"):
        spec = node.spec
        required_inputs = frozenset(
            reverse_inputs.get(key, key) for key in spec.required_inputs
        )
        optional_inputs = frozenset(
            reverse_inputs.get(key, key) for key in spec.optional_inputs
        )
        node_outputs = frozenset(plan.map_outputs.get(key, key) for key in spec.outputs)
        optional_outputs = {
            plan.map_outputs.get(output_name, output_name): frozenset(
                reverse_inputs.get(dep, dep) for dep in deps
            )
            for output_name, deps in spec.optional_outputs.items()
        }
        return required_inputs, optional_inputs, node_outputs, optional_outputs

    if isinstance(node, NeighborListBuilder):
        return frozenset({"positions"}), frozenset({"cell", "pbc"}), frozenset(), {}

    if isinstance(node, DerivativeStep):
        required_inputs: set[str] = set()
        optional_inputs: set[str] = set()
        outputs_set: set[str] = set()
        optional_outputs: dict[str, frozenset[str]] = {}
        for name, (source, _wrt, mode) in node.outputs.items():
            required_inputs.add(source)
            if mode == "stress":
                optional_inputs |= {"cell", "pbc"}
                optional_outputs[name] = frozenset({"cell", "pbc"})
            else:
                outputs_set.add(name)
        return (
            frozenset(required_inputs),
            frozenset(optional_inputs),
            frozenset(outputs_set),
            optional_outputs,
        )

    raise TypeError(f"Unsupported node type: {type(node)}")


def _derive_node_display_contract(
    plan: _NodePlan,
) -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    dict[str, frozenset[str]],
]:
    """Return one node contract tailored for repr display."""

    node = plan.node
    if isinstance(node, NeighborListBuilder):
        required_inputs = frozenset({"positions"})
        optional_inputs = frozenset({"cell", "pbc"})
        if node.config.format == "coo":
            return (
                required_inputs,
                optional_inputs,
                frozenset({"edge_index", "neighbor_ptr"}),
                {"unit_shifts": frozenset({"cell", "pbc"})},
            )
        return (
            required_inputs,
            optional_inputs,
            frozenset({"neighbor_matrix", "num_neighbors"}),
            {"neighbor_shifts": frozenset({"cell", "pbc"})},
        )
    return _derive_node_contract(plan)


def _derive_pipeline_contract(node_plans: list[_NodePlan]) -> _ComposableContract:
    required_inputs: set[str] = set()
    optional_inputs: set[str] = set()
    outputs_set: set[str] = set()
    optional_outputs: dict[str, frozenset[str]] = {}
    produced: set[str] = set()

    for plan in node_plans:
        step_required, step_optional_inputs, step_outputs, step_optional_outputs = (
            _derive_node_contract(plan)
        )
        external_required = step_required - produced
        external_optional = step_optional_inputs - produced - external_required
        required_inputs |= external_required
        optional_inputs |= external_optional
        outputs_set |= step_outputs
        for output_name, deps in step_optional_outputs.items():
            external_deps = deps - produced
            if external_deps:
                optional_outputs[output_name] = frozenset(external_deps)
            else:
                outputs_set.add(output_name)
        produced |= step_outputs | set(step_optional_outputs)

    return _ComposableContract(
        required_inputs=frozenset(required_inputs),
        optional_inputs=frozenset(optional_inputs),
        outputs=frozenset(outputs_set - set(optional_outputs)),
        optional_outputs=optional_outputs,
    )


def _derive_display_pipeline_contract(
    node_plans: list[_NodePlan],
) -> _ComposableContract:
    """Return one display-oriented pipeline contract for repr output."""

    required_inputs: set[str] = set()
    optional_inputs: set[str] = set()
    outputs_set: set[str] = set()
    optional_outputs: dict[str, frozenset[str]] = {}
    produced: set[str] = set()

    for plan in node_plans:
        step_required, step_optional_inputs, step_outputs, step_optional_outputs = (
            _derive_node_display_contract(plan)
        )
        external_required = step_required - produced
        external_optional = step_optional_inputs - produced - external_required
        required_inputs |= external_required
        optional_inputs |= external_optional
        outputs_set |= step_outputs
        for output_name, deps in step_optional_outputs.items():
            external_deps = deps - produced
            if external_deps:
                optional_outputs[output_name] = frozenset(external_deps)
            else:
                outputs_set.add(output_name)
        produced |= step_outputs | set(step_optional_outputs)

    return _ComposableContract(
        required_inputs=frozenset(required_inputs),
        optional_inputs=frozenset(optional_inputs),
        outputs=frozenset(outputs_set - set(optional_outputs)),
        optional_outputs=optional_outputs,
    )


def _filter_public_display_contract(
    contract: _ComposableContract,
) -> _ComposableContract:
    """Drop internal neighbor transport keys from the public repr summary."""

    return _ComposableContract(
        required_inputs=frozenset(
            key
            for key in contract.required_inputs
            if key not in _INTERNAL_NEIGHBOR_KEYS
        ),
        optional_inputs=frozenset(
            key
            for key in contract.optional_inputs
            if key not in _INTERNAL_NEIGHBOR_KEYS
        ),
        outputs=frozenset(
            key for key in contract.outputs if key not in _INTERNAL_NEIGHBOR_KEYS
        ),
        optional_outputs={
            key: deps
            for key, deps in contract.optional_outputs.items()
            if key not in _INTERNAL_NEIGHBOR_KEYS
        },
    )


def _resolve_exported_keys(
    pipeline_contract: _ComposableContract,
    batch: Batch,
    requested_outputs: frozenset[str] | None,
) -> frozenset[str]:
    active_optional = frozenset(
        key
        for key, deps in pipeline_contract.optional_outputs.items()
        if all(hasattr(batch, dep) for dep in deps)
    )
    available = pipeline_contract.outputs | active_optional
    if requested_outputs is None:
        return available
    missing = requested_outputs - available
    if missing:
        raise KeyError(
            f"Requested exported outputs {missing} are unavailable for this batch."
        )
    return requested_outputs


def _split_by_autograd(
    models: tuple[_ComposableModel, ...],
) -> tuple[list[_ComposableModel], list[_ComposableModel]]:
    autograd: list[_ComposableModel] = []
    direct: list[_ComposableModel] = []
    seen_direct = False
    for model in models:
        if model.spec.use_autograd:
            if seen_direct:
                raise ValueError(
                    "Cannot interleave autograd and direct models in the simple path. "
                    "Use explicit wiring for complex compositions."
                )
            autograd.append(model)
        else:
            seen_direct = True
            direct.append(model)
    return autograd, direct


def _synthesize_neighbor_builders(
    models: list[_ComposableModel],
    neighbor_builder: NeighborListBuilder | None,
) -> list[NeighborListBuilder]:
    external_reqs = [
        model.spec.neighbor_config
        for model in models
        if model.spec.neighbor_config.source == "external"
    ]
    if not external_reqs:
        return []
    if neighbor_builder is not None:
        return [neighbor_builder]

    groups: dict[tuple[float, bool], int | None] = {}
    for req in external_reqs:
        key = (req.cutoff, req.half_list if req.half_list is not None else False)
        if key not in groups:
            groups[key] = req.max_neighbors
        elif req.max_neighbors is not None:
            existing = groups[key]
            groups[key] = (
                max(existing, req.max_neighbors)
                if existing is not None
                else req.max_neighbors
            )

    builders: list[NeighborListBuilder] = []
    for (cutoff, half_list), max_neighbors in groups.items():
        kwargs: dict[str, object] = {
            "cutoff": cutoff,
            "format": "matrix",
            "half_list": half_list,
        }
        if max_neighbors is not None:
            kwargs["max_neighbors"] = max_neighbors
        builders.append(NeighborListBuilder(**kwargs))
    return builders


def _synthesize_derivative_step(
    outputs: frozenset[str] | None,
) -> DerivativeStep | None:
    if outputs is None:
        return None
    forces = "forces" in outputs
    stresses = "stresses" in outputs
    if not forces and not stresses:
        return None
    return DerivativeStep(forces=forces, stresses=stresses)


def _compile_node_list(
    models: tuple[_ComposableModel, ...],
    outputs: set[str] | frozenset[str] | None,
    default_compute: frozenset[str],
    neighbor_builder: NeighborListBuilder | None,
) -> list[_ComposableModel | NeighborListBuilder | DerivativeStep]:
    autograd_models, direct_models = _split_by_autograd(models)
    all_models = list(models)
    builders = _synthesize_neighbor_builders(all_models, neighbor_builder)
    effective_outputs = frozenset(outputs) if outputs else default_compute
    derivative_step = _synthesize_derivative_step(effective_outputs)

    is_override = neighbor_builder is not None
    builder_by_key: dict[tuple[float, bool], NeighborListBuilder] = {
        (builder.config.cutoff, builder.config.half_list): builder
        for builder in builders
    }
    inserted: set[tuple[float, bool]] = set()
    override_inserted = False
    nodes: list[_ComposableModel | NeighborListBuilder | DerivativeStep] = []

    def _insert_builder_if_needed(model: _ComposableModel) -> None:
        nonlocal override_inserted
        req = model.spec.neighbor_config
        if req.source != "external":
            return
        if is_override:
            if not override_inserted:
                nodes.append(builders[0])
                override_inserted = True
            return
        key = (req.cutoff, req.half_list if req.half_list is not None else False)
        if key not in inserted and key in builder_by_key:
            nodes.append(builder_by_key[key])
            inserted.add(key)

    if autograd_models:
        for model in autograd_models:
            _insert_builder_if_needed(model)
            nodes.append(model)
        if derivative_step is not None:
            nodes.append(derivative_step)

    if direct_models:
        for model in direct_models:
            _insert_builder_if_needed(model)
            nodes.append(model)

    return nodes


def _format_display_key(
    key: str,
    *,
    optional: bool = False,
    grad_tracked: bool = False,
) -> str:
    """Return one display key with optional and autograd markers."""

    suffix = ""
    if grad_tracked:
        suffix += "*"
    if optional:
        suffix += "?"
    return f"{key}{suffix}"


def _format_display_keys(
    required: frozenset[str],
    optional: frozenset[str],
    *,
    grad_tracked: frozenset[str] = frozenset(),
) -> str:
    """Render one compact comma-separated key list for repr output."""

    rendered = [
        _format_display_key(key, grad_tracked=key in grad_tracked)
        for key in sorted(required)
    ]
    rendered.extend(
        _format_display_key(key, optional=True, grad_tracked=key in grad_tracked)
        for key in sorted(optional - required)
    )
    return ", ".join(rendered) if rendered else "\u2014"


def _format_node_label(node: object) -> str:
    """Return the label used for one effective runtime node in repr output."""

    return repr(node)


def _display_key_for_derivative_target(target: str) -> str:
    """Map one internal derivative target name to the user-facing repr key."""

    if target == "cell_scaling":
        return "cell"
    return target


def _build_runtime_model(
    models: list[nn.Module],
    nodes: tuple[object, ...],
    wiring: list[_WiringEdge],
) -> _ComposableRuntimeModel:
    """Build one runtime model without caching it on the wrapper."""

    model_names = ComposableModelWrapper._build_member_names(models)
    index_by_model = {model: idx for idx, model in enumerate(models)}
    explicit_derivative = next(
        (node for node in nodes if isinstance(node, DerivativeStep)),
        None,
    )
    if explicit_derivative is None:
        raw_nodes = _compile_node_list(
            tuple(models),
            _ALL_DERIVATIVE_OUTPUTS,
            _DEFAULT_OUTPUTS,
            None,
        )
    else:
        builders = _synthesize_neighbor_builders(models, None)
        builder_by_key = {
            (builder.config.cutoff, builder.config.half_list): builder
            for builder in builders
        }
        inserted_builders: set[tuple[float, bool]] = set()
        raw_nodes = []
        for node in nodes:
            if isinstance(node, DerivativeStep):
                raw_nodes.append(node)
                continue
            req = node.spec.neighbor_config
            if req.source == "external":
                key = (
                    req.cutoff,
                    req.half_list if req.half_list is not None else False,
                )
                if key not in inserted_builders and key in builder_by_key:
                    raw_nodes.append(builder_by_key[key])
                    inserted_builders.add(key)
            raw_nodes.append(node)
    map_inputs: list[dict[str, str] | None] = [None] * len(raw_nodes)
    map_outputs: list[dict[str, str] | None] = [None] * len(raw_nodes)
    names: list[str | None] = [None] * len(raw_nodes)

    for idx, node in enumerate(raw_nodes):
        if node in index_by_model:
            map_inputs[idx] = {}
            map_outputs[idx] = {}
            names[idx] = model_names[index_by_model[node]]

    for edge in wiring:
        source_node_idx = next(
            idx for idx, node in enumerate(raw_nodes) if node is edge.source
        )
        for target_input, source_output in edge.mapping.items():
            existing = map_outputs[source_node_idx].get(source_output)
            if existing is not None and existing != target_input:
                raise ValueError(
                    "One source output cannot be wired to multiple target input "
                    "names in the same composite."
                )
            map_outputs[source_node_idx][source_output] = target_input

    return _ComposableRuntimeModel(
        nodes=raw_nodes,
        outputs=_DEFAULT_OUTPUTS,
        map_inputs=map_inputs,
        map_outputs=map_outputs,
        names=names,
    )


def _render_composable_repr(
    composite: ComposableModelWrapper,
    runtime: _ComposableRuntimeModel,
) -> str:
    """Render one effective-pipeline repr for a composable wrapper."""

    display_contract = _filter_public_display_contract(
        _derive_display_pipeline_contract(runtime._node_plans)
    )
    derivative_targets = frozenset()
    if runtime._derivative_step_index is not None:
        derivative_node = runtime._node_plans[runtime._derivative_step_index].node
        if isinstance(derivative_node, DerivativeStep):
            derivative_targets = derivative_node.derivative_targets()
    display_derivative_targets = frozenset(
        _display_key_for_derivative_target(target) for target in derivative_targets
    )
    graph_connected_keys: set[str] = set(display_derivative_targets)

    name_by_model: dict[object, str] = {}
    for plan in runtime._node_plans:
        if plan.name is not None:
            name_by_model[plan.node] = plan.name

    wires_by_target: dict[object, list[str]] = {}
    for edge in composite._wiring:
        source_name = name_by_model.get(edge.source)
        if source_name is None:
            continue
        target_lines = wires_by_target.setdefault(edge.target, [])
        for target_input, source_output in edge.mapping.items():
            target_lines.append(f"{target_input} <- {source_name}.{source_output}")

    lines = [
        "ComposableModelWrapper(",
        "  inputs: "
        + _format_display_keys(
            display_contract.required_inputs,
            display_contract.optional_inputs,
        ),
        "  outputs: "
        + _format_display_keys(
            display_contract.outputs,
            frozenset(display_contract.optional_outputs),
        ),
    ]

    if runtime.has_autograd:
        lines.append("")
        lines.append("  -----")
        lines.append(
            "  grad enabled: "
            + _format_display_keys(
                display_derivative_targets,
                frozenset(),
                grad_tracked=display_derivative_targets,
            )
        )

    for index, plan in enumerate(runtime._node_plans):
        if (
            runtime._derivative_step_index is not None
            and index == runtime._derivative_step_index + 1
        ):
            lines.append("")
            lines.append("  -----")
            lines.append("  grad disabled")

        step_required, step_optional, step_outputs, step_optional_outputs = (
            _derive_node_display_contract(plan)
        )

        grad_inputs: frozenset[str] = frozenset()
        grad_outputs: frozenset[str] = frozenset()
        if (
            runtime._derivative_step_index is not None
            and index < runtime._derivative_step_index
            and not isinstance(plan.node, NeighborListBuilder)
        ):
            grad_inputs = frozenset(
                key
                for key in (step_required | step_optional)
                if key in graph_connected_keys
            )
            if hasattr(plan.node, "spec") and getattr(
                plan.node.spec, "use_autograd", False
            ):
                grad_outputs = frozenset(
                    step_outputs | frozenset(step_optional_outputs)
                )

        if isinstance(plan.node, DerivativeStep):
            grad_inputs = frozenset(
                key for key in step_required if key in graph_connected_keys
            )

        lines.append("")
        lines.append(f"  [{index}] {_format_node_label(plan.node)}")
        lines.append(
            "      inputs: "
            + _format_display_keys(
                step_required,
                step_optional,
                grad_tracked=grad_inputs,
            )
        )
        if plan.node in wires_by_target:
            lines.append("      wires: " + ", ".join(wires_by_target[plan.node]))
        if isinstance(plan.node, DerivativeStep):
            lines.append(
                "      grad targets: "
                + _format_display_keys(
                    display_derivative_targets,
                    frozenset(),
                )
            )
        lines.append(
            "      outputs: "
            + _format_display_keys(
                step_outputs,
                frozenset(step_optional_outputs),
                grad_tracked=grad_outputs,
            )
        )
        if grad_outputs:
            graph_connected_keys.update(step_outputs)
            graph_connected_keys.update(step_optional_outputs)

    lines.append(")")
    return "\n".join(lines)


def _resolve_model_inputs(
    plan: _NodePlan,
    batch: Batch,
    ctx: PipelineContext,
) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, context_key in plan.autograd_required_bindings:
        if not ctx.autograd_active:
            raise RuntimeError(
                f"Model input '{key}' requires an active autograd region."
            )
        if context_key not in ctx.autograd_registered_outputs:
            raise RuntimeError(
                f"Required autograd input '{context_key}' was not produced "
                f"by an earlier step in the active autograd region."
            )
        data[key] = ctx.autograd_registered_outputs[context_key]

    for key, context_key in plan.autograd_optional_bindings:
        if not ctx.autograd_active:
            continue
        if context_key in ctx.autograd_registered_outputs:
            data[key] = ctx.autograd_registered_outputs[context_key]

    for key, context_key in plan.required_bindings:
        data[key] = ctx.resolve(context_key, batch)

    for key, context_key in plan.optional_bindings:
        value = ctx.resolve_optional(context_key, batch)
        if value is not None:
            data[key] = value

    return data


def _adapt_neighbors(
    data: dict[str, object],
    plan: _NodePlan,
    batch: Batch,
    ctx: PipelineContext,
) -> None:
    if plan.neighbor_share_key is None or plan.spec is None:
        return
    neighbor_list = ctx.get_neighbor_list(*plan.neighbor_share_key)
    if neighbor_list is None:
        return
    positions = data.get("positions", batch.positions)
    cell = data.get("cell", getattr(batch, "cell", None))
    neighbor_data = neighbor_list.adapt(
        positions=positions,
        cell=cell,
        cutoff=plan.spec.neighbor_config.cutoff,
        format=plan.spec.neighbor_config.format,
        half_list=plan.spec.neighbor_config.half_list,
        max_neighbors=plan.spec.neighbor_config.max_neighbors,
    )
    data.update(neighbor_data)


def _publish_outputs(
    result: dict[str, object],
    plan: _NodePlan,
    ctx: PipelineContext,
) -> None:
    for key, value in result.items():
        published = plan.output_publish_plan.get(key)
        if published is None:
            mapped_key = key
            is_additive = False
            is_autograd_output = False
            qualified_key = None
        else:
            mapped_key, is_additive, is_autograd_output, qualified_key = published
        if is_additive:
            ctx.accumulate(mapped_key, value)
        else:
            ctx[mapped_key] = value
            if qualified_key is not None:
                ctx[qualified_key] = value
        if ctx.autograd_active and is_autograd_output:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Autograd output '{mapped_key}' must be a torch.Tensor, "
                    f"got {type(value)}"
                )
            ctx.register_autograd_output(mapped_key, value)


def _try_import_batch_neighbors(
    batch: Batch,
    cutoff: float,
    half_list: bool,
) -> NeighborList | None:
    batch_cutoff = getattr(batch, "_neighbor_list_cutoff", None)
    batch_half = getattr(batch, "_neighbor_list_half_list", None)
    if batch_cutoff is None or batch_cutoff != cutoff:
        return None
    if batch_half is None or batch_half != half_list:
        return None

    neighbor_matrix = getattr(batch, "neighbor_matrix", None)
    num_neighbors = getattr(batch, "num_neighbors", None)
    if neighbor_matrix is None or num_neighbors is None:
        return None

    return NeighborList(
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        neighbor_shifts=getattr(batch, "neighbor_shifts", None),
        batch_idx=batch.batch,
        fill_value=int(batch.positions.shape[0]),
        cutoff=cutoff,
        format="matrix",
        half_list=half_list,
    )


def _run_neighbor_builder(
    plan: _NodePlan,
    batch: Batch,
    ctx: PipelineContext,
) -> None:
    node = plan.node
    if not isinstance(node, NeighborListBuilder):
        return
    share_key = plan.neighbor_share_key
    if share_key is None:
        return
    cutoff, half_list = share_key

    if ctx.get_neighbor_list(cutoff, half_list) is not None:
        return

    batch_neighbor_list = _try_import_batch_neighbors(batch, cutoff, half_list)
    if batch_neighbor_list is not None:
        ctx.store_neighbor_list(cutoff, half_list, batch_neighbor_list)
        return

    values = node(
        positions=batch.positions,
        batch_idx=batch.batch,
        batch_ptr=batch.ptr,
        cell=getattr(batch, "cell", None),
        pbc=getattr(batch, "pbc", None),
    )
    ctx.store_neighbor_list(
        cutoff,
        half_list,
        NeighborList(
            neighbor_matrix=values["neighbor_matrix"],
            num_neighbors=values["num_neighbors"],
            neighbor_shifts=values.get("neighbor_shifts"),
            batch_idx=batch.batch,
            fill_value=int(batch.positions.shape[0]),
            cutoff=cutoff,
            format="matrix",
            half_list=half_list,
        ),
    )


def _execute_node(
    plan: _NodePlan,
    batch: Batch,
    ctx: PipelineContext,
    *,
    compute: set[str] | frozenset[str] | None = None,
) -> None:
    node = plan.node
    if isinstance(node, NeighborListBuilder):
        _run_neighbor_builder(plan, batch, ctx)
        return
    if isinstance(node, DerivativeStep):
        return

    if plan.has_adapt_input:
        data = node.adapt_input(batch, ctx, compute=compute)
    else:
        data = _resolve_model_inputs(plan, batch, ctx)

    if plan.needs_external_neighbors:
        _adapt_neighbors(data, plan, batch, ctx)

    result = node.forward(data)
    if plan.runtime_validator is not None:
        plan.runtime_validator(result, set(data))
    if plan.has_adapt_output:
        # Ensure additive outputs (e.g. energies) always pass through
        # adapt_output — they are accumulated in the pipeline context and
        # needed by DerivativeStep even when not explicitly requested.
        pipeline_compute = compute
        if pipeline_compute is not None and plan.spec is not None:
            pipeline_compute = set(pipeline_compute) | plan.spec.additive_outputs
        result = node.adapt_output(result, compute=pipeline_compute)
    _publish_outputs(result, plan, ctx)


def _run_pipeline(
    *,
    batch: Batch,
    ctx: PipelineContext,
    node_plans: list[_NodePlan],
    has_autograd: bool,
    derivative_step_index: int | None,
    derivative_produced: frozenset[str],
    neighbor_lifecycle: dict[tuple[float, bool], int],
    requested_outputs: frozenset[str] | None = None,
) -> None:
    if has_autograd and derivative_step_index is not None:
        derivative_node = node_plans[derivative_step_index].node
        if not isinstance(derivative_node, DerivativeStep):
            raise RuntimeError("Invalid derivative node configuration.")
        ctx.activate_autograd(batch, derivative_node.derivative_targets())

    for index, plan in enumerate(node_plans):
        if isinstance(plan.node, DerivativeStep):
            if not ctx.autograd_active:
                raise RuntimeError("Derivative step requires an active autograd region")
            plan.node.compute(batch, ctx)
            ctx.clear_autograd()
            continue

        effective_compute: set[str] | None = None
        if requested_outputs is not None:
            if (
                has_autograd
                and derivative_step_index is not None
                and index < derivative_step_index
            ):
                effective_compute = set(requested_outputs - derivative_produced)
            else:
                effective_compute = set(requested_outputs)

        _execute_node(plan, batch, ctx, compute=effective_compute)

        for key, last_idx in neighbor_lifecycle.items():
            if last_idx == index:
                ctx.remove_neighbor_list(*key)


def _materialize(
    *,
    ctx: PipelineContext,
    requested: frozenset[str],
    exported: frozenset[str],
    detach: bool = True,
) -> dict[str, Any]:
    del requested
    result: dict[str, Any] = {}
    for key in exported:
        if key not in ctx:
            continue
        value = ctx[key]
        if detach and isinstance(value, Tensor):
            value = value.detach()
        result[key] = value
    return result


class _ComposableRuntimeModel(nn.Module):
    """Private runtime object used by :class:`ComposableModelWrapper`."""

    def __init__(
        self,
        *models: _ComposableModel,
        nodes: list[_ComposableModel | NeighborListBuilder | DerivativeStep]
        | None = None,
        outputs: set[str] | frozenset[str] | None = None,
        neighbor_builder: NeighborListBuilder | None = None,
        map_inputs: list[dict[str, str] | None] | None = None,
        map_outputs: list[dict[str, str] | None] | None = None,
        names: list[str | None] | None = None,
        name: str | None = None,
        validate_runtime_outputs: bool = False,
    ) -> None:
        super().__init__()
        if nodes is not None and models:
            raise ValueError("Cannot combine positional models with nodes= keyword")
        if nodes is not None:
            raw_nodes = nodes
        elif models:
            raw_nodes = _compile_node_list(
                models,
                outputs,
                _DEFAULT_OUTPUTS,
                neighbor_builder,
            )
        else:
            raise ValueError(
                "Composable runtime requires either positional models or nodes="
            )

        count = len(raw_nodes)
        _map_inputs = map_inputs or [None] * count
        _map_outputs = map_outputs or [None] * count
        _names = names or [None] * count
        if (
            len(_map_inputs) != count
            or len(_map_outputs) != count
            or len(_names) != count
        ):
            raise ValueError(
                "map_inputs, map_outputs, and names must have the same length as nodes "
                f"({count})"
            )

        self._node_plans: list[_NodePlan] = []
        for index in range(count):
            self._node_plans.append(
                _NodePlan(
                    node=raw_nodes[index],
                    name=_names[index],
                    map_inputs=_map_inputs[index] or {},
                    map_outputs=_map_outputs[index] or {},
                    validate_runtime_outputs=validate_runtime_outputs,
                )
            )

        self._name = name
        self._outputs = frozenset(outputs) if outputs else None
        self._default_compute = frozenset(outputs) if outputs else _DEFAULT_OUTPUTS
        self._validate_runtime_outputs = validate_runtime_outputs
        self._derivative_step_index = _find_derivative_step(self._node_plans)
        self._derivative_produced = _precompute_derivative_outputs(
            self._node_plans,
            self._derivative_step_index,
        )
        self._neighbor_lifecycle = _compute_neighbor_lifecycle(self._node_plans)
        _compute_ambiguous_bare_keys(self._node_plans)
        _validate_steps(self._node_plans)
        self._register_wrapped_models()
        self.pipeline_contract = _derive_pipeline_contract(self._node_plans)
        self._validate_pipeline()

    @property
    def nodes(self) -> list[_ComposableModel | NeighborListBuilder | DerivativeStep]:
        """Return raw runtime nodes for introspection."""

        return [plan.node for plan in self._node_plans]

    @property
    def neighbor_requirements(self) -> list[NeighborConfig]:
        """Return distinct external neighbor requirements."""

        reqs = [
            plan.spec.neighbor_config
            for plan in self._node_plans
            if plan.spec is not None and plan.spec.neighbor_config.source == "external"
        ]
        if not reqs:
            return []
        unified = unify_neighbor_requirements(reqs)
        grouped: dict[tuple[float, bool], NeighborConfig] = {}
        for req in reqs:
            key = (req.cutoff, req.half_list if req.half_list is not None else False)
            grouped.setdefault(key, req)
        ordered = list(grouped.values())
        if len(ordered) == 1:
            return ordered
        return ordered if any(req.format == "matrix" for req in ordered) else [unified]

    @property
    def has_autograd(self) -> bool:
        """Return whether this composite has an autograd derivative boundary."""

        return self._derivative_step_index is not None

    def _register_wrapped_models(self) -> None:
        self._model_modules = nn.ModuleList(
            plan.node
            for plan in self._node_plans
            if isinstance(plan.node, nn.Module)
            and hasattr(plan.node, "spec")
            and isinstance(plan.node.spec, ModelConfig)
        )

    def _validate_pipeline(self) -> None:
        _check_requested_outputs(self._outputs, self.pipeline_contract)
        _check_duplicate_producers(self._node_plans)

    def _resolve_exported_keys(
        self,
        batch: Batch,
        requested_outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        requested = (
            frozenset(requested_outputs)
            if requested_outputs is not None
            else self._outputs
        )
        return _resolve_exported_keys(self.pipeline_contract, batch, requested)

    def forward(
        self,
        batch: Batch,
        *,
        compute: set[str] | frozenset[str] | None = None,
        detach: bool = True,
    ) -> dict[str, Any]:
        """Execute the composed runtime and return a plain output mapping."""

        requested = frozenset(compute) if compute else self._default_compute
        ctx = PipelineContext()
        _run_pipeline(
            batch=batch,
            ctx=ctx,
            node_plans=self._node_plans,
            has_autograd=self.has_autograd,
            derivative_step_index=self._derivative_step_index,
            derivative_produced=self._derivative_produced,
            neighbor_lifecycle=self._neighbor_lifecycle,
            requested_outputs=requested,
        )
        exported = self._resolve_exported_keys(batch, requested)
        return _materialize(
            ctx=ctx,
            requested=requested,
            exported=exported,
            detach=detach,
        )

    def compute(self, batch: Batch, ctx: PipelineContext) -> None:
        """Execute all nodes in order without output filtering."""

        _run_pipeline(
            batch=batch,
            ctx=ctx,
            node_plans=self._node_plans,
            has_autograd=self.has_autograd,
            derivative_step_index=self._derivative_step_index,
            derivative_produced=self._derivative_produced,
            neighbor_lifecycle=self._neighbor_lifecycle,
        )


class ComposableModelWrapper(BaseModelMixin, nn.Module):
    """Compose models with additive merge and explicit dependency wiring.

    Parameters
    ----------
    *nodes
        Ordered member models, explicit derivative steps, or nested
        composites.

    Attributes
    ----------
    models : nn.ModuleList
        Flat list of constituent model wrappers.
    nodes : tuple[object, ...]
        User-facing execution order including explicit derivative steps.

    Notes
    -----
    ``wire_output(source, target, mapping)`` interprets ``mapping`` as
    ``{target_input_key: source_output_key}``, matching the simplified design
    examples in ``_md/42_simplified_composable_models_design.md``.
    The composite ``repr`` renders an effective-pipeline summary with
    zero-based step indices. Explicit external neighbor requirements appear
    as synthesized :class:`~nvalchemi.models.neighbors.NeighborListBuilder`
    steps, while internal-neighbor wrappers remain represented on their own
    model step.
    """

    def __init__(self, *nodes: object) -> None:
        super().__init__()
        flat_nodes: list[object] = []
        wiring: list[_WiringEdge] = []
        for node in nodes:
            if isinstance(node, ComposableModelWrapper):
                flat_nodes.extend(list(node.nodes))
                wiring.extend(node._wiring)
                continue
            if isinstance(node, DerivativeStep):
                flat_nodes.append(node)
                continue
            if not isinstance(node, nn.Module):
                raise TypeError(
                    "ComposableModelWrapper expects nn.Module members or "
                    f"DerivativeStep nodes, got {type(node).__name__}."
                )
            if not hasattr(node, "spec"):
                raise TypeError(
                    "ComposableModelWrapper members must expose a 'spec' attribute."
                )
            flat_nodes.append(node)

        model_count = sum(
            1
            for node in flat_nodes
            if isinstance(node, nn.Module) and hasattr(node, "spec")
        )
        derivative_count = sum(
            1 for node in flat_nodes if isinstance(node, DerivativeStep)
        )
        if model_count == 0:
            raise ValueError("ComposableModelWrapper requires at least one model.")
        if derivative_count > 1:
            raise ValueError(
                "ComposableModelWrapper supports at most one DerivativeStep."
            )

        self.models = nn.ModuleList(
            [
                node
                for node in flat_nodes
                if isinstance(node, nn.Module) and hasattr(node, "spec")
            ]
        )
        self._nodes: tuple[object, ...] = tuple(flat_nodes)
        self._wiring: list[_WiringEdge] = wiring
        self._compiled_pipeline: _ComposableRuntimeModel | None = None
        self.spec = self._derive_model_config()

    def __add__(self, other: object) -> ComposableModelWrapper:
        """Return a flattened composite containing both operands."""

        return ComposableModelWrapper(self, other)

    def __repr__(self) -> str:
        """Return a side-effect-free effective-pipeline summary.

        Returns
        -------
        str
            Multiline summary of the current effective runtime plan. The
            display uses zero-based step indices, includes synthesized
            external neighbor-builder steps, shows explicit wires when
            present, and does not cache the internal compiled pipeline.
        """

        runtime = self._build_runtime_model()
        return _render_composable_repr(self, runtime)

    @property
    def nodes(self) -> tuple[object, ...]:
        """Return user-level composite nodes in execution order."""

        return self._nodes

    def wire_output(
        self,
        source: nn.Module,
        target: nn.Module,
        mapping: dict[str, str],
    ) -> ComposableModelWrapper:
        """Wire one source model output into one target model input.

        Parameters
        ----------
        source
            Upstream model that produces the values.
        target
            Downstream model that consumes the values.
        mapping
            Mapping from target input key to source output key.

        Returns
        -------
        ComposableModelWrapper
            The composite itself for fluent chaining.
        """

        members = set(self.models)
        if source not in members:
            raise ValueError("wire_output source model is not part of this composite.")
        if target not in members:
            raise ValueError("wire_output target model is not part of this composite.")

        source_outputs = source.spec.outputs | frozenset(source.spec.optional_outputs)
        target_inputs = target.spec.required_inputs | target.spec.optional_inputs
        for target_input, source_output in mapping.items():
            if target_input not in target_inputs:
                raise KeyError(
                    f"Target input '{target_input}' is not declared by "
                    f"{type(target).__name__}."
                )
            if source_output not in source_outputs:
                raise KeyError(
                    f"Source output '{source_output}' is not declared by "
                    f"{type(source).__name__}."
                )

        self._wiring.append(
            _WiringEdge(source=source, target=target, mapping=dict(mapping))
        )
        self._compiled_pipeline = None
        return self

    @property
    def pipeline_contract(self) -> _ComposableContract:
        """Expose the derived internal pipeline contract.

        Returns
        -------
        _ComposableContract
            Derived required inputs, optional inputs, and exported outputs
            for the current composite layout.
        """

        return self._pipeline.pipeline_contract

    @property
    def runtime_nodes(self) -> list[object]:
        """Expose the internal node list for runtime integrations.

        Returns
        -------
        list[object]
            Internal execution nodes, including synthesized neighbor
            builders and explicit derivative steps.
        """

        return self._pipeline.nodes

    @property
    def neighbor_requirements(self) -> list[NeighborConfig]:
        """Expose distinct external neighbor requirements.

        Returns
        -------
        list[NeighborConfig]
            Distinct structural neighbor-list requirements consumed by the
            composite runtime.
        """

        return self._pipeline.neighbor_requirements

    @property
    def _pipeline(self) -> _ComposableRuntimeModel:
        if self._compiled_pipeline is None:
            self._compiled_pipeline = self._compile_pipeline()
        return self._compiled_pipeline

    def _build_runtime_model(self) -> _ComposableRuntimeModel:
        """Build one runtime model without caching it on the wrapper."""

        return _build_runtime_model(
            list(self.models),
            self._nodes,
            self._wiring,
        )

    def _compile_pipeline(self) -> _ComposableRuntimeModel:
        """Compile member models and wiring into the internal runtime."""

        return self._build_runtime_model()

    @staticmethod
    def _build_member_names(models: list[nn.Module]) -> list[str]:
        """Return stable unique names for internal runtime nodes."""

        seen: dict[str, int] = {}
        names: list[str] = []
        for idx, model in enumerate(models):
            raw_name = getattr(model, "_name", None)
            if raw_name is None:
                raw_name = (
                    type(model)
                    .__name__.removesuffix("Wrapper")
                    .removesuffix("Model")
                    .lower()
                )
            count = seen.get(raw_name, 0)
            seen[raw_name] = count + 1
            names.append(raw_name if count == 0 else f"{raw_name}_{count}")
        return names

    def forward(
        self,
        batch: Batch,
        *,
        compute: set[str] | frozenset[str] | None = None,
        detach: bool = True,
    ) -> dict[str, Any]:
        """Execute the composed model.

        Parameters
        ----------
        batch
            Input batch providing the base fields for all model calls.
        compute
            Requested output keys.  When omitted, the composite defaults to
            ``{"energies", "forces"}``.
        detach
            If ``True``, detach exported tensors before returning them.

        Returns
        -------
        dict[str, Any]
            Output mapping restricted to the requested public keys.
        """

        public_compute = frozenset(compute) if compute is not None else _DEFAULT_OUTPUTS
        outputs = self._pipeline(batch, compute=public_compute, detach=detach)
        return {key: value for key, value in outputs.items() if key in public_compute}

    def compute(self, batch: Batch, ctx: PipelineContext) -> None:
        """Execute the low-level composed model into one runtime context.

        Parameters
        ----------
        batch
            Input batch providing the base fields for all model calls.
        ctx
            Runtime context populated in-place with published outputs.
        """

        self._pipeline.compute(batch, ctx)

    def _derive_model_config(self) -> ModelConfig:
        """Derive one execution contract for the composite itself."""

        required_inputs: set[str] = set()
        optional_inputs: set[str] = set()
        outputs: set[str] = set()
        optional_outputs: dict[str, frozenset[str]] = {}
        additive_outputs: set[str] = set()
        use_autograd = False
        neighbor_reqs: list[NeighborConfig] = []

        for node in self._nodes:
            if isinstance(node, DerivativeStep):
                use_autograd = True
                outputs.update(node.outputs)
                continue
            spec = node.spec
            required_inputs.update(spec.required_inputs)
            optional_inputs.update(spec.optional_inputs)
            outputs.update(spec.outputs)
            optional_outputs.update(spec.optional_outputs)
            additive_outputs.update(spec.additive_outputs)
            use_autograd = use_autograd or bool(spec.use_autograd)
            if spec.neighbor_config.source == "external":
                neighbor_reqs.append(spec.neighbor_config)

        # When no explicit DerivativeStep is present but autograd models
        # exist, the pipeline will synthesize an implicit derivative step
        # that produces forces and stresses.  Reflect that in the contract.
        has_explicit_derivative = any(
            isinstance(node, DerivativeStep) for node in self._nodes
        )
        if use_autograd and not has_explicit_derivative:
            outputs.update({"forces", "stresses"})
            additive_outputs.update({"forces", "stresses"})

        if not neighbor_reqs:
            neighbor_config = NeighborConfig()
        elif len(neighbor_reqs) == 1:
            neighbor_config = neighbor_reqs[0]
        else:
            neighbor_config = unify_neighbor_requirements(neighbor_reqs)

        return ModelConfig(
            required_inputs=frozenset(required_inputs),
            optional_inputs=frozenset(optional_inputs - required_inputs),
            outputs=frozenset(outputs),
            optional_outputs=optional_outputs,
            additive_outputs=frozenset(additive_outputs & outputs),
            use_autograd=use_autograd,
            neighbor_config=neighbor_config,
        )
