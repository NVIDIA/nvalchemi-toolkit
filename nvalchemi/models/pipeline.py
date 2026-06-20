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
"""Pipeline-based model composition.

:class:`PipelineModelWrapper` organizes models into **groups**, where each
group is a mini-pipeline with its own derivative computation strategy.
The top level sums outputs across groups.

Composition is available via the ``+`` operator for simple additive sums,
or via explicit ``PipelineModelWrapper`` construction for dependent
pipelines and custom derivative computation.

Motivating example — AIMNet2 + Ewald + DFTD3::

    pipe = PipelineModelWrapper(groups=[
        PipelineGroup(
            steps=[
                aimnet2,
                ewald,
            ],
            use_autograd=True,
        ),
        PipelineGroup(steps=[dftd3]),
    ])

See the module docstring or the proposal for full composition examples.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
from torch import nn

from nvalchemi._typing import (
    Energy,
    LatticeVectors,
    ModelOutputs,
    NodePositions,
    StrainDisplacement,
)
from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models._ops.neighbor_filter import prepare_neighbors_for_model
from nvalchemi.models._utils import (
    autograd_forces,
    autograd_forces_and_stresses,
    autograd_stresses,
    prepare_strain,
    sum_outputs,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

__all__ = [
    "NeighborAdaptation",
    "PipelineModelWrapper",
    "PipelineStep",
    "PipelineGroup",
]

# Sentinel for "attribute was not present on the object".
_MISSING = object()

# All neighbor-related attributes that may need saving/restoring when the
# pipeline temporarily adapts neighbor data for a step.
_NEIGHBOR_ATTRS = (
    "neighbor_matrix",
    "num_neighbors",
    "neighbor_matrix_shifts",
    "neighbor_list",
    "edge_ptr",
    "neighbor_list_shifts",
    "_neighbor_list_cutoff",
)
_PIPELINE_NEIGHBOR_SOURCES_ATTR = "_pipeline_neighbor_sources"

# Type alias for the user-provided derivative function.
DerivativeFn = Callable[
    [Energy, Batch, set[str]],  # (energy, data, requested_keys)
    dict[str, torch.Tensor],  # computed derivatives
]

NeighborAdaptation: TypeAlias = Literal["auto", "always", "never"]


@dataclass
class _AutogradStrainState:
    """Temporary affine-strain state for default stress derivatives."""

    displacement: StrainDisplacement | None = None
    cell_for_stress: LatticeVectors | None = None
    unstrained_positions: NodePositions | None = None
    unstrained_cell: LatticeVectors | None = None


@dataclass(frozen=True)
class _NeighborSource:
    """A pre-built source neighbor list used by one or more pipeline steps."""

    source_id: int
    config: NeighborConfig


@dataclass(frozen=True)
class _StepNeighborPlan:
    """Neighbor-list source and runtime adaptation needs for one step."""

    source_id: int
    target_config: NeighborConfig
    needs_cutoff_adaptation: bool
    needs_format_conversion: bool


@dataclass(frozen=True)
class _NeighborSourceData:
    """Tensor references captured from a built neighbor-list source."""

    source_id: int
    config: NeighborConfig
    neighbor_matrix: torch.Tensor | None = None
    num_neighbors: torch.Tensor | None = None
    neighbor_matrix_shifts: torch.Tensor | None = None
    neighbor_list: torch.Tensor | None = None
    edge_ptr: torch.Tensor | None = None
    neighbor_list_shifts: torch.Tensor | None = None


class _PipelineNeighborListHook:
    """Build all planned pipeline neighbor-list sources before compute."""

    def __init__(
        self,
        sources: list[_NeighborSource],
        hooks: list[NeighborListHook],
        stage: Any,
    ) -> None:
        if len(sources) != len(hooks):
            raise ValueError("sources and hooks must have the same length.")
        self.sources = sources
        self.hooks = hooks
        self.stage = stage
        self.frequency = 1

    def __call__(self, ctx: Any, stage: Any) -> None:
        source_data: list[_NeighborSourceData] = []
        for source, hook in zip(self.sources, self.hooks, strict=True):
            hook(ctx, stage)
            source_data.append(self._capture_source_data(ctx.batch, source, hook))

        ctx.batch.__dict__[_PIPELINE_NEIGHBOR_SOURCES_ATTR] = tuple(source_data)
        self._restore_default_source(ctx.batch, source_data[0])

    @staticmethod
    def _capture_source_data(
        batch: Batch,
        source: _NeighborSource,
        hook: NeighborListHook,
    ) -> _NeighborSourceData:
        # MATRIX tensors live on the hook's persistent buffers; COO tensors
        # are written onto the batch by the hook after each source build.
        if source.config.format == NeighborListFormat.MATRIX:
            return _NeighborSourceData(
                source_id=source.source_id,
                config=source.config,
                neighbor_matrix=getattr(hook, "_neighbor_matrix", None),
                num_neighbors=getattr(hook, "_num_neighbors", None),
                neighbor_matrix_shifts=getattr(
                    hook,
                    "_neighbor_matrix_shifts",
                    None,
                ),
            )

        return _NeighborSourceData(
            source_id=source.source_id,
            config=source.config,
            neighbor_list=getattr(batch, "neighbor_list", None),
            edge_ptr=getattr(batch, "edge_ptr", None),
            neighbor_list_shifts=getattr(batch, "neighbor_list_shifts", None),
        )

    @staticmethod
    def _restore_default_source(
        batch: Batch,
        source_data: _NeighborSourceData,
    ) -> None:
        for attr in _NEIGHBOR_ATTRS:
            batch.__dict__.pop(attr, None)
        if source_data.neighbor_matrix is not None:
            batch.__dict__["neighbor_matrix"] = source_data.neighbor_matrix
            batch.__dict__["num_neighbors"] = source_data.num_neighbors
            if source_data.neighbor_matrix_shifts is not None:
                batch.__dict__["neighbor_matrix_shifts"] = (
                    source_data.neighbor_matrix_shifts
                )
        if source_data.neighbor_list is not None:
            batch.__dict__["neighbor_list"] = source_data.neighbor_list
            batch.__dict__["edge_ptr"] = source_data.edge_ptr
            if source_data.neighbor_list_shifts is not None:
                batch.__dict__["neighbor_list_shifts"] = (
                    source_data.neighbor_list_shifts
                )
        batch.__dict__["_neighbor_list_cutoff"] = source_data.config.cutoff


@dataclass(eq=False)
class PipelineStep:
    """Wraps a model with an output rename mapping.

    Only needed when a model's output key doesn't match the downstream
    input key.  For models that don't need renaming, pass the bare model
    directly — the pipeline normalizes it internally.

    Parameters
    ----------
    model : BaseModelMixin
        The model to wrap.
    wire : dict[str, str]
        Output-to-attribute rename mapping.  Each entry
        ``{output_key: data_attribute}`` causes the pipeline to write the
        model's ``output_key`` value onto ``data.data_attribute`` before
        downstream models execute.  Downstream models that declare
        ``data_attribute`` in their ``required_inputs`` will then receive
        it automatically.

    Examples
    --------
    AIMNet2 produces ``"charges"`` (per-atom partial charges), but the
    Ewald model expects ``"node_charges"`` as a required input::

        PipelineStep(aimnet2, wire={"charges": "node_charges"})

    After AIMNet2 runs, the pipeline writes its ``"charges"`` output
    onto ``data.node_charges``.  When Ewald runs next, its
    ``adapt_input()`` finds ``data.node_charges`` and uses it.

    If a model's output keys already match downstream input keys, no
    wire mapping is needed — pass the bare model::

        PipelineGroup(steps=[model_a, model_b])  # auto-wired
    """

    model: BaseModelMixin
    wire: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineGroup:
    """A group of steps that share a derivative computation strategy.

    Steps within a group execute in order (for wiring).
    Groups execute in declaration order.

    ``steps`` accepts bare :class:`BaseModelMixin` instances or
    :class:`PipelineStep` wrappers.  Bare models are normalized to
    ``PipelineStep(model, wire={})`` internally.

    Parameters
    ----------
    steps : list[BaseModelMixin | PipelineStep]
        Ordered list of models (or wrapped models) in this group.
    use_autograd : bool
        If ``True``, sub-models produce energies only; the group sums
        them and calls ``derivative_fn`` to compute forces, stresses,
        and any other requested derivatives from the summed energy.
        If ``False`` (default), each sub-model computes its own outputs
        and the group sums them directly.
    derivative_fn : DerivativeFn | None
        Custom derivative function called after energy summation in
        autograd groups.  Receives ``(energy, data, requested)`` where
        ``energy`` is the summed group energy (on the autograd graph),
        ``data`` is the batch (with ``positions.requires_grad=True``),
        and ``requested`` is the set of output keys that still need to
        be computed (e.g. ``{"forces", "stress"}``).

        When ``None`` (default), the pipeline uses a built-in function
        that computes forces as ``-dE/dr`` and stresses via the affine
        strain trick (see :func:`~nvalchemi.models._utils.prepare_strain`).

        Only meaningful when ``use_autograd=True``.
    """

    steps: list[BaseModelMixin | PipelineStep]
    use_autograd: bool = False
    derivative_fn: DerivativeFn | None = None


class PipelineModelWrapper(nn.Module, BaseModelMixin):
    """Compose multiple models via a grouped pipeline.

    Models are organized into :class:`PipelineGroup` instances, where each
    group has a derivative computation strategy.  Within a group, steps
    execute in order so that upstream outputs can wire into downstream
    inputs.  The pipeline sums outputs across groups using
    :func:`~nvalchemi.models._utils.sum_outputs`.

    The pipeline's default ``model_config.active_outputs`` is synthesized as the
    **union of all sub-model** ``model_config.active_outputs`` **sets** at
    construction time, so it honestly reflects what the sub-models are
    configured to produce.  The user can then expand or narrow it.

    Parameters
    ----------
    groups : list[PipelineGroup]
        Ordered list of groups.  Groups execute in declaration order.
    additive_keys : set[str], optional
        Keys whose values are summed across groups.  Defaults to
        ``{"energy", "forces", "stress"}``.
    neighbor_adaptation : NeighborAdaptation, optional
        Policy for sharing or splitting neighbor-list source cutoffs across
        sub-models: ``"auto"`` (default), ``"always"``, or ``"never"``.
        ``"auto"`` adapts a source list for a smaller cutoff only when the
        source cutoff is at most ``max_cutoff_ratio`` times the target;
        ``"always"`` builds one max-cutoff source and filters for every
        tighter model; ``"never"`` builds exact cutoff source groups and
        skips cutoff filtering.
    max_cutoff_ratio : float, optional
        Maximum allowed ratio between a source cutoff and a target cutoff when
        ``neighbor_adaptation="auto"``.  Defaults to ``1.5``.

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling what the pipeline computes.
    """

    def __init__(
        self,
        groups: list[PipelineGroup],
        additive_keys: set[str] | None = None,
        *,
        neighbor_adaptation: NeighborAdaptation = "auto",
        max_cutoff_ratio: float = 1.5,
    ) -> None:
        super().__init__()
        if neighbor_adaptation not in ("auto", "always", "never"):
            raise ValueError(
                "PipelineModelWrapper: neighbor_adaptation must be one of "
                "'auto', 'always', or 'never'."
            )
        if max_cutoff_ratio < 1.0:
            raise ValueError("PipelineModelWrapper: max_cutoff_ratio must be >= 1.0.")
        self._neighbor_adaptation = neighbor_adaptation
        self._max_cutoff_ratio = float(max_cutoff_ratio)
        self._neighbor_sources: list[_NeighborSource] = []
        self._step_neighbor_plans: dict[int, _StepNeighborPlan] = {}
        # Normalize bare models to PipelineStep(model, wire={})
        self.groups: list[PipelineGroup] = []
        for group in groups:
            normalized: list[PipelineStep] = []
            for step in group.steps:
                if isinstance(step, PipelineStep):
                    normalized.append(step)
                else:
                    normalized.append(PipelineStep(model=step))
            self.groups.append(
                PipelineGroup(
                    steps=normalized,
                    use_autograd=group.use_autograd,
                    derivative_fn=group.derivative_fn,
                )
            )
        self._models = nn.ModuleList(
            s.model
            for g in self.groups
            for s in g.steps  # type: ignore[misc]
        )
        self.additive_keys = additive_keys or {"energy", "forces", "stress"}

        # Check wiring and collect inputs that must come from the batch.
        batch_required = self._check_wiring()
        # Synthesize a unified ModelConfig from all sub-models.
        self.model_config = self._build_model_config(batch_required)
        self._configure_sub_models()

    # ------------------------------------------------------------------
    # ModelConfig synthesis
    # ------------------------------------------------------------------

    def _source_config_from_steps(
        self,
        source_id: int,
        step_configs: list[NeighborConfig],
        source_cutoff: float,
    ) -> _NeighborSource:
        """Create a source config for a group of target neighbor configs."""
        has_matrix = any(nc.format == NeighborListFormat.MATRIX for nc in step_configs)
        source_format = (
            NeighborListFormat.MATRIX if has_matrix else NeighborListFormat.COO
        )
        skin = max(nc.skin for nc in step_configs)
        return _NeighborSource(
            source_id=source_id,
            config=NeighborConfig(
                cutoff=source_cutoff,
                format=source_format,
                half_list=step_configs[0].half_list,
                skin=skin,
            ),
        )

    def _record_neighbor_group(
        self,
        source: _NeighborSource,
        step_items: list[tuple[PipelineStep, NeighborConfig]],
    ) -> None:
        """Record one source and per-step plans for its assigned steps."""
        self._neighbor_sources.append(source)
        for step, target_config in step_items:
            source_config = source.config
            needs_cutoff = (source_config.cutoff - target_config.cutoff) > 1e-6
            if self._neighbor_adaptation == "never":
                needs_cutoff = False
            self._step_neighbor_plans[id(step)] = _StepNeighborPlan(
                source_id=source.source_id,
                target_config=target_config,
                needs_cutoff_adaptation=needs_cutoff,
                needs_format_conversion=(source_config.format != target_config.format),
            )

    def _build_neighbor_plan(
        self,
        step_items: list[tuple[PipelineStep, NeighborConfig]],
    ) -> NeighborConfig | None:
        """Build private source and per-step neighbor plans.

        Returns the largest/default source config to expose through
        ``self.model_config.neighbor_config``.
        """
        self._neighbor_sources = []
        self._step_neighbor_plans = {}

        if not step_items:
            return None

        first_half_list = step_items[0][1].half_list
        for _, nc in step_items:
            if nc.half_list != first_half_list:
                raise ValueError(
                    "PipelineModelWrapper: sub-models have different half_list "
                    f"values ({nc.half_list} vs {first_half_list}). "
                    "All sub-models must use the same half_list value."
                )

        if self._neighbor_adaptation == "always":
            source_cutoff = max(nc.cutoff for _, nc in step_items)
            step_configs = [nc for _, nc in step_items]
            source = self._source_config_from_steps(0, step_configs, source_cutoff)
            self._record_neighbor_group(source, step_items)
            return source.config

        if self._neighbor_adaptation == "never":
            groups: dict[float, list[tuple[PipelineStep, NeighborConfig]]] = {}
            for item in step_items:
                groups.setdefault(item[1].cutoff, []).append(item)
            for source_id, source_cutoff in enumerate(sorted(groups, reverse=True)):
                group_items = groups[source_cutoff]
                source = self._source_config_from_steps(
                    source_id,
                    [nc for _, nc in group_items],
                    source_cutoff,
                )
                self._record_neighbor_group(source, group_items)
            return self._neighbor_sources[0].config

        remaining = sorted(step_items, key=lambda item: item[1].cutoff, reverse=True)
        source_id = 0
        while remaining:
            source_cutoff = remaining[0][1].cutoff
            assigned: list[tuple[PipelineStep, NeighborConfig]] = []
            unassigned: list[tuple[PipelineStep, NeighborConfig]] = []
            for item in remaining:
                target_cutoff = item[1].cutoff
                # Tolerance avoids splitting groups over insignificant float noise.
                can_adapt = (
                    source_cutoff <= target_cutoff * self._max_cutoff_ratio + 1e-6
                )
                if can_adapt:
                    assigned.append(item)
                else:
                    unassigned.append(item)

            source = self._source_config_from_steps(
                source_id,
                [nc for _, nc in assigned],
                source_cutoff,
            )
            self._record_neighbor_group(source, assigned)
            remaining = unassigned
            source_id += 1

        return self._neighbor_sources[0].config

    def _build_model_config(
        self, batch_required: set[str] | None = None
    ) -> ModelConfig:
        """Synthesize a unified :class:`ModelConfig` from all sub-model configs.

        Merges capability and runtime fields across every sub-model in every
        group to produce a single config that honestly represents the full
        pipeline.

        Parameters
        ----------
        batch_required : set[str] | None
            Required inputs that must come from the batch (not produced
            by any step in the pipeline).  These are added to the
            pipeline's ``required_inputs``.

        Synthesis rules:

        - **outputs**: union of all sub-model ``outputs``.  For autograd
          groups, ``"forces"`` and ``"stress"`` are added because the
          group can derive them from the summed energy.
        - **autograd_outputs**: union of per-model ``autograd_outputs`` for
          direct groups; ``{"forces", "stress"}`` for autograd groups.
        - **required_inputs**: union of all sub-model ``required_inputs``.
        - **active_outputs**: union of all sub-model ``active_outputs``.
        - **supports_pbc**: ``True`` only if *every* sub-model supports PBC.
        - **needs_pbc**: ``True`` if *any* sub-model needs PBC.
        - **neighbor_config**: synthesized from the pipeline's neighbor-list
          plan.  Exposes the largest/default source config for compatibility.
          All sub-models must agree on ``half_list``.
        """
        all_outputs: set[str] = set()
        all_inputs: set[str] = set()
        all_autograd_outputs: set[str] = set()
        default_active: set[str] = set()
        needs_pbc = False
        supports_pbc = True

        neighbor_step_items: list[tuple[PipelineStep, NeighborConfig]] = []

        for group in self.groups:
            for step in group.steps:
                cfg = step.model.model_config
                all_outputs |= cfg.outputs
                all_inputs |= cfg.required_inputs
                default_active |= cfg.active_outputs
                if group.use_autograd:
                    # Group-level autograd can produce forces/stresses
                    # from the summed energy — add them to outputs.
                    all_outputs |= {"forces", "stress"}
                    all_autograd_outputs |= {"forces", "stress"}
                else:
                    all_autograd_outputs |= cfg.autograd_outputs
                if cfg.needs_pbc:
                    needs_pbc = True
                if not cfg.supports_pbc:
                    supports_pbc = False
                if cfg.neighbor_config is not None:
                    neighbor_step_items.append((step, cfg.neighbor_config))

        neighbor_config = self._build_neighbor_plan(neighbor_step_items)

        return ModelConfig(
            outputs=frozenset(all_outputs),
            autograd_outputs=frozenset(all_autograd_outputs),
            required_inputs=frozenset(all_inputs | (batch_required or set())),
            supports_pbc=supports_pbc,
            needs_pbc=needs_pbc,
            neighbor_config=neighbor_config,
            active_outputs=default_active,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def extra_repr(self) -> str:
        """Show pipeline structure: groups, steps, wire mappings, and autograd strategy."""
        lines = []
        for i, group in enumerate(self.groups):
            tag = "autograd" if group.use_autograd else "direct"
            if group.derivative_fn is not None:
                tag += ", custom_fn"
            lines.append(f"group[{i}] ({tag}):")
            for j, step in enumerate(group.steps):
                name = type(step.model).__name__
                wire_str = f", wire={step.wire}" if step.wire else ""
                lines.append(f"  step[{j}]: {name}{wire_str}")
        active = sorted(self.model_config.active_outputs)
        lines.append(f"active_outputs={{{', '.join(active)}}}")
        return "\n".join(lines)

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute embeddings is not meaningful for pipeline models.
        Call compute_embeddings on individual sub-models instead."""
        raise NotImplementedError(
            "PipelineModelWrapper does not produce unified embeddings.  "
            "Call compute_embeddings on individual sub-models instead."
        )

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for pipeline models.
        Export individual sub-models instead."""
        raise NotImplementedError(
            "PipelineModelWrapper does not support direct export.  "
            "Export individual sub-models instead."
        )

    # ------------------------------------------------------------------
    # Validation and configuration
    # ------------------------------------------------------------------

    def _check_wiring(self) -> set[str]:
        """Verify that the pipeline's data flow graph is satisfiable.

        Walks through all groups and steps in declaration order,
        accumulating the set of output keys (after wire renaming) that
        each step produces.  Inputs that are not produced by any prior
        step become **required inputs of the pipeline** — they must be
        present on the input batch at runtime.

        Returns
        -------
        set[str]
            Required inputs that must come from the batch (not produced
            by any step in the pipeline).
        """
        # Fields always present on a Batch — no need to wire these.
        batch_fields = {
            "positions",
            "atomic_numbers",
            "atomic_masses",
            "cell",
            "pbc",
            "energy",
            "forces",
        }
        available: set[str] = set(batch_fields)
        batch_required: set[str] = set()

        for group in self.groups:
            for step in group.steps:
                cfg = step.model.model_config
                # Inputs not produced by prior steps must come from
                # the batch — propagate them as pipeline required_inputs.
                missing = set(cfg.required_inputs) - available
                batch_required |= missing

                # Build the effective output names (after wire renaming)
                renamed_outputs: set[str] = set()
                for out_key in cfg.outputs:
                    if out_key in step.wire:
                        renamed_outputs.add(step.wire[out_key])
                    else:
                        renamed_outputs.add(out_key)
                available |= renamed_outputs

        return batch_required

    def _configure_sub_models(self) -> None:
        """Compute per-step active_output overrides.

        For autograd groups the pipeline handles forces/stress via autograd,
        so sub-models should only produce energy.  Rather than permanently
        mutating the sub-model's ``model_config`` (which would break reuse
        of the same model instance in other pipelines or standalone), we
        store the overrides in ``_step_active_overrides`` and apply them
        temporarily during the forward pass.
        """
        self._step_active_overrides: dict[int, set[str]] = {}

        for group in self.groups:
            if group.use_autograd:
                for step in group.steps:
                    new_active = set(step.model.model_config.active_outputs)
                    # Strip derivatives that the pipeline computes via
                    # autograd, but keep keys the model produces
                    # analytically (e.g. Ewald/PME with hybrid_forces=True
                    # returns detached kernel forces and virial).
                    direct = step.model.direct_derivative_keys()
                    new_active -= {"forces", "stress"} - direct
                    self._step_active_overrides[id(step)] = new_active

    def _call_step(
        self,
        step: PipelineStep,
        data: AtomicData | Batch,
        **kwargs: Any,
    ) -> ModelOutputs:
        """Call a step's model, temporarily applying overrides.

        Two kinds of temporary overrides are applied and restored:

        1. **active_outputs** — for autograd groups, sub-models skip
           forces/stress (computed by the group after energy summation).
        2. **neighbor data** — the step's neighbor plan selects one captured
           source list, then filters or converts it only when the plan
           requires a model-specific cutoff or format.
        """
        override = self._step_active_overrides.get(id(step))
        neighbor_plan = self._step_neighbor_plans.get(id(step))

        saved_neighbors: dict[str, Any] | None = None
        saved_active: set[str] | None = None

        if neighbor_plan is not None:
            saved_neighbors = self._select_step_neighbors(data, neighbor_plan)

        if override is not None:
            cfg = step.model.model_config
            saved_active = cfg.active_outputs
            cfg.active_outputs = override

        try:
            return step.model(data, **kwargs)
        finally:
            if saved_active is not None:
                step.model.model_config.active_outputs = saved_active
            if saved_neighbors is not None:
                self._restore_step_neighbors(data, saved_neighbors)

    # ------------------------------------------------------------------
    # Neighbor adaptation
    # ------------------------------------------------------------------

    @staticmethod
    def _save_neighbor_attrs(data: Batch) -> dict[str, Any]:
        """Save current shadowed neighbor attrs for later restoration."""
        saved: dict[str, Any] = {}
        for attr in _NEIGHBOR_ATTRS:
            if attr in data.__dict__:
                saved[attr] = data.__dict__[attr]
            else:
                saved[attr] = _MISSING
        return saved

    def _get_source_data(
        self,
        data: Batch,
        plan: _StepNeighborPlan,
    ) -> _NeighborSourceData | None:
        """Return captured source data for a step, or None for canonical data."""
        sources = getattr(data, _PIPELINE_NEIGHBOR_SOURCES_ATTR, None)
        if sources is None:
            return None
        for source in sources:
            if source.source_id == plan.source_id:
                return source
        raise RuntimeError(
            "PipelineModelWrapper: missing neighbor source "
            f"{plan.source_id} on batch. Ensure make_neighbor_hooks() hooks "
            "are registered."
        )

    @staticmethod
    def _write_source_to_data(
        data: Batch,
        source_data: _NeighborSourceData,
    ) -> None:
        """Shadow one captured source onto canonical batch neighbor attrs."""
        for attr in _NEIGHBOR_ATTRS:
            data.__dict__.pop(attr, None)

        if source_data.neighbor_matrix is not None:
            data.__dict__["neighbor_matrix"] = source_data.neighbor_matrix
            data.__dict__["num_neighbors"] = source_data.num_neighbors
            if source_data.neighbor_matrix_shifts is not None:
                data.__dict__["neighbor_matrix_shifts"] = (
                    source_data.neighbor_matrix_shifts
                )

        if source_data.neighbor_list is not None:
            data.__dict__["neighbor_list"] = source_data.neighbor_list
            data.__dict__["edge_ptr"] = source_data.edge_ptr
            if source_data.neighbor_list_shifts is not None:
                data.__dict__["neighbor_list_shifts"] = source_data.neighbor_list_shifts

        data.__dict__["_neighbor_list_cutoff"] = source_data.config.cutoff

    def _select_step_neighbors(
        self,
        data: Batch,
        plan: _StepNeighborPlan,
    ) -> dict[str, Any]:
        """Select a source neighbor list and optionally adapt it for one step."""
        saved = self._save_neighbor_attrs(data)

        source_data = self._get_source_data(data, plan)
        if source_data is not None:
            self._write_source_to_data(data, source_data)

        if plan.needs_cutoff_adaptation or plan.needs_format_conversion:
            adapted = prepare_neighbors_for_model(
                data,
                plan.target_config.cutoff,
                plan.target_config.format,
                data.num_nodes,
            )

            if (
                plan.target_config.format == NeighborListFormat.MATRIX
                and "neighbor_matrix" in adapted
            ):
                num_neighbors = adapted["num_neighbors"]
                max_k = num_neighbors.max() if num_neighbors.numel() > 0 else 0
                adapted["neighbor_matrix"] = adapted["neighbor_matrix"][
                    :, :max_k
                ].contiguous()
                shifts = adapted.get("neighbor_matrix_shifts")
                if shifts is not None:
                    adapted["neighbor_matrix_shifts"] = shifts[:, :max_k].contiguous()

            for key, value in adapted.items():
                data.__dict__[key] = value

        data.__dict__["_neighbor_list_cutoff"] = plan.target_config.cutoff
        return saved

    @staticmethod
    def _restore_step_neighbors(
        data: Batch,
        saved: dict[str, Any],
    ) -> None:
        """Restore neighbor data on *data* from *saved* state."""
        for attr, value in saved.items():
            if value is _MISSING:
                # Attribute wasn't in __dict__ before — remove the shadow
                # so the original group-stored value becomes visible again.
                data.__dict__.pop(attr, None)
            else:
                data.__dict__[attr] = value

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _resolve_inputs(
        self,
        step: PipelineStep,
        context: dict[PipelineStep, ModelOutputs],
        data: Batch | AtomicData,
    ) -> None:
        """Write resolved upstream outputs onto *data* for this step's model.

        For each input the model needs, check if an upstream model produced
        it (via *context*).  Applies wire renaming.  Only writes to *data*
        what this step actually needs — *data* is not polluted with all
        intermediate tensors.
        """
        needed = step.model.model_config.required_inputs
        for ctx_step, ctx_out in context.items():
            card = ctx_step.model.model_config
            for out_key in card.outputs:
                value = ctx_out.get(out_key)
                if value is None:
                    continue
                data_attr = ctx_step.wire.get(out_key, out_key)
                if data_attr in needed:
                    # Use object.__setattr__ for wired intermediate
                    # values (e.g. charges [N]) that may not match the
                    # Batch system-group length validation.
                    object.__setattr__(data, data_attr, value)

    # ------------------------------------------------------------------
    # Neighbor hook factory
    # ------------------------------------------------------------------

    def make_neighbor_hooks(
        self, max_neighbors: int | None = None
    ) -> list[NeighborListHook | _PipelineNeighborListHook]:
        """Return neighbor hooks required by the pipeline's neighbor-list plan.

        Parameters
        ----------
        max_neighbors : int | None, optional
            Maximum neighbors per atom for MATRIX format.  When ``None``
            (default), auto-estimated from the cutoff at first use.
        """
        from nvalchemi.dynamics.base import DynamicsStage  # noqa: PLC0415

        if not self._neighbor_sources:
            return []

        hooks = [
            NeighborListHook(
                source.config,
                skin=source.config.skin,
                max_neighbors=max_neighbors,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
            for source in self._neighbor_sources
        ]

        if len(hooks) == 1:
            return hooks

        return [
            _PipelineNeighborListHook(
                sources=self._neighbor_sources,
                hooks=hooks,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run all sub-models and accumulate outputs.

        For groups with ``use_autograd=True``, sub-models produce energies
        only.  The group sums them and calls the derivative function
        (default or user-provided) to compute forces, stresses, and any
        other requested derivatives from the summed energy.

        What gets computed is driven by ``self.model_config.active_outputs``.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch.

        Returns
        -------
        ModelOutputs
            Combined outputs across all groups.
        """
        # Determine what derivatives are requested beyond energies.
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        training_with_grad = self.training and torch.is_grad_enabled()
        requested_derivatives = self.model_config.active_outputs - {"energy"}

        # Collect all autograd_inputs that need requires_grad
        grad_keys: set[str] = set()
        for group in self.groups:
            if group.use_autograd:
                for step in group.steps:
                    grad_keys |= step.model.model_config.autograd_inputs
            else:
                for step in group.steps:
                    card = step.model.model_config
                    if card.autograd_outputs & step.model.model_config.active_outputs:
                        grad_keys |= card.autograd_inputs

        # Forward context: tracks each step's outputs without
        # polluting data with all intermediate tensors.
        context: dict[PipelineStep, ModelOutputs] = {}

        autograd_groups = [g for g in self.groups if g.use_autograd]
        group_outputs: list[ModelOutputs] = []
        autograd_count = len(autograd_groups)
        effective_autograd_count = autograd_count + int(training_with_grad)
        autograd_idx = 0

        for group in self.groups:
            if group.use_autograd:
                group_out = self._run_autograd_group(
                    group,
                    data,
                    context,
                    requested_derivatives,
                    autograd_idx,
                    effective_autograd_count,
                    grad_keys,
                    training=training_with_grad,
                    **kwargs,
                )
                autograd_idx += 1
            else:
                group_out = self._run_direct_group(
                    group,
                    data,
                    context,
                    **kwargs,
                )

            group_outputs.append(group_out)

        result = sum_outputs(*group_outputs, additive_keys=self.additive_keys)
        if training_with_grad:
            return result

        # Detach all tensors from the computation graph.
        detached: ModelOutputs = OrderedDict()
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
            else:
                detached[key] = value
        return detached

    def _run_direct_group(
        self,
        group: PipelineGroup,
        data: AtomicData | Batch,
        context: dict[PipelineStep, ModelOutputs],
        **kwargs: Any,
    ) -> ModelOutputs:
        """Run a direct group: each model computes its own outputs, summed."""
        step_outputs: list[ModelOutputs] = []
        for step in group.steps:
            self._resolve_inputs(step, context, data)
            out = self._call_step(step, data, **kwargs)
            step_outputs.append(out)
            context[step] = out
        return sum_outputs(*step_outputs, additive_keys=self.additive_keys)

    @staticmethod
    def _detach_data_tensors(data: AtomicData | Batch) -> None:
        """Detach tensor fields currently attached to a graph data object."""
        if isinstance(data, Batch):
            for group in data._storage.groups.values():
                for key, value in list(group.items()):
                    if isinstance(value, torch.Tensor):
                        group[key] = value.detach()
        else:
            # AtomicData.model_dump() defaults to Python mode, so tensor fields
            # remain tensors; only JSON serialization converts them to lists.
            for key, value in list(data.model_dump(exclude_none=True).items()):
                if isinstance(value, torch.Tensor):
                    data[key] = value.detach()

        # Detach runtime attributes stored directly on the object.
        for key, value in list(vars(data).items()):
            if key.startswith("_") or key in {"device", "keys"}:
                continue
            if isinstance(value, torch.Tensor):
                object.__setattr__(data, key, value.detach())

    def _run_autograd_group(
        self,
        group: PipelineGroup,
        data: AtomicData | Batch,
        context: dict[PipelineStep, ModelOutputs],
        requested_derivatives: set[str],
        autograd_idx: int,
        autograd_count: int,
        grad_keys: set[str],
        *,
        training: bool,
        **kwargs: Any,
    ) -> ModelOutputs:
        """Run an autograd group: sum energies, then compute derivatives.

        When ``derivative_fn`` is ``None``, the pipeline uses the default
        derivative computation (forces + stresses via affine strain).
        When ``derivative_fn`` is provided, the user's function receives
        the summed energy, the batch, and the set of requested keys.

        Before running group steps, tensors listed in ``grad_keys`` are
        detached into fresh autograd leaves without cloning.  Pipeline steps
        must therefore treat input tensors as read-only: boundary cleanup
        detaches graph state but does not roll back in-place value mutations.
        Stress strain is applied after fresh leaf creation so the strain graph
        remains connected.  The ``finally`` block restores unstrained fields
        and detaches tensors left on the graph data object.

        Direct additive derivative outputs from hybrid models are added to the
        autograd derivatives from the summed energy.  Non-additive outputs are
        carried through with last-write-wins behavior.
        """
        need_stresses = self._needs_default_stresses(group, data, requested_derivatives)
        self._prepare_autograd_leaves(data, grad_keys)
        strain = self._apply_default_stress_strain(data, need_stresses)

        try:
            step_outputs = self._run_autograd_steps(group, data, context, **kwargs)
            return self._build_autograd_group_output(
                group,
                data,
                step_outputs,
                requested_derivatives,
                strain,
                retain_graph=autograd_idx < (autograd_count - 1),
                training=training,
            )
        finally:
            self._restore_default_stress_strain(data, strain)
            self._detach_data_tensors(data)

    @staticmethod
    def _needs_default_stresses(
        group: PipelineGroup,
        data: AtomicData | Batch,
        requested_derivatives: set[str],
    ) -> bool:
        """Return whether the default derivative path needs affine strain."""
        return (
            group.derivative_fn is None
            and "stress" in requested_derivatives
            and isinstance(data, Batch)
            and hasattr(data, "cell")
            and data.cell is not None
        )

    @staticmethod
    def _prepare_autograd_leaves(data: AtomicData | Batch, grad_keys: set[str]) -> None:
        """Replace requested gradient tensors with fresh autograd leaves."""
        for key in grad_keys:
            tensor = getattr(data, key, None)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                fresh = tensor.detach().requires_grad_(True)
                data[key] = fresh

    @staticmethod
    def _apply_default_stress_strain(
        data: AtomicData | Batch,
        need_stresses: bool,
    ) -> _AutogradStrainState:
        """Apply affine strain for default stress derivatives when needed."""
        strain = _AutogradStrainState()
        if not need_stresses:
            return strain

        strain.unstrained_positions = data.positions
        strain.unstrained_cell = data.cell
        strain.cell_for_stress = data.cell
        scaled_pos, scaled_cell, strain.displacement = prepare_strain(
            data.positions,
            data.cell,
            data.batch_idx,
        )
        data["positions"] = scaled_pos
        data["cell"] = scaled_cell
        return strain

    @staticmethod
    def _restore_default_stress_strain(
        data: AtomicData | Batch,
        strain: _AutogradStrainState,
    ) -> None:
        """Restore unstrained tensors after default stress derivative setup."""
        if strain.unstrained_positions is not None:
            data["positions"] = strain.unstrained_positions
        if strain.unstrained_cell is not None:
            data["cell"] = strain.unstrained_cell

    def _run_autograd_steps(
        self,
        group: PipelineGroup,
        data: AtomicData | Batch,
        context: dict[PipelineStep, ModelOutputs],
        **kwargs: Any,
    ) -> list[ModelOutputs]:
        """Run all steps in an autograd group and record their outputs."""
        step_outputs: list[ModelOutputs] = []
        for step in group.steps:
            self._resolve_inputs(step, context, data)
            out = self._call_step(step, data, **kwargs)
            step_outputs.append(out)
            context[step] = out
        return step_outputs

    def _build_autograd_group_output(
        self,
        group: PipelineGroup,
        data: AtomicData | Batch,
        step_outputs: list[ModelOutputs],
        requested_derivatives: set[str],
        strain: _AutogradStrainState,
        *,
        retain_graph: bool,
        training: bool,
    ) -> ModelOutputs:
        """Build an autograd group output from step outputs and derivatives."""
        group_energy = None
        for output in step_outputs:
            energy = output.get("energy")
            if energy is not None:
                group_energy = energy if group_energy is None else group_energy + energy

        group_out: ModelOutputs = OrderedDict()
        if group_energy is not None:
            group_out["energy"] = group_energy

        if group_energy is not None and requested_derivatives:
            already_produced = set(group_out.keys())
            needed = requested_derivatives - already_produced

            if needed:
                if group.derivative_fn is not None:
                    derivs = group.derivative_fn(group_energy, data, needed)
                else:
                    derivs = self._default_derivatives(
                        group_energy,
                        data,
                        needed,
                        displacement=strain.displacement,
                        orig_cell=strain.cell_for_stress,
                        retain_graph=retain_graph,
                        training=training,
                    )
                group_out.update(derivs)

        for output in step_outputs:
            for key, value in output.items():
                if value is not None and key in self.additive_keys and key != "energy":
                    if key in group_out and group_out[key] is not None:
                        group_out[key] = group_out[key] + value
                    else:
                        group_out[key] = value

        for output in step_outputs:
            for key, value in output.items():
                if (
                    value is not None
                    and key not in self.additive_keys
                    and key not in group_out
                ):
                    group_out[key] = value

        return group_out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the full pipeline (topology + model weights) to a file.

        The saved file contains:

        - ``"config"`` — pipeline topology (groups, wire mappings,
          autograd flags, additive keys).
        - ``"state_dict"`` — model weights for all sub-models.
        - ``"active_outputs"`` — current ``model_config.active_outputs``.

        Custom ``derivative_fn`` callables are **not** serialized.  When
        loading a pipeline that used a custom function, pass it again
        via :meth:`load`.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        config = []
        for group in self.groups:
            steps_cfg = [
                {
                    "model_class": f"{type(step.model).__module__}.{type(step.model).__qualname__}",
                    "wire": step.wire,
                }
                for step in group.steps
            ]
            config.append(
                {
                    "steps": steps_cfg,
                    "use_autograd": group.use_autograd,
                    "has_derivative_fn": group.derivative_fn is not None,
                }
            )

        torch.save(
            {
                "config": config,
                "state_dict": self.state_dict(),
                "additive_keys": sorted(self.additive_keys),
                "active_outputs": sorted(self.model_config.active_outputs),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        models: list[BaseModelMixin],
        derivative_fns: dict[int, DerivativeFn] | None = None,
    ) -> "PipelineModelWrapper":
        """Load a pipeline from a file saved with :meth:`save`.

        Models must be provided in the same order they appear in the
        saved config (flattened across groups).  The topology (groups,
        wire mappings, autograd flags) is restored from the file.

        Parameters
        ----------
        path : str | Path
            Path to a file created by :meth:`save`.
        models : list[BaseModelMixin]
            Pre-constructed model instances, one per step in the
            original pipeline (flattened across groups, in order).
        derivative_fns : dict[int, DerivativeFn] | None, optional
            Mapping from group index to custom derivative function.
            Required for groups that were saved with
            ``has_derivative_fn=True``.

        Returns
        -------
        PipelineModelWrapper

        Raises
        ------
        ValueError
            If the number of models doesn't match the saved config, or
            if a group requires a derivative_fn that wasn't provided.
        """
        checkpoint = torch.load(path, weights_only=True)
        config = checkpoint["config"]
        derivative_fns = derivative_fns or {}

        # Count total steps in config.
        total_steps = sum(len(g["steps"]) for g in config)
        if len(models) != total_steps:
            raise ValueError(
                f"Expected {total_steps} models (from saved config), got {len(models)}."
            )

        # Rebuild groups from config + provided models.
        model_iter = iter(models)
        groups: list[PipelineGroup] = []
        for i, group_cfg in enumerate(config):
            steps: list[PipelineStep] = []
            for step_cfg in group_cfg["steps"]:
                model = next(model_iter)
                steps.append(PipelineStep(model=model, wire=step_cfg["wire"]))
            dfn = derivative_fns.get(i)
            if group_cfg["has_derivative_fn"] and dfn is None:
                raise ValueError(
                    f"Group {i} requires a derivative_fn but none was "
                    f"provided in derivative_fns[{i}]."
                )
            groups.append(
                PipelineGroup(
                    steps=steps,
                    use_autograd=group_cfg["use_autograd"],
                    derivative_fn=dfn,
                )
            )

        additive_keys = set(checkpoint.get("additive_keys", []))
        pipe = cls(groups=groups, additive_keys=additive_keys or None)
        pipe.load_state_dict(checkpoint["state_dict"])

        # Restore active_outputs.
        saved_active = checkpoint.get("active_outputs")
        if saved_active is not None:
            pipe.model_config.active_outputs = set(saved_active)

        return pipe

    @staticmethod
    def _default_derivatives(
        energy: Energy,
        data: Batch | AtomicData,
        requested: set[str],
        *,
        displacement: StrainDisplacement | None,
        orig_cell: LatticeVectors | None,
        retain_graph: bool,
        training: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Built-in derivative computation for autograd groups.

        Computes forces as ``-dE/dr`` and stresses via the affine strain
        trick (when ``displacement`` is provided).  If neither forces nor
        stresses are requested, returns an empty dict.
        """
        result: dict[str, torch.Tensor] = {}
        need_forces = "forces" in requested
        need_stresses = displacement is not None and "stress" in requested

        if need_forces and need_stresses:
            if orig_cell is None:
                raise RuntimeError(
                    "orig_cell is required when computing autograd stresses."
                )
            num_graphs = data.num_graphs if isinstance(data, Batch) else 1
            forces, stress = autograd_forces_and_stresses(
                energy,
                data.positions,
                displacement,
                orig_cell,
                num_graphs,
                training=training,
                retain_graph=retain_graph,
            )
            return {"forces": forces, "stress": stress}

        if need_forces:
            forces = autograd_forces(
                energy,
                data.positions,
                training=training,
                retain_graph=retain_graph,
            )
            return {"forces": forces}

        if need_stresses:
            if orig_cell is None:
                raise RuntimeError(
                    "orig_cell is required when computing autograd stresses."
                )
            num_graphs = data.num_graphs if isinstance(data, Batch) else 1
            stress = autograd_stresses(
                energy,
                displacement,
                orig_cell,
                num_graphs,
                training=training,
                retain_graph=retain_graph,
            )
            return {"stress": stress}

        return result
