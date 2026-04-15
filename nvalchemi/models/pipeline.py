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
from typing import Any

import torch
from torch import nn

from nvalchemi._typing import Energy, LatticeVectors, ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models._ops.neighbor_filter import prepare_neighbors_for_model
from nvalchemi.models._utils import (
    autograd_forces,
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

__all__ = ["PipelineModelWrapper", "PipelineStep", "PipelineGroup"]

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

# Type alias for the user-provided derivative function.
DerivativeFn = Callable[
    [Energy, Batch, set[str]],  # (energy, data, requested_keys)
    dict[str, torch.Tensor],  # computed derivatives
]


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

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling what the pipeline computes.
    """

    def __init__(
        self,
        groups: list[PipelineGroup],
        additive_keys: set[str] | None = None,
    ) -> None:
        super().__init__()
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
        - **neighbor_config**: synthesized at the **maximum cutoff** across
          all sub-models.  Uses ``MATRIX`` format if any sub-model requires
          it, ``COO`` otherwise.
          All sub-models must agree on ``half_list``.
        """
        all_outputs: set[str] = set()
        all_inputs: set[str] = set()
        all_autograd_outputs: set[str] = set()
        default_active: set[str] = set()
        needs_pbc = False
        supports_pbc = True

        sub_neighbor_configs: list[NeighborConfig] = []

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
                    sub_neighbor_configs.append(cfg.neighbor_config)

        # Synthesize neighbor_config at max cutoff
        neighbor_config: NeighborConfig | None = None
        if sub_neighbor_configs:
            for nc in sub_neighbor_configs:
                if nc.half_list != sub_neighbor_configs[0].half_list:
                    raise ValueError(
                        "PipelineModelWrapper: sub-models have different half_list "
                        f"values ({nc.half_list} vs {sub_neighbor_configs[0].half_list}). "
                        "All sub-models must use the same half_list value."
                    )
            max_cutoff = max(nc.cutoff for nc in sub_neighbor_configs)
            has_matrix = any(
                nc.format == NeighborListFormat.MATRIX for nc in sub_neighbor_configs
            )
            chosen_format = (
                NeighborListFormat.MATRIX if has_matrix else NeighborListFormat.COO
            )
            skin_vals = [nc.skin for nc in sub_neighbor_configs if nc.skin is not None]
            neighbor_config = NeighborConfig(
                cutoff=max_cutoff,
                format=chosen_format,
                half_list=sub_neighbor_configs[0].half_list,
                skin=max(skin_vals) if skin_vals else 0.0,
            )

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
        """Compute per-step active_output and neighbor overrides.

        For autograd groups the pipeline handles forces/stress via autograd,
        so sub-models should only produce energy.  Rather than permanently
        mutating the sub-model's ``model_config`` (which would break reuse
        of the same model instance in other pipelines or standalone), we
        store the overrides in ``_step_active_overrides`` and apply them
        temporarily during the forward pass.

        For neighbor adaptation, the pipeline's unified neighbor config may
        have a larger cutoff or different format than an individual step's
        model.  Steps that need adaptation are flagged in
        ``_step_needs_neighbor_adapt`` and handled in :meth:`_call_step`.
        """
        self._step_active_overrides: dict[int, set[str]] = {}
        self._step_needs_neighbor_adapt: dict[int, bool] = {}
        pipeline_nc = self.model_config.neighbor_config

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

            for step in group.steps:
                step_nc = step.model.model_config.neighbor_config
                if pipeline_nc is None or step_nc is None:
                    self._step_needs_neighbor_adapt[id(step)] = False
                else:
                    needs = (
                        step_nc.format != pipeline_nc.format
                        or (pipeline_nc.cutoff - step_nc.cutoff) > 1e-6
                    )
                    self._step_needs_neighbor_adapt[id(step)] = needs

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
        2. **neighbor data** — when the pipeline's unified neighbor config
           differs from the step's model (larger cutoff or different
           format), the batch's neighbor tensors are swapped to the
           model-specific version for the duration of the call.
        """
        override = self._step_active_overrides.get(id(step))
        needs_neighbor_adapt = self._step_needs_neighbor_adapt.get(id(step), False)

        saved_neighbors: dict[str, Any] | None = None
        saved_active: set[str] | None = None

        if needs_neighbor_adapt:
            saved_neighbors = self._adapt_step_neighbors(step, data)

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

    def _adapt_step_neighbors(
        self,
        step: PipelineStep,
        data: Batch,
    ) -> dict[str, Any]:
        """Filter/convert neighbor data on *data* for this step's model.

        Uses :func:`prepare_neighbors_for_model` to produce neighbor
        tensors matching the step's ``neighbor_config`` (cutoff + format),
        then writes them onto *data* so the model's ``adapt_input`` sees
        the correct data without needing to call the conversion itself.

        Returns the saved attribute values for :meth:`_restore_step_neighbors`.
        """
        nc = step.model.model_config.neighbor_config
        # nc is guaranteed non-None by _step_needs_neighbor_adapt guard.
        if nc is None:
            raise ValueError(
                f"PipelineModelWrapper: step {step} has no neighbor config"
            )

        adapted = prepare_neighbors_for_model(
            data, nc.cutoff, nc.format, data.num_nodes
        )

        # Trim MATRIX K-dimension to actual max neighbors.
        if nc.format == NeighborListFormat.MATRIX and "neighbor_matrix" in adapted:
            nn = adapted["num_neighbors"]
            max_k = nn.max() if nn.numel() > 0 else 0
            adapted["neighbor_matrix"] = adapted["neighbor_matrix"][
                :, :max_k
            ].contiguous()
            shifts = adapted.get("neighbor_matrix_shifts")
            if shifts is not None:
                adapted["neighbor_matrix_shifts"] = shifts[:, :max_k].contiguous()

        # Save current values (check __dict__ to distinguish
        # instance-level attrs from group-stored Batch properties).
        saved: dict[str, Any] = {}
        for attr in _NEIGHBOR_ATTRS:
            if attr in data.__dict__:
                saved[attr] = data.__dict__[attr]
            else:
                saved[attr] = _MISSING

        # Write adapted tensors onto data.
        for key, value in adapted.items():
            data.__dict__[key] = value

        # Stamp cutoff so any residual prepare_neighbors_for_model calls
        # inside the model are no-ops.
        data.__dict__["_neighbor_list_cutoff"] = nc.cutoff

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
    ) -> list[NeighborListHook]:
        """Return a single :class:`NeighborListHook` for the composite neighbor config.

        Parameters
        ----------
        max_neighbors : int | None, optional
            Maximum neighbors per atom for MATRIX format.  When ``None``
            (default), auto-estimated from the cutoff at first use.
        """
        from nvalchemi.dynamics.base import DynamicsStage  # noqa: PLC0415

        nc = self.model_config.neighbor_config
        if nc is None:
            return []
        return [
            NeighborListHook(
                nc,
                skin=nc.skin,
                max_neighbors=max_neighbors,
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
        autograd_idx = 0

        for group in self.groups:
            if group.use_autograd:
                group_out = self._run_autograd_group(
                    group,
                    data,
                    context,
                    requested_derivatives,
                    autograd_idx,
                    autograd_count,
                    grad_keys,
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

    def _run_autograd_group(
        self,
        group: PipelineGroup,
        data: AtomicData | Batch,
        context: dict[PipelineStep, ModelOutputs],
        requested_derivatives: set[str],
        autograd_idx: int,
        autograd_count: int,
        grad_keys: set[str],
        **kwargs: Any,
    ) -> ModelOutputs:
        """Run an autograd group: sum energies, then compute derivatives.

        When ``derivative_fn`` is ``None``, the pipeline uses the default
        derivative computation (forces + stresses via affine strain).
        When ``derivative_fn`` is provided, the user's function receives
        the summed energy, the batch, and the set of requested keys.
        """
        use_default_derivs = group.derivative_fn is None
        need_stresses = (
            use_default_derivs
            and "stress" in requested_derivatives
            and isinstance(data, Batch)
            and hasattr(data, "cell")
            and data.cell is not None
        )

        # Enable requires_grad on positions for force computation.
        # We detach + clone first to ensure a fresh leaf tensor.  Without
        # this, positions from a previous step may still carry graph
        # references (e.g. from in-place updates by the integrator),
        # causing "backward through the graph a second time" errors.
        # NOTE: This must happen BEFORE strain preparation so that
        # prepare_strain can build a graph through the fresh leaves.
        for key in grad_keys:
            tensor = getattr(data, key, None)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                fresh = tensor.detach().clone().requires_grad_(True)
                data[key] = fresh

        # Set up strain AFTER detach+clone (if stresses needed in default
        # path).  This scales positions and cell through a displacement
        # tensor so dE/d(displacement) gives the stress.  The fresh leaf
        # tensors created above ensure the strain graph is not severed.
        displacement = None
        orig_positions = None
        orig_cell = None
        if need_stresses:
            orig_positions = data.positions
            orig_cell = data.cell
            scaled_pos, scaled_cell, displacement = prepare_strain(
                data.positions,
                data.cell,
                data.batch_idx,
            )
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        # Run all models in the group.
        step_outputs: list[ModelOutputs] = []
        for step in group.steps:
            self._resolve_inputs(step, context, data)
            out = self._call_step(step, data, **kwargs)
            step_outputs.append(out)
            context[step] = out

        # Sum energies across all steps in the group.
        group_energy = None
        for o in step_outputs:
            e = o.get("energy")
            if e is not None:
                group_energy = e if group_energy is None else group_energy + e

        needs_retain = autograd_idx < (autograd_count - 1)
        group_out: ModelOutputs = OrderedDict()
        if group_energy is not None:
            group_out["energy"] = group_energy

        # Compute derivatives from the summed energy.
        if group_energy is not None and requested_derivatives:
            already_produced = set(group_out.keys())
            needed = requested_derivatives - already_produced

            if needed:
                if group.derivative_fn is not None:
                    # User override — full control.
                    derivs = group.derivative_fn(group_energy, data, needed)
                else:
                    # Default: forces + stresses.
                    derivs = self._default_derivatives(
                        group_energy,
                        data,
                        needed,
                        displacement=displacement,
                        orig_cell=orig_cell,
                        retain_graph=needs_retain,
                    )
                group_out.update(derivs)

        # Sum direct additive outputs from step outputs (e.g. hybrid-force
        # models that return detached kernel forces and virial/stress)
        # alongside the autograd derivatives computed above.  For hybrid
        # electrostatic models the kernel returns dE/dR|_q (forces) and
        # dE/d(strain)|_q (stress) while autograd provides the charge
        # chain-rule terms (dE/dq)(dq/dR) and (dE/dq)(dq/d(strain)).
        for o in step_outputs:
            for key, val in o.items():
                if val is not None and key in self.additive_keys and key != "energy":
                    if key in group_out and group_out[key] is not None:
                        group_out[key] = group_out[key] + val
                    else:
                        group_out[key] = val

        # Carry through non-additive keys from step outputs.
        for o in step_outputs:
            for key, val in o.items():
                if (
                    val is not None
                    and key not in self.additive_keys
                    and key not in group_out
                ):
                    group_out[key] = val

        # Restore original positions/cell if strain was applied.
        if orig_positions is not None:
            data["positions"] = orig_positions
        if orig_cell is not None:
            data["cell"] = orig_cell

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
        displacement: torch.Tensor | None,
        orig_cell: LatticeVectors | None,
        retain_graph: bool,
    ) -> dict[str, torch.Tensor]:
        """Built-in derivative computation for autograd groups.

        Computes forces as ``-dE/dr`` and stresses via the affine strain
        trick (when ``displacement`` is provided).  If neither forces nor
        stresses are requested, returns an empty dict.
        """
        result: dict[str, torch.Tensor] = {}
        need_stresses = displacement is not None and "stress" in requested

        if "forces" in requested:
            result["forces"] = autograd_forces(
                energy,
                data.positions,
                retain_graph=retain_graph or need_stresses,
            )
        if need_stresses:
            num_graphs = data.num_graphs if isinstance(data, Batch) else 1
            result["stress"] = autograd_stresses(
                energy,
                displacement,
                orig_cell,
                num_graphs,
                retain_graph=retain_graph,
            )
        return result
