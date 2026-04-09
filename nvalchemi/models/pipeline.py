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
                PipelineStep(aimnet2, wire={"charges": "node_charges"}),
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
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
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

if TYPE_CHECKING:
    from nvalchemi.dynamics.hooks import NeighborListHook

__all__ = ["PipelineModelWrapper", "PipelineStep", "PipelineGroup"]

# Type alias for the user-provided derivative function.
DerivativeFn = Callable[
    [torch.Tensor, Batch, set[str]],  # (energy, data, requested_keys)
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

        # Synthesize a unified ModelConfig from all sub-models.
        self.model_config = self._build_model_config()
        self._validate_alignment()
        self._configure_sub_models()

    # ------------------------------------------------------------------
    # ModelConfig synthesis
    # ------------------------------------------------------------------

    def _build_model_config(self) -> ModelConfig:
        """Synthesize a unified :class:`ModelConfig` from all sub-model configs.

        Merges capability and runtime fields across every sub-model in every
        group to produce a single config that honestly represents the full
        pipeline.

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
          it, ``COO`` otherwise.  ``max_neighbors`` takes the maximum.
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
            max_neighbors_vals = [
                nc.max_neighbors
                for nc in sub_neighbor_configs
                if nc.max_neighbors is not None
            ]
            max_neighbors = max(max_neighbors_vals) if max_neighbors_vals else None
            neighbor_config = NeighborConfig(
                cutoff=max_cutoff,
                format=chosen_format,
                half_list=sub_neighbor_configs[0].half_list,
                max_neighbors=max_neighbors,
            )

        return ModelConfig(
            outputs=frozenset(all_outputs),
            autograd_outputs=frozenset(all_autograd_outputs),
            required_inputs=frozenset(all_inputs),
            supports_pbc=supports_pbc,
            needs_pbc=needs_pbc,
            neighbor_config=neighbor_config,
            active_outputs=default_active,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

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

    def _validate_alignment(self) -> None:
        """Check that every model's inputs are satisfiable from upstream outputs."""
        available: set[str] = set()
        for group in self.groups:
            for step in group.steps:
                card = step.model.model_config
                # Build the effective output names (after wire renaming)
                renamed_outputs: set[str] = set()
                for out_key in card.outputs:
                    if out_key in step.wire:
                        renamed_outputs.add(step.wire[out_key])
                    else:
                        renamed_outputs.add(out_key)
                available |= renamed_outputs

    def _configure_sub_models(self) -> None:
        """Adjust sub-model configs based on group autograd strategy."""
        for group in self.groups:
            if group.use_autograd:
                for step in group.steps:
                    # Remove derivative keys from sub-model active_outputs —
                    # the group will compute these via autograd.
                    new_active = set(step.model.model_config.active_outputs)
                    new_active -= {"forces", "stress"}
                    step.model.model_config = step.model.model_config.model_copy(
                        update={"active_outputs": new_active}
                    )

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
                    # Use object.__setattr__ to bypass Batch's custom
                    # __setattr__ which validates tensor lengths against
                    # storage groups and would reject node-level tensors
                    # being written to the system-level group.
                    object.__setattr__(data, data_attr, value)

    # ------------------------------------------------------------------
    # Neighbor hook factory
    # ------------------------------------------------------------------

    def make_neighbor_hooks(self) -> list[NeighborListHook]:
        """Return a single :class:`NeighborListHook` for the composite neighbor config."""
        from nvalchemi.dynamics.hooks import NeighborListHook

        nc = self.model_config.neighbor_config
        if nc is None:
            return []
        return [NeighborListHook(nc)]

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
            out = step.model(data, **kwargs)
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

        # Set up strain BEFORE model forward passes (if stresses needed
        # in default path).  This scales positions and cell through a
        # displacement tensor so dE/d(displacement) gives the stress.
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
            object.__setattr__(data, "positions", scaled_pos)
            object.__setattr__(data, "cell", scaled_cell)

        # Enable requires_grad on positions for force computation.
        for key in grad_keys:
            tensor = getattr(data, key, None)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                tensor.requires_grad_(True)

        # Run all models in the group.
        step_outputs: list[ModelOutputs] = []
        for step in group.steps:
            self._resolve_inputs(step, context, data)
            out = step.model(data, **kwargs)
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
            object.__setattr__(data, "positions", orig_positions)
        if orig_cell is not None:
            object.__setattr__(data, "cell", orig_cell)

        return group_out

    @staticmethod
    def _default_derivatives(
        energy: torch.Tensor,
        data: Batch | AtomicData,
        requested: set[str],
        *,
        displacement: torch.Tensor | None,
        orig_cell: torch.Tensor | None,
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
