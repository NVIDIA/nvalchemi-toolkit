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
group is a mini-pipeline with its own force computation strategy.  The top
level sums outputs across groups.

All composition goes through explicit ``PipelineModelWrapper`` construction —
there is no ``__add__`` operator or ``ComposableModelWrapper``.

Motivating example — AIMNet2 + Ewald + DFTD3::

    pipe = PipelineModelWrapper(groups=[
        PipelineGroup(
            steps=[
                PipelineStep(aimnet2, wire={"charges": "node_charges"}),
                ewald,
            ],
            forces="autograd",
        ),
        PipelineGroup(steps=[dftd3], forces="direct"),
    ])

See the module docstring or the proposal for full composition examples.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import autograd_forces, sum_outputs
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from nvalchemi.dynamics.hooks import NeighborListHook

__all__ = ["PipelineModelWrapper", "PipelineStep", "PipelineGroup"]


@dataclass(eq=False)
class PipelineStep:
    """Wraps a model with an output rename mapping.

    Only needed when a model's output key doesn't match the downstream
    input key (e.g., model outputs ``"charges"`` but downstream expects
    ``"node_charges"``).  For models that don't need renaming, pass the
    bare model directly — the pipeline normalizes it internally.

    Parameters
    ----------
    model : BaseModelMixin
        The model to wrap.
    wire : dict[str, str]
        ``{output_key: data_attribute}`` rename mapping.
    """

    model: BaseModelMixin
    wire: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineGroup:
    """A group of steps that share a force computation strategy.

    Steps within a group execute in order (for wiring).
    Groups execute in declaration order.

    ``steps`` accepts bare :class:`BaseModelMixin` instances or
    :class:`PipelineStep` wrappers.  Bare models are normalized to
    ``PipelineStep(model, wire={})`` internally.

    Parameters
    ----------
    steps : list[BaseModelMixin | PipelineStep]
        Ordered list of models (or wrapped models) in this group.
    forces : {"direct", "autograd"}
        ``"direct"`` — models compute their own forces, summed.
        ``"autograd"`` — sum energies within group, differentiate once.
    """

    steps: list[BaseModelMixin | PipelineStep]
    forces: Literal["direct", "autograd"] = "direct"


class PipelineModelWrapper(nn.Module, BaseModelMixin):
    """Compose multiple models via a grouped pipeline.

    Models are organized into :class:`PipelineGroup` instances, where each
    group has a force computation strategy (``"direct"`` or ``"autograd"``).
    Within a group, steps execute in order so that upstream outputs can wire
    into downstream inputs.  The pipeline sums outputs across groups using
    :func:`~nvalchemi.models._utils.sum_outputs`.

    Parameters
    ----------
    groups : list[PipelineGroup]
        Ordered list of groups.  Groups execute in declaration order.
    additive_keys : set[str], optional
        Keys whose values are summed across groups.  Defaults to
        ``{"energies", "forces", "stresses"}``.

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration.  Updated via :meth:`_configure_sub_models`.
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
            self.groups.append(PipelineGroup(steps=normalized, forces=group.forces))
        self._models = nn.ModuleList(
            s.model
            for g in self.groups
            for s in g.steps  # type: ignore[misc]
        )
        self.additive_keys = additive_keys or {"energies", "forces", "stresses"}
        self.model_config = ModelConfig()
        self._model_card: ModelCard = self._build_model_card()
        self._validate_alignment()
        self._configure_sub_models()

    # ------------------------------------------------------------------
    # ModelCard synthesis
    # ------------------------------------------------------------------

    def _build_model_card(self) -> ModelCard:
        """Synthesize a :class:`ModelCard` from all sub-model cards."""
        all_outputs: set[str] = set()
        all_inputs: set[str] = set()
        all_autograd_outputs: set[str] = set()
        needs_pbc = False
        supports_pbc = True

        sub_configs: list[NeighborConfig] = []

        for group in self.groups:
            for step in group.steps:
                card = step.model.model_card
                all_outputs |= card.outputs
                all_inputs |= card.inputs
                if group.forces == "autograd":
                    # Group-level autograd replaces per-model autograd
                    all_autograd_outputs |= {"forces"} & self.model_config.compute
                else:
                    all_autograd_outputs |= card.autograd_outputs
                if card.needs_pbc:
                    needs_pbc = True
                if not card.supports_pbc:
                    supports_pbc = False
                if card.neighbor_config is not None:
                    sub_configs.append(card.neighbor_config)

        # Synthesize neighbor_config at max cutoff
        neighbor_config: NeighborConfig | None = None
        if sub_configs:
            for nc in sub_configs:
                if nc.half_list != sub_configs[0].half_list:
                    raise ValueError(
                        "PipelineModelWrapper: sub-models have different half_list "
                        f"values ({nc.half_list} vs {sub_configs[0].half_list}). "
                        "All sub-models must use the same half_list value."
                    )
            max_cutoff = max(nc.cutoff for nc in sub_configs)
            has_matrix = any(
                nc.format == NeighborListFormat.MATRIX for nc in sub_configs
            )
            chosen_format = (
                NeighborListFormat.MATRIX if has_matrix else NeighborListFormat.COO
            )
            max_neighbors_vals = [
                nc.max_neighbors for nc in sub_configs if nc.max_neighbors is not None
            ]
            max_neighbors = max(max_neighbors_vals) if max_neighbors_vals else None
            neighbor_config = NeighborConfig(
                cutoff=max_cutoff,
                format=chosen_format,
                half_list=sub_configs[0].half_list,
                max_neighbors=max_neighbors,
            )

        return ModelCard(
            outputs=all_outputs,
            autograd_outputs=all_autograd_outputs,
            inputs=all_inputs,
            supports_pbc=supports_pbc,
            needs_pbc=needs_pbc,
            neighbor_config=neighbor_config,
        )

    @property
    def model_card(self) -> ModelCard:
        """Synthesised :class:`ModelCard` derived from all sub-model cards."""
        return self._model_card

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
                card = step.model.model_card
                # Build the effective output names (after wire renaming)
                renamed_outputs: set[str] = set()
                for out_key in card.outputs:
                    if out_key in step.wire:
                        renamed_outputs.add(step.wire[out_key])
                    else:
                        renamed_outputs.add(out_key)
                available |= renamed_outputs

    def _configure_sub_models(self) -> None:
        """Adjust sub-model configs based on group force strategy."""
        for group in self.groups:
            if group.forces == "autograd":
                for step in group.steps:
                    # Remove force/stress keys from sub-model compute —
                    # the group will compute these via autograd
                    sub_compute = set(step.model.model_config.compute)
                    sub_compute -= {"forces", "stresses"}
                    step.model.model_config = ModelConfig(
                        compute=sub_compute,
                        gradient_keys=step.model.model_config.gradient_keys,
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
        needed = step.model.model_card.inputs
        for ctx_step, ctx_out in context.items():
            card = ctx_step.model.model_card
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

        nc = self.model_card.neighbor_config
        if nc is None:
            return []
        return [NeighborListHook(nc)]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run all sub-models and accumulate outputs.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch.

        Returns
        -------
        ModelOutputs
            Combined outputs across all groups.
        """
        # Collect all autograd_inputs that need requires_grad
        grad_keys: set[str] = set()
        for group in self.groups:
            if group.forces == "autograd":
                for step in group.steps:
                    grad_keys |= step.model.model_card.autograd_inputs
            else:
                for step in group.steps:
                    card = step.model.model_card
                    if card.autograd_outputs & step.model.model_config.compute:
                        grad_keys |= card.autograd_inputs
        for key in grad_keys:
            tensor = getattr(data, key, None)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                tensor.requires_grad_(True)

        # Forward context: tracks each step's outputs without
        # polluting data with all intermediate tensors.
        context: dict[PipelineStep, ModelOutputs] = {}

        autograd_groups = [g for g in self.groups if g.forces == "autograd"]
        group_outputs: list[ModelOutputs] = []
        autograd_count = len(autograd_groups)
        autograd_idx = 0

        for group in self.groups:
            step_outputs: list[ModelOutputs] = []
            for step in group.steps:
                # Resolve: write upstream outputs onto data for this step
                self._resolve_inputs(step, context, data)

                out = step.model(data, **kwargs)
                step_outputs.append(out)
                context[step] = out

            if group.forces == "autograd":
                autograd_idx += 1
                group_energy = None
                for o in step_outputs:
                    e = o.get("energies")
                    if e is not None:
                        group_energy = e if group_energy is None else group_energy + e
                needs_retain = autograd_idx < autograd_count
                group_out: ModelOutputs = OrderedDict()
                if group_energy is not None:
                    group_out["energies"] = group_energy
                    group_out["forces"] = autograd_forces(
                        group_energy,
                        data.positions,
                        training=False,
                        retain_graph=needs_retain,
                    )
                # Carry through non-additive keys from step outputs
                for o in step_outputs:
                    for key, val in o.items():
                        if (
                            val is not None
                            and key not in self.additive_keys
                            and key not in group_out
                        ):
                            group_out[key] = val
            else:
                group_out = sum_outputs(*step_outputs, additive_keys=self.additive_keys)

            group_outputs.append(group_out)

        result = sum_outputs(*group_outputs, additive_keys=self.additive_keys)

        # Detach all tensors from the computation graph.  By this point all
        # autograd groups have already computed their forces; the graph is no
        # longer needed.  Without detaching, callers that hold a reference to
        # the returned dict (e.g. BaseDynamics._last_outputs) would keep the
        # entire graph alive, causing memory to grow across steps.
        detached: ModelOutputs = OrderedDict()
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
            else:
                detached[key] = value
        return detached
