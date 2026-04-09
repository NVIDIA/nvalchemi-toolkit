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

:class:`ComposableModelWrapper` combines two or more
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible models whose
composable outputs (energy, forces, stress) are summed element-wise.
Non-composable outputs produced by sub-models are written back to the batch
on a last-write-wins basis.

Typical usage via the ``+`` operator (when models support it)::

    combined = model_a + model_b

Or directly::

    from nvalchemi.models.composable import ComposableModelWrapper

    combined = ComposableModelWrapper(lj_model, mlip_model)
    combined.model_config.compute_stresses = True

The composite model synthesises a :class:`~nvalchemi.models.base.ModelCard`
from all sub-model cards, picking the most permissive neighbor configuration
(maximum cutoff, full list, MATRIX if any sub-model uses MATRIX).
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from nvalchemi.dynamics.hooks import NeighborListHook

__all__ = ["ComposableModelWrapper"]

_COMPOSABLE_KEYS: frozenset[str] = frozenset({"energy", "forces", "stress"})


class ComposableModelWrapper(nn.Module, BaseModelMixin):
    """Compose multiple models by summing their composable outputs.

    Parameters
    ----------
    *models : BaseModelMixin
        Two or more model wrappers to compose.  Any nested
        :class:`ComposableModelWrapper` instances are flattened into the
        top-level list so that the composition is always a single flat layer.

    Attributes
    ----------
    models : nn.ModuleList
        Flat list of constituent model wrappers.
    """

    def __init__(self, *models: BaseModelMixin) -> None:
        super().__init__()

        # Flatten any nested ComposableModelWrappers.
        flat: list[BaseModelMixin] = []
        for m in models:
            if isinstance(m, ComposableModelWrapper):
                flat.extend(list(m.models))
            else:
                flat.append(m)

        # Guard: energy-first composition for multiple autograd models is not yet
        # implemented. Summing pre-computed forces is memory-inefficient for two
        # autograd models.
        n_autograd = sum(
            1
            for m in flat
            if getattr(m, "model_card", None) is not None
            and m.model_card.forces_via_autograd  # type: ignore[union-attr]
        )
        if n_autograd > 1:
            raise NotImplementedError(
                "Composing two or more autograd-forces models is not yet supported. "
                "Energy-first composition (sum energies, single autograd pass) is "
                "required for memory correctness but not yet implemented. "
            )

        self.models: nn.ModuleList = nn.ModuleList(flat)  # type: ignore[arg-type]
        # Use the property setter so all sub-models share the same ModelConfig
        # instance from construction; in-place mutations (e.g.
        # wrapper.model_config.compute_stresses = True) then propagate
        # automatically because every sub-model holds a reference to the same
        # object.
        self.model_config = ModelConfig()
        self._model_card: ModelCard = self._build_model_card()

    # ------------------------------------------------------------------
    # model_config property (shadows class-level attr from BaseModelMixin)
    # ------------------------------------------------------------------

    @property
    def model_config(self) -> ModelConfig:  # type: ignore[override]
        """Mutable configuration controlling which outputs are computed."""
        return self._model_config

    @model_config.setter
    def model_config(self, config: ModelConfig) -> None:
        self._model_config = config
        for model in self.models:
            model.model_config = config  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    def _build_model_card(self) -> ModelCard:
        cards = [m.model_card for m in self.models]  # type: ignore[union-attr]

        forces_via_autograd = any(c.forces_via_autograd for c in cards)
        supports_energies = all(c.supports_energies for c in cards)
        supports_forces = all(c.supports_forces for c in cards)
        supports_stresses = all(c.supports_stresses for c in cards)
        supports_pbc = all(c.supports_pbc for c in cards)
        needs_pbc = any(c.needs_pbc for c in cards)
        supports_non_batch = all(c.supports_non_batch for c in cards)

        # Synthesise neighbor_config from sub-models that have one.
        sub_configs = [
            c.neighbor_config for c in cards if c.neighbor_config is not None
        ]
        if sub_configs:
            # Validate that all sub-models have the same half_list value.
            for nc in sub_configs:
                if nc.half_list != sub_configs[0].half_list:
                    raise ValueError(
                        "ComposableModelWrapper: a sub-model has a different half_list value in its "
                        "NeighborConfig.  All sub-models must use the same half_list value when "
                        f"composed. Got {nc.half_list} and {sub_configs[0].half_list}."
                    )

                half_list = sub_configs[0].half_list
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
            neighbor_config: NeighborConfig | None = NeighborConfig(
                cutoff=max_cutoff,
                format=chosen_format,
                half_list=half_list,
                max_neighbors=max_neighbors,
            )
        else:
            neighbor_config = None

        # Warn if two sub-models both claim to include the same physics term,
        # which would cause double-counting in the composable composition.
        n_dispersion = sum(1 for c in cards if c.includes_dispersion)
        if n_dispersion > 1:
            warnings.warn(
                "ComposableModelWrapper: two or more sub-models have includes_dispersion=True. "
                "This may double-count dispersion interactions. Verify your model checkpoints.",
                UserWarning,
                stacklevel=3,
            )
        n_elec = sum(1 for c in cards if c.includes_long_range_electrostatics)
        if n_elec > 1:
            warnings.warn(
                "ComposableModelWrapper: two or more sub-models have "
                "includes_long_range_electrostatics=True. "
                "This may double-count long-range electrostatic interactions.",
                UserWarning,
                stacklevel=3,
            )

        return ModelCard(
            forces_via_autograd=forces_via_autograd,
            supports_energies=supports_energies,
            supports_forces=supports_forces,
            supports_stresses=supports_stresses,
            supports_pbc=supports_pbc,
            needs_pbc=needs_pbc,
            supports_non_batch=supports_non_batch,
            neighbor_config=neighbor_config,
        )

    @property
    def model_card(self) -> ModelCard:
        """Synthesised :class:`ModelCard` derived from all sub-model cards."""
        return self._model_card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        # Composite models do not have a unified embedding space.
        return {}

    # ------------------------------------------------------------------
    # Methods that are not meaningful for composite models
    # ------------------------------------------------------------------

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute embeddings is not meaningful for composite models.
        Call compute_embeddings on individual sub-models instead."""
        raise NotImplementedError(
            "ComposableModelWrapper does not produce unified embeddings.  "
            "Call compute_embeddings on individual sub-models instead."
        )

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for composite models.
        Export individual sub-models instead."""
        raise NotImplementedError(
            "ComposableModelWrapper does not support direct export.  "
            "Export individual sub-models instead."
        )

    # ------------------------------------------------------------------
    # Neighbor hook factory
    # ------------------------------------------------------------------

    def make_neighbor_hooks(self) -> list[NeighborListHook]:
        """Return a single :class:`NeighborListHook` for the composite neighbor config.

        A single composite hook at the maximum cutoff is used for all sub-models.
        This avoids running multiple neighbor-list algorithms per dynamics step.
        The cost of reformatting (e.g. neighbor matrix → neighbor list) at a
        synchronization point is preferable to computing separate neighbor lists.

        The import is deferred to avoid circular imports between
        ``nvalchemi.models`` and ``nvalchemi.dynamics``.
        """
        from nvalchemi.dynamics.hooks import NeighborListHook

        nc = self.model_card.neighbor_config
        if nc is None:
            return []
        return [NeighborListHook(nc)]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict[str, Tensor]:
        """Run all sub-models left-to-right and accumulate composable outputs.

        Composable outputs (``"energy"``, ``"forces"``, ``"stress"``) are
        summed across models.  All other outputs are written back to *data*
        on a last-write-wins basis.

        Parameters
        ----------
        data : Batch
            Input batch.  Neighbor data must already be populated (e.g. by
            a :class:`~nvalchemi.dynamics.hooks.NeighborListHook`).

        Returns
        -------
        OrderedDict[str, Tensor]
            Accumulated composable outputs in canonical order (energy →
            forces → stress), containing only the keys that are present in
            at least one sub-model's output.
        """
        accumulated: dict[str, Tensor] = {}

        for model in self.models:
            result = model(data, **kwargs)  # type: ignore[operator]
            if result is None:
                continue
            for key, val in result.items():
                if val is None:
                    continue
                if key in _COMPOSABLE_KEYS:
                    if key in accumulated:
                        accumulated[key] = accumulated[key] + val
                    else:
                        accumulated[key] = val
                else:
                    # Non-additive: write back to batch (last-write-wins).
                    # Use object.__setattr__ to bypass Batch's custom __setattr__
                    # which tries to route tensors into data groups and requires
                    # tensors to have a well-defined len().
                    object.__setattr__(data, key, val)

        # Return in canonical key order.
        out: OrderedDict[str, Tensor] = OrderedDict()
        for key in ("energy", "forces", "stress"):
            if key in accumulated:
                out[key] = accumulated[key]

        return out
