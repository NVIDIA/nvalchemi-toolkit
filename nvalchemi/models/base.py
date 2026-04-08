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
"""Base interfaces and shared configuration helpers for model wrappers."""

from __future__ import annotations

import enum
from collections.abc import KeysView
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.models.neighbors import NeighborList

__all__ = [
    "BaseModelMixin",
    "ModelConfig",
    "NeighborConfig",
    "NeighborListFormat",
    "PipelineContext",
    "_UNSET",
    "_resolve_config",
]

_C = TypeVar("_C", bound=BaseModel)


class _UnsetType:
    """Sentinel used to distinguish omitted kwargs from explicit ``None``."""

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        """Return a readable sentinel representation."""

        return "<UNSET>"

    def __bool__(self) -> bool:
        """Treat the sentinel as falsy."""

        return False


_UNSET: Any = _UnsetType()


class NeighborListFormat(str, enum.Enum):
    """Storage format for neighbor data written to model inputs.

    Attributes
    ----------
    COO : str
        Coordinate-format neighbor list.  Internally this is exposed as
        ``edge_index`` / ``neighbor_ptr`` data.
    MATRIX : str
        Dense neighbor-matrix format.  Neighbors are stored as
        ``neighbor_matrix`` and ``num_neighbors`` tensors.
    """

    COO = "coo"
    MATRIX = "matrix"


class NeighborConfig(BaseModel):
    """Neighbor-list requirement advertised by a model wrapper.

    The composable runtime reads this config to decide whether a model
    builds neighbors internally, consumes an externally prepared list,
    or does not require neighbors at all.

    Attributes
    ----------
    source : {"none", "internal", "external"}
        Declares who is responsible for neighbor-list construction.
    cutoff : float | None
        Interaction cutoff radius in the same units as positions.
    format : {"coo", "matrix"} | None
        Required neighbor-list storage layout for externally supplied data.
    half_list : bool | None
        Whether a half-list is required when ``source="external"``.
    max_neighbors : int | None
        Optional fixed matrix width for matrix-format consumers.
    """

    source: Annotated[
        Literal["none", "internal", "external"],
        Field(description="Who is responsible for neighbor-list construction."),
    ] = "none"
    cutoff: Annotated[
        float | None,
        Field(description="Interaction cutoff radius."),
    ] = None
    format: Annotated[
        Literal["coo", "matrix"] | None,
        Field(description="Required neighbor-list storage layout."),
    ] = None
    half_list: Annotated[
        bool | None,
        Field(description="Whether the list is half or full."),
    ] = None
    max_neighbors: Annotated[
        int | None,
        Field(description="Optional maximum neighbors per atom for matrix format."),
    ] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_requirement(self) -> NeighborConfig:
        if self.source == "external" and self.format is None:
            raise ValueError("External neighbor requirements must declare a format.")
        return self


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Execution contract for one model wrapper.

    ``ModelConfig`` replaces the old boolean runtime config and describes
    the actual interface consumed by :class:`ComposableModelWrapper`.

    Attributes
    ----------
    required_inputs : frozenset[str]
        Input keys that must be present for this model to execute.
    optional_inputs : frozenset[str], default=frozenset()
        Input keys that may be supplied when available.
    outputs : frozenset[str], default=frozenset()
        Output keys always produced by the model.
    optional_outputs : dict[str, frozenset[str]], default_factory=dict
        Conditionally produced outputs keyed by their external dependencies.
    additive_outputs : frozenset[str], default=frozenset()
        Outputs that should be accumulated across composed models.
    use_autograd : bool, default=True
        Whether this model participates in the autograd-connected region.
    autograd_inputs : frozenset[str], default=frozenset()
        Inputs that must come from tensors tracked inside the autograd region.
    autograd_outputs : frozenset[str], default=frozenset()
        Outputs published into the active autograd region.
    pbc_mode : {"non-pbc", "pbc", "any"} | None, default=None
        Periodic-boundary support contract.
    neighbor_config : NeighborConfig, default=NeighborConfig()
        Structural neighbor-list requirement for this model.
    """

    required_inputs: frozenset[str] = frozenset()
    optional_inputs: frozenset[str] = frozenset()
    outputs: frozenset[str] = frozenset()
    optional_outputs: dict[str, frozenset[str]] = field(default_factory=dict)
    additive_outputs: frozenset[str] = frozenset()
    use_autograd: bool = True
    autograd_inputs: frozenset[str] = frozenset()
    autograd_outputs: frozenset[str] = frozenset()
    pbc_mode: Literal["non-pbc", "pbc", "any"] | None = None
    neighbor_config: NeighborConfig = field(default_factory=NeighborConfig)

    def __post_init__(self) -> None:
        """Validate internal consistency of the execution contract."""

        input_overlap = self.required_inputs & self.optional_inputs
        if input_overlap:
            raise ValueError(
                f"required_inputs and optional_inputs overlap: {input_overlap}"
            )

        all_inputs = self.required_inputs | self.optional_inputs
        all_outputs = self.outputs | frozenset(self.optional_outputs)

        if not self.autograd_inputs <= all_inputs:
            extra = self.autograd_inputs - all_inputs
            raise ValueError(
                f"autograd_inputs {extra} not in required_inputs | optional_inputs"
            )

        if not self.autograd_outputs <= all_outputs:
            extra = self.autograd_outputs - all_outputs
            raise ValueError(
                f"autograd_outputs {extra} not in outputs | optional_outputs"
            )

        if not self.additive_outputs <= all_outputs:
            extra = self.additive_outputs - all_outputs
            raise ValueError(
                f"additive_outputs {extra} not in outputs | optional_outputs"
            )

        for output_name, deps in self.optional_outputs.items():
            if not deps <= all_inputs:
                extra = deps - all_inputs
                raise ValueError(
                    f"optional_outputs['{output_name}'] deps {extra} "
                    f"not in required_inputs | optional_inputs"
                )

        overlap = self.autograd_outputs & self.additive_outputs
        if overlap:
            raise ValueError(
                f"autograd_outputs and additive_outputs overlap: {overlap}"
            )


def _resolve_config(
    config_cls: type[_C],
    config: _C | None,
    overrides: dict[str, Any],
) -> _C:
    """Resolve one config object from an optional base config and overrides.

    Parameters
    ----------
    config_cls
        Pydantic config class to instantiate when *config* is not provided.
    config
        Optional prebuilt config object.
    overrides
        Keyword-style overrides. Entries whose value is :data:`_UNSET`
        are ignored.

    Returns
    -------
    _C
        Resolved configuration instance.
    """

    filtered = {key: value for key, value in overrides.items() if value is not _UNSET}
    if config is not None and filtered:
        return config.model_copy(update=filtered)
    if config is not None:
        return config
    return config_cls(**filtered)


class PipelineContext:
    """Shared runtime state for a composable model execution.

    The context stores intermediate outputs, shared neighbor lists, and the
    tensors needed to run one explicit autograd region.

    Notes
    -----
    Wrapper code should treat :class:`PipelineContext` as a small behavioral
    contract.  In practice wrappers typically use :meth:`resolve` and
    :meth:`resolve_optional` to read values and do not need to depend on the
    concrete storage layout.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._neighbor_lists: dict[tuple[float, bool], NeighborList] = {}
        self._autograd_active: bool = False
        self._autograd_derivative_targets: dict[str, Tensor] = {}
        self._autograd_input_overrides: dict[str, Tensor] = {}
        self._autograd_registered_outputs: dict[str, Tensor] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return one stored value without consulting the batch."""

        return self._store.get(key, default)

    def keys(self) -> KeysView[str]:
        """Return the published key view."""

        return self._store.keys()

    def resolve(self, key: str, batch: Batch) -> Any:
        """Resolve one required key from autograd state, context, or batch."""

        if self._autograd_active and key in self._autograd_input_overrides:
            return self._autograd_input_overrides[key]
        if key in self._store:
            return self._store[key]
        try:
            return getattr(batch, key)
        except AttributeError:
            raise KeyError(
                f"Key '{key}' not found in PipelineContext or Batch"
            ) from None

    def resolve_optional(self, key: str, batch: Batch, default: Any = None) -> Any:
        """Resolve one optional key from autograd state, context, or batch."""

        if self._autograd_active and key in self._autograd_input_overrides:
            return self._autograd_input_overrides[key]
        if key in self._store:
            return self._store[key]
        try:
            return getattr(batch, key)
        except AttributeError:
            return default

    def accumulate(self, key: str, value: Tensor) -> None:
        """Accumulate one additive tensor output."""

        if key in self._store:
            self._store[key] = self._store[key] + value
        else:
            self._store[key] = value

    def store_neighbor_list(
        self,
        cutoff: float,
        half_list: bool,
        neighbor_list: NeighborList,
    ) -> None:
        """Store one neighbor payload under its structural share key."""

        self._neighbor_lists[(cutoff, half_list)] = neighbor_list

    def get_neighbor_list(
        self,
        cutoff: float,
        half_list: bool,
    ) -> NeighborList | None:
        """Return one stored neighbor payload by exact share key."""

        return self._neighbor_lists.get((cutoff, half_list))

    def remove_neighbor_list(self, cutoff: float, half_list: bool) -> None:
        """Drop one stored neighbor payload."""

        self._neighbor_lists.pop((cutoff, half_list), None)

    @property
    def autograd_active(self) -> bool:
        """Return whether the autograd region is active."""

        return self._autograd_active

    @property
    def autograd_derivative_targets(self) -> dict[str, Tensor]:
        """Return tracked derivative targets."""

        return self._autograd_derivative_targets

    @property
    def autograd_input_overrides(self) -> dict[str, Tensor]:
        """Return autograd input overrides."""

        return self._autograd_input_overrides

    @property
    def autograd_registered_outputs(self) -> dict[str, Tensor]:
        """Return outputs registered during the active autograd region."""

        return self._autograd_registered_outputs

    def activate_autograd(self, batch: Batch, targets: frozenset[str]) -> None:
        """Enable gradient tracking on the requested derivative targets."""

        self._autograd_active = True
        needs_positions = "positions" in targets or "cell_scaling" in targets
        needs_cell_scaling = "cell_scaling" in targets

        if needs_positions and not needs_cell_scaling:
            positions = batch.positions.detach().requires_grad_(True)
            self._autograd_derivative_targets["positions"] = positions
            self._autograd_input_overrides["positions"] = positions
            return

        if needs_cell_scaling:
            positions = batch.positions.detach().requires_grad_(True)
            cell = getattr(batch, "cell", None)
            if cell is None:
                # No cell available — fall back to positions-only autograd.
                # DerivativeStep._active_derivatives will skip stresses
                # because cell_scaling is not in the active targets.
                self._autograd_derivative_targets["positions"] = positions
                self._autograd_input_overrides["positions"] = positions
                return
            positions_scaled, cell_scaled, cell_scaling = self._prepare_stress_scaling(
                positions,
                cell,
                batch.batch,
            )
            self._autograd_derivative_targets["positions"] = positions_scaled
            self._autograd_input_overrides["positions"] = positions_scaled
            self._autograd_derivative_targets["cell_scaling"] = cell_scaling
            self._autograd_input_overrides["cell"] = cell_scaled

    def clear_autograd(self) -> None:
        """Clear all active autograd state."""

        self._autograd_derivative_targets.clear()
        self._autograd_input_overrides.clear()
        self._autograd_registered_outputs.clear()
        self._autograd_active = False

    def register_autograd_output(self, name: str, value: Tensor) -> None:
        """Register one autograd-visible output tensor."""

        self._autograd_registered_outputs[name] = value

    def get_autograd_target(self, name: str) -> Tensor:
        """Return one tracked derivative target."""

        return self._autograd_derivative_targets[name]

    @staticmethod
    def _prepare_stress_scaling(
        positions: Tensor,
        cell: Tensor,
        batch_idx: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare the affine scaling tensors used for stress derivatives."""

        scaling = torch.eye(3, dtype=positions.dtype, device=positions.device)
        if cell.ndim == 3:
            scaling = scaling.repeat(cell.shape[0], 1, 1)
        scaling = scaling.requires_grad_(True)

        if scaling.ndim == 3:
            atom_scaling = torch.index_select(scaling, 0, batch_idx)
            positions_scaled = (positions.unsqueeze(1) @ atom_scaling).squeeze(1)
        else:
            positions_scaled = positions @ scaling

        cell_scaled = cell @ scaling if cell.ndim == 2 else torch.bmm(cell, scaling)
        return positions_scaled, cell_scaled, scaling


class BaseModelMixin:
    """Mixin implementing the wrapper boundary for composable models.

    Subclasses are expected to provide a class-level :class:`ModelConfig`
    and a regular ``forward(data)`` implementation.  The default
    ``adapt_input`` and ``adapt_output`` helpers are sufficient for most
    wrappers whose inputs map directly onto batch or context keys.

    Attributes
    ----------
    spec : ModelConfig
        Static execution contract describing model inputs, outputs,
        autograd participation, and neighbor requirements.
    """

    spec: ModelConfig

    @property
    def model_config(self) -> ModelConfig:
        """Return the execution contract for this model."""

        return self.spec

    @model_config.setter
    def model_config(self, config: ModelConfig) -> None:
        """Update the execution contract for this model."""

        self.spec = config

    def __add__(self, other: object) -> object:
        """Compose this model with another model or composite.

        Parameters
        ----------
        other
            Another model wrapper or an existing
            :class:`~nvalchemi.models.composable.ComposableModelWrapper`.

        Returns
        -------
        object
            A :class:`~nvalchemi.models.composable.ComposableModelWrapper`
            containing both operands.
        """

        from nvalchemi.models.composable import ComposableModelWrapper

        return ComposableModelWrapper(self, other)

    def adapt_input(
        self,
        batch: Any,
        ctx: PipelineContext,
        *,
        compute: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build one model input mapping from the batch and runtime context.

        Parameters
        ----------
        batch
            Input batch-like object.
        ctx
            Runtime execution context.
        compute
            Requested outputs for this call. Present for wrappers that
            want to skip backend work when only a subset of outputs is
            needed.

        Returns
        -------
        dict[str, Any]
            Input mapping passed to ``forward``.

        Raises
        ------
        KeyError
            If a required input declared by :attr:`spec` is missing.
        """

        del compute
        data: dict[str, Any] = {}
        external_neighbor_keys = frozenset(
            {
                "edge_index",
                "neighbor_ptr",
                "unit_shifts",
                "neighbor_matrix",
                "num_neighbors",
                "neighbor_shifts",
                "fill_value",
            }
        )
        for key in self.spec.required_inputs:
            try:
                data[key] = ctx.resolve(key, batch)
            except KeyError:
                if (
                    self.spec.neighbor_config.source == "external"
                    and key in external_neighbor_keys
                ):
                    continue
                raise
        for key in self.spec.optional_inputs:
            value = ctx.resolve_optional(key, batch)
            if value is not None:
                data[key] = value
        return data

    def adapt_output(
        self,
        raw_output: dict[str, Any],
        *,
        compute: set[str] | None = None,
    ) -> dict[str, Any]:
        """Filter one raw backend output mapping to the requested keys.

        Parameters
        ----------
        raw_output
            Raw mapping returned by ``forward``.
        compute
            Requested output keys.  ``None`` returns the raw mapping unchanged.

        Returns
        -------
        dict[str, Any]
            Published output mapping.
        """

        if compute is None:
            return raw_output
        retained_keys = set(compute)
        retained_keys.update(
            (self.spec.outputs | frozenset(self.spec.optional_outputs))
            - self.spec.additive_outputs
        )
        return {
            key: value
            for key, value in raw_output.items()
            if key in retained_keys and value is not None
        }
