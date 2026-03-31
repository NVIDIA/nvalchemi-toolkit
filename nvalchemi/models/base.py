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
from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from pydantic import BaseModel
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.models.contracts import (
    NeighborListProfile,
    PotentialProfile,
    StepProfile,
)
from nvalchemi.models.metadata import ModelCard
from nvalchemi.models.results import CalculatorResults

if TYPE_CHECKING:
    from nvalchemi.models.neighbors import NeighborListBuilderConfig

_C = TypeVar("_C", bound=BaseModel)


class _UnsetType:
    """Sentinel for distinguishing 'not provided' from ``None``."""

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False


_UNSET: Any = _UnsetType()


def _resolve_config(
    config_cls: type[_C],
    config: _C | None,
    overrides: dict[str, Any],
) -> _C:
    """Resolve a config from an optional config object and/or keyword overrides.

    Parameters
    ----------
    config_cls
        The Pydantic config class to construct.
    config
        An existing config instance, or ``None``.
    overrides
        Keyword arguments whose values are **not** :data:`_UNSET`.
        Keys that map to :data:`_UNSET` are silently dropped.

    Returns
    -------
    _C
        A fully resolved config instance built from *config* and/or
        *overrides*.
    """

    overrides = {k: v for k, v in overrides.items() if v is not _UNSET}
    if config is not None and overrides:
        return config.model_copy(update=overrides)
    if config is not None:
        return config
    return config_cls(**overrides)


@dataclass(slots=True)
class RuntimeState:
    """Internal execution state shared across one composite evaluation.

    Attributes
    ----------
    input_overrides : dict[str, Any]
        Runtime-injected tensors that shadow batch or result values
        (e.g. gradient-tracked positions).
    derivative_targets : dict[str, torch.Tensor]
        Named tensors with ``requires_grad=True`` set up by potentials
        during the ``prepare()`` phase.
    requested_derivative_targets : frozenset[str]
        Derivative target names aggregated from all steps before
        the main loop begins (e.g. ``{"positions", "cell_scaling"}``).
    """

    input_overrides: dict[str, Any] = field(default_factory=dict)
    derivative_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    requested_derivative_targets: frozenset[str] = frozenset()


@dataclass(slots=True)
class ForwardContext:
    """Per-call context created by the base ``forward()`` method.

    Parameters
    ----------
    outputs
        The resolved set of outputs requested for this call.
    results
        Accumulated results from earlier pipeline steps (may be ``None``).
    runtime_state
        Derivative-tracking state injected by the composite (may be ``None``).
    """

    outputs: frozenset[str]
    results: CalculatorResults | None = None
    runtime_state: RuntimeState | None = None


class _CalculationStep(nn.Module, abc.ABC):
    """Low-level base class for explicit steps in a composite calculator.

    This is the raw resolved-profile primitive.  Direct subclasses must
    build a fully resolved profile before calling ``super().__init__``.

    Higher-level bases :class:`Potential` and :class:`_NeighborListStep`
    provide convenience ``__init__`` methods that resolve the profile
    automatically from the class-level ``card`` and keyword overrides.
    Wrapper authors should subclass those instead of this class.

    Parameters
    ----------
    profile
        Resolved step contract for this instance.
    name
        Optional human-readable step name.
    device
        Optional execution device.  ``None`` means no fixed execution
        device (the step operates on whatever device its input tensors
        are on).  Model-backed steps should set this to a concrete
        device.  This is optional runtime state, not part of the
        card/profile contract.
    """

    profile: StepProfile
    model_card: ModelCard | None = None

    def __init__(
        self,
        profile: StepProfile,
        *,
        name: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self._device: torch.device | None = (
            torch.device(device) if device is not None else None
        )
        self.profile = profile
        self.step_name = name or type(self).__name__
        self._declared_inputs: frozenset[str] = (
            profile.required_inputs | profile.optional_inputs
        )

    @property
    def device(self) -> torch.device | None:
        """Return the step's execution device, or ``None`` if unset."""

        return self._device

    def to(self, *args: Any, **kwargs: Any) -> _CalculationStep:
        """Move the step and update the tracked execution device."""

        result = super().to(*args, **kwargs)
        if args and isinstance(args[0], (torch.device, str, int)):
            result._device = torch.device(args[0])
        elif "device" in kwargs and kwargs["device"] is not None:
            result._device = torch.device(kwargs["device"])
        return result

    @staticmethod
    def freeze_parameters(module: nn.Module) -> None:
        """Freeze all parameters on a module for inference-only evaluation."""

        for p in module.parameters():
            p.requires_grad_(False)

    def active_outputs(self, outputs: Iterable[str] | None = None) -> frozenset[str]:
        """Return the requested outputs for the current call.

        Parameters
        ----------
        outputs
            Explicit output keys requested by the caller.  When ``None``
            the step's ``default_result_keys`` are used.

        Returns
        -------
        frozenset[str]
            Validated set of output keys.

        Raises
        ------
        ValueError
            If any requested key is not in the step's ``result_keys``.
        """

        active = (
            self.profile.default_result_keys if outputs is None else frozenset(outputs)
        )
        self.validate_requested_outputs(active)
        return active

    def required_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return the declared required inputs for one output request.

        Parameters
        ----------
        outputs
            Explicit output keys, or ``None`` for defaults.

        Returns
        -------
        frozenset[str]
            Batch or result keys the step must receive.
        """

        self.active_outputs(outputs)
        return self.profile.required_inputs

    def optional_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return the declared optional inputs for one output request.

        Parameters
        ----------
        outputs
            Explicit output keys, or ``None`` for defaults.

        Returns
        -------
        frozenset[str]
            Batch or result keys the step may use when available.
        """

        self.active_outputs(outputs)
        return self.profile.optional_inputs

    def validate_requested_outputs(self, outputs: frozenset[str]) -> None:
        """Validate requested outputs against the step specification.

        Parameters
        ----------
        outputs
            Output keys to validate.

        Raises
        ------
        ValueError
            If any key is not in the step's ``result_keys``.
        """

        unsupported = outputs - self.profile.result_keys
        if unsupported:
            raise ValueError(
                f"Unsupported outputs requested for {self.step_name}: "
                f"{sorted(unsupported)}. Supported: "
                f"{sorted(self.profile.result_keys)}."
            )

    # -- input resolution (private plumbing) ----------------------------------

    @staticmethod
    def _resolve_input(
        batch: Batch,
        key: str,
        *,
        results: CalculatorResults | None = None,
        runtime_state: RuntimeState | None = None,
    ) -> Any:
        """Resolve an input value from runtime state, results, or batch."""

        if runtime_state is not None and key in runtime_state.input_overrides:
            return runtime_state.input_overrides[key]
        if results is not None and key in results:
            return results[key]
        if hasattr(batch, key):
            return getattr(batch, key)
        raise KeyError(f"Missing required input {key!r}.")

    @staticmethod
    def _has_input(
        batch: Batch,
        key: str,
        *,
        results: CalculatorResults | None = None,
        runtime_state: RuntimeState | None = None,
    ) -> bool:
        """Return whether an input can be resolved from a supported source."""

        if runtime_state is not None and key in runtime_state.input_overrides:
            return True
        if results is not None and key in results:
            return True
        return hasattr(batch, key)

    def _validate_input_declaration(self, key: str) -> None:
        """Ensure *key* is declared by the step contract."""

        if key not in self._declared_inputs:
            raise RuntimeError(
                f"{self.step_name} tried to access undeclared input {key!r}. "
                f"Declared inputs: {sorted(self._declared_inputs)}."
            )

    # -- public input accessors (subclass API) --------------------------------

    def require_input(
        self,
        batch: Batch,
        key: str,
        ctx: ForwardContext,
    ) -> Any:
        """Resolve one declared required input for the current request.

        Parameters
        ----------
        batch
            Current input batch.
        key
            Name of the input to resolve.
        ctx
            Forward context carrying accumulated results and runtime state.

        Returns
        -------
        Any
            Resolved input value (tensor or other).

        Raises
        ------
        RuntimeError
            If *key* was not declared in the step's input contract.
        KeyError
            If *key* cannot be found in any source.
        """

        self._validate_input_declaration(key)
        return self._resolve_input(
            batch,
            key,
            results=ctx.results,
            runtime_state=ctx.runtime_state,
        )

    def optional_input(
        self,
        batch: Batch,
        key: str,
        ctx: ForwardContext,
    ) -> Any | None:
        """Resolve one declared optional input for the current request.

        Parameters
        ----------
        batch
            Current input batch.
        key
            Name of the input to resolve.
        ctx
            Forward context carrying accumulated results and runtime state.

        Returns
        -------
        Any or None
            Resolved input value, or ``None`` when the key is absent.

        Raises
        ------
        RuntimeError
            If *key* was not declared in the step's input contract.
        """

        self._validate_input_declaration(key)
        if not self._has_input(
            batch,
            key,
            results=ctx.results,
            runtime_state=ctx.runtime_state,
        ):
            return None
        return self._resolve_input(
            batch,
            key,
            results=ctx.results,
            runtime_state=ctx.runtime_state,
        )

    # -- shared helpers (subclass API) ----------------------------------------

    def resolve_periodic_inputs(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> tuple[Any | None, Any | None]:
        """Resolve optional ``cell`` and ``pbc`` inputs with both-or-neither enforcement.

        Returns
        -------
        tuple
            ``(cell, pbc)`` -- both ``None`` when non-periodic, both present
            when periodic.

        Raises
        ------
        ValueError
            If exactly one of ``cell`` / ``pbc`` is provided.
        """

        cell = self.optional_input(batch, "cell", ctx)
        pbc = self.optional_input(batch, "pbc", ctx)
        if (cell is None) != (pbc is None):
            raise ValueError(
                f"{self.step_name} requires both 'cell' and 'pbc' when "
                "periodic data is provided."
            )
        return cell, pbc

    def normalize_graph_scalar(
        self,
        batch: Batch,
        key: str,
        ctx: ForwardContext,
        *,
        default: float,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Resolve a graph-level scalar and normalize to shape ``[B]``.

        Parameters
        ----------
        batch
            Current batch.
        key
            Name of the optional input to resolve.
        ctx
            Forward context.
        default
            Fill value when the input is absent.
        dtype
            Target dtype for the returned tensor.
        device
            Target device for the returned tensor.
        """

        value = self.optional_input(batch, key, ctx)
        if value is None:
            tensor = torch.full(
                (batch.num_graphs,),
                fill_value=default,
                device=device,
                dtype=dtype,
            )
        else:
            tensor = torch.as_tensor(value, device=device, dtype=dtype)

        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 1:
            tensor = tensor.squeeze(-1)
        if tensor.shape[0] != batch.num_graphs:
            raise ValueError(
                f"{key!r} must have one value per graph. "
                f"Expected {batch.num_graphs}, got {tuple(tensor.shape)}."
            )
        return tensor

    def build_results(
        self,
        ctx: ForwardContext,
        **values: torch.Tensor | None,
    ) -> CalculatorResults:
        """Build a :class:`CalculatorResults` containing only active requested keys.

        Parameters
        ----------
        ctx
            Forward context (used to read ``ctx.outputs``).
        **values
            Result name to tensor mapping.  Keys not in ``ctx.outputs`` or
            whose value is ``None`` are silently skipped.
        """

        out = CalculatorResults()
        for key, value in values.items():
            if key in ctx.outputs and value is not None:
                out[key] = value
        return out

    # -- derivative target declaration ----------------------------------------

    def requested_derivative_targets(
        self,
        outputs: frozenset[str],
    ) -> frozenset[str]:
        """Return derivative targets this step needs for the given outputs.

        Default returns empty.  Derivative steps (e.g.
        :class:`EnergyDerivativesStep`) override to declare what targets
        they need (e.g. ``{"positions", "cell_scaling"}``).

        Parameters
        ----------
        outputs
            The resolved set of outputs requested from this step.
        """

        return frozenset()

    # -- pre-loop hook --------------------------------------------------------

    def prepare(
        self,
        batch: Batch,
        runtime_state: RuntimeState,
        outputs: frozenset[str],
    ) -> None:
        """Optional pre-loop hook called before the composite main loop.

        Override to set up runtime state (e.g. gradient tracking) before
        any step in the pipeline executes.  Default is a no-op.

        Parameters
        ----------
        batch
            The current input batch.
        runtime_state
            Shared runtime state for this composite evaluation.
        outputs
            The resolved set of outputs requested from this step.
        """

    # -- template method ------------------------------------------------------

    def forward(
        self,
        batch: Batch,
        *,
        results: CalculatorResults | None = None,
        outputs: Iterable[str] | None = None,
        _runtime_state: RuntimeState | None = None,
    ) -> CalculatorResults:
        """Run the step against read-only inputs and accumulated results.

        Wrapper authors should override :meth:`compute`, not this method.
        """

        active = self.active_outputs(outputs)
        if not active:
            return CalculatorResults()
        ctx = ForwardContext(
            outputs=active,
            results=results,
            runtime_state=_runtime_state,
        )
        return self.compute(batch, ctx)

    @abc.abstractmethod
    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Model-specific computation.  Override this, not ``forward()``.

        Parameters
        ----------
        batch
            The current input batch.
        ctx
            Forward context with resolved outputs, results, and runtime state.
        """


class Potential(_CalculationStep, abc.ABC):
    """Convenience base class for energy-producing calculation steps.

    Wrapper authors subclass this instead of :class:`_CalculationStep`.
    The ``__init__`` resolves the instance profile from the class-level
    ``card`` plus any keyword overrides, so subclasses never need to call
    ``card.to_profile(...)`` directly.

    Parameters
    ----------
    name
        Optional human-readable step name.
    device
        Optional execution device.
    **profile_overrides
        Forwarded to ``type(self).card.to_profile(**profile_overrides)``
        to produce the resolved instance profile.
    """

    profile: PotentialProfile
    model_card: ModelCard | None

    def __init__(
        self,
        *,
        name: str | None = None,
        device: torch.device | str | None = None,
        **profile_overrides: Any,
    ) -> None:
        resolved = type(self).card.to_profile(**profile_overrides)
        super().__init__(resolved, name=name, device=device)

    def neighbor_list_builder_config(
        self,
        **overrides: Any,
    ) -> NeighborListBuilderConfig | None:
        """Return a builder config matching this potential's external contract.

        Parameters
        ----------
        **overrides
            Keyword overrides applied on top of the advertised external
            neighbor requirement. This is intended for user-facing tweaks
            such as a larger cutoff or different reuse settings.

        Returns
        -------
        NeighborListBuilderConfig | None
            A ready-to-use builder config for external-neighbor potentials,
            or ``None`` when this potential manages neighbors internally or
            does not use neighbors.

        Raises
        ------
        ValueError
            If the potential advertises an external neighbor requirement
            without enough information to build a concrete config.
        """

        requirement = self.profile.neighbor_requirement
        if requirement.source != "external":
            return None
        if requirement.cutoff is None:
            raise ValueError(
                f"{self.step_name} advertises an external neighbor requirement "
                "without a cutoff, so a NeighborListBuilderConfig cannot be created."
            )

        from nvalchemi.models.neighbors import NeighborListBuilderConfig

        config_data: dict[str, Any] = {
            "neighbor_list_name": requirement.name,
            "cutoff": requirement.cutoff,
            "format": requirement.format,
            "reuse_if_available": True,
        }
        if requirement.half_list is not None:
            config_data["half_list"] = requirement.half_list
        config_data.update(overrides)
        return NeighborListBuilderConfig(**config_data)

    def prepare(
        self,
        batch: Batch,
        runtime_state: RuntimeState,
        outputs: frozenset[str],
    ) -> None:
        """Set up gradient tracking if this potential participates.

        Calls :meth:`_ensure_derivative_targets` for the intersection
        of this potential's declared ``gradient_setup_targets`` and the
        pipeline's requested targets.
        """

        targets = (
            self.profile.gradient_setup_targets
            & runtime_state.requested_derivative_targets
        )
        if targets:
            self._ensure_derivative_targets(batch, runtime_state, targets)

    @staticmethod
    def _prepare_stress_scaling(
        positions: torch.Tensor,
        cell: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Set up the affine scaling tensors used for stress via autograd.

        Parameters
        ----------
        positions
            Atomic positions ``[V, 3]`` (assumed Angstrom).
        cell
            Unit-cell matrix ``[3, 3]`` or ``[B, 3, 3]`` (assumed Angstrom).
        batch_idx
            Graph membership index per atom ``[V]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(positions_scaled, cell_scaled, scaling)`` where
            *scaling* is the ``[3, 3]`` or ``[B, 3, 3]`` identity with
            ``requires_grad=True``.
        """

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

    @staticmethod
    def _ensure_derivative_targets(
        batch: Batch,
        runtime_state: RuntimeState,
        targets: frozenset[str],
    ) -> None:
        """Set up gradient tracking for the requested derivative targets.

        This helper is **idempotent and incremental**: targets already
        present in ``runtime_state.derivative_targets`` are skipped.  If
        ``"positions"`` was registered from a prior call and
        ``"cell_scaling"`` is now requested, the helper upgrades
        positions to the stress-scaled version.

        Parameters
        ----------
        batch
            The current input batch.
        runtime_state
            Shared runtime state for this composite evaluation.
        targets
            Derivative target names to set up
            (e.g. ``{"positions", "cell_scaling"}``).
        """

        already_set = frozenset(runtime_state.derivative_targets.keys())
        needed = targets - already_set

        if not needed:
            return

        needs_positions = "positions" in needed or (
            "cell_scaling" in needed and "positions" not in already_set
        )
        needs_cell_scaling = "cell_scaling" in needed

        if needs_positions and not needs_cell_scaling:
            positions = batch.positions.detach().requires_grad_(True)
            runtime_state.derivative_targets["positions"] = positions
            runtime_state.input_overrides["positions"] = positions
            return

        if needs_cell_scaling:
            positions = batch.positions.detach().requires_grad_(True)
            cell = getattr(batch, "cell", None)
            if cell is None:
                raise ValueError("cell_scaling target requires batch.cell.")
            positions_scaled, cell_scaled, cell_scaling = (
                Potential._prepare_stress_scaling(positions, cell, batch.batch)
            )
            runtime_state.derivative_targets["positions"] = positions_scaled
            runtime_state.input_overrides["positions"] = positions_scaled
            runtime_state.derivative_targets["cell_scaling"] = cell_scaling
            runtime_state.input_overrides["cell"] = cell_scaled

    @staticmethod
    def normalize_system_energies(
        energy: torch.Tensor,
        *,
        num_graphs: int,
        source_name: str,
    ) -> torch.Tensor:
        """Normalize per-system energy outputs to shape ``[B, 1]``.

        Parameters
        ----------
        energy
            Raw energy tensor from the backend kernel.
        num_graphs
            Expected number of systems in the current batch.
        source_name
            Human-readable potential name used in error messages.

        Returns
        -------
        torch.Tensor
            Energy tensor reshaped to ``[B, 1]``.

        Raises
        ------
        ValueError
            If the tensor shape is unexpected or the batch dimension
            does not match *num_graphs*.
        """

        if energy.ndim == 0:
            energy = energy.reshape(1, 1)
        elif energy.ndim == 1:
            energy = energy.unsqueeze(-1)
        elif energy.ndim != 2 or energy.shape[-1] != 1:
            raise ValueError(
                f"Unexpected {source_name} energy shape: {tuple(energy.shape)}."
            )
        if energy.shape[0] != num_graphs:
            raise ValueError(
                f"{source_name} returned the wrong number of system energies. "
                f"Expected {num_graphs}, got {energy.shape[0]}."
            )
        return energy


class _NeighborListStep(_CalculationStep, abc.ABC):
    """Convenience base class for neighbor-list-producing steps.

    The ``__init__`` resolves the instance profile from the class-level
    ``card`` plus any keyword overrides, so subclasses never need to call
    ``card.to_profile(...)`` directly.

    Parameters
    ----------
    name
        Optional human-readable step name.
    **profile_overrides
        Forwarded to ``type(self).card.to_profile(**profile_overrides)``
        to produce the resolved instance profile.
    """

    profile: NeighborListProfile

    def __init__(
        self,
        *,
        name: str | None = None,
        **profile_overrides: Any,
    ) -> None:
        resolved = type(self).card.to_profile(**profile_overrides)
        super().__init__(resolved, name=name)
