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

from collections import OrderedDict
from collections.abc import Iterable

from torch import nn

from nvalchemi.data import Batch
from nvalchemi.models.base import (
    RuntimeState,
    _CalculationStep,
)
from nvalchemi.models.contracts import PipelineContract
from nvalchemi.models.results import CalculatorResults


class CompositeCalculator(nn.Module):
    """Generic ordered composition of pipeline steps.

    Steps are executed in insertion order.  Results are merged after
    each step, with additive keys accumulated across producers.  Steps
    may override ``prepare()`` to set up shared ``RuntimeState`` before
    the main loop (e.g. gradient tracking for energy differentiation).

    Attributes
    ----------
    outputs : frozenset[str]
        Default output keys used when :meth:`forward` is called
        without an explicit ``outputs`` argument.
    steps : nn.ModuleDict
        Ordered mapping of step name to :class:`_CalculationStep`.
    pipeline_contract : PipelineContract
        Aggregated contract describing the full pipeline's supported
        result keys and additive keys.

    Examples
    --------
    Build a composite that produces energies, forces, and stresses:

    >>> from nvalchemi.models import CompositeCalculator, EnergyDerivativesStep
    >>> calculator = CompositeCalculator(
    ...     potential,
    ...     EnergyDerivativesStep(),
    ...     outputs=["energies", "forces", "stresses"],
    ... )
    >>> results = calculator(batch)
    """

    def __init__(
        self,
        *steps: _CalculationStep,
        outputs: Iterable[str] | None = None,
    ) -> None:
        """Initialise a composite calculator from an ordered sequence of steps.

        Parameters
        ----------
        *steps
            Calculation steps executed in insertion order.  Each step
            must have a unique ``step_name``.
        outputs
            Default output keys requested when :meth:`forward` is
            called without an explicit ``outputs`` argument.  When
            ``None`` an empty set is used (each ``forward()`` call
            must then supply outputs explicitly).  Default ``None``.

        Raises
        ------
        ValueError
            If two steps share the same ``step_name`` or if any
            requested output key is not supported by the pipeline.
        """

        super().__init__()
        self.outputs = frozenset(outputs or ())

        named_steps: "OrderedDict[str, _CalculationStep]" = OrderedDict()
        for step in steps:
            if step.step_name in named_steps:
                raise ValueError(f"Duplicate step name: {step.step_name!r}")
            named_steps[step.step_name] = step

        self.steps = nn.ModuleDict(named_steps)
        self.pipeline_contract = self._assemble_contract()
        self._validate_requested_outputs(self.outputs)

    def forward(
        self,
        batch: Batch,
        *,
        results: CalculatorResults | None = None,
        outputs: Iterable[str] | None = None,
    ) -> CalculatorResults:
        """Run the pipeline and return results.

        Parameters
        ----------
        batch
            Input batch of atomic systems.
        results
            Pre-existing results to seed the working container.
            Default ``None``.
        outputs
            Explicit output keys.  When ``None`` the constructor-level
            ``outputs`` are used.  Default ``None``.

        Returns
        -------
        CalculatorResults
            Merged outputs from all participating steps.
        """

        active_outputs = self._active_outputs(outputs)
        working = results.copy() if results is not None else CalculatorResults()
        self._validate_external_inputs(batch, working, active_outputs)
        child_requests = self._resolve_child_requests(active_outputs)

        runtime_state = RuntimeState()
        all_derivative_targets: set[str] = set()
        for step, step_outputs_requested in zip(
            self.steps.values(),
            child_requests,
            strict=True,
        ):
            if step_outputs_requested:
                all_derivative_targets |= step.requested_derivative_targets(
                    step_outputs_requested,
                )
        runtime_state.requested_derivative_targets = frozenset(all_derivative_targets)

        for step, step_outputs_requested in zip(
            self.steps.values(),
            child_requests,
            strict=True,
        ):
            if step_outputs_requested:
                step.prepare(batch, runtime_state, step_outputs_requested)

        for step, step_outputs_requested in zip(
            self.steps.values(),
            child_requests,
            strict=True,
        ):
            if not step_outputs_requested:
                continue
            step_outputs = step(
                batch,
                results=working,
                outputs=step_outputs_requested,
                _runtime_state=runtime_state,
            )
            additive_keys = step.profile.additive_result_keys
            working.merge(step_outputs, additive_keys=additive_keys)

        return working

    def _assemble_contract(self) -> PipelineContract:
        """Build the transparent contract for the pipeline.

        Returns
        -------
        PipelineContract
            Aggregated contract with all step names, supported result
            keys, and additive result keys.
        """

        supported_results: set[str] = set()
        additive_results: set[str] = set()
        step_names = list(self.steps.keys())
        for step in self.steps.values():
            profile = step.profile
            supported_results |= profile.result_keys
            additive_results |= profile.additive_result_keys

        return PipelineContract(
            steps=tuple(step_names),
            result_keys=frozenset(supported_results),
            additive_result_keys=frozenset(additive_results),
        )

    def _validate_requested_outputs(self, active_outputs: frozenset[str]) -> None:
        """Validate a requested output set against the assembled contract.

        Parameters
        ----------
        active_outputs
            Output keys to validate.

        Raises
        ------
        ValueError
            If any key is not supported by the pipeline.
        """

        unsupported = active_outputs - self.pipeline_contract.result_keys
        if unsupported:
            raise ValueError(
                f"Unsupported outputs requested: {sorted(unsupported)}. "
                f"Composite supports: {sorted(self.pipeline_contract.result_keys)}."
            )

    def required_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Resolve caller-supplied required inputs for one output request.

        Parameters
        ----------
        outputs
            Explicit output keys, or ``None`` for constructor defaults.

        Returns
        -------
        frozenset[str]
            Batch keys the caller must supply.
        """

        active_outputs = self._active_outputs(outputs)
        return self._required_external_inputs(active_outputs)

    def optional_inputs(self, outputs: Iterable[str] | None = None) -> frozenset[str]:
        """Resolve caller-visible optional inputs for one output request.

        Parameters
        ----------
        outputs
            Explicit output keys, or ``None`` for constructor defaults.

        Returns
        -------
        frozenset[str]
            Batch keys the caller may optionally supply.
        """

        active_outputs = self._active_outputs(outputs)
        return self._optional_external_inputs(active_outputs)

    def _active_outputs(self, outputs: Iterable[str] | None = None) -> frozenset[str]:
        """Return the requested outputs for the current call.

        Parameters
        ----------
        outputs
            Explicit output keys, or ``None`` for constructor defaults.
            Default ``None``.

        Returns
        -------
        frozenset[str]
            Validated set of active output keys.
        """

        active_outputs = self.outputs if outputs is None else frozenset(outputs)
        self._validate_requested_outputs(active_outputs)
        return active_outputs

    def _required_external_inputs(
        self,
        active_outputs: frozenset[str],
    ) -> frozenset[str]:
        """Resolve caller-supplied required inputs for one output request.

        Parameters
        ----------
        active_outputs
            Validated output keys for this call.

        Returns
        -------
        frozenset[str]
            Batch keys the caller must supply (those not produced by
            earlier steps).
        """

        child_requests = self._resolve_child_requests(active_outputs)
        available_results: set[str] = set()
        required_inputs: set[str] = set()

        for step, request in zip(self.steps.values(), child_requests, strict=True):
            if not request:
                continue
            required_inputs |= step.required_inputs(request) - available_results
            available_results |= request

        return frozenset(required_inputs)

    def _optional_external_inputs(
        self,
        active_outputs: frozenset[str],
    ) -> frozenset[str]:
        """Resolve caller-visible optional inputs for one output request.

        Parameters
        ----------
        active_outputs
            Validated output keys for this call.

        Returns
        -------
        frozenset[str]
            Batch keys the caller may optionally supply (excluding
            those already required).
        """

        child_requests = self._resolve_child_requests(active_outputs)
        available_results: set[str] = set()
        required_inputs: set[str] = set()
        optional_inputs: set[str] = set()

        for step, request in zip(self.steps.values(), child_requests, strict=True):
            if not request:
                continue
            step_required = step.required_inputs(request)
            step_optional = step.optional_inputs(request)
            required_inputs |= step_required - available_results
            optional_inputs |= step_optional - available_results
            available_results |= request

        return frozenset(optional_inputs - required_inputs)

    def _validate_external_inputs(
        self,
        batch: Batch,
        results: CalculatorResults,
        active_outputs: frozenset[str],
    ) -> None:
        """Validate external inputs required by the assembled pipeline.

        Parameters
        ----------
        batch
            Current input batch.
        results
            Pre-existing results seeded into the pipeline.
        active_outputs
            Validated output keys for this call.

        Raises
        ------
        KeyError
            If any required input is missing from both *results* and
            *batch*.
        """

        required = self._required_external_inputs(active_outputs)
        missing = [key for key in required if key not in results and not hasattr(batch, key)]
        if missing:
            raise KeyError(f"Missing required inputs: {sorted(missing)}.")

    def _resolve_child_requests(
        self,
        active_outputs: frozenset[str],
    ) -> tuple[frozenset[str], ...]:
        """Resolve the outputs each step should produce for this call.

        Uses backward dependency analysis to determine which steps are
        needed, then expands additive keys so that every producer
        contributes.

        Parameters
        ----------
        active_outputs
            Validated output keys for this call.

        Returns
        -------
        tuple[frozenset[str], ...]
            Per-step output requests aligned with ``self.steps``.
        """

        downstream_needs: set[str] = set(active_outputs)
        requests_rev: list[frozenset[str]] = []
        steps = tuple(self.steps.values())

        for step in reversed(steps):
            profile = step.profile
            request = frozenset(downstream_needs & profile.result_keys)
            requests_rev.append(request)
            downstream_needs -= profile.result_keys
            downstream_needs |= step.required_inputs(request)

        raw_requests = tuple(reversed(requests_rev))

        all_resolved: set[str] = set()
        for request in raw_requests:
            all_resolved |= request
        expand_keys = all_resolved & self.pipeline_contract.additive_result_keys

        if not expand_keys:
            return raw_requests

        expanded_requests: list[frozenset[str]] = []
        for request, step in zip(raw_requests, steps, strict=True):
            step_expand = expand_keys & step.profile.result_keys
            expanded_requests.append(request | step_expand)
        return tuple(expanded_requests)
