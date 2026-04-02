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
"""Hook registry mixin for workflow engines."""

from __future__ import annotations

from enum import Enum

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks._protocol import Hook


class HookRegistryMixin:
    """Mixin providing flat-list hook storage and dispatch.

    The host class must provide a ``step_count`` attribute. Override
    ``_build_context`` to populate workflow-specific fields.

    Set ``_stage_type`` on the host class to restrict which stage enum
    types may be registered. When set, ``register_hook`` raises
    :class:`TypeError` if ``hook.stage`` is not an instance of the
    declared type(s).

    Attributes
    ----------
    hooks : list[Hook]
        Flat list of registered hooks.
    step_count : int
        Current step number (must be provided by the engine).
    _stage_type : type[Enum] | tuple[type[Enum], ...] | None
        Accepted stage enum type(s). ``None`` disables validation.
    """

    hooks: list[Hook]
    step_count: int
    _stage_type: type[Enum] | tuple[type[Enum], ...] | None = None

    def _init_hooks(self, hooks: list[Hook] | None = None) -> None:
        """Initialize hook storage and register provided hooks.

        Parameters
        ----------
        hooks : list[Hook] | None
            Optional list of hooks to register.
        """
        self.hooks = []
        if hooks:
            for hook in hooks:
                self.register_hook(hook)

    def register_hook(self, hook: Hook) -> None:
        """Register a hook.

        Parameters
        ----------
        hook : Hook
            Hook to register.

        Raises
        ------
        ValueError
            If ``hook.frequency`` is not a positive integer.
        TypeError
            If ``hook.stage`` is not an instance of the accepted
            ``_stage_type`` declared on the engine **and** the hook does
            not define ``_runs_on_stage`` (cross-category hooks that
            manage their own stage dispatch bypass this check).
        """
        if not isinstance(hook.frequency, int) or hook.frequency < 1:
            raise ValueError(
                f"Hook frequency must be a positive integer, got {hook.frequency}"
            )
        stage_type = self._stage_type
        # Hooks that define ``_runs_on_stage`` handle stage dispatch
        # themselves (e.g. cross-category hooks); skip the type check.
        has_custom_dispatch = getattr(hook, "_runs_on_stage", None) is not None
        if (
            stage_type is not None
            and not has_custom_dispatch
            and not isinstance(hook.stage, stage_type)
        ):
            expected = (
                stage_type.__name__
                if isinstance(stage_type, type)
                else " | ".join(t.__name__ for t in stage_type)
            )
            raise TypeError(
                f"Hook {hook!r} has stage={hook.stage!r} "
                f"(type {type(hook.stage).__name__}), but this engine "
                f"only accepts {expected} stages."
            )
        self.hooks.append(hook)

    def _build_context(self, batch: Batch) -> HookContext:
        """Build a HookContext for the current state.

        Override in subclasses to populate workflow-specific fields.

        Parameters
        ----------
        batch : Batch
            Current batch being processed.

        Returns
        -------
        HookContext
            Context object for hooks.
        """
        return HookContext(
            batch=batch,
            step_count=self.step_count,
            model=getattr(self, "model", None),
        )

    def _call_hooks(self, stage: Enum, batch: Batch) -> None:
        """Call hooks registered for the given stage, gated by frequency.

        Hooks fire when ``self.step_count % hook.frequency == 0``.
        Hooks that define ``_runs_on_stage`` are called when that method
        returns ``True``; otherwise, the default check is
        ``stage == hook.stage``.

        Parameters
        ----------
        stage : Enum
            Current workflow stage.
        batch : Batch
            Current batch being processed.
        """
        ctx = self._build_context(batch)
        for hook in self.hooks:
            runs_on_stage = getattr(hook, "_runs_on_stage", None)
            if runs_on_stage is not None:
                if not runs_on_stage(stage):
                    continue
            elif stage != hook.stage:
                continue
            if self.step_count % hook.frequency == 0:
                hook(ctx, stage)
