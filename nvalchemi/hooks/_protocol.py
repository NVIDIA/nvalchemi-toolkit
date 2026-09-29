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
"""Hook protocol definition."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nvalchemi.hooks._context import HookContext


@runtime_checkable
class Hook(Protocol):
    """Protocol for hooks that observe or modify workflow state.

    Attributes
    ----------
    frequency : int
        How often the hook runs (every N steps).
    stage : Enum | None
        The stage enum value at which this hook runs, or ``None`` for
        hooks that are stage-agnostic until registered with a specific
        engine.
    on_register : Callable[[object], None], optional
        Optional lifecycle method called once by the registry after the
        hook passes stage/frequency validation and before it is stored.
        Hooks that mutate workflow topology or configuration should document
        their ordering assumptions because registration order is user-owned.
    """

    frequency: int
    stage: Enum | None

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Execute the hook.

        Only called when the registry determines the hook should fire at
        the dispatched stage.  By default, hooks fire when
        ``stage == self.stage``.  To fire at multiple stages, define a
        ``_runs_on_stage(self, stage: Enum) -> bool`` method that returns
        ``True`` for each relevant stage.

        Frequency gating is handled by the registry: hooks are only
        called when ``step_count % frequency == 0``.

        Parameters
        ----------
        ctx : HookContext
            Snapshot of the current workflow state. Workflow engines may pass
            a :class:`HookContext` subclass with additional fields.
        stage : Enum
            The stage being dispatched.
        """
        ...


@runtime_checkable
class CheckpointableHook(Protocol):
    """Protocol for hooks that own restart-critical runtime state.

    Most hooks should remain stateless and omit this protocol. Hooks that
    affect resumed training semantics can opt in by exposing ``state_dict``
    and ``load_state_dict``. Pydantic-backed hooks should use
    ``model_dump()`` inside their ``state_dict`` implementation for
    declarative fields and add only the extra runtime state they own.
    """

    def state_dict(self) -> Mapping[str, Any]:
        """Return hook state to store with a training checkpoint."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore hook state from a training checkpoint."""
        ...


@runtime_checkable
class StatefulHook(Hook, CheckpointableHook, Protocol):
    """Protocol for a hook whose state evolves with the workflow it observes.

    Composed rather than redeclared: :class:`Hook` already carries
    ``frequency`` and ``stage`` — the whole of "run this every N steps, at
    that point in the step" — and :class:`CheckpointableHook` already carries
    ``state_dict`` / ``load_state_dict``.  What neither says is whether a given
    dispatch is allowed to *change* anything, or when accumulated changes
    become visible to other workers.  This protocol adds exactly those two
    members, and nothing else.

    The pattern it describes recurs well beyond any one workflow: an adaptive
    bias depositing hills, an NEB run promoting its climbing image, an
    adaptive thermostat retuning its coupling, a neighbour list widening its
    skin.  All of them are read-only while forces are being computed, mutate
    after the step, and synchronise only occasionally.  Without a shared
    protocol each one invents its own vocabulary for the same three ideas.

    Attributes
    ----------
    read_only : bool
        ``False`` for a hook that mutates its own state when dispatched.
        Engines that evaluate the same step more than once — priming forces,
        re-evaluating under a proposed replica-exchange assignment — use this
        to tell a dispatch that must happen exactly once from one that is
        safe to repeat.

    Notes
    -----
    Domain hook families may keep a signature suited to their semantics
    rather than ``__call__(ctx, stage)``, as
    :class:`~nvalchemi.training.hooks.update.TrainingUpdateHook` does, in
    which case an orchestrator owns protocol compliance on their behalf.  The
    attributes below are the part worth sharing regardless.

    .. warning::

        ``isinstance`` against a runtime-checkable Protocol checks that the
        members *exist*, never that they have the right signature.  An object
        whose ``__call__`` takes something other than ``(ctx, stage)`` — any
        ``nn.Module``, for one, whose ``__call__`` is its forward — passes the
        check and then fails when dispatched.  Use ``isinstance`` to ask
        whether a hook owns state, not to decide that an arbitrary object is
        safe to call as a hook; that is the registering engine's business.
    """

    read_only: bool = False

    def commit(self) -> None:
        """Publish pending state at a synchronisation boundary.

        Optional; a hook whose state is local to one worker needs no
        synchronisation and may leave this a no-op.  Never called on the hot
        path — engines call it at whatever boundary they define as safe (an
        epoch, a segment, the end of a run), at most once per boundary.
        """
        ...
