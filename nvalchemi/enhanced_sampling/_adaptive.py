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
"""Battery for biases whose internal state evolves during sampling.

``AdaptivePotentialMixin`` is one of the composable mixins described in the
``BiasPotential`` docstring: a bias mixes in only what applies to it.  It
carries no energy, no forces, and no model half — a bias that is adaptive
*and* conservative mixes in both this and
:class:`~nvalchemi.enhanced_sampling.ConservativeBias`; a bias that is
adaptive and non-conservative (ABF) mixes in only this one.

The mixin exists to make one guarantee enforceable by the runner: **evaluation
never mutates state**.  ``evaluate()`` is read-only and compile-friendly;
every history-dependent change happens in :meth:`update`, which the runner
calls exactly once per due step, after the integration step has finished.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import nn

from nvalchemi.dynamics.base import DynamicsStage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nvalchemi.data import Batch
    from nvalchemi.enhanced_sampling._bias import BiasResult

__all__ = ["AdaptivePotentialMixin"]


class AdaptivePotentialMixin:
    """Mixin for a bias whose state changes as sampling proceeds.

    Mix in alongside whatever else applies, **this mixin first**::

        class WellTemperedMetaDynamicsBias(AdaptivePotentialMixin, ConservativeBias):
            ...   # conservative and adaptive

        class AdaptiveBiasingForce(AdaptivePotentialMixin):
            ...   # adaptive, but no energy to differentiate

    Order matters and is enforced.  :class:`ConservativeBias` inherits
    ``nn.Module``, which already defines ``state_dict`` / ``load_state_dict``
    for buffers; putting this mixin second would let those shadow the ones
    here and silently drop the bias history from every checkpoint.  With the
    mixin first, :meth:`state_dict` calls up the MRO and merges both.
    ``__init_subclass__`` raises ``TypeError`` on the wrong order rather than
    letting a checkpoint quietly lose data.

    The runner detects the capability with ``hasattr(bias, "update")``; there
    is no registration and no requirement to inherit this class.  A bias may
    implement ``update`` structurally instead.

    Attributes
    ----------
    update_frequency:
        Dynamics steps between :meth:`update` calls.  ``1`` means every step.
    observation_stage:
        Which stage the frame handed to :meth:`update` is captured at.

        * ``AFTER_STEP`` (default) — post-step coordinates.  What
          metadynamics wants: a hill is deposited at the configuration the
          system actually reached.
        * ``AFTER_COMPUTE`` — captured while ``batch.forces`` still holds the
          **unbiased** physical forces, before any bias contribution is
          added.  What ABF requires; observing biased forces would feed the
          estimator its own output.

    Notes
    -----
    State version
        :meth:`bump_state_version` records that the bias changed.  The runner
        reads :attr:`state_version` to decide whether forces in the batch are
        stale and need re-priming, and (from PR 5) to validate that an
        accepted replica-exchange state assignment is coherent.  A bias that
        mutates state inside :meth:`update` must call it.
    """

    update_frequency: int = 1
    observation_stage: DynamicsStage = DynamicsStage.AFTER_STEP

    # Incremented by bump_state_version(); never reset.
    _state_version: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject an MRO that would let ``nn.Module`` shadow this mixin.

        Parameters
        ----------
        **kwargs:
            Forwarded cooperatively up the MRO.

        Raises
        ------
        TypeError
            If ``nn.Module`` precedes this mixin, which would make
            :meth:`state_dict` unreachable and drop bias history from
            checkpoints without any error.
        """
        super().__init_subclass__(**kwargs)
        mro = cls.__mro__
        if nn.Module in mro and mro.index(AdaptivePotentialMixin) > mro.index(
            nn.Module
        ):
            raise TypeError(
                f"{cls.__name__}: AdaptivePotentialMixin must come before "
                f"nn.Module (and therefore before ConservativeBias) in the "
                f"base list, otherwise nn.Module.state_dict shadows the "
                f"mixin's and the bias history is silently dropped from "
                f"checkpoints. Write "
                f"'class {cls.__name__}(AdaptivePotentialMixin, ConservativeBias)'."
            )

    @property
    def state_version(self) -> int:
        """Monotonic counter of state-changing updates applied to this bias."""
        return self._state_version

    def bump_state_version(self) -> None:
        """Record that the bias state changed.

        Call from :meth:`update` whenever the change affects the energy the
        next :meth:`evaluate` will return.  An update that only accumulates
        statistics without changing the applied bias (an ABF bin below its
        minimum-sample threshold, say) should **not** bump — bumping forces
        the runner to re-prime forces for no reason.
        """
        self._state_version += 1

    def update(self, frames: Batch, result: BiasResult) -> None:
        """Consume a captured frame after the integration step finishes.

        Called by the runner exactly once per due step.  Free to mutate
        state, allocate, grow storage, and communicate — none of this is on
        the compiled path.

        Parameters
        ----------
        frames:
            The captured observation, stamped by the runner with
            ``walker_id``, ``thermodynamic_state_id``, and ``sampling_step``.
            Captured at :attr:`observation_stage`.
        result:
            The detached :class:`BiasResult` this bias returned during the
            preceding force evaluation.

        Raises
        ------
        NotImplementedError
            If the subclass does not override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} mixes in AdaptivePotentialMixin but does "
            "not implement update(frames, result)."
        )

    def commit_epoch(self) -> None:
        """Synchronise pending state at a consistency-epoch boundary.

        Default is a no-op: a bias whose history is local to one walker needs
        no synchronisation.  Multi-walker shared-history biases override this
        to publish and merge.  Called only at epoch boundaries, never on the
        hot path.
        """
        return None

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return the bias state for checkpointing, merged up the MRO.

        Cooperative: when mixed into an ``nn.Module`` bias this calls
        ``nn.Module.state_dict`` first, so buffers (umbrella centers, wall
        thresholds) and bias history land in one mapping.  A bias with real
        history (hills, reference frames, ABF bins) overrides this, calls
        ``super().state_dict()``, and adds its own entries.

        Parameters
        ----------
        *args, **kwargs:
            Forwarded to the next ``state_dict`` in the MRO, if any.

        Returns
        -------
        dict[str, Any]
            Zarr-representable state, always including ``state_version``.
        """
        parent = getattr(super(), "state_dict", None)
        state: dict[str, Any] = dict(parent(*args, **kwargs)) if parent else {}
        state["state_version"] = self._state_version
        return state

    def load_state_dict(
        self, state: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Restore bias state produced by :meth:`state_dict`.

        Strips the mixin's own keys before delegating, so that
        ``nn.Module.load_state_dict`` does not reject them as unexpected
        under its default ``strict=True``.

        Parameters
        ----------
        state:
            The mapping previously returned by :meth:`state_dict`.
        *args, **kwargs:
            Forwarded to the next ``load_state_dict`` in the MRO, if any.

        Returns
        -------
        Any
            Whatever the next ``load_state_dict`` returns, or ``None``.
        """
        remaining = dict(state)
        self._state_version = int(remaining.pop("state_version", 0))
        parent = getattr(super(), "load_state_dict", None)
        if parent is not None:
            return parent(remaining, *args, **kwargs)
        return None
