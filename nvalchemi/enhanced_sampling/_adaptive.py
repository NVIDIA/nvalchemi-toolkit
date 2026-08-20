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

import math
from typing import TYPE_CHECKING, Any

from torch import nn

from nvalchemi.dynamics.base import DynamicsStage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nvalchemi.data import Batch
    from nvalchemi.enhanced_sampling._bias import BiasResult

__all__ = ["AdaptivePotentialMixin"]


def _values_agree(left: Any, right: Any) -> bool:
    """Return whether two fingerprint entries describe the same setting.

    Floats are compared with a relative tolerance so that storing a value
    through float32 and reading it back does not read as a configuration
    change, while a real difference (0.2 against 0.9) still does.

    Parameters
    ----------
    left, right:
        Fingerprint entries: scalars, strings, ``None``, or flat sequences.

    Returns
    -------
    bool
        ``True`` when the two agree.
    """
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_agree(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-12)
    return bool(left == right)


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
        stale and need re-priming.  It is also checkpointed per bias, ready
        for validating that an accepted replica-exchange state assignment is
        coherent — which the exchange does not consume yet.  A bias that
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

    def config_fingerprint(self) -> Mapping[str, Any]:
        """Return the constructor configuration the saved state depends on.

        Default is empty, which disables the check.  A bias whose persisted
        state is only meaningful under the settings that produced it should
        override this and list them.

        The distinction is between *state* and *configuration*.  An ABF
        histogram is state; the ``cv_range`` that decides what its bins mean
        is configuration.  Restoring the first without the second silently
        reinterprets every bin — bin 5 stops meaning ``r = 1.55`` and starts
        meaning ``r = 3.1``, with the same numbers in it.

        Include configuration held as a **buffer** too.  ``nn.Module``
        restores buffers by overwriting, so without the check a mismatched
        setting is not merely unvalidated: the caller's value is silently
        replaced by the checkpoint's, which is the opposite of what asking
        for it meant.

        Returns
        -------
        Mapping[str, Any]
            Zarr-representable scalars, strings, ``None``, or flat sequences
            of those.  Tensors must be converted to lists.
        """
        return {}

    def _check_config_fingerprint(self, saved: Mapping[str, Any]) -> None:
        """Reject a saved fingerprint that disagrees with this bias.

        Parameters
        ----------
        saved:
            The fingerprint recorded when the state was written.

        Raises
        ------
        ValueError
            If any recorded setting differs from the live one.
        """
        live = dict(self.config_fingerprint())
        differences = []
        for key in sorted(set(saved) | set(live)):
            was, now = saved.get(key, "<absent>"), live.get(key, "<absent>")
            if not _values_agree(was, now):
                differences.append(f"  {key}: checkpoint {was!r} vs bias {now!r}")
        if differences:
            raise ValueError(
                f"{type(self).__name__} "
                f"{getattr(self, 'name', '<unnamed>')!r}: the saved state was "
                f"produced under a different configuration:\n"
                + "\n".join(differences)
                + "\nThe stored state is only meaningful under the settings "
                "that produced it, so restoring it here would reinterpret it "
                "rather than continue it. Rebuild the bias with the "
                "checkpoint's configuration, or start a fresh run."
            )

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
            Zarr-representable state, always including ``state_version`` and,
            when :meth:`config_fingerprint` is overridden, the configuration
            that state is only meaningful under.
        """
        parent = getattr(super(), "state_dict", None)
        state: dict[str, Any] = dict(parent(*args, **kwargs)) if parent else {}
        state["state_version"] = self._state_version
        fingerprint = dict(self.config_fingerprint())
        if fingerprint:
            state["config_fingerprint"] = fingerprint
        return state

    def load_state_dict(
        self, state: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Restore bias state produced by :meth:`state_dict`.

        Strips the mixin's own keys before delegating, so that
        ``nn.Module.load_state_dict`` does not reject them as unexpected
        under its default ``strict=True``.  A recorded
        :meth:`config_fingerprint` is checked *before* delegating, since the
        delegate overwrites buffers and a check afterwards would come too
        late to stop the caller's configuration being replaced.

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

        Raises
        ------
        ValueError
            If the state was written under a different
            :meth:`config_fingerprint`.
        """
        remaining = dict(state)
        self._state_version = int(remaining.pop("state_version", 0))
        saved = remaining.pop("config_fingerprint", None)
        if saved is not None:
            # Before delegating: nn.Module.load_state_dict overwrites buffers,
            # so a mismatch caught afterwards would already have replaced the
            # caller's configuration with the checkpoint's.
            self._check_config_fingerprint(saved)
        parent = getattr(super(), "load_state_dict", None)
        if parent is not None:
            return parent(remaining, *args, **kwargs)
        return None
