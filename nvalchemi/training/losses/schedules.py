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
r"""Concrete weight schedules for loss functions.

Four Pydantic-validated schedules are provided: :class:`ConstantWeight`,
:class:`LinearWeight`, :class:`CosineWeight`, and :class:`PiecewiseWeight`.
Each satisfies the runtime-checkable
:class:`~nvalchemi.training.losses.base.LossWeightSchedule` protocol and
can be supplied inside :class:`ComposedLossFunction`'s ``weights``
sequence or on the left of ``schedule * leaf``.

The concrete schedules always receive both the global step and epoch.
When ``per_epoch=False`` (the default), schedule windows and boundaries
advance by global step. When ``per_epoch=True``, they advance by epoch,
which lets loss weights follow optimizers or learning-rate schedulers
that update once per epoch.

Notation
--------
Each schedule maps a scalar schedule index :math:`t` to a weight
:math:`w(t)`. :math:`t` is the global step when ``per_epoch=False`` (the
default) and the epoch when ``per_epoch=True``, and :math:`T` denotes the
``num_steps`` window length of the ramp schedules.

Serialization note
------------------

Schedules live in :class:`ComposedLossFunction`'s ``weights`` argument
rather than on leaves, and are reconstructed by the upstream
``TrainingStrategy`` from their ``(instance, spec)`` pair — the same
pattern used for models and optimizers (see
:mod:`nvalchemi.training._checkpoint`). A concrete schedule class still
round-trips standalone via :func:`~nvalchemi.training.create_model_spec`.

Adding a new schedule
---------------------

You can write any callable ``(step: int, epoch: int) -> float`` with a
``per_epoch`` attribute and it will satisfy the
:class:`~nvalchemi.training.losses.base.LossWeightSchedule` protocol.

To participate in :class:`~nvalchemi.training.strategy.TrainingStrategy`
checkpointing, custom schedule classes must also be spec-serializable.
Custom schedule classes must implement ``to_spec()`` returning a
:class:`~nvalchemi.training._spec.BaseSpec` so strategy checkpoints can
rebuild them. The built-in Pydantic schedule base provides this method
from ``model_dump()``.

Alternatively, subclass
:class:`~nvalchemi.training.losses.base._BaseWeightSchedule`:

1. Inherit to pick up ``per_epoch`` and the frozen Pydantic config.
2. Implement ``__call__(step: int, epoch: int) -> float``; use
   ``self._map_schedule_index(step, epoch)`` for schedules that advance
   over a single training counter.
"""

from __future__ import annotations

import bisect
import math
from typing import Annotated, TypeAlias

from pydantic import Field, model_validator

from nvalchemi.training.losses.base import _BaseWeightSchedule

_PositiveSteps: TypeAlias = Annotated[
    int,
    Field(
        gt=0,
        description="Positive length of the schedule window in steps or epochs.",
    ),
]


class ConstantWeight(_BaseWeightSchedule):
    """Time-invariant loss weight that returns ``value`` at every update.

    ``ConstantWeight`` is the simplest :class:`LossWeightSchedule`: it ignores
    both the global step and the epoch and always yields :attr:`value`. Reach
    for it when a component's contribution is fixed but you still want to
    express the weight as a schedule object -- for instance to keep a uniform
    type across a :class:`ComposedLossFunction`'s ``weights`` sequence, or to
    scale a leaf loss with the ``schedule * leaf`` operator. A bare ``float``
    weight behaves identically; the schedule form is mostly for symmetry and
    serialization.

    Examples
    --------
    >>> from nvalchemi.training.losses import ConstantWeight
    >>> w = ConstantWeight(value=2.5)
    >>> w(step=0, epoch=0), w(step=1000, epoch=9)
    (2.5, 2.5)

    Multiply a leaf loss to build a weighted component::

        >>> weighted = ConstantWeight(value=10.0) * ForceMSELoss()

    Notes
    -----
    Instances are frozen (immutable) per the shared ``_BaseWeightSchedule``
    config, so a schedule can be safely reused across components.
    """

    value: Annotated[float, Field(description="Constant weight value.")]

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        """Return :attr:`value`, ignoring ``step`` and ``epoch``."""
        return float(self.value)


class _RampSchedule(_BaseWeightSchedule):
    """Shared base for linear / cosine ramps from ``start`` to ``end``.

    Subclasses only differ in the curve applied to the clamped fraction
    ``t in [0, 1]``. The index is the global step when ``per_epoch=False``
    and the epoch when ``per_epoch=True``.
    """

    start: Annotated[float, Field(description="Weight at schedule index 0.")]
    end: Annotated[float, Field(description="Weight at schedule index `num_steps`.")]
    num_steps: _PositiveSteps

    def _ramp_fraction(self, step: int, epoch: int) -> float | None:
        """Return the clamped fraction ``t in [0, 1]`` or ``None`` outside the window.

        ``None`` means the caller should return the boundary value
        (``start`` for ``idx <= 0``; ``end`` for ``idx >= num_steps``).
        Otherwise the return is the raw linear fraction; subclasses apply
        their own curve to it.
        """
        idx = self._map_schedule_index(step, epoch)
        if idx <= 0 or idx >= self.num_steps:
            return None
        return idx / self.num_steps


class LinearWeight(_RampSchedule):
    """Loss weight that ramps linearly from ``start`` to ``end``.

    ``LinearWeight`` interpolates a component's weight along a straight line:
    it returns :attr:`start` at schedule index ``0`` and :attr:`end` at index
    :attr:`num_steps`, moving proportionally in between. Use it to phase a loss
    term in or out gradually -- for example warming a force or stress term up
    from ``0`` over the first few thousand updates, or annealing an auxiliary
    term down toward the end of training. The schedule index is the global step
    when ``per_epoch=False`` (default) and the epoch when ``per_epoch=True``,
    and the value is clamped to :attr:`start` for index ``<= 0`` and to
    :attr:`end` for index ``>= num_steps``.

    Examples
    --------
    >>> from nvalchemi.training.losses import LinearWeight
    >>> w = LinearWeight(start=0.0, end=1.0, num_steps=10)
    >>> w(step=0, epoch=0), w(step=5, epoch=0), w(step=100, epoch=0)
    (0.0, 0.5, 1.0)

    Advance the ramp once per epoch instead of per step::

        >>> w = LinearWeight(start=0.2, end=1.0, num_steps=10, per_epoch=True)

    Notes
    -----
    ``num_steps`` must be strictly positive. Instances are frozen (immutable)
    per the shared ``_BaseWeightSchedule`` config.
    """

    def __call__(self, step: int, epoch: int) -> float:
        """Linear ramp from ``start`` to ``end``, clamped at both ends."""
        frac = self._ramp_fraction(step, epoch)
        if frac is None:
            return float(
                self.start if self._map_schedule_index(step, epoch) <= 0 else self.end
            )
        return float(self.start + (self.end - self.start) * frac)


class CosineWeight(_RampSchedule):
    r"""Loss weight that eases from ``start`` to ``end`` on a half-cosine curve.

    ``CosineWeight`` interpolates like :class:`LinearWeight` but follows a
    half-cosine (smooth ``ease-in/ease-out``) path: it starts and ends nearly
    flat and changes fastest near the midpoint. Writing :math:`s` for
    ``start``, :math:`e` for ``end``, and :math:`T` for ``num_steps``, the
    weight at schedule index :math:`t` is

    .. math::

        w(t) = s + (e - s)\,\frac{1 - \cos(\pi \tau)}{2},
        \qquad \tau = \operatorname{clamp}\!\left(\frac{t}{T},\, 0,\, 1\right).

    The clamp on :math:`\tau` yields :math:`w = s` for :math:`t \le 0` (where
    :math:`\cos 0 = 1`) and :math:`w = e` for :math:`t \ge T` (where
    :math:`\cos \pi = -1`), with the fastest change at the midpoint
    :math:`t = T/2`. Prefer it over a linear ramp when you want a gentler onset
    and settle for a term, which can avoid the abrupt gradient shifts a sharp
    linear turn-on causes. The schedule index :math:`t` is the global step when
    ``per_epoch=False`` (default) and the epoch when ``per_epoch=True``.

    Examples
    --------
    >>> from nvalchemi.training.losses import CosineWeight
    >>> w = CosineWeight(start=0.0, end=1.0, num_steps=10)
    >>> w(step=0, epoch=0), round(w(step=5, epoch=0), 3), w(step=100, epoch=0)
    (0.0, 0.5, 1.0)

    Anneal a weight downward on the cosine curve::

        >>> w = CosineWeight(start=1.0, end=0.1, num_steps=5000)

    Notes
    -----
    ``num_steps`` must be strictly positive. Instances are frozen (immutable)
    per the shared ``_BaseWeightSchedule`` config.
    """

    def __call__(self, step: int, epoch: int) -> float:
        """Half-cosine interpolation, clamped at both ends."""
        frac = self._ramp_fraction(step, epoch)
        if frac is None:
            return float(
                self.start if self._map_schedule_index(step, epoch) <= 0 else self.end
            )
        # Half-cosine: cos(0)=1 at index=0 -> start; cos(pi)=-1 at num_steps -> end.
        curve = 0.5 * (1.0 - math.cos(math.pi * frac))
        return float(self.start + (self.end - self.start) * curve)


class PiecewiseWeight(_BaseWeightSchedule):
    r"""Step-function loss weight that switches value at fixed boundaries.

    ``PiecewiseWeight`` holds a constant weight within each interval and jumps
    to the next value once the schedule index crosses a boundary. Given
    boundaries :math:`b_0 < b_1 < \dots < b_{k-1}` and values
    :math:`v_0, \dots, v_k`, the weight at schedule index :math:`t` is the value
    of the interval that contains :math:`t`:

    .. math::

        w(t) =
        \begin{cases}
          v_0 & t < b_0, \\
          v_j & b_{j-1} \le t < b_j \quad (1 \le j \le k-1), \\
          v_k & t \ge b_{k-1}.
        \end{cases}

    Equivalently :math:`w(t) = v_j` with
    :math:`j = \bigl|\{\, m : b_m \le t \,\}\bigr|`, the number of boundaries the
    index has reached or passed (each interval is closed on the left). Use it
    for stage-wise or curriculum-style training where a term should be on/off or
    held at discrete levels rather than ramped continuously -- for example
    enabling a stress term only after a warm-up phase. The schedule index
    :math:`t` is the global step when ``per_epoch=False`` (default) and the
    epoch when ``per_epoch=True``.

    Examples
    --------
    >>> from nvalchemi.training.losses import PiecewiseWeight
    >>> w = PiecewiseWeight(boundaries=(10, 20), values=(0.1, 0.5, 0.9))
    >>> w(step=5, epoch=0), w(step=15, epoch=0), w(step=25, epoch=0)
    (0.1, 0.5, 0.9)

    Switch weights per epoch instead of per step::

        >>> w = PiecewiseWeight(
        ...     boundaries=(5,), values=(0.0, 1.0), per_epoch=True
        ... )

    Notes
    -----
    ``values`` must have exactly ``len(boundaries) + 1`` entries and
    ``boundaries`` must be strictly increasing and non-negative; an ``after``
    validator raises ``ValueError`` otherwise. Fields are tuples (not lists) so
    instances stay hashable under the frozen model config.
    """

    boundaries: Annotated[
        tuple[int, ...],
        Field(
            description=(
                "Strictly increasing, non-negative schedule-index boundaries."
            ),
        ),
    ]
    values: Annotated[
        tuple[float, ...],
        Field(description="Values for each interval; length len(boundaries) + 1."),
    ]

    @model_validator(mode="after")
    def _check_boundaries_and_values(self) -> PiecewiseWeight:
        """Enforce strictly-increasing non-negative boundaries and correct length."""
        if len(self.values) != len(self.boundaries) + 1:
            raise ValueError(
                f"values must have length len(boundaries) + 1; got "
                f"len(values)={len(self.values)}, "
                f"len(boundaries)={len(self.boundaries)}"
            )
        prev = -1
        for b in self.boundaries:
            if b < 0:
                raise ValueError(
                    f"boundaries must be non-negative; got {self.boundaries}"
                )
            if b <= prev:
                raise ValueError(
                    f"boundaries must be strictly increasing; got {self.boundaries}"
                )
            prev = b
        return self

    def __call__(self, step: int, epoch: int) -> float:
        """Return the value of the interval containing the schedule index.

        ``bisect_right`` gives the count of boundaries that the index has
        reached or passed, which is the index into :attr:`values`.
        """
        idx = bisect.bisect_right(
            self.boundaries, self._map_schedule_index(step, epoch)
        )
        return float(self.values[idx])
