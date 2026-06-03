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
"""Composable :class:`torch.nn.Module`-based loss-function abstractions.

Leaf loss terms are tensor-to-tensor :class:`BaseLossFunction` instances
whose :meth:`~BaseLossFunction.forward` returns the raw, unweighted loss
tensor. :class:`ComposedLossFunction` owns the per-component weighting
(either floats or :class:`LossWeightSchedule` instances) and, by default,
normalizes the resolved weights so they sum to ``1.0`` at every call.
This keeps weight scheduling a *relative* knob and leaves the learning
rate as the sole *absolute* magnitude control.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import torch
from torch import nn

from nvalchemi.training.losses.base import LossWeightSchedule


class ComposedLossOutput(TypedDict):
    """Output returned by :class:`ComposedLossFunction`.

    This is solely used as a type hint, and not as a concrete data
    structure; it's used to signal to users that the emitted dict
    from composed losses will always at least contain the keys within
    this ``TypedDict``.

    The mapping always contains ``total_loss`` and four per-component
    sub-mappings keyed by component name. ``per_component_weight`` holds
    the effective (possibly normalized) weight actually applied to each
    component at this call; ``per_component_raw_weight`` holds the
    pre-normalization resolved weight — identical to
    ``per_component_weight`` when ``normalize_weights=False`` and useful
    for logging the underlying schedule value regardless of
    normalization. ``per_component_sample`` carries per-component
    **weighted** per-sample loss tensors of shape ``(B,)``, detached;
    see :attr:`BaseLossFunction.per_sample_loss` for the per-leaf
    populate-or-skip contract.
    """

    total_loss: torch.Tensor
    per_component_total: dict[str, torch.Tensor]
    per_component_weight: dict[str, float]
    per_component_raw_weight: dict[str, float]
    per_component_sample: dict[str, torch.Tensor]


def assert_same_shape(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
    prediction_key: str | None = None,
    target_key: str | None = None,
    strict: bool = False,
) -> None:
    """Raise :class:`ValueError` when ``pred`` and ``target`` are not compatible.

    Checks dtype equality first (a dtype mismatch is usually a bug
    upstream of shape), then the shape compatibility policy selected by
    ``strict``.

    Shape policy
    ------------
    ``strict=False`` (default) accepts any pair of shapes that is
    broadcast-compatible via :func:`torch.broadcast_shapes`. This is
    convenient for custom losses that legitimately broadcast (e.g. a
    per-graph scale against a per-component target) but is a trap for
    elementwise losses: ``(B, 1)`` vs ``(B, 3)`` passes, and the
    subsequent ``pred - target`` silently broadcasts into a ``(B, 3)``
    residual — usually not what you intend.

    ``strict=True`` requires ``pred.shape == target.shape`` exactly. All
    built-in leaf losses (:class:`EnergyMSELoss`, :class:`ForceMSELoss`,
    :class:`StressMSELoss`) pass ``strict=True`` because their elementwise
    arithmetic would otherwise corrupt the scalar loss under a
    broadcast-compatible-but-unequal pair. Custom
    :class:`BaseLossFunction` subclasses that do elementwise arithmetic
    should also pass ``strict=True``.

    Parameters
    ----------
    pred : torch.Tensor
        Prediction tensor.
    target : torch.Tensor
        Target tensor whose dtype must equal ``pred``'s and whose shape
        must be compatible with ``pred``'s under the selected policy.
    name : str
        Calling loss-term's class name, used as a prefix in the error
        message (typically ``type(self).__name__``).
    prediction_key : str, optional
        Key the prediction tensor was pulled from in the composed
        mapping. When provided, included in the error message.
    target_key : str, optional
        Key the target tensor was pulled from in the composed mapping.
        When provided, included in the error message.
    strict : bool, default False
        When ``True``, require ``pred.shape == target.shape``. When
        ``False``, only require broadcast compatibility.

    Raises
    ------
    ValueError
        If ``pred.dtype != target.dtype``, or if the shape policy is
        violated (broadcast-incompatible for ``strict=False``, unequal
        for ``strict=True``).
    """
    pred_fragment = (
        f"prediction_key={prediction_key!r}"
        if prediction_key is not None
        else "prediction"
    )
    target_fragment = (
        f"target_key={target_key!r}" if target_key is not None else "target"
    )
    if pred.dtype != target.dtype:
        raise ValueError(
            f"{name}: prediction and target dtype mismatch; "
            f"{pred_fragment} has dtype {pred.dtype}, "
            f"{target_fragment} has dtype {target.dtype}."
        )
    if strict:
        if pred.shape != target.shape:
            raise ValueError(
                f"{name}: prediction and target shape must match exactly "
                f"for elementwise loss; {pred_fragment} has shape "
                f"{tuple(pred.shape)}, {target_fragment} has shape "
                f"{tuple(target.shape)}."
            )
        return
    try:
        torch.broadcast_shapes(pred.shape, target.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name}: prediction and target shape mismatch; "
            f"{pred_fragment} has shape {tuple(pred.shape)}, "
            f"{target_fragment} has shape {tuple(target.shape)}."
        ) from exc


class ReductionContext(dict):
    """Lightweight metadata bag flowing through the loss template pipeline.

    A plain ``dict`` subclass used to pass metadata between
    :meth:`BaseLossFunction.normalize`, :meth:`~BaseLossFunction.mask`,
    and :meth:`~BaseLossFunction.reduce`. Using a bare ``dict`` instead
    of ``TypedDict(total=False)`` keeps the type ``torch.compile``-safe
    (Dynamo rejects ``TypedDict`` with optional keys).

    Conventional keys
    -----------------
    ``"weights"`` : torch.Tensor
        Per-sample weights for the final reduction. For energy losses
        with ``per_atom=True`` this carries atom counts ``(B, 1)``; for
        force losses it may carry per-atom or per-component weights.
    """


class BaseLossFunction(nn.Module, abc.ABC):
    """Abstract :class:`torch.nn.Module` base for ALCHEMI loss functions.

    ``BaseLossFunction`` implements a **template-method**
    :meth:`forward` pipeline that orchestrates five overridable hooks:

    1. :meth:`validate` — shape / dtype checks.
    2. :meth:`normalize` — pre-process ``pred`` and ``target``
       (e.g. per-atom energy division) and return a
       :class:`ReductionContext` for downstream hooks.
    3. :meth:`mask` — produce a boolean validity tensor
       (e.g. ``torch.isfinite``, padding masks).
    4. :meth:`compute_residual` — **abstract**; the only method every
       leaf *must* implement. Receives ``pred``, ``target``, and the
       validity ``mask`` produced by step 3.
    5. :meth:`reduce` — collapse the residual tensor and validity mask
       into a scalar loss and populate :attr:`per_sample_loss`.

    Loss authors subclass ``BaseLossFunction`` and override
    :meth:`compute_residual` at a minimum. Normalization, masking, and
    reduction come free via the defaults, or can be overridden
    individually for domain-specific behaviour (e.g. per-atom energy
    division in :meth:`normalize`, padding-aware force masking in
    :meth:`mask`, graph-balanced force reduction in :meth:`reduce`).

    Leaves are weightless — weighting and scheduling live on
    :class:`ComposedLossFunction`. Operator sugar
    (``scalar * leaf``, ``leaf + leaf``, ``sum([...])``) produces a
    composition; see :class:`ComposedLossFunction` for semantics.

    Attributes
    ----------
    per_sample_loss : torch.Tensor | None
        Detached per-graph loss tensor of shape ``(B,)`` left as a side
        effect of the most recent :meth:`forward` call, or ``None`` when
        the loss does not naturally compute a per-graph view (or when
        ``forward`` has never been called). Intended for logging and
        diagnostics only — gradients flow through the scalar returned by
        :meth:`forward`, not through this attribute.
    """

    def __init__(self) -> None:
        """Initialize the base loss as a stateless :class:`nn.Module`."""
        super().__init__()
        self.per_sample_loss: torch.Tensor | None = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Template-method pipeline: validate → normalize → mask → residual → reduce.

        Subclasses should **not** override this method. Override the
        individual hooks instead. Extra keyword arguments (``batch``,
        ``batch_idx``, ``num_nodes_per_graph``, etc.) are forwarded to
        every hook via ``**kwargs``.
        """
        self.per_sample_loss = None
        self.validate(pred, target)
        pred, target, ctx = self.normalize(pred, target, **kwargs)
        valid = self.mask(pred, target, ctx, **kwargs)
        residual = self.compute_residual(pred, target, valid)
        return self.reduce(residual, valid, ctx, **kwargs)

    def validate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Check shape and dtype compatibility of ``pred`` and ``target``.

        Default implementation calls :func:`assert_same_shape` with
        ``strict=True`` when ``prediction_key`` / ``target_key``
        attributes are present on the instance.
        """
        assert_same_shape(
            pred,
            target,
            name=type(self).__name__,
            prediction_key=getattr(self, "prediction_key", None),
            target_key=getattr(self, "target_key", None),
            strict=True,
        )

    def normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, ReductionContext]:
        """Pre-process prediction and target before residual computation.

        Returns a ``(pred, target, ctx)`` triple. The default
        implementation is the identity — ``ctx`` is empty.

        Override to inject per-atom energy division, or any other
        pre-processing that should be available to all loss authors as a
        composable step.
        """
        return pred, target, ReductionContext()

    def mask(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return a boolean validity mask for ``target``.

        The default implementation returns an all-``True`` mask matching
        ``target``'s shape. Override to exclude non-finite entries,
        padding, or any other invalid positions.
        """
        return torch.ones_like(target, dtype=torch.bool)

    @abc.abstractmethod
    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return the per-element residual tensor.

        This is the only hook that **must** be overridden. The ``valid``
        mask (from :meth:`mask`) is provided so the leaf can zero out
        invalid positions before computing the residual (important for
        operations like ``vector_norm`` where masking after the
        reduction would be incorrect).
        """

    def reduce(
        self,
        residual: torch.Tensor,
        valid: torch.Tensor,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Collapse a residual tensor to a scalar loss.

        The default implementation computes a validity-weighted mean:
        ``(residual * valid_float).sum() / valid_float.sum()``, where
        ``valid_float`` incorporates optional ``ctx["weights"]``.

        Override for domain-specific reductions (graph-balanced force
        reduction, RMSD, etc.). Implementations should also populate
        :attr:`per_sample_loss` with a detached ``(B,)`` tensor when a
        per-graph decomposition is available.
        """
        valid_weights = valid.to(dtype=residual.dtype)
        weights = ctx.get("weights")
        if weights is not None:
            valid_weights = valid_weights * weights.expand_as(residual)
        scalar = residual.mul(valid_weights).sum() / valid_weights.sum().clamp_min(1.0)
        self._populate_per_sample_loss(residual)
        return scalar

    def _populate_per_sample_loss(self, residual: torch.Tensor) -> None:
        """Set :attr:`per_sample_loss` when the residual has a per-graph shape."""
        if residual.ndim == 1:
            self.per_sample_loss = residual.detach()
        elif residual.ndim == 2 and residual.shape[-1] == 1:
            self.per_sample_loss = residual.squeeze(-1).detach()

    # Arithmetic dunders — return ComposedLossFunction.
    def __mul__(self, other: Any) -> ComposedLossFunction:
        """Return ``ComposedLossFunction([self], weights=[other])``.

        ``other`` may be a :class:`float`/:class:`int` or a
        :class:`LossWeightSchedule`.
        """
        match other:
            case bool():
                return NotImplemented
            case int() | float() | LossWeightSchedule():
                return ComposedLossFunction([self], weights=[other])
            case _:
                return NotImplemented

    def __rmul__(self, other: Any) -> ComposedLossFunction:
        """Mirror of :meth:`__mul__` for ``scalar * loss``."""
        return self.__mul__(other)

    def __add__(self, other: Any) -> ComposedLossFunction:
        """Return ``self + other`` flattening any existing composition.

        Both operands get weight ``1.0`` unless they are themselves
        compositions, in which case their existing weights are preserved.
        """
        if isinstance(other, ComposedLossFunction):
            return ComposedLossFunction(
                [self, *other.components],
                weights=[1.0, *other._weights],
                normalize_weights=other.normalize_weights,
            )
        if isinstance(other, BaseLossFunction):
            return ComposedLossFunction([self, other], weights=[1.0, 1.0])
        return NotImplemented

    def __radd__(self, other: Any) -> BaseLossFunction | ComposedLossFunction:
        """Return ``self`` when seeded with integer ``0`` (for :func:`sum`)."""
        if other == 0:
            return self
        if isinstance(other, (BaseLossFunction, ComposedLossFunction)):
            return self.__add__(other)
        return NotImplemented


def _resolve_weight(
    weight: LossWeightSchedule | float,
    step: int,
    epoch: int | None,
    *,
    context: str,
) -> float:
    """Resolve a single weight (float or schedule) to a finite float.

    Parameters
    ----------
    weight
        Either a plain scalar or a :class:`LossWeightSchedule`.
    step, epoch
        Training counters forwarded to the schedule.
    context
        Caller-supplied name (typically the component's class name) used
        in error messages.

    Raises
    ------
    ValueError
        If a ``per_epoch=True`` schedule is evaluated with
        ``epoch is None`` or the schedule returns a non-finite value.
    TypeError
        If the schedule returns a non-numeric value.
    """
    if not isinstance(weight, LossWeightSchedule):
        coerced = float(weight)
        if not math.isfinite(coerced):
            raise ValueError(
                f"{context}: weight {weight!r} is not finite; "
                "weights must be finite floats."
            )
        return coerced
    if weight.per_epoch and epoch is None:
        raise ValueError(
            f"epoch must be provided when the {context} loss weight "
            "schedule has per_epoch=True. Pass epoch=<current_epoch> to "
            "the loss, or set per_epoch=False on the schedule."
        )
    try:
        value = weight(step, epoch or 0)
    except TypeError as exc:
        raise TypeError(
            f"{type(weight).__name__} does not satisfy the "
            "LossWeightSchedule contract: __call__ must accept "
            "(step: int, epoch: int) and return a float."
        ) from exc
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{type(weight).__name__} returned {type(value).__name__}; "
            "LossWeightSchedule.__call__ must return float."
        )
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(
            f"{type(weight).__name__} for {context} returned non-finite "
            f"weight {coerced!r}; schedules must return finite floats."
        )
    return coerced


def _component_names(components: Sequence[BaseLossFunction]) -> tuple[str, ...]:
    """Return class names with suffixes applied to duplicate component types."""
    raw_names = tuple(type(comp).__name__ for comp in components)
    counts: dict[str, int] = {}
    for name in raw_names:
        counts[name] = counts.get(name, 0) + 1
    next_index: dict[str, int] = {}
    names: list[str] = []
    for name in raw_names:
        if counts[name] > 1:
            idx = next_index.get(name, 0)
            next_index[name] = idx + 1
            names.append(f"{name}_{idx}")
        else:
            names.append(name)
    return tuple(names)


class ComposedLossFunction(nn.Module):
    """Weighted sum of :class:`BaseLossFunction` components.

    This class owns the per-component weighting — leaves are weightless.
    Weights may be plain floats or :class:`LossWeightSchedule` instances;
    they are resolved to floats at call time. By default the resolved
    weights are normalized to sum to ``1.0`` so scheduling controls
    *relative* contributions while the learning rate controls the
    absolute loss magnitude. Opt out with ``normalize_weights=False``.

    Components live in an :class:`torch.nn.ModuleList` for
    ``.modules()`` / ``.state_dict()`` / nested-``__repr__`` support.
    When a component is itself a :class:`ComposedLossFunction`, its
    components and weights are flattened into the parent element-wise so
    ``(A + B) + C`` is equivalent to ``A + B + C``.

    Parameters
    ----------
    components
        Loss terms to combine; must contain at least one element.
    weights
        Optional per-component weights. When provided, ``weights`` must
        have the same length as ``components`` at construction time
        (i.e. top-level components — child weights inside nested
        compositions are multiplied element-wise by the parent weight
        during flattening). A ``None`` entry is shorthand for ``1.0``,
        so ``weights=[None, 2.0, None]`` means "component 1 gets 2×,
        others default". Passing ``weights=None`` defaults every
        component to ``1.0``.
    normalize_weights
        When ``True`` (default), resolved weights are divided by their
        sum at each call so the effective weights sum to ``1.0``. A
        zero-sum raises :class:`ValueError`. When ``False``, raw
        weighted sums are returned.

    Attributes
    ----------
    components
        :class:`torch.nn.ModuleList` of the flattened leaf components.
    normalize_weights
        Whether effective weights are renormalized to sum to ``1.0``.
    """

    def __init__(
        self,
        components: Sequence[BaseLossFunction | ComposedLossFunction],
        *,
        weights: Sequence[LossWeightSchedule | float | None] | None = None,
        normalize_weights: bool = True,
    ) -> None:
        """Store flattened components, their weights, and the normalization flag."""
        super().__init__()
        components = tuple(components)
        if len(components) == 0:
            raise ValueError("components must contain at least one loss term")
        for i, comp in enumerate(components):
            if not isinstance(comp, (BaseLossFunction, ComposedLossFunction)):
                raise TypeError(
                    f"components[{i}] must be a BaseLossFunction or "
                    f"ComposedLossFunction, got "
                    f"{type(comp).__name__}"
                )

        if weights is None:
            raw_weights: list[LossWeightSchedule | float] = [1.0] * len(components)
        else:
            raw_weights = [1.0 if w is None else w for w in weights]
            if len(raw_weights) != len(components):
                raise ValueError(
                    f"weights has length {len(raw_weights)} but components has "
                    f"length {len(components)}; lengths must match."
                )
            for i, w in enumerate(raw_weights):
                match w:
                    case bool():
                        valid = False
                    case int() | float() | LossWeightSchedule():
                        valid = True
                    case _:
                        valid = False
                if not valid:
                    raise TypeError(
                        f"weights[{i}] must be a float or LossWeightSchedule, "
                        f"got {type(w).__name__}."
                    )

        flat_components: list[BaseLossFunction] = []
        flat_weights: list[LossWeightSchedule | float] = []
        for comp, parent_w in zip(components, raw_weights, strict=True):
            if isinstance(comp, ComposedLossFunction):
                for child_comp, child_w in zip(
                    comp.components, comp._weights, strict=True
                ):
                    flat_components.append(child_comp)
                    flat_weights.append(_compose_weights(parent_w, child_w))
            else:
                flat_components.append(comp)
                flat_weights.append(parent_w)

        self.components: nn.ModuleList = nn.ModuleList(flat_components)
        self._weights: list[LossWeightSchedule | float] = flat_weights
        self.normalize_weights: bool = normalize_weights

    def _resolve_raw_and_effective(
        self, step: int, epoch: int | None
    ) -> tuple[tuple[str, ...], list[float], list[float]]:
        """Resolve raw and effective weights in a single pass.

        Returns a triple ``(names, raw, effective)`` where ``raw`` holds
        the per-component resolved floats (pre-normalization) and
        ``effective`` holds the weights that will actually be applied —
        identical to ``raw`` when :attr:`normalize_weights` is ``False``
        and ``raw / sum(raw)`` otherwise. When normalization is enabled
        the raw weights must sum to a strictly positive float; a sum
        that is non-positive (negative, zero, or non-finite from
        cancellation) is rejected with :class:`ValueError` because the
        resulting normalization either flips every contribution's sign
        or blows up. Individual raw weights may themselves be negative
        as long as their sum is positive.
        """
        names = _component_names(tuple(self.components))
        raw = [
            _resolve_weight(w, step, epoch, context=name)
            for w, name in zip(self._weights, names, strict=True)
        ]
        if not self.normalize_weights:
            return names, raw, list(raw)
        total = sum(raw)
        if not math.isfinite(total) or total <= 0.0:
            resolved = dict(zip(names, raw, strict=True))
            raise ValueError(
                "ComposedLossFunction: cannot normalize weights whose sum "
                f"is not strictly positive (sum={total!r}). Resolved "
                f"weights at step={step}, epoch={epoch}: {resolved}. "
                "Choose weights whose sum is a finite positive float or "
                "set normalize_weights=False."
            )
        effective = [w / total for w in raw]
        return names, raw, effective

    def current_weight(self, step: int = 0, epoch: int | None = None) -> list[float]:
        """Resolve each component's weight to a float for ``(step, epoch)``.

        When :attr:`normalize_weights` is ``True`` the returned list sums
        to ``1.0``; otherwise it is the raw resolved weights. With
        normalization enabled the raw sum must be a strictly positive
        float or :class:`ValueError` is raised.

        Parameters
        ----------
        step
            Current global training step.
        epoch
            Current training epoch, or ``None`` when unused.

        Returns
        -------
        list[float]
            One effective weight per component, in order.

        Raises
        ------
        ValueError
            If normalization is enabled and the raw weights do not sum
            to a strictly positive, finite float.
        """
        _, _, effective = self._resolve_raw_and_effective(step, epoch)
        return effective

    def weight_factors(
        self, step: int = 0, epoch: int | None = None
    ) -> dict[str, float]:
        """Return a flat ``{component_name: effective_weight}`` dict.

        Duplicate class names get numeric suffixes (``_0``, ``_1``, ...)
        applied to *all* colliding entries, not only the duplicates.
        """
        names = _component_names(tuple(self.components))
        effective = self.current_weight(step=step, epoch=epoch)
        return dict(zip(names, effective, strict=True))

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        step: int = 0,
        epoch: int | None = None,
        **kwargs: Any,
    ) -> ComposedLossOutput:
        """Return the weighted total loss and per-component diagnostics.

        Each component is called with the routed ``pred`` / ``target``
        tensors, then its raw loss is scaled by the effective weight for
        this step. The output's
        ``per_component_total`` contains ``effective_weight * raw_loss``
        per component; ``per_component_weight`` holds the scalar weights
        that were applied (after normalization, if enabled);
        ``per_component_raw_weight`` holds the pre-normalization
        resolved weights so schedule ramps remain observable on
        single-component normalized compositions; see
        :attr:`BaseLossFunction.per_sample_loss` for the
        ``per_component_sample`` contract.
        """
        names, raw_weights, effective = self._resolve_raw_and_effective(step, epoch)

        per_component_total: dict[str, torch.Tensor] = {}
        per_component_sample: dict[str, torch.Tensor] = {}
        per_component_weight: dict[str, float] = dict(
            zip(names, effective, strict=True)
        )
        per_component_raw_weight: dict[str, float] = dict(
            zip(names, raw_weights, strict=True)
        )
        total: torch.Tensor | None = None

        for name, comp, weight in zip(names, self.components, effective, strict=True):
            prediction_key = getattr(comp, "prediction_key", None)
            target_key = getattr(comp, "target_key", None)
            if prediction_key is None:
                raise AttributeError(
                    f"{type(comp).__name__} cannot be used in "
                    "ComposedLossFunction without a prediction_key attribute."
                )
            if target_key is None:
                raise AttributeError(
                    f"{type(comp).__name__} cannot be used in "
                    "ComposedLossFunction without a target_key attribute."
                )
            try:
                pred = predictions[prediction_key]
            except KeyError as exc:
                raise KeyError(
                    f"{type(comp).__name__}: prediction mapping is missing "
                    f"key {prediction_key!r}"
                ) from exc
            try:
                target = targets[target_key]
            except KeyError as exc:
                raise KeyError(
                    f"{type(comp).__name__}: target mapping is missing "
                    f"key {target_key!r}"
                ) from exc
            if not isinstance(pred, torch.Tensor):
                raise TypeError(
                    f"{type(comp).__name__}: prediction mapping key "
                    f"{prediction_key!r} must resolve to torch.Tensor, "
                    f"got {type(pred).__name__}."
                )
            if not isinstance(target, torch.Tensor):
                raise TypeError(
                    f"{type(comp).__name__}: target mapping key "
                    f"{target_key!r} must resolve to torch.Tensor, "
                    f"got {type(target).__name__}."
                )
            # Guard against stale diagnostics from custom leaves that forget to clear.
            comp.per_sample_loss = None
            raw = comp(pred, target, **kwargs)
            if not isinstance(raw, torch.Tensor):
                raise TypeError(
                    f"{type(comp).__name__} returned "
                    f"{type(raw).__name__} from forward(); "
                    "BaseLossFunction subclasses must return a torch.Tensor."
                )
            contribution = weight * raw
            per_component_total[name] = contribution
            sample = comp.per_sample_loss
            if sample is not None:
                if not isinstance(sample, torch.Tensor):
                    raise TypeError(
                        f"{type(comp).__name__} (component {name!r}) set "
                        f"per_sample_loss to {type(sample).__name__}; "
                        "must be a torch.Tensor or None."
                    )
                if sample.ndim != 1:
                    raise ValueError(
                        f"{type(comp).__name__} (component {name!r}) set "
                        f"per_sample_loss with shape {tuple(sample.shape)}; "
                        "must be a 1-D tensor of shape (B,)."
                    )
                per_component_sample[name] = (weight * sample).detach()
            total = contribution if total is None else total + contribution

        if total is None:
            raise RuntimeError("ComposedLossFunction has no components.")

        return cast(
            ComposedLossOutput,
            {
                "total_loss": total,
                "per_component_total": per_component_total,
                "per_component_weight": per_component_weight,
                "per_component_raw_weight": per_component_raw_weight,
                "per_component_sample": per_component_sample,
            },
        )

    def __mul__(self, other: Any) -> ComposedLossFunction:
        """Scale every component weight by a float ``other``.

        Only float/int scalars are accepted. Schedules are rejected with
        :class:`TypeError`: compose schedules onto the individual
        components before combining, or multiply the composition by a
        plain float.
        """
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            if isinstance(other, LossWeightSchedule):
                raise TypeError(
                    "Multiplying a ComposedLossFunction by a "
                    "LossWeightSchedule is not supported. Scale each "
                    "component individually (e.g. schedule * EnergyMSELoss()) "
                    "and compose the results, or multiply by a float."
                )
            return NotImplemented
        scale = float(other)
        scaled_weights = [_compose_weights(scale, w) for w in self._weights]
        return ComposedLossFunction(
            list(self.components),
            weights=scaled_weights,
            normalize_weights=self.normalize_weights,
        )

    def __rmul__(self, other: Any) -> ComposedLossFunction:
        """Mirror of :meth:`__mul__` for ``scalar * composition``."""
        return self.__mul__(other)

    def __add__(self, other: Any) -> ComposedLossFunction:
        """Return ``self + other`` flattening any existing composition.

        The result inherits :attr:`normalize_weights` from ``self``.
        Adding two compositions with mismatched ``normalize_weights``
        raises :class:`ValueError` — combine them explicitly via
        :class:`ComposedLossFunction` to pick the intended flag.
        """
        if isinstance(other, ComposedLossFunction):
            if self.normalize_weights != other.normalize_weights:
                raise ValueError(
                    "Cannot add ComposedLossFunctions with mismatched "
                    f"normalize_weights (self={self.normalize_weights}, "
                    f"other={other.normalize_weights}). Construct the "
                    "combined composition explicitly via "
                    "ComposedLossFunction(..., normalize_weights=...)."
                )
            return ComposedLossFunction(
                [*self.components, *other.components],
                weights=[*self._weights, *other._weights],
                normalize_weights=self.normalize_weights,
            )
        if isinstance(other, BaseLossFunction):
            return ComposedLossFunction(
                [*self.components, other],
                weights=[*self._weights, 1.0],
                normalize_weights=self.normalize_weights,
            )
        return NotImplemented

    def __radd__(self, other: Any) -> ComposedLossFunction:
        """Return ``self`` when seeded with integer ``0`` (for :func:`sum`)."""
        if other == 0:
            return self
        if isinstance(other, BaseLossFunction):
            return ComposedLossFunction(
                [other, *self.components],
                weights=[1.0, *self._weights],
                normalize_weights=self.normalize_weights,
            )
        return NotImplemented

    def extra_repr(self) -> str:
        """Expose component count and normalization alongside the default repr."""
        return (
            f"num_components={len(self.components)}, "
            f"normalize_weights={self.normalize_weights}"
        )


def _compose_weights(
    outer: LossWeightSchedule | float,
    inner: LossWeightSchedule | float,
) -> LossWeightSchedule | float:
    """Return ``outer * inner`` as a weight, keeping floats where possible.

    If either operand is a schedule, the result is a
    :class:`_ProductWeight` that resolves ``outer(step, epoch) *
    inner(step, epoch)`` lazily. Pure float × float collapses to a float.
    """
    outer_is_schedule = isinstance(outer, LossWeightSchedule)
    inner_is_schedule = isinstance(inner, LossWeightSchedule)
    if not outer_is_schedule and not inner_is_schedule:
        return float(outer) * float(inner)
    return _ProductWeight(outer, inner)


@dataclass(frozen=True)
class _ProductWeight:
    """Lazy product of two weights — either operand may be a schedule or a float.

    Needed for nested composition flattening: when a parent composition
    has a non-unity weight and a child's weight is a
    :class:`LossWeightSchedule`, the product cannot be resolved at
    construction time because the schedule is a callable of
    ``(step, epoch)``. :class:`_ProductWeight` captures both operands
    and evaluates the product at call time while structurally
    satisfying the :class:`LossWeightSchedule` protocol (``per_epoch``
    attribute + ``__call__``).
    """

    left: LossWeightSchedule | float
    right: LossWeightSchedule | float
    per_epoch: bool = field(init=False)

    def __post_init__(self) -> None:
        """Derive ``per_epoch`` from the two operands."""
        combined = bool(
            getattr(self.left, "per_epoch", False)
            or getattr(self.right, "per_epoch", False)
        )
        # Frozen dataclass → must go through object.__setattr__.
        object.__setattr__(self, "per_epoch", combined)

    def __call__(self, step: int, epoch: int) -> float:
        """Return ``left(step, epoch) * right(step, epoch)``."""
        left = self.left(step, epoch) if callable(self.left) else float(self.left)
        right = self.right(step, epoch) if callable(self.right) else float(self.right)
        return float(left) * float(right)
