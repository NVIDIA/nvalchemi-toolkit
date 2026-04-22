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
"""Left-to-right composition helper for user-supplied data transforms.

A single :class:`Compose` class handles either per-sample transforms
(:data:`~nvalchemi._typing.SampleTransform`, taking and returning
``(AtomicData, metadata)``) or per-batch transforms
(:data:`~nvalchemi._typing.BatchTransform`, taking and returning
:class:`~nvalchemi.data.Batch`). The composition kind is fixed at
construction by inspecting each callable's type annotations; mixing
sample and batch transforms in the same :class:`Compose` is rejected.

Transforms must *return* their (possibly mutated) output; in-place
mutation without returning is not supported.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal, Sequence, get_type_hints

from plum import dispatch, overload

from nvalchemi._typing import BatchTransform, SampleTransform
from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch

TransformKind = Literal["sample", "batch"]


def _gate_transforms(
    transforms: Sequence[SampleTransform | BatchTransform],
) -> TransformKind | None:
    """Validate ``transforms`` and determine the composition kind.

    For each callable, if type hints are available the first positional
    parameter's annotation is inspected: :class:`AtomicData` marks a
    sample transform, :class:`Batch` marks a batch transform, anything
    else raises :class:`TypeError`. Untyped callables are skipped (no
    classification contribution). After all callables are processed,
    the typed ones must agree on a single kind, or :class:`TypeError`
    is raised.

    Parameters
    ----------
    transforms : Sequence[SampleTransform | BatchTransform]
        Callables to validate.

    Returns
    -------
    {"sample", "batch"} or None
        The composition kind, or ``None`` if no callable carried
        classifiable annotations (empty input or every entry untyped).

    Raises
    ------
    TypeError
        If a callable is not callable, carries a typed first parameter
        that is neither :class:`AtomicData` nor :class:`Batch`, or if
        typed callables disagree on the composition kind.
    """
    kind: TransformKind | None = None
    classifying_index: int | None = None

    for i, fn in enumerate(transforms):
        if not callable(fn):
            raise TypeError(
                f"Compose: transform at index {i} is not callable: "
                f"got {type(fn).__name__}"
            )

        # Resolve annotations. For non-function callables (e.g., class
        # instances with __call__), descend into __call__.
        hint_source: Any = fn
        if not (
            inspect.isfunction(fn) or inspect.ismethod(fn) or inspect.isbuiltin(fn)
        ):
            call = getattr(type(fn), "__call__", None)
            if call is not None:
                hint_source = call
        try:
            hints = get_type_hints(hint_source)
        except Exception:
            hints = {}

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue

        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if not positional:
            continue

        first_name = positional[0].name
        if first_name not in hints:
            # Untyped first parameter: skip this callable entirely.
            continue

        first_ann = hints[first_name]
        if first_ann is AtomicData:
            this_kind: TransformKind = "sample"
        elif first_ann is Batch:
            this_kind = "batch"
        else:
            raise TypeError(
                f"Compose: transform at index {i} ({fn!r}) has typed first "
                f"parameter {first_ann!r}; expected AtomicData (sample "
                f"transform) or Batch (batch transform)"
            )

        if kind is None:
            kind = this_kind
            classifying_index = i
        elif kind != this_kind:
            raise TypeError(
                f"Compose: cannot mix sample and batch transforms; "
                f"transform at index {classifying_index} is {kind!r} but "
                f"transform at index {i} is {this_kind!r}"
            )

    return kind


class Compose:
    """Compose a sequence of transforms into a single callable.

    Transforms are applied left-to-right: the output of transform
    ``i`` becomes the input to transform ``i + 1``. An empty sequence
    acts as the identity.

    A :class:`Compose` is either a *sample* composition or a *batch*
    composition, inferred at construction from the type annotations of
    the supplied callables. Each typed callable must declare its first
    positional parameter as :class:`~nvalchemi.data.AtomicData` (sample
    transform) or :class:`~nvalchemi.data.Batch` (batch transform);
    untyped callables are accepted and take on the kind established by
    the typed ones. Mixing sample and batch transforms in the same
    :class:`Compose` raises :class:`TypeError`. A composition with no
    typed callables defaults to sample semantics.

    Parameters
    ----------
    transforms : Sequence[SampleTransform | BatchTransform]
        Transforms to compose in application order. Any iterable is
        accepted and eagerly materialized into an immutable
        :class:`tuple`. Each entry must be callable.

    Attributes
    ----------
    transforms : tuple[SampleTransform, ...] | tuple[BatchTransform, ...]
        The composed transforms, stored in application order.
    is_batch : bool
        ``True`` if this is a batch composition, ``False`` if it is a
        sample composition. ``False`` for an empty / all-untyped
        composition (defaults to sample semantics).

    Raises
    ------
    TypeError
        If any entry is not callable, if a typed first parameter is
        neither :class:`AtomicData` nor :class:`Batch`, or if typed
        callables disagree on the composition kind.
    RuntimeError
        (At call time) If any transform raises; the original exception
        is wrapped in a :class:`RuntimeError` identifying the failing
        index, with the original exception attached via ``__cause__``.

    Examples
    --------
    Per-sample composition:

    >>> def shift_positions(data, metadata):  # doctest: +SKIP
    ...     new_data = data.replace(positions=data.positions + 1.0)
    ...     return new_data, metadata
    >>> def tag_origin(data, metadata):  # doctest: +SKIP
    ...     metadata["origin"] = "shifted"
    ...     return data, metadata
    >>> compose = Compose([shift_positions, tag_origin])  # doctest: +SKIP
    >>> new_data, new_meta = compose(data, {})  # doctest: +SKIP

    Per-batch composition:

    >>> def scale_positions(batch):  # doctest: +SKIP
    ...     return batch.replace(positions=batch.positions * 2.0)
    >>> compose = Compose([scale_positions])  # doctest: +SKIP
    >>> new_batch = compose(batch)  # doctest: +SKIP
    """

    __slots__ = ("transforms", "is_batch")

    def __init__(self, transforms: Sequence[SampleTransform | BatchTransform]) -> None:
        self.transforms: tuple[SampleTransform, ...] | tuple[BatchTransform, ...] = (
            tuple(transforms)
        )
        kind = _gate_transforms(self.transforms)
        self.is_batch: bool = kind == "batch"

    @overload
    def __call__(self, data: Batch) -> Batch:  # noqa: F811
        """Apply a batch composition to a :class:`Batch`."""
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as e:
                raise RuntimeError(
                    f"Compose: transform[{i}] ({transform!r}) raised {type(e).__name__}"
                ) from e
        return data

    @overload
    def __call__(  # noqa: F811
        self, data: AtomicData, metadata: dict[str, Any]
    ) -> tuple[AtomicData, dict[str, Any]]:
        """Apply a sample composition to an ``(AtomicData, metadata)`` pair."""
        for i, transform in enumerate(self.transforms):
            try:
                data, metadata = transform(data, metadata)
            except Exception as e:
                raise RuntimeError(
                    f"Compose: transform[{i}] ({transform!r}) raised {type(e).__name__}"
                ) from e
        return data, metadata

    @dispatch
    def __call__(  # noqa: F811
        self,
        data: AtomicData | Batch,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[AtomicData, dict[str, Any]] | Batch:
        """Apply the composed transforms left-to-right.

        Parameters
        ----------
        data : AtomicData or Batch
            For a sample composition, the :class:`AtomicData` sample.
            For a batch composition, the :class:`Batch`.
        metadata : dict[str, Any], optional
            Per-sample metadata dict. Required for sample compositions
            and must be omitted for batch compositions.

        Returns
        -------
        tuple[AtomicData, dict[str, Any]] or Batch
            The output of the final transform. For a sample composition,
            the ``(data, metadata)`` tuple. For a batch composition, the
            :class:`Batch`. An empty composition returns its input
            unchanged (identity).

        Raises
        ------
        TypeError
            If the call arity does not match the composition kind
            (e.g., passing ``metadata`` to a batch composition, or
            omitting it for a sample composition).
        RuntimeError
            If any transform raises; the message identifies the failing
            index and the original exception is attached via
            ``__cause__``.
        """
        ...

    def __repr__(self) -> str:
        """Return a developer-readable representation listing the transforms."""
        return f"Compose({list(self.transforms)!r})"
