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
"""Teacher scoring interfaces for knowledge distillation."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

import torch

from nvalchemi.models.base import ModelConfig, NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.training.runtime import evaluating

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "BUILTIN_SIGNALS",
    "InProcessTeacherScorer",
    "NeighborListPolicy",
    "SUPPORTED_SIGNALS",
    "SignalLevel",
    "TeacherLabels",
    "TeacherScorer",
    "TeacherSignal",
    "scorer_fields",
    "signal_fields",
    "signal_for_field",
]

SignalLevel: TypeAlias = Literal["node", "system"]
"""Batch level a teacher signal is attached at."""

TeacherLabels: TypeAlias = dict[str, tuple[torch.Tensor, SignalLevel]]
"""Teacher signals for one batch, keyed by the batch field they populate."""

NeighborListPolicy: TypeAlias = Literal["rebuild", "reuse"]
"""Where :class:`InProcessTeacherScorer` takes the teacher's neighbor list from."""

_NEIGHBOR_LIST_POLICIES: frozenset[str] = frozenset({"rebuild", "reuse"})
"""The two values of :data:`NeighborListPolicy`."""

_TEACHER_FIELD_PREFIX = "teacher_"
"""Namespace every teacher field lives in, clear of a batch's own fields."""

_SIGNAL_LEVELS: frozenset[str] = frozenset({"node", "system"})
"""The two levels a teacher signal may be attached at."""


@dataclasses.dataclass(frozen=True)
class TeacherSignal:
    """Model output, batch field, level, and shape rule behind one teacher signal.

    A signal names what the teacher produces — *model_output*, a key in its
    outputs — and where the label lands: *field*, at *level*. A signal with
    ``model_output=None`` is one the scorer derives outside the forward pass,
    such as the built-in ``embeddings`` through
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`; a custom
    signal must name an output. Pass an instance to
    :class:`InProcessTeacherScorer` beside the built-in names to label a
    teacher output the built-in table does not cover.

    Parameters
    ----------
    name : str
        Signal name, as a scorer is asked for it.
    model_output : str | None
        Key of the teacher output the label is read from, or ``None`` for a
        signal the scorer derives by a route of its own.
    field : str
        Batch field the label populates; must start with ``teacher_``.
    level : SignalLevel
        ``"node"`` for one row per atom, ``"system"`` for one per graph.
    normalize : Callable[[torch.Tensor, Batch], torch.Tensor] | None, optional
        Reshapes the detached raw output to the field's canonical shape, given
        the batch it was produced for. Default ``None`` (keep the shape).
    extra_fields : tuple[str, ...], optional
        Companion fields the signal writes beside *field*, each in the
        ``teacher_*`` namespace. Default ``()``.

    Raises
    ------
    ValueError
        If *field* or a companion field falls outside the ``teacher_*``
        namespace, or *level* is neither ``"node"`` nor ``"system"``.

    Examples
    --------
    >>> from nvalchemi.training.distillation import TeacherSignal
    >>> charges = TeacherSignal("charges", "charges", "teacher_charges", "node")
    >>> charges.fields
    ('teacher_charges',)
    """

    name: str
    model_output: str | None
    field: str
    level: SignalLevel
    normalize: Callable[[torch.Tensor, Batch], torch.Tensor] | None = None
    extra_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Hold the fields to the ``teacher_*`` namespace and the level to the two."""
        foreign = [
            field
            for field in self.fields
            if not field.startswith(_TEACHER_FIELD_PREFIX)
        ]
        if foreign:
            raise ValueError(
                f"Teacher signal {self.name!r} must populate the 'teacher_*' "
                "namespace, so a batch's own reference fields are never overwritten; "
                f"got {foreign!r}."
            )
        if self.level not in _SIGNAL_LEVELS:
            raise ValueError(
                f"Teacher signal {self.name!r} must be attached at 'node' or "
                f"'system'; got level {self.level!r}."
            )

    @property
    def fields(self) -> tuple[str, ...]:
        """Return every batch field the signal populates, its own first."""
        return (self.field, *self.extra_fields)


def _energy_shape(value: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Return a per-graph energy as ``(B, 1)``."""
    return value.unsqueeze(-1) if value.ndim == 1 else value


def _atomic_energies_shape(value: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Return per-atom energies as ``(V,)``."""
    return value.reshape(-1)


def _stress_shape(value: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Return a per-graph stress as ``(B, 3, 3)``."""
    return value.reshape(-1, 3, 3)


BUILTIN_SIGNALS: Mapping[str, TeacherSignal] = MappingProxyType(
    {
        "energy": TeacherSignal(
            "energy", "energy", "teacher_energy", "system", _energy_shape
        ),
        "forces": TeacherSignal("forces", "forces", "teacher_forces", "node"),
        "stress": TeacherSignal(
            "stress", "stress", "teacher_stress", "system", _stress_shape
        ),
        "atomic_energies": TeacherSignal(
            "atomic_energies",
            "atomic_energies",
            "teacher_atomic_energies",
            "node",
            _atomic_energies_shape,
        ),
        "embeddings": TeacherSignal(
            "embeddings", None, "teacher_node_embeddings", "node"
        ),
    }
)
"""Built-in teacher signals by name; a scorer takes these names or a :class:`TeacherSignal`."""

SUPPORTED_SIGNALS: frozenset[str] = frozenset(BUILTIN_SIGNALS)
"""Names of the built-in signals, the strings :class:`InProcessTeacherScorer` resolves."""

_DERIVED_SIGNALS: frozenset[str] = frozenset({"embeddings"})
"""Built-in signals the scorer derives outside the forward pass, by name."""

_DENSE_NEIGHBOR_KEYS = frozenset(
    {"neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"}
)
"""Node-level neighbor tensors a ``MATRIX`` build writes."""

_SPARSE_NEIGHBOR_KEYS = frozenset({"neighbor_list", "neighbor_list_shifts"})
"""Edge-level neighbor tensors a ``COO`` build writes."""

_NEIGHBOR_KEYS = _DENSE_NEIGHBOR_KEYS | _SPARSE_NEIGHBOR_KEYS
"""Ephemeral neighbor keys; the distillation package's shared definition."""

_STORABLE_DTYPES = (torch.float16, torch.float32, torch.float64)
"""Floating-point dtypes an ALCHEMI Zarr store can hold."""

_EMBEDDING_KEYS = frozenset({"node_embeddings", "graph_embeddings"})
"""Batch keys that :meth:`compute_embeddings` implementations write in place."""

_COUNT_LEVELS = frozenset({"atoms", "system"})
"""Batch levels whose storage holds the atom and graph counts and never leaves."""

_CUTOFF_ATTR = "_neighbor_list_cutoff"
"""Batch attribute recording the cutoff a neighbor list was built at."""

_PIPELINE_SOURCES_ATTR = "_pipeline_neighbor_sources"
"""Instance-dict attribute holding a composed pipeline's per-source neighbor lists."""

_SHADOWED_NEIGHBOR_ATTRS = _NEIGHBOR_KEYS | {"edge_ptr", _CUTOFF_ATTR}
"""Instance-dict neighbor attributes snapshotted and restored around a rebuild."""

_REQUIRED_NEIGHBOR_KEYS: dict[NeighborListFormat, tuple[str, ...]] = {
    NeighborListFormat.COO: ("neighbor_list",),
    NeighborListFormat.MATRIX: ("neighbor_matrix", "num_neighbors"),
}
"""Batch keys a teacher reads its neighbor list from, by format."""

_BATCH_STATE_ATTRS = frozenset({"device", "keys"})
"""Public instance-dict entries of a batch that hold state rather than a field."""


def _resolve_signals(
    signals: Iterable[str | TeacherSignal],
) -> dict[str, TeacherSignal]:
    """Return the spec behind every entry of *signals*, keyed by name, in order.

    A name is looked up in :data:`BUILTIN_SIGNALS`; a :class:`TeacherSignal` is
    taken as given. A repeated entry is folded into one.

    Raises
    ------
    KeyError
        If a name is not a built-in signal.
    ValueError
        If two different specs share a name, or two specs claim one field.
    """
    resolved: dict[str, TeacherSignal] = {}
    for signal in signals:
        if isinstance(signal, TeacherSignal):
            spec = signal
        else:
            spec = BUILTIN_SIGNALS.get(signal)
            if spec is None:
                raise KeyError(
                    f"Unknown teacher signal {signal!r}; supported signals are "
                    f"{sorted(BUILTIN_SIGNALS)!r}, and any other must be given "
                    "as a TeacherSignal."
                )
        previous = resolved.setdefault(spec.name, spec)
        if previous != spec:
            raise ValueError(
                f"Teacher signals must have distinct names; got two specs named "
                f"{spec.name!r}."
            )
    owners: dict[str, str] = {}
    for spec in resolved.values():
        for field in spec.fields:
            owner = owners.setdefault(field, spec.name)
            if owner != spec.name:
                raise ValueError(
                    f"Teacher signals must populate distinct fields; got {field!r} "
                    f"from both {owner!r} and {spec.name!r}."
                )
    return resolved


def signal_fields(signals: Iterable[str | TeacherSignal]) -> tuple[str, ...]:
    """Return every batch field the given signals populate, sorted.

    A signal may populate companion fields alongside its own; all are reported
    so a consumer can prepare for the whole set a scorer will write.

    Parameters
    ----------
    signals : Iterable[str | TeacherSignal]
        Built-in signal names or :class:`TeacherSignal` specs.

    Returns
    -------
    tuple[str, ...]
        Batch field names, deduplicated and sorted.

    Raises
    ------
    KeyError
        If a name is not a built-in signal.
    ValueError
        If two specs share a name or claim one field.
    """
    fields: set[str] = set()
    for spec in _resolve_signals(signals).values():
        fields.update(spec.fields)
    return tuple(sorted(fields))


def signal_for_field(
    field: str, signals: Iterable[str | TeacherSignal] | None = None
) -> str | None:
    """Return the signal that populates *field*, or ``None`` when none does.

    Parameters
    ----------
    field : str
        Batch field name to resolve back to the signal that writes it.
    signals : Iterable[str | TeacherSignal] | None, optional
        Signals to search, as built-in names or :class:`TeacherSignal` specs.
        Default ``None`` (the built-in signals).

    Returns
    -------
    str | None
        Name of the signal populating *field*, as its own or a companion
        field, or ``None`` when no searched signal writes it.
    """
    specs = (
        BUILTIN_SIGNALS.values()
        if signals is None
        else _resolve_signals(signals).values()
    )
    for spec in specs:
        if field in spec.fields:
            return spec.name
    return None


def _reject_foreign_fields(fields: Iterable[str], subject: str) -> None:
    """Raise ``ValueError`` when any of *fields* falls outside ``teacher_*``.

    *subject* opens the message and names where the fields came from: a
    scorer's declared ``label_fields`` or the labels it actually returned.
    """
    foreign = sorted(
        field for field in fields if not field.startswith(_TEACHER_FIELD_PREFIX)
    )
    if foreign:
        raise ValueError(
            f"{subject} must populate the 'teacher_*' namespace, so a batch's own "
            f"reference fields are never overwritten; got {foreign!r}. Rename each "
            "into the namespace, or stop the scorer writing it."
        )


def _node_embedding_shapes(teacher: BaseModelMixin) -> dict[str, tuple[int, ...]]:
    """Return the teacher's embedding shapes, empty when it publishes none."""
    try:
        return teacher.embedding_shapes or {}
    except NotImplementedError:
        return {}


def _planned_neighbor_sources(teacher: BaseModelMixin) -> int:
    """Return how many neighbor lists *teacher* consumes per batch."""
    factory = getattr(teacher, "make_neighbor_hooks", None)
    if not callable(factory):
        return 1
    hooks = factory()
    if not isinstance(hooks, list) or not hooks:
        return 1
    sources = getattr(hooks[0], "sources", None)
    return len(sources) if isinstance(sources, (list, tuple)) else 1


def _check_reusable_neighbors(batch: Batch, config: NeighborConfig) -> None:
    """Raise unless *batch* carries a list the teacher's format can consume.

    Only what a rebuild would otherwise have to guess is checked: the keys the
    teacher's format reads, in storage or shadowed in the instance dictionary,
    and the cutoff stamp when the batch carries one. Whether the list holds
    each pair once or twice is recorded nowhere, so matching it to the
    teacher's ``half_list`` is the caller's.
    """
    required = _REQUIRED_NEIGHBOR_KEYS[config.format]
    missing = [
        key for key in required if key not in batch.__dict__ and key not in batch
    ]
    if missing:
        raise ValueError(
            f"neighbor_list='reuse' needs the batch to carry the "
            f"{config.format.value!r} neighbor list the teacher consumes, but it has "
            f"no {missing!r}. Build the list before scoring, or pass "
            "neighbor_list='rebuild' to let the scorer build the teacher's own."
        )
    cutoff = getattr(batch, _CUTOFF_ATTR, None)
    if cutoff is not None and float(cutoff) != float(config.cutoff):
        raise ValueError(
            f"neighbor_list='reuse' needs the batch's neighbor list built at the "
            f"teacher's cutoff {config.cutoff!r}, but it is stamped with cutoff "
            f"{cutoff!r}. Build the list at the teacher's cutoff, or pass "
            "neighbor_list='rebuild'."
        )


def _snapshot_grad_flags(batch: Batch, config: ModelConfig) -> dict[str, bool]:
    """Return the ``requires_grad`` flag of every input the teacher may enable."""
    keys = {"positions"} | set(config.gradient_keys) | set(config.autograd_inputs)
    flags: dict[str, bool] = {}
    for key in keys:
        value = getattr(batch, key, None)
        if isinstance(value, torch.Tensor):
            flags[key] = value.requires_grad
    return flags


def _restore_grad_flags(batch: Batch, flags: dict[str, bool]) -> None:
    """Restore the ``requires_grad`` flags captured by :func:`_snapshot_grad_flags`."""
    for key, flag in flags.items():
        value = getattr(batch, key, None)
        if isinstance(value, torch.Tensor) and value.requires_grad != flag:
            value.requires_grad_(flag)


@contextmanager
def _isolated_neighbors(
    batch: Batch,
    config: NeighborConfig | None,
    neighbor_list: NeighborListPolicy = "rebuild",
) -> Iterator[None]:
    """Give the teacher the neighbor list *neighbor_list* names, restoring state on exit.

    ``"rebuild"`` snapshots the node-level neighbor tensors, the edge group,
    and every neighbor attribute in the batch's instance dictionary — where a
    composed pipeline shadows its default source's list — builds the teacher's
    own list, so the teacher can resolve nothing else, and restores all of it
    afterwards. ``"reuse"`` consumes the batch's list after
    :func:`_check_reusable_neighbors` and builds nothing. The per-source table
    a composed pipeline captures under ``_pipeline_neighbor_sources`` is
    hidden for the whole block either way, because a composed teacher consults
    it before anything canonical.

    Parameters
    ----------
    batch : Batch
        Batch to score on; mutated for the duration of the block under
        ``"rebuild"``.
    config : NeighborConfig | None
        Neighbor requirements of the teacher, or ``None`` for a model that
        needs no neighbor list, which makes the block a no-op.
    neighbor_list : NeighborListPolicy, optional
        ``"rebuild"`` or ``"reuse"``. Default ``"rebuild"``.

    Yields
    ------
    None

    Raises
    ------
    ValueError
        If *neighbor_list* is ``"reuse"`` and the batch carries no list the
        teacher's format reads, or one stamped with another cutoff.
    """
    saved_sources = (
        {_PIPELINE_SOURCES_ATTR: batch.__dict__.pop(_PIPELINE_SOURCES_ATTR)}
        if _PIPELINE_SOURCES_ATTR in batch.__dict__
        else {}
    )
    try:
        if config is None:
            yield
            return
        if neighbor_list == "reuse":
            _check_reusable_neighbors(batch, config)
            yield
            return

        atoms = batch._atoms_group
        saved_nodes = (
            {key: atoms[key] for key in _NEIGHBOR_KEYS if key in atoms}
            if atoms is not None
            else {}
        )
        saved_edges = batch.pop_level("edges")
        saved_shadows = {
            name: batch.__dict__.pop(name)
            for name in _SHADOWED_NEIGHBOR_ATTRS
            if name in batch.__dict__
        }
        if atoms is not None:
            for key in saved_nodes:
                del atoms[key]
        try:
            compute_neighbors(batch, config=config)
            yield
        finally:
            if atoms is not None:
                for key in _NEIGHBOR_KEYS:
                    if key in atoms:
                        del atoms[key]
                for key, value in saved_nodes.items():
                    atoms[key] = value
            if saved_edges is None:
                batch.drop_level("edges")
            else:
                batch.set_level("edges", saved_edges)
            for name in _SHADOWED_NEIGHBOR_ATTRS:
                batch.__dict__.pop(name, None)
            batch.__dict__.update(saved_shadows)
    finally:
        batch.__dict__.update(saved_sources)


def _restore_at_level(batch: Batch, key: str, value: torch.Tensor, level: str) -> None:
    """Write *value* back to *batch* under *key* at the *level* it was taken from.

    The value is split along the level's own pointer so
    :meth:`~nvalchemi.data.Batch.add_key` lands it on that level instead of
    the one the attribute registry would pick.
    """
    ptr = batch.level_ptr(level).tolist()
    rows = [value[start:stop] for start, stop in zip(ptr[:-1], ptr[1:], strict=True)]
    batch.add_key(key, rows, level=level, overwrite=True)


def _field_shadows(batch: Batch) -> dict[str, Any]:
    """Return the instance-dict entries a model can shadow *batch*'s fields with.

    The private entries are left out: they are the neighbor provenance stamps
    and the captured source table, which :func:`_isolated_neighbors` owns.
    """
    return {
        name: value
        for name, value in batch.__dict__.items()
        if not name.startswith("_") and name not in _BATCH_STATE_ATTRS
    }


@contextmanager
def _isolated_fields(batch: Batch) -> Iterator[None]:
    """Score with *batch*, restoring on exit every field the teacher writes.

    A composed teacher wires one stage into the next through the batch: the
    pipeline writes an intermediate such as ``charges`` straight into the
    instance dictionary, where it shadows the batch's own field of that name,
    and an autograd group replaces each of its gradient inputs with a fresh
    leaf in storage. Neither is rolled back, so a teacher scored on a live
    batch would hand the student its charges in place of the batch's, or a
    positions tensor cut loose from the graph the student built it on.

    Both are undone by recording every stored field and every shadow by
    reference — no tensor is copied, so the cost is one dictionary per level —
    and afterwards dropping what appeared and putting back what was replaced.
    A field the teacher deleted outright is re-added at the level it came
    from. A tensor a teacher edits in place is not recovered; nothing short
    of cloning the batch could.

    Parameters
    ----------
    batch : Batch
        Batch the teacher runs on; its fields are restored on exit.

    Yields
    ------
    None
    """
    levels = {
        level: (
            {field: batch[field] for field in fields},
            int(batch.level_ptr(level)[-1]),
        )
        for level, fields in batch.level_keys.items()
    }
    shadows = _field_shadows(batch)
    try:
        yield
    finally:
        current = batch.level_keys
        for level in current:
            if level not in levels:
                batch.drop_level(level)
        for level, (saved, cardinality) in levels.items():
            fields = current.get(level)
            if fields is None:
                continue
            for field in fields - set(saved):
                del batch[field]
            for field, value in saved.items():
                if field not in batch:
                    _restore_at_level(batch, field, value, level)
                elif batch[field] is not value:
                    batch[field] = value
            if not saved and not cardinality and level not in _COUNT_LEVELS:
                batch.drop_level(level)
        for name in _field_shadows(batch):
            if name not in shadows:
                del batch.__dict__[name]
        batch.__dict__.update(shadows)


@runtime_checkable
class TeacherScorer(Protocol):
    """Structural interface for objects that produce teacher signals for a batch.

    An implementation declares the ``signals`` it emits and returns, for one
    :class:`~nvalchemi.data.Batch`, ``{batch field: (tensor, level)}`` with
    levels ``"node"`` or ``"system"`` as :meth:`~nvalchemi.data.Batch.add_key`
    takes them. Tensors must be detached and live on the batch's device.

    It may also publish ``label_fields``, the sequence of batch fields
    :meth:`label` populates (never a bare string); consumers read it through
    :func:`scorer_fields`. An implementation naming a built-in signal is read
    as writing every field that signal populates, so one that writes fewer,
    or one whose signals are its own, must declare ``label_fields``. The
    protocol will not grow required members.

    See Also
    --------
    InProcessTeacherScorer : Scorer that evaluates a teacher in this process.
    nvalchemi.training.distillation.labeling.label_dataset : Offline consumer.
    """

    signals: frozenset[str]

    def label(self, batch: Batch) -> TeacherLabels:
        """Return ``{batch field: (detached tensor, level)}`` for *batch*."""
        ...


def scorer_fields(scorer: TeacherScorer) -> tuple[str, ...] | None:
    """Return the batch fields *scorer* populates, or ``None`` when they cannot be known.

    A ``label_fields`` declaration is taken at its word; otherwise a scorer
    whose signals are all built in gets :func:`signal_fields` of them,
    companion fields included; otherwise the fields are unknown, since a
    custom signal may map onto any field.

    ``None`` is not ``()``: a scorer that labels nothing declares ``()``, while
    an undeclared scorer with a custom signal resolves to ``None``, which a
    consumer must treat as unknown rather than as nothing to check.

    Parameters
    ----------
    scorer : TeacherScorer
        Scorer to resolve the fields of.

    Returns
    -------
    tuple[str, ...] | None
        Batch field names the scorer writes, or ``None`` when they cannot be
        determined without scoring a batch.

    Raises
    ------
    TypeError
        If *scorer* declares ``label_fields`` as a string, which would
        otherwise resolve to its characters.
    """
    declared = getattr(scorer, "label_fields", None)
    if isinstance(declared, str):
        raise TypeError(
            "label_fields must be a sequence of field names, not a single "
            f"string; got {declared!r} — declare ({declared!r},) to mean one "
            "field."
        )
    if declared is not None:
        return tuple(declared)
    if frozenset(scorer.signals) <= SUPPORTED_SIGNALS:
        return signal_fields(scorer.signals)
    return None


class InProcessTeacherScorer:
    """Score a batch with a teacher model loaded in the current process.

    The scorer owns the teacher's evaluation contract: it narrows
    ``active_outputs`` to the outputs the requested signals need, builds and
    afterwards restores whatever neighbor list the teacher requires — or, on
    request, consumes the one the batch already carries — picks the grad mode
    the teacher's autograd outputs need, detaches every result, and normalizes
    each signal to its canonical shape. The batch is left exactly as it was
    found, so a scorer can be called mid-training on a live batch.

    Each signal is a :class:`TeacherSignal` mapping one teacher output to one
    batch field at one level. The built-in ones (:data:`BUILTIN_SIGNALS`) are
    requested by name: ``energy`` to ``teacher_energy`` ``(B, 1)`` and
    ``stress`` to ``teacher_stress`` ``(B, 3, 3)`` at system level;
    ``forces`` to ``teacher_forces`` ``(V, 3)``, ``atomic_energies`` to
    ``teacher_atomic_energies`` ``(V,)``, and ``embeddings`` (from
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`) to
    ``teacher_node_embeddings`` ``(V, D)`` at node level. Any other teacher
    output is requested as a :class:`TeacherSignal` of its own. The resolved
    specs are published as ``signal_specs`` and the fields they write as
    ``label_fields``.

    Parameters
    ----------
    teacher : BaseModelMixin
        Model wrapper producing the signals. Placed in evaluation mode at
        construction and for the duration of every :meth:`label` call, with
        the mode it arrived in restored afterwards; its parameters and their
        ``requires_grad`` flags are never modified.
    signals : Iterable[str | TeacherSignal]
        Signals to produce: built-in names or :class:`TeacherSignal` specs,
        each naming an output in the teacher's declared ``outputs``.
    dtype : torch.dtype | None, optional
        Cast floating-point outputs to this dtype. Any floating-point dtype is
        accepted; whether a store can hold it is checked by
        :func:`~nvalchemi.training.distillation.labeling.label_dataset`, and a
        labeled store reads back at the reading dataset's ``positions`` dtype
        regardless. Default ``None`` (keep the teacher's dtype).
    neighbor_list : NeighborListPolicy, optional
        ``"rebuild"`` builds the teacher's own neighbor list for every call,
        hiding whatever list the batch carries, and rolls it back afterwards.
        ``"reuse"`` hands the teacher the batch's list instead: the batch must
        carry the keys the teacher's format reads, and a cutoff stamp, when
        the batch has one, must equal the teacher's; otherwise :meth:`label`
        raises rather than falling back. Whether the list holds each pair once
        or twice is not recorded on the batch, so a reused list must match the
        teacher's ``half_list`` by construction. Default ``"rebuild"``.

    Raises
    ------
    ValueError
        If *signals* is empty, names a signal that is neither built in nor a
        :class:`TeacherSignal`, gives two specs one name or one field, names a
        model output the teacher does not declare, gives a custom spec no
        model output, requests ``"embeddings"`` from a teacher that publishes
        no node-embedding shape, *dtype* is not a floating-point dtype,
        *neighbor_list* is neither ``"rebuild"`` nor ``"reuse"``, or *teacher*
        is a composition planning more than one neighbor-list source.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InProcessTeacherScorer
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> labels = scorer.label(batch)  # doctest: +SKIP
    >>> labels["teacher_forces"][1]  # doctest: +SKIP
    'node'

    A teacher output the built-in table does not cover is labeled through a
    spec of its own:

    >>> from nvalchemi.training.distillation import TeacherSignal
    >>> charges = TeacherSignal("charges", "charges", "teacher_charges", "node")
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", charges])  # doctest: +SKIP
    >>> scorer.label_fields  # doctest: +SKIP
    ('teacher_charges', 'teacher_energy')

    Notes
    -----
    Under the default ``neighbor_list="rebuild"`` the teacher's list is built
    for the forward pass and rolled back, and a list a composed pipeline keeps
    as an instance attribute, along with its captured per-source table, is
    hidden from the teacher for the whole of scoring. ``"reuse"`` is for the
    case where the student has already built the list the teacher needs, in
    the teacher's format and at its cutoff, and one build per step is one too
    many; the scorer then checks only what it cannot infer and refuses the
    batch by name when the list is missing or stamped with another cutoff. A
    teacher composition planning more than one neighbor-list source is refused
    at construction, because the scorer builds one list per batch; compose it
    to plan a single list instead (``neighbor_adaptation="always"`` or a large
    enough ``max_cutoff_ratio``).
    ``requires_grad`` on ``positions`` and the teacher's autograd inputs is
    restored after each call, as is every field a composed teacher writes onto
    the batch to wire one stage into the next.
    """

    def __init__(
        self,
        teacher: BaseModelMixin,
        signals: Iterable[str | TeacherSignal],
        *,
        dtype: torch.dtype | None = None,
        neighbor_list: NeighborListPolicy = "rebuild",
    ) -> None:
        """Validate the requested signals against the teacher's declared outputs."""
        requested = list(signals)
        if not requested:
            raise ValueError("At least one teacher signal must be requested; got [].")
        unsupported = sorted(
            {
                signal
                for signal in requested
                if not isinstance(signal, TeacherSignal)
                and signal not in BUILTIN_SIGNALS
            }
        )
        if unsupported:
            raise ValueError(
                f"Teacher signals must be names from {sorted(BUILTIN_SIGNALS)!r} or "
                f"TeacherSignal specs; got unsupported {unsupported!r}."
            )
        specs = dict(sorted(_resolve_signals(requested).items()))
        underived = sorted(
            name
            for name, spec in specs.items()
            if spec.model_output is None and name not in _DERIVED_SIGNALS
        )
        if underived:
            raise ValueError(
                f"Teacher signals {underived!r} name no model output, and the scorer "
                f"derives only {sorted(_DERIVED_SIGNALS)!r} without one; give each a "
                "model_output the teacher declares."
            )
        required = frozenset(
            spec.model_output
            for spec in specs.values()
            if spec.model_output is not None
        )
        declared = teacher.model_config.outputs
        missing = required - declared
        if missing:
            raise ValueError(
                f"Teacher cannot produce the outputs required by signals "
                f"{sorted(specs)!r}; got outputs={sorted(declared)!r}, "
                f"missing {sorted(missing)!r}."
            )
        if "embeddings" in specs and "node_embeddings" not in _node_embedding_shapes(
            teacher
        ):
            raise ValueError(
                "Teacher must publish a ``node_embeddings`` shape to serve the "
                f"``embeddings`` signal; got {sorted(_node_embedding_shapes(teacher))!r}."
            )
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point dtype; got {dtype!r}.")
        if neighbor_list not in _NEIGHBOR_LIST_POLICIES:
            raise ValueError(
                f"neighbor_list must be 'rebuild' or 'reuse'; got {neighbor_list!r}."
            )
        planned = _planned_neighbor_sources(teacher)
        if planned > 1:
            raise ValueError(
                f"Teacher plans {planned!r} neighbor-list sources, but a scorer "
                "builds one list per batch; compose the teacher to plan a single "
                'list with neighbor_adaptation="always" or a max_cutoff_ratio of '
                "at least its largest-to-smallest cutoff ratio."
            )

        self.teacher = teacher
        self.signals = frozenset(specs)
        self.signal_specs: Mapping[str, TeacherSignal] = MappingProxyType(specs)
        self.label_fields = signal_fields(specs.values())
        self.dtype = dtype
        self.neighbor_list = neighbor_list
        self._required_outputs = required
        evaluate = getattr(teacher, "eval", None)
        if callable(evaluate):
            evaluate()

    def label(self, batch: Batch) -> TeacherLabels:
        """Return the requested teacher signals for *batch*.

        Parameters
        ----------
        batch : Batch
            Batch to score. Restored to its incoming state before returning,
            including neighbor tensors, any pre-existing embeddings, and any
            field the teacher writes while scoring.

        Returns
        -------
        TeacherLabels
            Mapping from batch field name to ``(detached tensor, level)``.

        Raises
        ------
        RuntimeError
            If the teacher omits an output or embedding a requested signal
            needs.
        ValueError
            If ``neighbor_list="reuse"`` and *batch* carries no list the
            teacher's format reads, or one stamped with another cutoff.
        """
        config = self.teacher.model_config
        previous_active = set(config.active_outputs)
        grad_flags = _snapshot_grad_flags(batch, config)
        try:
            self.teacher.set_config("active_outputs", set(self._required_outputs))
            with (
                evaluating(self.teacher)
                if isinstance(self.teacher, torch.nn.Module)
                else nullcontext(),
                _isolated_neighbors(batch, config.neighbor_config, self.neighbor_list),
                _isolated_fields(batch),
            ):
                labels = self._forward_labels(batch) if self._required_outputs else {}
                if "embeddings" in self.signals:
                    labels.update(self._embedding_labels(batch))
        finally:
            self.teacher.set_config("active_outputs", previous_active)
            _restore_grad_flags(batch, grad_flags)
        return labels

    def _forward_labels(self, batch: Batch) -> TeacherLabels:
        """Run the teacher forward pass and collect its detached signals."""
        config = self.teacher.model_config
        grad_mode = (
            torch.enable_grad()
            if config.autograd_outputs & self._required_outputs
            else torch.no_grad()
        )
        with grad_mode:
            outputs = self.teacher(batch)
        labels: TeacherLabels = {}
        for spec in self.signal_specs.values():
            if spec.model_output is None:
                continue
            value = outputs.get(spec.model_output)
            if value is None:
                raise RuntimeError(
                    f"Teacher returned no {spec.model_output!r} output for the "
                    f"{spec.name!r} signal."
                )
            labels[spec.field] = (self._finalize(spec, value, batch), spec.level)
        del outputs
        return labels

    def _embedding_labels(self, batch: Batch) -> TeacherLabels:
        """Compute node embeddings without leaving them attached to *batch*.

        Pre-existing embeddings are cleared first, so the ``node_embeddings``
        read back is the teacher's rather than a stale value already on the
        batch, and are restored into the level they came from: ``del`` drops
        the key from every level, after which :meth:`Batch.__setitem__` would
        route it by the attribute registry rather than the incoming layout.
        """
        saved_levels = {
            key: level
            for level, fields in batch.level_keys.items()
            for key in _EMBEDDING_KEYS & fields
        }
        saved_values = {key: batch[key] for key in saved_levels}
        for key in saved_levels:
            del batch[key]
        saved_tracked = {
            level: names & _EMBEDDING_KEYS
            for level, names in (batch.keys or {}).items()
        }
        for level in saved_tracked:
            batch.keys[level] -= _EMBEDDING_KEYS
        try:
            with torch.no_grad():
                self.teacher.compute_embeddings(batch)
            if "node_embeddings" not in batch:
                raise RuntimeError(
                    "Teacher compute_embeddings() must write ``node_embeddings`` onto "
                    f"the batch; got {sorted(key for key in _EMBEDDING_KEYS if key in batch)!r}."
                )
            spec = self.signal_specs["embeddings"]
            value = self._finalize(spec, batch["node_embeddings"].clone(), batch)
            return {spec.field: (value, spec.level)}
        finally:
            for key in _EMBEDDING_KEYS:
                if key in batch:
                    del batch[key]
            for key, value in saved_values.items():
                _restore_at_level(batch, key, value, saved_levels[key])
            for level, names in saved_tracked.items():
                batch.keys[level] = (batch.keys[level] - _EMBEDDING_KEYS) | names

    def _finalize(
        self, spec: TeacherSignal, value: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        """Detach *value*, normalize it to *spec*'s canonical shape, and cast it."""
        value = value.detach()
        if spec.normalize is not None:
            value = spec.normalize(value, batch)
        if self.dtype is not None and value.is_floating_point():
            value = value.to(self.dtype)
        return value
