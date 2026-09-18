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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import torch

from nvalchemi._typing import Energy, Forces, NodePositions
from nvalchemi.models.base import ModelConfig, NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.models.base import BaseModelMixin

__all__ = [
    "InProcessTeacherScorer",
    "SUPPORTED_SIGNALS",
    "SignalLevel",
    "TeacherLabels",
    "TeacherScorer",
    "hessian_vector_product",
    "scorer_fields",
    "signal_fields",
    "signal_for_field",
]

SignalLevel: TypeAlias = Literal["node", "system"]
"""Batch level a teacher signal is attached at."""

TeacherLabels: TypeAlias = dict[str, tuple[torch.Tensor, SignalLevel]]
"""Teacher signals for one batch, keyed by the batch field they populate."""


@dataclasses.dataclass(frozen=True)
class _SignalSpec:
    """Model output, batch field, and level backing one teacher signal."""

    model_output: str | None
    field: str
    level: SignalLevel
    extra_fields: tuple[str, ...] = ()


_HVP_PROBE_FIELD = "teacher_hvp_probe"
"""Field holding the direction a stored Hessian-vector product was taken along."""

_SIGNAL_SPECS: dict[str, _SignalSpec] = {
    "energy": _SignalSpec("energy", "teacher_energy", "system"),
    "forces": _SignalSpec("forces", "teacher_forces", "node"),
    "stress": _SignalSpec("stress", "teacher_stress", "system"),
    "atomic_energies": _SignalSpec(
        "atomic_energies", "teacher_atomic_energies", "node"
    ),
    "embeddings": _SignalSpec(None, "teacher_node_embeddings", "node"),
    "hessian": _SignalSpec(
        None, "teacher_hvp", "node", extra_fields=(_HVP_PROBE_FIELD,)
    ),
}
"""Supported teacher signals, keyed by signal name."""

SUPPORTED_SIGNALS: frozenset[str] = frozenset(_SIGNAL_SPECS)
"""Teacher signal names :class:`InProcessTeacherScorer` can produce."""

_TEACHER_FIELD_PREFIX = "teacher_"
"""Namespace every teacher field lives in, clear of a batch's own fields."""

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

_CUTOFF_TOLERANCE = 1e-6
"""Absolute tolerance when matching a pre-built neighbor list to a cutoff."""

_CUTOFF_ATTR = "_neighbor_list_cutoff"
"""Batch attribute recording the cutoff a neighbor list was built at."""

_HALF_LIST_ATTR = "_neighbor_list_half"
"""Batch attribute recording whether a neighbor list holds each pair once."""

_PIPELINE_SOURCES_ATTR = "_pipeline_neighbor_sources"
"""Instance-dict attribute holding a composed pipeline's per-source neighbor lists."""

_SHADOWED_NEIGHBOR_ATTRS = _NEIGHBOR_KEYS | {
    "edge_ptr",
    _CUTOFF_ATTR,
    _HALF_LIST_ATTR,
}
"""Instance-dict neighbor attributes snapshotted and restored around a rebuild."""


def signal_fields(signals: Iterable[str]) -> tuple[str, ...]:
    """Return every batch field the named signals populate, sorted.

    A signal may populate companion fields alongside its own; all are reported
    so a consumer can prepare for the whole set a scorer will write.

    Parameters
    ----------
    signals : Iterable[str]
        Signal names, each of which must be in :data:`SUPPORTED_SIGNALS`.

    Returns
    -------
    tuple[str, ...]
        Batch field names, deduplicated and sorted.

    Raises
    ------
    KeyError
        If a name is not a supported signal.
    """
    fields: set[str] = set()
    for name in signals:
        spec = _SIGNAL_SPECS.get(name)
        if spec is None:
            raise KeyError(
                f"Unknown teacher signal {name!r}; supported signals are "
                f"{sorted(SUPPORTED_SIGNALS)!r}."
            )
        fields.add(spec.field)
        fields.update(spec.extra_fields)
    return tuple(sorted(fields))


def signal_for_field(field: str) -> str | None:
    """Return the signal that populates *field*, or ``None`` when no signal does.

    Parameters
    ----------
    field : str
        Batch field name to resolve back to the signal that writes it.

    Returns
    -------
    str | None
        Name of the signal populating *field*, or ``None`` when *field* is not
        one a supported signal writes.
    """
    for name, spec in _SIGNAL_SPECS.items():
        if field == spec.field or field in spec.extra_fields:
            return name
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


def _normalize_signal_shape(signal: str, value: torch.Tensor) -> torch.Tensor:
    """Reshape a raw teacher output to the canonical shape for *signal*."""
    match signal:
        case "energy":
            return value.unsqueeze(-1) if value.ndim == 1 else value
        case "atomic_energies":
            return value.reshape(-1)
        case "stress":
            return value.reshape(-1, 3, 3)
        case _:
            return value


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


def _matches_neighbor_config(batch: Batch, config: NeighborConfig) -> bool:
    """Return whether *batch* already carries a list the teacher can consume."""
    if config.half_list or getattr(batch, _HALF_LIST_ATTR, None) is not False:
        return False
    cutoff = getattr(batch, _CUTOFF_ATTR, None)
    if cutoff is None or abs(float(cutoff) - config.cutoff) > _CUTOFF_TOLERANCE:
        return False
    required = (
        ("neighbor_list",)
        if config.format == NeighborListFormat.COO
        else ("neighbor_matrix", "num_neighbors")
    )
    # The provenance stamps live in the instance dict, so while a shadowed list
    # is present they describe that list rather than anything in storage.
    if _NEIGHBOR_KEYS & batch.__dict__.keys():
        return all(key in batch.__dict__ for key in required)
    return all(key in batch for key in required)


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
def _evaluating(teacher: BaseModelMixin) -> Iterator[None]:
    """Score with *teacher* in evaluation mode, restoring the mode it arrived in.

    A teacher put back in training mode after construction would otherwise
    sample dropout and update batch-norm statistics while it scores. A teacher
    that is not an :class:`~torch.nn.Module` has no mode and is left alone.
    """
    evaluate = getattr(teacher, "eval", None)
    restore = getattr(teacher, "train", None)
    training = bool(getattr(teacher, "training", False)) and callable(evaluate)
    if training:
        evaluate()
    try:
        yield
    finally:
        if training and callable(restore):
            restore()


@contextmanager
def _isolated_neighbors(batch: Batch, config: NeighborConfig | None) -> Iterator[None]:
    """Build the teacher's neighbor list on *batch*, restoring prior state on exit.

    A pre-built list is reused only when it is a known full list at the
    teacher's cutoff and format: the core stamps a list's cutoff
    (``_neighbor_list_cutoff``) but not its half-list provenance, so a
    ``half_list=True`` teacher and any batch without a ``_neighbor_list_half``
    stamp get a rebuild. Every list built here is stamped, and a caller holding
    a full list may stamp it ``False`` to opt into reuse.

    A rebuild snapshots the node-level neighbor tensors, the edge group, and
    every neighbor attribute in the batch's instance dictionary — where a
    composed pipeline shadows its default source's list — so the teacher can
    resolve nothing but the list built here, and restores all of it afterwards.
    The per-source table a composed pipeline captures under
    ``_pipeline_neighbor_sources`` is hidden for the whole block, reuse
    included, because a composed teacher consults it before anything canonical.

    Parameters
    ----------
    batch : Batch
        Batch to build neighbors on; mutated for the duration of the block.
    config : NeighborConfig | None
        Neighbor requirements of the teacher, or ``None`` for a model that
        needs no neighbor list.

    Yields
    ------
    None
    """
    saved_sources = (
        {_PIPELINE_SOURCES_ATTR: batch.__dict__.pop(_PIPELINE_SOURCES_ATTR)}
        if _PIPELINE_SOURCES_ATTR in batch.__dict__
        else {}
    )
    try:
        if config is None or _matches_neighbor_config(batch, config):
            yield
            return

        atoms = batch._atoms_group
        saved_nodes = (
            {key: atoms[key] for key in _NEIGHBOR_KEYS if key in atoms}
            if atoms is not None
            else {}
        )
        saved_edges = batch._storage.groups.pop("edges", None)
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
            setattr(batch, _HALF_LIST_ATTR, config.half_list)
            yield
        finally:
            if atoms is not None:
                for key in _NEIGHBOR_KEYS:
                    if key in atoms:
                        del atoms[key]
                for key, value in saved_nodes.items():
                    atoms[key] = value
            if saved_edges is None:
                batch._storage.groups.pop("edges", None)
            else:
                batch._storage.groups["edges"] = saved_edges
            for name in _SHADOWED_NEIGHBOR_ATTRS:
                batch.__dict__.pop(name, None)
            batch.__dict__.update(saved_shadows)
    finally:
        batch.__dict__.update(saved_sources)


@contextmanager
def _isolated_embeddings(batch: Batch) -> Iterator[None]:
    """Clear the embedding fields of *batch*, restoring them on exit.

    Pre-existing embeddings are cleared first, so the ``node_embeddings`` read
    back inside the block is the model's rather than a stale value already on
    the batch, and are restored into the group they came from: ``del`` drops
    the key from every group, after which :meth:`Batch.__setitem__` would route
    it by the attribute registry rather than the incoming layout. Tensors read
    inside the block outlive it, autograd graph included.

    Parameters
    ----------
    batch : Batch
        Batch whose embedding fields are cleared for the duration of the block.

    Yields
    ------
    None
    """
    saved_groups = {}
    for key in _EMBEDDING_KEYS:
        group = batch._storage.group_from_attr(key)
        if group is not None:
            saved_groups[key] = (group, group[key])
            del batch[key]
    saved_tracked = {
        level: names & _EMBEDDING_KEYS for level, names in (batch.keys or {}).items()
    }
    for level in saved_tracked:
        batch.keys[level] -= _EMBEDDING_KEYS
    try:
        yield
    finally:
        for key in _EMBEDDING_KEYS:
            if key in batch:
                del batch[key]
        for key, (group, value) in saved_groups.items():
            group[key] = value
        for level, names in saved_tracked.items():
            batch.keys[level] = (batch.keys[level] - _EMBEDDING_KEYS) | names


def hessian_vector_product(
    energy: Energy,
    positions: NodePositions,
    probe: NodePositions,
    *,
    create_graph: bool = False,
) -> Forces:
    r"""Return the product of an energy's position Hessian with a probe vector.

    The Hessian of a batch is block-diagonal over its graphs, so one
    double-backward pass over the summed energy returns the per-graph products
    stacked into one ``(V, 3)`` tensor, at the cost of two backward passes
    rather than :math:`3V`:

    .. math::

        (\mathbf{H}\mathbf{v})_{ia} =
        \sum_{b\beta} \frac{\partial^2 E}{\partial r_{ia} \partial r_{ib\beta}}
        v_{ib\beta}
        = \frac{\partial}{\partial r_{ia}}
        \left( \nabla_{\mathbf{r}} E \cdot \mathbf{v} \right).

    Both the teacher's label and the student's prediction go through this
    function, so the two are the same estimator of the same quantity.

    Parameters
    ----------
    energy : Energy
        Energy of shape ``(B, 1)``, carrying an autograd graph back to
        *positions*.
    positions : NodePositions
        Positions of shape ``(V, 3)`` the energy is differentiated with respect
        to, with ``requires_grad`` enabled.
    probe : NodePositions
        Probe direction of shape ``(V, 3)``.
    create_graph : bool, optional
        Whether the returned product stays attached to the graph, which a
        student prediction needs and a teacher label does not. Default
        ``False``.

    Returns
    -------
    Forces
        Hessian-vector product of shape ``(V, 3)``, in force units per length.

    Raises
    ------
    RuntimeError
        If the energy does not carry an autograd graph back to *positions*, or
        if the model is not twice differentiable.

    Notes
    -----
    A gradient the positions do not enter — a linear energy's — has a zero
    Hessian, and the product is returned as zeros rather than refused; with
    ``create_graph`` it stays attached to the energy's graph so a loss can
    still backpropagate through it. The Hessian is the curvature of the
    *energy*, so a direct-force teacher contributes curvature its own force
    head need not agree with.
    """
    try:
        gradient = torch.autograd.grad(energy.sum(), positions, create_graph=True)[0]
        product = (
            torch.autograd.grad(
                (gradient * probe).sum(),
                positions,
                create_graph=create_graph,
                allow_unused=True,
            )[0]
            if gradient.requires_grad
            else None
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Hessian-vector products differentiate the energy twice with respect "
            "to positions, so the model must be twice differentiable and its "
            "energy must carry an autograd graph back to positions with "
            f"requires_grad enabled; got {exc}."
        ) from exc
    if product is None:
        product = torch.zeros_like(positions)
        return product + 0.0 * energy.sum() if create_graph else product
    return product


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
    as writing every field that signal populates, so one that writes fewer
    must declare ``label_fields``. The protocol will not grow required
    members.

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
    whose signals are all in :data:`SUPPORTED_SIGNALS` gets
    :func:`signal_fields` of them, companion fields included; otherwise the
    fields are unknown, since a custom signal may map onto any field.

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
    afterwards restores whatever neighbor list the teacher requires, picks the
    grad mode the teacher's autograd outputs need, detaches every result, and
    normalizes each signal to its canonical shape. The batch is left exactly
    as it was found, so a scorer can be called mid-training on a live batch.

    Each signal maps to one batch field at one level: ``energy`` to
    ``teacher_energy`` ``(B, 1)`` and ``stress`` to ``teacher_stress``
    ``(B, 3, 3)`` at system level; ``forces`` to ``teacher_forces`` ``(V, 3)``,
    ``atomic_energies`` to ``teacher_atomic_energies`` ``(V,)``, ``embeddings``
    (from :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`) to
    ``teacher_node_embeddings`` ``(V, D)``, and ``hessian`` (from
    :meth:`label_hvp`) to ``teacher_hvp`` ``(V, 3)`` at node level, beside the
    probe direction it was taken along in ``teacher_hvp_probe``. The fields are
    published as ``label_fields``.

    Parameters
    ----------
    teacher : BaseModelMixin
        Model wrapper producing the signals. Placed in evaluation mode at
        construction and for the duration of every :meth:`label` call, with
        the mode it arrived in restored afterwards; its parameters and their
        ``requires_grad`` flags are never modified.
    signals : Iterable[str]
        Signal names to produce, each in :data:`SUPPORTED_SIGNALS`.
    dtype : torch.dtype | None, optional
        Cast floating-point outputs to this dtype. Any floating-point dtype is
        accepted; whether a store can hold it is checked by
        :func:`~nvalchemi.training.distillation.labeling.label_dataset`, and a
        labeled store reads back at the reading dataset's ``positions`` dtype
        regardless. Default ``None`` (keep the teacher's dtype).
    probe_seed : int | None, optional
        Seed of the generator the ``hessian`` probe direction is drawn from.
        Default ``None`` (draw from the global RNG, a fresh direction per
        labeling). Reassignable between calls; see the Notes.

    Raises
    ------
    ValueError
        If *signals* is empty, names an unsupported signal, requires a model
        output the teacher does not declare, requests ``"embeddings"`` from a
        teacher that publishes no node-embedding shape, requests ``"hessian"``
        from a teacher that declares no ``energy`` output, *dtype* is not a
        floating-point dtype, or *teacher* is a composition planning more than
        one neighbor-list source.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InProcessTeacherScorer
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> labels = scorer.label(batch)  # doctest: +SKIP
    >>> labels["teacher_forces"][1]  # doctest: +SKIP
    'node'

    Notes
    -----
    A pre-built neighbor list is reused only when it is a known full list at
    the teacher's cutoff and format; anything else is rebuilt for the forward
    pass and rolled back, and a list a composed pipeline keeps as an instance
    attribute, along with its captured per-source table, is hidden from the
    teacher for the whole of scoring. A teacher composition planning more than
    one neighbor-list source is refused at construction, because the scorer
    builds one list per batch; compose it to plan a single list instead
    (``neighbor_adaptation="always"`` or a large enough ``max_cutoff_ratio``).
    ``requires_grad`` on ``positions`` and the teacher's autograd inputs is
    restored after each call.

    Forward-pass signals share one teacher pass; ``embeddings`` adds a second,
    and ``hessian`` an energy-only pass plus two backward passes, so a Hessian
    label costs roughly three to four times an energy-and-force one. A redrawn
    probe is a new objective: leave ``probe_seed`` unset wherever coverage of
    the Hessian comes from redrawing (training, offline labeling) and pin it
    where a number is compared across passes, as
    :class:`~nvalchemi.training.distillation.DistillationStrategy` does per
    validation batch.
    """

    def __init__(
        self,
        teacher: BaseModelMixin,
        signals: Iterable[str],
        *,
        dtype: torch.dtype | None = None,
        probe_seed: int | None = None,
    ) -> None:
        """Validate the requested signals against the teacher's declared outputs."""
        requested = frozenset(signals)
        if not requested:
            raise ValueError(
                f"At least one teacher signal must be requested; got {sorted(requested)!r}."
            )
        unsupported = requested - SUPPORTED_SIGNALS
        if unsupported:
            raise ValueError(
                f"Teacher signals must be names from {sorted(SUPPORTED_SIGNALS)!r}; "
                f"got unsupported {sorted(unsupported)!r}."
            )
        required = frozenset(
            spec.model_output
            for spec in (_SIGNAL_SPECS[name] for name in requested)
            if spec.model_output is not None
        )
        declared = teacher.model_config.outputs
        missing = required - declared
        if missing:
            raise ValueError(
                f"Teacher cannot produce the outputs required by signals "
                f"{sorted(requested)!r}; got outputs={sorted(declared)!r}, "
                f"missing {sorted(missing)!r}."
            )
        if (
            "embeddings" in requested
            and "node_embeddings" not in _node_embedding_shapes(teacher)
        ):
            raise ValueError(
                "Teacher must publish a ``node_embeddings`` shape to serve the "
                f"``embeddings`` signal; got {sorted(_node_embedding_shapes(teacher))!r}."
            )
        if "hessian" in requested and "energy" not in declared:
            raise ValueError(
                "The ``hessian`` signal differentiates the teacher's energy "
                "twice, so the teacher must declare an ``energy`` output; got "
                f"outputs={sorted(declared)!r}."
            )
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point dtype; got {dtype!r}.")
        planned = _planned_neighbor_sources(teacher)
        if planned > 1:
            raise ValueError(
                f"Teacher plans {planned!r} neighbor-list sources, but a scorer "
                "builds one list per batch; compose the teacher to plan a single "
                'list with neighbor_adaptation="always" or a max_cutoff_ratio of '
                "at least its largest-to-smallest cutoff ratio."
            )

        self.teacher = teacher
        self.signals = requested
        self.label_fields = signal_fields(requested)
        self.dtype = dtype
        self.probe_seed = probe_seed
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
            including neighbor tensors and any pre-existing embeddings.

        Returns
        -------
        TeacherLabels
            Mapping from batch field name to ``(detached tensor, level)``.

        Raises
        ------
        RuntimeError
            If the teacher omits an output or embedding a requested signal
            needs.
        """
        config = self.teacher.model_config
        previous_active = set(config.active_outputs)
        grad_flags = _snapshot_grad_flags(batch, config)
        try:
            self.teacher.set_config("active_outputs", set(self._required_outputs))
            with (
                _evaluating(self.teacher),
                _isolated_neighbors(batch, config.neighbor_config),
            ):
                labels = self._forward_labels(batch) if self._required_outputs else {}
                if "embeddings" in self.signals:
                    labels.update(self._embedding_labels(batch))
                if "hessian" in self.signals:
                    labels.update(self._hessian_labels(batch))
        finally:
            self.teacher.set_config("active_outputs", previous_active)
            _restore_grad_flags(batch, grad_flags)
        return labels

    def label_hvp(self, batch: Batch, probe: NodePositions) -> Forces:
        """Return the teacher's Hessian-vector product along *probe*.

        The teacher's energy is differentiated twice with respect to the
        positions of *batch*, on a pass narrowed to the energy under the same
        neighbor-list isolation as :meth:`label`; the batch is left as it was
        found.

        Parameters
        ----------
        batch : Batch
            Batch to differentiate the teacher's energy on.
        probe : NodePositions
            Probe direction of shape ``(V, 3)``, matching the batch's positions.

        Returns
        -------
        Forces
            Detached Hessian-vector product of shape ``(V, 3)``, cast to
            ``dtype`` when one is configured.

        Raises
        ------
        RuntimeError
            If the teacher returns no energy, or is not twice differentiable
            with respect to positions.

        Examples
        --------
        >>> import torch
        >>> from nvalchemi.training.distillation import InProcessTeacherScorer
        >>> scorer = InProcessTeacherScorer(teacher, ["hessian"])  # doctest: +SKIP
        >>> probe = torch.randn_like(batch.positions)  # doctest: +SKIP
        >>> scorer.label_hvp(batch, probe).shape  # doctest: +SKIP
        torch.Size([12, 3])

        Notes
        -----
        One product is one forward and two backward passes; a Hutchinson
        average over ``k`` probes is ``k`` calls, left to the caller since the
        loss consumes one materialized target per batch.
        """
        config = self.teacher.model_config
        previous_active = set(config.active_outputs)
        grad_flags = _snapshot_grad_flags(batch, config)
        try:
            self.teacher.set_config("active_outputs", {"energy"})
            with _isolated_neighbors(batch, config.neighbor_config):
                positions = batch.positions
                with torch.enable_grad():
                    positions.requires_grad_(True)
                    energy = self.teacher(batch).get("energy")
                    if energy is None:
                        raise RuntimeError(
                            "Teacher returned no 'energy' output for the "
                            "'hessian' signal."
                        )
                    value = hessian_vector_product(energy, positions, probe)
        finally:
            self.teacher.set_config("active_outputs", previous_active)
            _restore_grad_flags(batch, grad_flags)
        return self._finalize("hessian", value)

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
        for name in sorted(self.signals):
            spec = _SIGNAL_SPECS[name]
            if spec.model_output is None:
                continue
            value = outputs.get(spec.model_output)
            if value is None:
                raise RuntimeError(
                    f"Teacher returned no {spec.model_output!r} output for the "
                    f"{name!r} signal."
                )
            labels[spec.field] = (self._finalize(name, value), spec.level)
        del outputs
        return labels

    def _embedding_labels(self, batch: Batch) -> TeacherLabels:
        """Compute node embeddings without leaving them attached to *batch*."""
        with _isolated_embeddings(batch):
            with torch.no_grad():
                self.teacher.compute_embeddings(batch)
            if "node_embeddings" not in batch:
                raise RuntimeError(
                    "Teacher compute_embeddings() must write ``node_embeddings`` onto "
                    f"the batch; got {sorted(key for key in _EMBEDDING_KEYS if key in batch)!r}."
                )
            spec = _SIGNAL_SPECS["embeddings"]
            value = self._finalize("embeddings", batch["node_embeddings"].clone())
        return {spec.field: (value, spec.level)}

    def _hessian_labels(self, batch: Batch) -> TeacherLabels:
        """Draw a probe and return the teacher's product with it, probe included.

        The probe is standard normal on the batch's device and dtype, from the
        global stream unless ``probe_seed`` names one of its own; it travels
        with the product because the loss compares two products taken along
        one direction.
        """
        spec = _SIGNAL_SPECS["hessian"]
        probe = self._draw_probe(batch.positions)
        value = self.label_hvp(batch, probe)
        return {
            spec.field: (value, spec.level),
            _HVP_PROBE_FIELD: (self._finalize("hessian", probe), spec.level),
        }

    def _draw_probe(self, positions: NodePositions) -> NodePositions:
        """Return a standard-normal direction shaped like *positions*."""
        if self.probe_seed is None:
            return torch.randn_like(positions)
        generator = torch.Generator(device=positions.device)
        generator.manual_seed(self.probe_seed)
        return torch.randn(
            positions.shape,
            generator=generator,
            dtype=positions.dtype,
            device=positions.device,
        )

    def _finalize(self, signal: str, value: torch.Tensor) -> torch.Tensor:
        """Detach *value*, normalize it to the canonical shape, and cast it."""
        value = _normalize_signal_shape(signal, value.detach())
        if self.dtype is not None and value.is_floating_point():
            value = value.to(self.dtype)
        return value
