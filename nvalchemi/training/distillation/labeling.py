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
"""Offline labeling of a dataset with teacher signals."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    _get_cat_dim,
)
from nvalchemi.training.distillation._labels import (
    _attach_teacher_labels,
    _prune_empty_edges,
    _reject_foreign_fields,
)
from nvalchemi.training.distillation.scoring import (
    _DENSE_NEIGHBOR_KEYS,
    _NEIGHBOR_KEYS,
    _STORABLE_DTYPES,
    scorer_fields,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes.backends.zarr import StoreLike
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
    from nvalchemi.training.distillation.scoring import TeacherScorer

__all__ = ["label_dataset"]


_FieldSchema: TypeAlias = dict[str, tuple[str, torch.dtype]]
"""Store level and dtype of every field a labeled chunk persists."""

_STORE_LEVELS = {"node": "atom", "edge": "edge", "system": "system"}
"""Store level names for the batch levels a writer persists."""

_REPORTED_MISMATCHES = 4
"""Number of disagreeing store arrays named before an integrity error truncates."""


@dataclasses.dataclass(frozen=True)
class _StoreState:
    """Sample counts and field schema of an existing labeled store."""

    active: int
    total: int
    schema: _FieldSchema


def _torn_store_error(detail: str) -> ValueError:
    """Return the error raised for a store an interrupted run left inconsistent."""
    return ValueError(
        "Store is inconsistent, so a resumed run cannot line up with the dataset: "
        f"{detail}. This is what a labeling run interrupted mid-append leaves "
        "behind; truncate the store back to its committed samples or label into a "
        "fresh one."
    )


def _store_array(reader: AtomicDataZarrReader, field: str) -> Any | None:
    """Return the Zarr array backing *field*, or ``None`` when the store has none."""
    for group in ("core", "custom"):
        if group in reader._root and field in reader._root[group]:
            return reader._root[group][field]
    return None


def _check_store_integrity(reader: AtomicDataZarrReader) -> None:
    """Raise when a store's arrays disagree about how many samples it holds.

    ``AtomicDataZarrWriter.append`` extends the pointer arrays, then the masks,
    then every field array, and commits ``num_samples`` last, so a run
    interrupted anywhere in that sequence leaves arrays of different lengths.
    Resuming from the sample mask would then write every remaining sample at an
    offset the pointers do not agree with, which no later read reports. Only
    array metadata is inspected, never sample data.
    """
    committed = reader._root.attrs.get("num_samples")
    if committed is None:
        raise _torn_store_error("the store records no committed sample count")
    num_samples = int(committed)
    meta = reader._root["meta"]
    pointers = {"atoms_ptr": reader._atoms_ptr, "edges_ptr": reader._edges_ptr}
    for name, pointer in pointers.items():
        if int(pointer[0].item()) != 0 or bool((pointer[1:] < pointer[:-1]).any()):
            raise _torn_store_error(
                f"meta/{name} is not a non-decreasing pointer array starting at zero; "
                f"got {pointer.tolist()!r}"
            )
    totals = {
        "atom": int(reader._atoms_ptr[-1].item()),
        "edge": int(reader._edges_ptr[-1].item()),
        "system": num_samples,
    }
    lengths = {
        "meta/atoms_ptr": (int(reader._atoms_ptr.numel()), num_samples + 1),
        "meta/edges_ptr": (int(reader._edges_ptr.numel()), num_samples + 1),
        "meta/samples_mask": (int(reader._samples_mask.numel()), num_samples),
    }
    for name, expected in (("atoms_mask", "atom"), ("edges_mask", "edge")):
        if name in meta:
            lengths[f"meta/{name}"] = (int(meta[name].shape[0]), totals[expected])
    for field, level in reader.field_levels.items():
        array = _store_array(reader, field)
        if array is None:
            raise _torn_store_error(
                f"the store declares field {field!r} but holds no array for it"
            )
        cat_dim = _get_cat_dim(field) % len(array.shape)
        lengths[field] = (int(array.shape[cat_dim]), totals[level])
    mismatched = [
        f"{name} holds {found!r} rows where {expected!r} are committed"
        for name, (found, expected) in lengths.items()
        if found != expected
    ]
    if mismatched:
        reported = ", ".join(mismatched[:_REPORTED_MISMATCHES])
        remaining = len(mismatched) - _REPORTED_MISMATCHES
        raise _torn_store_error(
            f"{num_samples!r} samples are committed but {reported}"
            + (f", and {remaining!r} further arrays disagree" if remaining > 0 else "")
        )


def _store_schema(reader: AtomicDataZarrReader) -> _FieldSchema:
    """Return the level and dtype of every field an existing store holds.

    Runs after :func:`_check_store_integrity`, so every declared field is known
    to have an array. Dtypes come from an empty slice, which reads no chunk.
    """
    schema: _FieldSchema = {}
    for field, level in reader.field_levels.items():
        array = _store_array(reader, field)
        schema[field] = (level, torch.from_numpy(array[:0]).dtype)
    return schema


def _existing_store_state(store: StoreLike) -> _StoreState | None:
    """Return the state of *store*, or ``None`` when it cannot be read."""
    try:
        reader = AtomicDataZarrReader(store)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    try:
        _check_store_integrity(reader)
        return _StoreState(
            active=len(reader),
            total=int(reader._samples_mask.numel()),
            schema=_store_schema(reader),
        )
    finally:
        reader.close()


def _batch_schema(batch: Batch) -> _FieldSchema:
    """Return the level and dtype of every field a writer would persist for *batch*."""
    schema: _FieldSchema = {}
    for level, names in (batch.keys or {}).items():
        for name in names:
            if name in batch:
                schema[name] = (_STORE_LEVELS.get(level, level), batch[name].dtype)
    return schema


def _check_chunk_schema(
    reference: _FieldSchema, outgoing: _FieldSchema, indices: Sequence[int]
) -> None:
    """Raise when a chunk would write a different schema than the store holds.

    ``AtomicDataZarrWriter.append`` extends only the arrays a store already
    holds and silently ignores everything else, so a chunk whose fields drift
    from the store's would leave arrays at different lengths rather than fail,
    and one whose dtypes drift would have its labels quietly cast.
    """
    chunk = f"the chunk covering samples {indices[0]!r}-{indices[-1]!r}"
    extra = sorted(set(outgoing) - set(reference))
    missing = sorted(set(reference) - set(outgoing))
    if extra or missing:
        raise ValueError(
            "Every labeled chunk must write the fields the store holds; "
            f"{chunk} writes extra {extra!r} and is missing {missing!r}."
        )
    drifted = ", ".join(
        f"{name} is stored as {reference[name]!r} but arrives as {outgoing[name]!r}"
        for name in sorted(reference)
        if reference[name] != outgoing[name]
    )
    if drifted:
        raise ValueError(
            "Every labeled chunk must write the levels and dtypes the store holds; "
            f"in {chunk}, {drifted}."
        )


def _check_storable_dtypes(outgoing: _FieldSchema) -> None:
    """Raise when a chunk carries a floating-point dtype no store can hold.

    Storability is the store's rule rather than the scorer's, so a scorer is
    free to label in whatever floating-point precision the student trains in.
    Only the chunk that defines a fresh store's schema is checked, because
    every later chunk is already held to that schema.
    """
    unstorable = ", ".join(
        f"{name} arrives as {dtype!r}"
        for name, (_, dtype) in sorted(outgoing.items())
        if dtype.is_floating_point and dtype not in _STORABLE_DTYPES
    )
    if unstorable:
        raise ValueError(
            "Every floating-point field must arrive in a dtype an ALCHEMI Zarr "
            f"store can hold; {unstorable}, and the storable dtypes are "
            f"{list(_STORABLE_DTYPES)!r}."
        )


def _strip_unstorable(
    batch: Batch, keep: frozenset[str], ephemeral: frozenset[str]
) -> None:
    """Drop fields that must not reach the store, keeping *keep* intact.

    Removes *ephemeral*, the neighbor tensors this run rebuilds rather than
    stores, plus anything else that appeared on *batch* during labeling. An
    edge group left with no fields is dropped as well, so the store's edge
    pointers do not record edges that no array backs.
    """
    for key in ephemeral | (frozenset(_batch_schema(batch)) - keep):
        if key in batch:
            del batch[key]
    _prune_empty_edges(batch)


def label_dataset(
    dataset: BatchDatasetProtocol,
    scorer: TeacherScorer,
    store: StoreLike,
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
    resume: bool = True,
    keep_neighbors: bool = False,
) -> int:
    """Label *dataset* with teacher signals and persist the result to *store*.

    Walks *dataset* in contiguous chunks of *batch_size* samples, scores each
    chunk with *scorer*, attaches every returned signal to the chunk as a
    batch field, and writes the augmented chunk to a Zarr store. The store
    ends up holding the original fields plus the teacher fields, so the
    labeled dataset is read back through the ordinary
    :class:`~nvalchemi.data.datapipes.backends.zarr.AtomicDataZarrReader` /
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` path.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Source dataset. Only ``__len__`` and ``load_batches`` are used, so
        both :class:`~nvalchemi.data.datapipes.dataset.Dataset` and
        :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
        qualify.
    scorer : TeacherScorer
        Scorer producing the teacher signals for each chunk.
    store : StoreLike
        Destination Zarr store: a path, a zarr store instance, or a dict.
    batch_size : int, optional
        Number of samples scored per forward pass. Default ``32``.
    device : torch.device | str | None, optional
        Device to move each chunk to before scoring. Default ``None``
        (score on whatever device the dataset emits).
    resume : bool, optional
        If ``True`` (default), an existing store is treated as a partial run:
        the first ``len(store)`` samples are skipped and labeling continues
        from there. If ``False``, an existing store is an error.
    keep_neighbors : bool, optional
        If ``False`` (default), a source neighbor list is dropped rather than
        stored: the cutoff a list was built at lives on the batch and not in
        the store, so a reloaded list is one no consumer can check. Set
        ``True`` to carry a sparse (``COO``) source list over anyway; the dense
        tensors are dropped either way. Default ``False``.

    Returns
    -------
    int
        Number of samples labeled by this call. ``0`` when a resumed store
        already covers the whole dataset.

    Raises
    ------
    ValueError
        If *batch_size* is not positive, *scorer* declares or returns a batch
        field outside the ``teacher_*`` namespace, *store* exists but cannot be
        read as an ALCHEMI Zarr store, *resume* is ``False`` and *store*
        exists, *store* holds soft-deleted samples, *store* holds arrays that
        disagree about how many samples it contains, a chunk carries a
        floating-point field in a dtype a store cannot hold, or a chunk would
        write a different field set, level, or dtype than the store holds.
    TypeError
        If *scorer* declares ``label_fields`` as a single string.

    Examples
    --------
    >>> from nvalchemi.training.distillation import label_dataset
    >>> scorer = InProcessTeacherScorer(teacher, ["energy", "forces"])  # doctest: +SKIP
    >>> label_dataset(dataset, scorer, "labeled.zarr", batch_size=64)  # doctest: +SKIP
    1024

    Notes
    -----
    The first write defines the store schema: every field present on the first
    chunk — teacher fields included — becomes a store array, and later chunks
    only extend arrays that already exist. Every chunk is therefore checked
    against that schema, on fresh and resumed runs alike, and one whose fields,
    levels, or dtypes differ is rejected instead of silently misaligning arrays
    or casting labels into the stored precision. Resuming also counts on stored
    sample *i* being dataset sample *i*, which soft-deleted samples break, so a
    store with deletions is rejected rather than continued from the wrong
    offset, and on the store's arrays agreeing about how many samples it holds,
    which an interrupted append breaks: a store whose pointers, masks, and field
    arrays disagree with its committed sample count is rejected rather than
    resumed from an offset that would misplace every remaining sample.

    Labels are attached with ``overwrite=True``, so a scorer writing outside
    the ``teacher_*`` namespace would replace the reference field of that name
    — the very label the student is trained against — and persist the
    replacement. A scorer's declared ``label_fields`` is therefore held to the
    namespace before any chunk is written, and the fields it actually returns
    are held to it again per chunk, which is what polices a scorer declaring
    nothing.

    Every source field is carried over except the neighbor tensors. The dense
    ones (``neighbor_matrix``, ``num_neighbors``, ``neighbor_matrix_shifts``)
    cannot append into a fixed-width store array at all, and a sparse list is
    dropped because the cutoff it was built at is a batch attribute the store
    does not hold: a reloaded list is one nothing downstream can check, which
    is why the scorer rebuilds rather than trust one. Build the student's list
    from the stored positions with a
    :class:`~nvalchemi.hooks.NeighborListHook` at ``BEFORE_FORWARD``, or pass
    ``keep_neighbors=True`` to store the sparse list anyway. A store written
    under one ``keep_neighbors`` setting and resumed under the other is refused
    by the per-chunk schema check like any other field-set drift.

    This store is the consumption path for training on teacher labels: point a
    reader at it and the teacher fields arrive alongside the reference labels,
    at the levels recorded here — but not necessarily in the dtype recorded
    here. A dataset coerces every floating-point field it reads to the dtype of
    its own ``positions``
    (:meth:`~nvalchemi.data.AtomicData.check_fp_dtype_consistency`), so labels
    stored as float16 or float64 come back at the reading dataset's precision.
    The stored dtype governs what the store costs, not what training sees.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}.")

    declared = scorer_fields(scorer)
    if declared is not None:
        _reject_foreign_fields(declared, "A scorer's label_fields")

    state = _existing_store_state(store)
    if state is None and isinstance(store, (str, Path)) and Path(store).exists():
        raise ValueError(
            "Store path exists but is not a readable ALCHEMI Zarr store; got "
            f"{store!s}."
        )
    if state is not None and not resume:
        raise ValueError(
            f"Store already exists with {state.active!r} samples and resume is False; "
            "pass resume=True to continue labeling or write to a fresh store."
        )
    if state is not None and state.active != state.total:
        raise ValueError(
            f"Store holds {state.total - state.active!r} soft-deleted samples, so a "
            "resumed run cannot line up with the dataset; defragment the store or "
            "label into a fresh one."
        )

    total = len(dataset)
    start = state.active if state is not None else 0
    schema = state.schema if state is not None else None
    if start >= total:
        return 0

    writer = AtomicDataZarrWriter(store)
    ephemeral = _DENSE_NEIGHBOR_KEYS if keep_neighbors else _NEIGHBOR_KEYS
    labeled = 0
    for begin in range(start, total, batch_size):
        indices = list(range(begin, min(begin + batch_size, total)))
        batch = dataset.load_batches([indices])[0]
        if device is not None:
            batch = batch.to(device)
        loaded_fields = frozenset(_batch_schema(batch))
        labels = scorer.label(batch)
        _reject_foreign_fields(labels, "Teacher labels")
        _attach_teacher_labels(batch, labels)
        _strip_unstorable(batch, loaded_fields | frozenset(labels), ephemeral)
        outgoing = _batch_schema(batch)
        if schema is None:
            _check_storable_dtypes(outgoing)
            writer.write(batch)
            schema = outgoing
        else:
            _check_chunk_schema(schema, outgoing, indices)
            writer.append(batch)
        labeled += batch.num_graphs
    return labeled
