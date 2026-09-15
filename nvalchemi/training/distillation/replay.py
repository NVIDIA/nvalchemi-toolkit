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
"""Replay buffer of generated frames and the reference/replay mixing loader."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch

from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.data.datapipes.samplers import MultiDatasetBatchSampler

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol

__all__ = ["ReplayBuffer", "ReplayEviction", "build_mixed_loader"]

ReplayEviction: TypeAlias = Literal["fifo", "uncertainty"]
"""Policy choosing which frames leave a replay buffer that is over capacity."""

_GROUP_LEVELS = {"atoms": "node", "edges": "edge", "system": "system"}
"""Batch level each storage group holds, used to report a schema mismatch."""

_SCHEMA_REMEDY = (
    "Label the reference dataset with label_dataset, requesting the signals the "
    "propagator's scorer produces, and store it in the shape a replay frame "
    "has: the structure, whatever propagator state travels with it, and the "
    "teacher_* labels, with none of the energy, forces, or stress the labeling "
    "hook strips."
)
"""Remedy naming the replay-frame contract both mixture sources have to meet."""


def _frame_schema(frames: Batch) -> frozenset[str]:
    """Return the ``level.field`` names :meth:`Batch.append` intersects over."""
    return frozenset(
        f"{_GROUP_LEVELS.get(name, name)}.{key}"
        for name, group in frames._storage.groups.items()
        for key in group.keys()
    )


def _frame_dtypes(frames: Batch) -> dict[str, torch.dtype]:
    """Return the dtype every ``level.field`` of *frames* is stored at."""
    return {
        f"{_GROUP_LEVELS.get(name, name)}.{key}": group[key].dtype
        for name, group in frames._storage.groups.items()
        for key in group.keys()
    }


def _schema_levels(schema: Iterable[str]) -> frozenset[str]:
    """Return the batch levels *schema* holds at least one field at."""
    return frozenset(name.partition(".")[0] for name in schema)


def _emitted_device(
    dataset: BatchDatasetProtocol, probe: Batch | None = None
) -> torch.device:
    """Return the concrete device *dataset* emits its batches on.

    A declaration is preferred where one settles the question: a
    ``target_device`` a :class:`~nvalchemi.data.datapipes.dataset.Dataset` was
    opened with, or the device an
    :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    already holds its batch on. Two cases are not settled by a declaration and
    are answered by looking at a batch instead. A composition such as
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` declares
    neither attribute, so reading declarations alone reports ``None`` — no
    constraint — and lets a CUDA-resident anchor be paired with a host-memory
    replay buffer that only fails once a segment's loader collates them. And an
    index-less ``cuda`` declaration, which a
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` opened without a device
    reports, names whichever device is current rather than a specific one, so
    it is resolved to the indexed device a batch actually arrives on.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Dataset to resolve the emission device of.
    probe : Batch | None, optional
        A batch already drawn from *dataset*, used instead of drawing one.
        Default ``None`` (draw ``load_batches([[0]])`` when a probe is needed).

    Returns
    -------
    torch.device
        Device batches are emitted on. Always a device: a source that declares
        nothing is measured, rather than reported as an absent constraint.
    """
    target = getattr(dataset, "target_device", None)
    resident = getattr(dataset, "in_memory_batch", None)
    declared = (
        torch.device(target)
        if target is not None
        else None
        if resident is None
        else resident.device
    )
    if declared is not None and not (
        declared.type == "cuda" and declared.index is None
    ):
        return declared
    if probe is None:
        probe = dataset.load_batches([[0]])[0]
    return probe.device


def _same_device(left: torch.device | None, right: torch.device | None) -> bool:
    """Return whether two emitted devices collate without a cross-device copy.

    An index-less device such as ``cuda`` names whichever device of that type is
    current, so it is compared by type alone; two indexed devices have to name
    the same one, because ``cuda:0`` and ``cuda:1`` concatenate no better than a
    host tensor and a device tensor do. A dataset no longer reaches this
    comparison as an index-less CUDA device, because :func:`_emitted_device`
    resolves that against a batch first; the wildcard is left for a
    ``replay_device`` a caller names index-less itself.
    """
    if left is None or right is None:
        return True
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or left.index == right.index


def _check_mixture_sources(
    reference_dataset: BatchDatasetProtocol, replay_buffer: ReplayBuffer
) -> None:
    """Reject two sources that cannot be collated into one training batch.

    The reference schema is read from a one-sample probe batch rather than from
    ``field_names``, which a Zarr-backed
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` answers with the arrays
    it stores while an
    :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    answers with the whole canonical key set.

    Fields are compared by dtype as well as by name. Collation casts the
    second part of a mixed batch to the dtype the first part carries, and which
    source leads a chunk follows whichever child dataset the prefetch happens
    to draw from first, so an anchor labeled at a different precision than the
    generated frames would change the targets' dtype from chunk to chunk with
    nothing to show for it.

    Raises
    ------
    ValueError
        If one source holds a batch level the other lacks, if they carry
        different fields, if they carry a field at different dtypes, or if they
        emit their batches on different devices.
    """
    probe = reference_dataset.load_batches([[0]])[0]
    reference_schema = _frame_schema(probe)
    replay_schema = replay_buffer.schema
    reference_levels = _schema_levels(reference_schema)
    replay_levels = _schema_levels(replay_schema)
    if reference_levels != replay_levels:
        raise ValueError(
            "Both mixture sources must hold the same batch levels, because "
            "collation zero-fills a level only one of them carries instead of "
            f"dropping it; got {sorted(reference_levels)!r} on the reference "
            f"dataset and {sorted(replay_levels)!r} on the replay buffer, "
            f"differing in {sorted(reference_levels ^ replay_levels)!r}. "
            f"{_SCHEMA_REMEDY}"
        )
    if reference_schema != replay_schema:
        raise ValueError(
            "Both mixture sources must carry the same fields, because collation "
            "keeps only the fields both hold and drops the rest out of every "
            f"mixed batch; got {sorted(reference_schema - replay_schema)!r} on "
            "the reference dataset alone and "
            f"{sorted(replay_schema - reference_schema)!r} on the replay buffer "
            f"alone. {_SCHEMA_REMEDY}"
        )
    reference_dtypes = _frame_dtypes(probe)
    replay_dtypes = _frame_dtypes(replay_buffer.dataset.in_memory_batch)
    mismatched = sorted(
        name
        for name in reference_dtypes
        if reference_dtypes[name] != replay_dtypes[name]
    )
    if mismatched:
        detail = "; ".join(
            f"{name!r} at {reference_dtypes[name]!s} on the reference dataset "
            f"and {replay_dtypes[name]!s} on the replay buffer"
            for name in mismatched
        )
        raise ValueError(
            "Both mixture sources must carry each field at one dtype, because "
            "collation casts the second part of a mixed batch to the dtype of "
            "the first and the two sources take turns leading a chunk; got "
            f"{detail}. Label the reference dataset with the cast_to the "
            "on-policy scorer uses — the student's parameter dtype — or cast "
            "it in a batch transform."
        )
    reference_device = _emitted_device(reference_dataset, probe)
    replay_device = _emitted_device(replay_buffer.dataset)
    if not _same_device(reference_device, replay_device):
        raise ValueError(
            "Both mixture sources must emit batches on one device, because "
            "collation concatenates their tensors; got reference on "
            f"{reference_device!s} and replay on {replay_device!s}. Pass "
            "ReplayBuffer(device=...) — OnPolicyConfig.replay_device from a "
            "segment loop — to stage generated frames where the reference "
            "dataset lives."
        )


def _batch_allocation(replay_ratio: float, batch_size: int) -> tuple[int, int]:
    """Return the ``(reference, replay)`` sample counts of one mixed batch."""
    replay = int(replay_ratio * batch_size + 0.5)
    return batch_size - replay, replay


def _minimum_batch_size(replay_ratio: float) -> int:
    """Return the smallest batch size giving both mixture sources a sample.

    The ratio algebra alone is not enough: :func:`_batch_allocation` rounds a
    half sample up into the replay share, so a size where the reference share
    lands exactly on that boundary still starves it. The count is therefore
    walked up until the allocator itself agrees, which takes one step at most.
    """
    size = ceil(0.5 / min(replay_ratio, 1.0 - replay_ratio))
    while min(_batch_allocation(replay_ratio, size)) == 0:
        size += 1
    return size


def _batch_size_remedy(replay_ratio: float) -> str:
    """Return the remedy clause naming a batch size the allocator does accept."""
    remedy = f"raise batch_size to at least {_minimum_batch_size(replay_ratio)}"
    if replay_ratio == 0.5:
        return remedy
    return f"{remedy}, or move replay_ratio toward 0.5"


def _single_source_loader(
    dataset: BatchDatasetProtocol,
    *,
    batch_size: int,
    num_batches: int | None,
    shuffle: bool,
    generator: torch.Generator | None,
    seed: int,
) -> DataLoader:
    """Return a loader over one source, sized to *num_batches* when given."""
    if num_batches is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    single = MultiDataset(dataset)
    return DataLoader(
        single,
        batch_sampler=MultiDatasetBatchSampler(
            single,
            batch_size=batch_size,
            samples_per_dataset=(batch_size,),
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
            seed=seed,
        ),
    )


class ReplayBuffer:
    """Hold generated frames for replay, behind one frozen key schema.

    The buffer is an :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    grown one segment at a time, which makes it a plain
    :class:`~nvalchemi.data.datapipes.dataset.BatchDatasetProtocol` source that
    a :class:`~nvalchemi.data.datapipes.dataloader.DataLoader` or a
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` consumes like
    any other dataset. It starts empty and materializes on the first
    :meth:`extend`.

    The schema check is the point of the class rather than a safety net.
    :meth:`~nvalchemi.data.Batch.append` keeps only the keys both sides hold,
    so a single unlabeled frame appended to a labeled buffer would silently
    strip ``teacher_*`` from *every* frame already stored and leave the loss
    with a missing target several segments later. The first :meth:`extend`
    freezes the incoming schema — levels included, because ``append`` merges
    group by group — and every later one must match it exactly.

    A stored frame is a *training sample*, not a propagator state: it carries
    the structure the student generated — positions, cell, atomic numbers,
    velocities — and the ``teacher_*`` labels, and nothing that describes the
    run that produced it.
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` is what enforces
    that contract on the way in, dropping the ephemeral neighbor tensors, the
    dynamics bookkeeping fields, and the ``energy``, ``forces``, and ``stress``
    the propagator overwrote with the student's own predictions. That last one
    is what keeps a replay frame from carrying a self-label under the name a
    reference target uses: on-policy losses read ``teacher_*``, and
    :func:`build_mixed_loader` requires the reference dataset mixed with the
    buffer to carry the same fields, so an anchor holding reference ``energy``
    or ``forces`` of its own is rejected rather than quietly stripped of them.
    Supervising a mixed batch from teacher labels and reference labels at once
    is masked-composition work that is not modeled yet.

    Over capacity, ``eviction="fifo"`` drops the oldest frames by rebuilding
    the resident batch from the kept indices.

    Parameters
    ----------
    capacity : int | None, optional
        Maximum number of frames kept. Default ``None`` (unbounded), which
        grows for the whole run — bound it on long runs, on any run whose
        frames stay on the propagator's device, and on any run whose objective
        reads a batch as a sample of the current policy, since a uniform draw
        over a buffer nothing is ever retired from is a draw over every policy
        the run has had.
    eviction : {"fifo", "uncertainty"}, optional
        Policy deciding which frames leave a full buffer. Default ``"fifo"``.
        ``"uncertainty"`` is reserved for uncertainty-steered sampling and is
        not implemented yet.
    device : torch.device | str | None, optional
        Device the buffer keeps frames on, and emits them from. Default
        ``None``, which adopts the device the first :meth:`extend` arrives on
        and normalizes every later one to it — a buffer fed by two capture
        routes on different devices holds one device's frames rather than a
        mixture no concatenation can take. A segment loop resolves
        ``OnPolicyConfig.replay_device`` into this argument and names the
        mixture's device explicitly, because its frames arrive from a
        host-memory sink rather than from the propagator; ``"cpu"`` stages
        generated frames off the accelerator.

    Raises
    ------
    ValueError
        If *capacity* is not positive.
    NotImplementedError
        If ``eviction="uncertainty"`` is selected.

    Examples
    --------
    >>> from nvalchemi.training.distillation import ReplayBuffer
    >>> buffer = ReplayBuffer(capacity=4096)
    >>> buffer.extend(labeled_frames)  # doctest: +SKIP
    >>> len(buffer)  # doctest: +SKIP
    128

    Notes
    -----
    Frames are owned, not aliased: the batch that seeds the buffer is copied,
    and later ones are concatenated into fresh tensors, so a propagator may
    keep integrating the batch it handed over.
    """

    def __init__(
        self,
        *,
        capacity: int | None = None,
        eviction: ReplayEviction = "fifo",
        device: torch.device | str | None = None,
    ) -> None:
        """Validate the capacity and eviction policy of an empty buffer."""
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be positive or None; got {capacity!r}.")
        if eviction == "uncertainty":
            raise NotImplementedError(
                "Uncertainty-steered eviction is reserved for committee-based "
                f"frame selection and is not implemented yet; got {eviction!r}, "
                "use 'fifo'."
            )
        self.capacity = capacity
        self.eviction = eviction
        self.device = device
        self._dataset: InMemoryDataset | None = None
        self._schema: frozenset[str] = frozenset()

    def __len__(self) -> int:
        """Return the number of frames currently held."""
        return 0 if self._dataset is None else len(self._dataset)

    @property
    def dataset(self) -> InMemoryDataset:
        """Dataset view of the stored frames, for a loader to draw from."""
        if self._dataset is None:
            raise RuntimeError(
                "ReplayBuffer holds no frames yet; call extend() before reading "
                "its dataset."
            )
        return self._dataset

    @property
    def schema(self) -> frozenset[str]:
        """Frozen ``level.field`` schema every frame must match, empty until filled."""
        return self._schema

    def extend(self, frames: Batch) -> None:
        """Add *frames* to the buffer and evict down to capacity.

        Parameters
        ----------
        frames : Batch
            Frames to store, one graph each. The first call freezes the
            buffer's key schema; later calls must match it. It also pins the
            buffer's device unless one was named at construction.

        Raises
        ------
        ValueError
            If the key schema of *frames* differs from the buffer's.
        """
        if frames.num_graphs == 0:
            return
        if self.device is None:
            self.device = frames.device
        else:
            frames = frames.to(self.device)
        incoming = _frame_schema(frames)
        if self._dataset is None:
            self._schema = incoming
            self._dataset = InMemoryDataset(
                in_memory_batch=frames.clone(), device=self.device
            )
        else:
            self._check_schema(incoming)
            self._dataset.in_memory_batch.append(frames)
        self._evict()

    def clear(self) -> None:
        """Drop every stored frame and unfreeze the key schema.

        The buffer returns to the state it was constructed in, so the next
        :meth:`extend` freezes its schema afresh. That is what lets a restart
        replace a live buffer's contents with the frames a checkpoint carries
        rather than merge the two.
        """
        self._dataset = None
        self._schema = frozenset()

    def _check_schema(self, incoming: frozenset[str]) -> None:
        """Reject frames whose keys or levels differ from the frozen schema."""
        if incoming == self._schema:
            return
        raise ValueError(
            "Replay frames must carry the buffer's key schema, because appending "
            "keeps only the keys both sides hold; got extra "
            f"{sorted(incoming - self._schema)!r} and missing "
            f"{sorted(self._schema - incoming)!r}."
        )

    def _evict(self) -> None:
        """Drop the oldest frames until the buffer fits its capacity."""
        if self._dataset is None or self.capacity is None:
            return
        resident = self._dataset.in_memory_batch
        if resident.num_graphs <= self.capacity:
            return
        kept = torch.arange(
            resident.num_graphs - self.capacity,
            resident.num_graphs,
            device=resident.device,
        )
        self._dataset.in_memory_batch = resident.index_select(kept)


def build_mixed_loader(
    reference_dataset: BatchDatasetProtocol | None,
    replay_buffer: ReplayBuffer,
    *,
    replay_ratio: float,
    batch_size: int,
    num_batches: int | None = None,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a loader drawing a fixed reference/replay mixture in every batch.

    The two sources are composed into a
    :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` and drawn by a
    :class:`~nvalchemi.data.datapipes.samplers.MultiDatasetBatchSampler` with
    the ratio resolved to whole samples of *batch_size*. The composition is
    therefore *exact* per batch rather than an average — with
    ``replay_ratio=0.25`` and ``batch_size=8`` every optimizer step sees six
    reference samples and two replay samples — and the achievable granularity
    is ``1 / batch_size``. A ratio strictly between 0 and 1 that rounds either
    source down to no samples at all is rejected rather than silently trained
    as a single-source run.

    **Rebuild this loader after every segment.** The batch sampler reads the
    child dataset lengths once, in its constructor, and a buffer that has grown
    since is invisible to it: the loader keeps drawing from the prefix the
    sampler was built against and the newest frames — the on-policy ones — are
    never sampled.

    Parameters
    ----------
    reference_dataset : BatchDatasetProtocol | None
        Anchor dataset, typically a teacher-labeled store. ``None`` means the
        run trains on generated data only and requires ``replay_ratio=1.0``.
    replay_buffer : ReplayBuffer
        Buffer of generated frames. An empty buffer falls back to a
        reference-only loader, which is the shape of a run whose first segment
        has not been stored yet.
    replay_ratio : float
        Fraction of every batch drawn from *replay_buffer*, in ``[0, 1]``.
    batch_size : int
        Samples per batch across both sources.
    num_batches : int | None, optional
        Batches per epoch, honored on every path. Default ``None`` (the
        sampler's own ``"dataset_size"`` policy, and one pass over a lone
        source); pass the number of optimizer steps a segment runs to size the
        epoch to the segment.
    shuffle : bool, optional
        Randomize sample order within each child and within each batch.
        Default ``True``.
    generator : torch.Generator | None, optional
        Generator for reproducible mixing. Default ``None``. Used wherever a
        batch sampler draws, which is every path except an unsized
        single-source fallback; that one draws from the global RNG.
    seed : int, optional
        Base seed the batch sampler draws from when it owns its generator,
        combined with the epoch a caller sets on it. Default ``0``. Ignored
        when *generator* is given, and on the unsized single-source fallback.

    Returns
    -------
    DataLoader
        Loader yielding :class:`~nvalchemi.data.Batch` objects of the requested
        composition.

    Raises
    ------
    ValueError
        If *replay_ratio* is outside ``[0, 1]``, if both sources are empty, if
        *reference_dataset* is ``None`` while ``replay_ratio < 1``, if the two
        sources carry different batch levels or fields, if they carry a field
        at different dtypes, if they emit on
        different devices, or if the ratio allocates no samples at all to one
        of them.

    Examples
    --------
    >>> from nvalchemi.training.distillation import build_mixed_loader
    >>> loader = build_mixed_loader(  # doctest: +SKIP
    ...     reference_dataset,
    ...     buffer,
    ...     replay_ratio=0.25,
    ...     batch_size=8,
    ...     num_batches=64,
    ... )

    Notes
    -----
    Collation is not a merge, so the two sources have to carry one schema.
    :meth:`~nvalchemi.data.Batch.append` keeps only the fields both hold within
    a shared storage group — the rest are dropped out of every mixed batch —
    while a whole level only one side holds is *zero-filled* for the other's
    samples instead, which fabricates targets rather than losing them. Both
    differences are compared here, on a one-sample probe batch from each side
    rather than on ``field_names``: a Zarr-backed
    :class:`~nvalchemi.data.datapipes.dataset.Dataset` reports the arrays it
    stores there while the buffer's
    :class:`~nvalchemi.data.datapipes.in_memory_dataset.InMemoryDataset`
    reports the whole canonical key set, so the two never agree on that.

    The schema both sides have to meet is the replay-frame contract: the
    structure, the propagator state travelling with it, and the ``teacher_*``
    labels, with none of the ``energy``, ``forces``, or ``stress`` the labeling
    hook strips. A reference dataset carrying plain reference labels under those
    names is therefore rejected here — label it with
    :func:`~nvalchemi.training.distillation.label_dataset`, requesting the same
    signals the propagator's scorer produces, and the anchor becomes mixable.
    On-policy losses read ``teacher_*``.

    The sampler draws with replacement, so a replay buffer smaller than its
    per-batch allocation oversamples rather than failing.
    """
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError(f"replay_ratio must lie in [0, 1]; got {replay_ratio!r}.")

    if len(replay_buffer) == 0:
        if reference_dataset is None:
            raise ValueError(
                "build_mixed_loader needs something to draw from; got an empty "
                "replay buffer and reference_dataset=None."
            )
        return _single_source_loader(
            reference_dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
            seed=seed,
        )

    if reference_dataset is None:
        if replay_ratio != 1.0:
            raise ValueError(
                "A replay_ratio below 1 mixes in reference data, so a reference "
                "dataset is required; got reference_dataset=None and "
                f"replay_ratio={replay_ratio!r}."
            )
        return _single_source_loader(
            replay_buffer.dataset,
            batch_size=batch_size,
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
            seed=seed,
        )

    _check_mixture_sources(reference_dataset, replay_buffer)
    reference_samples, replay_samples = _batch_allocation(replay_ratio, batch_size)
    if 0.0 < replay_ratio < 1.0 and min(reference_samples, replay_samples) == 0:
        raise ValueError(
            f"replay_ratio={replay_ratio!r} allocates {reference_samples} "
            f"reference and {replay_samples} replay samples of "
            f"batch_size={batch_size!r}, so one source never reaches an "
            f"optimizer step; {_batch_size_remedy(replay_ratio)}."
        )
    mixed = MultiDataset(reference_dataset, replay_buffer.dataset, output_strict=False)
    return DataLoader(
        mixed,
        batch_sampler=MultiDatasetBatchSampler(
            mixed,
            batch_size=batch_size,
            samples_per_dataset=(reference_samples, replay_samples),
            num_batches=num_batches,
            shuffle=shuffle,
            generator=generator,
            seed=seed,
        ),
    )
