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
"""Transactional, pickle-free Zarr checkpoints for any set of stateful objects.

Writing state transactionally and refusing to restore a torn store is not
specific to any one workflow: enhanced-sampling restart needs it, plain MD
restart needs it, and an NEB restart will need it.  So this lives beside the
package rather than inside one subpackage, and it is written against
:class:`Stateful` — the ``state_dict`` / ``load_state_dict`` pair that hooks,
integrators, biases, ladders and ``nn.Module`` already satisfy.

Layout, extending the existing ``AtomicData`` Zarr record in place::

    checkpoint.zarr/
      meta/, core/, custom/     the batch, via AtomicDataZarrWriter (optional)
      checkpoint/
        manifest                committed metadata — WRITTEN LAST
        <component>/            one group per named Stateful; '/' nests

Component names are caller-chosen and may nest (``"biases/umbrella"``), so a
checkpoint can be inspected group by group rather than as one opaque blob.
``manifest`` is reserved.

State is stored as Zarr arrays (tensors) and group attributes (scalars,
strings, nested mappings).  There are **no pickle payloads**: a checkpoint is
readable by anything that can read Zarr, and loading one cannot execute code.

Transactionality
----------------
Components are written first, each checksummed, and the manifest is written
last.  A checkpoint interrupted at any point therefore has no manifest, and
:func:`load_checkpoint` rejects a store without one rather than restoring a
torn half-state.  Checksums are verified on read, so a store that was
truncated *after* the manifest landed is also caught.

The whole store is built beside the destination and moved into place only
once it is complete, so saving over an existing checkpoint cannot damage it.
Writing in place would: a rewrite interrupted before the new manifest lands
leaves the *old* manifest attesting to component data that has already been
replaced, which fails its own checksum — so an interrupted save would destroy
a restart point that was valid a moment earlier, which is the opposite of
what manifest-last is for.  Writing in place also simply fails for a store
holding arrays, because an array cannot be created where one already exists;
saving every epoch to one path is the ordinary workflow and has to work.

The cover is total.  Every component carries its own digest, and
``batch_checksum`` covers ``meta/``, ``core/``, and ``custom/`` — the
positions, velocities, pointer arrays, and any extra per-graph fields that
``AtomicDataZarrWriter`` writes outside the component path.  Checksumming only
the component state would attest to the model and integrator while leaving the
coordinates unguarded, which is the half of the checkpoint a reader is most
likely to trust blindly.

Relationship to ``training/_checkpoint.py``
-------------------------------------------
Training has its own checkpoint layer with a different storage format — a
directory of ``.pt`` files with per-component specs, indices, and
model/optimizer/scheduler associations.  It is not refactored onto this
module; that is a format migration with its own compatibility story.  What
this module fixes is the smaller thing: a *third* copy of "write
transactionally, checksum, refuse a torn store" is not created the next time a
workflow needs restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import torch
import zarr
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointContents",
    "CheckpointManifest",
    "Stateful",
    "load_checkpoint",
    "save_checkpoint",
]

CHECKPOINT_FORMAT_VERSION = 1

_ROOT = "checkpoint"
_MANIFEST_NAME = "manifest"
_MANIFEST = f"{_ROOT}/{_MANIFEST_NAME}"

_SCALAR_TYPES = (bool, int, float, str)

# Zarr groups that together hold the batch.  These are written by
# AtomicDataZarrWriter rather than through _encode_state, so they need their
# own integrity cover — without it the manifest would attest only to the
# component state and a corrupted core/positions would restore silently.
_BATCH_GROUPS = ("meta", "core", "custom")


@runtime_checkable
class Stateful(Protocol):
    """Anything that can hand out its state and take it back.

    The whole contract this module needs.  ``nn.Module``, ``BaseDynamics``,
    ``ReplicaExchange``, every ``CheckpointableHook`` and every adaptive bias
    already satisfy it, which is the point: a checkpoint is a mapping of name
    to *something that knows its own state*, not a per-workflow schema.

    .. warning::

        ``isinstance`` against a runtime-checkable Protocol checks that the
        members *exist*, never that they behave.  Use it to ask whether an
        object owns state, not to decide that an arbitrary object is safe to
        check-point.
    """

    def state_dict(self) -> Mapping[str, Any]:
        """Return this object's state as a Zarr-representable mapping."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore this object's state from a mapping."""
        ...


class CheckpointManifest(BaseModel):
    """Committed checkpoint metadata, written after every component.

    Its presence is the commit marker: a store without a manifest is an
    incomplete write and is refused.

    Attributes
    ----------
    format_version:
        Layout version, for forward migration.
    components:
        Names of the component groups written.
    checksums:
        SHA-256 per component, verified on read.  Must have an entry for
        every name in :attr:`components`, and no others.
    batch_checksum:
        SHA-256 over every array in ``meta/``, ``core/``, and ``custom/`` —
        the batch itself.  Kept separate from :attr:`checksums` because those
        name component groups the reader walks, while this covers arrays
        written by ``AtomicDataZarrWriter``.  Empty exactly when no batch was
        saved.
    num_graphs:
        Graph count, validated against the restored batch.  ``None`` when no
        batch was saved.
    batch_fields:
        Per-graph fields persisted alongside the batch through the custom
        array API, because ``AtomicDataZarrWriter`` only writes the fields it
        recognises.  Recorded so the reader restores exactly what was saved.
    batch_field_dtypes:
        ``str(dtype)`` per entry in :attr:`batch_fields`.
    compatibility:
        Caller-defined fingerprint of the configuration that wrote the
        checkpoint — class names, hyperparameters, whatever restoring into a
        different value would silently corrupt.  This module stores and
        returns it; deciding what a mismatch *means* belongs to the caller,
        which is why there is a ``validate`` hook on :func:`load_checkpoint`
        rather than a comparison here.
    metadata:
        Caller-defined bookkeeping — step counts, epoch indices, timestamps.
        Never interpreted here.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: int = CHECKPOINT_FORMAT_VERSION
    components: list[str] = Field(default_factory=list)
    checksums: dict[str, str] = Field(default_factory=dict)
    batch_checksum: str = ""
    num_graphs: int | None = None
    batch_fields: list[str] = Field(default_factory=list)
    batch_field_dtypes: dict[str, str] = Field(default_factory=dict)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_component_is_covered(self) -> CheckpointManifest:
        """Reject a manifest whose integrity cover has gaps.

        In a manifest-gated format the manifest is the authority on what the
        store contains, so a declared component without a checksum is not
        "unverified" — it is invalid.  Treating a missing entry as permission
        to skip verification would make the cover opt-out: deleting one key
        from the manifest attributes is enough to leave that component free to
        modify.  The same goes for the batch.

        Returns
        -------
        CheckpointManifest
            The validated manifest.

        Raises
        ------
        ValueError
            If a component has no checksum, a checksum names no component, a
            saved batch has no checksum, or a checksum claims a batch that was
            never saved.
        """
        declared = set(self.components)
        covered = set(self.checksums)

        uncovered = sorted(declared - covered)
        if uncovered:
            raise ValueError(
                f"Checkpoint manifest declares component(s) {uncovered} with "
                "no checksum. Every declared component must be covered; a "
                "missing entry is a tampered or truncated manifest, not a "
                "component that may be read unverified."
            )
        orphaned = sorted(covered - declared)
        if orphaned:
            raise ValueError(
                f"Checkpoint manifest has checksum(s) for {orphaned}, which it "
                "does not declare as components. The manifest is inconsistent "
                "with itself."
            )
        if self.num_graphs is not None and not self.batch_checksum:
            raise ValueError(
                "Checkpoint manifest records a batch but has no "
                "batch_checksum. The positions, velocities and per-graph "
                "identity would then be restored unverified."
            )
        if self.num_graphs is None and self.batch_checksum:
            raise ValueError(
                "Checkpoint manifest has a batch_checksum but records no "
                "batch. The manifest is inconsistent with itself."
            )
        return self


@dataclass(frozen=True)
class CheckpointContents:
    """What a verified checkpoint held.

    Attributes
    ----------
    manifest:
        The committed manifest.
    states:
        Component name to decoded state.  Present whether or not
        :func:`load_checkpoint` applied it, so a caller that must control
        ordering — an integrator whose per-system state has to be allocated
        against the restored batch before it can be written into — can apply
        that component itself.
    batch:
        The restored batch, or ``None`` when none was saved.
    """

    manifest: CheckpointManifest
    states: dict[str, dict[str, Any]] = field(default_factory=dict)
    batch: Batch | None = None


def _qualified_name(obj: Any) -> str:
    """Return ``module.ClassName`` for *obj*'s type.

    Useful for building a :attr:`CheckpointManifest.compatibility` entry: a
    restore into a different class is the failure this names.

    Parameters
    ----------
    obj:
        Any object.

    Returns
    -------
    str
        Fully-qualified class name.
    """
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve ``"torch.float32"`` back to the dtype object.

    Parameters
    ----------
    name:
        The ``str(dtype)`` form.

    Returns
    -------
    torch.dtype
        The resolved dtype.

    Raises
    ------
    ValueError
        If the name does not resolve to a dtype.
    """
    candidate = getattr(torch, name.rsplit(".", 1)[-1], None)
    if not isinstance(candidate, torch.dtype):
        raise ValueError(f"Checkpoint: unknown tensor dtype {name!r}.")
    return candidate


def _encode_state(group: zarr.Group, state: Mapping[str, Any]) -> None:
    """Write a nested state mapping into *group*.

    Tensors become arrays; scalars, strings, ``None``, and flat sequences
    become attributes; nested mappings become subgroups.  A per-key kind tag
    is stored so the decoder never has to guess.

    Parameters
    ----------
    group:
        Destination Zarr group.
    state:
        Mapping of tensors, scalars, and nested mappings.

    Raises
    ------
    TypeError
        If a value is none of the supported kinds.  Refusing here is
        deliberate: the alternative is a pickle payload, which would make a
        checkpoint executable and unreadable outside Python.
    """
    kinds: dict[str, str] = {}
    scalars: dict[str, Any] = {}
    dtypes: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}

    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            kinds[key] = "tensor"
            dtypes[key] = str(value.dtype)
            array = value.detach().cpu().contiguous().numpy()
            # Zarr materialises a 0-d array as shape (1,), so the true shape
            # is recorded separately and reapplied on decode. Without it a
            # scalar buffer (a step counter, a deposition count) comes back
            # rank-1 and fails its own component checksum on restore.
            shapes[key] = list(array.shape)
            group.create_array(key, shape=array.shape, dtype=array.dtype)
            if array.size:
                group[key][...] = array
        elif isinstance(value, Mapping):
            kinds[key] = "group"
            _encode_state(group.require_group(key), value)
        elif value is None or isinstance(value, _SCALAR_TYPES):
            kinds[key] = "scalar"
            scalars[key] = value
        elif isinstance(value, (list, tuple)) and all(
            v is None or isinstance(v, _SCALAR_TYPES) for v in value
        ):
            kinds[key] = "scalar"
            scalars[key] = list(value)
        else:
            raise TypeError(
                f"Checkpoint: cannot store {key!r} of type "
                f"{type(value).__name__}. State must be tensors, scalars, "
                "strings, flat sequences of those, or nested mappings — a "
                "checkpoint carries no pickle payloads."
            )

    group.attrs["kinds"] = kinds
    group.attrs["scalars"] = scalars
    group.attrs["dtypes"] = dtypes
    group.attrs["shapes"] = shapes


def _decode_state(group: zarr.Group, device: torch.device | str) -> dict[str, Any]:
    """Read back a mapping written by :func:`_encode_state`.

    Parameters
    ----------
    group:
        Source Zarr group.
    device:
        Device to place restored tensors on.

    Returns
    -------
    dict[str, Any]
        The restored mapping.
    """
    kinds = dict(group.attrs.get("kinds", {}))
    scalars = dict(group.attrs.get("scalars", {}))
    dtypes = dict(group.attrs.get("dtypes", {}))
    shapes = dict(group.attrs.get("shapes", {}))

    state: dict[str, Any] = {}
    for key, kind in kinds.items():
        if kind == "tensor":
            array = np.asarray(group[key][...])
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            # Missing for checkpoints written before shapes were recorded;
            # falling back to the stored shape reads those exactly as before.
            if key in shapes:
                tensor = tensor.reshape(tuple(shapes[key]))
            state[key] = tensor.to(device=device, dtype=_torch_dtype(dtypes[key]))
        elif kind == "group":
            state[key] = _decode_state(group[key], device)
        else:
            state[key] = scalars.get(key)
    return state


def _component_checksum(state: Mapping[str, Any]) -> str:
    """Return a SHA-256 over a component's contents.

    Order-independent by construction (keys are walked sorted), so the digest
    depends on the state and not on dict insertion order.

    Parameters
    ----------
    state:
        The component state.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()

    def _walk(mapping: Mapping[str, Any], prefix: str) -> None:
        for key in sorted(mapping):
            value = mapping[key]
            digest.update(f"{prefix}{key}".encode())
            if isinstance(value, torch.Tensor):
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(value, Mapping):
                _walk(value, f"{prefix}{key}/")
            else:
                digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    _walk(state, "")
    return digest.hexdigest()


def _batch_checksum(root: zarr.Group) -> str:
    """Return a SHA-256 over every array holding the batch.

    Covers ``meta/``, ``core/``, and ``custom/`` — positions, velocities,
    forces, the CSR pointer arrays, and any extra per-graph fields.  These are
    written by ``AtomicDataZarrWriter``, not by :func:`_encode_state`, so they
    are outside the per-component checksum path and need this.

    Reads the arrays back from the store rather than hashing the in-memory
    batch, so the write-side and read-side digests are computed over exactly
    the same bytes.  The cost is one extra full read of the batch on write;
    integrity that only sometimes holds is not worth the saving.

    Parameters
    ----------
    root:
        The opened checkpoint root group.

    Returns
    -------
    str
        Hex digest, empty-string-safe if the groups are absent.
    """
    digest = hashlib.sha256()
    for group_name in _BATCH_GROUPS:
        if group_name not in root:
            continue
        group = root[group_name]
        for key in sorted(group.array_keys()):
            array = group[key]
            digest.update(f"{group_name}/{key}".encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(tuple(array.shape)).encode())
            digest.update(np.ascontiguousarray(array[...]).tobytes())
    return digest.hexdigest()


def _check_names(components: Mapping[str, Any]) -> None:
    """Reject component names that cannot be stored.

    Parameters
    ----------
    components:
        The caller's component mapping.

    Raises
    ------
    ValueError
        If a name is empty, has an empty path segment, or collides with the
        reserved manifest group.
    """
    for name in components:
        parts = name.split("/")
        if not name or any(not part for part in parts):
            raise ValueError(
                f"Checkpoint: component name {name!r} is empty or has an empty "
                "path segment. Names become Zarr groups, and '/' nests them."
            )
        if parts[0] == _MANIFEST_NAME:
            raise ValueError(
                f"Checkpoint: component name {name!r} collides with the "
                f"reserved {_MANIFEST!r} group, which is the commit marker."
            )


def save_checkpoint(
    path: str | Path,
    components: Mapping[str, Stateful],
    *,
    batch: Batch | None = None,
    batch_fields: Sequence[str] = (),
    compatibility: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointManifest:
    """Write a transactional checkpoint.

    Order matters and is the whole guarantee: batch, then each component,
    then the manifest.  An interruption anywhere before the last step leaves
    a store with no manifest, which :func:`load_checkpoint` refuses.

    The store is built beside *path* and moved into place only once it is
    complete, so an interrupted save leaves any checkpoint already there
    intact and loadable.  Saving repeatedly to one path — every epoch, say —
    is therefore both safe and supported.  On success whatever was at *path*
    is replaced.

    Parameters
    ----------
    path:
        Destination store.
    components:
        Mapping of component name to :class:`Stateful`.  Names may nest with
        ``/`` (``"biases/umbrella"``); ``"manifest"`` is reserved.
    batch:
        Batch to save alongside the components, or ``None`` for a
        state-only checkpoint.
    batch_fields:
        Per-graph fields on *batch* to persist explicitly.
        ``AtomicDataZarrWriter.write`` only stores the fields it recognises,
        so anything else — walker identity, a state assignment — is dropped
        without complaint unless it is named here.
    compatibility:
        Fingerprint of the configuration doing the writing, stored verbatim
        and handed back on load.  Nothing here interprets it.
    metadata:
        Caller bookkeeping — step counts, epoch indices — stored verbatim.

    Returns
    -------
    CheckpointManifest
        The manifest that was committed.

    Raises
    ------
    ValueError
        If a component name is empty or reserved, or *batch_fields* names a
        field the batch does not carry.
    TypeError
        If any component's state holds a value that cannot be stored without
        pickling.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.writing-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        manifest = _write_store(
            staging,
            components,
            batch=batch,
            batch_fields=batch_fields,
            compatibility=compatibility,
            metadata=metadata,
        )
        _move_into_place(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def _move_into_place(staging: Path, destination: Path) -> None:
    """Replace *destination* with *staging*, keeping one of them whole.

    Two renames rather than one, because a directory cannot be renamed onto a
    populated directory.  The old store is moved aside first and removed only
    after the new one is in place, so a failure at any point leaves either the
    previous checkpoint or the new one — never a mixture of the two.

    Parameters
    ----------
    staging:
        The freshly written store.
    destination:
        Where it belongs.
    """
    if not destination.exists():
        os.replace(staging, destination)
        return

    superseded = destination.with_name(f"{destination.name}.superseded-{os.getpid()}")
    shutil.rmtree(superseded, ignore_errors=True)
    os.replace(destination, superseded)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(superseded, destination)
        raise
    shutil.rmtree(superseded, ignore_errors=True)


def _write_store(
    path: Path,
    components: Mapping[str, Stateful],
    *,
    batch: Batch | None,
    batch_fields: Sequence[str],
    compatibility: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> CheckpointManifest:
    """Write a complete store at *path*, manifest last.

    Parameters
    ----------
    path:
        Destination, which must not already exist.
    components:
        Name to :class:`Stateful`.
    batch:
        Batch to save, or ``None``.
    batch_fields:
        Per-graph fields to persist explicitly.
    compatibility:
        Configuration fingerprint.
    metadata:
        Caller bookkeeping.

    Returns
    -------
    CheckpointManifest
        The manifest that was committed.
    """
    _check_names(components)

    batch_field_dtypes: dict[str, str] = {}
    if batch is not None:
        writer = AtomicDataZarrWriter(str(path))
        writer.write(batch)
        for name in batch_fields:
            value = getattr(batch, name, None)
            if value is None:
                raise ValueError(
                    f"Checkpoint: batch_fields names {name!r}, which the batch "
                    "does not carry. Drop it, or stamp the field before "
                    "saving — a field that is silently skipped comes back "
                    "missing on restore."
                )
            writer.add_custom(name, value.reshape(-1), level="system")
            batch_field_dtypes[name] = str(value.dtype)
    elif batch_fields:
        raise ValueError(
            "Checkpoint: batch_fields was given without a batch to take them from."
        )

    root = zarr.open_group(str(path), mode="a")
    # Computed now, while the store holds only the batch: the digest must not
    # depend on the component groups written next.
    batch_checksum = _batch_checksum(root) if batch is not None else ""
    container = root.require_group(_ROOT)

    checksums: dict[str, str] = {}
    for name, component in components.items():
        state = dict(component.state_dict())
        group = container
        for part in name.split("/"):
            group = group.require_group(part)
        _encode_state(group, state)
        checksums[name] = _component_checksum(state)

    manifest = CheckpointManifest(
        components=sorted(components),
        checksums=checksums,
        batch_checksum=batch_checksum,
        num_graphs=batch.num_graphs if batch is not None else None,
        batch_fields=list(batch_fields),
        batch_field_dtypes=batch_field_dtypes,
        compatibility=dict(compatibility or {}),
        metadata=dict(metadata or {}),
    )

    # Written last: this is the commit.
    manifest_group = container.require_group(_MANIFEST_NAME)
    manifest_group.attrs["manifest"] = manifest.model_dump()
    return manifest


def load_checkpoint(
    path: str | Path,
    components: Mapping[str, Stateful] | None = None,
    *,
    device: torch.device | str = "cpu",
    validate: Callable[[CheckpointManifest], None] | None = None,
) -> CheckpointContents:
    """Read a checkpoint, refusing anything not fully committed.

    Every component is decoded and checksum-verified before *any* of them is
    applied, so a store that fails verification leaves the caller's objects
    untouched.

    Parameters
    ----------
    path:
        Source store.
    components:
        Mapping of component name to :class:`Stateful` to restore into.
        Every name must be present in the checkpoint.  Omit a component —
        or pass ``None`` for all of them — to have its state decoded and
        returned without being applied, which is what a caller with an
        ordering constraint needs.
    device:
        Device to place restored tensors on.
    validate:
        Called with the manifest after integrity checks pass and before any
        state is applied — and before the check that *components* names only
        things the checkpoint holds, since a configuration mismatch is usually
        what made a component absent.  This is where a caller compares
        :attr:`CheckpointManifest.compatibility` against its own
        configuration and raises, because what a mismatch *means* is
        domain knowledge this module does not have.

    Returns
    -------
    CheckpointContents
        The manifest, every component's decoded state, and the batch when one
        was saved.

    Raises
    ------
    ValueError
        If the store has no committed manifest; if the manifest is internally
        inconsistent; if a declared component is missing; if any checksum,
        component or batch, does not match; or if *components* names something
        the checkpoint does not hold.
    """
    root = zarr.open_group(str(path), mode="r")
    if _MANIFEST not in root:
        raise ValueError(
            f"Checkpoint at {path} has no committed manifest, so it was never "
            "finished — the manifest is written last, after every component. "
            "Treat this store as an interrupted write and discard it."
        )
    try:
        manifest = CheckpointManifest(**dict(root[_MANIFEST].attrs["manifest"]))
    except PydanticValidationError as exc:
        # Surface the same way as every other checkpoint failure — one
        # ValueError naming the store — rather than a nested pydantic report.
        reasons = "; ".join(str(err["msg"]) for err in exc.errors())
        raise ValueError(
            f"Checkpoint at {path} has an invalid manifest: {reasons}"
        ) from exc

    if manifest.format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint at {path} has format_version "
            f"{manifest.format_version}, but this build reads version "
            f"{CHECKPOINT_FORMAT_VERSION}."
        )

    container = root[_ROOT]
    states: dict[str, dict[str, Any]] = {}
    for name in manifest.components:
        group: Any = container
        for part in name.split("/"):
            if part not in group:
                raise ValueError(
                    f"Checkpoint at {path} declares component {name!r} in its "
                    "manifest but the group is missing; the store is corrupt."
                )
            group = group[part]
        state = _decode_state(group, device)
        actual = _component_checksum(state)
        # Unconditional: the manifest validator guarantees the entry exists,
        # so no path reads a component without checking it.
        expected = manifest.checksums[name]
        if actual != expected:
            raise ValueError(
                f"Checkpoint at {path}: component {name!r} failed its checksum "
                f"(expected {expected[:12]}…, got {actual[:12]}…). The store "
                "was modified or truncated after the manifest was written."
            )
        states[name] = state

    batch = None
    if manifest.num_graphs is not None:
        actual = _batch_checksum(root)
        if actual != manifest.batch_checksum:
            raise ValueError(
                f"Checkpoint at {path}: the batch failed its checksum "
                f"(expected {manifest.batch_checksum[:12]}…, got "
                f"{actual[:12]}…). One of meta/, core/, or custom/ was "
                "modified or truncated after the manifest was written — "
                "positions, velocities, or per-graph identity can no longer "
                "be trusted."
            )
        batch = _read_batch(path, manifest, device)

    # After integrity, before the structural check below: a configuration
    # mismatch usually *causes* the missing component ("this checkpoint was
    # written without replica exchange"), and the domain can say so in terms
    # the caller acts on. A bare "does not hold component(s) ['exchange']"
    # would pre-empt that with the symptom.
    if validate is not None:
        validate(manifest)

    missing = sorted(set(components or {}) - set(manifest.components))
    if missing:
        raise ValueError(
            f"Checkpoint at {path} does not hold component(s) {missing}, which "
            f"the caller asked to restore. It holds {manifest.components}."
        )

    for name, component in (components or {}).items():
        component.load_state_dict(states[name])

    return CheckpointContents(manifest=manifest, states=states, batch=batch)


def _read_batch(
    path: str | Path, manifest: CheckpointManifest, device: torch.device | str
) -> Batch:
    """Reconstruct the batch, extra per-graph fields included.

    Parameters
    ----------
    path:
        Source store.
    manifest:
        The committed manifest, read for the expected graph count and the
        extra fields to restore.
    device:
        Device for the restored batch.

    Returns
    -------
    Batch
        The restored batch.

    Raises
    ------
    ValueError
        If the store holds a different number of graphs than the manifest
        recorded.
    """
    reader = AtomicDataZarrReader(str(path))
    if len(reader) != manifest.num_graphs:
        raise ValueError(
            f"Checkpoint at {path} holds {len(reader)} graph(s) but its "
            f"manifest records {manifest.num_graphs}; the store is corrupt."
        )

    data_list = [AtomicData(**reader.read(i)[0]) for i in range(len(reader))]
    batch = Batch.from_data_list(data_list).to(device)

    root = zarr.open_group(str(path), mode="r")
    custom = root["custom"] if "custom" in root else None
    for name in manifest.batch_fields:
        if custom is not None and name in custom:
            values = np.asarray(custom[name][...])
            batch[name] = torch.from_numpy(np.ascontiguousarray(values)).to(
                device=device,
                dtype=_torch_dtype(manifest.batch_field_dtypes[name]),
            )
    return batch
