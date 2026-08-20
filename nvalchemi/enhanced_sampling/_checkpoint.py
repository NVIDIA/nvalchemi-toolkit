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
"""Transactional Zarr checkpoints for enhanced sampling.

Layout, extending the existing ``AtomicData`` Zarr record in place::

    checkpoint.zarr/
      meta/, core/, custom/     walker batch, via AtomicDataZarrWriter
      sampling/
        manifest               committed metadata — WRITTEN LAST
        dynamics/              integrator, thermostat, and RNG counters
        biases/<name>/         each bias's state_dict()
        runner/                walker-id allocation and epoch counters
        exchange/              reserved; replica exchange is not implemented

State is stored as Zarr arrays (tensors) and group attributes (scalars,
strings, nested mappings).  There are **no pickle payloads**: a checkpoint is
readable by anything that can read Zarr, and loading one cannot execute code.

Transactionality
----------------
Components are written first, each checksummed, and the manifest is written
last.  A checkpoint interrupted at any point therefore has no manifest, and
:func:`read_checkpoint` rejects a store without one rather than restoring a
torn half-state.  Checksums are verified on read, so a store that was
truncated *after* the manifest landed is also caught.

The cover is total.  Every ``sampling/`` component carries its own digest,
and ``batch_checksum`` covers ``meta/``, ``core/``, and ``custom/`` — the
positions, velocities, forces, pointer arrays, and walker identity that
``AtomicDataZarrWriter`` writes outside the component path.  Checksumming
only the sampling state would attest to the bias and integrator while
leaving the coordinates unguarded, which is the half of the checkpoint a
reader is most likely to trust blindly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
    from pathlib import Path

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointManifest",
    "read_checkpoint",
    "write_checkpoint",
]

CHECKPOINT_FORMAT_VERSION = 1

_SAMPLING = "sampling"
_MANIFEST = "sampling/manifest"

# Graph-level fields the runner owns.  ``AtomicDataZarrWriter.write`` only
# persists fields it recognises, so these are added explicitly through
# ``add_custom`` — without that, walker identity would be silently dropped and
# a "restored" run would come back with fresh ids and default state
# assignments.
_IDENTITY_FIELDS = ("walker_id", "thermodynamic_state_id")

_SCALAR_TYPES = (bool, int, float, str)

# Zarr groups that together hold the walker batch.  These are written by
# AtomicDataZarrWriter rather than through _encode_state, so they need their
# own integrity cover — without it the manifest would attest only to the
# sampling/* state and a corrupted core/positions would restore silently.
_BATCH_GROUPS = ("meta", "core", "custom")


class CheckpointManifest(BaseModel):
    """Committed checkpoint metadata, written after every component.

    Its presence is the commit marker: a store without a manifest is an
    incomplete write and is refused.

    Attributes
    ----------
    format_version:
        Layout version, for forward migration.
    sampling_step:
        Dynamics step count the checkpoint was taken at.
    sampling_epoch:
        Consistency epoch the checkpoint was taken at.
    steps_per_epoch:
        Epoch length in force evaluations.
    num_graphs:
        Walker count, validated against the restored batch.
    components:
        Names of the ``sampling/`` groups written.
    checksums:
        SHA-256 per ``sampling/`` component, verified on read.  Must have an
        entry for every name in :attr:`components`, and no others.
    batch_checksum:
        SHA-256 over every array in ``meta/``, ``core/``, and ``custom/`` —
        the walker batch itself.  Kept separate from :attr:`checksums`
        because those name ``sampling/`` groups the reader walks, while this
        covers arrays written by ``AtomicDataZarrWriter``.
    model_class:
        Fully-qualified model wrapper class, validated on restore.
    dynamics_class:
        Fully-qualified dynamics class, validated on restore.
    bias_classes:
        Bias name to fully-qualified class, validated on restore.
    exchange_config:
        Replica-exchange configuration fingerprint, or ``None`` when the run
        had no exchange.  Validated on restore: the ladder decides what a
        swap *means*, so restoring into a different one — or into no exchange
        at all — has to be refused rather than silently accepted.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: int = CHECKPOINT_FORMAT_VERSION
    sampling_step: int
    sampling_epoch: int
    steps_per_epoch: int
    num_graphs: int
    components: list[str] = Field(default_factory=list)
    checksums: dict[str, str] = Field(default_factory=dict)
    batch_checksum: str = ""
    model_class: str = ""
    dynamics_class: str = ""
    bias_classes: dict[str, str] = Field(default_factory=dict)
    exchange_config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _every_component_is_covered(self) -> CheckpointManifest:
        """Reject a manifest whose integrity cover has gaps.

        In a manifest-gated format the manifest is the authority on what the
        store contains, so a declared component without a checksum is not
        "unverified" — it is invalid.  Treating a missing entry as permission
        to skip verification would make the cover opt-out: deleting one key
        from the manifest attributes is enough to leave that component free to
        modify.  The same goes for the walker batch.

        Returns
        -------
        CheckpointManifest
            The validated manifest.

        Raises
        ------
        ValueError
            If a component has no checksum, a checksum names no component, or
            the batch checksum is absent.
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
        if not self.batch_checksum:
            raise ValueError(
                "Checkpoint manifest has no batch_checksum. The walker batch "
                "— positions, velocities, and walker identity — would then be "
                "restored unverified."
            )
        return self


def _qualified_name(obj: Any) -> str:
    """Return ``module.ClassName`` for *obj*'s type.

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

    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            kinds[key] = "tensor"
            dtypes[key] = str(value.dtype)
            array = value.detach().cpu().contiguous().numpy()
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

    state: dict[str, Any] = {}
    for key, kind in kinds.items():
        if kind == "tensor":
            array = np.asarray(group[key][...])
            tensor = torch.from_numpy(np.ascontiguousarray(array))
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
    """Return a SHA-256 over every array holding the walker batch.

    Covers ``meta/``, ``core/``, and ``custom/`` — positions, velocities,
    forces, the CSR pointer arrays, and the runner's identity fields.  These
    are written by ``AtomicDataZarrWriter``, not by :func:`_encode_state`, so
    they are outside the per-component checksum path and need this.

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


def write_checkpoint(
    path: str | Path,
    batch: Batch,
    components: Mapping[str, Mapping[str, Any]],
    *,
    sampling_step: int,
    sampling_epoch: int,
    steps_per_epoch: int,
    model_class: str = "",
    dynamics_class: str = "",
    bias_classes: Mapping[str, str] | None = None,
    exchange_config: Mapping[str, Any] | None = None,
) -> CheckpointManifest:
    """Write a transactional checkpoint.

    Order matters and is the whole guarantee: walker batch, then each
    component, then the manifest.  An interruption anywhere before the last
    step leaves a store with no manifest, which :func:`read_checkpoint`
    refuses.

    Parameters
    ----------
    path:
        Destination store.
    batch:
        The live walker batch.
    components:
        Mapping of component name (``"dynamics"``, ``"biases/umbrella"``,
        ``"runner"``) to its state mapping.
    sampling_step:
        Step count at the checkpoint.
    sampling_epoch:
        Epoch at the checkpoint.
    steps_per_epoch:
        Epoch length.
    model_class:
        Fully-qualified model class, recorded for restore-time validation.
    dynamics_class:
        Fully-qualified dynamics class, likewise.
    bias_classes:
        Bias name to fully-qualified class, likewise.
    exchange_config:
        Replica-exchange configuration fingerprint, or ``None``.

    Returns
    -------
    CheckpointManifest
        The manifest that was committed.
    """
    writer = AtomicDataZarrWriter(str(path))
    writer.write(batch)

    # AtomicDataZarrWriter.write persists only the fields it recognises, so
    # the runner's identity fields have to be added through the custom-array
    # API or they are dropped without complaint.
    for field in _IDENTITY_FIELDS:
        value = getattr(batch, field, None)
        if value is not None:
            writer.add_custom(field, value.reshape(-1), level="system")

    root = zarr.open_group(str(path), mode="a")
    # Computed now, while the store holds only the batch: the digest must not
    # depend on the sampling/* groups written next.
    batch_checksum = _batch_checksum(root)
    sampling = root.require_group(_SAMPLING)

    checksums: dict[str, str] = {}
    for name, state in components.items():
        group = sampling
        for part in name.split("/"):
            group = group.require_group(part)
        _encode_state(group, state)
        checksums[name] = _component_checksum(state)

    manifest = CheckpointManifest(
        sampling_step=sampling_step,
        sampling_epoch=sampling_epoch,
        steps_per_epoch=steps_per_epoch,
        num_graphs=batch.num_graphs,
        components=sorted(components),
        checksums=checksums,
        batch_checksum=batch_checksum,
        model_class=model_class,
        dynamics_class=dynamics_class,
        bias_classes=dict(bias_classes or {}),
        exchange_config=dict(exchange_config) if exchange_config else None,
    )

    # Written last: this is the commit.
    manifest_group = root.require_group(_MANIFEST)
    manifest_group.attrs["manifest"] = manifest.model_dump()
    return manifest


def read_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[Batch, dict[str, dict[str, Any]], CheckpointManifest]:
    """Read a checkpoint, refusing anything not fully committed.

    Parameters
    ----------
    path:
        Source store.
    device:
        Device to place restored tensors on.

    Returns
    -------
    tuple[Batch, dict[str, dict[str, Any]], CheckpointManifest]
        The walker batch, the component states, and the manifest.

    Raises
    ------
    ValueError
        If the store has no committed manifest; if the manifest is internally
        inconsistent (a declared component with no checksum, a checksum for no
        component, or no batch checksum); if a declared component is missing;
        or if any checksum, component or batch, does not match.
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

    sampling = root[_SAMPLING]
    states: dict[str, dict[str, Any]] = {}
    for name in manifest.components:
        group: Any = sampling
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

    actual = _batch_checksum(root)
    if actual != manifest.batch_checksum:
        raise ValueError(
            f"Checkpoint at {path}: the walker batch failed its checksum "
            f"(expected {manifest.batch_checksum[:12]}…, got {actual[:12]}…). "
            "One of meta/, core/, or custom/ was modified or truncated after "
            "the manifest was written — positions, velocities, or walker "
            "identity can no longer be trusted."
        )

    return _read_batch(path, manifest, device), states, manifest


def _read_batch(
    path: str | Path, manifest: CheckpointManifest, device: torch.device | str
) -> Batch:
    """Reconstruct the walker batch, identity fields included.

    Parameters
    ----------
    path:
        Source store.
    manifest:
        The committed manifest, read for the expected walker count.
    device:
        Device for the restored batch.

    Returns
    -------
    Batch
        The restored batch.

    Raises
    ------
    ValueError
        If the store holds a different number of walkers than the manifest
        recorded.
    """
    reader = AtomicDataZarrReader(str(path))
    if len(reader) != manifest.num_graphs:
        raise ValueError(
            f"Checkpoint at {path} holds {len(reader)} walker(s) but its "
            f"manifest records {manifest.num_graphs}; the store is corrupt."
        )

    data_list = [AtomicData(**reader.read(i)[0]) for i in range(len(reader))]
    batch = Batch.from_data_list(data_list).to(device)

    root = zarr.open_group(str(path), mode="r")
    custom = root["custom"] if "custom" in root else None
    for field in _IDENTITY_FIELDS:
        if custom is not None and field in custom:
            values = np.asarray(custom[field][...])
            batch[field] = torch.from_numpy(np.ascontiguousarray(values)).to(
                device=device, dtype=torch.long
            )
    return batch
