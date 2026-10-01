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
"""Recommended distributed runtime manager for nvalchemi workflows."""

from __future__ import annotations

import contextlib
import logging
import os
import pickle
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Generic, TypeVar

import torch
from physicsnemo.distributed import (
    DistributedManager,
    PhysicsNeMoUninitializedDistributedManagerWarning,
)
from torch import distributed as dist

logger = logging.getLogger(__name__)

__all__ = [
    "DistributedManager",
    "PhysicsNeMoUninitializedDistributedManagerWarning",
    "CollectivePhase",
    "ProcessGroupContext",
    "collective_error_sync",
    "collective_device",
    "resolve_global_rank",
    "resolve_world_size",
]


@dataclass(frozen=True, init=False, slots=True)
class ProcessGroupContext:
    """Read-only topology and communication-device details for a supplied group.

    The context never creates, destroys, or otherwise owns the process group.
    Construct it after the caller has initialized the group and confirmed that
    this process is a member.

    Parameters
    ----------
    process_group : torch.distributed.ProcessGroup
        An initialized process group that includes this process.
    execution_device : torch.device or str, optional
        Communication-device hint. Gloo always uses CPU. NCCL resolves a CUDA
        hint first, then an initialized ``DistributedManager`` CUDA device,
        then the caller's current CUDA device.

    Raises
    ------
    TypeError
        If *process_group* is not a concrete PyTorch process group.
    RuntimeError
        If distributed is uninitialized, the group is unusable, or NCCL has no
        usable CUDA device.
    ValueError
        If the process is not a member or an explicit CUDA index is invalid.
    NotImplementedError
        If the group's backend is neither Gloo nor NCCL.
    """

    process_group: dist.ProcessGroup
    rank: int
    world_size: int
    backend: str
    collective_device: torch.device

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        *,
        execution_device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(process_group, dist.ProcessGroup):
            raise TypeError("process_group must be an actual torch ProcessGroup")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")

        try:
            rank = dist.get_rank(group=process_group)
            world_size = dist.get_world_size(group=process_group)
            backend_value = dist.get_backend(group=process_group)
        except RuntimeError as exc:
            raise RuntimeError("process_group is not initialized or usable") from exc
        if rank < 0 or world_size <= 0:
            raise ValueError("the current process is not a member of process_group")

        backend = _normalize_backend(backend_value)
        if backend not in {"gloo", "nccl"}:
            raise NotImplementedError(
                f"process-group backend {backend!r} is not supported"
            )
        device = _resolve_group_collective_device(backend, execution_device)

        object.__setattr__(self, "process_group", process_group)
        object.__setattr__(self, "rank", int(rank))
        object.__setattr__(self, "world_size", int(world_size))
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "collective_device", device)

    def global_rank(self, group_rank: int) -> int:
        """Translate a group-local rank to its global rank.

        Parameters
        ----------
        group_rank : int
            Rank within this context's process group.

        Returns
        -------
        int
            The corresponding global rank.

        Raises
        ------
        TypeError
            If *group_rank* is not a non-boolean integral value.
        ValueError
            If *group_rank* is outside this group's rank range.
        """
        if isinstance(group_rank, bool) or not isinstance(group_rank, Integral):
            raise TypeError("group_rank must be a non-boolean integer")
        local_rank = int(group_rank)
        if local_rank < 0 or local_rank >= self.world_size:
            raise ValueError(
                f"group_rank must be in [0, {self.world_size}), got {local_rank}"
            )
        return int(dist.get_global_rank(self.process_group, local_rank))

    @contextlib.contextmanager
    def communication_scope(self) -> Iterator[None]:
        """Select NCCL's resolved device for scoped communication, then restore it."""
        if self.backend == "gloo":
            yield
            return
        with torch.cuda.device(self.collective_device):
            yield


def _normalize_backend(backend: Any) -> str:
    """Return a stable lower-case backend name for PyTorch enum/string values."""
    value = getattr(backend, "value", backend)
    name = str(value).lower().rsplit(".", 1)[-1]
    return name


def _resolve_group_collective_device(
    backend: str, execution_device: torch.device | str | None
) -> torch.device:
    """Resolve a supplied group's device without changing CUDA's current device."""
    if backend == "gloo":
        return torch.device("cpu")

    hint: torch.device | None = None
    if execution_device is not None:
        try:
            hint = torch.device(execution_device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid execution_device {execution_device!r}") from exc

    if hint is not None and hint.type == "cuda":
        return _validated_cuda_device(hint, explicit=True)
    if hint is not None and hint.type != "cpu":
        raise ValueError("execution_device must be a CPU or CUDA device")

    if DistributedManager.is_initialized():
        manager_device = torch.device(DistributedManager().device)
        if manager_device.type == "cuda":
            return _validated_cuda_device(manager_device, explicit=False)

    if not torch.cuda.is_available():
        raise RuntimeError("NCCL requires a usable CUDA device")
    try:
        current_index = int(torch.cuda.current_device())
    except RuntimeError as exc:
        raise RuntimeError("NCCL requires a usable current CUDA device") from exc
    return _validated_cuda_device(torch.device("cuda", current_index), explicit=False)


def _validated_cuda_device(device: torch.device, *, explicit: bool) -> torch.device:
    """Check CUDA availability and index, preserving explicit-hint failures."""
    if not torch.cuda.is_available():
        if explicit:
            raise RuntimeError("the explicit CUDA execution device is unavailable")
        raise RuntimeError("NCCL requires a usable CUDA device")
    index = device.index
    if index is None:
        try:
            index = int(torch.cuda.current_device())
        except RuntimeError as exc:
            raise RuntimeError("NCCL requires a usable current CUDA device") from exc
    count = int(torch.cuda.device_count())
    if index < 0 or index >= count:
        if explicit:
            raise ValueError(
                f"explicit CUDA device index {index} is outside [0, {count})"
            )
        raise RuntimeError(f"CUDA device index {index} is outside [0, {count})")
    return torch.device("cuda", index)


_T = TypeVar("_T")


def _validated_phase_label(value: object) -> str:
    """Validate and copy a phase label into an exact built-in string."""
    if not isinstance(value, str):
        raise TypeError("phase must be a string")
    label = str.__str__(value)
    if not label.strip():
        raise ValueError("phase must be non-empty")
    return label


class CollectivePhase(Generic[_T]):
    """Mutable local phase and metadata state yielded by ``collective_error_sync``.

    Instances are created by the context-manager helper. Their ``records`` are
    available only after every local member has completed the shared exchange.
    """

    __slots__ = ("_phase", "_metadata", "_records")

    def __init__(self) -> None:
        raise TypeError(
            "CollectivePhase instances are created by collective_error_sync"
        )

    @classmethod
    def _create(cls, phase: str) -> CollectivePhase[Any]:
        instance = object.__new__(cls)
        instance._phase = phase
        instance._metadata = None
        instance._records = None
        return instance

    @property
    def phase(self) -> str:
        """Current local phase label used when reporting an ordinary failure."""
        return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        self._phase = _validated_phase_label(value)

    @property
    def metadata(self) -> _T | None:
        """Local control metadata to serialize once when the body exits."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: _T | None) -> None:
        self._metadata = value

    @property
    def records(self) -> tuple[_T | None, ...]:
        """Metadata snapshots ordered by group-local rank after successful exit."""
        if self._records is None:
            raise RuntimeError("collective records are available only after success")
        return self._records

    def _complete(self, records: tuple[_T | None, ...]) -> None:
        self._records = records


def _safe_exception_message(exc: Exception) -> str:
    """Format an exception message without letting a broken ``__str__`` escape."""
    try:
        return str.__str__(str(exc))
    except Exception:
        return "exception message unavailable"


def _encode_error_envelope(phase: str, exc: Exception) -> bytes:
    """Encode only primitive failure details; metadata is omitted on failure."""
    envelope = (
        "error",
        _validated_phase_label(phase),
        str.__str__(type(exc).__name__),
        _safe_exception_message(exc),
    )
    return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_envelope(encoded: bytes) -> tuple[Any, ...]:
    """Decode and minimally validate the primitive envelope from a peer."""
    # Process-group members exchange helper-generated envelopes as trusted peers.
    envelope = pickle.loads(encoded)  # noqa: S301
    if (
        not isinstance(envelope, tuple)
        or not envelope
        or envelope[0] not in {"success", "error"}
    ):
        raise RuntimeError("received an invalid collective error envelope")
    if envelope[0] == "success":
        if (
            len(envelope) != 3
            or not isinstance(envelope[1], str)
            or not isinstance(envelope[2], bytes)
        ):
            raise RuntimeError("received an invalid success envelope")
    elif (
        len(envelope) != 4
        or not isinstance(envelope[1], str)
        or not isinstance(envelope[2], str)
        or not isinstance(envelope[3], str)
    ):
        raise RuntimeError("received an invalid error envelope")
    return envelope


@contextlib.contextmanager
def collective_error_sync(
    context: ProcessGroupContext,
    *,
    phase: str,
    error_type: type[Exception] = RuntimeError,
) -> Iterator[CollectivePhase[Any]]:
    """Agree ordinary local failures and exchange successful control metadata.

    Every process-group member must enter matching calls in the same order.
    Exactly one object all-gather carries either local failure details or the
    pickled metadata bytes. The lowest failing group-local rank supplies the
    shared diagnostic.

    Parameters
    ----------
    context : ProcessGroupContext
        Initialized group context shared by this rank.
    phase : str
        Initial non-empty phase label; the yielded object's label can change.
    error_type : type[Exception], default RuntimeError
        Exception class used for the same distributed diagnostic on every rank.

    Yields
    ------
    CollectivePhase
        Mutable phase and metadata state; ``records`` is populated only after
        a successful group exchange.

    Raises
    ------
    TypeError, ValueError
        If the context, phase, or configured exception type is invalid.
    Exception
        The configured class if a rank's body or metadata serialization fails.
    """
    if not isinstance(context, ProcessGroupContext):
        raise TypeError("context must be a ProcessGroupContext")
    phase = _validated_phase_label(phase)
    if not isinstance(error_type, type) or not issubclass(error_type, Exception):
        raise TypeError("error_type must be an Exception subclass")
    try:
        error_type("message validation")
    except Exception as exc:
        raise TypeError("error_type must accept a message") from exc

    state = CollectivePhase._create(phase)
    local_exception: Exception | None = None
    try:
        yield state
    except Exception as exc:
        local_exception = exc

    if local_exception is None:
        try:
            metadata_bytes = pickle.dumps(
                state.metadata, protocol=pickle.HIGHEST_PROTOCOL
            )
        except Exception as exc:
            local_exception = exc

    if local_exception is None:
        encoded = pickle.dumps(
            ("success", state.phase, metadata_bytes), protocol=pickle.HIGHEST_PROTOCOL
        )
    else:
        encoded = _encode_error_envelope(state.phase, local_exception)

    gathered: list[bytes | None] = [None] * context.world_size
    with context.communication_scope():
        dist.all_gather_object(gathered, encoded, group=context.process_group)

    decoded = [_decode_envelope(item) for item in gathered]
    failures = [(rank, item) for rank, item in enumerate(decoded) if item[0] == "error"]
    if failures:
        failing_rank, envelope = failures[0]
        _, failed_phase, exception_type, message = envelope
        diagnostic = (
            f"distributed {failed_phase} failed at rank={failing_rank} "
            f"phase={failed_phase}: {exception_type}: {message}"
        )
        shared_exception = error_type(diagnostic)
        if failing_rank == context.rank and local_exception is not None:
            raise shared_exception from local_exception
        raise shared_exception from None

    # Metadata is application-defined and exchanged only among trusted members.
    records = tuple(pickle.loads(item[2]) for item in decoded)  # noqa: S301
    state._complete(records)


def resolve_world_size() -> int:
    """Resolve world size from PhysicsNeMo, torch.distributed, or environment."""
    if DistributedManager.is_initialized():
        return int(DistributedManager().world_size)
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return world_size


def resolve_global_rank(global_rank: int | None = None) -> int:
    """Resolve global rank from an explicit value, distributed state, or env."""
    if global_rank is not None:
        return int(global_rank)
    if DistributedManager.is_initialized():
        return int(DistributedManager().rank)
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    rank = int(os.environ.get("RANK", 0))
    return rank


def collective_device(fallback: torch.device | str = "cpu") -> torch.device:
    """Resolve the rank-local device for distributed tensor collectives."""
    if dist.is_available() and dist.is_initialized():
        try:
            backend = dist.get_backend()
        except RuntimeError:
            backend = None
        if backend != "nccl":
            return torch.device("cpu")
    if DistributedManager.is_initialized():
        device = torch.device(DistributedManager().device)
    elif torch.cuda.is_available():
        index = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", index)
    else:
        device = torch.device(fallback)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


# Full-precision fp32 lands far below this; reduced precision far above.
_REDUCED_PRECISION_THRESHOLD = 1e-5
_warned_reduced_precision = False


def pin_fp32() -> None:
    """Force full-precision fp32 matmul and convolution.

    A distributed forward pads to different shapes than a single-process one, so
    under reduced-precision fp32 (TF32) the backend can pick a different kernel
    for each and the results separate by far more than fp32 rounding. Also sets
    ``NVIDIA_TF32_OVERRIDE``, which is what reaches ``mp.spawn`` / ``torchrun``
    workers -- they inherit the environment, not the torch flags. Call before the
    process builds a CUDA context.

    Returns
    -------
    None
    """
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for holder in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(holder, "fp32_precision"):
            try:
                holder.fp32_precision = "ieee"
            except Exception:  # pragma: no cover - varies by torch version
                # Non-fatal: the primary flags above already pin precision, and
                # this attribute only exists on some torch versions.
                logger.warning("could not set fp32_precision", exc_info=True)


@contextlib.contextmanager
def pinned_fp32() -> Iterator[None]:
    """Pin full-precision fp32 for the duration of the block, then restore.

    The scoped counterpart to :func:`pin_fp32`, for callers that need one
    comparison at full precision without changing the rest of the process.
    Prefer :func:`pin_fp32` for a whole run: it also sets the environment
    variable that ``mp.spawn`` / ``torchrun`` workers inherit, and restoring
    that on exit would unpin the workers.

    Yields
    ------
    None
    """
    saved: list[tuple[Any, str, Any]] = [
        (
            torch.backends.cuda.matmul,
            "allow_tf32",
            torch.backends.cuda.matmul.allow_tf32,
        ),
        (torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32),
    ]
    saved.extend(
        (holder, "fp32_precision", holder.fp32_precision)
        for holder in (torch.backends.cuda.matmul, torch.backends.cudnn)
        if hasattr(holder, "fp32_precision")
    )
    # Reading the global precision raises once legacy (``allow_tf32``) and new
    # (``fp32_precision``) APIs have both been written, which ``pin_fp32`` does.
    try:
        saved_precision = torch.get_float32_matmul_precision()
    except RuntimeError:  # pragma: no cover - depends on prior calls
        saved_precision = None
    try:
        pin_fp32()
        yield
    finally:
        for holder, attr, value in saved:
            try:
                setattr(holder, attr, value)
            except Exception:  # pragma: no cover - varies by torch version
                logger.warning("could not restore %s", attr, exc_info=True)
        if saved_precision is not None:
            torch.set_float32_matmul_precision(saved_precision)


def _is_reduced_precision(device: str | torch.device | None = None) -> bool:
    """Whether fp32 matmul currently runs on the reduced-precision path.

    Measured rather than read off the backend flags: which kernel runs depends on
    torch version, backend and shape.
    """
    if not torch.cuda.is_available():
        return False
    try:
        gen = torch.Generator(device="cpu").manual_seed(0)
        a = torch.randn(512, 512, generator=gen).to(device or "cuda")
        b = torch.randn(512, 512, generator=gen).to(device or "cuda")
        ref = a.double() @ b.double()
        err = (((a @ b).double() - ref).abs().max() / ref.abs().max()).item()
    except Exception:  # pragma: no cover - a probe must not break a forward
        logger.debug("fp32 precision probe failed", exc_info=True)
        return False
    return err > _REDUCED_PRECISION_THRESHOLD


def warn_if_reduced_precision(
    device: str | torch.device | None = None,
) -> None:
    """Warn once per process if reduced-precision fp32 is in force."""
    global _warned_reduced_precision
    if _warned_reduced_precision or not torch.cuda.is_available():
        return
    _warned_reduced_precision = True
    if not _is_reduced_precision(device):
        return
    warnings.warn(
        "Reduced-precision fp32 (TF32) is enabled; distributed and "
        "single-process results can then differ by much more than fp32 rounding. "
        "Call nvalchemi.distributed.pin_fp32() before building models if this "
        "run must match a reference.",
        UserWarning,
        stacklevel=3,
    )
