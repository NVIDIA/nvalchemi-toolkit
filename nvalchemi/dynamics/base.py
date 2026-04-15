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
"""
Base classes and protocols for molecular dynamics simulations.

This module provides the foundational abstractions for running dynamics
simulations, including hook protocols for extensibility, the base
dynamics class that coordinates model evaluation with integrator updates,
and the ``FusedStage`` class for fusing multiple dynamics stages on a
single GPU with shared batch and forward pass.

Inheritance structure::

    object
    └── _CommunicationMixin          # inter-rank communication base
        └── BaseDynamics(_CommunicationMixin)
            └── FusedStage(BaseDynamics)

``BaseDynamics`` inherits from ``_CommunicationMixin``, so all dynamics
subclasses automatically have communication capabilities for pipeline
execution without needing explicit multiple inheritance.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Callable, Sequence
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
)

import torch
from jaxtyping import Bool
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from torch import distributed as dist

from nvalchemi._typing import AtomsLike, ModelOutputs
from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.hooks._registry import HookRegistryMixin
from nvalchemi.models.base import BaseModelMixin

if TYPE_CHECKING:
    from nvalchemi.dynamics.sampler import SizeAwareSampler
    from nvalchemi.dynamics.sinks import DataSink


__all__ = [
    "Hook",
    "DynamicsStage",
    "ConvergenceHook",
    "DistributedPipeline",
    "BufferConfig",
]


class BufferConfig(BaseModel):
    """Buffer capacities for pipeline communication.

    Required by :class:`_CommunicationMixin` whenever the stage
    participates in inter-rank communication (i.e. ``prior_rank`` or
    ``next_rank`` is set).  Buffers are lazily created via
    ``Batch.empty()`` on the first simulation step, once a concrete
    batch is available as a template.

    Attributes
    ----------
    num_systems : int
        Maximum number of graphs the buffer can hold.
    num_nodes : int
        Total node (atom) capacity across all graphs.
    num_edges : int
        Total edge capacity across all graphs.
    """

    num_systems: Annotated[
        int, Field(ge=0, description="Maximum number of graphs the buffer can hold.")
    ]
    num_nodes: Annotated[
        int, Field(ge=0, description="Total node (atom) capacity across all graphs.")
    ]
    num_edges: Annotated[
        int, Field(ge=0, description="Total edge capacity across all graphs.")
    ]


class DynamicsStage(Enum):
    """
    Enumeration of stages in the dynamics step where hooks can fire.

    Each stage corresponds to a specific point in the simulation step,
    allowing hooks to be triggered before or after key operations.

    Attributes
    ----------
    BEFORE_STEP : int
        Fired at the very beginning of a step, before any operations.
    BEFORE_PRE_UPDATE : int
        Fired before the pre_update (first half of integrator) is called.
    AFTER_PRE_UPDATE : int
        Fired after the pre_update completes.
    BEFORE_COMPUTE : int
        Fired before the model forward pass (force/energy computation).
    AFTER_COMPUTE : int
        Fired after the model forward pass completes.
    BEFORE_POST_UPDATE : int
        Fired before the post_update (second half of integrator) is called.
    AFTER_POST_UPDATE : int
        Fired after the post_update completes.
    AFTER_STEP : int
        Fired at the very end of a step, after all operations.
    ON_CONVERGE : int
        Fired when a convergence criterion is met (e.g., for optimizers).
    """

    BEFORE_STEP = 0
    BEFORE_PRE_UPDATE = 1
    AFTER_PRE_UPDATE = 2
    BEFORE_COMPUTE = 3
    AFTER_COMPUTE = 4
    BEFORE_POST_UPDATE = 5
    AFTER_POST_UPDATE = 6
    AFTER_STEP = 7
    ON_CONVERGE = 8


class _ConvergenceCriterion(BaseModel):
    """A single convergence criterion evaluated against a tensor key on ``Batch``.

    This is an internal model and should not be instantiated directly by
    users.  Instead, pass ``dict`` mappings to :class:`ConvergenceHook`,
    which will validate and construct ``_ConvergenceCriterion`` instances
    automatically.

    The evaluation pipeline is:

    1.  Retrieve ``getattr(batch, key)``; raise ``KeyError`` if absent.
    2.  If ``custom_op`` is provided, delegate entirely to it and return.
    3.  If the tensor is node-level (its first dimension matches
        ``batch.num_nodes``), scatter-reduce it to graph-level using
        ``batch.batch_idx`` as the group index.
    4.  Otherwise the tensor is assumed to be graph-level and is squeezed
        to 1-D ``(B,)`` if it has a trailing singleton dimension.
    5.  If ``reduce_op`` is not ``None``, apply the requested reduction
        along ``reduce_dims`` **before** step 3/4 (i.e. within each
        node / graph entry).
    6.  Compare the resulting ``(B,)`` tensor against ``threshold``.

    Attributes
    ----------
    key : str
        Tensor key to measure convergence against (e.g. ``"forces"``).
    threshold : float
        Convergence threshold; values ≤ this are considered converged.
    reduce_dims : int | list[int]
        Dimension(s) to reduce over when ``reduce_op`` is not ``None``.
        Defaults to ``-1``.
    reduce_op : {``"min"``, ``"max"``, ``"norm"``, ``"mean"``, ``"sum"``} or ``None``
        Reduction applied to the raw tensor before the graph-level
        aggregation.  ``None`` (default) skips this step and expects
        the key to already be at graph-level or to be a node-level
        vector that will be scatter-reduced.
    custom_op : Callable[[torch.Tensor], Bool[Tensor, " B"]] | None
        Custom callable that receives the raw tensor and must return
        a boolean ``(B,)`` mask.  When provided, ``reduce_op``,
        ``reduce_dims``, and ``threshold`` are ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: Annotated[str, Field(description="Tensor key to measure convergence against.")]
    threshold: Annotated[
        float,
        Field(
            description="Threshold for convergence; values"
            " smaller than or equal to `threshold` are considered converged."
        ),
    ]
    reduce_dims: Annotated[
        int | list[int], Field(description="Dimension(s) to reduce over.")
    ] = -1
    reduce_op: Annotated[
        Literal["min", "max", "norm", "mean", "sum"] | None,
        Field(
            description="Operation used to reduce non-scalar criteria."
            " None skips reductions and expects the key to already be"
            " graph-level or a node-level vector suitable for"
            " scatter-reduce."
        ),
    ] = None
    custom_op: Annotated[
        Callable[[torch.Tensor], Bool[torch.Tensor, " B"]] | None,  # noqa: F722, F821
        Field(
            description="Custom operation that wraps the convergence"
            " logic and returns a bool tensor indicating which samples"
            " have converged."
        ),
    ] = None

    def __repr__(self) -> str:
        """Return a human-readable summary of this criterion."""
        if self.custom_op is not None:
            op_name = getattr(self.custom_op, "__name__", repr(self.custom_op))
            return f"_ConvergenceCriterion(key={self.key!r}, custom_op={op_name})"
        parts = [f"key={self.key!r}", f"threshold={self.threshold}"]
        if self.reduce_op is not None:
            parts.append(f"reduce_op={self.reduce_op!r}")
            parts.append(f"reduce_dims={self.reduce_dims!r}")
        return f"_ConvergenceCriterion({', '.join(parts)})"

    def _reduce_within_entry(self, target: torch.Tensor) -> torch.Tensor:
        """Apply ``reduce_op`` along ``reduce_dims`` within each entry.

        Parameters
        ----------
        target : torch.Tensor
            The raw tensor retrieved from the batch.

        Returns
        -------
        torch.Tensor
            The reduced tensor.  If ``reduce_op`` is ``None``, the
            input is returned unchanged.
        """
        if self.reduce_op is None:
            return target
        match self.reduce_op:
            case "min":
                return torch.amin(target, self.reduce_dims)
            case "max":
                return torch.amax(target, self.reduce_dims)
            case "norm":
                return torch.linalg.vector_norm(target, dim=self.reduce_dims)
            case "mean":
                return torch.mean(target, dim=self.reduce_dims)
            case "sum":
                return torch.sum(target, dim=self.reduce_dims)
        # Unreachable because of the Literal type, but satisfies the
        # type checker and gives a clear error for bad runtime values.
        raise ValueError(f"Unknown reduce_op: {self.reduce_op!r}")

    @staticmethod
    def _scatter_reduce_to_graph(
        values: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Scatter-reduce a 1-D node-level tensor to graph-level via max.

        Parameters
        ----------
        values : torch.Tensor
            1-D tensor of shape ``(V,)`` with per-node values.
        batch_idx : torch.Tensor
            Integer tensor mapping each node to its graph index.
        num_graphs : int
            Number of graphs in the batch.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(B,)`` with per-graph reduced values
            (using ``max`` as the scatter operation).
        """
        out = torch.full(
            (num_graphs,), float("-inf"), dtype=values.dtype, device=values.device
        )
        out.scatter_reduce_(0, batch_idx, values, reduce="amax", include_self=False)
        return out

    def __call__(self, batch: Batch) -> Bool[torch.Tensor, " B"]:  # noqa: F722, F821
        """Evaluate this criterion against a batch.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.

        Returns
        -------
        Bool[Tensor, " B"]
            Per-sample boolean mask where ``True`` indicates that the
            sample satisfies this convergence criterion.

        Raises
        ------
        KeyError
            If ``self.key`` is not present on ``batch``.
        """
        target: torch.Tensor | None = getattr(batch, self.key, None)
        if target is None:
            available = list(batch.model_dump(exclude_none=True))
            raise KeyError(
                "Key for convergence check not found;"
                f" expected={self.key!r}, available={available}"
            )

        if self.custom_op is not None:
            return self.custom_op(target)

        target = self._reduce_within_entry(target)

        is_node_level = (
            target.shape[0] == batch.num_nodes and batch.num_nodes != batch.num_graphs
        )

        if is_node_level:
            if target.dim() > 1:
                target = target.view(target.shape[0], -1).amax(dim=-1)
            reduced = self._scatter_reduce_to_graph(
                target, batch.batch_idx, batch.num_graphs
            )
        else:
            reduced = target.squeeze(-1) if target.dim() == 2 else target

        return reduced <= self.threshold


CommMode: TypeAlias = Literal["sync", "async_recv", "fully_async"]


class _CommunicationMixin:
    """Base class providing inter-rank communication and buffer management.

    ``BaseDynamics`` inherits from this class, so all dynamics subclasses
    automatically have communication capabilities for pipeline execution.

    This class manages active batch buffers, overflow sinks, and inter-rank
    communication for distributed pipeline execution.

    Parameters
    ----------
    prior_rank : int | None
        Rank to receive data from.  ``None`` marks this stage as the
        first in its sub-pipeline (no upstream).  Defaults to ``-1``
        (unset), which tells :meth:`DistributedPipeline.setup` to
        auto-assign based on stage ordering.  Set explicitly to
        ``None`` or a rank integer to prevent auto-assignment.
    next_rank : int | None
        Rank to send graduated samples to.  ``None`` marks this stage
        as the last in its sub-pipeline (no downstream).  Defaults to
        ``-1`` (unset), with the same auto-assignment semantics as
        ``prior_rank``.
    sinks : list[DataSink] | None
        Priority-ordered overflow sinks.
    active_batch : Batch | None
        The currently active working batch.
    max_batch_size : int
        Maximum samples in the active batch.
    done : bool
        Whether this stage has no more work.
    device_type : str
        Device type string (e.g., ``"cuda"``, ``"cpu"``).
    comm_mode : CommMode
        Communication mode for inter-rank buffer synchronization.
        One of ``"sync"``, ``"async_recv"``, or ``"fully_async"``.
        Default ``"sync"``.
    buffer_config : BufferConfig
        Pre-allocation capacities for send/recv communication buffers.
        **Required** when ``prior_rank`` or ``next_rank`` is set to a
        valid rank.  A ``ValueError`` is raised at construction if
        omitted for a stage that has neighbors.
    **kwargs
        Forwarded to the next class in the MRO (cooperative init).

    Attributes
    ----------
    prior_rank : int | None
        Rank of the previous pipeline stage.  ``-1`` means unset
        (will be auto-assigned by :meth:`DistributedPipeline.setup`),
        ``None`` means no upstream, and a non-negative integer is the
        explicit source rank.
    next_rank : int | None
        Rank of the next pipeline stage, with the same conventions
        as ``prior_rank``.
    sinks : list[DataSink]
        Overflow sinks in priority order.
    active_batch : Batch | None
        Current working batch.
    max_batch_size : int
        Maximum active batch capacity.
    done : bool
        Whether this stage is finished.
    sampler : SizeAwareSampler | None
        Size-aware sampler for inflight batching, or ``None`` for
        external batch handling (i.e. the typical looping over
        dataloader approach). Defaults to ``None``.
    refill_frequency : int
        How often to check and refill graduated samples from the
        sampler; no-op if ``sampler`` is not provided.
    device_type : str
        Device type string (e.g., ``"cuda"``, ``"cpu"``).
    comm_mode : CommMode
        Communication mode for inter-rank buffer synchronization.
    buffer_config : BufferConfig | None
        Buffer capacities, or ``None`` for isolated stages.
    _pending_recv_handle : Any
        Stored ``irecv`` handle when receive is deferred (non-sync modes).
        ``None`` when no receive is pending.
    _pending_send_handle : Any
        Stored ``isend`` handle when send is deferred (``"fully_async"``).
        ``None`` when no send is pending.
    _stream : torch.cuda.Stream | None
        The CUDA stream created when entering the context manager.
        ``None`` when outside a ``with`` block or on non-CUDA devices.
    _stream_ctx : torch.cuda.StreamContext | None
        The active stream context wrapping ``_stream``.
        ``None`` when outside a ``with`` block or on non-CUDA devices.

    Examples
    --------
    >>> from nvalchemi.dynamics.base import BaseDynamics, BufferConfig
    >>> cfg = BufferConfig(num_systems=10, num_nodes=500, num_edges=2000)
    >>> dyn = BaseDynamics(model=model, prior_rank=0, buffer_config=cfg, max_batch_size=50)
    >>> dyn.is_first_stage
    False
    """

    def __init__(
        self,
        *,
        prior_rank: int | None = -1,
        next_rank: int | None = -1,
        sinks: Sequence[DataSink] | None = None,
        active_batch: Batch | None = None,
        max_batch_size: int = 100,
        done: bool = False,
        sampler: SizeAwareSampler | None = None,
        refill_frequency: int = 1,
        device_type: str | None = None,
        comm_mode: CommMode = "async_recv",
        buffer_config: BufferConfig | None = None,
        debug_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the communication mixin.

        Parameters
        ----------
        prior_rank : int | None, optional
            Rank to receive data from (previous stage). Default None.
        next_rank : int | None, optional
            Rank to send graduated samples to (next stage). Default None.
        sinks : Sequence[DataSink] | None, optional
            Priority-ordered overflow sinks. Default None (empty list).
        active_batch : Batch | None, optional
            The currently active working batch. Default None.
        max_batch_size : int, optional
            Maximum samples in the active batch. Default 100.
        done : bool, optional
            Whether this stage has no more work. Default False.
        sampler : SizeAwareSampler | None, optional
            Size-aware sampler for inflight batching. When provided,
            enables inflight batching where graduated (converged/finished)
            samples are automatically replaced with fresh ones from the dataset.
            Default ``None``, which expects batches to come from dataloaders.
        refill_frequency : int, optional
            How often (in steps) to check for graduated samples and
            request replacements from the sampler. Only used when
            ``sampler`` is not None. Default 1.
        device_type : str, optional
            Device type string (e.g., ``"cuda"``, ``"cpu"``). Defaults
            to ``None``, which will perform auto placement.
        comm_mode : CommMode, optional
            Communication mode controlling blocking behavior of inter-rank
            buffer synchronization.  ``"sync"`` (default) blocks on receive
            immediately.  ``"async_recv"`` defers the receive wait until
            ``_complete_pending_recv`` is called.  ``"fully_async"``
            additionally stores the send handle and drains it at the start
            of the next ``_prestep_sync_buffers`` call.
        buffer_config : BufferConfig | None, optional
            Pre-allocation capacities for send/recv buffers.  Buffers
            are created lazily via ``Batch.empty()`` on the first step
            using the first concrete batch as a template.  **Required**
            when ``prior_rank`` or ``next_rank`` is a valid rank;
            raises ``ValueError`` otherwise.  Default ``None`` (only
            valid for isolated stages with no neighbors).
        debug_mode : bool, optional
            When ``True``, emit detailed ``loguru.debug`` diagnostics
            for inter-rank communication. Default ``False``.
        **kwargs : Any
            Forwarded to the next class in the MRO (cooperative init).
        """
        super().__init__(**kwargs)
        self.prior_rank = prior_rank
        self.next_rank = next_rank
        self.sinks: list[DataSink] = list(sinks) if sinks is not None else []
        self.active_batch = active_batch
        self.max_batch_size = max_batch_size
        self.done = done
        self.sampler = sampler
        self.refill_frequency = refill_frequency
        if not device_type:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = device_type
        if comm_mode not in ("sync", "async_recv", "fully_async"):
            raise ValueError(
                f"Invalid comm_mode={comm_mode!r}. "
                f"Expected one of: 'sync', 'async_recv', 'fully_async'."
            )
        self.comm_mode: CommMode = comm_mode
        self._pending_recv_handle: Any = None
        self._pending_send_handle: Any = None
        self._stream: torch.cuda.Stream | None = None
        self._stream_ctx: torch.cuda.StreamContext | None = None
        if isinstance(buffer_config, dict):
            buffer_config = BufferConfig(**buffer_config)
        if buffer_config is not None and not isinstance(buffer_config, BufferConfig):
            raise TypeError(
                f"Buffer configuration invalid; got a {type(buffer_config)} object."
            )
        self.buffer_config = buffer_config
        if self.has_neighbor and self.buffer_config is None:
            raise ValueError(
                "buffer_config is required when prior_rank or next_rank is set. "
                "Pre-allocated buffers are mandatory for inter-rank communication."
            )
        self.send_buffer: Batch | None = None
        self.recv_buffer: Batch | None = None
        self._recv_template: Batch | None = None
        self.debug_mode = debug_mode

    @property
    def has_neighbor(self) -> bool:
        """Convenient property to see if rank is isolated"""
        next_rank = self.next_rank is not None and self.next_rank != -1
        prior_rank = self.prior_rank is not None and self.prior_rank != -1
        return next_rank or prior_rank

    def _ensure_buffers(self, template: Batch) -> None:
        """Lazily create send/recv buffers from the first concrete batch.

        Called automatically at the start of the first communication
        step when ``buffer_config`` is set.  Uses *template* (the first
        real batch) to determine attribute keys, dtypes, and trailing
        shapes for ``Batch.empty()``.

        Parameters
        ----------
        template : Batch
            A concrete batch to use as a template for buffer creation.
        """
        if self.buffer_config is None:
            return
        cfg = self.buffer_config
        if self.send_buffer is None and self.next_rank is not None:
            self.send_buffer = Batch.empty(
                num_systems=cfg.num_systems,
                num_nodes=cfg.num_nodes,
                num_edges=cfg.num_edges,
                template=template,
                device=self.device,
            )
        if self.recv_buffer is None and self.prior_rank is not None:
            self.recv_buffer = Batch.empty(
                num_systems=cfg.num_systems,
                num_nodes=cfg.num_nodes,
                num_edges=cfg.num_edges,
                template=template,
                device=self.device,
            )

    @property
    def is_final_stage(self) -> bool:
        """Return whether this is the last stage in the pipeline.

        Returns
        -------
        bool
            ``True`` if ``next_rank`` is ``None``.
        """
        return self.next_rank is None

    @property
    def is_first_stage(self) -> bool:
        """Return whether this is the first stage in the pipeline.

        Returns
        -------
        bool
            ``True`` if ``prior_rank`` is ``None``.
        """
        return self.prior_rank is None

    @property
    def inflight_mode(self) -> bool:
        """Return whether inflight batching is enabled.

        Inflight batching is enabled when a sampler is configured.
        In this mode, graduated samples are automatically replaced with
        fresh ones from the dataset.

        Returns
        -------
        bool
            ``True`` if ``sampler`` is not ``None``.
        """
        return self.sampler is not None

    @property
    def local_rank(self) -> int:
        """Get the node-local rank for this process.

        Returns ``0`` when ``torch.distributed`` is not initialized.

        Returns
        -------
        int
            The local rank on this node.
        """
        if dist.is_initialized():
            return dist.get_node_local_rank()
        return 0

    @property
    def device(self) -> torch.device:
        """Compute the torch device for this rank.

        For CUDA-like devices (``"cuda"``, ``"xpu"``, etc.), returns
        ``torch.device(f"{device_type}:{local_rank}")``.  For CPU,
        returns ``torch.device("cpu")`` because PyTorch CPU devices
        do not use ordinal indices — all ranks share a single CPU
        device.

        Returns
        -------
        torch.device
            Device for this rank.
        """
        match self.device_type:
            case "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "Requested CUDA device type, but not available to PyTorch."
                    )
                return torch.device(f"cuda:{self.local_rank}")
            case "cpu":
                return torch.device("cpu")
            case _:
                try:
                    device = torch.device(f"{self.device_type}:{self.local_rank}")
                    return device
                except Exception as e:
                    raise RuntimeError(
                        f"Unable to create device={self.device_type}:{self.local_rank}"
                        f" with exception: {e}"
                    )

    @property
    def stream(self) -> torch.cuda.Stream | None:
        """Return the active CUDA stream, if any.

        Returns ``None`` when outside a ``with`` block or on non-CUDA
        devices.

        Returns
        -------
        torch.cuda.Stream | None
            The CUDA stream created by ``__enter__``, or ``None``.
        """
        return self._stream

    def __enter__(self) -> _CommunicationMixin:
        """Enter the stream context manager.

        On CUDA devices, creates a new ``torch.cuda.Stream`` and enters
        a ``torch.cuda.StreamContext`` so that all subsequent GPU
        operations execute on the dedicated stream.  On non-CUDA devices
        this is a no-op.

        Returns
        -------
        _CommunicationMixin
            This instance.
        """
        if self.device_type == "cuda" and torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device=self.device)
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the stream context manager.

        Exits the ``torch.cuda.StreamContext`` (if one was entered) and
        clears the stored stream references.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type, if any.
        exc_val : BaseException | None
            Exception value, if any.
        exc_tb : Any
            Exception traceback, if any.
        """
        if self._stream_ctx is not None:
            self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stream = None
        self._stream_ctx = None

    @property
    def active_batch_size(self) -> int:
        """Return the number of samples currently in the active batch.

        Returns
        -------
        int
            Number of graphs in the active batch, or 0 if no batch.
        """
        if self.active_batch is None:
            return 0
        return self.active_batch.num_graphs or 0

    @property
    def active_batch_has_room(self) -> bool:
        """Return whether the active batch can accept more samples.

        Returns
        -------
        bool
            ``True`` if the active batch is below ``max_batch_size``.
        """
        return self.active_batch_size < self.max_batch_size

    @property
    def room_in_active_batch(self) -> int:
        """Return the number of additional samples the active batch can hold.

        Returns
        -------
        int
            Remaining capacity.
        """
        return max(0, self.max_batch_size - self.active_batch_size)

    @property
    def _send_buffer_capacity(self) -> int:
        """Return the number of additional graphs the send buffer can accept.

        When ``send_buffer`` is ``None`` (no pre-allocated buffer), returns
        ``sys.maxsize`` to indicate no capacity constraint—the system sends
        live batches directly without a fixed-size buffer.

        Returns
        -------
        int
            Remaining capacity in the send buffer, or ``sys.maxsize`` when
            there is no pre-allocated send buffer.
        """
        if self.send_buffer is None:
            return sys.maxsize
        return self.send_buffer.system_capacity - self.send_buffer.num_graphs

    def _buffer_to_batch(self, incoming_batch: Batch) -> None:
        """Route received data into the active batch or overflow sinks.

        If the active batch has room, samples are appended directly.
        Otherwise, excess samples are written to overflow sinks in
        priority order.

        Parameters
        ----------
        incoming_batch : Batch
            Batch of samples received from the prior stage.
        """
        if incoming_batch.num_graphs == 0:
            return

        if self.active_batch is None:
            if incoming_batch.num_graphs <= self.max_batch_size:
                # reform the batch without padding
                self.active_batch = Batch.from_data_list(
                    incoming_batch.to_data_list(), device=incoming_batch.device
                )
            else:
                # slice out samples that will fit in the active batch
                # and move the rest to overflow
                data_list = incoming_batch.to_data_list()
                fit = data_list[: self.max_batch_size]
                overflow = data_list[self.max_batch_size :]
                self.active_batch = Batch.from_data_list(
                    fit, device=incoming_batch.device
                )
                self._overflow_to_sinks(
                    Batch.from_data_list(overflow, device=incoming_batch.device)
                )
            return

        room = self.room_in_active_batch
        if room <= 0:
            self._overflow_to_sinks(incoming_batch)
            return

        data_list = incoming_batch.to_data_list()
        if len(data_list) <= room:
            existing = self.active_batch.to_data_list()
            self.active_batch = Batch.from_data_list(
                existing + data_list, device=incoming_batch.device
            )
        else:
            fit = data_list[:room]
            overflow = data_list[room:]
            existing = self.active_batch.to_data_list()
            self.active_batch = Batch.from_data_list(
                existing + fit, device=incoming_batch.device
            )
            self._overflow_to_sinks(
                Batch.from_data_list(overflow, device=incoming_batch.device)
            )

    def _recv_to_batch(self, incoming: Batch) -> None:
        """Stage incoming data through the recv buffer into the active batch.

        When ``recv_buffer`` is available, copies *incoming* data into the
        pre-allocated receive buffer via :meth:`Batch.put`, then routes the
        buffer contents into the active batch via :meth:`_buffer_to_batch`.
        When ``recv_buffer`` is ``None``, falls back to routing *incoming*
        directly.

        Parameters
        ----------
        incoming : Batch
            Batch received from the prior stage (via ``irecv`` / ``wait``).
        """
        if incoming.num_graphs > 0 and self._recv_template is None:
            self._recv_template = incoming
        if self.recv_buffer is not None and incoming.num_graphs > 0:
            mask = torch.ones(incoming.num_graphs, dtype=torch.bool, device=self.device)
            self.recv_buffer.put(incoming, mask=mask)
            self._buffer_to_batch(self.recv_buffer)
            self.recv_buffer.zero()
        else:
            self._buffer_to_batch(incoming)

    def _overflow_to_sinks(
        self, batch: Batch, mask: torch.Tensor | None = None
    ) -> None:
        """Write overflow samples to the first sink with available capacity.

        Parameters
        ----------
        batch : Batch
            Overflow samples to store.
        mask : torch.Tensor | None, optional
            Boolean mask for selective writing. Forwarded to sink.write().

        Raises
        ------
        RuntimeError
            If no sink has capacity for the overflow.
        """
        for sink in self.sinks:
            if not sink.is_full:
                sink.write(batch, mask=mask)
                return
        raise RuntimeError(
            f"All sinks are full. Cannot store {batch.num_graphs} overflow samples."
        )

    def _batch_to_buffer(self, mask: torch.Tensor) -> None:
        """Move graduated samples from the active batch into the send buffer.

        Uses ``send_buffer.put`` to copy samples where *mask* is ``True``
        into the pre-allocated send buffer, then trims the active batch
        to a new tight :class:`~nvalchemi.data.Batch` without the
        graduated samples (or *None* if all were graduated).

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape ``(active_batch.num_graphs,)`` where
            ``True`` marks a graduated (converged) sample.

        Raises
        ------
        RuntimeError
            If ``active_batch`` or ``send_buffer`` is ``None``.
        """
        if self.active_batch is None:
            raise RuntimeError("No active batch to extract from.")
        if self.send_buffer is None:
            raise RuntimeError("No send buffer to write to.")

        self.send_buffer.put(self.active_batch, mask=mask)
        self.active_batch = self.active_batch.trim(copied_mask=mask)

    def _drain_sinks_to_batch(self) -> None:
        """Pull samples from overflow sinks into the active batch.

        Iterates through sinks in priority order.  For each non-empty
        sink, drains its contents and routes them into the active batch
        via :meth:`_buffer_to_batch`.  Stops early when the active batch
        has no more room.

        If the drained batch is larger than the remaining room,
        ``_buffer_to_batch`` handles the partial-fit logic (accepts what
        fits, overflows the rest back to sinks).

        Notes
        -----
        Called by ``_prestep_sync_buffers`` after processing incoming
        data from the prior rank, to backfill any remaining capacity
        with previously overflowed samples.
        """
        for sink in self.sinks:
            if self.room_in_active_batch <= 0:
                break
            if len(sink) == 0:
                continue
            overflow = sink.drain()
            self._buffer_to_batch(overflow)

    def _prestep_sync_buffers(self) -> None:
        """Synchronize buffers before a dynamics step.

        If this stage has a prior rank, zeros the send buffer and receive
        buffer, then receives data from the prior stage via ``Batch.irecv``.

        In ``"sync"`` mode the receive completes inline and incoming
        data is staged through the receive buffer (if present) into the
        active batch via :meth:`_recv_to_batch`.  In ``"async_recv"`` and
        ``"fully_async"`` modes the receive handle is stored in
        ``_pending_recv_handle`` and the caller must invoke
        ``_complete_pending_recv`` before accessing ``active_batch``.

        In ``"fully_async"`` mode, any pending send handle from the
        previous iteration is drained (awaited) at the top of this
        method before posting the new receive.

        After processing incoming data (or when there is no prior rank),
        any remaining capacity in the active batch is backfilled from
        overflow sinks via :meth:`_drain_sinks_to_batch`.

        Notes
        -----
        This method should be called before ``dynamics.step()`` in the
        pipeline loop.  When using a non-sync ``comm_mode``, call
        ``_complete_pending_recv`` between this method and ``step()``.
        """
        if self.comm_mode == "fully_async" and self._pending_send_handle is not None:
            if self.debug_mode:
                logger.debug("[rank {}] draining pending async send", self.global_rank)
            self._pending_send_handle.wait()
            self._pending_send_handle = None

        if self.prior_rank is not None:
            if self.send_buffer is not None:
                self.send_buffer.zero()
            if self.recv_buffer is not None:
                self.recv_buffer.zero()

            template = self.recv_buffer or self._recv_template
            if self.debug_mode:
                logger.debug(
                    "[rank {}] posting irecv from rank {} (template={})",
                    self.global_rank,
                    self.prior_rank,
                    template is not None,
                )
            handle = Batch.irecv(
                src=self.prior_rank, device=self.device, template=template
            )

            if self.comm_mode == "sync":
                incoming = handle.wait()
                if self.debug_mode:
                    logger.debug(
                        "[rank {}] sync recv complete, {} graphs from rank {}",
                        self.global_rank,
                        incoming.num_graphs,
                        self.prior_rank,
                    )
                self._recv_to_batch(incoming)
            else:
                self._pending_recv_handle = handle

        # In async modes, drain happens in _complete_pending_recv after recv.
        if self.comm_mode == "sync" or self.prior_rank is None:
            self._drain_sinks_to_batch()

    def _complete_pending_recv(self) -> None:
        """Finalize any deferred receive before compute needs the data.

        In ``"sync"`` mode this is a no-op because
        ``_prestep_sync_buffers`` already completed the receive inline.
        In ``"async_recv"`` and ``"fully_async"`` modes, this method
        calls ``wait()`` on the stored receive handle and stages the
        incoming batch through the receive buffer (if present) into the
        active batch via :meth:`_recv_to_batch`.

        After processing incoming data, any remaining capacity in the
        active batch is backfilled from overflow sinks via
        :meth:`_drain_sinks_to_batch`.

        Notes
        -----
        Must be called after ``_prestep_sync_buffers`` and before any
        method that reads ``active_batch`` (e.g., ``step()``).
        """
        if self._pending_recv_handle is not None:
            if self.debug_mode:
                logger.debug(
                    "[rank {}] waiting on pending async recv", self.global_rank
                )
            incoming = self._pending_recv_handle.wait()
            if self.debug_mode:
                logger.debug(
                    "[rank {}] async recv complete, {} graphs",
                    self.global_rank,
                    incoming.num_graphs,
                )
            self._recv_to_batch(incoming)
            self._pending_recv_handle = None
        self._drain_sinks_to_batch()

    def _manage_send_handle(self, handle: Any) -> None:
        """Store or wait on a send handle based on communication mode.

        Parameters
        ----------
        handle
            The send handle returned by ``Batch.isend``.
        """
        if self.comm_mode == "fully_async":
            self._pending_send_handle = handle
        else:
            handle.wait()

    def _populate_send_buffer(self, converged_indices: torch.Tensor) -> None:
        """Populate the send buffer with converged graphs.

        Creates a boolean mask from the converged indices and copies those
        graphs into the send buffer. Does NOT send — the caller is responsible
        for issuing the ``isend``.

        Parameters
        ----------
        converged_indices : torch.Tensor
            Integer indices of converged samples (already truncated to capacity).
        """
        mask = torch.zeros(
            self.active_batch.num_graphs,
            dtype=torch.bool,
            device=self.device,
        )
        mask[converged_indices] = True
        self._batch_to_buffer(mask)
        if self.debug_mode:
            logger.debug(
                "[rank {}] populated send buffer with {} converged graphs",
                self.global_rank,
                converged_indices.numel(),
            )

    def _remove_converged_final_stage(self, converged_indices: torch.Tensor) -> None:
        """Remove converged graphs on the final stage and route to sinks.

        Extracts converged graphs, removes them from the active batch,
        and writes them to configured sinks if available.

        Parameters
        ----------
        converged_indices : torch.Tensor
            Integer indices of converged samples.
        """
        graduated = self.active_batch.index_select(converged_indices)
        all_indices = set(range(self.active_batch.num_graphs))
        remaining = sorted(all_indices - set(converged_indices.tolist()))
        if remaining:
            self.active_batch = self.active_batch.index_select(remaining)
        else:
            self.active_batch = None
        if self.debug_mode:
            logger.debug(
                "[rank {}] final stage, {} converged graphs removed",
                self.global_rank,
                converged_indices.numel(),
            )
        if self.sinks:
            self._overflow_to_sinks(graduated)

    def _poststep_sync_buffers(
        self, converged_indices: torch.Tensor | None = None
    ) -> None:
        """Synchronize buffers after a dynamics step.

        If ``converged_indices`` is provided and a next rank exists with
        available capacity, the converged samples are copied into
        ``send_buffer`` via :meth:`_populate_send_buffer`.  The send
        buffer is then unconditionally sent to the next rank — even if
        empty (``num_graphs == 0`` after zeroing) — so the downstream
        ``irecv`` always completes without deadlock.

        On the final stage, converged samples are extracted via
        ``index_select`` and written to the first available sink.

        Back-pressure behavior
        ----------------------
        Only as many converged samples as fit in the remaining buffer
        capacity are copied and sent.  Excess converged samples remain
        in the active batch and become no-ops until the next step when
        buffer capacity may be available.

        Parameters
        ----------
        converged_indices : torch.Tensor | None, optional
            Integer indices of converged samples in the active batch.
            Typically obtained from ``BaseDynamics._check_convergence()``.
            If ``None``, no samples are graduated.
        """
        has_converged = converged_indices is not None and converged_indices.numel() > 0

        if has_converged:
            if self.next_rank is not None:
                send_capacity = self._send_buffer_capacity
                if send_capacity > 0:
                    if converged_indices.numel() > send_capacity:
                        converged_indices = converged_indices[:send_capacity]
                    self._populate_send_buffer(converged_indices)
            if self.is_final_stage:
                self._remove_converged_final_stage(converged_indices)

        if self.next_rank is not None:
            handle = self.send_buffer.isend(dst=self.next_rank)
            self._manage_send_handle(handle)

    @property
    def global_rank(self) -> int:
        """Get the global rank for this process.

        Returns
        -------
        int
            Global rank across all nodes, or 0 if distributed is not initialized.
        """
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        return rank

    def __or__(self, other: BaseDynamics) -> DistributedPipeline:
        """Compose two stages into a ``DistributedPipeline`` via ``stage_a | stage_b``.

        Chaining is supported::

            a | b | c   →   (a | b) | c

        The first ``|`` creates a two-stage pipeline via this method.
        Subsequent ``|`` calls hit ``DistributedPipeline.__or__``,
        which appends stages and re-wires source/sink dependencies.

        Parameters
        ----------
        other : BaseDynamics
            The next stage to chain after this one.

        Returns
        -------
        DistributedPipeline
            A pipeline containing both stages mapped to sequential ranks.
            Source/sink dependencies (``prior_rank`` / ``next_rank``) are
            wired when ``DistributedPipeline.setup()`` is called (e.g.
            via the context manager or ``run()``).
        """
        return DistributedPipeline(stages={0: self, 1: other})

    def __add__(self, other: BaseDynamics) -> FusedStage:
        """Fuse two dynamics into a ``FusedStage`` via ``dyn_a + dyn_b``.

        Creates a ``FusedStage`` where both dynamics share a single batch
        and forward pass. Each dynamics applies masked updates to samples
        based on their status code.

        Parameters
        ----------
        other : BaseDynamics
            The dynamics to fuse with this one.

        Returns
        -------
        FusedStage
            A fused stage containing both dynamics with status codes 0 and 1.

        Raises
        ------
        TypeError
            If either ``self`` or ``other`` is not a ``BaseDynamics`` instance.
        """
        # FusedStage is defined later in this file
        if not isinstance(self, BaseDynamics):
            raise TypeError(
                "Both operands of + must be BaseDynamics instances. "
                f"self is {type(self).__name__}, not BaseDynamics."
            )
        if not isinstance(other, BaseDynamics):
            raise TypeError(
                "Both operands of + must be BaseDynamics instances. "
                f"other is {type(other).__name__}, not BaseDynamics."
            )
        return FusedStage(sub_stages=[(0, self), (1, other)])


class BaseDynamics(HookRegistryMixin, _CommunicationMixin):
    """Base class for all dynamics simulations.

    This class coordinates a ``BaseModelMixin`` model with a numerical
    integrator to evolve a ``Batch`` of atomic systems over time. It manages
    the step loop, hook execution at stage boundaries, and model evaluation.

    ``BaseDynamics`` inherits from ``HookRegistryMixin`` for hook storage
    and from ``_CommunicationMixin`` for inter-rank communication and
    buffer management for pipeline execution. All dynamics subclasses
    automatically have communication capabilities.

    The public interface centers on three methods. ``run(batch)``
    is the top-level entry point: it repeatedly calls ``step()`` for
    ``n_steps`` iterations and is the only method most users need.
    ``n_steps`` can be set at construction time or passed to ``run()``.
    ``step(batch)`` executes a single simulation step, orchestrating the
    full hook-wrapped sequence ``pre_update → compute → post_update``, with
    hooks fired at each stage boundary, followed by convergence checking.
    Subclasses should generally NOT override ``step``.
    ``compute(batch)`` performs the model forward pass: it calls
    ``model(batch)`` which must return a fully adapted ``ModelOutputs`` dict,
    validates outputs against ``__needs_keys__``, and writes results (forces,
    energies, stresses) back to the batch in-place.
    Subclasses should generally NOT override ``compute``.

    Attributes
    ----------
    model : BaseModelMixin
        The neural network potential model.
    step_count : int
        The current step number, starting from 0.
    hooks : list[Hook]
        Flat list of registered hooks.
    model_is_conservative : bool
        Indicates that the model uses automatic differentiation
        to obtain forces.
    convergence_hook : ConvergenceHook
        Hook that evaluates composable convergence criteria.  Defaults
        to a single forces-based criterion with threshold ``0.05``.
    n_steps : int | None
        Total number of simulation steps for ``run()``.  ``None``
        means the step count must be supplied when calling ``run()``.
    exit_status : int
        Status code threshold for graduated samples. Samples with
        ``status >= exit_status`` are treated as no-ops during
        ``step()`` — their positions and velocities are preserved
        through the integrator. Default is 1.
    __needs_keys__ : set[str]
        Set of output keys that this dynamics requires from the model.
        Empty by default on ``BaseDynamics``. Subclasses declare their
        own requirements (e.g., typically forces for optimization and MD).
        Checked in ``_validate_model_outputs()`` after each forward pass.
    __provides_keys__ : set[str]
        Set of keys that this dynamics produces or updates on the batch
        beyond model outputs. Empty by default. Subclasses declare what
        additional state they provide (e.g., ``{"velocities", "positions"}``
        for velocity verlet). Used for validation and buffer preallocation.

    Notes
    -----

    Developers implementing a new integrator should override
    ``pre_update(batch)`` and ``post_update(batch)`` to implement the
    integration scheme. These are called around ``compute()`` — ``pre_update``
    before, ``post_update`` after. For example, Velocity Verlet updates
    positions in ``pre_update`` and velocities in ``post_update``. The
    class-level sets ``__needs_keys__`` and ``__provides_keys__`` declare
    what outputs the dynamics requires from the model and what additional
    state it produces; requirements are checked in ``_validate_model_outputs()``
    after each forward pass.
    ``masked_update(batch, mask)`` is used by ``FusedStage`` to apply
    ``pre_update``/``post_update`` only to a subset of samples in a batched
    setting. Models must be ``BaseModelMixin`` instances — plain
    ``nn.Module`` is not accepted.

    Examples
    --------
    >>> model = MyPotentialModel()
    >>> dynamics = BaseDynamics(model, n_steps=1000)
    >>> dynamics.run(batch)
    """

    _stage_type = DynamicsStage

    __needs_keys__: set[str] = set()
    __provides_keys__: set[str] = set()

    _mutable_fields: tuple[str, ...] = ("positions", "velocities")

    _bookkeeping_keys: dict[str, Callable[[int, torch.device], torch.Tensor]] = {
        "status": lambda n, dev: torch.zeros(n, 1, dtype=torch.long, device=dev),
        "system_id": lambda n, dev: torch.full(
            (n, 1), -1, dtype=torch.long, device=dev
        ),
    }

    @classmethod
    def register_bookkeeping_key(
        cls,
        key: str,
        init_fn: Callable[[int, torch.device], torch.Tensor],
    ) -> None:
        """Register a graph-level bookkeeping field to survive refill_check.

        Parameters
        ----------
        key : str
            Field name on Batch.
        init_fn : Callable[[int, torch.device], torch.Tensor]
            Factory that creates a zero-initialized tensor of shape (n, 1)
            for n systems on the given device.
        """
        cls._bookkeeping_keys = {**cls._bookkeeping_keys, key: init_fn}

    def __init__(
        self,
        model: BaseModelMixin,
        hooks: list[Hook] | None = None,
        convergence_hook: Any = None,
        n_steps: int | None = None,
        exit_status: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the dynamics engine.

        Parameters
        ----------
        model : BaseModelMixin
            The neural network potential model.
        hooks : list[Hook] | None, optional
            Initial list of hooks to register. Each hook will be
            organized by its `stage` attribute.
        convergence_hook : ConvergenceHook | dict | None, optional
            Hook that evaluates composable convergence criteria.
        n_steps : int | None, optional
            Total number of simulation steps. If provided, ``run()``
            will use this value when called without an explicit
            ``n_steps`` argument. Default is ``None``.
            If a dict is provided, it is unpacked as
            ``ConvergenceHook(**convergence_hook)``.
            If ``None``, no convergence will be assessed.
        exit_status : int, optional
            Status code threshold for graduated samples. Samples with
            ``status >= exit_status`` are treated as no-ops during
            ``step()`` — their positions and velocities are preserved
            through the integrator. Default is 1. Subclasses like
            ``FusedStage`` may compute this dynamically.
        **kwargs : Any
            Additional keyword arguments forwarded to the next class
            in the MRO (for cooperative multiple inheritance).
        """
        super().__init__(**kwargs)
        if not isinstance(model, BaseModelMixin):
            raise TypeError(
                f"Expected a `BaseModelMixin` instance, got {type(model).__name__}."
                " Please wrap your model with a `BaseModelMixin` subclass."
            )
        self.model = model
        self.step_count: int = 0
        if isinstance(convergence_hook, dict):
            convergence_hook = ConvergenceHook(**convergence_hook)
        self.convergence_hook = convergence_hook
        self.n_steps = n_steps
        self.exit_status = exit_status
        self.model_config = model.model_config
        self.current_hook_stage: DynamicsStage | None = None
        self._init_hooks(hooks)

        self._last_converged: torch.Tensor | None = None

    @property
    def model_is_conservative(self) -> bool:
        """Returns whether or not the model uses conservative forces."""
        return "forces" in self.model_config.autograd_outputs

    def __repr__(self) -> str:
        """Return a human-readable summary of the dynamics engine."""
        cls = type(self).__name__
        model_cls = type(self.model).__name__
        conservative = self.model_is_conservative
        n_hooks = len(self.hooks)
        return (
            f"{cls}("
            f"model={model_cls}, "
            f"n_steps={self.n_steps}, "
            f"step_count={self.step_count}, "
            f"conservative={conservative}, "
            f"convergence_hook={self.convergence_hook!r}, "
            f"hooks={n_hooks})"
        )

    def _call_hooks(self, stage: DynamicsStage, batch: Batch) -> None:
        """Execute hooks for the given stage with dynamics-specific tracking."""
        self.current_hook_stage = stage
        super()._call_hooks(stage, batch)

    def _build_context(self, batch: Batch) -> HookContext:
        """Build a dynamics-specific HookContext."""
        if self._last_converged is not None:
            _mask = torch.zeros(
                batch.num_graphs, dtype=torch.bool, device=batch.positions.device
            )
            _mask[self._last_converged] = True
        else:
            _mask = None
        return HookContext(
            batch=batch,
            step_count=self.step_count,
            model=self.model,
            converged_mask=_mask,
            global_rank=self.global_rank,
            workflow=self,
        )

    def _open_hooks(self) -> None:
        """Enter context-manager hooks registered on this stage.

        Calls ``__enter__`` on every hook that supports the context-manager
        protocol.  A ``seen`` set prevents double-entering hooks.

        Called automatically at the start of :meth:`run`.
        """
        seen: set[int] = set()
        for hook in self.hooks:
            hook_id = id(hook)
            if hook_id not in seen and hasattr(hook, "__enter__"):
                seen.add(hook_id)
                hook.__enter__()

    def _close_hooks(self) -> None:
        """Exit context-manager hooks, falling back to ``close()`` otherwise.

        For hooks that support the context-manager protocol, calls
        ``__exit__(None, None, None)``.  For hooks that only expose a
        ``close()`` method (e.g. ``ProfilerHook``), calls ``close()``
        directly.  A ``seen`` set prevents double-closing hooks.

        Called automatically at the end of :meth:`run`.
        """
        seen: set[int] = set()
        for hook in self.hooks:
            hook_id = id(hook)
            if hook_id in seen:
                continue
            seen.add(hook_id)
            if hasattr(hook, "__exit__"):
                hook.__exit__(None, None, None)
            elif hasattr(hook, "close"):
                hook.close()

    def _check_convergence(self, batch: Batch) -> torch.Tensor | None:
        """Return indices of converged samples, or None if none converged.

        Delegates to ``self.convergence_hook.evaluate(batch)`` if a
        convergence hook is configured.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.

        Returns
        -------
        torch.Tensor | None
            Integer tensor of converged sample indices, or ``None``.
        """
        if self.convergence_hook is None:
            return None
        return self.convergence_hook.evaluate(batch)

    def _validate_model_outputs(self, outputs: ModelOutputs) -> None:
        """Validate that model outputs satisfy the dynamics requirements.

        Iterates over ``__needs_keys__`` and checks that each declared key
        is present and not ``None`` in the model outputs.

        Parameters
        ----------
        outputs : ModelOutputs
            The model outputs to validate.

        Raises
        ------
        RuntimeError
            If a required output key is missing or ``None``.
        """
        for key in self.__needs_keys__:
            if outputs.get(key) is None:
                raise RuntimeError(
                    f"{type(self).__name__} requires '{key}' "
                    f"(declared in __needs_keys__), but the model did not "
                    f"produce it. Check your model's ModelConfig and "
                    f"ModelConfig to ensure '{key}' is supported and enabled,"
                    " or that no hooks are missing."
                )

    def _validate_batch_keys(self, batch: Batch) -> None:
        """Validate that the batch contains all keys declared in ``__provides_keys__``.

        This is a diagnostic helper — not wired into the hot path. It can be
        called after ``step()`` to verify that the dynamics produced the keys
        it claims to provide.

        Parameters
        ----------
        batch : Batch
            The batch to validate.

        Raises
        ------
        RuntimeError
            If a declared provides-key is ``None`` on the batch.
        """
        for key in self.__provides_keys__:
            val = getattr(batch, key, None)
            if val is None:
                raise RuntimeError(
                    f"{type(self).__name__} declares '{key}' in "
                    f"__provides_keys__, but batch.{key} is None after "
                    f"compute. This may indicate a misconfigured model or "
                    f"dynamics."
                )

    # ------------------------------------------------------------------
    # Per-system integrator state management
    # ------------------------------------------------------------------

    def _init_state(self, batch: Batch) -> None:
        """Allocate per-system integrator state from the first concrete batch.

        No-op in the base class.  Subclasses that require per-system state
        (e.g. all warp-kernel integrators) override this to build a
        system-only :class:`~nvalchemi.data.Batch` and assign it to
        ``self._state``.

        Parameters
        ----------
        batch : Batch
            The first concrete batch; used to determine M, device, and dtype.
        """

    def _make_new_state(self, n: int, template_batch: Batch) -> "Batch | None":
        """Return default state for *n* newly admitted systems.

        No-op in the base class (returns ``None``).  Subclasses override
        to produce a system-only :class:`~nvalchemi.data.Batch` with
        default/reset state for *n* replacement systems.

        Parameters
        ----------
        n : int
            Number of new systems to create state for.
        template_batch : Batch
            The updated active batch; provides device and dtype.

        Returns
        -------
        Batch | None
            A system-only batch with *n* rows, or ``None`` if this
            dynamics does not maintain per-system state.
        """
        return None

    def _ensure_state_initialized(self, batch: Batch) -> None:
        """Lazily initialize per-system integrator state on the first call.

        Calls :meth:`_init_state` the first time this method is invoked
        (i.e. when ``self._state`` does not yet exist).  Subsequent calls
        are no-ops.  This is invoked automatically at the start of
        :meth:`step` and :meth:`masked_update` so that concrete subclasses
        never need to call it explicitly.

        Parameters
        ----------
        batch : Batch
            The current batch; forwarded to ``_init_state`` if needed.
        """
        if not hasattr(self, "_state"):
            self._init_state(batch)

    def _sync_state_to_batch(
        self,
        remaining_indices: "torch.Tensor",
        n_new: int,
        template_batch: Batch,
    ) -> None:
        """Synchronize ``self._state`` after an inflight batch refill.

        Called by :meth:`_refill_check` after graduated systems have been
        removed and replacement systems appended.  Removes state rows for
        graduated systems and appends fresh default state for the new ones.

        If this dynamics has no ``_state`` (e.g. :class:`DemoDynamics`),
        this method is a no-op.

        Parameters
        ----------
        remaining_indices : torch.Tensor
            Integer indices of systems that remain in the new batch, in
            order.  Used to slice ``self._state`` via ``index_select``.
        n_new : int
            Number of newly admitted replacement systems appended after
            the remaining ones.  State for these is produced by
            :meth:`_make_new_state`.
        template_batch : Batch
            The updated batch (remaining + replacements); provides device
            and dtype for :meth:`_make_new_state`.
        """
        if not hasattr(self, "_state"):
            return

        if remaining_indices.numel() > 0:
            remaining_state: "Batch | None" = self._state.index_select(
                remaining_indices
            )
        else:
            remaining_state = None

        new_state: "Batch | None" = None
        if n_new > 0:
            new_state = self._make_new_state(n_new, template_batch)

        if remaining_state is not None and new_state is not None:
            remaining_state.append(new_state)
            self._state = remaining_state
        elif remaining_state is not None:
            self._state = remaining_state
        elif new_state is not None:
            self._state = new_state
        else:
            del self._state

    def pre_update(self, batch: Batch) -> None:
        """
        Perform the first half of the integration step.

        This method is a no-op in the base class and should be
        overridden by integrator subclasses (e.g., Velocity Verlet
        would update positions here).

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data, modified in-place.
        """
        pass

    def post_update(self, batch: Batch) -> None:
        """
        Perform the second half of the integration step.

        This method is a no-op in the base class and should be
        overridden by integrator subclasses (e.g., Velocity Verlet
        would update velocities here).

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data, modified in-place.
        """
        pass

    # Class-level mapping from model output keys to batch attribute names.
    # Override in subclasses to handle non-standard batch attribute names.
    _OUTPUT_KEY_TO_BATCH_ATTR: dict[str, str] = {
        "energy": "energy",
        "forces": "forces",
        "stress": "stress",
    }

    def compute(self, batch: Batch | AtomsLike) -> ModelOutputs:
        """
        Perform the model forward pass to compute forces and energies.

        This method:
        1. Runs the model forward pass, which should enable gradients
        2. Adapts outputs to the standard format
        3. Validates outputs against dynamics requirements
        4. Writes known keys back to the batch in-place via
           :attr:`_OUTPUT_KEY_TO_BATCH_ATTR`
        5. Detaches all output tensors from the computation graph and
           exposes them as ``_last_outputs`` for custom dynamics subclasses
           that need charges, embeddings, or other non-standard outputs.
        6. Clears ``requires_grad`` on batch tensors that the model
           enabled for autograd (e.g. positions), so downstream hooks
           can safely perform in-place operations.

        The detach in step 5 is deliberate: model wrappers may return
        tensors that are still attached to the autograd graph (e.g. MACE
        returns energies on the graph even after computing forces
        internally).  Since ``compute()`` is a terminal consumer — values
        have already been copied into the batch — holding the graph would
        cause memory to grow without bound across dynamics steps.  Callers
        that need the live graph (e.g. training loops computing a loss)
        should call ``model(batch)`` directly instead of ``compute()``.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data. Will have forces and
            energies updated in-place.

        Returns
        -------
        ModelOutputs
            OrderedDict containing the model outputs (energies, forces,
            and any other computed properties).  All tensors are detached
            from the computation graph.

        Raises
        ------
        RuntimeError
            If the model outputs do not satisfy the dynamics requirements
            specified by ``__needs_keys__``.
        """
        self._last_outputs = None

        # model.forward() is responsible for returning a fully adapted ModelOutputs dict.
        # adapt_output() must NOT be called again here; each wrapper handles adaptation
        # internally and returns canonical keys directly from forward().
        outputs: ModelOutputs = self.model(batch)
        self._validate_model_outputs(outputs)

        # Write known keys to batch via copy_, then detach all tensors from the
        # computation graph.  Models like MACE return energies still attached to
        # the graph after internal autograd; without detaching, _last_outputs
        # would keep the entire forward graph alive until the next compute().
        detached: ModelOutputs = OrderedDict()
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
            else:
                detached[key] = value
        # Explicitly break all references to graph-attached tensors before
        # any further work.  Without this, the local `outputs` dict and its
        # values keep the autograd graph alive for the remainder of compute().
        del outputs

        for out_key, batch_attr in self._OUTPUT_KEY_TO_BATCH_ATTR.items():
            value = detached.get(out_key)
            if value is not None:
                target = getattr(batch, batch_attr, None)
                if target is not None:
                    target.copy_(value.view(target.shape))

        # Clear requires_grad on batch tensors that the model enabled for
        # autograd force/stress computation.  After compute() the gradients
        # have been extracted into forces/stress and the graph is detached,
        # so keeping requires_grad=True would only cause spurious errors in
        # downstream hooks that perform in-place operations (e.g. copy_).
        # Always include "positions" — it is an implicit model input that
        # may not appear in autograd_inputs but can still carry grad from
        # the forward pass.
        cfg = self.model_config
        grad_keys: set[str] = {"positions"}
        grad_keys |= cfg.gradient_keys
        if cfg.autograd_outputs & cfg.active_outputs:
            grad_keys |= cfg.autograd_inputs
        for key in grad_keys:
            value = getattr(batch, key, None)
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.requires_grad_(False)

        self._last_outputs = detached

        return detached

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """
        Execute a single dynamics step with the full hook-wrapped sequence.

        The step proceeds as follows:
        1. BEFORE_STEP hooks
        2. BEFORE_PRE_UPDATE hooks -> pre_update() -> AFTER_PRE_UPDATE hooks
        3. BEFORE_COMPUTE hooks -> compute() -> AFTER_COMPUTE hooks
        4. BEFORE_POST_UPDATE hooks -> post_update() -> AFTER_POST_UPDATE hooks
        5. AFTER_STEP hooks
        6. Check convergence and fire ON_CONVERGE hooks if any samples converged
        7. Increment step_count

        Samples with ``status >= exit_status`` are treated as no-ops for the
        integrator (pre_update/post_update). Their positions and velocities
        are preserved through the step. This enables back-pressure handling
        in pipeline mode where converged samples may remain in the active
        batch when the send buffer is full.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.

        Returns
        -------
        tuple[Batch, torch.Tensor | None]
            The updated batch after the step, and a 1-D integer tensor
            of converged sample indices (or ``None`` if nothing converged).
        """
        self._ensure_state_initialized(batch)

        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        active_mask: torch.Tensor | None = None
        if hasattr(batch, "status") and batch.status is not None:
            status = (
                batch.status.squeeze(-1) if batch.status.dim() == 2 else batch.status
            )
            active_mask = status[: batch.num_graphs] < self.exit_status

        saved: dict[str, torch.Tensor] = {}
        if active_mask is not None:
            node_mask_occupied = torch.repeat_interleave(
                active_mask, batch.num_nodes_per_graph
            )
            node_mask = torch.zeros(
                batch.num_nodes, dtype=torch.bool, device=batch.device
            )
            node_mask[: len(node_mask_occupied)] = node_mask_occupied
            sys_mask = ~active_mask
            for field in self._mutable_fields:
                val = getattr(batch, field, None)
                if val is None:
                    continue
                if val.shape[0] == batch.num_nodes:
                    saved[field] = val[~node_mask].clone()
                elif val.shape[0] == batch.num_graphs:
                    saved[field] = val[sys_mask].clone()

        self._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch)
        self.pre_update(batch)
        self._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, batch)
        self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        self.compute(batch)
        self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
        self._call_hooks(DynamicsStage.BEFORE_POST_UPDATE, batch)
        self.post_update(batch)
        self._call_hooks(DynamicsStage.AFTER_POST_UPDATE, batch)
        if active_mask is not None:
            with torch.no_grad():
                for field, sv in saved.items():
                    val = getattr(batch, field)
                    if val.shape[0] == batch.num_nodes:
                        val[~node_mask] = sv
                    else:
                        val[sys_mask] = sv

        self._call_hooks(DynamicsStage.AFTER_STEP, batch)

        converged = self._check_convergence(batch)
        self._last_converged = converged
        if converged is not None:
            self._call_hooks(DynamicsStage.ON_CONVERGE, batch)

        self.step_count += 1

        return batch, converged

    def run(self, batch: Batch, n_steps: int | None = None) -> Batch:
        """
        Run the dynamics simulation for a specified number of steps.

        This is a convenience method that repeatedly calls ``step()``.
        The step count can be set at construction time via the
        ``n_steps`` parameter, or passed directly to this method.
        A value passed here takes precedence over the instance
        attribute.

        Parameters
        ----------
        batch : Batch
            The initial batch of atomic data.
        n_steps : int | None, optional
            The number of steps to run.  If ``None``, falls back to
            ``self.n_steps``.  If both are ``None``, raises
            ``ValueError``.

        Returns
        -------
        Batch
            The batch after all steps have been executed.

        Raises
        ------
        ValueError
            If no step count is available (both the argument and
            ``self.n_steps`` are ``None``).
        """
        resolved = n_steps if n_steps is not None else self.n_steps
        if resolved is None:
            raise ValueError(
                "No step count provided. Either pass `n_steps` to run() "
                "or set it at construction time via "
                f"`{type(self).__name__}(..., n_steps=N)`."
            )
        self._open_hooks()
        try:
            for _ in range(resolved):
                batch, _converged = self.step(batch)
                # Early exit when every system has satisfied the convergence
                # criteria (sampler-free / Mode 1 only).
                if (
                    self.sampler is None
                    and _converged is not None
                    and _converged.numel() == batch.num_graphs
                ):
                    break
        finally:
            self._close_hooks()
        return batch

    def refill_check(self, batch: Batch, exit_status: int) -> Batch | None:
        """Replace graduated samples via index-select and append.

        Graduated graphs (``status >= exit_status``) are written to sinks,
        then removed via :meth:`Batch.index_select` on the remaining indices.
        Replacement samples from the sampler are appended via
        :meth:`Batch.append`.  Dynamics-specific bookkeeping fields are
        written into the result batch via the ``_bookkeeping_keys`` registry.

        Parameters
        ----------
        batch : Batch
            The current batch with a ``status`` field.
        exit_status : int
            Status code indicating graduation.

        Returns
        -------
        Batch | None
            A new batch with graduated graphs replaced by fresh samples,
            or ``None`` if no active samples remain (sampler exhausted
            and all graduated) — in which case ``self.done`` is set to
            ``True``.

        Raises
        ------
        RuntimeError
            If ``self.sampler`` is ``None``.
        """
        if self.sampler is None:
            raise RuntimeError("refill_check requires a sampler to be configured.")

        status = batch.status
        if status.dim() == 2:
            status = status.squeeze(-1)
        graduated_mask = status >= exit_status

        if not graduated_mask.any():
            return batch

        remaining_indices = torch.where(~graduated_mask)[0]

        if self.sinks and graduated_mask.any():
            self._overflow_to_sinks(batch, mask=graduated_mask)

        n_remaining = remaining_indices.numel()

        if remaining_indices.numel() > 0:
            result = batch.index_select(remaining_indices)
        else:
            result = None

        # Budget-based replacement: compute total atom/edge headroom
        # from the remaining systems and the sampler's constraints.
        remaining_atoms = (
            int(batch.num_nodes_per_graph[remaining_indices].sum())
            if n_remaining > 0
            else 0
        )
        atom_budget = (
            self.sampler.max_atoms - remaining_atoms
            if self.sampler.max_atoms is not None
            else None
        )
        edges_per_graph = batch.num_edges_per_graph
        remaining_edges = (
            int(edges_per_graph[remaining_indices].sum())
            if n_remaining > 0 and edges_per_graph.numel() > 0
            else 0
        )
        edge_budget = (
            self.sampler.max_edges - remaining_edges
            if self.sampler.max_edges is not None
            else None
        )
        max_new = (
            self.sampler.max_batch_size - n_remaining
            if self.sampler.max_batch_size is not None
            else None
        )
        replacements = self.sampler.request_replacements_budget(
            atom_budget=atom_budget,
            edge_budget=edge_budget,
            max_count=max_new,
        )

        if result is not None and replacements:
            repl_batch = Batch.from_data_list(replacements, device=batch.device)
            result.append(repl_batch)
        elif result is None and replacements:
            result = Batch.from_data_list(replacements, device=batch.device)

        if result is not None:
            n_total = result.num_graphs
            device = result.device
            for key, default_fn in self._bookkeeping_keys.items():
                new_tensor = default_fn(n_total, device)
                remaining_vals = getattr(batch, key, None)
                if remaining_vals is not None and n_remaining > 0:
                    src = remaining_vals[remaining_indices]
                    src = src.unsqueeze(-1) if src.dim() == 1 else src
                    new_tensor[:n_remaining] = src
                result[key] = new_tensor

            self._sync_state_to_batch(remaining_indices, len(replacements), result)
            return result

        if self.sampler.exhausted:
            self.done = True
        self._sync_state_to_batch(remaining_indices, 0, batch)
        return None

    def masked_update(
        self,
        batch: Batch,
        mask: Bool[torch.Tensor, "B"],  # noqa: F722, F821
    ) -> None:
        """
        Apply pre_update and post_update only to selected samples in the batch.

        This method allows selective updates where only some graphs in the
        batch are modified. Unmasked samples retain their original positions
        and velocities.

        The mask is a boolean tensor of shape (B,) where B is the number of
        graphs. True values indicate samples that should be updated.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data, modified in-place.
        mask : Bool[Tensor, "B"]
            Boolean mask selecting which graphs to update. Shape (B,)
            where B is the number of graphs in the batch.

        Notes
        -----
        This method expands the graph-level mask to node-level using
        `batch.batch_idx` to correctly index per-node tensors like positions
        and velocities.
        """
        # lazy init — FusedStage sub-stages never have step() called on them directly
        self._ensure_state_initialized(batch)

        node_mask_occupied = torch.repeat_interleave(mask, batch.num_nodes_per_graph)
        node_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=batch.device)
        node_mask[: len(node_mask_occupied)] = node_mask_occupied
        sys_mask = ~mask

        saved: dict[str, torch.Tensor] = {}
        for field in self._mutable_fields:
            val = getattr(batch, field, None)
            if val is None:
                continue
            if val.shape[0] == batch.num_nodes:
                saved[field] = val[~node_mask].clone()
            elif val.shape[0] == batch.num_graphs:
                saved[field] = val[sys_mask].clone()

        self.pre_update(batch)
        self.post_update(batch)

        with torch.no_grad():
            for field, sv in saved.items():
                val = getattr(batch, field)
                if val.shape[0] == batch.num_nodes:
                    val[~node_mask] = sv
                else:
                    val[sys_mask] = sv

    def _masked_pre_update(
        self,
        batch: Batch,
        mask: Bool[torch.Tensor, "B"],  # noqa: F722, F821
    ) -> None:
        """Run only pre_update for masked samples, restoring non-masked state.

        Used by :class:`FusedStage` to interleave pre_update across all
        sub-stages before the shared compute, so that forces are evaluated at
        the post-pre_update positions (required for BAOAB Langevin and
        velocity-Verlet-based integrators).
        """
        self._ensure_state_initialized(batch)

        node_mask = mask[batch.batch_idx]
        sys_mask = ~mask

        saved: dict[str, torch.Tensor] = {}
        for field in self._mutable_fields:
            val = getattr(batch, field, None)
            if val is None:
                continue
            if val.shape[0] == batch.num_nodes:
                saved[field] = val[~node_mask].clone()
            elif val.shape[0] == batch.num_graphs:
                saved[field] = val[sys_mask].clone()

        self.pre_update(batch)

        with torch.no_grad():
            for field, sv in saved.items():
                val = getattr(batch, field)
                if val.shape[0] == batch.num_nodes:
                    val[~node_mask] = sv
                else:
                    val[sys_mask] = sv

    def _masked_post_update(
        self,
        batch: Batch,
        mask: Bool[torch.Tensor, "B"],  # noqa: F722, F821
    ) -> None:
        """Run only post_update for masked samples, restoring non-masked state.

        Called by :class:`FusedStage` after the shared compute so that
        post_update (e.g. the final BAOAB velocity half-kick) uses forces at
        the new positions.
        """
        node_mask = mask[batch.batch_idx]
        sys_mask = ~mask

        saved: dict[str, torch.Tensor] = {}
        for field in self._mutable_fields:
            val = getattr(batch, field, None)
            if val is None:
                continue
            if val.shape[0] == batch.num_nodes:
                saved[field] = val[~node_mask].clone()
            elif val.shape[0] == batch.num_graphs:
                saved[field] = val[sys_mask].clone()

        self.post_update(batch)

        with torch.no_grad():
            for field, sv in saved.items():
                val = getattr(batch, field)
                if val.shape[0] == batch.num_nodes:
                    val[~node_mask] = sv
                else:
                    val[sys_mask] = sv


class ConvergenceHook:
    """Hook that evaluates composable convergence criteria and optionally
    migrates converged samples between pipeline stages.

    Wraps one or more :class:`_ConvergenceCriterion` instances and combines
    their results with AND semantics: a sample is converged only when
    **every** criterion is satisfied.

    When ``source_status`` and ``target_status`` are both provided, the
    hook also performs status migration — updating ``batch.status`` for
    converged samples that match ``source_status``.  This enables the
    single-loop execution strategy used by :class:`FusedStage`.

    When used as a standalone convergence detector (both ``source_status``
    and ``target_status`` are ``None``), call :meth:`evaluate` directly
    or let :class:`BaseDynamics` use it via ``_check_convergence``.

    Attributes
    ----------
    criteria : list[_ConvergenceCriterion]
        The individual convergence criteria.
    frequency : int
        Execute every N steps.
    stage : DynamicsStage
        The stage at which this hook fires (``AFTER_STEP``).
    source_status : int | None
        Status code of samples to check for convergence.  ``None``
        disables status migration.
    target_status : int | None
        Status code to assign to converged samples.  ``None``
        disables status migration.

    Examples
    --------
    >>> # Backward-compatible fmax-only hook
    >>> hook = ConvergenceHook.from_fmax(0.05)
    >>> converged = hook.evaluate(batch)

    >>> # Multi-criteria hook for FusedStage with status migration
    >>> hook = ConvergenceHook(
    ...     criteria=[
    ...         {"key": "forces", "threshold": 0.05, "reduce_op": "norm", "reduce_dims": -1},
    ...         {"key": "energy_change", "threshold": 1e-6},
    ...     ],
    ...     source_status=0,
    ...     target_status=1,
    ... )
    """

    def __init__(
        self,
        criteria: (
            _ConvergenceCriterion
            | list[_ConvergenceCriterion]
            | dict
            | list[dict]
            | None
        ) = None,
        source_status: int | None = None,
        target_status: int | None = None,
        frequency: int = 1,
    ) -> None:
        """Initialize the convergence hook.

        Parameters
        ----------
        criteria : _ConvergenceCriterion | list[...] | dict | list[dict] | None
            Convergence criterion specification(s).  Dicts are validated
            and converted to ``_ConvergenceCriterion`` instances.  If
            ``None``, defaults to a single forces-based criterion (max
            per-atom force norm) with threshold ``0.05``.
        source_status : int | None, optional
            Status code to check.  ``None`` disables status migration.
        target_status : int | None, optional
            Status code to assign on convergence.  ``None`` disables
            status migration.
        frequency : int, optional
            Execute every N steps. Default 1.
        """
        self.frequency = frequency
        self.stage = DynamicsStage.AFTER_STEP
        self.source_status = source_status
        self.target_status = target_status

        if criteria is None:
            self.criteria: list[_ConvergenceCriterion] = [
                _ConvergenceCriterion(
                    key="forces", threshold=0.05, reduce_op="norm", reduce_dims=-1
                )
            ]
        elif isinstance(criteria, _ConvergenceCriterion):
            self.criteria = [criteria]
        elif isinstance(criteria, dict):
            self.criteria = [_ConvergenceCriterion(**criteria)]
        elif isinstance(criteria, (list, tuple)):
            normalized: list[_ConvergenceCriterion] = []
            for item in criteria:
                if isinstance(item, dict):
                    normalized.append(_ConvergenceCriterion(**item))
                elif isinstance(item, _ConvergenceCriterion):
                    normalized.append(item)
                else:
                    raise TypeError(
                        "Each criterion must be a dict or"
                        f" _ConvergenceCriterion, got {type(item).__name__}"
                    )
            self.criteria = normalized
        else:
            raise TypeError(
                "criteria must be a dict, _ConvergenceCriterion, or list"
                f" thereof, got {type(criteria).__name__}"
            )

    def __repr__(self) -> str:
        """Return a human-readable summary of the convergence hook."""
        inner = ", ".join(repr(c) for c in self.criteria)
        parts = [f"criteria=[{inner}]"]
        if self.source_status is not None:
            parts.append(f"source_status={self.source_status}")
        if self.target_status is not None:
            parts.append(f"target_status={self.target_status}")
        parts.append(f"frequency={self.frequency}")
        return f"ConvergenceHook({', '.join(parts)})"

    @classmethod
    def from_fmax(
        cls,
        threshold: float = 0.05,
        source_status: int | None = None,
        target_status: int | None = None,
        frequency: int = 1,
    ) -> ConvergenceHook:
        """Create a forces-based convergence hook (fmax-compatible).

        This is a convenience constructor for backward compatibility
        with the original ``convergence_fmax`` parameter.

        Parameters
        ----------
        threshold : float, optional
            Maximum force threshold.  Default ``0.05``.
        source_status : int | None, optional
            Status code to check.  ``None`` disables status migration.
        target_status : int | None, optional
            Status code to assign on convergence.  ``None`` disables
            status migration.
        frequency : int, optional
            Execute every N steps.  Default 1.

        Returns
        -------
        ConvergenceHook
            Hook with a single forces-based (max force norm) criterion.
        """
        return cls.from_forces(
            threshold=threshold,
            frequency=frequency,
            source_status=source_status,
            target_status=target_status,
        )

    @classmethod
    def from_forces(
        cls,
        threshold: float,
        frequency: int = 1,
        source_status: int | None = None,
        target_status: int | None = None,
    ) -> ConvergenceHook:
        """Construct from force-norm threshold (reads 'forces' key, norm reduction).

        Parameters
        ----------
        threshold : float
            Force threshold; systems with max force norm <= threshold are converged.
        frequency : int, optional
            Evaluate every N steps. Default 1.
        source_status : int | None, optional
            Status code that eligible systems must have. Default None (any status).
        target_status : int | None, optional
            Status code to assign to converged systems. Default None (no status change).

        Returns
        -------
        ConvergenceHook
            Hook that evaluates max per-atom force norm against ``threshold``.
        """
        return cls(
            criteria=[
                {
                    "key": "forces",
                    "threshold": threshold,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
            frequency=frequency,
            source_status=source_status,
            target_status=target_status,
        )

    @property
    def num_criteria(self) -> int:
        """Return the number of individual criteria."""
        return len(self.criteria)

    def evaluate(self, batch: Batch) -> torch.Tensor | None:
        """Evaluate all criteria and return indices of converged samples.

        Pre-allocates a ``(N_criteria, B)`` boolean tensor, evaluates
        each criterion to fill one row, then AND-reduces across
        criteria.  Returns the integer indices of converged samples,
        or ``None`` if no samples have converged.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.

        Returns
        -------
        torch.Tensor | None
            1-D integer tensor of converged sample indices, or ``None``
            if no samples satisfy all criteria.
        """
        n_criteria = len(self.criteria)
        n_graphs = batch.num_graphs

        results = torch.ones(
            n_criteria,
            n_graphs,
            dtype=torch.bool,
            device=batch.positions.device,
        )

        for i, criterion in enumerate(self.criteria):
            results[i] = criterion(batch)

        converged_mask = torch.all(results, dim=0)

        if not converged_mask.any():
            return None
        return torch.where(converged_mask)[0]

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Evaluate convergence and optionally migrate sample status.

        When ``source_status`` and ``target_status`` are both set,
        converged samples whose ``batch.status`` matches
        ``source_status`` are migrated to ``target_status``.

        If ``batch`` lacks ``status`` when status migration is
        configured, the migration step is silently skipped.

        Parameters
        ----------
        ctx : HookContext
            The hook context containing the current batch.
        stage : Enum
            The stage being dispatched.
        """
        batch = ctx.batch
        converged = self.evaluate(batch)
        if converged is None:
            return

        if self.source_status is not None and self.target_status is not None:
            if not hasattr(batch, "status") or batch.status is None:
                return

            status = batch.status
            if status.dim() == 2:
                status = status.squeeze(-1)

            converged_mask = torch.zeros(
                batch.num_graphs, dtype=torch.bool, device=status.device
            )
            converged_mask[converged] = True

            status_mask = status == self.source_status
            migrate = converged_mask & status_mask

            if migrate.any():
                flat_status = (
                    batch.status.view(-1) if batch.status.dim() == 2 else batch.status
                )
                flat_status[migrate] = self.target_status


class FusedStage(BaseDynamics):
    """Composite dynamics engine fusing multiple sub-stages on a single GPU.

    ``FusedStage`` composes multiple ``BaseDynamics`` sub-stages to share one
    ``Batch`` and one model forward pass per step, avoiding redundant forward
    passes when multiple simulation phases (e.g., relaxation then MD) operate
    on the same batch.

    Unlike ``BaseDynamics``, **``step(batch)``** is overridden. Instead of the
    standard ``pre_update → compute → post_update`` loop, ``FusedStage``
    performs: (1) a single ``compute()`` call on the full batch, then (2)
    iterates over sub-stages, applying ``masked_update(batch, mask)`` on each
    sub-stage's dynamics for samples whose ``batch.status`` matches that
    sub-stage's status code. Only ONE forward pass happens per step regardless
    of the number of sub-stages. **``run(batch)``** is also overridden —
    the ``n_steps`` attribute (inherited from ``BaseDynamics``) and any
    ``n_steps`` argument passed to ``run()`` are both the **maximum** number
    of steps; the loop runs until all samples have migrated to the
    ``exit_status``, the sampler is exhausted, or ``n_steps`` is reached. Convergence-driven migration is handled
    by ``ConvergenceHook`` instances auto-registered between adjacent
    sub-stages: when samples converge in sub-stage *i*, their ``batch.status``
    is updated to sub-stage *i+1*'s code, causing them to be processed by the
    next dynamics on the following step. The ``+`` operator composes
    sub-stages: ``dyn_a + dyn_b`` creates a ``FusedStage``, and
    ``fused + dyn_c`` appends a third sub-stage. The ``|`` operator (inherited
    from ``BaseDynamics`` via ``_CommunicationMixin``) creates a ``DistributedPipeline``
    for multi-rank execution instead.

    Developers generally do NOT subclass ``FusedStage``. Instead, create
    ``BaseDynamics`` subclasses (integrators) and compose them using ``+``.
    ``FusedStage`` handles orchestration automatically. The key requirement is
    that sub-stage dynamics must implement ``masked_update`` correctly
    (inherited from ``BaseDynamics``) and that the batch must have a
    ``status`` tensor.

    Hook Firing Semantics
    ~~~~~~~~~~~~~~~~~~~~~
    Because ``FusedStage`` shares a single forward pass across all sub-stages,
    hook firing differs from standalone ``BaseDynamics`` execution. The
    following hooks fire **on each sub-stage** during ``_step_impl``:

    **Fired on sub-stages (in order):**

    - ``BEFORE_STEP`` — at the start of each fused step, before any work.
    - ``AFTER_COMPUTE`` — after the shared model forward pass completes.
    - ``BEFORE_PRE_UPDATE`` — before each sub-stage's ``masked_update``
      (fires even when no samples match the sub-stage's status code).
    - ``AFTER_POST_UPDATE`` — after each sub-stage's ``masked_update``
      (fires even when no samples match the sub-stage's status code).
    - ``AFTER_STEP`` — after all masked updates are complete.
    - ``ON_CONVERGE`` — when a sub-stage's ``_check_convergence`` detects
      converged samples.

    **NOT fired on sub-stages:**

    - ``BEFORE_COMPUTE`` — the forward pass is shared across all sub-stages,
      not executed per-sub-stage; there is no meaningful "before compute"
      point for individual sub-stages.
    - ``AFTER_PRE_UPDATE`` — ``masked_update`` combines ``pre_update`` and
      ``post_update`` atomically; there is no intermediate hook point.
    - ``BEFORE_POST_UPDATE`` — same reason as ``AFTER_PRE_UPDATE``.

    **Step count semantics:** Each sub-stage's ``step_count`` is incremented
    alongside the ``FusedStage``'s own ``step_count`` after every fused step,
    ensuring that hook frequency (e.g., ``every_n_steps``) is respected
    correctly across all sub-stages.

    Parameters
    ----------
    sub_stages : list[tuple[int, BaseDynamics]]
        Ordered ``(status_code, dynamics)`` pairs. Status codes are
        auto-assigned starting from 0 when using the ``+`` operator.
    entry_status : int
        Status code assigned to incoming samples (default: 0).
    exit_status : int
        Status code that triggers graduation to the next pipeline stage.
        Auto-set to ``len(sub_stages)`` (one past the last sub-stage code).
    compile_step : bool
        If ``True``, replace ``self.step`` with
        ``torch.compile(self.step, **compile_kwargs)``.
    compile_kwargs : dict
        Keyword arguments forwarded to ``torch.compile``.
    **kwargs
        Additional keyword arguments forwarded to ``BaseDynamics``.

    Attributes
    ----------
    sub_stages : list[tuple[int, BaseDynamics]]
        Ordered ``(status_code, dynamics)`` pairs.
    entry_status : int
        Status code for incoming samples.
    exit_status : int
        Status code that triggers graduation.
    compile_step : bool
        Whether the step method is compiled.
    compile_kwargs : dict
        Arguments passed to ``torch.compile``.
    __needs_keys__ : set[str]
        Union of all sub-stage ``__needs_keys__`` sets.  Populated
        automatically during ``__init__``.
    __provides_keys__ : set[str]
        Union of all sub-stage ``__provides_keys__`` sets.  Populated
        automatically during ``__init__``.

    Examples
    --------
    >>> from nvalchemi.dynamics import FusedStage, BaseDynamics
    >>> dynamics0 = BaseDynamics(model=model)
    >>> dynamics1 = BaseDynamics(model=model)
    >>> fused = FusedStage(sub_stages=[(0, dynamics0), (1, dynamics1)])
    >>> fused.exit_status
    2
    """

    def __init__(
        self,
        sub_stages: list[tuple[int, BaseDynamics]],
        *,
        entry_status: int = 0,
        exit_status: int = -1,
        compile_step: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
        init_fn: Callable[[Batch], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the fused stage.

        Parameters
        ----------
        sub_stages : list[tuple[int, BaseDynamics]]
            Ordered ``(status_code, dynamics)`` pairs.
        entry_status : int, optional
            Status code assigned to incoming samples. Default 0.
        exit_status : int, optional
            Status code that triggers graduation. Auto-set to ``len(sub_stages)``
            if -1. Default -1.
        compile_step : bool, optional
            If ``True``, compile the step method with ``torch.compile``.
            Default ``False``.
        compile_kwargs : dict[str, Any] | None, optional
            Keyword arguments for ``torch.compile``. Default ``None``.
        init_fn : Callable[[Batch], None] | None, optional
            Optional callback invoked on the initial batch immediately after
            ``sampler.build_initial_batch()`` returns, before the first step.
            Use this to populate fields that the sampler does not set, such
            as ``velocities`` or ``forces``.  Only called in Mode 2 (inflight
            batching with ``batch=None``).  Default ``None``.
        **kwargs : Any
            Additional keyword arguments forwarded to ``BaseDynamics``.

        Raises
        ------
        ValueError
            If sub-stages have different ``device_type`` values.
        """
        first_dynamics = sub_stages[0][1]
        model = first_dynamics.model

        device_types = {dyn.device_type for _, dyn in sub_stages}
        if len(device_types) > 1:
            per_stage = {code: dyn.device_type for code, dyn in sub_stages}
            raise ValueError(
                f"All sub-stages in a FusedStage must share the same "
                f"device_type, but got: {per_stage}. A FusedStage runs "
                f"on a single device with a shared batch and forward pass."
            )

        super().__init__(model=model, **kwargs)

        self.sub_stages = sub_stages

        self.__needs_keys__ = set().union(
            *(dyn.__needs_keys__ for _, dyn in sub_stages)
        )
        self.__provides_keys__ = set().union(
            *(dyn.__provides_keys__ for _, dyn in sub_stages)
        )

        self.entry_status = entry_status
        self.compile_kwargs: dict[str, Any] = (
            compile_kwargs if compile_kwargs is not None else {}
        )
        self.compile_step = compile_step
        self._compiled_step: (
            Callable[[Batch], tuple[Batch, torch.Tensor | None]] | None
        ) = None

        if exit_status == -1:
            self.exit_status = len(self.sub_stages)
        else:
            self.exit_status = exit_status

        self.convergence_check_frequency: int = 1

        self.init_fn = init_fn

        self.fused_hooks: list[Hook] = []

        for i in range(len(self.sub_stages)):
            source_code, source_dynamics = self.sub_stages[i]
            if i + 1 < len(self.sub_stages):
                target_code, _ = self.sub_stages[i + 1]
            else:
                # The last stage graduates directly to exit_status when it
                # declares a convergence criterion.
                if source_dynamics.convergence_hook is None:
                    continue
                target_code = self.exit_status

            # Remove duplicate migration hooks with the same (source_status, target_status)
            # to prevent double-fire after __add__ reconstruction.
            source_dynamics.hooks = [
                h
                for h in source_dynamics.hooks
                if not (
                    isinstance(h, ConvergenceHook)
                    and hasattr(h, "source_status")
                    and h.source_status == source_code
                    and hasattr(h, "target_status")
                    and h.target_status == target_code
                )
            ]

            criteria = None
            if source_dynamics.convergence_hook is not None:
                criteria = source_dynamics.convergence_hook.criteria
            hook = ConvergenceHook(
                criteria=criteria,
                source_status=source_code,
                target_status=target_code,
            )
            source_dynamics.register_hook(hook)

        for status_code, dynamics in self.sub_stages:
            if dynamics.n_steps is not None:
                counter_key = f"n_steps_counter_{status_code}"
                BaseDynamics.register_bookkeeping_key(
                    counter_key,
                    lambda n, dev: torch.zeros(n, 1, dtype=torch.long, device=dev),
                )

        if self.compile_step:
            self.compile()

    def __repr__(self) -> str:
        """Return a human-readable summary of the fused stage."""
        stages_repr = ", ".join(
            f"{code}:{type(dyn).__name__}" for code, dyn in self.sub_stages
        )
        compiled = self._compiled_step is not None
        return (
            f"FusedStage("
            f"sub_stages=[{stages_repr}], "
            f"entry_status={self.entry_status}, "
            f"exit_status={self.exit_status}, "
            f"compiled={compiled}, "
            f"step_count={self.step_count})"
        )

    def compile(self, **kwargs: Any) -> FusedStage:
        """Compile the fused step with ``torch.compile``.

        Merges *kwargs* with any ``compile_kwargs`` stored at construction
        time (values passed here take precedence), then wraps
        ``_step_impl`` with ``torch.compile``.  Calling this method also
        sets ``compile_step = True`` so that the ``step`` dispatch path
        uses the compiled callable.

        This method is idempotent in intent but **will** re-compile if
        called again (e.g. with different kwargs).

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments forwarded to ``torch.compile``.  Merged
            with ``compile_kwargs`` from ``__init__``; values here win.

        Returns
        -------
        FusedStage
            This instance, enabling fluent chaining such as
            ``fused.compile(fullgraph=True).run(batch)``.
        """
        merged = {**self.compile_kwargs, **kwargs}
        self.compile_kwargs = merged
        self.compile_step = True
        self._compiled_step = torch.compile(self._step_impl, **merged)
        return self

    def __enter__(self) -> FusedStage:
        """Enter the stream context and propagate to all sub-stages.

        Calls the parent ``__enter__`` to create and enter a CUDA stream
        context, then sets every sub-stage's ``_stream`` reference to the
        same stream so that all computation runs on a single dedicated
        stream.

        If ``compile_step`` is ``True`` but compilation has not yet been
        performed (e.g. because the stage was created via ``__add__``),
        compilation is triggered here automatically.

        Returns
        -------
        FusedStage
            This instance.
        """
        super().__enter__()
        for _, dynamics in self.sub_stages:
            dynamics._stream = self._stream
        if self.compile_step and self._compiled_step is None:
            self.compile()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the stream context and clear sub-stage stream references.

        Clears every sub-stage's ``_stream`` reference, then delegates
        to the parent ``__exit__`` to exit the ``StreamContext`` and
        clean up.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type, if any.
        exc_val : BaseException | None
            Exception value, if any.
        exc_tb : Any
            Exception traceback, if any.
        """
        for _, dynamics in self.sub_stages:
            dynamics._stream = None
        super().__exit__(exc_type, exc_val, exc_tb)

    def _sync_state_to_batch(
        self,
        remaining_indices: "torch.Tensor",
        n_new: int,
        template_batch: Batch,
    ) -> None:
        """Fan out state sync to all sub-stages.

        ``FusedStage`` itself holds no ``_state``; each sub-stage does.
        This override delegates to every sub-stage so that inflight
        batch refills (via :meth:`~BaseDynamics._refill_check`) keep
        each sub-stage's ``_state`` aligned with the new batch
        composition.

        Parameters
        ----------
        remaining_indices : torch.Tensor
            Integer indices of systems that remain after graduation.
        n_new : int
            Number of newly admitted replacement systems.
        template_batch : Batch
            The updated batch; provides device/dtype for new-state init.
        """
        for _, sub_stage in self.sub_stages:
            sub_stage._sync_state_to_batch(remaining_indices, n_new, template_batch)

    def _ensure_bookkeeping_fields(self, batch: Batch) -> None:
        """Auto-initialize status and registered bookkeeping fields if absent.

        Parameters
        ----------
        batch : Batch
            The batch to check and initialize fields on.
        """
        for key, default_fn in self._bookkeeping_keys.items():
            if getattr(batch, key, None) is None:
                batch[key] = default_fn(batch.num_graphs, batch.device)

    def register_fused_hook(self, hook: Hook) -> None:
        """Register a hook that fires at the FusedStage level on the full batch.

        Unlike hooks registered on individual sub-stages (which only receive
        the sub-batched view), fused hooks observe the complete batch at the
        ``BEFORE_STEP``, ``AFTER_STEP``, ``BEFORE_COMPUTE``, and
        ``AFTER_COMPUTE`` stages of every fused step.

        Parameters
        ----------
        hook : Hook
            The hook to register.  Only ``BEFORE_STEP`` and ``AFTER_STEP``
            stages are meaningful at the fused level; other stages are
            silently accepted but will not fire during normal execution.

        Raises
        ------
        ValueError
            If ``hook.frequency`` is not a positive integer.
        """
        if not isinstance(hook.frequency, int) or hook.frequency < 1:
            raise ValueError(
                f"Hook {hook!r} has frequency={hook.frequency!r}. "
                "frequency must be a positive integer (>= 1)."
            )
        self.fused_hooks.append(hook)

    def _call_fused_hooks(self, stage: DynamicsStage, batch: Batch) -> None:
        """Invoke all fused hooks registered for the given stage.

        Parameters
        ----------
        stage : DynamicsStage
            The hook stage to fire.
        batch : Batch
            The current full batch.
        """
        ctx = self._build_context(batch)
        for hook in self.fused_hooks:
            runs_on_stage = getattr(hook, "_runs_on_stage", None)
            if runs_on_stage is not None:
                if not runs_on_stage(stage):
                    continue
            elif stage != hook.stage:
                continue
            if self.step_count % hook.frequency == 0:
                hook(ctx, stage)

    def _step_impl(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Internal step implementation (may be compiled).

        Performs the following sequence:
        1. Fire BEFORE_STEP hooks (fused, self, sub-stages).
        2. For each sub-stage: fire BEFORE_PRE_UPDATE, run pre_update
           (positions advance to r(t+dt)).
        3. Fire BEFORE_COMPUTE hooks (fused / self, e.g. NeighborListHook
           with new positions) → single shared forward pass → AFTER_COMPUTE.
        4. For each sub-stage: run post_update (final velocity kick at
           r(t+dt) forces), fire AFTER_POST_UPDATE.
        5. Fire AFTER_STEP hooks on each sub-stage.
        6. Snapshot status, check convergence per sub-stage and fire
           ON_CONVERGE if triggered.
        7. Increment step_count for FusedStage and all sub-stages.
        8. Identify samples that newly graduated during this step.

        Parameters
        ----------
        batch : Batch
            The batch with a ``status`` field.

        Returns
        -------
        tuple[Batch, torch.Tensor | None]
            The updated batch, and a 1-D integer tensor of sample indices
            that newly graduated (reached ``exit_status``) during this step,
            or ``None`` if no samples graduated.
        """
        self._ensure_bookkeeping_fields(batch)

        self._call_fused_hooks(DynamicsStage.BEFORE_STEP, batch)
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        for _, dynamics in self.sub_stages:
            dynamics._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        # Phase 1 — pre_update for each sub-stage.
        # This moves positions to r(t+dt) so that the shared compute can
        # evaluate forces at the correct (updated) positions.
        status = batch.status
        if status.dim() == 2:
            status = status.squeeze(-1)

        stage_active_masks: list[torch.Tensor] = []
        for status_code, dynamics in self.sub_stages:
            mask = status == status_code
            stage_active_masks.append(mask)
            dynamics._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch)
            if mask.any():
                dynamics._masked_pre_update(batch, mask)

        # Phase 2 — shared forward pass at the updated positions.
        self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)

        outputs: ModelOutputs = self.compute(batch)

        # TODO: update this when `batch` structure is done
        for key, tensor in outputs.items():
            if key not in ("forces", "energy"):
                batch[key] = tensor

        self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
        for _, dynamics in self.sub_stages:
            dynamics._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

        # Phase 3 — post_update for each sub-stage, now with forces at r(t+dt).
        for status_code, dynamics in self.sub_stages:
            mask = status == status_code
            if mask.any():
                dynamics._masked_post_update(batch, mask)
            dynamics._call_hooks(DynamicsStage.AFTER_POST_UPDATE, batch)

        for _, dynamics in self.sub_stages:
            dynamics._call_hooks(DynamicsStage.AFTER_STEP, batch)

        self._call_hooks(DynamicsStage.AFTER_STEP, batch)
        self._call_fused_hooks(DynamicsStage.AFTER_STEP, batch)

        for i, (status_code, dynamics) in enumerate(self.sub_stages):
            if dynamics.n_steps is None:
                continue

            counter_key = f"n_steps_counter_{status_code}"

            if getattr(batch, counter_key, None) is None:
                batch[counter_key] = torch.zeros(
                    batch.num_graphs, 1, dtype=torch.long, device=batch.device
                )

            counter = getattr(batch, counter_key)
            cur_status = (
                batch.status.squeeze(-1) if batch.status.dim() == 2 else batch.status
            )
            active = cur_status == status_code

            counter[active] += 1

            next_status = (
                self.sub_stages[i + 1][0]
                if i + 1 < len(self.sub_stages)
                else self.exit_status
            )

            migrate = active & (counter.squeeze(-1) >= dynamics.n_steps)
            if migrate.any():
                batch.status.view(-1)[migrate] = next_status
                counter[migrate] = 0  # Reset for next system in this slot

        pre_converge_status = batch.status.clone()
        if pre_converge_status.dim() == 2:
            pre_converge_status = pre_converge_status.squeeze(-1)

        for active_mask, (_, dynamics) in zip(
            stage_active_masks, self.sub_stages, strict=True
        ):
            converged = dynamics._check_convergence(batch)
            if converged is None:
                dynamics._last_converged = None
                continue

            stage_converged = torch.zeros(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            )
            stage_converged[converged] = True
            stage_converged &= active_mask

            if not stage_converged.any():
                dynamics._last_converged = None
                continue

            dynamics._last_converged = torch.where(stage_converged)[0]
            dynamics._call_hooks(DynamicsStage.ON_CONVERGE, batch)

        self.step_count += 1
        for _, dynamics in self.sub_stages:
            dynamics.step_count += 1

        post_status = batch.status
        if post_status.dim() == 2:
            post_status = post_status.squeeze(-1)
        newly_graduated = (pre_converge_status < self.exit_status) & (
            post_status >= self.exit_status
        )
        exit_converged: torch.Tensor | None = (
            torch.where(newly_graduated)[0] if newly_graduated.any() else None
        )

        return batch, exit_converged

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Execute one fused step: single forward pass + masked updates.

        If ``compile_step=True`` was set, this delegates to the compiled
        step implementation.

        Parameters
        ----------
        batch : Batch
            The batch with a ``status`` field.

        Returns
        -------
        tuple[Batch, torch.Tensor | None]
            The updated batch, and a 1-D integer tensor of sample indices
            that newly graduated (reached ``exit_status``) during this step,
            or ``None`` if no samples graduated.
        """
        if self._compiled_step is not None:
            return self._compiled_step(batch)
        return self._step_impl(batch)

    def __call__(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Call the ``step`` method on a batch."""
        return self.step(batch)

    @staticmethod
    def all_complete(batch: Batch, exit_status: int) -> bool:
        """Check if all samples have reached the exit status.

        Parameters
        ----------
        batch : Batch
            The current batch.
        exit_status : int
            The status code that indicates completion.

        Returns
        -------
        bool
            ``True`` if every sample has ``status == exit_status``.
        """
        if not hasattr(batch, "status") or batch.status is None:
            return False
        status = batch.status
        if status.dim() == 2:
            status = status.squeeze(-1)
        return bool((status == exit_status).all())

    def _open_hooks(self) -> None:
        """Enter context-manager hooks on this stage, fused hooks, and sub-stages."""
        super()._open_hooks()

        seen: set[int] = set()
        for hook in self.fused_hooks:
            hook_id = id(hook)
            if hook_id not in seen and hasattr(hook, "__enter__"):
                seen.add(hook_id)
                hook.__enter__()

        for _, dynamics in self.sub_stages:
            dynamics._open_hooks()

    def _close_hooks(self) -> None:
        """Exit context-manager hooks on this stage, fused hooks, and sub-stages."""
        super()._close_hooks()

        seen: set[int] = set()
        for hook in self.fused_hooks:
            hook_id = id(hook)
            if hook_id in seen:
                continue
            seen.add(hook_id)
            if hasattr(hook, "__exit__"):
                hook.__exit__(None, None, None)
            elif hasattr(hook, "close"):
                hook.close()

        for _, dynamics in self.sub_stages:
            dynamics._close_hooks()

    def run(
        self, batch: Batch | None = None, n_steps: int | None = None
    ) -> Batch | None:
        """Run the fused stage until all samples converge or the sampler is exhausted.

        Supports two modes of execution:

        **Mode 1 (external batch loop):** When ``batch`` is provided,
        runs the dynamics until ``all_complete`` or until ``n_steps``
        have been executed (whichever comes first).

        **Mode 2 (inflight batching):** When ``batch is None`` and a
        sampler is configured, builds the initial batch from the
        sampler and replaces graduated samples every
        ``refill_frequency`` steps.

        .. note::

            In Mode 2, ``refill_check`` replaces graduated samples by
            extracting remaining graphs via :meth:`Batch.index_select`,
            requesting replacements from the sampler, and appending them
            via :meth:`Batch.append`.  This produces a **new** ``Batch``
            object; the ``batch = result`` reassignment in the loop body
            updates the local reference.  ``None`` is returned when the
            sampler is exhausted and no active samples remain, which
            triggers termination.

        Parameters
        ----------
        batch : Batch | None, optional
            The initial batch. If ``None``, uses the sampler to build one.
        n_steps : int | None, optional
            Maximum number of steps to run.  When ``None``, falls back to
            ``self.n_steps``.  When both are ``None``, the loop runs until
            ``all_complete`` (Mode 1) or sampler exhaustion (Mode 2).
            Sub-stages that have no exit criterion (e.g. a plain MD stage)
            will loop forever without a step limit, so always pass
            ``n_steps`` when such a stage is the final sub-stage.
            Note: sub-stages with ``n_steps`` set use that value as a per-system
            step budget for automatic migration to the next stage.

        Returns
        -------
        Batch | None
            The batch after all steps, or ``None`` if the sampler was
            exhausted and all samples graduated.

        Raises
        ------
        ValueError
            If ``batch is None`` and no sampler is configured.
        """
        if batch is None:
            if self.sampler is None:
                raise ValueError("No batch provided and no sampler configured.")
            batch = self.sampler.build_initial_batch()
            if self.init_fn is not None:
                self.init_fn(batch)
            self.active_batch = batch

        # Ensure bookkeeping fields are present before the loop begins.
        self._ensure_bookkeeping_fields(batch)

        resolved_steps = n_steps if n_steps is not None else self.n_steps

        self._open_hooks()
        try:
            # Prime forces before the first step so that pre_update can use
            # them.  _step_impl now runs pre_update BEFORE compute, so without
            # this initial forward pass the first step would integrate with
            # zero (uninitialised) forces.
            self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
            self.compute(batch)
            self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

            step_num = 0
            while True:
                batch, _converged = self.step(batch)

                if (
                    self.sampler is not None
                    and (step_num + 1) % self.refill_frequency == 0
                ):
                    result = self.refill_check(batch, self.exit_status)
                    if result is None:
                        self.active_batch = None
                        return None
                    batch = result
                    self.active_batch = batch
                elif (
                    self.sampler is None
                    and (step_num + 1) % self.convergence_check_frequency == 0
                    and self.all_complete(batch, self.exit_status)
                ):
                    break

                step_num += 1

                if resolved_steps is not None and step_num >= resolved_steps:
                    break

            return batch
        finally:
            self._close_hooks()

    def refill_check(self, batch: Batch, exit_status: int) -> Batch | None:
        """Replace graduated samples and clear stale convergence indices.

        Delegates to the parent :meth:`BaseDynamics.refill_check` to remove
        graduated graphs and append replacements from the sampler.  When the
        batch composition changes, ``_last_converged`` is cleared on this
        ``FusedStage`` and all its sub-stages so that subsequent hooks do not
        receive an invalid ``converged_mask``.

        Parameters
        ----------
        batch : Batch
            The current batch with a ``status`` field.
        exit_status : int
            Status code indicating graduation.

        Returns
        -------
        Batch | None
            A new batch with graduated graphs replaced by fresh samples,
            or ``None`` if no active samples remain.
        """
        result = super().refill_check(batch, exit_status)
        # Clear stale convergence indices since the batch configuration has changed
        self._last_converged = None
        for _, dynamics in self.sub_stages:
            dynamics._last_converged = None
        return result

    def __add__(self, other: BaseDynamics) -> FusedStage:
        """Append a sub-stage to this fused stage via ``fused + dyn``.

        Parameters
        ----------
        other : BaseDynamics
            The dynamics to append to this fused stage.

        Returns
        -------
        FusedStage
            A new fused stage with the additional sub-stage appended.

        Raises
        ------
        TypeError
            If ``other`` is not a ``BaseDynamics`` instance.

        Notes
        -----
        Compilation is deferred when composing via ``+``.  If the source
        ``FusedStage`` had ``compile_step=True``, the returned stage
        preserves that intent but does **not** compile eagerly.  Call
        ``.compile()`` explicitly or enter the context manager to trigger
        compilation.
        """
        if not isinstance(other, BaseDynamics):
            raise TypeError(
                "Cannot append stage: other must be a BaseDynamics instance. "
                f"Got {type(other).__name__} instead."
            )
        next_code = len(self.sub_stages)
        new_sub_stages = list(self.sub_stages) + [(next_code, other)]
        new_fused = FusedStage(
            sub_stages=new_sub_stages,
            entry_status=self.entry_status,
            compile_step=False,
            compile_kwargs=self.compile_kwargs,
        )
        # Defer compilation to __enter__ or an explicit .compile() call.
        new_fused.compile_step = self.compile_step
        return new_fused


class DistributedPipeline:
    """Orchestrates multi-rank pipeline execution.

    Maps GPU ranks to pipeline stages and coordinates the distributed
    step loop.  Each rank executes only its assigned stage.

    Parameters
    ----------
    stages : dict[int, BaseDynamics]
        Mapping from rank to its assigned pipeline stage.
    synchronized : bool
        If ``True``, insert a global ``dist.barrier()`` across **all**
        pipeline ranks after every ``step()`` call, forcing every rank to
        complete its current step before any rank proceeds to the next one.
        This is primarily useful for debugging ordering or deadlock issues
        because it eliminates all inter-rank timing skew.

        .. note::

           This is distinct from the per-stage ``comm_mode`` parameter on
           ``_CommunicationMixin``, which controls the blocking behavior
           of *pairwise* ``isend``/``irecv`` between adjacent stages.
           ``synchronized`` enforces a *global* synchronization point
           across the entire pipeline and will significantly reduce
           throughput; it should be disabled (``False``) in production.

    Attributes
    ----------
    stages : dict[int, BaseDynamics]
        Rank-to-stage mapping.
    synchronized : bool
        Whether a global ``dist.barrier()`` is inserted after every step.
    _dist_initialized : bool
        Whether this DistributedPipeline instance initialized the distributed
        process group (used to determine cleanup responsibility).

    Examples
    --------
    >>> # Context manager usage (recommended):
    >>> pipeline = DistributedPipeline(stages={0: opt_stage, 1: md_stage})
    >>> with pipeline:
    ...     pipeline.run()
    ...
    >>> # Manual usage:
    >>> pipeline = DistributedPipeline(stages={0: opt_stage, 1: md_stage})
    >>> pipeline.init_distributed()
    >>> pipeline.setup()
    >>> pipeline.run()
    >>> pipeline.cleanup()
    >>> # Composing multiple pipelines together
    >>> full_pipeline = pipe1 | pipe2 | pipe3
    >>> with full_pipeline:
    ...     pipeline.run()
    ...
    """

    def __init__(
        self,
        stages: dict[int, BaseDynamics],
        synchronized: bool = False,
        debug_mode: bool = False,
        **dist_kwargs: Any,
    ) -> None:
        """Initialize the pipeline.

        Parameters
        ----------
        stages : dict[int, BaseDynamics]
            Mapping from global rank to pipeline stage.
        synchronized : bool, optional
            If ``True``, insert a global ``dist.barrier()`` across all
            pipeline ranks after every step, preventing any rank from
            advancing until all ranks have completed the current step.
            Useful for debugging but significantly reduces throughput.
            See the class-level docstring for how this differs from the
            per-stage ``comm_mode``.  Default ``False``.
        debug_mode : bool, optional
            When ``True``, emit detailed ``loguru.debug`` diagnostics
            for inter-rank communication and pipeline orchestration.
            Propagated to all stages during ``setup()``. Default ``False``.
        **dist_kwargs : Any
            Additional keyword arguments for ``torch.distributed.init_process_group``.
        """
        dist_kwargs.setdefault(
            "backend", "nccl" if dist.is_nccl_available() else "gloo"
        )
        self.stages = stages
        self.synchronized = synchronized
        self._dist_initialized: bool = False
        self._dist_kwargs = dist_kwargs
        self._done_tensor: torch.Tensor | None = None
        self.debug_mode = debug_mode
        self._templates_shared: bool = False

    def __or__(self, other: BaseDynamics | DistributedPipeline) -> DistributedPipeline:
        """Append a stage or merge another pipeline via the ``|`` operator.

        Supports three composition patterns::

            pipeline | dynamics        # append one stage
            pipeline | pipeline        # merge two pipelines
            dyn1 | dyn2 | dyn3 | ...  # left-associative chaining

        Ranks in the resulting pipeline are renumbered to form a
        contiguous sequence. Source/sink dependencies (``prior_rank`` /
        ``next_rank``) are wired when ``setup()`` is called (e.g. via the
        context manager or ``run()``).

        Parameters
        ----------
        other : BaseDynamics | DistributedPipeline
            A single dynamics stage to append, or another pipeline
            whose stages will be absorbed (renumbered) after this
            pipeline's stages.

        Returns
        -------
        DistributedPipeline
            A new pipeline containing all stages with stages mapped to
            contiguous ranks.

        Raises
        ------
        TypeError
            If ``other`` is not a ``BaseDynamics`` or
            ``DistributedPipeline`` instance.
        RuntimeError
            If ``torch.distributed`` is already initialized. Pipeline
            composition must occur before entering the pipeline context
            or calling ``run()``.
        """
        if dist.is_initialized():
            raise RuntimeError(
                "Cannot compose pipelines after torch.distributed has been "
                "initialized. Build the full pipeline topology before "
                "entering the pipeline context or calling run()."
            )
        if isinstance(other, DistributedPipeline):
            base_rank = max(self.stages.keys()) + 1
            new_stages = {**self.stages}
            for i, rank in enumerate(sorted(other.stages.keys())):
                new_stages[base_rank + i] = other.stages[rank]
            pipeline = DistributedPipeline(
                stages=new_stages,
                synchronized=self.synchronized or other.synchronized,
                **self._dist_kwargs,
            )
        elif isinstance(other, BaseDynamics):
            next_rank = max(self.stages.keys()) + 1
            new_stages = {**self.stages, next_rank: other}
            pipeline = DistributedPipeline(
                stages=new_stages,
                synchronized=self.synchronized,
                **self._dist_kwargs,
            )
        else:
            raise TypeError(
                f"Right operand of | must be a BaseDynamics or "
                f"DistributedPipeline instance, got {type(other).__name__}."
            )
        return pipeline

    def _validate_world_size(self) -> None:
        """Validate that the distributed world size matches the expected ranks.

        Compares ``torch.distributed.get_world_size()`` against the number
        of configured pipeline stages. A mismatch indicates that the
        ``torchrun`` launch configuration does not match the pipeline
        topology.

        This method is a no-op if ``torch.distributed`` is not initialized
        (e.g., during local testing).

        Raises
        ------
        RuntimeError
            If the world size does not match the number of configured
            pipeline stages.
        """
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        expected = len(self.stages)
        if world_size != expected:
            raise RuntimeError(
                f"DistributedPipeline expects {expected} ranks (stages configured "
                f"for ranks {sorted(self.stages.keys())}), but "
                f"torch.distributed world_size is {world_size}. "
            )

    def setup(self) -> None:
        """Wire up ``prior_rank`` / ``next_rank`` between adjacent stages.

        Sorts stages by rank and connects each stage to its
        predecessor and successor.

        Raises
        ------
        ValueError
            If fewer than 2 stages are provided, or if adjacent stages
            have mismatched buffer configurations.
        RuntimeError
            If the world size does not match the number of configured
            pipeline stages.
        """
        sorted_ranks = sorted(self.stages.keys())
        if len(sorted_ranks) < 2:
            raise ValueError("Pipeline requires at least 2 stages.")

        self._validate_world_size()

        for i, rank in enumerate(sorted_ranks):
            stage = self.stages[rank]
            if stage.prior_rank == -1:
                stage.prior_rank = sorted_ranks[i - 1] if i > 0 else None
            if stage.next_rank == -1:
                stage.next_rank = (
                    sorted_ranks[i + 1] if i < len(sorted_ranks) - 1 else None
                )

        for i in range(len(sorted_ranks) - 1):
            rank = sorted_ranks[i]
            next_rank = sorted_ranks[i + 1]
            sender = self.stages[rank]
            receiver = self.stages[next_rank]
            s_cfg = getattr(sender, "buffer_config", None)
            r_cfg = getattr(receiver, "buffer_config", None)
            if s_cfg is None or r_cfg is None:
                raise ValueError(
                    "All stages in a DistributedPipeline must have buffer_config set. "
                    f"Stage on rank {rank} has buffer_config={s_cfg}, "
                    f"stage on rank {next_rank} has buffer_config={r_cfg}."
                )
            if s_cfg != r_cfg:
                raise ValueError(
                    f"Buffer configuration mismatch between rank {rank} "
                    f"and rank {next_rank}: sender has "
                    f"BufferConfig(num_systems={s_cfg.num_systems}, "
                    f"num_nodes={s_cfg.num_nodes}, num_edges={s_cfg.num_edges}), "
                    f"receiver has "
                    f"BufferConfig(num_systems={r_cfg.num_systems}, "
                    f"num_nodes={r_cfg.num_nodes}, num_edges={r_cfg.num_edges}). "
                    f"Adjacent stages must use identical buffer configurations."
                )

        n_stages = len(sorted_ranks)
        device = self.local_stage.device
        self._done_tensor = torch.zeros(n_stages, dtype=torch.int32, device=device)
        # move model to device if it isn't there already
        model = self.local_stage.model
        if not callable(getattr(model, "to", None)):
            raise RuntimeError(
                "Model expected to possess `to()` method for device"
                f" and casting behavior. Passed model is type {type(model)}"
                " so ensure class contains this method."
            )
        else:
            self.local_stage.model = model.to(device)

        for stage in self.stages.values():
            stage.debug_mode = self.debug_mode

    def _share_templates(self) -> None:
        """Compute batch schema templates for all stages via local iteration.

        Since ``DistributedPipeline`` has all stages in ``self.stages`` on
        every rank, templates can be computed locally without inter-rank
        communication. Inflight (first) stages build their initial batch
        from the sampler and cache an ``empty_like`` template. Downstream
        stages derive their template from the upstream stage's cached
        template.

        This method is idempotent; repeated calls are no-ops once
        templates have been computed.
        """
        if self._templates_shared:
            return
        self._templates_shared = True

        for rank in sorted(self.stages.keys()):
            stage = self.stages[rank]

            if stage.is_first_stage and stage.inflight_mode:
                if stage.active_batch is None:
                    stage.active_batch = stage.sampler.build_initial_batch()
                if stage.active_batch is not None:
                    if stage.active_batch.device != stage.device:
                        stage.active_batch = stage.active_batch.to(stage.device)
                    stage._recv_template = Batch.empty_like(
                        stage.active_batch, device=stage.device
                    )
                    if self.debug_mode:
                        logger.debug(
                            "[rank {}] computed template from inflight sampler",
                            rank,
                        )
            elif stage.prior_rank is not None:
                upstream = self.stages[stage.prior_rank]
                if upstream._recv_template is not None:
                    stage._recv_template = Batch.empty_like(
                        upstream._recv_template, device=stage.device
                    )
                    if self.debug_mode:
                        logger.debug(
                            "[rank {}] computed template from upstream rank {}",
                            rank,
                            stage.prior_rank,
                        )

    @property
    def local_rank(self) -> int:
        """Get the local rank for this process."""
        rank = 0
        if dist.is_initialized():
            rank = dist.get_node_local_rank()
        return rank

    @property
    def global_rank(self) -> int:
        """Get the global rank for this process."""
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()
        return rank

    @property
    def local_stage(self) -> BaseDynamics:
        """Get the stage associated with the rank this is executed on."""
        return self.stages[self.global_rank]

    def step(self) -> None:
        """Execute one timestep for the local rank's stage.

        The stage (a ``BaseDynamics`` subclass) handles both the dynamics
        step and buffer synchronization.

        Supports two modes for the first stage:

        **Mode 1 (external batch loop):** Standard flow where the first
        stage receives from ``_prestep_sync_buffers`` like other stages.

        **Mode 2 (inflight batching):** When the first stage has
        ``inflight_mode=True`` (i.e., a sampler is configured), it builds
        the initial batch from the sampler and refills graduated samples
        instead of receiving from a prior stage.

        When ``self.synchronized`` is ``True``, a global
        ``dist.barrier()`` is issued at the end of each step so that no
        rank advances until every rank in the pipeline has finished the
        current step.

        Raises
        ------
        RuntimeError
            If ``torch.distributed`` is not initialized, or if the world
            size does not match the number of configured pipeline stages.
        KeyError
            If the current rank is not in the global rank stage mapping.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Call torch.distributed.init_process_group() first."
            )

        rank = self.global_rank
        if rank not in self.stages:
            raise KeyError(f"Rank {rank} is not assigned to any pipeline stage.")

        stage = self.stages[rank]
        stage_type = type(stage).__name__

        if stage.is_first_stage and stage.inflight_mode:
            n_graphs = stage.active_batch.num_graphs if stage.active_batch else 0
            if self.debug_mode:
                logger.debug(
                    "[rank {}] inflight step begin | stage={} batch_size={}",
                    rank,
                    stage_type,
                    n_graphs,
                )
            if stage.active_batch is None and not stage.done:
                try:
                    stage.active_batch = stage.sampler.build_initial_batch()
                except RuntimeError:
                    stage.active_batch = None
                if stage.active_batch is not None:
                    if stage.active_batch.device != stage.device:
                        stage.active_batch = stage.active_batch.to(stage.device)
                    if self.debug_mode:
                        logger.debug(
                            "[rank {}] built initial batch, {} graphs",
                            rank,
                            stage.active_batch.num_graphs,
                        )
                else:
                    if self.debug_mode:
                        logger.debug(
                            "[rank {}] sampler exhausted at build, marking done", rank
                        )
                    stage.done = True

            if stage.active_batch is not None:
                stage._ensure_buffers(stage.active_batch)
            elif stage._recv_template is not None:
                stage._ensure_buffers(stage._recv_template)

            if stage.active_batch is not None:
                stage.active_batch, converged_indices = stage.step(stage.active_batch)
                n_conv = (
                    converged_indices.numel() if converged_indices is not None else 0
                )
                if self.debug_mode:
                    logger.debug(
                        "[rank {}] step done | converged={} remaining={}",
                        rank,
                        n_conv,
                        stage.active_batch.num_graphs if stage.active_batch else 0,
                    )

                stage._poststep_sync_buffers(converged_indices)

                if hasattr(stage, "exit_status"):
                    exit_status = stage.exit_status
                else:
                    exit_status = 1
                if stage.active_batch is not None:
                    result = stage.refill_check(stage.active_batch, exit_status)
                    stage.active_batch = result
                    if result is None:
                        if self.debug_mode:
                            logger.debug(
                                "[rank {}] sampler exhausted, marking done", rank
                            )
                        stage.done = True
                else:
                    if self.debug_mode:
                        logger.debug(
                            "[rank {}] active_batch is None after poststep, "
                            "rebuilding from sampler",
                            rank,
                        )
                    try:
                        stage.active_batch = stage.sampler.build_initial_batch()
                    except RuntimeError:
                        stage.active_batch = None
                    if stage.active_batch is not None:
                        if stage.active_batch.device != stage.device:
                            stage.active_batch = stage.active_batch.to(stage.device)
                    else:
                        if self.debug_mode:
                            logger.debug(
                                "[rank {}] sampler exhausted, marking done", rank
                            )
                        stage.done = True
            elif stage.next_rank is not None:
                if self.debug_mode:
                    logger.debug(
                        "[rank {}] done, sending empty buffer to rank {}",
                        rank,
                        stage.next_rank,
                    )
                stage.send_buffer.isend(dst=stage.next_rank).wait()
        else:
            n_graphs = stage.active_batch.num_graphs if stage.active_batch else 0
            if self.debug_mode:
                logger.debug(
                    "[rank {}] downstream step begin | stage={} batch_size={}",
                    rank,
                    stage_type,
                    n_graphs,
                )
            if stage.active_batch is not None:
                stage._ensure_buffers(stage.active_batch)

            stage._prestep_sync_buffers()
            stage._complete_pending_recv()

            converged_indices = None
            if stage.active_batch is not None and stage.active_batch.num_graphs > 0:
                stage.active_batch, converged_indices = stage.step(stage.active_batch)
                n_conv = (
                    converged_indices.numel() if converged_indices is not None else 0
                )
                if self.debug_mode:
                    logger.debug(
                        "[rank {}] step done | converged={} remaining={}",
                        rank,
                        n_conv,
                        stage.active_batch.num_graphs if stage.active_batch else 0,
                    )
            elif stage.active_batch is not None:
                if self.debug_mode:
                    logger.debug(
                        "[rank {}] skipping step, active_batch has 0 graphs", rank
                    )

            stage._poststep_sync_buffers(converged_indices)

            # Auto-terminate downstream stages: if the upstream is done
            # and this stage has no remaining work, mark it as done.
            n_active = (
                stage.active_batch.num_graphs if stage.active_batch is not None else 0
            )
            upstream_done = (
                self._done_tensor is not None
                and stage.prior_rank is not None
                and bool(self._done_tensor[stage.prior_rank])
            )
            if upstream_done and n_active == 0 and not stage.done:
                if self.debug_mode:
                    logger.debug(
                        "[rank {}] upstream rank {} done and no active work, marking done",
                        rank,
                        stage.prior_rank,
                    )
                stage.done = True

        if self.synchronized:
            if self.debug_mode:
                logger.debug("[rank {}] waiting at barrier", rank)
            dist.barrier()

    def _sync_done_flags(self) -> bool:
        """Synchronize ``done`` flags across all ranks via ``all_reduce``.

        Each rank writes its local stage's ``done`` status into the
        shared ``_done_tensor`` at its position, then an ``all_reduce``
        (``MAX``) broadcasts the flags so every rank sees the global
        state.

        Returns
        -------
        bool
            ``True`` if **all** stages report ``done``.
        """
        if self._done_tensor is None:
            raise RuntimeError("_done_tensor is not initialized. Call setup() first.")

        stage = self.local_stage
        self._done_tensor[self.global_rank] = int(stage.done)

        if dist.is_initialized():
            dist.all_reduce(self._done_tensor, op=dist.ReduceOp.MAX)

        all_done = bool(self._done_tensor.all())
        if self.debug_mode:
            logger.debug(
                "[rank {}] done_flags={} all_done={}",
                self.global_rank,
                self._done_tensor.tolist(),
                all_done,
            )
        return all_done

    def run(self) -> None:
        """Run the pipeline loop until all stages report done.

        After each ``step()``, an ``all_reduce`` synchronizes the
        ``done`` flags across all ranks so that every process can
        observe the global termination state.
        """
        self.setup()
        self._share_templates()
        iteration = 0
        while True:
            if self.debug_mode:
                logger.debug(
                    "[rank {}] === pipeline iteration {} ===",
                    self.global_rank,
                    iteration,
                )
            self.step()
            if self._sync_done_flags():
                if self.debug_mode:
                    logger.debug("[rank {}] all stages done, exiting", self.global_rank)
                break
            iteration += 1

    def init_distributed(self) -> None:
        """Initialize the ``torch.distributed`` process group.

        If ``torch.distributed`` is already initialized, this method is
        a no-op.  Otherwise, it calls
        ``torch.distributed.init_process_group(**self._dist_kwargs)``.

        The backend and other distributed options are configured via the
        constructor's ``**dist_kwargs`` parameter.

        Notes
        -----
        When launching with ``torchrun``, the process group is typically
        already initialized.  This method provides a convenient fallback
        for scripts that do not use ``torchrun``.
        """
        if dist.is_initialized():
            return
        dist.init_process_group(**self._dist_kwargs)
        self._dist_initialized = True

    def cleanup(self) -> None:
        """Destroy the ``torch.distributed`` process group.

        Only destroys the process group if it was initialized by this
        ``DistributedPipeline`` instance (via :meth:`init_distributed`).  If the
        process group was externally initialized (e.g., by ``torchrun``),
        this method is a no-op.
        """
        if self._dist_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._dist_initialized = False

    def __enter__(self) -> DistributedPipeline:
        """Enter the pipeline context manager.

        Calls :meth:`init_distributed` and :meth:`setup` in sequence.
        The ``setup()`` call also initializes the distributed
        ``_done_tensor`` used for coordinated termination.

        Returns
        -------
        DistributedPipeline
            This pipeline instance.
        """
        self.init_distributed()
        self.setup()
        self._share_templates()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the pipeline context manager.

        Calls :meth:`cleanup` to destroy the process group if it was
        initialized by this pipeline.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type, if any.
        exc_val : BaseException | None
            Exception value, if any.
        exc_tb : Any
            Exception traceback, if any.
        """
        self.cleanup()
