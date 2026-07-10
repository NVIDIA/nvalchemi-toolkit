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
"""Neighbor list hook for on-the-fly neighbor list construction.

This module provides :class:`NeighborListHook`, which runs at the
``BEFORE_COMPUTE`` stage to compute or refresh the neighbor list stored in
the batch before the model forward pass.  It supports an optional Verlet
skin buffer to avoid recomputing neighbors every step.

Both ``MATRIX`` and ``COO`` neighbor formats are supported for dynamic
updates (i.e. updates each dynamics step).  For ``COO`` format the hook
creates or replaces the edges group on the batch each step so that
``batch.neighbor_list`` (shape ``(E, 2)``) and ``batch.neighbor_list_shifts``
(shape ``(E, 3)``, PBC only) are always up to date.  The companion
``Batch.edge_ptr`` property derives the per-atom CSR pointer on demand.

Pre-allocation
--------------
The hook maintains *staging buffers* — persistent GPU tensors that are
refreshed each step via ``Tensor.copy_()`` — to avoid per-step dynamic
allocation inside the ``neighbor_list`` dispatcher.

For PBC systems, ``NeighborListHook`` runs the ``nvalchemiops`` selector
once per batch shape and caches the resulting explicit strategy name (for
example ``batch_cell_list_pair_centric`` or ``batch_cluster_tile``). For
non-PBC systems it keeps the legacy size threshold between naive and cell-list
methods. The selected paths normally allocate auxiliary tensors on demand with
CPU-GPU syncs (e.g. ``.item()`` calls), so
:meth:`NeighborListHook._alloc_nl_kwargs` computes these **once** when the batch
shape is first seen (or changes) and caches them in
``NeighborListHook._buf_nl_kwargs``:

* *Naive, no PBC*: no extra kwargs needed.
* *Naive, PBC*: ``shift_range_per_dimension``, ``num_shifts_per_system``,
  ``max_shifts_per_system``, and ``max_atoms_per_system``.
* *Cell list*: seven cell-list scratch tensors via ``allocate_cell_list``.
* *Cluster tile*: batch cluster-tile sort/group/tile scratch tensors.

**NPT note**: geometry-dependent kwargs (shift ranges, cell-list sizes) are
fixed when the staging buffers are first allocated for a given ``(N, B)``
shape.  For NPT (variable-cell) simulations the pre-computed values may
become stale as the cell changes; accuracy is maintained by keeping the
cutoff + skin well below the shortest cell dimension throughout the run.
"""

from __future__ import annotations

from enum import Enum

import torch
from nvalchemiops.neighbors.base_dispatch import neighbor_list_strategy_run_args
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors import neighbor_list, suggest_neighbor_list_method

try:
    from nvalchemiops.torch.neighbors.batch_cell_list import (
        estimate_batch_cell_list_sizes,
    )
except ImportError:
    estimate_batch_cell_list_sizes = None

try:
    from nvalchemiops.torch.neighbors.batch_cluster_tile import (
        allocate_batch_cluster_tile_list,
        estimate_batch_max_tiles_per_group,
    )
except ImportError:
    allocate_batch_cluster_tile_list = None
    estimate_batch_max_tiles_per_group = None

try:
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list,
        compute_naive_num_shifts,
    )
except ImportError:
    allocate_cell_list = None
    compute_naive_num_shifts = None

try:
    from nvalchemiops.torch.neighbors.rebuild_detection import (
        batch_neighbor_list_needs_rebuild as _batch_nl_needs_rebuild,
    )
except ImportError:
    _batch_nl_needs_rebuild = None

try:
    from nvalchemi.dynamics._ops.neighbor_list_rebuild import (
        batch_neighbor_list_rebuild_inplace as _batch_nl_rebuild_inplace,
    )
except ImportError:
    _batch_nl_rebuild_inplace = None

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import _write_neighbor_data_to_batch


class _NLGraphCache:
    """Per-shape CUDA graph cache for the neighbor-list dispatch chain.

    Used by :class:`NeighborListHook` when constructed with
    ``use_cuda_graph=True``. Owns a private capture stream (one per cache
    instance) so the integrator's default-stream work is unaffected.
    On a NL hook call:

    * ``matches(key)`` returns True if the captured graph corresponds to
      the current ``(N, B, dtype, max_neighbors, has_pbc, has_cell)``
      tuple. The hook builds the key from its already-tracked allocator
      gates; cache invalidation rides along free with the existing
      shape-change detection.
    * ``replay()`` schedules the captured graph on the cache's private
      stream and joins back to the caller's current stream.
    * ``capture(fn, key)`` records ``fn()`` into a fresh
      ``torch.cuda.CUDAGraph`` under ``wp.ScopedStream`` so Warp launches
      attach to the captured stream (the default stream switch
      otherwise creates an event dependency the capture rejects).
      Caller is responsible for warming up ``fn`` once eagerly before
      calling capture so the first-build adaptive-K pass settles
      ``max_neighbors`` before the captured graph fixes it in place.

    Lifetime: a single cache instance lives for the lifetime of the
    hook. ``invalidate()`` drops the captured graph (and lets it be
    re-captured on the next call) — currently unused but useful for
    callers that want to force re-capture without changing shape.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=device)
        # Lazy-imported to avoid pulling warp at module import time when
        # the user hasn't opted into capture.
        import warp as _wp

        self._wp = _wp
        self._wp_stream = _wp.Stream(cuda_stream=self._stream.cuda_stream)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._key: tuple | None = None
        # Latches True if a capture attempt fails. Once set, the cache
        # permanently declines to (re-)capture and the hook stays on the
        # eager path: a failed ``torch.cuda.graph`` leaves the capture
        # stream poisoned, so retrying every step would re-poison the
        # context and cascade ``cudaErrorStreamCaptureInvalidated`` into
        # unrelated work. The eager dispatch is correct on its own; the
        # graph is only a replay optimisation.
        self._capture_disabled: bool = False

    @staticmethod
    def make_key(
        N: int,
        B: int,
        dtype: torch.dtype,
        max_neighbors: int,
        has_pbc: bool,
        has_cell: bool,
    ) -> tuple:
        """Build the cache key from the same shape signals the hook uses
        to gate buffer reallocation.
        """
        return (N, B, str(dtype), max_neighbors, has_pbc, has_cell)

    def matches(self, key: tuple) -> bool:
        """Return True if the captured graph is valid for this key."""
        return self._graph is not None and self._key == key

    def replay(self) -> None:
        """Schedule the captured graph on the cache's stream and join
        back to the caller's current stream."""
        current = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            self._graph.replay()  # type: ignore[union-attr]
        current.wait_stream(self._stream)

    def capture(self, fn, key: tuple) -> bool:
        """Capture ``fn()`` into a fresh CUDA graph for this key.

        The caller must have already invoked ``fn`` once eagerly outside
        the captured region so any warmup-only work (kernel JIT,
        first-build adaptive-K resize) finishes before capture freezes
        the launch dim.

        Returns ``True`` on a successful capture and ``False`` if capture
        failed (in which case the cache permanently disables capture and
        the caller stays on the eager path). A failed
        ``torch.cuda.graph`` can leave the capture stream in an aborted
        stream-capture state that poisons every later CUDA call with
        ``cudaErrorStreamCaptureInvalidated`` / "Offset increment outside
        graph capture"; on failure we abort the partial graph and hard-
        synchronise the device so the context is clean for subsequent
        (unrelated) work.
        """
        if self._capture_disabled:
            return False
        graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(current)
        try:
            with self._wp.ScopedStream(self._wp_stream):
                with torch.cuda.stream(self._stream):
                    # Warmup-in-stream so the warp kernel modules are
                    # loaded for the captured stream's CUDA context. The
                    # first such load goes through ``cuModuleLoadData``,
                    # which isn't capture-safe — running once outside
                    # ``torch.cuda.graph`` ensures the cache hits inside.
                    fn()
                    torch.cuda.synchronize(self._device)
                    with torch.cuda.graph(graph, stream=self._stream):
                        fn()
            current.wait_stream(self._stream)
            torch.cuda.synchronize(self._device)
        except Exception:  # noqa: BLE001 — any capture failure → eager fallback
            # Permanently disable capture for this cache and drop any
            # partially-built graph, then resynchronise so the aborted
            # stream-capture state cannot leak into later CUDA work.
            self._capture_disabled = True
            self._graph = None
            self._key = None
            del graph
            self._resync_after_failed_capture()
            return False
        self._graph = graph
        self._key = key
        return True

    def _resync_after_failed_capture(self) -> None:
        """Drain both the capture stream and the device after an aborted
        capture so the poisoned stream-capture state is cleared before any
        further (unrelated) CUDA work runs.

        Best-effort: each step is guarded so a secondary error while
        recovering can't mask the original fallback.
        """
        try:
            self._stream.synchronize()
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            torch.cuda.synchronize(self._device)
        except Exception:  # noqa: BLE001, S110
            pass

    def invalidate(self) -> None:
        """Drop the captured graph; next call will re-capture."""
        self._graph = None
        self._key = None


class NeighborListHook:
    """Compute and cache neighbor lists before each model evaluation.

    This hook runs at :attr:`~DynamicsStage.BEFORE_COMPUTE` and writes
    neighbor data into the batch so that the model's ``adapt_input`` can
    read it.  An optional Verlet skin buffer avoids rebuilding the list
    every step: the list is only recomputed when the maximum atomic
    displacement since the last build exceeds ``config.skin / 2``, or when
    the set of active systems changes (detected via ``system_id``).

    For ``MATRIX`` format the following tensors are written to the atoms
    group of the batch (and thus accessible as ``batch.neighbor_matrix``
    etc.):

    * ``neighbor_matrix`` — shape ``(N, max_neighbors)``, int32
    * ``num_neighbors``   — shape ``(N,)``, int32
    * ``neighbor_matrix_shifts`` — shape ``(N, max_neighbors, 3)``, int32
      (only written when PBC is active)

    For ``COO`` format the edges group of the batch is created or replaced
    on every rebuild, making the following accessible:

    * ``batch.neighbor_list`` — shape ``(E, 2)``, int32 (nvalchemi convention)
    * ``batch.neighbor_list_shifts`` — shape ``(E, 3)``, int32 (only when PBC active)
    * ``batch.edge_ptr`` — shape ``(N+1,)``, int32, derived on demand via
      the :attr:`~nvalchemi.data.Batch.edge_ptr` property

    Parameters
    ----------
    config : NeighborConfig
        Neighbor list configuration read from the model config.
    skin : float, optional
        Verlet skin distance in the same length units as positions.
        The neighbor list is searched out to ``cutoff + skin`` so that
        atoms crossing the skin boundary but not the bare cutoff are
        already included.  The list is only rebuilt when any atom has
        moved more than ``skin / 2`` since the previous build (requires
        ``nvalchemiops >= 0.4``); set to ``0.0`` (default) to rebuild
        every step.
    max_neighbors : int | None, optional
        Maximum number of neighbors per atom for MATRIX format.  When
        ``None`` (default), auto-estimated from the cutoff via
        ``estimate_max_neighbors(cutoff)``.  Ignored for COO format.
    stage : Enum | None, optional
        The workflow stage at which this hook runs.  Defaults to
        ``DynamicsStage.BEFORE_COMPUTE``.
    method : str | None, optional
        Explicit ``nvalchemiops`` neighbor-list method to use.  When ``None``
        (default), the hook selects an appropriate method from the batch shape
        and periodic-cell metadata.
    """

    def __init__(
        self,
        config: NeighborConfig,
        skin: float = 0.0,
        max_neighbors: int | None = None,
        stage: Enum | None = None,
        method: str | None = None,
    ) -> None:
        self.config = config
        self.skin = skin
        self.stage = stage
        self.method = method
        self._max_neighbors_override = max_neighbors
        self.frequency = 1
        self._neighbor_list_flag = config.format == NeighborListFormat.COO

        # CUDA graph capture is opt-in via ``use_cuda_graph``. When True
        # the hook captures the ``neighbor_list`` call once per shape
        # (after the first-build adaptive-K pass settles ``max_neighbors``)
        # and replays the graph on subsequent calls. Caller asserts CUDA
        # tensors throughout — there is no implicit CPU fallback inside
        # the captured region. Currently only the cell-list dispatch path
        # captures cleanly; naive-PBC capture requires the
        # nvalchemiops.torch wrappers to thread the wrap-scratch kwargs
        # through (see ``allocate_naive_pbc_wrap_scratch`` upstream).
        self._use_cuda_graph = use_cuda_graph
        self._graph_cache: _NLGraphCache | None = None

        # Skin-buffer state: populated after the first build.
        self._ref_positions: torch.Tensor | None = None
        self._rebuild_flags: torch.Tensor | None = None

        # Neighbor Matrix state: populated after the first build.
        self._neighbor_matrix: torch.Tensor | None = None
        self._col_range: torch.Tensor | None = None
        self._num_neighbors: torch.Tensor | None = None
        self._neighbor_matrix_shifts: torch.Tensor | None = None

        # Shape the staging buffers were allocated for; used to detect when
        # re-allocation is needed (e.g. inflight batching with variable load).
        self._alloc_N: int | None = None
        self._alloc_B: int | None = None

        # Staging buffers — persistent GPU tensors refreshed each step via
        # copy_() to avoid per-step dynamic allocation inside the dispatcher.
        self._buf_positions: torch.Tensor | None = None
        self._buf_batch_idx: torch.Tensor | None = None
        self._buf_batch_ptr: torch.Tensor | None = None
        self._buf_cell: torch.Tensor | None = None  # PBC only
        self._buf_pbc: torch.Tensor | None = None  # PBC only

        # Algorithm-specific pre-allocated kwargs forwarded to neighbor_list.
        self._buf_nl_kwargs: dict[str, torch.Tensor | int] = {}
        self._neighbor_list_method: str | None = None

        # Set by ``_alloc_nl_kwargs`` based on the selected base algorithm;
        # gates ``_NLGraphCache.capture`` since only the cell-list path is
        # currently graph-capturable (the naive dispatcher chain doesn't yet
        # thread wrap-scratch kwargs through to the warp launcher).
        self._using_cell_list_path: bool = False

        # Adaptive K-dimension state.
        self._actual_max_k: torch.Tensor | None = None  # GPU scalar from last build
        self._first_build: bool = True  # Force sync check after first kernel call

    # ------------------------------------------------------------------
    # Main hook entry point
    # ------------------------------------------------------------------
    # Under DD the framework owns compilation, so drop torch.compile here.
    # ``dynamic=True`` still produces an 8 s upfront compile and an
    # occasional ~1.8 s recompile when halo size changes; eager is fast
    # enough, so we avoid both.
    @torch.compiler.disable
    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Recompute the neighbor list if needed and write it to the batch.

        When ``skin > 0`` and ``nvalchemiops`` provides
        :func:`~nvalchemiops.torch.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild`,
        the list is only rebuilt when at least one atom has moved more than
        ``skin / 2`` since the previous build.  The reference positions are
        updated in-place on the GPU (no clone) whenever a rebuild occurs.
        """
        self._rebuild(ctx.batch)

    @torch.compiler.disable
    def _init_ref_positions(self, positions: torch.Tensor) -> None:
        """One-time clone of positions into the skin-buffer reference.

        Marked ``@torch.compiler.disable`` because the attribute assignment
        is a Python mutation that creates a graph break.  Called once per
        ``(N, B)`` shape from ``_rebuild`` *before* any CUDA-graph capture
        so the captured graph sees a non-None ``_ref_positions`` and
        records the skin-check kernel into the replay region.
        """
        self._ref_positions = positions.detach().clone()

    def _captured_region(self) -> None:
        """Per-step GPU work that runs on stable-identity staging buffers.

        Three operations make up the captured region:

        * **Skin check** (``_batch_nl_rebuild_inplace``) — writes
          ``self._rebuild_flags`` in place. Skipped when ``skin == 0`` or
          before ``_ref_positions`` is initialised, in which case the
          captured graph records no skin-check kernel and downstream
          ``neighbor_list`` runs with the all-True flags from
          ``_alloc_staging_buffers``.
        * **Neighbor list dispatch** — reads from the staging buffers
          and writes the persistent output tensors.
        * **Mark stale** — uses ``masked_fill_`` instead of
          boolean-mask assignment because the latter goes through
          ``nonzero()``, a dynamic-shape op that would invalidate
          capture.

        Runs eagerly when ``use_cuda_graph=False``. When
        ``use_cuda_graph=True`` the same callable is captured into a
        ``torch.cuda.CUDAGraph`` and replayed on subsequent matching
        calls. ``cell_inv`` is held in ``self._cell_inv_cache`` (refreshed
        eagerly outside this region) so the captured graph reads a
        stable tensor identity; for NPT runs the cache must be updated
        in place (currently NVT/NVE-only).
        """
        if self.skin > 0.0 and self._ref_positions is not None:
            cell_inv = self._cell_inv_cache if self._buf_cell is not None else None
            if _batch_nl_rebuild_inplace is not None:
                _batch_nl_rebuild_inplace(
                    reference_positions=self._ref_positions,
                    current_positions=self._buf_positions,
                    batch_idx=self._buf_batch_idx,
                    rebuild_flags=self._rebuild_flags,
                    skin_distance_threshold=self.skin / 2,
                    update_reference_positions=True,
                    cell=self._buf_cell,
                    cell_inv=cell_inv,
                    pbc=self._buf_pbc,
                )
            elif _batch_nl_needs_rebuild is not None:
                # Out-of-place fallback (older nvalchemiops): allocates a
                # fresh tensor each call, which is not capture-safe. Only
                # safe to hit when ``use_cuda_graph=False``.
                self._rebuild_flags = _batch_nl_needs_rebuild(
                    reference_positions=self._ref_positions,
                    current_positions=self._buf_positions,
                    batch_idx=self._buf_batch_idx,
                    skin_distance_threshold=self.skin / 2,
                    update_reference_positions=True,
                    cell=self._buf_cell,
                    cell_inv=cell_inv,
                    pbc=self._buf_pbc,
                )

        self._invoke_neighbor_list(rebuild_flags=self._rebuild_flags)

        stale = self._col_range.unsqueeze(0) >= self._num_neighbors.unsqueeze(1)
        self._neighbor_matrix.masked_fill_(stale, self._alloc_N)
        if self._neighbor_matrix_shifts is not None:
            self._neighbor_matrix_shifts.masked_fill_(stale.unsqueeze(-1), 0)

    def _invoke_neighbor_list(self, rebuild_flags: torch.Tensor | None) -> None:
        """Single-call dispatch into ``nvalchemiops.torch.neighbors.neighbor_list``.

        Factored out so the same call site is used both eagerly (cache
        miss) and inside the captured graph; ``rebuild_flags`` is the
        only argument that varies between the two — the first-build
        adaptive-K path passes ``None`` to force a full rebuild even
        when the skin check would have skipped systems.
        """
        neighbor_list(
            positions=self._buf_positions,
            cutoff=self.config.cutoff + self.skin,
            cell=self._buf_cell,
            pbc=self._buf_pbc,
            max_neighbors=self._max_neighbors,
            half_fill=self.config.half_list,
            batch_ptr=self._buf_batch_ptr,
            batch_idx=self._buf_batch_idx,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            rebuild_flags=rebuild_flags,
            method=self._neighbor_list_method,
            **self._buf_nl_kwargs,
        )

    def _make_graph_cache_key(
        self,
        N: int,
        B: int,
        dtype: torch.dtype,
        pbc: torch.Tensor | None,
        cell: torch.Tensor | None,
    ) -> tuple:
        """Build the cache key from the same shape signals the alloc
        gate already tracks. Lazily creates the cache on first call so
        we only pay the stream-allocation cost once we know the device.

        Returns the key (the hook then asks the cache whether it
        matches). When ``use_cuda_graph`` is False, returns a sentinel
        the cache will never match — the eager path takes over.
        """
        if not self._use_cuda_graph:
            return ()
        if self._graph_cache is None:
            if self._buf_positions is None:
                raise RuntimeError(
                    "graph cache key requested before staging buffers were "
                    "allocated — _alloc_staging_buffers must run first"
                )
            self._graph_cache = _NLGraphCache(self._buf_positions.device)
        has_pbc = pbc is not None
        has_cell = cell is not None
        return _NLGraphCache.make_key(
            N=N,
            B=B,
            dtype=dtype,
            max_neighbors=self._max_neighbors,
            has_pbc=has_pbc,
            has_cell=has_cell,
        )

    # ------------------------------------------------------------------
    # Neighbor list construction
    # ------------------------------------------------------------------

    def _rebuild(self, batch: Batch) -> None:
        """Build the neighbor list and write results into the batch."""
        positions = batch.positions  # (N, 3)
        batch_ptr = batch.batch_ptr  # (B+1,)
        N = batch.num_nodes
        B = batch.num_graphs

        # Detect PBC.  getattr avoids a try/except which is a graph break.
        pbc = getattr(batch, "pbc", None)  # (B, 3) bool or None
        cell = getattr(batch, "cell", None)  # (B, 3, 3) float or None

        # Drop cell + pbc when the system is non-periodic — mirrors
        # ``nvalchemi.neighbors.compute_neighbors``. The underlying
        # naive kernel defaults to ``wrap_positions=True`` and will fold
        # slightly-negative coordinates through the cell, silently
        # dropping neighbors. Matters for distributed halo batches that
        # carry cell + ``pbc=[False]*3`` so downstream code knows the
        # lattice shape but shouldn't wrap positions.
        #
        # PBC is fixed for the lifetime of an NVT/NVE run — distributed
        # halo_exchange creates a fresh padded_batch each step so a
        # ``id()``-keyed cache always misses. Resolve once on the first
        # call and hold the decision; a wrapper that *legitimately*
        # changes PBC mid-run (rare; would require an NPT-style box
        # reshape on a non-PBC system) can ``del`` ``_pbc_drop_decided``
        # to force re-evaluation.
        if pbc is not None:
            if not getattr(self, "_pbc_drop_decided", False):
                self._pbc_drop_decided = True
                self._pbc_drop_value = not bool(pbc.any())
            if self._pbc_drop_value:
                pbc = None
                cell = None

        # ------------------------------------------------------------------
        # Allocate (or reallocate) the output tensors when shape changes.
        # Reallocation also resets the skin-buffer state so that the first
        # subsequent step forces a full rebuild and re-initialises
        # _ref_positions for the new atom count.
        # ------------------------------------------------------------------
        if self._neighbor_matrix is None or self._neighbor_matrix.shape[0] != N:
            self._alloc_output_tensors(N, batch, pbc)

        # ------------------------------------------------------------------
        # (Re)allocate staging buffers and algorithm kwargs on shape change.
        # ------------------------------------------------------------------
        if self._alloc_N != N or self._alloc_B != B:
            # Composition changed — check K before staging realloc.
            self._check_and_resize_k(N, batch.device, pbc)
            self._alloc_staging_buffers(
                N,
                B,
                positions.dtype,
                batch.device,
                cell,
                pbc,
                batch_ptr,
                positions=positions,
                batch_idx=batch.batch_idx,
            )
            self._alloc_N = N
            self._alloc_B = B

        # Refresh staging buffers from the current batch. Stays eager — the
        # source tensor identity (``positions``, ``batch_ptr`` etc.) changes
        # each call, so the copies cannot be recorded into a CUDA graph
        # without the graph-update API.
        self._copy_to_staging_buffers(positions, batch_ptr, batch.batch_idx, cell, pbc)

        # ------------------------------------------------------------------
        # Refresh ``_cell_inv_cache`` eagerly. The skin-check kernel reads
        # this tensor identity from inside the captured region; computing
        # the inverse here keeps ``torch.linalg.inv_ex`` (allocating) out
        # of the captured graph. Cache hits after the first call for
        # NVT/NVE; NPT runs need an in-place update path (currently
        # unsupported under capture).
        # ------------------------------------------------------------------
        if self._buf_cell is not None:
            cached = getattr(self, "_cell_inv_cache", None)
            cached_id = getattr(self, "_cell_inv_cache_id", None)
            if cached is None or cached_id != id(self._buf_cell):
                cached = torch.linalg.inv_ex(self._buf_cell)[0].contiguous()
                self._cell_inv_cache = cached
                self._cell_inv_cache_id = id(self._buf_cell)

        # ------------------------------------------------------------------
        # Captured region: skin check + neighbor-list dispatch + mark-stale.
        #
        # Replay path: when ``use_cuda_graph`` is on AND the current call
        # shape matches the cached graph, replay it. The graph records
        # the three steps in order, all reading/writing stable-identity
        # buffers — staging copies happen above before replay so the
        # captured kernels see the freshly-staged inputs.
        # ------------------------------------------------------------------
        cache_key = self._make_graph_cache_key(N, B, positions.dtype, pbc, cell)
        if self._graph_cache is not None and self._graph_cache.matches(cache_key):
            self._graph_cache.replay()
        else:
            self._captured_region()

            # ------------------------------------------------------------------
            # Adaptive K: first-build check (runs once, then never again).
            # This is the only per-step adaptive K code.  After the first
            # build, all checks are gated on structural events (N/B change)
            # inside _alloc_output_tensors / _alloc_staging_buffers.
            # ------------------------------------------------------------------
            if self._first_build:
                self._first_build = False
                self._actual_max_k = self._num_neighbors.max()
                grew = self._check_and_resize_k(N, batch.device, pbc)
                if grew:
                    # K was too small — re-run kernel with larger buffers
                    # before capture so the captured graph fixes the right K.
                    # Skin check is skipped here (full rebuild via
                    # rebuild_flags=None), so we call ``_invoke_neighbor_list``
                    # directly rather than ``_captured_region``.
                    self._invoke_neighbor_list(rebuild_flags=None)
                # Initialise the skin-buffer reference *before* capture so
                # the captured graph records the skin-check kernel. If we
                # captured here with ``_ref_positions=None`` the Python
                # ``if`` inside ``_captured_region`` would erase the
                # skin-check from the graph and every subsequent replay
                # would do a full rebuild.
                if self.skin > 0.0 and self._ref_positions is None:
                    self._init_ref_positions(positions)
                # Adaptive-K may have grown ``_max_neighbors`` since we
                # built the key above. Rebuild it so the captured graph
                # is keyed on the post-resize value — otherwise the next
                # call's key (computed with the grown ``_max_neighbors``)
                # wouldn't match and we'd re-capture every call.
                cache_key = self._make_graph_cache_key(N, B, positions.dtype, pbc, cell)

            # Capture the dispatch into a CUDA graph for replay on
            # subsequent steps. Skipped when the cache is disabled, a prior
            # capture failed (``_capture_disabled`` — stay eager), the
            # dispatch chose the naive path (whose dispatcher chain
            # doesn't yet thread the wrap-scratch kwargs through), or
            # first-build adaptive-K hasn't settled. ``capture`` returns
            # False on failure (and self-heals the CUDA context); the eager
            # results already in the buffers are used regardless.
            if (
                self._graph_cache is not None
                and not self._graph_cache._capture_disabled
                and not self._first_build
                and self._using_cell_list_path
            ):
                self._graph_cache.capture(self._captured_region, cache_key)

        # ------------------------------------------------------------------
        # Post-processing: write results to batch (shared with compute_neighbors)
        # ------------------------------------------------------------------
        _write_neighbor_data_to_batch(
            batch=batch,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            format=NeighborListFormat.COO
            if self._neighbor_list_flag
            else NeighborListFormat.MATRIX,
            cutoff=self.config.cutoff,
        )

    # ------------------------------------------------------------------
    # Staging buffer management
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _alloc_output_tensors(
        self,
        N: int,
        batch: "Batch",
        pbc: torch.Tensor | None,
    ) -> None:
        """Allocate neighbor-matrix output tensors for atom count *N*.

        Marked ``@torch.compiler.disable`` because it calls
        ``estimate_max_neighbors`` (CPU work), allocates tensors with
        dynamic shapes, and mutates Python attributes — all graph breaks.
        Called only when the atom count changes.
        """
        device = batch.device
        max_nbrs = self._max_neighbors_override
        if max_nbrs is None:
            max_nbrs = estimate_max_neighbors(
                cutoff=self.config.cutoff + self.skin,
            )
        # Non-PBC hard cap: an atom can see at most (N_system - 1)
        # neighbors without periodic images.  We use max_num_nodes
        # (not max_num_nodes - 1) so that K has one sentinel slot
        # to distinguish "all used" from "overflow" in the adaptive check.
        # Round up to nearest 16 for memory-aligned kernel performance.
        if pbc is None and batch.max_num_nodes > 0:
            cap = ((batch.max_num_nodes + 15) // 16) * 16
            max_nbrs = min(max_nbrs, cap)
        self._max_neighbors = max_nbrs
        self._neighbor_matrix = torch.full(
            (N, max_nbrs), N, dtype=torch.int32, device=device
        )
        self._col_range = torch.arange(max_nbrs, device=device, dtype=torch.int32)
        self._num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        if pbc is not None:
            self._neighbor_matrix_shifts = torch.zeros(
                N, max_nbrs, 3, dtype=torch.int32, device=device
            )
        # Reset skin-buffer state so __call__ re-initialises _ref_positions.
        self._ref_positions = None
        self._rebuild_flags = None
        # Reset adaptive K state so first build triggers a sync check.
        self._first_build = True
        self._actual_max_k = None

    @torch.compiler.disable
    def _check_and_resize_k(
        self,
        N: int,
        device: torch.device,
        pbc: torch.Tensor | None,
    ) -> bool:
        """Sync on actual max K and grow/shrink the neighbor matrix if needed.

        Called on structural events (first build, N/B change, cell volume
        change).  The sync cost is acceptable because these events are
        infrequent and the calling code path is already off the compile graph.

        Returns ``True`` if K was grown (caller must re-run the kernel).
        Shrinking trims the existing buffers in-place — no re-run needed.
        """
        if self._actual_max_k is None:
            return False
        actual = int(self._actual_max_k.item())

        if actual >= self._max_neighbors:
            # Overflow — grow with 1.5x headroom and round to nearest 16.  Must re-run kernel.
            self._max_neighbors = ((int(actual * 1.5) + 15) // 16) * 16
            self._realloc_k(N, device, pbc)
            return True
        elif actual < (1 / 2) * self._max_neighbors and actual > 0:
            # 2x+ overestimate — trim existing buffers in-place.
            new_k = ((int(actual * 2) + 15) // 16) * 16
            # Never shrink below the user-provided override — it serves as a
            # hard floor.  We may grow above it on overflow, but not below.
            if self._max_neighbors_override is not None:
                new_k = max(new_k, self._max_neighbors_override)
            if new_k < self._max_neighbors:
                self._max_neighbors = new_k
                self._neighbor_matrix = self._neighbor_matrix[:, :new_k].contiguous()
                self._col_range = self._col_range[:new_k]
                if self._neighbor_matrix_shifts is not None:
                    self._neighbor_matrix_shifts = self._neighbor_matrix_shifts[
                        :, :new_k
                    ].contiguous()
            # num_neighbors unchanged — still valid.
        return False

    @torch.compiler.disable
    def _realloc_k(
        self,
        N: int,
        device: torch.device,
        pbc: torch.Tensor | None,
    ) -> None:
        """Reallocate neighbor-matrix buffers at the current N with a new K.

        Preserves N (no staging-buffer realloc needed) but resets the
        skin state to force a full rebuild on the next step.
        """
        max_nbrs = self._max_neighbors
        self._neighbor_matrix = torch.full(
            (N, max_nbrs), N, dtype=torch.int32, device=device
        )
        self._col_range = torch.arange(max_nbrs, device=device, dtype=torch.int32)
        self._num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        if pbc is not None:
            self._neighbor_matrix_shifts = torch.zeros(
                N, max_nbrs, 3, dtype=torch.int32, device=device
            )
        else:
            self._neighbor_matrix_shifts = None
        # Reset skin state to force a full rebuild.
        self._ref_positions = None
        self._rebuild_flags = None

    @torch.compiler.disable
    def _alloc_staging_buffers(
        self,
        N: int,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        batch_ptr: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> None:
        """Allocate persistent staging buffers for the current (N, B) shape."""
        self._buf_positions = torch.zeros(N, 3, dtype=dtype, device=device)
        self._buf_batch_idx = torch.zeros(N, dtype=torch.int32, device=device)
        self._buf_batch_ptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
        if cell is not None:
            self._buf_cell = torch.zeros(B, 3, 3, dtype=dtype, device=device)
            self._buf_pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)
        else:
            self._buf_cell = None
            self._buf_pbc = None
        # Pre-allocate rebuild_flags as all-True so that the very first step
        # (before _ref_positions is set and the skin check runs) forces a full
        # neighbor-list build for every system.  The in-place op zeroes this
        # buffer at the start of each subsequent call before writing fresh values.
        self._rebuild_flags = torch.ones(B, dtype=torch.bool, device=device)
        # Pre-allocate algorithm-specific kwargs to eliminate on-demand CPU syncs
        # from the neighbor_list dispatcher.  Use the actual batch_ptr (if provided)
        # to compute max_atoms_per_system correctly — the staging buffer is still
        # all-zeros at this point and would give max_atoms = 0.
        ptr = batch_ptr if batch_ptr is not None else self._buf_batch_ptr
        alloc_positions = positions if positions is not None else self._buf_positions
        alloc_batch_idx = batch_idx if batch_idx is not None else self._buf_batch_idx
        self._alloc_nl_kwargs(
            N, B, alloc_positions, alloc_batch_idx, ptr, cell, pbc, device, dtype
        )

    def _copy_to_staging_buffers(
        self,
        positions: torch.Tensor,
        batch_ptr: torch.Tensor,
        batch_idx: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> None:
        """Refresh staging buffers from the current batch."""
        self._buf_positions.copy_(positions)
        self._buf_batch_ptr.copy_(batch_ptr)
        self._buf_batch_idx.copy_(batch_idx)
        if self._buf_cell is not None and cell is not None:
            self._buf_cell.copy_(cell)
        if self._buf_pbc is not None and pbc is not None:
            self._buf_pbc.copy_(pbc)

    # ------------------------------------------------------------------
    # Algorithm-specific pre-allocation
    # ------------------------------------------------------------------

    def _alloc_nl_kwargs(
        self,
        N: int,
        B: int,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Pre-allocate algorithm-specific kwargs to remove CPU-GPU syncs.

        The ops dispatcher exposes a host-only cost model that can select
        fine-grained strategies such as ``batch_cell_list_pair_centric`` and
        ``batch_cluster_tile``.  Run that selector once when staging buffers are
        allocated, cache the chosen method, and pre-allocate scratch tensors for
        the selected base algorithm.

        ``_using_cell_list_path`` is set from the selected base method so that
        :class:`_NLGraphCache` capture is only attempted on the cell-list
        dispatch path (the naive dispatcher chain doesn't yet thread the
        wrap-scratch kwargs through to the warp launcher).
        """
        self._buf_nl_kwargs = {}
        batch_ptr = batch_ptr.detach().to(dtype=torch.int32).contiguous()
        self._neighbor_list_method = self._select_neighbor_list_method(
            N, B, batch_ptr, cell, pbc, dtype
        )
        base_method = self._base_neighbor_list_method(self._neighbor_list_method)
        self._using_cell_list_path = base_method.endswith("cell_list")

        if base_method.endswith("cluster_tile"):
            if (
                allocate_batch_cluster_tile_list is None
                or estimate_batch_max_tiles_per_group is None
                or cell is None
            ):
                return
            alloc_cell = cell.to(dtype).contiguous()
            max_tiles_per_group = estimate_batch_max_tiles_per_group(
                batch_ptr, self.config.cutoff + self.skin, alloc_cell
            )
            (
                sorted_atom_index,
                sort_inv,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                batch_idx_sorted,
                batch_ptr_padded,
                group_system,
                group_ptr,
                group_ctr_x,
                group_ctr_y,
                group_ctr_z,
                group_ext_x,
                group_ext_y,
                group_ext_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
            ) = allocate_batch_cluster_tile_list(
                batch_ptr,
                device,
                dtype=dtype,
                max_tiles_per_group=max_tiles_per_group,
            )
            self._buf_nl_kwargs = {
                "max_tiles_per_group": max_tiles_per_group,
                "sorted_atom_index": sorted_atom_index,
                "sort_inv": sort_inv,
                "sorted_pos_x": sorted_pos_x,
                "sorted_pos_y": sorted_pos_y,
                "sorted_pos_z": sorted_pos_z,
                "batch_idx_sorted": batch_idx_sorted,
                "batch_ptr_padded": batch_ptr_padded,
                "group_system": group_system,
                "group_ptr": group_ptr,
                "group_ctr_x": group_ctr_x,
                "group_ctr_y": group_ctr_y,
                "group_ctr_z": group_ctr_z,
                "group_ext_x": group_ext_x,
                "group_ext_y": group_ext_y,
                "group_ext_z": group_ext_z,
                "num_tiles": num_tiles,
                "tile_row_group": tile_row_group,
                "tile_col_group": tile_col_group,
                "tile_system": tile_system,
            }
            return

        if base_method.endswith("cell_list"):
            if estimate_batch_cell_list_sizes is None or allocate_cell_list is None:
                return  # nvalchemiops too old; fall back to dynamic allocation
            if cell is not None and pbc is not None:
                # PBC: use the actual cell geometry.
                alloc_cell = cell.to(dtype).contiguous()
                alloc_pbc = pbc
            else:
                # Non-PBC: synthesise a bounding-box cell from current positions
                # with a 1.5x pad so that position drift during the simulation
                # doesn't overflow the pre-allocated cell-list arrays.
                expanded_idx = batch_idx.unsqueeze(1).expand_as(positions)
                pos_min = torch.full((B, 3), float("inf"), dtype=dtype, device=device)
                pos_min.scatter_reduce_(0, expanded_idx, positions, reduce="amin")
                pos_max = torch.full((B, 3), float("-inf"), dtype=dtype, device=device)
                pos_max.scatter_reduce_(0, expanded_idx, positions, reduce="amax")
                cell_lengths = (pos_max - pos_min) * 1.5 + 0.1 * (
                    self.config.cutoff + self.skin
                )
                alloc_cell = torch.diag_embed(cell_lengths)  # (B, 3, 3)
                alloc_pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)

            max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
                alloc_cell, alloc_pbc, self.config.cutoff + self.skin
            )
            (
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            ) = allocate_cell_list(
                N, int(max_total_cells), neighbor_search_radius, device
            )
            self._buf_nl_kwargs = {
                "cells_per_dimension": cells_per_dimension,
                "neighbor_search_radius": neighbor_search_radius,
                "atom_periodic_shifts": atom_periodic_shifts,
                "atom_to_cell_mapping": atom_to_cell_mapping,
                "atoms_per_cell_count": atoms_per_cell_count,
                "cell_atom_start_indices": cell_atom_start_indices,
                "cell_atom_list": cell_atom_list,
            }
            return

        # Naive algorithm.
        if cell is not None and pbc is not None:
            # PBC naive: pre-compute shift-range tensors so the dispatcher
            # does not call compute_naive_num_shifts (which has .item()) on
            # the hot path.
            if compute_naive_num_shifts is None:
                return
            shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
                cell.to(dtype).contiguous(),
                self.config.cutoff + self.skin,
                pbc,
            )
            max_atoms = int((batch_ptr[1:] - batch_ptr[:-1]).max().item())
            self._buf_nl_kwargs = {
                "shift_range_per_dimension": shift_range,
                "num_shifts_per_system": num_shifts,
                "max_shifts_per_system": max_shifts,
                "max_atoms_per_system": max_atoms,
            }
        # No-PBC naive: no extra kwargs required — the kernel has no
        # CPU-sync allocations in this branch.

    def _select_neighbor_list_method(
        self,
        N: int,
        B: int,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> str:
        """Choose the explicit method to use inside the compiled hot path."""
        if self.method is not None:
            return self.method
        fallback = "cell_list" if N // max(B, 1) >= 2000 else "naive"
        if cell is None or pbc is None:
            return fallback
        try:
            return suggest_neighbor_list_method(
                batch_ptr,
                cell.to(dtype).contiguous(),
                pbc,
                self.config.cutoff + self.skin,
                half_fill=self.config.half_list,
                return_neighbor_list=False,
                positions_dtype=dtype,
            )
        except (RuntimeError, NotImplementedError, ValueError):
            return fallback

    @staticmethod
    def _base_neighbor_list_method(method: str | None) -> str:
        """Return the dispatcher base method for a fine-grained strategy name."""
        if method is None:
            return "naive"
        try:
            return neighbor_list_strategy_run_args(method)[0]
        except ValueError:
            return method
