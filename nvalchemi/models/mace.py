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

"""MACE model wrapper.

Wraps any MACE model (``MACE``, ``ScaleShiftMACE``, etc.) as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible wrapper, ready for
use in any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine or standalone
inference / fine-tuning.

Usage
-----
Load a named foundation-model checkpoint::

    from nvalchemi.models.mace import MACEWrapper
    import torch

    model = MACEWrapper.from_checkpoint("medium-0b2", device=torch.device("cuda"))

Or wrap an already-instantiated model::

    mace_model = torch.load("my_mace.pt", weights_only=False)
    model = MACEWrapper(mace_model)

For dynamics, register :class:`~nvalchemi.hooks.NeighborListHook`
with ``format=NeighborListFormat.COO`` so that ``neighbor_list`` and
``neighbor_list_shifts`` are populated before each model call::

    from nvalchemi.hooks import NeighborListHook
    from nvalchemi.dynamics.base import DynamicsStage

    nl_hook = NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **conservatively** via MACE's internal autograd, so
  ``"forces"`` is in ``autograd_outputs``.
* ``node_attrs`` (one-hot atomic-number encodings) are computed via a
  pre-built GPU lookup table — no CPU round-trips per step.
* For PBC systems, both ``neighbor_list_shifts`` (integer image indices ``[E, 3]``)
  and pre-computed ``shifts`` (physical Å vectors ``[E, 3]``) are passed to
  MACE.  ``shifts`` is always required by ``prepare_graph``; ``neighbor_list_shifts``
  is additionally used when ``compute_displacement=True`` (stress path).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from nvalchemi.training._spec import BaseSpec

_torch_version = version("torch")

__all__ = ["MACEWrapper"]


def _load_atomic_energies(
    atomic_energies: Mapping[int | str, float] | None,
    atomic_energies_path: Path | str | None,
) -> dict[int, float] | None:
    """Return normalized atomic energy overrides from inline values or JSON."""
    if atomic_energies is not None and atomic_energies_path is not None:
        raise ValueError("Specify only one of atomic_energies or atomic_energies_path.")
    if atomic_energies_path is not None:
        raw = json.loads(Path(atomic_energies_path).read_text())
    else:
        raw = atomic_energies
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("atomic_energies must be a mapping of atomic number to E0.")
    normalized: dict[int, float] = {}
    for atomic_number, energy in raw.items():
        try:
            z = int(atomic_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"atomic energy key {atomic_number!r} is not an atomic number."
            ) from exc
        normalized[z] = float(energy)
    return normalized


def _apply_atomic_energies(
    model: nn.Module, atomic_energies: Mapping[int | str, float]
) -> None:
    """Overwrite MACE atomic reference energies in-place."""
    if not hasattr(model, "atomic_energies_fn"):
        raise AttributeError("MACE model has no atomic_energies_fn module.")
    atomic_energies_fn = model.atomic_energies_fn
    if not hasattr(atomic_energies_fn, "atomic_energies"):
        raise AttributeError(
            "MACE model atomic_energies_fn has no atomic_energies tensor."
        )
    if not hasattr(model, "atomic_numbers"):
        raise AttributeError("MACE model has no atomic_numbers tensor.")

    target = atomic_energies_fn.atomic_energies
    if not isinstance(target, torch.Tensor):
        raise TypeError("MACE atomic_energies_fn.atomic_energies is not a tensor.")
    normalized = _load_atomic_energies(atomic_energies, None)
    if normalized is None:
        return
    atomic_numbers = torch.as_tensor(model.atomic_numbers).detach().cpu().tolist()
    index_by_z = {int(z): index for index, z in enumerate(atomic_numbers)}
    unknown = sorted(set(normalized) - set(index_by_z))
    if unknown:
        raise ValueError(
            "Atomic energy overrides contain elements not supported by the "
            f"MACE checkpoint: {unknown}."
        )

    updated = target.detach().clone()
    flat = updated.reshape(-1)
    if flat.numel() != len(atomic_numbers):
        raise ValueError(
            "MACE atomic_energies tensor size does not match model.atomic_numbers: "
            f"{flat.numel()} != {len(atomic_numbers)}."
        )
    for atomic_number, energy in normalized.items():
        flat[index_by_z[atomic_number]] = torch.as_tensor(
            energy, dtype=target.dtype, device=target.device
        )
    with torch.no_grad():
        target.copy_(updated)


def _patch_e3nn_irrep_len_for_compile() -> None:
    """Patch ``e3nn.o3.Irrep.__len__`` for ``torch.compile`` compatibility.

    TorchDynamo may treat ``Irrep`` as a sequence while building guards.
    Some e3nn versions override ``__len__`` to raise
    ``NotImplementedError`` even though ``Irrep`` subclasses ``tuple``.
    Restoring ``tuple.__len__`` keeps the tuple semantics without
    modifying the installed package on disk.
    """
    try:
        from e3nn.o3 import Irrep

        if Irrep.__len__ is not tuple.__len__:
            Irrep.__len__ = tuple.__len__
    except ImportError:
        pass


# cuEquivariance support — distributed wiring for cueq-converted MACE.
#
# cueq fuses the InteractionBlock's ``conv_tp`` into one opaque CUDA op that
# absorbs the gather + tensor-product + scatter, hiding the scatter the halo
# correction normally intercepts. That op is declared in ``custom_ops`` with
# ``scatter_outputs=(0,)`` so its output gets the halo exchange instead. The
# other cueq ops are node-local and get pass-through wrappers.


def _mace_uses_cueq(model: nn.Module) -> bool:
    """True iff any submodule of *model* is a cuequivariance kernel.

    Walks ``named_modules`` and checks class module path. Cheap enough to
    call per-``distribution_spec`` access — MACE's module tree is shallow.
    """
    for mod in model.modules():
        mod_path = type(mod).__module__
        if mod_path.startswith("cuequivariance"):
            return True
    return False


_MACE_CUEQ_SPEC_CACHE: Any = None


def _mace_cueq_spec() -> Any:
    """Return the MACE MLIPSpec for the cueq path.

    Extends the MPNN halo spec with ``custom_ops`` for the cueq kernels the
    forward touches: ``fused_tensor_product`` (+ its two backwards) with
    ``scatter_outputs=(0,)`` so halo rows on the message tensor get exchanged,
    and the node-local ops (``uniform_1d``, ``indexed_linear_B/C``,
    ``segmented_transpose``) as pass-throughs.

    Resolves ``torch.ops.cuequivariance`` lazily so importing this module
    doesn't require ``cuequivariance``; raises ``ImportError`` if called
    when it isn't registered.
    """
    global _MACE_CUEQ_SPEC_CACHE
    if _MACE_CUEQ_SPEC_CACHE is not None:
        return _MACE_CUEQ_SPEC_CACHE

    # Probe the op namespace — raises AttributeError if cueq isn't loaded.
    try:
        cueq_ops = torch.ops.cuequivariance
        # Pass the op packets; OpAdapter resolves ``.default`` itself.
        fused_fwd = cueq_ops.fused_tensor_product
        fused_bwd = cueq_ops.fused_tensor_product_bwd
        fused_bwd_bwd = cueq_ops.fused_tensor_product_bwd_bwd
        uniform_1d = cueq_ops.uniform_1d
        indexed_linear_B = cueq_ops.indexed_linear_B
        indexed_linear_C = cueq_ops.indexed_linear_C
        segmented_transpose = cueq_ops.segmented_transpose
    except AttributeError as e:
        raise ImportError(
            "torch.ops.cuequivariance.* not registered. Install "
            "cuequivariance (pip install 'nvalchemi-toolkit[mace]') and "
            "import cuequivariance_torch before loading a cueq MACE model."
        ) from e

    from nvalchemi.distributed.spec import (
        SPEC_MPNN_HALO,
        OpAdapter,
    )

    _custom_ops = (
        # Conv-fusion: output[0] is the message tensor whose halo rows carry
        # partial sums; the exchange brings them to owner values.
        OpAdapter(fused_fwd, scatter_outputs=[0]),
        OpAdapter(fused_bwd, scatter_outputs=[0]),
        OpAdapter(fused_bwd_bwd, scatter_outputs=[0]),
        # Node-local kernels — pass-through.
        OpAdapter(uniform_1d),
        OpAdapter(indexed_linear_B),
        OpAdapter(indexed_linear_C),
        OpAdapter(segmented_transpose),
    )
    # cueq fuses the tensor products / linear / symmetric contractions but does
    # NOT replace e3nn's ``SphericalHarmonics``, whose ``forward`` calls a
    # scripted kernel and an in-place ``sh.mul_(cat)`` normalization that both
    # mishandle sharded tensors (the scripted kernel faults; the in-place op
    # fails under compile). Marshal the whole ``SphericalHarmonics.forward`` so
    # both run on a plain local tensor.
    from nvalchemi.distributed._core.adapter import MethodAdapter

    _marshal = (
        MethodAdapter(
            "e3nn.o3",
            "SphericalHarmonics",
            "forward",
            mode="marshal",
        ),
    )
    from nvalchemi.distributed.spec import CompilePolicy, ForceStrategy

    # cueq fused kernels + the SphericalHarmonics marshal on the MPNN-halo
    # preset, plus the MACE compile policy: forces come from autograd over a
    # compiled energy-only forward.
    _MACE_CUEQ_SPEC_CACHE = SPEC_MPNN_HALO.with_adapters(
        *_custom_ops, *_marshal
    ).with_compile(CompilePolicy(
            static_shapes=True,
            force_strategy=ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY,
        ))
    return _MACE_CUEQ_SPEC_CACHE


_MACE_SCRIPTED_SPEC_CACHE: Any = None


def _mace_scripted_spec() -> Any:
    """Return the MACE non-cueq MLIPSpec with scripted-op marshalling wired.

    Plain (non-cueq) MACE runs e3nn's ``SphericalHarmonics`` layer, whose
    ``forward`` calls a scripted kernel and an in-place ``sh.mul_(cat)``
    normalization. On the distributed halo path the scripted kernel mishandles
    sharded tensors and faults. Auto-discovery only wraps scripted *submodules*,
    not a scripted *function* called from a plain ``forward``, so MACE marshals
    the whole ``SphericalHarmonics.forward`` explicitly. Cached.
    """
    global _MACE_SCRIPTED_SPEC_CACHE
    if _MACE_SCRIPTED_SPEC_CACHE is not None:
        return _MACE_SCRIPTED_SPEC_CACHE

    from nvalchemi.distributed._core.adapter import MethodAdapter
    from nvalchemi.distributed.spec import SPEC_MPNN_HALO

    # Marshal the whole SphericalHarmonics.forward: unwrap the sharded input to
    # its local tensor once so both the scripted kernel and the in-place
    # ``sh.mul_(cat)`` run on a plain local tensor, then re-wrap the output.
    _marshal = (
        MethodAdapter(
            "e3nn.o3",
            "SphericalHarmonics",
            "forward",
            mode="marshal",
        ),
    )
    from nvalchemi.distributed.spec import (  # noqa: PLC0415
        CompilePolicy,
        ForceStrategy,
    )

    # The SphericalHarmonics marshal on the MPNN-halo preset, plus the MACE
    # compile policy (forces via autograd over a compiled energy-only forward).
    _MACE_SCRIPTED_SPEC_CACHE = SPEC_MPNN_HALO.with_adapters(*_marshal).with_compile(
        CompilePolicy(
            static_shapes=True,
            force_strategy=ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY,
        )
    )
    return _MACE_SCRIPTED_SPEC_CACHE


@OptionalDependency.MACE.require
class MACEWrapper(nn.Module, BaseModelMixin):
    """Wrapper for any MACE model implementing the :class:`~nvalchemi.models.base.BaseModelMixin` interface.

    Accepts any MACE model variant (``MACE``, ``ScaleShiftMACE``, cuEq-converted
    models, ``torch.compile``-d models, etc.).  The wrapper handles:

    * One-hot ``node_attrs`` encoding via a pre-built GPU lookup table
      (no CPU round-trip per step).
    * Gradient enabling on ``positions`` for conservative force / stress
      computation.
    * PBC via both ``neighbor_list_shifts`` (integer image indices) and pre-computed
      ``shifts`` (physical Å vectors from ``neighbor_list_shifts @ cell``) passed to
      MACE.  ``shifts`` is always required; ``neighbor_list_shifts`` is additionally
      consumed when ``compute_displacement=True`` (stress path).

    Parameters
    ----------
    model : nn.Module
        An instantiated MACE model.  Any subclass of ``mace.modules.MACE``
        is accepted.

    Attributes
    ----------
    model : nn.Module
        The underlying MACE model.
    model_config : ModelConfig
        Mutable configuration controlling which outputs are computed.
    """

    model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        *,
        reconstruction_spec: "BaseSpec | None" = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._checkpoint_spec = reconstruction_spec

        # e3nn's ``Irrep.__len__`` raises under TorchDynamo guard-building, so
        # any compiled MACE needs this idempotent compat shim before the first
        # traced forward.
        _patch_e3nn_irrep_len_for_compile()

        # Cache the model dtype — determined at construction, stable thereafter.
        self._cached_model_dtype: torch.dtype = next(model.parameters()).dtype

        # Pre-build a one-hot lookup table: shape [max_z + 1, num_elements].
        # At runtime, node_attrs = _node_emb.index_select(0, atomic_numbers)
        # — a single GPU op, no CPU round-trips.
        z_table: list[int] = model.atomic_numbers.tolist()
        node_emb = torch.zeros(max(z_table) + 1, len(z_table))
        for i, z in enumerate(z_table):
            node_emb[z, i] = 1.0
        # Place on the model's device+dtype so _node_attrs needs no per-step
        # conversion. Use the model's device (from_checkpoint moves the inner
        # model before calling cls(model), so no later .to() is guaranteed).
        model_device = next(model.parameters()).device
        node_emb = node_emb.to(device=model_device, dtype=self._cached_model_dtype)
        # persistent=False: derived from model.atomic_numbers, excluded from
        # state_dict but still tracked for device / dtype moves.
        self.register_buffer("_node_emb", node_emb, persistent=False)
        self.model_config = ModelConfig(
            # ``atomic_energies`` (per-atom energy = MACE's raw ``node_energy``)
            # is a normal output; the distributed force path requests it to get
            # a per-node energy to differentiate, and callers may ask for it too.
            outputs=frozenset(
                {"energy", "forces", "stress", "hessian", "atomic_energies"}
            ),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"unit_shifts", "cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    def checkpoint_spec(self) -> "BaseSpec | None":
        """Return the factory spec used to reconstruct this wrapper, if known.

        Wrappers created by :meth:`from_checkpoint` store a callable spec for
        that factory so strategy checkpoints can rebuild optimized MACE models
        without introspecting the transformed inner MACE module constructor.
        Wrappers around arbitrary live modules return ``None`` and use the
        generic constructor-introspection fallback.
        """
        return self._checkpoint_spec

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        hidden_dim: int = self.model.products[0].linear.irreps_out.dim
        return {
            "node_embeddings": (hidden_dim,),
            "graph_embeddings": (hidden_dim,),
        }

    @property
    def distribution_spec(self) -> Any:
        """MLIPSpec for MACE under domain decomposition.

        MACE uses the MPNN halo spec: every message-passing layer scatters over
        edges into ``node_feats`` (halo rows kept in sync), and a final
        per-graph scatter over node energies produces total energy (halo rows
        dropped, then all-reduced across ranks). For cueq-converted checkpoints
        the fused ``conv_tp`` kernel hides that scatter, so the spec recovers
        halo correctness via ``custom_ops`` (see :func:`_mace_cueq_spec`).

        Memoized on first access. The per-checkpoint addition over the base
        spec is the message-passing halo refresh: ``neighbor_refresh_adapters``
        discovers the concrete InteractionBlocks and declares their per-node
        output halo-corrected. ``NVALCHEMI_MACE_NO_REFRESH=1`` drops it
        (debug only).
        """
        cached = getattr(self, "_dist_spec_cache", None)
        if cached is None:
            import os  # noqa: PLC0415

            from nvalchemi.distributed import neighbor_refresh_adapters  # noqa: PLC0415

            base = (
                _mace_cueq_spec()
                if _mace_uses_cueq(self.model)
                else _mace_scripted_spec()
            )
            refresh = (
                ()
                if os.environ.get("NVALCHEMI_MACE_NO_REFRESH") == "1"
                else neighbor_refresh_adapters(self.model.interactions)
            )
            cached = base.with_adapters(*refresh)
            self._dist_spec_cache = cached
        return cached

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Interaction cutoff in Angstroms, read from ``model.r_max``."""
        r_max = self.model.r_max
        return r_max.item() if isinstance(r_max, torch.Tensor) else float(r_max)

    @property
    def _model_dtype(self) -> torch.dtype:
        """Return the current dtype of the model's parameters (live, not cached).

        Reading from parameters() directly ensures this stays correct after
        `.half()` or `.to(dtype=...)` calls post-construction.

        Note: calling `.to(dtype=...)` after construction with cuEquivariance or
        `torch.compile` enabled is unsupported and may produce incorrect results.
        Use `from_checkpoint` with the desired `dtype` parameter instead.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # MACE MP models default to float64
            return torch.float64

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def _node_attrs(self, data: Batch) -> torch.Tensor:
        """One-hot encode atomic numbers via the pre-built lookup table.

        Uses a single ``index_select`` on GPU — no CPU round-trips.
        ``_node_emb`` is already on the correct device and dtype (set at
        construction and kept in sync by ``nn.Module``'s ``.to()``
        machinery), so no per-step device/dtype conversion is needed.
        """
        return self._node_emb.index_select(0, data.atomic_numbers.long())

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Build the input dict expected by ``MACE.forward``.

        Handles ``AtomicData -> Batch`` promotion, ``node_attrs`` encoding,
        gradient enabling on ``positions``, transposing ``edge_index`` from
        nvalchemi's ``[E, 2]`` to MACE's ``[2, E]`` convention, zero-filling
        of ``neighbor_list_shifts`` / ``cell`` for non-PBC systems, and
        pre-computation of physical ``shifts`` vectors from
        ``neighbor_list_shifts @ cell``.

        Expects COO neighbor data (``neighbor_list``, optionally
        ``neighbor_list_shifts``) to be present on the batch.  When used
        in a :class:`~nvalchemi.models.pipeline.PipelineModelWrapper`,
        the pipeline handles format conversion and cutoff filtering
        before calling this model.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; an ``AtomicData`` is promoted to a single-graph
            ``Batch``.
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            MACE inputs: ``positions``, ``node_attrs``, ``batch``, ``ptr``,
            ``edge_index`` ``[2, E]``, ``shifts`` ``[E, 3]``, ``cell`` ``[B, 3, 3]``.

        Notes
        -----
        Does not call ``super().adapt_input()`` (``Batch`` has no ``model_dump``);
        gradient enabling on ``positions`` is handled here.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype
        device = data.positions.device
        B = data.num_graphs

        # nvalchemi (E, 2) -> MACE COO (2, E)
        edge_index = data.neighbor_list.long().T  # [2, E]
        E = edge_index.shape[1]

        # Enable grad on positions for force/stress; clone first so we never
        # mutate a caller leaf in-place, but pass a grad-requiring tensor through
        # unchanged (it already carries its upstream graph).
        positions = data.positions.to(dtype=dtype)
        compute_forces = "forces" in self.model_config.active_outputs
        compute_stresses = "stress" in self.model_config.active_outputs
        if (compute_forces or compute_stresses) and not positions.requires_grad:
            positions = positions.clone()
            positions.requires_grad_(True)

        # neighbor_list_shifts: integer PBC image indices [E, 3], cast to float for
        # MACE's cell @ neighbor_list_shifts contraction.  Zero for non-PBC systems.
        neighbor_list_shifts_raw = getattr(data, "neighbor_list_shifts", None)
        if neighbor_list_shifts_raw is None:
            neighbor_list_shifts = torch.zeros(E, 3, dtype=dtype, device=device)
        else:
            neighbor_list_shifts = neighbor_list_shifts_raw.to(
                dtype=dtype, device=device
            )

        # cell: [B, 3, 3].  Identity matrix for non-PBC systems.
        cell_raw = getattr(data, "cell", None)
        if cell_raw is None:
            cell = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .contiguous()
            )
        else:
            cell = cell_raw.to(dtype=dtype, device=device)

        # Physical shifts [E, 3] = neighbor_list_shifts @ cell[graph]; MACE's
        # energy/force-only path reads data["shifts"] directly. Drop sentinel
        # edges (endpoint == n_atoms) first — a sentinel sender is out of bounds
        # and would fault the sender-indexed gathers below.
        # Convention: shifts[e] = neighbor_list_shifts[e] @ cell[graph_of_sender_e]
        # matching get_symmetric_displacement in mace.modules.utils.
        n_atoms = positions.shape[0]
        valid = (edge_index[0] < n_atoms) & (edge_index[1] < n_atoms)
        edge_index = edge_index[:, valid]
        neighbor_list_shifts = neighbor_list_shifts[valid]

        sender = edge_index[0]  # [E] — source node indices
        batch_per_edge = data.batch_idx[sender]
        shifts = torch.einsum("eb,ebc->ec", neighbor_list_shifts, cell[batch_per_edge])

        node_attrs = self._node_attrs(data)

        return {
            "positions": positions,
            "node_attrs": node_attrs,
            # MACE requires int64 for graph-topology tensors.
            "batch": data.batch_idx.long(),
            "ptr": data.batch_ptr.long(),
            "edge_index": edge_index,  # [2, E] — MACE convention
            "neighbor_list_shifts": neighbor_list_shifts,
            "unit_shifts": neighbor_list_shifts,  # mace-torch compat: prepare_graph reads data["unit_shifts"]
            "shifts": shifts,
            "cell": cell,
        }

    def adapt_output(
        self, raw_output: dict[str, Any], data: AtomicData | Batch
    ) -> ModelOutputs:
        """Map MACE raw outputs to nvalchemi standard keys.

        Normalizes ``energy`` shape, forwards ``forces`` / ``stress`` / ``hessian``
        when present, and exposes MACE's ``node_energy`` as ``atomic_energies``,
        then delegates to the base auto-mapper.

        Parameters
        ----------
        raw_output : dict[str, Any]
            The dict returned by ``MACE.forward``.
        data : AtomicData | Batch
            The input system the outputs were computed for.

        Returns
        -------
        ModelOutputs
            The standardized outputs (subset of ``energy``, ``forces``,
            ``stress``, ``hessian``, ``atomic_energies``).
        """
        energy = raw_output["energy"]
        mapped: dict[str, Any] = {
            "energy": energy.unsqueeze(-1) if energy.ndim == 1 else energy,
        }
        if raw_output.get("forces") is not None:
            mapped["forces"] = raw_output["forces"]
        if raw_output.get("stress") is not None:
            mapped["stress"] = raw_output["stress"]
        if raw_output.get("hessian") is not None:
            mapped["hessian"] = raw_output["hessian"]
        # Per-atom energy = MACE's raw ``node_energy``. The base auto-mapper
        # keeps it only when ``atomic_energies`` is active, so it is free
        # otherwise.
        if raw_output.get("node_energy") is not None:
            mapped["atomic_energies"] = raw_output["node_energy"]

        return super().adapt_output(mapped, data)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the MACE model for the active outputs.

        Pure and distribution-agnostic: computes exactly what
        ``model_config.active_outputs`` requests. Forces/stress run MACE's
        internal autograd; an ``atomic_energies``-only request runs an
        energy-only forward returning per-atom energy.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system (with neighbor data attached).
        **kwargs
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        ModelOutputs
            The standardized outputs for the active set.
        """
        active = self.model_config.active_outputs & self.model_config.outputs
        compute_forces = "forces" in active
        compute_stresses = "stress" in active

        model_inputs = self.adapt_input(data, **kwargs)
        raw_output = self.model.forward(
            model_inputs,
            compute_force=compute_forces,
            compute_stress=compute_stresses,
            # compute_displacement enables the MACE displacement trick required
            # for stress computation via autograd through cell @ neighbor_list_shifts.
            compute_displacement=compute_stresses,
            # Train mode retains the autograd graph through forces/stresses so
            # force/stress losses can backprop; eval mode (inference, MD, DD)
            # keeps the cheaper no-create-graph path.
            training=self.training,
        )
        result = self.adapt_output(raw_output, data)
        return result

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute node and graph embeddings without forces or stresses.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; an ``AtomicData`` is promoted to a ``Batch``.
        **kwargs
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        AtomicData | Batch
            *data*, with ``node_embeddings`` ``[N, hidden_dim]`` and
            ``graph_embeddings`` ``[B, hidden_dim]`` (sum-pooled) written in
            place. ``model_config`` is not mutated.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        model_inputs = self.adapt_input(data, **kwargs)

        # Pass flags as local kwargs — never mutate self.model_config.
        raw_output = self.model.forward(
            model_inputs,
            compute_force=False,
            compute_stress=False,
            compute_displacement=False,
            training=False,
        )

        node_feats = raw_output.get("node_feats")
        if node_feats is None:
            raise RuntimeError(
                "MACE model did not return 'node_feats'. "
                "Ensure the model is a standard MACE variant."
            )

        # Write to the atoms group directly: a plain attribute set would route to
        # the system group and block the later per-graph graph_embeddings write.
        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = node_feats
        else:
            data.node_embeddings = node_feats

        hidden_dim = node_feats.shape[-1]
        graph_embeddings = torch.zeros(
            data.num_graphs,
            hidden_dim,
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            data.batch_idx.long().unsqueeze(-1).expand(-1, hidden_dim),
            node_feats,
        )
        data.graph_embeddings = graph_embeddings
        return data

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device = torch.device("cpu"),
        enable_cueq: bool = False,
        dtype: torch.dtype | None = None,
        compile_model: bool = False,
        atomic_energies: Mapping[int | str, float] | None = None,
        atomic_energies_path: Path | str | None = None,
        **compile_kwargs: Any,
    ) -> "MACEWrapper":
        """Load a MACE model from a checkpoint and return a :class:`MACEWrapper`.

        Accepts local file paths or named MACE-MP foundation-model checkpoints
        (e.g. ``"medium-0b2"``), which are downloaded automatically to the
        MACE cache directory.

        Operations are applied in this order:

        1. **Load** — ``torch.load`` the checkpoint to the specified device.
        2. **dtype** — cast model weights to the requested dtype.
        3. **cuEq** — convert to cuEquivariance format for GPU speedup.
        4. **compile** — ``torch.compile``; freezes parameters and sets eval
           mode.  The model is **inference-only** after this step.

        For best GPU throughput, use ``device=torch.device("cuda")``,
        ``enable_cueq=True``, ``dtype=torch.float32``, and
        ``compile_model=True``.  Example::

            model = MACEWrapper.from_checkpoint(
                "medium-mpa-0",
                device=torch.device("cuda"),
                dtype=torch.float32,
                enable_cueq=True,
                compile_model=True,
            )

        Parameters
        ----------
        checkpoint_path : Path | str
            Local path to a ``.pt`` file, or a named checkpoint string such as
            ``"medium-0b2"``.
        device : torch.device, optional
            Target device.  Defaults to CPU.
        enable_cueq : bool, optional
            Convert to cuEquivariance format for GPU speedup.  Defaults to
            ``False``.  Requires the ``cuequivariance`` package.
        dtype : torch.dtype | None, optional
            If set, cast model weights to this dtype before cuEq conversion.
        compile_model : bool, optional
            Apply ``torch.compile``.  Sets eval mode and freezes parameters;
            the model is **inference-only** after this step.
        atomic_energies : Mapping[int | str, float] | None, optional
            Per-element E0 overrides keyed by atomic number.
        atomic_energies_path : Path | str | None, optional
            JSON file containing per-element E0 overrides keyed by atomic number.
        **compile_kwargs
            Forwarded to ``torch.compile``.

        Returns
        -------
        MACEWrapper

        Raises
        ------
        ImportError
            If ``mace-torch`` is not installed, or if ``enable_cueq=True``
            and ``cuequivariance`` is not installed.
        ValueError
            If ``enable_cueq=True`` and ``device`` is not a CUDA device.
        """
        OptionalDependency.MACE.is_available() or OptionalDependency.MACE._raise_error(
            "MACEWrapper.from_checkpoint"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from mace.calculators.foundations_models import download_mace_mp_checkpoint

        target_device = torch.device(device)
        cached_path = download_mace_mp_checkpoint(checkpoint_path)
        model: nn.Module = torch.load(
            cached_path, weights_only=False, map_location=target_device
        )

        # Step 1: dtype conversion.
        if dtype is not None:
            model.to(dtype=dtype)

        # Step 2: cuEq conversion.
        if enable_cueq:
            try:
                import cuequivariance  # noqa: F401
            except ImportError:
                raise ImportError(
                    "cuequivariance is required for enable_cueq=True. "
                    "Install it with: pip install 'nvalchemi-toolkit[mace]'"
                )
            from mace.cli.convert_e3nn_cueq import run as _convert_mace_weights

            if target_device.type != "cuda":
                raise ValueError(
                    "nvalchemi Toolkit MACE cuEquivariance conversion requires "
                    "a CUDA device."
                )
            with torch.cuda.device(target_device):
                model = _convert_mace_weights(
                    model,
                    return_model=True,
                    device="cuda",
                )

        model = model.to(target_device)

        atomic_energy_overrides = _load_atomic_energies(
            atomic_energies, atomic_energies_path
        )
        if atomic_energy_overrides is not None:
            _apply_atomic_energies(model, atomic_energy_overrides)

        # Step 3: torch.compile the model for single-process inference —
        # inference-only after this point. 
        if compile_model:
            # Apply the e3nn compile-compat shim before tracing. (It is also
            # applied idempotently in __init__, but compile runs first here.)
            _patch_e3nn_irrep_len_for_compile()
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            model = torch.compile(model, **compile_kwargs)

        from nvalchemi.training._spec import create_model_spec

        checkpoint_spec = create_model_spec(
            cls.from_checkpoint,
            checkpoint_path=str(checkpoint_path),
            enable_cueq=enable_cueq,
            dtype=dtype,
            compile_model=compile_model,
            atomic_energies=atomic_energies,
            atomic_energies_path=atomic_energies_path,
            **compile_kwargs,
        )
        wrapper = cls(model, reconstruction_spec=checkpoint_spec)
        wrapper.eval()
        return wrapper

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Serialize the underlying MACE model without the wrapper.

        The exported file can be reloaded as a plain MACE ``nn.Module`` and
        used with the standard MACE / ASE interface.

        Parameters
        ----------
        path : Path
            Output path.
        as_state_dict : bool, optional
            If ``True``, save only the ``state_dict``; otherwise pickle the
            full model object.  Defaults to ``False``.
        """
        if as_state_dict:
            torch.save(self.model.state_dict(), path)
        else:
            torch.save(self.model, path)
