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

import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

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

_torch_version = version("torch")

__all__ = ["MACEWrapper"]


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

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        # Cache the model dtype — determined at construction, stable thereafter.
        self._cached_model_dtype: torch.dtype = next(model.parameters()).dtype

        # Pre-build a one-hot lookup table: shape [max_z + 1, num_elements].
        # At runtime, node_attrs = _node_emb.index_select(0, atomic_numbers)
        # — a single GPU op, no CPU round-trips.
        z_table: list[int] = model.atomic_numbers.tolist()
        node_emb = torch.zeros(max(z_table) + 1, len(z_table))
        for i, z in enumerate(z_table):
            node_emb[z, i] = 1.0
        # Cast to model device+dtype so _node_attrs needs no per-step conversion.
        # Must use the model's device here: from_checkpoint moves the inner model
        # to the target device before calling cls(model), so the buffer must be
        # placed on that device from construction rather than relying on a
        # subsequent .to() call that never happens.
        model_device = next(model.parameters()).device
        node_emb = node_emb.to(device=model_device, dtype=self._cached_model_dtype)
        # persistent=False: derived from model.atomic_numbers, excluded from
        # state_dict but still tracked for device / dtype moves.
        self.register_buffer("_node_emb", node_emb, persistent=False)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress", "hessian"}),
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

        .. note::
            This method does **not** call ``super().adapt_input()`` because
            :class:`~nvalchemi.data.Batch` does not implement ``model_dump()``,
            which the base implementation requires.  Gradient enabling on
            ``positions`` is handled manually here instead.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype
        device = data.positions.device
        B = data.num_graphs

        # nvalchemi (E, 2) -> MACE COO (2, E)
        edge_index = data.neighbor_list.long().T  # [2, E]
        E = edge_index.shape[1]

        # Cast positions to model dtype, then enable gradients on the converted
        # tensor.  We always clone before enabling grad so that data.positions
        # is never mutated in-place (which would happen when dtype already
        # matches and .to() returns the same storage).
        positions = data.positions.to(dtype=dtype)
        compute_forces = "forces" in self.model_config.active_outputs
        compute_stresses = "stress" in self.model_config.active_outputs
        if compute_forces or compute_stresses:
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

        # Pre-compute physical shift vectors [E, 3].
        # MACE's prepare_graph always reads data["shifts"] (physical Å vectors)
        # directly; it only recomputes them internally when
        # compute_displacement=True (stress path).  We must supply "shifts" for
        # the energy/force-only path.
        # Convention: shifts[e] = neighbor_list_shifts[e] @ cell[graph_of_sender_e]
        # matching get_symmetric_displacement in mace.modules.utils.
        sender = edge_index[0]  # [E] — source node indices
        batch_per_edge = data.batch_idx[sender]
        shifts = torch.einsum("eb,ebc->ec", neighbor_list_shifts, cell[batch_per_edge])
        return {
            "positions": positions,
            "node_attrs": self._node_attrs(data),
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
        """Map MACE output keys to nvalchemi standard keys.

        MACE uses ``"energy"`` / ``"stress"`` / ``"hessian"``; nvalchemi
        expects ``"energy"`` / ``"stress"`` / ``"hessian"``.
        Renaming happens *before* calling ``super()`` so the base auto-mapper
        sees the canonical key names.
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

        return super().adapt_output(mapped, data)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the MACE model and return the output."""
        model_inputs = self.adapt_input(data, **kwargs)

        compute_forces = "forces" in (
            self.model_config.active_outputs & self.model_config.outputs
        )
        compute_stresses = "stress" in (
            self.model_config.active_outputs & self.model_config.outputs
        )

        raw_output = self.model.forward(
            model_inputs,
            compute_force=compute_forces,
            compute_stress=compute_stresses,
            # compute_displacement enables the MACE displacement trick required
            # for stress computation via autograd through cell @ neighbor_list_shifts.
            compute_displacement=compute_stresses,
            training=False,  # Only inference supported right now.
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

        Writes ``node_embeddings`` (shape ``[N, hidden_dim]``) and
        ``graph_embeddings`` (shape ``[B, hidden_dim]``, sum-pooled over atoms)
        into *data* in-place and returns it.  Does **not** mutate
        ``model_config``.
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

        # Write node embeddings directly to the atoms group to avoid the
        # default "system" routing in MultiLevelStorage for unknown keys.
        # If we wrote via `data.node_embeddings = ...`, it would land in the
        # system group (batch_size = [N]) and then block the graph_embeddings
        # write (batch_size = [B]) from going to the same group.
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
        """
        OptionalDependency.MACE.is_available() or OptionalDependency.MACE._raise_error(
            "MACEWrapper.from_checkpoint"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from mace.calculators.foundations_models import download_mace_mp_checkpoint

        cached_path = download_mace_mp_checkpoint(checkpoint_path)
        model: nn.Module = torch.load(
            cached_path, weights_only=False, map_location=device
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

            model = _convert_mace_weights(model, return_model=True, device=device)

        model = model.to(device)

        # Step 3: torch.compile — inference-only after this point.
        if compile_model:
            _patch_e3nn_irrep_len_for_compile()
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            model = torch.compile(model, **compile_kwargs)

        return cls(model)

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
