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
"""PET (Point Edge Transformer) model wrapper.

Wraps the pure-torch :class:`metatrain.pet.modules.backend.PETBackend` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model. ``PETBackend``
is the structure-preprocessing / featurization / prediction core of the PET
architecture, operating purely on :class:`torch.Tensor` objects (no
``metatomic.torch.System`` / ``metatensor.torch.TensorMap`` at call time), so it
is ``torch.compile``-friendly.

:class:`PETWrapper` owns a ``PETBackend`` (built from hypers + atomic types) and
adds only the nvalchemi-specific glue:

* translating a :class:`~nvalchemi.data.Batch` into the concatenated plain
  tensors the backend expects (:meth:`PETWrapper.adapt_input`, mirroring
  :func:`metatrain.pet.modules.structures.concatenate_structures`);
* driving the three backend building blocks
  (:meth:`~metatrain.pet.modules.backend.PETBackend.preprocess`,
  :meth:`~metatrain.pet.modules.backend.PETBackend.calculate_features`,
  :meth:`~metatrain.pet.modules.backend.PETBackend.predict`);
* gradient / affine-strain wiring for conservative forces and stress;
* the flat composition / scaler buffers decoded from the checkpoint.

Usage
-----
Load a PET checkpoint (e.g. ``pet-mad-xs-v1.5.0.ckpt``)::

    from nvalchemi.models.pet import PETWrapper
    import torch

    model = PETWrapper.from_checkpoint(
        "pet-mad-xs-v1.5.0.ckpt",
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

Notes
-----
* Forces and stress are derived from the energy via autograd
  (``autograd_outputs = {"forces", "stress"}``). The non-conservative PET
  heads are intentionally skipped.
* The upstream composition model and scaler — originally wrapped as
  ``metatomic.torch.AtomisticModel`` with serialized ``TensorMap`` buffers
  — are decoded once at :meth:`PETWrapper.from_checkpoint` time into two
  flat torch buffers (``composition_energy``, ``scale_energy``) so the
  forward path has no metatensor dependency.
* Only the ``energy`` output is registered on the backend; the long-range
  module is skipped entirely.
"""

from __future__ import annotations

import contextlib
import sys
import types
import warnings

# metatrain.pet.__init__ imports metatrain.pet.trainer, which imports
# metatrain.utils.distributed.slurm, which pulls in `hostlist`. That package
# is a SLURM-only helper that the nvalchemi dev environment explicitly
# disables via the `override-dependencies` block in `pyproject.toml`. When
# it isn't present, stub a minimal module so the import chain resolves and
# the tests can still exercise PET's pure-torch code paths.
if "hostlist" not in sys.modules:
    try:
        import hostlist  # noqa: F401
    except ImportError:
        sys.modules["hostlist"] = types.ModuleType("hostlist")

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import (
    autograd_forces_and_stresses,
    autograd_stresses,
    prepare_strain,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PETWrapper"]


# ---------------------------------------------------------------------------
# Hyper-parameter normalisation
# ---------------------------------------------------------------------------


_REQUIRED_HYPERS: tuple[str, ...] = (
    "cutoff",
    "cutoff_function",
    "cutoff_width",
    "d_pet",
    "d_node",
    "d_head",
    "d_feedforward",
    "num_heads",
    "num_gnn_layers",
    "num_attention_layers",
    "normalization",
    "activation",
    "attention_temperature",
    "transformer_type",
    "featurizer_type",
)

# The single-block, scalar ``energy`` output shape passed to
# ``PETBackend.add_output``. The block key ``energy___0`` and shape ``[1]`` are
# what ``metatrain.pet.model.PET._add_output`` derives for a standard scalar
# energy target (a single TensorMap block with key name ``"_"`` / value ``0``
# and one property).
_ENERGY_OUTPUT_SHAPES: dict[str, list[int]] = {"energy___0": [1]}


def _normalize_hypers(hypers: dict[str, Any]) -> dict[str, Any]:
    """Validate *hypers* and fill in the keys ``PETBackend`` reads directly.

    Returns a copy with ``num_neighbors_adaptive`` / ``adaptive_cutoff_method`` /
    ``cutoff_width_adaptive`` / ``system_conditioning`` defaulted when absent
    (checkpoints already carry these after
    :meth:`metatrain.pet.PET.upgrade_checkpoint`; hand-built hypers may not).

    Parameters
    ----------
    hypers : dict[str, Any]
        Hyper-parameter dict pulled from the checkpoint's ``model_hypers`` (or
        constructed by hand for tests).

    Returns
    -------
    dict[str, Any]
        Normalised copy.

    Raises
    ------
    ValueError
        When a required key is missing.
    """
    missing = [key for key in _REQUIRED_HYPERS if key not in hypers]
    if missing:
        raise ValueError(f"PET hypers are missing required keys: {missing}")
    normalized = dict(hypers)
    normalized.setdefault("num_neighbors_adaptive", None)
    normalized.setdefault("adaptive_cutoff_method", "grid")
    normalized.setdefault("cutoff_width_adaptive", 1.0)
    normalized.setdefault("system_conditioning", False)
    return normalized


# ---------------------------------------------------------------------------
# State-dict filtering helpers
# ---------------------------------------------------------------------------


_HEAD_PREFIXES: tuple[str, ...] = (
    "node_heads.",
    "edge_heads.",
    "node_last_layers.",
    "edge_last_layers.",
)


def _filter_state_dict(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filter a (upgraded) PET checkpoint state dict down to the backend.

    The current metatrain checkpoint layout (>= v14) nests the pure-torch core
    under a ``backend.`` prefix and keeps the additive composition model,
    scaler, long-range featurizer and ``finetune_config`` outside it. This keeps
    only the ``backend.*`` keys (with the prefix stripped) and, among those,
    only the ``energy`` readout heads/last-layers — dropping any other output
    (e.g. ``non_conservative_forces`` / ``non_conservative_stress``) so the
    result loads into an energy-only :class:`PETBackend` with ``strict=True``.

    Parameters
    ----------
    raw_sd : dict[str, torch.Tensor]
        State dict from the upgraded ``wrapped_model_checkpoint``'s
        ``model_state_dict``.

    Returns
    -------
    dict[str, torch.Tensor]
        Filtered state dict, keyed for ``PETBackend.load_state_dict``.
    """
    filtered: dict[str, torch.Tensor] = {}
    for key, value in raw_sd.items():
        if not key.startswith("backend."):
            # Drops additive_models.*, scaler.*, long_range_featurizer.*,
            # finetune_config, etc.
            continue
        name = key[len("backend.") :]
        if name.startswith(_HEAD_PREFIXES):
            # name == "<which>_heads.<output>.<...>"; keep only energy.
            output_name = name.split(".")[1]
            if output_name != "energy":
                continue
        filtered[name] = value
    return filtered


def _decode_tensor_map_values(buffer: torch.Tensor) -> torch.Tensor:
    """Decode a serialized ``TensorMap`` buffer to a flat tensor.

    The metatrain composition / scaler modules store their weights as
    ``TensorMap`` byte buffers (uint8 tensors). Decoding with
    ``metatensor.torch.load_buffer`` returns a :class:`TensorMap`; the
    single block's ``values`` tensor carries the actual numeric weights.

    Parameters
    ----------
    buffer : torch.Tensor
        ``uint8`` tensor holding the serialized TensorMap.

    Returns
    -------
    torch.Tensor
        The decoded ``values`` tensor from the single block of the
        TensorMap.
    """
    import metatensor.torch as mts

    tensor_map = mts.load_buffer(buffer)
    return tensor_map.block(0).values


@contextlib.contextmanager
def _ignore_nonleaf_grad_warning():
    """Silence the benign non-leaf ``.grad`` warning emitted under compile.

    When Dynamo's builder wraps a grad-tracking (non-leaf, ``requires_grad``)
    tensor as a graph input, it reads its ``.grad`` and PyTorch emits a harmless
    ``UserWarning``. The autograd graph is left intact, so forces via
    ``autograd.grad`` still work. Mirrors the helper in
    ``metatrain/pet/tests/test_backend.py``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*grad attribute of a Tensor that is not a leaf Tensor.*",
            category=UserWarning,
        )
        yield


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


@OptionalDependency.PET.require
class PETWrapper(nn.Module, BaseModelMixin):
    """:class:`~nvalchemi.models.base.BaseModelMixin` wrapper around PET.

    Builds and owns a :class:`metatrain.pet.modules.backend.PETBackend` (from
    *hypers* and *atomic_types*) and drives its three building blocks. Handles:

    * translating a :class:`~nvalchemi.data.Batch` into the concatenated plain
      tensors consumed by :meth:`PETBackend.preprocess`
      (:meth:`adapt_input`);
    * enabling gradients on ``positions`` when autograd outputs are active, and
      wiring the affine strain trick
      (:func:`~nvalchemi.models._utils.prepare_strain`) for stress;
    * applying the flat composition / scaler buffers decoded from the
      checkpoint at load time;
    * producing :class:`~nvalchemi._typing.ModelOutputs` with ``energy``,
      ``forces``, and ``stress``.

    Parameters
    ----------
    atomic_types : Sequence[int]
        Atomic numbers in species-index order.
    hypers : dict[str, Any]
        PET hyper-parameters. Must include the keys listed in
        :data:`_REQUIRED_HYPERS`.
    composition_energy : torch.Tensor
        Per-species reference energy, shape ``[num_species]``. Indexed by the
        species index (not by atomic number).
    scale_energy : torch.Tensor
        Scalar (0-dim) tensor used as the global energy scale.

    Attributes
    ----------
    backend : PETBackend
        Underlying pure-torch PET core.
    atomic_types : list[int]
        Copy of the atomic-number list used to build the backend.
    hypers : dict[str, Any]
        Normalised copy of the hyper-parameters.
    model_config : ModelConfig
        Capability declaration with ``active_outputs`` defaulting to
        ``{"energy", "forces", "stress"}``.
    """

    def __init__(
        self,
        atomic_types: Sequence[int],
        hypers: dict[str, Any],
        composition_energy: torch.Tensor,
        scale_energy: torch.Tensor,
    ) -> None:
        from metatrain.pet.modules.backend import PETBackend

        super().__init__()

        self.atomic_types = list(atomic_types)
        self.hypers = _normalize_hypers(hypers)
        # Width of the adaptive-cutoff smooth taper; not cached on ``PETBackend``
        # itself (unlike the other hypers), so the wrapper keeps its own copy
        # to pass into ``preprocess`` at call time.
        self._cutoff_width_adaptive = float(self.hypers["cutoff_width_adaptive"])

        # Build the core from hypers + atomic species and register the single
        # scalar energy output (heads + last layers).
        self.backend = PETBackend(self.hypers, self.atomic_types)
        self.backend.add_output("energy", _ENERGY_OUTPUT_SHAPES)

        # Set to True by `from_checkpoint(compile_model=True)`, which compiles
        # the three backend methods; controls the Dynamo config patching applied
        # around the backend calls in `forward` / `compute_embeddings`.
        self._compiled = False

        # Per-species reference energy (shape [num_species]) indexed by species
        # index (backend.species_to_species_index lookup), not atomic number.
        # Non-persistent: decoded at `from_checkpoint` time from the raw
        # metatensor buffer; saved back via `export_model` using the same route.
        self.register_buffer(
            "composition_energy", composition_energy.clone(), persistent=False
        )
        # Scalar global scaler. Kept as a 0-dim tensor for broadcast.
        self.register_buffer(
            "scale_energy", scale_energy.clone().reshape(()), persistent=False
        )

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell", "neighbor_list_shifts"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=float(self.hypers["cutoff"]),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return node/graph embedding shapes.

        Embeddings concatenate the per-layer node features with the
        cutoff-weighted, neighbor-summed per-layer edge features (see
        :meth:`compute_embeddings`), so the dimension is
        ``num_readout_layers * (d_node + d_pet)``.
        """
        dim = self.backend.num_readout_layers * (
            self.backend.d_node + self.backend.d_pet
        )
        return {"node_embeddings": (dim,), "graph_embeddings": (dim,)}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Interaction cutoff in Angstroms."""
        return float(self.backend.cutoff)

    @property
    def _model_dtype(self) -> torch.dtype:
        """Return the current dtype of the backend's parameters.

        Read live from ``parameters()`` so it stays correct after
        ``.to(dtype=...)`` calls.
        """
        try:
            return next(self.backend.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # Backend invocation (compile-aware)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _backend_ctx(self):
        """Context for calling the backend building blocks.

        When the backend methods have been ``torch.compile``-d
        (``self._compiled``), the Dynamo flags required to capture the
        data-dependent ``max_edges_per_node`` size must be active while the
        compiled functions trace (lazily, on first call) — so the calls
        themselves run inside the ``config.patch`` context, matching
        ``metatrain/pet/tests/test_backend.py``. The benign non-leaf ``.grad``
        warning Dynamo emits while building grad-tracking graph inputs (when
        autograd forces / stress are active) is silenced. In eager mode this is
        a no-op.
        """
        if self._compiled:
            with (
                torch._dynamo.config.patch(
                    capture_scalar_outputs=True,
                    capture_dynamic_output_shape_ops=True,
                    specialize_int=True,
                ),
                _ignore_nonleaf_grad_warning(),
            ):
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def adapt_input(
        self, data: AtomicData | Batch, **_kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Translate a :class:`~nvalchemi.data.Batch` into backend input tensors.

        Produces the concatenated, plain-tensor structure representation that
        :meth:`PETBackend.preprocess` consumes — the nvalchemi analogue of
        :func:`metatrain.pet.modules.structures.concatenate_structures`. All the
        edge manipulation (NEF reshaping, adaptive cutoffs, reversed-neighbor
        indexing) then happens inside the backend.

        Handles ``AtomicData -> Batch`` promotion and gradient enabling on
        ``positions`` when an autograd output is active. Strain handling (for
        stress) is done by :meth:`forward` **before** calling this method, so
        that the scaled positions/cell flow through the full featurisation.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data. ``AtomicData`` inputs are promoted to a single-graph
            ``Batch``.
        **_kwargs : Any
            Ignored — kept for interface compatibility.

        Returns
        -------
        dict[str, torch.Tensor]
            Keyword arguments for :meth:`PETBackend.preprocess`:
            ``positions``, ``centers``, ``neighbors``, ``species``, ``cells``,
            ``cell_shifts``, ``system_indices``.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype

        # Cast positions to model dtype and enable gradients when needed.
        # Clone so the original batch tensor is never mutated in-place.
        positions = data.positions.to(dtype=dtype)
        if self.model_config.autograd_outputs & self.model_config.active_outputs:
            positions = positions.clone()
            positions.requires_grad_(True)
        # Store the prepared positions back on the batch so that downstream
        # autograd calls in `forward` can reach them via `data.positions`.
        data["positions"] = positions

        return self._collect_backend_inputs(data, dtype)

    def _collect_backend_inputs(
        self, data: Batch, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Gather the :meth:`PETBackend.preprocess` kwargs from a prepared batch.

        Reads ``data.positions`` as-is (the caller is responsible for any dtype
        cast / gradient setup) and assembles the remaining structure tensors,
        mirroring :func:`metatrain.pet.modules.structures.concatenate_structures`.

        Parameters
        ----------
        data : Batch
            Batch whose ``positions`` are already prepared.
        dtype : torch.dtype
            Model dtype, used to cast ``cells``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keyword arguments for :meth:`PETBackend.preprocess`.
        """
        positions = data.positions
        device = positions.device
        num_graphs = int(data.num_graphs)

        centers = data.neighbor_list[:, 0].long()
        neighbors = data.neighbor_list[:, 1].long()
        species = data.atomic_numbers.long()
        system_indices = data.batch_idx.long()

        # Integer PBC shifts [E, 3] — zero for non-PBC systems.
        raw_shifts = getattr(data, "neighbor_list_shifts", None)
        if raw_shifts is None:
            cell_shifts = torch.zeros(
                centers.shape[0], 3, dtype=torch.long, device=device
            )
        else:
            cell_shifts = raw_shifts.to(dtype=torch.long, device=device)

        # Cell [B, 3, 3] — identity for non-PBC systems.
        raw_cell = getattr(data, "cell", None)
        if raw_cell is None:
            cells = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(num_graphs, -1, -1)
                .contiguous()
            )
        else:
            cells = raw_cell.to(dtype=dtype, device=device)

        return {
            "positions": positions,
            "centers": centers,
            "neighbors": neighbors,
            "species": species,
            "cells": cells,
            "cell_shifts": cell_shifts,
            "system_indices": system_indices,
        }

    def adapt_output(
        self,
        raw_output: dict[str, torch.Tensor | None],
        data: AtomicData | Batch,
    ) -> ModelOutputs:
        """Map raw PET outputs to the standard :class:`ModelOutputs` layout.

        Parameters
        ----------
        raw_output : dict[str, torch.Tensor | None]
            Dict with optional ``energy``, ``forces``, ``stress`` tensors.
        data : AtomicData | Batch
            Original input batch (unused here but forwarded to
            :meth:`BaseModelMixin.adapt_output`).

        Returns
        -------
        ModelOutputs
            Ordered dict keyed by the wrapper's active outputs.
        """
        mapped: dict[str, torch.Tensor] = {}
        energy = raw_output.get("energy")
        if energy is not None:
            mapped["energy"] = energy.unsqueeze(-1) if energy.ndim == 1 else energy
        if raw_output.get("forces") is not None:
            mapped["forces"] = raw_output["forces"]
        if raw_output.get("stress") is not None:
            mapped["stress"] = raw_output["stress"]
        return super().adapt_output(mapped, data)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the PET backend and return energy / forces / stress.

        The energy comes from
        :meth:`PETBackend.preprocess` ->
        :meth:`PETBackend.calculate_features` ->
        :meth:`PETBackend.predict` (the latter already sums the node and
        cutoff-weighted edge contributions over all readout layers). The flat
        scaler / composition buffers are then applied.

        Conservative forces are derived via :func:`torch.autograd.grad` of the
        total energy with respect to positions. Stresses use the affine-strain
        trick from :func:`~nvalchemi.models._utils.prepare_strain` /
        :func:`~nvalchemi.models._utils.autograd_stresses`.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch.
        **kwargs : Any
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        ModelOutputs
            Dict with the active output keys populated.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        active = self.model_config.active_outputs & self.model_config.outputs
        compute_forces = "forces" in active
        compute_stresses = "stress" in active

        # Set up the affine strain BEFORE adapt_input so the scaled positions
        # and cell flow through the full featurisation.
        displacement: torch.Tensor | None = None
        orig_positions: torch.Tensor | None = None
        orig_cell: torch.Tensor | None = None
        if compute_stresses and getattr(data, "cell", None) is not None:
            scaled_pos, scaled_cell, displacement = prepare_strain(
                data.positions.to(self._model_dtype),
                data.cell.to(self._model_dtype),
                data.batch_idx,
            )
            orig_positions = data.positions
            orig_cell = data.cell
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        inputs = self.adapt_input(data, **kwargs)
        positions = data.positions  # updated in-place by adapt_input

        with self._backend_ctx():
            batch_data = self.backend.preprocess(
                inputs["positions"],
                inputs["centers"],
                inputs["neighbors"],
                inputs["species"],
                inputs["cells"],
                inputs["cell_shifts"],
                inputs["system_indices"],
                self._cutoff_width_adaptive,
            )
            node_features_list, edge_features_list = self.backend.calculate_features(
                batch_data
            )
            atomic_predictions, _, _ = self.backend.predict(
                node_features_list,
                edge_features_list,
                batch_data,
                inputs["cells"],
                inputs["system_indices"],
                ["energy"],
            )

        per_atom = atomic_predictions["energy"][0]  # [N, 1]
        species_idx = self.backend.species_to_species_index[inputs["species"]]
        # Scaler first, then composition (matches upstream PET ordering).
        per_atom = self.scale_energy * per_atom
        per_atom = per_atom + self.composition_energy[species_idx].unsqueeze(-1)

        num_graphs = int(data.num_graphs)
        energy = torch.zeros(
            num_graphs, 1, dtype=per_atom.dtype, device=per_atom.device
        )
        energy.scatter_add_(0, inputs["system_indices"].unsqueeze(-1), per_atom)

        result: dict[str, torch.Tensor] = {"energy": energy}

        need_stress = (
            compute_stresses and displacement is not None and orig_cell is not None
        )
        if compute_forces and need_stress:
            # A single backward for both forces and stress. Two separate
            # ``autograd.grad`` calls would run backward twice over the same
            # graph, which clashes with ``torch.compile``'s donated-buffer
            # optimization (it requires create_graph=retain_graph=False).
            forces, stress = autograd_forces_and_stresses(
                energy,
                positions,
                displacement,
                orig_cell,
                num_graphs,
            )
            result["forces"] = forces
            result["stress"] = stress
        elif compute_forces:
            (grad,) = torch.autograd.grad(energy.sum(), positions)
            result["forces"] = -grad
        elif need_stress:
            result["stress"] = autograd_stresses(
                energy,
                displacement,
                orig_cell,
                num_graphs,
            )

        # Restore the batch's original positions/cell if strain was applied,
        # so the caller sees no mutation from the stress trick.
        if orig_positions is not None and orig_cell is not None:
            data["positions"] = orig_positions
            data["cell"] = orig_cell

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute node and graph embeddings without autograd.

        The node embedding is the concatenation of the per-layer node features
        with the cutoff-weighted, neighbor-summed per-layer edge features —
        matching ``metatrain.pet.model.PET._get_output_features``::

            node = cat(node_features_list, dim=1)
            edge = (cat(edge_features_list, dim=2) * cutoff_factors).sum(neighbors)
            feats = cat([node, edge], dim=1)

        Writes ``node_embeddings``
        (``[N, num_readout_layers*(d_node+d_pet)]``) and ``graph_embeddings``
        (``[B, ...]``, sum-pooled over atoms) into *data* and returns it. Does
        **not** mutate ``model_config``.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data.
        **kwargs : Any
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        AtomicData | Batch
            The same batch with ``node_embeddings`` and ``graph_embeddings``
            attached.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        with torch.no_grad():
            # Build inputs without enabling gradients on positions (embeddings
            # are autograd-free), so adapt_input's grad toggle is bypassed.
            data["positions"] = data.positions.to(dtype=self._model_dtype)
            inputs = self._collect_backend_inputs(data, self._model_dtype)
            with self._backend_ctx():
                batch_data = self.backend.preprocess(
                    inputs["positions"],
                    inputs["centers"],
                    inputs["neighbors"],
                    inputs["species"],
                    inputs["cells"],
                    inputs["cell_shifts"],
                    inputs["system_indices"],
                    self._cutoff_width_adaptive,
                )
                node_features_list, edge_features_list = (
                    self.backend.calculate_features(batch_data)
                )

            node_features = torch.cat(node_features_list, dim=1)
            edge_features = torch.cat(edge_features_list, dim=2)
            edge_features = (
                edge_features * batch_data["cutoff_factors"][:, :, None]
            ).sum(dim=1)
            node_feats = torch.cat([node_features, edge_features], dim=1)

        # Write node embeddings directly to the atoms group to avoid the
        # default "system" routing used by `setattr` on unknown keys.
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
        dtype: torch.dtype | None = None,
        compile_model: bool = False,
        **compile_kwargs: Any,
    ) -> "PETWrapper":
        """Load a PET checkpoint from disk and return a wrapped instance.

        Expects a ``torch.load``-friendly dict written by
        :class:`metatrain.pet.PET`. The outer dict is an LLPR wrapper; the
        PET core lives in ``wrapped_model_checkpoint`` if that key is present,
        otherwise directly at the top level. The (possibly old) checkpoint is
        brought to the current layout via
        :meth:`metatrain.pet.PET.upgrade_checkpoint` before its hypers /
        ``state_dict`` are read — old checkpoints (e.g. ``pet-mad-xs-v1.5.0``,
        version 11) predate the ``backend.`` state-dict prefix.

        Metatomic's torch extension **must** be importable before
        :func:`torch.load` is called, otherwise the checkpoint's
        ``ScriptObject`` metadata fails to unpickle. We import
        :mod:`metatomic.torch` lazily here to enforce that without polluting
        module import time.

        Parameters
        ----------
        checkpoint_path : Path | str
            Path to a PET checkpoint file (``.ckpt`` / ``.pt``).
        device : torch.device, optional
            Target device. Defaults to CPU.
        dtype : torch.dtype | None, optional
            If set, cast the backend and composition/scaler buffers to this
            dtype before returning.
        compile_model : bool, optional
            ``torch.compile`` the three backend building blocks
            (``preprocess`` / ``calculate_features`` / ``predict``). Sets eval
            mode and freezes parameters; the model is **inference-only** after
            this step. Conservative forces and stress via autograd still work
            through the compiled graph (the combined single-backward in
            :meth:`forward` keeps it compatible with ``torch.compile``'s
            donated-buffer optimization). Note that the backend's
            data-dependent ``max_edges_per_node`` size requires
            ``capture_scalar_outputs`` / ``capture_dynamic_output_shape_ops`` to
            be set, which :meth:`forward` applies via ``torch._dynamo.config``
            patching at call time (see :meth:`_backend_ctx`).

            Models using the ``'grid'`` adaptive-cutoff method (pet-mad
            <= v1.5.0) cannot be compiled at all — autograd backward through the
            compiled grid cutoff aborts — so ``compile_model=True`` for such a
            model raises ``ValueError``. Use a ``'solver'``-method checkpoint
            (pet-mad >= v1.6.0) to compile, or run the grid model eagerly.
        **compile_kwargs
            Forwarded verbatim to each ``torch.compile`` call, so the caller
            chooses the compilation options (e.g. ``fullgraph=True``,
            ``mode=...``, ``dynamic=...``).

        Returns
        -------
        PETWrapper

        Raises
        ------
        OptionalDependencyError
            When :mod:`metatrain` is not installed.
        ValueError
            When ``compile_model`` is requested for a ``'grid'`` adaptive-cutoff
            model (see ``compile_model`` above).
        """
        if not OptionalDependency.PET.is_available():
            OptionalDependency.PET._raise_error(f"{cls.__qualname__}.from_checkpoint")
        # Make sure metatomic's custom torch ops are registered before
        # torch.load, otherwise the ScriptObject metadata unpickling fails.
        import metatomic.torch  # noqa: F401
        from metatrain.pet import PET

        raw = torch.load(str(checkpoint_path), weights_only=False, map_location=device)
        if isinstance(raw, dict) and "wrapped_model_checkpoint" in raw:
            raw = raw["wrapped_model_checkpoint"]

        # Bring an old checkpoint up to the current model version (adds the
        # `backend.` prefix and any missing hypers). Mutates `raw` in place.
        raw = PET.upgrade_checkpoint(raw)

        model_data = raw["model_data"]
        hypers = dict(model_data["model_hypers"])
        atomic_types = list(model_data["dataset_info"].atomic_types)
        # Prefer the latest weights (``model_state_dict``); fall back to the best
        # epoch (``best_model_state_dict``). Exported / best-only checkpoints
        # (e.g. pet-mad-xs-v1.6.0) carry only the latter.
        raw_sd = raw.get("model_state_dict") or raw.get("best_model_state_dict")
        if raw_sd is None:
            raise KeyError(
                "Checkpoint has neither 'model_state_dict' nor 'best_model_state_dict'."
            )

        composition_values = _decode_tensor_map_values(
            raw_sd["additive_models.0.energy_composition_buffer"]
        )  # [num_species, 1]
        composition_energy = composition_values.squeeze(-1).clone()
        scale_values = _decode_tensor_map_values(
            raw_sd["scaler.energy_scaler_buffer"]
        )  # [1, 1]
        scale_energy = scale_values.reshape(()).clone()

        backend_sd = _filter_state_dict(raw_sd)
        wrapper = cls(
            atomic_types=atomic_types,
            hypers=hypers,
            composition_energy=composition_energy,
            scale_energy=scale_energy,
        )
        wrapper.backend.load_state_dict(backend_sd, strict=True)

        if dtype is not None:
            wrapper.backend = wrapper.backend.to(dtype=dtype)
            wrapper.composition_energy = wrapper.composition_energy.to(dtype=dtype)
            wrapper.scale_energy = wrapper.scale_energy.to(dtype=dtype)

        wrapper = wrapper.to(device)

        if compile_model:
            wrapper.eval()
            for param in wrapper.parameters():
                param.requires_grad = False
            # The 'grid' adaptive-cutoff method (what pet-mad <= v1.5.0 was
            # trained with) cannot be safely compiled: autograd backward through
            # the compiled grid cutoff aborts at the C++ level (with
            # ``fullgraph=True`` always, and intermittently even without it).
            # The 'solver' method (pet-mad >= v1.6.0) is fully compatible,
            # including ``fullgraph=True``. Refuse to compile a grid model so the
            # user gets a clear error instead of a hard crash at the first
            # backward.
            uses_grid_adaptive = (
                wrapper.backend.num_neighbors_adaptive is not None
                and str(wrapper.backend.adaptive_cutoff_method).lower() == "grid"
            )
            if uses_grid_adaptive:
                raise ValueError(
                    "compile_model=True is not supported for PET models using "
                    "the 'grid' adaptive-cutoff method (e.g. pet-mad-xs "
                    "<= v1.5.0): autograd backward through the compiled grid "
                    "cutoff aborts. Load a checkpoint trained with the 'solver' "
                    "method (e.g. pet-mad-xs >= v1.6.0) to use torch.compile, or "
                    "run the grid model in eager mode (compile_model=False)."
                )
            wrapper.backend.preprocess = torch.compile(
                wrapper.backend.preprocess, **compile_kwargs
            )
            wrapper.backend.calculate_features = torch.compile(
                wrapper.backend.calculate_features, **compile_kwargs
            )
            wrapper.backend.predict = torch.compile(
                wrapper.backend.predict, **compile_kwargs
            )
            wrapper._compiled = True
        return wrapper

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Serialize the wrapper to disk in a pure-torch layout.

        Writes a plain dict containing the backend ``state_dict``, the
        hyper-parameters, the atomic-type list, and the composition/scaler
        buffers. The output is **not** a metatrain / metatomic checkpoint — it
        is a self-contained snapshot that can be reloaded by constructing
        ``PETWrapper(atomic_types, hypers, ...)`` and calling
        ``load_state_dict`` on its backend.

        Parameters
        ----------
        path : Path
            Output path.
        as_state_dict : bool, optional
            If ``True``, save only the backend's ``state_dict``. Defaults to
            ``False`` (saves the full snapshot).
        """
        if as_state_dict:
            torch.save(self.backend.state_dict(), path)
        else:
            snapshot = {
                "backend_state_dict": self.backend.state_dict(),
                "hypers": self.hypers,
                "atomic_types": self.atomic_types,
                "composition_energy": self.composition_energy.detach().cpu(),
                "scale_energy": self.scale_energy.detach().cpu(),
            }
            torch.save(snapshot, path)
