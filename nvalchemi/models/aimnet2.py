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
"""AIMNet2 model wrapper.

Wraps an AIMNet2 ``nn.Module`` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model, ready for
use in any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine or standalone
inference.

Usage
-----
Load from a checkpoint (downloads if needed)::

    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    wrapper = AIMNet2Wrapper.from_checkpoint("aimnet2", device="cuda")

Or wrap an already-loaded ``nn.Module``::

    raw_model = torch.load("aimnet2.pt", weights_only=False)
    wrapper = AIMNet2Wrapper(raw_model)

Notes
-----
* Energy is the primitive differentiable output. Forces and stresses are
  derived via autograd (``autograd_outputs={"forces", "stress"}``).
* AIMNet2 also predicts partial charges, which are available as a direct
  output (``"charges" in model_config.outputs``).
* Coulomb and D3 dispersion contributions are **disabled** inside the
  calculator — use :class:`~nvalchemi.models.pipeline.PipelineModelWrapper`
  to compose with :class:`~nvalchemi.models.ewald.EwaldModelWrapper` or
  :class:`~nvalchemi.models.dftd3.DFTD3ModelWrapper` for long-range
  interactions.
* AIMNet2 runs in **float32 only**. The wrapper enforces this.
* NSE (Neutral Spin Equilibrated) models are auto-detected at construction
  time. When detected, ``spin_charges`` is added to the output set.
* The wrapper uses an **external neighbor list** (MATRIX format) provided
  by :class:`~nvalchemi.dynamics.hooks.NeighborListHook`.  The neighbor
  matrix is converted to AIMNet2's internal ``nbmat`` format (with a
  padding row) before the model forward pass.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._optional import OptionalDependency
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed._core.context import current_dd_context
from nvalchemi.distributed._core.enums import Scope
from nvalchemi.distributed.helpers import (
    localize,
    refresh_neighbors,
    system_sum,
    to_local,
)
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

__all__ = ["AIMNet2Wrapper"]


_AIMNET2_HALO_SPEC_CACHE: Any = None


def _aimnet2_halo_spec() -> Any:
    """Build AIMNet2's halo MLIPSpec (local neighbors, owned+ghost storage).

    Returns
    -------
    Any
        A memoized ``MLIPSpec``. The conv kernel and Coulomb heads refresh
        their ghost rows before running locally, and the per-system sum
        counts owned atoms only.
    """
    global _AIMNET2_HALO_SPEC_CACHE
    if _AIMNET2_HALO_SPEC_CACHE is not None:
        return _AIMNET2_HALO_SPEC_CACHE
    from dataclasses import replace  # noqa: PLC0415

    from aimnet.modules.aev import ConvSV  # noqa: PLC0415
    from aimnet.modules.lr import LRCoulomb, SRCoulomb  # noqa: PLC0415

    from nvalchemi.distributed.graph_padder import DenseBatchPadder  # noqa: PLC0415
    from nvalchemi.distributed.spec import (  # noqa: PLC0415
        SPEC_MPNN_HALO,
        CompilePolicy,
        ForceStrategy,
        MethodAdapter,
        PythonAdapter,
    )
    helpers = (
        PythonAdapter(
            module_path="aimnet.nbops",
            attr_name="mol_sum",
            replacement=_distributed_mol_sum,
        ),
        MethodAdapter(ConvSV, "forward", _distributed_conv_sv_forward),
        MethodAdapter(LRCoulomb, "forward", _distributed_coulomb_forward),
        MethodAdapter(SRCoulomb, "forward", _distributed_coulomb_forward),
    )
    _AIMNET2_HALO_SPEC_CACHE = replace(
        SPEC_MPNN_HALO,
        distribution=replace(
            SPEC_MPNN_HALO.distribution,
            adapters=helpers,
            # Only positions are sharded; atomic_numbers stay plain because they
            # feed an embedding that would otherwise mix tensor types in backward.
            shard_fields=("positions",),
        ),
        # Forces come from autograd over an energy-only forward; the per-system
        # sum already yields the global energy. The dense neighbor matrix is
        # padded to fixed shapes for compile.
        compile=CompilePolicy(
            static_shapes=True,
            force_strategy=ForceStrategy.FRAMEWORK_FROM_GLOBAL_ENERGY,
            graph_padder=DenseBatchPadder(),
        ),
    )
    return _AIMNET2_HALO_SPEC_CACHE


@OptionalDependency.AIMNET.require
class AIMNet2Wrapper(nn.Module, BaseModelMixin):
    """Wrapper for AIMNet2 interatomic potentials.

    Energy is always computed as the primitive differentiable output via
    the raw AIMNet2 model. Forces and stresses are derived from energy
    via autograd. Partial charges and node embeddings (AIM features) are
    taken directly from the model outputs.

    The wrapper declares an **external** MATRIX-format neighbor list
    requirement at the model's AEV cutoff. The
    :class:`~nvalchemi.dynamics.hooks.NeighborListHook` (or the pipeline's
    synthesized hook) populates ``neighbor_matrix`` on the batch before
    each forward pass.  The wrapper converts this to AIMNet2's internal
    ``nbmat`` format (with a padding row for the padding atom).

    Coulomb and D3 dispersion are disabled.  Use
    :class:`~nvalchemi.models.pipeline.PipelineModelWrapper` to compose
    AIMNet2 with electrostatics or dispersion models.

    Parameters
    ----------
    model : nn.Module
        An AIMNet2 model (loaded from checkpoint or instantiated
        directly).  Use :meth:`from_checkpoint` for the common
        construction path.
    compile_model : bool, optional
        ``torch.compile`` the AIMNet2 module forward via the calculator's
        kernel-aware compile path (single-process inference). Distributed
        compilation is a separate switch, ``DistributedModel(..., compile=True)``.
    compile_kwargs : dict[str, Any] | None, optional
        Forwarded to ``torch.compile`` when ``compile_model=True``.
    train : bool | None, optional
        Whether AIMNet2Calculator should keep the model trainable. Defaults
        to the wrapped module's current training mode.

    Attributes
    ----------
    model_config : ModelConfig
        Configuration with capability and runtime fields.
    model : nn.Module
        The underlying AIMNet2 model. If you want your model
        to be compiled, wrap with ``torch.compile(model, **kwargs)``
        before passing here.
    """

    model: nn.Module

    def __init__(
        self,
        model: nn.Module,
        *,
        compile_model: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
        train: bool | None = None,
    ) -> None:
        from aimnet.calculators import AIMNet2Calculator

        super().__init__()
        self.model = model
        calculator_train = model.training if train is None else train

        # Build a calculator for its pad/unpad utilities and its own
        # kernel-aware ``torch.compile`` of the module forward (``compile_model``).
        self._calculator = AIMNet2Calculator(
            model=model,
            device=str(next(model.parameters()).device),
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=compile_model,
            compile_kwargs=compile_kwargs,
            train=calculator_train,
        )

        # Detect NSE (Neutral Spin Equilibrated) models.
        raw_model = model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        self._is_nse = getattr(raw_model, "num_charge_channels", 1) == 2
        if self._is_nse:
            if "spin_charges" not in self._calculator.keys_out:
                self._calculator.keys_out = [*self._calculator.keys_out, "spin_charges"]

        # Extract cutoff from the loaded model.
        self._cutoff = self._extract_cutoff(raw_model)

        # Build the model config with external neighbor list.
        outputs = {"energy", "forces", "stress", "charges"}
        if self._is_nse:
            outputs.add("spin_charges")

        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset({"charge"}),
            optional_inputs=frozenset({"cell", "mult"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self._cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=False,
                # max_neighbors left as None — NeighborListHook will
                # auto-estimate via estimate_max_neighbors(cutoff).
            ),
            active_outputs={"energy", "forces", "charges"},
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
        compile_model: bool = False,
        **compile_kwargs: Any,
    ) -> "AIMNet2Wrapper":
        """Load an AIMNet2 model from a checkpoint and return a wrapped instance.

        Uses ``AIMNet2Calculator`` to resolve and load the checkpoint, then
        extracts the raw ``nn.Module`` and wraps it.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to an AIMNet2 checkpoint file, or a model alias recognized by
            ``AIMNet2Calculator`` (e.g. ``"aimnet2"``).
        device : torch.device | str, optional
            Target device. Defaults to ``"cpu"``.
        compile_model : bool, optional
            ``torch.compile`` the AIMNet2 model for single-process inference.
            Distributed compilation is a separate switch,
            ``DistributedModel(..., compile=True)``.
        **compile_kwargs
            Forwarded to ``torch.compile`` when ``compile_model=True``.

        Returns
        -------
        AIMNet2Wrapper
            The wrapped, fp32 model on *device*.
        """
        from aimnet.calculators import AIMNet2Calculator

        # Resolve + load the checkpoint only; the raw module is extracted and
        # this calculator discarded, so it never compiles. The wrapper's own
        # calculator performs the compile (via ``compile_model``).
        train = not compile_model
        calc = AIMNet2Calculator(
            model=str(checkpoint_path),
            device=str(device),
            needs_coulomb=False,
            needs_dispersion=False,
            compile_model=False,
            train=False,
        )
        raw_model = calc.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        # AIMNet2 runs in float32 only: its AEV kernel rejects non-fp32 input,
        # so enforce fp32 on the parameters here.
        raw_model = raw_model.float()
        # The throwaway loader calculator above was built with ``train=False``,
        # which froze (``requires_grad_(False)``) the parameters of the shared
        # module we just extracted. AIMNet2Calculator only *disables* grad for
        # ``train=False`` and never re-enables it for ``train=True``, so a
        # checkpoint loaded for training (``train=not compile_model``) would
        # otherwise come back with every parameter frozen and be impossible to
        # fine-tune. Restore the requested grad state before handing the module
        # to the wrapper's own calculator.
        if train:
            for param in raw_model.parameters():
                param.requires_grad_(True)
        return cls(
            raw_model,
            compile_model=compile_model,
            compile_kwargs=dict(compile_kwargs) if compile_kwargs else None,
            train=train,
        )

    @staticmethod
    def _extract_cutoff(raw_model: nn.Module) -> float:
        """Extract the AEV interaction cutoff from the loaded model."""
        aev = getattr(raw_model, "aev", None)
        if aev is None:
            return 5.0  # default AIMNet2 cutoff
        rc_s = getattr(aev, "rc_s", None)
        rc_v = getattr(aev, "rc_v", None)
        values = [float(v) for v in (rc_s, rc_v) if v is not None]
        return max(values) if values else 5.0

    # ------------------------------------------------------------------
    # Distributed hooks
    # ------------------------------------------------------------------

    @property
    def distribution_spec(self) -> Any:
        """MLIPSpec describing AIMNet2 under domain decomposition.

        Returns
        -------
        Any
            The halo ``MLIPSpec`` (owned+ghost local-neighbor storage). The
            conv kernel and Coulomb heads refresh their ghost rows each layer,
            and the per-system sum counts owned atoms only. Whether the
            distributed forward is compiled is decided by
            ``DistributedModel(..., compile=True)``.
        """
        return _aimnet2_halo_spec()

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """AIM-feature embedding shapes produced by this model.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Maps ``"node_embeddings"`` to its per-node feature shape
            ``(aim_dim,)``, read from the model's AEV output size.
        """
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        aim_dim = 256
        aev = getattr(raw_model, "aev", None)
        if aev is not None:
            output_size = getattr(aev, "output_size", None)
            if output_size is not None:
                aim_dim = int(output_size)
        return {"node_embeddings": (aim_dim,)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute AIM-feature node embeddings and attach them to *data*.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; an ``AtomicData`` is promoted to a single-graph
            ``Batch``.
        **kwargs
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        AtomicData | Batch
            *data*, with ``node_embeddings`` ``[N, aim_dim]`` written in place
            when the model exposes AIM features.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        model_input = self.adapt_input(data, **kwargs)
        n_real = data.num_nodes

        with torch.no_grad():
            raw_output = self._calculator.model(model_input)

        if "aim" in raw_output:
            data.add_key("node_embeddings", [raw_output["aim"][:n_real]], level="node")
        return data

    # ------------------------------------------------------------------
    # adapt_input / adapt_output
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Build the flat input dict expected by ``AIMNet2.forward``.

        Appends a single padding atom and converts the external neighbor matrix
        to AIMNet2's ``nbmat`` layout. Enables gradients on ``positions`` when an
        autograd output is active.

        Parameters
        ----------
        data : AtomicData | Batch
            The input system; an ``AtomicData`` is promoted to a single-graph
            ``Batch``. Requires ``neighbor_matrix`` (from NeighborListHook).
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        dict[str, Any]
            AIMNet2 inputs with a trailing padding atom (index ``N``):
            ``coord`` ``[N+1, 3]``, ``numbers`` ``[N+1]`` (pad row = 0),
            ``mol_idx`` ``[N+1]`` (sorted ascending), ``nbmat`` ``[N+1, K]``
            (unused slots = ``N``), ``charge`` ``[n_systems]``, and optional
            ``cell`` / ``shifts`` / ``mult``.

        Notes
        -----
        Does not call ``super().adapt_input()``: AIMNet2 uses its own key
        conventions (``coord`` / ``numbers`` / ``nbmat``).
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        # Enable grad on positions if any autograd output is active.
        if self.model_config.autograd_outputs & self.model_config.active_outputs:
            data.positions.requires_grad_(True)

        N = data.num_nodes
        device = data.positions.device

        # Core per-atom fields (pre-padding).
        coord = data.positions.to(torch.float32)
        numbers = data.atomic_numbers.to(torch.long)
        mol_idx = data.batch_idx.to(torch.long)

        # Append a padding atom (position 0, Z=0) so AIMNet2's "last atom is
        # padding" convention holds; it joins the last (highest-id) system.
        pad_pos = torch.zeros(1, 3, dtype=coord.dtype, device=device)
        pad_z = torch.zeros(1, dtype=torch.long, device=device)
        pad_mol = torch.full((1,), data.num_graphs - 1, dtype=torch.long, device=device)
        coord = torch.cat([coord, pad_pos], dim=0)
        numbers = torch.cat([numbers, pad_z], dim=0)
        mol_idx = torch.cat([mol_idx, pad_mol], dim=0)

        # Charge (and optional NSE multiplicity).
        charge = getattr(data, "charge", None)
        if charge is None:
            charge = torch.zeros(data.num_graphs, dtype=torch.float32, device=device)
        if charge.ndim == 0:
            charge = charge.unsqueeze(0)
        elif charge.ndim > 1:
            charge = charge.squeeze(-1)

        result: dict[str, torch.Tensor] = {
            "coord": coord,
            "numbers": numbers,
            "mol_idx": mol_idx,
            "charge": charge.to(torch.float32),
        }

        # Optional PBC cell.
        cell = getattr(data, "cell", None)
        if cell is not None:
            # AIMNet2 expects one cell matrix per system, even for single systems.
            if cell.ndim == 2:
                cell = cell.unsqueeze(0)
            result["cell"] = cell.to(torch.float32)

        # NSE multiplicity.
        if self._is_nse:
            mult = getattr(data, "mult", None)
            if mult is not None:
                result["mult"] = mult

        # Neighbor matrix to AIMNet2 nbmat, with a self-referencing padding row
        # appended. Unused slots hold the sentinel ``N`` so calc_masks masks them.
        nbmat_pad_value = N
        neighbor_matrix = getattr(data, "neighbor_matrix", None)
        if neighbor_matrix is not None:
            nbmat = neighbor_matrix.to(torch.long)
            K = nbmat.shape[1]
            padding_row = torch.full(
                (1, K), nbmat_pad_value, dtype=torch.long, device=device
            )
            result["nbmat"] = torch.cat([nbmat, padding_row], dim=0)

            neighbor_matrix_shifts = getattr(data, "neighbor_matrix_shifts", None)
            if neighbor_matrix_shifts is not None:
                # Pad at fp32 explicitly: ``torch.cat`` auto-promotes mismatched
                # operands, and any fp64 shifts would fault the fp32-only AEV kernel.
                shifts_padding = torch.zeros(
                    1, K, 3, dtype=torch.float32, device=device
                )
                result["shifts"] = torch.cat(
                    [neighbor_matrix_shifts.to(torch.float32), shifts_padding],
                    dim=0,
                )

        return result

    def adapt_output(
        self, model_output: dict[str, Any], data: AtomicData | Batch
    ) -> ModelOutputs:
        """Map AIMNet2 outputs to nvalchemi standard keys.

        Per-atom direct outputs (``charges`` / ``spin_charges``) carry the
        padding-atom row appended by :meth:`adapt_input` (plus any padding rows
        added under compiled DD); they are sliced back to ``data.num_nodes``.
        ``energy`` (per-system) and ``forces`` (autograd over real positions)
        need no strip.

        Parameters
        ----------
        model_output : dict[str, Any]
            Raw outputs from the AIMNet2 forward pass.
        data : AtomicData | Batch
            The input system the outputs were computed for.

        Returns
        -------
        ModelOutputs
            Standardized outputs: ``energy`` ``[n_systems, 1]`` plus any active
            ``forces`` / ``stress`` / ``charges`` / ``spin_charges``.
        """
        n_real = data.num_nodes
        output: ModelOutputs = OrderedDict()

        def _strip(t: Any) -> Any:
            if t is not None and hasattr(t, "shape") and t.shape[0] > n_real:
                return t[:n_real]
            return t

        energy = model_output.get("energy")
        if energy is not None:
            output["energy"] = energy.unsqueeze(-1) if energy.ndim == 1 else energy

        if "forces" in self.model_config.active_outputs and "forces" in model_output:
            output["forces"] = model_output["forces"]
        if "stress" in self.model_config.active_outputs and "stress" in model_output:
            output["stress"] = model_output["stress"]
        # Guard on not-None so the energy-only autograd path (which passes
        # just {energy, forces}) never injects a ``charges=None``.
        if (
            "charges" in self.model_config.active_outputs
            and model_output.get("charges") is not None
        ):
            output["charges"] = _strip(model_output["charges"])
        if (
            "spin_charges" in self.model_config.active_outputs
            and model_output.get("spin_charges") is not None
        ):
            output["spin_charges"] = _strip(model_output["spin_charges"])

        return output

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the AIMNet2 model for the active outputs.

        Pure and distribution-agnostic: computes exactly what
        ``model_config.active_outputs`` requests. Energy is the primitive output
        (summed per-system, over owned atoms plus an all-reduce under domain
        decomposition); forces and stresses are derived via autograd.

        For stresses, the affine strain trick from
        :func:`~nvalchemi.models._utils.prepare_strain` scales positions and cell
        through a displacement tensor so ``dE/d(displacement)`` gives the strain.

        In a pipeline with ``use_autograd=True``, the pipeline handles
        derivative computation externally — it strips forces/stresses
        from ``active_outputs`` so this method only computes energy.

        Parameters
        ----------
        data : AtomicData | Batch
            Input batch with positions, atomic numbers, charge, and
            ``neighbor_matrix`` (from NeighborListHook).
        **kwargs
            Forwarded to :meth:`adapt_input`.

        Returns
        -------
        ModelOutputs
            The standardized outputs for the active set.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        compute_forces = "forces" in (
            self.model_config.active_outputs & self.model_config.outputs
        )
        compute_stresses = "stress" in (
            self.model_config.active_outputs & self.model_config.outputs
        )

        # AIMNet2's kernels require fp32. Cast before ``prepare_strain`` so the
        # scaled cell does not inherit an fp64 dtype the AEV kernel rejects.
        if data.positions.dtype != torch.float32:
            data["positions"] = data.positions.to(torch.float32)
        if (
            hasattr(data, "cell")
            and data.cell is not None
            and data.cell.dtype != torch.float32
        ):
            data["cell"] = data.cell.to(torch.float32)

        # Set up affine strain BEFORE adapt_input so the scaled positions
        # flow through the model forward pass.
        displacement = None
        orig_positions = None
        orig_cell = None
        if compute_stresses and hasattr(data, "cell") and data.cell is not None:
            scaled_pos, scaled_cell, displacement = prepare_strain(
                data.positions, data.cell, data.batch_idx
            )
            orig_positions = data.positions
            orig_cell = data.cell
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        model_input = self.adapt_input(data, **kwargs)
        raw_output = self._calculator.model(model_input)

        # ``energy`` is already per-system; the per-atom charge outputs carry the
        # appended pad atom, which adapt_output slices back to the real count.
        result: dict[str, Any] = {"energy": raw_output["energy"]}
        if "charges" in self.model_config.active_outputs:
            result["charges"] = raw_output.get("charges")
        if "spin_charges" in self.model_config.active_outputs:
            result["spin_charges"] = raw_output.get("spin_charges")

        # Autograd-derived forces/stresses. Under domain decomposition the
        # framework computes forces externally (FRAMEWORK_FROM_GLOBAL_ENERGY),
        # so this eager autograd block only runs in single-process / pipeline
        # use_autograd=False paths.
        if compute_forces and compute_stresses and displacement is not None:
            energy = result["energy"]
            forces, stress = autograd_forces_and_stresses(
                energy,
                data.positions,
                displacement,
                orig_cell,
                data.num_graphs,
                training=self.training,
            )
            result["forces"] = forces
            result["stress"] = stress
        elif compute_forces:
            energy = result["energy"]
            forces = -torch.autograd.grad(
                energy,
                data.positions,
                grad_outputs=torch.ones_like(energy),
                create_graph=self.training,
                retain_graph=compute_stresses or self.training,
            )[0]
            result["forces"] = forces
        elif compute_stresses and displacement is not None:
            result["stress"] = autograd_stresses(
                result["energy"],
                displacement,
                orig_cell,
                data.num_graphs,
                training=self.training,
            )

        # Restore original positions/cell if strain was applied.
        if orig_positions is not None:
            data["positions"] = orig_positions
            data["cell"] = orig_cell

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Serialize the underlying AIMNet2 model without the wrapper.

        Parameters
        ----------
        path : Path
            Output path.
        as_state_dict : bool, optional
            If ``True``, save only the ``state_dict``; otherwise pickle the full
            model object. Defaults to ``False``.
        """
        raw_model = self.model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        if as_state_dict:
            torch.save(raw_model.state_dict(), path)
        else:
            torch.save(raw_model, path)


# ======================================================================
# Distributed-aware replacements for ``aimnet`` helpers, declared as adapters
# on the halo spec so stock ``AIMNet2.forward`` runs unchanged under domain
# decomposition (per-system sum over owned atoms + ghost refresh before the
# local kernels).
# ======================================================================


def _distributed_coulomb_forward(
    original: Any, head_self: Any, data: dict[str, Any]
) -> Any:
    """Adapter for AIMNet2's Coulomb heads (LRCoulomb / SRCoulomb).

    Refreshes the input charges so boundary owners see up-to-date ghost values,
    then runs the stock head. Serves both eager and compiled.

    Parameters
    ----------
    original : Any
        The unbound stock ``forward`` being wrapped.
    head_self : Any
        The Coulomb head instance.
    data : dict[str, Any]
        The AIMNet2 input dict.

    Returns
    -------
    Any
        The stock head output computed over refreshed local tensors.
    """
    key_in = getattr(head_self, "key_in", "charges")
    data_p = localize(data)
    q = data_p.get(key_in)
    ctx = current_dd_context()
    if q is not None and (ctx.compiling or q.shape[0] >= ctx.n_padded):
        data_p[key_in] = refresh_neighbors(q)
    return original(head_self, data_p)


def _distributed_mol_sum(x: Any, data: dict[str, Any]) -> Any:
    """Halo-aware replacement for :func:`aimnet.nbops.mol_sum`.

    Sums each system's owned per-atom contributions, then all-reduces across
    ranks so every rank holds the global per-system value.

    Parameters
    ----------
    x : Any
        Per-atom contributions to reduce.
    data : dict[str, Any]
        The AIMNet2 input dict, providing ``mol_idx``.

    Returns
    -------
    Any
        A plain ``[n_systems_global, *F]`` tensor replicated on every rank.
    """
    n_sys = int(current_dd_context().n_systems_global)
    return system_sum(to_local(x), data["mol_idx"], n_sys, scope=Scope.OWNED)


def _distributed_conv_sv_forward(
    original: Any, conv_self: Any, data: dict[str, Any], a: Any
) -> Any:
    """Adapter for ``aimnet.modules.aev.ConvSV.forward``.

    Refreshes the ghost rows of the conv input, then runs the stock kernel on
    the local neighbor matrix. Serves both eager and compiled.

    Parameters
    ----------
    original : Any
        The unbound stock ``forward`` being wrapped.
    conv_self : Any
        The ``ConvSV`` instance.
    data : dict[str, Any]
        The AIMNet2 input dict.
    a : Any
        The per-atom feature tensor fed to the conv kernel.

    Returns
    -------
    Any
        The stock conv output computed over refreshed local tensors.
    """
    return original(conv_self, localize(data), refresh_neighbors(to_local(a)))
