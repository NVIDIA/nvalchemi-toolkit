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

"""UMA (fairchem-core) model wrapper.

Wraps a fairchem ``MLIPPredictUnit`` as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model.

UMA is multi-task — a single backbone checkpoint (``uma-s-1p1``,
``uma-s-1p2``, ``uma-m-1p1``) ships with heads for OMol25 (molecules),
OMat24 (crystals), OC20 (catalysis), ODAC23 (direct air capture), and
OMC (molecular crystals). Task selection happens at checkpoint load
time via ``task_name=`` and is baked into the wrapper — matching the
"one wrapper, one model" pattern used by :class:`~nvalchemi.models.mace.MACEWrapper`.

Usage
-----
::

    from nvalchemi.models.uma import UMAWrapper

    # OMol molecular potential
    mol_wrapper = UMAWrapper.from_checkpoint(
        "uma-s-1p1", task_name="omol", device="cuda"
    )

    # OMat bulk-crystal potential (same checkpoint, different head)
    mat_wrapper = UMAWrapper.from_checkpoint(
        "uma-s-1p1", task_name="omat", device="cuda"
    )

Notes
-----
* Energy is the primitive differentiable output; forces and (for
  periodic tasks) stress are derived via autograd.
* OMol requires a total-charge field; the wrapper reads ``charge`` off
  the input ``AtomicData`` / ``Batch`` and defaults to 0 if absent.
  Spin multiplicity (``spin`` on the batch) defaults to 1 for OMol and
  0 for periodic tasks.
* ``active_outputs`` is task-aware: ``{"energy", "forces"}`` for
  molecular tasks, ``{"energy", "forces", "stress"}`` for periodic.

``torch.compile``
-----------------
Unlike :class:`~nvalchemi.models.mace.MACEWrapper` /
:class:`~nvalchemi.models.aimnet2.AIMNet2Wrapper`, the UMA wrapper does
**not** expose a ``compile_model`` flag. fairchem owns compilation
internally: it is a field on ``fairchem.core.units.mlip_unit.api.inference.InferenceSettings``
(``compile: bool``), not a ``torch.compile(model)`` call. Reach it
through :meth:`from_checkpoint`'s ``inference_settings`` argument:

* ``inference_settings="turbo"`` — fairchem's optimized preset, which
  sets ``compile=True`` **and** ``tf32=True`` / ``merge_mole=True`` /
  ``activation_checkpointing=False``. Best for long simulations with
  fixed atomic composition; it changes numerics relative to ``"default"``.
* For a pure compile toggle, pass an ``InferenceSettings`` instance with
  ``compile=True`` and the other fields left at their ``"default"``
  values::

      from fairchem.core.units.mlip_unit.api.inference import (
          InferenceSettings,
      )

      wrapper = UMAWrapper.from_checkpoint(
          "uma-s-1p1",
          task_name="omat",
          device="cuda",
          inference_settings=InferenceSettings(compile=True),
      )
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

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
    from fairchem.core.units.mlip_unit import MLIPPredictUnit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

__all__ = ["UMATask", "UMAWrapper"]


# UMA task heads. The ``Literal`` is the single source of truth for valid task
# names; the membership set is derived from it.
UMATask = Literal["omol", "omat", "oc20", "odac", "omc"]
_UMA_TASKS: frozenset[str] = frozenset(get_args(UMATask))

# Tasks that declare PBC (stress supported). ``omol`` is molecular — no stress.
_PBC_TASKS: frozenset[str] = frozenset({"omat", "oc20", "odac", "omc"})


@OptionalDependency.UMA.require
class UMAWrapper(nn.Module, BaseModelMixin):
    """Wrapper for fairchem's UMA (Universal Models for Atoms).

    Wraps a :class:`fairchem.core.units.mlip_unit.MLIPPredictUnit` — the
    level at which energy / forces / stress are computed by fairchem
    (the raw backbone only produces node embeddings). Task is fixed at
    construction; ``active_outputs`` reflects what that task supports.

    Parameters
    ----------
    predict_unit : fairchem.core.units.mlip_unit.MLIPPredictUnit
        Pre-loaded UMA prediction unit. Use :meth:`from_checkpoint` for
        the typical construction path that resolves a registered
        checkpoint name and downloads via HuggingFace Hub.
    task_name : str
        UMA task: one of ``_UMA_TASKS``. Determines which per-task head
        in the multi-task model is used and which inputs (charge, spin)
        must be populated.
    train : bool
        ``False`` (default) freezes all weights for inference (lossless for
        autograd forces); ``True`` keeps fairchem's trainable/frozen split
        so weights stay exposed for fine-tuning.

    Attributes
    ----------
    model_config : ModelConfig
        Task-dependent outputs + autograd + neighbor config.
    task_name : str
        The UMA task this wrapper is pinned to.
    """

    def __init__(
        self,
        predict_unit: "MLIPPredictUnit",
        task_name: UMATask = "omol",
        train: bool = False,
    ) -> None:
        super().__init__()
        if task_name not in _UMA_TASKS:
            raise ValueError(
                f"UMAWrapper task_name {task_name!r} must be one of {get_args(UMATask)}"
            )

        # Validate that the checkpoint actually supports this task —
        # surface the error at construction, not on first forward.
        available = list(predict_unit.dataset_to_tasks.keys())
        if task_name not in available:
            raise ValueError(
                f"Checkpoint does not ship a '{task_name}' head. "
                f"Available: {available}. Load a different checkpoint "
                f"or pick one of the available tasks."
            )

        self.predict_unit = predict_unit
        self.task_name = task_name
        self._is_pbc_task = task_name in _PBC_TASKS
        self._cutoff = self._extract_cutoff()

        # Under turbo/compile, the first forward must feed CPU input to dodge a
        # fairchem lazy-merge device bug; this one-shot flag clears after that
        # forward (tracked here rather than via fairchem's private init flag).
        _settings = getattr(predict_unit, "inference_settings", None)
        self._cpu_route_first_forward = _settings is not None and bool(
            getattr(_settings, "merge_mole", False)
            or getattr(_settings, "compile", False)
        )

        # Task-dependent output set. Energy + forces are universal;
        # stress only makes sense for periodic tasks.
        outputs: set[str] = {"energy", "forces"}
        autograd_outputs: set[str] = {"forces"}
        active_outputs: set[str] = {"energy", "forces"}
        if self._is_pbc_task:
            outputs.add("stress")
            autograd_outputs.add("stress")
            active_outputs.add("stress")

        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            autograd_outputs=frozenset(autograd_outputs),
            autograd_inputs=frozenset({"positions"}),
            # All optional (not required for OMol) to keep one config shape
            # across tasks — the adapter fills charge/spin defaults.
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell", "charge", "spin", "tags"}),
            supports_pbc=True,
            needs_pbc=self._is_pbc_task,
            neighbor_config=NeighborConfig(
                cutoff=self._cutoff,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
            active_outputs=active_outputs,
        )

        # Inference (train=False): freeze all weights — conservative forces
        # come from autograd w.r.t. positions, so this is lossless and avoids
        # building a weight-grad graph each forward. Training: leave fairchem's
        # loaded trainable/frozen split intact so weights stay exposed.
        self._train = train
        if not train:
            for p in self.predict_unit.model.parameters():
                p.requires_grad_(False)
        self.train(train)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        name_or_path: str | Path,
        task_name: UMATask = "omol",
        device: str | torch.device = "cpu",
        inference_settings: "InferenceSettings | str" = "default",
        overrides: dict | None = None,
        train: bool = False,
    ) -> "UMAWrapper":
        """Resolve and load a UMA checkpoint.

        Accepts either a registered model name
        (``"uma-s-1p1"`` / ``"uma-s-1p2"`` / ``"uma-m-1p1"``) or a
        local filesystem path to a ``.pt`` file. The multi-task
        checkpoints ship all five task heads; ``task_name`` picks which
        one the wrapper exposes via :attr:`model_config.active_outputs`.

        Parameters
        ----------
        name_or_path : str | Path
            Registered model name (see
            ``fairchem.core.calculate.pretrained_mlip.available_models``)
            or a local file path.
        task_name : str
            One of ``omol``, ``omat``, ``oc20``, ``odac``, ``omc``.
            Defaults to ``omol`` (molecular chemistry) — the most
            common entry point; override explicitly for crystals /
            catalysis.
        device : str | torch.device
            Target device for inference. Defaults to ``"cpu"``.
        inference_settings : InferenceSettings | str
            fairchem inference configuration. Either a preset name
            (``"default"`` or ``"turbo"``) or a
            ``fairchem.core.units.mlip_unit.api.inference.InferenceSettings``
            instance. ``torch.compile`` is reached through this argument
            — see the module docstring's *torch.compile* section.
            Defaults to ``"default"``.
        overrides : dict | None
            Optional overrides forwarded to fairchem's inference-settings
            builder.
        train : bool
            If ``False`` (default), freeze all weights for inference —
            lossless, since conservative forces come from autograd on
            positions. If ``True``, keep fairchem's loaded trainable/frozen
            split so weights remain exposed for fine-tuning. Note: the
            ``forward`` path goes through fairchem's inference ``predict``
            (eval mode, detached forces); gradient-based training requires a
            separate path through the raw model.
        """
        import os as _os

        from fairchem.core.calculate import pretrained_mlip  # noqa: PLC0415
        from fairchem.core.units.mlip_unit import load_predict_unit  # noqa: PLC0415

        if isinstance(device, torch.device):
            device = device.type

        name_str = str(name_or_path)
        if name_str in pretrained_mlip.available_models:
            predict_unit = pretrained_mlip.get_predict_unit(
                name_str,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        elif _os.path.isfile(name_str):
            predict_unit = load_predict_unit(
                name_str,
                inference_settings=inference_settings,
                overrides=overrides,
                device=device,
            )
        else:
            raise ValueError(
                f"{name_str!r} is neither a registered model name nor a "
                f"local file path. Known names: "
                f"{sorted(pretrained_mlip.available_models)}"
            )
        return cls(predict_unit, task_name=task_name, train=train)

    def _extract_cutoff(self) -> float:
        """Pull the radial cutoff from the loaded backbone.

        UMA's ASE calculator uses a 6.0 Å radius when external graph
        generation is enabled; the inference-settings default is 6.0.
        Prefer the backbone's own ``r_max`` attribute when present.
        """
        backbone = getattr(self.predict_unit.model.module, "backbone", None)
        if backbone is not None:
            r_max = getattr(backbone, "r_max", None)
            if r_max is not None:
                return float(r_max.item() if hasattr(r_max, "item") else r_max)
            cutoff = getattr(backbone, "cutoff", None)
            if cutoff is not None:
                return float(cutoff)
        return 6.0

    # ------------------------------------------------------------------
    # BaseModelMixin contract
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Radial cutoff (Å) for neighbor-list construction."""
        return self._cutoff

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Shape of the per-node backbone embedding.

        eSCN-MD's backbone produces ``[N, (lmax+1)², sphere_channels]``.
        eSEN variants share the same layout. Readable off the loaded
        backbone's ``sph_feature_size`` / ``sphere_channels`` attrs.
        """
        backbone = getattr(self.predict_unit.model.module, "backbone", None)
        if backbone is None:
            raise RuntimeError(
                "predict_unit.model.module has no .backbone attribute — "
                "embedding_shapes cannot be inferred."
            )
        sph = int(getattr(backbone, "sph_feature_size"))
        ch = int(getattr(backbone, "sphere_channels"))
        return {"node_embeddings": (sph, ch)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Run the backbone only and attach node embeddings.

        UMA/eSEN backbones return ``{"embedding": [N, sph, ch], "batch": [N]}``;
        we attach the embedding as a node property so pipelines can
        consume it without re-running the heads.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        fc_data = self.adapt_input(data, **kwargs)
        fc_data = fc_data.to(next(self.predict_unit.model.parameters()).device)
        backbone = self.predict_unit.model.module.backbone
        with torch.no_grad():
            out = backbone(fc_data)
        if "embedding" in out:
            data.add_key("node_embeddings", [out["embedding"]], level="node")
        return data

    # ------------------------------------------------------------------
    # adapt_input / adapt_output
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> Any:
        """Convert an nvalchemi ``AtomicData`` / ``Batch`` to a fairchem
        ``AtomicData`` — tensor-native, no ASE round trip.

        Tensors stay on ``data.positions.device`` throughout, preserving
        GPU residency and autograd. ``edge_index`` is left empty (shape
        ``(2, 0)``); fairchem's ``MLIPPredictUnit`` rebuilds the graph
        internally — this matches the default ``FAIRChemCalculator``
        path (``r_edges=False``), so outputs are equivalent.

        Charge/spin defaults follow the ASE-calculator convention:
        per-system LongTensors, 0 for periodic tasks, 0 for OMol unless
        the caller provides them on the batch.
        """
        from fairchem.core.datasets.atomic_data import (  # noqa: PLC0415
            AtomicData as FCAtomicData,
        )

        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        device = data.positions.device
        target_dtype = self.predict_unit.inference_settings.base_precision_dtype

        pos = data.positions.to(target_dtype)
        atomic_numbers = data.atomic_numbers.to(torch.long)
        batch_idx = data.batch_idx.to(torch.long)
        n_systems = int(data.num_graphs)
        total_atoms = pos.shape[0]

        # Per-system atom counts — precomputed on the batch.
        natoms = data.num_nodes_per_graph.to(torch.long)

        # Batch already shapes cell/pbc as (B, 3, 3) / (B, 3) — just cast.
        cell = getattr(data, "cell", None)
        if cell is None:
            cell = torch.zeros(n_systems, 3, 3, dtype=target_dtype, device=device)
        else:
            cell = cell.to(target_dtype)

        pbc = getattr(data, "pbc", None)
        if pbc is None:
            pbc = torch.full(
                (n_systems, 3), self._is_pbc_task, dtype=torch.bool, device=device
            )
        else:
            pbc = pbc.to(torch.bool)

        # charge/spin: the typed AtomicData fields are float (B, 1); fairchem
        # wants per-system long (B,), so flatten + cast (also handles raw (B,)).
        charge = getattr(data, "charge", None)
        if charge is None:
            charge = torch.zeros(n_systems, dtype=torch.long, device=device)
        else:
            charge = charge.to(torch.long).reshape(n_systems)

        spin = getattr(data, "spin", None)
        if spin is None:
            # spin is the multiplicity (only used by the OMol head); default to
            # the closed-shell singlet (1), matching FAIRChemCalculator. Open-
            # shell systems must set it explicitly. Periodic heads ignore spin.
            spin_default = 0 if self._is_pbc_task else 1
            spin = torch.full(
                (n_systems,), spin_default, dtype=torch.long, device=device
            )
        else:
            spin = spin.to(torch.long).reshape(n_systems)

        # Empty edges — predict_unit rebuilds the graph internally.
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        cell_offsets = torch.empty((0, 3), dtype=target_dtype, device=device)
        nedges = torch.zeros(n_systems, dtype=torch.long, device=device)

        fixed = torch.zeros(total_atoms, dtype=torch.long, device=device)

        # fairchem tags = nvalchemi atom_categories (sub-surface/surface/
        # adsorbate for OC20/ODAC); defaults to zeros when unset.
        cats = getattr(data, "atom_categories", None)
        if cats is None:
            tags = torch.zeros(total_atoms, dtype=torch.long, device=device)
        else:
            tags = cats.to(torch.long).reshape(total_atoms)

        return FCAtomicData(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            batch=batch_idx,
            sid=[""] * n_systems,
            dataset=[self.task_name] * n_systems,
        )

    def adapt_output(
        self, raw: dict, data: AtomicData | Batch | None = None
    ) -> ModelOutputs:
        """Map fairchem's prediction dict to nvalchemi's output keys.

        Fairchem returns tensors keyed by ``"energy"`` (per-system),
        ``"forces"`` (per-atom), and optionally ``"stress"``
        (per-system). The shapes already match our expectations.
        """
        out: dict[str, torch.Tensor] = {}
        active = self.model_config.active_outputs

        if "energy" in active:
            energy = raw["energy"]
            # Ensure per-system 2D shape (n_graphs, 1) — matches LJ/MACE.
            if energy.dim() == 1:
                energy = energy.unsqueeze(-1)
            out["energy"] = energy
        if "forces" in active:
            out["forces"] = raw["forces"]
        if "stress" in active and "stress" in raw:
            stress = raw["stress"]
            if stress.dim() == 2 and stress.shape[-1] == 9:
                # fairchem sometimes flattens stress; reshape to (n, 3, 3).
                stress = stress.reshape(-1, 3, 3)
            out["stress"] = stress

        return out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the UMA predict unit on *data*.

        Pipeline: ``adapt_input`` → ``MLIPPredictUnit.predict`` →
        ``adapt_output``. ``predict`` handles internal graph generation,
        task routing, and element-reference undoing — we don't need to
        compute neighbors ourselves at this layer.

        Turbo workaround: fairchem applies ``torch.compile`` + MoLE merge
        lazily on the first ``predict`` (``MLIPPredictUnit._lazy_init``).
        With ``merge_mole=True`` that merge leaves the charge/spin embeddings
        on CPU when the first batch is GPU-resident, crashing the forward.
        fairchem's own ASE calculator dodges this by feeding CPU data on that
        first call, so we route only the first wrapper forward through CPU
        input; ``predict`` moves it back to the model device, so inference
        stays on-device and later forwards use the input's real device.
        """
        fc_data = self.adapt_input(data, **kwargs)

        # First turbo/compile forward only — see the docstring.
        if self._cpu_route_first_forward:
            fc_data = fc_data.to(torch.device("cpu"))
            self._cpu_route_first_forward = False

        raw = self.predict_unit.predict(fc_data, undo_element_references=True)
        return self.adapt_output(raw, data=data)
