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
"""Tests for UMAWrapper (fairchem-core predict-unit wrapper).

Organised in tiers:

* **Structural tests** (``Test*`` classes using ``_Mock*`` predict units)
  exercise ``adapt_input`` / ``adapt_output`` / forward composition,
  task-name validation, and model-config correctness — no checkpoint
  needed, fast, always run when ``fairchem-core`` is importable.
* **Distribution-spec tests** (``TestMLIPSpec``) assert the domain-
  decomposition halo policy and custom-op registration carried on the
  wrapper's ``distribution_spec`` — also mock-only, no checkpoint.
* **Checkpoint tests** load a real fairchem checkpoint (default
  ``uma-s-1p1``, override via ``NVALCHEMI_UMA_CKPT`` / ``NVALCHEMI_UMA_DEVICE``)
  and cover forward-equivalence vs ``FAIRChemCalculator``, charged-input
  response, NVE energy conservation (``@slow``), and the turbo /
  ``torch.compile`` device path (``@slow``, CUDA only). They skip
  cleanly when the gated checkpoint cannot be downloaded.
"""

from __future__ import annotations

import math
import os
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip(
    "fairchem.core", reason="fairchem-core not installed; skipping UMA tests"
)

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from fairchem.core.datasets.atomic_data import AtomicData as FCAtomicData  # noqa: E402

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph  # noqa: E402
from nvalchemi.dynamics.integrators.nve import NVE  # noqa: E402
from nvalchemi.models import uma as uma_module  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402
from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper  # noqa: E402
from nvalchemi.models.uma import (  # noqa: E402
    _UMA_TASKS,
    UMAWrapper,
    _complete_cell,
    _distributed_edgewise_gather,
    _distributed_partition_graph,
    _fold_into_cell,
    _resolve_inference_settings,
)

_CKPT = os.environ.get("NVALCHEMI_UMA_CKPT", "uma-s-1p1")
_DEVICE = os.environ.get(
    "NVALCHEMI_UMA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)


# ===========================================================================
# Structural tests — mock predict unit, no checkpoint
# ===========================================================================


class _MockInferenceSettings:
    base_precision_dtype = torch.float32
    external_graph_gen = False


class _MockBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r_max = torch.tensor(6.0)
        self.sph_feature_size = 16  # (lmax+1)² with lmax=3
        self.sphere_channels = 128
        # A trainable weight so the train/freeze flag is observable in tests.
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, data: FCAtomicData) -> dict:
        n = data.pos.shape[0]
        return {
            "embedding": torch.zeros(
                n,
                self.sph_feature_size,
                self.sphere_channels,
                dtype=data.pos.dtype,
                device=data.pos.device,
            ),
            "batch": data.batch,
        }


class _MockInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MockBackbone()


class _MockModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = _MockInnerModel()


class _MockPredictUnit:
    """Minimal stand-in for ``fairchem.core.units.mlip_unit.MLIPPredictUnit``.

    Records the ``FCAtomicData`` handed to ``predict`` so tests can
    inspect the tensor-native conversion. Returns a synthetic
    energy/forces dict matching the prediction schema.
    """

    def __init__(self, supported_tasks: list[str] | None = None) -> None:
        if supported_tasks is None:
            supported_tasks = list(_UMA_TASKS)
        self.dataset_to_tasks = {t: [] for t in supported_tasks}
        self.inference_settings = _MockInferenceSettings()
        self.model = _MockModelWrapper()
        self.last_data: FCAtomicData | None = None

    def predict(self, data: FCAtomicData, undo_element_references: bool = True) -> dict:
        self.last_data = data
        n_graphs = data.num_graphs
        n_atoms = data.pos.shape[0]
        # Differentiable energy = sum of per-atom position L2 norms,
        # grouped by graph. Lets autograd tests through (forces != 0).
        norms = data.pos.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()
        energy = torch.zeros(n_graphs, dtype=data.pos.dtype, device=data.pos.device)
        energy.scatter_add_(0, data.batch, norms)
        return {
            "energy": energy,
            "forces": torch.zeros(
                n_atoms, 3, dtype=data.pos.dtype, device=data.pos.device
            ),
        }


@pytest.fixture
def mock_pu() -> _MockPredictUnit:
    return _MockPredictUnit()


@pytest.fixture
def mock_omol(mock_pu) -> UMAWrapper:
    return UMAWrapper(mock_pu, task_name="omol")


@pytest.fixture
def mock_omat(mock_pu) -> UMAWrapper:
    return UMAWrapper(mock_pu, task_name="omat")


def _make_propane() -> AtomicData:
    """Propane C3H8 — 11 atoms, molecular (no PBC)."""
    positions = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [1.5260, 0.0000, 0.0000],
            [2.0330, 1.4360, 0.0000],
            [-0.5093, 1.0222, 0.0000],
            [-0.5093, -0.5111, 0.8853],
            [-0.5093, -0.5111, -0.8853],
            [2.0319, -0.5111, 0.8853],
            [2.0319, -0.5111, -0.8853],
            [3.1193, 1.4360, 0.0000],
            [1.6763, 1.9471, 0.8853],
            [1.6763, 1.9471, -0.8853],
        ],
        dtype=torch.float32,
    )
    numbers = torch.tensor([6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    return AtomicData(positions=positions, atomic_numbers=numbers)


def _make_periodic_cu() -> AtomicData:
    """Cubic Cu cell (1 atom per cell, a=3.615 Å) — minimal periodic system."""
    a = 3.615
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    numbers = torch.tensor([29], dtype=torch.long)
    cell = (torch.eye(3, dtype=torch.float32) * a).unsqueeze(0)
    pbc = torch.tensor([[True, True, True]])
    return AtomicData(positions=positions, atomic_numbers=numbers, cell=cell, pbc=pbc)


class TestConstruction:
    def test_invalid_task_name_raises(self, mock_pu):
        with pytest.raises(ValueError, match="task_name"):
            UMAWrapper(mock_pu, task_name="not_a_task")

    def test_unsupported_checkpoint_task_raises(self):
        # A predict unit that only ships the omol head.
        pu = _MockPredictUnit(supported_tasks=["omol"])
        with pytest.raises(ValueError, match="does not ship"):
            UMAWrapper(pu, task_name="omat")

    def test_task_stored(self, mock_omol):
        assert mock_omol.task_name == "omol"

    def test_cutoff_from_backbone(self, mock_omol):
        assert math.isclose(mock_omol.cutoff, 6.0, abs_tol=1e-6)

    def test_inference_freezes_weights(self, mock_omol):
        """train=False (default) freezes the underlying weights for inference."""
        params = list(mock_omol.predict_unit.model.parameters())
        assert params and all(not p.requires_grad for p in params)
        assert mock_omol.training is False

    def test_train_keeps_weights_trainable(self, mock_pu):
        """train=True leaves weights trainable/exposed for fine-tuning."""
        w = UMAWrapper(mock_pu, task_name="omol", train=True)
        assert any(p.requires_grad for p in w.predict_unit.model.parameters())
        assert w.training is True

    def test_from_checkpoint_forwards_default_preset_unchanged(self, mock_pu):
        from fairchem.core.calculate import pretrained_mlip

        with (
            patch.object(pretrained_mlip, "available_models", ["uma-test"]),
            patch.object(
                pretrained_mlip, "get_predict_unit", return_value=mock_pu
            ) as get_predict_unit,
        ):
            UMAWrapper.from_checkpoint("uma-test")

        get_predict_unit.assert_called_once_with(
            "uma-test",
            inference_settings="default",
            overrides=None,
            device="cpu",
        )


class TestModelConfig:
    def test_omol_active_outputs(self, mock_omol):
        active = mock_omol.model_config.active_outputs
        assert "energy" in active and "forces" in active
        assert "stress" not in active  # molecular — no stress

    def test_omat_active_outputs(self, mock_omat):
        active = mock_omat.model_config.active_outputs
        assert active >= {"energy", "forces", "stress"}

    def test_needs_pbc_by_task(self, mock_omol, mock_omat):
        assert mock_omol.model_config.needs_pbc is False
        assert mock_omat.model_config.needs_pbc is True

    def test_supports_pbc_always(self, mock_omol, mock_omat):
        assert mock_omol.model_config.supports_pbc is True
        assert mock_omat.model_config.supports_pbc is True

    def test_neighbor_config(self, mock_omol):
        nc = mock_omol.model_config.neighbor_config
        assert nc.cutoff == mock_omol.cutoff
        assert nc.format is NeighborListFormat.COO
        assert nc.half_list is False

    def test_autograd_forces(self, mock_omol):
        assert "forces" in mock_omol.model_config.autograd_outputs
        assert "positions" in mock_omol.model_config.autograd_inputs

    def test_autograd_stress_for_periodic(self, mock_omat):
        assert "stress" in mock_omat.model_config.autograd_outputs


class TestAdaptInput:
    def test_molecular_single_system(self, mock_omol):
        batch = Batch.from_data_list([_make_propane()])
        fc = mock_omol.adapt_input(batch)

        assert isinstance(fc, FCAtomicData)
        assert fc.pos.shape == (11, 3)
        assert fc.atomic_numbers.shape == (11,)
        assert fc.natoms.tolist() == [11]
        assert fc.cell.shape == (1, 3, 3)
        assert fc.pbc.shape == (1, 3)
        assert fc.pbc.any().item() is False  # omol — no PBC
        assert fc.edge_index.shape == (2, 0)
        assert fc.nedges.tolist() == [0]
        assert fc.charge.tolist() == [0]  # default for omol
        assert fc.spin.tolist() == [1]  # OMol default multiplicity = singlet
        assert fc.dataset == ["omol"]

    def test_periodic_single_system(self, mock_omat):
        batch = Batch.from_data_list([_make_periodic_cu()])
        fc = mock_omat.adapt_input(batch)
        # Periodic heads ignore spin; default stays 0.
        assert fc.spin.tolist() == [0]

        assert fc.pos.shape == (1, 3)
        assert fc.cell.shape == (1, 3, 3)
        assert fc.pbc.all().item() is True
        assert fc.dataset == ["omat"]

    def test_multi_system_batch(self, mock_omol):
        batch = Batch.from_data_list([_make_propane(), _make_propane()])
        fc = mock_omol.adapt_input(batch)

        assert fc.pos.shape == (22, 3)
        assert fc.natoms.tolist() == [11, 11]
        assert fc.batch.tolist() == [0] * 11 + [1] * 11
        assert len(fc.dataset) == 2
        assert fc.dataset == ["omol", "omol"]

    def test_accepts_atomicdata_directly(self, mock_omol):
        fc = mock_omol.adapt_input(_make_propane())
        assert fc.num_graphs == 1
        assert fc.pos.shape == (11, 3)

    def test_preserves_device(self, mock_pu):
        """adapt_input must keep tensors on data.positions.device."""
        w = UMAWrapper(mock_pu, task_name="omol")
        data = _make_propane()
        # CPU-only test; device preservation is the invariant.
        fc = w.adapt_input(data)
        assert fc.pos.device == data.positions.device
        assert fc.atomic_numbers.device == data.positions.device
        assert fc.batch.device == data.positions.device

    def test_preserves_gradient_flow(self, mock_omol):
        """positions with requires_grad should flow into the FC AtomicData."""
        data = _make_propane()
        data.positions.requires_grad_(True)
        fc = mock_omol.adapt_input(data)
        # Tensor-native: pos is the same storage (dtype conversion is
        # identity for matching dtypes), so requires_grad carries.
        assert fc.pos.requires_grad

    def test_target_dtype_cast(self, mock_pu):
        """When predict_unit declares fp32, input fp64 positions cast down."""
        mock_pu.inference_settings.base_precision_dtype = torch.float32
        w = UMAWrapper(mock_pu, task_name="omol")
        data = AtomicData(
            positions=torch.randn(5, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        )
        fc = w.adapt_input(data)
        assert fc.pos.dtype == torch.float32
        assert fc.cell.dtype == torch.float32

    def test_passes_explicit_charge_spin(self, mock_pu):
        w = UMAWrapper(mock_pu, task_name="omol")
        data = _make_propane()
        # Batch-level charge/spin should flow through.
        batch = Batch.from_data_list([data])
        batch.charge = torch.tensor([-1], dtype=torch.long)
        batch.spin = torch.tensor([2], dtype=torch.long)
        fc = w.adapt_input(batch)
        assert fc.charge.tolist() == [-1]
        assert fc.spin.tolist() == [2]

    def test_passes_tags(self, mock_pu):
        """Per-atom atom_categories (fairchem tags, e.g. OC20/ODAC) pass through."""
        w = UMAWrapper(mock_pu, task_name="oc20")
        data = AtomicData(
            positions=torch.zeros(4, 3),
            atomic_numbers=torch.tensor([1, 1, 1, 1], dtype=torch.long),
            atom_categories=torch.tensor([0, 1, 2, 1], dtype=torch.long),
        )
        fc = w.adapt_input(data)
        assert fc.tags.tolist() == [0, 1, 2, 1]

    def test_tags_default_zero(self, mock_omol):
        """Without atom_categories, the adapter fills tags with zeros."""
        fc = mock_omol.adapt_input(_make_propane())
        assert fc.tags.tolist() == [0] * 11


class TestCompleteCell:
    """``_complete_cell`` — the zero-vector completion ``adapt_input`` applies."""

    @pytest.mark.parametrize(
        "cell",
        [
            torch.diag(torch.tensor([3.0, 4.0, 0.0])),  # 2D
            torch.tensor([[0.0] * 3, [1.0, 2.0, 0.5], [0.0] * 3]),  # 1D, tilted
            torch.diag(torch.tensor([0.0, 0.0, 7.0])),  # 1D along z
            torch.zeros(3, 3),  # molecule
        ],
    )
    def test_matches_ase_up_to_orientation(self, cell):
        """Same kept vectors, orthonormal right-handed fill-ins, like ASE."""
        from ase.geometry import complete_cell

        out = _complete_cell(cell[None])[0]
        ref = torch.tensor(complete_cell(cell.numpy()), dtype=cell.dtype)
        kept = cell.any(dim=-1)
        torch.testing.assert_close(out[kept], cell[kept])
        # The added vectors span the same subspace as ASE's (orthogonal
        # complement of the kept ones) and are unit length.
        added = out[~kept]
        torch.testing.assert_close(added.norm(dim=-1), torch.ones(len(added)))
        proj = added @ ref[~kept].T
        torch.testing.assert_close(
            proj @ proj.T, torch.eye(len(added)), atol=1e-6, rtol=0
        )
        assert torch.linalg.det(out) > 0

    def test_full_cell_unchanged_and_batched(self):
        cells = torch.stack(
            [TestFoldIntoCell._CELL, torch.diag(torch.tensor([3.0, 4.0, 0.0]))]
        )
        out = _complete_cell(cells)
        torch.testing.assert_close(out[0], cells[0])
        torch.testing.assert_close(out[1], torch.diag(torch.tensor([3.0, 4.0, 1.0])))


class TestFoldIntoCell:
    """``_fold_into_cell`` — the lattice fold ``adapt_input`` applies."""

    _CELL = torch.tensor([[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.5, 6.0]])

    @staticmethod
    def _frac(pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        return pos @ torch.linalg.inv(cell)

    @staticmethod
    def _fold1(pos: torch.Tensor, cell: torch.Tensor, pbc=(True, True, True)):
        """Fold a single-system *pos* with a ``[3, 3]`` cell."""
        return _fold_into_cell(
            pos,
            cell[None],
            torch.tensor([pbc]),
            torch.zeros(pos.shape[0], dtype=torch.long),
        )

    def test_triclinic_all_periodic_lands_in_unit_cell(self):
        frac_in = torch.tensor([[0.2, 0.3, 0.4], [1.2, -0.7, 2.4], [-3.1, 2.5, -1.9]])
        pos = frac_in @ self._CELL
        out = self._fold1(pos, self._CELL)
        frac_out = self._frac(out, self._CELL)
        assert (frac_out >= -1e-6).all() and (frac_out < 1.0).all()
        # Only whole-lattice-vector translations.
        delta = self._frac(pos - out, self._CELL)
        torch.testing.assert_close(delta, delta.round(), atol=1e-5, rtol=0)
        torch.testing.assert_close(out[0], pos[0])

    def test_face_maps_to_zero(self):
        cell = torch.eye(3) * 5.0
        pos = torch.tensor([[5.0, 0.0, 10.0]])
        torch.testing.assert_close(self._fold1(pos, cell), torch.zeros(1, 3))

    def test_non_periodic_direction_untouched(self):
        pos = torch.tensor([[7.5, -2.5, 23.0]])
        out = self._fold1(pos, torch.eye(3) * 5.0, pbc=(True, True, False))
        torch.testing.assert_close(out, torch.tensor([[2.5, 2.5, 23.0]]))

    def test_no_periodic_direction_is_identity(self):
        pos = torch.tensor([[17.0, -9.0, 3.0]])
        out = self._fold1(pos, torch.eye(3) * 5.0, pbc=(False, False, False))
        torch.testing.assert_close(out, pos)

    def test_singular_cell_is_identity(self):
        pos = torch.tensor([[17.0, -9.0, 3.0]])
        torch.testing.assert_close(self._fold1(pos, torch.zeros(3, 3)), pos)

    def test_batched_per_system_cells_and_pbc(self):
        """Each atom folds with its own system's cell/pbc; (B,1,...) shapes OK."""
        cells = torch.stack([torch.eye(3) * 4.0, torch.eye(3) * 10.0])[:, None]
        pbc = torch.tensor([[True, True, True], [False, True, True]])[:, None]
        pos = torch.tensor([[9.0, -1.0, 4.0], [13.0, 13.0, -1.0]])
        out = _fold_into_cell(pos, cells, pbc, torch.tensor([0, 1]))
        expected = torch.tensor([[1.0, 3.0, 0.0], [13.0, 3.0, 9.0]])
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize(
        ("cell", "pbc", "pos", "expected"),
        [
            # 2D slab, zero c: folds in-plane, z untouched.
            (
                torch.diag(torch.tensor([3.0, 4.0, 0.0])),
                (True, True, False),
                [[7.0, -5.0, 13.0]],
                [[1.0, 3.0, 13.0]],
            ),
            # 1D wire along z, zero a and b: folds z only.
            (
                torch.diag(torch.tensor([0.0, 0.0, 7.0])),
                (False, False, True),
                [[1.0, -2.0, -15.0]],
                [[1.0, -2.0, 6.0]],
            ),
            # 1D wire along a tilted a: folds along a, keeps the orthogonal part.
            (
                torch.tensor([[2.0, 2.0, 0.0], [0.0] * 3, [0.0] * 3]),
                (True, False, False),
                [[5.5, 7.0, 3.0]],
                [[-0.5, 1.0, 3.0]],
            ),
        ],
    )
    def test_adapt_input_folds_low_dimensional_cells(
        self, mock_omat, cell, pbc, pos, expected
    ):
        """Zero lattice vectors are completed, so 1D/2D systems fold along periodic axes."""
        data = AtomicData(
            positions=torch.tensor(pos),
            atomic_numbers=torch.tensor([29]),
            cell=cell[None],
            pbc=torch.tensor([pbc]),
        )
        fc = mock_omat.adapt_input(data)
        torch.testing.assert_close(fc.pos, torch.tensor(expected))
        assert torch.linalg.det(fc.cell).abs().item() > 0

    def test_gradient_passes_through_unchanged(self):
        pos = torch.tensor([[9.0, -1.0, 4.0]], requires_grad=True)
        out = self._fold1(pos, torch.eye(3) * 4.0)
        (out * torch.tensor([[1.0, 2.0, 3.0]])).sum().backward()
        torch.testing.assert_close(pos.grad, torch.tensor([[1.0, 2.0, 3.0]]))


class TestAdaptInputWrapping:
    """``adapt_input`` folds periodic positions for fairchem, never in place."""

    @staticmethod
    def _unwrapped(pbc: list[bool]) -> AtomicData:
        a = 5.0
        return AtomicData(
            positions=torch.tensor([[1.0, 1.0, 1.0], [11.5, -3.0, 16.0]]),
            atomic_numbers=torch.tensor([29, 29], dtype=torch.long),
            cell=(torch.eye(3) * a).unsqueeze(0),
            pbc=torch.tensor([pbc]),
        )

    def test_folds_periodic_positions(self, mock_omat):
        fc = mock_omat.adapt_input(self._unwrapped([True, True, True]))
        expected = torch.tensor([[1.0, 1.0, 1.0], [1.5, 2.0, 1.0]])
        torch.testing.assert_close(fc.pos, expected)

    def test_mixed_pbc_leaves_vacuum_axis(self, mock_omat):
        fc = mock_omat.adapt_input(self._unwrapped([True, True, False]))
        expected = torch.tensor([[1.0, 1.0, 1.0], [1.5, 2.0, 16.0]])
        torch.testing.assert_close(fc.pos, expected)

    def test_data_positions_unchanged(self, mock_omat, mock_pu):
        batch = Batch.from_data_list([self._unwrapped([True, True, True])])
        before = batch.positions.clone()
        mock_omat(batch)
        torch.testing.assert_close(batch.positions, before, atol=0, rtol=0)
        assert not torch.equal(mock_pu.last_data.pos, before)

    def test_gradient_reaches_data_positions(self, mock_omat):
        batch = Batch.from_data_list([self._unwrapped([True, True, True])])
        batch.positions.requires_grad_(True)
        fc = mock_omat.adapt_input(batch)
        assert fc.pos.requires_grad
        fc.pos.sum().backward()
        torch.testing.assert_close(batch.positions.grad, torch.ones(2, 3))

    def test_pipeline_autograd_stress_ignores_lattice_shifts(self, mock_omat):
        """An autograd pipeline strains positions and cell before ``adapt_input``.

        Folded atoms must move with the strained lattice, or every atom shifted
        by a lattice vector corrupts the stress. The mock energy depends only on
        the folded coordinates, so shifted and pre-wrapped inputs must agree.
        """
        cell = TestFoldIntoCell._CELL[None]
        wrapped = torch.tensor([[0.5, 1.0, 1.5], [3.0, 4.0, 5.0], [1.0, 0.5, 0.2]])
        images = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.0], [-1.0, 3.0, -2.0]])
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[mock_omat], use_autograd=True)]
        )

        def evaluate(pos: torch.Tensor) -> dict[str, torch.Tensor]:
            data = AtomicData(
                positions=pos,
                atomic_numbers=torch.full((3,), 26),
                cell=cell,
                pbc=torch.tensor([[True, True, True]]),
            )
            out = pipe(Batch.from_data_list([data]))
            return {k: out[k].detach() for k in ("energy", "forces", "stress")}

        ref = evaluate(wrapped)
        assert ref["stress"].abs().max() > 1e-2
        shifted = evaluate(wrapped + images @ cell[0])
        for key in ref:
            torch.testing.assert_close(shifted[key], ref[key])


def _fairchem_graph(wrapper: UMAWrapper, data: AtomicData, version: int) -> dict:
    """Graph fairchem builds from ``adapt_input``'s output (no checkpoint needed)."""
    from fairchem.core.graph.compute import generate_graph

    fc = wrapper.adapt_input(Batch.from_data_list([data]))
    return generate_graph(
        fc,
        cutoff=wrapper._cutoff,
        max_neighbors=1000,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=version,
        pbc=fc.pbc,
    )


def _assert_same_graph(ours: dict, ref: dict) -> None:
    """Same edges with the same edge vectors, independent of edge order."""

    def canonical(graph: dict) -> torch.Tensor:
        vec = graph["edge_distance_vec"].to(torch.float64)
        edges = torch.cat([graph["edge_index"].T.to(torch.float64), vec], dim=1)
        order = np.lexsort(edges.round(decimals=4).numpy().T[::-1])
        return edges[torch.from_numpy(order)]

    torch.testing.assert_close(canonical(ours), canonical(ref), atol=1e-4, rtol=0)


@pytest.mark.parametrize("version", [1, 2], ids=["radius_pbc_v1", "radius_pbc_v2"])
class TestFairchemGraphInputs:
    """The graph fairchem builds from adapted inputs, without a checkpoint."""

    def test_unwrapped_atoms_get_the_wrapped_graph(
        self, mock_omat, monkeypatch, version
    ):
        ref_data = _rattled_fe_333()
        far = _shift_atoms(ref_data, {0: (2, 0, 0), 1: (0, -2, 1)})
        ref = _fairchem_graph(mock_omat, ref_data, version)
        _assert_same_graph(_fairchem_graph(mock_omat, far, version), ref)

        # Without the fold, the builder's image scan misses pairs of far atoms.
        monkeypatch.setattr(uma_module, "_fold_into_cell", lambda pos, *_: pos)
        broken = _fairchem_graph(mock_omat, far, version)
        assert broken["edge_index"].shape[1] < ref["edge_index"].shape[1]

    @pytest.mark.parametrize(
        ("pbc", "shifts"),
        [
            ((True, True, False), {0: (2, 0, 0), 7: (-1, 3, 0), 20: (0, -2, 0)}),
            ((False, False, True), {0: (0, 0, 2), 9: (0, 0, -3)}),
        ],
        ids=["2d_slab", "1d_wire"],
    )
    def test_zero_vacuum_vectors_get_the_vacuum_padded_graph(
        self, mock_omat, monkeypatch, version, pbc, shifts
    ):
        ref_data = _vacuum_padded(pbc, seed=11)
        ref = _fairchem_graph(mock_omat, ref_data, version)
        zero = _zero_vacuum_vectors(ref_data)
        for data in (zero, _shift_atoms(zero, shifts)):
            _assert_same_graph(_fairchem_graph(mock_omat, data, version), ref)

        # Without completion, the builder divides by the zero cell volume.
        monkeypatch.setattr(uma_module, "_complete_cell", lambda cell: cell)
        with pytest.raises(RuntimeError):
            _fairchem_graph(mock_omat, zero, version)


class TestAdaptOutput:
    def test_molecular_energy_forces(self, mock_omol):
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(5, 3),
        }
        out = mock_omol.adapt_output(raw)
        assert "energy" in out and "forces" in out
        # Ensure per-system 2D shape.
        assert out["energy"].shape == (1, 1)
        assert "stress" not in out

    def test_energy_already_2d(self, mock_omol):
        raw = {
            "energy": torch.tensor([[1.5]]),
            "forces": torch.zeros(5, 3),
        }
        out = mock_omol.adapt_output(raw)
        assert out["energy"].shape == (1, 1)

    def test_periodic_stress_shape(self, mock_omat):
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(1, 3),
            "stress": torch.zeros(1, 3, 3),
        }
        out = mock_omat.adapt_output(raw)
        assert out["stress"].shape == (1, 3, 3)

    def test_stress_flat_reshape(self, mock_omat):
        """fairchem sometimes returns stress flattened to (B, 9)."""
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(1, 3),
            "stress": torch.zeros(1, 9),
        }
        out = mock_omat.adapt_output(raw)
        assert out["stress"].shape == (1, 3, 3)


class TestForward:
    def test_composes(self, mock_omol):
        batch = Batch.from_data_list([_make_propane()])
        out = mock_omol(batch)
        assert "energy" in out and "forces" in out
        assert out["energy"].shape == (1, 1)
        assert out["forces"].shape == (11, 3)

    def test_records_input_at_predict(self, mock_omol, mock_pu):
        batch = Batch.from_data_list([_make_propane()])
        mock_omol(batch)
        assert mock_pu.last_data is not None
        assert isinstance(mock_pu.last_data, FCAtomicData)
        assert mock_pu.last_data.pos.shape == (11, 3)

    def test_batched(self, mock_omol):
        batch = Batch.from_data_list([_make_propane(), _make_propane()])
        out = mock_omol(batch)
        assert out["energy"].shape == (2, 1)
        assert out["forces"].shape == (22, 3)


class TestInferenceSettingsSpec:
    """``from_checkpoint`` accepts a key=value spec besides presets and instances."""

    def test_spec_builds_inference_settings(self):
        settings = _resolve_inference_settings(
            "compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"
        )
        assert (
            settings.compile,
            settings.merge_mole,
            settings.tf32,
            settings.activation_checkpointing,
        ) == (
            False,
            False,
            True,
            False,
        )

    def test_presets_and_instances_pass_through(self):
        assert _resolve_inference_settings("batch") == "batch"
        settings = _resolve_inference_settings("tf32=true")
        assert _resolve_inference_settings(settings) is settings

    def test_unknown_field_is_rejected_not_dropped(self):
        with pytest.raises(ValueError, match="compyle"):
            _resolve_inference_settings("compyle=false")

    def test_values_take_their_field_types(self):
        settings = _resolve_inference_settings(
            "base_precision_dtype=float64,edge_chunk_size=512,execution_mode=general,max_atoms=none"
        )
        assert settings.base_precision_dtype is torch.float64
        assert settings.edge_chunk_size == 512
        assert settings.execution_mode == "general"
        assert settings.max_atoms is None

    @pytest.mark.parametrize(
        "spec",
        [
            "compile=1",
            "edge_chunk_size=true",
            "base_precision_dtype=float7",
            "predict_untrained_forces=x",
        ],
    )
    def test_value_that_does_not_fit_the_field_is_rejected(self, spec):
        with pytest.raises(ValueError, match="does not fit"):
            _resolve_inference_settings(spec)


class _MockTask:
    def __init__(self, name: str, prop: str) -> None:
        self.name, self.property = name, prop


class _GateablePredictUnit(_MockPredictUnit):
    """Mock with fairchem's derivative-gating surface: a ``regress_config`` the
    head also holds, and the two task tables post-processing indexes by."""

    def __init__(self) -> None:
        super().__init__()
        inner = self.model.module
        self.regress = type(
            "RegressConfig",
            (),
            {
                "forces": True,
                "stress": True,
                "hessian": False,
                "direct_forces": False,
                "direct_stress": False,
            },
        )()
        inner.backbone.regress_config = self.regress
        inner.output_heads = {
            "efs": type("Head", (), {"regress_config": self.regress})()
        }
        omat = [_MockTask(f"omat_{p}", p) for p in ("energy", "forces", "stress")]
        inner._tasks = {t.name: t for t in omat}
        inner._dataset_to_tasks = {name: [] for name in _UMA_TASKS}
        inner._dataset_to_tasks["omat"] = omat
        self.dataset_to_tasks = inner._dataset_to_tasks
        self.seen: list[tuple[bool, bool, list[str]]] = []

    def predict(self, data: FCAtomicData, undo_element_references: bool = True) -> dict:
        tasks = self.model.module._tasks
        self.seen.append((self.regress.forces, self.regress.stress, sorted(tasks)))
        return super().predict(data, undo_element_references)


class TestDerivativeGating:
    """``active_outputs`` decides which autograd derivatives fairchem computes."""

    def test_energy_only_skips_forces_and_stress_then_restores(self):
        pu = _GateablePredictUnit()
        wrapper = UMAWrapper(pu, task_name="omat")
        batch = Batch.from_data_list([_make_periodic_cu()])

        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper(batch)
        assert "forces" not in out or out["forces"] is None
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        wrapper(batch)

        assert pu.seen[0] == (False, False, ["omat_energy"])
        assert pu.seen[1] == (True, True, ["omat_energy", "omat_forces", "omat_stress"])
        assert pu.model.module._dataset_to_tasks["omat"] is pu.dataset_to_tasks["omat"]

    def test_forces_without_stress_keeps_forces_only(self):
        pu = _GateablePredictUnit()
        wrapper = UMAWrapper(pu, task_name="omat")
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper(Batch.from_data_list([_make_periodic_cu()]))
        assert pu.seen[0] == (True, False, ["omat_energy", "omat_forces"])

    def test_regates_after_fairchem_rebuilds_the_model_mid_run(self):
        """fairchem 2.22's merge_mole fallback rebuilds an unmerged model after the
        first composition change; energy-only must be re-applied on the next call."""
        pu = _GateablePredictUnit()
        wrapper = UMAWrapper(pu, task_name="omat")
        wrapper.model_config.active_outputs = {"energy"}
        batch = Batch.from_data_list([_make_periodic_cu()])
        wrapper(batch)
        # Simulate the rebuild: derivatives back on, full task table restored.
        pu.regress.forces, pu.regress.stress = True, True
        inner = pu.model.module
        inner._tasks.update(
            {
                t.name: t
                for t in inner._dataset_to_tasks["omat"]
                + [
                    _MockTask("omat_forces", "forces"),
                    _MockTask("omat_stress", "stress"),
                ]
            }
        )
        wrapper(batch)
        assert pu.seen[-1] == (False, False, ["omat_energy"])

    def test_wrapper_built_after_another_gated_the_unit_restores_derivatives(self):
        """A wrapper built after an energy-only one ran still sees the as-loaded tables."""
        pu = _GateablePredictUnit()
        batch = Batch.from_data_list([_make_periodic_cu()])
        energy_only = UMAWrapper(pu, task_name="omat")
        energy_only.model_config.active_outputs = {"energy"}
        energy_only(batch)
        UMAWrapper(pu, task_name="omat")(batch)
        assert pu.seen[-1] == (
            True,
            True,
            ["omat_energy", "omat_forces", "omat_stress"],
        )

    def test_unit_is_as_loaded_after_an_energy_only_call(self):
        """A calculator sharing the unit is not left without forces/stress."""
        pu = _GateablePredictUnit()
        wrapper = UMAWrapper(pu, task_name="omat")
        wrapper.model_config.active_outputs = {"energy"}
        wrapper(Batch.from_data_list([_make_periodic_cu()]))

        assert pu.seen[-1] == (False, False, ["omat_energy"])
        assert (pu.regress.forces, pu.regress.stress) == (True, True)
        assert sorted(pu.model.module._tasks) == [
            "omat_energy",
            "omat_forces",
            "omat_stress",
        ]
        assert len(pu.dataset_to_tasks["omat"]) == 3

    def test_unit_is_restored_when_predict_raises(self):
        pu = _GateablePredictUnit()
        wrapper = UMAWrapper(pu, task_name="omat")
        wrapper.model_config.active_outputs = {"energy"}
        with (
            patch.object(pu, "predict", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            wrapper(Batch.from_data_list([_make_periodic_cu()]))
        assert (pu.regress.forces, pu.regress.stress) == (True, True)

    def test_unrecognised_layout_no_warning_when_all_producible_outputs_requested(
        self, mock_omol
    ):
        """omol has no stress: energy + forces is everything, nothing is discarded."""
        mock_omol.model_config.active_outputs = {"energy", "forces"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mock_omol(Batch.from_data_list([_make_propane()]))

    def test_unrecognised_layout_warns_once_and_computes_everything(self, mock_omat):
        mock_omat.model_config.active_outputs = {"energy"}
        batch = Batch.from_data_list([_make_periodic_cu()])
        with pytest.warns(UserWarning, match="cannot skip forces/stress"):
            mock_omat(batch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mock_omat(batch)  # second call must not warn again


# ===========================================================================
# Distribution spec — domain-decomposition halo policy (mock-only)
# ===========================================================================


class TestGraphPartitionAdapters:
    @pytest.mark.parametrize(
        ("rank", "edge_index", "expected_scatter_target"),
        [
            (0, torch.tensor([[4, 2], [0, 1]]), torch.tensor([0, 1])),
            (1, torch.tensor([[0, 1, 3], [2, 4, 3]]), torch.tensor([0, 2, 1])),
        ],
    )
    def test_partition_maps_global_receivers_to_owned_rows(
        self, rank, edge_index, expected_scatter_target
    ):
        ctx = Mock(rank=rank, world_size=2)
        ctx.gather_meta.owner_rank = torch.tensor([0, 0, 1, 1, 1])
        original = Mock(return_value={"edge_index": edge_index})
        backbone = Mock(otf_graph=True)
        data_dict = {
            "atomic_numbers": torch.arange(5),
            "batch": torch.zeros(5, dtype=torch.long),
        }

        graph = _distributed_partition_graph.__wrapped__.__wrapped__(
            ctx, original, backbone, data_dict
        )

        assert graph["edge_index"] is edge_index
        torch.testing.assert_close(data_dict["scatter_target"], expected_scatter_target)
        assert data_dict["scatter_target"].min().item() >= 0
        assert data_dict["scatter_target"].max().item() < len(
            data_dict["atomic_numbers"]
        )
        assert "gp_node_offset" not in data_dict
        assert backbone.otf_graph is True

    def test_partition_rejects_non_owned_receiver(self):
        ctx = Mock(rank=1, world_size=2)
        ctx.gather_meta.owner_rank = torch.tensor([0, 0, 1, 1, 1])
        original = Mock(return_value={"edge_index": torch.tensor([[0, 3], [1, 3]])})
        backbone = Mock(otf_graph=True)
        data_dict = {
            "atomic_numbers": torch.arange(5),
            "batch": torch.zeros(5, dtype=torch.long),
        }

        with pytest.raises(RuntimeError, match="outside this rank's owned atom block"):
            _distributed_partition_graph.__wrapped__.__wrapped__(
                ctx, original, backbone, data_dict
            )

    def test_edgewise_passes_scatter_target_to_forward_chunk(self):
        edgewise = Mock()
        expected = torch.randn(2, 4)
        edgewise.forward_chunk.return_value = expected
        x = torch.randn(2, 3, 4)
        x_full = torch.randn(5, 3, 4)
        x_edge = torch.randn(3, 8)
        edge_index = torch.tensor([[4, 0, 3], [2, 4, 3]])
        wigner = torch.randn(3, 2)
        wigner_inv_envelope = torch.randn(3, 2)
        scatter_target = torch.tensor([0, 2, 1])

        with patch(
            "nvalchemi.models.uma.refresh_neighbors", return_value=x_full
        ) as refresh:
            result = _distributed_edgewise_gather.__wrapped__(
                Mock(),
                Mock(),
                edgewise,
                x,
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                5,
                scatter_target,
            )

        assert result is expected
        refresh.assert_called_once_with(x)
        edgewise.forward_chunk.assert_called_once_with(
            x_full,
            2,
            x_edge,
            edge_index,
            wigner,
            wigner_inv_envelope,
            scatter_target,
        )

    def test_edgewise_requires_scatter_target(self):
        with pytest.raises(RuntimeError, match="did not provide scatter_target"):
            _distributed_edgewise_gather.__wrapped__(
                Mock(),
                Mock(),
                Mock(),
                torch.randn(2, 3, 4),
                torch.randn(3, 8),
                torch.tensor([[4, 0, 3], [2, 4, 3]]),
                torch.randn(3, 2),
                torch.randn(3, 2),
                5,
            )


class TestMLIPSpec:
    def test_inherits_uma_storage_modes(self, mock_omol):
        """Spec carries the halo storage policy (default modes).

        The per-block edge→node aggregation is owned-complete under the halo
        (ghost-shell) policy, so the correction is a per-block input refresh +
        boundary fold ``MethodAdapter``\\s on the fairchem backbone (see
        :meth:`test_registers_boundary_fold_adapters`), NOT a ``scatter_mode``
        override — so the policy keeps the preset's default ``halo_read`` gather
        mode. (The old ``scatter="local"`` override and the ``ScatterOutputs``
        Triton ``custom_ops`` both belonged to the retired ``gp_utils``/
        replicated design.)
        """
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

        spec = mock_omol.distribution_spec()
        policy = spec.distribution.policy
        assert isinstance(policy, HaloStoragePolicy)
        assert policy.gather_mode == "halo_read"
        assert spec.system_reductions is True

    def test_no_triton_custom_ops(self, mock_omol):
        """No ``custom_ops``: the retired ``gp_utils``/replicated design
        registered five ``torch.ops.fairchem._kernel_*`` OpAdapters (two carrying
        ``ScatterOutputs``); the current halo design corrects at the fairchem
        module boundary via fold adapters instead, so ``custom_ops`` is empty."""
        spec = mock_omol.distribution_spec()
        assert spec.distribution.custom_ops == ()

    def test_registers_boundary_fold_adapters(self, mock_omol):
        """The per-block edge→node correction and the owned-only + all-reduce
        energy/element-reference reduction are carried by method/function fold
        adapters on the fairchem backbone (lowered onto ``third_party_helpers``),
        which replaced the retired ``ScatterOutputs`` Triton OpAdapters.

        Under the refresh-only halo policy the edge→node folds reduce to a pure
        input refresh (owned-complete); under graph-parallel the same adapters
        become an all-reduce — the point here is only that they are declared.
        """
        spec = mock_omol.distribution_spec()
        helpers = spec.distribution.third_party_helpers
        methods = {
            (h.class_name, h.method_name) for h in helpers if hasattr(h, "method_name")
        }
        funcs = {
            (h.module_path.split(".")[-1], h.attr_name)
            for h in helpers
            if hasattr(h, "attr_name")
        }
        # Per-block input refresh + the two edge→node aggregation recombines that
        # replaced the ScatterOutputs OpAdapters.
        assert ("eSCNMD_Block", "forward") in methods
        assert ("Edgewise", "forward") in methods
        assert ("EdgeDegreeEmbedding", "forward") in methods
        # Owned-only + all_reduce per-system energy reduction, patched on both
        # module bindings of ``reduce_node_to_system``.
        assert ("outputs", "reduce_node_to_system") in funcs
        assert ("escn_md", "reduce_node_to_system") in funcs
        # Element-reference undo summed over owned atoms only.
        assert ("ElementReferences", "undo_refs") in methods
        # MoLE composition-consistency guard (version-selected between the
        # fairchem<=2.19 and >=2.21 method names).
        assert ("eSCNMDBackbone", "_get_composition_info") in methods or (
            "eSCNMDMoeBackbone",
            "_get_merged_mole_consistency_info",
        ) in methods

    def test_graph_partition_registers_partition_adapters(self, mock_omol):
        from nvalchemi.distributed.config import StrategyKind

        spec = mock_omol.distribution_spec(StrategyKind.GRAPH_PARTITION)
        helpers = spec.distribution.third_party_helpers
        replacements = {
            (helper.class_name, helper.method_name): helper.replacement
            for helper in helpers
            if hasattr(helper, "method_name")
        }

        assert replacements[("eSCNMDBackbone", "_generate_graph")] is (
            _distributed_partition_graph
        )
        assert replacements[("Edgewise", "forward")] is _distributed_edgewise_gather


# ===========================================================================
# Checkpoint tests — real fairchem checkpoint (skipped without HF access)
# ===========================================================================


def _load_predict_unit():
    """Load a fresh UMA predict unit; skip if HF access or download fails."""
    from fairchem.core.calculate import pretrained_mlip
    from huggingface_hub.errors import GatedRepoError

    try:
        return pretrained_mlip.get_predict_unit(_CKPT, device=_DEVICE)
    except GatedRepoError as e:
        pytest.skip(f"no HF access to UMA checkpoint {_CKPT}: {e}")
    except Exception as e:  # noqa: BLE001 — top-level guard for CI portability
        pytest.skip(f"could not load UMA checkpoint {_CKPT}: {e}")


@pytest.fixture(scope="module")
def predict_unit():
    """UMA predict unit shared by the wrappers and reference calculators.

    ``UMAWrapper`` gates derivatives only for the duration of each call and
    restores the unit afterwards, so the calculators sharing it see the
    as-loaded forces and stress -- these comparisons double as the end-to-end
    check of that.
    """
    return _load_predict_unit()


@pytest.fixture(scope="module")
def calc_omol(predict_unit):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit=predict_unit, task_name="omol")


@pytest.fixture(scope="module")
def calc_omat(predict_unit):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit=predict_unit, task_name="omat")


@pytest.fixture(scope="module")
def wrapper_omol(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omol")


@pytest.fixture(scope="module")
def wrapper_omat(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omat")


def _propane_atoms() -> Atoms:
    """Propane C3H8 — OMol test system."""
    positions = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [1.5260, 0.0000, 0.0000],
            [2.0330, 1.4360, 0.0000],
            [-0.5093, 1.0222, 0.0000],
            [-0.5093, -0.5111, 0.8853],
            [-0.5093, -0.5111, -0.8853],
            [2.0319, -0.5111, 0.8853],
            [2.0319, -0.5111, -0.8853],
            [3.1193, 1.4360, 0.0000],
            [1.6763, 1.9471, 0.8853],
            [1.6763, 1.9471, -0.8853],
        ]
    )
    numbers = [6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
    atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def _bcc_fe_atoms() -> Atoms:
    """bcc Fe, 2x2x2 supercell — OMat test system (16 atoms)."""
    return bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)


def _atomicdata_from_ase(atoms: Atoms) -> AtomicData:
    """Convert an ASE ``Atoms`` into our ``AtomicData`` (CPU tensors)."""
    pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)
    numbers = torch.as_tensor(np.asarray(atoms.get_atomic_numbers()), dtype=torch.long)
    kwargs: dict = {"positions": pos, "atomic_numbers": numbers}
    if np.any(atoms.pbc):
        kwargs["cell"] = torch.as_tensor(
            np.asarray(atoms.cell.array), dtype=torch.float32
        ).unsqueeze(0)
        kwargs["pbc"] = torch.as_tensor(
            np.asarray(atoms.pbc), dtype=torch.bool
        ).reshape(1, 3)
    return AtomicData(**kwargs)


def _bcc_fe_batch(device: str | torch.device, seed: int = 42) -> Batch:
    """bcc Fe 2x2x2 (16 atoms) on *device* with MB velocities at 300 K.

    Carries positions / atomic_numbers / atomic_masses / cell / pbc /
    velocities — ready for NVE and for the GPU-resident turbo forward.
    """
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    n = len(atoms)
    positions = torch.as_tensor(
        np.asarray(atoms.positions), dtype=torch.float32, device=device
    )
    numbers = torch.as_tensor(
        np.asarray(atoms.get_atomic_numbers()), dtype=torch.long, device=device
    )
    masses = torch.full((n,), 55.845, dtype=torch.float32, device=device)
    cell = torch.as_tensor(
        np.asarray(atoms.cell.array), dtype=torch.float32, device=device
    ).unsqueeze(0)
    pbc = torch.ones(1, 3, dtype=torch.bool, device=device)

    kB = 8.617333262e-5  # eV/K
    g = torch.Generator(device="cpu").manual_seed(seed)
    vel = torch.randn(n, 3, generator=g).to(device) * float(
        (kB * 300.0 / 55.845) ** 0.5
    )
    vel -= vel.mean(dim=0)  # zero net momentum

    data = AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        atomic_masses=masses,
        cell=cell,
        pbc=pbc,
        velocities=vel,
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1, device=device, dtype=torch.float32),
    )
    return Batch.from_data_list([data])


# ---------------------------------------------------------------------------
# Forward equivalence vs FAIRChemCalculator
# ---------------------------------------------------------------------------


class TestOMolEquivalence:
    """Propane molecular energy/forces match ``FAIRChemCalculator``."""

    @pytest.fixture(autouse=True)
    def _setup(self, wrapper_omol, calc_omol):
        self.wrapper = wrapper_omol
        self.calc = calc_omol
        self.atoms = _propane_atoms()

    def _reference(self) -> dict[str, np.ndarray]:
        atoms = self.atoms.copy()
        atoms.info = dict(self.atoms.info)
        atoms.calc = self.calc
        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
        }

    def _wrapper_result(self) -> dict[str, np.ndarray]:
        data = _atomicdata_from_ase(self.atoms)
        batch = Batch.from_data_list([data])
        batch.charge = torch.tensor([0], dtype=torch.long)
        batch.spin = torch.tensor([1], dtype=torch.long)
        out = self.wrapper(batch)
        return {
            "energy": float(out["energy"].detach().cpu().numpy().flatten()[0]),
            "forces": out["forces"].detach().cpu().numpy(),
        }

    def test_energy_matches(self):
        ref = self._reference()
        ours = self._wrapper_result()
        # fp32 precision — 1e-4 eV absolute covers round-trip jitter.
        assert np.isclose(ours["energy"], ref["energy"], atol=1e-4, rtol=1e-5), (
            f"energy mismatch: ours={ours['energy']:.6f} "
            f"ref={ref['energy']:.6f} diff={ours['energy'] - ref['energy']:.2e}"
        )

    def test_forces_match(self):
        ref = self._reference()
        ours = self._wrapper_result()
        assert ours["forces"].shape == ref["forces"].shape
        np.testing.assert_allclose(ours["forces"], ref["forces"], atol=1e-4, rtol=1e-4)


class TestOMatEquivalence:
    """bcc Fe 2x2x2 energy/forces/stress match ``FAIRChemCalculator``."""

    @pytest.fixture(autouse=True)
    def _setup(self, wrapper_omat, calc_omat):
        self.wrapper = wrapper_omat
        self.calc = calc_omat
        self.atoms = _bcc_fe_atoms()

    def _reference(self) -> dict[str, np.ndarray]:
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": atoms.get_stress(voigt=False),
        }

    def _wrapper_result(self) -> dict[str, np.ndarray]:
        data = _atomicdata_from_ase(self.atoms)
        batch = Batch.from_data_list([data])
        out = self.wrapper(batch)
        return {
            "energy": float(out["energy"].detach().cpu().numpy().flatten()[0]),
            "forces": out["forces"].detach().cpu().numpy(),
            "stress": out["stress"].detach().cpu().numpy()[0],
        }

    def test_energy_matches(self):
        ref = self._reference()
        ours = self._wrapper_result()
        assert np.isclose(ours["energy"], ref["energy"], atol=1e-4, rtol=1e-5), (
            f"energy mismatch: ours={ours['energy']:.6f} "
            f"ref={ref['energy']:.6f} diff={ours['energy'] - ref['energy']:.2e}"
        )

    def test_forces_match(self):
        ref = self._reference()
        ours = self._wrapper_result()
        np.testing.assert_allclose(ours["forces"], ref["forces"], atol=1e-4, rtol=1e-4)

    def test_stress_matches(self):
        ref = self._reference()
        ours = self._wrapper_result()
        # Reference is (3, 3); ours is (3, 3) after the adapt_output path.
        np.testing.assert_allclose(
            ours["stress"].reshape(3, 3),
            ref["stress"].reshape(3, 3),
            atol=1e-4,
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Charged inputs — total charge must propagate into the OMol head
# ---------------------------------------------------------------------------


class TestChargedInputs:
    """OMol energies respond to (and correctly use) the total-charge input."""

    def _wrapper_energy(self, wrapper, charge: int, spin: int) -> float:
        batch = Batch.from_data_list([_atomicdata_from_ase(_propane_atoms())])
        batch.charge = torch.tensor([charge], dtype=torch.long)
        batch.spin = torch.tensor([spin], dtype=torch.long)
        return float(wrapper(batch)["energy"].detach().cpu().flatten()[0])

    def test_charge_changes_energy(self, wrapper_omol):
        """Neutral singlet vs anion doublet — charge must alter the energy."""
        e_neutral = self._wrapper_energy(wrapper_omol, charge=0, spin=1)
        e_anion = self._wrapper_energy(wrapper_omol, charge=-1, spin=2)
        assert math.isfinite(e_neutral) and math.isfinite(e_anion)
        assert abs(e_anion - e_neutral) > 1e-3, (
            f"charge had no effect: neutral={e_neutral:.6f} anion={e_anion:.6f}"
        )

    def test_charged_matches_calculator(self, wrapper_omol, calc_omol):
        """A charged wrapper run matches ``FAIRChemCalculator`` with the same
        charge/spin — confirming charge is passed through correctly, not just
        that *something* changed."""
        atoms = _propane_atoms()
        atoms.info["charge"] = -1
        atoms.info["spin"] = 2
        atoms.calc = calc_omol
        ref = atoms.get_potential_energy()
        ours = self._wrapper_energy(wrapper_omol, charge=-1, spin=2)
        assert np.isclose(ours, ref, atol=1e-4, rtol=1e-5), (
            f"charged energy mismatch: ours={ours:.6f} ref={ref:.6f}"
        )


# ---------------------------------------------------------------------------
# Lattice-vector invariance — unwrapped MD positions (no WrapPeriodicHook)
# ---------------------------------------------------------------------------


def _rattled_fe_333(seed: int = 7) -> AtomicData:
    """Rattled bcc Fe 3x3x3 (54 atoms, 8.61 A cell): fairchem scans +-1 image."""
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (3, 3, 3)
    atoms.rattle(stdev=0.05, seed=seed)
    atoms.wrap()
    return _atomicdata_from_ase(atoms)


def _shift_atoms(
    data: AtomicData, shifts: dict[int, tuple[int, int, int]]
) -> AtomicData:
    """Copy of *data* with atom ``i`` translated by ``shifts[i]`` lattice vectors."""
    pos = data.positions.clone()
    cell = data.cell.reshape(3, 3)
    for i, n in shifts.items():
        pos[i] += torch.tensor(n, dtype=pos.dtype) @ cell
    return AtomicData(
        positions=pos, atomic_numbers=data.atomic_numbers, cell=data.cell, pbc=data.pbc
    )


def _evaluate(wrapper: UMAWrapper, data: AtomicData) -> dict[str, torch.Tensor]:
    batch = Batch.from_data_list([data]).to(_DEVICE)
    out = wrapper(batch)
    return {k: out[k].detach().cpu() for k in ("energy", "forces", "stress")}


def _assert_same_outputs(ours: dict, ref: dict) -> None:
    torch.testing.assert_close(ours["energy"], ref["energy"], atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(ours["forces"], ref["forces"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ours["stress"], ref["stress"], atol=1e-5, rtol=1e-4)


class TestUnwrappedPositions:
    """Energy/forces/stress are invariant to per-atom lattice translations."""

    _SHIFTS = {
        0: (1, 0, 0),
        5: (-1, 0, 0),
        11: (0, 2, 0),
        17: (0, 0, -2),
        23: (3, -3, 0),
        29: (-3, 1, 3),
        41: (2, -1, -3),
    }

    def test_forces_and_stress_active(self, wrapper_omat):
        active = set(wrapper_omat.model_config.active_outputs)
        assert {"energy", "forces", "stress"} <= active

    def test_lattice_vector_invariance(self, wrapper_omat):
        ref_data = _rattled_fe_333()
        ref = _evaluate(wrapper_omat, ref_data)
        assert ref["forces"].abs().max() > 1e-2  # rattled: non-trivial forces
        ours = _evaluate(wrapper_omat, _shift_atoms(ref_data, self._SHIFTS))
        _assert_same_outputs(ours, ref)

    def test_far_atom_regression(self, wrapper_omat, monkeypatch):
        """An atom two cells out lost its neighbours before the fold; now matches."""
        ref_data = _rattled_fe_333()
        far = _shift_atoms(ref_data, {0: (2, 0, 0), 1: (0, -2, 1)})
        ref = _evaluate(wrapper_omat, ref_data)
        _assert_same_outputs(_evaluate(wrapper_omat, far), ref)

        # Without the fold, fairchem's +-1 image scan misses pairs.
        monkeypatch.setattr(uma_module, "_fold_into_cell", lambda pos, *_: pos)
        broken = _evaluate(wrapper_omat, far)
        assert (broken["energy"] - ref["energy"]).abs().max() > 1e-2

    def test_mixed_pbc_slab_matches_wrapped(self, wrapper_omat):
        """pbc=(T, T, F): in-plane translations are folded, z is not touched."""
        atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (3, 3, 3)
        atoms.rattle(stdev=0.05, seed=3)
        atoms.center(vacuum=8.0, axis=2)
        atoms.pbc = (True, True, False)
        atoms.wrap()
        ref_data = _atomicdata_from_ase(atoms)
        ref = _evaluate(wrapper_omat, ref_data)
        shifted = _shift_atoms(ref_data, {0: (2, 0, 0), 7: (-1, 3, 0)})
        _assert_same_outputs(_evaluate(wrapper_omat, shifted), ref)


def _vacuum_padded(pbc: tuple[bool, bool, bool], seed: int) -> AtomicData:
    """Rattled bcc Fe with vacuum along every non-periodic axis (non-singular cell)."""
    repeats = tuple(3 if periodic else 2 for periodic in pbc)
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * repeats
    atoms.rattle(stdev=0.05, seed=seed)
    for axis, periodic in enumerate(pbc):
        if not periodic:
            atoms.center(vacuum=8.0, axis=axis)
    atoms.pbc = pbc
    atoms.wrap()
    return _atomicdata_from_ase(atoms)


def _zero_vacuum_vectors(data: AtomicData) -> AtomicData:
    """Copy of *data* whose non-periodic lattice vectors are zero (singular cell)."""
    cell = data.cell.clone()
    cell[0, ~data.pbc.reshape(3)] = 0.0
    return AtomicData(
        positions=data.positions.clone(),
        atomic_numbers=data.atomic_numbers,
        cell=cell,
        pbc=data.pbc,
    )


class TestLowDimensionalCells:
    """1D/2D cells with zero lattice vectors match the vacuum-padded cell.

    Non-periodic directions get no images, so a zero vector and an explicit
    vacuum vector describe the same system. Without completing the zero
    vectors, fairchem's graph builder sees a zero cell volume. Stress is
    normalised by that volume, so only energy and forces are compared.
    """

    @pytest.mark.parametrize(
        ("pbc", "shifts"),
        [
            ((True, True, False), {0: (2, 0, 0), 7: (-1, 3, 0), 20: (0, -2, 0)}),
            ((False, False, True), {0: (0, 0, 2), 9: (0, 0, -3)}),
        ],
        ids=["2d_slab", "1d_wire"],
    )
    def test_zero_vacuum_vectors_match_vacuum_padded_cell(
        self, wrapper_omat, pbc, shifts
    ):
        ref_data = _vacuum_padded(pbc, seed=11)
        ref = _evaluate(wrapper_omat, ref_data)
        assert ref["forces"].abs().max() > 1e-2  # rattled: non-trivial forces

        zero = _zero_vacuum_vectors(ref_data)
        for data in (zero, _shift_atoms(zero, shifts)):
            ours = _evaluate(wrapper_omat, data)
            assert torch.isfinite(ours["energy"]).all()
            torch.testing.assert_close(
                ours["energy"], ref["energy"], atol=1e-4, rtol=1e-6
            )
            torch.testing.assert_close(
                ours["forces"], ref["forces"], atol=1e-4, rtol=1e-4
            )


# ---------------------------------------------------------------------------
# NVE energy conservation (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNVEStability:
    """NVE drift over a short trajectory must stay below 1 meV/atom."""

    def test_bcc_fe_300k(self, wrapper_omat):
        """1000-step NVE on bcc Fe 2x2x2 at 300 K — drift < 1 meV/atom."""
        from nvalchemi.dynamics.base import DynamicsStage

        n_steps = int(os.environ.get("NVALCHEMI_UMA_NVE_STEPS", 1000))
        dt_fs = float(os.environ.get("NVALCHEMI_UMA_NVE_DT_FS", 0.5))
        stride = max(1, n_steps // 10)
        threshold_ev_per_atom = 1e-3

        batch = _bcc_fe_batch(_DEVICE)
        n_atoms = batch.num_nodes
        nve = NVE(wrapper_omat, dt=dt_fs)

        trajectory: list[tuple[int, float]] = []

        def _total_energy(b: Batch) -> float:
            pe = b.energy.squeeze(-1).sum().item()
            ke = kinetic_energy_per_graph(
                b.velocities, b.atomic_masses, b.batch_idx, b.num_graphs
            )
            return pe + ke.squeeze(-1).sum().item()

        def _energy_probe(ctx, stage):
            if ctx.step_count % stride == 0 or ctx.step_count == n_steps:
                trajectory.append((ctx.step_count, _total_energy(ctx.batch)))

        _energy_probe.stage = DynamicsStage.AFTER_STEP
        _energy_probe.frequency = 1
        nve.register_hook(_energy_probe)

        nve.run(batch, n_steps=n_steps)

        assert trajectory, "no energy samples recorded"
        e0, e_final = trajectory[0][1], trajectory[-1][1]
        drift_per_atom = abs(e_final - e0) / n_atoms
        print(
            f"\nNVE stability ({_CKPT}, bcc Fe 2x2x2, 300 K, {n_steps} @ {dt_fs} fs): "
            f"drift {drift_per_atom * 1e3:.4f} meV/atom"
        )
        assert drift_per_atom < threshold_ev_per_atom, (
            f"NVE drift {drift_per_atom * 1e3:.3f} meV/atom exceeds "
            f"1 meV/atom over {n_steps} steps"
        )


# ---------------------------------------------------------------------------
# Turbo / torch.compile device path (slow, CUDA only)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="turbo/compile device-placement path is CUDA-specific",
)
class TestTurboCompile:
    """UMA under ``inference_settings="turbo"`` (compile + MoLE merge).

    Reproduces the original failing scenario — a GPU-resident first
    forward under turbo, which used to crash with a CPU/CUDA device
    mismatch (fairchem's lazy MoLE merge ran before its normal device move).
    ``UMAWrapper.forward`` moves the model and first input to CUDA before
    Fairchem performs the merge.
    """

    @pytest.fixture(scope="class")
    def wrapper_turbo(self) -> UMAWrapper:
        from huggingface_hub.errors import GatedRepoError

        try:
            return UMAWrapper.from_checkpoint(
                _CKPT, task_name="omat", device="cuda", inference_settings="turbo"
            )
        except GatedRepoError as e:
            pytest.skip(f"no HF access to UMA checkpoint {_CKPT}: {e}")
        except Exception as e:  # noqa: BLE001 — top-level guard for CI portability
            pytest.skip(f"could not load UMA checkpoint {_CKPT}: {e}")

    def test_gpu_resident_first_forward(self, wrapper_turbo: UMAWrapper) -> None:
        """A GPU-resident first forward (lazy merge + compile) must run and
        return finite, on-device outputs — the case that used to crash."""
        out = wrapper_turbo(_bcc_fe_batch("cuda"))
        assert out["energy"].shape == (1, 1)
        assert out["forces"].shape == (16, 3)
        assert out["stress"].shape == (1, 3, 3)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        assert out["forces"].device.type == "cuda"

    def test_second_forward_after_init(self, wrapper_turbo: UMAWrapper) -> None:
        """After lazy init, a fresh GPU batch still runs on-device."""
        out = wrapper_turbo(_bcc_fe_batch("cuda"))
        assert torch.isfinite(out["energy"]).all()
        assert out["forces"].device.type == "cuda"
