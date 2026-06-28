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
"""Tests for :mod:`nvalchemi.training._checkpoint`."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import EnergyMSELoss, TrainingStage
from nvalchemi.training._checkpoint import (
    CheckpointManifest,
    load_checkpoint,
    save_checkpoint,
)
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn

# ---------------------------------------------------------------------------
# Helper classes (reused from original file)
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Custom activation: SwiGLU-style gated activation with a learnable scale.

    Splits the input channel dimension in half, applies SiLU to one half and
    multiplies by the other, then scales by a learnable parameter. Exercises
    :func:`create_model_spec`/:func:`save_checkpoint` against a module that
    owns its own :class:`~torch.nn.Parameter` rather than delegating to a
    stock layer.
    """

    def __init__(self, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return self.scale * (a * torch.nn.functional.silu(b))


class CustomMLPBlock(nn.Module):
    """Pre-norm MLP with a custom activation, a SwiGLU expansion, and dropout.

    The block is deliberately non-trivial: it stacks :class:`LayerNorm`, an
    expansion :class:`Linear` that feeds :class:`SwiGLU` (which halves the
    channel dimension), a projection :class:`Linear`, and :class:`Dropout`,
    with an optional residual connection. It stress tests the spec layer in
    three ways:

    1. ``__init__`` takes a mix of ints, floats, booleans, and a
       :class:`torch.dtype` --- the latter routed through the custom type
       serializer registry.
    2. The module owns parameters at multiple nesting depths (top-level
       :class:`Linear` weights, plus the :class:`SwiGLU` scale parameter).
    3. The forward pass is stateless w.r.t. shape up to the channel
       dimension, so round-tripped weights must reproduce outputs exactly
       (modulo dropout, which we disable by calling ``.eval()``).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.1,
        eps: float = 1e-5,
        activation_scale: float = 1.0,
        use_residual: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if hidden_features % 2 != 0:
            raise ValueError(
                f"hidden_features must be even for SwiGLU, got {hidden_features}"
            )
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_residual = use_residual

        self.norm = nn.LayerNorm(in_features, eps=eps, dtype=dtype)
        self.expand = nn.Linear(in_features, hidden_features, dtype=dtype)
        self.activation = SwiGLU(init_scale=activation_scale)
        self.project = nn.Linear(hidden_features // 2, in_features, dtype=dtype)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.expand(h)
        h = self.activation(h)
        h = self.project(h)
        h = self.dropout(h)
        return x + h if self.use_residual else h


@dataclass
class NotModule:
    """Non-module class used to verify load rejects non-nn.Module specs."""

    arg_a: int
    arg_b: str


def checkpoint_training_fn(
    model: BaseModelMixin,
    batch: Batch,
) -> dict[str, torch.Tensor]:
    """Importable training function used by strategy checkpoint tests."""
    return default_training_fn(model, batch)


class _NoOpCheckpointHook:
    """Simple observer hook used by checkpoint restart tests."""

    stage = TrainingStage.BEFORE_BATCH
    frequency = 1

    def __call__(self, ctx: Any, stage: TrainingStage) -> None:
        """Observe a training stage without mutating state."""
        del ctx, stage


def _make_checkpoint_batch(n_atoms: int = 3, seed: int = 0) -> Batch:
    """Build a small batch with energy targets for strategy checkpoint tests."""
    generator = torch.Generator().manual_seed(seed)
    data = AtomicData(
        positions=torch.randn(n_atoms, 3, generator=generator),
        atomic_numbers=torch.randint(
            1, 10, (n_atoms,), dtype=torch.long, generator=generator
        ),
        atomic_masses=torch.ones(n_atoms),
        energy=torch.randn(1, 1, generator=generator),
    )
    return Batch.from_data_list([data])


def _make_checkpoint_strategy(num_steps: int = 4) -> TrainingStrategy:
    """Create a serializable demo training strategy for checkpoint tests."""
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    torch.manual_seed(0)
    model = DemoModelWrapper(DemoModel(num_atom_types=20, hidden_dim=8))
    return TrainingStrategy(
        models=model,
        optimizer_configs=OptimizerConfig(
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": 1e-3},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 2, "gamma": 0.5},
        ),
        num_steps=num_steps,
        training_fn=checkpoint_training_fn,
        loss_fn=EnergyMSELoss(),
        devices=[torch.device("cpu")],
    )


def _make_multi_optimizer_checkpoint_strategy() -> TrainingStrategy:
    """Create a serializable strategy with two optimizer/scheduler pairs."""
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    torch.manual_seed(0)
    model = DemoModelWrapper(DemoModel(num_atom_types=20, hidden_dim=8))
    return TrainingStrategy(
        models=model,
        optimizer_configs=[
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": 1e-3},
                scheduler_cls=torch.optim.lr_scheduler.StepLR,
                scheduler_kwargs={"step_size": 2, "gamma": 0.5},
            ),
            OptimizerConfig(
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"lr": 1e-2},
                scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
                scheduler_kwargs={"gamma": 0.9},
            ),
        ],
        num_steps=1,
        training_fn=checkpoint_training_fn,
        loss_fn=EnergyMSELoss(),
        devices=[torch.device("cpu")],
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestSaveCheckpointSingleModel:
    """Basic dict-based single-model checkpoint save/load behavior."""

    def test_save_creates_manifest_and_model_dir(self, tmp_path: Path) -> None:
        """Directory layout: manifest.json + models/{name}/ created on save."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        assert (tmp_path / "manifest.json").is_file()
        assert (tmp_path / "models" / "main" / "spec.json").is_file()
        assert (tmp_path / "models" / "main" / "checkpoints" / "0.pt").is_file()

    def test_save_load_basic_roundtrip(self, tmp_path: Path) -> None:
        """Save one model, load it, verify weights match."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx = save_checkpoint(tmp_path, models={"main": (model, spec)})
        assert idx == 0

        result = load_checkpoint(tmp_path)
        assert isinstance(result, CheckpointManifest)
        assert "main" in result.models
        reloaded, reloaded_spec = result.models["main"]
        assert isinstance(reloaded, nn.Linear)
        assert torch.equal(reloaded.weight, model.weight)
        assert torch.equal(reloaded.bias, model.bias)
        assert reloaded_spec.timestamp == spec.timestamp

    def test_autoincrement_from_manifest(self, tmp_path: Path) -> None:
        """Three sequential saves auto-increment indices 0, 1, 2."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx0 = save_checkpoint(tmp_path, models={"main": (model, spec)})
        idx1 = save_checkpoint(tmp_path, models={"main": (model, spec)})
        idx2 = save_checkpoint(tmp_path, models={"main": (model, spec)})
        assert (idx0, idx1, idx2) == (0, 1, 2)

    def test_explicit_index(self, tmp_path: Path) -> None:
        """Explicit ``checkpoint_index=5`` writes to ``checkpoints/5.pt``."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx = save_checkpoint(
            tmp_path, models={"main": (model, spec)}, checkpoint_index=5
        )
        assert idx == 5
        assert (tmp_path / "models" / "main" / "checkpoints" / "5.pt").is_file()

    def test_spec_consistency_check_raises_on_mismatch(self, tmp_path: Path) -> None:
        """Saving a different spec under the same model name raises ValueError."""
        model_a = nn.Linear(4, 2)
        spec_a = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model_a, spec_a)})

        model_b = nn.Linear(8, 2)
        spec_b = create_model_spec(nn.Linear, in_features=8, out_features=2)
        with pytest.raises(ValueError, match="in_features"):
            save_checkpoint(tmp_path, models={"main": (model_b, spec_b)})

    def test_load_latest_from_manifest(self, tmp_path: Path) -> None:
        """Default load returns the latest checkpoint from the manifest."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        for i in (0, 2, 5):
            save_checkpoint(
                tmp_path, models={"main": (model, spec)}, checkpoint_index=i
            )

        # Overwrite index 5 with mutated weights.
        mutated = nn.Linear(4, 2)
        with torch.no_grad():
            mutated.weight.copy_(mutated.weight + 100.0)
        save_checkpoint(tmp_path, models={"main": (mutated, spec)}, checkpoint_index=5)

        result = load_checkpoint(tmp_path)
        reloaded, _ = result.models["main"]
        assert torch.allclose(reloaded.weight, mutated.weight)

    def test_load_explicit_index(self, tmp_path: Path) -> None:
        """Loading specific checkpoint indices returns the correct weights."""
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)

        model_a = nn.Linear(4, 2)
        model_b = nn.Linear(4, 2)
        with torch.no_grad():
            model_b.weight.copy_(model_b.weight + 10.0)

        save_checkpoint(tmp_path, models={"main": (model_a, spec)}, checkpoint_index=1)
        save_checkpoint(tmp_path, models={"main": (model_b, spec)}, checkpoint_index=2)

        loaded_a = load_checkpoint(tmp_path, checkpoint_index=1).models["main"][0]
        loaded_b = load_checkpoint(tmp_path, checkpoint_index=2).models["main"][0]
        assert torch.allclose(loaded_a.weight, model_a.weight)
        assert torch.allclose(loaded_b.weight, model_b.weight)

    def test_load_missing_manifest_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError when no manifest.json exists."""
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_checkpoint(tmp_path)

    def test_non_module_build_raises(self, tmp_path: Path) -> None:
        """Spec for a non-nn.Module class raises RuntimeError on load."""
        spec = create_model_spec(NotModule, arg_a=5, arg_b="hello")

        # Manually stage the directory layout so load_checkpoint can parse it.
        model_dir = tmp_path / "models" / "main"
        ckpt_dir = model_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (model_dir / "spec.json").write_text(spec.model_dump_json(indent=2))
        torch.save({}, ckpt_dir / "0.pt")
        manifest = {
            "schema_version": 1,
            "checkpoint_index": 0,
            "models": ["main"],
            "optimizers": [],
            "schedulers": [],
            "associations": {},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(RuntimeError, match="expected nn.Module"):
            load_checkpoint(tmp_path)

    def test_load_weights_only_true(self, tmp_path: Path) -> None:
        """Every ``torch.load`` call uses ``weights_only=True``."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        import nvalchemi.training._checkpoint as ckpt_mod

        real_load = ckpt_mod.torch.load
        with patch.object(ckpt_mod.torch, "load", wraps=real_load) as mock_load:
            load_checkpoint(tmp_path)

        assert mock_load.call_count >= 1
        for call in mock_load.call_args_list:
            assert call.kwargs.get("weights_only") is True, (
                f"torch.load called without weights_only=True: {call}"
            )


class TestMultiModel:
    """Two or more named models in a single checkpoint."""

    @staticmethod
    def _save_student_teacher(
        tmp_path: Path,
    ) -> tuple[nn.Module, nn.Module]:
        """Save a student/teacher pair and return the original modules."""
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        with torch.no_grad():
            teacher.weight.copy_(teacher.weight + 5.0)
        s_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        t_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(
            tmp_path,
            models={"student": (student, s_spec), "teacher": (teacher, t_spec)},
        )
        return student, teacher

    def test_save_load_two_models(self, tmp_path: Path) -> None:
        """Both student and teacher round-trip correctly."""
        student, teacher = self._save_student_teacher(tmp_path)
        result = load_checkpoint(tmp_path)
        assert set(result.models) == {"student", "teacher"}

        loaded_student, _ = result.models["student"]
        loaded_teacher, _ = result.models["teacher"]
        assert torch.equal(loaded_student.weight, student.weight)
        assert torch.equal(loaded_teacher.weight, teacher.weight)

    def test_models_have_independent_weights(self, tmp_path: Path) -> None:
        """Perturbed models are distinguishable after load."""
        student, teacher = self._save_student_teacher(tmp_path)
        result = load_checkpoint(tmp_path)
        loaded_s = result.models["student"][0]
        loaded_t = result.models["teacher"][0]
        # The teacher was shifted by +5.0 so they should differ.
        assert not torch.equal(loaded_s.weight, loaded_t.weight)

    def test_model_names_in_manifest(self, tmp_path: Path) -> None:
        """Manifest lists both model names."""
        self._save_student_teacher(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert sorted(manifest["models"]) == ["student", "teacher"]

    def test_model_subdirectories_exist(self, tmp_path: Path) -> None:
        """Per-model subdirectories are created under ``models/``."""
        self._save_student_teacher(tmp_path)
        assert (tmp_path / "models" / "student").is_dir()
        assert (tmp_path / "models" / "teacher").is_dir()


class TestOptimizerCheckpoint:
    """Optimizer state round-trip through the checkpoint layer."""

    @staticmethod
    def _train_steps(
        model: nn.Module, optimizer: torch.optim.Optimizer, n_steps: int
    ) -> None:
        """Run *n_steps* fake training steps to build up optimizer state."""
        for _ in range(n_steps):
            x = torch.randn(2, model.in_features)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_save_load_optimizer_state_dict(self, tmp_path: Path) -> None:
        """Optimizer state_dict round-trips through save/load."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.01, momentum=0.9)

        self._train_steps(model, optimizer, 3)
        original_state = optimizer.state_dict()

        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
        )
        result = load_checkpoint(tmp_path)
        loaded_opt, _ = result.optimizers["opt"]
        loaded_state = loaded_opt.state_dict()

        # Compare param_groups (excluding 'params' which are tensor ids).
        for orig_pg, loaded_pg in zip(
            original_state["param_groups"], loaded_state["param_groups"]
        ):
            for key in ("lr", "momentum", "weight_decay"):
                assert orig_pg[key] == loaded_pg[key]

    def test_optimizer_param_groups_preserved(self, tmp_path: Path) -> None:
        """LR, momentum, weight_decay survive round-trip."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.8, weight_decay=1e-4
        )
        opt_spec = create_model_spec(
            torch.optim.SGD, lr=0.05, momentum=0.8, weight_decay=1e-4
        )
        self._train_steps(model, optimizer, 1)

        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
        )
        result = load_checkpoint(tmp_path)
        loaded_pg = result.optimizers["opt"][0].param_groups[0]
        assert loaded_pg["lr"] == pytest.approx(0.05)
        assert loaded_pg["momentum"] == pytest.approx(0.8)
        assert loaded_pg["weight_decay"] == pytest.approx(1e-4)

    def test_optimizer_step_state_preserved(self, tmp_path: Path) -> None:
        """Momentum buffers match: original and reloaded produce same results."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.01, momentum=0.9)

        self._train_steps(model, optimizer, 5)

        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
        )
        result = load_checkpoint(tmp_path)
        loaded_model, _ = result.models["main"]
        loaded_opt, _ = result.optimizers["opt"]

        # Run M more steps on both and verify weights converge identically.
        torch.manual_seed(99)
        inputs = [torch.randn(2, 4) for _ in range(3)]

        for x in inputs:
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for x in inputs:
            loss = loaded_model(x).sum()
            loss.backward()
            loaded_opt.step()
            loaded_opt.zero_grad()

        for p_orig, p_loaded in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p_orig, p_loaded, atol=1e-6)


class TestSchedulerCheckpoint:
    """Scheduler state round-trip through the checkpoint layer."""

    def test_save_load_scheduler_state_dict(self, tmp_path: Path) -> None:
        """CosineAnnealingLR state_dict round-trips."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        sched_spec = create_model_spec(
            torch.optim.lr_scheduler.CosineAnnealingLR, T_max=10
        )

        for _ in range(5):
            scheduler.step()
        original_state = scheduler.state_dict()

        associations = {"main": {"optimizers": ["opt"], "schedulers": ["sched"]}}
        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
            schedulers={"sched": (scheduler, sched_spec)},
            associations=associations,
        )
        result = load_checkpoint(tmp_path)
        loaded_sched, _ = result.schedulers["sched"]
        loaded_state = loaded_sched.state_dict()

        for key in original_state:
            assert original_state[key] == loaded_state[key], (
                f"scheduler state key {key!r} differs"
            )

    def test_lr_trajectory_preserved(self, tmp_path: Path) -> None:
        """LR trajectory matches after reload: step N, save, reload, step M more."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        sched_spec = create_model_spec(
            torch.optim.lr_scheduler.CosineAnnealingLR, T_max=20
        )

        # Step N=5 times before save.
        for _ in range(5):
            scheduler.step()

        associations = {"main": {"optimizers": ["opt"], "schedulers": ["sched"]}}
        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
            schedulers={"sched": (scheduler, sched_spec)},
            associations=associations,
        )
        result = load_checkpoint(tmp_path)
        loaded_sched, _ = result.schedulers["sched"]

        # Step M=10 more times on both; LR must match at every step.
        for step in range(10):
            scheduler.step()
            loaded_sched.step()
            lr_orig = scheduler.get_last_lr()[0]
            lr_loaded = loaded_sched.get_last_lr()[0]
            assert lr_orig == pytest.approx(lr_loaded), (
                f"LR mismatch at step {step}: {lr_orig} vs {lr_loaded}"
            )


class TestAssociations:
    """Model-to-optimizer-to-scheduler linkage via associations."""

    def test_associations_stored_in_manifest(self, tmp_path: Path) -> None:
        """Associations dict appears in manifest.json."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.01)

        associations = {"main": {"optimizers": ["opt"], "schedulers": []}}
        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
            associations=associations,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["associations"] == associations

    def test_load_wires_optimizer_to_correct_model(self, tmp_path: Path) -> None:
        """Optimizer param groups reference the associated model's parameters."""
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        s_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        t_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.01)

        associations = {"student": {"optimizers": ["s_opt"], "schedulers": []}}
        save_checkpoint(
            tmp_path,
            models={
                "student": (student, s_spec),
                "teacher": (teacher, t_spec),
            },
            optimizers={"s_opt": (optimizer, opt_spec)},
            associations=associations,
        )
        result = load_checkpoint(tmp_path)
        loaded_opt, _ = result.optimizers["s_opt"]
        loaded_student, _ = result.models["student"]

        # The optimizer's param groups should reference the loaded student's
        # parameters (same data pointers).
        opt_param_ids = {id(p) for pg in loaded_opt.param_groups for p in pg["params"]}
        student_param_ids = {id(p) for p in loaded_student.parameters()}
        assert opt_param_ids == student_param_ids

    def test_load_wires_scheduler_to_correct_optimizer(self, tmp_path: Path) -> None:
        """Scheduler references the correct optimizer on load."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        sched_spec = create_model_spec(
            torch.optim.lr_scheduler.CosineAnnealingLR, T_max=10
        )

        associations = {"main": {"optimizers": ["opt"], "schedulers": ["sched"]}}
        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
            schedulers={"sched": (scheduler, sched_spec)},
            associations=associations,
        )
        result = load_checkpoint(tmp_path)
        loaded_sched, _ = result.schedulers["sched"]
        loaded_opt, _ = result.optimizers["opt"]

        # The scheduler's internal optimizer should be the loaded one.
        assert loaded_sched.optimizer is loaded_opt

    def test_single_model_fallback_no_associations(self, tmp_path: Path) -> None:
        """One model + one optimizer, no associations: load succeeds via fallback."""
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        opt_spec = create_model_spec(torch.optim.SGD, lr=0.01)

        # No associations at all.
        save_checkpoint(
            tmp_path,
            models={"main": (model, m_spec)},
            optimizers={"opt": (optimizer, opt_spec)},
        )
        result = load_checkpoint(tmp_path)
        assert "opt" in result.optimizers

    def test_scheduler_attaches_to_second_optimizer(self, tmp_path: Path) -> None:
        """Regression: scheduler must wrap the optimizer it was saved with.

        With multiple optimizers on the same model, the scheduler must attach
        to the specific optimizer it was constructed with, not simply the
        first optimizer in the manifest's association list. Auto-inference
        cannot disambiguate two optimizers sharing parameters, so explicit
        associations are required to pin the scheduler to the SGD optimizer.
        """
        model = nn.Linear(4, 2)
        m_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)

        # Two optimizers with different classes to make identity unambiguous.
        opt_adam = torch.optim.Adam(model.parameters(), lr=0.01)
        opt_sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        adam_spec = create_model_spec(torch.optim.Adam, lr=0.01)
        sgd_spec = create_model_spec(torch.optim.SGD, lr=0.1)

        # Scheduler on the SGD optimizer (NOT Adam).
        scheduler = torch.optim.lr_scheduler.StepLR(opt_sgd, step_size=10, gamma=0.5)
        sched_spec = create_model_spec(
            torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.5
        )

        # Explicit associations: the scheduler must wrap ``sgd``. Listing
        # ``sgd`` first among optimizers ensures the load-time wiring logic
        # picks it as the scheduler's optimizer.
        associations = {
            "m": {"optimizers": ["sgd", "adam"], "schedulers": ["step"]},
        }
        save_checkpoint(
            tmp_path,
            models={"m": (model, m_spec)},
            optimizers={"adam": (opt_adam, adam_spec), "sgd": (opt_sgd, sgd_spec)},
            schedulers={"step": (scheduler, sched_spec)},
            associations=associations,
        )

        result = load_checkpoint(tmp_path)
        loaded_scheduler, _ = result.schedulers["step"]
        loaded_sgd, _ = result.optimizers["sgd"]
        loaded_adam, _ = result.optimizers["adam"]

        assert loaded_scheduler.optimizer is loaded_sgd, (
            "scheduler should wrap the SGD optimizer it was saved with"
        )
        assert loaded_scheduler.optimizer is not loaded_adam

        # Verify the manifest recorded the association correctly.
        assert "step" in result.associations.get("m", {}).get("schedulers", [])


class TestCheckpointCustomMLPBlock:
    """Stress tests: serialize a non-trivial custom block end-to-end.

    The target is :class:`CustomMLPBlock` --- a pre-norm MLP wrapping a custom
    :class:`SwiGLU` activation, an expansion/projection :class:`Linear` pair,
    :class:`LayerNorm`, and :class:`Dropout`. These tests exercise the spec +
    checkpoint pipeline against a module that mixes several param types
    (weights, bias, learnable activation scale, LayerNorm affine params) and
    several kwarg types (int, float, bool, :class:`torch.dtype`).
    """

    @staticmethod
    def _make_spec(**overrides: object) -> tuple[dict[str, object], object]:
        """Build default CustomMLPBlock kwargs + spec with optional overrides."""
        kwargs: dict[str, object] = {
            "in_features": 8,
            "hidden_features": 16,
            "dropout": 0.25,
            "eps": 1e-6,
            "activation_scale": 0.5,
            "use_residual": True,
            "dtype": torch.float32,
        }
        kwargs.update(overrides)
        return kwargs, create_model_spec(CustomMLPBlock, **kwargs)

    def test_roundtrip_preserves_all_params(self, tmp_path: Path) -> None:
        """All named parameters survive a save/load round-trip bit-exactly."""
        kwargs, spec = self._make_spec()
        model = CustomMLPBlock(**kwargs)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        save_checkpoint(tmp_path, models={"main": (model, spec)})
        result = load_checkpoint(tmp_path)
        reloaded, reloaded_spec = result.models["main"]
        assert isinstance(reloaded, CustomMLPBlock)
        assert reloaded_spec.timestamp == spec.timestamp

        original_params = dict(model.named_parameters())
        reloaded_params = dict(reloaded.named_parameters())
        assert set(original_params) == set(reloaded_params)
        for name, tensor in original_params.items():
            assert torch.equal(reloaded_params[name], tensor), (
                f"parameter {name!r} differs after round-trip"
            )

    def test_roundtrip_preserves_forward_output(self, tmp_path: Path) -> None:
        """Forward pass output is identical after round-trip."""
        kwargs, spec = self._make_spec(dropout=0.5)
        model = CustomMLPBlock(**kwargs).eval()
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        result = load_checkpoint(tmp_path)
        reloaded, _ = result.models["main"]
        reloaded.eval()

        x = torch.randn(4, kwargs["in_features"])
        with torch.no_grad():
            y_original = model(x)
            y_reloaded = reloaded(x)
        assert torch.equal(y_original, y_reloaded)

    def test_spec_json_is_pure_json(self, tmp_path: Path) -> None:
        """spec.json contains only JSON-native types; no pickled blobs."""
        kwargs, spec = self._make_spec()
        model = CustomMLPBlock(**kwargs)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        raw = (tmp_path / "models" / "main" / "spec.json").read_text()
        parsed = json.loads(raw)
        assert parsed["dtype"] == "torch.float32"
        for key in ("cls_path", "timestamp"):
            assert key in parsed
        assert parsed["cls_path"].endswith(".CustomMLPBlock")

    def test_dtype_kwarg_round_trips(self, tmp_path: Path) -> None:
        """torch.float64 dtype kwarg survives JSON round-trip."""
        kwargs, spec = self._make_spec(dtype=torch.float64)
        model = CustomMLPBlock(**kwargs)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        result = load_checkpoint(tmp_path)
        _, reloaded_spec = result.models["main"]
        reloaded = result.models["main"][0]
        assert reloaded_spec.dtype is torch.float64
        assert reloaded.expand.weight.dtype is torch.float64
        assert reloaded.norm.weight.dtype is torch.float64

    def test_activation_parameter_checkpointed(self, tmp_path: Path) -> None:
        """Learnable ``SwiGLU.scale`` survives the round-trip."""
        kwargs, spec = self._make_spec(activation_scale=0.5)
        model = CustomMLPBlock(**kwargs)
        with torch.no_grad():
            model.activation.scale.fill_(7.5)

        save_checkpoint(tmp_path, models={"main": (model, spec)})
        result = load_checkpoint(tmp_path)
        reloaded, _ = result.models["main"]
        assert torch.equal(reloaded.activation.scale, torch.tensor(7.5))

    def test_autoincrement_multiple_checkpoints(self, tmp_path: Path) -> None:
        """Autoincrement + per-index reload preserves correct weights."""
        kwargs, spec = self._make_spec()
        model = CustomMLPBlock(**kwargs)

        snapshots: list[torch.Tensor] = []
        for step in range(3):
            with torch.no_grad():
                model.project.weight.add_(float(step) + 1.0)
            snapshots.append(model.project.weight.detach().clone())
            idx = save_checkpoint(tmp_path, models={"main": (model, spec)})
            assert idx == step

        for step, snapshot in enumerate(snapshots):
            result = load_checkpoint(tmp_path, checkpoint_index=step)
            reloaded, _ = result.models["main"]
            assert torch.equal(reloaded.project.weight, snapshot), (
                f"checkpoint {step} did not reload its own weights"
            )

    def test_hyperparameter_mismatch_raises(self, tmp_path: Path) -> None:
        """Saving a second spec with different hyperparameters must fail."""
        kwargs_a, spec_a = self._make_spec(hidden_features=16)
        model_a = CustomMLPBlock(**kwargs_a)
        save_checkpoint(tmp_path, models={"main": (model_a, spec_a)})

        kwargs_b, spec_b = self._make_spec(hidden_features=32)
        model_b = CustomMLPBlock(**kwargs_b)
        with pytest.raises(ValueError, match="hidden_features"):
            save_checkpoint(tmp_path, models={"main": (model_b, spec_b)})


class TestSecurityAST:
    """AST-level security invariants for ``_checkpoint.py``."""

    _CHECKPOINT_PATH = (
        Path(__file__).resolve().parents[2]
        / "nvalchemi"
        / "training"
        / "_checkpoint.py"
    )
    _FORBIDDEN_MODULES = frozenset({"pickle", "cloudpickle", "dill", "marshal"})

    def _tree(self) -> ast.AST:
        return ast.parse(self._CHECKPOINT_PATH.read_text())

    def test_no_pickle_imports(self) -> None:
        """No imports of pickle, cloudpickle, dill, or marshal."""
        tree = self._tree()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in self._FORBIDDEN_MODULES, (
                        f"_checkpoint.py:{node.lineno} imports forbidden "
                        f"module {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root = node.module.split(".")[0]
                assert root not in self._FORBIDDEN_MODULES, (
                    f"_checkpoint.py:{node.lineno} imports from forbidden "
                    f"module {node.module!r}"
                )

    def test_torch_load_always_weights_only(self) -> None:
        """Every ``torch.load(...)`` call has ``weights_only=True``."""
        tree = self._tree()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            ):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
            assert "weights_only" in kw, (
                f"_checkpoint.py:{node.lineno} torch.load() missing weights_only= kwarg"
            )
            val = kw["weights_only"]
            assert isinstance(val, ast.Constant) and val.value is True, (
                f"_checkpoint.py:{node.lineno} torch.load(weights_only=...) "
                f"must be literal True, got {ast.dump(val)}"
            )

    def test_torch_save_uses_state_dict(self) -> None:
        """``torch.save`` never receives a raw module/optimizer object.

        The implementation extracts ``state_dict()`` in the caller and
        passes the dict to ``_save_component``, which calls ``torch.save``
        with a plain variable. We verify that no ``torch.save`` call has
        a first argument that is a bare attribute access on ``self``
        (e.g., ``torch.save(model, ...)`` or ``torch.save(self.model, ...)``)
        --- only plain names (like ``state_dict``) or subscripts are
        acceptable.
        """
        tree = self._tree()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "save"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            ):
                continue
            assert node.args, (
                f"_checkpoint.py:{node.lineno} torch.save() called with no args"
            )
            first = node.args[0]
            # Must NOT be a bare attribute on self (which would indicate
            # saving a raw object).  Acceptable: Name, Subscript, or Call.
            if isinstance(first, ast.Attribute):
                assert not (
                    isinstance(first.value, ast.Name) and first.value.id == "self"
                ), (
                    f"_checkpoint.py:{node.lineno} torch.save() first arg "
                    f"appears to be a raw object (self.{first.attr}), "
                    f"expected a state_dict result"
                )


class TestSchemaVersion:
    """Manifest schema versioning and forward-compatibility guard."""

    def test_save_writes_schema_version(self, tmp_path: Path) -> None:
        """``manifest.json`` contains ``schema_version`` after save."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "schema_version" in manifest
        assert manifest["schema_version"] == 1

    def test_load_v0_manifest_without_schema_key(self, tmp_path: Path) -> None:
        """A manifest missing ``schema_version`` (v0) loads successfully."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        # Strip schema_version to simulate a v0 manifest.
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("schema_version")
        manifest_path.write_text(json.dumps(manifest, indent=2))

        result = load_checkpoint(tmp_path)
        assert "main" in result.models
        reloaded, _ = result.models["main"]
        assert torch.equal(reloaded.weight, model.weight)

    def test_future_schema_version_raises(self, tmp_path: Path) -> None:
        """A manifest with a newer schema version raises ``ValueError``."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        # Bump to a future version.
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = 999
        manifest_path.write_text(json.dumps(manifest, indent=2))

        with pytest.raises(ValueError, match="newer than supported"):
            load_checkpoint(tmp_path)

    def test_schema_version_preserved_across_saves(self, tmp_path: Path) -> None:
        """Successive saves always write the current schema version."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        for _ in range(3):
            save_checkpoint(tmp_path, models={"main": (model, spec)})

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["schema_version"] == 1

    def test_manifest_pydantic_validation(self, tmp_path: Path) -> None:
        """Malformed manifest.json triggers Pydantic ``ValidationError``."""
        from pydantic import ValidationError

        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        # Corrupt the manifest: models should be list[str], not a string.
        manifest_path = tmp_path / "manifest.json"
        raw = json.loads(manifest_path.read_text())
        raw["models"] = "not-a-list"
        manifest_path.write_text(json.dumps(raw))

        with pytest.raises(ValidationError):
            load_checkpoint(tmp_path)

    def test_manifest_model_dump_roundtrip(self) -> None:
        """``CheckpointManifest`` round-trips through JSON serialization."""
        original = CheckpointManifest(
            checkpoint_index=3,
            models=["a", "b"],
            optimizers=["opt_a"],
            schedulers=[],
            associations={"a": {"optimizers": ["opt_a"], "schedulers": []}},
        )
        dumped = original.model_dump_json()
        restored = CheckpointManifest.model_validate_json(dumped)
        assert restored == original


class TestCheckpointGPU:
    """GPU-specific checkpoint round-trip tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_load_model_on_gpu(self, tmp_path: Path) -> None:
        """Round-trip a model whose parameters live on CUDA."""
        device = torch.device("cuda")
        model = nn.Linear(4, 2).to(device)
        with torch.no_grad():
            model.weight.add_(1.25)
            model.bias.add_(-0.5)

        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        idx = save_checkpoint(tmp_path, models={"main": (model, spec)})
        assert idx == 0

        # Verify saved tensors are CUDA-resident.
        saved = torch.load(
            tmp_path / "models" / "main" / "checkpoints" / "0.pt",
            weights_only=True,
        )
        assert saved["weight"].is_cuda
        assert saved["bias"].is_cuda

        result = load_checkpoint(tmp_path)
        reloaded, _ = result.models["main"]
        assert torch.allclose(reloaded.weight.cpu(), model.weight.cpu())
        assert torch.allclose(reloaded.bias.cpu(), model.bias.cpu())


class TestDtypeRoundtrip:
    """Regression: ``torch.dtype`` kwargs rehydrate as real dtype objects."""

    def test_dtype_kwarg_roundtrip(self, tmp_path: Path) -> None:
        """A spec with a ``torch.dtype`` kwarg round-trips bit-exactly.

        Atul's concern A3: when a spec carries a ``torch.dtype`` (e.g.
        ``torch.float32``), the saved ``spec.json`` uses a string
        representation, but on load the field must rehydrate to the
        actual :class:`torch.dtype` so that ``spec.build()`` can hand it
        to modules expecting a dtype (not a string).
        """
        spec = create_model_spec(
            nn.Linear, in_features=4, out_features=2, dtype=torch.float32
        )
        model = spec.build()
        assert model.weight.dtype == torch.float32

        save_checkpoint(tmp_path, models={"m": (model, spec)})
        result = load_checkpoint(tmp_path)
        loaded_model, loaded_spec = result.models["m"]

        assert loaded_model.weight.dtype == torch.float32
        # The spec's dtype field must rehydrate as a torch.dtype (not a string).
        assert loaded_spec.dtype == torch.float32
        assert isinstance(loaded_spec.dtype, torch.dtype)


class TestEMACheckpoint:
    """Tests for round-tripping EMA (``AveragedModel``) wrappers.

    ``torch.optim.swa_utils.AveragedModel`` takes the base model as a
    positional ``__init__`` argument; :func:`create_model_spec` only
    accepts kwargs. The supported workflow is therefore to save the base
    model (and optionally the inner averaged module) as ordinary
    :class:`nn.Module`\\ s, then reconstruct the ``AveragedModel`` wrapper
    in user code after loading.
    """

    def test_ema_base_model_roundtrip(self, tmp_path: Path) -> None:
        """Save base + EMA inner module, reconstruct EMA wrapper on load."""
        from torch.optim.swa_utils import AveragedModel

        base = nn.Linear(4, 2)
        ema = AveragedModel(base)
        # Simulate training: perturb base weights, then update EMA.
        for _ in range(3):
            with torch.no_grad():
                base.weight.add_(torch.randn_like(base.weight) * 0.1)
            ema.update_parameters(base)

        base_spec = create_model_spec(nn.Linear, in_features=4, out_features=2)

        # Save both the base and the EMA's inner module. ``ema.module`` is
        # a plain ``nn.Linear`` with the averaged state_dict.
        save_checkpoint(
            tmp_path,
            models={
                "base": (base, base_spec),
                "ema_inner": (ema.module, base_spec),
            },
        )

        result = load_checkpoint(tmp_path)
        loaded_base, _ = result.models["base"]
        loaded_ema_inner, _ = result.models["ema_inner"]

        # Base and EMA inner weights round-trip.
        assert torch.allclose(loaded_base.weight, base.weight)
        assert torch.allclose(loaded_ema_inner.weight, ema.module.weight)

        # Reconstruct the EMA wrapper: copy averaged weights into the new
        # ``AveragedModel``'s inner module so forward-pass output matches.
        reconstructed_ema = AveragedModel(loaded_base)
        reconstructed_ema.module.load_state_dict(loaded_ema_inner.state_dict())
        x = torch.randn(1, 4)
        with torch.no_grad():
            assert torch.allclose(reconstructed_ema(x), ema(x))


class TestLoadCheckpointKwargs:
    """Tests for the ``map_location`` and ``model_names`` kwargs."""

    def test_map_location_cpu(self, tmp_path: Path) -> None:
        """``map_location='cpu'`` places the loaded model on CPU."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"m": (model, spec)})

        result = load_checkpoint(tmp_path, map_location="cpu")
        loaded, _ = result.models["m"]
        assert loaded.weight.device.type == "cpu"
        # State-dict tensors must also be on CPU.
        for v in loaded.state_dict().values():
            assert v.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_map_location_cuda(self, tmp_path: Path) -> None:
        """``map_location='cuda'`` places the loaded model on CUDA."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"m": (model, spec)})

        result = load_checkpoint(tmp_path, map_location="cuda")
        loaded, _ = result.models["m"]
        assert loaded.weight.device.type == "cuda"

    def test_model_names_loads_only_specified_model(self, tmp_path: Path) -> None:
        """``model_names`` restricts loading to the selected models only."""
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(
            tmp_path,
            models={"student": (m1, spec), "teacher": (m2, spec)},
        )

        result = load_checkpoint(tmp_path, model_names={"teacher"})
        assert list(result.models.keys()) == ["teacher"]
        assert result.optimizers == {}
        assert result.schedulers == {}
        # Associations on the result remain informational (reflect on-disk state).
        assert isinstance(result.associations, dict)

    def test_model_names_multi_select(self, tmp_path: Path) -> None:
        """``model_names`` with multiple names loads all of them and their
        associated optimizers/schedulers (union)."""
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        m3 = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)

        opt1 = torch.optim.Adam(m1.parameters(), lr=0.01)
        opt2 = torch.optim.Adam(m2.parameters(), lr=0.02)
        opt_spec_1 = create_model_spec(torch.optim.Adam, lr=0.01)
        opt_spec_2 = create_model_spec(torch.optim.Adam, lr=0.02)

        save_checkpoint(
            tmp_path,
            models={"a": (m1, spec), "b": (m2, spec), "c": (m3, spec)},
            optimizers={"a_opt": (opt1, opt_spec_1), "b_opt": (opt2, opt_spec_2)},
        )

        result = load_checkpoint(tmp_path, model_names={"a", "b"})
        assert set(result.models.keys()) == {"a", "b"}
        assert "c" not in result.models
        # Both associated optimizers come along, c's (nonexistent) does not.
        assert set(result.optimizers.keys()) == {"a_opt", "b_opt"}

    def test_model_names_includes_associated_components(self, tmp_path: Path) -> None:
        """``model_names`` also loads the associated optimizers/schedulers."""
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)

        # Give the student an optimizer; teacher gets nothing.
        opt1 = torch.optim.Adam(m1.parameters(), lr=0.01)
        opt_spec = create_model_spec(torch.optim.Adam, lr=0.01)

        save_checkpoint(
            tmp_path,
            models={"student": (m1, spec), "teacher": (m2, spec)},
            optimizers={"s_opt": (opt1, opt_spec)},
        )

        # Loading student pulls in its associated optimizer.
        result = load_checkpoint(tmp_path, model_names={"student"})
        assert list(result.models.keys()) == ["student"]
        assert "s_opt" in result.optimizers

        # Loading teacher picks up no optimizer (none associated).
        result = load_checkpoint(tmp_path, model_names={"teacher"})
        assert list(result.models.keys()) == ["teacher"]
        assert result.optimizers == {}

    def test_model_names_unknown_raises_keyerror(self, tmp_path: Path) -> None:
        """Unknown names in ``model_names`` raise :class:`KeyError` listing them."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(
            tmp_path,
            models={"student": (model, spec), "teacher": (model, spec)},
        )

        with pytest.raises(KeyError, match="nonexistent"):
            load_checkpoint(tmp_path, model_names={"nonexistent"})

        # Multiple unknowns — both should be reported.
        with pytest.raises(KeyError) as excinfo:
            load_checkpoint(tmp_path, model_names={"nonexistent", "ghost"})
        msg = str(excinfo.value)
        assert "nonexistent" in msg
        assert "ghost" in msg
        # The error message must list the available model names.
        assert "student" in msg
        assert "teacher" in msg


class TestStrategyCheckpoint:
    """High-level strategy checkpoint round-trip behavior."""

    def test_strategy_checkpoint_loads_builtin_result_and_restart_state(
        self, tmp_path: Path
    ) -> None:
        """Saving a strategy returns a restartable high-level load dict."""
        strategy = _make_checkpoint_strategy(num_steps=4)
        strategy.train_batch(_make_checkpoint_batch(seed=1))
        assert strategy.step_count == 1
        assert strategy.global_step_count == 1
        assert strategy.batch_count == 1

        idx = save_checkpoint(tmp_path, strategy=strategy)
        assert idx == 0
        assert (tmp_path / "strategy.json").is_file()

        loaded = load_checkpoint(tmp_path)
        assert isinstance(loaded, dict)
        assert loaded["checkpoint_index"] == 0
        assert loaded["source"]["format"] == "native"

        restored = loaded["strategy"]
        assert isinstance(restored, TrainingStrategy)
        assert restored.step_count == 1
        assert restored.global_step_count == 1
        assert restored.batch_count == 1
        assert restored.epoch_count == strategy.epoch_count
        assert restored.epoch_step_count == strategy.epoch_step_count
        assert restored.single_model_input is True

        main_entry = loaded["models"]["main"]
        assert main_entry["model"] is restored.models["main"]
        assert set(main_entry["optimizers"]) == {"main_optimizer"}
        assert set(main_entry["schedulers"]) == {"main_scheduler"}
        assert restored._resume_optimizer_state is True
        assert restored._optimizers[0].state_dict()["state"]
        assert restored._lr_schedulers[0].state_dict()["last_epoch"] == 1

        restored.run(
            [
                _make_checkpoint_batch(seed=2),
                _make_checkpoint_batch(seed=3),
                _make_checkpoint_batch(seed=4),
            ]
        )
        assert restored.step_count == 4
        assert restored.global_step_count == 4

    def test_loaded_strategy_reuses_restored_optimizer_state(
        self, tmp_path: Path
    ) -> None:
        """A resumed run advances checkpointed optimizer state instead of resetting."""
        strategy = _make_checkpoint_strategy(num_steps=2)
        strategy.run(
            [
                _make_checkpoint_batch(seed=1),
                _make_checkpoint_batch(seed=2),
            ]
        )
        saved_steps = [
            float(state["step"])
            for state in strategy._optimizers[0].state.values()
            if "step" in state
        ]
        assert saved_steps
        save_checkpoint(tmp_path, strategy=strategy)

        loaded = load_checkpoint(tmp_path)
        restored = loaded["strategy"]
        restored.num_steps = 3
        restored.run([_make_checkpoint_batch(seed=3)])
        resumed_steps = [
            float(state["step"])
            for state in restored._optimizers[0].state.values()
            if "step" in state
        ]

        assert resumed_steps
        assert min(resumed_steps) == min(saved_steps) + 1.0

    def test_restored_strategy_can_save_next_checkpoint(self, tmp_path: Path) -> None:
        """A resumed strategy can continue writing checkpoints in the same root."""
        strategy = _make_checkpoint_strategy(num_steps=4)
        strategy.train_batch(_make_checkpoint_batch(seed=1))
        save_checkpoint(tmp_path, strategy=strategy)

        loaded = load_checkpoint(tmp_path)
        restored = loaded["strategy"]
        restored.run(
            [
                _make_checkpoint_batch(seed=2),
                _make_checkpoint_batch(seed=3),
                _make_checkpoint_batch(seed=4),
            ]
        )

        idx = save_checkpoint(tmp_path, strategy=restored)
        assert idx == 1
        assert (tmp_path / "models" / "main" / "checkpoints" / "1.pt").is_file()
        assert (
            tmp_path / "optimizers" / "main_optimizer" / "checkpoints" / "1.pt"
        ).is_file()
        assert (
            tmp_path / "schedulers" / "main_scheduler" / "checkpoints" / "1.pt"
        ).is_file()

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["checkpoint_index"] == 1
        strategy_metadata = json.loads((tmp_path / "strategy.json").read_text())
        assert strategy_metadata["runtime_state"]["step_count"] == 4
        assert strategy_metadata["runtime_state"]["global_step_count"] == 4
        indexed_metadata = json.loads(
            (tmp_path / "strategy" / "checkpoints" / "1.json").read_text()
        )
        assert indexed_metadata["runtime_state"]["step_count"] == 4
        assert indexed_metadata["runtime_state"]["global_step_count"] == 4

        reloaded = load_checkpoint(tmp_path, checkpoint_index=1)
        assert reloaded["strategy"].step_count == 4
        assert reloaded["strategy"].global_step_count == 4

    def test_strategy_metadata_is_loaded_by_checkpoint_index(
        self, tmp_path: Path
    ) -> None:
        """Explicit checkpoint indices restore matching strategy counters."""
        strategy = _make_checkpoint_strategy(num_steps=4)
        strategy.train_batch(_make_checkpoint_batch(seed=1))
        save_checkpoint(tmp_path, strategy=strategy)

        loaded = load_checkpoint(tmp_path)
        restored = loaded["strategy"]
        restored.run(
            [
                _make_checkpoint_batch(seed=2),
                _make_checkpoint_batch(seed=3),
                _make_checkpoint_batch(seed=4),
            ]
        )
        save_checkpoint(tmp_path, strategy=restored)

        loaded_first = load_checkpoint(tmp_path, checkpoint_index=0)
        loaded_second = load_checkpoint(tmp_path, checkpoint_index=1)

        assert loaded_first["strategy"].step_count == 1
        assert loaded_second["strategy"].step_count == 4
        assert (tmp_path / "strategy" / "checkpoints" / "0.json").is_file()
        assert (tmp_path / "strategy" / "checkpoints" / "1.json").is_file()

    def test_multi_optimizer_strategy_schedulers_keep_optimizer_edges(
        self, tmp_path: Path
    ) -> None:
        """Schedulers in multi-optimizer strategies reload on the right optimizer."""
        strategy = _make_multi_optimizer_checkpoint_strategy()
        save_checkpoint(tmp_path, strategy=strategy)

        loaded = load_checkpoint(tmp_path)
        restored = loaded["strategy"]

        assert set(loaded["models"]["main"]["optimizers"]) == {
            "main_optimizer_0",
            "main_optimizer_1",
        }
        assert set(loaded["models"]["main"]["schedulers"]) == {
            "main_scheduler_0",
            "main_scheduler_1",
        }
        assert restored._lr_schedulers[0] is not None
        assert restored._lr_schedulers[1] is not None
        assert restored._lr_schedulers[0].optimizer is restored._optimizers[0]
        assert restored._lr_schedulers[1].optimizer is restored._optimizers[1]

    def test_register_hook_preserves_loaded_optimizer_state(
        self, tmp_path: Path
    ) -> None:
        """Adding observer hooks after load does not discard optimizer state."""
        strategy = _make_checkpoint_strategy(num_steps=2)
        strategy.train_batch(_make_checkpoint_batch(seed=1))
        save_checkpoint(tmp_path, strategy=strategy)

        restored = load_checkpoint(tmp_path)["strategy"]
        assert restored._resume_optimizer_state is True

        restored.register_hook(_NoOpCheckpointHook())

        assert restored._resume_optimizer_state is True

    def test_map_location_overrides_restored_strategy_device(
        self, tmp_path: Path
    ) -> None:
        """A CPU map_location keeps the restored strategy runnable on CPU."""
        strategy = _make_checkpoint_strategy(num_steps=2)
        strategy.train_batch(_make_checkpoint_batch(seed=1))
        save_checkpoint(tmp_path, strategy=strategy)

        for metadata_path in (
            tmp_path / "strategy.json",
            tmp_path / "strategy" / "checkpoints" / "0.json",
        ):
            metadata = json.loads(metadata_path.read_text())
            metadata["devices"] = ["cuda"]
            metadata_path.write_text(json.dumps(metadata))

        restored = load_checkpoint(tmp_path, map_location="cpu")["strategy"]

        assert restored.devices == [torch.device("cpu")]
        assert all(
            parameter.device.type == "cpu"
            for parameter in restored.models["main"].parameters()
        )
        optimizer_params = restored._optimizers[0].param_groups[0]["params"]
        assert all(parameter.device.type == "cpu" for parameter in optimizer_params)

    def test_strategy_can_be_second_positional_argument(self, tmp_path: Path) -> None:
        """``save_checkpoint(path, strategy)`` is accepted as high-level UX."""
        strategy = _make_checkpoint_strategy(num_steps=2)
        idx = save_checkpoint(tmp_path, strategy)
        assert idx == 0
        loaded = load_checkpoint(tmp_path)
        assert isinstance(loaded["strategy"], TrainingStrategy)

    def test_strategy_methods_save_and_load_restartable_checkpoint(
        self, tmp_path: Path
    ) -> None:
        """``TrainingStrategy`` exposes one-off save/load checkpoint helpers."""
        strategy = _make_checkpoint_strategy(num_steps=3)
        strategy.train_batch(_make_checkpoint_batch(seed=1))

        idx = strategy.save_checkpoint(tmp_path)

        assert idx == 0
        restored = TrainingStrategy.load_checkpoint(tmp_path, map_location="cpu")
        assert isinstance(restored, TrainingStrategy)
        assert restored.step_count == 1
        assert restored.batch_count == 1
        assert restored._resume_optimizer_state is True

        restored.run(
            [
                _make_checkpoint_batch(seed=2),
                _make_checkpoint_batch(seed=3),
            ]
        )
        assert restored.step_count == 3

        idx = restored.save_checkpoint(tmp_path)
        assert idx == 1
        reloaded = TrainingStrategy.load_checkpoint(tmp_path, checkpoint_index=1)
        assert reloaded.step_count == 3
        assert reloaded.batch_count == 3

    def test_strategy_load_checkpoint_requires_strategy_metadata(
        self, tmp_path: Path
    ) -> None:
        """The strategy convenience loader rejects component-only checkpoints."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        with pytest.raises(
            ValueError, match="checkpoint saved from a TrainingStrategy"
        ):
            TrainingStrategy.load_checkpoint(tmp_path)

    def test_validator_callback_wraps_failures(self, tmp_path: Path) -> None:
        """Validators receive model entries and errors name the checkpoint/model."""
        model = nn.Linear(4, 2)
        spec = create_model_spec(nn.Linear, in_features=4, out_features=2)
        save_checkpoint(tmp_path, models={"main": (model, spec)})

        seen: list[str] = []

        def passing_validator(
            model_name: str,
            entry: dict[str, Any],
            loaded: dict[str, Any],
        ) -> None:
            seen.append(model_name)
            assert isinstance(entry["model"], nn.Linear)
            assert loaded["source"]["path"] == str(tmp_path)

        loaded = load_checkpoint(tmp_path, validators=[passing_validator])
        assert isinstance(loaded, CheckpointManifest)
        assert seen == ["main"]

        def failing_validator(
            model_name: str,
            entry: dict[str, Any],
            loaded: dict[str, Any],
        ) -> None:
            del model_name, entry, loaded
            raise RuntimeError("unsupported elements")

        with pytest.raises(ValueError, match="unsupported elements"):
            load_checkpoint(tmp_path, validators=[failing_validator])

    def test_mace_adapter_uses_training_safe_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """MACE adapter loads local files with cuEq/compile disabled by default."""
        checkpoint_path = tmp_path / "mace.pt"
        checkpoint_path.write_bytes(b"placeholder")
        wrapper = nn.Linear(1, 1)

        with patch(
            "nvalchemi.models.mace.MACEWrapper.from_checkpoint",
            return_value=wrapper,
        ) as mocked:
            with pytest.warns(UserWarning, match="trusted"):
                loaded = load_checkpoint(
                    checkpoint_path,
                    adapter="mace",
                    adapter_kwargs={"dtype": torch.float32},
                )

        assert loaded["strategy"] is None
        assert loaded["models"]["main"]["model"] is wrapper
        assert loaded["models"]["main"]["spec"] is None
        assert loaded["source"]["format"] == "mace"
        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        assert kwargs["enable_cueq"] is False
        assert kwargs["compile_model"] is False
        assert kwargs["dtype"] is torch.float32
