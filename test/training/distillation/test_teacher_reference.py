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
"""Tests for once-per-root teacher checkpoints of :mod:`nvalchemi.training.distillation`."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStrategy,
    _checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from nvalchemi.training._checkpoint import _model_fingerprint, _strategy_components
from nvalchemi.training._spec import BaseSpec, create_model_spec
from nvalchemi.training.distillation import DistillationStrategy
from test.training.conftest import _build_batch, _build_demo_model
from test.training.distillation.conftest import (
    _build_direct_force_model,
    _build_direct_force_teacher,
    _DirectForceTeacher,
)


class _SourcedTeacher(_DirectForceTeacher):
    """Direct-force teacher that names the file its weights were loaded from."""

    def __init__(self, model: Any, source: str) -> None:
        """Record the source alongside the wrapped model."""
        super().__init__(model)
        self.source = source

    def checkpoint_spec(self) -> BaseSpec:
        """Return the factory spec that reloads this teacher from its source."""
        return create_model_spec(_teacher_from_file, path=self.source)


class _OrderedPair(nn.Module):
    """Module registering the same two buffers in a caller-chosen order."""

    def __init__(self, *names: str) -> None:
        """Register the named buffers in the order given."""
        super().__init__()
        values = {"alpha": torch.arange(6.0), "beta": torch.full((4,), 0.5)}
        for name in names:
            self.register_buffer(name, values[name])


class _ElementTable(nn.Module):
    """Module holding a per-element buffer of the length a periodic table needs."""

    def __init__(self) -> None:
        """Hold one 95-entry table, the shape ``dftd3`` registers."""
        super().__init__()
        self.register_buffer("rcov", torch.rand(95))


class _ExtraStateModule(nn.Module):
    """Module whose state dict carries a non-tensor entry beside its weights."""

    def __init__(self) -> None:
        """Hold one small linear layer."""
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def get_extra_state(self) -> dict[str, str]:
        """Return the arbitrary object torch stores under ``_extra_state``."""
        return {"provenance": "hand-written"}

    def set_extra_state(self, state: dict[str, str]) -> None:
        """Accept the object :meth:`get_extra_state` produced."""
        del state


def _teacher_from_file(path: str) -> _SourcedTeacher:
    """Rebuild the direct-force teacher stored at *path*."""
    teacher = _SourcedTeacher(_build_direct_force_model(seed=2), path)
    teacher.load_state_dict(torch.load(path, weights_only=True))
    return teacher


def _write_teacher(tmp_path: Path, *, seed: int = 2) -> _SourcedTeacher:
    """Return a teacher whose weights sit in a file it can be reloaded from."""
    source = tmp_path / "teacher.pt"
    teacher = _SourcedTeacher(_build_direct_force_model(seed=seed), str(source))
    torch.save(teacher.state_dict(), source)
    return teacher


def _finetuned_teacher_checkpoint(tmp_path: Path) -> Path:
    """Return a native checkpoint of a teacher fine-tuned away from its source."""
    teacher = _write_teacher(tmp_path)
    with torch.no_grad():
        for parameter in teacher.parameters():
            parameter.add_(0.25)
    root = tmp_path / "finetuned"
    save_checkpoint(root, models={"teacher": (teacher, teacher.checkpoint_spec())})
    return root


def _load_role_model(root: Path, name: str, *, checkpoint_index: int = -1) -> Any:
    """Return one model of a native checkpoint, the way the CLI loads a role."""
    loaded = load_checkpoint(
        root, checkpoint_index=checkpoint_index, model_names={name}
    )
    models = loaded["models"] if isinstance(loaded, Mapping) else loaded.models
    entry = models[name]
    return entry["model"] if isinstance(entry, Mapping) else entry[0]


def _student_energy_fn(
    models: Mapping[str, Any], batch: Batch
) -> dict[str, torch.Tensor]:
    """Return the student's energy, the way a hand-rolled two-model loop would."""
    return {"predicted_energy": models["student"](batch)["energy"]}


def _make_strategy(teacher: Any, *, num_steps: int = 2) -> DistillationStrategy:
    """Return an offline distillation strategy over demo models."""
    return DistillationStrategy(
        models={"student": _build_demo_model(), "teacher": teacher},
        optimizer_configs={
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
                )
            ]
        },
        loss_fn=EnergyMSELoss(target_key="teacher_energy")
        + ForceMSELoss(target_key="teacher_forces"),
        num_steps=num_steps,
    )


def _make_plain_strategy(teacher: Any, *, num_steps: int = 2) -> TrainingStrategy:
    """Return the two-model strategy a hand-rolled loop declares nothing from."""
    return TrainingStrategy(
        models={"student": _build_demo_model(), "teacher": teacher},
        optimizer_configs={
            "student": [
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 1e-2}
                )
            ]
        },
        loss_fn=EnergyMSELoss(),
        num_steps=num_steps,
        training_fn=_student_energy_fn,
    )


def _undeclared_teacher_root(tmp_path: Path, teacher: Any) -> Path:
    """Return a root holding a teacher its writer never named in the manifest."""
    strategy = _make_plain_strategy(teacher)
    root = tmp_path / "checkpoints"
    strategy.save_checkpoint(root)
    strategy.save_checkpoint(root)
    return root


def _batches(count: int = 4) -> list[Batch]:
    """Return a short stream of unlabeled training batches."""
    return [_build_batch(seed=index) for index in range(count)]


def _manifest(root: Path) -> dict[str, Any]:
    """Return the parsed checkpoint manifest under *root*."""
    return json.loads((root / "manifest.json").read_text())


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* back over the checkpoint manifest under *root*."""
    (root / "manifest.json").write_text(json.dumps(manifest))


def _teacher_weight_file(root: Path, index: int) -> Path:
    """Return the path a teacher checkpoint at *index* would occupy."""
    return root / "models" / "teacher" / "checkpoints" / f"{index}.pt"


def _teacher_weight_files(root: Path) -> list[str]:
    """Return the names of the teacher weight files *root* holds."""
    return sorted(
        path.name for path in _teacher_weight_file(root, 0).parent.glob("*.pt")
    )


def _teacher_state(root: Path, index: int) -> dict[str, torch.Tensor]:
    """Return the teacher weights a load at *index* hands back."""
    return _load_role_model(root, "teacher", checkpoint_index=index).state_dict()


class TestTeacherStoredOncePerRoot:
    def test_the_first_checkpoint_holds_the_teacher_weights(
        self, tmp_path: Path
    ) -> None:
        """The teacher is written once, and the manifest says where."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"

        strategy.save_checkpoint(root)

        manifest = _manifest(root)
        assert set(manifest["model_references"]) == {"teacher"}
        assert manifest["model_references"]["teacher"]["checkpoint_index"] == 0
        assert _teacher_weight_file(root, 0).is_file()
        assert (root / "models" / "student" / "checkpoints" / "0.pt").is_file()
        assert (root / "models" / "teacher" / "spec.json").is_file()

    def test_later_checkpoints_reference_the_stored_copy(self, tmp_path: Path) -> None:
        """Three saves write the teacher once and point at index 0 twice."""
        strategy = _make_strategy(_build_direct_force_teacher(seed=2))
        root = tmp_path / "checkpoints"

        for _ in range(3):
            strategy.save_checkpoint(root)

        assert _manifest(root)["checkpoint_index"] == 2
        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0
        assert _teacher_weight_file(root, 0).is_file()
        assert not _teacher_weight_file(root, 1).exists()
        assert not _teacher_weight_file(root, 2).exists()

    def test_a_source_less_teacher_is_stored_without_a_warning(
        self, tmp_path: Path
    ) -> None:
        """Naming no source costs nothing now that the copy is per root."""
        strategy = _make_strategy(_build_direct_force_teacher(seed=2))
        root = tmp_path / "checkpoints"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            strategy.save_checkpoint(root)
            strategy.save_checkpoint(root)

        assert not [w for w in caught if "names no source" in str(w.message)]
        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0

    def test_the_reference_fingerprints_the_stored_weights(
        self, tmp_path: Path
    ) -> None:
        """A reader can tell what sits at the stored index without loading it."""
        teacher = _write_teacher(tmp_path)
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"

        strategy.save_checkpoint(root)

        reference = _manifest(root)["model_references"]["teacher"]
        assert reference["rebuild"] == "stored"
        assert reference["fingerprint"] == _model_fingerprint(teacher)

    def test_a_different_teacher_is_refused_rather_than_repointing(
        self, tmp_path: Path
    ) -> None:
        """A root holds one teacher, so storing a second one is refused at save."""
        teacher = _write_teacher(tmp_path)
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)

        with torch.no_grad():
            for parameter in teacher.parameters():
                parameter.add_(0.5)

        with pytest.raises(ValueError, match="already holds a different copy"):
            strategy.save_checkpoint(root)
        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0
        assert not _teacher_weight_file(root, 1).exists()

    def test_a_copy_the_manifest_does_not_name_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """The copy on disk counts, so a root written without a reference is one too."""
        root = _undeclared_teacher_root(tmp_path, _build_direct_force_teacher(seed=2))
        assert _manifest(root)["model_references"] == {}
        assert _teacher_weight_files(root) == ["0.pt", "1.pt"]
        strategy = _make_strategy(_build_direct_force_teacher(seed=11))

        with pytest.raises(ValueError, match="already holds a different copy"):
            strategy.save_checkpoint(root)
        assert not _teacher_weight_file(root, 2).exists()

    def test_a_copy_the_manifest_does_not_name_is_reused(self, tmp_path: Path) -> None:
        """Continuing such a root references the copy rather than storing a second."""
        teacher = _write_teacher(tmp_path)
        expected = {key: tensor.clone() for key, tensor in teacher.state_dict().items()}
        root = _undeclared_teacher_root(tmp_path, teacher)
        strategy = _make_strategy(_write_teacher(tmp_path))

        strategy.save_checkpoint(root)

        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 1
        assert _teacher_weight_files(root) == ["0.pt", "1.pt"]
        for index in range(3):
            restored = _teacher_state(root, index)
            for key, tensor in expected.items():
                torch.testing.assert_close(restored[key], tensor)

    def test_a_non_declaring_save_keeps_the_root_reference(
        self, tmp_path: Path
    ) -> None:
        """A writer that declares nothing orphans none of the indices reading the copy."""
        teacher = _write_teacher(tmp_path)
        expected = {key: tensor.clone() for key, tensor in teacher.state_dict().items()}
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)

        save_checkpoint(root, models=_strategy_components(strategy)[0])

        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0
        assert _teacher_weight_files(root) == ["0.pt"]
        restored = _teacher_state(root, 1)
        for key, tensor in expected.items():
            torch.testing.assert_close(restored[key], tensor)

    def test_a_non_declaring_save_refuses_a_different_copy(
        self, tmp_path: Path
    ) -> None:
        """Such a writer is held to the one-copy rule for the model it carries."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)
        models = dict(_strategy_components(strategy)[0])
        models["teacher"] = (_build_direct_force_teacher(seed=11), models["teacher"][1])

        with pytest.raises(ValueError, match="already holds a different copy"):
            save_checkpoint(root, models=models)
        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0
        assert not _teacher_weight_file(root, 1).exists()

    def test_a_named_reference_reads_no_weights_at_save(self, tmp_path: Path) -> None:
        """A manifest that names the copy is trusted, so saving reads no weights back."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"

        with patch.object(
            _checkpoint.torch, "load", wraps=_checkpoint.torch.load
        ) as mock_load:
            for _ in range(4):
                strategy.save_checkpoint(root)

        teacher_dir = str(root / "models" / "teacher")
        assert not [
            call for call in mock_load.call_args_list if teacher_dir in str(call)
        ]
        assert _manifest(root)["checkpoint_index"] == 3
        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0

    def test_every_index_restores_the_teacher_it_was_written_against(
        self, tmp_path: Path
    ) -> None:
        """A non-latest index reads back its own teacher, not a later save's."""
        teacher = _write_teacher(tmp_path)
        expected = {key: tensor.clone() for key, tensor in teacher.state_dict().items()}
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        for _ in range(3):
            strategy.save_checkpoint(root)

        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 0
        for index in range(3):
            restored = DistillationStrategy.load_checkpoint(
                root,
                checkpoint_index=index,
                training_fn="nvalchemi.training.distillation.default_distillation_fn",
            )
            restored_teacher = restored.models["teacher"].state_dict()
            for key, tensor in expected.items():
                torch.testing.assert_close(restored_teacher[key], tensor)

    def test_a_missing_stored_copy_is_written_again(self, tmp_path: Path) -> None:
        """Re-storing identical weights repairs a root whose stored file went away."""
        teacher = _write_teacher(tmp_path)
        expected = {key: tensor.clone() for key, tensor in teacher.state_dict().items()}
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)
        _teacher_weight_file(root, 0).unlink()

        strategy.save_checkpoint(root)

        assert _manifest(root)["model_references"]["teacher"]["checkpoint_index"] == 1
        assert _teacher_weight_file(root, 1).is_file()
        restored = DistillationStrategy.load_checkpoint(
            root,
            checkpoint_index=0,
            training_fn="nvalchemi.training.distillation.default_distillation_fn",
        )
        restored_teacher = restored.models["teacher"].state_dict()
        for key, tensor in expected.items():
            torch.testing.assert_close(restored_teacher[key], tensor)

    def test_restoring_reads_the_teacher_back_from_the_stored_index(
        self, tmp_path: Path
    ) -> None:
        """A rebuilt strategy holds the teacher's own weights, not fresh ones."""
        teacher = _write_teacher(tmp_path)
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)

        restored = DistillationStrategy.load_checkpoint(
            root, training_fn="nvalchemi.training.distillation.default_distillation_fn"
        )

        restored_teacher = restored.models["teacher"]
        assert restored_teacher is not teacher
        for key, tensor in teacher.state_dict().items():
            torch.testing.assert_close(restored_teacher.state_dict()[key], tensor)

    def test_a_fine_tuned_teacher_restores_its_fine_tuned_weights(
        self, tmp_path: Path
    ) -> None:
        """The canonical distill-a-fine-tuned-teacher run restarts on its own weights."""
        teacher = _load_role_model(_finetuned_teacher_checkpoint(tmp_path), "teacher")
        expected = {key: tensor.clone() for key, tensor in teacher.state_dict().items()}
        strategy = _make_strategy(teacher)
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)

        restored = DistillationStrategy.load_checkpoint(
            root, training_fn="nvalchemi.training.distillation.default_distillation_fn"
        )

        restored_teacher = restored.models["teacher"]
        assert restored_teacher is not teacher
        for key, tensor in expected.items():
            torch.testing.assert_close(restored_teacher.state_dict()[key], tensor)

    def test_a_replaced_stored_copy_is_refused(self, tmp_path: Path) -> None:
        """Weights that are no longer the checkpointed ones fail loudly."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)
        torch.save(
            _build_direct_force_teacher(seed=11).state_dict(),
            _teacher_weight_file(root, 0),
        )

        with pytest.raises(ValueError, match="stored once per checkpoint root"):
            DistillationStrategy.load_checkpoint(
                root,
                training_fn="nvalchemi.training.distillation.default_distillation_fn",
            )

    def test_a_reference_without_a_fingerprint_is_refused(self, tmp_path: Path) -> None:
        """A manifest that cannot be checked is refused rather than trusted."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)
        manifest = _manifest(root)
        del manifest["model_references"]["teacher"]["fingerprint"]
        _write_manifest(root, manifest)

        with pytest.raises(ValueError, match="carries no fingerprint"):
            DistillationStrategy.load_checkpoint(
                root,
                training_fn="nvalchemi.training.distillation.default_distillation_fn",
            )

    def test_a_reference_without_an_index_is_refused(self, tmp_path: Path) -> None:
        """A manifest naming no stored index says so instead of failing on a path."""
        strategy = _make_strategy(_write_teacher(tmp_path))
        root = tmp_path / "checkpoints"
        strategy.save_checkpoint(root)
        manifest = _manifest(root)
        del manifest["model_references"]["teacher"]["checkpoint_index"]
        _write_manifest(root, manifest)

        with pytest.raises(ValueError, match="no checkpoint_index"):
            DistillationStrategy.load_checkpoint(
                root,
                training_fn="nvalchemi.training.distillation.default_distillation_fn",
            )

    def test_a_restarted_run_trains_on_identically(self, tmp_path: Path) -> None:
        """Interrupting and restoring a run reaches the weights it would have."""
        torch.manual_seed(0)
        uninterrupted = _make_strategy(_write_teacher(tmp_path), num_steps=4)
        uninterrupted.run(_batches())

        torch.manual_seed(0)
        interrupted = _make_strategy(_write_teacher(tmp_path), num_steps=2)
        interrupted.run(_batches(2))
        root = tmp_path / "checkpoints"
        interrupted.save_checkpoint(root)
        resumed = DistillationStrategy.load_checkpoint(
            root, training_fn="nvalchemi.training.distillation.default_distillation_fn"
        )
        resumed.num_steps = 4
        resumed.run(_batches()[2:])

        assert resumed.step_count == uninterrupted.step_count == 4
        expected = uninterrupted.models["student"].state_dict()
        for key, tensor in resumed.models["student"].state_dict().items():
            torch.testing.assert_close(tensor, expected[key])


class TestModelFingerprint:
    def test_a_reduced_precision_copy_fingerprints_differently(self) -> None:
        """Precision is part of a model's identity, not normalized away."""
        module = nn.Linear(64, 64)

        digest = _model_fingerprint(module)
        reduced = _model_fingerprint(module.to(torch.bfloat16))

        assert digest != reduced

    def test_a_widened_copy_fingerprints_differently(self) -> None:
        """Widening preserves the values, so the dtype alone tells the copies apart."""
        module = nn.Linear(64, 64)

        digest = _model_fingerprint(module)
        widened = _model_fingerprint(module.to(torch.float64))

        assert widened["num_elements"] == digest["num_elements"]
        assert widened["digest"] != digest["digest"]

    def test_a_change_in_a_tensors_final_values_is_detected(self) -> None:
        """The sample spans the whole index range, so no tensor ends in a blind tail."""
        module = nn.Linear(256, 256)

        digest = _model_fingerprint(module)
        with torch.no_grad():
            module.weight.reshape(-1)[-1] += 1.0

        assert _model_fingerprint(module) != digest

    def test_a_change_late_in_a_per_element_table_is_detected(self) -> None:
        """A table small enough to hash whole has no gaps between samples to hide in."""
        module = _ElementTable()

        digest = _model_fingerprint(module)
        with torch.no_grad():
            module.rcov[92] += 1.0

        assert _model_fingerprint(module) != digest

    def test_two_orderings_of_the_same_state_fingerprint_alike(self) -> None:
        """State-dict insertion order is not part of a model's identity."""
        forwards = _model_fingerprint(_OrderedPair("alpha", "beta"))
        backwards = _model_fingerprint(_OrderedPair("beta", "alpha"))

        assert forwards == backwards

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_the_same_weights_fingerprint_alike_on_either_device(self) -> None:
        """A teacher saved on GPU verifies against the reference a CPU load wrote."""
        module = _ElementTable()
        module.linear = nn.Linear(128, 128)

        on_host = _model_fingerprint(module)
        on_device = _model_fingerprint(module.cuda())

        assert on_host == on_device

    def test_a_module_carrying_extra_state_is_fingerprinted(self) -> None:
        """A non-tensor state-dict entry is hashed rather than crashing the save."""
        module = _ExtraStateModule()

        digest = _model_fingerprint(module)

        assert digest["num_tensors"] == 2
        assert digest["num_elements"] == 10
        assert digest["digest"] != _model_fingerprint(nn.Linear(4, 2))["digest"]
