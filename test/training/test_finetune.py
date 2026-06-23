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
"""Tests for fine-tuning strategy conveniences and hooks."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from nvalchemi.hooks._context import HookContext
from nvalchemi.training import (
    CheckpointHook,
    EnergyMSELoss,
    FineTuningStrategy,
    OptimizerConfig,
)
from nvalchemi.training._spec import create_model_spec
from nvalchemi.training.hooks import ModulePatchHook, TrainableParameterHook
from nvalchemi.training.strategy import TrainingStrategy


class _OnRegisterRecorder:
    """Record workflow state when the hook is registered."""

    frequency = 1
    stage = None

    def __init__(self) -> None:
        self.saw_aux_projection = False
        self.saw_optimizer_filter = False

    def _runs_on_stage(self, stage: Enum) -> bool:  # noqa: ARG002
        return False

    def on_register(self, workflow: Any) -> None:
        self.saw_aux_projection = hasattr(
            workflow.models["main"].model, "aux_projection"
        )
        self.saw_optimizer_filter = workflow._optimizer_parameter_names is not None

    def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
        return


def _optimizer_param_ids(strategy: TrainingStrategy) -> set[int]:
    """Return ids of every parameter present in strategy optimizers."""
    return {
        id(param)
        for optimizer in strategy._optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    }


class TestModulePatchHook:
    def test_replaces_existing_module_and_adds_new_child(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        replacement = nn.Linear(8, 1)
        aux_spec = create_model_spec(nn.Linear, in_features=8, out_features=2)
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "hooks": [
                    ModulePatchHook(
                        patches={
                            "main.model.projection": replacement,
                            "main.model.aux_projection": aux_spec,
                        }
                    )
                ],
            }
        )

        model = strategy.models["main"].model
        assert model.projection is replacement
        assert isinstance(model.aux_projection, nn.Linear)
        assert model.aux_projection.out_features == 2

    def test_warns_when_direct_module_instance_is_shared(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        shared = nn.Linear(8, 1)
        with pytest.warns(UserWarning, match="parameters will be shared"):
            TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "hooks": [
                        ModulePatchHook(
                            patches={
                                "main.model.aux_a": shared,
                                "main.model.aux_b": shared,
                            }
                        )
                    ],
                }
            )

    def test_missing_parent_raises(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(AttributeError, match="missing parent"):
            TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "hooks": [
                        ModulePatchHook(
                            patches={"main.model.not_real.head": nn.Linear(8, 1)}
                        )
                    ],
                }
            )

    def test_late_registration_raises_when_optimizers_exist(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "loss_fn": EnergyMSELoss()}
        )
        strategy.train_batch(batch)

        with pytest.raises(RuntimeError, match="before optimizers are built"):
            strategy.register_hook(
                ModulePatchHook(patches={"main.model.projection": nn.Linear(8, 1)})
            )


class TestTrainableParameterHook:
    def test_patterns_must_match_parameters(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="did not match any parameter"):
            TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "hooks": [
                        TrainableParameterHook(
                            freeze_patterns=("main.model.not_real.*",)
                        )
                    ],
                }
            )

    def test_freezes_excluded_parameters_and_restores_requires_grad(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "loss_fn": EnergyMSELoss(),
                "freeze_patterns": ("main.model.*",),
                "trainable_patterns": ("main.model.projection.*",),
            }
        )
        named_params = dict(strategy.models["main"].named_parameters())
        requires_grad_before = {
            name: param.requires_grad for name, param in named_params.items()
        }
        projection_before = {
            name: param.detach().clone()
            for name, param in named_params.items()
            if name.startswith("model.projection.")
        }
        joint_mlp_before = {
            name: param.detach().clone()
            for name, param in named_params.items()
            if name.startswith("model.joint_mlp.")
        }

        strategy.run([batch])

        optimizer_ids = _optimizer_param_ids(strategy)
        named_after = dict(strategy.models["main"].named_parameters())
        for name, param in named_after.items():
            assert param.requires_grad is requires_grad_before[name]
            qualified = f"main.{name}"
            if qualified.startswith("main.model.projection."):
                assert id(param) in optimizer_ids
                assert param.grad is not None
            elif qualified.startswith("main.model.joint_mlp."):
                assert id(param) not in optimizer_ids
                assert param.grad is None

        assert any(
            not torch.equal(before, named_after[name])
            for name, before in projection_before.items()
        )
        assert all(
            torch.equal(before, named_after[name])
            for name, before in joint_mlp_before.items()
        )

    def test_trainable_patterns_alone_are_an_allow_list(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "loss_fn": EnergyMSELoss(),
                "trainable_patterns": ("main.model.projection.*",),
            }
        )

        strategy.run([batch])

        optimizer_ids = _optimizer_param_ids(strategy)
        for name, param in strategy.models["main"].named_parameters():
            qualified = f"main.{name}"
            if qualified.startswith("main.model.projection."):
                assert id(param) in optimizer_ids
            elif qualified.startswith("main.model.joint_mlp."):
                assert id(param) not in optimizer_ids

    def test_trainable_patterns_temporarily_unfreeze_matched_parameters(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        model = baseline_strategy_kwargs["models"].model
        for parameter in model.projection.parameters():
            parameter.requires_grad_(False)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.projection.named_parameters()
        }

        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "loss_fn": EnergyMSELoss(),
                "trainable_patterns": ("main.model.projection.*",),
            }
        )

        strategy.run([batch])

        optimizer_ids = _optimizer_param_ids(strategy)
        for parameter in model.projection.parameters():
            assert parameter.requires_grad is False
            assert id(parameter) in optimizer_ids
        assert any(
            not torch.equal(parameter, before[name])
            for name, parameter in model.projection.named_parameters()
        )

    def test_optimizer_only_mode_preserves_excluded_gradients(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "loss_fn": EnergyMSELoss(),
                "freeze_patterns": ("main.model.*",),
                "trainable_patterns": ("main.model.projection.*",),
                "freeze_mode": "optimizer_only",
            }
        )
        named_params = dict(strategy.models["main"].named_parameters())
        requires_grad_before = {
            name: param.requires_grad for name, param in named_params.items()
        }

        strategy.run([batch])

        optimizer_ids = _optimizer_param_ids(strategy)
        for name, param in strategy.models["main"].named_parameters():
            assert param.requires_grad is requires_grad_before[name]
            qualified = f"main.{name}"
            if qualified.startswith("main.model.joint_mlp."):
                assert id(param) not in optimizer_ids
                assert param.grad is not None

    def test_optimizer_only_mode_clears_excluded_gradients_between_batches(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "loss_fn": EnergyMSELoss(),
                "freeze_patterns": ("main.model.*",),
                "trainable_patterns": ("main.model.projection.*",),
                "freeze_mode": "optimizer_only",
                "optimizer_configs": OptimizerConfig(
                    optimizer_cls=torch.optim.SGD,
                    optimizer_kwargs={"lr": 0.0},
                ),
            }
        )

        strategy.train_batch(batch)
        first_grads = {
            name: param.grad.detach().clone()
            for name, param in strategy.models["main"].named_parameters()
            if name.startswith("model.joint_mlp.") and param.grad is not None
        }

        strategy.train_batch(batch)
        second_grads = {
            name: param.grad.detach().clone()
            for name, param in strategy.models["main"].named_parameters()
            if name.startswith("model.joint_mlp.") and param.grad is not None
        }

        assert first_grads
        assert first_grads.keys() == second_grads.keys()
        for name, grad in first_grads.items():
            assert torch.allclose(second_grads[name], grad)

    def test_late_registration_warns_when_optimizers_exist(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any
    ) -> None:
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "loss_fn": EnergyMSELoss()}
        )
        strategy.run([batch])

        with pytest.warns(UserWarning, match="optimizers were built"):
            strategy.register_hook(
                TrainableParameterHook(freeze_patterns=("main.model.projection.*",))
            )


class TestFineTuningStrategy:
    def test_generated_hooks_register_before_explicit_hooks(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        recorder = _OnRegisterRecorder()
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
                "freeze_patterns": ("main.model.*",),
                "trainable_patterns": ("main.model.projection.*",),
                "hooks": [recorder],
            }
        )

        assert isinstance(strategy.hooks[0], ModulePatchHook)
        assert isinstance(strategy.hooks[1], TrainableParameterHook)
        assert strategy.hooks[2] is recorder
        assert recorder.saw_aux_projection is True
        assert recorder.saw_optimizer_filter is True

    def test_from_pretrained_checkpoint_starts_fresh_finetuning_run(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any, tmp_path: Path
    ) -> None:
        source = TrainingStrategy(
            **{**baseline_strategy_kwargs, "hooks": [_OnRegisterRecorder()]}
        )
        source.train_batch(batch)
        source.save_checkpoint(tmp_path)
        source_state = {
            name: parameter.detach().clone()
            for name, parameter in source.models["main"].named_parameters()
        }

        strategy = FineTuningStrategy.from_pretrained_checkpoint(
            tmp_path,
            optimizer_configs=OptimizerConfig(
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.0},
            ),
            training_fn=baseline_strategy_kwargs["training_fn"],
            loss_fn=EnergyMSELoss(),
            trainable_patterns=("main.model.projection.*",),
            num_steps=1,
        )

        assert strategy.step_count == 0
        assert strategy.batch_count == 0
        assert strategy.num_epochs is None
        assert strategy.num_steps == 1
        assert isinstance(strategy.hooks[0], TrainableParameterHook)
        assert not any(isinstance(hook, _OnRegisterRecorder) for hook in strategy.hooks)
        loaded_state = dict(strategy.models["main"].named_parameters())
        assert loaded_state.keys() == source_state.keys()
        for name, parameter in loaded_state.items():
            assert torch.equal(parameter, source_state[name])

    def test_from_pretrained_checkpoint_preserves_multi_model_mapping(
        self, tmp_path: Path
    ) -> None:
        from test.training.conftest import _build_demo_model
        from test.training.test_strategy import dict_demo_training_fn

        source = TrainingStrategy(
            models={"student": _build_demo_model(), "teacher": _build_demo_model()},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
            loss_fn=EnergyMSELoss(),
            num_steps=5,
        )
        source.save_checkpoint(tmp_path)

        strategy = FineTuningStrategy.from_pretrained_checkpoint(
            tmp_path,
            optimizer_configs={
                "student": [
                    OptimizerConfig(
                        optimizer_cls=torch.optim.SGD,
                        optimizer_kwargs={"lr": 0.0},
                    )
                ]
            },
            training_fn=dict_demo_training_fn,
            loss_fn=EnergyMSELoss(),
            trainable_patterns=("student.model.projection.*",),
            num_steps=1,
        )

        assert set(strategy.models) == {"student", "teacher"}
        assert strategy.num_steps == 1
        assert strategy.optimizer_configs.keys() == {"student"}

    def test_from_pretrained_checkpoint_reuses_loss_and_optimizer_config(
        self, baseline_strategy_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        source = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "optimizer_configs": OptimizerConfig(
                    optimizer_cls=torch.optim.AdamW,
                    optimizer_kwargs={"lr": 2e-3, "weight_decay": 1e-4},
                ),
            }
        )
        source.save_checkpoint(tmp_path)

        strategy = FineTuningStrategy.from_pretrained_checkpoint(
            tmp_path,
            use_original_loss=True,
            use_original_opt_class=True,
            training_fn=baseline_strategy_kwargs["training_fn"],
            num_steps=1,
        )

        assert len(strategy.loss_fn.components) == len(source.loss_fn.components)
        [optimizer_config] = strategy.optimizer_configs["main"]
        assert optimizer_config.optimizer_cls is torch.optim.AdamW
        assert optimizer_config.optimizer_kwargs["lr"] == pytest.approx(1e-5)
        assert optimizer_config.optimizer_kwargs["weight_decay"] == pytest.approx(1e-4)

    def test_from_pretrained_checkpoint_reuse_requires_strategy_metadata(
        self, baseline_strategy_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        source = TrainingStrategy(**baseline_strategy_kwargs)
        source.save_checkpoint(tmp_path)
        (tmp_path / "strategy.json").unlink()
        (tmp_path / "strategy" / "checkpoints" / "0.json").unlink()

        with pytest.raises(ValueError, match="no strategy metadata"):
            FineTuningStrategy.from_pretrained_checkpoint(
                tmp_path,
                use_original_loss=True,
                training_fn=baseline_strategy_kwargs["training_fn"],
                optimizer_configs=baseline_strategy_kwargs["optimizer_configs"],
                num_steps=1,
            )

    def test_from_pretrained_checkpoint_rejects_models_override(
        self, baseline_strategy_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        source = TrainingStrategy(**baseline_strategy_kwargs)
        source.save_checkpoint(tmp_path)

        with pytest.raises(ValueError, match="loads models from checkpoint_dir"):
            FineTuningStrategy.from_pretrained_checkpoint(
                tmp_path,
                models=baseline_strategy_kwargs["models"],
                optimizer_configs=baseline_strategy_kwargs["optimizer_configs"],
                training_fn=baseline_strategy_kwargs["training_fn"],
                loss_fn=EnergyMSELoss(),
                num_steps=1,
            )

    def test_roundtrip_preserves_finetuning_fields(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        patch_spec = create_model_spec(nn.Linear, in_features=8, out_features=1)
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "module_patches": {"main.model.aux_projection": patch_spec},
                "freeze_patterns": ("main.model.*",),
                "trainable_patterns": ("main.model.projection.*",),
                "freeze_mode": "optimizer_only",
            }
        )
        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        restored = FineTuningStrategy.from_spec_dict(
            spec,
            models=baseline_strategy_kwargs["models"],
            hooks=[],
        )

        assert restored.freeze_patterns == ("main.model.*",)
        assert restored.trainable_patterns == ("main.model.projection.*",)
        assert restored.freeze_mode == "optimizer_only"
        assert set(restored.module_patches) == {"main.model.aux_projection"}
        assert isinstance(restored.hooks[0], ModulePatchHook)
        assert isinstance(restored.hooks[1], TrainableParameterHook)
        assert restored._optimizer_parameter_names is not None

    def test_checkpoint_load_restores_filtered_optimizer_state(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Any, tmp_path: Path
    ) -> None:
        """Fine-tuning checkpoints rebuild optimizers with filtered parameters."""
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "trainable_patterns": ("main.model.projection.*",),
                "hooks": [
                    CheckpointHook(
                        tmp_path / "checkpoints",
                        step_interval=1,
                        async_save=False,
                    )
                ],
                "num_epochs": None,
                "num_steps": 1,
            }
        )
        strategy.train_batch(batch)

        restored = FineTuningStrategy.load_checkpoint(
            tmp_path / "checkpoints",
            map_location="cpu",
        )

        assert restored.step_count == 1
        optimizer_ids = _optimizer_param_ids(restored)
        for name, parameter in restored.models["main"].named_parameters():
            if name.startswith("model.projection."):
                assert id(parameter) in optimizer_ids
            else:
                assert id(parameter) not in optimizer_ids

    def test_direct_module_patch_serialization_raises(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
            }
        )
        with pytest.raises(TypeError, match="BaseSpec"):
            strategy.to_spec_dict()
