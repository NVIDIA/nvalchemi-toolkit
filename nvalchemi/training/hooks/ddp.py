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
"""DistributedDataParallel setup hook for training strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import DistributedSampler, RandomSampler

from nvalchemi.hooks._context import TrainContext
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.distributed import (
    destroy_distributed,
    distributed_device,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed_initialized,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nvalchemi.data.batch import Batch
    from nvalchemi.distributed import DistributedManager
    from nvalchemi.training.strategy import TrainingStrategy

__all__ = ["DDPHook"]


def _manager_process_group(manager: DistributedManager | None) -> Any:
    """Return a process group exposed by a structural manager, if any."""
    if manager is None:
        return None
    for name in ("process_group", "group", "get_process_group"):
        if not hasattr(manager, name):
            continue
        value = getattr(manager, name)
        if callable(value):
            try:
                return value()
            except TypeError:
                continue
        return value
    return None


def _sampler_is_distributed(sampler: Any) -> bool:
    """Return whether ``sampler`` is a torch distributed sampler."""
    return isinstance(sampler, DistributedSampler)


def _infer_shuffle(dataloader: Any, configured: bool | None) -> bool:
    """Infer sampler shuffling from the original dataloader when unspecified."""
    if configured is not None:
        return configured
    return isinstance(getattr(dataloader, "sampler", None), RandomSampler)


class DDPHook(BaseModel):
    """Wrap training models with ``DistributedDataParallel`` at setup time.

    ``DDPHook`` is a standard training hook that runs at
    :attr:`~nvalchemi.training.TrainingStage.SETUP`. It initializes
    ``torch.distributed`` from torchrun environment variables when needed,
    optionally uses ``TrainingStrategy.distributed_manager`` for rank/device
    metadata, wraps selected models in
    :class:`torch.nn.parallel.DistributedDataParallel`, and injects a
    :class:`torch.utils.data.DistributedSampler` into supported dataloaders.

    Parameters
    ----------
    model_keys : tuple[str, ...] | None, optional
        Named models to wrap. ``None`` wraps all models that have optimizer
        configs.
    find_unused_parameters : bool | None, optional
        Forwarded to ``DistributedDataParallel``. ``None`` uses the external
        manager's setting when present, otherwise ``False``.
    broadcast_buffers : bool | None, optional
        Forwarded to ``DistributedDataParallel``. ``None`` uses the external
        manager's setting when present, otherwise ``False``.
    static_graph : bool, optional
        Forwarded to ``DistributedDataParallel``.
    process_group : Any, optional
        Explicit process group. Defaults to a process group exposed by the
        external distributed manager or PyTorch's default group.
    backend : str | None, optional
        Backend used when this hook initializes ``torch.distributed``.
    auto_init : bool, optional
        If ``True``, initialize ``torch.distributed`` when ``WORLD_SIZE > 1``
        and no manager/process group has already initialized communication.
    shuffle : bool | None, optional
        Distributed sampler shuffle policy. ``None`` mirrors whether the
        original dataloader used ``RandomSampler``.
    sampler_drop_last : bool | None, optional
        ``DistributedSampler.drop_last``. ``None`` mirrors the dataloader's
        batch-level ``drop_last`` setting when discoverable.
    seed : int, optional
        Distributed sampler seed.
    """

    model_keys: tuple[str, ...] | None = None
    find_unused_parameters: bool | None = None
    broadcast_buffers: bool | None = None
    static_graph: bool = False
    process_group: Any | None = None
    backend: str | None = None
    auto_init: bool = True
    shuffle: bool | None = None
    sampler_drop_last: bool | None = None
    seed: Annotated[int, Field(ge=0)] = 0

    frequency: ClassVar[int] = 1
    stage: ClassVar[TrainingStage] = TrainingStage.SETUP

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="forbid",
    )

    _original_models: dict[str, torch.nn.Module] = PrivateAttr(default_factory=dict)
    _initialized_process_group: bool = PrivateAttr(default=False)
    _manager: DistributedManager | None = PrivateAttr(default=None)
    _strategy: Any | None = PrivateAttr(default=None)
    _is_wrapped: bool = PrivateAttr(default=False)

    def prepare_strategy(self, strategy: TrainingStrategy) -> None:
        """Prepare rank/device state before the strategy moves models."""
        manager = strategy.distributed_manager
        self._manager = manager
        if self.auto_init:
            self._initialized_process_group = init_distributed(
                manager,
                backend=self.backend,
            )
        world_size = get_world_size(manager)
        if world_size <= 1:
            return
        device = distributed_device(
            manager,
            strategy.devices[0],
            prefer_cuda=self.backend != "gloo",
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
            strategy.devices = [device]

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Run DDP setup when the strategy dispatches ``TrainingStage.SETUP``."""
        if stage is not TrainingStage.SETUP:
            return
        strategy = ctx.workflow
        if strategy is None:
            raise RuntimeError("DDPHook requires a TrainContext.workflow.")
        self._wrap_models(strategy)
        strategy.active_dataloader = self.prepare_dataloader(strategy.active_dataloader)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Restore original models and clean up process groups owned by this hook."""
        self.close()

    def close(self) -> None:
        """Restore wrapped models and destroy process group if this hook created it."""
        if self._original_models:
            strategy = self._strategy
            for key, model in self._original_models.items():
                if strategy is not None:
                    strategy.models[key] = model
            self._original_models.clear()
        self._strategy = None
        self._is_wrapped = False
        if self._initialized_process_group:
            destroy_distributed(self._manager)
            self._initialized_process_group = False

    def _target_model_keys(self, strategy: TrainingStrategy) -> tuple[str, ...]:
        """Return model keys this hook should wrap."""
        if self.model_keys is not None:
            keys = self.model_keys
        else:
            keys = tuple(strategy.optimizer_configs)
        missing = [key for key in keys if key not in strategy.models]
        if missing:
            raise KeyError(
                f"DDPHook model_keys include unknown model(s) {missing}; "
                f"available model keys: {sorted(strategy.models)}."
            )
        return keys

    def _wrap_models(self, strategy: TrainingStrategy) -> None:
        """Wrap selected strategy models in DistributedDataParallel."""
        if self._is_wrapped:
            return
        manager = strategy.distributed_manager
        world_size = get_world_size(manager)
        initialized = is_distributed_initialized(manager)
        if world_size <= 1:
            return
        if not initialized:
            raise RuntimeError(
                "DDPHook requires initialized distributed communication when "
                "world_size > 1. Launch with torchrun, initialize "
                "torch.distributed before strategy.run(), or provide an "
                "initialized distributed_manager."
            )

        process_group = self.process_group or _manager_process_group(manager)
        self._strategy = strategy
        for key in self._target_model_keys(strategy):
            model = strategy.models[key]
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                continue
            self._original_models[key] = model
            strategy.models[key] = self._build_ddp(model, process_group)
        self._is_wrapped = True

    def _build_ddp(
        self,
        model: torch.nn.Module,
        process_group: Any | None,
    ) -> torch.nn.parallel.DistributedDataParallel:
        """Construct a DDP wrapper for ``model``."""
        kwargs: dict[str, Any] = {
            "find_unused_parameters": self._resolve_ddp_flag(
                "find_unused_parameters",
                default=False,
            ),
            "broadcast_buffers": self._resolve_ddp_flag(
                "broadcast_buffers",
                default=False,
            ),
            "static_graph": self.static_graph,
        }
        if process_group is not None:
            kwargs["process_group"] = process_group
        device = next(model.parameters()).device
        if device.type == "cuda":
            device_index = 0 if device.index is None else device.index
            kwargs["device_ids"] = [device_index]
            kwargs["output_device"] = device_index
        return torch.nn.parallel.DistributedDataParallel(model, **kwargs)

    def _resolve_ddp_flag(self, name: str, *, default: bool) -> bool:
        """Resolve a DDP boolean option from hook field, manager, or default."""
        value = getattr(self, name)
        if value is not None:
            return bool(value)
        if self._manager is not None and hasattr(self._manager, name):
            return bool(getattr(self._manager, name))
        return default

    def prepare_dataloader(
        self,
        dataloader: Iterable[Batch] | None,
    ) -> Iterable[Batch] | None:
        """Inject a DistributedSampler into supported dataloaders."""
        if dataloader is None:
            return None
        manager = self._manager
        world_size = get_world_size(manager)
        if world_size <= 1:
            return dataloader
        if isinstance(dataloader, TorchDataLoader):
            return self._prepare_torch_dataloader(dataloader)
        try:
            from nvalchemi.data.datapipes.dataloader import DataLoader as NVCDataLoader
        except ImportError:
            NVCDataLoader = None
        if NVCDataLoader is not None and isinstance(dataloader, NVCDataLoader):
            return self._prepare_nvalchemi_dataloader(dataloader)
        return dataloader

    def _build_sampler(self, dataloader: Any, *, drop_last: bool) -> DistributedSampler:
        """Create a DistributedSampler for ``dataloader``."""
        manager = self._manager
        return DistributedSampler(
            dataloader.dataset,
            num_replicas=get_world_size(manager),
            rank=get_rank(manager),
            shuffle=_infer_shuffle(dataloader, self.shuffle),
            seed=self.seed,
            drop_last=drop_last,
        )

    def _prepare_nvalchemi_dataloader(self, dataloader: Any) -> Any:
        """Mutate the AtomicData-native dataloader sampler in place."""
        if _sampler_is_distributed(getattr(dataloader, "sampler", None)):
            return dataloader
        drop_last = (
            dataloader.drop_last
            if self.sampler_drop_last is None
            else self.sampler_drop_last
        )
        dataloader.sampler = self._build_sampler(dataloader, drop_last=drop_last)
        return dataloader

    def _prepare_torch_dataloader(self, dataloader: TorchDataLoader) -> TorchDataLoader:
        """Return a replacement torch DataLoader with a DistributedSampler."""
        if _sampler_is_distributed(getattr(dataloader, "sampler", None)):
            return dataloader
        nested_sampler = getattr(
            getattr(dataloader, "batch_sampler", None), "sampler", None
        )
        if _sampler_is_distributed(nested_sampler):
            return dataloader
        if getattr(dataloader, "batch_size", None) is None:
            raise ValueError(
                "DDPHook cannot inject DistributedSampler into a DataLoader "
                "constructed with batch_sampler. Pass a distributed-aware "
                "batch_sampler instead."
            )

        batch_sampler = getattr(dataloader, "batch_sampler", None)
        dataloader_drop_last = bool(getattr(batch_sampler, "drop_last", False))
        sampler_drop_last = (
            dataloader_drop_last
            if self.sampler_drop_last is None
            else self.sampler_drop_last
        )
        sampler = self._build_sampler(dataloader, drop_last=sampler_drop_last)
        kwargs: dict[str, Any] = {
            "batch_size": dataloader.batch_size,
            "sampler": sampler,
            "num_workers": dataloader.num_workers,
            "collate_fn": dataloader.collate_fn,
            "pin_memory": dataloader.pin_memory,
            "drop_last": dataloader_drop_last,
            "timeout": dataloader.timeout,
            "worker_init_fn": dataloader.worker_init_fn,
            "generator": dataloader.generator,
            "persistent_workers": dataloader.persistent_workers,
        }
        multiprocessing_context = getattr(dataloader, "multiprocessing_context", None)
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context
        prefetch_factor = getattr(dataloader, "prefetch_factor", None)
        if dataloader.num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        pin_memory_device = getattr(dataloader, "pin_memory_device", "")
        if pin_memory_device:
            kwargs["pin_memory_device"] = pin_memory_device
        if hasattr(dataloader, "in_order"):
            kwargs["in_order"] = dataloader.in_order
        return TorchDataLoader(dataloader.dataset, **kwargs)
