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

from collections.abc import Callable
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler

from nvalchemi.data.datapipes.dataloader import DataLoader as ALCHEMIDataLoader
from nvalchemi.data.datapipes.dataset import Dataset as ALCHEMIDataset
from nvalchemi.data.datapipes.multidataset import MultiDataset as ALCHEMIMultiDataset
from nvalchemi.data.datapipes.samplers import (
    DistributedSamplerProtocol,
    MultiDatasetBatchSampler,
)
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


def _sampler_is_distributed(
    sampler: Any, sampler_cls: Callable[..., Any] = DistributedSampler
) -> bool:
    """Return whether ``sampler`` is already a configured distributed sampler."""
    if isinstance(sampler, DistributedSamplerProtocol):
        return True
    return isinstance(sampler_cls, type) and isinstance(sampler, sampler_cls)


def _accepts_distributed_sampler_defaults(sampler_cls: Callable[..., Any]) -> bool:
    """Return whether a sampler factory accepts PyTorch distributed kwargs."""
    if sampler_cls is DistributedSampler or (
        isinstance(sampler_cls, type) and issubclass(sampler_cls, DistributedSampler)
    ):
        return True
    try:
        parameters = signature(sampler_cls).parameters
    except (TypeError, ValueError):
        return False
    if any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return True
    return {"num_replicas", "rank"}.issubset(parameters)


def _is_default_sampler_cls(sampler_cls: Callable[..., Any]) -> bool:
    """Return whether ``sampler_cls`` is the default PyTorch sampler family."""
    return sampler_cls is DistributedSampler or (
        isinstance(sampler_cls, type) and issubclass(sampler_cls, DistributedSampler)
    )


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
    :class:`torch.nn.parallel.DistributedDataParallel`, and injects the
    configured distributed sampler into dataloaders with ``dataset`` and
    ``sampler`` attributes.

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
    sampler_cls : Callable[..., Any], optional
        Sampler class or factory used for supported dataloaders. The callable is
        invoked as ``sampler_cls(dataset, **sampler_kwargs)``. The default is
        :class:`torch.utils.data.DistributedSampler`.
    sampler_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to ``sampler_cls``. For the default
        ``DistributedSampler`` and sampler callables that accept PyTorch's
        distributed sampler keywords, missing ``num_replicas``, ``rank``,
        ``shuffle``, ``seed``, and ``drop_last`` values are inferred from the
        manager and dataloader before user-provided kwargs are applied.
    """

    model_keys: tuple[str, ...] | None = None
    find_unused_parameters: bool | None = None
    broadcast_buffers: bool | None = None
    static_graph: bool = False
    process_group: Any | None = None
    backend: str | None = None
    auto_init: bool = True
    sampler_cls: Callable[..., Any] = DistributedSampler
    sampler_kwargs: dict[str, Any] = Field(default_factory=dict)

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
        """Inject the configured sampler into dataloaders that expose one."""
        if dataloader is None:
            return None
        manager = self._manager
        world_size = get_world_size(manager)
        if world_size <= 1:
            return dataloader
        # Only dataloader-like objects with sampler/dataset attributes can be
        # rewritten here; arbitrary iterables are left as caller-managed inputs.
        if not hasattr(dataloader, "sampler"):
            return dataloader
        if not hasattr(dataloader, "dataset"):
            raise ValueError(
                "DDPHook cannot inject a distributed sampler into a dataloader "
                "with no dataset attribute."
            )

        # Preserve dataloaders that are already distributed-aware, including
        # batch samplers that either are distributed samplers or wrap one.
        sampler = getattr(dataloader, "sampler", None)
        if _sampler_is_distributed(sampler, self.sampler_cls):
            return dataloader
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        nested_sampler = getattr(batch_sampler, "sampler", None)
        if _sampler_is_distributed(batch_sampler, self.sampler_cls):
            return dataloader
        if _sampler_is_distributed(nested_sampler, self.sampler_cls):
            return dataloader

        drop_last = self._dataloader_drop_last(dataloader)
        # nvalchemi dataloaders benefit from sampler objects that emit complete
        # batches, especially MultiDataset where per-dataset composition matters.
        if _is_default_sampler_cls(self.sampler_cls) and isinstance(
            dataloader, ALCHEMIDataLoader
        ):
            return self._prepare_nvalchemi_dataloader(dataloader, drop_last=drop_last)

        # Generic dataloaders get a sample-level sampler first; if their sampler
        # attribute is immutable, rebuild a replacement dataloader around it.
        sampler = self._build_sampler(dataloader, drop_last=drop_last)
        if self._assign_dataloader_sampler(dataloader, sampler):
            return dataloader
        return self._rebuild_dataloader_with_sampler(
            dataloader,
            sampler,
            drop_last=drop_last,
        )

    def _uses_distributed_sampler_defaults(self) -> bool:
        """Return whether sampler construction should apply torch defaults."""
        return _accepts_distributed_sampler_defaults(self.sampler_cls)

    def _build_sampler_kwargs(
        self, dataloader: Any, *, drop_last: bool
    ) -> dict[str, Any]:
        """Return kwargs for the configured sampler class or factory."""
        kwargs: dict[str, Any] = {}
        if self._uses_distributed_sampler_defaults():
            manager = self._manager
            configured_shuffle = self.sampler_kwargs.get("shuffle")
            kwargs.update(
                {
                    "num_replicas": get_world_size(manager),
                    "rank": get_rank(manager),
                    "shuffle": _infer_shuffle(dataloader, configured_shuffle),
                    "seed": 0,
                    "drop_last": drop_last,
                }
            )
        kwargs.update(self.sampler_kwargs)
        return kwargs

    def _build_sampler(self, dataloader: Any, *, drop_last: bool) -> Any:
        """Create the configured distributed sampler for ``dataloader``."""
        return self.sampler_cls(
            dataloader.dataset,
            **self._build_sampler_kwargs(dataloader, drop_last=drop_last),
        )

    def _prepare_nvalchemi_dataloader(
        self,
        dataloader: ALCHEMIDataLoader,
        *,
        drop_last: bool,
    ) -> ALCHEMIDataLoader:
        """Install a batched distributed sampler for nvalchemi dataloaders."""
        if dataloader.batch_sampler is not None:
            raise ValueError(
                "DDPHook cannot replace a non-distributed batch_sampler on "
                "nvalchemi.data.datapipes.DataLoader. Pass a distributed-aware "
                "batch_sampler or let DDPHook install the default one."
            )

        dataset = dataloader.dataset
        kwargs = self._build_sampler_kwargs(dataloader, drop_last=drop_last)
        if isinstance(dataset, ALCHEMIMultiDataset):
            dataloader.batch_sampler = MultiDatasetBatchSampler(
                dataset,
                batch_size=dataloader.batch_size,
                **kwargs,
            )
        elif isinstance(dataset, ALCHEMIDataset):
            sampler = DistributedSampler(dataset, **kwargs)
            dataloader.batch_sampler = BatchSampler(
                sampler,
                batch_size=dataloader.batch_size,
                drop_last=drop_last,
            )
        else:
            raise TypeError(
                "DDPHook expected nvalchemi.data.datapipes.DataLoader.dataset to be "
                "Dataset or MultiDataset when installing the default distributed "
                f"batch sampler; got {type(dataset).__name__}."
            )

        dataloader.sampler = None
        return dataloader

    def _dataloader_drop_last(self, dataloader: Any) -> bool:
        """Infer whether the dataloader drops incomplete batches."""
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if hasattr(batch_sampler, "drop_last"):
            return bool(batch_sampler.drop_last)
        return bool(getattr(dataloader, "drop_last", False))

    def _assign_dataloader_sampler(self, dataloader: Any, sampler: Any) -> bool:
        """Try to assign ``sampler`` directly to ``dataloader.sampler``."""
        try:
            dataloader.sampler = sampler
        except (AttributeError, ValueError):
            return False
        return getattr(dataloader, "sampler", None) is sampler

    def _rebuild_dataloader_with_sampler(
        self,
        dataloader: Any,
        sampler: Any,
        *,
        drop_last: bool,
    ) -> Any:
        """Return a replacement dataloader when the sampler attribute is immutable."""
        if getattr(dataloader, "batch_size", None) is None:
            raise ValueError(
                "DDPHook cannot inject DistributedSampler into a DataLoader "
                "constructed with batch_sampler. Pass a distributed-aware "
                "batch_sampler instead."
            )
        kwargs: dict[str, Any] = {
            "batch_size": dataloader.batch_size,
            "sampler": sampler,
            "drop_last": drop_last,
        }
        for name in (
            "num_workers",
            "collate_fn",
            "pin_memory",
            "timeout",
            "worker_init_fn",
            "generator",
            "persistent_workers",
        ):
            if hasattr(dataloader, name):
                kwargs[name] = getattr(dataloader, name)
        if hasattr(dataloader, "multiprocessing_context"):
            multiprocessing_context = getattr(dataloader, "multiprocessing_context")
            if multiprocessing_context is not None:
                kwargs["multiprocessing_context"] = multiprocessing_context
        if getattr(dataloader, "num_workers", 0) > 0:
            prefetch_factor = getattr(dataloader, "prefetch_factor", None)
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor
        pin_memory_device = getattr(dataloader, "pin_memory_device", "")
        if pin_memory_device:
            kwargs["pin_memory_device"] = pin_memory_device
        if hasattr(dataloader, "in_order"):
            kwargs["in_order"] = dataloader.in_order
        try:
            return type(dataloader)(dataloader.dataset, **kwargs)
        except TypeError as exc:
            raise ValueError(
                "DDPHook could not assign dataloader.sampler and could not "
                "rebuild the dataloader with the configured sampler."
            ) from exc
