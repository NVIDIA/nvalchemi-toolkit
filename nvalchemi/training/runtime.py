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
"""Runtime helpers for dataloading, device placement, and parallelism setup."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils.data import DataLoader

__all__ = [
    "configure_dataloader",
    "configure_parallelism",
    "freeze_unconfigured_models",
    "move_to_devices",
    "train_configured_models",
]


@contextmanager
def freeze_unconfigured_models(
    models: dict[str, torch.nn.Module] | torch.nn.ModuleDict,
    optimizer_configs: Mapping[str, object],
) -> Iterator[None]:
    """Temporarily eval/freeze models omitted from optimizer configs.

    Parameters
    ----------
    models : dict[str, torch.nn.Module] | torch.nn.ModuleDict
        Named models participating in a training run.
    optimizer_configs : Mapping[str, object]
        Optimizer configuration keyed by model name. Models absent from this
        mapping are temporarily switched to eval mode and have all parameters
        marked ``requires_grad=False``.

    Yields
    ------
    None
        Control while omitted models are frozen.
    """
    state: dict[str, tuple[bool, list[tuple[torch.nn.Parameter, bool]]]] = {}
    for key, model in models.items():
        if key in optimizer_configs:
            continue
        param_states: list[tuple[torch.nn.Parameter, bool]] = []
        for param in model.parameters():
            param_states.append((param, param.requires_grad))
            param.requires_grad_(False)
        state[key] = (model.training, param_states)
        model.eval()
    try:
        yield
    finally:
        for key, (training, param_states) in state.items():
            models[key].train(training)
            for param, requires_grad in param_states:
                param.requires_grad_(requires_grad)


@contextmanager
def train_configured_models(
    models: dict[str, torch.nn.Module] | torch.nn.ModuleDict,
    optimizer_configs: Mapping[str, object],
) -> Iterator[None]:
    """Temporarily put optimizer-configured models in training mode.

    Parameters
    ----------
    models : dict[str, torch.nn.Module] | torch.nn.ModuleDict
        Named models participating in a training run.
    optimizer_configs : Mapping[str, object]
        Optimizer configuration keyed by model name. Models present in this
        mapping are switched to training mode while the context is active.

    Yields
    ------
    None
        Control while configured models are in training mode.
    """
    state = {
        key: model.training for key, model in models.items() if key in optimizer_configs
    }
    for key in state:
        models[key].train()
    try:
        yield
    finally:
        for key, training in state.items():
            models[key].train(training)


def move_to_devices(
    models: torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict,
    devices: Sequence[torch.device],
    *,
    non_blocking: bool = False,
) -> torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict:
    """Move one model or named models to device(s), preserving input shape.

    Parameters
    ----------
    models : torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict
        Single module or named modules. Named modules are assigned devices in
        insertion order.
    devices : Sequence[torch.device]
        One device broadcasts to all models; otherwise length must match the
        number of models.
    non_blocking : bool, optional
        Forwarded to :meth:`torch.nn.Module.to`.

    Returns
    -------
    torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict
        The same input shape after in-place ``.to(...)`` calls.

    Raises
    ------
    ValueError
        If ``devices`` has length other than ``1`` or the number of models.
    """
    if isinstance(models, (dict, torch.nn.ModuleDict)):
        if len(devices) not in (1, len(models)):
            raise ValueError(
                f"devices must have length 1 or len(models)={len(models)}; "
                f"got {len(devices)}."
            )
        expanded = list(devices) if len(devices) != 1 else list(devices) * len(models)
        for model, device in zip(models.values(), expanded, strict=True):
            model.to(device, non_blocking=non_blocking)
        return models
    if len(devices) != 1:
        raise ValueError(
            f"single-model device assignment requires exactly one device; "
            f"got {len(devices)}."
        )
    return models.to(devices[0], non_blocking=non_blocking)


def configure_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool | None = None,
    sampler: Any = None,
    batch_sampler: Any = None,
    collate_fn: Callable | None = None,
    **dl_kwargs: Any,
) -> DataLoader:
    """Thin wrapper around :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    dataset : Any
    batch_size : int
    shuffle : bool | None, optional
        Defaults to ``False`` when ``sampler`` is provided and ``True``
        otherwise. Passing ``True`` with ``sampler`` raises ``ValueError``.
    sampler : Any, optional
        Optional sample-ordering object forwarded to ``DataLoader``.
    batch_sampler : Any, optional
        Optional batch sampler forwarded to ``DataLoader``.
    collate_fn : Callable | None, optional
    **dl_kwargs : Any
        Forwarded to ``DataLoader``.

    Returns
    -------
    torch.utils.data.DataLoader

    Raises
    ------
    ValueError
        If ``shuffle=True`` and ``sampler`` are both provided.
    """
    if shuffle is True and sampler is not None:
        raise ValueError("shuffle=True is incompatible with sampler.")
    resolved_shuffle = sampler is None if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=resolved_shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        **dl_kwargs,
    )


def configure_parallelism(
    models: torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict,
    *,
    strategy: str = "none",
) -> torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict:
    """Configure model parallelism, preserving input shape.

    Parameters
    ----------
    models : torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict
    strategy : str, optional

    Returns
    -------
    torch.nn.Module | dict[str, torch.nn.Module] | torch.nn.ModuleDict

    Raises
    ------
    NotImplementedError
        For any strategy other than ``"none"``.
    """
    if strategy == "none":
        return models
    raise NotImplementedError(
        f"Unsupported parallelism strategy: {strategy!r}; "
        "supported strategies: ['none']"
    )
