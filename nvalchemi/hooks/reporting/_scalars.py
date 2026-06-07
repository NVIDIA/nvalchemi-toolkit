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
"""Scalar extraction helpers for reporting sinks."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import TypeAlias

import torch

from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.reporting._state import ReportingState

ScalarCallback: TypeAlias = Callable[[HookContext, Enum], object]

_COMPONENT_SCALAR_SPECS = (
    ("per_component_total", "total"),
    ("per_component_weight", "weight"),
    ("per_component_raw_weight", "raw_weight"),
)


@dataclass(frozen=True, kw_only=True)
class ScalarSnapshot:
    """Scalar reporting payload for one hook event.

    Attributes
    ----------
    stage : str
        Hook stage name associated with the snapshot.
    scalars : dict[str, float]
        Flat scalar mapping using slash-separated semantic keys.
    timestamp_s : float
        Wall-clock timestamp from :func:`time.time`.
    elapsed_s : float | None
        Seconds since the reporting state was created, when available.
    event_count : int | None
        Reporting orchestrator event count, when available.
    step_count : int | None
        Workflow step count from the hook context, when available.
    batch_count : int | None
        Training batch count from the hook context, when available.
    epoch_step_count : int | None
        Training epoch-local batch count from the hook context, when available.
    epoch : int | None
        Training epoch from the hook context, when available.
    global_rank : int
        Distributed rank from the hook context.
    """

    stage: str
    scalars: dict[str, float]
    timestamp_s: float = field(default_factory=time.time)
    elapsed_s: float | None = None
    event_count: int | None = None
    step_count: int | None = None
    batch_count: int | None = None
    epoch_step_count: int | None = None
    epoch: int | None = None
    global_rank: int = 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary representation.

        Returns
        -------
        dict[str, object]
            Snapshot metadata plus the scalar mapping.
        """
        return {
            "stage": self.stage,
            "timestamp_s": self.timestamp_s,
            "elapsed_s": self.elapsed_s,
            "event_count": self.event_count,
            "step_count": self.step_count,
            "batch_count": self.batch_count,
            "epoch_step_count": self.epoch_step_count,
            "epoch": self.epoch,
            "global_rank": self.global_rank,
            "scalars": dict(self.scalars),
        }


def collect_scalars(
    ctx: HookContext,
    stage: Enum,
    state: ReportingState | None = None,
    *,
    custom_scalars: Mapping[str, ScalarCallback] | None = None,
    include_losses: bool = True,
    include_optimizer_lrs: bool = True,
    include_dynamics: bool = False,
) -> ScalarSnapshot:
    """Collect scalar values from a hook context.

    Parameters
    ----------
    ctx : HookContext
        Workflow hook context.
    stage : Enum
        Hook stage being reported.
    state : ReportingState | None, optional
        Shared reporting state used for event count and elapsed time metadata.
    custom_scalars : Mapping[str, ScalarCallback] | None, optional
        Additional scalar callbacks. Each callback receives ``(ctx, stage)`` and
        may return either a scalar value or a nested mapping of scalar values.
    include_losses : bool, default True
        When ``True``, extract ``ctx.loss`` and ``ctx.losses`` values.
    include_optimizer_lrs : bool, default True
        When ``True``, extract learning rates from optimizer parameter groups.
    include_dynamics : bool, default False
        When ``True``, extract default dynamics observables from the current
        batch and dynamics context.

    Returns
    -------
    ScalarSnapshot
        Snapshot containing flat scalar keys and hook metadata.

    Raises
    ------
    TypeError
        If a custom scalar or loss value has an unsupported type.
    ValueError
        If a value expected to be scalar contains multiple elements.
    """
    scalars: dict[str, float] = {}
    if include_losses:
        scalars.update(extract_loss_scalars(ctx))
    if include_optimizer_lrs:
        scalars.update(extract_optimizer_lr_scalars(ctx))
    if include_dynamics:
        scalars.update(extract_dynamics_scalars(ctx))
    if custom_scalars is not None:
        for name, callback in custom_scalars.items():
            value = callback(ctx, stage)
            if value is None:
                continue
            if isinstance(value, Mapping):
                scalars.update(extract_scalars(value, prefix=name))
            else:
                scalars[name] = _to_float(value, name)

    return ScalarSnapshot(
        stage=stage.name,
        scalars=scalars,
        elapsed_s=time.monotonic() - state.started_at_s if state is not None else None,
        event_count=state.event_count if state is not None else None,
        step_count=getattr(ctx, "step_count", None),
        batch_count=getattr(ctx, "batch_count", None),
        epoch_step_count=getattr(ctx, "epoch_step_count", None),
        epoch=getattr(ctx, "epoch", None),
        global_rank=ctx.global_rank,
    )


def extract_loss_scalars(ctx: HookContext) -> dict[str, float]:
    """Extract scalar loss values from a hook context.

    Parameters
    ----------
    ctx : HookContext
        Workflow hook context. Training contexts may expose ``loss`` and
        ``losses`` attributes.

    Returns
    -------
    dict[str, float]
        Flat loss scalar mapping. Composed-loss outputs use keys such as
        ``loss/energy/total`` and ``loss/energy/weight``.

    Raises
    ------
    TypeError
        If a loss value has an unsupported type.
    ValueError
        If a value expected to be scalar contains multiple elements.
    """
    scalars: dict[str, float] = {}
    loss = getattr(ctx, "loss", None)
    if loss is not None:
        scalars["loss/total"] = _to_float(loss, "loss/total")

    losses = getattr(ctx, "losses", None)
    if losses is None:
        return scalars
    if not isinstance(losses, Mapping):
        raise TypeError(f"ctx.losses must be a mapping, got {type(losses).__name__}.")

    if "total_loss" in losses:
        scalars["loss/total"] = _to_float(losses["total_loss"], "loss/total")
    for source_key, target_suffix in _COMPONENT_SCALAR_SPECS:
        _extract_component_scalars(
            losses,
            source_key=source_key,
            target_suffix=target_suffix,
            scalars=scalars,
        )
    _extract_component_sample_means(losses, scalars)

    composed_loss_keys = _composed_loss_keys()
    for name, value in losses.items():
        if name in composed_loss_keys:
            continue
        if value is None:
            continue
        if isinstance(value, Mapping):
            scalars.update(extract_scalars(value, prefix=f"loss/{name}"))
        else:
            scalars[f"loss/{name}"] = _to_float(value, f"loss/{name}")
    return scalars


def extract_dynamics_scalars(ctx: HookContext) -> dict[str, float]:
    """Extract scalar dynamics observables from a hook context.

    Parameters
    ----------
    ctx : HookContext
        Workflow hook context. Dynamics contexts may expose ``converged_mask``
        and batches may expose energy, force, velocity, mass, and status fields.

    Returns
    -------
    dict[str, float]
        Flat scalar mapping containing available dynamics observables.

    Raises
    ------
    ValueError
        If a present tensor cannot be reduced because it is empty.
    """
    batch = ctx.batch
    scalars: dict[str, float] = {}

    energy = _get_tensor_attr(batch, "energy")
    if energy is not None:
        scalars["energy"] = _tensor_mean_to_float(energy, "energy")

    forces = _get_tensor_attr(batch, "forces")
    if forces is not None:
        if forces.numel() == 0:
            raise ValueError("'fmax' cannot reduce an empty forces tensor.")
        scalars["fmax"] = _to_float(
            torch.linalg.vector_norm(forces.detach(), dim=-1).max(),
            "fmax",
        )

    temperature = _temperature_scalar(batch)
    if temperature is not None:
        scalars["temperature"] = temperature

    converged_mask = _get_tensor_attr(ctx, "converged_mask")
    if converged_mask is not None:
        scalars["converged_fraction"] = _tensor_mean_to_float(
            converged_mask.float(),
            "converged_fraction",
        )

    active_fraction = _active_fraction_scalar(ctx)
    if active_fraction is not None:
        scalars["active_fraction"] = active_fraction

    return scalars


def extract_optimizer_lr_scalars(ctx: HookContext) -> dict[str, float]:
    """Extract learning-rate scalars from optimizer parameter groups.

    Parameters
    ----------
    ctx : HookContext
        Workflow hook context. Training contexts may expose an ``optimizers``
        sequence.

    Returns
    -------
    dict[str, float]
        Flat optimizer learning-rate mapping.
    """
    optimizers = getattr(ctx, "optimizers", None)
    if not optimizers:
        return {}
    scalars: dict[str, float] = {}
    for optimizer_idx, optimizer in enumerate(optimizers):
        param_groups = getattr(optimizer, "param_groups", ())
        for group_idx, group in enumerate(param_groups):
            if "lr" not in group:
                continue
            key = _optimizer_lr_key(
                optimizer_count=len(optimizers),
                optimizer_idx=optimizer_idx,
                group_count=len(param_groups),
                group_idx=group_idx,
            )
            scalars[key] = _to_float(group["lr"], key)
    return scalars


def extract_scalars(
    values: Mapping[str, object],
    *,
    prefix: str | None = None,
) -> dict[str, float]:
    """Extract scalar leaves from a nested mapping.

    Parameters
    ----------
    values : Mapping[str, object]
        Mapping whose leaves must be scalar Python numbers or scalar tensors.
    prefix : str | None, optional
        Optional key prefix added before every extracted scalar.

    Returns
    -------
    dict[str, float]
        Flat slash-separated scalar mapping.

    Raises
    ------
    TypeError
        If a key is not a string or a leaf has an unsupported type.
    ValueError
        If a tensor leaf contains multiple elements.
    """
    scalars: dict[str, float] = {}
    root = prefix.strip("/") if prefix else ""
    for name, value in values.items():
        if not isinstance(name, str):
            raise TypeError(f"Scalar keys must be strings, got {type(name).__name__}.")
        if value is None:
            continue
        key = _join_key(root, name)
        if isinstance(value, Mapping):
            scalars.update(extract_scalars(value, prefix=key))
        else:
            scalars[key] = _to_float(value, key)
    return scalars


def _extract_component_scalars(
    losses: Mapping[str, object],
    *,
    source_key: str,
    target_suffix: str,
    scalars: dict[str, float],
) -> None:
    values = losses.get(source_key)
    if values is None:
        return
    if not isinstance(values, Mapping):
        raise TypeError(f"losses[{source_key!r}] must be a mapping.")
    for component, value in values.items():
        if not isinstance(component, str):
            raise TypeError(
                f"Loss component names must be strings, got {type(component).__name__}."
            )
        key = f"loss/{component}/{target_suffix}"
        scalars[key] = _to_float(value, key)


def _extract_component_sample_means(
    losses: Mapping[str, object],
    scalars: dict[str, float],
) -> None:
    values = losses.get("per_component_sample")
    if values is None:
        return
    if not isinstance(values, Mapping):
        raise TypeError("losses['per_component_sample'] must be a mapping.")
    for component, value in values.items():
        if not isinstance(component, str):
            raise TypeError(
                f"Loss component names must be strings, got {type(component).__name__}."
            )
        key = f"loss/{component}/sample_mean"
        scalars[key] = _tensor_mean_to_float(value, key)


def _optimizer_lr_key(
    *,
    optimizer_count: int,
    optimizer_idx: int,
    group_count: int,
    group_idx: int,
) -> str:
    if optimizer_count == 1 and group_count == 1:
        return "optimizer/lr"
    if optimizer_count == 1:
        return f"optimizer/group_{group_idx}/lr"
    if group_count == 1:
        return f"optimizer/{optimizer_idx}/lr"
    return f"optimizer/{optimizer_idx}/group_{group_idx}/lr"


def _join_key(prefix: str, name: str) -> str:
    clean_name = name.strip("/")
    return clean_name if not prefix else f"{prefix}/{clean_name}"


def _get_tensor_attr(obj: object, name: str) -> torch.Tensor | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, torch.Tensor) else None


def _temperature_scalar(batch: object) -> float | None:
    velocities = _get_tensor_attr(batch, "velocities")
    atomic_masses = _get_tensor_attr(batch, "atomic_masses")
    batch_idx = _get_tensor_attr(batch, "batch_idx")
    num_nodes_per_graph = _get_tensor_attr(batch, "num_nodes_per_graph")
    num_graphs = getattr(batch, "num_graphs", None)
    if (
        velocities is None
        or atomic_masses is None
        or batch_idx is None
        or num_nodes_per_graph is None
        or not isinstance(num_graphs, int)
    ):
        return None
    from nvalchemi.dynamics.hooks._utils import temperature_per_graph  # noqa: PLC0415

    temperature = temperature_per_graph(
        velocities,
        atomic_masses,
        batch_idx,
        num_graphs,
        num_nodes_per_graph,
    )
    return _tensor_mean_to_float(temperature, "temperature")


def _active_fraction_scalar(ctx: HookContext) -> float | None:
    status = _get_tensor_attr(ctx.batch, "status")
    exit_status = getattr(getattr(ctx, "workflow", None), "exit_status", None)
    num_graphs = getattr(ctx.batch, "num_graphs", None)
    if (
        status is None
        or not isinstance(exit_status, int)
        or not isinstance(num_graphs, int)
    ):
        return None
    status = status.squeeze(-1) if status.dim() == 2 else status
    active_mask = status[:num_graphs] < exit_status
    return _tensor_mean_to_float(active_mask.float(), "active_fraction")


def _composed_loss_keys() -> frozenset[str]:
    try:
        from nvalchemi.training.losses.composition import ComposedLossOutput
    except ImportError:
        return frozenset(
            ("total_loss", "per_component_sample")
            + tuple(source_key for source_key, _ in _COMPONENT_SCALAR_SPECS)
        )
    return frozenset(ComposedLossOutput.__annotations__)


def _to_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name!r} must be scalar, got tensor with shape {tuple(value.shape)}."
            )
        return float(value.detach().reshape(-1)[0].item())
    raise TypeError(
        f"{name!r} must be a scalar number or scalar tensor, "
        f"got {type(value).__name__}."
    )


def _tensor_mean_to_float(value: object, name: str) -> float:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name!r} must be a tensor, got {type(value).__name__}.")
    if value.numel() == 0:
        raise ValueError(f"{name!r} cannot reduce an empty tensor.")
    tensor = value.detach()
    if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
        tensor = tensor.float()
    return float(tensor.mean().item())
