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
"""Declarative recipes that configure a ``BaseDynamics`` they do not own.

``BaseDynamics`` owns the stepping loop.  A workflow built on top of it —
enhanced sampling, NEB, a relaxation schedule, an equation-of-state scan —
differs from plain dynamics only in *what it configures*: which engine, which
hooks, how long.  Expressing each of those as its own runner with its own
``run()`` would give every future workflow a second loop to choose between,
and the two would drift.

:class:`DynamicsStrategy` is the alternative, and it mirrors
:class:`~nvalchemi.training.strategy.TrainingStrategy`: a Pydantic model that
validates the whole configuration up front, builds the engine, and serialises
to a spec.  The difference from the training precedent is deliberate —
``TrainingStrategy`` owns its loop because nothing below it does, whereas a
dynamics strategy delegates to the engine it builds.

Workflows become sibling subclasses that override :meth:`build_hooks`::

    class EnhancedSampling(DynamicsStrategy): ...
    class NEB(DynamicsStrategy): ...
    class Relax(DynamicsStrategy): ...
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["DynamicsStrategy"]

_JSON_SCALARS = (bool, int, float, str)


def _json_ready(value: Any, where: str) -> Any:
    """Return *value* in a form :func:`json.dumps` accepts.

    Tensors become nested lists, dtypes and devices their string form, paths
    their string form, and containers are converted element-wise.  Anything
    else raises rather than being coerced to a ``repr``: a spec that
    serialises to a string nothing can read back is worse than one that says
    it cannot represent the configuration.

    Parameters
    ----------
    value:
        The value to convert.
    where:
        Dotted path to *value*, used in the error.

    Returns
    -------
    Any
        A JSON-representable equivalent.

    Raises
    ------
    TypeError
        If *value* has no JSON form.
    """
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, f"{where}[{key!r}]")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, f"{where}[{i}]") for i, item in enumerate(value)]
    raise TypeError(
        f"to_spec_dict: {where} is a {type(value).__name__}, which has no "
        "JSON form. A spec is declarative configuration; move a live object "
        "to a runtime argument, or give the value a representable form."
    )


class DynamicsStrategy(BaseModel):
    """Declarative recipe that builds and runs a configured ``BaseDynamics``.

    Construction validates the configuration; :meth:`build` turns it into a
    live engine; :meth:`run` drives that engine.  The strategy never steps the
    dynamics itself — :meth:`run` delegates to ``BaseDynamics.run``, so there
    is exactly one stepping loop in the toolkit.

    Attributes
    ----------
    engine:
        The ``BaseDynamics`` subclass to build.  A class, not an instance: the
        strategy is a recipe, and the same recipe can build an engine more
        than once (per rank, per restart).
    engine_kwargs:
        Constructor arguments for *engine* beyond ``model``, ``hooks`` and
        ``n_steps`` — timestep, temperature, friction, and so on.
    n_steps:
        Default duration, used when :meth:`run` is called without one.
    extra_hooks:
        Caller-supplied hooks, appended after whatever :meth:`build_hooks`
        contributes.  Excluded from :meth:`to_spec_dict`, because a hook is a
        live object rather than a declarative knob.

    Notes
    -----
    Subclassing
        Override :meth:`build_hooks` to contribute the hooks the workflow
        needs.  Call ``super().build_hooks()`` and extend, so *extra_hooks*
        keeps working::

            def build_hooks(self) -> list[Hook]:
                return [WalkerIdentityHook(), *super().build_hooks()]

    Engine reuse
        :meth:`run` builds the engine once and caches it, so consecutive calls
        continue the same trajectory rather than restarting it with a fresh
        thermostat and step counter.  Call :meth:`build` directly to own the
        engine yourself.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    engine: type[BaseDynamics] = Field(
        description="BaseDynamics subclass this strategy builds."
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor arguments for the engine beyond model/hooks/n_steps.",
    )
    n_steps: int | None = Field(
        default=None, description="Default duration when run() is given none."
    )
    extra_hooks: list[Any] = Field(
        default_factory=list,
        exclude=True,
        description="Caller-supplied hooks; runtime objects, not serialised.",
    )

    _engine: BaseDynamics | None = PrivateAttr(default=None)

    def build_hooks(self) -> list[Hook]:
        """Return the hooks this strategy contributes to the engine.

        The base implementation contributes only :attr:`extra_hooks`.
        Subclasses override to prepend their own and should call
        ``super().build_hooks()`` rather than dropping it.

        Returns
        -------
        list[Hook]
            Hooks to register on the built engine, in registration order.
        """
        return list(self.extra_hooks)

    def build(self, model: BaseModelMixin) -> BaseDynamics:
        """Construct the engine for *model*.

        Parameters
        ----------
        model:
            The potential the engine will call.

        Returns
        -------
        BaseDynamics
            A freshly constructed engine with this strategy's hooks
            registered.  Each call builds a new one; :meth:`run` caches
            instead.
        """
        return self.engine(
            model=model,
            hooks=self.build_hooks(),
            n_steps=self.n_steps,
            **self.engine_kwargs,
        )

    def dynamics(self, model: BaseModelMixin) -> BaseDynamics:
        """Return the cached engine for *model*, building it on first use.

        Parameters
        ----------
        model:
            The potential the engine will call.

        Returns
        -------
        BaseDynamics
            The engine this strategy drives.  Stable across calls, so step
            counters and thermostat state persist.

        Raises
        ------
        ValueError
            If *model* is not the one the cached engine was built for.  The
            cache exists so consecutive :meth:`run` calls continue a single
            trajectory; handing it a second potential would either return an
            engine evaluating the first — a trajectory for the wrong model,
            with nothing to show for it — or quietly start a second
            trajectory sharing the first one's hooks and counters.  Call
            :meth:`build` for an independent engine, or construct a second
            strategy.
        """
        if self._engine is None:
            self._engine = self.build(model)
        elif self._engine.model is not model:
            raise ValueError(
                f"{type(self).__name__}: this strategy is already driving an "
                f"engine built for a {type(self._engine.model).__name__}, and "
                f"a different {type(model).__name__} was passed. A strategy "
                "caches its engine so consecutive run() calls continue one "
                "trajectory, which a second potential would silently "
                "invalidate. Use build(model) for an independent engine, or "
                "construct a second strategy."
            )
        return self._engine

    def run(
        self,
        batch: Batch,
        model: BaseModelMixin,
        n_steps: int | None = None,
    ) -> Batch:
        """Run the configured dynamics.

        Delegates to ``BaseDynamics.run``; the strategy contributes
        configuration, not a second stepping loop.

        Parameters
        ----------
        batch:
            The initial batch.
        model:
            The potential the engine calls.
        n_steps:
            Duration; falls back to :attr:`n_steps`.

        Returns
        -------
        Batch
            The batch after all steps.
        """
        engine = self.dynamics(model)
        return engine.run(
            batch, n_steps=n_steps if n_steps is not None else self.n_steps
        )

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialise the declarative knobs to a JSON-ready dict.

        ``extra_hooks`` is excluded: a hook is a live object, not a knob.
        Subclasses that add live fields should exclude them the same way and
        extend this dict with their own declarative settings.

        ``engine_kwargs`` is converted rather than copied.  Several
        integrators take tensor-valued controls — ``NVTLangevin`` annotates
        ``temperature`` as ``float | torch.Tensor`` — so copying them
        verbatim produces a dict that :func:`json.dumps` refuses, for a
        configuration the engine itself accepts.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.

        Raises
        ------
        TypeError
            If an ``engine_kwargs`` value has no JSON form, naming the key.
        """
        return {
            "engine": f"{self.engine.__module__}.{self.engine.__qualname__}",
            "engine_kwargs": _json_ready(dict(self.engine_kwargs), "engine_kwargs"),
            "n_steps": self.n_steps,
        }
