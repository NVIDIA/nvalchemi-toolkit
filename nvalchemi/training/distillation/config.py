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
"""Configuration of the on-policy generate-label-train segment loop."""

from __future__ import annotations

import copy
import inspect
import json
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    get_args,
)

import torch
from jaxtyping import Bool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from nvalchemi._serialization import _cls_path_of, _import_callable
from nvalchemi.data.batch import Batch
from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sinks import DataSink, ResizableSink
from nvalchemi.hooks import DynamicsContext
from nvalchemi.training._spec_utils import SpecSerializable
from nvalchemi.training.distillation.hooks import TeacherLabelHook
from nvalchemi.training.distillation.replay import (
    FIFO,
    AdmissionPolicy,
    EvictionPolicy,
    ReplayEviction,
    _batch_allocation,
    _batch_size_remedy,
)
from nvalchemi.training.distillation.scoring import (
    BUILTIN_SIGNALS,
    InProcessTeacherScorer,
    TeacherScorer,
    TeacherSignal,
    _isolated_neighbors,
    _planned_neighbor_sources,
)
from nvalchemi.training.distillation.seeding import (
    InitialStructures,
    InitialStructuresSource,
)
from nvalchemi.training.runtime import evaluating

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["OnPolicyConfig", "OnPolicySettings", "ResizableSink", "SpecSerializable"]

_SPEC_SCALARS = (bool, int, float, str, torch.dtype, torch.device)
"""Propagator constructor argument types a spec can carry verbatim."""

_RUNTIME_DYNAMICS_ARGS = frozenset(
    {"model", "hooks", "convergence_hook", "sinks", "sampler", "active_batch"}
)
"""Propagator constructor arguments held by the runtime rather than the spec."""

_LIVE_COLLABORATOR_ARGS = _RUNTIME_DYNAMICS_ARGS - {"model", "active_batch"}
"""Runtime propagator arguments a rebuild neither rebinds nor restores."""

_SOURCE_CLS_KEY = "source_cls"
"""Recipe key naming the class that rebuilds a custom initial-structures source."""

_SCORER_CLS_KEY = "scorer_cls"
"""Recipe key naming the class that rebuilds a custom teacher scorer."""

_RECORDED_SPEC_ATTR = "_recipe_spec"
"""Attribute holding the spec a recipe-built propagator was built from."""

_RECIPE_OBJECT_KEYS = frozenset({"dynamics", "teacher_scorer", "initial_structures"})
"""Recipe entries that reference an object rather than carrying a scalar setting."""

_RUNTIME_ONLY_FALLBACKS = {
    "capture_sink": "stages frames in host memory",
    "replay_admission": "admits every captured frame",
    "divergence": "flags non-finite positions or forces as divergence",
}
"""What a rebuilt loop does in place of each runtime-only object a recipe omits."""


def _dynamics_spec_dict(dynamics: BaseDynamics) -> dict[str, Any]:
    """Return the ``{"cls_path", "kwargs"}`` reference a propagator rebuilds from.

    A propagator built from a recipe remembers the reference it was built from
    and round-trips as a copy of it. A setting mutated on the live object
    therefore does not travel. Any other propagator is introspected: its
    constructor arguments are read back off same-named attributes. That fails
    for a propagator that normalizes its arguments into private internals,
    which every shipped integrator and optimizer does, so a hand-built one of
    those is refused rather than rebuilt from the constructor's defaults. A
    ``torch.dtype`` or ``torch.device`` argument travels as its name and is
    read back for a constructor annotated to take one.

    The student and every other live collaborator (hooks, a convergence hook,
    sinks, a sampler) are left out and re-registered at rebuild time, just as
    :meth:`~nvalchemi.training.TrainingStrategy.to_spec_dict` leaves the
    strategy's hooks out. A propagator carrying such a collaborator is
    reported rather than silently rebuilt without it. The reference is a
    dotted path and keyword arguments rather than a
    :class:`~nvalchemi.training._spec.BaseSpec`, because a dynamics
    constructor imports its ``BaseModelMixin`` annotation under
    ``TYPE_CHECKING``, where it does not resolve.

    Raises
    ------
    ValueError
        If no import reaches the propagator's class, or if the propagator
        neither remembers a reference nor exposes the constructor arguments it
        was built with.
    """
    live = _live_collaborators(dynamics)
    recorded = getattr(dynamics, _RECORDED_SPEC_ATTR, None)
    if isinstance(recorded, Mapping):
        _warn_live_collaborators(live)
        return {**recorded, "kwargs": dict(recorded.get("kwargs", {}))}
    # Resolve the path first, so a propagator no recipe can name is refused
    # before the collaborator warning describes a spec that is never written.
    try:
        cls_path = _cls_path_of(type(dynamics))
    except TypeError as exc:
        raise ValueError(
            f"OnPolicyConfig.dynamics is a {type(dynamics).__name__} defined "
            f"where no import reaches it ({exc}), so no recipe names it. Move "
            "the class to module scope, build the propagator from a recipe — "
            "OnPolicyConfig.from_spec_dict keeps the reference it built from — "
            "or re-supply dynamics at construction."
        ) from exc
    kwargs, unserializable = _introspected_dynamics_kwargs(dynamics)
    _warn_live_collaborators(sorted(set(live) | set(unserializable)))
    return {"cls_path": cls_path, "kwargs": kwargs}


def _live_collaborators(dynamics: BaseDynamics) -> list[str]:
    """Return the collaborators a propagator holds that a rebuilt one would not.

    The collaborators are read off the live propagator rather than off the
    constructor arguments it can be introspected for. A collaborator
    registered after construction therefore counts, and a propagator built
    from a recipe, which is never introspected, is checked too.

    Two collaborators are left out. The student is rebound at rebuild time.
    The segment loop registers its
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` for the length
    of a run and removes it afterwards, and a rebuilt loop registers its own,
    just as :class:`~nvalchemi.training.distillation.DistillationStrategy`
    keeps its own internal hooks out of its spec. Reporting that hook would
    fire at every mid-segment checkpoint and say nothing.
    """
    held = [
        name
        for name in _LIVE_COLLABORATOR_ARGS
        if name != "hooks" and getattr(dynamics, name, None)
    ]
    if any(
        not isinstance(hook, TeacherLabelHook)
        for hook in getattr(dynamics, "hooks", None) or ()
    ):
        held.append("hooks")
    return sorted(held)


def _source_spec_dict(structures: InitialStructuresSource) -> dict[str, Any]:
    """Return the recipe block *structures* is rebuilt from.

    An :class:`~nvalchemi.training.distillation.InitialStructures` is named by
    the store it reads and its budgets. Another source is named by its own
    ``to_spec_dict`` block plus its class path, and
    :meth:`OnPolicyConfig.from_spec_dict` hands the block back to that class's
    ``from_spec_dict``. A source offering neither method has no stable
    position to serialize and is refused.

    Raises
    ------
    ValueError
        If *structures* is neither an ``InitialStructures`` nor a source with
        ``to_spec_dict`` and a ``from_spec_dict`` classmethod.
    """
    if isinstance(structures, InitialStructures):
        return structures.to_spec_dict()
    if not isinstance(structures, SpecSerializable):
        raise ValueError(
            f"OnPolicyConfig.initial_structures is a {type(structures).__name__}, "
            "which no recipe can name: a source without to_spec_dict and "
            "from_spec_dict has no stable position to serialize. Give it "
            "to_spec_dict() and a from_spec_dict() classmethod to become a recipe "
            "reference, use InitialStructures over a store, or keep the run "
            "unserialized and re-supply the source at construction."
        )
    return {
        _SOURCE_CLS_KEY: _cls_path_of(type(structures)),
        **structures.to_spec_dict(),
    }


def _source_from_spec_dict(
    block: Mapping[str, Any], device: torch.device | str | None
) -> InitialStructuresSource:
    """Rebuild the source :func:`_source_spec_dict` described, on *device* when given.

    The device override applies to the store an ``InitialStructures`` reads. A
    custom source collates wherever its own ``from_spec_dict`` decides.
    """
    source_cls = block.get(_SOURCE_CLS_KEY)
    if source_cls is not None:
        rest = {key: value for key, value in block.items() if key != _SOURCE_CLS_KEY}
        return _import_callable(source_cls).from_spec_dict(rest)
    if device is not None:
        block = {**block, "dataset": {**block["dataset"], "device": str(device)}}
    return InitialStructures.from_spec_dict(block)


def _warn_live_collaborators(omitted: list[str]) -> None:
    """Report the collaborators a rebuilt propagator starts without."""
    if not omitted:
        return
    warnings.warn(
        f"The propagator holds runtime objects under {omitted!r} that no recipe "
        "describes, so they are omitted and a rebuilt propagator starts "
        "without them. Re-register them on the rebuilt dynamics, or "
        "re-supply the whole propagator at construction.",
        UserWarning,
        stacklevel=4,
    )


def _and_list(items: list[str]) -> str:
    """Join *items* as prose: ``"a"``, ``"a and b"``, or ``"a, b, and c"``."""
    if len(items) < 3:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _spec_scalar(value: Any) -> Any:
    """Return the JSON-ready form of a constructor argument a spec carries."""
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    return value


def _dynamics_signature(target: Callable[..., Any]) -> inspect.Signature:
    """Return *target*'s signature, falling back to unresolved annotations.

    A propagator constructor annotates its model as ``BaseModelMixin``, a name
    that :mod:`nvalchemi.dynamics` imports only under ``TYPE_CHECKING``.
    Resolving the string annotations of a propagator written in that style
    therefore raises :exc:`NameError`. The unresolved signature still keeps
    the parameter names and defaults a recipe reads. Its annotations stay the
    strings the source wrote, and :func:`_decoded_dynamics_kwargs` matches
    those strings as well as resolved objects.

    Core's :func:`~nvalchemi._serialization._callable_signature` must stay
    strict: :mod:`nvalchemi.training._spec` turns the annotations it resolves
    into pydantic field types, and a string there would build the wrong spec
    rather than a lenient one. Rebuilding a propagator calls its constructor
    directly and needs no annotation object at all.
    """
    try:
        return inspect.signature(target, eval_str=True)
    except NameError:
        return inspect.signature(target)


def _init_kwargs_from_attrs(dynamics: BaseDynamics) -> dict[str, Any]:
    """Read a propagator's constructor arguments back off its own attributes."""
    kwargs: dict[str, Any] = {}
    for name, parameter in _dynamics_signature(type(dynamics)).parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        try:
            kwargs[name] = getattr(dynamics, name)
        except AttributeError:
            continue
    return kwargs


def _introspected_dynamics_kwargs(
    dynamics: BaseDynamics,
) -> tuple[dict[str, Any], list[str]]:
    """Return a propagator's serializable constructor arguments and what was dropped."""
    try:
        signature = _dynamics_signature(type(dynamics))
        attributes = _init_kwargs_from_attrs(dynamics)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"OnPolicyConfig.dynamics is a {type(dynamics).__name__} whose "
            f"constructor cannot be read back ({exc}), so no recipe describes "
            "it. Build the propagator from a recipe — OnPolicyConfig."
            "from_spec_dict keeps the reference it built from — or re-supply "
            "dynamics at construction."
        ) from exc
    kwargs: dict[str, Any] = {}
    omitted: list[str] = []
    for name, value in attributes.items():
        if name in _RUNTIME_DYNAMICS_ARGS:
            continue
        if value is None or isinstance(value, _SPEC_SCALARS):
            kwargs[name] = _spec_scalar(value)
        elif value:
            omitted.append(name)
    missing = sorted(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in {"self", "model"}
        and name not in kwargs
        and (
            parameter.default is inspect.Parameter.empty
            or (name not in attributes and name not in _RUNTIME_DYNAMICS_ARGS)
        )
    )
    if missing:
        raise ValueError(
            f"OnPolicyConfig.dynamics is a {type(dynamics).__name__} that does "
            f"not expose its {missing!r} as attributes, so no recipe describes "
            "it: rebuilding it would fall back to the constructor's own "
            "defaults for arguments this propagator was not built with. Build "
            "the propagator from a recipe — OnPolicyConfig.from_spec_dict "
            "keeps the reference it built from — or re-supply dynamics at "
            "construction."
        )
    return kwargs, sorted(omitted)


def _annotation_accepts(annotation: Any, scalar: type) -> bool:
    """Return whether a constructor annotation takes *scalar*, on its own or in a union.

    An annotation arrives in one of two forms, and both are matched: the
    object :func:`_dynamics_signature` resolves it to, or the source string it
    leaves when a propagator's module hides an import behind
    ``TYPE_CHECKING``. A string is scanned for the dotted name as a whole
    token. Every spelling of one union therefore matches
    (``torch.dtype | None``, ``Optional[torch.dtype]``,
    ``Union[torch.dtype, None]``), while a bare ``dtype`` that names something
    else does not.
    """
    named = f"torch.{scalar.__name__}"
    if isinstance(annotation, str):
        return named in re.findall(r"[\w.]+", annotation)
    return scalar in (get_args(annotation) or (annotation,))


def _decoded_dynamics_kwargs(
    target: Callable[..., Any], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return recipe kwargs with torch dtypes and devices read back from their names."""
    try:
        parameters = _dynamics_signature(target).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    decoded = dict(kwargs)
    for name, value in kwargs.items():
        annotation = getattr(parameters.get(name), "annotation", None)
        if not isinstance(value, str):
            continue
        if _annotation_accepts(annotation, torch.dtype):
            decoded[name] = getattr(torch, value)
        elif _annotation_accepts(annotation, torch.device):
            decoded[name] = torch.device(value)
    return decoded


def _recorded_dynamics_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-ready copy of *spec* a rebuilt propagator remembers.

    Each keyword argument is encoded the way an introspected one is, with a
    ``torch.dtype`` or ``torch.device`` written as its name. Each must then
    have a JSON representation, because the propagator round-trips as this
    reference and a checkpoint writes it out verbatim.

    Raises
    ------
    ValueError
        If a keyword argument has no JSON representation.
    """
    kwargs = {
        name: _spec_scalar(value) for name, value in spec.get("kwargs", {}).items()
    }
    for name, value in kwargs.items():
        try:
            json.dumps(value)
        except TypeError as exc:
            raise ValueError(
                f"OnPolicyConfig.dynamics names {name!r} as a "
                f"{type(value).__name__}, which JSON cannot carry ({exc}), and "
                "a recipe is written as JSON. Give the argument a value JSON "
                "represents — a number, a string, a bool, or a torch dtype or "
                "device, which travel as their names — or re-supply dynamics "
                "at construction."
            ) from exc
    return {**spec, "kwargs": kwargs}


def _dynamics_from_spec_dict(
    spec: Mapping[str, Any], student: BaseModelMixin
) -> BaseDynamics:
    """Rebuild the propagator around *student* and record the reference on it.

    The reference is checked and copied before the propagator is built. A
    keyword argument no recipe can carry is therefore refused where it
    entered, rather than at the first checkpoint that writes it out. The copy
    also keeps the propagator's record of what it was built from separate
    from the caller's mapping and from any spec emitted later.

    Raises
    ------
    ValueError
        If ``cls_path`` names something that cannot be imported, if a keyword
        argument has no JSON representation, or if *spec* builds something that
        is not a :class:`~nvalchemi.dynamics.base.BaseDynamics`.
    """
    try:
        target = _import_callable(spec["cls_path"])
    except (ImportError, AttributeError, TypeError) as exc:
        raise ValueError(
            f"OnPolicyConfig.dynamics 'cls_path' {spec['cls_path']!r} could not "
            f"be imported: {exc}"
        ) from exc
    recorded = _recorded_dynamics_spec(spec)
    dynamics = target(
        model=student, **_decoded_dynamics_kwargs(target, spec.get("kwargs", {}))
    )
    if not isinstance(dynamics, BaseDynamics):
        raise ValueError(
            f"OnPolicyConfig.dynamics rebuilt a {type(dynamics).__name__} from "
            f"{spec['cls_path']!r}; expected a BaseDynamics propagator."
        )
    object.__setattr__(dynamics, _RECORDED_SPEC_ATTR, recorded)
    return dynamics


def _signal_spec_dict(spec: TeacherSignal) -> str | dict[str, Any]:
    """Return a built-in signal as its name and a custom one as its fields.

    A ``normalize`` callable has no JSON form, so it is dropped with a
    warning. The rebuilt scorer then keeps the teacher output's own shape.
    A signal whose companion fields that callable produces is refused
    instead: without it, the rebuilt signal would declare companions nothing
    produces, and :class:`~nvalchemi.training.distillation.TeacherSignal`
    rejects such a spec on reload.

    Raises
    ------
    ValueError
        If *spec* declares ``extra_fields`` produced by a ``normalize``
        callable.
    """
    if BUILTIN_SIGNALS.get(spec.name) == spec:
        return spec.name
    if spec.extra_fields and spec.normalize is not None:
        raise ValueError(
            f"Teacher signal {spec.name!r} produces companion fields "
            f"{list(spec.extra_fields)!r} through a normalize callable, which no "
            "recipe describes, and a signal rebuilt without it would declare "
            "companions nothing produces. Re-supply the scorer at construction, "
            "or drop the companion fields."
        )
    if spec.normalize is not None:
        warnings.warn(
            f"Teacher signal {spec.name!r} carries a normalize callable, which no "
            "recipe describes; the rebuilt scorer keeps the teacher output's own "
            "shape.",
            UserWarning,
            stacklevel=2,
        )
    return {
        "name": spec.name,
        "model_output": spec.model_output,
        "field": spec.field,
        "level": spec.level,
        "extra_fields": list(spec.extra_fields),
    }


def _signal_from_spec(entry: str | Mapping[str, Any]) -> str | TeacherSignal:
    """Return a recipe signal entry as a built-in name or a rebuilt :class:`TeacherSignal`.

    Raises
    ------
    TypeError
        If a custom entry is missing a field or names an unknown one.
    ValueError
        If a custom entry breaks the ``teacher_*`` namespace or level rules.
    """
    if isinstance(entry, str):
        return entry
    fields = dict(entry)
    fields["extra_fields"] = tuple(fields.get("extra_fields", ()))
    return TeacherSignal(**fields)


def _scorer_spec_dict(
    scorer: TeacherScorer, teacher: BaseModelMixin | None
) -> dict[str, Any]:
    """Return the recipe block *scorer* is rebuilt from.

    An :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` is
    named by its signals, dtype, probe seed, neighbor-list policy, autocast
    mode, and the strategy model name ``"teacher"``. An autocast dtype is
    written by name, as the label dtype is. Another scorer travels as a
    :class:`~nvalchemi.training.distillation.SpecSerializable`: its own
    ``to_spec_dict`` block plus its class
    path, which :func:`_scorer_from_spec_dict` hands back to that class's
    ``from_spec_dict``. A scorer offering neither method is refused.

    Raises
    ------
    ValueError
        If *scorer* is neither an ``InProcessTeacherScorer`` over *teacher*
        nor a ``SpecSerializable``.
    """
    if not isinstance(scorer, InProcessTeacherScorer):
        if isinstance(scorer, SpecSerializable):
            return {
                _SCORER_CLS_KEY: _cls_path_of(type(scorer)),
                **scorer.to_spec_dict(),
            }
        raise ValueError(
            f"OnPolicyConfig.teacher_scorer is a {type(scorer).__name__}, which "
            "no recipe describes: an InProcessTeacherScorer travels as a signal "
            "set over the strategy's own teacher, and any other scorer through "
            "its own to_spec_dict() and from_spec_dict() classmethod under "
            f"scorer_cls. Implement to_spec_dict/from_spec_dict on "
            f"{type(scorer).__name__}, or re-supply teacher_scorer at "
            "construction."
        )
    if teacher is not None and scorer.teacher is not teacher:
        raise ValueError(
            "OnPolicyConfig.teacher_scorer scores with a "
            f"{type(scorer.teacher).__name__} that is not the strategy's "
            "models['teacher'], and a recipe references the teacher by that "
            "name rather than serializing a second model. Score with the "
            "strategy's teacher, or re-supply the scorer at construction."
        )
    dtype = scorer.dtype
    autocast = scorer.autocast
    return {
        "teacher": "teacher",
        "signals": [_signal_spec_dict(spec) for spec in scorer.signal_specs.values()],
        "dtype": None if dtype is None else str(dtype).removeprefix("torch."),
        "probe_seed": scorer.probe_seed,
        "neighbor_list": scorer.neighbor_list,
        "autocast": (
            str(autocast).removeprefix("torch.")
            if isinstance(autocast, torch.dtype)
            else autocast
        ),
    }


def _scorer_from_spec_dict(
    block: Mapping[str, Any], teacher: BaseModelMixin
) -> TeacherScorer:
    """Rebuild the scorer :func:`_scorer_spec_dict` described, over *teacher*.

    The teacher is passed to an ``InProcessTeacherScorer``. A custom scorer is
    rebuilt by its own ``from_spec_dict`` from its block alone.
    """
    scorer_cls = block.get(_SCORER_CLS_KEY)
    if scorer_cls is not None:
        rest = {key: value for key, value in block.items() if key != _SCORER_CLS_KEY}
        return _import_callable(scorer_cls).from_spec_dict(rest)
    dtype = block.get("dtype")
    autocast = block.get("autocast", False)
    return InProcessTeacherScorer(
        teacher,
        [_signal_from_spec(entry) for entry in block["signals"]],
        dtype=None if dtype is None else getattr(torch, dtype),
        probe_seed=block.get("probe_seed"),
        neighbor_list=block.get("neighbor_list", "rebuild"),
        autocast=getattr(torch, autocast) if isinstance(autocast, str) else autocast,
    )


def _probe_propagator(probe: Batch, dynamics: BaseDynamics) -> Batch | None:
    """Run one ``compute()`` on *probe* and check the result against the declared keys.

    :meth:`~nvalchemi.dynamics.base.BaseDynamics.check_initial_batch`
    compares the declared keys with the initial structures. This function
    compares them with what ``compute()`` actually does. A propagator whose
    declarations no longer match its implementation is therefore rejected
    here rather than on the first step of a long run. Examples are a
    ``__needs_keys__`` output the student does not produce, or a field that
    ``compute()`` reads but nothing declared. The cost is one student forward
    pass at construction, which front-loads the kernel and CUDA
    initialization the first step pays anyway.

    The forward pass runs with the same isolation the scorer uses. The model
    is held in evaluation mode, and every submodule's own ``training`` flag is
    restored afterwards. ``compute()`` restores the ``requires_grad`` flags it
    enables. The propagator's ``_last_outputs`` is put back. The probe batch
    is the caller's own row, moved to the model's device. A graph model gets
    the neighbor list its ``neighbor_config`` declares, built on the probe and
    rolled back afterwards; during a run, a hook builds the propagator's list
    instead. A model that plans more than one neighbor-list source is not
    probed, and a warning says so. The probe builds exactly one list, and the
    check must not reject a propagator the loop can run.

    Parameters
    ----------
    probe : Batch
        One-row batch already checked for the declared structure fields.
    dynamics : BaseDynamics
        Propagator to probe.

    Returns
    -------
    Batch | None
        The probe, on the model's device, carrying the outputs ``compute()``
        wrote. ``None`` when the propagator was not probed.

    Raises
    ------
    ValueError
        If the model produced no output for a declared ``__needs_keys__``
        entry, if ``compute()`` read a field the probe does not carry, or if
        a declared ``__provides_keys__`` entry is absent afterwards.

    Warns
    -----
    UserWarning
        If the model plans more than one neighbor-list source. The
        propagator's declarations then go unchecked until its first step.
    """
    model = getattr(dynamics, "model", None)
    if model is None:
        return None
    if _planned_neighbor_sources(model) > 1:
        warnings.warn(
            f"{type(dynamics).__name__} was not probed at construction: its "
            f"model {type(model).__name__} plans more than one neighbor-list "
            "source and the probe builds exactly one list, so its declared keys "
            "are first checked against compute() on the first step. Pass "
            "probe=False to silence this.",
            UserWarning,
            stacklevel=2,
        )
        return None
    parameters = getattr(model, "parameters", None)
    device = (
        next((parameter.device for parameter in parameters()), None)
        if callable(parameters)
        else None
    )
    if device is not None and probe.device != device:
        probe = probe.to(device)
    neighbor_config = getattr(
        getattr(model, "model_config", None), "neighbor_config", None
    )
    name = type(dynamics).__name__
    last_outputs = getattr(dynamics, "_last_outputs", None)
    try:
        held = (
            evaluating(model) if isinstance(model, torch.nn.Module) else nullcontext()
        )
        with held, _isolated_neighbors(probe, neighbor_config):
            dynamics.compute(probe)
        dynamics._validate_batch_keys(probe)
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"{name}.compute() read a field the initial structures do not carry "
            f"({exc.args[0]!s}). Declare it in __provides_keys__ so the "
            "structures are checked for it at construction, or add it to the "
            "structures."
        ) from exc
    except RuntimeError as exc:
        raise ValueError(
            f"{name}'s declared keys do not match what its compute() did on one "
            f"initial structure: {exc}"
        ) from exc
    finally:
        dynamics._last_outputs = last_outputs
    return probe


def _probe_criterion(
    probe: Batch, dynamics: BaseDynamics, criterion: ConvergenceHook
) -> None:
    """Dispatch a copy of *criterion* once on *probe* and check that it migrates status.

    *probe* carries the outputs of one ``compute()``. It is stamped with the
    ``status`` the run gives its structures and dispatched to a deep copy of
    the criterion, exactly as the propagator dispatches the live one. The hook
    must therefore read every key its criteria name, and the ``status`` column
    must migrate to ``target_status`` exactly where
    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.evaluate_mask` says the
    structure converged. Whether the structure converges depends on the data
    and is not checked; whether the migration works is. Dispatching a copy
    leaves the live criterion untouched, because the lifecycle registers and
    removes that object by identity. A criterion that names a key the row does
    not carry is not dispatched, because a hook may write that key during the
    step and one ``compute()`` cannot show that. The check is skipped instead,
    with a warning that names the key.

    Parameters
    ----------
    probe : Batch
        One-row batch :func:`_probe_propagator` returned.
    dynamics : BaseDynamics
        Propagator the criterion will be registered on.
    criterion : ConvergenceHook
        Live criterion the lifecycle drives.

    Raises
    ------
    ValueError
        If the criterion raised while reading the row, or if the status column
        did not migrate where the criterion converged.

    Warns
    -----
    UserWarning
        If a criterion reads a key the probed row does not carry.
    """
    missing = sorted({rule.key for rule in criterion.criteria if rule.key not in probe})
    if missing:
        warnings.warn(
            f"The convergence criterion reads {missing!r}, which one compute() of "
            f"{type(dynamics).__name__} on an initial structure did not produce, so "
            "whether it fires cannot be checked at construction. A hook writing "
            "the key during the step is fine; a key nothing writes never converges.",
            UserWarning,
            stacklevel=2,
        )
        return
    hook = copy.deepcopy(criterion)
    probe["status"] = torch.full(
        (probe.num_graphs, 1), hook.source_status, dtype=torch.long, device=probe.device
    )
    try:
        converged = hook.evaluate_mask(probe)
        hook(
            DynamicsContext(batch=probe, step_count=0, workflow=dynamics),
            DynamicsStage.AFTER_STEP,
        )
    except (KeyError, AttributeError, RuntimeError) as exc:
        raise ValueError(
            "The convergence criterion failed on one initial structure carrying "
            f"the propagator's outputs: {exc}"
        ) from exc
    status = probe["status"].view(-1)
    expected = torch.where(
        converged,
        torch.full_like(status, hook.target_status),
        torch.full_like(status, hook.source_status),
    )
    if not torch.equal(status, expected):
        raise ValueError(
            "The convergence criterion fired on one initial structure but the "
            f"status column did not migrate where it converged; got status "
            f"{status.tolist()!r} for converged {converged.tolist()!r}, migrating "
            f"{hook.source_status!r} to {hook.target_status!r}. A criterion the "
            "lifecycle drives has to write batch.status itself, as ConvergenceHook "
            "does."
        )


def _propagator_tree(dynamics: BaseDynamics) -> Iterator[BaseDynamics]:
    """Yield *dynamics* and every propagator it composes, each exactly once.

    A :class:`~nvalchemi.dynamics.FusedStage` keeps its integrators in
    ``sub_stages``, and a pipeline keeps its stages in ``stages``. A check that
    reads only the root therefore misses the propagator that actually runs.
    Nodes are compared by identity, because one integrator reached through two
    sub-stages is a single propagator.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator at the root of the composition.

    Yields
    ------
    BaseDynamics
        Every propagator in the tree, the root first.
    """
    seen: list[BaseDynamics] = []
    pending: list[BaseDynamics] = [dynamics]
    while pending:
        node = pending.pop()
        if any(node is visited for visited in seen):
            continue
        seen.append(node)
        yield node
        pending.extend(sub for _, sub in getattr(node, "sub_stages", ()))
        stages = getattr(node, "stages", ())
        pending.extend(stages.values() if isinstance(stages, Mapping) else stages)


def _status_migrators(dynamics: BaseDynamics) -> list[ConvergenceHook]:
    """Return every status migrator on *dynamics* and its sub-stages.

    A status migrator is a :class:`~nvalchemi.dynamics.base.ConvergenceHook`
    with both ``source_status`` and ``target_status`` set. Every place a
    propagator can hold one is searched. That covers its registered hooks (on
    a :class:`~nvalchemi.dynamics.FusedStage`, the hooks registered at the
    fused level) and its ``convergence_hook``. It also covers the same two
    places on every sub-stage, where a fused stage puts the migrators it
    builds itself: one on every sub-stage except the last, and one on the
    last whenever it declares a ``convergence_hook``.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator to walk.

    Returns
    -------
    list[ConvergenceHook]
        The migrators, in the order they were found.
    """
    return [
        hook
        for propagator in _propagator_tree(dynamics)
        for hook in (*propagator.hooks, propagator.convergence_hook)
        if isinstance(hook, ConvergenceHook)
        and hook.source_status is not None
        and hook.target_status is not None
    ]


def _competing_migrators(
    dynamics: BaseDynamics, criterion: ConvergenceHook
) -> list[ConvergenceHook]:
    """Return the status migrators already on *dynamics* that are not *criterion*.

    The lifecycle is about to install *criterion* as the propagator's
    ``convergence_hook`` and as a registered hook, so *criterion* itself is
    not a competitor wherever the walk finds it.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the lifecycle is being installed on.
    criterion : ConvergenceHook
        The lifecycle's own criterion.

    Returns
    -------
    list[ConvergenceHook]
        The competing criteria, in the order they were found.
    """
    return [hook for hook in _status_migrators(dynamics) if hook is not criterion]


def _check_sole_migrator(dynamics: BaseDynamics, criterion: ConvergenceHook) -> None:
    """Reject *dynamics* unless *criterion* would be its only status migrator.

    The config runs this check when it is built, and the lifecycle runs it
    again when the run starts, because a hook can be registered on the
    propagator in between.

    Raises
    ------
    ValueError
        If *dynamics* carries a status-migrating criterion other than
        *criterion*.
    """
    competing = _competing_migrators(dynamics, criterion)
    if not competing:
        return
    migrations = [(hook.source_status, hook.target_status) for hook in competing]
    raise ValueError(
        "The relaxation lifecycle owns graduation for this run, so the "
        "propagator must carry no other status-migrating ConvergenceHook; "
        f"got {migrations!r} beside the configured "
        f"({criterion.source_status!r}, {criterion.target_status!r}). "
        "Remove it, or drop fmax or convergence_hook and let the propagator "
        "manage its own lifecycle. A FusedStage builds one for every "
        "sub-stage except the last, and for the last one when it declares a "
        "convergence_hook."
    )


def _check_structure_status(state: Batch, criterion: ConvergenceHook) -> None:
    """Reject a criterion that migrates off a status no initial structure holds.

    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.__call__` migrates only the
    graphs whose status equals its ``source_status``. A criterion whose
    ``source_status`` no initial structure holds leaves the convergence path
    inert: no structure ever converges, so none freezes or graduates through
    it, and generation never runs out of trajectories through it, so the
    exhaustion warning never fires. The divergence path ignores
    ``source_status``, so a diverged structure would still freeze, graduate,
    and be counted in the boundary warning.

    Parameters
    ----------
    state : Batch
        Initial batch, already stamped with the run's own bookkeeping.
    criterion : ConvergenceHook
        Criterion driving the trajectory lifecycle.

    Raises
    ------
    ValueError
        If *state* carries no ``status`` column, or if no graph of it carries
        the criterion's ``source_status``.
    """
    if "status" not in state:
        raise ValueError(
            "The initial batch carries no status column, so nothing could migrate "
            f"off source_status={criterion.source_status!r}. A relaxation "
            "lifecycle graduates structures on that column. An "
            "InitialStructuresSource driving a lifecycle stamps status zeros and "
            "system_ids on the batch its initial_batch returns, as "
            "InitialStructures does."
        )
    statuses = sorted({int(value) for value in state["status"].view(-1).tolist()})
    if criterion.source_status in statuses:
        return
    raise ValueError(
        "A converged graph migrates off the status its initial structure "
        f"carries; got source_status={criterion.source_status!r} against "
        f"initial statuses {statuses!r}. The run stamps that status itself "
        "rather than reading it from the structures, so nothing would ever "
        "freeze or graduate. Pass source_status=0, or pass the threshold as "
        "fmax instead."
    )


class OnPolicySettings(BaseModel):
    """Declarative settings of one on-policy distillation segment loop.

    Every field is a JSON scalar, so the settings validate without a
    propagator, a teacher, or a store. A recipe with invalid settings is
    therefore rejected before any teacher is loaded. :class:`OnPolicyConfig`
    inherits these settings and adds the live objects the loop drives. That
    class and the strategy check whether those objects work with the loop.

    Parameters
    ----------
    replay_ratio : float
        Fraction of every training batch drawn from the replay buffer. The
        reference dataset supplies the rest.
    training_steps_per_segment : int
        Optimizer steps taken per segment, one per training batch.
    batch_size : int, optional
        Samples per training batch, counting both sources of the mixture.
        Default ``8``.
    generation_steps : int, optional
        Propagator steps generated per segment. Default ``100``.
    label_frequency : int, optional
        Propagator steps between teacher labelings. Each segment's last frame
        is labeled in addition. Default ``100``.
    replay_capacity : int | None, optional
        Frame capacity of the replay buffer. Default ``None`` (unbounded). A
        Boltzmann objective needs a bound; see the :class:`OnPolicyConfig`
        Notes.
    replay_eviction : {"fifo"}, optional
        Eviction policy of the replay buffer, as a name a recipe can store. A
        policy instance is passed to :class:`OnPolicyConfig` instead. Default
        ``"fifo"``.
    replay_device : str | None, optional
        Device the replay buffer keeps frames on. An index-less ``cuda`` means
        this rank's current CUDA device. Default ``None`` uses the
        device the reference dataset emits its batches on, or host memory when
        there is no reference dataset.
    seed : int, optional
        Base seed of every segment's mixture sampler. Default ``0``.
    rank_seed_stride : int, optional
        Seed offset between neighboring ranks on a multi-rank launch. Rank
        ``r`` adds ``r * rank_seed_stride`` to its seeds. Default
        ``1_000_003``.
    require_wrapped_student : bool, optional
        Whether a multi-rank run refuses to start unless the ``SETUP`` stage
        replaced the student with a wrapper that owns it, as a ``DDPHook``
        does. Default ``True``.
    samples_equilibrium : bool | None, optional
        Whether the propagator samples an equilibrium ensemble, the ensemble a
        distribution-matching objective is defined on. Default ``None`` infers
        the answer from the propagator: a stage whose class declares
        :attr:`~nvalchemi.dynamics.BaseDynamics.samples_equilibrium` false, as
        the relaxation optimizers do, or a convergence criterion counts as not
        sampling one. ``True`` or ``False`` overrides that inference for the
        objective's guard.
    fmax : float | None, optional
        Max force norm below which a generated trajectory counts as finished.
        Setting it turns on the trajectory lifecycle of a relaxation run, in
        which converged structures leave the batch and fresh initial
        structures replace them. Default ``None`` ends no trajectory, which
        suits a molecular-dynamics run.
    weight_sync_frequency : int, optional
        Segments between weight syncs to the propagator. Default ``1``, the
        only accepted value while the propagator shares the student module.
    probe : bool, optional
        Whether :class:`OnPolicyConfig` runs the propagator's ``compute()`` on
        one initial structure at construction to check it against its
        declared keys. ``False`` defers any mismatch to the first step.
        Default ``True``.
    restart : {"error", "reseed", "resume"}, optional
        What a run does with a restart bundle it cannot consume. A bundle
        cannot be consumed when it was written on, or is restored onto, more
        than one rank, or when this rank's shard cannot take its position.
        ``"error"`` refuses to start. ``"reseed"`` drops the bundle with a
        warning and starts generation cold. ``"resume"`` refuses as
        ``"error"`` does, and also refuses a restore that carries no bundle at
        all. Default ``"error"``.

    Raises
    ------
    ValueError
        If a count or the threshold is not positive, if ``replay_ratio`` falls
        outside ``[0, 1]`` or is exactly ``0``, if rounding the ratio against
        the batch size gives one mixture source no sample in any batch, or if
        ``weight_sync_frequency`` is not ``1``.

    Examples
    --------
    >>> from nvalchemi.training.distillation import OnPolicySettings
    >>> settings = OnPolicySettings(replay_ratio=0.25, training_steps_per_segment=32)
    >>> settings.batch_size
    8

    Notes
    -----
    ``label_frequency`` is the throughput setting. It counts against the
    propagator's cumulative ``step_count``, so the labeling cadence does not
    restart at a segment boundary. Each segment also labels the frame it ends
    on, and the cadence dispatch adjacent to that forced label is skipped.
    With ``generation_steps`` a multiple of ``label_frequency``, each
    trajectory is therefore labeled ``generation_steps // label_frequency``
    times per segment, which is once per segment only when the two are equal.
    The first segment pays one more, for the cadence dispatch at step ``0``.
    ``training_steps_per_segment`` counts training batches. It equals the
    number of optimizer steps only while every batch takes one step.
    ``fmax`` is compared against the student's forces, which are the forces
    the propagator follows. It therefore measures the convergence of the
    relaxation itself.

    Size ``replay_capacity`` as a multiple of the trajectory count. Otherwise
    FIFO eviction removes only part of one labeled step's frames, which
    over-represents the structures at the back of the batch. Space the
    ``seed`` of replicate runs by at least
    ``num_steps // training_steps_per_segment``, because the sampler adds the
    segment index to it. See :ref:`training-distillation-api`.

    On a multi-rank launch, each rank adds ``rank * rank_seed_stride`` to
    ``seed`` and, through
    :meth:`~nvalchemi.dynamics.BaseDynamics.seed_offset`, to every
    ``random_seed`` that ``dynamics`` and its sub-stages expose. Ranks
    therefore draw from the reference dataset independently and apply
    different thermostat noise to the structures they were dealt. Both
    streams add a step counter to their seed, so the stride has to stay above
    every counter the run reaches; the default, ``1_000_003``, does so for a
    run whose counters stay below it. Set a different stride when a replicate
    launch's seeds would land on another rank's stride. A stage that holds
    randomness the offset cannot move, such as a :class:`torch.Generator`
    with no integer ``random_seed``, is named in a warning, and the caller
    must give it a rank-distinct seed.

    A multi-rank run also checks that the ``SETUP`` stage replaced the student
    with a gradient-synchronizing wrapper. A wrapper that works in place, such
    as FSDP2's ``fully_shard`` or hook-based synchronization, leaves the
    student object unchanged and fails that check.
    ``require_wrapped_student=False`` waives the check for such a wrapper, and
    keeping the ranks' students in step then becomes the caller's
    responsibility.
    """

    replay_ratio: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Fraction of every training batch drawn from the replay buffer; "
                "the rest comes from the reference dataset."
            ),
        ),
    ]
    training_steps_per_segment: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Training batches drawn from each segment's mixture, one "
                "optimizer step each unless an update hook vetoes the step."
            ),
        ),
    ]
    batch_size: Annotated[
        int,
        Field(
            default=8,
            gt=0,
            description=(
                "Samples per training batch, split between the reference "
                "dataset and the replay buffer at replay_ratio."
            ),
        ),
    ] = 8
    generation_steps: Annotated[
        int,
        Field(
            default=100,
            gt=0,
            description="Propagator steps generated per segment.",
        ),
    ] = 100
    label_frequency: Annotated[
        int,
        Field(
            default=100,
            gt=0,
            description=(
                "Propagator steps between teacher labelings, on top of the "
                "segment's own last frame. Larger values trade label density "
                "for generation throughput."
            ),
        ),
    ] = 100
    replay_capacity: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Frames the replay buffer keeps; None leaves it unbounded.",
        ),
    ] = None
    replay_eviction: Annotated[
        ReplayEviction,
        Field(
            default="fifo",
            description=(
                "Policy retiring frames from a full replay buffer, named as a "
                "recipe spells it; 'fifo' drops the oldest frames first."
            ),
        ),
    ] = "fifo"
    replay_device: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Device the replay buffer keeps frames on, named as a string. "
                "None uses the device the reference dataset emits its batches "
                "on, so the mixture collates on one device, or host memory when "
                "the run has no reference dataset. An index-less 'cuda' names "
                "the device this rank has made current, which under a launcher "
                "is the one it pinned this rank to."
            ),
        ),
    ] = None
    seed: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description=(
                "Base seed of every segment's mixture sampler, combined with the "
                "segment index so consecutive segments draw different reference "
                "samples and replicate runs can be made independent."
            ),
        ),
    ] = 0
    rank_seed_stride: Annotated[
        int,
        Field(
            default=1_000_003,
            gt=0,
            description=(
                "Seed-space distance between neighboring ranks: rank r moves "
                "seed, and every random_seed the propagator exposes, by "
                "r * rank_seed_stride. Both streams add a step counter to the "
                "base seed, so keep it above the run's step count. Set a "
                "different stride when a replicate launch's seeds would land "
                "on another rank's stride."
            ),
        ),
    ] = 1_000_003
    require_wrapped_student: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether a multi-rank run refuses to start unless the SETUP "
                "stage replaced models['student'] with a wrapper owning it, "
                "the way a DDPHook does. False skips that check with a warning "
                "for wrappers working in place (FSDP2 fully_shard, hook-based "
                "gradient synchronization) and leaves keeping the ranks' "
                "students in step to the caller."
            ),
        ),
    ] = True
    fmax: Annotated[
        float | None,
        Field(
            default=None,
            gt=0.0,
            description=(
                "Max force norm below which a generated trajectory counts as "
                "finished, which is what turns a relaxation run into a "
                "lifecycle. None runs no trajectory lifecycle: nothing graduates "
                "and nothing is backfilled, which is what a molecular-dynamics "
                "run wants."
            ),
        ),
    ] = None
    weight_sync_frequency: Annotated[
        int,
        Field(
            default=1,
            gt=0,
            description=(
                "Segments between weight syncs to the propagator. Reserved: "
                "must be 1 while the propagator shares the student module."
            ),
        ),
    ] = 1
    probe: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Whether the propagator's compute() is run on one initial "
                "structure at construction to check its declared keys against "
                "what it does. False skips that forward and defers a mismatch "
                "to the first step."
            ),
        ),
    ] = True
    samples_equilibrium: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Whether the propagator samples an equilibrium ensemble. None "
                "infers it: a stage declaring samples_equilibrium=False on its "
                "class, as the relaxation optimizers do, or a convergence "
                "criterion is treated as not sampling equilibrium. True or "
                "False overrides that inference for a distribution-matching "
                "objective's guard."
            ),
        ),
    ] = None
    restart: Annotated[
        Literal["error", "reseed", "resume"],
        Field(
            default="error",
            description=(
                "What a run does with a restart bundle it cannot consume. A "
                "checkpoint carries rank zero's trajectory, replay frames, and "
                "initial-structure position alone, so a bundle written on or "
                "restored onto more than one rank, or whose position this "
                "rank's shard cannot take, is one the run cannot consume. "
                "'error' refuses to start. 'reseed' drops the bundle with a "
                "warning and starts generation cold on every rank. 'resume' "
                "additionally refuses a restore that carries no bundle."
            ),
        ),
    ] = "error"

    model_config = ConfigDict(extra="forbid")

    @field_validator("replay_device", mode="before")
    @classmethod
    def _name_replay_device(cls, value: Any) -> Any:
        """Accept a ``torch.device`` and store it as the string every reader expects."""
        return str(value) if isinstance(value, torch.device) else value

    @model_validator(mode="after")
    def _validate_weight_sync(self) -> OnPolicySettings:
        """Reject any ``weight_sync_frequency`` other than the reserved value 1."""
        if self.weight_sync_frequency != 1:
            raise ValueError(
                "weight_sync_frequency must be 1: the propagator holds the same "
                "student module the trainer updates, so an eager run is never out "
                f"of sync; got {self.weight_sync_frequency!r}. Larger values are "
                "reserved for the compiled and asynchronous teacher paths."
            )
        return self

    @model_validator(mode="after")
    def _validate_mixture(self) -> OnPolicySettings:
        """Reject a mixture that no batch can actually be drawn from."""
        if self.replay_ratio == 0.0:
            raise ValueError(
                "replay_ratio=0 trains on reference data only, which is "
                "offline distillation paying for generation it never uses; "
                "drop on_policy and call run() with a loader over the labeled "
                "dataset instead."
            )
        reference_samples, replay_samples = _batch_allocation(
            self.replay_ratio, self.batch_size
        )
        if self.replay_ratio >= 1.0 or min(reference_samples, replay_samples) > 0:
            return self
        raise ValueError(
            "The mixture is drawn as whole samples, so replay_ratio and "
            "batch_size must together give each source at least one sample per "
            f"batch; got replay_ratio={self.replay_ratio!r} with "
            f"batch_size={self.batch_size!r}, which puts {reference_samples} "
            f"reference and {replay_samples} generated samples in every batch "
            "and leaves one source out of training entirely. To fix it, "
            f"{_batch_size_remedy(self.replay_ratio)}."
        )


def _on_policy_settings(recipe: Mapping[str, Any]) -> OnPolicySettings:
    """Validate an ``on_policy`` block's scalar settings, ignoring its object entries.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        ``on_policy`` block, as :meth:`OnPolicyConfig.to_spec_dict` produces
        it or as a recipe file carries it.

    Returns
    -------
    OnPolicySettings
        The settings the block sets, with the config's own defaults filled in.

    Raises
    ------
    pydantic.ValidationError
        If a setting is out of range, of the wrong type, or unknown.

    Notes
    -----
    A clean pass means the block's settings are self-consistent, not that the
    run will start. The propagator, the scorer, and the initial-structure
    store the block names are skipped. Whether they compose with the segment
    loop is for :meth:`DistillationStrategy.run` to decide, not this function.
    """
    return OnPolicySettings.model_validate(
        {key: value for key, value in recipe.items() if key not in _RECIPE_OBJECT_KEYS}
    )


class OnPolicyConfig(OnPolicySettings):
    """Settings and live objects of one on-policy distillation segment loop.

    Each *segment* has two phases. The *generation* phase runs the student's
    own propagator for ``generation_steps`` steps and labels frames with the
    teacher as it goes. The *training* phase then takes
    ``training_steps_per_segment`` optimizer steps. Its batches are a
    *mixture* of the reference dataset and the replay buffer, split at
    ``replay_ratio``. The propagator holds the module the trainer updates, so
    each segment generates from a fresher policy than the last. The scalar
    settings come from :class:`OnPolicySettings`, which this class inherits so
    that a recipe stays flat. :attr:`settings` returns a detached copy of them
    for a pre-flight check or a restart bundle.

    The propagator can be any :class:`~nvalchemi.dynamics.base.BaseDynamics`.
    A relaxation optimizer such as :class:`~nvalchemi.dynamics.optimizers.FIRE`
    drives the loop exactly as a thermostat does. The initial structures must
    carry every field the propagator updates in place through
    ``__provides_keys__``. For every shipped propagator these include
    ``velocities``, and the variable-cell ones also need a ``cell``. The model
    outputs named by ``__needs_keys__`` are computed before the first step, so
    they need not be present. Construction checks one row, so a missing field
    is a construction error rather than a failure on the first step. The
    propagator's ``compute()`` then runs once on that row. This rejects
    declarations that no longer match the implementation, such as a
    ``__needs_keys__`` output the student never produces or a field
    ``compute()`` reads that nothing declared. The cost is one student forward
    pass at construction. A graph model is probed with the neighbor list its
    ``neighbor_config`` declares, built on the row and rolled back afterwards.
    A model that plans more than one neighbor-list source is not probed.

    A relaxation propagator needs a *trajectory lifecycle*. Relaxations
    converge, and a converged structure that keeps being propagated fills the
    replay buffer with near-duplicates of a frame it already holds. ``fmax``
    turns the lifecycle on. A converged structure freezes and is stored once,
    as the minimum it reached. At the segment boundary it *graduates*, that
    is, it leaves the batch, and the initial structures *backfill* a fresh
    structure in its place for as long as the source holds rows. An
    :class:`~nvalchemi.training.distillation.InitialStructures` built with
    ``recycle=True`` wraps to its first row instead of letting the batch
    narrow. Generation ends when the last trajectory finishes, and the
    remaining training steps draw on the frames already in the buffer.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator that generates on-policy frames. It holds the student
        module.
    teacher_scorer : TeacherScorer
        Scorer that labels generated frames. A custom scorer that declares
        ``label_fields`` lets the fields it writes be known before the run.
    initial_structures : InitialStructuresSource
        Structures the generated trajectories start from. They are served
        from one position that backfills and restarts share. A multi-rank
        launch deals them out strided, every ``world_size``-th structure to
        each rank. Pass an
        :class:`~nvalchemi.training.distillation.InitialStructures`, any other
        object that implements the protocol, or a bare dataset, which is
        wrapped in an ``InitialStructures``.
    capture_sink : DataSink | None, optional
        Sink that stages each segment's labeled frames until the segment
        boundary drains them into the replay buffer. A
        :class:`~nvalchemi.dynamics.sinks.GPUBuffer` keeps the staged frames on
        the generation device and avoids a device-to-host copy per frame.
        Default ``None`` builds a host-memory sink per segment.
    replay_eviction : {"fifo"} | EvictionPolicy, optional
        Eviction policy of the replay buffer: the name ``"fifo"`` or a live
        :class:`~nvalchemi.training.distillation.EvictionPolicy` instance.
        Default ``"fifo"``.
    replay_admission : AdmissionPolicy | None, optional
        Predicate that selects which captured frames each segment admits into
        the replay buffer. Default ``None`` admits every captured frame.
    convergence_hook : ConvergenceHook | None, optional
        Live criterion that decides when a generated trajectory is finished,
        used instead of the ``fmax`` threshold. Default ``None``.
    divergence : Callable[[Batch], Bool[torch.Tensor, "G"]] | None, optional
        Predicate over the live frame that returns one boolean per graph,
        ``True`` where the trajectory diverged. The lifecycle freezes a flagged
        graph on the step it is flagged and keeps it out of both capture
        routes. At the segment boundary it retires the graph and backfills its
        place. Default ``None`` uses
        :func:`~nvalchemi.training.distillation.nonfinite_divergence`.

    Raises
    ------
    ValueError
        If a setting is out of range, if both ``fmax`` and
        ``convergence_hook`` are set, if ``convergence_hook`` cannot manage
        the lifecycle, if ``initial_structures`` recycles while no criterion is
        set, if a criterion is paired with a
        :class:`~nvalchemi.distributed.DomainParallel` propagator or a
        multi-sub-stage :class:`~nvalchemi.dynamics.FusedStage`, if
        ``initial_structures`` is neither a source nor a dataset, if the
        initial structures lack a field the propagator needs for its first
        step, or if the propagator's ``compute()`` on one row contradicts its
        declared keys.

    Examples
    --------
    >>> from nvalchemi.training.distillation import (  # doctest: +SKIP
    ...     InProcessTeacherScorer,
    ...     OnPolicyConfig,
    ...     InitialStructures,
    ... )
    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=NVTLangevin(student, dt=0.5, temperature=300.0),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     initial_structures=InitialStructures(dataset),
    ...     replay_ratio=0.25,
    ...     training_steps_per_segment=32,
    ...     batch_size=16,
    ...     generation_steps=50,
    ...     label_frequency=10,
    ...     replay_capacity=8192,
    ... )

    The same loop over relaxation paths. Each structure graduates once its max
    force norm falls below ``0.05``, and the next initial structure takes its
    place:

    >>> config = OnPolicyConfig(  # doctest: +SKIP
    ...     dynamics=FIRE(student, dt=0.1),
    ...     teacher_scorer=InProcessTeacherScorer(teacher, ["energy", "forces"]),
    ...     initial_structures=InitialStructures(dataset, recycle=True),
    ...     fmax=0.05,
    ...     replay_ratio=0.25,
    ...     training_steps_per_segment=32,
    ...     batch_size=16,
    ...     generation_steps=50,
    ...     label_frequency=10,
    ... )

    Notes
    -----
    Any :class:`~nvalchemi.training.distillation.TeacherScorer` may drive
    generation. A custom scorer can declare ``label_fields``. The declaration
    lets :class:`~nvalchemi.training.distillation.DistillationStrategy` check
    the generated fields against ``reference_dataset`` at construction. It
    also keeps :class:`~nvalchemi.training.distillation.TeacherLabelHook` from
    re-scoring a frame that is dispatched again. A custom ``teacher_*`` field
    the scorer writes is an ordinary loss target, so the reference dataset and
    any validation data must carry it too.

    The loop sizes ``capture_sink``. A segment captures at most one frame per
    trajectory per labeled step, including the forced last frame. The sink
    must therefore hold ``(generation_steps + 1)`` frames per trajectory in
    the propagated batch. A configured sink with less capacity is grown
    through ``resize(capacity)`` when it satisfies
    :class:`~nvalchemi.dynamics.ResizableSink`, and rejected otherwise. The
    sink must also be empty when a segment starts, because everything it holds
    is drained into the replay buffer as generated frames.

    ``capture_sink`` is runtime-only, like ``dynamics`` and
    ``teacher_scorer``, so no recipe names it. A recipe does not name a policy
    instance either. :attr:`settings` records a custom ``replay_eviction`` as
    ``"fifo"`` with a warning, and a config rebuilt from those settings evicts
    FIFO until the policy is supplied again.

    ``fmax`` stays a plain number that a recipe can hold.
    :attr:`convergence_criterion` is the live criterion the lifecycle drives.
    For an ``fmax`` threshold, it is built once with
    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.from_fmax`, migrating
    status ``0`` to the propagator's ``exit_status``. The same object is
    returned on every read, because the lifecycle registers that one object
    and removes it by identity. A criterion that has to be a live hook goes to
    ``convergence_hook`` instead. Setting both ``fmax`` and
    ``convergence_hook`` is refused.

    A ``convergence_hook`` must migrate status on every step, from the status
    ``0`` the run stamps its structures with. A hook that only reports
    convergence would freeze and graduate nothing. A hook that skips steps
    graduates a structure late, so both capture routes would store it. The
    construction probe also dispatches a copy of the criterion to the probed
    row. A criterion that raises on the propagator's outputs, or whose firing
    leaves ``status`` unchanged, is therefore refused here. A criterion that
    reads a key no ``compute()`` produces is not dispatched, because a hook
    may write that key during the step; a warning names the key instead.
    ``probe=False`` skips this dispatch along with the forward pass it reads.

    ``divergence`` ends a trajectory before it converges. It has the same
    shape as :class:`~nvalchemi.training.distillation.AdmissionPolicy`: it
    takes the live frame and returns one boolean per graph. The default flags
    a graph whose positions or forces are no longer finite. Use a custom
    predicate for a student whose forces explode to finite but unphysical
    values, or for a criterion on the energy. Like ``replay_admission``, the
    predicate is runtime-only. A predicate that returns anything but one
    boolean per graph is refused on its first dispatch, and the error names
    the shape it returned.

    For the duration of the loop, the criterion also becomes the propagator's
    convergence detector. It must be the only hook that migrates status, so a
    propagator that carries a second migrating
    :class:`~nvalchemi.dynamics.base.ConvergenceHook` is refused. A
    :class:`~nvalchemi.dynamics.FusedStage` builds such a hook for every
    sub-stage except the last, and for the last one whenever it declares a
    ``convergence_hook``. Only a single-sub-stage fused stage without a
    criterion of its own is therefore accepted. A
    :class:`~nvalchemi.distributed.DomainParallel` propagator is refused with
    a criterion as well: its step dispatches no ``ON_GRADUATE``, the stage
    the converged route captures at, so its minima would never be stored. A
    migrator already on the propagator is refused when the config is built,
    and one registered afterwards is refused when the run starts. See
    :ref:`training-distillation-api` for the capture routes and the backfill.

    Distribution-matching objectives are defined on equilibrium ensembles, and
    a relaxation path is not one. A
    :class:`~nvalchemi.training.distillation.BoltzmannMatchingLoss` is
    therefore refused at construction beside a relaxation propagator or any
    convergence criterion, unless ``samples_equilibrium=True`` declares that
    the propagator does sample an equilibrium ensemble. ``False`` refuses the
    term whatever the propagator is. The term should also have a bounded
    ``replay_capacity``. Energy, force, and atomic-energy matching are
    pointwise, so they distill a relaxation path exactly as they distill a
    trajectory.
    """

    dynamics: Annotated[
        BaseDynamics,
        Field(
            description=(
                "Propagator generating on-policy frames from the student. Any "
                "BaseDynamics: an integrator for trajectories, an optimizer for "
                "relaxation paths."
            )
        ),
    ]
    teacher_scorer: Annotated[
        TeacherScorer,
        Field(
            description=(
                "Scorer producing the teacher signals for generated frames. A "
                "label_fields declaration on a custom one lets the strategy "
                "check its fields against reference_dataset up front and makes a "
                "teacher_* field of its own usable as a loss target."
            )
        ),
    ]
    initial_structures: Annotated[
        InitialStructuresSource,
        Field(
            description=(
                "Structures the generated trajectories start from: any "
                "InitialStructuresSource, of which InitialStructures is the "
                "reference. The initial batch, the backfill, and a restart all "
                "read from its one position. A bare dataset is wrapped in an "
                "unbudgeted InitialStructures."
            )
        ),
    ]
    replay_eviction: Annotated[
        ReplayEviction | EvictionPolicy,
        Field(
            default="fifo",
            description=(
                "Policy retiring frames from a full replay buffer: 'fifo', or a "
                "live EvictionPolicy instance, which is runtime-only and "
                "recorded as 'fifo' in the declarative settings."
            ),
        ),
    ] = "fifo"
    replay_admission: Annotated[
        AdmissionPolicy | None,
        Field(
            default=None,
            description=(
                "Predicate over a batch of captured frames returning one boolean "
                "per graph; frames it refuses never enter the replay buffer. "
                "Runtime-only: no recipe names it."
            ),
        ),
    ] = None
    capture_sink: Annotated[
        DataSink | None,
        Field(
            default=None,
            description=(
                "Runtime-only sink the labeling hook stages each segment's "
                "labeled frames in until the segment boundary drains them into "
                "the replay buffer; None builds a host-memory sink per segment, "
                "and a GPUBuffer keeps the staging on the generation device. "
                "The loop sizes it to (generation_steps + 1) frames per "
                "trajectory, growing a ResizableSink through resize(capacity) "
                "and refusing a smaller sink otherwise."
            ),
        ),
    ] = None
    convergence_hook: Annotated[
        ConvergenceHook | None,
        Field(
            default=None,
            description=(
                "Live criterion deciding when a generated trajectory is "
                "finished, in place of the fmax threshold. No recipe "
                "describes it, so it is runtime-only."
            ),
        ),
    ] = None
    divergence: Annotated[
        Callable[[Batch], Bool[torch.Tensor, "G"]] | None,
        Field(
            default=None,
            description=(
                "Predicate over the live frame returning one boolean per graph, "
                "set where the trajectory diverged; the lifecycle moves the "
                "verdict onto the batch's device, then freezes and retires "
                "those graphs uncaptured. None flags non-finite positions or "
                "forces. Runtime-only: no recipe names it."
            ),
        ),
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _probed: bool = PrivateAttr(default=False)
    _convergence_criterion: ConvergenceHook | None = PrivateAttr(default=None)

    @property
    def settings(self) -> OnPolicySettings:
        """Detached copy of the declarative settings, for a recipe or a restart bundle.

        Returns
        -------
        OnPolicySettings
            The scalar settings of this config, validated on their own. The
            copy holds no reference to the live objects.

        Warns
        -----
        UserWarning
            If ``replay_eviction`` is a policy instance other than
            :class:`~nvalchemi.training.distillation.FIFO`. The copy records
            it as ``"fifo"``.
        """
        values = {name: getattr(self, name) for name in OnPolicySettings.model_fields}
        eviction = self.replay_eviction
        if not isinstance(eviction, str):
            if not isinstance(eviction, FIFO):
                warnings.warn(
                    f"replay_eviction is a {type(eviction).__name__} instance, "
                    "which no recipe or restart bundle can name, so the "
                    "declarative settings record 'fifo'; a config rebuilt from "
                    "them evicts FIFO until the policy is re-supplied at "
                    "construction.",
                    UserWarning,
                    stacklevel=2,
                )
            values["replay_eviction"] = "fifo"
        return OnPolicySettings.model_validate(values)

    @property
    def convergence_criterion(self) -> ConvergenceHook | None:
        """Return the criterion the trajectory lifecycle drives, or ``None``.

        A ``convergence_hook`` is returned as is. An ``fmax`` threshold is
        turned into a criterion on first read, migrating status ``0`` to the
        propagator's ``exit_status``. The same object is returned for the life
        of the config, because the lifecycle registers it on the propagator
        and removes it again by identity.

        Returns
        -------
        ConvergenceHook | None
            The live criterion, or ``None`` for a run managing no lifecycle.
        """
        if self.convergence_hook is not None:
            return self.convergence_hook
        if self.fmax is None:
            return None
        if self._convergence_criterion is None:
            self._convergence_criterion = ConvergenceHook.from_fmax(
                float(self.fmax),
                source_status=0,
                target_status=self.dynamics.exit_status,
            )
        return self._convergence_criterion

    @model_validator(mode="before")
    @classmethod
    def _coerce_initial_structures(cls, data: Any) -> Any:
        """Pass a source through, wrap a bare dataset, and refuse anything else."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        structures = data.get("initial_structures")
        if structures is None or isinstance(structures, InitialStructuresSource):
            return data
        if isinstance(structures, BatchDatasetProtocol):
            data["initial_structures"] = InitialStructures(structures)
            return data
        raise ValueError(
            "OnPolicyConfig.initial_structures must be an InitialStructuresSource "
            "(an object with probe, initial_batch, shard, exhausted, draw, "
            "state_dict, and load_state_dict, as InitialStructures implements "
            "them) or a BatchDatasetProtocol dataset to wrap in one; got "
            f"{type(structures).__name__!r}. Pass an InitialStructures, another "
            "source, or a dataset."
        )

    @model_validator(mode="after")
    def _validate_convergence_hook(self) -> OnPolicyConfig:
        """Validate a ``convergence_hook``; an ``fmax`` threshold needs no checks."""
        if self.convergence_hook is None:
            return self
        if self.fmax is not None:
            raise ValueError(
                "Set fmax or convergence_hook, not both; got "
                f"fmax={self.fmax!r}, convergence_hook={self.convergence_hook!r}. "
                "Drop the threshold to keep the hook, or drop the hook to keep a "
                "config a recipe can describe."
            )
        exit_status = self.dynamics.exit_status
        migrates = (
            self.convergence_hook.source_status is not None
            and self.convergence_hook.target_status is not None
        )
        if not migrates:
            raise ValueError(
                "The convergence hook of a relaxation loop has to migrate "
                f"status; got source_status={self.convergence_hook.source_status!r} "
                f"and target_status={self.convergence_hook.target_status!r}. A "
                "converged graph freezes in the propagator's step on its status. "
                "It graduates out of the batch on that status too. Pass "
                f"source_status=0 with target_status={exit_status!r}, or pass "
                "the threshold as fmax instead."
            )
        if self.convergence_hook.target_status < exit_status:
            raise ValueError(
                "Converged graphs must migrate to at least the propagator's "
                "exit status, which is what graduates them out of the active "
                f"batch; got target_status="
                f"{self.convergence_hook.target_status!r} against "
                f"dynamics.exit_status={exit_status!r}."
            )
        if self.convergence_hook.frequency != 1:
            raise ValueError(
                "The convergence hook of a relaxation loop has to run on every "
                f"step; got frequency={self.convergence_hook.frequency!r}. A "
                "structure is captured on the step it converges. It has to be "
                "frozen and left out of the path capture on that same step. A "
                "hook that skips steps would store it by both routes and keep "
                "propagating it until the next firing. Pass frequency=1, or pass "
                "the threshold as fmax instead."
            )
        return self

    @model_validator(mode="after")
    def _validate_lifecycle_shape(self) -> OnPolicyConfig:
        """Reject a lifecycle the structures or the propagator cannot carry."""
        managed = self.fmax is not None or self.convergence_hook is not None
        if getattr(self.initial_structures, "recycle", False) and not managed:
            raise ValueError(
                "The initial-structures source sets recycle=True; got "
                f"fmax={self.fmax!r} and "
                f"convergence_hook={self.convergence_hook!r}. Recycling wraps "
                "the backfill to the front of the rows, and only a run managing "
                "a trajectory lifecycle ever backfills. Pass fmax or a "
                "convergence_hook, or drop recycle."
            )
        if not managed:
            return self
        if isinstance(self.dynamics, DomainParallel):
            raise ValueError(
                "A DomainParallel propagator cannot carry a trajectory lifecycle: "
                "its step dispatches no ON_GRADUATE stage, so no converged frame "
                f"would ever be captured; got dynamics={self.dynamics!r} with "
                f"fmax={self.fmax!r} and convergence_hook={self.convergence_hook!r}. "
                "Pass the dynamics it wraps, or drop fmax and convergence_hook."
            )
        _check_sole_migrator(self.dynamics, self.convergence_criterion)
        return self

    @model_validator(mode="after")
    def _validate_structure_fields(self) -> OnPolicyConfig:
        """Check one row against the propagator's declared keys, its compute(), and the criterion.

        The forward pass and the criterion dispatch run only with
        ``probe=True``, and only once per instance. The after-validators run
        again when the config is passed into a strategy, and that second pass
        skips both.
        """
        probe = self.initial_structures.probe()
        self.dynamics.check_initial_batch(probe)
        if self._probed or not self.probe:
            return self
        probed = _probe_propagator(probe, self.dynamics)
        criterion = self.convergence_criterion
        if probed is not None and criterion is not None:
            _probe_criterion(probed, self.dynamics, criterion)
        self._probed = True
        return self

    def to_spec_dict(self, *, teacher: BaseModelMixin | None = None) -> dict[str, Any]:
        """Serialize the segment loop to a recipe's JSON-ready ``on_policy`` block.

        Every setting :class:`OnPolicySettings` declares round-trips as itself.
        The three live objects round-trip as references instead. The
        propagator becomes the spec it rebuilds from, and the student is
        rebound at construction. The scorer becomes its signal set, its dtype,
        its probe seed, and the name of the strategy model it scores with.
        ``initial_structures`` becomes the store it reads and the budgets it
        was given, without its position: the position is state a restart
        bundle carries, not configuration. Another
        :class:`~nvalchemi.training.distillation.InitialStructuresSource`
        becomes its own ``to_spec_dict`` block under its class path.

        A *runtime-only field* holds a live object that no recipe can
        describe. This method leaves such a field out of the block, and the
        caller re-supplies it at construction. Runtime-only fields include
        ``convergence_hook``, ``capture_sink``, ``replay_admission``,
        ``divergence``, and a policy instance on ``replay_eviction``, which
        the settings record as ``"fifo"``, silently for a
        :class:`~nvalchemi.training.distillation.FIFO` and with a warning for
        any other policy. The ``fmax`` threshold beside ``convergence_hook``
        is a
        setting, and it travels. The hooks, sinks, and convergence hook a
        propagator may carry are left out too, and so is the in-flight state
        of a run, which travels in a checkpoint rather than in a spec.

        Parameters
        ----------
        teacher : BaseModelMixin | None, optional
            Model the recipe's ``"teacher"`` reference resolves to, checked
            against the scorer's own. Default ``None`` (unchecked).

        Returns
        -------
        dict[str, Any]
            JSON-ready ``on_policy`` block suitable for :func:`json.dumps`.

        Raises
        ------
        ValueError
            If no spec can describe the propagator (no import reaches its
            class, or a hand-built one hides the arguments it was built with),
            if the scorer is neither an
            :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`
            over *teacher* nor a :class:`SpecSerializable`, if the initial
            structures' dataset holds its samples in memory, or if the initial
            structures are a source with no ``to_spec_dict`` /
            ``from_spec_dict`` to be named by.

        Warns
        -----
        UserWarning
            If the propagator carries hooks or other live collaborators that a
            rebuilt one starts without, if ``convergence_hook`` is set, if
            ``capture_sink``, ``replay_admission``, or ``divergence`` is set,
            or if ``replay_eviction`` is a policy instance other than
            :class:`~nvalchemi.training.distillation.FIFO`.
        """
        spec: dict[str, Any] = {
            "dynamics": _dynamics_spec_dict(self.dynamics),
            "teacher_scorer": _scorer_spec_dict(self.teacher_scorer, teacher),
            "initial_structures": _source_spec_dict(self.initial_structures),
            **self.settings.model_dump(mode="json"),
        }
        omitted = [
            name for name in _RUNTIME_ONLY_FALLBACKS if getattr(self, name) is not None
        ]
        if omitted:
            names = _and_list([f"OnPolicyConfig.{name}" for name in omitted])
            fallbacks = _and_list([_RUNTIME_ONLY_FALLBACKS[name] for name in omitted])
            if len(omitted) == 1:
                verb, omission, pronoun = "holds a runtime object", "it is", "it"
            else:
                verb, omission, pronoun = "hold runtime objects", "they are", "them"
            warnings.warn(
                f"{names} {verb} no recipe describes, so {omission} omitted; a "
                f"rebuilt loop {fallbacks}. Re-supply {pronoun} at construction.",
                UserWarning,
                stacklevel=2,
            )
        if self.convergence_hook is not None:
            warnings.warn(
                "OnPolicyConfig.convergence_hook is a live ConvergenceHook no "
                "recipe describes, so it is omitted; set fmax to keep the "
                "criterion in the recipe, or re-supply the hook at "
                "construction.",
                UserWarning,
                stacklevel=2,
            )
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        student: BaseModelMixin,
        teacher: BaseModelMixin,
        device: torch.device | str | None = None,
    ) -> OnPolicyConfig:
        """Rebuild a segment loop from a :meth:`to_spec_dict` ``on_policy`` block.

        The settings are validated first, on their own, as
        :class:`OnPolicySettings`. A block carrying an out-of-range scalar is
        therefore refused before a store is opened or a propagator is built.

        Parameters
        ----------
        spec : Mapping[str, Any]
            ``on_policy`` block produced by :meth:`to_spec_dict`, optionally
            after a JSON round trip.
        student : BaseModelMixin
            Model the rebuilt propagator generates with. It must be the same
            module object the strategy trains, because that is what makes the
            data on-policy.
        teacher : BaseModelMixin
            Model the rebuilt scorer labels with.
        device : torch.device | str | None, optional
            Device the rebuilt initial structures collate onto. A
            ``replay_device`` set in the block follows it too. Default
            ``None`` (the device the block recorded). A strategy rebuilding its
            loop passes its own primary device, so a checkpoint written on one
            device restores its data where the run now trains.

        Returns
        -------
        OnPolicyConfig
            Config equal to the serialized one on every field a recipe carries.
            The rebuilt initial structures open at their first row. Only a
            restart bundle resumes them mid-run.

        Raises
        ------
        ValueError
            If a propagator keyword argument has no JSON representation, if
            the propagator spec builds something that is not a
            :class:`~nvalchemi.dynamics.base.BaseDynamics`, or if the rebuilt
            config is invalid.
        """
        settings = _on_policy_settings(spec)
        if device is not None and settings.replay_device is not None:
            settings = settings.model_copy(update={"replay_device": str(device)})
        return cls(
            **settings.model_dump(),
            dynamics=_dynamics_from_spec_dict(spec["dynamics"], student),
            teacher_scorer=_scorer_from_spec_dict(spec["teacher_scorer"], teacher),
            initial_structures=_source_from_spec_dict(
                spec["initial_structures"], device
            ),
        )
