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

import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.sinks import DataSink
from nvalchemi.training.distillation.replay import (
    FIFO,
    AdmissionPolicy,
    EvictionPolicy,
    ReplayEviction,
    _batch_allocation,
    _batch_size_remedy,
)
from nvalchemi.training.distillation.scoring import (
    TeacherScorer,
    _isolated_neighbors,
    _planned_neighbor_sources,
)
from nvalchemi.training.distillation.seeding import (
    InitialStructures,
    InitialStructuresSource,
    _check_structure_fields,
)
from nvalchemi.training.runtime import evaluating

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = ["OnPolicyConfig", "OnPolicySettings", "ResizableSink"]


@runtime_checkable
class ResizableSink(Protocol):
    """Sink the segment loop can grow to the capacity one segment needs.

    A :class:`~nvalchemi.dynamics.sinks.DataSink` fixes its capacity at
    construction, so a ``capture_sink`` configured smaller than the frames one
    segment captures is refused unless it also satisfies this protocol, in
    which case the loop calls ``resize`` before the segment starts.
    """

    capacity: int

    def resize(self, capacity: int) -> None:
        """Grow the sink so it holds at least *capacity* frames."""
        ...


def _probe_propagator(probe: Batch, dynamics: BaseDynamics) -> Batch | None:
    """Run one ``compute()`` on *probe* and hold the propagator to its declarations.

    :func:`~nvalchemi.training.distillation.seeding._check_structure_fields`
    compares the declared keys with the initial structures; this compares them
    with what ``compute()`` actually does, so a propagator whose declarations
    have drifted from its implementation — a ``__needs_keys__`` output the
    student does not produce, a field read that nothing declared — is refused
    here rather than on the first step of a long run. The cost is one student
    forward at construction, which front-loads the kernel and CUDA
    initialization the first step pays anyway.

    The forward runs under the scorer's isolation: the model is held in
    evaluation mode with every submodule's own flag restored afterwards,
    ``compute()`` restores the ``requires_grad``
    flags it enables, the propagator's ``_last_outputs`` is put back, and the
    probe batch is the caller's own row, moved to the model's device. A graph
    model gets the neighbor list its ``neighbor_config`` declares, built on the
    probe and rolled back afterwards, since a propagator's list is otherwise a
    hook's to build; a model planning more than one neighbor-list source is not
    probed, because that builder makes exactly one list and the check must not
    refuse a propagator the loop can run, and a warning says so.

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
        wrote; ``None`` when the propagator was not probed.

    Raises
    ------
    ValueError
        If the model produced no output for a declared ``__needs_keys__``
        entry, if ``compute()`` read a field the probe does not carry, or if
        a declared ``__provides_keys__`` entry is absent afterwards.

    Warns
    -----
    UserWarning
        If the model plans more than one neighbor-list source, so the
        propagator's declarations go unchecked until its first step.
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


class OnPolicySettings(BaseModel):
    """Declarative settings of one on-policy distillation segment loop.

    Every field is a JSON scalar, so the whole set validates without a
    propagator, a teacher, or a store, and a recipe's settings can be refused
    before a teacher is loaded. :class:`OnPolicyConfig` inherits them and adds
    the live objects the loop drives; whether those objects compose with the
    loop is settled there and in the strategy.

    Parameters
    ----------
    replay_ratio : float
        Fraction of every training batch drawn from the replay buffer.
    training_steps_per_segment : int
        Optimizer steps taken per segment, one per training batch.
    batch_size : int, optional
        Samples per training batch, across both mixture sources. Default ``8``.
    generation_steps : int, optional
        Propagator steps generated per segment. Default ``100``.
    label_frequency : int, optional
        Propagator steps between teacher labelings, on top of each segment's
        last frame. Default ``100``.
    replay_capacity : int | None, optional
        Frame capacity of the replay buffer. Default ``None`` (unbounded).
    replay_eviction : {"fifo"}, optional
        Eviction policy of the replay buffer, named for a recipe. Default
        ``"fifo"``; a policy instance goes on :class:`OnPolicyConfig`.
    replay_device : str | None, optional
        Device the replay buffer keeps frames on. Default ``None`` (where the
        reference dataset emits its batches; host memory without one).
    seed : int, optional
        Base seed of every segment's mixture sampler. Default ``0``.
    weight_sync_frequency : int, optional
        Segments between weight syncs to the propagator. Default ``1``, the
        only accepted value while the propagator shares the student module.
    probe : bool, optional
        Whether :class:`OnPolicyConfig` runs the propagator's ``compute()`` on
        one initial structure at construction to hold it to its declared keys.
        Default ``True``; ``False`` defers any mismatch to the first step.

    Raises
    ------
    ValueError
        If a count is not positive, if ``replay_ratio`` falls outside
        ``[0, 1]`` or is exactly ``0``, if the ratio and the batch size
        together round a mixture source out of every batch, or if
        ``weight_sync_frequency`` is not ``1``.

    Examples
    --------
    >>> from nvalchemi.training.distillation import OnPolicySettings
    >>> settings = OnPolicySettings(replay_ratio=0.25, training_steps_per_segment=32)
    >>> settings.batch_size
    8

    Notes
    -----
    ``label_frequency`` is the throughput setting, counted against the
    propagator's cumulative ``step_count`` so the cadence does not restart at a
    segment boundary; each segment also labels the frame it ends on, and the
    cadence dispatch adjacent to that forced label is passed over, so
    ``generation_steps`` a multiple of ``label_frequency`` labels each
    trajectory once per segment. ``training_steps_per_segment`` is a budget of
    training batches, which is a budget of optimizer steps only while every
    batch takes one. Size ``replay_capacity`` as a multiple of the trajectory
    count, since FIFO eviction otherwise cuts a segment's contribution mid-step
    and over-represents the back of the batch, and space the ``seed`` of
    replicate runs by at least ``num_steps // training_steps_per_segment``,
    since the sampler adds it to the segment index. See
    :ref:`training-distillation-api`.
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
                "Device the replay buffer holds frames on, named as a string. "
                "Generated frames reach it from a host-memory sink, so None "
                "stages them where the reference dataset actually emits its "
                "own batches — the mixture is collated before training moves "
                "it — and leaves them in host memory when the run has no "
                "reference dataset. Set it only to override that, and load the "
                "reference dataset there too."
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

    model_config = ConfigDict(extra="forbid")

    @field_validator("replay_device", mode="before")
    @classmethod
    def _name_replay_device(cls, value: Any) -> Any:
        """Accept a torch.device for a setting every reader names as a string."""
        return str(value) if isinstance(value, torch.device) else value

    @model_validator(mode="after")
    def _validate_weight_sync(self) -> OnPolicySettings:
        """Hold the reserved sync setting at 1 until the decoupled paths land."""
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
        """Reject a mixture no batch can actually be drawn from."""
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
            "The mixture is drawn as whole samples of a batch, so replay_ratio "
            "and batch_size only mean something together; got replay_ratio="
            f"{self.replay_ratio!r} with batch_size={self.batch_size!r}, which "
            f"puts {reference_samples} reference and {replay_samples} generated "
            "samples in every batch and leaves one source out of training "
            f"entirely; {_batch_size_remedy(self.replay_ratio)}."
        )


class OnPolicyConfig(OnPolicySettings):
    """One on-policy distillation segment loop, settings and live objects together.

    A *generation* phase runs the student's own propagator for
    ``generation_steps`` steps, labeling frames with the teacher as it goes; a
    *training* phase then takes ``training_steps_per_segment`` optimizer steps
    on batches mixed from the reference dataset and the replay buffer at
    ``replay_ratio``. The propagator holds the module the trainer updates, so
    each segment generates from a fresher policy than the last. The scalar half
    is :class:`OnPolicySettings`, inherited so a recipe stays flat; :attr:`settings`
    is the detached copy a pre-flight or a restart bundle carries.

    The propagator is any :class:`~nvalchemi.dynamics.base.BaseDynamics`, so a
    relaxation optimizer such as :class:`~nvalchemi.dynamics.optimizers.FIRE`
    drives the loop exactly as a thermostat does. Initial structures must carry
    whatever it updates in place through ``__provides_keys__`` — ``velocities``
    for every shipped propagator, plus a ``cell`` for the variable-cell ones;
    the model outputs of ``__needs_keys__`` are primed before the first step —
    and one row is checked here, so a missing field is a construction error
    rather than a failure on the first step. The propagator's ``compute()``
    then runs once on that row, so declarations that have drifted from the
    implementation — a ``__needs_keys__`` output the student never produces, a
    field ``compute()`` reads that nothing declared — are refused here too, at
    the cost of one student forward at construction. A graph model is probed
    with the neighbor list its ``neighbor_config`` declares, built on the row
    and rolled back; a model planning more than one neighbor-list source is
    not probed.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator generating on-policy frames, holding the student module.
    teacher_scorer : TeacherScorer
        Scorer labeling generated frames. Declaring ``label_fields`` on a
        custom one makes the fields it writes knowable up front.
    initial_structures : InitialStructuresSource
        Structures the generated trajectories start from, behind the cursor a
        restart resumes: an :class:`~nvalchemi.training.distillation.InitialStructures`,
        any other object implementing the protocol, or a bare dataset, which is
        wrapped.
    capture_sink : DataSink | None, optional
        Sink each segment's labeled frames are staged in before the segment
        boundary drains them into the replay buffer. Default ``None``, a
        host-memory sink built per segment; a
        :class:`~nvalchemi.dynamics.sinks.GPUBuffer` keeps the staging on the
        generation device instead of paying a device-to-host copy per frame.
    replay_eviction : {"fifo"} | EvictionPolicy, optional
        The setting widened to a live
        :class:`~nvalchemi.training.distillation.EvictionPolicy` instance.
        Default ``"fifo"``.
    replay_admission : AdmissionPolicy | None, optional
        Predicate masking the frames each segment admits into the replay
        buffer. Default ``None`` (every captured frame enters).

    Raises
    ------
    ValueError
        If a setting is out of range, if ``initial_structures`` is neither a
        source nor a dataset, if the initial structures lack a field the
        propagator opens its step with, or if the propagator's ``compute()``
        on one row contradicts its declared keys.

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

    Notes
    -----
    Any :class:`~nvalchemi.training.distillation.TeacherScorer` may drive
    generation. Declaring ``label_fields`` on a custom one lets
    :class:`~nvalchemi.training.distillation.DistillationStrategy` check the
    generated fields against ``reference_dataset`` at construction and keeps
    :class:`~nvalchemi.training.distillation.TeacherLabelHook` from re-scoring
    a re-dispatched frame; a custom ``teacher_*`` field it writes is an
    ordinary loss target the reference dataset and any validation data must carry too.

    The loop owns the sizing of ``capture_sink``: a segment captures at most
    one frame per trajectory per labeled step, the forced last frame included,
    so the sink has to hold ``(generation_steps + 1)`` frames per trajectory
    of the batch being propagated. A configured sink with less capacity is
    resized through ``resize(capacity)`` when it satisfies
    :class:`ResizableSink` and refused otherwise, and it has to be empty when a
    segment starts, since everything
    it holds is drained into the replay buffer as generated frames. It is
    runtime-only, like ``dynamics`` and ``teacher_scorer``: no recipe names it,
    and neither does one name a policy instance — :attr:`settings` records a
    custom ``replay_eviction`` as ``"fifo"`` with a warning, and a config
    rebuilt from it evicts FIFO until the policy is re-supplied.
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
                "Structures the generated trajectories are seeded from, behind "
                "the cursor the initial batch and a restart share: any "
                "InitialStructuresSource, of which InitialStructures is the "
                "reference. A bare dataset is wrapped in an unbudgeted one."
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
                "Sink the labeling hook stages each segment's labeled frames in "
                "before they are drained into the replay buffer. None builds a "
                "host-memory sink per segment; a GPUBuffer keeps the staging on "
                "the generation device. The loop sizes it to (generation_steps "
                "+ 1) frames per trajectory, resizing through resize(capacity) "
                "when the sink is a ResizableSink and refusing a smaller one "
                "otherwise. "
                "Runtime-only: no recipe names it."
            ),
        ),
    ] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _probed: bool = PrivateAttr(default=False)

    @property
    def settings(self) -> OnPolicySettings:
        """Detached copy of the declarative half, for a recipe or a bundle.

        Returns
        -------
        OnPolicySettings
            The scalars this config carries, validated on their own and holding
            no reference back to the live objects beside them.

        Warns
        -----
        UserWarning
            If ``replay_eviction`` is a policy instance other than
            :class:`~nvalchemi.training.distillation.FIFO`, which the copy
            records as ``"fifo"``.
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
            "— probe, initial_batch, shard, exhausted, draw, state_dict, and "
            "load_state_dict, as InitialStructures implements them — or a "
            "BatchDatasetProtocol dataset to wrap in one; got "
            f"{type(structures).__name__!r}."
        )

    @model_validator(mode="after")
    def _validate_structure_fields(self) -> OnPolicyConfig:
        """Check one row against the propagator's declarations, then its compute().

        The forward runs once per instance and only with ``probe=True``: the
        after-validators run again when the config is passed into a strategy,
        and that pass skips it.
        """
        probe = self.initial_structures.probe()
        _check_structure_fields(probe, self.dynamics)
        if self.probe and not self._probed:
            _probe_propagator(probe, self.dynamics)
            self._probed = True
        return self
