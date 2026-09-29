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
"""Offline knowledge-distillation strategy built on :class:`TrainingStrategy`."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import Field, PrivateAttr, model_validator

from nvalchemi._serialization import _dtype_deserialize, _import_cls
from nvalchemi._typing import ModelOutputs
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import TrainingStage
from nvalchemi.training import _spec_utils as strategy_spec
from nvalchemi.training import _strategy_validation as strategy_validation
from nvalchemi.training.distillation._attach import _attach_teacher_labels
from nvalchemi.training.distillation.scoring import (
    _EMBEDDING_KEYS,
    _TEACHER_FIELD_PREFIX,
    InProcessTeacherScorer,
    signal_fields,
    signal_for_field,
)
from nvalchemi.training.losses.composition import loss_target_keys
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.hooks._context import TrainContext
    from nvalchemi.training import ValidationConfig
    from nvalchemi.training.losses.composition import (
        BaseLossFunction,
        ComposedLossFunction,
    )

__all__ = ["DistillationStrategy", "default_distillation_fn"]

_REQUIRED_MODELS = frozenset({"student", "teacher"})
"""Model names every distillation strategy must be given."""

_PREDICTION_KEY_PREFIX = "predicted_"
"""Prefix the stock training function publishes every student output under."""


def default_distillation_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Run the student forward pass and prefix output keys with ``predicted_``.

    The teacher is never called here: teacher knowledge reaches the loss as
    ``teacher_*`` batch fields, either written offline by
    :func:`~nvalchemi.training.distillation.label_dataset` or attached to the
    batch by :meth:`DistillationStrategy.attach_teacher_labels`.

    Parameters
    ----------
    models : Mapping[str, BaseModelMixin]
        Named models of the strategy; only ``"student"`` is read.
    batch : Batch
        Input batch of atomic graphs.

    Returns
    -------
    dict[str, torch.Tensor]
        Predictions keyed by ``predicted_<output_name>`` with ``None`` outputs
        omitted.
    """
    outputs: ModelOutputs = models["student"](batch)
    return {
        f"{_PREDICTION_KEY_PREFIX}{key}": value
        for key, value in outputs.items()
        if value is not None
    }


def _derived_teacher_signals(loss_fn: ComposedLossFunction) -> frozenset[str]:
    """Return the built-in teacher signals the loss composition's targets require.

    A ``teacher_*`` target no built-in signal populates is a custom teacher
    field — one :func:`~nvalchemi.training.distillation.label_dataset` persisted
    from a custom scorer — that the batch must already carry, so it is passed
    over here rather than refused.
    """
    signals: set[str] = set()
    for key in loss_target_keys(loss_fn):
        signal = (
            signal_for_field(key) if key.startswith(_TEACHER_FIELD_PREFIX) else None
        )
        if signal is not None:
            signals.add(signal)
    return frozenset(signals)


def _set_rebuild_overrides(
    strategy_cls: type[DistillationStrategy], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the *overrides* that are set, checked against *strategy_cls*'s signature.

    A subclass overriding ``from_spec_dict`` with the base signature of an
    earlier release knows nothing of a later optional keyword, so an unset
    override is dropped instead of forwarded and a plain rebuild keeps working.
    A set one the subclass cannot take is refused here, where the keyword and
    the class can be named, rather than as a bare ``TypeError`` from the call.
    """
    parameters = inspect.signature(strategy_cls.from_spec_dict).parameters
    takes_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    forwarded: dict[str, Any] = {}
    for name, value in overrides.items():
        if value is None:
            continue
        if not takes_var_keyword and name not in parameters:
            raise TypeError(
                f"from_spec_dict: {strategy_cls.__name__}.from_spec_dict does not "
                f"accept the {name!r} keyword, so the supplied {value!r} cannot be "
                f"applied; add {name} to its signature or drop the override."
            )
        forwarded[name] = value
    return forwarded


def _student_label_dtype(student: BaseModelMixin) -> torch.dtype | None:
    """Return the dtype teacher labels are cast to for *student*.

    The first floating-point parameter decides, floored at single precision: a
    store reads labels back at the dataset's ``positions`` dtype, float32 for
    essentially every dataset, so a narrower on-the-fly label would disagree
    with the persisted one. A student exposing no parameters gets ``None``,
    which keeps the teacher's own dtype.
    """
    parameters = getattr(student, "parameters", None)
    if not callable(parameters):
        return None
    for parameter in parameters():
        if parameter.is_floating_point():
            if parameter.dtype.itemsize < torch.float32.itemsize:
                return torch.float32
            return parameter.dtype
    return None


class _TeacherLabelHook:
    """Label the batch a forward pass is about to consume, training or validation.

    The first batch the seam labels raises one :class:`UserWarning` per strategy
    naming the missing fields, since from then on every such batch costs a
    teacher pass that a store written by
    :func:`~nvalchemi.training.distillation.label_dataset` would have spared.
    """

    frequency = 1
    stage = TrainingStage.BEFORE_FORWARD

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:  # noqa: ARG002
        """Attach the teacher fields the upcoming batch is missing."""
        strategy: DistillationStrategy = ctx.workflow
        if ctx.batch is None or not strategy.label_missing:
            return
        missing = strategy._missing_teacher_fields(ctx.batch)
        if not missing:
            return
        strategy.attach_teacher_labels(ctx.batch)
        if strategy._warned_label_seam:
            return
        strategy._warned_label_seam = True
        warnings.warn(
            "DistillationStrategy is labeling batches on the fly: a batch reached "
            f"the forward pass without the teacher fields {missing!r}, so a teacher "
            "pass now runs for every such batch, in training and validation alike. "
            "Label the dataset ahead of time with label_dataset to avoid the cost, "
            "or set label_missing=False to surface it as a missing target instead.",
            UserWarning,
            stacklevel=2,
        )


class DistillationStrategy(TrainingStrategy):
    """Train a student against a frozen teacher's signals.

    A :class:`~nvalchemi.training.TrainingStrategy` over the named models
    ``"student"`` and ``"teacher"``. The teacher is frozen by omission — it must
    not appear in ``optimizer_configs`` — while the student and any auxiliary
    model must be configured with one. Teacher knowledge reaches the loss as
    ``teacher_*`` batch fields, so a built-in term distills by pointing its
    ``target_key`` at one (``EnergyMSELoss(target_key="teacher_energy")``) and
    :class:`~nvalchemi.training.distillation.AtomicEnergyMatchingLoss` reads
    ``teacher_atomic_energies``; mixing teacher and reference targets is
    ordinary loss composition.

    The signals the teacher is asked for are derived from the ``teacher_*``
    targets the training loss and a ``validation_config`` loss read, or named
    in ``teacher_signals``, which must cover the derived set. They are checked
    against the teacher's outputs at construction, as are both losses'
    prediction keys against the outputs the student actually computes whenever
    the effective training or validation function is the stock
    :func:`default_distillation_fn`. Nothing re-runs on assignment, so pass
    ``validation_config`` to the constructor or, when rebuilding from a spec
    that excludes it, to :meth:`from_spec_dict`. Every resolved signal is required
    on every batch: a batch missing any resolved field is labeled on the fly by
    an internal ``BEFORE_FORWARD`` hook, in training and validation alike,
    unless ``label_missing=False`` leaves it to surface as a missing loss
    target, while a store written by
    :func:`~nvalchemi.training.distillation.label_dataset` with the same signal
    set trains with no teacher pass at all. A ``teacher_*`` target no built-in
    signal populates is a custom field such a store carries: it is neither
    derived nor attached, and a batch lacking it fails as a missing target.
    See :ref:`training-distillation-api` for the full contract.

    Raises
    ------
    ValueError
        If ``models`` is not a named mapping containing ``"student"`` and
        ``"teacher"``, if the teacher is given an optimizer config, if the
        student or an auxiliary model is not, if a loss component reads a
        prediction the student does not compute or names one outside the
        ``predicted_`` namespace under the stock ``training_fn``, if an explicit
        ``teacher_signals`` omits a signal a loss needs, if no built-in teacher
        signal is requested at all, if the teacher cannot produce a requested signal,
        if the teacher is a composition that plans more than one neighbor-list
        source, or if ``label_dtype`` is not a floating-point dtype.

    Examples
    --------
    Distill energies, forces, and the teacher's per-atom energy decomposition
    from a store written by :func:`label_dataset`:

    >>> import torch
    >>> from nvalchemi.training import EnergyMSELoss, ForceMSELoss, OptimizerConfig
    >>> from nvalchemi.training.distillation import (
    ...     DistillationStrategy,
    ...     AtomicEnergyMatchingLoss,
    ... )
    >>> loss_fn = (
    ...     EnergyMSELoss(target_key="teacher_energy")
    ...     + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    ...     + 0.1 * AtomicEnergyMatchingLoss()
    ... )
    >>> strategy = DistillationStrategy(  # doctest: +SKIP
    ...     models={"student": student, "teacher": teacher},
    ...     optimizer_configs={
    ...         "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
    ...     },
    ...     loss_fn=loss_fn,
    ...     num_steps=1_000,
    ... )
    >>> strategy.run(labeled_loader)  # doctest: +SKIP

    Notes
    -----
    Labeling runs with autocast disabled, and labels are cast to ``label_dtype``
    when one is given. By default it is inferred as the student's first
    floating-point parameter dtype, never below single precision, so a
    ``bfloat16`` or ``float16`` student gets float32 labels and needs
    ``dtype_policy="prediction_to_target"`` on its loss terms; a float64 student
    reads float32 back from a store and needs a ``dtype_policy`` too. Set
    ``label_dtype`` explicitly when the inference guesses wrong, as for a
    mixed-precision student whose first parameter is not representative or a
    student exposing no parameters at all, which otherwise keeps the teacher's
    own dtype. Labels are attached to the device-placed copy the strategy trains
    on, not the caller's batch, so a loader replaying the same systems costs one
    teacher pass per epoch, and the first batch labeled this way raises one
    :class:`UserWarning` naming the fields it lacked.

    Teacher conservativeness is not validated: a teacher predicting forces from
    its own head is first class, since every signal is detached before the
    student sees it. A teacher composition planning more than one neighbor-list
    source is refused, as by
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer`.

    The labeling hook is never serialized; :meth:`from_spec_dict` and
    :meth:`load_checkpoint` re-register it ahead of the caller's hooks and
    replace any carried copy. ``ValidationConfig(use_ema="auto")`` validates
    the EMA student against the live teacher (``model_source="mixed"``), while
    ``use_ema="always"`` also demands an inference-slot entry for the teacher.
    Checkpoints serialize every model, teacher included, so size the checkpoint
    interval for a large teacher.
    """

    teacher_signals: Annotated[
        frozenset[str] | None,
        Field(
            description=(
                "Teacher signals produced for every scored batch. ``None`` "
                "derives them from the ``teacher_*`` targets the training and "
                "validation losses read; an explicit set must cover those and "
                "may request more, at the cost of re-scoring every batch a "
                "store labeled without the extra fields delivers."
            )
        ),
    ] = None
    label_missing: Annotated[
        bool,
        Field(
            description=(
                "Whether a batch lacking the required ``teacher_*`` fields is "
                "labeled on the fly by a teacher forward pass, in training and "
                "validation alike. ``False`` skips the teacher, so an unlabeled "
                "batch surfaces as a missing loss target."
            )
        ),
    ] = True
    label_dtype: Annotated[
        torch.dtype | None,
        Field(
            description=(
                "Floating-point dtype teacher labels are cast to when a batch is "
                "labeled on the fly. ``None`` infers it from the student's first "
                "floating-point parameter, never below float32; an explicit dtype "
                "is passed to the teacher scorer verbatim."
            )
        ),
    ] = None

    _scorer: InProcessTeacherScorer | None = PrivateAttr(default=None)
    _teacher_fields: tuple[str, ...] = PrivateAttr(default=())
    _warned_label_seam: bool = PrivateAttr(default=False)

    @property
    def teacher_scorer(self) -> InProcessTeacherScorer:
        """Scorer producing the resolved teacher signals for one batch."""
        if self._scorer is None:
            raise RuntimeError(
                "DistillationStrategy has no teacher scorer; it is built during "
                "validation and must not be cleared."
            )
        return self._scorer

    @model_validator(mode="before")
    @classmethod
    def _default_distillation_training_fn(cls, data: Any) -> Any:
        """Fall back to the stock student-forward training function."""
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("training_fn") is None:
            normalized["training_fn"] = default_distillation_fn
        return normalized

    @model_validator(mode="before")
    @classmethod
    def _prepend_labeling_hook(cls, data: Any) -> Any:
        """Put the internal teacher-labeling hook ahead of the caller's hooks.

        A seam carried in the incoming hooks is replaced rather than kept, so
        rebuilding a strategy from a live one's ``hooks`` leaves exactly one
        labeling hook, still ahead of every caller hook.
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["hooks"] = [
            _TeacherLabelHook(),
            *(
                hook
                for hook in (normalized.get("hooks") or [])
                if not isinstance(hook, _TeacherLabelHook)
            ),
        ]
        return normalized

    @model_validator(mode="after")
    def _validate_distillation(self) -> DistillationStrategy:
        """Enforce the student/teacher contract and resolve the teacher signals."""
        missing_models = _REQUIRED_MODELS - set(self.models)
        if self.single_model_input or missing_models:
            raise ValueError(
                "DistillationStrategy needs a named-model mapping holding "
                f"'student' and 'teacher'; got models={sorted(self.models)!r}."
            )
        if "teacher" in self.optimizer_configs:
            raise ValueError(
                "The teacher is frozen by omission, so it must not appear in "
                f"optimizer_configs; got {sorted(self.optimizer_configs)!r}."
            )
        unconfigured = set(self.models) - set(self.optimizer_configs) - {"teacher"}
        if unconfigured:
            raise ValueError(
                "Every model but the teacher must be given an optimizer config; "
                f"got unconfigured {sorted(unconfigured)!r}."
            )
        if self.label_dtype is not None and not self.label_dtype.is_floating_point:
            raise ValueError(
                "label_dtype must be a floating-point dtype or None; got "
                f"{self.label_dtype!r}."
            )
        self._validate_student_outputs()
        signals = self._resolve_teacher_signals()
        self._scorer = InProcessTeacherScorer(
            self.models["teacher"],
            signals,
            dtype=(
                _student_label_dtype(self.models["student"])
                if self.label_dtype is None
                else self.label_dtype
            ),
        )
        self._teacher_fields = signal_fields(signals)
        return self

    def _validate_student_outputs(self) -> None:
        """Check both losses' prediction keys against what the student computes.

        Only under the stock function: ``default_distillation_fn`` emits exactly
        ``active_outputs`` intersected with ``outputs``, so a narrowed student
        is caught here rather than on its first batch. The validation loss is
        checked whenever ``validation_fn`` falling back to ``training_fn`` is
        the stock one.
        """
        if self.training_fn is default_distillation_fn:
            self._validate_prediction_keys(self.loss_fn.components, "training")
        validation = self.validation_config
        if validation is None or validation.loss_fn is None:
            return
        if (validation.validation_fn or self.training_fn) is default_distillation_fn:
            self._validate_prediction_keys(validation.loss_fn.components, "validation")

    def _validate_prediction_keys(
        self, components: Sequence[BaseLossFunction], side: str
    ) -> None:
        """Check one composition's prediction keys, naming *side* in every error."""
        student = self.models["student"]
        declared = student.model_config.outputs
        active = student.output_data()
        for component in components:
            key = getattr(component, "prediction_key", None)
            if key is None:
                continue
            label = f"{side} loss component {type(component).__name__!r}"
            if not key.startswith(_PREDICTION_KEY_PREFIX):
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which "
                    "default_distillation_fn never emits: it publishes every "
                    f"student output under {_PREDICTION_KEY_PREFIX}<output>. "
                    "Rename the key into that namespace, or pass a training_fn "
                    "that owns its own convention."
                )
            output = key.removeprefix(_PREDICTION_KEY_PREFIX)
            if output in active:
                continue
            if output in _EMBEDDING_KEYS:
                raise ValueError(
                    f"The {label} reads prediction_key={key!r}, which the stock "
                    "training_fn cannot produce: embeddings come from the "
                    "student's compute_embeddings(), not from its forward pass. "
                    "Pass a training_fn that calls compute_embeddings and returns "
                    f"the embedding under {key!r}."
                )
            if output in declared:
                raise ValueError(
                    "Student declares but does not compute the output required by "
                    f"the {label} reading prediction_key={key!r}; got "
                    f"active_outputs={sorted(active)!r}, missing {output!r}. Add "
                    "it to the student's model_config.active_outputs."
                )
            raise ValueError(
                f"Student cannot produce the output required by the {label} "
                f"reading prediction_key={key!r}; got outputs={sorted(declared)!r}, "
                f"missing {output!r}."
            )

    def _resolve_teacher_signals(self) -> frozenset[str]:
        """Return the signals both losses need, widened by an explicit request."""
        derived = {"training": _derived_teacher_signals(self.loss_fn)}
        validation = self.validation_config
        if validation is not None and validation.loss_fn is not None:
            derived["validation"] = _derived_teacher_signals(validation.loss_fn)
        required: frozenset[str] = frozenset().union(*derived.values())
        resolved = required if self.teacher_signals is None else self.teacher_signals
        uncovered = {
            side: sorted(signals - resolved)
            for side, signals in derived.items()
            if signals - resolved
        }
        if uncovered:
            raise ValueError(
                "teacher_signals must cover every teacher target the training and "
                f"validation losses read; got {sorted(resolved)!r}, missing "
                f"{uncovered!r}."
            )
        if not resolved:
            raise ValueError(
                "DistillationStrategy needs at least one teacher signal; got no "
                "built-in teacher_* target in the training or validation loss and "
                f"teacher_signals={self.teacher_signals!r}. A custom teacher_* field "
                "the batch already carries is not a signal; name one in "
                "teacher_signals or read a built-in teacher target."
            )
        return resolved

    def attach_teacher_labels(self, batch: Batch) -> bool:
        """Attach the teacher fields *batch* is missing, and report whether it did.

        A batch already carrying every resolved ``teacher_*`` field is left
        untouched, so pre-labeling a batch that later reaches :meth:`run` costs
        one teacher pass; a batch carrying only some of them is re-scored in
        full, since a partial set was written for a different signal set. The
        teacher runs with autocast disabled, so the labels match what
        :func:`~nvalchemi.training.distillation.label_dataset` persisted
        wherever the store returns the label dtype.

        Parameters
        ----------
        batch : Batch
            Batch to label in place, already on the teacher's device.

        Returns
        -------
        bool
            ``True`` when the teacher ran, ``False`` when *batch* already
            carried every resolved field.
        """
        if not self._missing_teacher_fields(batch):
            return False
        with torch.autocast(device_type=batch.device.type, enabled=False):
            labels = self.teacher_scorer.label(batch)
        _attach_teacher_labels(batch, labels)
        return True

    def _missing_teacher_fields(self, batch: Batch) -> list[str]:
        """Return the resolved teacher fields *batch* does not carry."""
        return [field for field in self._teacher_fields if field not in batch]

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize declarative distillation knobs to a JSON-ready dict.

        The bundle names its own class under ``strategy_cls``, which
        :meth:`from_spec_dict` builds.

        Returns
        -------
        dict[str, Any]
            JSON-ready bundle suitable for :func:`json.dumps`.
        """
        spec = super().to_spec_dict()
        spec["strategy_cls"] = f"{type(self).__module__}.{type(self).__qualname__}"
        spec["teacher_signals"] = (
            None if self.teacher_signals is None else sorted(self.teacher_signals)
        )
        spec["label_missing"] = self.label_missing
        spec["label_dtype"] = (
            None if self.label_dtype is None else str(self.label_dtype)
        )
        return spec

    @classmethod
    def from_spec_dict(
        cls,
        spec: Mapping[str, Any],
        *,
        models: strategy_validation.ModelInput | None = None,
        hooks: Sequence[Any] | None = None,
        training_fn: Any = None,
        validation_config: ValidationConfig | None = None,
    ) -> DistillationStrategy:
        """Rebuild a :class:`DistillationStrategy` from ``to_spec_dict`` output.

        A ``strategy_cls`` naming a subclass dispatches to that class's own
        ``from_spec_dict`` with the spec and every runtime override, so the
        strategy a spec names is the one that runs; a subclass adding a runtime
        keyword must widen this call with it. An optional keyword is forwarded
        only when it is set, so a subclass overriding ``from_spec_dict`` without
        it still rebuilds from a plain spec.

        Parameters
        ----------
        spec : Mapping[str, Any]
            A dict produced by :meth:`to_spec_dict`, optionally after a JSON
            round-trip.
        models : BaseModelMixin | dict[str, BaseModelMixin] | None, optional
            Runtime model override(s). Distillation models are not serialized
            in full, so the student and teacher are normally re-supplied here.
        hooks : Sequence[Any] | None, optional
            Runtime hooks; defaults to an empty list.
        training_fn : Any, optional
            Runtime callable or dotted-path override.
        validation_config : ValidationConfig | None, optional
            Runtime validation configuration. Specs exclude it because it
            carries a live loader, so a validation-only ``teacher_*`` target is
            resolved by passing the config here rather than assigning it
            afterwards, which re-runs no validator.

        Returns
        -------
        DistillationStrategy
            A freshly validated strategy of the class *spec* names, ready to
            :meth:`run`.

        Raises
        ------
        ValueError
            If *spec* is missing a required key, if its ``strategy_cls`` entry
            is not a dotted class path string, or if that path resolves to a
            class that is not a :class:`DistillationStrategy` subclass.
        TypeError
            If a runtime keyword is supplied but the subclass *spec* names
            overrides ``from_spec_dict`` without accepting it.
        """
        required = ("optimizer_configs", "devices", "loss_fn_spec")
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"from_spec_dict: spec is missing required key(s) {missing}. "
                f"Expected keys: {list(required)}."
            )
        raw_strategy_cls = spec.get("strategy_cls")
        if raw_strategy_cls is not None:
            if not isinstance(raw_strategy_cls, str):
                raise ValueError(
                    "from_spec_dict: 'strategy_cls' must be a dotted class path "
                    f"string; got {type(raw_strategy_cls).__name__}."
                )
            imported = _import_cls(raw_strategy_cls)
            if not issubclass(imported, cls):
                raise ValueError(
                    f"from_spec_dict: {raw_strategy_cls!r} must resolve to a "
                    f"{cls.__name__} subclass."
                )
            if imported is not cls:
                return imported.from_spec_dict(
                    spec,
                    models=models,
                    hooks=hooks,
                    training_fn=training_fn,
                    **_set_rebuild_overrides(
                        imported, {"validation_config": validation_config}
                    ),
                )
        model_input = strategy_spec._models_from_spec_and_overrides(
            spec.get("model_specs", {}),
            models,
            single_model_input=strategy_spec._single_model_input_from_spec(
                spec.get("single_model_input")
            ),
        )
        return cls(
            models=model_input,
            optimizer_configs=strategy_spec._optimizer_configs_from_spec(
                spec["optimizer_configs"]
            ),
            num_epochs=spec.get("num_epochs"),
            num_steps=spec.get("num_steps"),
            epoch_step_modifier=spec.get("epoch_step_modifier", 1.0),
            hooks=list(hooks) if hooks is not None else [],
            training_fn=strategy_spec._training_fn_from_spec(spec, training_fn),
            loss_fn=strategy_spec._loss_fn_from_spec(spec["loss_fn_spec"]),
            devices=strategy_spec._devices_from_spec(spec["devices"]),
            validation_config=validation_config,
            teacher_signals=spec.get("teacher_signals"),
            label_missing=spec.get("label_missing", True),
            label_dtype=(
                None
                if spec.get("label_dtype") is None
                else _dtype_deserialize(spec["label_dtype"])
            ),
        )
