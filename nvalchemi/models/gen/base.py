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
"""Model-side generative API: config and mixin.

This is the *non-energy* counterpart to
:class:`~nvalchemi.models.base.BaseModelMixin`. Where ``BaseModelMixin`` is
MLIP/energy-oriented (energy, forces, stress, neighbor lists, autograd,
pipeline composition), a generative flow model predicts a velocity field over a
flow state and shares none of that energy machinery. The two are therefore kept
**separate** (see :class:`GenerativeModelMixin`): a generative model inherits
this mixin, not ``BaseModelMixin``.

Two pieces live here:

* :class:`GenerativeModelConfig` — a pydantic config schema describing a
  generative model's intents, modalities, and variable-atom support. It is set
  as ``self.model_config`` in a wrapper's ``__init__`, mirroring the
  ``BaseModelMixin`` pattern (the config lives with the model, not on the
  :class:`~nvalchemi.gen.generator.AtomGenerator`).
* :class:`GenerativeModelMixin` — the model mixin providing
  ``forward`` (raw output), ``adapt_output`` (raw -> :class:`ModelOutputs`),
  and ``condition`` (required), plus optional ``to_batch`` and
  ``prior_template`` hooks. It owns no scheduler, sampler, or guidance.

The ``Modality`` and ``GenerativeIntent`` enums are defined in
:mod:`nvalchemi.gen.enums` and re-exported here for convenience.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from collections.abc import Mapping
from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.enums import GenerativeIntent, Modality
from nvalchemi.gen.generator import default_condition

__all__ = [
    "ArtifactT",
    "GenerativeIntent",
    "GenerativeModelConfig",
    "GenerativeModelMixin",
    "Modality",
]


#: Type alias for the output artifact field of :class:`GenerativeModelConfig`.
#:
#: The artifact a generative model produces is identified by its
#: :class:`Modality` (e.g. :attr:`Modality.CRYSTAL`).
ArtifactT: TypeAlias = Modality

#: Intents that *produce* an output artifact. Used to split
#: :attr:`GenerativeModelConfig.intent_modality_map` into input-facing and
#: output-facing modalities. The remaining intents (``condition``, ``complete``,
#: ``transform``, ``connect``) are input-facing.
_OUTPUT_INTENTS: frozenset[GenerativeIntent] = frozenset(
    {
        GenerativeIntent.CREATE,
        GenerativeIntent.SAMPLE,
        GenerativeIntent.PROPOSE,
        GenerativeIntent.DECODE,
    }
)


class GenerativeModelConfig(BaseModel):
    """Pydantic config schema for a generative (non-energy) model.

    A :class:`GenerativeModelConfig` is the contract between a generative model
    wrapper and the rest of nvalchemi. Every
    :class:`GenerativeModelMixin` subclass must set a ``self.model_config``
    instance in its ``__init__`` (there is deliberately no class-level default,
    so each wrapper owns its own config object — mirroring
    :class:`~nvalchemi.models.base.BaseModelMixin`).

    Attributes
    ----------
    intents
        The operational roles the model can play.
    supports_variable_atoms
        Whether the model accepts systems with varying atom counts.
    output_artifact
        The primary output artifact modality (e.g. :attr:`Modality.CRYSTAL`).
    intent_modality_map
        Mapping from each supported intent to the modalities that intent
        operates on. Every intent in :attr:`intents` must have an entry here.
    consumes_fields
        Batch fields the model's conditioning reads (empty means
        unconditional). Declared here so a
        :class:`~nvalchemi.gen.pipeline.GenerationPipeline` can validate stage
        links at construction; a generator writes *something* by definition,
        so declarations are required, not optional.
    produces_fields
        Batch fields the model's generated output carries (written or
        forwarded). Distinct namespace from
        :attr:`active_prediction_outputs`, which keys ``ModelOutputs`` (e.g.
        ``"flow"``), not batch fields.
    active_prediction_outputs
        Output keys the model predicts (e.g. ``{"flow"}``). ``None`` defaults
        to ``{"flow"}`` at use time.

    Examples
    --------
    >>> from nvalchemi.models.gen.base import GenerativeModelConfig
    >>> from nvalchemi.gen.enums import GenerativeIntent, Modality
    >>> cfg = GenerativeModelConfig(
    ...     intents={GenerativeIntent.CREATE, GenerativeIntent.SAMPLE},
    ...     supports_variable_atoms=True,
    ...     output_artifact=Modality.CRYSTAL,
    ...     intent_modality_map={
    ...         GenerativeIntent.CREATE: frozenset({Modality.CRYSTAL}),
    ...         GenerativeIntent.SAMPLE: frozenset({Modality.CRYSTAL}),
    ...     },
    ...     consumes_fields=frozenset({"positions", "atomic_numbers"}),
    ...     produces_fields=frozenset({"positions", "atomic_numbers", "cell"}),
    ... )
    >>> Modality.CRYSTAL in cfg.output_modalities
    True

    Notes
    -----
    ``extra="forbid"``: unknown constructor keywords raise
    :class:`pydantic.ValidationError`, matching the
    :class:`~nvalchemi.models.base.ModelConfig` pattern.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    intents: Annotated[
        set[GenerativeIntent],
        Field(description="Operational roles the model can play."),
    ]
    supports_variable_atoms: Annotated[
        bool,
        Field(description="Whether the model accepts variable atom counts."),
    ]
    output_artifact: Annotated[
        ArtifactT,
        Field(description="Primary output artifact modality."),
    ]
    intent_modality_map: Annotated[
        dict[GenerativeIntent, frozenset[Modality]],
        Field(
            description=(
                "Mapping from each supported intent to the modalities it "
                "operates on. Every intent in `intents` must have an entry."
            )
        ),
    ]
    consumes_fields: Annotated[
        frozenset[str],
        Field(
            description=(
                "Batch fields the model's conditioning reads (empty = "
                "unconditional). Required: feeds GenerationPipeline link "
                "validation."
            )
        ),
    ]
    produces_fields: Annotated[
        frozenset[str],
        Field(
            description=(
                "Batch fields the model's generated output carries "
                "(written or forwarded). Required: feeds GenerationPipeline "
                "link validation."
            )
        ),
    ]
    active_prediction_outputs: Annotated[
        set[str] | None,
        Field(
            default=None,
            description=(
                "Output keys the model predicts (e.g. {'flow'}). None "
                "defaults to {'flow'} at use time."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def assert_intents_in_map(self) -> GenerativeModelConfig:
        """Assert every intent in :attr:`intents` has a map entry.

        Returns
        -------
        GenerativeModelConfig
            The validated config.

        Raises
        ------
        ValueError
            If any intent in :attr:`intents` is missing from
            :attr:`intent_modality_map`.
        """
        missing = self.intents - set(self.intent_modality_map)
        if missing:
            missing_str = ", ".join(sorted(i.value for i in missing))
            raise ValueError(f"intents missing from intent_modality_map: {missing_str}")
        return self

    @property
    def input_modalities(self) -> frozenset[Modality]:
        """Modalities consumed by the model's input-facing intents.

        Returns
        -------
        frozenset of Modality
            Union of modalities over intents that are NOT output-producing
            (``condition``, ``complete``, ``transform``, ``connect``).
        """
        input_intents = self.intents - _OUTPUT_INTENTS
        return (
            frozenset().union(*(self.intent_modality_map[i] for i in input_intents))
            if input_intents
            else frozenset()
        )

    @property
    def output_modalities(self) -> frozenset[Modality]:
        """Modalities produced by the model's output-producing intents.

        Returns
        -------
        frozenset of Modality
            Union of modalities over output-producing intents
            (``create``, ``sample``, ``propose``, ``decode``), plus
            :attr:`output_artifact`.
        """
        output_intents = self.intents & _OUTPUT_INTENTS
        produced = (
            frozenset().union(*(self.intent_modality_map[i] for i in output_intents))
            if output_intents
            else frozenset()
        )
        return produced | {self.output_artifact}


class GenerativeModelMixin(abc.ABC):
    """Mixin class for models designed for generative workflows.

    This is the counterpart to :class:`~nvalchemi.models.base.BaseModelMixin`.
    It mirrors the two-step output pattern — ``forward`` returns raw output,
    :meth:`adapt_output` structures it into :class:`ModelOutputs` — but for a
    velocity/flow model rather than an energy/MLIP model.

    Concrete implementations must provide:

    - ``model_config`` attribute — a :class:`GenerativeModelConfig` set in
      ``__init__`` (enforced by :meth:`__init_subclass__`).
    - :meth:`forward` — raw model output for one forward call.

    The mixin provides defaults for:

    - :meth:`adapt_output` — maps raw output to :class:`ModelOutputs`.
    - :meth:`condition` — passes an already-built :class:`Batch` (or
      :class:`AtomicData`) through, replicated by ``num_samples``;
      delegates to the module-level
      :func:`~nvalchemi.gen.default_condition`, the same default the
      :class:`~nvalchemi.gen.generator.AtomGenerator` falls back to.

    Optional hooks (define on subclasses as needed; the base does not
    provide them, so the :class:`~nvalchemi.gen.generator.AtomGenerator`
    raises :class:`TypeError` when they are absent):

    - ``to_batch(sample, cond_batch) -> Batch`` — map a sample
      :class:`~tensordict.TensorDict` (e.g. the denoised endpoint under
      ``"x1"``) into a :class:`Batch`. ``cond_batch``
      is ``None`` for unconditional generation. Built by the
      :class:`~nvalchemi.gen.generator.AtomGenerator` via
      ``model.condition`` and passed here for materialization context.
    - ``generate(*, num_samples=1, rng=None, cond=None, **kwargs)
      -> TensorDict`` — a model-supplied
      :class:`~nvalchemi.gen.generator.GeneratingFunction` (without the
      leading ``model`` argument, since it is a method). When present, the
      :class:`~nvalchemi.gen.generator.AtomGenerator` uses it as the fallback
      generation source when no ``generator_func`` is supplied.

    Note that the intention behind classes that include this mixin is
    to not fully own the generation process: instead that's offloaded
    to the :class:`~nvalchemi.gen.generator.AtomGenerator`, which in
    turn allows you to define a workflow that *uses* this model for
    generation.
    """

    model_config: GenerativeModelConfig

    # model_config must be set as an instance attribute in each subclass
    # __init__: self.model_config = GenerativeModelConfig(...). There is
    # intentionally NO class-level default (see BaseModelMixin for the same
    # rationale). __init_subclass__ wraps __init__ to enforce this at
    # construction time.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Hook applied to every concrete subclass at class-creation time.

        Wraps the subclass ``__init__`` so that after construction,
        ``self.model_config`` is verified to exist and be a
        :class:`GenerativeModelConfig`. This catches the common mistake of
        forgetting to set ``model_config`` with a clear error instead of a late
        ``AttributeError`` deep in a forward pass.

        Parameters
        ----------
        **kwargs
            Forwarded to the superclass hook.
        """
        super().__init_subclass__(**kwargs)
        # Inject extra_repr onto the concrete class so it takes precedence over
        # ``nn.Module.extra_repr`` (which precedes this mixin in the MRO when a
        # wrapper is declared as ``class W(nn.Module, GenerativeModelMixin)``).
        if "extra_repr" not in cls.__dict__:
            cls.extra_repr = GenerativeModelMixin._config_extra_repr
        if "__init__" in cls.__dict__:
            import functools

            original_init = cls.__init__

            @functools.wraps(original_init)
            def _checked_init(self: Any, *args: Any, **kw: Any) -> None:
                original_init(self, *args, **kw)
                cfg = getattr(self, "model_config", None)
                if not isinstance(cfg, GenerativeModelConfig):
                    raise TypeError(
                        f"{type(self).__name__}.__init__() must set "
                        f"self.model_config = GenerativeModelConfig(...). "
                        f"See GenerativeModelMixin docstring for details."
                    )

            cls.__init__ = _checked_init  # type: ignore[attr-defined]

    def adapt_output(self, raw: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Map raw model output to :class:`ModelOutputs`.

        The default builds an :class:`OrderedDict` keyed by
        :attr:`GenerativeModelConfig.active_prediction_outputs` (defaulting to
        ``{"flow"}``). A dict-like ``raw`` fills matching keys; a single
        :class:`~torch.Tensor` is placed under the ``"flow"`` key (or the
        single configured key).

        Parameters
        ----------
        raw
            Raw output from :meth:`forward`.
        data
            Source structure data (for context/metadata).

        Returns
        -------
        ModelOutputs
            ``OrderedDict`` with the active prediction outputs.
        """
        keys = self.model_config.active_prediction_outputs or {"flow"}
        output: ModelOutputs = OrderedDict((k, None) for k in sorted(keys))
        if isinstance(raw, Mapping):
            for key in output:
                if key in raw:
                    output[key] = raw[key]
        elif isinstance(raw, Tensor):
            key = "flow" if "flow" in output else next(iter(output))
            output[key] = raw
        return output

    def condition(
        self, cond: AtomicData | Batch | None, num_samples: int = 1
    ) -> Batch | None:
        """Ingest a conditioning spec into a conditioning :class:`Batch`.

        Delegates to the module-level
        :func:`~nvalchemi.gen.default_condition` — the same default
        the :class:`~nvalchemi.gen.generator.AtomGenerator` falls back to when a
        model defines no ``condition`` of its own. An already-built
        :class:`Batch` (or :class:`AtomicData`) passes through with each
        conditioning graph replicated ``num_samples`` times; ``None`` passes
        through as ``None`` for unconditional generation; any other tensor
        container passes through unchanged (replication semantics for
        non-batch containers belong to the model or generating function).

        Parameters
        ----------
        cond
            An already-built :class:`Batch` or :class:`AtomicData`,
            or ``None`` for unconditional generation.
        num_samples
            Number of independent draws per conditioning graph.

        Returns
        -------
        Batch | None
            The conditioning batch, tiled so each conditioning graph appears
            ``num_samples`` times, or ``None`` for unconditional generation.
        """
        return default_condition(cond, num_samples)

    @staticmethod
    def _config_extra_repr(self: Any) -> str:
        """Format the generative config for ``nn.Module.__repr__``.

        Parameters
        ----------
        self
            The wrapper instance (injected onto concrete subclasses).

        Returns
        -------
        str
            A short summary of intents and output artifact.
        """
        cfg = getattr(self, "model_config", None)
        if not isinstance(cfg, GenerativeModelConfig):
            return "model_config=<not set>"
        intents = ", ".join(sorted(i.value for i in cfg.intents))
        return f"intents={{{intents}}}, output_artifact={cfg.output_artifact.value}"
