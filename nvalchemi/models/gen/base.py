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
pipeline composition), a generative model owns a family-specific sampling
procedure (a diffusion sampler's loop, a GAN decoder pass, a packing or
population loop) and shares none of that energy machinery. The two are
kept separate: a generative model builds on this mixin, not
``BaseModelMixin``.

Two pieces live here:

* :class:`GenerativeModelConfig` — a pydantic config schema describing a
  generative model's capability surface: variable-atom support, batch-field
  declarations, and prediction-output keys. It is set as ``self.model_config``
  in a wrapper's ``__init__``, mirroring the ``BaseModelMixin`` pattern (the
  config lives with the model, not on the
  :class:`~nvalchemi.gen.generator.AtomisticGenerator`).
* :class:`GenerativeModelMixin` — the model mixin providing
  ``forward`` (raw output) and ``adapt_output`` (raw -> :class:`ModelOutputs`).
  It owns no conditioning, scheduler, sampler, or guidance.
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch

__all__ = [
    "GenerativeModelConfig",
    "GenerativeModelMixin",
]


class GenerativeModelConfig(BaseModel):
    """Pydantic config schema for a generative (non-energy) model.

    A :class:`GenerativeModelConfig` is the contract between a generative model
    wrapper and the rest of nvalchemi. Every
    :class:`GenerativeModelMixin` subclass must set a ``self.model_config``
    instance in its ``__init__`` (no class-level default,
    so each wrapper owns its own config object — mirroring
    :class:`~nvalchemi.models.base.BaseModelMixin`).

    The base schema holds only the four capability fields below. Models with
    model-specific runtime fields should subclass it (e.g. ``class
    ClariConfig(GenerativeModelConfig): temperature: float = 1.0``) — the
    mixin's ``isinstance`` enforcement accepts subclasses.

    Attributes
    ----------
    supports_variable_atoms
        Whether the model accepts systems with varying atom counts.
    required_inputs
        Batch fields the model's conditioning reads (empty means
        unconditional). Declared here so a
        :class:`~nvalchemi.gen.pipeline.GenerationPipeline` can validate stage
        links at construction; a generator writes *something* by definition,
        so declarations are required, not optional.
    outputs
        Batch fields the model's generated output carries (written or
        forwarded). Distinct namespace from
        :attr:`prediction_outputs`, which keys ``ModelOutputs`` (e.g.
        ``"flow"``), not batch fields.
    prediction_outputs
        Output keys the model predicts (e.g. ``{"flow"}``). ``None`` defaults
        to ``{"flow"}`` at use time.

    Examples
    --------
    >>> from nvalchemi.models.gen.base import GenerativeModelConfig
    >>> cfg = GenerativeModelConfig(
    ...     supports_variable_atoms=True,
    ...     required_inputs=frozenset({"positions", "atomic_numbers"}),
    ...     outputs=frozenset({"positions", "atomic_numbers", "cell"}),
    ... )
    >>> cfg.prediction_outputs is None
    True

    Notes
    -----
    ``extra="forbid"``: unknown constructor keywords raise
    :class:`pydantic.ValidationError`, matching the
    :class:`~nvalchemi.models.base.ModelConfig` pattern.
    """

    model_config = ConfigDict(extra="forbid")

    supports_variable_atoms: Annotated[
        bool,
        Field(description="Whether the model accepts variable atom counts."),
    ]
    required_inputs: Annotated[
        frozenset[str],
        Field(
            description=(
                "Batch fields the model's conditioning reads (empty = "
                "unconditional). Required: feeds GenerationPipeline link "
                "validation."
            )
        ),
    ]
    outputs: Annotated[
        frozenset[str],
        Field(
            description=(
                "Batch fields the model's generated output carries "
                "(written or forwarded). Required: feeds GenerationPipeline "
                "link validation."
            )
        ),
    ]
    prediction_outputs: Annotated[
        set[str] | None,
        Field(
            default=None,
            description=(
                "Output keys the model predicts (e.g. {'flow'}). None "
                "defaults to {'flow'} at use time."
            ),
        ),
    ] = None


class GenerativeModelMixin(abc.ABC):
    """Mixin class for models designed for generative workflows.

    This is the counterpart to :class:`~nvalchemi.models.base.BaseModelMixin`.
    It mirrors the two-step output pattern — ``forward`` returns raw output,
    :meth:`adapt_output` structures it into :class:`ModelOutputs` — with none
    of the energy machinery (no neighbor lists, no autograd plumbing).

    Concrete implementations must provide:

    - ``model_config`` attribute — a :class:`GenerativeModelConfig` (or a
      subclass of it) set in ``__init__`` (enforced by
      :meth:`__init_subclass__`).
    - :meth:`forward` — raw model output for one forward call.

    The mixin provides defaults for:

    - :meth:`adapt_output`: maps raw output to :class:`ModelOutputs`, keyed
      by :attr:`GenerativeModelConfig.prediction_outputs` (defaulting to
      ``{"flow"}``).

    ``forward`` / :meth:`adapt_output` are the model-side contract: the
    :class:`~nvalchemi.gen.generator.AtomisticGenerator` never reads them;
    the generating function calls them. Materialization and conditioning
    belong to the procedure, so the mixin defines neither. A model may define
    a ``to_batch(sample, cond_batch=None) -> Batch`` helper for its
    generating function to call (the demos do); nothing in the framework
    looks for it.

    Classes using this mixin do not own the generation process: that lives
    with the :class:`~nvalchemi.gen.generator.GeneratingFunction` (driven by
    the :class:`~nvalchemi.gen.generator.AtomisticGenerator`); the model is what
    the function samples from.
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
        :attr:`GenerativeModelConfig.prediction_outputs` (defaulting to
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
        keys = self.model_config.prediction_outputs or {"flow"}
        output: ModelOutputs = OrderedDict((k, None) for k in sorted(keys))
        if isinstance(raw, Mapping):
            for key in output:
                if key in raw:
                    output[key] = raw[key]
        elif isinstance(raw, Tensor):
            key = "flow" if "flow" in output else next(iter(output))
            output[key] = raw
        return output

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
            A short summary of the declared batch-field contracts.
        """
        cfg = getattr(self, "model_config", None)
        if not isinstance(cfg, GenerativeModelConfig):
            return "model_config=<not set>"
        consumes = ", ".join(sorted(cfg.required_inputs))
        produces = ", ".join(sorted(cfg.outputs))
        return f"required_inputs={{{consumes}}}, outputs={{{produces}}}"
