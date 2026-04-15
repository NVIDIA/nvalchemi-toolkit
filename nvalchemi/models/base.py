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
from __future__ import annotations

import abc
import warnings
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi._typing import AtomsLike, ModelOutputs
from nvalchemi.data import AtomicData, Batch

warnings.simplefilter("once", UserWarning)


class NeighborListFormat(str, Enum):
    """Storage format for neighbor data written to the batch.

    Attributes
    ----------
    COO : str
        Coordinate (sparse) format.  Internally ``edge_index`` is stored as
        ``[E, 2]`` (each row is a ``[source, target]`` pair).  Model boundary
        adapters (e.g. ``MACEWrapper.adapt_input``) transpose to the
        conventional ``[2, E]`` layout expected by most GNN-based MLIPs.
    MATRIX : str
        Dense neighbor-matrix format.  Neighbors are stored as a
        ``neighbor_matrix`` tensor of shape ``[N, max_neighbors]`` (global
        atom indices) together with a ``num_neighbors`` tensor of shape
        ``[N]``.  Used by Warp interaction kernels (e.g. Lennard-Jones) that
        benefit from fixed-width rows.
    """

    COO = "coo"  # internal (E, 2); model boundary adapters transpose to (2, E)
    MATRIX = "matrix"


class NeighborConfig(BaseModel):
    """Configuration for on-the-fly neighbor list construction.

    An instance of this class attached to a :class:`ModelConfig` signals that
    the model requires a neighbor list and describes the format and parameters
    it expects.  At runtime a :class:`~nvalchemi.hooks.NeighborListHook`
    reads this config to compute and cache the appropriate neighbor data.

    Attributes
    ----------
    cutoff : float
        Interaction cutoff radius in the same length units as positions.
    format : NeighborListFormat
        Whether to build a dense neighbor matrix (``MATRIX``) or a sparse
        edge-index list (``COO``).  Defaults to ``COO``.
    half_list : bool
        If ``True``, each pair ``(i, j)`` with ``i < j`` appears only once.
        Newton's third law is applied inside the interaction kernel to recover
        forces on both atoms.  Defaults to ``False``.
    skin : float
        Verlet skin distance.  The neighbor list is only rebuilt when any atom
        has moved more than ``skin / 2`` since the last build.  Set to ``0.0``
        (default) to rebuild every step.
    """

    cutoff: float
    format: NeighborListFormat = NeighborListFormat.COO
    half_list: bool = False
    skin: float = 0.0


class ModelConfig(BaseModel):
    """Unified model configuration combining capability declaration and
    runtime control.

    A ``ModelConfig`` has two kinds of fields:

    - **Capability fields** (frozen at construction) describe what the
      model checkpoint can do.  These use ``frozenset`` to signal
      immutability.  They are set once by the wrapper's ``__init__`` and
      should not be changed at runtime.
    - **Runtime fields** (mutable) control what the model should compute
      on each forward pass.  These can be changed freely by the user.

    ``outputs`` and ``required_inputs`` use free-form strings so new
    properties can be added without modifying this class.  Well-known
    output keys: ``energy``, ``forces``, ``stresses``, ``hessians``,
    ``dipoles``, ``charges``, ``embeddings``.

    Attributes
    ----------
    outputs : frozenset[str]
        All properties the model can produce (frozen).
    autograd_outputs : frozenset[str]
        Subset of ``outputs`` computed via autograd (frozen).
    autograd_inputs : frozenset[str]
        Input keys needing ``requires_grad_(True)`` for autograd (frozen).
    required_inputs : frozenset[str]
        Extra inputs beyond ``{positions, atomic_numbers}`` that the
        model requires (frozen).
    optional_inputs : frozenset[str]
        Extra inputs the model can optionally use if present (frozen).
    supports_pbc : bool
        Whether the model supports periodic boundary conditions (frozen).
    needs_pbc : bool
        Whether the model requires PBC inputs (frozen).
    neighbor_config : NeighborConfig | None
        Neighbor list requirements (frozen).
    active_outputs : set[str]
        Properties to compute this run (mutable).  Defaults to
        ``outputs`` if not explicitly set.
    gradient_keys : set[str]
        Extra input keys to enable gradients for beyond those implied
        by ``autograd_inputs`` (mutable).
    """

    # ── Capability fields (frozen at construction) ──────────────────────

    outputs: Annotated[
        frozenset[str],
        Field(
            default_factory=lambda: frozenset({"energy"}),
            description="All properties the model can produce.",
        ),
    ]
    autograd_outputs: Annotated[
        frozenset[str],
        Field(
            default_factory=frozenset,
            description="Subset of outputs computed via autograd.",
        ),
    ]
    autograd_inputs: Annotated[
        frozenset[str],
        Field(
            default_factory=lambda: frozenset({"positions"}),
            description="Input keys needing requires_grad for autograd outputs.",
        ),
    ]
    required_inputs: Annotated[
        frozenset[str],
        Field(
            default_factory=frozenset,
            description="Extra required inputs beyond {positions, atomic_numbers}.",
        ),
    ]
    optional_inputs: Annotated[
        frozenset[str],
        Field(
            default_factory=frozenset,
            description="Extra inputs used if present, silently skipped if absent.",
        ),
    ]
    supports_pbc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether the model supports periodic boundary conditions.",
        ),
    ]
    needs_pbc: Annotated[
        bool,
        Field(
            default=False,
            description="Whether the model requires PBC inputs.",
        ),
    ]
    neighbor_config: Annotated[
        NeighborConfig | None,
        Field(
            default=None,
            description="Neighbor list requirements. None means no neighbor list.",
        ),
    ]

    # ── Runtime fields (mutable) ────────────────────────────────────────

    active_outputs: Annotated[
        set[str] | None,
        Field(
            default=None,
            description=(
                "Properties to compute this run. "
                "None means use all outputs (the default)."
            ),
        ),
    ]
    gradient_keys: Annotated[
        set[str],
        Field(
            default_factory=set,
            description="Extra input keys to enable gradients for.",
        ),
    ]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _default_active_outputs(self) -> "ModelConfig":
        """Default active_outputs to outputs if not explicitly set."""
        if self.active_outputs is None:
            # Use object.__setattr__ because we're inside validation
            object.__setattr__(self, "active_outputs", set(self.outputs))
        return self

    @property
    def needs_neighborlist(self) -> bool:
        """Convenience accessor: ``True`` when the model requires a neighbor list."""
        return self.neighbor_config is not None


class BaseModelMixin(abc.ABC):
    """Abstract mixin providing a standardized interface for model wrappers.

    All external MLIP wrappers should inherit from this mixin (alongside
    ``nn.Module``) to ensure a consistent interface for dynamics engines,
    composition pipelines, and downstream tooling.

    Concrete implementations must provide:

    - ``model_config`` attribute — a :class:`ModelConfig` instance set in
      ``__init__``.
    - ``embedding_shapes`` property — expected shapes of computed
      embeddings.
    - ``compute_embeddings()`` — compute and attach embeddings to the
      input data structure.

    The mixin provides default implementations of:

    - ``input_data()`` — set of required input keys derived from the
      model config.
    - ``output_data()`` — set of active outputs intersected with
      supported outputs (warns on unsupported requests).
    - ``adapt_input()`` — enable gradients on required tensors and
      collect input dict.
    - ``adapt_output()`` — map raw model output to :class:`ModelOutputs`
      ordered dict.
    """

    # model_config must be set as an instance attribute in each subclass __init__:
    #   self.model_config = ModelConfig(outputs=..., ...)
    # There is intentionally NO class-level default to prevent all instances from
    # sharing a single ModelConfig object (which would cause mutations in one wrapper
    # to silently affect all others).  __init_subclass__ wraps __init__ to enforce
    # this at construction time — a missing model_config raises TypeError.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Hook applied to every concrete subclass at class-creation time.

        Performs two injections:

        1. **extra_repr** — ``nn.Module.__repr__`` calls
           ``self.extra_repr()`` but its default returns ``""``.  Since
           wrappers inherit ``nn.Module`` before ``BaseModelMixin``
           (required for PyTorch), ``Module.extra_repr`` wins in the MRO.
           This hook injects our version directly onto each concrete
           wrapper class so it takes precedence.
        2. **model_config post-init check** — wraps the subclass
           ``__init__`` so that after construction,
           ``self.model_config`` is verified to exist.  This catches
           the common mistake of forgetting to set ``model_config`` in
           ``__init__`` with a clear error instead of a late
           ``AttributeError`` deep in a forward pass.
        """
        super().__init_subclass__(**kwargs)
        if "extra_repr" not in cls.__dict__:
            cls.extra_repr = BaseModelMixin._config_extra_repr

        # Wrap __init__ to verify model_config is set after construction.
        if "__init__" in cls.__dict__:
            import functools

            original_init = cls.__init__

            @functools.wraps(original_init)
            def _checked_init(self: Any, *args: Any, **kw: Any) -> None:
                original_init(self, *args, **kw)
                if not hasattr(self, "model_config"):
                    raise TypeError(
                        f"{type(self).__name__}.__init__() must set "
                        f"self.model_config = ModelConfig(...).  "
                        f"See BaseModelMixin docstring for details."
                    )

            cls.__init__ = _checked_init  # type: ignore[attr-defined]

    @staticmethod
    def _config_extra_repr(self: Any) -> str:
        """Format the model config for ``nn.Module.__repr__``."""
        cfg = getattr(self, "model_config", None)
        if cfg is None:
            return "model_config=<not set>"

        parts = []

        outputs = sorted(cfg.outputs)
        active = sorted(cfg.active_outputs)
        parts.append(f"outputs={{{', '.join(outputs)}}}")
        if set(active) != set(outputs):
            parts.append(f"active_outputs={{{', '.join(active)}}}")
        if cfg.autograd_outputs:
            parts.append(
                f"autograd_outputs={{{', '.join(sorted(cfg.autograd_outputs))}}}"
            )
        if cfg.required_inputs:
            parts.append(
                f"required_inputs={{{', '.join(sorted(cfg.required_inputs))}}}"
            )
        if cfg.optional_inputs:
            parts.append(
                f"optional_inputs={{{', '.join(sorted(cfg.optional_inputs))}}}"
            )

        if cfg.supports_pbc or cfg.needs_pbc:
            pbc_parts = []
            if cfg.supports_pbc:
                pbc_parts.append("supports_pbc")
            if cfg.needs_pbc:
                pbc_parts.append("needs_pbc")
            parts.append(f"pbc=[{', '.join(pbc_parts)}]")

        if cfg.neighbor_config is not None:
            nc = cfg.neighbor_config
            nc_str = f"cutoff={nc.cutoff}, format={nc.format.value}"
            if nc.half_list:
                nc_str += ", half_list"
            parts.append(f"neighbors=({nc_str})")

        return "\n".join(parts)

    @property
    @abc.abstractmethod
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Retrieves the expected shapes of the node, edge, and graph embeddings."""
        ...

    @abc.abstractmethod
    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """
        Compute embeddings at different levels of a batch of atomic graphs.

        Parameters
        ----------
        data : AtomicData | Batch
            Input atomic data containing positions, atomic numbers, etc.

        Returns
        -------
        AtomicData | Batch
            Data structure with embeddings attached in-place.

        Raises
        ------
        NotImplementedError
            If the model does not support embeddings computation
        """
        ...

    def direct_derivative_keys(self) -> set[str]:
        """Return output keys this model computes analytically in ``forward()``.

        When this model participates in a pipeline autograd group, the
        pipeline strips ``"forces"`` and ``"stress"`` from sub-model
        ``active_outputs`` so it can compute them via autograd on the
        summed energy.  Keys returned by this method are **kept** in
        ``active_outputs`` — the pipeline collects them from the model
        output and sums them with the autograd-derived derivatives.

        Override this in models that produce analytical forces or stress
        alongside an energy that carries autograd information (e.g.
        Ewald/PME with ``hybrid_forces=True``).

        Returns
        -------
        set[str]
            Keys (e.g. ``{"forces", "stress"}``) that the model produces
            analytically and should be summed with autograd derivatives.
            Default: empty set (all derivatives come from autograd).
        """
        return set()

    def set_config(self, key: str, value: Any) -> None:
        """Set a mutable field on :attr:`model_config`.

        Convenience method equivalent to
        ``self.model_config.<key> = value`` with validation that the
        field exists and is mutable.

        Parameters
        ----------
        key : str
            Name of a mutable ``ModelConfig`` field (e.g.
            ``"active_outputs"``, ``"gradient_keys"``).
        value
            New value for the field.

        Raises
        ------
        AttributeError
            If *key* is not a field on :class:`ModelConfig`.
        """
        if not hasattr(self.model_config, key):
            raise AttributeError(
                f"ModelConfig has no field '{key}'.  "
                f"Available fields: {list(self.model_config.model_fields)}"
            )
        setattr(self.model_config, key, value)

    def adapt_input(
        self, data: AtomicData | Batch | AtomsLike, **kwargs: Any
    ) -> dict[str, Any]:
        """Adapt framework batch data to external model input format.

        The base implementation enables ``requires_grad`` on tensors that
        need gradients (determined by ``model_config.autograd_inputs`` and
        ``model_config.gradient_keys``), then collects all keys declared
        by :meth:`input_data` into a dict.

        Subclasses should call ``super().adapt_input(data)`` and then add
        or transform entries as needed for their underlying model.

        Parameters
        ----------
        data : AtomicData | Batch | AtomsLike
            Framework data structure.

        Returns
        -------
        dict[str, Any]
            Input in the format expected by the external model.
        """
        effective_grad_keys = set(self.model_config.gradient_keys)
        # Enable grad on autograd_inputs if any autograd output is active
        if self.model_config.autograd_outputs & self.model_config.active_outputs:
            effective_grad_keys |= self.model_config.autograd_inputs
        for key in effective_grad_keys:
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(
                    f"'{key}' required for gradient computation, but not found in batch."
                )
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"'{key}' set to require gradients, but is {type(value)} (not a tensor)."
                )
            value.requires_grad_(True)
        # Collect required input data
        input_dict = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value
        # Collect optional input data (include if present, skip if not)
        for key in self.model_config.optional_inputs:
            value = getattr(data, key, None)
            if value is not None:
                input_dict[key] = value
        return input_dict

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """Adapt external model output to :class:`ModelOutputs` format.

        Returns an OrderedDict keyed by :meth:`output_data` entries,
        populated from *model_output* where keys match.

        .. note::

            Returned tensors may still be attached to the autograd
            computation graph (e.g. energies from autograd-force models
            like MACE).  This is intentional — the model does not know
            whether the caller needs the graph (e.g. pipeline
            shared-autograd groups).  **Callers that do not need the
            graph are responsible for detaching.**

        Parameters
        ----------
        model_output : Any
            Raw output from the external model.
        data : AtomicData | Batch
            Original input data (may be needed for context/metadata).

        Returns
        -------
        ModelOutputs
            OrderedDict with expected output keys and their values
            (or ``None`` if not present).  Tensors may be graph-attached.
        """
        output = OrderedDict((key, None) for key in self.output_data())
        if isinstance(model_output, dict):
            for key in output:
                value = model_output.get(key)
                if value is not None:
                    if key == "energy" and value.ndim == 1:
                        value = value.unsqueeze(-1)
                    output[key] = value
        return output

    def add_output_head(self, prefix: str) -> None:
        """
        Add an output head to the model.

        Parameters
        ----------
        prefix : str
            Prefix for the output head
        """
        raise NotImplementedError

    def input_data(self) -> set[str]:
        """Return the set of **required** input keys.

        Base implementation derives keys from the model config:
        ``{positions, atomic_numbers}`` plus neighbor-list keys
        (from ``neighbor_config``), ``pbc`` (if ``needs_pbc``),
        and any extra keys in ``model_config.required_inputs``.

        Optional inputs (``model_config.optional_inputs``) are handled
        separately in :meth:`adapt_input` and are NOT included here.

        Returns
        -------
        set[str]
            Set of required input keys.
        """
        base = {"positions", "atomic_numbers"}
        nc = self.model_config.neighbor_config
        if nc is not None:
            if nc.format == NeighborListFormat.COO:
                base.add("neighbor_list")
            elif nc.format == NeighborListFormat.MATRIX:
                base |= {"neighbor_matrix", "num_neighbors"}
        if self.model_config.needs_pbc:
            base.add("pbc")
        return base | set(self.model_config.required_inputs)

    def output_data(self) -> set[str]:
        """Return the set of keys the model will compute this run.

        Intersects ``active_outputs`` with ``outputs``.
        Warns if any active keys are not supported by the model.

        Returns
        -------
        set[str]
            Set of output keys that are both active and supported.
        """
        active = self.model_config.active_outputs
        supported = self.model_config.outputs
        unsupported = active - supported
        if unsupported:
            warnings.warn(
                f"Requested {unsupported} but model only supports {supported}.",
                UserWarning,
                stacklevel=2,
            )
        return active & supported

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """
        Export the current model without the ``BaseModelMixin`` interface.
        """
        raise NotImplementedError

    def __add__(self, other: "BaseModelMixin") -> "PipelineModelWrapper":
        """Compose two models additively via the ``+`` operator.

        Returns a :class:`~nvalchemi.models.pipeline.PipelineModelWrapper`
        where each model occupies its own group with
        ``use_autograd=False``, so energy, forces, and stress from
        both models are summed element-wise.

        This is the simplest composition pattern — suitable when each model
        computes its own forces independently (analytically or via its own
        internal autograd).  For dependent pipelines where one model's
        output feeds into another's input, or for shared-autograd groups
        that differentiate the summed energy of multiple models, use the
        explicit :class:`~nvalchemi.models.pipeline.PipelineModelWrapper`
        constructor with :class:`~nvalchemi.models.pipeline.PipelineGroup`
        and :class:`~nvalchemi.models.pipeline.PipelineStep`.

        Parameters
        ----------
        other : BaseModelMixin
            Another model to compose with.

        Returns
        -------
        PipelineModelWrapper
            A pipeline that sums the outputs of both models.

        Examples
        --------
        >>> combined = lj_model + ewald_model
        >>> combined = mace_model + dftd3_model
        >>> combined = model_a + model_b + model_c  # chains naturally
        """
        from nvalchemi.models.pipeline import (  # noqa: PLC0415
            PipelineGroup,
            PipelineModelWrapper,
        )

        # If the left-hand side is already a pipeline of direct groups
        # (produced by a previous +), flatten into it instead of nesting.
        if isinstance(self, PipelineModelWrapper):
            new_groups = list(self.groups) + [PipelineGroup(steps=[other])]
            return PipelineModelWrapper(groups=new_groups)
        return PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[self]),
                PipelineGroup(steps=[other]),
            ]
        )

    def make_neighbor_hooks(self, max_neighbors: int | None = None) -> list:
        """Return a list of :class:`~nvalchemi.hooks.NeighborListHook` instances
        for this model's neighbor configuration.

        Returns an empty list if the model does not require a neighbor list.
        Defers the import to avoid circular imports.

        Parameters
        ----------
        max_neighbors : int | None, optional
            Maximum neighbors per atom for MATRIX format.  When ``None``
            (default), auto-estimated from the cutoff at first use.
        """
        from nvalchemi.dynamics.base import DynamicsStage  # noqa: PLC0415
        from nvalchemi.hooks import NeighborListHook  # noqa: PLC0415

        nc = self.model_config.neighbor_config
        if nc is None:
            return []
        return [
            NeighborListHook(
                nc,
                skin=nc.skin,
                max_neighbors=max_neighbors,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ]
