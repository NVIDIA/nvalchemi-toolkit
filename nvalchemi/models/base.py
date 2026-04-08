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
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

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

    An instance of this class attached to a :class:`ModelCard` signals that
    the model requires a neighbor list and describes the format and parameters
    it expects.  At runtime a :class:`~nvalchemi.dynamics.hooks.NeighborListHook`
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
    max_neighbors : int | None
        Maximum number of neighbors per atom.  Required when
        ``format=MATRIX``; ignored for ``COO``.
    algorithm : str
        Neighbor-finding algorithm.  ``"auto"`` (default) selects naïve
        O(N²) search for small systems and a cell-list algorithm for larger
        ones.  Explicit choices are ``"naive"`` and ``"cell_list"``.
    """

    cutoff: float
    format: NeighborListFormat = NeighborListFormat.COO
    half_list: bool = False
    skin: float = 0.0
    max_neighbors: int | None = None


class ModelConfig(BaseModel):
    """Runtime configuration for what a model should compute.

    ``compute`` is a set of string keys naming the properties the model
    should produce on each forward pass.  Well-known keys:
    ``energies``, ``forces``, ``stresses``, ``hessians``, ``dipoles``,
    ``charges``, ``embeddings``.

    ``gradient_keys`` names additional input keys that should have
    ``requires_grad_(True)`` set before the forward pass.

    Attributes
    ----------
    compute : set[str]
        Set of output property names to compute.
    gradient_keys : set[str]
        Extra input keys to enable gradients for (beyond those implied
        by ``model_card.autograd_inputs``).
    """

    compute: set[str] = Field(default_factory=lambda: {"energies", "forces"})
    gradient_keys: set[str] = Field(default_factory=set)


class ModelCard(BaseModel):
    """Immutable capability declaration for a model checkpoint.

    A ModelCard describes what a specific checkpoint can do — its
    outputs, inputs, autograd behavior, and structural requirements.
    It is determined by the architecture and checkpoint, not by the
    user's runtime choices (those go in ``ModelConfig``).

    - **Inference:** The wrapper constructs the card once at ``__init__``
      from the checkpoint's properties.  Users cannot modify it.
    - **Training:** The user creates the card upfront to declare what
      the model will be trained to produce.  Training scripts use the
      card to determine what losses to compute and how derivatives
      are taken.

    The card is frozen (immutable after construction) and serializable
    via Pydantic, so it can be saved alongside the checkpoint.

    ``outputs`` and ``inputs`` use free-form strings so new properties
    can be added without touching this class.  Well-known keys:
    energies, forces, stresses, hessians, dipoles, charges, embeddings.

    Attributes
    ----------
    outputs : set[str]
        What the model can produce.
    autograd_outputs : set[str]
        Which outputs are computed via autograd (subset of ``outputs``).
    autograd_inputs : set[str]
        Which inputs require gradients for autograd outputs.
        Default covers the common case (forces = -dE/dr).
    inputs : set[str]
        Extra inputs beyond {positions, atomic_numbers}.
        Neighbor-list keys are auto-derived from ``neighbor_config``.
    supports_pbc : bool
        Whether the model supports periodic boundary conditions.
    needs_pbc : bool
        Whether the model requires periodic boundary conditions.
    neighbor_config : NeighborConfig | None
        Neighbor list requirements.  ``None`` means the model does not
        use a neighbor list.
    """

    outputs: set[str] = Field(default_factory=lambda: {"energies"})
    autograd_outputs: set[str] = Field(default_factory=set)
    autograd_inputs: set[str] = Field(default_factory=lambda: {"positions"})
    inputs: set[str] = Field(default_factory=set)
    supports_pbc: bool = False
    needs_pbc: bool = False
    neighbor_config: NeighborConfig | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    - ``model_card`` property — immutable :class:`ModelCard` describing
      the checkpoint's capabilities.
    - ``embedding_shapes`` property — expected shapes of computed
      embeddings.
    - ``compute_embeddings()`` — compute and attach embeddings to the
      input data structure.

    The mixin provides default implementations of:

    - ``input_data()`` — set of required input keys derived from the
      model card.
    - ``output_data()`` — set of requested outputs intersected with
      supported outputs (warns on unsupported requests).
    - ``adapt_input()`` — enable gradients on required tensors and
      collect input dict.
    - ``adapt_output()`` — map raw model output to :class:`ModelOutputs`
      ordered dict.
    """

    # model_config must be set as an instance attribute in each subclass __init__:
    #   self.model_config = ModelConfig()
    # There is intentionally NO class-level default to prevent all instances from
    # sharing a single ModelConfig object (which would cause mutations in one wrapper
    # to silently affect all others).

    @property
    @abc.abstractmethod
    def model_card(self) -> ModelCard:
        """Retrieves the model card for the model.

        The model card is a Pydantic model that contains
        information about the model's capabilities and requirements.
        """
        ...

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

        This method should extract meaningful representations from the model
        at node (atomic), edge (bond), and/or graph/system (structure) levels.
        The concrete implementation should check if the model supports
        computing embeddings, as well as perform validation on `kwargs`
        to make sure they are valid for the model.

        The method should add graph, node, and/or edge embeddings to the `Batch`
        data structure in-place.

        Parameters
        ----------
        data : AtomicData | Batch
            Input atomic data containing positions, atomic numbers, etc.

        Returns
        -------
        AtomicData | Batch
            Standardized `AtomicData` or `Batch` data structure mutated in place.

        Raises
        ------
        NotImplementedError
            If the model does not support embeddings computation
        """
        ...

    def adapt_input(
        self, data: AtomicData | Batch | AtomsLike, **kwargs: Any
    ) -> dict[str, Any]:
        """Adapt framework batch data to external model input format.

        The base implementation enables ``requires_grad`` on tensors that
        need gradients (determined by ``model_card.autograd_inputs`` and
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
        # Enable grad on autograd_inputs if any autograd output is requested
        if self.model_card.autograd_outputs & self.model_config.compute:
            effective_grad_keys |= self.model_card.autograd_inputs
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
        # Collect input data
        input_dict = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
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
            :meth:`BaseDynamics.compute() <nvalchemi.dynamics.base.BaseDynamics.compute>`
            and :meth:`PipelineModelWrapper.forward()
            <nvalchemi.models.pipeline.PipelineModelWrapper.forward>`
            both detach automatically.

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
                    if key == "energies" and value.ndim == 1:
                        value = value.unsqueeze(-1)
                    output[key] = value
        return output

    def add_output_head(self, prefix: str) -> None:
        """
        Add an output head to the model.

        This method should create a multilayer perceptron block for
        mapping input embeddings to a desired output shape.

        Parameters
        ----------
        prefix : str
            Prefix for the output head
        """
        raise NotImplementedError

    def input_data(self) -> set[str]:
        """Return the set of keys expected in the input data.

        Base implementation derives keys from the model card:
        ``{positions, atomic_numbers}`` plus neighbor-list keys
        (from ``neighbor_config``), ``pbc`` (if ``needs_pbc``),
        and any extra keys in ``model_card.inputs``.

        Returns
        -------
        set[str]
            Set of required input keys.
        """
        base = {"positions", "atomic_numbers"}
        nc = self.model_card.neighbor_config
        if nc is not None:
            if nc.format == NeighborListFormat.COO:
                base.add("edge_index")
            elif nc.format == NeighborListFormat.MATRIX:
                base |= {"neighbor_matrix", "num_neighbors"}
        if self.model_card.needs_pbc:
            base.add("pbc")
        return base | self.model_card.inputs

    def output_data(self) -> set[str]:
        """Return the set of keys the model will compute this run.

        Intersects ``model_config.compute`` with ``model_card.outputs``.
        Warns if any requested keys are not supported by the model.

        Returns
        -------
        set[str]
            Set of output keys that are both requested and supported.
        """
        requested = self.model_config.compute
        supported = self.model_card.outputs
        unsupported = requested - supported
        if unsupported:
            warnings.warn(
                f"Requested {unsupported} but model only supports {supported}.",
                UserWarning,
                stacklevel=2,
            )
        return requested & supported

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """
        Export the current model without the ``BaseModelMixin`` interface.

        The idea behind this method is to allow users to use the trained
        model with the same interface as the corresponding 'upstream' version,
        so that they can re-use validation code that might have been written
        for the upstream case (e.g. ``ase.Calculator`` instances).

        Essentially, this method should recreate the equivalent base class
        (by checking MRO), then run ``torch.save`` and serialize the
        model either directly or as its ``state_dict``.
        """
        raise NotImplementedError

    def __add__(self, other: "BaseModelMixin") -> "PipelineModelWrapper":
        """Compose two models additively via the ``+`` operator.

        Returns a :class:`~nvalchemi.models.pipeline.PipelineModelWrapper`
        where each model occupies its own ``"direct"``-force group, so
        energies, forces, and stresses from both models are summed
        element-wise.

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

    def make_neighbor_hooks(self) -> list:
        """Return a list of :class:`~nvalchemi.dynamics.hooks.NeighborListHook` instances
        for this model's neighbor configuration.

        Returns an empty list if the model does not require a neighbor list.
        Defers the import to avoid circular imports.
        """
        from nvalchemi.dynamics.hooks import NeighborListHook  # noqa: PLC0415

        nc = self.model_card.neighbor_config
        if nc is None:
            return []
        return [NeighborListHook(nc)]
