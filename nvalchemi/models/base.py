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
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi._typing import AtomsLike, ModelOutputs
from nvalchemi.data import AtomicData, Batch

warnings.simplefilter("once", UserWarning)


class NeighborListFormat(str, Enum):
    """Storage format for neighbor data written to the batch.

    Attributes
    ----------
    COO : str
        Coordinate (sparse) format.  Internally ``neighbor_list`` is stored as
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
    MATRIX = "matrix"  # dense neighbor-matrix format


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
    """
    Configuration structure for a given model.

    All models that inherit from `BaseModelMixin` should have a `model_config`
    attribute that is an instance of this class, which can be used to
    change the behavior of the model.

    Attributes
    ----------
    compute_forces : bool, default True
        Set to enable or disable force computation.
    compute_stresses : bool, default False
        Set to enable or disable stress computation.
    compute_hessians : bool, default False
        Set to enable or disable Hessian computation.
    compute_dipoles : bool, default False
        Set to enable or disable dipole computation.
    gradient_keys : set[str], default set()
        Set of keys to enable gradients for in the `Batch` of `AtomicData` structure.
    """

    compute_forces: Annotated[
        bool,
        Field(description="Set to enable or disable force computation."),
    ] = True
    compute_stresses: Annotated[
        bool,
        Field(description="Set to enable or disable stress computation."),
    ] = False
    compute_hessians: Annotated[
        bool,
        Field(description="Set to enable or disable Hessian computation."),
    ] = False
    compute_dipoles: Annotated[
        bool,
        Field(description="Set to enable or disable dipole computation."),
    ] = False
    compute_charges: Annotated[
        bool,
        Field(description="Set to enable or disable charge computation."),
    ] = False
    compute_embeddings: Annotated[
        bool,
        Field(description="Set to enable or disable embedding computation."),
    ] = False
    compute_energies: Annotated[
        bool,
        Field(description="Set to enable or disable energies computation."),
    ] = True
    gradient_keys: Annotated[
        set[str],
        Field(
            description="Set of keys to compute gradients for in the `Batch` of `AtomicData` structure.",
            default_factory=set,
        ),
    ]


class ModelCard(BaseModel):
    """
    Model card for a given model.

    This model card is a Pydantic model that contains information about the model's
    capabilities and requirements.

    A new model wrapper should return this data structure as the `model_card` property.
    """

    forces_via_autograd: Annotated[
        bool, Field(description="Whether the model predicts forces via autograd.")
    ]
    supports_node_embeddings: Annotated[
        bool, Field(description="Whether the model supports computing embeddings.")
    ] = False
    supports_edge_embeddings: Annotated[
        bool, Field(description="Whether the model supports computing edge embeddings.")
    ] = False
    supports_graph_embeddings: Annotated[
        bool,
        Field(description="Whether the model supports computing graph embeddings."),
    ] = False
    supports_energies: Annotated[
        bool, Field(description="Whether the model supports energies computation.")
    ] = True
    supports_forces: Annotated[
        bool, Field(description="Whether the model supports forces computation.")
    ] = False
    supports_stresses: Annotated[
        bool,
        Field(description="Whether the model supports stresses/virials computation."),
    ] = False
    supports_hessians: Annotated[
        bool,
        Field(
            description="Whether the model supports computing the Hessians of the energy."
        ),
    ] = False
    supports_pbc: Annotated[
        bool,
        Field(description="Whether the model supports periodic boundary conditions."),
    ] = False
    needs_pbc: Annotated[
        bool,
        Field(
            description="Whether the model needs periodic boundary conditions parameters as part of its input."
        ),
    ]
    needs_node_charges: Annotated[
        bool,
        Field(
            description="Whether the model needs partial atomic charges as part of its input."
        ),
    ] = False
    needs_system_charges: Annotated[
        bool,
        Field(
            description="Whether the model needs the total system charge as part of its input."
        ),
    ] = False
    supports_dipoles: Annotated[
        bool,
        Field(
            description="Whether the model explicitly supports computing the dipole moments."
        ),
    ] = False
    supports_non_batch: Annotated[
        bool, Field(description="Whether the model supports non-batch input.")
    ] = False
    neighbor_config: Annotated[
        NeighborConfig | None,
        Field(
            description=(
                "Neighbor list requirements for this model.  ``None`` means the "
                "model does not use a neighbor list.  When set, a "
                "``NeighborListHook`` should be registered with the dynamics "
                "engine to supply the required neighbor data before each "
                "``compute()`` call."
            )
        ),
    ] = None

    includes_dispersion: Annotated[
        bool,
        Field(
            description="Whether the model already incorporates dispersion (e.g. D3) in its energy."
        ),
    ] = False
    includes_long_range_electrostatics: Annotated[
        bool,
        Field(
            description="Whether the model already incorporates long-range electrostatics in its energy."
        ),
    ] = False

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def needs_neighborlist(self) -> bool:
        """Convenience accessor: ``True`` when the model requires a neighbor list."""
        return self.neighbor_config is not None


# To support a new output key, add an entry here AND the corresponding
# ``compute_<suffix>`` / ``supports_<suffix>`` fields to ModelConfig / ModelCard.
_OUTPUT_KEY_TO_CONFIG_SUFFIX: dict[str, str] = {
    "forces": "forces",
    "stress": "stresses",
    "hessians": "hessians",
    "dipole": "dipoles",
    "charges": "charges",
    "energy": "energies",
}


class BaseModelMixin(abc.ABC):
    """
    Abstract MixIn class providing a homogenized interface for wrapper models
    from external machine learning interatomic potential projects.

    This mixin defines the core interface that all external model wrappers
    should implement to ensure consistency across different model types.

    The mixin provides abstract methods for:

    - Computing embeddings at different graph levels
    - Predicting energies and forces
    - Defining expected output shapes
    - Adapting inputs and outputs between framework and external model formats

    A concrete implementation of this mixin should utilize the following
    functions to implement predictions:

    - ``_adapt_input``, which adapts the input batch to the model's expected format
    - ``_adapt_output``, which adapts the model's output to the framework's expected format
    - ``validate_batch``, which ensures that the input batch is compatible with the model
    - ``compute_embeddings``, which computes embeddings at different graph levels

    The mixin also defines several properties that must be implemented to specify
    model capabilities; when adding a new model, these properties must be implemented.

    - ``model_card``: Pydantic model that contains information about the model's
      capabilities and requirements
    - ``embedding_shapes``: Expected shapes of node, edge, and graph embeddings

    The workflow for using this mixin is:

    1. Implement all required properties to specify model capabilities
    2. Implement ``_adapt_input`` to convert framework data to model format
    3. Implement ``parse_output`` to convert model output to framework format
    4. Implement prediction methods based on supported capabilities
    5. Use ``validate_batch`` to ensure input compatibility
    6. Call ``parse_output`` to write model outputs to the ``Batch`` data structure

    Raises
    ------
    NotImplementedError
        If any required abstract methods or properties are not implemented
    ValueError
        If input validation fails in `validate_batch`
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
        """
        Adapt framework batch data to external model input format.

        The base implementation will check the `model_config` to determine
        what input keys need gradients enabled, depending on what is required.

        A subclass implementation should call this, in addition to doing
        whatever is needed to extract `Batch` inputs into arguments for
        the underlying model `forward` call.

        The method should return a dictionary of input arguments that will be
        unpacked in the actual `forward` and/or `__call__` methods.

        Parameters
        ----------
        batch : Batch
            Framework batch data

        Returns
        -------
        dict[str, Any]
            Input in the format expected by the external model
            (could be dict, custom object, etc.)
        """
        # Build effective gradient keys without mutating the shared ModelConfig.
        effective_grad_keys = set(self.model_config.gradient_keys)
        if self.model_config.compute_forces or self.model_config.compute_stresses:
            effective_grad_keys.add("positions")
        # enable gradients on tensors that need them
        for key in effective_grad_keys:
            if getattr(data, key, None) is None:
                raise KeyError(
                    f"'{key}' required for gradient computation, but not found in batch."
                )
            value = getattr(data, key, None)
            if value is not None and isinstance(value, torch.Tensor):
                value.requires_grad_(True)
            elif not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"'{key}' set to require gradients, but is {type(value)} (not a tensor)."
                )
        # prefill with input data requirement expectations
        input_dict = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value
        return input_dict

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """
        Adapt external model output to the framework's standard output format (ModelOutputs).

        This implementation returns a ModelOutputs (OrderedDict) with keys from output_data(),
        initialized to None, and populates with values from model_output if present and if we
        can match the key names generically. It is unlikely that this will perfectly match
        key names for all models, so it is imperative to manually check and override this
        implementation in a subclass.

        Parameters
        ----------
        model_output : Any
            Raw output from the external model
        data : AtomicData | Batch
            Original input data (may be needed for context/metadata)

        Returns
        -------
        ModelOutputs
            OrderedDict with expected output keys and their values (or None if not present).
        """
        output = OrderedDict((key, None) for key in self.output_data())
        if isinstance(model_output, dict):
            for key in output:
                value = model_output.get(key, None)
                if value is not None:
                    # insert key-specific logic here
                    match key:
                        case "energy":
                            if value.ndim == 1:
                                # energy needs to be [N, 1] shape
                                value.unsqueeze_(-1)
                        case _:
                            pass
                    output[key] = value
        return output

    def add_output_head(self, prefix: str) -> None:
        """
        Add an output head to the model.

        This method should create an multilayer perceptron block for
        mapping input embeddings to a desired output shape. The logic
        for this should differentiate based on invariant/equivariant
        models - specifically those that use `e3nn` layers.

        The method should then save the output head to a `output_heads`
        `ModuleDict` attribute.

        Parameters
        ----------
        prefix : str
            Prefix for the output head
        """
        raise NotImplementedError

    def input_data(self) -> set[str]:
        """
        Returns a set of keys that are expected to be in the input data.

        This method provides the base logic that is generally common across
        all models, but can be overridden by subclasses to add more expected
        keys.

        Returns
        -------
        set[str]
            Set of keys that are expected to be in the input data.
        """
        expected_keys = {"positions", "atomic_numbers"}
        card = self.model_card
        if card.needs_pbc:
            expected_keys.add("pbc")
        nb = card.neighbor_config
        if nb is not None:
            if nb.format == NeighborListFormat.COO:
                expected_keys.add("neighbor_list")
            elif nb.format == NeighborListFormat.MATRIX:
                expected_keys.add("neighbor_matrix")
                expected_keys.add("num_neighbors")
        if card.needs_node_charges:
            expected_keys.add("charges")
        if card.needs_system_charges:
            expected_keys.add("charge")
        return expected_keys

    @staticmethod
    def _verify_request(
        model_config: ModelConfig,
        model_card: ModelCard,
        key: str,
    ) -> bool:
        """
        Verify that a requested computation is supported by the model.

        This method checks if a specific computation (forces, stresses, dipoles, hessians, or charges)
        is both requested in the model configuration and supported by the model card.
        If the computation is requested but not supported, it logs a warning.

        Parameters
        ----------
        model_config : ModelConfig
            The model configuration containing computation settings.
        model_card : ModelCard
            The model card containing capability information.
        key : str
            The type of computation to verify.

        Returns
        -------
        bool
            True if the computation is both requested and supported by the model, False otherwise.
        """

        is_requested = getattr(model_config, f"compute_{key}")
        is_supported = getattr(model_card, f"supports_{key}")
        if is_requested and not is_supported:
            warnings.warn(
                f"Model does not support {key}, but compute_{key} is set to True.",
                UserWarning,
            )
        return is_requested and is_supported

    def output_data(self) -> set[str]:
        """
        Returns a set of keys that are expected to be computed by the model
        and written to the `AtomicData` or `Batch` data structure.

        This method provides the base logic that is generally common across
        all models, but can be overridden by subclasses to add more expected
        keys.

        Returns
        -------
        set[str]
            Set of keys that are expected to be computed by the model
            and written to the `AtomicData` or `Batch` data structure.
        """
        expected_keys = set()
        for output_key, config_suffix in _OUTPUT_KEY_TO_CONFIG_SUFFIX.items():
            if getattr(self.model_config, f"compute_{config_suffix}", False):
                if self._verify_request(
                    self.model_config, self.model_card, config_suffix
                ):
                    expected_keys.add(output_key)
        return expected_keys

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

    def __add__(self, other: "BaseModelMixin") -> "ComposableModelWrapper":
        """Compose two models additively via the ``+`` operator.

        Returns an :class:`ComposableModelWrapper` that sums energies, forces,
        and stresses from both models.

        Parameters
        ----------
        other : BaseModelMixin
            Another model to add.
        """
        from nvalchemi.models.composable import ComposableModelWrapper  # noqa: PLC0415

        return ComposableModelWrapper(self, other)

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
