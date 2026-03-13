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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from beartype import beartype
from torch import nn

from nvalchemi._typing import (
    AtomicNumbers,
    BatchIndices,
    ModelOutputs,
    NodePositions,
)
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelCard

# only the wrapper model is exported
__all__ = ["DemoModelWrapper"]


@dataclass(eq=False)
class DemoModel(nn.Module):
    """
    This model is a simple demo model that computes the energies
    and conservative forces given an atomic point cloud (not graph!).

    The model is not intended to be used in production, but rather to
    serve as a simple example of how to implement a model in the framework,
    as well as useful for testing and debugging.

    The following elements are design considerations:

    - ``@dataclass(eq=False)`` provides a straight-forward way for users
      to specify arguments to construct the model.
    - Type annotations are used to set expected input types and output keys,
      and ``beartype`` decorator is used to enforce type checking at runtime.
    """

    def __init__(self, num_atom_types: int = 100, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            self.num_atom_types, self.hidden_dim, padding_idx=0, max_norm=1.0
        )
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.projection = nn.Linear(self.hidden_dim, 1)

    @beartype
    def forward(
        self,
        atomic_numbers: AtomicNumbers,
        positions: NodePositions,
        batch_indices: BatchIndices | None = None,
        compute_forces: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the demo model.

        Computes energies and forces for a given molecular system using atomic
        embeddings and coordinate embeddings with residual connections.

        This model is **not** invariant or equivariant to translation/rotation.

        Parameters
        ----------
        atomic_numbers : AtomicNumbers
            Atomic numbers for each atom in the system.
        positions : CoordinatesTensor
            Cartesian coordinates of atoms.
        batch_indices : BatchIndices, optional
            Batch indices for each atom.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - energies: Predicted energy values. Shape: (batch_size, 1)
            - forces: Computed forces via automatic differentiation.
                     Shape: (batch_size, 3)
        """
        atom_z = self.embedding(atomic_numbers)
        coord_z = self.coord_embedding(positions)
        embedding = self.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))
        # effectively a residual connection
        embedding = embedding + atom_z + coord_z
        node_energy = self.projection(embedding)
        # scatter add to get energies of the graph
        if batch_indices is not None:
            num_graphs = batch_indices.max() + 1
            energies = torch.zeros(
                (num_graphs, 1),
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energies.scatter_add_(0, batch_indices.unsqueeze(-1), node_energy)
        else:
            energies = node_energy.sum(dim=0, keepdim=True)
        return_dict = {"energies": energies}
        # forces may be present in the output
        if compute_forces:
            forces = -torch.autograd.grad(
                energies,
                inputs=[positions],
                grad_outputs=torch.ones_like(energies),
                create_graph=self.training,
                retain_graph=self.training,
            )[0]
            return_dict["forces"] = forces
        return return_dict


class DemoModelWrapper(DemoModel, BaseModelMixin):
    """
    Wrapper for the demo model that implements the BaseModelMixin interface.

    This wrapper is used to implement the BaseModelMixin interface for the demo model.

    This demonstrates how `BaseModelMixin` adds elements to the underlying model
    that help standardize the model's interface and behavior, and allow for the
    original model to be used without modification.
    """

    @property
    def model_card(self) -> ModelCard:
        """Returns the model card for the demo model.

        This serves as an immutable specification of the model's
        capabilities and requirements.

        Fields will all default to False, and are expected to be
        set to True if the model expects an input, or directly
        outputs a particular property.

        Returns
        -------
        ModelCard
            Model card for the demo model.
        """
        return ModelCard(
            forces_via_autograd=True,
            supports_energies=True,
            supports_forces=True,
            supports_stresses=False,
            supports_hessians=False,
            supports_dipoles=False,
            supports_non_batch=True,
            needs_neighborlist=False,
            needs_pbc=False,
            model_name=self.__class__.__name__,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Returns the expected shapes of the embeddings for the demo model.

        This serves as an immutable specification of the model's
        expected embedding shapes.

        Returns
        -------
        dict[str, tuple[int, ...]]
            Expected shapes of the embeddings for the demo model.
        """
        return {
            "node_embeddings": (self.hidden_dim,),
            "graph_embedding": (self.hidden_dim,),
        }

    @property
    def dtype(self) -> torch.dtype:
        """Returns the projection layer's datatype for casting."""
        return self.projection.weight.dtype

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """
        Adapts the input data to the model's expected format.

        This method is used to convert the input data to the format expected by the model.
        """
        model_inputs = super().adapt_input(data, **kwargs)
        # type cast parameters to match the model's expected types
        model_inputs["atomic_numbers"] = data.atomic_numbers
        model_inputs["positions"] = data.positions.to(self.dtype)
        if isinstance(data, Batch):
            model_inputs["batch_indices"] = data.batch
        else:
            model_inputs["batch_indices"] = None
        # pass model config to the behavior of the underlying model
        model_inputs["compute_forces"] = self.model_config.compute_forces
        return model_inputs

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """
        Computes the embeddings for the demo model.

        This method is used to compute the embeddings for the demo model.

        This implementation assumes that the underlying model does not
        expose embeddings, and relies on the developer to effectively
        duplicate the logic of the `forward` call and extract embeddings.

        For models that *do* expose embeddings, the implementation can be
        simplified by calling `forward` or other methods.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data to compute embeddings for.

        Returns
        -------
        AtomicData | Batch
            Input data with computed embeddings.
        """
        model_inputs = self.adapt_input(data, **kwargs)
        atom_z = self.embedding(model_inputs["atomic_numbers"])
        coord_z = self.coord_embedding(model_inputs["positions"])
        embedding = self.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))
        embedding = embedding + atom_z + coord_z
        if isinstance(data, Batch):
            batch_indices = data.batch
        else:
            batch_indices = torch.zeros_like(model_inputs["atomic_numbers"])
        num_graphs = 1 if isinstance(data, AtomicData) else data.batch_size
        # scatter add to compute graph-level embeddings; while you could retrieve hidden_dim
        # directly and more easily, retrieving it from the property is to show off the interface
        graph_embedding_shape = self.embedding_shapes["graph_embedding"]
        graph_embedding = torch.zeros(
            (num_graphs, *graph_embedding_shape),
            device=embedding.device,
            dtype=embedding.dtype,
        )
        graph_embedding.scatter_add_(0, batch_indices.unsqueeze(-1), embedding)
        # write embeddings to data structure
        data.graph_embeddings = graph_embedding
        data.node_embeddings = embedding
        return data

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """
        Adapts the model output to the framework's expected format.

        The super() implementation will provide the initial OrderedDict with keys
        that are expected to be present in the model output. This method will then
        map the model outputs to this OrderedDict.

        Technically, this is not necessary for the demo model, but it is included
        to demonstrate how to override the super() implementation.
        """
        output = super().adapt_output(model_output, data)
        energies = model_output["energies"]
        # this shows how to handle unbatched data and conform to expected output shapes
        if isinstance(data, AtomicData) and energies.ndim == 1:
            energies.unsqueeze_(-1)
        output["energies"] = energies
        if self.model_config.compute_forces:
            output["forces"] = model_output["forces"]
        # can check that none of the expected keys are missing
        for key, value in output.items():
            if value is None:
                raise KeyError(
                    f"Key '{key}' not found in model output but is supported and requested."
                )
        return output

    @beartype
    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """
        Forward pass of the demo model wrapper.

        This wraps the underlying model's forward pass, adapting both the
        input and outputs to the framework's expected format.

        Parameters
        ----------
        data : AtomicData | Batch
            Input data to use for the forward pass.

        Returns
        -------
        ModelOutputs
            Output data from the forward pass, formatted with the correct key
            naming conventions.
        """
        model_inputs = self.adapt_input(data, **kwargs)
        model_outputs = super().forward(**model_inputs)
        return self.adapt_output(model_outputs, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """
        Export the demo model without the ``BaseModelMixin`` interface.

        This is not intended to be used in production, but more as a
        demonstration of how to export the model as you may with an
        actual subclass.

        Parameters
        ----------
        path : Path
            Path to export the model to.
        as_state_dict : bool, optional
            Whether to export the model as a state dictionary.
            Defaults to False, which pickles the model entirely.
        """
        base_cls = self.__class__.__mro__[1]
        # thankfully, this model has no required arguments so this
        # actually works
        base_model = base_cls()
        for name, module in self.named_children():
            setattr(base_model, name, module)
        if as_state_dict:
            torch.save(base_model.state_dict(), path)
        else:
            torch.save(base_model, path)
