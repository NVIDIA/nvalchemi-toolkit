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
"""Node-embedding matching loss and the projector that makes it cross-architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from jaxtyping import Bool, Float

from nvalchemi._typing import BatchIndices, NodeEmbeddings
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)
from nvalchemi.training.losses.reductions import per_graph_sum

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch

__all__ = ["EmbeddingMatchingLoss", "EmbeddingProjector"]

_NodeMask: TypeAlias = Bool[torch.Tensor, "V H"]
_PerGraphValues: TypeAlias = Float[torch.Tensor, "B"]

_PROJECTOR_REMEDY = (
    "Register an EmbeddingProjector of the student's width by the teacher's as "
    "an auxiliary model named 'projector', with an optimizer config of its own, "
    "and train with training_fn=embedding_distillation_fn, which routes the "
    "student's embeddings through it."
)
"""What to do about a student and teacher whose embedding widths differ."""


class EmbeddingProjector(torch.nn.Module, BaseModelMixin):
    """Learnable map from the student's embedding width to the teacher's.

    Embedding matching compares two representations component by component, so
    it needs them to have the same width. Architectures rarely do, and the
    student's width is a capacity decision rather than something to give up, so
    the widths are reconciled by a small learned map on the *student* side —
    trained jointly with the student, against fixed teacher targets.

    The projection is deliberately not applied to the teacher. A learnable map
    on the target side is optimized to be easy to hit, and the pair minimizes
    the objective by collapsing the teacher's representation rather than by
    teaching the student anything.

    ``EmbeddingProjector`` is registered as an ordinary named model of
    :class:`~nvalchemi.training.distillation.DistillationStrategy` — the only
    place a module's parameters are visible to ``setup_optimizers``, since a
    loss term's are not — so it needs an ``optimizer_configs`` entry like the
    student. Its parameters are saved and restored with every other model in a
    checkpoint, and thrown away at the end of training: the distilled artifact
    is the student alone, which is why the projector never sits between the
    student and its outputs. Every constructor argument is kept as an attribute
    of the same name, because that is what a checkpoint's model spec is rebuilt
    from — a projector restores at the shape and bias it was built with rather
    than at the defaults.

    Being a named model makes it a :class:`~nvalchemi.models.base.BaseModelMixin`,
    but it is an adapter rather than a model of a physical system: it declares
    no outputs, needs no neighbor list, and its
    :meth:`~torch.nn.Module.forward` takes the embedding tensor it maps rather
    than a batch of atomic graphs.
    :meth:`compute_embeddings` is the batch-shaped entry point, replacing the
    embeddings on the data it is handed. The class ships from the module that
    defines the term it serves, since it exists for that objective alone and
    its module path is written into every checkpoint spec that carries it.

    Parameters
    ----------
    in_features : int
        Width of the student's node embeddings.
    out_features : int
        Width of the teacher's node embeddings.
    hidden_features : int | None, optional
        Width of one hidden layer, making the projector a two-layer perceptron
        with a :class:`~torch.nn.SiLU` nonlinearity. Default ``None``, a single
        linear map.
    bias : bool, optional
        Whether the linear layers carry a bias. Default ``True``.

    Raises
    ------
    ValueError
        If a width is not positive.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.distillation import EmbeddingProjector
    >>> projector = EmbeddingProjector(64, 256)
    >>> projector(torch.zeros(10, 64)).shape
    torch.Size([10, 256])

    Notes
    -----
    A linear projector is the default because it is the weakest map that can
    reconcile the widths, and a weak map is the point: a projector with enough
    capacity to fit the teacher's representation from any student
    representation makes the loss satisfiable without the student learning the
    teacher's structure. Reach for ``hidden_features`` only when a linear map
    leaves the term stuck far above zero while the energy and force terms
    converge.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | None = None,
        bias: bool = True,
    ) -> None:
        """Build the linear or two-layer map between the two widths."""
        super().__init__()
        for name, width in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("hidden_features", hidden_features),
        ):
            if width is not None and width <= 0:
                raise ValueError(
                    f"EmbeddingProjector widths must be positive; got {name}={width!r}."
                )
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.bias = bias
        self.model_config = ModelConfig(
            outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
        )
        self.projection = (
            torch.nn.Linear(in_features, out_features, bias=bias)
            if hidden_features is None
            else torch.nn.Sequential(
                torch.nn.Linear(in_features, hidden_features, bias=bias),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_features, out_features, bias=bias),
            )
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the per-node shape this projector maps embeddings to."""
        return {"node_embeddings": (self.out_features,)}

    def forward(self, embeddings: NodeEmbeddings) -> NodeEmbeddings:
        """Return *embeddings* mapped from the student's width to the teacher's.

        Parameters
        ----------
        embeddings : NodeEmbeddings
            Student node embeddings of shape ``(V, in_features)``.

        Returns
        -------
        NodeEmbeddings
            Projected embeddings of shape ``(V, out_features)``.
        """
        return self.projection(embeddings)

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Replace the node embeddings on *data* with their projection, in place.

        Parameters
        ----------
        data : AtomicData | Batch
            Data already carrying node embeddings, as the student's own
            :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`
            leaves it.
        **kwargs : Any
            Unused; accepted for interface compatibility.

        Returns
        -------
        AtomicData | Batch
            *data*, with projected embeddings.

        Raises
        ------
        KeyError
            If *data* carries no ``node_embeddings`` to project.
        """
        del kwargs
        projected = self(data["node_embeddings"])
        # Write to the atoms group directly: a plain attribute set would route
        # per-atom embeddings to the system group.
        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = projected
        else:
            data.node_embeddings = projected
        return data

    def extra_repr(self) -> str:
        """Human-readable width summary for :class:`nn.Module`'s repr."""
        return (
            f"in_features={self.in_features!r}, "
            f"out_features={self.out_features!r}, "
            f"hidden_features={self.hidden_features!r}, "
            f"bias={self.bias!r}"
        )


class EmbeddingMatchingLoss(BaseLossFunction):
    r"""Mean-squared-error loss on per-atom representations.

    What a teacher knows that its energies and forces do not say is how it
    represents an atom's environment, and this term supervises the student with
    it directly. Prediction and target are node-level tensors of shape
    ``(V, H)``: the student's node embeddings against the teacher's
    ``embeddings`` signal, which
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` writes to
    ``teacher_node_embeddings``. The per-component residual is

    .. math::

        \rho_{iah} = \left(\hat{z}_{iah} - z_{iah}\right)^2,

    for component :math:`h` of atom :math:`a` of graph :math:`i`, and is
    reduced according to ``normalize_by_atom_count``. Writing
    :math:`\mathcal{V}_i` for the atoms of graph :math:`i` that ``mask``
    accepts and :math:`M_i = |\mathcal{V}_i|`:

    - ``normalize_by_atom_count=True`` (default): each graph's mean residual is
      averaged over graphs, so every structure contributes equally regardless
      of size,

      .. math::

          L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{\max(H M_i, 1)}
          \sum_{a \in \mathcal{V}_i} \sum_{h=1}^{H} \rho_{iah}.

    - ``normalize_by_atom_count=False``: one global mean over every valid
      component, so a large structure dominates a small one.

    Parameters
    ----------
    target_key : str, default "teacher_node_embeddings"
        Target container key for the teacher's node embeddings.
    prediction_key : str, default "predicted_node_embeddings"
        Prediction container key for the student's node embeddings, which the
        stock student forward pass does not produce — see the Notes.
    normalize_by_atom_count : bool, default True
        When ``True``, compute a mean residual per graph, then mean over
        graphs. When ``False``, compute one global mean over valid components.
    ignore_nonfinite : bool, default True
        When ``True``, components whose target is ``NaN`` or infinite are
        excluded from both loss value and gradient.
    dtype_policy : {"strict", "prediction_to_target", "target_to_prediction"}, default "strict"
        How to handle prediction/target dtype mismatches before validation.

    Raises
    ------
    ValueError
        If the student's and teacher's embedding widths differ, or if the
        graph-balanced reduction is requested without ``batch_idx`` and
        ``num_graphs`` metadata.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.training.distillation import EmbeddingMatchingLoss
    >>> loss_fn = EmbeddingMatchingLoss()
    >>> pred = torch.tensor([[0.0, 2.0], [0.0, 0.0]])
    >>> target = torch.zeros(2, 2)
    >>> batch_idx = torch.tensor([0, 1])
    >>> loss_fn(pred, target, batch_idx=batch_idx, num_graphs=2)
    tensor(1.)

    See Also
    --------
    EmbeddingProjector : Learnable width adapter for cross-architecture runs.

    Notes
    -----
    Embeddings do not come out of a forward pass — they come from
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, a second
    pass over the batch — on the teacher side and the student side alike. The
    stock
    :func:`~nvalchemi.training.distillation.default_distillation_fn` therefore
    cannot serve this term, and
    :class:`~nvalchemi.training.distillation.DistillationStrategy` refuses it at
    construction rather than on the first batch. Train with
    :func:`~nvalchemi.training.distillation.embedding_distillation_fn`, which
    runs both passes and routes the student's embeddings through a
    ``"projector"`` model when one is registered.

    Two representations of the same environment agree only up to whatever
    symmetry each architecture's embedding space carries, which nothing here
    can quotient out: a permutation of the student's channels, or a rotation of
    an equivariant block, is a perfect match this term reports as a large
    error. That is what the learnable :class:`EmbeddingProjector` absorbs, and
    it is why a residual floor on this term is normal and not by itself a sign
    the student has stopped learning. Weight the term as a regularizer against
    energy and force terms that carry the physical targets.
    """

    requires_eval_grad: bool = False

    def __init__(
        self,
        *,
        target_key: str = "teacher_node_embeddings",
        prediction_key: str = "predicted_node_embeddings",
        normalize_by_atom_count: bool = True,
        ignore_nonfinite: bool = True,
        dtype_policy: DTypePolicy = "strict",
    ) -> None:
        """Configure attribute keys and embedding reduction semantics."""
        super().__init__(dtype_policy=dtype_policy)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.normalize_by_atom_count = normalize_by_atom_count
        self.ignore_nonfinite = ignore_nonfinite

    def validate(self, pred: NodeEmbeddings, target: NodeEmbeddings) -> None:
        """Check the two representations agree on shape, width included."""
        if (
            pred.ndim == target.ndim == 2
            and pred.shape[0] == target.shape[0]
            and pred.shape[-1] != target.shape[-1]
        ):
            raise ValueError(
                "EmbeddingMatchingLoss compares representations component by "
                "component, so the student's embedding width must equal the "
                f"teacher's; got student {tuple(pred.shape)} against teacher "
                f"{tuple(target.shape)}. {_PROJECTOR_REMEDY}"
            )
        super().validate(pred, target)

    def mask(
        self,
        pred: NodeEmbeddings,
        target: NodeEmbeddings,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> _NodeMask:
        """Return one validity flag per embedding component."""
        if self.ignore_nonfinite:
            return torch.isfinite(target)
        return torch.ones_like(target, dtype=torch.bool)

    def compute_residual(
        self,
        pred: NodeEmbeddings,
        target: NodeEmbeddings,
        valid: _NodeMask,
    ) -> NodeEmbeddings:
        """Return squared component residuals, zeroing invalid components."""
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def reduce(
        self,
        residual: NodeEmbeddings,
        valid: _NodeMask,
        ctx: ReductionContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reduce squared component residuals to a scalar loss."""
        valid_components = valid.to(dtype=residual.dtype)
        if not self.normalize_by_atom_count:
            return residual.sum() / valid_components.sum().clamp_min(1.0)
        batch: Batch | None = kwargs.get("batch")
        batch_idx: BatchIndices | None = kwargs.get("batch_idx")
        num_graphs: int | None = kwargs.get("num_graphs")
        if batch is not None:
            if batch_idx is None:
                batch_idx = getattr(batch, "batch_idx", None)
            if num_graphs is None:
                num_graphs = getattr(batch, "num_graphs", None)
        per_graph_residual, per_graph_counts = self._per_graph_terms(
            residual, valid_components, batch_idx, num_graphs
        )
        per_sample = per_graph_residual / per_graph_counts.clamp_min(1.0)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()

    def _per_graph_terms(
        self,
        residual: NodeEmbeddings,
        valid_components: NodeEmbeddings,
        batch_idx: BatchIndices | None,
        num_graphs: int | None,
    ) -> tuple[_PerGraphValues, _PerGraphValues]:
        """Return per-graph residual sums and valid component counts."""
        if batch_idx is None or num_graphs is None:
            raise ValueError(
                "EmbeddingMatchingLoss needs batch_idx and num_graphs metadata for "
                f"its graph-balanced reduction; got batch_idx={batch_idx!r}, "
                f"num_graphs={num_graphs!r}."
            )
        return (
            per_graph_sum(residual.sum(dim=-1), batch_idx, num_graphs=num_graphs),
            per_graph_sum(
                valid_components.sum(dim=-1), batch_idx, num_graphs=num_graphs
            ),
        )

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )
