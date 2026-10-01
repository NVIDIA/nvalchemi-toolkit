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
from jaxtyping import Bool

from nvalchemi._typing import NodeEmbeddings
from nvalchemi.data.batch import Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.training.losses.composition import (
    BaseLossFunction,
    DTypePolicy,
    ReductionContext,
)
from nvalchemi.training.losses.reductions import masked_mean

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData

__all__ = ["EmbeddingMatchingLoss", "EmbeddingProjector"]

_NodeMask: TypeAlias = Bool[torch.Tensor, "V H"]

_PROJECTOR_REMEDY = (
    "Register an EmbeddingProjector(student_width, teacher_width) as an "
    "auxiliary model named 'projector', with an optimizer config of its own, "
    "and train with training_fn=embedding_distillation_fn, which routes the "
    "student's embeddings through it."
)
"""Remedy for a student and teacher whose embedding widths differ."""


class EmbeddingProjector(torch.nn.Module, BaseModelMixin):
    """Learnable map from the student's embedding width to the teacher's.

    Embedding matching compares the student's and teacher's per-atom
    representations component by component, so the two widths must agree.
    Different architectures rarely share a width, and the student's width is a
    capacity decision. The projector reconciles the widths with a small map on
    the *student* side, trained jointly against fixed teacher targets. It is
    never applied to the teacher: a learnable map on the target side minimizes
    the objective by collapsing the teacher's representation.

    The projector is registered as an ordinary named model of
    :class:`~nvalchemi.training.distillation.DistillationStrategy`, because
    named models are the only place ``setup_optimizers`` sees a module's
    parameters. It therefore needs an ``optimizer_configs`` entry and is
    checkpointed like every other model. It is discarded at the end of
    training, because the distilled artifact is the student alone. Every
    constructor argument is stored as an attribute of the same name, from which
    a checkpoint's model spec rebuilds the projector. The projector is an
    adapter rather than a model of a physical system: it declares no outputs
    and needs no neighbor list. :meth:`forward` maps an embedding tensor, and
    :meth:`compute_embeddings` replaces the embeddings on a batch in place.

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
    frozen_student : bool, optional
        Whether the student's representation is frozen on purpose, so that the
        embedding term (the
        :class:`~nvalchemi.training.distillation.EmbeddingMatchingLoss`
        component of the loss) trains this projector alone. When ``True``,
        :func:`~nvalchemi.training.distillation.embedding_distillation_fn`
        accepts student embeddings that are detached from the student's
        trainable parameters, such as those of a frozen trunk beside a
        trainable head. When ``False``, it treats them as an accident and
        refuses them. Default ``False``.

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
    reconcile the widths. A projector with enough capacity to fit the teacher's
    representation from any student representation satisfies the loss without
    the student learning anything. Set ``hidden_features`` only when the
    embedding term stalls under a linear map while the energy and force terms
    converge.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | None = None,
        bias: bool = True,
        frozen_student: bool = False,
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
        self.frozen_student = frozen_student
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
        if isinstance(data, Batch):
            data.add_key(
                "node_embeddings",
                list(projected.split(data.num_nodes_list)),
                level="node",
                overwrite=True,
            )
        else:
            data.node_embeddings = projected
        return data

    def extra_repr(self) -> str:
        """Human-readable width summary for :class:`nn.Module`'s repr."""
        return (
            f"in_features={self.in_features!r}, "
            f"out_features={self.out_features!r}, "
            f"hidden_features={self.hidden_features!r}, "
            f"bias={self.bias!r}, "
            f"frozen_student={self.frozen_student!r}"
        )


class EmbeddingMatchingLoss(BaseLossFunction):
    r"""Mean-squared-error loss on per-atom representations.

    The prediction is the student's node embeddings. The target is the
    teacher's ``embeddings`` signal, which
    :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` writes to
    ``teacher_node_embeddings``. Both are node-level tensors of shape
    ``(V, H)``. The per-component residual
    :math:`\rho_{iah} = (\hat{z}_{iah} - z_{iah})^2` is reduced according to
    ``normalize_by_atom_count``. By default, with :math:`\mathcal{V}_i` the
    atoms of graph :math:`i` that ``mask`` accepts and
    :math:`M_i = |\mathcal{V}_i|`, the loss is

    .. math::

        L = \frac{1}{B} \sum_{i=1}^{B} \frac{1}{\max(H M_i, 1)}
        \sum_{a \in \mathcal{V}_i} \sum_{h=1}^{H} \rho_{iah}

    so every structure contributes equally regardless of size. When
    ``normalize_by_atom_count`` is ``False``, the loss is one global mean over
    every valid component.

    Parameters
    ----------
    target_key : str, default "teacher_node_embeddings"
        Target container key for the teacher's node embeddings.
    prediction_key : str, default "predicted_node_embeddings"
        Prediction container key for the student's node embeddings. The stock
        student forward pass does not produce them; see the Notes.
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
    On both sides, embeddings come from
    :meth:`~nvalchemi.models.base.BaseModelMixin.compute_embeddings`, a second
    pass over the batch. The stock
    :func:`~nvalchemi.training.distillation.default_distillation_fn` therefore
    cannot serve this term, and
    :class:`~nvalchemi.training.distillation.DistillationStrategy` refuses it
    at construction. Train with
    :func:`~nvalchemi.training.distillation.embedding_distillation_fn` instead.

    Two architectures' representations of one environment agree only up to
    the symmetries of each embedding space, such as a channel permutation or a
    rotation of an equivariant block. Both are linear maps, so a linear
    :class:`EmbeddingProjector` absorbs them; without a projector they are
    matched component by component and leave a residual floor. Differences
    that no linear map closes, such as a feature one architecture builds and
    the other does not, leave a floor with or without one. Weight the term as
    a regularizer beside the terms that carry the physical targets.
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
        """Check that the two representations agree in shape, width included."""
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
        """Reduce squared component residuals to a scalar loss.

        :func:`~nvalchemi.training.losses.reductions.masked_mean` takes the
        global mean over valid entries, or the graph-balanced one when
        ``normalize_by_atom_count`` is set, in at least float32 either way.
        """
        loss, per_sample = masked_mean(
            residual,
            valid,
            graph_balanced=self.normalize_by_atom_count,
            batch_idx=kwargs.get("batch_idx"),
            num_graphs=kwargs.get("num_graphs"),
            batch=kwargs.get("batch"),
            loss_name=type(self).__name__,
        )
        if per_sample is not None:
            self.per_sample_loss = per_sample.detach()
        return loss

    def extra_repr(self) -> str:
        """Human-readable hyperparameter summary for :class:`nn.Module`'s repr."""
        return (
            f"target_key={self.target_key!r}, "
            f"prediction_key={self.prediction_key!r}, "
            f"normalize_by_atom_count={self.normalize_by_atom_count!r}, "
            f"ignore_nonfinite={self.ignore_nonfinite!r}, "
            f"dtype_policy={self.dtype_policy!r}"
        )
