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
"""The conservative-bias battery.

A bias in this toolkit is an **additive potential** — the same shape as
``DFTD3ModelWrapper`` and ``LennardJonesModelWrapper``, which is why it is
built from the same parts rather than from a parallel hierarchy of its own:

* It produces :data:`~nvalchemi._typing.ModelOutputs`, like every model.  There
  is no bias-specific result type.  Diagnostics travel as ``diagnostics/<key>``
  entries and the producer's state revision as ``state_version``; both are
  general ``ModelOutputs`` conventions, not enhanced-sampling ones.
* It is a :class:`~nvalchemi.models.base.BaseModelMixin`, which supplies
  ``model_config.active_outputs``, ``distribution_spec``, and ``+``
  composition.  There is no separate ``BiasPotential`` protocol: "produces
  ``ModelOutputs`` from a ``Batch``" is what ``BaseModelMixin`` already means,
  and a second name for it would only be a second thing to keep in sync.
* A bias whose state evolves during sampling additionally mixes in
  :class:`~nvalchemi.enhanced_sampling.AdaptivePotentialMixin`, which is the
  shared ``StatefulHook`` lifecycle — ``frequency``, ``stage``, ``read_only``,
  ``commit`` — and not a lifecycle invented here.

:class:`ConservativeBias` itself is thin.  The autograd work lives in
:func:`~nvalchemi.models._utils.isolated_energy_derivatives`, because
evaluating an energy against a detached view of a live ``Batch`` and handing
back derivatives with no graph attached is useful to anything that
differentiates against a batch it does not own — an NEB spring term, a
hand-written wall — and not only to biases.

Why the *applied* outputs are a closed set
------------------------------------------
``ModelOutputs`` is an open mapping, and a bias may put any key in it.  Only a
defined set is ever *applied*: for each one the runner must know the
destination buffer (``batch.energy``, ``batch.forces``, ``batch.stress``),
whether it is per-graph or per-atom, and how it combines across biases.  An
unrecognised applied key has none of that, so
``EnhancedSampling._check_destinations`` raises rather than dropping a
contribution in silence.  The open extension point is ``diagnostics/<key>``:
arbitrary tensors, no shape contract, reported and never applied.  The split
is between what the runner acts on and what it merely records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor, nn

from nvalchemi.models._utils import isolated_energy_derivatives
from nvalchemi.models.base import BaseModelMixin, ModelConfig

if TYPE_CHECKING:
    from pathlib import Path

    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.distributed.config import StrategyKind
    from nvalchemi.distributed.spec import MLIPSpec

__all__ = ["ConservativeBias"]


class ConservativeBias(nn.Module, BaseModelMixin):
    """A bias defined by a differentiable energy.

    Subclass and override :meth:`energy` to return a per-graph bias energy
    ``[B, 1]``; :meth:`forward` derives atomic forces and tensile-positive
    Cauchy stress from it and returns detached
    :data:`~nvalchemi._typing.ModelOutputs`.

    Composed as ``nn.Module, BaseModelMixin`` — the house multiple-inheritance
    idiom (``LennardJonesModelWrapper(nn.Module, BaseModelMixin)``,
    ``TrainingStrategy(BaseModel, HookRegistryMixin)``).
    ``BaseModelMixin.__init_subclass__`` is cooperative, so it composes with
    :class:`~nvalchemi.enhanced_sampling.AdaptivePotentialMixin` for a bias
    that is adaptive as well as conservative.  The cost of the abstraction is
    the two ``BaseModelMixin`` abstract methods (:attr:`embedding_shapes` and
    :meth:`compute_embeddings`), stubbed here exactly as
    ``LennardJonesModelWrapper`` and ``DFTD3ModelWrapper`` stub them.

    .. note::

        Subclasses **must** call ``super().__init__(name=...)``.  That is the
        ``nn.Module`` requirement (attribute assignment before
        ``Module.__init__`` raises), and ``BaseModelMixin.__init_subclass__``
        additionally verifies ``self.model_config`` is set afterwards.

    Notes
    -----
    How the derivatives are taken — the symmetric strain leaf, the
    substitute-and-restore, the detachment — is
    :func:`~nvalchemi.models._utils.isolated_energy_derivatives` and is
    documented there.  It is deliberately not restated here: one mechanism
    described in two places is one that can drift.  What follows is what is
    true of *this class* and not of that function.

    Stress rather than virial
        :meth:`forward` emits ``"stress"``, never ``"virial"``.  That is a
        policy choice, not a consequence of the maths: both are documented
        toolkit conventions, and tensile-positive Cauchy stress is the one the
        rest of the toolkit speaks — every model wrapper emits it,
        :func:`~nvalchemi.models._utils.sum_outputs` treats it as additive,
        and the NPT/NPH integrators read ``batch.stress``.  So a bias
        contribution sums with model output with no volume conversion at the
        boundary.  ``"virial"`` stays available for a hand-written bias that
        produces one directly.

    What gates stress here
        The helper computes stress for any periodic batch with a non-zero
        cell.  This class adds one gate on top: ``"stress"`` must be in
        ``model_config.active_outputs``.  Pass ``compute_stress=False`` to
        drop it at construction, or flip ``active_outputs`` at runtime as on
        any model wrapper.

    ``torch.compile`` compatibility
        :meth:`forward` is eager, because the helper it delegates to calls
        ``requires_grad_()``.  Compile :meth:`energy` — the hot path —
        instead; ``EnhancedSampling(compile_biases=True)`` does exactly that.
        Keep data-dependent Python branches out of :meth:`energy` for the
        same reason; put them in a :meth:`forward` override, which is eager by
        construction, as ``HarmonicUmbrellaBias`` does for its bounds check.
    """

    def __init__(self, name: str, *, compute_stress: bool = True) -> None:
        """Initialise the bias and declare its output capabilities.

        Parameters
        ----------
        name:
            Unique identifier, used as a dict key in
            ``EnhancedSampling(biases={...})`` and as a checkpoint group name.
        compute_stress:
            When ``False``, ``"stress"`` is dropped from
            ``model_config.active_outputs`` and the strain leaf is skipped
            entirely.  Use for force-only biases.
        """
        super().__init__()
        self.name = name
        outputs = {"energy", "forces", "stress"}
        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions", "cell"}),
            supports_pbc=True,
            needs_pbc=False,
            active_outputs=outputs if compute_stress else {"energy", "forces"},
        )

    # ------------------------------------------------------------------
    # BaseModelMixin required surface
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """No embeddings: a bias potential is a closed-form energy term."""
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Computing embeddings is not meaningful for a bias potential."""
        raise NotImplementedError(f"{type(self).__name__} does not produce embeddings.")

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Not implemented — a bias has no underlying model to export."""
        raise NotImplementedError(
            f"{type(self).__name__} has no exportable model; use state_dict()."
        )

    def distribution_spec(
        self, strategy: StrategyKind | None = None
    ) -> MLIPSpec | None:
        """Return ``None``: a bias does not claim domain-decomposition support.

        ``None`` is not an oversight, and it is not "unsupported forever" — it
        makes ``DistributedModel`` raise ``DistributionError`` rather than
        shard a bias whose cross-rank semantics are unverified (an explicit
        ``DistributedModel(bias, cfg, spec=...)`` remains the escape hatch for
        a caller who knows better).

        The default cannot be a halo preset, because a bias is not necessarily
        local the way a cutoff potential is.  A CV can couple atoms in
        different domains by construction: ``pair_distance`` over two atoms on
        opposite sides of the cell has no cutoff, and an RMSD bias reads every
        atom.  ``SPEC_LJ_HALO`` is correct for LJ precisely because a halo
        exchange covers its interaction range; nothing guarantees that for an
        arbitrary CV.

        A bias that *is* local should override this and declare its outputs,
        e.g.::

            def distribution_spec(self, strategy=None):
                return MLIPSpec(
                    distribution=DistributionSpec(policy=HaloStoragePolicy()),
                    outputs={
                        "energy": OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE),
                        "forces": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
                        "stress": OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE),
                    },
                )

        Parameters
        ----------
        strategy:
            Accepted for the framework contract; ignored by the default.

        Returns
        -------
        MLIPSpec | None
            Always ``None`` unless a subclass overrides.
        """
        return None

    # ------------------------------------------------------------------
    # Bias surface
    # ------------------------------------------------------------------

    def _align_device(self, reference: Tensor) -> None:
        """Move this bias's buffers to *reference*'s device if they differ.

        A bias holds its parameters as buffers (window centers, stiffness,
        wall thresholds), and a user who builds the bias before moving the
        batch to GPU would otherwise hit a bare "expected all tensors to be on
        the same device" from inside the energy expression, naming neither the
        bias nor the fix.  Moving once here is cheaper than making every bias
        author remember ``.to(device)``.

        Eager-only, like the rest of :meth:`forward`.

        Parameters
        ----------
        reference:
            Any tensor from the live batch; its device is the target.
        """
        buffer = next(self.buffers(), None)
        if buffer is not None and buffer.device != reference.device:
            self.to(reference.device)

    def energy(self, current: Batch) -> Tensor:
        """Return bias energy ``[B, 1]`` (eV).

        Must be differentiable w.r.t. ``current.positions`` and/or
        ``current.cell``.  Depending on only one of the two is allowed: a
        position-independent term (a volume restraint, say) yields zero forces
        rather than an error.

        Parameters
        ----------
        current:
            A *read-only view* of the live batch whose ``positions`` and
            ``cell`` have been replaced by their strained counterparts from
            :func:`~nvalchemi.models._utils.prepare_strain`.  Do not assign to
            any batch field inside this method.

        Returns
        -------
        Tensor
            Shape ``[B, 1]``.

        Raises
        ------
        NotImplementedError
            If the subclass does not override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement energy(self, current: Batch) -> Tensor"
        )

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Return the bias contribution for *data*, fully detached.

        Read-only with respect to the bias: it deposits nothing, writes no
        storage, and communicates with no other worker, so it is safe to call
        repeatedly on the same batch.  An adaptive bias changes its state in
        :meth:`~nvalchemi.enhanced_sampling.AdaptivePotentialMixin.update`
        instead, which the runner delivers once per due step after the
        integrator has finished.

        *data* is treated as read-only: every field
        :func:`~nvalchemi.models._utils.isolated_energy_derivatives`
        substitutes is restored before this returns, including on an
        exception.

        Parameters
        ----------
        data:
            The live batch from the dynamics step.
        **kwargs:
            Unused; accepted for interface compatibility with model wrappers.

        Returns
        -------
        ModelOutputs
            ``energy``, ``forces``, and — when ``"stress"`` is active and the
            batch is periodic — ``stress``, restricted to
            ``model_config.active_outputs``.  Every tensor satisfies
            ``requires_grad=False`` and ``grad_fn is None``.

            Not validated here:
            :func:`~nvalchemi.models._utils.validate_contribution` runs where
            the contribution is *consumed* — that is where an unchecked value
            does damage, by being added into a batch buffer, retained in a
            history, or written to a checkpoint — so
            :class:`~nvalchemi.enhanced_sampling.EnhancedSampling` calls it on
            every bias it applies.  Call it directly to self-check a bias used
            outside the runner.

        Notes
        -----
        Called with ``allow_unused=True``, so :meth:`energy` need not depend on
        both positions and strain — see that parameter on
        :func:`~nvalchemi.models._utils.isolated_energy_derivatives` for what
        a partial dependence yields.
        """
        self._align_device(data.positions)
        active = self.model_config.active_outputs or set()
        outputs = isolated_energy_derivatives(
            self.energy,
            data,  # type: ignore[arg-type]
            want_forces="forces" in active,
            want_stress="stress" in active,
            # A bias need not depend on both positions and strain. A pure
            # volume restraint has no position dependence, and its zero force
            # is an answer, not an error.
            allow_unused=True,
        )
        for key in list(outputs):
            if key not in active:
                del outputs[key]
        return outputs
