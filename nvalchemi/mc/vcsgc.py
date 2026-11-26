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
"""Variance-constrained semi-grand-canonical Monte Carlo transmutation moves."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.mc.sgc import SGC

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["VCSGC"]


class VCSGC(SGC):
    r"""Batched variance-constrained semi-grand-canonical (VC-SGC) Monte Carlo.

    Binary VC-SGC after Sadigh et al., Phys. Rev. B **85**, 184203 (2012). It
    reuses :class:`SGC`'s single-site transmutation proposal but replaces the
    linear reservoir term with a quadratic one in the concentration
    :math:`c = N_B / N` of ``concentration_species`` (called B below; the other
    species is A). Each graph samples

    .. math::

        P(\sigma) \propto \exp\!\left\{-\beta\left[E(\sigma)
            + N\left(\phi\,c + \kappa\,c^2\right)\right]\right\},

    so one transmutation that changes :math:`N_B` by :math:`\Delta n = \pm 1`
    is accepted with :math:`\min\{1, e^{-\beta[\Delta E + \Delta n\,(\phi +
    2\kappa\tilde c)]}\}`, where :math:`\tilde c` is the mean of the old and
    new concentrations. That midpoint form is exact, not an approximation.

    **What it measures.** At the saddle point of the concentration integral,
    the slope of the fixed-composition free energy at the ensemble mean
    :math:`\bar c` is

    .. math::

        \mu_B - \mu_A = \frac{1}{N}\frac{\partial F}{\partial c}\bigg|_{\bar c}
            = -(\phi + 2\kappa\bar c),

    returned per graph by :meth:`exchange_chemical_potential`. With a
    barostatted MD stage (hybrid MC-MD), :math:`F` is the Gibbs free energy.
    Integrating this slope over :math:`\bar c` gives :math:`F(c)` up to a
    constant, including compositions inside a miscibility gap, which plain
    :class:`SGC` cannot hold: there the concentration jumps to one side.

    **Choosing** :math:`\kappa`. The combined per-atom weight
    :math:`f(c) + \kappa c^2` must be convex for a composition to be stable,
    so hold a composition inside a gap with :math:`\kappa > -\min f''(c) / 2`.
    The concentration then fluctuates with variance
    :math:`\sigma_c^2 \approx k_B T / [N (f''(\bar c) + 2\kappa)]`: a larger
    :math:`\kappa` pins :math:`\bar c` more tightly but accepts fewer
    composition-changing moves. :math:`\kappa = 0` recovers :class:`SGC` with
    :math:`\mu_B - \mu_A = -\phi`.

    .. warning::

       Below that threshold the composition distribution turns bimodal while
       :math:`\bar c` can still sit at the target (the walker alternates between
       the two coexisting phases), and :meth:`exchange_chemical_potential` of
       such a mean is meaningless. Check that the sampled concentration is
       unimodal, e.g. that its standard deviation is close to
       :math:`\sigma_c` above, before using :math:`\bar c`. For a regular
       solution with :math:`\Omega = 0.3` eV at 700 K (threshold 0.18 eV),
       :math:`\kappa = 0.05` eV gives :math:`\bar c = 0.50` but a most-probable
       :math:`c` of 0.02, while :math:`\kappa = 0.5` eV holds :math:`c = 0.50
       \pm 0.015`.

    **Parametrisation.** Pass either ``phi`` or ``target_concentration``
    :math:`c_0`, which sets :math:`\phi = -2\kappa c_0` so that the weight is
    :math:`N\kappa(c - c_0)^2` up to a constant and
    :math:`\mu_B - \mu_A = \Delta\mu_\mathrm{ref} + 2\kappa(c_0 - \bar c)`. A
    phase-diagram scan grids over ``target_concentration`` directly.

    **Reference exchange potential.** ``reference_exchange_potential``
    :math:`\Delta\mu_\mathrm{ref}` makes :math:`\phi` an excess over a
    calibrated reference: the linear coefficient actually sampled is
    :math:`\phi - \Delta\mu_\mathrm{ref}`, so everything above holds with
    :math:`\mu_B - \mu_A` measured from :math:`\Delta\mu_\mathrm{ref}`
    (and :math:`\kappa = 0` is :class:`SGC` with
    :math:`\mu_B - \mu_A = \Delta\mu_\mathrm{ref} - \phi`). It defaults to 0.

    .. warning::

       Set it for machine-learned potentials. Their per-element energy offsets
       put the system's own :math:`\mu_B - \mu_A` eV away from zero, and the
       constraint can only supply :math:`2\kappa|c_0 - \bar c| \le 2\kappa`.
       Without a reference the walker therefore runs to the favoured end member
       instead of sampling near :math:`c_0`. Use the value calibrated from
       pure-element runs at the same temperature (the same
       :math:`\Delta\mu_\mathrm{ref}` an SGC scan is centred on), in the same
       convention: :math:`\mu_B - \mu_A` with B = ``concentration_species``.

    ``phi``, ``kappa`` and ``reference_exchange_potential`` are in eV and
    are *intensive*: Sadigh et al.'s variance parameter is
    :math:`\kappa_\mathrm{S} = \kappa / N`, so a fixed ``kappa`` gives the same
    stability criterion at every system size. Other codes use dimensionless or
    :math:`k_B T`-scaled parameters, so values do not transfer without
    conversion.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        species: Sequence[int],
        kappa: float | torch.Tensor,
        phi: float | torch.Tensor | None = None,
        target_concentration: float | torch.Tensor | None = None,
        concentration_species: int | None = None,
        reference_exchange_potential: float | torch.Tensor = 0.0,
        **kwargs: Any,
    ) -> None:
        r"""Initialize a binary VC-SGC sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        species
            The two allowed atomic numbers.
        kappa
            Non-negative variance-constraint strength in eV, a scalar or one
            value per graph. Must be positive when ``target_concentration`` is
            given.
        phi
            Linear constraint parameter in eV, a scalar or one value per graph.
            Mutually exclusive with ``target_concentration``.
        target_concentration
            Target fraction :math:`c_0 \in [0, 1]` of ``concentration_species``,
            a scalar or one value per graph. Sets ``phi = -2 * kappa * c0``.
        concentration_species
            Atomic number whose fraction is :math:`c`. Defaults to
            ``species[1]``.
        reference_exchange_potential
            Calibrated :math:`\Delta\mu_\mathrm{ref} = \mu_B - \mu_A` in eV,
            a scalar or one value per graph; ``phi`` is measured relative to
            it. Defaults to 0. See the class docstring's warning.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.

        Raises
        ------
        ValueError
            If ``species`` is not a pair, ``concentration_species`` is not one of
            them, both or neither of ``phi`` and ``target_concentration`` are
            given, ``kappa`` is negative (or not positive with a target), a
            target lies outside ``[0, 1]``, a parameter is not scalar or
            one-dimensional, or per-graph parameters differ in length.
        """
        # The linear reservoir of SGC is replaced by the VC-SGC term in
        # _chemical_delta; zero potentials only satisfy SGC's own validation.
        super().__init__(
            model=model,
            temperature=temperature,
            species=species,
            chemical_potentials={int(number): 0.0 for number in species},
            **kwargs,
        )
        if len(self.species) != 2:
            raise ValueError(
                "VC-SGC is implemented for binary systems: species must contain two atomic numbers"
            )
        self.concentration_species = int(
            self.species[1] if concentration_species is None else concentration_species
        )
        if self.concentration_species not in self.species:
            raise ValueError(
                "concentration_species must be one of the configured species"
            )
        if (phi is None) == (target_concentration is None):
            raise ValueError("pass exactly one of phi and target_concentration")

        self.kappa = self._parameter(kappa, "kappa")
        if torch.any(self.kappa < 0):
            raise ValueError("kappa must be non-negative")
        if target_concentration is not None:
            target = self._parameter(target_concentration, "target_concentration")
            if torch.any((target < 0) | (target > 1)):
                raise ValueError("target_concentration must lie in [0, 1]")
            if torch.any(self.kappa <= 0):
                raise ValueError(
                    "kappa must be positive when target_concentration is given"
                )
            self.phi = -2.0 * self.kappa * target
        else:
            self.phi = self._parameter(phi, "phi")
        self.reference_exchange_potential = self._parameter(
            reference_exchange_potential, "reference_exchange_potential"
        )
        try:
            # The coefficient of N*c actually sampled: phi is an excess over the reference.
            self._linear = self.phi - self.reference_exchange_potential
        except RuntimeError as exc:
            raise ValueError(
                "per-graph phi, kappa and reference_exchange_potential must have equal lengths"
            ) from exc

    @staticmethod
    def _parameter(value: float | torch.Tensor, name: str) -> torch.Tensor:
        """Detach a scalar or per-graph parameter and check its rank."""
        tensor = torch.as_tensor(value, dtype=torch.float64).detach().clone()
        if tensor.ndim > 1:
            raise ValueError(f"{name} must be scalar or one-dimensional per graph")
        return tensor

    def _per_graph(self, value: torch.Tensor, batch: Batch, name: str) -> torch.Tensor:
        """Broadcast a VC-SGC parameter to one value per graph on the batch device."""
        value = self._on_device(name, value, batch.device, batch.positions.dtype)
        if value.ndim == 0:
            return value.expand(batch.num_graphs)
        if value.shape != (batch.num_graphs,):
            raise ValueError(
                f"{name} tensors must have one value per graph "
                f"(expected {(batch.num_graphs,)}, got {tuple(value.shape)})"
            )
        return value

    def _solute_counts(self, batch: Batch) -> torch.Tensor:
        """Per-graph count of ``concentration_species`` (no device-to-host sync)."""
        is_solute = (batch.atomic_numbers.reshape(-1) == self.concentration_species).to(
            batch.positions.dtype
        )
        counts = torch.zeros(
            batch.num_graphs, dtype=batch.positions.dtype, device=batch.device
        )
        counts.index_add_(0, batch.batch_idx.to(torch.long), is_solute)
        return counts

    def concentration(self, batch: Batch) -> torch.Tensor:
        """Return the per-graph fraction :math:`c` of ``concentration_species``.

        Parameters
        ----------
        batch
            Batch whose current atom types are counted.

        Returns
        -------
        torch.Tensor
            Concentration per graph, shape ``[num_graphs]``.
        """
        return self._solute_counts(batch) / batch.num_nodes_per_graph.to(
            batch.positions.dtype
        )

    def exchange_chemical_potential(
        self, mean_concentration: torch.Tensor, batch: Batch | None = None
    ) -> torch.Tensor:
        r"""Return :math:`\mu_B - \mu_A = \Delta\mu_\mathrm{ref} - \phi - 2\kappa\bar c` per graph, in eV.

        Parameters
        ----------
        mean_concentration
            Ensemble-averaged concentration :math:`\bar c`, one value per graph
            (or a scalar when the parameters are scalar).
        batch
            Optional batch supplying the target device and dtype; per-graph
            parameters are broadcast against its graph count.

        Returns
        -------
        torch.Tensor
            Exchange chemical potential per graph.
        """
        if batch is not None:
            linear = self._per_graph(self._linear, batch, "phi")
            kappa = self._per_graph(self.kappa, batch, "kappa")
            mean = torch.as_tensor(
                mean_concentration, dtype=linear.dtype, device=linear.device
            )
        else:
            mean = torch.as_tensor(mean_concentration, dtype=self._linear.dtype)
            linear, kappa = self._linear.to(mean.device), self.kappa.to(mean.device)
        return -(linear + 2.0 * kappa * mean)

    def _chemical_delta(self, batch: Batch) -> torch.Tensor:
        """Return the VC-SGC term ``dn * (phi - reference + 2 * kappa * c_mid)`` per graph.

        Called after :meth:`_propose` has applied the trial transmutation, so
        the counted atom types are the trial state for active graphs.
        """
        if (
            self._proposal_old is None
            or self._proposal_new is None
            or self._active is None
        ):
            raise RuntimeError("VC-SGC constraint term requested without a proposal")
        dtype = batch.positions.dtype
        solute = self.concentration_species
        delta_n = (self._proposal_new == solute).to(dtype) - (
            self._proposal_old == solute
        ).to(dtype)
        n_atoms = batch.num_nodes_per_graph.to(dtype)
        # Midpoint of old and trial concentrations; exact for a quadratic weight.
        c_mid = (self._solute_counts(batch) - 0.5 * delta_n) / n_atoms
        linear = self._per_graph(self._linear, batch, "phi")
        kappa = self._per_graph(self.kappa, batch, "kappa")
        delta = delta_n * (linear + 2.0 * kappa * c_mid)
        return torch.where(self._active, delta, torch.zeros_like(delta))
