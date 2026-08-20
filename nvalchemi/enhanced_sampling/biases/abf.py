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
"""Adaptive biasing force over a pair-distance collective variable."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling._adaptive import AdaptivePotentialMixin
from nvalchemi.enhanced_sampling._bias import BiasResult
from nvalchemi.enhanced_sampling.cv.pair_distance import pair_displacement

if TYPE_CHECKING:
    from nvalchemi.data import Batch

__all__ = ["AdaptiveBiasingForce"]


class AdaptiveBiasingForce(AdaptivePotentialMixin, nn.Module):
    r"""Estimate and cancel the mean force along a pair distance.

    Metadynamics fills a basin with hills; ABF instead measures the mean
    force :math:`\partial A/\partial \xi` in each bin of the collective
    variable and applies its negative, so the residual force along the CV
    averages to zero and the walker diffuses freely.  The estimate *is* the
    free-energy gradient, so no reweighting is needed at the end —
    :meth:`free_energy` integrates it directly.

    For the pair distance :math:`r = |\mathbf{r}_j - \mathbf{r}_i|` the
    estimator is

    .. math::

        \frac{\partial A}{\partial r} = \left\langle
            -\frac{(\mathbf{F}_j - \mathbf{F}_i)\cdot\hat{\mathbf{u}}}{2}
            \;-\; \frac{2 k_B T}{r} \right\rangle_r

    with :math:`\hat{\mathbf{u}} = (\mathbf{r}_j - \mathbf{r}_i)/r`.  The
    second term is the **metric correction**, and it is not optional: see
    Notes.

    Parameters
    ----------
    atom_indices:
        The pair ``(i, j)``, shape ``[2]`` for the same pair in every graph
        or ``[B, 2]`` for one per graph.  Indices are local to each graph.
    temperature:
        Simulation temperature in Kelvin.  Enters the metric correction, so
        it must match the thermostat; a mismatch biases the estimate.
    cv_range:
        ``(r_min, r_max)`` in angstrom, the binned interval.  Outside it no
        force is applied and no sample is recorded.
    name:
        Unique bias identifier.
    n_bins:
        Number of uniform bins across ``cv_range``.
    min_samples:
        Samples a bin needs before *any* bias force is applied from it.  An
        estimate from a handful of samples is noise, and applying it would
        drive the walker on the strength of that noise.
    full_samples:
        Samples at which the applied fraction reaches 1.  Defaults to
        ``2 * min_samples``.  Between the two the force ramps linearly;
        equal to ``min_samples`` it is a step, applying nothing until the
        threshold and full force at it.
    max_force:
        Optional cap on ``|dA/dr|`` in eV/A.  A bin that has been visited
        once at a bad geometry can hold a large estimate; the cap bounds
        what that can do to the trajectory.
    update_frequency:
        Steps between :meth:`update` calls.  ``1`` (every step) is the usual
        choice — ABF wants every uncorrelated sample it can get.

    Raises
    ------
    ValueError
        For a malformed ``atom_indices``, a non-positive temperature, an
        empty or inverted ``cv_range``, fewer than one bin, a negative
        ``min_samples``, ``full_samples < min_samples``, or a non-positive
        ``max_force``.

    Notes
    -----
    The metric correction is not optional
        Projecting Cartesian forces onto :math:`\nabla\xi` and averaging
        gives the mean force in the *constrained* ensemble, not the gradient
        of the free energy of the unconstrained one.  The two differ by the
        Jacobian of the coordinate change — for a distance in three
        dimensions the shell volume grows as :math:`r^2`, contributing
        :math:`-2k_B T/r`.

        Omitting it does not produce noise, it produces a smoothly wrong
        answer: two non-interacting particles would be reported as having a
        flat PMF when the true one is :math:`-2k_B T\ln r`.  That is why
        this class takes an atom **pair** rather than a general ``cv``
        callable — the correction is specific to this coordinate, and
        accepting an arbitrary CV would mean applying a distance-shaped
        correction to something that is not a distance.

    Force-only, and therefore excluded from replica exchange
        :meth:`evaluate` returns forces with ``energy=None``.  There is no
        potential to report: the applied force is not the gradient of any
        function the bias holds, which is exactly what makes ABF
        non-conservative.  ``supplies_exchange_energy`` is ``False``, and
        :class:`~nvalchemi.enhanced_sampling.ReplicaExchange` refuses such a
        bias rather than evaluating an acceptance rule that needs a
        cross-state bias energy.

    Observation ordering
        ``observation_stage`` is ``AFTER_COMPUTE``, where ``batch.forces``
        still holds the **unbiased** physical force.  Observing after the
        bias is applied would feed the estimator its own output, and it
        would converge to whatever it had already decided.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.enhanced_sampling import AdaptiveBiasingForce
    >>> abf = AdaptiveBiasingForce(
    ...     atom_indices=torch.tensor([0, 1]),
    ...     temperature=300.0,
    ...     cv_range=(2.0, 6.0),
    ...     n_bins=40,
    ... )
    >>> int(abf.bin_counts.sum())
    0
    """

    #: The acceptance rule needs a cross-state bias energy; this has none.
    supplies_exchange_energy: bool = False

    def __init__(
        self,
        atom_indices: Tensor,
        temperature: float,
        cv_range: tuple[float, float],
        *,
        name: str = "abf",
        n_bins: int = 100,
        min_samples: int = 200,
        full_samples: int | None = None,
        max_force: float | None = None,
        update_frequency: int = 1,
    ) -> None:
        super().__init__()

        indices = torch.as_tensor(atom_indices, dtype=torch.long)
        if indices.dim() == 1:
            if indices.numel() != 2:
                raise ValueError(
                    f"AdaptiveBiasingForce: atom_indices must name exactly two "
                    f"atoms, got {indices.numel()}."
                )
        elif indices.dim() != 2 or indices.shape[-1] != 2:
            raise ValueError(
                f"AdaptiveBiasingForce: atom_indices must have shape [2] or "
                f"[B, 2], got {tuple(indices.shape)}."
            )
        if bool((indices < 0).any()):
            raise ValueError(
                f"AdaptiveBiasingForce: atom_indices must be non-negative "
                f"per-graph indices, got {indices.tolist()}."
            )
        if bool((indices[..., 0] == indices[..., 1]).any()):
            raise ValueError(
                f"AdaptiveBiasingForce: atom_indices names the same atom twice "
                f"({indices.tolist()}); the distance would be identically zero."
            )

        if temperature <= 0.0:
            raise ValueError(
                f"AdaptiveBiasingForce: temperature must be positive, got "
                f"{temperature}. It scales the metric correction, so it must "
                "match the thermostat."
            )

        low, high = (float(v) for v in cv_range)
        if not high > low:
            raise ValueError(
                f"AdaptiveBiasingForce: cv_range must be increasing, got "
                f"({low}, {high})."
            )
        if low < 0.0:
            raise ValueError(
                f"AdaptiveBiasingForce: cv_range lower bound must be "
                f"non-negative for a distance, got {low}."
            )
        if int(n_bins) < 1:
            raise ValueError(
                f"AdaptiveBiasingForce: n_bins must be at least 1, got {n_bins}."
            )
        if int(min_samples) < 0:
            raise ValueError(
                f"AdaptiveBiasingForce: min_samples must be non-negative, got "
                f"{min_samples}."
            )
        ramp_end = 2 * int(min_samples) if full_samples is None else int(full_samples)
        if ramp_end < int(min_samples):
            raise ValueError(
                f"AdaptiveBiasingForce: full_samples ({ramp_end}) must be at "
                f"least min_samples ({int(min_samples)}) — the force ramps up "
                "between the two."
            )
        if max_force is not None and max_force <= 0.0:
            raise ValueError(
                f"AdaptiveBiasingForce: max_force must be positive, got {max_force}."
            )
        if int(update_frequency) < 1:
            raise ValueError(
                f"AdaptiveBiasingForce: update_frequency must be at least 1, "
                f"got {update_frequency}."
            )

        self.name = name
        self.temperature = float(temperature)
        self.cv_range = (low, high)
        self.n_bins = int(n_bins)
        self.min_samples = int(min_samples)
        self.full_samples = ramp_end
        self.max_force = None if max_force is None else float(max_force)
        self.update_frequency = int(update_frequency)
        self.observation_stage = DynamicsStage.AFTER_COMPUTE

        dtype = torch.get_default_dtype()
        self.register_buffer("atom_indices", indices)
        self.register_buffer("bin_counts", torch.zeros(self.n_bins, dtype=torch.long))
        self.register_buffer("force_sum", torch.zeros(self.n_bins, dtype=dtype))

    # ------------------------------------------------------------------
    # Geometry and binning
    # ------------------------------------------------------------------

    @property
    def bin_width(self) -> float:
        """Return the width of one CV bin, in angstrom."""
        low, high = self.cv_range
        return (high - low) / self.n_bins

    @property
    def bin_centers(self) -> Tensor:
        """Return the CV value at the middle of each bin, shape ``[n_bins]``."""
        low, _ = self.cv_range
        offsets = torch.arange(
            self.n_bins, dtype=self.force_sum.dtype, device=self.force_sum.device
        )
        return low + (offsets + 0.5) * self.bin_width

    def bin_index(self, values: Tensor) -> Tensor:
        """Return the bin each CV value falls in, shape ``[...]``.

        Values outside ``cv_range`` are reported as ``-1``; they contribute
        no sample and receive no force, so they belong to no bin.

        Parameters
        ----------
        values:
            CV values in angstrom.

        Returns
        -------
        Tensor
            Integer bin indices, or ``-1`` for out-of-range values.
        """
        index, in_range = self._bin_of(torch.as_tensor(values))
        return torch.where(in_range, index, torch.full_like(index, -1))

    def _align_device(self, reference: Tensor) -> None:
        """Move this bias's buffers to *reference*'s device if they differ.

        Parameters
        ----------
        reference:
            Any tensor from the live batch; its device is the target.
        """
        if self.force_sum.device != reference.device:
            self.to(reference.device)

    def _geometry(self, current: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(distance, unit_vector, global_i, global_j)``.

        Parameters
        ----------
        current:
            The live batch.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Distances ``[B]``, unit displacements ``[B, 3]``, and the two
            global row indices ``[B]`` the bias force is written to.
        """
        # pair_displacement validates atom_indices and applies the minimum
        # image convention, so the resolution repeated below is known good.
        delta = pair_displacement(current, self.atom_indices)  # [B, 3]
        distance = torch.linalg.vector_norm(delta, dim=-1)  # [B]
        unit = delta / distance.clamp(min=torch.finfo(delta.dtype).tiny).unsqueeze(-1)

        offsets = current.batch_ptr[:-1]  # [B]
        indices = self.atom_indices.to(offsets.device)
        if indices.dim() == 1:
            indices = indices.unsqueeze(0).expand(offsets.numel(), 2)
        return distance, unit, offsets + indices[:, 0], offsets + indices[:, 1]

    def _bin_of(self, distance: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(bin_index, in_range)`` for CV values *distance*.

        Parameters
        ----------
        distance:
            CV values, shape ``[B]``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bin indices clamped into range, and a boolean mask marking which
            samples actually fell inside ``cv_range``.
        """
        low, high = self.cv_range
        raw = ((distance - low) / self.bin_width).floor().to(torch.long)
        in_range = (distance >= low) & (distance < high)
        return raw.clamp(0, self.n_bins - 1), in_range

    # ------------------------------------------------------------------
    # Estimate
    # ------------------------------------------------------------------

    def mean_force(self) -> Tensor:
        """Return the per-bin mean-force estimate, shape ``[n_bins]``.

        Returns
        -------
        Tensor
            ``dA/dr`` in eV/A per bin.  Unvisited bins are ``nan`` rather
            than zero: a bin with no samples has no estimate, and zero is a
            perfectly plausible mean force that would hide that.
        """
        counts = self.bin_counts.to(self.force_sum.dtype)
        estimate = self.force_sum / counts
        return torch.where(self.bin_counts > 0, estimate, torch.nan)

    def _applied_gradient(self) -> Tensor:
        """Return the ramped, capped ``dA/dr`` actually applied, ``[n_bins]``.

        Returns
        -------
        Tensor
            Zero wherever no force should be applied yet, so this is safe to
            index with any bin.
        """
        counts = self.bin_counts.to(self.force_sum.dtype)
        estimate = torch.where(
            self.bin_counts > 0, self.force_sum / counts.clamp(min=1.0), 0.0
        )
        if self.max_force is not None:
            estimate = estimate.clamp(-self.max_force, self.max_force)
        # Reuses ramp_fraction rather than repeating it: two copies of the
        # same schedule are two places for the endpoint to disagree.
        return estimate * self.ramp_fraction()

    def ramp_fraction(self) -> Tensor:
        """Return the applied fraction per bin, shape ``[n_bins]``.

        Zero at or below ``min_samples``, one at or above ``full_samples``,
        linear between.  Ramping rather than switching on at the threshold
        avoids a force discontinuity of exactly the size the threshold exists
        to prevent.

        ``full_samples == min_samples`` is a step: nothing until the
        threshold, full force at it.  That is a legitimate choice — it is the
        classic hard-threshold form, and it is what ``min_samples=0`` gives
        by default — so it is handled rather than rejected.  It needs its own
        branch because the linear form would divide by a zero span, and
        clamping that span to 1 would delay full force by one sample.

        A bin with no samples has no estimate, so it reports zero whatever
        the thresholds are.

        Returns
        -------
        Tensor
            Fractions in ``[0, 1]``.
        """
        counts = self.bin_counts.to(self.force_sum.dtype)
        span = self.full_samples - self.min_samples
        if span <= 0:
            fraction = (self.bin_counts >= self.full_samples).to(counts.dtype)
        else:
            fraction = ((counts - self.min_samples) / span).clamp(0.0, 1.0)
        return torch.where(self.bin_counts > 0, fraction, torch.zeros_like(fraction))

    # ------------------------------------------------------------------
    # BiasPotential
    # ------------------------------------------------------------------

    def evaluate(self, current: Batch) -> BiasResult:
        """Return the bias force, with no energy.

        Read-only: the estimate is applied but never updated here.

        Parameters
        ----------
        current:
            The live batch.

        Returns
        -------
        BiasResult
            ``forces`` only, plus per-walker diagnostics: the CV value, the
            ``applied_gradient`` actually used (ramped and capped, so not the
            same as :meth:`mean_force`), the bin's sample count, its ramp
            fraction, and whether the walker was inside ``cv_range``.
            ``energy`` is ``None`` because there is none to report.
        """
        self._align_device(current.positions)

        with torch.no_grad():
            distance, unit, global_i, global_j = self._geometry(current)
            bins, in_range = self._bin_of(distance)

            gradient = self._applied_gradient()[bins]  # [B]
            gradient = torch.where(in_range, gradient, torch.zeros_like(gradient))

            # V_bias = -A, so F_bias = +grad A = (dA/dr) grad r, and
            # grad_i r = -u while grad_j r = +u.
            contribution = gradient.unsqueeze(-1) * unit  # [B, 3]
            forces = torch.zeros_like(current.positions)
            forces.index_add_(0, global_j, contribution)
            forces.index_add_(0, global_i, -contribution)

            # Named applied_gradient, not mean_force: this is the ramped and
            # capped value actually used, which is not what mean_force()
            # returns. Reusing that name would invite reading a
            # threshold-suppressed zero as a measured zero mean force.
            observables = {
                "cv": distance.unsqueeze(-1),
                "applied_gradient": gradient.unsqueeze(-1),
                "samples": self.bin_counts[bins].unsqueeze(-1),
                "ramp": self.ramp_fraction()[bins].unsqueeze(-1),
                "in_range": in_range.to(forces.dtype).unsqueeze(-1),
            }

        return BiasResult(forces=forces, observables=observables)

    def update(self, frames: Batch, result: BiasResult) -> None:
        """Accumulate one mean-force sample per walker.

        Parameters
        ----------
        frames:
            The ``AFTER_COMPUTE`` capture, whose ``forces`` are the unbiased
            physical forces.
        result:
            This bias's own preceding result; unused, since the estimator
            must not see its own output.

        Raises
        ------
        ValueError
            If the captured frame carries no forces to project.
        """
        physical = getattr(frames, "forces", None)
        if physical is None:
            raise ValueError(
                f"AdaptiveBiasingForce {self.name!r}: the observed frame has no "
                "forces, so there is no mean force to sample. This bias "
                "observes at AFTER_COMPUTE, where the batch must already carry "
                "the physical forces from the model."
            )

        with torch.no_grad():
            self._align_device(physical)
            distance, unit, global_i, global_j = self._geometry(frames)
            bins, in_range = self._bin_of(distance)

            # Projected force along the CV, then the Jacobian term. See the
            # class Notes for why the second is not optional.
            projected = (
                -((physical[global_j] - physical[global_i]) * unit).sum(dim=-1) / 2.0
            )
            metric = 2.0 * KB_EV * self.temperature / distance
            sample = (projected - metric).to(self.force_sum.dtype)

            keep = in_range & torch.isfinite(sample)
            if not bool(keep.any()):
                return

            bins = bins[keep]
            self.force_sum.index_add_(0, bins, sample[keep])
            self.bin_counts.index_add_(0, bins, torch.ones_like(bins))

            # Only bump when the applied force actually changed. A bin still
            # below its threshold contributes nothing, so re-priming forces
            # over it would be pure cost — this is the case
            # AdaptivePotentialMixin.bump_state_version documents. Asking
            # ramp_fraction rather than re-deriving the threshold keeps the
            # two from disagreeing at the endpoint.
            changed = bool((self.ramp_fraction()[bins] > 0.0).any())

        if changed:
            self.bump_state_version()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def config_fingerprint(self) -> dict[str, Any]:
        """Return the settings the accumulated histogram is only valid under.

        ``cv_range`` and ``n_bins`` decide what each bin *means*; restoring a
        histogram under a different pair silently relabels every bin.
        ``temperature`` scales the metric correction already folded into
        ``force_sum``, so samples taken at 300 K cannot be extended at 900 K.
        ``atom_indices`` names the coordinate itself.  ``min_samples`` /
        ``full_samples`` / ``max_force`` change what the same counts apply.

        Returns
        -------
        dict[str, Any]
            The checked configuration.
        """
        return {
            "atom_indices": self.atom_indices.reshape(-1).tolist(),
            "cv_range": list(self.cv_range),
            "n_bins": self.n_bins,
            "temperature": self.temperature,
            "min_samples": self.min_samples,
            "full_samples": self.full_samples,
            "max_force": self.max_force,
        }

    def free_energy(self) -> Tensor:
        """Return the PMF at each bin center, shape ``[n_bins]``.

        Integrates the mean-force estimate by the trapezoid rule, shifted so
        the minimum over sampled bins is zero.  This is the payoff of the
        method: unlike metadynamics there is nothing to deconvolve, because
        the accumulated quantity already *is* the free-energy gradient.

        Returns
        -------
        Tensor
            Free energy in eV.  Bins never visited are ``nan``.

        Raises
        ------
        RuntimeError
            If no bin has been sampled, or if an unvisited bin sits between
            two visited ones.  Integration carries the profile across the
            gap, so a hole in the middle would silently contaminate every
            value beyond it.
        """
        visited = self.bin_counts > 0
        if not bool(visited.any()):
            raise RuntimeError(
                f"AdaptiveBiasingForce {self.name!r}: no bin has been sampled, "
                "so there is no free energy to report. Check that cv_range "
                "covers the CV values the run actually visits."
            )

        indices = visited.nonzero(as_tuple=False).squeeze(-1)
        first, last = int(indices[0]), int(indices[-1])
        if not bool(visited[first : last + 1].all()):
            missing = (~visited[first : last + 1]).nonzero(as_tuple=False).squeeze(
                -1
            ) + first
            raise RuntimeError(
                f"AdaptiveBiasingForce {self.name!r}: bin(s) "
                f"{missing.tolist()} were never visited but lie between bins "
                f"{first} and {last} that were. Integrating the mean force "
                "carries the profile across the gap, so every value beyond it "
                "would be wrong by an unknown constant. Run longer, widen the "
                "bins, or narrow cv_range."
            )

        gradient = self.force_sum[first : last + 1] / self.bin_counts[
            first : last + 1
        ].to(self.force_sum.dtype)

        profile = torch.full_like(self.force_sum, torch.nan)
        if gradient.numel() == 1:
            profile[first] = 0.0
            return profile

        midpoints = 0.5 * (gradient[:-1] + gradient[1:]) * self.bin_width
        integrated = torch.cat(
            [
                torch.zeros(1, dtype=gradient.dtype, device=gradient.device),
                torch.cumsum(midpoints, dim=0),
            ]
        )
        profile[first : last + 1] = integrated - integrated.min()
        return profile

    def __repr__(self) -> str:
        """Return a concise description of the bias."""
        low, high = self.cv_range
        sampled = int((self.bin_counts > 0).sum())
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"cv_range=({low:g}, {high:g}), n_bins={self.n_bins}, "
            f"sampled_bins={sampled}/{self.n_bins}, "
            f"samples={int(self.bin_counts.sum())})"
        )
