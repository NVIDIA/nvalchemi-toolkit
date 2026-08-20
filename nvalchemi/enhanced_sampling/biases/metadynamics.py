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
"""Well-tempered metadynamics over one or more collective variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling._adaptive import AdaptivePotentialMixin
from nvalchemi.enhanced_sampling._bias import ConservativeBias
from nvalchemi.enhanced_sampling.cv._periodic import periodic_difference

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nvalchemi.data import Batch
    from nvalchemi.enhanced_sampling._bias import BiasResult

__all__ = ["WellTemperedMetaDynamicsBias"]

_HISTORY_MODES = ("shared", "state", "walker")
_STORAGE_POLICIES = ("preallocated", "grow", "fifo")


class WellTemperedMetaDynamicsBias(AdaptivePotentialMixin, ConservativeBias):
    r"""Gaussian hills deposited along a CV, with well-tempered damping.

    .. math::

        V(s, t) = \sum_i h_i \exp\!\left(
            -\sum_d \frac{(s_d - c_{i,d})^2}{2\sigma_d^2}\right)

    Each deposition adds one hill per walker at that walker's current CV
    value.  In the well-tempered scheme the height decays where the bias has
    already accumulated,

    .. math:: h_t = h_0 \exp\!\left(-\frac{V(s_t)}{k_B T (\gamma - 1)}\right)

    which makes the sum converge instead of filling forever.  At convergence
    the free energy is recovered as
    :math:`F(s) = -\frac{\gamma}{\gamma - 1} V(s)` (Barducci, Bussi &
    Parrinello 2008), available from :meth:`free_energy`.

    Passing ``bias_factor=None`` gives standard metadynamics — the
    :math:`\gamma \to \infty` limit, where every hill has height ``h_0`` and
    :math:`F(s) = -V(s)`.

    Parameters
    ----------
    cv:
        Differentiable ``cv(batch) -> Tensor[B, D]``.
    height:
        Initial hill height ``h_0`` in eV.
    sigma:
        Hill width per CV component; scalar or shape ``[D]``.
    temperature:
        Simulation temperature in Kelvin, used for the well-tempered damping.
    bias_factor:
        ``gamma > 1``, or ``None`` for standard metadynamics.
    name:
        Unique bias identifier.
    update_frequency:
        Dynamics steps between depositions (the deposition pace).
    storage:
        ``"preallocated"``, ``"grow"``, or ``"fifo"`` — see Notes.
    max_hills:
        Capacity.  Required for ``"preallocated"`` and ``"fifo"``; the
        initial chunk for ``"grow"``.
    history:
        ``"shared"`` (every walker sees every hill), ``"state"`` (hills
        belong to the thermodynamic state that deposited them), or
        ``"walker"`` (each walker sees only its own).
    periods:
        Period per CV component, ``0`` for non-periodic.  Applied to the
        ``s - c`` difference so a hill near a branch cut still repels from
        both sides.
    ramp_depositions:
        Number of *deposition events* over which a freshly added hill ramps
        from zero to full height.  ``0`` activates immediately.  Counted in
        depositions rather than dynamics steps because ``energy()`` sees the
        hill table, not the step counter.
    compute_stress:
        Passed through to :class:`ConservativeBias`.

    Raises
    ------
    ValueError
        For an invalid storage policy, history mode, non-positive height or
        width, ``bias_factor <= 1``, or a missing capacity.

    Notes
    -----
    Storage policies
        ``preallocated`` keeps tensor shapes fixed for the whole run and
        **raises** when capacity is exhausted, rather than silently changing
        the physics.  It is the compile-stable choice.

        ``grow`` allocates in chunks of ``max_hills``; each growth changes
        the hill-tensor shape, which forces a recompile if ``energy()`` is
        compiled.

        ``fifo`` bounds memory by discarding the oldest hill.  This is
        **scientifically meaningful, not merely a cache eviction**: the
        accumulated bias is no longer the integral of everything deposited,
        so the well-tempered convergence argument no longer applies and
        :meth:`free_energy` is not a valid estimator.  Chosen deliberately
        for exploration, and the natural policy for the RMSD variant.

    Multi-walker history
        ``shared`` is the multiple-walker scheme: every walker feels every
        hill, so ``B`` walkers fill a basin roughly ``B`` times faster.
        ``state`` keeps a separate history per thermodynamic state, which is
        what a replica-exchange ladder needs.  ``walker`` runs ``B``
        independent metadynamics simulations in one batch.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.enhanced_sampling import (
    ...     WellTemperedMetaDynamicsBias, pair_distance,
    ... )
    >>> idx = torch.tensor([0, 5])
    >>> bias = WellTemperedMetaDynamicsBias(
    ...     cv=lambda b: pair_distance(b, idx),
    ...     height=0.01, sigma=0.1, temperature=300.0, bias_factor=10.0,
    ...     max_hills=1000,
    ... )
    >>> bias.hill_count.item()
    0
    """

    def __init__(
        self,
        cv: Callable[[Batch], Tensor],
        height: float,
        sigma: Tensor | float,
        temperature: float,
        *,
        bias_factor: float | None = None,
        name: str = "metadynamics",
        update_frequency: int = 500,
        storage: Literal["preallocated", "grow", "fifo"] = "preallocated",
        max_hills: int | None = None,
        history: Literal["shared", "state", "walker"] = "shared",
        periods: Tensor | None = None,
        ramp_depositions: int = 0,
        compute_stress: bool = True,
    ) -> None:
        super().__init__(name=name, compute_stress=compute_stress)

        if storage not in _STORAGE_POLICIES:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: storage must be one of "
                f"{list(_STORAGE_POLICIES)}, got {storage!r}."
            )
        if history not in _HISTORY_MODES:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: history must be one of "
                f"{list(_HISTORY_MODES)}, got {history!r}."
            )
        if height <= 0.0:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: height must be positive, got "
                f"{height}. A non-positive hill would attract the walker to "
                "where it has already been."
            )
        if bias_factor is not None and bias_factor <= 1.0:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: bias_factor must be greater "
                f"than 1, got {bias_factor}. gamma = 1 divides by zero in the "
                "well-tempered height; pass None for standard metadynamics."
            )
        if int(update_frequency) < 1:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: update_frequency must be at "
                f"least 1, got {update_frequency}."
            )
        if int(ramp_depositions) < 0:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: ramp_depositions must be "
                f"non-negative, got {ramp_depositions}."
            )
        if storage in ("preallocated", "fifo") and max_hills is None:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: storage={storage!r} needs an "
                "explicit max_hills — it is the whole point of the policy, "
                "either the ceiling that raises or the ring that overwrites."
            )
        capacity = int(max_hills) if max_hills is not None else 256
        if capacity < 1:
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: max_hills must be at least 1, "
                f"got {max_hills}."
            )

        sigma_t = torch.as_tensor(sigma, dtype=torch.get_default_dtype()).reshape(-1)
        if bool((sigma_t <= 0).any()):
            raise ValueError(
                f"WellTemperedMetaDynamicsBias: sigma must be positive, got "
                f"{sigma_t.tolist()}. A zero width is a delta function with no "
                "gradient anywhere but one point."
            )

        self.cv = cv
        self.storage = storage
        self.history = history
        self.update_frequency = int(update_frequency)
        self.ramp_depositions = int(ramp_depositions)
        self.height = float(height)
        self.temperature = float(temperature)
        self.bias_factor = None if bias_factor is None else float(bias_factor)
        self._capacity = capacity

        self.register_buffer("sigma", sigma_t)
        if periods is None:
            self.periods: Tensor | None = None
        else:
            self.register_buffer(
                "periods",
                torch.as_tensor(periods, dtype=torch.get_default_dtype()).reshape(-1),
            )
        self._allocate(capacity, dim=sigma_t.numel())

        # This bias reads thermodynamic_state_id only in "state" history, and
        # even then the hills follow the state rather than the energy varying
        # by state at fixed history — but the assignment does change which
        # hills a walker feels, so a temperature ladder would need the
        # combined acceptance rule that is not implemented.
        self.state_dependent_for_exchange = history == "state"

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _allocate(self, capacity: int, dim: int) -> None:
        """Create or replace the hill buffers at *capacity*.

        Parameters
        ----------
        capacity:
            Number of hill slots.
        dim:
            CV dimension.
        """
        dtype = self.sigma.dtype
        device = self.sigma.device
        self.register_buffer(
            "hill_centers", torch.zeros(capacity, dim, dtype=dtype, device=device)
        )
        self.register_buffer(
            "hill_heights", torch.zeros(capacity, dtype=dtype, device=device)
        )
        self.register_buffer(
            "hill_owner", torch.full((capacity,), -1, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "hill_step", torch.full((capacity,), -1, dtype=torch.long, device=device)
        )
        # hill_count saturates at capacity (how many slots are live);
        # hills_written keeps counting, which is what makes the FIFO ring
        # position unambiguous once it has wrapped.
        self.register_buffer(
            "hill_count", torch.zeros((), dtype=torch.long, device=device)
        )
        self.register_buffer(
            "hills_written", torch.zeros((), dtype=torch.long, device=device)
        )
        self.register_buffer(
            "deposits", torch.zeros((), dtype=torch.long, device=device)
        )

    def _grow(self) -> None:
        """Double-buffer the hill tensors by one more chunk."""
        extra = self._capacity
        device = self.sigma.device

        def _extend(buffer: Tensor, fill: float | int) -> Tensor:
            pad = torch.full(
                (extra, *buffer.shape[1:]), fill, dtype=buffer.dtype, device=device
            )
            return torch.cat([buffer, pad], dim=0)

        self.hill_centers = _extend(self.hill_centers, 0.0)
        self.hill_heights = _extend(self.hill_heights, 0.0)
        self.hill_owner = _extend(self.hill_owner, -1)
        self.hill_step = _extend(self.hill_step, -1)

    @property
    def capacity(self) -> int:
        """Return the current number of hill slots."""
        return int(self.hill_centers.shape[0])

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def _owner_key(self, current: Batch, n_graphs: int) -> Tensor:
        """Return the history key per graph, shape ``[B]``.

        Parameters
        ----------
        current:
            The live batch.
        n_graphs:
            Number of graphs.

        Returns
        -------
        Tensor
            Per-graph key matched against ``hill_owner``.  All ``-1`` under
            ``"shared"``, which the mask then ignores.
        """
        device = self.hill_owner.device
        if self.history == "state":
            ids = getattr(current, "thermodynamic_state_id", None)
        elif self.history == "walker":
            ids = getattr(current, "walker_id", None)
        else:
            return torch.full((n_graphs,), -1, dtype=torch.long, device=device)
        if ids is None:
            return torch.zeros(n_graphs, dtype=torch.long, device=device)
        return ids.reshape(-1).to(device=device, dtype=torch.long)

    def _hill_scale(self, step: Tensor) -> Tensor:
        """Return each hill's ramp factor, shape ``[capacity]``.

        A hill that switched on at full height would deliver a force
        discontinuity at the deposition event; ramping over
        ``ramp_depositions`` spreads that over a window instead.

        Parameters
        ----------
        step:
            Current deposition counter.

        Returns
        -------
        Tensor
            Factors in ``[0, 1]``.
        """
        if self.ramp_depositions <= 0:
            return torch.ones_like(self.hill_heights)
        age = (step - self.hill_step).to(self.hill_heights.dtype)
        return torch.clamp(age / float(self.ramp_depositions), min=0.0, max=1.0)

    def gaussian_sum(self, values: Tensor, owner_key: Tensor) -> Tensor:
        """Return the accumulated bias at *values*, shape ``[B]``.

        Contains no data-dependent Python branch, so it compiles with
        ``fullgraph=True``: the live-hill mask comes from comparing an
        ``arange`` against the count tensor rather than slicing by an int.

        Parameters
        ----------
        values:
            CV values, shape ``[B, D]``.
        owner_key:
            History key per graph, shape ``[B]``.

        Returns
        -------
        Tensor
            Bias energy per graph, shape ``[B]``.
        """
        slots = torch.arange(self.hill_centers.shape[0], device=values.device)
        live = slots < self.hill_count

        delta = periodic_difference(
            values.unsqueeze(1),  # [B, 1, D]
            self.hill_centers.unsqueeze(0),  # [1, H, D]
            self.periods,
        )  # [B, H, D]
        exponent = 0.5 * ((delta / self.sigma) ** 2).sum(dim=-1)  # [B, H]
        gaussians = torch.exp(-exponent)

        scale = self._hill_scale(self.deposits)  # [H]
        weights = self.hill_heights * scale  # [H]

        mask = live.unsqueeze(0)  # [1, H]
        if self.history != "shared":
            mask = mask & (self.hill_owner.unsqueeze(0) == owner_key.unsqueeze(1))
        return (gaussians * weights.unsqueeze(0) * mask).sum(dim=-1)

    def energy(self, current: Batch) -> Tensor:
        """Return the accumulated metadynamics bias, shape ``[B, 1]``.

        Parameters
        ----------
        current:
            Batch with strained positions supplied by
            :meth:`ConservativeBias.evaluate`.

        Returns
        -------
        Tensor
            Shape ``[B, 1]`` in eV.
        """
        values = self.cv(current)
        owner = self._owner_key(current, values.shape[0])
        return self.gaussian_sum(values, owner).unsqueeze(-1)

    # ------------------------------------------------------------------
    # Deposition
    # ------------------------------------------------------------------

    def _well_tempered_height(self, bias_at_cv: Tensor) -> Tensor:
        """Return the damped hill height per walker.

        Parameters
        ----------
        bias_at_cv:
            Accumulated bias at the deposition point, shape ``[B]``.

        Returns
        -------
        Tensor
            Heights, shape ``[B]``.
        """
        if self.bias_factor is None:
            return torch.full_like(bias_at_cv, self.height)
        damping = KB_EV * self.temperature * (self.bias_factor - 1.0)
        return self.height * torch.exp(-bias_at_cv / damping)

    def _next_slots(self, count: int) -> Tensor:
        """Return the slot indices *count* new hills will occupy.

        Parameters
        ----------
        count:
            Number of hills about to be deposited.

        Returns
        -------
        Tensor
            Slot indices, shape ``[count]``.

        Raises
        ------
        RuntimeError
            If ``preallocated`` capacity is exhausted.  Raising rather than
            evicting is the point of the policy: silently dropping hills
            would change the physics of a converging run without saying so.
        """
        written = int(self.hills_written)
        capacity = self.capacity

        if self.storage == "fifo":
            # Ring buffer: hill j always lands in slot j % capacity, so the
            # oldest is the one overwritten however many times it has wrapped.
            return (
                torch.arange(count, device=self.hill_heights.device) + written
            ) % capacity

        if written + count > capacity:
            if self.storage == "preallocated":
                raise RuntimeError(
                    f"WellTemperedMetaDynamicsBias {self.name!r}: hill storage "
                    f"is full ({capacity} hills) and storage='preallocated'. "
                    f"Raise max_hills, lengthen update_frequency, or switch to "
                    "storage='grow' (recompiles when it resizes) or "
                    "storage='fifo' (bounded memory, but discards the oldest "
                    "hills and so is no longer a converging well-tempered run)."
                )
            while written + count > self.capacity:
                self._grow()

        return torch.arange(count, device=self.hill_heights.device) + written

    def update(self, frames: Batch, result: BiasResult) -> None:
        """Deposit one hill per walker at its current CV value.

        Parameters
        ----------
        frames:
            Post-step frame captured by the runner.
        result:
            The bias's own result from the preceding force evaluation.
        """
        with torch.no_grad():
            values = self.cv(frames).detach()  # [B, D]
            owner = self._owner_key(frames, values.shape[0])
            bias_at_cv = self.gaussian_sum(values, owner)
            heights = self._well_tempered_height(bias_at_cv)

            count = values.shape[0]
            slots = self._next_slots(count)

            self.hill_centers[slots] = values.to(self.hill_centers.dtype)
            self.hill_heights[slots] = heights.to(self.hill_heights.dtype)
            self.hill_owner[slots] = owner
            self.hill_step[slots] = self.deposits

            self.deposits += 1
            self.hills_written += count
            self.hill_count = torch.clamp(self.hills_written, max=self.capacity)
        self.bump_state_version()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def free_energy(self, values: Tensor, owner_key: Tensor | None = None) -> Tensor:
        """Return the free-energy estimate at *values*.

        ``F(s) = -(gamma / (gamma - 1)) V(s)`` for well-tempered, or
        ``F(s) = -V(s)`` for standard metadynamics.

        Parameters
        ----------
        values:
            CV values, shape ``[B, D]``.
        owner_key:
            History key per graph.  Defaults to the shared history.

        Returns
        -------
        Tensor
            Free energy in eV, shape ``[B]``, up to an additive constant.

        Raises
        ------
        RuntimeError
            Under ``storage="fifo"``, where discarded hills mean the
            accumulated bias is no longer the integral of everything
            deposited and the well-tempered relation does not hold.
        """
        if self.storage == "fifo":
            raise RuntimeError(
                f"WellTemperedMetaDynamicsBias {self.name!r}: free_energy() is "
                "not valid under storage='fifo'. Discarding the oldest hills "
                "means the accumulated bias is no longer the integral of "
                "everything deposited, so the well-tempered relation between "
                "bias and free energy does not hold. Use 'preallocated' or "
                "'grow' for a run you intend to reconstruct a profile from."
            )
        if owner_key is None:
            owner_key = torch.full(
                (values.shape[0],),
                -1 if self.history == "shared" else 0,
                dtype=torch.long,
                device=values.device,
            )
        bias = self.gaussian_sum(values, owner_key)
        if self.bias_factor is None:
            return -bias
        return -bias * (self.bias_factor / (self.bias_factor - 1.0))

    def load_state_dict(
        self, state: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Restore state, resizing the hill buffers to match the checkpoint.

        ``storage="grow"`` means the saved capacity is whatever the run had
        reached, which is almost never the constructor's initial chunk —
        ``nn.Module.load_state_dict`` would reject the size mismatch.

        Parameters
        ----------
        state:
            Mapping from :meth:`state_dict`.
        *args, **kwargs:
            Forwarded up the MRO.

        Returns
        -------
        Any
            Whatever the next ``load_state_dict`` returns.
        """
        centers = state.get("hill_centers")
        if centers is not None and tuple(centers.shape) != tuple(
            self.hill_centers.shape
        ):
            self._allocate(int(centers.shape[0]), int(centers.shape[1]))
        return super().load_state_dict(state, *args, **kwargs)

    def __repr__(self) -> str:
        """Return a concise description of the bias."""
        gamma = "None" if self.bias_factor is None else f"{self.bias_factor:g}"
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"hills={int(self.hill_count)}/{self.capacity}, "
            f"storage={self.storage!r}, history={self.history!r}, "
            f"bias_factor={gamma})"
        )
