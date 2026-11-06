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
"""The deposit-history ring buffer shared by the depositing biases.

Metadynamics stores hills; RMSD metadynamics stores reference structures.
What they store differs — a CV centre and a height against a centred
coordinate set — but *how* they store it does not: a fixed-width table of
slots, a per-slot owner key selecting whose history a slot belongs to, a
monotonic written counter, and one of three policies for what happens when
the table fills.

Keeping one implementation here is not only tidiness.  The two copies had
already drifted: the guard that refuses a ``"fifo"`` ring too small to hold a
single deposition existed on the hill side and not on the reference side, so
``max_references`` below the walker count silently discarded references from
the deposition being written rather than the oldest one.

Buffer names stay with the subclass.  ``hill_owner`` and ``reference_owner``
are the names in every checkpoint written so far, and they read better in
their own files than a shared ``owner`` would; the mixin is told which
attributes to use rather than imposing its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from nvalchemi.data import Batch

__all__ = ["HISTORY_MODES", "STORAGE_POLICIES", "DepositHistoryMixin"]

#: Whose history a deposit belongs to.
HISTORY_MODES = ("shared", "state", "walker")

#: What happens when the table fills.
STORAGE_POLICIES = ("preallocated", "grow", "fifo")

#: Batch field carrying the owner key, per history mode.
_OWNER_FIELDS = {"state": "thermodynamic_state_id", "walker": "walker_id"}


class DepositHistoryMixin:
    """Slot allocation and owner keys for a bias that deposits into a table.

    A subclass supplies the names of its own buffers and the noun it calls a
    deposit, then gets :attr:`capacity`, :meth:`_grow`, :meth:`_next_slots`
    and :meth:`_owner_key` for free.  It is expected to set ``self.storage``
    and ``self.history`` in ``__init__``, having validated them with
    :meth:`validate_history_options`.

    Attributes
    ----------
    _deposit_noun:
        Singular noun for one deposit — ``"hill"``, ``"reference"``.  Used in
        the error messages, so it should read naturally in a sentence.
    _capacity_option:
        Constructor keyword that sets the table width — ``"max_hills"``.
    _full_advice:
        What to tell the caller when ``"preallocated"`` capacity runs out.
        Specific to the bias, since the policies differ in what they cost.
    _deposit_buffers:
        ``(attribute, fill)`` for each buffer :meth:`_grow` extends.  The
        first one's device is the device the slot indices are built on.
    _written_attr:
        Attribute holding the monotonic count of deposits ever written —
        ``"hills_written"``.  It keeps counting past the capacity, which is
        what makes the FIFO ring position unambiguous once it has wrapped.
    _owner_attr:
        Attribute holding the per-slot owner key — ``"hill_owner"``.
    """

    _deposit_noun: str = "deposit"
    _capacity_option: str = "capacity"
    _full_advice: str = ""
    _deposit_buffers: tuple[tuple[str, float | int], ...] = ()
    _written_attr: str = ""
    _owner_attr: str = ""

    storage: str
    history: str
    name: str

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def validate_history_options(cls, storage: str, history: str) -> None:
        """Refuse a storage policy or history mode that does not exist.

        Called from ``__init__`` before anything is allocated, so a typo is a
        construction error rather than a surprise at the first deposition.

        Parameters
        ----------
        storage:
            Requested storage policy.
        history:
            Requested history mode.

        Raises
        ------
        ValueError
            If either is not one of the known values.
        """
        if storage not in STORAGE_POLICIES:
            raise ValueError(
                f"{cls.__name__}: storage must be one of "
                f"{list(STORAGE_POLICIES)}, got {storage!r}."
            )
        if history not in HISTORY_MODES:
            raise ValueError(
                f"{cls.__name__}: history must be one of "
                f"{list(HISTORY_MODES)}, got {history!r}."
            )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Return the current number of slots."""
        return int(getattr(self, self._deposit_buffers[0][0]).shape[0])

    def _grow(self) -> None:
        """Extend every deposit buffer by one more chunk."""
        extra = self._capacity  # type: ignore[attr-defined]
        device = getattr(self, self._deposit_buffers[0][0]).device
        for attribute, fill in self._deposit_buffers:
            buffer = getattr(self, attribute)
            pad = torch.full(
                (extra, *buffer.shape[1:]), fill, dtype=buffer.dtype, device=device
            )
            setattr(self, attribute, torch.cat([buffer, pad], dim=0))

    def _next_slots(self, count: int) -> Tensor:
        """Return the slot indices *count* new deposits will occupy.

        Parameters
        ----------
        count:
            Number of deposits about to be written, one per walker.

        Returns
        -------
        Tensor
            Slot indices, shape ``[count]``.

        Raises
        ------
        RuntimeError
            If ``"preallocated"`` capacity is exhausted.  Raising rather than
            evicting is the point of the policy: silently dropping deposits
            would change the physics of a converging run without saying so.
            Also if a ``"fifo"`` ring is smaller than one deposition, where
            the overwriting would be within the deposition rather than of the
            oldest entry.
        """
        noun = self._deposit_noun
        written = int(getattr(self, self._written_attr))
        device = getattr(self, self._deposit_buffers[0][0]).device
        capacity = self.capacity

        if self.storage == "fifo":
            if count > capacity:
                raise RuntimeError(
                    f"{type(self).__name__} {self.name!r}: one deposition "
                    f"writes {count} {noun}(s) — one per walker — but "
                    f"{self._capacity_option} is {capacity}. A ring that "
                    f"cannot hold a single deposition does not discard the "
                    f"*oldest* {noun}, which is what storage='fifo' means: it "
                    f"would silently drop {noun}s deposited at the same "
                    f"instant, keeping whichever {capacity} of the {count} "
                    f"walkers happened to be written last. Raise "
                    f"{self._capacity_option} to at least the walker count."
                )
            # Ring buffer: deposit j always lands in slot j % capacity, so the
            # oldest is the one overwritten however many times it has wrapped.
            # count <= capacity, checked above, so the slots within one
            # deposition are distinct and none overwrites another.
            return (torch.arange(count, device=device) + written) % capacity

        if written + count > capacity:
            if self.storage == "preallocated":
                raise RuntimeError(
                    f"{type(self).__name__} {self.name!r}: {noun} storage is "
                    f"full ({capacity} {noun}s) and storage='preallocated'. "
                    f"{self._full_advice}"
                )
            while written + count > self.capacity:
                self._grow()

        return torch.arange(count, device=device) + written

    # ------------------------------------------------------------------
    # Ownership
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
            Per-graph key matched against the per-slot owner buffer.  All
            ``-1`` under ``"shared"``, which the mask then ignores.

        Raises
        ------
        ValueError
            If the batch does not carry the field the history mode needs, or
            carries one of the wrong length.
        """
        noun = self._deposit_noun
        device = getattr(self, self._owner_attr).device
        field = _OWNER_FIELDS.get(self.history)
        if field is None:
            return torch.full((n_graphs,), -1, dtype=torch.long, device=device)

        ids = getattr(current, field, None)
        if ids is None:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: history="
                f"{self.history!r} needs batch.{field}, which this batch "
                f"does not carry. Falling back to a single owner would put "
                f"every {noun} under one key and silently collapse the "
                f"per-{field} histories into one shared history — the "
                f"opposite of what history={self.history!r} asks for. "
                "EnhancedSampling stamps this field on every step; a bias "
                "driven directly must set it, or use history='shared' if "
                "one history really is intended."
            )
        if ids.numel() != n_graphs:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: batch.{field} has "
                f"{ids.numel()} entries but the batch has {n_graphs} "
                f"graph(s). A shorter tensor broadcasts across graphs, which "
                f"would file every {noun} under one walker's key without "
                "raising."
            )
        return ids.reshape(-1).to(device=device, dtype=torch.long)
