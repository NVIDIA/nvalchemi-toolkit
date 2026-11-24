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
"""A block scheduler for alternating MC and MD on one Toolkit batch."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from nvalchemi.data import Batch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.mc.base import BaseMonteCarlo

__all__ = ["HybridMCMD"]


class HybridMCMD:
    """Alternate batched MC and MD blocks without a host-side state handoff.

    MC and MD may share one model object or use two. Two models let each phase
    run the inference path that suits it -- e.g. an unmerged UMA for SGC trials,
    whose composition changes every step, and a MoLE-merged UMA for MD, whose
    composition is fixed within a block (``before_md_block`` re-merges it). The
    two must represent the same potential (same checkpoint and task, differing
    only in how it is evaluated): the chain samples one ensemble only if MC
    and MD see one energy surface. With two models every MC block re-evaluates
    its baseline energy with the MC model (one call per block) instead of
    adopting MD's, so acceptance ratios never mix the two evaluation paths.

    With ``mc_energy_only=True`` the shared model's ``active_outputs`` is
    narrowed to ``{"energy"}`` for each MC block and restored for MD, so a
    wrapper that honours ``active_outputs`` (``UMAWrapper`` does) skips the
    forces/stress backward on every MC trial. Each MC block then starts by
    re-evaluating its baseline energy under the same narrowed outputs
    (:meth:`~nvalchemi.mc.base.BaseMonteCarlo.refresh_energy`), rather than
    adopting MD's, so every acceptance ratio compares energies from one
    evaluation path -- one extra model call per block.

    ``md_steps=0`` gives MC-only blocks (e.g. rigid-lattice SGC); with an
    ``after_md_block`` hook (e.g. a volume relaxation) the hook still runs
    between MC blocks on a fresh force/stress evaluation.
    """

    def __init__(
        self,
        mc: BaseMonteCarlo,
        md: BaseDynamics,
        mc_steps: int,
        md_steps: int,
        mc_energy_only: bool = False,
        before_md_block: Callable[[Batch], None] | None = None,
        after_md_block: Callable[[Batch], None] | None = None,
    ) -> None:
        """Initialize the hybrid scheduler.

        Parameters
        ----------
        mc
            A GPU-resident MC sampler.
        md
            A Toolkit molecular-dynamics integrator using the same model.
        mc_steps
            MC proposals per graph in each hybrid block.
        md_steps
            MD integration steps in each hybrid block.
        mc_energy_only
            Evaluate energies only (no forces/stress) during MC blocks.
        before_md_block
            Called with the batch before each MD block's first force evaluation
            (including the initial one), after MC has updated the species. Use
            it to prepare a separate MD model for the current composition.
        after_md_block
            Called with the batch after each MD block (and after the initial
            force evaluation), before MC adopts ``batch.energy``. It must leave
            ``batch.energy`` evaluated at the geometry it leaves behind, e.g. a
            cell relaxation that re-evaluates the model.
        """
        if mc_steps < 0 or md_steps < 0:
            raise ValueError("mc_steps and md_steps must both be non-negative")
        if mc_energy_only and "energy" not in mc.model.model_config.outputs:
            raise ValueError("mc_energy_only requires a model that outputs energy")
        self.mc = mc
        self.md = md
        self.mc_steps = mc_steps
        self.md_steps = md_steps
        self.mc_energy_only = mc_energy_only
        self.before_md_block = before_md_block
        self.after_md_block = after_md_block
        self.separate_models = mc.model is not md.model

    @contextlib.contextmanager
    def _mc_outputs(self) -> Iterator[None]:
        """Narrow the shared model to energy-only for one MC block, then restore."""
        if not self.mc_energy_only:
            yield
            return
        config = self.mc.model.model_config
        full = set(config.active_outputs)
        config.active_outputs = {"energy"}
        try:
            yield
        finally:
            config.active_outputs = full

    def run_mc_block(self, batch: Batch) -> None:
        """Run one block of ``mc_steps`` MC proposals.

        Applies ``mc_energy_only`` (narrowed outputs plus the baseline
        re-evaluation) and, with separate MC and MD models, the baseline
        re-evaluation with the MC model. Prefer :meth:`run` with ``on_block``
        for per-block observables; a loop that reimplements :meth:`run` must
        call this, not ``self.mc.run``, or both are silently bypassed.

        Parameters
        ----------
        batch
            The shared hybrid batch, advanced in place.
        """
        if self.mc_steps == 0:
            return
        with self._mc_outputs():
            if self.mc_energy_only or self.separate_models:
                self.mc.refresh_energy(batch)
            self.mc.run(batch, n_steps=self.mc_steps)

    def prepare_md_block(self, batch: Batch) -> None:
        """Run ``before_md_block`` (if set) ahead of an MD block's first force call."""
        if self.before_md_block is not None:
            self.before_md_block(batch)

    def run_md_block(self, batch: Batch, n_steps: int | None = None) -> None:
        """Run one MD block and hand the resulting energy to MC.

        ``md.run`` evaluates forces before its first step, so derivatives left
        by a rejected MC trial never reach the integrator. With zero steps
        only that evaluation runs (for ``after_md_block``).

        Parameters
        ----------
        batch
            The shared hybrid batch, advanced in place.
        n_steps
            MD steps; defaults to ``md_steps``.
        """
        n_steps = self.md_steps if n_steps is None else n_steps
        self.prepare_md_block(batch)
        if n_steps > 0:
            self.md.run(batch, n_steps=n_steps)
        else:
            self.md.compute(batch)
        if self.after_md_block is not None:
            self.after_md_block(batch)
        self.mc.synchronize(batch)

    def run(
        self,
        batch: Batch,
        n_blocks: int,
        on_block: Callable[[Batch, int], bool | None] | None = None,
    ) -> Batch:
        """Run alternating MC and MD blocks on the supplied batch.

        Parameters
        ----------
        batch
            The shared hybrid batch, advanced in place.
        n_blocks
            Number of MC + MD blocks.
        on_block
            Called after every block as ``on_block(batch, block_index)``, once
            MC has adopted the block's final energy; record observables here.
            Returning ``True`` stops the run early.

        Returns
        -------
        Batch
            The same batch object.
        """
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if getattr(batch, "forces", None) is None:
            raise ValueError("hybrid MC-MD requires preallocated batch.forces")
        # Without MD steps or a hook nothing changes the geometry between MC
        # blocks, and MC's own accepted energies are already on the batch.
        md_between_blocks = self.md_steps > 0 or self.after_md_block is not None
        with self.md:
            self.run_md_block(batch, n_steps=0)
            for block in range(n_blocks):
                self.run_mc_block(batch)
                if md_between_blocks:
                    self.run_md_block(batch)
                if on_block is not None and on_block(batch, block):
                    break
        return batch
