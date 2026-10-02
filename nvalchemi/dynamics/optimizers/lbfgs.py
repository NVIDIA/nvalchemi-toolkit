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
"""
L-BFGS and L-BFGS+variable-cell geometry optimizers.

L-BFGS steps along a quasi-Newton direction built from the last
``history_size`` position/force differences, bounded by a ``maxstep`` trust
region.  One force evaluation per step; no energy.

* ``LBFGS``             — fixed-cell coordinate optimizer.
* ``LBFGSVariableCell`` — variable-cell optimizer.

Both classes delegate to ``lbfgs_step_coord`` and ``lbfgs_step_coord_cell``
from :mod:`nvalchemiops.torch.lbfgs`.  The step is placed entirely in
``pre_update``; ``post_update`` is a no-op.

The two-loop recursion iterates over ``history_size`` in Python, so each
step launches noticeably more Warp kernels than a FIRE2 step.  Fewer total
steps usually still wins on wall-clock time once the model forward pass
dominates, but with a cheap model this per-step launch overhead can make
L-BFGS slower than FIRE2 despite converging in fewer steps.

Hyperparameters:

* ``history_size``  — stored curvature pairs (default 6); fixed at allocation
* ``curvature_eps`` — pair acceptance floor (default ``None``: by dtype)
* ``maxstep``       — maximum displacement per step (default 0.2)

State spans two levels: per-system scalars and a segmented ``"lbfgs_dofs"``
level (one row per atom, plus two per system for variable cell) holding
the history.  Positions must not be edited between steps (e.g. by
``WrapPeriodicHook``): the next step differences them against the last.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._ops._bridge import _make_two_level_state_batch
from nvalchemi.dynamics._ops.cell_align import cell_alignment_offenders
from nvalchemi.dynamics._ops.lbfgs import (
    LBFGSCellState,
    LBFGSState,
    lbfgs_prepare_cell_state,
    lbfgs_prepare_state,
    lbfgs_step_coord,
    lbfgs_step_coord_cell,
)
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook, _aligned_periodic
from nvalchemi.hooks.periodic import WrapPeriodicHook

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["LBFGS", "LBFGSVariableCell"]

_LBFGS_DEFAULTS = dict(
    history_size=6,
    curvature_eps=None,
    maxstep=0.2,
)

_DOF_LEVEL = "lbfgs_dofs"
_PER_DOF = ("x_base", "force_base", "direction", "s_history", "y_history")
_PER_SYSTEM = (
    "ys", "yy", "alpha_hist", "beta_hist", "ss", "gg", "d0", "dmax", "dquad",
    "alpha_step", "iteration", "end", "n_loop", "history_count",
)  # fmt: skip
_CELL_PER_DOF = ("ext_positions", "ext_forces")
_CELL_PER_SYSTEM = (
    "ref_cell", "ref_cell_inv", "kappa", "phi", "phi_inv", "d_phi",
    "cell_dof_a", "cell_dof_b", "cell_force_a", "cell_force_b",
)  # fmt: skip


#: Axis-reversal permutation (swaps x <-> z, leaves y): the one piece
#: ``_anti_transpose``, ``_axis_reverse_vectors`` and ``_axis_reverse_matrix``
#: share.  Built lazily per dtype/device by those helpers rather than fixed
#: at import time, and cached (see ``_axis_reverse`` below) since
#: ``LBFGSVariableCell.pre_update`` reaches it six times a step.
_AXIS_REVERSE = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))


@functools.lru_cache(maxsize=None)
def _axis_reverse(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Cached per (dtype, device): read-only in every caller, so the same
    # tensor is safe to share, and each cache miss is a small host-to-device
    # copy that would otherwise repeat on every one of the six calls this
    # makes per step, on top of the per-step launch overhead the module
    # docstring warns about.
    return torch.tensor(_AXIS_REVERSE, dtype=dtype, device=device)


def _anti_transpose(cell: torch.Tensor) -> torch.Tensor:
    r"""Reflect a row-convention cell across its anti-diagonal for the backend.

    ``nvalchemiops.dynamics.optimizers.lbfgs._lbfgs_step_coord_cell_impl``
    computes the deformation gradient as ``Phi = H @ H_ref^-1`` from
    whatever ``cell`` it is handed, packs only ``Phi``'s *lower*-triangular
    six entries (``(0,0),(1,0),(2,0),(1,1),(2,1),(2,2)``) as the cell's
    degrees of freedom, and reads the conjugate force off the same six
    slots.  ``nvalchemi`` (like ASE) stores lattice vectors as rows, aligned
    so ``a`` (row 0) is simplest and ``c`` (row 2) general; nvalchemiops'
    own :func:`~nvalchemiops.dynamics.utils.cell_filter.align_cell` fills
    the *same* row-major layout (verified: its kernel literally constructs
    ``wp.mat33d(a, 0, 0,  b*cosg, b*sing, 0,  c1, c2, c3)``, i.e. row 0
    simplest too) — so nvalchemi's row-aligned ``H`` already satisfies the
    backend's own structural assumption, and needs no transform to reach
    *a* valid chart.

    But not *this* chart.  For two row-aligned cells ``H``, ``H0`` (both
    lower-triangular as plain matrices), the true physical deformation
    gradient relating them — the one whose entries actually have the
    physical meaning the force formula below assumes — is forced to be
    *upper*-triangular: :math:`\Phi_{\text{true}}^T = H_0^{-1} H` is a
    product of lower-triangular matrices (lower-triangular itself), so
    :math:`\Phi_{\text{true}}` is its transpose.  Feeding ``H`` directly
    makes the backend's internal ``Phi`` lower-triangular but *wrong*
    (confirmed: its force formula then only matches a from-scratch
    finite-difference gradient to ~0.1% relative error, exact only in the
    degenerate case ``H == H0``).  Transposing alone fixes that formula
    exactly (machine precision, not ~0.1%) but makes ``Phi`` upper
    triangular — so the packer's fixed lower-triangular six-slot reading
    now sees structural zeros exactly where ``Phi``'s real shear content
    lives, permanently losing it: an optimizer built this way cannot
    correct a stale shear-type reference cell no matter how long it runs
    (measured: bounded force/stress oscillation that never tightens, for
    any nonzero shear between reference and current cell, however small).

    The fix is the transform that is simultaneously (a) exact for the
    force formula and (b) keeps the packed ``Phi`` lower-triangular: not a
    transpose, but an *anti-transpose* — reflection across the
    anti-diagonal, ``H -> J H^T J`` with ``J`` the axis-reversal
    permutation (:data:`_AXIS_REVERSE`).  Equivalently: relabel which
    physical axis is "x" vs "z" *and* which lattice vector is "a" vs "c",
    together.  ``J (H0^{-1} H)^T J`` is then a product of two
    anti-transposes, each of which maps lower-triangular to lower-triangular
    (reflecting across the anti-diagonal reverses which off-diagonal
    entries survive the same way reversing both row and column order
    would), so the packed ``Phi`` stays lower-triangular and loses nothing.
    Verified end-to-end against finite differences on a periodic LJ system
    with a genuinely non-trivial (non-diagonal) reference: the backend's
    predicted ``dE/dPhi`` under this transform matches a central-difference
    estimate to ~1e-9 relative error, and a full relaxation from a sheared
    stale reference converges to the correct equilibrium — see
    ``TestLBFGSVariableCellPhysicalCorrectness`` in
    ``test/dynamics/test_lbfgs.py``.

    Positions and forces are per-atom 3-vectors, not matrices, so they
    carry no row/column ambiguity, but *do* need the same axis relabeling
    applied component-wise — see :func:`_axis_reverse_vectors`.  Cauchy
    stress is symmetric (:math:`\sigma = \sigma^T`) so it needs no
    transpose, but *does* still need the axis relabeling (conjugation by
    ``J``, no transpose) to stay paired with the relabeled cell — see
    :func:`_axis_reverse_matrix`.  Skipping the stress relabeling alone
    reintroduces a large (measured: ~67% relative) force error, because
    the force formula contracts stress against ``Phi`` in the relabeled
    frame.
    """
    j = _axis_reverse(cell.dtype, cell.device)
    return (j @ cell.transpose(-1, -2) @ j).contiguous()


def _axis_reverse_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Swap the x/z components of each row vector: ``vectors @ J``.

    Keeps per-atom positions/forces consistent with :func:`_anti_transpose`'s
    cell relabeling.  An involution: applying it twice is the identity, used
    both to feed the backend and to undo the relabeling on the way out.
    """
    j = _axis_reverse(vectors.dtype, vectors.device)
    return (vectors @ j).contiguous()


def _axis_reverse_matrix(matrices: torch.Tensor) -> torch.Tensor:
    """Conjugate each per-system matrix by the axis reversal: ``J @ M @ J``.

    No transpose — unlike :func:`_anti_transpose`, this is for quantities
    (stress) that already transform like a rank-2 Cartesian tensor rather
    than like a cell whose rows are lattice vectors.
    """
    j = _axis_reverse(matrices.dtype, matrices.device)
    return (j @ matrices @ j).contiguous()


def _build_state(
    atoms_per_system: torch.Tensor,
    history_size: int,
    dtype: torch.dtype,
    dev: torch.device,
    *,
    cell: torch.Tensor | None = None,
    cell_force_scale: float = 1.0,
) -> Batch:
    segments = atoms_per_system.to(torch.int32) + (0 if cell is None else 2)
    opt = lbfgs_prepare_state(
        int(segments.sum()),
        segments.numel(),
        dtype=dtype,
        device=dev,
        history_size=history_size,
    )
    system = {k: getattr(opt, k) for k in _PER_SYSTEM}
    dofs = {k: getattr(opt, k) for k in _PER_DOF}
    if cell is not None:
        atom_ptr = torch.nn.functional.pad(
            atoms_per_system.cumsum(0, dtype=torch.int32), (1, 0)
        )
        # The anti-transpose of a row-aligned cell is itself lower-triangular
        # (see _anti_transpose), so it passes lbfgs_prepare_cell_state's
        # alignment gate directly — no need to admit the row form and then
        # overwrite the chart afterward.
        cs = lbfgs_prepare_cell_state(
            atom_ptr,
            _anti_transpose(cell),
            cell_force_scale=cell_force_scale,
            dtype=dtype,
            device=dev,
        )
        system |= {k: getattr(cs, k) for k in _CELL_PER_SYSTEM}
        dofs |= {k: getattr(cs, k) for k in _CELL_PER_DOF}
    return _make_two_level_state_batch(
        system, dofs, segments, dev, level_name=_DOF_LEVEL
    )


def _warn_if_wraps_positions(dynamics: BaseDynamics) -> None:
    """Warn when ``WrapPeriodicHook`` is registered on *dynamics* or its ``FusedStage``.

    L-BFGS differences consecutive positions (``s = x - x_base``) to build its
    curvature history.  A hook that edits positions between steps — most
    commonly ``WrapPeriodicHook``, the natural thing to copy over from an NVE
    example — inserts a spurious lattice-translation jump into that history.
    This doesn't error; it silently corrupts the search direction, which has
    been measured to inflate the number of steps to converge by 50-500x, or
    to stall convergence outright.  See the module docstring.
    """
    own_and_enclosing = (*dynamics.hooks, *dynamics._enclosing_hooks)
    if any(isinstance(h, WrapPeriodicHook) for h in own_and_enclosing):
        warnings.warn(
            f"{type(dynamics).__name__} has WrapPeriodicHook registered. "
            "Editing positions between steps corrupts L-BFGS's curvature "
            "history and can silently inflate the number of steps to "
            "converge by 50-500x, or stall convergence entirely. Remove "
            "WrapPeriodicHook from this optimizer (or its FusedStage), or "
            "use FIRE2/FIRE2VariableCell if periodic wrapping is required.",
            UserWarning,
            stacklevel=2,
        )


def _ops_state(state: Batch) -> LBFGSState:
    return LBFGSState(**{k: state[k] for k in _PER_DOF + _PER_SYSTEM})


def _ops_cell_state(state: Batch) -> LBFGSCellState:
    # The packed topology is the segmented level's own; never stored.
    return LBFGSCellState(
        ext_batch_idx=state._storage.groups[_DOF_LEVEL].batch_idx.int(),
        ext_atom_ptr=state.level_ptr(_DOF_LEVEL),
        **{k: state[k] for k in _CELL_PER_SYSTEM + _CELL_PER_DOF},
    )


class _LBFGSMixin:
    """Allocation-time contract shared by ``LBFGS`` and ``LBFGSVariableCell``.

    Not a :class:`~nvalchemi.dynamics.base.BaseDynamics` subclass on its
    own — mixed into both concrete optimizers (which supply that base)
    rather than being one itself, so the state-shape test's
    ``BaseDynamics`` subclass scan
    (``test_state_management._discover_dynamics_implementations``) doesn't
    pick up an incomplete class missing ``__needs_keys__`` /
    ``__provides_keys__``.

    Holds the constructor envelope both optimizers share, the
    ``history_size`` fixed-at-allocation property, and the
    ``_init_state`` / ``_make_new_state`` / ``post_update`` bodies, which
    are otherwise identical between the two and differ only in the cell
    kwargs :meth:`_extra_state_kwargs` supplies.
    """

    # ``post_update`` below is unconditionally ``pass``, so a FusedStage
    # sub-stage never needs the masked save/restore dance around it — see
    # ``BaseDynamics._post_update_is_noop`` and ``_masked_post_update``.
    _post_update_is_noop = True

    def __init__(
        self,
        model: BaseModelMixin,
        history_size: int = _LBFGS_DEFAULTS["history_size"],
        curvature_eps: float | None = _LBFGS_DEFAULTS["curvature_eps"],
        maxstep: float = _LBFGS_DEFAULTS["maxstep"],
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        if (
            not isinstance(history_size, int)
            or isinstance(history_size, bool)
            or history_size <= 0
        ):
            raise ValueError(
                f"history_size must be a positive int; got {history_size!r}"
            )
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        self._history_size = history_size
        self.curvature_eps = curvature_eps
        self.maxstep = maxstep

    @property
    def history_size(self) -> int:
        """Stored curvature pairs; fixed once state is allocated."""
        return self._history_size

    @history_size.setter
    def history_size(self, value: int) -> None:
        raise AttributeError(
            "history_size is fixed when optimizer state is allocated; "
            "construct a new optimizer to change it"
        )

    def _extra_state_kwargs(self, batch: Batch, n: int) -> dict[str, Any]:
        """Extra ``_build_state`` kwargs; overridden by ``LBFGSVariableCell``."""
        del batch, n
        return {}

    def _check_hook_compatibility(self) -> None:
        # Runs every step (see BaseDynamics._check_hook_compatibility), not
        # just at first admission like the identical call in _init_state
        # below: a WrapPeriodicHook registered after the first step — on
        # this optimizer directly, or on an enclosing FusedStage — must
        # still be caught.  The two calls can both fire on step one; that's
        # harmless (same message, deduplicated by Python's default warning
        # filter), and simpler than threading "already warned" state through
        # both call sites.
        _warn_if_wraps_positions(self)

    def _init_state(self, batch: Batch) -> None:
        _warn_if_wraps_positions(self)
        self._state = _build_state(
            batch.num_nodes_per_graph,
            self.history_size,
            batch.positions.dtype,
            batch.device,
            **self._extra_state_kwargs(batch, batch.num_graphs),
        )

    def _make_new_state(self, n: int, template_batch: Batch) -> Batch:
        return _build_state(
            template_batch.num_nodes_per_graph[-n:],
            self.history_size,
            template_batch.positions.dtype,
            template_batch.device,
            **self._extra_state_kwargs(template_batch, n),
        )

    def post_update(self, batch: Batch) -> None:
        """No-op; forces from new positions are used on the next step."""


class LBFGS(_LBFGSMixin, BaseDynamics):
    """Fixed-cell L-BFGS geometry optimizer.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.
    history_size : int
        Stored curvature pairs.  Must be a positive int; checked at
        construction.  Fixed once state is allocated.  Default 6.
    curvature_eps : float, optional
        Curvature-pair acceptance floor.  Default ``None`` (by dtype).
    maxstep : float
        Maximum displacement per step.  Not validated: ``maxstep <= 0``
        disables the trust region (unbounded step length per
        :func:`nvalchemiops.dynamics.optimizers.lbfgs._alpha_cap`), which is
        occasionally useful but easy to trigger by accident (e.g. a stray
        negative sign) since nothing raises.  Default 0.2.
    n_steps : int, optional
        Total steps for :meth:`run`.
    hooks : list[Hook], optional
        Initial hooks.
    convergence_hook : ConvergenceHook or dict, optional
        Convergence criterion.
    **kwargs
        Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.

    Attributes
    ----------
    __needs_keys__ : set[str]
        ``{"forces"}``.
    __provides_keys__ : set[str]
        ``{"positions"}``.
    """

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"positions"}

    def pre_update(self, batch: Batch) -> None:
        """Full L-BFGS step using current forces.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* updated in-place.
        """
        # Detach positions to avoid "non-leaf .grad accessed" warning from
        # wp.from_torch.  In-place updates still apply to the batch storage.
        lbfgs_step_coord(
            batch.positions.detach(),
            batch.forces,
            _ops_state(self._state),
            batch.batch_idx.int(),
            maxstep=self.maxstep,
            curvature_eps=self.curvature_eps,
        )


class LBFGSVariableCell(_LBFGSMixin, BaseDynamics):
    """Variable-cell L-BFGS geometry optimizer.

    Relaxes atomic coordinates and the cell together, driven by the model's
    stress.  Cells must be aligned: install
    :class:`~nvalchemi.dynamics.hooks.AlignCellHook` (``frequency=1``) on this
    optimizer or its ``FusedStage``, or pass pre-aligned cells.

    Parameters
    ----------
    model : BaseModelMixin
        The neural network potential model.  Must produce ``"stress"``.
    history_size : int
        Stored curvature pairs.  Must be a positive int; checked at
        construction.  Fixed once state is allocated.  Default 6.
    curvature_eps : float, optional
        Curvature-pair acceptance floor.  Default ``None`` (by dtype).
    maxstep : float
        Maximum displacement per step.  Not validated: ``maxstep <= 0``
        disables the trust region (unbounded step length per
        :func:`nvalchemiops.dynamics.optimizers.lbfgs._alpha_cap`), which is
        occasionally useful but easy to trigger by accident (e.g. a stray
        negative sign) since nothing raises.  Default 0.2.
    n_steps : int, optional
        Total steps for :meth:`run`.
    hooks : list[Hook], optional
        Initial hooks.
    convergence_hook : ConvergenceHook or dict, optional
        Convergence criterion.
    cell_force_scale : float
        Fixed once state is allocated — unlike ``FIRE2VariableCell``'s
        ``cell_force_scale``, which is a plain mutable attribute read every
        step; the two share a name and purpose but not a mutability
        contract.  Multiplier on the atom count normalizing stress-derived
        cell forces; raise it to move the cell less per step.  Default 1.0.
    **kwargs
        Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.

    Attributes
    ----------
    __needs_keys__ : set[str]
        ``{"forces", "stress"}``.
    __provides_keys__ : set[str]
        ``{"positions", "cell"}``.
    """

    __needs_keys__: set[str] = {"forces", "stress"}
    __provides_keys__: set[str] = {"positions", "cell"}

    def __init__(
        self,
        model: BaseModelMixin,
        history_size: int = _LBFGS_DEFAULTS["history_size"],
        curvature_eps: float | None = _LBFGS_DEFAULTS["curvature_eps"],
        maxstep: float = _LBFGS_DEFAULTS["maxstep"],
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        *,
        cell_force_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(cell_force_scale) or cell_force_scale <= 0:
            raise ValueError(
                f"cell_force_scale must be finite and positive; got {cell_force_scale}"
            )
        super().__init__(
            model=model,
            history_size=history_size,
            curvature_eps=curvature_eps,
            maxstep=maxstep,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        self._cell_force_scale = cell_force_scale

    @property
    def cell_force_scale(self) -> float:
        """Cell-force normalization multiplier; fixed once state is allocated."""
        return self._cell_force_scale

    @cell_force_scale.setter
    def cell_force_scale(self, value: float) -> None:
        raise AttributeError(
            "cell_force_scale is fixed when optimizer state is allocated; "
            "construct a new optimizer to change it"
        )

    def _reference_cells(self, batch: Batch, n: int) -> torch.Tensor:
        """Aligned cells of the last *n* systems, for the chart.  Never writes *batch*."""
        # Own hooks, or those of an enclosing FusedStage: both run at
        # BEFORE_STEP, before this stage's first pre_update.
        #
        # KNOWN LIMITATION: `_enclosing_hooks` is a live back-pointer set by
        # FusedStage.__init__ (nvalchemi/dynamics/base.py).  Deriving a new
        # stage via `+` from a FusedStage this optimizer belongs to repoints
        # `_enclosing_hooks` at the *new* stage's (possibly empty) hooks, so
        # if you keep running the original stage afterward, an AlignCellHook
        # registered on it can stop being found here even though it is still
        # registered — see FusedStage.__add__ for the full explanation.
        align_hooks = [
            h
            for h in (*self.hooks, *self._enclosing_hooks)
            if isinstance(h, AlignCellHook)
        ]
        if any(h.frequency != 1 for h in align_hooks):
            raise ValueError(
                "LBFGSVariableCell requires AlignCellHook(frequency=1): its "
                "reference chart assumes every admitted cell is aligned "
                "before the next step."
            )
        cell = batch.cell.detach()
        if align_hooks:
            aligned = _aligned_periodic(batch)
            if aligned is not None:
                cell = aligned[1]
        cell = cell[-n:]
        # Same criterion AlignCellHook uses to decide whether a cell needs
        # realigning at all, so a cell it considers already-aligned can
        # never fail here, and vice versa — see cell_alignment_offenders.
        bad = torch.nonzero(cell_alignment_offenders(cell)).flatten()
        if bad.numel():
            systems = (bad + batch.num_graphs - n).tolist()
            skew = torch.triu(cell[bad], 1).abs().amax(dim=(1, 2))
            fix = (
                "AlignCellHook aligns only periodic systems; set pbc for "
                "these, or align their cells before the run."
                if align_hooks
                else "Install AlignCellHook() in hooks, or align the cells "
                "before the run."
            )
            raise ValueError(
                f"LBFGSVariableCell: cell for system(s) {systems} is not "
                f"aligned (upper off-diagonal up to {skew.max().item():.3e}, "
                "or left-handed). " + fix
            )
        return cell

    def _extra_state_kwargs(self, batch: Batch, n: int) -> dict[str, Any]:
        return dict(
            cell=self._reference_cells(batch, n),
            cell_force_scale=self.cell_force_scale,
        )

    def pre_update(self, batch: Batch) -> None:
        """Full L-BFGS variable-cell step using current forces and stress.

        Parameters
        ----------
        batch : Batch
            Current batch; *positions* and *cell* updated in-place.
        """
        # batch.stress is tensile-positive Cauchy stress -W/V (eV/A^3);
        # ops converts it to the cell force internally.
        #
        # The backend wants the anti-transposed cell (see _anti_transpose)
        # and axis-reversed positions/forces/stress to match.  It mutates
        # positions and cell in place and requires contiguous tensors, so
        # these must be transformed *copies*, not views — writes to a
        # non-contiguous view would not land back in the batch, and in any
        # case the backend rejects non-contiguous inputs outright.
        cell_fed = _anti_transpose(batch.cell.detach())
        positions_fed = _axis_reverse_vectors(batch.positions.detach())
        lbfgs_step_coord_cell(
            positions_fed,
            cell_fed,
            _axis_reverse_vectors(batch.forces),
            _axis_reverse_matrix(batch.stress),
            _ops_state(self._state),
            _ops_cell_state(self._state),
            batch.batch_idx.int(),
            maxstep=self.maxstep,
            curvature_eps=self.curvature_eps,
        )
        batch.positions.detach().copy_(_axis_reverse_vectors(positions_fed))
        batch.cell.detach().copy_(_anti_transpose(cell_fed))
