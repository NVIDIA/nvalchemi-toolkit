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
Bring your own model with a Warp kernel: from architecture to spec
====================================================================

The same arc as :doc:`04_byo_pytorch_mpnn`, but now the model embeds
one performance-critical Warp kernel — the case where
:class:`~nvalchemi.distributed.spec.OpAdapter` carries weight.

A Warp kernel is opaque to ShardTensor's ``__torch_function__``: it
calls ``wp.from_torch(t)`` internally, bypassing the dispatch
machinery. :func:`trace_and_validate`'s auto-fix rules can't
synthesise the wrap because they don't know how to read the kernel's
input/output semantics. The wrapper author declares those semantics
once on the spec via an :class:`OpAdapter`, the registry installs a
distribution-aware handler, and from there everything flows.

Walkthrough:

1. **Architecture.** A pairwise interaction kernel computed in Warp:
   ``E_ij = ε * exp(-(r_ij / σ)²)`` summed per receiver. Wrapped as
   ``@torch.library.custom_op`` so PyTorch sees it as a regular op.
2. **Wrapper.** :class:`BaseModelMixin` adapter — distribution-unaware.
   Same shape as example 04: model_config + adapt_input + adapt_output.
3. **Single-process verification.** Confirm the Warp op + autograd
   forces work on a small cluster.
4. **First validation attempt — without OpAdapter.** Show what
   ``trace_and_validate`` reports when the spec doesn't carry the
   kernel's distribution semantics.
5. **Author the OpAdapter + retry.** Declare ``output_transforms={0:
   ScatterOutputs()}`` on the spec; re-run validation.
6. **Persist the working spec** for production use.

For the easy path (pure PyTorch, no opaque kernels), see
:doc:`04_byo_pytorch_mpnn`.

.. note::

    Read-oriented walkthrough. The ``main()`` block at the bottom is
    runnable; everything above it is illustrative.

    To execute the validation + spec save::

        python examples/distributed/05_byo_graph_transformer.py
"""

from __future__ import annotations

import os
import tempfile

# Warp's default kernel cache (~/.cache/warp/) may be read-only in
# sandboxed dev environments; route to a writable temp location so
# the kernel below compiles on first launch. Mirrors the same shim
# the validator harness installs in spawned workers.
os.environ.setdefault(
    "WARP_CACHE_PATH",
    os.path.join(tempfile.gettempdir(), "nvalchemi-tutorial-warp-cache"),
)

from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import warp as wp  # noqa: E402
from torch import Tensor, nn  # noqa: E402

# ====================================================================
# 1. The Warp kernel + custom_op wrap
# ====================================================================
#
# The kernel computes a Gaussian pairwise interaction summed
# per-receiver. One thread per edge:
#
#   E[receiver] += ε * exp(-(r_ij / σ)²)
#
# A real production kernel would be richer (multi-channel, learnable
# parameters, fused gradient). The semantics that matter for
# distribution are the same:
#
# * **Per-edge, write-to-receiver scatter pattern.** Halo rows of the
#   output buffer accumulate partial sums when the receiver is a halo
#   atom — exactly the case :class:`ScatterOutputs` reverse-exchanges.
# * **Reads sender + receiver positions.** Under halo storage,
#   positions cover (owned + halo) rows on each rank, so cross-rank
#   pairs are served by local halo copies — no input transformation
#   needed.


@wp.kernel
def _gaussian_pair_kernel(
    edge_sender: wp.array(dtype=wp.int64),  # (E,)
    edge_receiver: wp.array(dtype=wp.int64),  # (E,)
    positions: wp.array2d(dtype=wp.float32),  # (n_atoms, 3)
    epsilon: wp.float32,
    inv_sigma: wp.float32,
    cutoff_sq: wp.float32,
    energy_per_atom: wp.array(dtype=wp.float32),  # (n_atoms,) — accumulator
) -> None:
    """One thread per edge. Adds ``ε exp(-(r/σ)²)`` into the
    receiver's slot. Skips edges with ``r² > cutoff²``."""
    e = wp.tid()
    i = edge_sender[e]
    j = edge_receiver[e]

    dx = positions[i, 0] - positions[j, 0]
    dy = positions[i, 1] - positions[j, 1]
    dz = positions[i, 2] - positions[j, 2]
    r2 = dx * dx + dy * dy + dz * dz

    if r2 > cutoff_sq:
        return

    r_over_sigma = wp.sqrt(r2) * inv_sigma
    contribution = epsilon * wp.exp(-r_over_sigma * r_over_sigma)
    wp.atomic_add(energy_per_atom, j, contribution)


@torch.library.custom_op(
    "tutorial::gaussian_pair_energy",
    mutates_args=(),
)
def gaussian_pair_energy(
    edge_index: Tensor,  # (2, E) — (sender, receiver), int64
    positions: Tensor,  # (n_atoms, 3), float32
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> Tensor:
    """Per-atom Gaussian-pair energy. Output shape ``(n_atoms,)``.

    For each receiver atom, accumulates ``ε exp(-(r/σ)²)`` over its
    incoming edges within the cutoff. Halo-storage friendly:
    ``positions`` is read rank-locally from a halo-padded tensor; the
    spec's :class:`OpAdapter` reverse-exchanges the per-atom output's
    halo rows back to owners after the kernel returns.
    """
    n_atoms = int(positions.shape[0])
    dev = positions.device
    wp_dev = f"cuda:{dev.index}" if dev.type == "cuda" else "cpu"

    energy_per_atom = torch.zeros(n_atoms, dtype=torch.float32, device=dev)
    wp.launch(
        kernel=_gaussian_pair_kernel,
        dim=int(edge_index.shape[1]),
        inputs=[
            wp.from_torch(edge_index[0].contiguous().to(torch.int64)),
            wp.from_torch(edge_index[1].contiguous().to(torch.int64)),
            wp.from_torch(positions.contiguous().to(torch.float32)),
            float(epsilon),
            float(1.0 / sigma),
            float(cutoff * cutoff),
        ],
        outputs=[wp.from_torch(energy_per_atom)],
        device=wp_dev,
    )
    return energy_per_atom


@gaussian_pair_energy.register_fake
def _gaussian_pair_energy_fake(
    edge_index: Tensor,
    positions: Tensor,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> Tensor:
    """Meta-tensor implementation for ``torch.compile``. Returns a
    correctly-shaped tensor of zeros without launching the kernel."""
    return torch.zeros(positions.shape[0], dtype=torch.float32, device=positions.device)


# Why register a fake / meta path?
# ``register_fake`` lets PyTorch's tracing machinery (``torch.compile``,
# ``torch.export``, FakeTensor mode) reason about output shapes
# without launching the kernel. A custom_op without a fake kernel
# silently breaks under any tracing path that hits the op. Strictly
# optional for the eager-mode validator, but cheap to add.


# ====================================================================
# 1.5. Backward via ``register_autograd``
# ====================================================================
#
# Autograd through the Warp op needs an explicit backward formula. A
# real production kernel would supply a Warp-implemented backward
# (``wp.atomic_add`` into a per-atom grad buffer, paired with
# ``wp.tape``); for the walkthrough we use a PyTorch-implemented
# backward — readable, autograd-correct, good enough for the
# OpAdapter pedagogy. Choice of Warp vs PyTorch backward is
# orthogonal to distribution semantics; the OpAdapter on the spec
# governs both forward and backward distributed plumbing.
#
# Closed-form gradient. For each edge ``(i → j)`` with
# ``r = ‖x_i − x_j‖`` and ``r² < cutoff²``::
#
#   ∂E_j / ∂x_i = (-2 ε / σ²) · exp(−(r/σ)²) · (x_i − x_j)
#   ∂E_j / ∂x_j = -∂E_j / ∂x_i
#
# Multiply by the upstream gradient ``grad_E[j]`` and scatter into
# per-atom ``grad_positions``.


def _gaussian_pair_energy_setup_context(ctx, inputs, output):
    """Save the inputs we need for backward. ``custom_op`` doesn't
    automatically save anything (unlike ``torch.autograd.Function``);
    the setup_context callback is where the explicit save lives."""
    edge_index, positions, epsilon, sigma, cutoff = inputs
    ctx.save_for_backward(edge_index, positions)
    ctx.epsilon = float(epsilon)
    ctx.sigma = float(sigma)
    ctx.cutoff = float(cutoff)


def _gaussian_pair_energy_backward(ctx, grad_output: Tensor):
    """Backward formula. Computes ``∂L/∂positions`` from
    ``∂L/∂energy_per_atom``; non-tensor inputs return ``None``."""
    edge_index, positions = ctx.saved_tensors
    epsilon, sigma, cutoff = ctx.epsilon, ctx.sigma, ctx.cutoff

    sender = edge_index[0].long()
    receiver = edge_index[1].long()
    rij = positions[sender] - positions[receiver]  # (E, 3)
    r2 = (rij * rij).sum(dim=-1)  # (E,)
    cutoff_sq = cutoff * cutoff

    # ε exp(-(r/σ)²) for valid (within-cutoff) edges; zero otherwise.
    inv_sigma_sq = 1.0 / (sigma * sigma)
    energy_per_edge = epsilon * torch.exp(-r2 * inv_sigma_sq)
    valid = r2 < cutoff_sq
    energy_per_edge = torch.where(
        valid, energy_per_edge, torch.zeros_like(energy_per_edge)
    )

    # ∂(grad·E_per_edge)/∂x_sender = (−2/σ²) · grad_output[receiver]
    #                                · energy_per_edge · (x_sender − x_receiver)
    factor = (-2.0 * inv_sigma_sq) * grad_output[receiver] * energy_per_edge
    edge_grad = factor.unsqueeze(-1) * rij  # (E, 3)

    grad_positions = torch.zeros_like(positions)
    # Rebind the return: single-process ``index_add_`` is in-place, but
    # under domain decomposition the halo-correction handler returns a
    # *new* cross-rank-corrected tensor.
    grad_positions = grad_positions.index_add_(0, sender, edge_grad)
    grad_positions = grad_positions.index_add_(0, receiver, -edge_grad)

    # Order matches the forward ABI: (edge_index, positions, eps, σ, c).
    return None, grad_positions, None, None, None


torch.library.register_autograd(
    "tutorial::gaussian_pair_energy",
    _gaussian_pair_energy_backward,
    setup_context=_gaussian_pair_energy_setup_context,
)


# ====================================================================
# 2. Architecture — model on top of the Warp op
# ====================================================================
#
# The model trivially wraps the kernel: positions + edge_index in,
# energy out. Forces come from autograd. A real model would have
# more bells and whistles (per-element parameters, multi-body terms);
# the point of this example is the *kernel*'s integration, not the
# physics.


class GaussianPairModel(nn.Module):
    """Energy + forces via the registered Warp op. Forces flow through
    :func:`register_autograd`'s backward formula.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 2.0,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int,
    ) -> dict[str, torch.Tensor]:
        # Warp custom op. Output shape (n_atoms,) — per-atom Gaussian
        # pair energy. Autograd path through this call uses the
        # backward registered above.
        atomic_energies = torch.ops.tutorial.gaussian_pair_energy.default(
            edge_index, positions, self.epsilon, self.sigma, self.cutoff
        )

        # Per-graph total via scatter_add. Rebind the return — under
        # domain decomposition the per_system_reduce dispatch returns a
        # new cross-rank-corrected tensor.
        total_energy = torch.zeros(
            n_graphs, device=positions.device, dtype=positions.dtype
        )
        total_energy = total_energy.scatter_add_(
            0, batch, atomic_energies.to(positions.dtype)
        )

        return {"energy": total_energy, "atomic_energies": atomic_energies}


# ====================================================================
# 3. Wrapper — BaseModelMixin
# ====================================================================
#
# Same shape as the BP wrapper in example 04. The wrapper has zero
# distribution-aware code. Distribution-specific behaviour is declared
# entirely on the spec, separately, in a later step.

from nvalchemi._typing import ModelOutputs  # noqa: E402
from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import (  # noqa: E402
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)


class GaussianPairWrapper(nn.Module, BaseModelMixin):
    def __init__(self, model: GaussianPairModel) -> None:
        super().__init__()
        self.model = model
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            neighbor_config=NeighborConfig(
                cutoff=model.cutoff, format=NeighborListFormat.COO
            ),
        )

    @property
    def cutoff(self) -> float:
        return self.model_config.neighbor_config.cutoff

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        return data

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        n_atoms = data.positions.shape[0]
        edge_index = data.neighbor_list.long().T  # (2, E)
        # Drop NL sentinel rows. Distribution-agnostic: the framework's
        # halo NL filter rewrites halo-receiver rows to the same
        # sentinel ``compute_neighbors`` already uses for genuine
        # padding rows, so this single line covers both regimes.
        valid = (edge_index[0] < n_atoms) & (edge_index[1] < n_atoms)
        edge_index = edge_index[:, valid]

        return {
            "edge_index": edge_index,
            "batch": data.batch_idx,
            "n_graphs": data.num_graphs,
        }

    def adapt_output(
        self, model_output: dict[str, Tensor], data: AtomicData | Batch
    ) -> ModelOutputs:
        out: ModelOutputs = OrderedDict()
        for key in self.model_config.active_outputs:
            if key in model_output:
                out[key] = model_output[key]
        return out

    def forward(self, data: AtomicData | Batch) -> ModelOutputs:
        positions = data.positions
        compute_forces = "forces" in self.model_config.active_outputs

        if compute_forces and not positions.requires_grad:
            positions.requires_grad_(True)

        kwargs = self.adapt_input(data)
        kwargs["positions"] = positions
        raw = self.model(**kwargs)

        if compute_forces:
            (forces_grad,) = torch.autograd.grad(
                outputs=raw["energy"].sum(),
                inputs=positions,
                create_graph=False,
            )
            raw["forces"] = -forces_grad

        return self.adapt_output(raw, data)


# ====================================================================
# 4. The OpAdapter that makes the kernel distribution-aware
# ====================================================================
#
# This is the only distribution-aware code in the whole walkthrough,
# and it lives on the *spec*, not on the wrapper. The wrapper author
# writes the spec once and either returns it from the wrapper's
# ``distribution_spec`` property (the canonical pattern — see MACE /
# UMA wrappers) or attaches it after construction. ``trace_and_validate``'s
# auto-fix rules can polish the surrounding settings (output_kinds,
# all_reduce_outputs) but cannot synthesise the :class:`OpAdapter`
# itself because they don't know what the kernel does.
#
# Spec hierarchy at a glance::
#
#     MLIPSpec          ← top-level: chemistry-aware (output_kinds,
#       │                  all_reduce_outputs, owned_only_outputs).
#       │                  Serializable; this is what gets saved as JSON.
#       ├── core: DistributionSpec      ← chemistry-free: storage policy
#       │     │                    + custom-op declarations. The piece
#       │     │                    a hypothetical non-MLIP user would
#       │     │                    still need.
#       │     ├── policy        ← HaloStoragePolicy / PlainShard (or None)
#       │     │                    — declares *how* local storage relates to
#       │     │                    placement and how scatters/gathers are
#       │     │                    dispatched (covered below).
#       │     └── custom_ops    ← tuple of OpAdapter, one per opaque
#       │                          (Warp/Triton) kernel that needs
#       │                          distribution-aware wrapping.
#       └── output_kinds        ← per-output classification used by
#                                  consolidation (covered below).
#
# What every section of the spec means is explained inline next to
# its construction.


def _build_distribution_spec(wrapper: GaussianPairWrapper):
    """Construct the spec for this wrapper. Centralised here so
    ``main()`` can attach it to the wrapper before the validation
    block, and the production sketch in ``main()`` can reload it from
    disk without rebuilding."""
    from nvalchemi.distributed.ops import HaloStoragePolicy, ScatterOutputs
    from nvalchemi.distributed.spec import (
        DistributionSpec,
        MLIPSpec,
        OpAdapter,
        OutputKind,
    )

    return MLIPSpec(
        # DistributionSpec carries the chemistry-free pieces: storage layout
        # (``policy``) and per-op distribution declarations
        # (``custom_ops``). A non-MLIP user of the same machinery would
        # still need this; everything chemistry-aware lives one layer
        # up on the MLIPSpec.
        distribution=DistributionSpec(
            # ``policy`` — the per-field storage policy. The shipped
            # policies (all from ``nvalchemi.distributed.ops``):
            #
            # * ``HaloStoragePolicy`` — each rank stores ``[owned | halo]``
            #                           rows of every per-atom tensor. Halo
            #                           rows duplicate owners on neighbouring
            #                           ranks, refreshed per layer. Used by
            #                           MACE / MPNN wrappers, LJ, Ewald, PME,
            #                           UMA, and this example.
            # * ``PlainShard``        — each rank stores only its ``n_owned``
            #                           rows; cross-rank lookups go through
            #                           ``all_to_all_v``. Used by AIMNet2.
            # * ``None``              — single-process / no cross-rank comms.
            #
            # ``HaloStoragePolicy`` accepts ``scatter_mode`` /
            # ``gather_mode`` overrides (default ``"halo_correction"`` /
            # ``"halo_read"``): the per-atom ``scatter_add_`` does a local
            # scatter then ``halo_reverse_exchange`` + ``halo_forward_exchange``
            # to push halo-row partials to their owners and refresh copies
            # (the MACE pattern), and ``index_select`` is served from the
            # refreshed local halo rows. The defaults are right here.
            policy=HaloStoragePolicy(),
            # ``custom_ops`` — one :class:`OpAdapter` per opaque kernel
            # (Warp / Triton / any ``@torch.library.custom_op``
            # registered with no native ShardTensor support). The
            # framework's __torch_function__ dispatch can't see *into*
            # such kernels (they call ``wp.from_torch(t)`` etc., which
            # strips the ShardTensor subclass), so the OpAdapter
            # declares the distribution semantics from the outside.
            custom_ops=(
                OpAdapter(
                    # Handle to the registered op. The dispatch
                    # registry installs a wrapper on this handle that
                    # runs the transforms below before/after each
                    # call.
                    op=torch.ops.tutorial.gaussian_pair_energy.default,
                    # ``arg_transforms`` adapt input tensors before
                    # the kernel sees them. Map ``arg_index ->
                    # transform``. Available transforms:
                    #
                    # * ``GatherInputs()``     — halo-pad an
                    #   owned-shape input ``(n_owned, *F)`` to
                    #   ``(n_padded, *F)`` via ``halo_forward_exchange``.
                    #   Use when the kernel needs to read halo rows
                    #   that aren't yet on this rank.
                    # * ``GatherInputsFull()`` — full-gather a sharded
                    #   input to global shape ``(n_global, *F)`` via
                    #   ``distributed_index_select``. The
                    #   sharded-storage analogue of ``GatherInputs``.
                    # * ``SliceOwned()``       — slice a halo-padded
                    #   ``(n_padded, *F)`` to ``(n_owned, *F)`` so the
                    #   kernel only sees owned rows. Pair with
                    #   ``AllReduceSum`` on the output for a
                    #   one-contribution-per-atom-globally pattern
                    #   (Ewald structure factor, PME charge spread).
                    #
                    # Empty here because our positions input is
                    # already halo-padded by the framework before
                    # ``DistributedModel`` calls the wrapper; the
                    # kernel reads it as-is.
                    arg_transforms={},
                    # ``output_transforms`` adapt output tensors after
                    # the kernel returns. Map ``output_index ->
                    # transform``. Available transforms:
                    #
                    # * ``ScatterOutputs()``    — halo-correct a
                    #   per-atom output: ``halo_reverse_exchange``
                    #   pushes halo-row partial sums to owners, then
                    #   ``halo_forward_exchange`` refreshes halo
                    #   copies. The right choice for any kernel that
                    #   writes into per-atom slots from edges. **This
                    #   is what we need here.**
                    # * ``AllReduceSum()``      — sum the per-rank
                    #   partial output across the mesh. Pair with
                    #   ``SliceOwned`` on the input side.
                    # * ``SliceOutputsOwned()`` — slice a sharded
                    #   global-shape output back to owned-only.
                    #   Inverse of ``GatherInputsFull``.
                    #
                    # Output 0 is the per-atom energy. Halo rows of
                    # that buffer hold partial scatter sums (each rank
                    # only sees its halo atom's edges); ScatterOutputs
                    # routes them to the receiver's owner.
                    output_transforms={0: ScatterOutputs()},
                ),
            ),
        ),
        # ``output_kinds`` classifies each named output for
        # consolidation. The consolidation step (after the wrapper's
        # forward returns) needs to know whether to halo-reverse
        # forces, all-reduce stress, or pass an already-global value
        # through unchanged.
        #
        # OutputKind values:
        #
        # * ``PER_NODE``  — one row per atom. Combine rule depends on
        #   whether the output is autograd-derived and whether it's
        #   declared in ``owned_only_outputs`` /
        #   ``all_reduce_outputs``. Forces, atomic_energies.
        # * ``PER_GRAPH`` — one row per system. Energy, stress.
        # * ``GLOBAL``    — already globally correct on every rank;
        #   consolidation passes it through. Rare; usually scalar
        #   metadata or replicated config tensors.
        # * ``UNKNOWN``   — migration default. Consolidation falls
        #   back to a shape-based heuristic and emits a warning. Avoid
        #   in production — declare every output explicitly.
        output_kinds={
            "energy": OutputKind.PER_GRAPH,
            "forces": OutputKind.PER_NODE,
        },
    )


# ====================================================================
# 5. Validation harness — single-process check + trace_and_validate
# ====================================================================


def _build_lattice_batch(n_per_side: int, device: torch.device) -> Batch:
    """Same dense non-periodic cluster the BP example uses — see example 04 for
    design notes. Non-periodic on purpose: the kernel computes
    ``positions[i] - positions[j]`` with no minimum-image shift, so it is only
    correct without periodic wraparound (under PBC the single-process neighbor
    list and the domain-decomposed halo resolve cross-boundary edges
    differently). The cell is kept so the partitioner has a box to decompose."""
    spacing = 2.5
    coords = torch.arange(n_per_side, dtype=torch.float32, device=device) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n_atoms = positions.shape[0]
    torch.manual_seed(0)
    positions = positions + 0.05 * torch.randn_like(positions)
    box = n_per_side * spacing
    atomic_numbers = torch.randint(
        low=1, high=20, size=(n_atoms,), device=device, dtype=torch.long
    )
    cell = torch.eye(3, device=device, dtype=torch.float32) * box
    pbc = torch.tensor([[False, False, False]], device=device)
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        cell=cell.unsqueeze(0),
        pbc=pbc,
    )
    return Batch.from_data_list([data], device=device)


# Module-level holder so the model factory below can attach a freshly
# built spec to each replica's ``distribution_spec`` property without
# re-deriving it. ``trace_and_validate`` calls the factory once in the
# launcher and once per spawned worker; reading the spec from a
# module-global keeps every replica in sync.
_SPEC_CACHE: Any = None


class _GaussianPairWrapperWithSpec(GaussianPairWrapper):
    """Subclass that exposes the cached spec via the
    ``distribution_spec`` property — what production wrappers do via
    a hand-authored property. We split the subclass out so the
    section-3 :class:`GaussianPairWrapper` stays distribution-unaware
    (the example's pedagogical point)."""

    @property
    def distribution_spec(self):
        return _SPEC_CACHE


def _model_factory():
    torch.manual_seed(123)
    return _GaussianPairWrapperWithSpec(GaussianPairModel(cutoff=5.0)).cuda()


def main() -> None:
    """Walkthrough: single-process check → first validation attempt →
    add OpAdapter → revalidate → save spec.

    Requires CUDA: ``trace_and_validate`` is a single-GPU multi-process
    spawn, and the Warp kernel itself is CUDA-only in this example.
    """
    from loguru import logger

    if not torch.cuda.is_available():
        logger.error("This walkthrough's main() requires CUDA.")
        return

    import os

    device = torch.device("cuda:0")

    # ----- Stage A: single-process verification -----
    logger.info("Stage A: single-process verification.")
    # Default to a genuinely-decomposed (non-degenerate) system so
    # trace_and_validate is real evidence; override with
    # NVALCHEMI_GP_N_PER_SIDE (e.g. 4) for a quicker degenerate smoke run.
    n_per_side = int(os.environ.get("NVALCHEMI_GP_N_PER_SIDE", "10"))
    sample_batch = _build_lattice_batch(n_per_side=n_per_side, device=device)

    # Plain wrapper (no spec) for the structural sanity check. We're
    # only confirming the kernel + autograd path runs.
    plain_wrapper = GaussianPairWrapper(GaussianPairModel(cutoff=5.0)).cuda()
    from nvalchemi.distributed.validate.reference import _ensure_neighbors

    _ensure_neighbors(sample_batch, plain_wrapper)
    out = plain_wrapper(sample_batch)
    logger.info(
        "  forward OK: energy={e:.4f}  |  "
        "forces shape={fs}, ‖forces‖∞={fmax:.4f}",
        e=out["energy"].item(),
        fs=tuple(out["forces"].shape),
        fmax=out["forces"].abs().max().item(),
    )

    # ----- Stage B: declare the OpAdapter spec on the wrapper -----
    # In a production wrapper this lives in a ``distribution_spec``
    # property. Here we cache it module-globally so every spawned
    # worker's factory sees the same spec instance — keeping the
    # single-arc flow tight without serialising it through the
    # validator's spec_dict path twice.
    global _SPEC_CACHE
    _SPEC_CACHE = _build_distribution_spec(plain_wrapper)
    logger.info(
        "Stage B: spec declared with one OpAdapter on "
        "torch.ops.tutorial.gaussian_pair_energy."
    )

    # ----- Stage C: validate -----
    logger.info("Stage C: trace_and_validate.")
    from nvalchemi.distributed.validate import trace_and_validate

    report = trace_and_validate(
        model_factory=_model_factory,
        sample_batch=sample_batch,
        world_size=2,
        device="cuda:0",
        atol=1e-4,
        rtol=1e-3,
    )

    report.log_summary(logger)

    # ----- Stage D: persist the spec -----
    spec_path = Path("gaussian_pair_spec.json")
    report.spec.save(spec_path)
    logger.info("Stage D: spec written to {p}", p=spec_path)

    # ----- Stage E: load round-trip + production sketch -----
    from nvalchemi.distributed.spec import MLIPSpec

    # Re-importing the wrapper module re-registers the op (it's tagged
    # at import time via ``@torch.library.custom_op``). For the
    # round-trip check we just make sure the saved JSON resolves the
    # op handle correctly via ``_resolve_op``.
    loaded_spec = MLIPSpec.load(spec_path)
    assert len(loaded_spec.distribution.custom_ops) == 1
    assert loaded_spec.distribution.custom_ops[0].scatter_outputs == (0,)
    logger.info(
        "Stage E: loaded spec round-trips cleanly; OpAdapter "
        "scatter_outputs preserved through the JSON form. "
        "Use ``DistributedModel(wrapper, cfg, spec=loaded_spec)`` "
        "in production runs."
    )


if __name__ == "__main__":
    main()
