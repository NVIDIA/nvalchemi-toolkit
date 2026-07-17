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
Bring your own PyTorch model: from architecture to a saved spec
=================================================================

This walkthrough takes a small interatomic potential — written in plain
PyTorch — through the canonical BYO arc:

1. **Architecture.** A Behler-Parrinello-style energy model: a per-atom
   radial descriptor + an MLP energy head. Returns *atomic_energies*
   only.
2. **Wrapper.** A :class:`BaseModelMixin` adapter — pure single-process
   PyTorch. No ``ShardTensor``, no ``_dist_ctx``, no spec.
3. **Run + inspect.** Build a sample, call the wrapper, look at the
   shapes and values it produces.
4. **Validate.** :func:`trace_and_validate` spawns a multi-rank run,
   compares per-output tensors against the single-process reference,
   and (if anything diverges) tells you what needs to change. The
   wrapper goes in unmodified — no distributed code is needed for an
   MPNN-halo model whose forward is just scatter-aggregations and
   autograd.
5. **Persist.** :meth:`MLIPSpec.save` writes the spec the validator
   discovered to disk; :meth:`MLIPSpec.load` reads it back. Production
   wrappers ship the saved JSON alongside the checkpoint.

For wrappers with non-PyTorch kernels (Warp / Triton) where the spec
needs an explicit :class:`OpAdapter` declaration, see
:doc:`05_byo_graph_transformer`. For the design rationale and the
adapter mechanics, see ``docs/userguide/distributed_byo.md``.

.. note::

    Read-oriented. ``main()`` at the bottom is runnable; the sections
    above are illustrative.

    To execute the walkthrough end-to-end (requires CUDA)::

        python examples/distributed/04_byo_pytorch_mpnn.py
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.spec import MLIPSpec
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

# ====================================================================
# 1. Architecture — Behler-Parrinello-style energy model
# ====================================================================
#
#   G[i] = Σ_j f(r_ij)        — per-atom radial sum over neighbours
#   E[i] = MLP(G[i], Z[i])     — per-atom energy (Z-conditioned)
#
# Returns *atomic_energies only*. Total energy and forces are computed
# at the wrapper boundary.


class BPDescriptor(nn.Module):
    """Per-atom radial descriptor: ``G[i] = Σ_j f(r_ij)``."""

    def __init__(self, feat_dim: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, feat_dim)
        self.register_buffer("centers", centers, persistent=True)
        self.gamma = 0.5 / ((cutoff / feat_dim) ** 2)

    def forward(
        self,
        positions: torch.Tensor,  # (n_atoms, 3)
        edge_index: torch.Tensor,  # (2, n_edges) — (sender, receiver)
    ) -> torch.Tensor:
        rij = positions[edge_index[1]] - positions[edge_index[0]]
        r = torch.linalg.vector_norm(rij, dim=-1, keepdim=True)
        cutoff_envelope = (
            0.5
            * (1.0 + torch.cos(torch.pi * r / self.cutoff))
            * (r < self.cutoff).float()
        )
        edge_feats = torch.exp(-self.gamma * (r - self.centers) ** 2) * cutoff_envelope

        G = torch.zeros(
            positions.shape[0],
            edge_feats.shape[-1],
            device=positions.device,
            dtype=positions.dtype,
        )
        receivers = edge_index[1].unsqueeze(-1).expand_as(edge_feats)
        # Rebind the return: single-process ``scatter_add_`` mutates in
        # place and returns ``self`` (a no-op rebind), but under domain
        # decomposition the dispatch handler returns a *new*
        # cross-rank-corrected tensor — without the rebind ``G`` would
        # silently stay zero.
        G = G.scatter_add_(0, receivers, edge_feats)
        return G


class BPModel(nn.Module):
    """Pure-PyTorch energy model. Returns ``{"atomic_energies": (n_atoms,)}``."""

    def __init__(self, feat_dim: int = 32, cutoff: float = 5.0) -> None:
        super().__init__()
        self.descriptor = BPDescriptor(feat_dim, cutoff)
        self.element_embedding = nn.Embedding(120, 8)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 8, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        G = self.descriptor(positions, edge_index)
        z_emb = self.element_embedding(atomic_numbers)
        atomic_energies = self.head(torch.cat([G, z_emb], dim=-1)).squeeze(-1)
        return {"atomic_energies": atomic_energies}


# ====================================================================
# 2. Wrapper — :class:`BaseModelMixin` adapter
# ====================================================================
#
# Pure single-process: no ``ShardTensor``, no ``_dist_ctx``, no spec.
# Distribution is the framework's problem; the wrapper just describes
# the model.


class BPWrapper(nn.Module, BaseModelMixin):
    """:class:`BaseModelMixin` adapter around :class:`BPModel`.

    Boundary concerns:

    * ``model_config`` declares the active outputs + autograd inputs.
    * ``adapt_input`` extracts the COO neighbour list and drops any
      sentinel rows the producer used for padding (a single
      distribution-agnostic filter that also catches halo-receiver
      sentinels under domain decomposition).
    * ``forward`` runs the model, scatter-aggregates per-atom energies
      into per-graph totals, and computes forces via autograd.
    """

    def __init__(self, model: BPModel) -> None:
        super().__init__()
        self.model = model
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            neighbor_config=NeighborConfig(
                cutoff=model.descriptor.cutoff, format=NeighborListFormat.COO
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
        # Drop NL sentinel rows: both the builder's genuine padding rows
        # (atoms with fewer neighbours than ``max_neighbors``) and, under
        # domain decomposition, the halo-receiver rows the framework
        # rewrote to the same sentinel to enforce one rank per edge. One
        # line covers both regimes.
        valid = (edge_index[0] < n_atoms) & (edge_index[1] < n_atoms)
        return {
            "atomic_numbers": data.atomic_numbers,
            "edge_index": edge_index[:, valid],
        }

    def adapt_output(
        self, model_output: dict[str, torch.Tensor], data: AtomicData | Batch
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

        n_graphs = data.num_graphs
        kwargs = self.adapt_input(data)
        kwargs["positions"] = positions
        raw = self.model(**kwargs)
        atomic_energies = raw["atomic_energies"]

        total_energy = torch.zeros(
            n_graphs, device=positions.device, dtype=positions.dtype
        )
        total_energy = total_energy.scatter_add_(0, data.batch_idx, atomic_energies)
        raw["energy"] = total_energy

        if compute_forces:
            (forces_grad,) = torch.autograd.grad(
                outputs=total_energy.sum(),
                inputs=positions,
                create_graph=False,
            )
            raw["forces"] = -forces_grad

        return self.adapt_output(raw, data)


# ====================================================================
# 3. Run the model + inspect the outputs (single-process)
# ====================================================================


def _build_lattice_batch(n_per_side: int, device: torch.device) -> Batch:
    """Non-periodic simple-cubic cluster. Dense enough that a 5 Å NL has real
    edges. Non-periodic on purpose: this descriptor uses
    ``positions[recv] - positions[send]`` with no minimum-image shift, so it is
    only correct without periodic wraparound — under PBC the single-process
    neighbor list (shift vectors) and the domain-decomposed halo (ghost images)
    resolve cross-boundary edges differently, so the two would not agree. A cell
    is still supplied so the partitioner has a box to decompose."""
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


def _model_factory():
    """Wrapper factory the validator uses. Called once in the
    launcher (reference) and once per spawned worker; the
    deterministic seed keeps every replica in sync.

    Set ``NVALCHEMI_BP_COMPILE=1`` to ``torch.compile`` the inner
    ``BPModel`` (the wrapper itself stays eager — only the model under
    the wrapper is compiled, mirroring ``MACEWrapper.from_checkpoint(
    compile_model=True)``). The optional ``NVALCHEMI_BP_COMPILE_BACKEND``
    overrides the backend (``inductor`` default; ``aot_eager`` for an
    Inductor-codegen-free run). Env-var-driven so the flag propagates
    cleanly through the validator's ``mp.spawn`` workers.
    """
    import os

    torch.manual_seed(123)
    model = BPModel(feat_dim=32, cutoff=5.0)
    if os.environ.get("NVALCHEMI_BP_COMPILE", "0") != "0":
        backend = os.environ.get("NVALCHEMI_BP_COMPILE_BACKEND", "inductor")
        model = torch.compile(model, backend=backend, fullgraph=False)
    return BPWrapper(model).cuda()


# ====================================================================
# 4. Validate via trace_and_validate
# 5. Persist the spec via MLIPSpec.save / load
# ====================================================================
#
# The validator returns a :class:`TraceReport` whose ``.spec`` is the
# working spec on success (or the closest variant on failure). We call
# ``.save()`` on that spec and round-trip it through ``.load()``.
# Production wrappers ship the saved JSON alongside their checkpoint.


def main() -> None:
    """Walkthrough: build → run → validate → save spec → load round-trip."""
    from loguru import logger

    if not torch.cuda.is_available():
        logger.error("This walkthrough's main() requires CUDA.")
        return

    import os

    device = torch.device("cuda:0")
    # A 2-rank partition only truly decomposes when the box is wide enough that
    # each rank's halo does NOT already cover every atom (box/ranks >
    # ~2*(cutoff+skin)); with spacing 2.5 and cutoff 5.0 that needs
    # n_per_side >= 10. The default validates on such a genuinely-decomposed
    # system (remote atoms on each rank), so trace_and_validate is real
    # evidence rather than the trivial-partition case it warns about. Override
    # with NVALCHEMI_BP_N_PER_SIDE (e.g. 4) for a quicker degenerate smoke run.
    n_per_side = int(os.environ.get("NVALCHEMI_BP_N_PER_SIDE", "10"))
    sample_batch = _build_lattice_batch(n_per_side=n_per_side, device=device)

    # ----- Section 3: run + inspect -----
    logger.info("Section 3: build, run, inspect.")
    wrapper = _model_factory()
    from nvalchemi.neighbors import compute_neighbors

    compute_neighbors(sample_batch, config=wrapper.model_config.neighbor_config)

    out = wrapper(sample_batch)
    logger.info(
        "  forward returned: "
        "energy={e:.4f} (shape {es}, dtype {ed})  |  "
        "forces shape={fs}, ‖forces‖∞={fmax:.4f}",
        e=out["energy"].item(),
        es=tuple(out["energy"].shape),
        ed=out["energy"].dtype,
        fs=tuple(out["forces"].shape),
        fmax=out["forces"].abs().max().item(),
    )

    # ----- Section 4: validate -----
    compile_state = os.environ.get("NVALCHEMI_BP_COMPILE", "0")
    backend = os.environ.get("NVALCHEMI_BP_COMPILE_BACKEND", "inductor")
    logger.info(
        "Section 4: trace_and_validate (NVALCHEMI_BP_COMPILE={c}, backend={b}).",
        c=compile_state,
        b=backend if compile_state != "0" else "n/a",
    )
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

    if not report.ok:
        logger.error("Validation did not pass; not saving spec.")
        return

    # ----- Section 5: persist + round-trip -----
    spec_path = Path("bp_model_spec.json")
    report.spec.save(spec_path)
    loaded_spec = MLIPSpec.load(spec_path)
    assert loaded_spec.distribution.policy == report.spec.distribution.policy
    assert loaded_spec.output_kinds == report.spec.output_kinds
    logger.info(
        "Section 5: spec written to {p} and round-tripped cleanly. "
        "Production usage::\n"
        "    spec = MLIPSpec.load({p!r})\n"
        "    wrapper = BPWrapper(BPModel(...))\n"
        "    dist_model = DistributedModel(wrapper, domain_cfg, spec=spec)",
        p=str(spec_path),
    )


if __name__ == "__main__":
    main()
