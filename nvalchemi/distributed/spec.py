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

"""Distribution spec — each model's declaration of what the distributed
framework must provide for it.

Two layers:

* :class:`DistributionSpec` (in ``_core/spec.py``) carries the
  framework-generic fields: a :class:`StoragePolicy`
  (:class:`HaloStoragePolicy`, or ``None`` for the local case) plus the
  ``custom_ops`` and ``third_party_helpers`` tuples.
* :class:`MLIPSpec` (this module) wraps it and adds output-classification
  sets (``owned_only_outputs``, ``all_reduce_outputs``).

:class:`MLIPSpec` is the public spec. The recommended form declares each
output once via ``outputs={name: OutputSpec(kind, reduce)}``; the parallel
``owned_only_outputs`` / ``all_reduce_outputs`` / ``output_kinds`` sets remain
for serialization. Models declare their spec via
``BaseModelMixin.distribution_spec``; the ``SPEC_*_HALO`` presets cover the
model families we target. See :meth:`MLIPSpec.to_dict` /
:meth:`MLIPSpec.from_dict` for the JSON wire format.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nvalchemi.distributed._core.adapter import (
    AdapterRegistry,
    FunctionAdapter,
    JitAdapter,
    MethodAdapter,
    OpAdapter,
    PythonAdapter,
)
from nvalchemi.distributed._core.spec import DistributionSpec
from nvalchemi.distributed._core.storage_policy import (
    GraphParallelPolicy,
    HaloStoragePolicy,
    StoragePolicy,
)
from nvalchemi.distributed.graph_padder import GraphPadder
from nvalchemi.distributed.output_kinds import OutputKind, OutputSpec, Reduce

__all__ = [
    "MLIPSpec",
    "DistributionSpec",
    "OpAdapter",
    "JitAdapter",
    "PythonAdapter",
    "FunctionAdapter",
    "MethodAdapter",
    "AdapterRegistry",
    "OutputKind",
    "OutputSpec",
    "Reduce",
    "CompilePolicy",
    "ForceStrategy",
    "GraphPadder",
    "replace_policy",
    "SPEC_MPNN_HALO",
    "SPEC_MPNN_GP",
    "SPEC_UMA_HALO",
    "SPEC_LJ_HALO",
    "SPEC_EWALD_HALO",
    "SPEC_PME_HALO",
    "SPEC_DFTD3_HALO",
]


def _merge_policies(a: StoragePolicy | None, b: StoragePolicy | None) -> Any:
    """Combine two storage policies into one that subsumes both.

    ``None`` (local) is the identity. Two halo policies keep halo and pick the
    more permissive scatter/gather mode. Same-class merges keep the class.
    """
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, HaloStoragePolicy) and isinstance(b, HaloStoragePolicy):
        scatter_order = ("local", "halo_correction")
        gather_order = ("local", "halo_read")
        return HaloStoragePolicy(
            scatter_mode=max(
                a.scatter_mode, b.scatter_mode, key=scatter_order.index
            ),
            gather_mode=max(a.gather_mode, b.gather_mode, key=gather_order.index),
        )
    if type(a) is type(b):
        return a
    raise ValueError(
        f"Cannot merge storage policies {type(a).__name__} and "
        f"{type(b).__name__}: halo is the only cross-rank storage policy."
    )


# Map ``replace_policy``'s short kwargs onto policy field names.
_POLICY_FIELD_ALIASES = {"scatter": "scatter_mode", "gather": "gather_mode"}


def replace_policy(spec: "MLIPSpec", **changes: Any) -> "MLIPSpec":
    """Build a new ``MLIPSpec`` with the storage policy's fields replaced.

    Convenience for wrapper-level overrides (e.g. UMA sets the gather mode to
    skip halo correction)::

        new_spec = replace_policy(spec, scatter="local")

    Parameters
    ----------
    spec : MLIPSpec
        The spec whose storage policy is being overridden.
    **changes : Any
        Field overrides. The short names ``scatter`` / ``gather`` map onto the
        policy's ``scatter_mode`` / ``gather_mode``.

    Returns
    -------
    MLIPSpec
        A new spec with the policy fields replaced.

    Raises
    ------
    ValueError
        If the spec has no storage policy to modify (a local ``None`` policy).
    """
    policy = spec.distribution.policy
    if policy is None:
        raise ValueError("replace_policy: spec has no storage policy to modify.")
    mapped = {_POLICY_FIELD_ALIASES.get(k, k): v for k, v in changes.items()}
    new_policy = dataclasses.replace(policy, **mapped)
    new_core = dataclasses.replace(spec.distribution, policy=new_policy)
    return dataclasses.replace(spec, distribution=new_core)


def _decode_output_kinds(raw: Any) -> dict[str, OutputKind]:
    """Decode the ``output_kinds`` slot from a serialized v2 dict.

    Accepts both the canonical sorted ``[[key, kind_value], ...]`` list-of-pairs
    and a plain ``{key: kind_value}`` dict. Unknown kind values raise so typos
    surface at load time rather than degrading to UNKNOWN later.
    """
    if not raw:
        return {}
    items: Any
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = raw  # iterable of [key, kind_value]
    out: dict[str, OutputKind] = {}
    for k, v in items:
        try:
            out[k] = OutputKind(v)
        except ValueError as e:
            raise ValueError(
                f"MLIPSpec.from_dict: unknown OutputKind value {v!r} for "
                f"output {k!r}; expected one of "
                f"{[k_.value for k_ in OutputKind]}"
            ) from e
    return out


class ForceStrategy(Enum):
    """How a model's forces are produced under a distributed forward.

    A single named choice from which the framework derives
    :attr:`CompilePolicy.forces_via_autograd` /
    :attr:`~CompilePolicy.consolidate_node_energy` /
    :attr:`~CompilePolicy.energy_output` — so a model declares intent once and
    cannot express an invalid combination.

    Members
    -------
    MODEL_INTERNAL
        The model computes forces inside its own forward; the framework runs the
        wrapper as-is and consolidates its outputs. E.g. UMA.
    FRAMEWORK_FROM_NODE_ENERGY
        The framework drives an energy-only forward returning per-node
        ``"atomic_energies"``, does the owned-only per-graph sum + cross-rank
        all-reduce, and takes ``forces = -dE/dx`` via autograd. The MACE pattern.
    FRAMEWORK_FROM_GLOBAL_ENERGY
        Same autograd force path, but the model consolidates the per-system
        ``"energy"`` inside its forward and the framework differentiates it
        as-is. The AIMNet2 pattern.
    """

    MODEL_INTERNAL = "model_internal"
    FRAMEWORK_FROM_NODE_ENERGY = "framework_from_node_energy"
    FRAMEWORK_FROM_GLOBAL_ENERGY = "framework_from_global_energy"


@dataclass(frozen=True)
class CompilePolicy:
    """How a model wants ``torch.compile`` driven under domain decomposition.

    ``static_shapes`` requests fixed-shape (capped) compilation so a compiled MD
    trajectory stays compiled across steps — the framework pads the graph to
    stable per-rank capacities. ``graph_padder`` is the
    :class:`GraphPadder` used for that padding; when ``None`` the framework uses
    the built-in COO ``edge_index`` padder (:class:`COOPadder`), so a standard
    MPNN declares nothing. ``force_strategy`` declares how forces are produced
    (see :class:`ForceStrategy`); the derived :attr:`forces_via_autograd` /
    :attr:`consolidate_node_energy` / :attr:`energy_output` properties follow
    from it.

    The policy is only the contract. Whether and how to compile is owned by
    :class:`DistributedModel` (constructed with ``compile=True`` /
    ``compile_kwargs=...``); the policy carries no compile switch of its own.
    """

    static_shapes: bool = True
    graph_padder: "GraphPadder | None" = None
    force_strategy: ForceStrategy = ForceStrategy.MODEL_INTERNAL

    @property
    def forces_via_autograd(self) -> bool:
        """True when the framework owns the energy-only forward + force autograd
        (any ``FRAMEWORK_FROM_*`` strategy)."""
        return self.force_strategy is not ForceStrategy.MODEL_INTERNAL

    @property
    def consolidate_node_energy(self) -> bool:
        """True when the model returns un-reduced per-node energy and the
        framework does the owned-only per-graph sum + all-reduce
        (``FRAMEWORK_FROM_NODE_ENERGY``)."""
        return self.force_strategy is ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY

    @property
    def energy_output(self) -> str:
        """The ``active_outputs`` key driven for the energy-only forward:
        ``"atomic_energies"`` for the per-node strategy, else ``"energy"``."""
        if self.force_strategy is ForceStrategy.FRAMEWORK_FROM_NODE_ENERGY:
            return "atomic_energies"
        return "energy"


@dataclass(frozen=True)
class MLIPSpec:
    """What an MLIP needs from the distributed framework.

    Wraps a :class:`DistributionSpec` and adds output-classification sets keyed
    by output name (``"forces"``, ``"stress"``, etc.). The framework reads it via
    ``BaseModelMixin.distribution_spec``.

    The recommended construction declares each output once via ``outputs=``::

        MLIPSpec(
            distribution=DistributionSpec(
                policy=HaloStoragePolicy(),
                custom_ops=(...),
            ),
            outputs={
                "energy": OutputSpec(OutputKind.PER_GRAPH),
                "forces": OutputSpec(OutputKind.PER_NODE),
                "stress": OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE),
            },
        )

    ``outputs`` is lowered in ``__post_init__`` onto the canonical fields
    (``output_kinds`` / ``owned_only_outputs`` / ``all_reduce_outputs``), which
    are what consolidation and serialization read.

    Parameters
    ----------
    distribution
        Required. A :class:`DistributionSpec` carrying the
        :class:`StoragePolicy` and escape-hatch tuples.
    owned_only_outputs
        Output keys whose per-atom values are already globally-correct on every
        rank (e.g. forces computed from replicated global state like Ewald/PME
        reciprocal ``S(k)``). Consolidation slices these to ``[:n_owned]`` rather
        than halo-reverse-summing.
    all_reduce_outputs
        Output keys whose value on each rank is a partial contribution that must
        be summed across the mesh to give the globally-correct value.
    output_kinds
        Per-output classification (:class:`OutputKind`) consumed by output
        consolidation. Outputs missing from this dict fall back to a shape-based
        heuristic (``shape[0] == n_padded`` ⇒ per-atom) and emit a one-shot
        warning.
    """

    distribution: DistributionSpec
    owned_only_outputs: frozenset[str] = field(default_factory=frozenset)
    all_reduce_outputs: frozenset[str] = field(default_factory=frozenset)
    output_kinds: dict[str, OutputKind] = field(default_factory=dict)
    # Whether per-system reductions (``mol_sum``-style scatters into
    # ``(n_systems, *F)`` accumulators) route through ``per_system_reduce``.
    system_reductions: bool = True
    # Name of a per-node energy output the framework reduces, owned-aware, into
    # the per-system ``"energy"`` on the eager halo path (owned-slice + per-graph
    # scatter + all-reduce). For electrostatics/dispersion wrappers (PME, Ewald,
    # DFTD3) whose per-atom energies are plain tensors that can't route through
    # ``per_system_reduce``: they emit raw per-atom energies under this key plus
    # a plain ``"energy"`` for the non-distributed case, and the framework
    # overrides ``"energy"`` with the owned-aware sum under decomposition.
    # ``None`` (default) means the wrapper owns its own per-system energy. The
    # compiled path does the same via
    # ``CompilePolicy.force_strategy=FRAMEWORK_FROM_NODE_ENERGY``.
    node_energy_key: str | None = None
    # ``outputs`` is the recommended declaration form (lowered onto the canonical
    # fields in ``__post_init__``); ``compile`` carries the :class:`CompilePolicy`
    # read by :class:`DistributedModel`. Both are excluded from eq/hash/serialization
    # — the canonical fields are the serialized source of truth, so a spec built
    # either way round-trips equal.
    outputs: "dict[str, OutputSpec] | None" = field(
        default=None, compare=False, hash=False
    )
    compile: "CompilePolicy | None" = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        # Lower ``outputs`` onto the three canonical fields additively, so
        # ``dataclasses.replace(preset, outputs={override})`` composes with the
        # preset's existing classification.
        if self.outputs:
            owned = frozenset(
                n for n, s in self.outputs.items() if s.reduce is Reduce.OWNED_ONLY
            )
            all_reduce = frozenset(
                n for n, s in self.outputs.items() if s.reduce is Reduce.ALL_REDUCE
            )
            kinds = dict(self.output_kinds)
            kinds.update(
                {
                    n: s.kind
                    for n, s in self.outputs.items()
                    if s.kind is not OutputKind.UNKNOWN
                }
            )
            object.__setattr__(
                self, "owned_only_outputs", self.owned_only_outputs | owned
            )
            object.__setattr__(
                self, "all_reduce_outputs", self.all_reduce_outputs | all_reduce
            )
            object.__setattr__(self, "output_kinds", kinds)
            # Clear ``outputs`` once consumed: the canonical fields are the truth.
            object.__setattr__(self, "outputs", None)

    def merge(self, other: "MLIPSpec") -> "MLIPSpec":
        """Merge two specs for a composed pipeline.

        Storage policy: merged via :func:`_merge_policies` (two halo policies
        keep halo and take the more permissive scatter/gather mode).
        Output-classification sets: union. Escape-hatch tuples: concatenated.
        ``system_reductions``: logical OR.
        """
        merged_policy = _merge_policies(self.distribution.policy, other.distribution.policy)
        merged_core = DistributionSpec(
            policy=merged_policy,
            custom_ops=self.distribution.custom_ops + other.distribution.custom_ops,
            third_party_helpers=(
                self.distribution.third_party_helpers + other.distribution.third_party_helpers
            ),
            # Union: the composed model promotes whatever either side promotes.
            shard_fields=tuple(
                dict.fromkeys(
                    self.distribution.shard_fields + other.distribution.shard_fields
                )
            ),
        )
        return MLIPSpec(
            distribution=merged_core,
            owned_only_outputs=self.owned_only_outputs | other.owned_only_outputs,
            all_reduce_outputs=self.all_reduce_outputs | other.all_reduce_outputs,
            system_reductions=self.system_reductions or other.system_reductions,
            node_energy_key=self.node_energy_key or other.node_energy_key,
        )

    def with_adapters(self, *adapters: Any) -> "MLIPSpec":
        """Return a copy with ``adapters`` added to the distribution.

        Each adapter is lowered onto ``custom_ops`` (``OpAdapter``) or
        ``third_party_helpers`` (everything else), composing with whatever the
        spec already declares. Lets a model take a preset and attach
        model-discovered adapters without rebuilding the spec by hand. All other
        settings are carried unchanged.
        """
        if not adapters:
            return self
        d = self.distribution
        new_core = DistributionSpec(
            policy=d.policy,
            custom_ops=d.custom_ops,
            third_party_helpers=d.third_party_helpers,
            shard_fields=d.shard_fields,
            adapters=adapters,  # lowered onto the split tuples in __post_init__
        )
        return MLIPSpec(
            distribution=new_core,
            owned_only_outputs=self.owned_only_outputs,
            all_reduce_outputs=self.all_reduce_outputs,
            output_kinds=dict(self.output_kinds),
            system_reductions=self.system_reductions,
            node_energy_key=self.node_energy_key,
            compile=self.compile,
        )

    def with_compile(self, policy: "CompilePolicy") -> "MLIPSpec":
        """Return a copy with the :class:`CompilePolicy` set (replacing any
        existing one). All other settings are carried unchanged."""
        return MLIPSpec(
            distribution=self.distribution,
            owned_only_outputs=self.owned_only_outputs,
            all_reduce_outputs=self.all_reduce_outputs,
            output_kinds=dict(self.output_kinds),
            system_reductions=self.system_reductions,
            node_energy_key=self.node_energy_key,
            compile=policy,
        )

    # ------------------------------------------------------------------
    # Serialization.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        Schema::

            {
              "version": 2,
              "core": <DistributionSpec.to_dict()>,
              "system_reductions": bool,
              "owned_only_outputs": [...],
              "all_reduce_outputs": [...],
              "output_kinds": [[key, kind], ...]
            }

        Op handles are encoded as ``"<namespace>::<name>"`` strings, resolved at
        load time provided the registering module has been imported.
        """
        return {
            "version": 2,
            "core": self.distribution.to_dict(),
            "system_reductions": self.system_reductions,
            "node_energy_key": self.node_energy_key,
            "owned_only_outputs": sorted(self.owned_only_outputs),
            "all_reduce_outputs": sorted(self.all_reduce_outputs),
            # Per-output classification, stored as a sorted list of
            # [key, kind_value] pairs so the JSON dump is deterministic and the
            # value side is a stable string rather than the OutputKind repr.
            "output_kinds": [
                [k, self.output_kinds[k].value] for k in sorted(self.output_kinds)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLIPSpec":
        """Inverse of :meth:`to_dict`.

        Resolves op qualnames — the caller must ensure the relevant
        op-registering modules have been imported first.
        """
        version = d.get("version")
        if version == 2:
            return cls(
                distribution=DistributionSpec.from_dict(d["core"]),
                system_reductions=d.get("system_reductions", True),
                node_energy_key=d.get("node_energy_key"),
                owned_only_outputs=frozenset(d.get("owned_only_outputs", [])),
                all_reduce_outputs=frozenset(d.get("all_reduce_outputs", [])),
                output_kinds=_decode_output_kinds(d.get("output_kinds", [])),
            )
        raise ValueError(
            f"MLIPSpec.from_dict: unsupported version {version}; "
            f"this build understands version=2."
        )

    def save(self, path: "str | Any") -> None:
        """Write the spec to ``path`` as JSON."""
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: "str | Any") -> "MLIPSpec":
        """Load a spec previously saved via :meth:`save`."""
        import json  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        return cls.from_dict(json.loads(Path(path).read_text()))

# ======================================================================
# Presets — one-liners for the model families we target directly.
# ======================================================================


_HALO_MLIP_POLICY = HaloStoragePolicy(
    scatter_mode="halo_correction",
    gather_mode="halo_read",
)


# Standard per-output declarations for any MLIP wrapper. PER_NODE = per-atom
# (forces, atomic_energies); PER_GRAPH = per-system (energy, stress); the default
# ``reduce=Reduce.NONE`` takes the per-kind consolidation. A wrapper with an extra
# output adds another ``OutputSpec``; one needing a cross-rank combine sets
# ``reduce=`` (e.g. UMA stress: ``OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE)``).
_STANDARD_MLIP_OUTPUTS: dict[str, OutputSpec] = {
    "energy": OutputSpec(OutputKind.PER_GRAPH),
    "forces": OutputSpec(OutputKind.PER_NODE),
    "stress": OutputSpec(OutputKind.PER_GRAPH),
    "atomic_energies": OutputSpec(OutputKind.PER_NODE),
}


SPEC_MPNN_HALO = MLIPSpec(
    distribution=DistributionSpec(policy=_HALO_MLIP_POLICY),
    outputs=dict(_STANDARD_MLIP_OUTPUTS),
)
"""Scatter-heavy MPNNs: MACE, NequIP, Allegro, ORB. Every edge-level update is a
``scatter_sum`` into per-atom features (the halo-correction handler keeps halo
rows in sync), plus a final per-graph ``scatter_sum`` on node energies (the
:func:`per_system_reduce` handler drops halo rows and all-reduces across ranks),
so stock ``model.forward`` produces globally-correct energy + forces with no
wrapper-side post-processing."""


# Graph-parallel outputs differ from the halo set only in their per-atom
# reduction: the per-layer node all-gather's reduce-scatter adjoint already sums
# each owned atom's cross-rank gradient, so forces come out globally-correct on
# their owning rank — passed through (``OWNED_ONLY``), never halo-reversed or
# divided by world size.
_GP_MLIP_OUTPUTS: dict[str, OutputSpec] = {
    "energy": OutputSpec(OutputKind.PER_GRAPH),
    "forces": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
    "stress": OutputSpec(OutputKind.PER_GRAPH),
    "atomic_energies": OutputSpec(OutputKind.PER_NODE, Reduce.OWNED_ONLY),
}


SPEC_MPNN_GP = MLIPSpec(
    distribution=DistributionSpec(
        policy=GraphParallelPolicy(),
        shard_fields=(),
    ),
    outputs=dict(_GP_MLIP_OUTPUTS),
)
"""Scatter-heavy MPNNs under the graph-parallel strategy: atoms split by a
balanced index range (no spatial halo), each rank owning the edges into its
nodes. The plain-interior promotion set (``shard_fields=()``) keeps the model on
plain owned-row tensors; per message-passing layer the framework all-gathers the
node features to a replicated tensor (reduce-scatter on the backward) so every
edge sees its source, and the final per-graph node-energy sum drops to owners and
all-reduces. The complement of :data:`SPEC_MPNN_HALO`: index-balanced and
locality-blind, it avoids the ghost overhead halo pays when the box approaches the
cutoff, at the cost of a full node all-gather each layer."""


SPEC_UMA_HALO = MLIPSpec(
    distribution=DistributionSpec(policy=_HALO_MLIP_POLICY),
    # UMA's Triton edge-permute kernels are registered at runtime by
    # UMAWrapper.distribution_spec; the module-level preset stays empty so
    # importers don't depend on fairchem.
    outputs=dict(_STANDARD_MLIP_OUTPUTS),
)
"""UMA (eSCN-family) via the halo storage policy, with fairchem graph parallel
disabled.

Each rank holds ``owned + halo`` rows and runs a standard full forward over them.
``UMAWrapper.distribution_spec`` layers in OpAdapters for the fused Triton
edge-permute kernels (edge→node aggregation gets per-layer halo correction); the
per-system reductions route through :func:`per_system_reduce`, and forces/stress
flow through plain autograd.
"""


SPEC_LJ_HALO = MLIPSpec(
    distribution=DistributionSpec(policy=_HALO_MLIP_POLICY),
    outputs=dict(_STANDARD_MLIP_OUTPUTS),
)
"""Lennard-Jones pair potential. Halo storage serves cross-rank neighbor pairs
from local halo copies after one halo exchange; forces come from direct
Warp-kernel writes (no autograd). The wrapper ends its forward with a
``scatter_add_`` aggregating per-atom energies to per-system totals — like
MACE's final ``scatter_sum`` — which ``system_reductions=True`` routes through
:func:`per_system_reduce` (slices halo rows off the source and all-reduces)."""


SPEC_EWALD_HALO = MLIPSpec(
    distribution=DistributionSpec(policy=_HALO_MLIP_POLICY),
    outputs=dict(_STANDARD_MLIP_OUTPUTS),
    # Reciprocal-stage-1 ops (partial structure factors) need owned-slice +
    # all-reduce; populated lazily in ``EwaldModelWrapper.distribution_spec`` to
    # avoid a warp import at spec-module load time.
)
"""Ewald summation: halo storage. Real-space pair interactions on halo-padded
inputs follow the standard halo path. Reciprocal-space dispatch is declarative
via ``custom_ops``: stage 1's handler does owned-slice + all-reduce so the
wrapper's ``forward`` stays distribution-agnostic. Per-atom energy scatter to
per-system totals routes through :func:`per_system_reduce`."""


SPEC_PME_HALO = MLIPSpec(
    distribution=DistributionSpec(policy=_HALO_MLIP_POLICY),
    outputs=dict(_STANDARD_MLIP_OUTPUTS),
    # Charge-spreading needs owned-slice + all-reduce so halo atoms don't
    # double-count the per-rank partial mesh. Custom_ops populated lazily in
    # ``PMEModelWrapper.distribution_spec``.
)
"""PME (Particle Mesh Ewald): halo storage. Real-space pair interactions on
halo-padded inputs follow the standard halo path; charge spreading gets an
owned-slice + all-reduce handler. Post-spread stages (FFT, Green's function,
IFFT, spline_gather, corrections) are replicated across ranks — they operate on
the all-reduced global mesh, so no dispatch is needed. Caveat: single-system
halo (``batch_idx=None``) hits a plain ``charges.sum()`` in
``pme_energy_corrections`` that double-counts halo rows; batched halo works
correctly via ``scatter_add_`` dispatch through :func:`per_system_reduce`."""


SPEC_DFTD3_HALO = MLIPSpec(
    distribution=DistributionSpec(policy=_HALO_MLIP_POLICY),
    outputs=dict(_STANDARD_MLIP_OUTPUTS),
)
"""DFT-D3(BJ) dispersion: halo storage, no global coupling. Coordination numbers,
C6 interpolation, and the two-body dispersion sum are all within-cutoff, so like
Lennard-Jones DFTD3 needs no cross-rank collective. The wrapper localizes
ShardTensor inputs for the Warp kernel, emits per-atom dispersion energies, and
reduces them with owned-slice + all-reduce; forces are direct per-atom.
One subtlety vs LJ: a ghost atom's coordination number (and its force term)
depend on the ghost's own neighbors, which reach a few angstrom beyond the
dispersion cutoff — so exact forces need a halo deeper than the cutoff
(``ghost_width >= cutoff + CN_counting_range``, set via ``skin``)."""
