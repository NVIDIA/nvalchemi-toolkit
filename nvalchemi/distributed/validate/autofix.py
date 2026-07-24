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

"""Auto-fix rule engine for ``trace_and_validate``.

When the initial inferred spec fails validation, the engine tries a
small corpus of rule-based mutations — each rule is a predicate +
spec transformation that captures one observed bug pattern. Rules are
tried in order of confidence; the first one whose predicate matches
and whose candidate spec hasn't been tried before becomes the next
attempt.

Each rule is independently testable: feed it a ``(spec, last_attempt)``
pair and check the proposed mutation. See
:mod:`test.distributed.test_validate_autofix`.
"""

from __future__ import annotations

from typing import Callable

from nvalchemi.distributed.spec import MLIPSpec, OpAdapter
from nvalchemi.distributed.validate.types import Attempt

__all__ = [
    "_next_fix_candidate",
    "_spec_signature",
    "_op_signature",
    "_rule_halo_to_local",
    "_rule_drop_extra_all_reduce",
    "_rule_add_per_graph_autograd_to_all_reduce",
    "_suspect_op_summary",
    "_RULES",
]


def _next_fix_candidate(
    current_spec: MLIPSpec, attempts: list[Attempt]
) -> tuple[MLIPSpec, str] | None:
    """Pick the next spec mutation to try. Returns ``(new_spec,
    rationale)`` or ``None`` when no rule applies.

    Rules are tried in order of confidence; each rule that hasn't
    already been attempted in this run gets a chance.
    """
    last = attempts[-1]
    tried_specs = {_spec_signature(a.spec) for a in attempts}

    for rule_name, rule_fn in _RULES:
        candidate = rule_fn(current_spec, last)
        if candidate is None:
            continue
        if _spec_signature(candidate) in tried_specs:
            continue
        return candidate, rule_name
    return None


def _spec_signature(spec: MLIPSpec) -> tuple:
    """Hashable signature for dedup of attempted specs."""
    from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
        policy_to_dict,
    )

    return (
        # The storage policy is a frozen dataclass; serialize to a deterministic
        # tuple via its dict representation.
        tuple(sorted(policy_to_dict(spec.distribution.policy).items())),
        spec.system_reductions,
        tuple(sorted(spec.owned_only_outputs)),
        tuple(sorted(spec.all_reduce_outputs)),
        tuple(_op_signature(o) for o in spec.distribution.custom_ops),
    )


def _op_signature(op_spec: OpAdapter) -> tuple:
    return (
        getattr(op_spec.op, "_schema", None) and op_spec.op._schema.name,
        op_spec.gather_inputs,
        op_spec.scatter_outputs,
        op_spec.owned_slice_inputs,
        op_spec.all_reduce_outputs,
    )


# Each rule: (description, fn). fn returns the candidate next spec or None.


def _rule_halo_to_local(spec: MLIPSpec, last: Attempt) -> MLIPSpec | None:
    """If the model used ``scatter='halo_correction'`` and the multi-rank
    output diverges, try ``scatter='local'`` — the UMA case (eSCN
    backbone is halo-unaware, edge_index covers the full graph, so
    halo_reverse double-counts cross-rank contributions).

    Predicate: scatter is currently halo_correction AND the failure has
    a non-trivial diff on at least one output AND a halo_correction
    handler fired during the multi-rank run.
    """
    from nvalchemi.distributed._core.storage_policy import (  # noqa: PLC0415
        HaloStoragePolicy,
    )
    from nvalchemi.distributed.spec import replace_policy  # noqa: PLC0415

    policy = spec.distribution.policy
    if (
        not isinstance(policy, HaloStoragePolicy)
        or policy.scatter_mode != "halo_correction"
    ):
        return None
    if last.handler_counts.get("halo_scatter_correction", 0) == 0:
        return None
    if max(last.max_abs_diff.values(), default=0.0) < 1e-6:
        return None
    return replace_policy(spec, scatter="local")


def _rule_add_per_graph_autograd_to_all_reduce(
    spec: MLIPSpec, last: Attempt
) -> MLIPSpec | None:
    """Promote a PER_GRAPH autograd output with a significant diff into
    ``all_reduce_outputs``.

    Signal: the output's ``output_kinds`` is PER_GRAPH, the wrapper's
    ``model_config.autograd_outputs`` includes it (we infer this
    indirectly from the spec — see implementation note), it has a
    non-trivial relative diff, and it isn't already in
    ``all_reduce_outputs``.

    Why this is the right fix. The strain-trick stress (and any
    similar per-graph autograd derivative) is:

    1. Computed via ``autograd.grad(replicated_E.sum(), strain)``.
    2. Backward through ``per_system_reduce`` all-reduces the upstream
       grad → each rank's local autograd produces ``world_size x
       per_rank_strain_contribution``.
    3. Consolidation's default for non-per-atom autograd does
       ``/world_size``, producing ``per_rank_strain_contribution`` —
       the *partial* contribution from this rank's owned atoms.

    The global stress is the sum across ranks. So the fix is
    ``/world_size`` (already done by consolidation) **plus**
    ``all_reduce(SUM)`` — which is exactly what
    ``all_reduce_outputs`` triggers for autograd-derived keys.

    Predicate gates:

    * ``spec.output_kinds[key] is OutputKind.PER_GRAPH``.
    * ``key`` is not already in ``spec.all_reduce_outputs``.
    * ``last.max_rel_diff[key]`` exceeds a small threshold (avoid
      promoting noise-floor diffs into spec changes).
    * Energy is ruled out separately because ``per_system_reduce``
      already replicates it in forward — so its consolidation diff
      should be zero before this rule fires; if it isn't, that's a
      different bug class.
    """
    import dataclasses  # noqa: PLC0415

    from nvalchemi.distributed.output_kinds import OutputKind  # noqa: PLC0415

    candidates: set[str] = set()
    for key, kind in spec.output_kinds.items():
        if kind is not OutputKind.PER_GRAPH:
            continue
        if key in spec.all_reduce_outputs:
            continue
        # Energy goes through per_system_reduce's forward all_reduce
        # already; if its diff is large, the cause is elsewhere
        # (typically halo coverage). Skip it so this rule doesn't
        # mask genuine bugs.
        if key == "energy":
            continue
        rel = last.max_rel_diff.get(key, 0.0)
        if rel < 1e-3:
            continue
        candidates.add(key)

    if not candidates:
        return None

    return dataclasses.replace(
        spec, all_reduce_outputs=spec.all_reduce_outputs | frozenset(candidates)
    )


def _rule_drop_extra_all_reduce(spec: MLIPSpec, last: Attempt) -> MLIPSpec | None:
    """If multi_E ~= ref_E x world_size on a key declared in
    ``all_reduce_outputs``, that key is being reduced an extra time
    (the wrapper's internals already replicate it). Drop the key.
    """
    import dataclasses  # noqa: PLC0415

    if not spec.all_reduce_outputs:
        return None
    # Heuristic: any output's abs diff is ~ ref magnitude.
    # Caller has access only to the diff dict, not the absolute ref;
    # treat very-large diffs as the signature.
    keys_to_drop = {
        k
        for k in spec.all_reduce_outputs
        if last.max_rel_diff.get(k, 0.0) > 0.5  # signal of duplication
    }
    if not keys_to_drop:
        return None
    return dataclasses.replace(
        spec, all_reduce_outputs=spec.all_reduce_outputs - keys_to_drop
    )


_RULES: list[tuple[str, Callable[[MLIPSpec, Attempt], MLIPSpec | None]]] = [
    (
        "halo_correction → local (suspected halo-unaware backbone double-count)",
        _rule_halo_to_local,
    ),
    (
        "add per-graph autograd output to all_reduce_outputs "
        "(strain-trick stress and similar per-rank-partial gradients)",
        _rule_add_per_graph_autograd_to_all_reduce,
    ),
    (
        "drop key from all_reduce_outputs (suspected extra reduction)",
        _rule_drop_extra_all_reduce,
    ),
]


def _suspect_op_summary(handler_counts: dict[str, int]) -> str:
    if not handler_counts:
        return "no handler firings observed; check that the wrapper sees ShardTensor inputs."
    top = sorted(handler_counts.items(), key=lambda kv: -kv[1])[:3]
    parts = [f"{name}×{count}" for name, count in top]
    return "Top handler firings: " + ", ".join(parts)
