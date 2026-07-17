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

"""Every preset declares :attr:`MLIPSpec.output_kinds` covering the standard
MLIP output set.

The shape-based heuristic in :mod:`nvalchemi.distributed.output_consolidation`
falls back when an output is undeclared, but emits a warning. This test
guards that the production presets stay declared so the warning is dead
code in normal operation. Adding a new preset that uses ``MLIPSpec``
without a corresponding declaration causes the relevant assertion to
fail here; declare on the preset (typically via
``outputs=dict(_STANDARD_MLIP_OUTPUTS)``) to silence.
"""

from __future__ import annotations

import pytest

from nvalchemi.distributed.output_kinds import OutputKind
from nvalchemi.distributed.spec import (
    SPEC_EWALD_HALO,
    SPEC_LJ_HALO,
    SPEC_MPNN_HALO,
    SPEC_PME_HALO,
    SPEC_UMA_HALO,
)

# Outputs that an MLIP wrapper might emit. ``energy`` / ``forces`` /
# ``stress`` cover all production wrappers; ``atomic_energies`` is
# emitted by some (LJ) as an intermediate. A preset is allowed to
# declare more than this — the assertion only requires *at minimum*
# that the standard set is covered.
_STANDARD_OUTPUTS: frozenset[str] = frozenset(
    {"energy", "forces", "stress", "atomic_energies"}
)


@pytest.mark.parametrize(
    "spec_name,spec",
    [
        ("SPEC_MPNN_HALO", SPEC_MPNN_HALO),
        ("SPEC_UMA_HALO", SPEC_UMA_HALO),
        ("SPEC_LJ_HALO", SPEC_LJ_HALO),
        ("SPEC_EWALD_HALO", SPEC_EWALD_HALO),
        ("SPEC_PME_HALO", SPEC_PME_HALO),
    ],
)
def test_preset_declares_standard_output_kinds(spec_name, spec) -> None:
    """Every preset MLIPSpec declares output_kinds for the standard
    MLIP output set (``energy``, ``forces``, ``stress``,
    ``atomic_energies``). Without a declaration, consolidation falls
    back to the shape heuristic and emits a one-shot warning."""
    declared = set(spec.output_kinds)
    missing = _STANDARD_OUTPUTS - declared
    assert not missing, (
        f"{spec_name}.output_kinds is missing {sorted(missing)}. "
        f"Add to the preset (e.g. via "
        f"``outputs=dict(_STANDARD_MLIP_OUTPUTS)`` in spec.py) "
        f"to ensure consolidation has explicit guidance and avoid the "
        f"shape-heuristic fallback warning at runtime."
    )


@pytest.mark.parametrize(
    "spec_name,spec",
    [
        ("SPEC_MPNN_HALO", SPEC_MPNN_HALO),
        ("SPEC_UMA_HALO", SPEC_UMA_HALO),
        ("SPEC_LJ_HALO", SPEC_LJ_HALO),
        ("SPEC_EWALD_HALO", SPEC_EWALD_HALO),
        ("SPEC_PME_HALO", SPEC_PME_HALO),
    ],
)
def test_preset_output_kinds_match_expected_shape(spec_name, spec) -> None:
    """Sanity: ``energy`` / ``stress`` are PER_GRAPH; ``forces`` /
    ``atomic_energies`` are PER_NODE. Catches accidental kind swaps
    (e.g. declaring ``forces=PER_GRAPH``).
    """
    expected = {
        "energy": OutputKind.PER_GRAPH,
        "stress": OutputKind.PER_GRAPH,
        "forces": OutputKind.PER_NODE,
        "atomic_energies": OutputKind.PER_NODE,
    }
    for key, expected_kind in expected.items():
        if key not in spec.output_kinds:
            continue
        assert spec.output_kinds[key] is expected_kind, (
            f"{spec_name}.output_kinds[{key!r}] = "
            f"{spec.output_kinds[key]!r}, expected {expected_kind!r}"
        )


def test_output_kind_values_round_trip_through_serialization() -> None:
    """Sanity: ``OutputKind`` values stay stable through
    :meth:`MLIPSpec.to_dict` / :meth:`from_dict`. The validator's
    ``mp.spawn`` path relies on this."""
    from nvalchemi.distributed.spec import MLIPSpec

    d = SPEC_MPNN_HALO.to_dict()
    loaded = MLIPSpec.from_dict(d)
    assert loaded.output_kinds == SPEC_MPNN_HALO.output_kinds


def test_undeclared_output_falls_back_to_heuristic() -> None:
    """An output key absent from ``output_kinds`` should *not* cause an
    error. Consolidation falls back to the shape heuristic and emits a
    warning, so wrappers that emit unusual debug outputs stay
    functional."""
    from nvalchemi.distributed.output_kinds import OutputKind

    spec = SPEC_MPNN_HALO
    assert spec.output_kinds.get("some_random_debug_output") is None
    # The actual heuristic-fallback path is exercised in
    # output_consolidation; here we only verify the spec doesn't
    # raise on access. Decoupling the two assertions lets this test
    # run on CPU without a halo metadata mock.
    assert OutputKind.UNKNOWN.value == "unknown"
