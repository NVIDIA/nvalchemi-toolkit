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
"""Private exact graph-orbit classification for CSP topology helpers."""

from __future__ import annotations


def _refine_colors(atomic_numbers: list[int], adjacency: list[list[bool]]) -> list[int]:
    """Refine atom labels by neighbor-color multisets until stable."""
    labels = {number: index for index, number in enumerate(sorted(set(atomic_numbers)))}
    colors = [labels[number] for number in atomic_numbers]
    while True:
        signatures = [
            (
                colors[index],
                tuple(sorted(colors[j] for j, edge in enumerate(row) if edge)),
            )
            for index, row in enumerate(adjacency)
        ]
        unique = {
            signature: color for color, signature in enumerate(sorted(set(signatures)))
        }
        refined = [unique[signature] for signature in signatures]
        # Color numbers may be renumbered between rounds; partition equality is
        # the relevant fixed-point condition.
        same_partition = all(
            (colors[i] == colors[j]) == (refined[i] == refined[j])
            for i in range(len(colors))
            for j in range(len(colors))
        )
        colors = refined
        if same_partition:
            return colors


def _find_mapping(
    source_adjacency: list[list[bool]],
    target_adjacency: list[list[bool]],
    candidates: list[list[int]],
    initial_mapping: list[int] | None = None,
) -> list[int] | None:
    """Find an exact graph mapping from ordered per-source candidates."""
    count = len(source_adjacency)
    if count != len(target_adjacency) or len(candidates) != count:
        return None

    mapping = [-1] * count if initial_mapping is None else initial_mapping.copy()
    if len(mapping) != count:
        return None
    used = [False] * count
    mapped_count = 0
    for source, target in enumerate(mapping):
        if target < 0:
            continue
        if target >= count or used[target] or target not in candidates[source]:
            return None
        if any(
            mapping[prior] >= 0
            and source_adjacency[source][prior]
            != target_adjacency[target][mapping[prior]]
            for prior in range(source)
        ):
            return None
        used[target] = True
        mapped_count += 1

    def search() -> bool:
        """Complete the partial mapping with minimum-options source choice."""
        nonlocal mapped_count
        if mapped_count == count:
            return True
        best_source = -1
        best_options: list[int] | None = None
        for source in range(count):
            if mapping[source] >= 0:
                continue
            options = [
                target
                for target in candidates[source]
                if not used[target]
                and all(
                    source_adjacency[source][prior]
                    == target_adjacency[target][mapping[prior]]
                    for prior in range(count)
                    if mapping[prior] >= 0
                )
            ]
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_source, best_options = source, options
                if len(options) == 1:
                    break
        if best_options is None:
            return False
        for target in best_options:
            mapping[best_source] = target
            used[target] = True
            mapped_count += 1
            if search():
                return True
            mapped_count -= 1
            mapping[best_source] = -1
            used[target] = False
        return False

    return mapping if search() else None


def _rooted_isomorphic(
    adjacency: list[list[bool]], colors: list[int], root: int, target: int
) -> bool:
    """Whether an automorphism preserving refined colors maps root to target."""
    if colors[root] != colors[target]:
        return False
    count = len(colors)
    candidates = [
        [
            destination
            for destination in range(count)
            if colors[source] == colors[destination]
        ]
        for source in range(count)
    ]
    initial_mapping = [-1] * count
    initial_mapping[root] = target
    return _find_mapping(adjacency, adjacency, candidates, initial_mapping) is not None


def topological_atom_types(
    atomic_numbers: list[int], adjacency: list[list[bool]]
) -> list[int]:
    """Return local IDs for exact label-preserving automorphism orbits."""
    count = len(atomic_numbers)
    if count == 0:
        return []
    colors = _refine_colors(atomic_numbers, adjacency)
    atom_types = [-1] * count
    next_type = 0
    for root in range(count):
        if atom_types[root] >= 0:
            continue
        atom_types[root] = next_type
        for target in range(root + 1, count):
            if atom_types[target] < 0 and _rooted_isomorphic(
                adjacency, colors, root, target
            ):
                atom_types[target] = next_type
        next_type += 1
    return atom_types


def connected_components(adjacency: list[list[bool]]) -> list[list[int]]:
    """Return graph components in order of their first atom."""
    count = len(adjacency)
    seen = [False] * count
    components: list[list[int]] = []
    for root in range(count):
        if seen[root]:
            continue
        seen[root] = True
        component = [root]
        for atom in component:
            for neighbor, edge in enumerate(adjacency[atom]):
                if edge and not seen[neighbor]:
                    seen[neighbor] = True
                    component.append(neighbor)
        components.append(component)
    return components


def find_isomorphism(
    source_numbers: list[int],
    source_adjacency: list[list[bool]],
    target_numbers: list[int],
    target_adjacency: list[list[bool]],
) -> list[int] | None:
    """Map source vertices to target vertices for an exact labelled isomorphism."""
    count = len(source_numbers)
    if count != len(target_numbers):
        return None
    if count == 0:
        return []
    source_degrees = [sum(row) for row in source_adjacency]
    target_degrees = [sum(row) for row in target_adjacency]
    candidates = [
        [
            target
            for target in range(count)
            if source_numbers[source] == target_numbers[target]
            and source_degrees[source] == target_degrees[target]
        ]
        for source in range(count)
    ]
    if any(not options for options in candidates):
        return None

    return _find_mapping(source_adjacency, target_adjacency, candidates)
