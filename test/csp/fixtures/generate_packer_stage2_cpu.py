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

"""Regenerate the Stage 2 CPU numeric fixture from pinned source kernels.

Run from the toolkit checkout with these environment variables set:

    WARP_CACHE_DIR=/path/to/warp-cache NVALCHEMI_CSP_SOURCE=/path/to/nvalchemi_csp python test/csp/fixtures/generate_packer_stage2_cpu.py
"""

from __future__ import annotations

import os
import pprint
import subprocess
import sys
from pathlib import Path

import torch
import warp as wp

PINNED_SOURCE = "a47102925287bc016fa787a581daf091291c6a6f"
SOURCE_FILES = (
    "nvalchemi_csp/kernels/contact_overlap.py",
    "nvalchemi_csp/kernels/contact_kernels.py",
    "nvalchemi_csp/kernels/packer_relaxation.py",
    "nvalchemi_csp/kernels/symmetry_helpers.py",
    "nvalchemi_csp/kernels/util.py",
)


def _verify_source_checkout(source: Path) -> str:
    """Require both committed and imported source files to match the pin."""
    source_head = subprocess.check_output(  # noqa: S603
        ["git", "-C", str(source), "rev-parse", "HEAD"],  # noqa: S607
        text=True,
    ).strip()
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "-C",
            str(source),
            "diff",
            "--exit-code",
            PINNED_SOURCE,
            "HEAD",
            "--",
            *SOURCE_FILES,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    for relative_path in SOURCE_FILES:
        pinned_content = subprocess.check_output(  # noqa: S603, S607
            [  # noqa: S607
                "git",
                "-C",
                str(source),
                "show",
                f"{PINNED_SOURCE}:{relative_path}",
            ],
        )
        working_content = (source / relative_path).read_bytes()
        if working_content != pinned_content:
            raise RuntimeError(
                "Imported source file differs from pinned revision "
                f"{PINNED_SOURCE}: {relative_path}"
            )
    return source_head


def main() -> None:
    source = Path(os.environ["NVALCHEMI_CSP_SOURCE"]).resolve()
    source_head = _verify_source_checkout(source)
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_DIR"]
    sys.path.insert(0, str(source))
    from nvalchemi_csp.kernels import contact_overlap as source_contacts
    from nvalchemi_csp.kernels.packer_relaxation import relaxation_step

    # The pinned source contact wrapper validates CUDA-only public inputs. The
    # private kernels and cell-list implementation also support Warp CPU, so
    # narrow the wrapper validation for deterministic CPU fixture generation.
    source_contacts._validate_force_inputs = (
        lambda cp, mcp, ids, mp, centers, rotations, cells, inv, sym, ops, contact, radius: (
            ids.shape[0],
            ids.shape[1],
            ops.shape[1],
            int(mp[-1]),
            cp.device,
        )
    )

    conf = torch.tensor([[-0.45, 0, 0], [0.45, 0, 0], [0, 0, 0]], dtype=torch.float32)
    conf_ptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    conformer_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    molecule_atom_ptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    centers = torch.tensor(
        [[[0.11, 0.09, 0.12], [0.87, 0.10, 0.11]]], dtype=torch.float32
    )
    angle = torch.tensor(0.37)
    first_rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), torch.tensor(0.0))),
            torch.stack((torch.sin(angle), torch.cos(angle), torch.tensor(0.0))),
            torch.tensor([0.0, 0.0, 1.0]),
        )
    )
    rotations = (
        torch.stack((first_rotation, torch.eye(3))).reshape(1, 2, 3, 3).contiguous()
    )
    cells = torch.tensor(
        [[[4.4, 0, 0], [1.0, 4.2, 0], [0.5, 0.4, 4.6]]], dtype=torch.float32
    )
    inverses = torch.linalg.inv(cells).contiguous()
    symmetry_table = torch.tensor(
        [
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [-1, 0, 0, 0, -1, 0, 0, 0, -1, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    contact_distances = torch.tensor(
        [[1.8, 2.2, 2.4], [2.2, 1.8, 2.4], [2.4, 2.4, 2.0]], dtype=torch.float32
    )
    radius = torch.tensor([1.0, 1.0], dtype=torch.float32)
    multiop = source_contacts.contact_overlap_forces(
        conformer_positions=conf,
        mol_conf_atom_ptr=conf_ptr,
        conformer_ids=conformer_ids,
        mol_atom_ptr=molecule_atom_ptr,
        centers_frac=centers,
        rotations=rotations,
        cells=cells,
        symop_table=symmetry_table,
        symop_indices=torch.tensor([[0, 1]], dtype=torch.int32),
        contact_map=contact_distances,
        molecule_radius=radius,
        cell_inverses=inverses,
    )
    contact_expected = {
        name: getattr(multiop, name).tolist()
        for name in ("forces", "torques", "virial", "total_overlap", "max_overlap")
    }

    step_cells, step_inverses = cells.clone(), inverses.clone()
    step_centers, step_rotations = centers.clone(), rotations.clone()
    steps = torch.zeros(1, dtype=torch.int32)
    reference = torch.tensor([torch.linalg.det(cells).item()], dtype=torch.float32)
    relaxation_step(
        active_mask=torch.ones(1, dtype=torch.int32),
        conformer_positions=conf,
        conformer_atom_offsets=conf_ptr,
        conformer_ids=conformer_ids,
        molecule_atom_offsets=molecule_atom_ptr,
        forces=multiop.forces,
        torques=multiop.torques,
        max_overlap=multiop.max_overlap,
        virial=multiop.virial,
        cells=step_cells,
        cell_inverses=step_inverses,
        centers_frac=step_centers,
        rotations=step_rotations,
        space_groups=torch.tensor([2], dtype=torch.int32),
        reference_volumes=reference,
        steps=steps,
        expanded_atom_count=6,
        step_scale=0.12,
        max_atom_step=0.3,
        cell_step_scale=0.01,
        max_cell_strain=0.003,
        external_pressure=1.0e-5,
    )
    one_step = {
        name: tensor.tolist()
        for name, tensor in (
            ("cells", step_cells),
            ("inverse_cells", step_inverses),
            ("centers", step_centers),
            ("rotations", step_rotations),
            ("steps", steps),
        )
    }

    p1 = source_contacts.contact_overlap_forces(
        conformer_positions=conf,
        mol_conf_atom_ptr=conf_ptr,
        conformer_ids=conformer_ids,
        mol_atom_ptr=molecule_atom_ptr,
        centers_frac=centers,
        rotations=rotations,
        cells=cells,
        symop_table=symmetry_table[:1].contiguous(),
        symop_indices=torch.tensor([[0]], dtype=torch.int32),
        contact_map=contact_distances,
        molecule_radius=radius,
        cell_inverses=inverses,
    )
    p1_expected = {
        name: getattr(p1, name).tolist()
        for name in ("forces", "torques", "virial", "total_overlap", "max_overlap")
    }
    output = {
        "provenance": {
            "source_commit": PINNED_SOURCE,
            "source_checkout_head": source_head,
            "verified_identical_source_files": list(SOURCE_FILES),
            "generation_command": (
                "WARP_CACHE_DIR=/path/to/warp-cache "
                "NVALCHEMI_CSP_SOURCE=/path/to/nvalchemi_csp "
                "python test/csp/fixtures/generate_packer_stage2_cpu.py"
            ),
            "scope": "fixed-state contact outputs and one source relaxation step; CPU Warp kernels",
        },
        "tolerances": {
            "contact_atol": 2.0e-4,
            "contact_rtol": 2.0e-4,
            "step_atol": 5.0e-4,
            "step_rtol": 5.0e-4,
        },
        "multiop_skew": {"contacts": contact_expected, "one_step": one_step},
        "p1_skew": {"contacts": p1_expected},
    }
    fixture_path = Path(__file__).with_name("packer_stage2_cpu.py")
    license_path = Path(__file__).parents[2] / "_license" / "header.txt"
    license_header = license_path.read_text().rstrip()
    fixture_text = (
        f"{license_header}\n\n"
        '"""Pinned Stage 2 CPU numerical reference fixture."""\n\n'
        "SOURCE_REFERENCE = "
        + pprint.pformat(output, sort_dicts=False, width=88)
        + "\n"
    )
    fixture_path.write_text(fixture_text)
    subprocess.run(  # noqa: S603 -- formatter invocation has fixed arguments.
        [sys.executable, "-m", "ruff", "format", str(fixture_path)], check=True
    )
    print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
