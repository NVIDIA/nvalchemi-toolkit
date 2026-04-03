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
"""DFT-D3(BJ) dispersion correction model wrapper.

Wraps the ``nvalchemiops`` DFT-D3(BJ) dispersion interaction as a
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model, ready to
drop into any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine.

Usage
-----
::

    from nvalchemi.models.dftd3 import DFTD3ModelWrapper
    from nvalchemi.dynamics.hooks import NeighborListHook

    # PBE-D3(BJ) parameters (Grimme 2010)
    model = DFTD3ModelWrapper(
        a1=0.4289,
        a2=4.4407,   # Bohr
        s8=0.7875,
    )

    nl_hook = NeighborListHook(model.model_card.neighbor_config)
    dynamics.register_hook(nl_hook)
    dynamics.model = model

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so :attr:`~ModelCard.forces_via_autograd` is ``False``.
* Positions and cell are converted from Å → Bohr before the kernel call
  and outputs are converted back to eV/Å.
* D3 parameters are loaded from a ``.pt`` cache file (default location
  ``~/.cache/nvalchemiops/dftd3_parameters.pt``).  When the file is absent,
  :func:`load_dftd3_params` calls :func:`extract_dftd3_parameters` which
  downloads the Fortran reference archive from the Grimme group website,
  parses it in-memory, and caches the result automatically.
* Stress/virial computation (needed for NPT/NPH) is available via
  ``model_config.compute_stresses = True``.
"""

from __future__ import annotations

import io
import re
import tarfile
from collections import OrderedDict
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._ops.neighbor_filter import prepare_neighbors_for_model
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)

__all__ = [
    "DFTD3ModelWrapper",
    "extract_dftd3_parameters",
    "save_dftd3_parameters",
    "load_dftd3_params",
]

# ---------------------------------------------------------------------------
# Unit conversion constants  (CODATA 2022)
# ---------------------------------------------------------------------------
BOHR_TO_ANGSTROM: float = 0.529177210544
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV: float = 27.211386245981

# ---------------------------------------------------------------------------
# DFT-D3 reference parameter source
# ---------------------------------------------------------------------------
# Official Grimme group distribution of the Fortran reference implementation.
# MD5 is checked after download to guard against silent corruption.
_DFTD3_TGZ_URL: str = (
    "https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz"
)
_DFTD3_TGZ_MD5: str = "a76c752e587422c239c99109547516d2"

_DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "nvalchemiops"
_DEFAULT_PARAM_FILE: Path = _DEFAULT_CACHE_DIR / "dftd3_parameters.pt"


# ---------------------------------------------------------------------------
# Private helpers for Fortran source parsing
# ---------------------------------------------------------------------------


def _download_and_extract_tgz(url: str) -> dict[str, str]:
    """Download a .tgz archive and return a dict of {basename: text} for .f/.F files."""
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content_bytes = response.content
    hasher = md5(usedforsecurity=False)
    hasher.update(content_bytes)
    if hasher.hexdigest() != _DFTD3_TGZ_MD5:
        raise ValueError(
            f"MD5 checksum mismatch for downloaded archive.\n"
            f"Expected: {_DFTD3_TGZ_MD5}\n"
            f"Got:      {hasher.hexdigest()}\n"
            "The archive may have changed. Provide a local dftd3_ref directory."
        )

    extracted: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(content_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith((".f", ".F")):
                fobj = tar.extractfile(member)
                if fobj is not None:
                    extracted[Path(member.name).name] = fobj.read().decode(
                        "utf-8", errors="ignore"
                    )
    return extracted


def _find_fortran_array(content: str, var_name: str) -> np.ndarray:
    """Parse a Fortran ``data var_name / ... /`` block and return float64 values."""
    lines = content.splitlines()
    in_block = False
    block_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!") or stripped.lower().startswith("c "):
            continue
        if not in_block:
            if re.match(rf"^\s*data\s+{var_name}\s*/\s*", line, re.IGNORECASE):
                in_block = True
                block_lines.append(line)
        else:
            block_lines.append(line)
            if "/" in line and not line.strip().startswith("!"):
                break

    if not block_lines:
        raise ValueError(f"Variable '{var_name}' not found in Fortran source")

    data_str = " ".join(block_lines)
    match = re.search(
        rf"data\s+{var_name}\s*/\s*(.*?)\s*/", data_str, re.DOTALL | re.IGNORECASE
    )
    if not match:
        raise ValueError(f"Failed to parse '{var_name}'")

    block = match.group(1)
    clean_lines = []
    for ln in block.split("\n"):
        if "!" in ln:
            ln = ln[: ln.index("!")]
        clean_lines.append(ln)
    block = " ".join(clean_lines)

    numbers = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", block)
    return np.array([float(n.replace("_wp", "")) for n in numbers], dtype=np.float64)


def _parse_pars_array(content: str) -> np.ndarray:
    """Parse the ``pars`` array from pars.f into an (N, 5) float64 array."""
    values: list[float] = []
    in_section = False

    for line in content.splitlines():
        if "pars(" in line.lower() and "=(" in line:
            in_section = True
        if not in_section:
            continue
        if "/)" in line:
            in_section = False
        if "!" in line:
            line = line[: line.index("!")]
        line = re.sub(r"pars\(", " ", line, flags=re.IGNORECASE)
        line = line.replace("=(/", " ").replace("/)", " ").replace(":", " ")
        numbers = re.findall(r"[-+]?\d+\.\d+[eEdD][-+]?\d+", line)
        values.extend(float(n.replace("D", "e").replace("d", "e")) for n in numbers)

    arr = np.array(values, dtype=np.float64)
    n = len(arr) // 5
    return arr[: n * 5].reshape(n, 5)


def _limit(encoded: int) -> tuple[int, int]:
    """Decode Fortran element encoding: returns (atomic_number, cn_index)."""
    atom, cn_idx = encoded, 1
    while atom > 100:
        atom -= 100
        cn_idx += 1
    return atom, cn_idx


def _build_c6_arrays(
    pars_records: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build c6ab [95,95,5,5] and cn_ref [95,95,5,5] from pars records."""
    c6ab = np.zeros((95, 95, 5, 5), dtype=np.float32)
    cn_ref = np.full((95, 95, 5, 5), -1.0, dtype=np.float32)
    cn_values: dict[int, dict[int, float]] = {e: {} for e in range(95)}

    for record in pars_records:
        c6_val, z_i_enc, z_j_enc, cn_i, cn_j = record
        iat, iadr = _limit(int(z_i_enc))
        jat, jadr = _limit(int(z_j_enc))
        if not (1 <= iat <= 94 and 1 <= jat <= 94):
            continue
        if not (1 <= iadr <= 5 and 1 <= jadr <= 5):
            continue
        ia, ja = iadr - 1, jadr - 1
        c6ab[iat, jat, ia, ja] = c6_val
        c6ab[jat, iat, ja, ia] = c6_val
        cn_values[iat].setdefault(ia, cn_i)
        cn_values[jat].setdefault(ja, cn_j)

    for elem in range(1, 95):
        for partner in range(1, 95):
            for ci in range(5):
                if ci in cn_values[elem]:
                    cn_ref[elem, partner, ci, :] = cn_values[elem][ci]

    return c6ab, cn_ref


# ---------------------------------------------------------------------------
# Public parameter extraction utilities
# ---------------------------------------------------------------------------


def extract_dftd3_parameters(
    dftd3_ref_dir: Path | str | None = None,
) -> dict[str, torch.Tensor]:
    """Extract DFT-D3 parameters from Fortran source and return as tensors.

    Either reads local Fortran source files (``dftd3.f`` + ``pars.f``) or
    downloads them in-memory from the Grimme group website, parses the raw
    Fortran data arrays, and returns the four parameter tensors required by
    the ``nvalchemiops`` DFT-D3 kernels.

    Parameters
    ----------
    dftd3_ref_dir : Path or str or None, optional
        Directory containing ``dftd3.f`` and ``pars.f`` from the reference
        Fortran implementation.  When ``None`` (default) the archive is
        downloaded automatically from
        ``https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz``
        and extracted in-memory — no files are written to disk.

    Returns
    -------
    dict[str, torch.Tensor]
        ``"rcov"``   — covalent radii, shape ``[95]``, Bohr (float32)
        ``"r4r2"``   — ⟨r⁴⟩/⟨r²⟩ expectation values, shape ``[95]`` (float32)
        ``"c6ab"``   — C6 reference coefficients, shape ``[95, 95, 5, 5]`` (float32)
        ``"cn_ref"`` — CN reference grid, shape ``[95, 95, 5, 5]`` (float32)

        Index 0 is reserved for padding; valid atomic numbers are 1–94.

    Raises
    ------
    FileNotFoundError
        If *dftd3_ref_dir* is given but the expected files are absent.
    RuntimeError
        If the download or archive extraction fails.
    ValueError
        If Fortran source parsing fails or the MD5 checksum does not match.
    """
    if dftd3_ref_dir is not None:
        dftd3_ref_dir = Path(dftd3_ref_dir)
        if not dftd3_ref_dir.exists():
            raise FileNotFoundError(f"dftd3_ref_dir not found: {dftd3_ref_dir}")
        dftd3_f = dftd3_ref_dir / "dftd3.f"
        pars_f = dftd3_ref_dir / "pars.f"
        for p in (dftd3_f, pars_f):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
        print(f"Reading DFT-D3 source files from: {dftd3_ref_dir}")
        dftd3_content = dftd3_f.read_text()
        pars_content = pars_f.read_text()
    else:
        print(f"Downloading DFT-D3 archive from {_DFTD3_TGZ_URL} ...")
        try:
            files = _download_and_extract_tgz(_DFTD3_TGZ_URL)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download DFT-D3 archive: {exc}\n"
                "Check your internet connection or provide a local dftd3_ref_dir."
            ) from exc
        for name in ("dftd3.f", "pars.f"):
            if name not in files:
                raise RuntimeError(f"'{name}' not found in downloaded archive.")
        dftd3_content = files["dftd3.f"]
        pars_content = files["pars.f"]
        print("  Download and extraction complete.")

    print("Parsing Fortran source files ...")
    r2r4_94 = _find_fortran_array(dftd3_content, "r2r4")
    rcov_94 = _find_fortran_array(dftd3_content, "rcov")
    pars_records = _parse_pars_array(pars_content)

    r4r2 = np.zeros(95, dtype=np.float32)
    r4r2[1:95] = r2r4_94.astype(np.float32)
    rcov = np.zeros(95, dtype=np.float32)
    rcov[1:95] = rcov_94.astype(np.float32)
    c6ab, cn_ref = _build_c6_arrays(pars_records)
    print("  Parsing complete.")

    return {
        "rcov": torch.from_numpy(rcov),
        "r4r2": torch.from_numpy(r4r2),
        "c6ab": torch.from_numpy(c6ab),
        "cn_ref": torch.from_numpy(cn_ref),
    }


def save_dftd3_parameters(
    parameters: dict[str, torch.Tensor],
    param_file: Path | str | None = None,
) -> Path:
    """Save extracted DFT-D3 parameters to disk.

    Writes the parameter dictionary produced by :func:`extract_dftd3_parameters`
    to a ``.pt`` file so that :func:`load_dftd3_params` can load it without
    re-downloading the Fortran sources.

    Parameters
    ----------
    parameters : dict[str, torch.Tensor]
        Parameter dict with keys ``"rcov"``, ``"r4r2"``, ``"c6ab"``, ``"cn_ref"``.
    param_file : Path or str or None, optional
        Destination path.  Defaults to
        ``~/.cache/nvalchemiops/dftd3_parameters.pt``.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    dest = Path(param_file) if param_file is not None else _DEFAULT_PARAM_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters, dest)
    print(f"DFT-D3 parameters saved to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Parameter loading helper
# ---------------------------------------------------------------------------


def load_dftd3_params(
    param_file: Path | str | None = None,
    auto_download: bool = True,
) -> "D3Parameters":
    """Load DFT-D3 reference parameters from disk, downloading if necessary.

    Parameters
    ----------
    param_file : Path or str or None
        Explicit path to a ``.pt`` file containing the D3 parameters as a
        ``state_dict``-style dictionary with keys ``"rcov"``, ``"r4r2"``,
        ``"c6ab"``, and ``"cn_ref"``.  If ``None``, the default cache path
        ``~/.cache/nvalchemiops/dftd3_parameters.pt`` is used.
    auto_download : bool
        When ``True`` and the parameter file does not exist, the Fortran
        reference archive is downloaded from the Grimme group website,
        parsed in-memory, and saved to *param_file* automatically.
        Set to ``False`` to require the file to be present.

    Returns
    -------
    D3Parameters
        Loaded parameter dataclass on CPU.

    Raises
    ------
    FileNotFoundError
        If the parameter file does not exist and ``auto_download=False``.
    RuntimeError
        If the auto-download or parsing step fails.
    """
    from nvalchemiops.torch.interactions.dispersion import D3Parameters  # lazy

    dest = Path(param_file) if param_file is not None else _DEFAULT_PARAM_FILE

    if not dest.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"DFT-D3 parameter file not found at '{dest}'.  "
                "Set auto_download=True or provide the file path explicitly."
            )
        params = extract_dftd3_parameters()
        save_dftd3_parameters(params, dest)

    state_dict = torch.load(str(dest), map_location="cpu", weights_only=True)
    return D3Parameters(
        rcov=state_dict["rcov"],
        r4r2=state_dict["r4r2"],
        c6ab=state_dict["c6ab"],
        cn_ref=state_dict["cn_ref"],
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class DFTD3ModelWrapper(nn.Module, BaseModelMixin):
    """DFT-D3(BJ) dispersion correction as a model wrapper.

    Wraps the Warp-accelerated ``dftd3`` kernel from ``nvalchemiops``.
    Positions are expected in Å; the wrapper converts to Bohr internally
    and returns energies in eV, forces in eV/Å, and virials in eV.

    Parameters
    ----------
    a1 : float
        BJ damping parameter :math:`a_1` (dimensionless, functional-specific).
    a2 : float
        BJ damping parameter :math:`a_2` in **Bohr** (functional-specific).
    s8 : float
        C8 coefficient scaling factor (dimensionless, functional-specific).
    cutoff : float, optional
        Interaction cutoff in Å.  Defaults to ``50.0`` Å (≈95 Bohr), which
        covers virtually all D3 interactions.
    k1 : float, optional
        Steepness of the CN counting function (1/Bohr).  Defaults to ``16.0``.
    k3 : float, optional
        Gaussian width for CN interpolation.  Defaults to ``-4.0``.
    s6 : float, optional
        C6 scaling factor.  Defaults to ``1.0``.
    max_neighbors : int, optional
        Maximum neighbors per atom for the neighbor matrix.  Defaults to 128.
    auto_download : bool, optional
        Automatically download D3 parameters if the cache file is missing.
        Defaults to ``True``.
    param_file : Path or str or None, optional
        Explicit path to the D3 parameter ``.pt`` file.  ``None`` uses the
        default cache at ``~/.cache/nvalchemiops/dftd3_parameters.pt``.

    Attributes
    ----------
    model_config : ModelConfig
        Mutable configuration controlling which outputs are computed.
        Set ``model.model_config.compute_stresses = True`` to enable
        virial computation for NPT/NPH simulations.
    rcov, r4r2, c6ab, cn_ref : nn.Buffer
        D3 reference parameters registered as module buffers so they move
        with ``.to(device)`` calls.
    """

    def __init__(
        self,
        a1: float,
        a2: float,
        s8: float,
        cutoff: float = 50.0,
        k1: float = 16.0,
        k3: float = -4.0,
        s6: float = 1.0,
        max_neighbors: int = 128,
        auto_download: bool = True,
        param_file: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.a1 = a1
        self.a2 = a2
        self.s8 = s8
        self.cutoff = cutoff
        self.k1 = k1
        self.k3 = k3
        self.s6 = s6
        self.max_neighbors = max_neighbors
        self.model_config = ModelConfig()
        self._model_card: ModelCard = self._build_model_card()

        # Load D3 parameters and register as buffers so .to(device) works.
        d3_params = load_dftd3_params(
            param_file=param_file, auto_download=auto_download
        )
        self.register_buffer("rcov", d3_params.rcov.float())
        self.register_buffer("r4r2", d3_params.r4r2.float())
        self.register_buffer("c6ab", d3_params.c6ab.float())
        self.register_buffer("cn_ref", d3_params.cn_ref.float())

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    def _build_model_card(self) -> ModelCard:
        return ModelCard(
            forces_via_autograd=False,
            supports_energies=True,
            supports_forces=True,
            supports_stresses=True,
            supports_pbc=True,
            needs_pbc=False,
            supports_non_batch=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.MATRIX,
                half_list=False,
                max_neighbors=self.max_neighbors,
            ),
        )

    @property
    def model_card(self) -> ModelCard:
        return self._model_card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute embeddings is not meaningful for D3 models."""
        raise NotImplementedError("DFTD3ModelWrapper does not produce embeddings.")

    # ------------------------------------------------------------------
    # Input / output key declarations
    # ------------------------------------------------------------------

    def input_data(self) -> set[str]:
        """Return required input keys.

        Overrides the base-class default to include ``atomic_numbers`` and
        the neighbor-matrix keys while omitting ``pbc`` (not needed for D3).
        """
        return {"positions", "atomic_numbers", "neighbor_matrix", "num_neighbors"}

    # ------------------------------------------------------------------
    # Input adaptation
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect required inputs from *data* without enabling gradients.

        Forces are computed analytically by the Warp kernel; autograd is
        not required on positions.
        """
        if not isinstance(data, Batch):
            raise TypeError(
                "DFTD3ModelWrapper requires a Batch input; "
                "got AtomicData.  Use Batch.from_data_list([data]) to wrap it."
            )

        input_dict: dict[str, Any] = {}
        for key in self.input_data():
            value = getattr(data, key, None)
            if value is None:
                raise KeyError(f"'{key}' required but not found in input data.")
            input_dict[key] = value

        input_dict["batch_idx"] = data.batch.to(torch.int32)
        input_dict["ptr"] = data.ptr.to(torch.int32)
        input_dict["num_graphs"] = data.num_graphs
        input_dict["fill_value"] = data.num_nodes

        # Collect neighbor tensors (with optional filtering to model cutoff).
        neighbor_dict = prepare_neighbors_for_model(
            data, self.cutoff, NeighborListFormat.MATRIX, data.num_nodes
        )
        input_dict["neighbor_matrix"] = neighbor_dict["neighbor_matrix"]
        input_dict["num_neighbors"] = neighbor_dict["num_neighbors"]
        input_dict["neighbor_shifts"] = neighbor_dict.get("neighbor_shifts", None)

        # Optional PBC cell.
        try:
            input_dict["cell"] = data.cell  # (B, 3, 3)
        except AttributeError:
            input_dict["cell"] = None

        return input_dict

    # ------------------------------------------------------------------
    # Output adaptation
    # ------------------------------------------------------------------

    def adapt_output(self, model_output: Any, data: AtomicData | Batch) -> ModelOutputs:
        """
        Adapt the model output to the framework output format.
        """
        output: ModelOutputs = OrderedDict()
        output["energies"] = model_output["energies"]
        if self.model_config.compute_forces:
            output["forces"] = model_output["forces"]
        if self.model_config.compute_stresses:
            if "virials" in model_output:
                # The dftd3 kernel accumulates the virial as W = -Σ r_ij ⊗ F_ij
                # (negative convention).  The framework convention for
                # batch.stresses is the positive physical virial W_phys = +Σ r_ij ⊗ F_ij
                # (energy units, eV).  Negate here to match LJ convention.
                output["stresses"] = -model_output["virials"]
            elif "stresses" in model_output:
                output["stresses"] = model_output["stresses"]
        return output

    def output_data(self) -> set[str]:
        """
        Return the set of keys that the model produces.
        """
        keys: set[str] = {"energies"}
        if self.model_config.compute_forces:
            keys.add("forces")
        if self.model_config.compute_stresses:
            keys.add("stresses")
        return keys

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the DFT-D3(BJ) kernel and return a :class:`ModelOutputs` dict.

        Parameters
        ----------
        data : Batch
            Batch containing ``positions``, ``atomic_numbers``,
            ``neighbor_matrix``, ``num_neighbors``, and optionally
            ``cell`` / ``neighbor_shifts`` (populated by
            :class:`~nvalchemi.dynamics.hooks.NeighborListHook`).

        Returns
        -------
        ModelOutputs
            OrderedDict with keys ``"energies"`` (shape ``[B, 1]``, eV),
            ``"forces"`` (shape ``[N, 3]``, eV/Å), and optionally
            ``"stresses"`` (shape ``[B, 3, 3]``, eV — the physical virial
            ``+Σ r_ij ⊗ F_ij``).
        """
        from nvalchemiops.torch.interactions.dispersion import (  # lazy
            D3Parameters,
            dftd3,
        )

        inp = self.adapt_input(data, **kwargs)

        positions = inp["positions"]  # (N, 3) Å
        numbers = inp["atomic_numbers"].to(torch.int32)  # (N,)
        neighbor_matrix = inp["neighbor_matrix"].contiguous()  # (N, K) int32
        neighbor_shifts = inp.get("neighbor_shifts")  # (N, K, 3) int32 or None
        batch_idx = inp["batch_idx"].contiguous()  # (N,) int32
        fill_value = inp["fill_value"]  # int
        B = inp["num_graphs"]

        # Convert positions and cell from Å to Bohr.
        positions_bohr = positions * ANGSTROM_TO_BOHR

        cell = inp.get("cell")
        cell_bohr: torch.Tensor | None = None
        if cell is not None:
            cell_bohr = cell * ANGSTROM_TO_BOHR

        # Also scale a2 from Bohr to Bohr (no conversion needed — a2 is
        # already stored in Bohr, matching the kernel's expectation).

        compute_virial = self.model_config.compute_stresses

        d3_params = D3Parameters(
            rcov=self.rcov,
            r4r2=self.r4r2,
            c6ab=self.c6ab,
            cn_ref=self.cn_ref,
        )

        result = dftd3(
            positions=positions_bohr,
            numbers=numbers,
            a1=self.a1,
            a2=self.a2,
            s8=self.s8,
            k1=self.k1,
            k3=self.k3,
            s6=self.s6,
            d3_params=d3_params,
            fill_value=fill_value,
            batch_idx=batch_idx,
            cell=cell_bohr,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            compute_virial=compute_virial,
            num_systems=B,
        )

        # dftd3 returns (energy[B], forces[N,3], coord_num[N])
        # or (energy[B], forces[N,3], coord_num[N], virial[B,3,3]) when compute_virial=True.
        if compute_virial:
            energy_ha, forces_ha_bohr, _coord_num, virial_ha = result
        else:
            energy_ha, forces_ha_bohr, _coord_num = result
            virial_ha = None

        # Convert units: Hartree → eV, Hartree/Bohr → eV/Å.
        energies_ev = energy_ha.to(positions.dtype) * HARTREE_TO_EV  # (B,)
        energies_ev = energies_ev.unsqueeze(-1)  # (B, 1)

        forces_ev_ang = forces_ha_bohr.to(positions.dtype) * (
            HARTREE_TO_EV / BOHR_TO_ANGSTROM
        )  # (N, 3)

        model_output: dict[str, Any] = {
            "energies": energies_ev,
            "forces": forces_ev_ang,
        }
        if virial_ha is not None:
            # Virial: Hartree → eV (purely energy units, no length scaling).
            model_output["virials"] = virial_ha.to(positions.dtype) * HARTREE_TO_EV

        return self.adapt_output(model_output, data)

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Export model is not implemented for D3 models."""
        raise NotImplementedError
