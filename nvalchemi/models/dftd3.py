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
from __future__ import annotations

import io
import re
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import torch
from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi.data import Batch
from nvalchemi.models.base import _UNSET, ForwardContext, Potential, _resolve_config
from nvalchemi.models.contracts import NeighborRequirement, PotentialCard
from nvalchemi.models.metadata import (
    DISPERSION,
    PAIRWISE,
    CheckpointInfo,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import neighbor_result_key
from nvalchemi.models.registry import resolve_known_artifact
from nvalchemi.models.results import CalculatorResults
from nvalchemi.models.utils import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    virial_to_stress,
)

__all__ = [
    "DFTD3Config",
    "DFTD3Potential",
    "extract_dftd3_parameters",
    "load_dftd3_params",
    "save_dftd3_parameters",
]

FUNCTIONAL_PARAMS: dict[str, dict[str, float]] = {
    "pbe": {"a1": 0.4289, "a2": 4.4407, "s6": 1.0, "s8": 0.7875},
    "wb97m": {"a1": 0.5660, "a2": 3.1280, "s6": 1.0, "s8": 0.3908},
    "r2scan": {"a1": 0.49484001, "a2": 5.73083694, "s6": 1.0, "s8": 0.78981345},
}


class DFTD3Config(BaseModel):
    """Configuration for :class:`DFTD3Potential`.

    When *functional* is set, the BJ damping parameters (``a1``, ``a2``,
    ``s6``, ``s8``) are overridden by built-in presets unless they were
    explicitly provided by the caller.

    Attributes
    ----------
    functional : str or None, default None
        XC functional name (e.g. ``"pbe"``, ``"r2scan"``).  When set,
        loads preset BJ parameters for this functional.
    a1 : float, default 0.4289
        BJ damping parameter *a1* (dimensionless).
    a2 : float, default 4.4407
        BJ damping parameter *a2* (assumed Bohr).
    s6 : float, default 1.0
        Scaling factor *s6* (dimensionless).
    s8 : float, default 0.7875
        Scaling factor *s8* (dimensionless).
    k1 : float, default 16.0
        Coordination number counting steepness.
    k3 : float, default -4.0
        Coordination number counting exponent.
    cutoff : float, default 15.0
        Interaction cutoff radius (assumed Angstrom).
    smoothing_on : float or None, default None
        Distance at which smoothing begins (assumed Angstrom).  Defaults to
        ``0.8 * cutoff``.
    smoothing_off : float or None, default None
        Distance at which the interaction vanishes (assumed Angstrom).
        Defaults to *cutoff*.
    param_path : Path or None, default None
        Local path to a pre-extracted DFT-D3 parameter file.
    auto_download : bool, default True
        Download and cache parameters automatically if missing.
    neighbor_list_name : str, default "default"
        Logical neighbor-list name used to namespace result keys.
    format : {"matrix"}, default "matrix"
        Neighbor-list storage format.
    """

    functional: Annotated[
        str | None, Field(description="XC functional name for preset BJ parameters.")
    ] = None
    a1: Annotated[
        float, Field(description="BJ damping parameter a1 (dimensionless).")
    ] = 0.4289
    a2: Annotated[
        float, Field(description="BJ damping parameter a2 (assumed Bohr).")
    ] = 4.4407
    s6: Annotated[float, Field(description="Scaling factor s6 (dimensionless).")] = 1.0
    s8: Annotated[float, Field(description="Scaling factor s8 (dimensionless).")] = (
        0.7875
    )
    k1: Annotated[float, Field(description="CN counting steepness.")] = 16.0
    k3: Annotated[float, Field(description="CN counting exponent.")] = -4.0
    cutoff: Annotated[
        float, Field(description="Interaction cutoff radius (assumed Angstrom).")
    ] = 15.0
    smoothing_on: Annotated[
        float | None, Field(description="Smoothing onset distance (assumed Angstrom).")
    ] = None
    smoothing_off: Annotated[
        float | None,
        Field(description="Smoothing termination distance (assumed Angstrom)."),
    ] = None
    param_path: Annotated[
        Path | None,
        Field(description="Local path to pre-extracted DFT-D3 parameter file."),
    ] = None
    auto_download: Annotated[
        bool,
        Field(description="Download and cache parameters automatically if missing."),
    ] = True
    neighbor_list_name: Annotated[
        str, Field(description="Logical neighbor-list name for result-key namespacing.")
    ] = "default"
    format: Annotated[
        Literal["matrix"], Field(description="Neighbor-list storage format.")
    ] = "matrix"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _apply_functional_and_defaults(self) -> Self:
        """Apply functional presets and default smoothing values.

        Raises
        ------
        ValueError
            If ``functional`` is not ``None`` and is not a recognised
            XC functional name.
        """

        if self.functional is not None:
            key = self.functional.lower()
            if key not in FUNCTIONAL_PARAMS:
                raise ValueError(
                    f"Unknown functional '{self.functional}'. "
                    f"Available: {sorted(FUNCTIONAL_PARAMS)}."
                )
            params = FUNCTIONAL_PARAMS[key]
            for name in ("a1", "a2", "s6", "s8"):
                if name not in self.model_fields_set:
                    setattr(self, name, params[name])
        if self.smoothing_on is None:
            self.smoothing_on = 0.8 * self.cutoff
        if self.smoothing_off is None:
            self.smoothing_off = self.cutoff
        return self


DFTD3PotentialCard = PotentialCard(
    required_inputs=frozenset(
        {
            "positions",
            "atomic_numbers",
            neighbor_result_key("default", "neighbor_matrix"),
            neighbor_result_key("default", "num_neighbors"),
        }
    ),
    optional_inputs=frozenset(
        {"cell", "pbc", neighbor_result_key("default", "neighbor_shifts")}
    ),
    result_keys=frozenset({"energies", "forces", "stresses"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies", "forces", "stresses"}),
    boundary_modes=frozenset({"non_pbc", "pbc"}),
    neighbor_requirement=NeighborRequirement(
        source="external",
        format="matrix",
        name="default",
    ),
    parameterized_by=frozenset({"neighbor_list_name", "functional"}),
)


def _extract_fortran_sources_from_archive_bytes(content_bytes: bytes) -> dict[str, str]:
    """Extract relevant Fortran sources from one DFT-D3 archive payload."""

    extracted: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(content_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith((".f", ".F")):
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    extracted[Path(member.name).name] = extracted_file.read().decode(
                        "utf-8",
                        errors="ignore",
                    )
    return extracted


def _extract_fortran_sources_from_archive(archive_path: Path | str) -> dict[str, str]:
    """Extract relevant Fortran sources from one local DFT-D3 archive."""

    archive_file = Path(archive_path)
    return _extract_fortran_sources_from_archive_bytes(archive_file.read_bytes())


def _find_fortran_array(content: str, var_name: str) -> np.ndarray:
    """Parse a Fortran ``data`` array block into a float64 array.

    Parameters
    ----------
    content
        Full text of the Fortran source file.
    var_name
        Variable name to locate in ``data`` statements.

    Returns
    -------
    np.ndarray
        One-dimensional float64 array of parsed values.

    Raises
    ------
    ValueError
        If the variable is not found or the block cannot be parsed.
    """

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
        raise ValueError(f"Variable '{var_name}' not found in Fortran source.")

    data_str = " ".join(block_lines)
    match = re.search(
        rf"data\s+{var_name}\s*/\s*(.*?)\s*/",
        data_str,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Failed to parse '{var_name}'.")

    block = match.group(1)
    clean_lines = []
    for line in block.split("\n"):
        if "!" in line:
            line = line[: line.index("!")]
        clean_lines.append(line)
    block = " ".join(clean_lines)
    numbers = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", block)
    return np.array(
        [float(number.replace("_wp", "")) for number in numbers], dtype=np.float64
    )


def _parse_pars_array(content: str) -> np.ndarray:
    """Parse the ``pars`` array from ``pars.f`` into an ``(N, 5)`` array.

    Parameters
    ----------
    content
        Full text of the ``pars.f`` Fortran source file.

    Returns
    -------
    np.ndarray
        Two-dimensional float64 array of shape ``(N, 5)``.
    """

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
        values.extend(
            float(number.replace("D", "e").replace("d", "e")) for number in numbers
        )

    array = np.array(values, dtype=np.float64)
    return array[: (len(array) // 5) * 5].reshape(-1, 5)


def _limit(encoded: int) -> tuple[int, int]:
    """Decode the Fortran element encoding into atomic number and CN index.

    Parameters
    ----------
    encoded
        Encoded integer from the Fortran reference data.

    Returns
    -------
    tuple[int, int]
        ``(atomic_number, cn_index)`` pair.
    """

    atom = encoded
    cn_index = 1
    while atom > 100:
        atom -= 100
        cn_index += 1
    return atom, cn_index


def _build_c6_arrays(pars_records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build c6 and CN reference arrays from parsed records.

    Parameters
    ----------
    pars_records
        ``(N, 5)`` array from :func:`_parse_pars_array`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(c6ab, cn_ref)`` arrays of shape ``(95, 95, 5, 5)`` each.
    """

    c6ab = np.zeros((95, 95, 5, 5), dtype=np.float32)
    cn_ref = np.full((95, 95, 5, 5), -1.0, dtype=np.float32)
    cn_values: dict[int, dict[int, float]] = {element: {} for element in range(95)}

    for record in pars_records:
        c6_value, z_i_enc, z_j_enc, cn_i, cn_j = record
        iat, iadr = _limit(int(z_i_enc))
        jat, jadr = _limit(int(z_j_enc))
        if not (1 <= iat <= 94 and 1 <= jat <= 94):
            continue
        if not (1 <= iadr <= 5 and 1 <= jadr <= 5):
            continue
        ia = iadr - 1
        ja = jadr - 1
        c6ab[iat, jat, ia, ja] = c6_value
        c6ab[jat, iat, ja, ia] = c6_value
        cn_values[iat].setdefault(ia, cn_i)
        cn_values[jat].setdefault(ja, cn_j)

    for element in range(1, 95):
        for partner in range(1, 95):
            for ci in range(5):
                if ci in cn_values[element]:
                    cn_ref[element, partner, ci, :] = cn_values[element][ci]
    return c6ab, cn_ref


def extract_dftd3_parameters(
    dftd3_ref_dir: Path | str | None = None,
) -> dict[str, torch.Tensor]:
    """Extract DFT-D3 parameter tensors from local Fortran sources.

    Parameters
    ----------
    dftd3_ref_dir
        Directory containing ``dftd3.f`` and ``pars.f``.  Default ``None``.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping with keys ``"rcov"``, ``"r4r2"``, ``"c6ab"``, ``"cn_ref"``.

    Raises
    ------
    FileNotFoundError
        If *dftd3_ref_dir* does not exist or required files are missing.
    ValueError
        If *dftd3_ref_dir* is ``None``.
    """

    if dftd3_ref_dir is not None:
        ref_dir = Path(dftd3_ref_dir)
        if not ref_dir.exists():
            raise FileNotFoundError(f"dftd3_ref_dir not found: {ref_dir}")
        dftd3_file = ref_dir / "dftd3.f"
        pars_file = ref_dir / "pars.f"
        for file_path in (dftd3_file, pars_file):
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        dftd3_content = dftd3_file.read_text()
        pars_content = pars_file.read_text()
    else:
        raise ValueError(
            "dftd3_ref_dir is required when extracting parameters directly. "
            "Use load_dftd3_params() or the models registry for downloaded artifacts."
        )

    r2r4_94 = _find_fortran_array(dftd3_content, "r2r4")
    rcov_94 = _find_fortran_array(dftd3_content, "rcov")
    pars_records = _parse_pars_array(pars_content)

    r4r2 = np.zeros(95, dtype=np.float32)
    r4r2[1:95] = r2r4_94.astype(np.float32)
    rcov = np.zeros(95, dtype=np.float32)
    rcov[1:95] = rcov_94.astype(np.float32)
    c6ab, cn_ref = _build_c6_arrays(pars_records)
    return {
        "rcov": torch.from_numpy(rcov),
        "r4r2": torch.from_numpy(r4r2),
        "c6ab": torch.from_numpy(c6ab),
        "cn_ref": torch.from_numpy(cn_ref),
    }


def extract_dftd3_parameters_from_archive(
    archive_path: Path | str,
) -> dict[str, torch.Tensor]:
    """Extract DFT-D3 parameter tensors from one downloaded reference archive.

    Parameters
    ----------
    archive_path
        Path to a ``.tar.gz`` archive containing ``dftd3.f`` and ``pars.f``.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping with keys ``"rcov"``, ``"r4r2"``, ``"c6ab"``, ``"cn_ref"``.

    Raises
    ------
    RuntimeError
        If the archive does not contain the required Fortran source files.
    """

    files = _extract_fortran_sources_from_archive(archive_path)
    for name in ("dftd3.f", "pars.f"):
        if name not in files:
            raise RuntimeError(f"'{name}' not found in downloaded DFT-D3 archive.")
    dftd3_content = files["dftd3.f"]
    pars_content = files["pars.f"]

    r2r4_94 = _find_fortran_array(dftd3_content, "r2r4")
    rcov_94 = _find_fortran_array(dftd3_content, "rcov")
    pars_records = _parse_pars_array(pars_content)

    r4r2 = np.zeros(95, dtype=np.float32)
    r4r2[1:95] = r2r4_94.astype(np.float32)
    rcov = np.zeros(95, dtype=np.float32)
    rcov[1:95] = rcov_94.astype(np.float32)
    c6ab, cn_ref = _build_c6_arrays(pars_records)
    return {
        "rcov": torch.from_numpy(rcov),
        "r4r2": torch.from_numpy(r4r2),
        "c6ab": torch.from_numpy(c6ab),
        "cn_ref": torch.from_numpy(cn_ref),
    }


def save_dftd3_parameters(
    parameters: dict[str, torch.Tensor],
    param_path: Path | str | None = None,
) -> Path:
    """Save extracted DFT-D3 parameters to a cache file.

    Parameters
    ----------
    parameters
        Tensor mapping as returned by :func:`extract_dftd3_parameters`.
    param_path
        Destination file path.  Default ``None``.

    Returns
    -------
    Path
        Resolved path where the parameters were written.

    Raises
    ------
    ValueError
        If *param_path* is ``None``.
    """

    if param_path is None:
        raise ValueError("param_path is required when saving DFT-D3 parameters.")
    destination = Path(param_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters, destination)
    return destination


def _resolve_dftd3_param_file(
    param_path: Path | str | None = None,
    *,
    auto_download: bool = True,
) -> tuple[Path, CheckpointInfo]:
    """Resolve the local DFT-D3 parameter file and its provenance.

    Parameters
    ----------
    param_path
        Explicit local path to a cached parameter file.  Default ``None``.
    auto_download
        Flag to download parameters automatically when *param_path*
        does not exist.  Default ``True``.

    Returns
    -------
    tuple[Path, CheckpointInfo]
        ``(resolved_path, checkpoint)`` pair.

    Raises
    ------
    FileNotFoundError
        If *param_path* is given, does not exist, and *auto_download*
        is ``False``.
    """

    if param_path is not None:
        destination = Path(param_path)
        checkpoint = CheckpointInfo(
            identifier=str(destination),
            source="local_path",
        )
        if not destination.exists():
            if not auto_download:
                raise FileNotFoundError(
                    f"DFT-D3 parameter file not found at '{destination}'. "
                    "Set auto_download=True or provide param_path explicitly."
                )
            resolved = resolve_known_artifact(
                "dftd3_parameters",
                family="dftd3",
                allow_download=True,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved.local_path, destination)
        return destination, checkpoint

    resolved = resolve_known_artifact(
        "dftd3_parameters",
        family="dftd3",
        allow_download=auto_download,
    )
    return resolved.local_path, resolved.checkpoint


def load_dftd3_params(
    param_path: Path | str | None = None,
    *,
    auto_download: bool = True,
) -> D3Parameters:
    """Load cached DFT-D3 parameters, downloading and caching if needed.

    Parameters
    ----------
    param_path
        Explicit local path to a cached parameter file.  Default ``None``.
    auto_download
        Flag to download parameters automatically when missing.
        Default ``True``.

    Returns
    -------
    D3Parameters
        Loaded reference parameters ready for the DFT-D3 kernel.
    """

    destination, _checkpoint = _resolve_dftd3_param_file(
        param_path,
        auto_download=auto_download,
    )
    state_dict = torch.load(str(destination), map_location="cpu", weights_only=True)
    return D3Parameters(
        rcov=state_dict["rcov"],
        r4r2=state_dict["r4r2"],
        c6ab=state_dict["c6ab"],
        cn_ref=state_dict["cn_ref"],
    )


class DFTD3Potential(Potential):
    """DFT-D3(BJ) dispersion correction using matrix neighbors.

    Implements the Becke-Johnson damped DFT-D3 dispersion correction
    as a composable pipeline step.  Requires an external matrix-format
    neighbor list.

    Attributes
    ----------
    card : PotentialCard
        Class-level contract card declaring required inputs and result keys.
    config : DFTD3Config
        Resolved configuration for this instance.
    model_card : ModelCard
        Provenance metadata describing the DFT-D3 parameter source.
    """

    card = DFTD3PotentialCard

    def __init__(
        self,
        config: DFTD3Config | None = None,
        *,
        functional: str | None = _UNSET,
        a1: float = _UNSET,
        a2: float = _UNSET,
        s6: float = _UNSET,
        s8: float = _UNSET,
        k1: float = _UNSET,
        k3: float = _UNSET,
        cutoff: float = _UNSET,
        smoothing_on: float | None = _UNSET,
        smoothing_off: float | None = _UNSET,
        param_path: Path | None = _UNSET,
        auto_download: bool = _UNSET,
        neighbor_list_name: str = _UNSET,
        format: Literal["matrix"] = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise a DFT-D3(BJ) dispersion correction potential.

        Accepts either a :class:`DFTD3Config` object, individual keyword
        arguments matching the config fields, or both (keyword arguments
        override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.  Default ``None``.
        functional
            XC functional name for preset BJ parameters.
        a1
            BJ damping parameter *a1* (dimensionless).
        a2
            BJ damping parameter *a2* (assumed Bohr).
        s6
            Scaling factor *s6* (dimensionless).
        s8
            Scaling factor *s8* (dimensionless).
        k1
            CN counting steepness.
        k3
            CN counting exponent.
        cutoff
            Interaction cutoff radius (assumed Angstrom).
        smoothing_on
            Smoothing onset distance (assumed Angstrom).
        smoothing_off
            Smoothing termination distance (assumed Angstrom).
        param_path
            Local path to a pre-extracted DFT-D3 parameter file.
        auto_download
            Flag to download and cache parameters automatically when
            missing.
        neighbor_list_name
            Logical neighbor-list name for result-key namespacing.
        format
            Neighbor-list storage format.
        name
            Human-readable step name.  Default ``None``.
        """

        config = _resolve_config(
            DFTD3Config,
            config,
            {
                "functional": functional,
                "a1": a1,
                "a2": a2,
                "s6": s6,
                "s8": s8,
                "k1": k1,
                "k3": k3,
                "cutoff": cutoff,
                "smoothing_on": smoothing_on,
                "smoothing_off": smoothing_off,
                "param_path": param_path,
                "auto_download": auto_download,
                "neighbor_list_name": neighbor_list_name,
                "format": format,
            },
        )
        super().__init__(
            name=name,
            required_inputs=frozenset(
                {
                    "positions",
                    "atomic_numbers",
                    neighbor_result_key(config.neighbor_list_name, "neighbor_matrix"),
                    neighbor_result_key(config.neighbor_list_name, "num_neighbors"),
                }
            ),
            optional_inputs=frozenset(
                {
                    "cell",
                    "pbc",
                    neighbor_result_key(config.neighbor_list_name, "neighbor_shifts"),
                }
            ),
            neighbor_requirement=NeighborRequirement(
                source="external",
                cutoff=config.cutoff,
                format="matrix",
                name=config.neighbor_list_name,
            ),
        )
        self.config = config
        param_file, checkpoint = _resolve_dftd3_param_file(
            config.param_path,
            auto_download=config.auto_download,
        )
        self._d3_params = load_dftd3_params(
            param_path=param_file,
            auto_download=False,
        )
        reference_xc_functional = (
            config.functional.lower() if config.functional is not None else None
        )
        self.model_card = ModelCard(
            model_family="dftd3",
            model_name=(
                str(config.param_path)
                if config.param_path is not None
                else "dftd3_parameters"
            ),
            reference_xc_functional=reference_xc_functional,
            provided_terms=(PhysicalTerm(kind=DISPERSION, variant=PAIRWISE),),
            checkpoint=checkpoint,
        )

    def required_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return required inputs for one requested output set.

        Parameters
        ----------
        outputs
            Subset of result keys to consider.  Default ``None`` (all).

        Returns
        -------
        frozenset[str]
            Required batch or result keys.  Includes ``"cell"`` and
            ``"pbc"`` when stresses are requested.
        """

        active = self.active_outputs(outputs)
        required = set(self.profile.required_inputs)
        if "stresses" in active:
            required |= {"cell", "pbc"}
        return frozenset(required)

    def optional_inputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> frozenset[str]:
        """Return optional inputs for one requested output set.

        Parameters
        ----------
        outputs
            Subset of result keys to consider.  Default ``None`` (all).

        Returns
        -------
        frozenset[str]
            Optional batch or result keys.  Excludes ``"cell"`` and
            ``"pbc"`` when stresses are requested (they become required).
        """

        active = self.active_outputs(outputs)
        optional = set(self.profile.optional_inputs)
        if "stresses" in active:
            optional -= {"cell", "pbc"}
        return frozenset(optional)

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Run the DFT-D3 kernel for the current batch.

        Parameters
        ----------
        batch
            Input :class:`~nvalchemi.data.Batch` with positions, atomic
            numbers, and matrix-format neighbor data.
        ctx
            Forward context carrying resolved outputs and runtime state.

        Returns
        -------
        CalculatorResults
            Mapping with ``"energies"`` and, when requested, ``"forces"``
            and ``"stresses"``.

        Raises
        ------
        ValueError
            If stresses are requested but periodic inputs are missing.
        """

        positions = self.require_input(batch, "positions", ctx)
        numbers = self.require_input(batch, "atomic_numbers", ctx).to(torch.int32)
        neighbor_matrix = self.require_input(
            batch,
            neighbor_result_key(self.config.neighbor_list_name, "neighbor_matrix"),
            ctx,
        ).contiguous()
        neighbor_shifts = self.optional_input(
            batch,
            neighbor_result_key(self.config.neighbor_list_name, "neighbor_shifts"),
            ctx,
        )
        if neighbor_shifts is None:
            neighbor_shifts = torch.zeros(
                (*neighbor_matrix.shape, 3),
                dtype=torch.int32,
                device=neighbor_matrix.device,
            )
        else:
            neighbor_shifts = neighbor_shifts.contiguous()

        cell, pbc = self.resolve_periodic_inputs(batch, ctx)
        periodic = (
            cell is not None
            and pbc is not None
            and bool(torch.as_tensor(pbc).any().item())
        )
        if "stresses" in ctx.outputs and not periodic:
            raise ValueError(
                "DFT-D3 stresses require periodic inputs 'cell' and 'pbc'."
            )

        positions_bohr = positions * ANGSTROM_TO_BOHR
        cell_bohr = None
        if cell is not None:
            cell_bohr = (
                cell if cell.ndim == 3 else cell.unsqueeze(0)
            ) * ANGSTROM_TO_BOHR

        result = dftd3(
            positions=positions_bohr,
            numbers=numbers,
            a1=self.config.a1,
            a2=self.config.a2,
            s8=self.config.s8,
            k1=self.config.k1,
            k3=self.config.k3,
            s6=self.config.s6,
            d3_params=self._d3_params,
            fill_value=-1,
            batch_idx=batch.batch.to(torch.int32),
            cell=cell_bohr,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            compute_virial="stresses" in ctx.outputs,
            num_systems=batch.num_graphs,
        )

        if "stresses" in ctx.outputs:
            energy_ha, forces_ha_bohr, _, virial_ha = result
        else:
            energy_ha, forces_ha_bohr, _ = result
            virial_ha = None

        energies = (energy_ha.to(positions.dtype) * HARTREE_TO_EV).unsqueeze(-1)
        forces = forces_ha_bohr.to(positions.dtype) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        stresses = None
        if virial_ha is not None and cell is not None:
            stresses = virial_to_stress(
                virial_ha.to(positions.dtype) * HARTREE_TO_EV,
                cell if cell.ndim == 3 else cell.unsqueeze(0),
            )

        return self.build_results(
            ctx,
            energies=energies,
            forces=forces if "forces" in ctx.outputs else None,
            stresses=stresses,
        )
