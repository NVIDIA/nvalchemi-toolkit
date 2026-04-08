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
:class:`~nvalchemi.models.base.BaseModelMixin`-compatible model for
composable execution.

Usage
-----
Instantiate a DFT-D3(BJ) wrapper with a built-in functional preset::

    model = DFTD3ModelWrapper(functional="pbe")

Or prewarm the cached parameter file for offline use::

    path = download_dftd3_parameters()
    model = DFTD3ModelWrapper(param_path=path)

Notes
-----
* Forces are computed **analytically** inside the Warp kernel (not via
  autograd), so ``spec.use_autograd`` is ``False``.
* Positions and cell tensors are converted from Å to Bohr before the kernel
  call, and kernel outputs are converted back to eV and eV/Å.
* Stress/virial computation is available when periodic cell data is present.
* D3 parameters are loaded from a cached ``.pt`` file (default location
  ``~/.cache/nvalchemi/dftd3``).  When the file is absent, the reference
  archive is downloaded from the ``simple-dftd3`` GitHub repository,
  verified with a SHA256 checksum, and cached automatically.
"""

from __future__ import annotations

import io
import re
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import torch
from nvalchemiops.torch.interactions.dispersion import D3Parameters, dftd3
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    _resolve_config,
)
from nvalchemi.models.utils import (
    _UNSET,
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    build_model_repr,
    collect_nondefault_repr_kwargs,
    download_cached_file,
    initialize_model_repr,
    mapping_get,
    normalize_batch_indices,
    resolve_matrix_neighbor_shifts,
    virial_to_stress,
)

__all__ = [
    "DFTD3Config",
    "DFTD3ModelWrapper",
    "DFTD3ParametersProcessor",
    "download_dftd3_parameters",
    "extract_dftd3_parameters",
    "load_dftd3_params",
    "save_dftd3_parameters",
]

FUNCTIONAL_PARAMS: dict[str, dict[str, float]] = {
    "bp": {"a1": 0.3946, "a2": 4.8516, "s6": 1.0, "s8": 3.2822},
    "b973c": {"a1": 0.37, "a2": 4.10, "s6": 1.0, "s8": 1.50},
    "b97-3c": {"a1": 0.37, "a2": 4.10, "s6": 1.0, "s8": 1.50},
    "blyp": {"a1": 0.4298, "a2": 4.2359, "s6": 1.0, "s8": 2.6996},
    "b3lyp": {"a1": 0.3981, "a2": 4.4211, "s6": 1.0, "s8": 1.9889},
    "pbe": {"a1": 0.4289, "a2": 4.4407, "s6": 1.0, "s8": 0.7875},
    "pbe0": {"a1": 0.4145, "a2": 4.8593, "s6": 1.0, "s8": 1.2177},
    "wb97m": {"a1": 0.5660, "a2": 3.1280, "s6": 1.0, "s8": 0.3908},
    "wb97x": {"a1": 0.0000, "a2": 5.4959, "s6": 1.0, "s8": 0.2641},
    "r2scan": {"a1": 0.49484001, "a2": 5.73083694, "s6": 1.0, "s8": 0.78981345},
}
DFTD3_ARCHIVE_URL = (
    "https://github.com/dftd3/simple-dftd3/archive/refs/heads/main.tar.gz"
)
DFTD3_ARCHIVE_SHA256 = (
    "0eb3e36bfb24dcd9bb1d1bece1531216b59539a8fde17ee80224af0653c92aa3"
)


class DFTD3Config(BaseModel):
    """Configuration for :class:`DFTD3ModelWrapper`."""

    functional: Annotated[
        str | None, Field(description="XC functional name for preset BJ parameters.")
    ] = None
    a1: Annotated[float, Field(description="BJ damping parameter a1.")] = 0.4289
    a2: Annotated[float, Field(description="BJ damping parameter a2 (Bohr).")] = 4.4407
    s6: Annotated[float, Field(description="Scaling factor s6.")] = 1.0
    s8: Annotated[float, Field(description="Scaling factor s8.")] = 0.7875
    k1: Annotated[float, Field(description="CN counting steepness.")] = 16.0
    k3: Annotated[float, Field(description="CN counting exponent.")] = -4.0
    cutoff: Annotated[
        float, Field(description="Interaction cutoff radius (Angstrom).")
    ] = 15.0
    smoothing_fraction: Annotated[
        float, Field(description="Fraction of cutoff where smoothing begins.")
    ] = 0.2
    param_path: Annotated[
        Path | None, Field(description="Local path to cached DFT-D3 parameters.")
    ] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _apply_functional_defaults(self) -> Self:
        """Apply known functional presets when available."""

        if self.functional is not None:
            params = FUNCTIONAL_PARAMS.get(self.functional.lower())
            if params is not None:
                for key in ("a1", "a2", "s6", "s8"):
                    if key not in self.model_fields_set:
                        setattr(self, key, params[key])
        return self


_DFTD3ModelConfig = ModelConfig(
    required_inputs=frozenset({"positions", "atomic_numbers", "neighbor_matrix"}),
    optional_inputs=frozenset(
        {"batch", "cell", "pbc", "neighbor_shifts", "num_neighbors"}
    ),
    outputs=frozenset({"energies", "forces"}),
    optional_outputs={"stresses": frozenset({"cell", "pbc"})},
    additive_outputs=frozenset({"energies", "forces", "stresses"}),
    use_autograd=False,
    pbc_mode="any",
    neighbor_config=NeighborConfig(
        source="external",
        cutoff=15.0,
        format="matrix",
        half_list=False,
    ),
)


def _extract_fortran_sources_from_archive_bytes(content_bytes: bytes) -> dict[str, str]:
    """Extract Fortran sources from one DFT-D3 reference archive payload."""

    extracted: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(content_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith((".f", ".F")):
                extracted_file = archive.extractfile(member)
                if extracted_file is not None:
                    extracted[Path(member.name).name] = extracted_file.read().decode(
                        "utf-8",
                        errors="ignore",
                    )
    return extracted


def _find_fortran_array(content: str, var_name: str) -> np.ndarray:
    """Parse one Fortran ``data`` array block into a float64 array."""

    match = re.search(
        rf"data\s+{var_name}\s*/\s*(.*?)\s*/",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Variable '{var_name}' not found in Fortran source.")
    block = re.sub(r"!.*", "", match.group(1))
    numbers = re.findall(r"[-+]?\d+\.\d+(?:_wp)?", block)
    return np.array(
        [float(number.replace("_wp", "")) for number in numbers],
        dtype=np.float64,
    )


def extract_dftd3_parameters(
    archive_path: Path | str,
) -> dict[str, torch.Tensor]:
    """Extract raw DFT-D3 reference tensors from one downloaded archive."""

    sources = _extract_fortran_sources_from_archive_bytes(
        Path(archive_path).read_bytes()
    )
    if not sources:
        raise ValueError("No DFT-D3 Fortran sources found in archive.")

    parmod = next(
        (content for name, content in sources.items() if "pars" in name.lower()), None
    )
    c6ab = next(
        (content for name, content in sources.items() if "c6" in name.lower()), None
    )
    if parmod is None or c6ab is None:
        raise ValueError("Failed to locate DFT-D3 parameter sources in archive.")

    rcov = torch.from_numpy(_find_fortran_array(parmod, "rcov")).float()
    r4r2 = torch.from_numpy(_find_fortran_array(parmod, "r4r2")).float()
    c6_raw = torch.from_numpy(_find_fortran_array(c6ab, "c6ab")).float()
    cn_raw = torch.from_numpy(_find_fortran_array(c6ab, "cnref")).float()
    return {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6ab": c6_raw.reshape(95, 95, 5, 5),
        "cn_ref": cn_raw.reshape(95, 95, 5, 5),
    }


def save_dftd3_parameters(
    parameters: dict[str, torch.Tensor],
    output_path: Path | str,
) -> Path:
    """Save extracted DFT-D3 parameters to a cached ``.pt`` file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters, output)
    return output


class DFTD3ParametersProcessor:
    """Helper for downloading, parsing, and caching DFT-D3 parameters.

    The processor encapsulates the standalone download / parse / convert path
    used by :class:`DFTD3ModelWrapper` when ``param_path`` is not provided.
    """

    @classmethod
    def extract_parameters_from_archive(
        cls,
        archive_path: Path | str,
    ) -> dict[str, torch.Tensor]:
        """Extract raw parameter tensors from one downloaded archive."""

        del cls
        return extract_dftd3_parameters(archive_path)

    @classmethod
    def resolve_param_file(
        cls,
        *,
        param_path: Path | None = None,
        cache_dir: Path | None = None,
        force: bool = False,
    ) -> Path:
        """Resolve a local cached parameter file, downloading if needed.

        Parameters
        ----------
        param_path
            Explicit path to a previously converted ``.pt`` parameter file.
        cache_dir
            Optional cache directory used for the downloaded archive and the
            converted parameter file.
        force
            When ``True``, re-download the archive and rebuild the converted
            parameter cache even if cached files already exist.

        Returns
        -------
        Path
            Path to the converted parameter file.

        Raises
        ------
        FileNotFoundError
            If an explicit ``param_path`` is provided but does not exist.
        ValueError
            If the downloaded archive checksum does not match the expected
            reference hash.
        """

        if param_path is not None:
            if not param_path.exists():
                raise FileNotFoundError(
                    f"DFT-D3 parameter file not found: {param_path}"
                )
            return param_path

        resolved_cache_dir = (
            cache_dir
            if cache_dir is not None
            else Path.home() / ".cache/nvalchemi/dftd3"
        )
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        converted = resolved_cache_dir / "dftd3_params.pt"
        if converted.exists() and not force:
            return converted

        archive_path = download_cached_file(
            url=DFTD3_ARCHIVE_URL,
            cache_dir=resolved_cache_dir,
            filename="dftd3.tgz",
            force=force,
        )
        archive_hash = sha256(archive_path.read_bytes()).hexdigest()
        if archive_hash != DFTD3_ARCHIVE_SHA256:
            raise ValueError(
                "DFT-D3 archive checksum mismatch. "
                f"Expected {DFTD3_ARCHIVE_SHA256}, got {archive_hash}."
            )
        parameters = cls.extract_parameters_from_archive(archive_path)
        return save_dftd3_parameters(parameters, converted)


def download_dftd3_parameters(
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Prewarm the cached DFT-D3 parameter file and return its path."""

    return DFTD3ParametersProcessor.resolve_param_file(
        cache_dir=cache_dir,
        force=force,
    )


def load_dftd3_params(
    param_path: Path | None = None,
    *,
    cache_dir: Path | None = None,
) -> D3Parameters:
    """Load one cached DFT-D3 parameter set into kernel-ready form."""

    resolved = DFTD3ParametersProcessor.resolve_param_file(
        param_path=param_path,
        cache_dir=cache_dir,
    )
    state_dict = torch.load(resolved, map_location="cpu", weights_only=True)
    return D3Parameters(
        rcov=state_dict["rcov"],
        r4r2=state_dict["r4r2"],
        c6ab=state_dict["c6ab"],
        cn_ref=state_dict["cn_ref"],
    )


class DFTD3ModelWrapper(nn.Module, BaseModelMixin):
    """DFT-D3(BJ) dispersion correction as a model wrapper.

    Parameters
    ----------
    config : DFTD3Config or None, optional
        Optional prebuilt DFT-D3 configuration object.
    functional, a1, a2, s6, s8, k1, k3, cutoff, smoothing_fraction, param_path
        Keyword overrides applied on top of ``config``.
    pbc_mode
        Optional override for the periodic-boundary support contract.
    device
        Execution device used for the DFT-D3 parameter tensors.
    name
        Optional stable display name used by the composable runtime.

    Attributes
    ----------
    spec : ModelConfig
        Execution contract describing the wrapper inputs, outputs, and
        neighbor-list requirements.
    """

    spec = _DFTD3ModelConfig

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
        smoothing_fraction: float = _UNSET,
        param_path: Path | None = _UNSET,
        pbc_mode: Literal["non-pbc", "pbc", "any"] | None = _UNSET,
        device: str | torch.device | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        base_config = config
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
                "smoothing_fraction": smoothing_fraction,
                "param_path": param_path,
            },
        )
        self._config = config
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self._d3_params = load_dftd3_params(config.param_path).to(device=self._device)
        self.spec = ModelConfig(
            required_inputs=_DFTD3ModelConfig.required_inputs,
            optional_inputs=_DFTD3ModelConfig.optional_inputs,
            outputs=_DFTD3ModelConfig.outputs,
            optional_outputs=dict(_DFTD3ModelConfig.optional_outputs),
            additive_outputs=_DFTD3ModelConfig.additive_outputs,
            use_autograd=_DFTD3ModelConfig.use_autograd,
            autograd_inputs=_DFTD3ModelConfig.autograd_inputs,
            autograd_outputs=_DFTD3ModelConfig.autograd_outputs,
            pbc_mode=_DFTD3ModelConfig.pbc_mode if pbc_mode is _UNSET else pbc_mode,
            neighbor_config=_DFTD3ModelConfig.neighbor_config.model_copy(
                update={"cutoff": config.cutoff}
            ),
        )

        explicit_values: dict[str, object] = {}
        if base_config is not None:
            for field_name in base_config.model_fields_set:
                explicit_values[field_name] = getattr(base_config, field_name)
        explicit_values.update(
            {
                key: value
                for key, value in {
                    "functional": functional,
                    "a1": a1,
                    "a2": a2,
                    "s6": s6,
                    "s8": s8,
                    "k1": k1,
                    "k3": k3,
                    "cutoff": cutoff,
                    "smoothing_fraction": smoothing_fraction,
                    "param_path": param_path,
                    "device": str(self._device),
                    "name": name,
                }.items()
                if value is not _UNSET
            }
        )
        defaults = DFTD3Config()
        initialize_model_repr(
            self,
            static_kwargs=collect_nondefault_repr_kwargs(
                explicit_values=explicit_values,
                defaults={
                    "functional": defaults.functional,
                    "a1": defaults.a1,
                    "a2": defaults.a2,
                    "s6": defaults.s6,
                    "s8": defaults.s8,
                    "k1": defaults.k1,
                    "k3": defaults.k3,
                    "cutoff": defaults.cutoff,
                    "smoothing_fraction": defaults.smoothing_fraction,
                    "param_path": defaults.param_path,
                    "device": "cpu",
                    "name": None,
                },
                order=(
                    "functional",
                    "a1",
                    "a2",
                    "s6",
                    "s8",
                    "k1",
                    "k3",
                    "cutoff",
                    "smoothing_fraction",
                    "param_path",
                    "device",
                    "name",
                ),
            ),
            kwarg_order=(
                "functional",
                "a1",
                "a2",
                "s6",
                "s8",
                "k1",
                "k3",
                "cutoff",
                "smoothing_fraction",
                "param_path",
                "device",
                "name",
            ),
        )

    def __repr__(self) -> str:
        """Return a compact constructor-style representation."""

        self._repr_kwargs["device"] = str(self._device)
        return build_model_repr(self)

    @property
    def device(self) -> torch.device:
        """Return the current execution device."""

        return self._device

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        """Move cached parameter tensors together with wrapper metadata."""

        result = super()._apply(fn)
        self._device = fn(torch.empty(0, device=self._device)).device
        if hasattr(self._d3_params, "to"):
            self._d3_params = self._d3_params.to(device=self._device)
        return result

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the DFT-D3(BJ) interaction kernel.

        Parameters
        ----------
        data
            Prepared input mapping resolved by the composable runtime. Required
            keys are ``positions``, ``atomic_numbers``, and ``neighbor_matrix``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping with ``energies`` and ``forces`` and, for periodic inputs,
            ``stresses``.
        """

        positions = data["positions"].to(device=self.device)
        numbers = data["atomic_numbers"].to(device=self.device, dtype=torch.int32)
        neighbor_matrix = data["neighbor_matrix"].to(device=self.device).contiguous()
        batch_idx, num_systems = normalize_batch_indices(
            positions, mapping_get(data, "batch")
        )

        cell = mapping_get(data, "cell")
        pbc = mapping_get(data, "pbc")
        periodic = (
            cell is not None
            and pbc is not None
            and bool(torch.as_tensor(pbc).any().item())
        )
        neighbor_shifts = resolve_matrix_neighbor_shifts(
            neighbor_matrix,
            mapping_get(data, "neighbor_shifts"),
            periodic=periodic,
            model_name=type(self).__name__,
            allow_missing_non_pbc=True,
        )

        positions_bohr = positions * ANGSTROM_TO_BOHR
        cell_bohr = None
        if cell is not None:
            cell_bohr = (cell if cell.ndim == 3 else cell.unsqueeze(0)).to(
                device=self.device,
                dtype=positions.dtype,
            ) * ANGSTROM_TO_BOHR

        smoothing_on = self._config.cutoff * (1.0 - self._config.smoothing_fraction)
        smoothing_off = self._config.cutoff
        result = dftd3(
            positions=positions_bohr,
            numbers=numbers,
            a1=self._config.a1,
            a2=self._config.a2,
            s8=self._config.s8,
            k1=self._config.k1,
            k3=self._config.k3,
            s6=self._config.s6,
            s5_smoothing_on=smoothing_on * ANGSTROM_TO_BOHR,
            s5_smoothing_off=smoothing_off * ANGSTROM_TO_BOHR,
            d3_params=self._d3_params,
            fill_value=-1,
            batch_idx=batch_idx.to(torch.int32),
            cell=cell_bohr,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_shifts,
            compute_virial=periodic,
            num_systems=num_systems,
        )

        if periodic:
            energy_ha, forces_ha_bohr, _, virial_ha = result
        else:
            energy_ha, forces_ha_bohr, _ = result
            virial_ha = None

        energies = energy_ha.to(positions.dtype)
        if energies.ndim == 1:
            energies = energies.unsqueeze(-1)
        energies = energies * HARTREE_TO_EV
        forces = forces_ha_bohr.to(positions.dtype) * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        outputs: dict[str, torch.Tensor] = {
            "energies": energies,
            "forces": forces,
        }
        if virial_ha is not None and cell is not None:
            outputs["stresses"] = virial_to_stress(
                virial_ha.to(positions.dtype) * HARTREE_TO_EV,
                cell if cell.ndim == 3 else cell.unsqueeze(0),
            )
        return outputs
