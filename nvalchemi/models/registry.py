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
"""Model registry for nvalchemi foundation model checkpoints.

Provides a lightweight registry of named foundation model checkpoints with
SHA-256 integrity verification and atomic download.

Usage
-----
::

    from nvalchemi.models.registry import list_foundation_models, download_and_verify, get_registry_entry

    print(list_foundation_models())
    path = download_and_verify(get_registry_entry("mace-mp-0b2-medium"))

Custom models can be registered for user-defined checkpoints::

    from nvalchemi.models.registry import register_model, ModelRegistryEntry

    register_model(ModelRegistryEntry(
        name="my-custom-mace",
        url="https://example.com/my_model.model",
        sha256="<sha256-hash>",
        model_class="MACEWrapper",
        description="Custom fine-tuned MACE model",
    ))
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ModelRegistryEntry",
    "register_model",
    "list_foundation_models",
    "get_registry_entry",
    "download_and_verify",
    "DEFAULT_CACHE_DIR",
]

DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "nvalchemi" / "models"


@dataclass
class ModelRegistryEntry:
    """Registry entry for a downloadable foundation model checkpoint.

    Attributes
    ----------
    name : str
        Canonical identifier for this model (e.g. ``"mace-mp-0b2-medium"``).
    url : str
        Direct download URL for the checkpoint file.
    sha256 : str
        Expected SHA-256 hex digest of the downloaded file.  Used to verify
        integrity and detect stale cached downloads.
    model_class : str
        Name of the wrapper class (e.g. ``"MACEWrapper"``).  String to avoid
        circular imports; the actual class is looked up at load time.
    default_kwargs : dict[str, Any]
        Default keyword arguments forwarded to ``from_checkpoint`` when loading
        via ``from_foundation_model``.
    description : str
        Human-readable description of the model.
    aliases : list[str]
        Alternative names that also resolve to this entry.
    """

    name: str
    url: str
    sha256: str
    model_class: str
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal registry store
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelRegistryEntry] = {}


def register_model(entry: ModelRegistryEntry) -> None:
    """Register a model entry by canonical name and any aliases.

    Parameters
    ----------
    entry : ModelRegistryEntry
        The entry to register.

    Raises
    ------
    ValueError
        If ``entry.name`` is already registered.
    """
    if entry.name in _REGISTRY:
        raise ValueError(
            f"Model '{entry.name}' is already registered. "
            "Use a unique name or call list_foundation_models() to see existing entries."
        )
    _REGISTRY[entry.name] = entry
    for alias in entry.aliases:
        if alias in _REGISTRY:
            warnings.warn(
                f"Alias '{alias}' is already registered; skipping alias registration.",
                UserWarning,
                stacklevel=2,
            )
        else:
            _REGISTRY[alias] = entry


def list_foundation_models() -> list[str]:
    """Return the sorted list of unique registered model canonical names.

    Aliases are excluded; only canonical names (``entry.name``) are returned.
    """
    return sorted({e.name for e in _REGISTRY.values()})


def get_registry_entry(name: str) -> ModelRegistryEntry:
    """Look up a registry entry by canonical name or alias.

    Parameters
    ----------
    name : str
        Model name or alias.

    Returns
    -------
    ModelRegistryEntry

    Raises
    ------
    KeyError
        If ``name`` is not found.  The error message lists available models.
    """
    if name not in _REGISTRY:
        available = list_foundation_models()
        raise KeyError(
            f"Model '{name}' not found in registry. Available models: {available}"
        )
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Download and verification
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify(
    entry: ModelRegistryEntry,
    cache_dir: Path | str | None = None,
    force_redownload: bool = False,
) -> Path:
    """Download a model checkpoint and verify its SHA-256 hash.

    Uses an atomic download strategy: writes to a temporary file, verifies the
    hash, then renames to the final destination.  This prevents partial writes
    from corrupting the cache.

    Parameters
    ----------
    entry : ModelRegistryEntry
        The registry entry describing the checkpoint to download.
    cache_dir : Path or str or None
        Directory to store downloaded checkpoints.  Defaults to
        ``~/.cache/nvalchemi/models``.
    force_redownload : bool
        If ``True``, re-download even if a cached file exists with a valid hash.

    Returns
    -------
    Path
        Local path to the verified checkpoint file.

    Raises
    ------
    RuntimeError
        If the downloaded file fails SHA-256 verification.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = entry.url.split("/")[-1]
    dest = cache_dir / filename

    def _verify(path: Path) -> bool:
        return _sha256_file(path) == entry.sha256

    # Use cached file if it exists and passes verification.
    if dest.exists() and not force_redownload:
        if _verify(dest):
            return dest
        else:
            warnings.warn(
                f"Cached checkpoint '{dest}' failed SHA-256 verification. "
                "Re-downloading.",
                UserWarning,
                stacklevel=2,
            )

    # Atomic download: write to temp file, verify, then rename.
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=cache_dir, prefix=f".{filename}.tmp")
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        print(f"Downloading {entry.name} from {entry.url} ...")
        urllib.request.urlretrieve(entry.url, tmp_path)  # noqa: S310
        if not _verify(tmp_path):
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded checkpoint for '{entry.name}' failed SHA-256 verification. "
                f"Expected: {entry.sha256}. "
                "The remote file may have been updated. "
                "Check the registry entry or report this at the project issue tracker."
            )
        tmp_path.rename(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return dest


# ---------------------------------------------------------------------------
# Built-in foundation model entries
# ---------------------------------------------------------------------------
# URLs and SHA-256 hashes for MACE-MP foundation models.
# Hashes must be updated if the remote files change.
# Source: https://github.com/ACEsuit/mace-mp

register_model(
    ModelRegistryEntry(
        name="mace-mp-0b2-small",
        url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/2023-12-10-mace-128-L0_energy_epoch-249.model",
        sha256="TODO_update_with_real_sha256",  # TODO: run sha256sum on the downloaded file
        model_class="MACEWrapper",
        default_kwargs={},
        description=(
            "MACE-MP-0b2 small (128 channels, L=0 equivariance). "
            "Universal potential trained on Materials Project DFT data."
        ),
        aliases=["mace-mp-small", "mace-mp-0b2-small"],
    )
)

register_model(
    ModelRegistryEntry(
        name="mace-mp-0b2-medium",
        url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/2023-12-10-mace-128-L1_energy_epoch-249.model",
        sha256="TODO_update_with_real_sha256",  # TODO: run sha256sum on the downloaded file
        model_class="MACEWrapper",
        default_kwargs={},
        description=(
            "MACE-MP-0b2 medium (128 channels, L=1 equivariance). "
            "Universal potential trained on Materials Project DFT data. "
            "Recommended default for most materials simulation tasks."
        ),
        aliases=["mace-mp-medium", "mace-mp", "mace-mp-0b2"],
    )
)

register_model(
    ModelRegistryEntry(
        name="mace-mp-0b2-large",
        url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/2023-12-10-mace-128-L2_energy_epoch-249.model",
        sha256="TODO_update_with_real_sha256",  # TODO: run sha256sum on the downloaded file
        model_class="MACEWrapper",
        default_kwargs={},
        description=(
            "MACE-MP-0b2 large (128 channels, L=2 equivariance). "
            "Universal potential trained on Materials Project DFT data. "
            "Higher accuracy at greater computational cost."
        ),
        aliases=["mace-mp-large"],
    )
)

register_model(
    ModelRegistryEntry(
        name="mace-mpa-0b3-medium",
        url="https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0b3/mace-mpa-0b3-medium.model",
        sha256="TODO_update_with_real_sha256",  # TODO: run sha256sum on the downloaded file
        model_class="MACEWrapper",
        default_kwargs={},
        description=(
            "MACE-MPA-0b3 medium. "
            "Universal potential trained on the Materials Project Alexandria dataset "
            "with broader coverage of chemical space."
        ),
        aliases=["mace-mpa-medium", "mace-mpa"],
    )
)
