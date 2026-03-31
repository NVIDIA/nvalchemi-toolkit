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

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nvalchemi.models.metadata import CheckpointInfo

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "nvalchemi" / "models"


class ArtifactProcessor(Protocol):
    """Materialize one final cached artifact from a downloaded source file."""

    def materialize(
        self,
        *,
        downloaded_path: Path,
        entry: KnownArtifactEntry,
        output_path: Path,
    ) -> Path:
        """Create or update the final cached artifact and return its path."""


class ArtifactResolver(Protocol):
    """Resolve one known artifact through custom family-specific logic."""

    def __call__(
        self,
        *,
        entry: KnownArtifactEntry,
        cache_dir: Path,
        force_redownload: bool,
        allow_download: bool,
    ) -> Path:
        """Return the resolved local path for one entry."""


@dataclass(frozen=True)
class KnownArtifactEntry:
    """Registry entry for one known downloadable or externally resolved artifact.

    Attributes
    ----------
    name : str
        Canonical artifact name used for registry look-up.
    family : str
        Model family this artifact belongs to (e.g. ``"mace"``).
    aliases : tuple[str, ...]
        Alternative names that resolve to this entry.
    url : str or None
        Direct download URL.
    sha256 : str or None
        Expected SHA-256 hex digest for integrity verification.
    md5 : str or None
        Expected MD5 hex digest for integrity verification.
    cache_subdir : str or None
        Subdirectory under the cache root.
    filename : str or None
        Explicit filename for the cached download.
    processor : ArtifactProcessor or None
        Post-download materialisation step (e.g. parameter extraction).
    materialized_filename : str or None
        Filename for the processed artifact on disk.
    resolver : ArtifactResolver or None
        Family-specific resolution logic (bypasses the default
        download path).
    metadata : dict[str, Any]
        Arbitrary key-value metadata stored alongside the entry.
    """

    name: str
    family: str
    aliases: tuple[str, ...] = ()
    url: str | None = None
    sha256: str | None = None
    md5: str | None = None
    cache_subdir: str | None = None
    filename: str | None = None
    processor: ArtifactProcessor | None = None
    materialized_filename: str | None = None
    resolver: ArtifactResolver | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedArtifact:
    """Resolved local artifact path plus registry provenance.

    Attributes
    ----------
    entry : KnownArtifactEntry
        The matched registry entry.
    local_path : Path
        Absolute path to the cached artifact on disk.
    checkpoint : CheckpointInfo
        Provenance metadata derived from the registry entry.
    """

    entry: KnownArtifactEntry
    local_path: Path
    checkpoint: CheckpointInfo


class Dftd3ParametersProcessor:
    """Convert the official DFT-D3 reference archive into cached parameter tensors."""

    def materialize(
        self,
        *,
        downloaded_path: Path,
        entry: KnownArtifactEntry,
        output_path: Path,
    ) -> Path:
        """Extract parameters from one downloaded archive and save them to disk."""

        from nvalchemi.models.dftd3 import (
            extract_dftd3_parameters_from_archive,
            save_dftd3_parameters,
        )

        del entry
        parameters = extract_dftd3_parameters_from_archive(downloaded_path)
        return save_dftd3_parameters(parameters, output_path)


__all__ = [
    "ArtifactProcessor",
    "ArtifactResolver",
    "DEFAULT_CACHE_DIR",
    "Dftd3ParametersProcessor",
    "KnownArtifactEntry",
    "ResolvedArtifact",
    "get_known_artifact",
    "list_known_artifacts",
    "register_known_artifact",
    "resolve_known_artifact",
]

_REGISTRY: dict[str, dict[str, KnownArtifactEntry]] = {}


def _normalize_family(family: str) -> str:
    """Normalize one family key for registry storage."""

    return family.strip().lower()


def _normalize_name(name: str) -> str:
    """Normalize one canonical name or alias for registry storage."""

    return name.strip().lower()


def register_known_artifact(entry: KnownArtifactEntry) -> None:
    """Register one known artifact entry for one model family.

    Parameters
    ----------
    entry
        The artifact entry to register.

    Raises
    ------
    ValueError
        If any of the entry's names or aliases is already registered
        under the same family.
    """

    family = _normalize_family(entry.family)
    mapping = _REGISTRY.setdefault(family, {})
    names = (entry.name, *entry.aliases)
    for raw_name in names:
        key = _normalize_name(raw_name)
        if key in mapping:
            raise ValueError(
                f"Known artifact '{raw_name}' is already registered for family "
                f"'{entry.family}'."
            )
    for raw_name in names:
        mapping[_normalize_name(raw_name)] = entry


def get_known_artifact(name: str, family: str) -> KnownArtifactEntry:
    """Return one known artifact entry by family and canonical name or alias.

    Parameters
    ----------
    name
        Canonical name or alias of the artifact.
    family
        Model family (e.g. ``"mace"``, ``"dftd3"``).

    Returns
    -------
    KnownArtifactEntry
        The matched registry entry.

    Raises
    ------
    KeyError
        If the family or name is not found in the registry.
    """

    family_key = _normalize_family(family)
    try:
        mapping = _REGISTRY[family_key]
    except KeyError as exc:
        raise KeyError(
            f"No known artifacts are registered for family '{family}'."
        ) from exc

    key = _normalize_name(name)
    if key not in mapping:
        available = sorted({entry.name for entry in mapping.values()})
        raise KeyError(
            f"Known artifact '{name}' not found for family '{family}'. "
            f"Available names: {available}"
        )
    return mapping[key]


def list_known_artifacts(family: str | None = None) -> list[str]:
    """List canonical known artifact names.

    Parameters
    ----------
    family
        Restrict listing to one model family.  When ``None``, all
        families are included.

    Returns
    -------
    list[str]
        Sorted list of canonical artifact names.
    """

    if family is not None:
        mapping = _REGISTRY.get(_normalize_family(family), {})
        return sorted({entry.name for entry in mapping.values()})
    return sorted(
        {
            entry.name
            for family_entries in _REGISTRY.values()
            for entry in family_entries.values()
        }
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _entry_cache_dir(entry: KnownArtifactEntry, cache_root: Path) -> Path:
    """Return the cache directory for one entry."""

    subdir = entry.cache_subdir or entry.family
    path = cache_root / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_destination(entry: KnownArtifactEntry, cache_dir: Path) -> Path:
    """Return the cached download destination for one entry."""

    if entry.filename is not None:
        filename = entry.filename
    elif entry.url is not None:
        filename = entry.url.rsplit("/", 1)[-1]
    else:
        raise ValueError(
            f"Known artifact '{entry.name}' has neither an explicit filename nor a URL."
        )
    return cache_dir / filename


def _materialized_destination(entry: KnownArtifactEntry, cache_dir: Path) -> Path:
    """Return the final materialized artifact path for one entry."""

    filename = entry.materialized_filename or entry.filename
    if filename is None and entry.url is not None:
        filename = entry.url.rsplit("/", 1)[-1]
    if filename is None:
        raise ValueError(
            f"Known artifact '{entry.name}' needs a materialized filename."
        )
    return cache_dir / filename


def _download_entry(
    entry: KnownArtifactEntry,
    *,
    cache_dir: Path,
    force_redownload: bool,
    allow_download: bool,
) -> Path:
    """Download one registry entry into the cache and optionally verify its hash."""

    if entry.url is None:
        raise ValueError(
            f"Known artifact '{entry.name}' does not define a direct download URL."
        )

    destination = _download_destination(entry, cache_dir)
    if destination.exists() and not force_redownload:
        sha256_ok = entry.sha256 is None or _sha256_file(destination) == entry.sha256
        md5_ok = True
        if entry.md5 is not None:
            hasher = hashlib.md5(usedforsecurity=False)
            hasher.update(destination.read_bytes())
            md5_ok = hasher.hexdigest() == entry.md5
        if sha256_ok and md5_ok:
            return destination
        destination.unlink(missing_ok=True)

    if not allow_download:
        raise FileNotFoundError(
            f"Known artifact '{entry.name}' is not cached at '{destination}' and "
            "downloads are disabled."
        )

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{destination.name}.tmp",
    )
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        urllib.request.urlretrieve(entry.url, tmp_path)  # noqa: S310
        if entry.sha256 is not None and _sha256_file(tmp_path) != entry.sha256:
            raise RuntimeError(
                f"Downloaded artifact '{entry.name}' failed SHA-256 verification."
            )
        if entry.md5 is not None:
            hasher = hashlib.md5(usedforsecurity=False)
            hasher.update(tmp_path.read_bytes())
            if hasher.hexdigest() != entry.md5:
                raise RuntimeError(
                    f"Downloaded artifact '{entry.name}' failed MD5 verification."
                )
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return destination


def resolve_known_artifact(
    name: str,
    family: str,
    *,
    force_redownload: bool = False,
    cache_dir: Path | str | None = None,
    allow_download: bool = True,
) -> ResolvedArtifact:
    """Resolve one known artifact to a local path and provenance metadata.

    Parameters
    ----------
    name
        Canonical name or alias of the artifact.
    family
        Model family (e.g. ``"mace"``, ``"dftd3"``).
    force_redownload
        Re-download even if a cached copy exists.
    cache_dir
        Root directory for cached downloads.  Defaults to
        :data:`DEFAULT_CACHE_DIR`.
    allow_download
        Allow network downloads.  When ``False`` and the artifact is
        not cached, raises :class:`FileNotFoundError`.

    Returns
    -------
    ResolvedArtifact
        Resolved local path and provenance metadata.

    Raises
    ------
    KeyError
        If the artifact is not registered.
    FileNotFoundError
        If the artifact is not cached and downloads are disabled.
    """

    entry = get_known_artifact(name, family)
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    entry_cache_dir = _entry_cache_dir(entry, cache_root)

    if entry.processor is not None:
        final_path = _materialized_destination(entry, entry_cache_dir)
        if not force_redownload and final_path.exists():
            local_path = final_path
        else:
            if entry.resolver is not None:
                downloaded = entry.resolver(
                    entry=entry,
                    cache_dir=entry_cache_dir,
                    force_redownload=force_redownload,
                    allow_download=allow_download,
                )
            else:
                downloaded = _download_entry(
                    entry,
                    cache_dir=entry_cache_dir,
                    force_redownload=force_redownload,
                    allow_download=allow_download,
                )
            local_path = entry.processor.materialize(
                downloaded_path=downloaded,
                entry=entry,
                output_path=final_path,
            )
    elif entry.resolver is not None:
        local_path = entry.resolver(
            entry=entry,
            cache_dir=entry_cache_dir,
            force_redownload=force_redownload,
            allow_download=allow_download,
        )
    else:
        local_path = _download_entry(
            entry,
            cache_dir=entry_cache_dir,
            force_redownload=force_redownload,
            allow_download=allow_download,
        )

    return ResolvedArtifact(
        entry=entry,
        local_path=Path(local_path),
        checkpoint=CheckpointInfo(
            identifier=name,
            url=entry.url,
            sha256=entry.sha256,
            source="registry",
        ),
    )


def _resolve_mace_foundation_checkpoint(
    *,
    entry: KnownArtifactEntry,
    cache_dir: Path,
    force_redownload: bool,
    allow_download: bool,
) -> Path:
    """Resolve one known MACE foundation-model checkpoint through mace-torch."""

    del cache_dir, force_redownload, allow_download
    upstream_name = str(entry.metadata.get("upstream_name", entry.name))
    try:
        from mace.calculators.foundations_models import (
            download_mace_mp_checkpoint,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "mace-torch is required to resolve named MACE artifacts from the "
            "models registry."
        ) from exc
    return Path(download_mace_mp_checkpoint(upstream_name))


def _register_builtin_mace_entries() -> None:
    """Register built-in known MACE foundation checkpoints."""

    entries = (
        KnownArtifactEntry(
            name="mace-mp-0b3-medium",
            family="mace",
            aliases=("mace-mp-0b3",),
            url=(
                "https://github.com/ACEsuit/mace-mp/releases/download/"
                "mace_mp_0b3/mace-mp-0b3-medium.model"
            ),
            resolver=_resolve_mace_foundation_checkpoint,
            metadata={
                "model_name": "mace-mp-0b3-medium",
                "upstream_name": "medium-0b3",
            },
        ),
        KnownArtifactEntry(
            name="mace-mpa-0-medium",
            family="mace",
            aliases=("mace-mpa-0",),
            url=(
                "https://github.com/ACEsuit/mace-mp/releases/download/"
                "mace_mpa_0/"
                "mace-mpa-0-medium.model"
            ),
            resolver=_resolve_mace_foundation_checkpoint,
            metadata={
                "model_name": "mace-mpa-0-medium",
                "upstream_name": "medium-mpa-0",
            },
        ),
    )
    for entry in entries:
        register_known_artifact(entry)


def _register_builtin_aimnet_entries() -> None:
    """Register the single built-in AIMNet2 checkpoint."""

    register_known_artifact(
        KnownArtifactEntry(
            name="aimnet2",
            family="aimnet2",
            url=(
                "https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/"
                "aimnet2_wb97m_d3_0.pt"
            ),
            filename="aimnet2_wb97m_d3_0.pt",
            cache_subdir="aimnet2",
            metadata={
                "model_name": "aimnet2",
                "reference_xc_functional": "wb97m",
            },
        )
    )


def _register_builtin_dftd3_entry() -> None:
    """Register the built-in processed DFT-D3 parameter artifact."""

    register_known_artifact(
        KnownArtifactEntry(
            name="dftd3_parameters",
            family="dftd3",
            aliases=("dftd3",),
            url="https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz",
            md5="a76c752e587422c239c99109547516d2",
            filename="dftd3.tgz",
            cache_subdir="nvalchemiops",
            processor=Dftd3ParametersProcessor(),
            materialized_filename="dftd3_parameters.pt",
            metadata={"model_name": "dftd3_parameters"},
        )
    )


_register_builtin_mace_entries()
_register_builtin_aimnet_entries()
_register_builtin_dftd3_entry()
