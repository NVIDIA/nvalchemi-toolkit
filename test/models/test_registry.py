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
"""Tests for nvalchemi.models.registry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

import nvalchemi.models.registry as _reg_module
from nvalchemi.models.registry import (
    ModelRegistryEntry,
    _sha256_file,
    download_and_verify,
    get_registry_entry,
    list_foundation_models,
    register_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(name: str, aliases: list[str]) -> ModelRegistryEntry:
    """Factory for a ModelRegistryEntry with a deterministic fake sha256."""
    fake_sha256 = hashlib.sha256(name.encode()).hexdigest()
    return ModelRegistryEntry(
        name=name,
        url=f"https://example.com/{name}.model",
        sha256=fake_sha256,
        model_class="FakeWrapper",
        description=f"Test entry for {name}",
        aliases=aliases,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace _REGISTRY with an empty dict for the duration of the test."""
    fresh: dict = {}
    monkeypatch.setattr(_reg_module, "_REGISTRY", fresh)
    return fresh


# ---------------------------------------------------------------------------
# TestBuiltins
# ---------------------------------------------------------------------------


class TestBuiltins:
    """Verify that the built-in foundation model entries are registered."""

    def test_list_contains_mace_mp_small(self):
        models = list_foundation_models()
        assert "mace-mp-0b2-small" in models

    def test_list_contains_mace_mp_medium(self):
        models = list_foundation_models()
        assert "mace-mp-0b2-medium" in models

    def test_list_contains_mace_mp_large(self):
        models = list_foundation_models()
        assert "mace-mp-0b2-large" in models

    def test_list_contains_mace_mpa_medium(self):
        models = list_foundation_models()
        assert "mace-mpa-0b3-medium" in models

    def test_list_is_sorted(self):
        models = list_foundation_models()
        assert models == sorted(models)

    def test_list_excludes_aliases(self):
        """Aliases like 'mace-mp-medium' must not appear in the canonical list."""
        models = list_foundation_models()
        assert "mace-mp-medium" not in models
        assert "mace-mp" not in models
        assert "mace-mp-small" not in models
        assert "mace-mp-large" not in models
        assert "mace-mp-0b2" not in models

    def test_list_no_duplicates(self):
        models = list_foundation_models()
        assert len(models) == len(set(models))

    def test_get_canonical_name(self):
        entry = get_registry_entry("mace-mp-0b2-medium")
        assert entry.name == "mace-mp-0b2-medium"

    def test_get_via_alias_medium(self):
        entry = get_registry_entry("mace-mp-medium")
        assert entry.name == "mace-mp-0b2-medium"

    def test_get_via_alias_mace_mp(self):
        entry = get_registry_entry("mace-mp")
        assert entry.name == "mace-mp-0b2-medium"

    def test_get_via_alias_mace_mp_0b2(self):
        entry = get_registry_entry("mace-mp-0b2")
        assert entry.name == "mace-mp-0b2-medium"

    def test_get_via_alias_small(self):
        entry = get_registry_entry("mace-mp-small")
        assert entry.name == "mace-mp-0b2-small"

    def test_get_via_alias_large(self):
        entry = get_registry_entry("mace-mp-large")
        assert entry.name == "mace-mp-0b2-large"


# ---------------------------------------------------------------------------
# TestRegisterModel
# ---------------------------------------------------------------------------


class TestRegisterModel:
    """Tests for register_model()."""

    def test_register_canonical(self, isolated_registry):
        e = _entry("model-a", [])
        register_model(e)
        assert "model-a" in isolated_registry
        assert isolated_registry["model-a"] is e

    def test_raises_on_duplicate_canonical(self, isolated_registry):
        e = _entry("model-dup", [])
        register_model(e)
        with pytest.raises(ValueError, match="already registered"):
            register_model(_entry("model-dup", []))

    def test_register_aliases(self, isolated_registry):
        e = _entry("model-b", ["alias-b1", "alias-b2"])
        register_model(e)
        assert isolated_registry["alias-b1"] is e
        assert isolated_registry["alias-b2"] is e

    def test_alias_collision_warns_and_preserves_original(self, isolated_registry):
        first = _entry("first-model", ["shared-alias"])
        register_model(first)
        second = _entry("second-model", ["shared-alias"])
        with pytest.warns(UserWarning, match="shared-alias"):
            register_model(second)
        # The original mapping is preserved
        assert isolated_registry["shared-alias"] is first

    def test_alias_collision_does_not_prevent_canonical_registration(
        self, isolated_registry
    ):
        first = _entry("first-model", ["shared-alias"])
        register_model(first)
        second = _entry("second-model", ["shared-alias"])
        with pytest.warns(UserWarning):
            register_model(second)
        assert "second-model" in isolated_registry


# ---------------------------------------------------------------------------
# TestGetRegistryEntry
# ---------------------------------------------------------------------------


class TestGetRegistryEntry:
    """Tests for get_registry_entry()."""

    def test_returns_entry_for_canonical(self, isolated_registry):
        e = _entry("my-model", [])
        register_model(e)
        assert get_registry_entry("my-model") is e

    def test_returns_entry_for_alias(self, isolated_registry):
        e = _entry("my-model", ["my-alias"])
        register_model(e)
        assert get_registry_entry("my-alias") is e

    def test_raises_key_error_for_unknown(self, isolated_registry):
        with pytest.raises(KeyError, match="not found in registry"):
            get_registry_entry("totally-unknown-model-xyz")

    def test_error_message_lists_available(self, isolated_registry):
        e = _entry("listed-model", [])
        register_model(e)
        with pytest.raises(KeyError, match="listed-model"):
            get_registry_entry("missing-model")


# ---------------------------------------------------------------------------
# TestSha256File
# ---------------------------------------------------------------------------


class TestSha256File:
    """Tests for _sha256_file()."""

    def test_known_bytes(self, tmp_path: Path):
        data = b"hello nvalchemi registry"
        expected = hashlib.sha256(data).hexdigest()
        p = tmp_path / "test.bin"
        p.write_bytes(data)
        assert _sha256_file(p) == expected

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert _sha256_file(p) == expected

    def test_large_file_chunked(self, tmp_path: Path):
        """File larger than 1 MiB exercises the chunked-read loop."""
        chunk = b"A" * (1 << 20)  # 1 MiB
        data = chunk * 3  # 3 MiB
        expected = hashlib.sha256(data).hexdigest()
        p = tmp_path / "large.bin"
        p.write_bytes(data)
        assert _sha256_file(p) == expected


# ---------------------------------------------------------------------------
# TestDownloadAndVerify
# ---------------------------------------------------------------------------


class TestDownloadAndVerify:
    """Tests for download_and_verify()."""

    def _write_entry_file(self, cache_dir: Path, entry: ModelRegistryEntry) -> Path:
        """Write a file whose contents hash to entry.sha256."""
        filename = entry.url.split("/")[-1]
        dest = cache_dir / filename
        # We'll use the entry name as the payload and compute its hash.
        # The entry fixture uses sha256(name.encode()) as sha256, so we write
        # the name bytes to get a matching file.
        dest.write_bytes(entry.name.encode())
        return dest

    def _make_entry_with_real_hash(
        self, name: str, payload: bytes
    ) -> ModelRegistryEntry:
        """Create a registry entry whose sha256 matches payload."""
        sha = hashlib.sha256(payload).hexdigest()
        return ModelRegistryEntry(
            name=name,
            url=f"https://example.com/{name}.model",
            sha256=sha,
            model_class="FakeWrapper",
        )

    def test_cache_hit_skips_urlretrieve(self, tmp_path: Path):
        payload = b"valid cached model weights"
        entry = self._make_entry_with_real_hash("cached-model", payload)
        filename = entry.url.split("/")[-1]
        (tmp_path / filename).write_bytes(payload)

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            result = download_and_verify(entry, cache_dir=tmp_path)
            mock_retrieve.assert_not_called()

        assert result == tmp_path / filename

    def test_cache_miss_calls_urlretrieve(self, tmp_path: Path):
        payload = b"fresh model weights"
        entry = self._make_entry_with_real_hash("fresh-model", payload)
        filename = entry.url.split("/")[-1]

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(payload)

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            result = download_and_verify(entry, cache_dir=tmp_path)

        assert result == tmp_path / filename
        assert result.read_bytes() == payload

    def test_stale_cache_warns_and_redownloads(self, tmp_path: Path):
        payload = b"correct model weights"
        entry = self._make_entry_with_real_hash("stale-model", payload)
        filename = entry.url.split("/")[-1]
        # Write stale (wrong) content
        (tmp_path / filename).write_bytes(b"stale garbage")

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(payload)

        with (
            patch("urllib.request.urlretrieve", side_effect=fake_retrieve),
            pytest.warns(UserWarning, match="SHA-256"),
        ):
            result = download_and_verify(entry, cache_dir=tmp_path)

        assert result.read_bytes() == payload

    def test_hash_mismatch_after_download_raises(self, tmp_path: Path):
        payload = b"correct model weights"
        entry = self._make_entry_with_real_hash("mismatch-model", payload)

        def fake_retrieve_bad(url, dest):
            Path(dest).write_bytes(b"corrupted bytes")

        with (
            patch("urllib.request.urlretrieve", side_effect=fake_retrieve_bad),
            pytest.raises(RuntimeError, match="SHA-256"),
        ):
            download_and_verify(entry, cache_dir=tmp_path)

    def test_hash_mismatch_cleans_up_temp_file(self, tmp_path: Path):
        payload = b"correct"
        entry = self._make_entry_with_real_hash("cleanup-model", payload)

        def fake_retrieve_bad(url, dest):
            Path(dest).write_bytes(b"corrupted")

        with (
            patch("urllib.request.urlretrieve", side_effect=fake_retrieve_bad),
            pytest.raises(RuntimeError),
        ):
            download_and_verify(entry, cache_dir=tmp_path)

        # No .*.tmp* files should remain
        leftover = list(tmp_path.glob(".*.tmp*"))
        assert leftover == [], f"Temp files were not cleaned up: {leftover}"

    def test_force_redownload_bypasses_valid_cache(self, tmp_path: Path):
        payload = b"valid cached content"
        entry = self._make_entry_with_real_hash("force-model", payload)
        filename = entry.url.split("/")[-1]
        (tmp_path / filename).write_bytes(payload)

        new_payload = b"freshly downloaded content"
        new_sha = hashlib.sha256(new_payload).hexdigest()
        fresh_entry = ModelRegistryEntry(
            name="force-model",
            url=entry.url,
            sha256=new_sha,
            model_class="FakeWrapper",
        )

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(new_payload)

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve) as mock_r:
            result = download_and_verify(
                fresh_entry, cache_dir=tmp_path, force_redownload=True
            )
            mock_r.assert_called_once()

        assert result.read_bytes() == new_payload

    def test_exception_mid_download_removes_temp_file(self, tmp_path: Path):
        payload = b"correct"
        entry = self._make_entry_with_real_hash("exception-model", payload)

        def fake_retrieve_raises(url, dest):
            Path(dest).write_bytes(b"partial")
            raise OSError("network error")

        with (
            patch("urllib.request.urlretrieve", side_effect=fake_retrieve_raises),
            pytest.raises(OSError, match="network error"),
        ):
            download_and_verify(entry, cache_dir=tmp_path)

        leftover = list(tmp_path.glob(".*.tmp*"))
        assert leftover == [], f"Temp files were not cleaned up: {leftover}"

    def test_cache_dir_created_if_absent(self, tmp_path: Path):
        payload = b"model data"
        entry = self._make_entry_with_real_hash("newdir-model", payload)
        new_dir = tmp_path / "deep" / "nested" / "cache"
        assert not new_dir.exists()

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(payload)

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            result = download_and_verify(entry, cache_dir=new_dir)

        assert new_dir.exists()
        assert result.exists()
