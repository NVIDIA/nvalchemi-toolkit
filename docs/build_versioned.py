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

"""Build the versioned Sphinx documentation site."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
DEFAULT_OUTPUT_DIR = DOCS_ROOT / "_build" / "site"
DEFAULT_SITE_URL = "https://nvidia.github.io/nvalchemi-toolkit"


def _run(command: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command from the repository root."""
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)  # noqa: S603


def _versioned_tags(output_dir: Path) -> list[str]:
    """Return release tags that were emitted by sphinx-multiversion."""
    result = subprocess.run(  # noqa: S603
        ["git", "tag", "--sort=-v:refname", "--list", "v[0-9]*"],  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    available_versions = {
        path.name
        for path in output_dir.iterdir()
        if path.is_dir() and (path / "index.html").exists()
    }
    return [
        tag
        for tag in result.stdout.splitlines()
        if tag.strip() and tag.strip() in available_versions
    ]


def _site_versions(output_dir: Path, site_url: str) -> list[dict[str, str]]:
    """Return PyData-compatible version switcher entries."""
    normalized_site_url = site_url.rstrip("/")
    versions = []
    if (output_dir / "main" / "index.html").exists():
        versions.append(
            {
                "name": "main (development)",
                "version": "main",
                "url": f"{normalized_site_url}/main/",
            }
        )

    versions.extend(
        [
            {
                "name": tag,
                "version": tag,
                "url": f"{normalized_site_url}/{tag}/",
            }
            for tag in _versioned_tags(output_dir)
        ]
    )
    return versions


def _write_version_manifests(output_dir: Path, site_url: str) -> None:
    """Write root and theme-static version switcher manifests."""
    versions = _site_versions(output_dir, site_url)
    manifest = json.dumps(versions, indent=2) + "\n"

    (output_dir / "versions.json").write_text(
        manifest,
        encoding="utf-8",
    )
    static_dir = output_dir / "_static"
    static_dir.mkdir(exist_ok=True)
    (static_dir / "switcher.json").write_text(manifest, encoding="utf-8")


def _write_root_redirect(output_dir: Path) -> None:
    """Write a root page that redirects to the main documentation."""
    (output_dir / "index.html").write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=main/">
    <link rel="canonical" href="main/">
    <title>ALCHEMI Toolkit documentation</title>
  </head>
  <body>
    <p><a href="main/">Continue to the ALCHEMI Toolkit documentation.</a></p>
    <script>window.location.replace("main/" + window.location.search + window.location.hash);</script>
  </body>
</html>
""",
        encoding="utf-8",
    )


def _write_legacy_404_redirect(output_dir: Path) -> None:
    """Write a GitHub Pages fallback for pre-versioned deep links."""
    (output_dir / "404.html").write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>ALCHEMI Toolkit documentation</title>
  </head>
  <body>
    <p>Redirecting to the current ALCHEMI Toolkit documentation.</p>
    <script>
      (function () {
        var root = "/nvalchemi-toolkit/";
        var path = window.location.pathname;
        if (path.indexOf(root) !== 0) {
          return;
        }
        var relative = path.slice(root.length);
        if (!relative || relative.indexOf("main/") === 0 || /^v\\d/.test(relative)) {
          return;
        }
        window.location.replace(root + "main/" + relative + window.location.search + window.location.hash);
      }());
    </script>
  </body>
</html>
""",
        encoding="utf-8",
    )


def build_versioned_docs(output_dir: Path, site_url: str) -> None:
    """Build the versioned documentation site into an output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    normalized_site_url = site_url.rstrip("/")
    build_env = os.environ.copy()
    build_env["DOCS_SITE_URL"] = normalized_site_url
    build_env["DOCS_SWITCHER_JSON_URL"] = f"{normalized_site_url}/_static/switcher.json"

    _run(["sphinx-multiversion", str(DOCS_ROOT), str(output_dir)], env=build_env)
    (output_dir / ".nojekyll").touch()
    _write_version_manifests(output_dir, site_url)
    _write_root_redirect(output_dir)
    _write_legacy_404_redirect(output_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the complete versioned site will be written.",
    )
    parser.add_argument(
        "--site-url",
        default=DEFAULT_SITE_URL,
        help="Public site URL used in versions.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the versioned documentation site."""
    args = parse_args()
    build_versioned_docs(args.output_dir, args.site_url)


if __name__ == "__main__":
    main()
