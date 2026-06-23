# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import logging
import os
import pathlib
import sys
from importlib.metadata import version

import dotenv
from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

# -- Load environment vars -----------------------------------------------------
# Note: To override, use environment variables (e.g. PLOT_GALLERY=True make html)
# Defaults will build API docs and execute examples
dotenv.load_dotenv()
# Enable plotting in example scripts so sphinx-gallery captures figure thumbnails.
os.environ.setdefault("NVALCHEMI_PLOT", "1")
doc_version = os.getenv("DOC_VERSION", "main")
current_doc_version = os.getenv("SPHINX_MULTIVERSION_NAME", doc_version)
docs_site_url = os.getenv("DOCS_SITE_URL", "https://nvidia.github.io/nvalchemi-toolkit")
docs_switcher_json_url = os.getenv(
    "DOCS_SWITCHER_JSON_URL",
    f"{docs_site_url.rstrip('/')}/_static/switcher.json",
)
plot_gallery = os.getenv("PLOT_GALLERY", "True").lower() in ("true", "1", "yes")
run_stale_examples = os.getenv("RUN_STALE_EXAMPLES", "False").lower() in (
    "true",
    "1",
    "yes",
)
filename_pattern = os.getenv(
    "FILENAME_PATTERN", r"/[0-9]+.*\.py"
)  # Match numbered .py files
logging.info(
    f"Doc config - version: {current_doc_version}, plot_gallery: {plot_gallery}, run_stale: {run_stale_examples}"
)

root = pathlib.Path(__file__).parent
release = version("nvalchemi-toolkit")

sys.path.insert(0, root.parent.as_posix())
# Add current folder to use sphinxext.py
sys.path.insert(0, os.path.dirname(__file__))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
version = ".".join(release.split(".")[:2])
project = "ALCHEMI Toolkit"
copyright = "2026, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
    "sphinx_favicon",
    "myst_parser",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx.ext.graphviz",
    "sphinx_gallery.gen_gallery",
]

source_suffix = [".rst", ".md"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
templates_path = ["_templates"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]
exclude_patterns = [
    "_build",
    "sphinxext.py",
    "Thumbs.db",
    ".DS_Store",
]
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/nvidia-sphinx-theme.css",
]
html_theme_options = {
    "logo": {
        "text": "ALCHEMI Toolkit",
        "image_light": "_static/NVIDIA-Logo-V-ForScreen-ForLightBG.png",
        "image_dark": "_static/NVIDIA-Logo-V-ForScreen-ForDarkBG.png",
    },
    "navbar_align": "content",
    "navigation_with_keys": True,
    "navbar_start": [
        "navbar-logo",
    ],
    "navbar_end": [
        "version-switcher",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "switcher": {
        "json_url": docs_switcher_json_url,
        "version_match": current_doc_version,
    },
    # The switcher manifest is generated after sphinx-multiversion finishes.
    "check_switcher": False,
    "external_links": [
        {
            "name": "Changelog",
            "url": "https://github.com/NVIDIA/nvalchemi-toolkit/blob/main/CHANGELOG.md",
        },
    ],
    "icon_links": [
        {
            # Label for this link
            "name": "Github",
            # URL where the link will redirect
            "url": "https://www.github.com/NVIDIA/nvalchemi-toolkit",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "show_toc_level": 2,
}
favicons = ["favicon.ico"]

# https://sphinx-gallery.github.io/stable/configuration.html

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["examples"],
    "plot_gallery": plot_gallery,
    "filename_pattern": filename_pattern,
    "ignore_pattern": r"(^_|utils\.py$)",  # Exclude files starting with _ or utils.py
    "image_srcset": ["1x"],
    "subsection_order": ExplicitOrder(
        [
            "../examples/basic",
            "../examples/intermediate",
            "../examples/advanced",
            "../examples/distributed",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
    "run_stale_examples": run_stale_examples,
    "backreferences_dir": "modules/backreferences",
    "doc_module": ("nvalchemi",),
    "reset_modules": (
        "matplotlib",
        "sphinxext.reset_torch",
    ),
    "reset_modules_order": "both",
    "show_memory": False,
    "exclude_implicit_doc": {r"load_model", r"load_default_package"},
    "log_level": {"backreference_missing": "warning", "gallery_examples": "debug"},
    # Suppress thumbnail generation warnings for examples without plots
    "thumbnail_size": (250, 250),
    "min_reported_time": 0,
    "capture_repr": ("_repr_html_", "__repr__"),
}

# mapping for other packages
intersphinx_mapping = {"torch": ("https://pytorch.org/docs/stable/", None)}

# -- Versioned documentation -------------------------------------------------
# Build the development docs from main and immutable release docs from v* tags.
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"
smv_tag_whitelist = r"^v\d+\.\d+\.\d+.*$"
smv_released_pattern = r"^refs/tags/v.*$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = True
