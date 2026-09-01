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
"""Modality and intent enums for the toolkit-level generative API.

The enums here describe *what* a generative model produces (its
modalities) and *how* it is used (its intents). They are deliberately
generic. ``Modality`` enumerates the artifact kinds a
generative model may ingest or emit; ``GenerativeIntent`` enumerates the
operational roles a model can play. A
:class:`~nvalchemi.models.gen.base.GenerativeModelConfig` binds a set of
intents to the modalities each intent operates on.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["GenerativeIntent", "Modality"]


class Modality(str, Enum):
    """Artifact kinds a generative model may ingest or emit.

    The set is intentionally broad so the same config schema can describe
    atomic point clouds, periodic crystals, and non-atomic modalities (text,
    spectra, embeddings, images). Atomic generative models in this toolkit
    typically operate on :attr:`point_cloud`, :attr:`graph`, or
    :attr:`crystal`.

    Attributes
    ----------
    POINT_CLOUD
        An unordered set of atoms without connectivity (coordinates + numbers).
    GRAPH
        An atomic graph with explicit edges (a neighbor list).
    CRYSTAL
        A periodic crystal: atoms plus a lattice/cell and periodicity flags.
    TEXT
        A text / SMILES / string conditioning artifact.
    SPECTRA
        A one-dimensional spectroscopic or signal artifact.
    EMBEDDING
        A dense latent embedding artifact.
    IMAGE
        A gridded image artifact.
    """

    POINT_CLOUD = "point_cloud"
    GRAPH = "graph"
    CRYSTAL = "crystal"
    TEXT = "text"
    SPECTRA = "spectra"
    EMBEDDING = "embedding"
    IMAGE = "image"


class GenerativeIntent(str, Enum):
    """Operational roles a generative model can play.

    Intents are split into **output-producing** roles (the model *creates* an
    artifact) and **input-facing** roles (the model *consumes* an artifact as a
    condition, completion target, or transformation input).
    :class:`~nvalchemi.models.gen.base.GenerativeModelConfig` uses this
    classification to derive :attr:`input_modalities` and
    :attr:`output_modalities`.

    Attributes
    ----------
    CREATE
        Generate an artifact from a prior (unconditional generation).
    CONDITION
        Generate an artifact conditioned on a given input artifact.
    COMPLETE
        Complete a partially-specified artifact.
    TRANSFORM
        Transform one artifact into another (e.g. map a structure to a latent).
    CONNECT
        Connect two artifacts (e.g. embed two modalities into a shared space).
    DECODE
        Decode a latent/embedding into a concrete artifact.
    SAMPLE
        Draw samples from a learned distribution.
    PROPOSE
        Propose candidate artifacts (e.g. crystal structure proposal).
    """

    CREATE = "create"
    CONDITION = "condition"
    COMPLETE = "complete"
    TRANSFORM = "transform"
    CONNECT = "connect"
    DECODE = "decode"
    SAMPLE = "sample"
    PROPOSE = "propose"
