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

"""Tests for the flat :class:`DomainConfig` model + :class:`HookScope`."""

from __future__ import annotations

from nvalchemi.distributed.config import DomainConfig, HookScope


class TestDomainConfig:
    def test_minimal(self):
        cfg = DomainConfig(cutoff=5.0)
        assert cfg.cutoff == 5.0
        assert cfg.skin == 0.0
        assert cfg.ghost_width is None
        assert cfg.mesh is None
        assert cfg.mesh_dim == "domain"
        assert cfg.grid_dims is None

    def test_full(self):
        cfg = DomainConfig(
            cutoff=5.0,
            skin=0.5,
            mesh="MESH_SENTINEL",
            mesh_dim="grid",
            ghost_width=7.0,
            grid_dims=(2, 2, 1),
        )
        assert cfg.cutoff == 5.0
        assert cfg.skin == 0.5
        assert cfg.ghost_width == 7.0
        assert cfg.mesh == "MESH_SENTINEL"
        assert cfg.mesh_dim == "grid"
        assert cfg.grid_dims == (2, 2, 1)

    def test_effective_ghost_width_defaults_to_cutoff_plus_skin(self):
        assert DomainConfig(cutoff=5.0).effective_ghost_width() == 5.0
        assert DomainConfig(cutoff=5.0, skin=1.5).effective_ghost_width() == 6.5

    def test_effective_ghost_width_explicit_overrides(self):
        cfg = DomainConfig(cutoff=5.0, skin=0.5, ghost_width=7.0)
        assert cfg.effective_ghost_width() == 7.0


class TestHookScope:
    def test_canonical_members(self):
        assert HookScope.LOCAL.value == "local"
        assert HookScope.GLOBAL.value == "global"
        assert HookScope.RANK_ZERO.value == "rank_zero"
