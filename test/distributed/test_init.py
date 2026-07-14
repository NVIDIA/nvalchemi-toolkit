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

"""Tests for nvalchemi.distributed.__init__ lazy import mechanism."""

from __future__ import annotations

import pytest


class TestLazyImports:
    """Test that __getattr__ lazy-imports public symbols on first access."""

    def test_import_domain_config(self) -> None:
        from nvalchemi.distributed import DomainConfig
        from nvalchemi.distributed.config import DomainConfig as Direct

        assert DomainConfig is Direct

    def test_import_hook_scope(self) -> None:
        from nvalchemi.distributed import HookScope
        from nvalchemi.distributed.config import HookScope as Direct

        assert HookScope is Direct

    def test_import_spatial_partitioner(self) -> None:
        from nvalchemi.distributed import SpatialPartitioner
        from nvalchemi.distributed.partitioner import SpatialPartitioner as Direct

        assert SpatialPartitioner is Direct

    def test_import_domain_parallel(self) -> None:
        from nvalchemi.distributed import DomainParallel
        from nvalchemi.distributed.domain_parallel import DomainParallel as Direct

        assert DomainParallel is Direct

    def test_import_sharded_batch(self) -> None:
        from nvalchemi.distributed import ShardedBatch
        from nvalchemi.distributed.sharded_batch import ShardedBatch as Direct

        assert ShardedBatch is Direct

    def test_import_particle_halo_config(self) -> None:
        from nvalchemi.distributed import ParticleHaloConfig
        from nvalchemi.distributed._core.particle_halo import (
            ParticleHaloConfig as Direct,
        )

        assert ParticleHaloConfig is Direct

    def test_import_reshard(self) -> None:
        from nvalchemi.distributed import reshard_by_destination
        from nvalchemi.distributed._core.reshard import reshard_by_destination as Direct

        assert reshard_by_destination is Direct

    def test_nonexistent_attribute_raises_attribute_error(self) -> None:
        import nvalchemi.distributed as dist_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = dist_mod.ThisDoesNotExist

    def test_all_names_importable(self) -> None:
        """Every name in __all__ should be importable via __getattr__."""
        import nvalchemi.distributed as dist_mod

        for name in dist_mod.__all__:
            obj = getattr(dist_mod, name)
            assert obj is not None, f"Failed to import {name}"
