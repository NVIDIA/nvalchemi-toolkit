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
"""Regression tests for the AOTAutograd plain->ShardTensor runtime-tangent shim.

Under ``torch.compile`` with a Dynamo graph break, a ShardTensor boundary
tensor's backward cotangent can be materialized by AOTAutograd as a *plain*
``torch.Tensor`` while the upstream subgraph traced a ShardTensor tangent.
PyTorch can coerce a runtime *subclass* tangent to its traced metadata (via
``__coerce_same_metadata_as_tangent__``) but has no hook for a runtime *plain*
tensor, so it raises ``...guessed its metadata incorrectly``.
:func:`~nvalchemi.distributed._core.shard_tensor._install_aot_plain_tangent_coercion`
monkeypatches ``AOTDispatchAutograd.process_runtime_tangent`` to rebuild the
ShardTensor from the plain tensor + the traced ``SubclassCreationMeta`` (lossless
for a ``Replicate`` boundary). These tests guard the shim install and the
lossless flatten/unflatten reconstruction it relies on.
"""

import torch

from nvalchemi.distributed._core.shard_tensor import (
    ShardTensor,
    _install_aot_plain_tangent_coercion,
)


def test_aot_plain_tangent_shim_installed_and_idempotent() -> None:
    """Importing ``shard_tensor`` installs the shim on
    ``AOTDispatchAutograd.process_runtime_tangent``; re-installing is a no-op."""
    from torch._functorch._aot_autograd.runtime_wrappers import AOTDispatchAutograd

    fn = AOTDispatchAutograd.process_runtime_tangent
    assert getattr(fn, "_mlip_plain_tangent_shim", False) is True

    _install_aot_plain_tangent_coercion()  # idempotent: must not re-wrap
    assert AOTDispatchAutograd.process_runtime_tangent is fn


def test_shardtensor_flatten_unflatten_roundtrip_is_lossless(_session_gloo_pg) -> None:
    """The shim rebuilds a ShardTensor from a plain inner + traced metadata via
    ``__tensor_unflatten__`` — the exact operation must round-trip losslessly."""
    local = torch.randn(5, 4, dtype=torch.float64)
    st = ShardTensor.wrap(local, mesh=_session_gloo_pg)

    inner_names, flatten_spec = st.__tensor_flatten__()
    # AOT hands back the plain inner(s); reconstruct exactly as the shim does.
    inner_dict = {name: getattr(st, name) for name in inner_names}
    rebuilt = type(st).__tensor_unflatten__(
        inner_dict, flatten_spec, st.shape, st.stride()
    )

    assert isinstance(rebuilt, ShardTensor)
    assert rebuilt.placements == st.placements
    assert rebuilt.shape == st.shape
    rebuilt_local = (
        rebuilt.to_local() if hasattr(rebuilt, "to_local") else rebuilt._local_tensor
    )
    torch.testing.assert_close(rebuilt_local, local)


def test_shim_reconstructs_shardtensor_from_plain_tangent(_session_gloo_pg) -> None:
    """End-to-end of the shim's hot path: a genuine ``SubclassCreationMeta`` for a
    ShardTensor + a PLAIN runtime tangent must come back as a ShardTensor whose
    local value equals the plain tensor (lossless for ``Replicate``). Without the
    shim this path raises ``...guessed its metadata incorrectly``."""
    import pytest
    from torch._functorch._aot_autograd.runtime_wrappers import AOTDispatchAutograd
    from torch._subclasses.fake_tensor import FakeTensorMode

    try:  # AOT internal API — skip (don't fail) if it drifts across torch versions
        from torch._functorch._aot_autograd.subclass_utils import create_subclass_meta

        # AOT records the SubclassCreationMeta over a FAKE subclass during
        # tracing (``SubclassCreationMeta.__post_init__`` asserts
        # ``is_fake(original_subclass)``), so build it under a FakeTensorMode.
        with FakeTensorMode():
            st = ShardTensor.wrap(
                torch.randn(6, 3, dtype=torch.float64), mesh=_session_gloo_pg
            )
            meta = create_subclass_meta([st], with_memory_format=True)[0]
        # AOT fills ``original_subclass_type`` during tracing;
        # ``create_subclass_meta`` leaves it ``None``. Set it to mirror the real
        # recorded metadata (this is the field the shim keys off).
        meta.original_subclass_type = ShardTensor
    except Exception as exc:  # pragma: no cover - version-drift guard
        pytest.skip(f"AOT subclass-meta construction API changed: {exc!r}")

    plain = torch.randn(6, 3, dtype=torch.float64)  # the plain runtime tangent
    out, _leaves = AOTDispatchAutograd.process_runtime_tangent(plain, meta)

    assert isinstance(out, ShardTensor)
    out_local = out.to_local() if hasattr(out, "to_local") else out._local_tensor
    torch.testing.assert_close(out_local, plain)
