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

"""Framework-generic distributed spec primitives.

This module holds :class:`DistributionSpec` — the framework-generic spec
carrying a
:class:`~nvalchemi.distributed._core.storage_policy.StoragePolicy`, the
tuple of :class:`OpAdapter` ``custom_ops``, and
the tuple of :class:`JitAdapter` / :class:`PythonAdapter`
``third_party_helpers``.

The adapter classes themselves (with their lifecycle methods + JSON
serialization) live in :mod:`nvalchemi.distributed._core.adapter`.

Chemistry-named output classification (``owned_only_outputs``,
``all_reduce_outputs`` keyed by output name) is on
:class:`~nvalchemi.distributed.spec.MLIPSpec` one layer up.

Part of the upstream-candidate ``_core/`` surface; must not import
from ``nvalchemi.models`` / ``nvalchemi.data`` / ``nvalchemi.dynamics`` /
``nvalchemi.distributed._chemistry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nvalchemi.distributed._core.adapter import (
    JitAdapter,
    OpAdapter,
    PythonAdapter,
    _adapter_from_dict,
    _op_qualname,
    _resolve_op,
)
from nvalchemi.distributed._core.storage_policy import (
    StoragePolicy,
    policy_from_dict,
    policy_to_dict,
)

__all__ = [
    "DistributionSpec",
    # Re-exports for convenient single-import construction.
    "OpAdapter",
    "JitAdapter",
    "PythonAdapter",
    "_op_qualname",
    "_resolve_op",
]


# ----------------------------------------------------------------------
# DistributionSpec — the framework-generic spec.
# ----------------------------------------------------------------------

#: Default per-atom fields the eager-DD path promotes to ShardTensors when a spec
#: doesn't narrow :attr:`DistributionSpec.shard_fields` — the common MLIP inputs.
DEFAULT_SHARD_FIELDS: tuple[str, ...] = ("positions", "charges", "atomic_numbers")


@dataclass(frozen=True)
class DistributionSpec:
    """Framework-generic distributed spec.

    Carries the field's :data:`StoragePolicy` (how local storage relates to its
    placement + the overlay-aware op behavior) and the declarative tuples of
    third-party touchpoints. No chemistry vocabulary appears here — output names
    like ``"stress"`` / ``"forces"`` live in :class:`MLIPSpec` one layer up.

    Parameters
    ----------
    policy
        A :class:`~nvalchemi.distributed._core.storage_policy.StoragePolicy`
        (:class:`HaloStoragePolicy` / :class:`PlainShard`),
        or ``None`` for the local (no cross-rank) case. The dispatch attaches it
        to each tensor as ``_storage_policy`` and routes ops through it.
    adapters
        The single declarative field for registering adapters — one tuple mixing
        :class:`OpAdapter` (opaque custom/triton kernels) with
        :class:`JitAdapter` / :class:`PythonAdapter` / :class:`MethodAdapter`
        (third-party callable replacements). At construction it is *lowered*:
        each :class:`OpAdapter` is appended to :attr:`custom_ops`, everything
        else to :attr:`third_party_helpers`, and ``adapters`` is cleared. The
        two split tuples remain the canonical storage all framework consumers
        read, so serialization and dispatch are unchanged — ``adapters`` is
        purely a unifying constructor convenience.
    custom_ops
        Tuple of :class:`OpAdapter` declaring custom-op wrap config
        (kernels registered via ``@torch.library.custom_op`` /
        ``@torch.library.triton_op``). May be passed directly or supplied via
        :attr:`adapters`.
    third_party_helpers
        Tuple of :class:`JitAdapter` / :class:`PythonAdapter` /
        :class:`MethodAdapter` — third-party callables that need a
        distributed-aware replacement. May be passed directly or via
        :attr:`adapters`.
    """

    policy: StoragePolicy | None = None
    custom_ops: tuple[OpAdapter, ...] = field(default_factory=tuple)
    third_party_helpers: tuple[Any, ...] = field(default_factory=tuple)
    # ^ ``Any`` rather than ``JitAdapter | PythonAdapter`` so the
    # dataclass type can be referenced before either adapter class is
    # imported in some tooling paths. The runtime discrimination is by
    # ``isinstance`` at use-site.
    adapters: tuple[Any, ...] = field(default_factory=tuple)
    # Which per-atom batch fields the eager-DD path promotes to ShardTensors (so
    # the model's primary ops dispatch on a ShardTensor and the per-layer halo
    # correction rides ``__torch_dispatch__``). Defaults to the common MLIP
    # inputs (:data:`DEFAULT_SHARD_FIELDS`); a model narrows it to exactly what it
    # needs — e.g. AIMNet2 declares ``("positions",)`` (``atomic_numbers`` feeds
    # an embedding→``Linear`` that must stay plain, since a ShardTensor through a
    # ``Linear`` mixes Tensor/DTensor in backward), and UMA declares ``()``
    # (plain-interior — promote nothing). Always a concrete tuple (no ``None``
    # sentinel) so ``()`` "promote nothing" can never collapse to the default
    # under truthiness. ``compare=False`` keeps the dataclass hashable.
    shard_fields: tuple[str, ...] = field(default=DEFAULT_SHARD_FIELDS, compare=False)

    def __post_init__(self) -> None:
        """Lower the unified :attr:`adapters` tuple onto the two canonical
        split tuples, then clear it. Discrimination is by type: an
        :class:`OpAdapter` is a custom-op wrap (``custom_ops``); anything else
        is a third-party callable replacement (``third_party_helpers``).

        Adapters passed via ``adapters`` are appended *after* any passed
        directly to ``custom_ops`` / ``third_party_helpers`` (so an explicit
        split list composes with the unified one). Frozen dataclass → write
        through ``object.__setattr__``.
        """
        self._validate()
        if not self.adapters:
            return
        extra_ops = tuple(a for a in self.adapters if isinstance(a, OpAdapter))
        extra_helpers = tuple(a for a in self.adapters if not isinstance(a, OpAdapter))
        object.__setattr__(self, "custom_ops", self.custom_ops + extra_ops)
        object.__setattr__(
            self, "third_party_helpers", self.third_party_helpers + extra_helpers
        )
        object.__setattr__(self, "adapters", ())

    def _validate(self) -> None:
        """Surface structurally-broken declarations at construction time rather
        than as an opaque failure mid-forward. Conservative on purpose — it only
        rejects what is unambiguously wrong (bad types, an op handle that does
        not resolve), never a merely-incomplete-but-valid spec."""

        # Duck-type the StoragePolicy interface (``scatter`` + ``to_local`` are
        # its defining op-behavior). ``isinstance`` against the runtime_checkable
        # Protocol is unreliable here because the Protocol declares a
        # ``placement`` property, so check the methods structurally instead.
        def _is_policy(p: Any) -> bool:
            return callable(getattr(p, "scatter", None)) and callable(
                getattr(p, "to_local", None)
            )

        if self.policy is not None and not _is_policy(self.policy):
            raise TypeError(
                f"DistributionSpec.policy must be a StoragePolicy or None, "
                f"got {type(self.policy).__name__}"
            )
        for op in self.custom_ops:
            if not isinstance(op, OpAdapter):
                raise TypeError(
                    f"DistributionSpec.custom_ops entries must be OpAdapter, "
                    f"got {type(op).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation. Used by :meth:`MLIPSpec.to_dict`
        (v2 schema, nested under ``"core"``).
        """
        d: dict[str, Any] = {
            "policy": policy_to_dict(self.policy),
            "custom_ops": [op.to_dict() for op in self.custom_ops],
            "third_party_helpers": [h.to_dict() for h in self.third_party_helpers],
        }
        # Omit when it equals the default so default specs serialize unchanged
        # (back-compatible with files written before shard_fields existed).
        if self.shard_fields != DEFAULT_SHARD_FIELDS:
            d["shard_fields"] = list(self.shard_fields)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DistributionSpec":
        """Inverse of :meth:`to_dict`."""
        return cls(
            policy=policy_from_dict(d["policy"]),
            custom_ops=tuple(OpAdapter.from_dict(e) for e in d.get("custom_ops", [])),
            third_party_helpers=tuple(
                _adapter_from_dict(e) for e in d.get("third_party_helpers", [])
            ),
            shard_fields=(
                tuple(d["shard_fields"])
                if "shard_fields" in d
                else DEFAULT_SHARD_FIELDS
            ),
        )
