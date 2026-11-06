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
"""Periodic-aware differences between collective-variable values."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["periodic_difference"]


def periodic_difference(
    values: Tensor, centers: Tensor, periods: Tensor | None = None
) -> Tensor:
    """Return ``values - centers``, wrapped into ``(-period/2, period/2]``.

    A dihedral restrained at ``+3.0 rad`` must not see a configuration at
    ``-3.0 rad`` as ``6.0 rad`` away; the true separation is ``0.28 rad`` the
    other way round.  Taking a raw difference makes the restraint pull the
    long way round the circle, which is both wrong and violently
    discontinuous at the branch cut.

    Parameters
    ----------
    values:
        Current CV values, shape ``[B, D]``.
    centers:
        Reference values, shape ``[B, D]`` (or broadcastable to it).
    periods:
        Period per CV component, shape ``[D]``.  A component whose period is
        ``0`` (or non-finite) is treated as non-periodic and its difference
        is returned unwrapped.  ``None`` means every component is
        non-periodic.

    Returns
    -------
    Tensor
        Wrapped difference, shape ``[B, D]``.

    Notes
    -----
    Differentiability
        ``round`` has zero gradient almost everywhere, so the wrap
        contributes nothing to ``d(delta)/d(values)`` — the derivative is the
        same as for an unwrapped difference, which is what a harmonic
        restraint needs.  The wrap is discontinuous exactly at
        ``delta = period/2``, the antipode of the center; that is inherent to
        a periodic CV, not an artefact here.
    """
    delta = values - centers
    if periods is None:
        return delta

    periods = periods.to(device=delta.device, dtype=delta.dtype)
    # A zero or non-finite period marks a non-periodic component; guard the
    # division so those components produce no wrap rather than NaN/Inf.
    active = torch.isfinite(periods) & (periods != 0)
    safe = torch.where(active, periods, torch.ones_like(periods))
    wrapped = delta - safe * torch.round(delta / safe)
    return torch.where(active, wrapped, delta)
