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

import pytest
import torch

from nvalchemi.csp.volume import (
    estimate_formula_unit_volume,
    get_default_atomic_volumes,
)


def test_default_atomic_volumes_are_fresh_and_volume_is_summed() -> None:
    first = get_default_atomic_volumes()
    second = get_default_atomic_volumes()
    assert first == second
    assert first[1] == 5.79
    assert first[6] == 13.4
    assert first[8] == 10.38
    first[1] = 1.0
    assert second[1] != 1.0
    assert estimate_formula_unit_volume(torch.tensor([6, 8])) == pytest.approx(23.78)


def test_custom_volume_table_replaces_defaults_and_reports_all_missing() -> None:
    assert estimate_formula_unit_volume(
        torch.tensor([6, 6]), atomic_volumes={6: 4.5}
    ) == pytest.approx(9.0)
    with pytest.raises(ValueError, match=r"\[8, 99\]"):
        estimate_formula_unit_volume(torch.tensor([8, 99]), atomic_volumes={6: 4.5})


@pytest.mark.parametrize(
    "table",
    [{0: 1.0}, {6: 0.0}, {6: float("inf")}, {6: float("nan")}],
)
def test_invalid_custom_volume_table_rejected(table: dict[int, float]) -> None:
    with pytest.raises(ValueError):
        estimate_formula_unit_volume(torch.tensor([6]), atomic_volumes=table)


@pytest.mark.parametrize(
    "atomic_numbers",
    [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([[6]], dtype=torch.int64),
        torch.tensor([6.0]),
    ],
)
def test_invalid_atomic_number_tensor_rejected(atomic_numbers: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        estimate_formula_unit_volume(atomic_numbers)


def test_missing_default_atomic_number_is_reported() -> None:
    with pytest.raises(ValueError, match="atomic number"):
        estimate_formula_unit_volume(torch.tensor([99]))
