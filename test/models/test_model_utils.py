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
"""Unit tests for the shared model-composition helpers.

Covers the two helpers any producer of additive outputs relies on:

* :func:`~nvalchemi.models._utils.validate_contribution` — detachment,
  shapes, stress/virial mutual exclusion, batch-size consistency, and
  finiteness of a :data:`~nvalchemi._typing.ModelOutputs` mapping.
* :func:`~nvalchemi.models._utils.isolated_energy_derivatives` — forces and
  tensile-positive Cauchy stress from an energy function, with the live
  ``Batch`` restored and no grad graph escaping.
* :func:`~nvalchemi.models._utils.aggregate_contributions` — the strict
  counterpart to ``sum_outputs``: summing, missing-entry handling, and the
  collisions it refuses to resolve silently.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import Tensor

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import (
    APPLIED_OUTPUT_KEYS,
    DIAGNOSTIC_PREFIX,
    STATE_VERSION_KEY,
    aggregate_contributions,
    isolated_energy_derivatives,
    validate_contribution,
)


def _outputs(**kwargs: Tensor | None) -> OrderedDict[str, Tensor | None]:
    """Build a ModelOutputs mapping from keyword arguments."""
    return OrderedDict(kwargs)


class TestValidateContribution:
    """Tests for the contribution checker."""

    def test_empty_mapping_accepted(self) -> None:
        validate_contribution(_outputs())

    @pytest.mark.parametrize("key", ["hessian", "dipole", "charges"])
    def test_an_output_nothing_can_apply_is_refused(self, key: str) -> None:
        """Dropping it silently is the failure this check exists to stop.

        ``aggregate_contributions`` keeps only the applied keys, so an
        unrecognised physical output would otherwise vanish between the
        producer and the buffer with the run carrying on as though it had been
        applied. ``hessian`` and ``dipole`` are the sharp cases: both are
        documented ``ModelOutputs`` keys for a full forward pass, and neither
        is something a consumer can add into a batch.
        """
        with pytest.raises(ValueError, match=f"\\['{key}'\\] cannot be applied"):
            validate_contribution(_outputs(**{key: torch.zeros(2, 3)}))

    def test_the_refusal_lists_every_offender_at_once(self) -> None:
        with pytest.raises(ValueError, match=r"\['dipole', 'hessian'\]"):
            validate_contribution(
                _outputs(hessian=torch.zeros(2, 2), dipole=torch.zeros(1, 3))
            )

    def test_the_refusal_names_both_ways_out(self) -> None:
        """Report it, or extend the framework — not "try a different key"."""
        with pytest.raises(ValueError) as excinfo:
            validate_contribution(_outputs(hessian=torch.zeros(2, 2)))
        message = str(excinfo.value)
        assert DIAGNOSTIC_PREFIX in message
        assert "add the output to the framework" in message

    @pytest.mark.parametrize("key", [*APPLIED_OUTPUT_KEYS, STATE_VERSION_KEY])
    def test_every_supported_key_is_accepted(self, key: str) -> None:
        """The check must not be stricter than the set it is enforcing."""
        value = {
            "energy": torch.zeros(1, 1),
            "forces": torch.zeros(3, 3),
            "stress": torch.zeros(1, 3, 3),
            "virial": torch.zeros(1, 3, 3),
            STATE_VERSION_KEY: torch.zeros(1, dtype=torch.int64),
        }[key]
        validate_contribution(_outputs(**{key: value}))

    def test_a_none_valued_unsupported_key_is_still_refused(self) -> None:
        """A key that is present is a claim, whether or not it carries data."""
        with pytest.raises(ValueError, match="cannot be applied"):
            validate_contribution(_outputs(hessian=None))

    def test_a_reported_namesake_is_fine(self) -> None:
        validate_contribution(
            _outputs(**{f"{DIAGNOSTIC_PREFIX}hessian": torch.zeros(9, 9)})
        )

    def test_detached_tensors_accepted(self) -> None:
        validate_contribution(
            _outputs(energy=torch.tensor([[1.0]]), forces=torch.zeros(3, 3))
        )

    def test_none_entries_skipped(self) -> None:
        validate_contribution(_outputs(energy=None, forces=torch.zeros(3, 3)))

    def test_requires_grad_energy_raises(self) -> None:
        bad = torch.tensor([[1.0]], requires_grad=True)
        with pytest.raises(ValueError, match="energy.*detached"):
            validate_contribution(_outputs(energy=bad))

    def test_requires_grad_forces_raises(self) -> None:
        bad = torch.zeros(3, 3, requires_grad=True)
        with pytest.raises(ValueError, match="forces.*detached"):
            validate_contribution(_outputs(forces=bad))

    def test_grad_fn_raises(self) -> None:
        x = torch.tensor([[1.0]], requires_grad=True)
        with pytest.raises(ValueError, match="energy.*detached"):
            validate_contribution(_outputs(energy=x * 2.0))

    def test_stress_and_virial_raises(self) -> None:
        with pytest.raises(ValueError, match="stress.*virial"):
            validate_contribution(
                _outputs(stress=torch.zeros(1, 3, 3), virial=torch.zeros(1, 3, 3))
            )

    def test_diagnostic_requires_grad_raises(self) -> None:
        bad = torch.zeros(3, requires_grad=True)
        with pytest.raises(ValueError, match="diagnostics/cv.*detached"):
            validate_contribution(_outputs(**{f"{DIAGNOSTIC_PREFIX}cv": bad}))

    def test_source_names_the_producer(self) -> None:
        with pytest.raises(ValueError, match="MyBias 'umbrella'"):
            validate_contribution(
                _outputs(energy=torch.zeros(2)), source="MyBias 'umbrella'"
            )

    # --- shape validation ---

    def test_energy_wrong_ndim_raises(self) -> None:
        """energy must be [B, 1]; a flat [B] tensor is rejected."""
        with pytest.raises(ValueError, match="energy.*\\[B, 1\\]"):
            validate_contribution(_outputs(energy=torch.zeros(2)))

    def test_energy_wrong_trailing_dim_raises(self) -> None:
        """energy last dim must be 1, not 3."""
        with pytest.raises(ValueError, match="energy.*\\[B, 1\\]"):
            validate_contribution(_outputs(energy=torch.zeros(2, 3)))

    def test_forces_wrong_ndim_raises(self) -> None:
        """forces must be [N, 3]; a 1-D tensor is rejected."""
        with pytest.raises(ValueError, match="forces.*\\[N, 3\\]"):
            validate_contribution(_outputs(forces=torch.zeros(9)))

    def test_forces_wrong_width_raises(self) -> None:
        """forces last dim must be 3, not 1."""
        with pytest.raises(ValueError, match="forces.*\\[N, 3\\]"):
            validate_contribution(_outputs(forces=torch.zeros(4, 1)))

    def test_stress_wrong_shape_raises(self) -> None:
        """stress must be [B, 3, 3]; a [B, 3] tensor is rejected."""
        with pytest.raises(ValueError, match="stress.*\\[B, 3, 3\\]"):
            validate_contribution(_outputs(stress=torch.zeros(2, 3)))

    def test_virial_wrong_shape_raises(self) -> None:
        """virial must be [B, 3, 3]."""
        with pytest.raises(ValueError, match="virial.*\\[B, 3, 3\\]"):
            validate_contribution(_outputs(virial=torch.zeros(2, 9)))

    def test_state_version_wrong_ndim_raises(self) -> None:
        """state_version must be 1-D."""
        with pytest.raises(ValueError, match="state_version.*\\[B\\]"):
            validate_contribution(
                _outputs(state_version=torch.zeros(2, 1, dtype=torch.int32))
            )

    def test_state_version_float_dtype_raises(self) -> None:
        """state_version must be an integer dtype."""
        with pytest.raises(ValueError, match="integer dtype"):
            validate_contribution(_outputs(state_version=torch.zeros(2)))

    def test_state_version_integer_accepted(self) -> None:
        validate_contribution(_outputs(state_version=torch.zeros(2, dtype=torch.int64)))

    def test_diagnostics_have_no_shape_contract(self) -> None:
        """A diagnostic of any rank is accepted; it is reported, not applied."""
        validate_contribution(
            _outputs(
                **{
                    f"{DIAGNOSTIC_PREFIX}cv": torch.zeros(2, 7, 5),
                    f"{DIAGNOSTIC_PREFIX}count": torch.zeros(3, dtype=torch.long),
                }
            )
        )

    # --- batch-size consistency ---

    def test_batch_size_mismatch_raises(self) -> None:
        """energy [2, 1] and virial [3, 3, 3] have inconsistent B."""
        with pytest.raises(ValueError, match="inconsistent"):
            validate_contribution(
                _outputs(energy=torch.zeros(2, 1), virial=torch.zeros(3, 3, 3))
            )

    def test_batch_size_consistent_accepted(self) -> None:
        """energy [2, 1] and virial [2, 3, 3] with matching B=2 are accepted."""
        validate_contribution(
            _outputs(energy=torch.zeros(2, 1), virial=torch.zeros(2, 3, 3))
        )

    # --- finiteness ---

    def test_energy_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="energy.*NaN or Inf"):
            validate_contribution(_outputs(energy=torch.tensor([[float("nan")]])))

    def test_energy_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="energy.*NaN or Inf"):
            validate_contribution(_outputs(energy=torch.tensor([[float("inf")]])))

    def test_forces_nan_raises(self) -> None:
        bad = torch.zeros(3, 3)
        bad[1, 2] = float("nan")
        with pytest.raises(ValueError, match="forces.*NaN or Inf"):
            validate_contribution(_outputs(forces=bad))

    def test_virial_inf_raises(self) -> None:
        bad = torch.zeros(1, 3, 3)
        bad[0, 0, 0] = float("-inf")
        with pytest.raises(ValueError, match="virial.*NaN or Inf"):
            validate_contribution(_outputs(virial=bad))

    def test_diagnostic_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="diagnostics/cv.*NaN or Inf"):
            validate_contribution(
                _outputs(**{f"{DIAGNOSTIC_PREFIX}cv": torch.tensor([float("nan")])})
            )

    def test_fully_populated_contribution_accepted(self) -> None:
        validate_contribution(
            _outputs(
                energy=torch.zeros(2, 1),
                forces=torch.zeros(6, 3),
                virial=torch.zeros(2, 3, 3),
                state_version=torch.zeros(2, dtype=torch.int64),
                **{f"{DIAGNOSTIC_PREFIX}bias/a/cv": torch.zeros(2)},
            )
        )


def _make_batch(*, periodic: bool, n_atoms: int = 3, box: float = 30.0) -> Batch:
    """Build a one-graph batch, periodic or not."""
    kwargs: dict = {
        "atomic_numbers": torch.ones(n_atoms, dtype=torch.long),
        "positions": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 1.0, 2.0], [2.0, 5.0, 1.0]][:n_atoms],
            dtype=torch.float64,
        ),
    }
    if periodic:
        kwargs["cell"] = torch.eye(3, dtype=torch.float64).unsqueeze(0) * box
        kwargs["pbc"] = torch.ones(1, 3, dtype=torch.bool)
    return Batch.from_data_list([AtomicData(**kwargs)])


def _quadratic(batch: Batch) -> Tensor:
    """E = 0.5 * ||r||^2, summed per graph."""
    return 0.5 * (batch.positions**2).sum().reshape(1, 1)


def _volume_only(batch: Batch) -> Tensor:
    """E = det(cell) — depends on the cell but not on positions."""
    return torch.linalg.det(batch.cell.reshape(-1, 3, 3)).reshape(-1, 1)


def _constant(batch: Batch) -> Tensor:
    """E = 3.0 — depends on neither positions nor cell, and carries no graph."""
    return torch.full((batch.num_graphs, 1), 3.0, dtype=batch.positions.dtype)


class TestIsolatedEnergyDerivatives:
    """Tests for the shared isolation + autograd helper."""

    def test_forces_match_analytic(self) -> None:
        batch = _make_batch(periodic=False)
        out = isolated_energy_derivatives(_quadratic, batch, want_stress=False)
        assert torch.allclose(out["forces"], -batch.positions)

    def test_energy_and_forces_are_detached(self) -> None:
        batch = _make_batch(periodic=False)
        out = isolated_energy_derivatives(_quadratic, batch, want_stress=False)
        for value in out.values():
            assert not value.requires_grad
            assert value.grad_fn is None

    def test_live_batch_is_restored(self) -> None:
        batch = _make_batch(periodic=True)
        positions, cell = batch.positions, batch.cell
        isolated_energy_derivatives(_quadratic, batch)
        assert batch.positions is positions
        assert batch.cell is cell
        assert not batch.positions.requires_grad
        assert batch.positions.grad_fn is None

    def test_live_batch_restored_after_exception(self) -> None:
        batch = _make_batch(periodic=True)
        positions, cell = batch.positions, batch.cell

        def boom(_: Batch) -> Tensor:
            raise RuntimeError("energy failed")

        with pytest.raises(RuntimeError, match="energy failed"):
            isolated_energy_derivatives(boom, batch)
        assert batch.positions is positions
        assert batch.cell is cell

    def test_stress_shape_and_symmetry(self) -> None:
        batch = _make_batch(periodic=True)
        out = isolated_energy_derivatives(_quadratic, batch)
        stress = out["stress"]
        assert stress.shape == (1, 3, 3)
        assert torch.allclose(stress, stress.mT, atol=1e-10)

    def test_no_stress_for_nonperiodic_batch(self) -> None:
        batch = _make_batch(periodic=False)
        assert "stress" not in isolated_energy_derivatives(_quadratic, batch)

    def test_want_stress_false_skips_stress(self) -> None:
        batch = _make_batch(periodic=True)
        assert "stress" not in isolated_energy_derivatives(
            _quadratic, batch, want_stress=False
        )

    def test_want_forces_false_still_gives_stress(self) -> None:
        batch = _make_batch(periodic=True)
        out = isolated_energy_derivatives(
            _volume_only, batch, want_forces=False, allow_unused=True
        )
        assert "forces" not in out
        assert out["stress"].shape == (1, 3, 3)

    def test_position_independent_energy_gives_zero_forces(self) -> None:
        batch = _make_batch(periodic=True)
        out = isolated_energy_derivatives(_volume_only, batch, allow_unused=True)
        assert torch.allclose(out["forces"], torch.zeros_like(out["forces"]))
        assert not torch.allclose(out["stress"], torch.zeros_like(out["stress"]))

    def test_position_independent_energy_raises_without_allow_unused(self) -> None:
        batch = _make_batch(periodic=True)
        with pytest.raises(RuntimeError):
            isolated_energy_derivatives(_volume_only, batch, allow_unused=False)

    def test_constant_energy_gives_zeros(self) -> None:
        """An energy with no graph at all yields zeros rather than raising."""
        batch = _make_batch(periodic=True)
        out = isolated_energy_derivatives(_constant, batch)
        assert torch.allclose(out["forces"], torch.zeros_like(out["forces"]))
        assert torch.allclose(out["stress"], torch.zeros_like(out["stress"]))

    def test_runs_inside_no_grad(self) -> None:
        """Dynamics runs under torch.no_grad(); the helper re-enables it."""
        batch = _make_batch(periodic=False)
        with torch.no_grad():
            out = isolated_energy_derivatives(_quadratic, batch, want_stress=False)
        assert torch.allclose(out["forces"], -batch.positions)


def _c(**kwargs) -> ModelOutputs:
    """Build a contribution mapping from keyword arguments."""
    return OrderedDict(kwargs)


def _diag(**kwargs) -> ModelOutputs:
    """Build a contribution carrying only namespaced diagnostics."""
    return OrderedDict(
        (f"{DIAGNOSTIC_PREFIX}{key}", value) for key, value in kwargs.items()
    )


class TestAggregateContributions:
    """Tests for contribution aggregation."""

    def test_empty_list_returns_empty_mapping(self) -> None:
        assert aggregate_contributions([]) == OrderedDict()

    def test_single_contribution_passthrough(self) -> None:
        e = torch.tensor([[1.0]])
        f = torch.zeros(3, 3)
        r = aggregate_contributions([_c(energy=e, forces=f)])
        assert torch.allclose(r["energy"], e)
        assert torch.allclose(r["forces"], f)

    def test_energy_summed(self) -> None:
        agg = aggregate_contributions(
            [_c(energy=torch.tensor([[1.0]])), _c(energy=torch.tensor([[2.0]]))]
        )
        assert torch.allclose(agg["energy"], torch.tensor([[3.0]]))

    def test_forces_summed(self) -> None:
        agg = aggregate_contributions(
            [_c(forces=torch.ones(4, 3)), _c(forces=torch.ones(4, 3) * 2.0)]
        )
        assert torch.allclose(agg["forces"], torch.ones(4, 3) * 3.0)

    def test_missing_fields_handled(self) -> None:
        agg = aggregate_contributions(
            [_c(energy=torch.tensor([[1.0]])), _c(forces=torch.zeros(2, 3))]
        )
        assert agg.get("energy") is not None
        assert agg.get("forces") is not None

    def test_none_entries_treated_as_zero(self) -> None:
        agg = aggregate_contributions(
            [_c(energy=torch.tensor([[1.0]]), forces=None), _c(energy=None)]
        )
        assert torch.allclose(agg["energy"], torch.tensor([[1.0]]))
        assert agg.get("forces") is None

    def test_virial_summed(self) -> None:
        agg = aggregate_contributions(
            [_c(virial=torch.ones(1, 3, 3)), _c(virial=torch.ones(1, 3, 3) * 2.0)]
        )
        assert torch.allclose(agg["virial"], torch.ones(1, 3, 3) * 3.0)

    def test_mixed_stress_and_virial_raises(self) -> None:
        """Mixing stress from one contribution and virial from another raises.

        The message must identify which indices contributed each field —
        converting between the two needs the cell volume, so the caller has
        to know which bias to fix.
        """
        with pytest.raises(ValueError, match="stress.*virial|virial.*stress"):
            aggregate_contributions(
                [_c(stress=torch.zeros(1, 3, 3)), _c(virial=torch.zeros(1, 3, 3))]
            )

    def test_mixed_stress_and_virial_error_identifies_indices(self) -> None:
        """Error message must identify which indices are responsible."""
        contributions = [
            _c(energy=torch.zeros(1, 1)),  # index 0 — no cell response
            _c(stress=torch.zeros(1, 3, 3)),  # index 1 — stress
            _c(energy=torch.zeros(1, 1)),  # index 2 — no cell response
            _c(virial=torch.zeros(1, 3, 3)),  # index 3 — virial
        ]
        with pytest.raises(ValueError, match=r"\[1\].*\[3\]|\[3\].*\[1\]"):
            aggregate_contributions(contributions)

    def test_all_stress_aggregates_correctly(self) -> None:
        """Multiple stress contributions are summed without raising."""
        agg = aggregate_contributions(
            [_c(stress=torch.ones(1, 3, 3)), _c(stress=torch.ones(1, 3, 3) * 2.0)]
        )
        assert agg.get("virial") is None
        assert torch.allclose(agg["stress"], torch.ones(1, 3, 3) * 3.0)

    def test_duplicate_diagnostic_key_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate diagnostic key"):
            aggregate_contributions(
                [
                    _diag(**{"bias/a/cv": torch.zeros(1)}),
                    _diag(**{"bias/a/cv": torch.ones(1)}),
                ]
            )

    def test_duplicate_diagnostic_error_identifies_indices(self) -> None:
        """The message must name both colliding contributions, not just the key.

        Diagnostics are merged rather than summed, so a collision silently
        dropping one bias's value is exactly what this guards against; the
        message has to say which two biases collided.
        """
        contributions = [
            _diag(**{"bias/a/cv": torch.zeros(1)}),  # index 0
            _c(energy=torch.zeros(1, 1)),
            _diag(**{"bias/b/cv": torch.zeros(1)}),
            _diag(**{"bias/a/cv": torch.ones(1)}),  # index 3
        ]
        with pytest.raises(ValueError) as excinfo:
            aggregate_contributions(contributions)
        message = str(excinfo.value)
        assert "contributions[0]" in message
        assert "contributions[3]" in message
        assert "bias/a/cv" in message

    def test_diagnostic_named_energy_is_not_summed_into_the_energy(self) -> None:
        """The ``diagnostics/`` prefix is what keeps the namespaces apart.

        A diagnostic called ``energy`` must not be added to the bias energy,
        which is the whole reason the prefix exists rather than a flat key.
        """
        agg = aggregate_contributions(
            [
                OrderedDict(
                    energy=torch.ones(1, 1),
                    **{f"{DIAGNOSTIC_PREFIX}energy": torch.tensor([100.0])},
                ),
                _c(energy=torch.ones(1, 1)),
            ]
        )
        assert torch.allclose(agg["energy"], torch.full((1, 1), 2.0))
        assert torch.allclose(agg[f"{DIAGNOSTIC_PREFIX}energy"], torch.tensor([100.0]))

    def test_distinct_diagnostic_keys_merged(self) -> None:
        agg = aggregate_contributions(
            [
                _diag(**{"bias/a/cv": torch.tensor([1.0])}),
                _diag(**{"bias/b/cv": torch.tensor([2.0])}),
            ]
        )
        assert f"{DIAGNOSTIC_PREFIX}bias/a/cv" in agg
        assert f"{DIAGNOSTIC_PREFIX}bias/b/cv" in agg

    def test_state_version_is_dropped(self) -> None:
        """An aggregate over several producers has no single state revision."""
        agg = aggregate_contributions(
            [
                _c(
                    energy=torch.ones(1, 1),
                    state_version=torch.tensor([3], dtype=torch.long),
                ),
                _c(
                    energy=torch.ones(1, 1),
                    state_version=torch.tensor([7], dtype=torch.long),
                ),
            ]
        )
        assert "state_version" not in agg
        assert torch.allclose(agg["energy"], torch.full((1, 1), 2.0))

    def test_different_registration_orders_same_result(self) -> None:
        """Aggregation must be order-independent (commutativity for sum)."""
        e1 = torch.tensor([[1.5]])
        e2 = torch.tensor([[0.5]])
        agg_ab = aggregate_contributions([_c(energy=e1), _c(energy=e2)])
        agg_ba = aggregate_contributions([_c(energy=e2), _c(energy=e1)])
        assert torch.allclose(agg_ab["energy"], agg_ba["energy"])


def _compile_kw(device: str) -> dict:
    """Compile kwargs for a fully-compilable path."""
    kw: dict = {"fullgraph": True}
    if device == "cuda":
        kw["backend"] = "inductor"
    return kw


class TestAggregateCompile:
    """``aggregate_contributions`` is on the compiled hot path for a biased
    run, so it must reach ``fullgraph=True`` for a fixed-size input list."""

    def test_aggregate_compiles_fullgraph(self, device: str) -> None:
        """aggregate_contributions compiles with fullgraph=True."""
        e1 = torch.ones(2, 1, device=device)
        e2 = torch.ones(2, 1, device=device) * 2.0
        f1 = torch.ones(8, 3, device=device)
        f2 = torch.ones(8, 3, device=device) * 0.5

        def _agg() -> ModelOutputs:
            return aggregate_contributions(
                [_c(energy=e1, forces=f1), _c(energy=e2, forces=f2)]
            )

        compiled = torch.compile(_agg, **_compile_kw(device))
        result = compiled()
        assert torch.allclose(result["energy"], torch.full((2, 1), 3.0, device=device))
