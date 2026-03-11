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
"""
Comprehensive tests for the dynamics base module.

Tests cover HookStageEnum, Hook protocol, and BaseDynamics class including
hook execution order, frequency gating, compute operations, and masked updates.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, Hook, HookStageEnum
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.models.demo import DemoModelWrapper

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_simple_batch(device: str = "cpu") -> Batch:
    """
    Create a simple test batch with two molecules.

    Returns a batch containing:
    - Molecule 1: CO2 (2 atoms)
    - Molecule 2: Water (3 atoms)

    Parameters
    ----------
    device : str
        Device to create tensors on.

    Returns
    -------
    Batch
        A batched data structure.
    """
    data1 = AtomicData(
        atomic_numbers=torch.tensor([6, 8], dtype=torch.long),
        positions=torch.randn(2, 3),
    )
    data2 = AtomicData(
        atomic_numbers=torch.tensor([1, 1, 8], dtype=torch.long),
        positions=torch.randn(3, 3),
    )
    batch = Batch.from_data_list([data1, data2], device=device)

    # Initialize forces and energies tensors for the compute() method
    # to write into via copy_()
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energies = torch.zeros(batch.num_graphs, 1)

    return batch


def create_single_molecule_batch(n_atoms: int = 5, device: str = "cpu") -> Batch:
    """
    Create a batch with a single molecule.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the molecule.
    device : str
        Device to create tensors on.

    Returns
    -------
    Batch
        A batched data structure with one molecule.
    """
    data = AtomicData(
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long),
        positions=torch.randn(n_atoms, 3),
    )
    batch = Batch.from_data_list([data], device=device)

    # Initialize forces and energies tensors
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energies = torch.zeros(batch.num_graphs, 1)

    return batch


# -----------------------------------------------------------------------------
# Concrete Hook for Testing
# -----------------------------------------------------------------------------


class RecordingHook:
    """
    A concrete hook implementation that records when it was called.

    This hook appends its name to a shared list each time it is invoked,
    allowing tests to verify execution order.

    Attributes
    ----------
    frequency : int
        Execute every N steps.
    stage : HookStageEnum
        Stage at which to fire.
    name : str
        Identifier for this hook.
    record_list : list
        Shared list to append name to when called.
    """

    def __init__(
        self,
        stage: HookStageEnum,
        record_list: list[str],
        name: str | None = None,
        frequency: int = 1,
    ) -> None:
        """
        Initialize the recording hook.

        Parameters
        ----------
        stage : HookStageEnum
            The stage at which this hook fires.
        record_list : list[str]
            List to append to when called.
        name : str | None
            Name identifier. Defaults to stage name.
        frequency : int
            How often to fire (every N steps).
        """
        self.stage = stage
        self.frequency = frequency
        self.name = name if name is not None else stage.name
        self.record_list = record_list

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Record that this hook was called."""
        self.record_list.append(self.name)


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestHookStageEnum:
    """Test suite for HookStageEnum enumeration."""

    def test_all_stages_exist(self) -> None:
        """Verify all 9 enum members exist."""
        expected_stages = [
            "BEFORE_STEP",
            "BEFORE_PRE_UPDATE",
            "AFTER_PRE_UPDATE",
            "BEFORE_COMPUTE",
            "AFTER_COMPUTE",
            "BEFORE_POST_UPDATE",
            "AFTER_POST_UPDATE",
            "AFTER_STEP",
            "ON_CONVERGE",
        ]

        actual_stages = [member.name for member in HookStageEnum]
        assert len(actual_stages) == 9
        assert set(actual_stages) == set(expected_stages)

    def test_enum_values_are_integers(self) -> None:
        """Verify enum values are integers (not strings)."""
        for member in HookStageEnum:
            assert isinstance(member.value, int)

    def test_enum_values_unique(self) -> None:
        """Verify all enum values are unique."""
        values = [member.value for member in HookStageEnum]
        assert len(values) == len(set(values))

    def test_enum_ordering(self) -> None:
        """Verify the logical ordering of stages."""
        assert HookStageEnum.BEFORE_STEP.value < HookStageEnum.BEFORE_PRE_UPDATE.value
        assert (
            HookStageEnum.BEFORE_PRE_UPDATE.value < HookStageEnum.AFTER_PRE_UPDATE.value
        )
        assert HookStageEnum.AFTER_PRE_UPDATE.value < HookStageEnum.BEFORE_COMPUTE.value
        assert HookStageEnum.BEFORE_COMPUTE.value < HookStageEnum.AFTER_COMPUTE.value
        assert (
            HookStageEnum.AFTER_COMPUTE.value < HookStageEnum.BEFORE_POST_UPDATE.value
        )
        assert (
            HookStageEnum.BEFORE_POST_UPDATE.value
            < HookStageEnum.AFTER_POST_UPDATE.value
        )
        assert HookStageEnum.AFTER_POST_UPDATE.value < HookStageEnum.AFTER_STEP.value


class TestHookProtocol:
    """Test suite for the Hook protocol."""

    def test_concrete_hook_satisfies_protocol(self) -> None:
        """Verify a concrete implementation satisfies the Hook protocol."""
        record_list: list[str] = []
        hook = RecordingHook(HookStageEnum.BEFORE_STEP, record_list)

        # Check that it satisfies the Protocol
        assert isinstance(hook, Hook)

        # Verify required attributes exist
        assert hasattr(hook, "frequency")
        assert hasattr(hook, "stage")
        assert callable(hook)

    def test_object_without_call_fails_protocol(self) -> None:
        """Verify an object missing __call__ fails the protocol check."""

        class MissingCallHook:
            frequency: int = 1
            stage: HookStageEnum = HookStageEnum.BEFORE_STEP

        incomplete_hook = MissingCallHook()
        assert not isinstance(incomplete_hook, Hook)

    def test_object_without_frequency_fails_protocol(self) -> None:
        """Verify an object missing frequency fails the protocol check."""

        class MissingFrequencyHook:
            stage: HookStageEnum = HookStageEnum.BEFORE_STEP

            def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
                pass

        incomplete_hook = MissingFrequencyHook()
        assert not isinstance(incomplete_hook, Hook)

    def test_object_without_stage_fails_protocol(self) -> None:
        """Verify an object missing stage fails the protocol check."""

        class MissingStageHook:
            frequency: int = 1

            def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
                pass

        incomplete_hook = MissingStageHook()
        assert not isinstance(incomplete_hook, Hook)


class TestBaseDynamics:
    """Test suite for BaseDynamics class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_no_convergence_hook_returns_none(self) -> None:
        """Verify returns None when convergence_hook is None (default)."""
        dynamics = BaseDynamics(self.model)
        batch = create_simple_batch()
        # Default convergence_hook is None, so _check_convergence returns None
        assert dynamics._check_convergence(batch) is None

    def test_all_converged(self) -> None:
        """Verify all samples detected when all below threshold."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([[0.01], [0.02]])
        result = dynamics._check_convergence(batch)
        assert result is not None
        assert len(result) == 2

    def test_some_converged(self) -> None:
        """Verify only converged samples are returned."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([[0.01], [0.10]])
        result = dynamics._check_convergence(batch)
        assert result is not None
        assert result.tolist() == [0]

    def test_none_converged(self) -> None:
        """Verify returns None when no samples are below threshold."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([[0.10], [0.20]])
        assert dynamics._check_convergence(batch) is None

    def test_convergence_threshold_boundary(self) -> None:
        """Verify exact threshold IS converged (uses <= comparison)."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        batch = create_simple_batch()
        # fmax exactly at threshold (0.05) should be converged with <= semantics
        batch["fmax"] = torch.tensor([[0.05], [0.04]])
        result = dynamics._check_convergence(batch)
        assert result is not None
        # Both samples should converge: 0.05 <= 0.05 and 0.04 <= 0.05
        assert result.tolist() == [0, 1]

    def test_custom_convergence_hook(self) -> None:
        """Verify custom convergence_hook threshold."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook=ConvergenceHook.from_fmax(0.10),
        )
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([[0.05], [0.15]])
        result = dynamics._check_convergence(batch)
        assert result is not None
        assert result.tolist() == [0]

    def test_default_convergence_hook(self) -> None:
        """Verify default convergence_hook is None."""
        dynamics = BaseDynamics(self.model)
        assert dynamics.convergence_hook is None

    def test_on_converge_hooks_fire(self) -> None:
        """Verify ON_CONVERGE hooks fire when convergence is detected."""
        record_list: list[str] = []
        hook = RecordingHook(
            HookStageEnum.ON_CONVERGE, record_list, name="converge_hook"
        )
        dynamics = BaseDynamics(
            self.model,
            hooks=[hook],
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([[0.01], [0.02]])
        dynamics.step(batch)
        assert "converge_hook" in record_list

    def test_on_converge_hooks_do_not_fire_when_no_convergence(self) -> None:
        """Verify ON_CONVERGE hooks do NOT fire when nothing converged."""
        record_list: list[str] = []
        hook = RecordingHook(
            HookStageEnum.ON_CONVERGE, record_list, name="converge_hook"
        )
        dynamics = BaseDynamics(
            self.model,
            hooks=[hook],
            convergence_hook=ConvergenceHook.from_fmax(0.05),
        )
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([[0.10], [0.20]])
        dynamics.step(batch)
        assert "converge_hook" not in record_list

    def test_convergence_hook_from_dict(self) -> None:
        """Verify BaseDynamics accepts a dict and converts to ConvergenceHook."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook={"criteria": [{"key": "fmax", "threshold": 0.03}]},
        )
        assert isinstance(dynamics.convergence_hook, ConvergenceHook)
        assert len(dynamics.convergence_hook.criteria) == 1
        assert dynamics.convergence_hook.criteria[0].key == "fmax"
        assert dynamics.convergence_hook.criteria[0].threshold == 0.03

    def test_convergence_hook_from_dict_multi_criteria(self) -> None:
        """Verify BaseDynamics accepts a dict with multiple criteria."""
        dynamics = BaseDynamics(
            self.model,
            convergence_hook={
                "criteria": [
                    {"key": "fmax", "threshold": 0.05},
                    {"key": "energies", "threshold": 0.001, "reduce_op": "max"},
                ],
            },
        )
        assert isinstance(dynamics.convergence_hook, ConvergenceHook)
        assert len(dynamics.convergence_hook.criteria) == 2

    def test_convergence_hook_object_passthrough(self) -> None:
        """Verify a ConvergenceHook instance is passed through unchanged."""
        hook = ConvergenceHook.from_fmax(0.05)
        dynamics = BaseDynamics(
            self.model,
            convergence_hook=hook,
        )
        assert dynamics.convergence_hook is hook


class TestConvergenceCriterion:
    """Test suite for the _ConvergenceCriterion internal model.

    Tests cover representation, graph-level key comparisons, node-level
    scatter-reduce behavior, reduce operations (norm, max, etc.), custom
    operations, and error handling for missing keys.
    """

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        from nvalchemi.dynamics.base import _ConvergenceCriterion

        self._ConvergenceCriterion = _ConvergenceCriterion

    def test_repr_basic(self) -> None:
        """Verify repr shows key and threshold when no custom_op or reduce_op."""
        criterion = self._ConvergenceCriterion(key="fmax", threshold=0.05)
        result = repr(criterion)
        assert "key='fmax'" in result
        assert "threshold=0.05" in result
        assert "reduce_op" not in result
        assert "custom_op" not in result

    def test_repr_with_reduce_op(self) -> None:
        """Verify repr shows reduce_op and reduce_dims when set."""
        criterion = self._ConvergenceCriterion(
            key="forces", threshold=0.05, reduce_op="norm", reduce_dims=-1
        )
        result = repr(criterion)
        assert "reduce_op='norm'" in result
        assert "reduce_dims=-1" in result

    def test_repr_with_custom_op(self) -> None:
        """Verify repr shows custom_op function name."""

        def my_custom_op(tensor: torch.Tensor) -> torch.Tensor:
            return tensor < 0.1

        criterion = self._ConvergenceCriterion(
            key="fmax", threshold=0.0, custom_op=my_custom_op
        )
        result = repr(criterion)
        assert "custom_op=my_custom_op" in result

    def test_graph_level_key_lower_than_threshold(self) -> None:
        """Verify batch with fmax tensor all below threshold returns all True."""
        batch = create_simple_batch()
        # fmax shape (B,) or (B, 1), all below 0.05
        batch["fmax"] = torch.tensor([0.01, 0.02])

        criterion = self._ConvergenceCriterion(key="fmax", threshold=0.05)
        result = criterion(batch)

        assert result.shape == (2,)
        assert result.all()

    def test_graph_level_key_above_threshold(self) -> None:
        """Verify fmax above threshold returns all False."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.10, 0.20])

        criterion = self._ConvergenceCriterion(key="fmax", threshold=0.05)
        result = criterion(batch)

        assert result.shape == (2,)
        assert not result.any()

    def test_graph_level_key_2d_squeezed(self) -> None:
        """Verify fmax as (B, 1) is correctly squeezed to (B,)."""
        batch = create_simple_batch()
        # Shape (B, 1) should be squeezed
        batch["fmax"] = torch.tensor([[0.01], [0.10]])

        criterion = self._ConvergenceCriterion(key="fmax", threshold=0.05)
        result = criterion(batch)

        assert result.shape == (2,)
        assert result[0].item()  # 0.01 < 0.05
        assert not result[1].item()  # 0.10 >= 0.05

    def test_graph_level_mixed(self) -> None:
        """Verify some above, some below returns correct mixed mask."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.10])

        criterion = self._ConvergenceCriterion(key="fmax", threshold=0.05)
        result = criterion(batch)

        assert result.shape == (2,)
        assert result[0].item()  # 0.01 < 0.05
        assert not result[1].item()  # 0.10 >= 0.05

    def test_missing_key_raises(self) -> None:
        """Verify key not on batch raises KeyError."""
        batch = create_simple_batch()

        criterion = self._ConvergenceCriterion(key="nonexistent_key", threshold=0.05)

        with pytest.raises(KeyError, match="Key for convergence check not found"):
            criterion(batch)

    def test_custom_op_called(self) -> None:
        """Verify custom_op receives the tensor and its result is returned."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.10])

        call_record = []

        def my_custom_op(tensor: torch.Tensor) -> torch.Tensor:
            call_record.append(tensor)
            # Return custom logic: only first is True
            return torch.tensor([True, False])

        criterion = self._ConvergenceCriterion(
            key="fmax", threshold=0.0, custom_op=my_custom_op
        )
        result = criterion(batch)

        # Verify custom_op was called
        assert len(call_record) == 1
        assert torch.allclose(call_record[0], torch.tensor([0.01, 0.10]))
        # Verify result comes from custom_op
        assert result[0].item()
        assert not result[1].item()

    def test_reduce_op_norm(self) -> None:
        """Verify forces (V, 3) with reduce_op='norm' reduces per-node."""
        batch = create_simple_batch()
        # batch has 5 nodes (2 + 3 atoms), 2 graphs
        # Create forces where norm per node varies
        # Mol1: 2 nodes, Mol2: 3 nodes
        forces = torch.zeros(5, 3)
        # Mol1 nodes: small norms
        forces[0] = torch.tensor([0.01, 0.0, 0.0])  # norm ~0.01
        forces[1] = torch.tensor([0.02, 0.0, 0.0])  # norm ~0.02
        # Mol2 nodes: one large norm
        forces[2] = torch.tensor([0.01, 0.0, 0.0])  # norm ~0.01
        forces[3] = torch.tensor([0.01, 0.0, 0.0])  # norm ~0.01
        forces[4] = torch.tensor([0.10, 0.0, 0.0])  # norm ~0.10

        # Set via atoms group directly: "forces" is a known node-level key
        batch.forces = forces

        criterion = self._ConvergenceCriterion(
            key="forces", threshold=0.05, reduce_op="norm", reduce_dims=-1
        )
        result = criterion(batch)

        # After norm: [0.01, 0.02, 0.01, 0.01, 0.10]
        # After scatter-reduce (max) to graph:
        # Graph0 (nodes 0,1): max=0.02 <= 0.05 -> True
        # Graph1 (nodes 2,3,4): max=0.10 > 0.05 -> False
        assert result.shape == (2,)
        assert result[0].item()  # 0.02 <= 0.05
        assert not result[1].item()  # 0.10 > 0.05

    def test_reduce_op_max(self) -> None:
        """Verify per-node tensor with reduce_op='max' scatter-reduces to graph max."""
        batch = create_simple_batch()
        # Create a per-node tensor (V, 2) and use max reduce
        values = torch.zeros(5, 2)
        values[0] = torch.tensor([0.01, 0.02])
        values[1] = torch.tensor([0.03, 0.01])
        values[2] = torch.tensor([0.01, 0.01])
        values[3] = torch.tensor([0.01, 0.01])
        values[4] = torch.tensor([0.10, 0.20])

        # Set directly in atoms group so the (5, 2) node-level tensor is stored correctly
        batch._atoms_group._data["test_values"] = values

        criterion = self._ConvergenceCriterion(
            key="test_values", threshold=0.05, reduce_op="max", reduce_dims=-1
        )
        result = criterion(batch)

        # After max(dim=-1): [0.02, 0.03, 0.01, 0.01, 0.20]
        # After scatter-reduce (max) to graph:
        # Graph0 (nodes 0,1): max=0.03 <= 0.05 -> True
        # Graph1 (nodes 2,3,4): max=0.20 > 0.05 -> False
        assert result.shape == (2,)
        assert result[0].item()  # 0.03 <= 0.05
        assert not result[1].item()  # 0.20 > 0.05

    def test_reduce_op_none_graph_level(self) -> None:
        """Verify reduce_op=None with graph-level key needs no reduction."""
        batch = create_simple_batch()
        batch["energy_change"] = torch.tensor([0.001, 0.002])

        criterion = self._ConvergenceCriterion(
            key="energy_change", threshold=0.01, reduce_op=None
        )
        result = criterion(batch)

        assert result.shape == (2,)
        assert result.all()  # Both 0.001 and 0.002 <= 0.01

    def test_node_level_scatter_reduce(self) -> None:
        """Verify forces (V, 3) with reduce_op='norm' correctly scatter-reduces."""
        batch = create_simple_batch()
        # batch has V=5 nodes, B=2 graphs
        # batch.batch should be [0, 0, 1, 1, 1] for mol1 (2 atoms), mol2 (3 atoms)

        # Create forces where per-node norms differ
        forces = torch.zeros(5, 3)
        # Graph 0 nodes (indices 0, 1)
        forces[0] = torch.tensor([0.01, 0.01, 0.01])  # norm ~0.017
        forces[1] = torch.tensor([0.02, 0.02, 0.02])  # norm ~0.035
        # Graph 1 nodes (indices 2, 3, 4)
        forces[2] = torch.tensor([0.01, 0.0, 0.0])  # norm ~0.01
        forces[3] = torch.tensor([0.01, 0.0, 0.0])  # norm ~0.01
        forces[4] = torch.tensor([0.08, 0.0, 0.0])  # norm ~0.08

        # Set via known node-level key so it routes to the atoms group
        batch.forces = forces

        criterion = self._ConvergenceCriterion(
            key="forces", threshold=0.05, reduce_op="norm", reduce_dims=-1
        )
        result = criterion(batch)

        # After norm: ~[0.017, 0.035, 0.01, 0.01, 0.08]
        # After scatter-reduce (max) per graph using batch.batch:
        # Graph0: max(0.017, 0.035) ~= 0.035 <= 0.05 -> True
        # Graph1: max(0.01, 0.01, 0.08) ~= 0.08 > 0.05 -> False
        assert result.shape == (2,)
        assert result[0].item()  # ~0.035 <= 0.05
        assert not result[1].item()  # ~0.08 > 0.05


class TestConvergenceHook:
    """Test suite for the ConvergenceHook class.

    Tests cover representation, the from_fmax convenience constructor,
    single and multi-criteria evaluation, dict input normalization,
    error handling for invalid types, AND semantics for composition,
    and status migration behavior.
    """

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        from nvalchemi.dynamics.base import ConvergenceHook, _ConvergenceCriterion

        self.ConvergenceHook = ConvergenceHook
        self._ConvergenceCriterion = _ConvergenceCriterion

    def test_repr(self) -> None:
        """Verify repr shows list of criteria."""
        cc = self.ConvergenceHook(
            criteria=[
                {"key": "fmax", "threshold": 0.05},
                {"key": "energy_change", "threshold": 1e-6},
            ]
        )
        result = repr(cc)
        assert "ConvergenceHook" in result
        assert "fmax" in result
        assert "energy_change" in result

    def test_from_fmax_convenience(self) -> None:
        """Verify from_fmax(0.05) creates single fmax criterion."""
        cc = self.ConvergenceHook.from_fmax(0.05)

        assert isinstance(cc.criteria, list)
        assert len(cc.criteria) == 1
        assert cc.criteria[0].key == "fmax"
        assert cc.criteria[0].threshold == 0.05

    def test_from_fmax_num_criteria(self) -> None:
        """Verify num_criteria returns 1 for from_fmax."""
        cc = self.ConvergenceHook.from_fmax(0.05)
        assert cc.num_criteria == 1

    def test_single_criterion_all_converged(self) -> None:
        """Verify single fmax criterion with all below returns all indices."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.02])

        cc = self.ConvergenceHook.from_fmax(0.05)
        result = cc.evaluate(batch)

        assert result is not None
        assert result.tolist() == [0, 1]

    def test_single_criterion_none_converged(self) -> None:
        """Verify all above threshold returns None."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.10, 0.20])

        cc = self.ConvergenceHook.from_fmax(0.05)
        result = cc.evaluate(batch)

        assert result is None

    def test_single_criterion_partial(self) -> None:
        """Verify some converged returns correct indices."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.10])

        cc = self.ConvergenceHook.from_fmax(0.05)
        result = cc.evaluate(batch)

        assert result is not None
        assert result.tolist() == [0]

    def test_multi_criteria_and_semantics(self) -> None:
        """Verify two criteria (fmax AND energy_change) require both to converge."""
        batch = create_simple_batch()
        # Sample 0: fmax OK, energy_change OK -> converged
        # Sample 1: fmax OK, energy_change NOT OK -> not converged
        batch["fmax"] = torch.tensor([0.01, 0.01])
        batch["energy_change"] = torch.tensor([1e-7, 1e-5])

        cc = self.ConvergenceHook(
            criteria=[
                {"key": "fmax", "threshold": 0.05},
                {"key": "energy_change", "threshold": 1e-6},
            ]
        )
        result = cc.evaluate(batch)

        # Only sample 0 meets BOTH criteria
        assert result is not None
        assert result.tolist() == [0]

    def test_dict_input_single(self) -> None:
        """Verify pass a single dict instead of _ConvergenceCriterion works."""
        cc = self.ConvergenceHook(criteria={"key": "fmax", "threshold": 0.05})

        assert isinstance(cc.criteria, list)
        assert len(cc.criteria) == 1
        assert isinstance(cc.criteria[0], self._ConvergenceCriterion)
        assert cc.criteria[0].key == "fmax"

    def test_dict_input_list(self) -> None:
        """Verify pass list of dicts is normalized."""
        cc = self.ConvergenceHook(
            criteria=[
                {"key": "fmax", "threshold": 0.05},
                {"key": "energy_change", "threshold": 1e-6},
            ]
        )

        assert isinstance(cc.criteria, list)
        assert len(cc.criteria) == 2
        assert all(isinstance(c, self._ConvergenceCriterion) for c in cc.criteria)

    def test_mixed_list_input(self) -> None:
        """Verify pass list of mixed dicts and _ConvergenceCriterion works."""
        crit = self._ConvergenceCriterion(key="energy_change", threshold=1e-6)
        cc = self.ConvergenceHook(
            criteria=[
                {"key": "fmax", "threshold": 0.05},
                crit,
            ]
        )

        assert isinstance(cc.criteria, list)
        assert len(cc.criteria) == 2
        assert cc.criteria[0].key == "fmax"
        assert cc.criteria[1] is crit

    def test_invalid_criteria_type_raises(self) -> None:
        """Verify pass an integer raises TypeError."""
        with pytest.raises(TypeError, match="criteria must be"):
            self.ConvergenceHook(criteria=42)  # type: ignore[arg-type]

    def test_invalid_list_item_raises(self) -> None:
        """Verify pass a list with an integer item raises TypeError."""
        with pytest.raises(TypeError, match="Each criterion must be"):
            self.ConvergenceHook(
                criteria=[{"key": "fmax", "threshold": 0.05}, 42]  # type: ignore[list-item]
            )

    def test_upfront_allocation_shape(self) -> None:
        """Verify with 3 criteria and 4-graph batch convergence works correctly."""
        # Create a batch with 4 graphs
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 8], dtype=torch.long),
                positions=torch.randn(2, 3),
            )
            for _ in range(4)
        ]
        batch = Batch.from_data_list(data_list)

        # Set up three criteria
        batch["fmax"] = torch.tensor([0.01, 0.10, 0.01, 0.01])
        batch["energy_change"] = torch.tensor([1e-7, 1e-7, 1e-5, 1e-7])
        batch["step_change"] = torch.tensor([0.001, 0.001, 0.001, 0.010])

        cc = self.ConvergenceHook(
            criteria=[
                {"key": "fmax", "threshold": 0.05},
                {"key": "energy_change", "threshold": 1e-6},
                {"key": "step_change", "threshold": 0.005},
            ]
        )
        result = cc.evaluate(batch)

        # Sample 0: fmax OK, energy_change OK, step_change OK -> converged
        # Sample 1: fmax NOT OK -> not converged
        # Sample 2: energy_change NOT OK -> not converged
        # Sample 3: step_change NOT OK -> not converged
        assert result is not None
        assert result.tolist() == [0]

    def test_custom_kernel_in_criteria(self) -> None:
        """Verify one criterion with custom_op composed with threshold criterion."""
        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.10])
        batch["custom_metric"] = torch.tensor([0.5, 0.5])

        def custom_kernel(values: torch.Tensor) -> torch.Tensor:
            # Custom logic: converge if value > 0.3
            return values > 0.3

        cc = self.ConvergenceHook(
            criteria=[
                {"key": "fmax", "threshold": 0.05},
                {"key": "custom_metric", "threshold": 0.0, "custom_op": custom_kernel},
            ]
        )
        result = cc.evaluate(batch)

        # Sample 0: fmax OK (0.01 <= 0.05), custom OK (0.5 > 0.3) -> converged
        # Sample 1: fmax NOT OK (0.10 > 0.05) -> not converged
        assert result is not None
        assert result.tolist() == [0]

    def test_default_criteria_is_fmax(self) -> None:
        """Verify ConvergenceHook() with no args defaults to fmax=0.05 criterion."""
        hook = self.ConvergenceHook()
        assert len(hook.criteria) == 1
        assert hook.criteria[0].key == "fmax"
        assert hook.criteria[0].threshold == 0.05

    def test_status_migration_on_call(self) -> None:
        """Verify status is migrated for converged samples matching source_status."""
        from unittest.mock import MagicMock

        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.10])
        # Sample 0 has status 0, sample 1 has status 1
        batch["status"] = torch.tensor([0, 1])

        hook = self.ConvergenceHook(
            criteria=[{"key": "fmax", "threshold": 0.05}],
            source_status=0,
            target_status=1,
        )

        # Create a mock dynamics object (unused by the hook)
        dynamics = MagicMock()

        # Call the hook
        hook(batch, dynamics)

        # Sample 0: converged (fmax 0.01 <= 0.05) AND status == source_status (0)
        #           -> status migrated to target_status (1)
        # Sample 1: not converged (fmax 0.10 > 0.05) -> status unchanged
        assert batch["status"][0].item() == 1
        assert batch["status"][1].item() == 1  # was already 1

    def test_no_status_migration_when_none(self) -> None:
        """Verify status is NOT modified when source/target status are None."""
        from unittest.mock import MagicMock

        batch = create_simple_batch()
        batch["fmax"] = torch.tensor([0.01, 0.10])
        batch["status"] = torch.tensor([0, 0])

        hook = self.ConvergenceHook(
            criteria=[{"key": "fmax", "threshold": 0.05}],
            source_status=None,
            target_status=None,
        )

        dynamics = MagicMock()
        hook(batch, dynamics)

        # Status should remain unchanged
        assert batch["status"][0].item() == 0
        assert batch["status"][1].item() == 0


class TestNStepsAttribute:
    """Test suite for the n_steps construction-time attribute."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def test_n_steps_default_is_none(self) -> None:
        """Verify n_steps defaults to None when not provided."""
        dynamics = BaseDynamics(self.model)
        assert dynamics.n_steps is None

    def test_n_steps_set_at_init(self) -> None:
        """Verify n_steps is stored when passed to __init__."""
        dynamics = BaseDynamics(self.model, n_steps=10)
        assert dynamics.n_steps == 10

    def test_run_uses_init_n_steps(self) -> None:
        """Verify run(batch) uses self.n_steps when no argument provided."""
        dynamics = BaseDynamics(self.model, n_steps=5)
        batch = create_simple_batch()
        dynamics.run(batch)
        assert dynamics.step_count == 5

    def test_run_argument_overrides_init(self) -> None:
        """Verify run(batch, n_steps=3) overrides self.n_steps."""
        dynamics = BaseDynamics(self.model, n_steps=10)
        batch = create_simple_batch()
        dynamics.run(batch, n_steps=3)
        assert dynamics.step_count == 3

    def test_run_raises_without_n_steps(self) -> None:
        """Verify run(batch) raises ValueError when both are None."""
        dynamics = BaseDynamics(self.model)
        batch = create_simple_batch()
        with pytest.raises(ValueError, match="No step count provided"):
            dynamics.run(batch)

    def test_run_positional_backward_compat(self) -> None:
        """Verify run(batch, 5) still works as a positional argument."""
        dynamics = BaseDynamics(self.model)
        batch = create_simple_batch()
        dynamics.run(batch, 5)
        assert dynamics.step_count == 5

    def test_n_steps_in_repr(self) -> None:
        """Verify n_steps appears in __repr__ output."""
        dynamics = BaseDynamics(self.model, n_steps=42)
        repr_str = repr(dynamics)
        assert "n_steps=42" in repr_str


class TestStepMasking:
    """Test suite for graduated sample masking in BaseDynamics.step().

    When samples have ``status >= exit_status``, they should be treated as
    no-ops for the integrator — their positions and velocities should be
    preserved through the step.
    """

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.model = DemoModelWrapper()

    def _make_multi_batch(
        self,
        n_graphs: int = 5,
        n_atoms_per_graph: int = 3,
    ) -> Batch:
        """Create a multi-graph batch with random positions and velocities.

        Parameters
        ----------
        n_graphs : int
            Number of graphs in the batch.
        n_atoms_per_graph : int
            Atoms per graph.

        Returns
        -------
        Batch
            Batch with pre-allocated ``forces``, ``energies``, and ``velocities``.
        """
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor(
                    [6] * n_atoms_per_graph,
                    dtype=torch.long,
                ),
                positions=torch.randn(n_atoms_per_graph, 3),
            )
            for _ in range(n_graphs)
        ]
        batch = Batch.from_data_list(data_list)
        batch.forces = torch.zeros(batch.num_nodes, 3)
        batch.energies = torch.zeros(batch.num_graphs, 1)
        # Initialize velocities to non-zero random values
        batch.velocities = torch.randn(batch.num_nodes, 3)
        return batch

    def test_step_masks_converged_samples(self) -> None:
        """Verify positions and velocities of converged samples are preserved.

        Converged samples (status >= exit_status) should have their state
        frozen through the integrator step.
        """
        dynamics = DemoDynamics(model=self.model, n_steps=1, dt=1.0, exit_status=1)

        batch = self._make_multi_batch(n_graphs=5, n_atoms_per_graph=3)
        # Set status: graphs 0, 1, 2 are active (status=0)
        #             graphs 3, 4 are graduated (status=1)
        batch["status"] = torch.tensor([[0], [0], [0], [1], [1]])

        # Clone positions and velocities of the converged samples (graphs 3 and 4)
        # Each graph has 3 atoms, so converged atoms are indices 9:15
        converged_positions_before = batch.positions[9:15].clone()
        converged_velocities_before = batch.velocities[9:15].clone()
        active_positions_before = batch.positions[:9].clone()

        # Run one step
        dynamics.step(batch)

        # Verify converged samples' positions and velocities are UNCHANGED
        assert torch.allclose(batch.positions[9:15], converged_positions_before), (
            "Converged samples' positions should be preserved"
        )
        assert torch.allclose(batch.velocities[9:15], converged_velocities_before), (
            "Converged samples' velocities should be preserved"
        )

        # Verify active samples' positions and velocities HAVE CHANGED
        # (DemoDynamics modifies positions/velocities for active samples)
        assert not torch.allclose(batch.positions[:9], active_positions_before), (
            "Active samples' positions should have changed"
        )
        # Note: velocities might also change due to the Velocity Verlet integrator

    def test_step_no_masking_when_all_active(self) -> None:
        """Verify all samples are updated when all have status < exit_status."""
        dynamics = DemoDynamics(model=self.model, n_steps=1, dt=1.0, exit_status=1)

        batch = self._make_multi_batch(n_graphs=3, n_atoms_per_graph=2)
        # All samples are active (status=0)
        batch["status"] = torch.tensor([[0], [0], [0]])

        positions_before = batch.positions.clone()

        dynamics.step(batch)

        # All positions should have changed
        assert not torch.allclose(batch.positions, positions_before), (
            "All samples should have been updated"
        )

    def test_step_no_masking_without_status_field(self) -> None:
        """Verify step runs normally without a status attribute."""
        dynamics = DemoDynamics(model=self.model, n_steps=1, dt=1.0, exit_status=1)

        batch = self._make_multi_batch(n_graphs=2, n_atoms_per_graph=2)
        # Don't set batch.status — it should be None or not exist

        positions_before = batch.positions.clone()

        # Should run without errors
        dynamics.step(batch)

        # Positions should have changed (normal step execution)
        assert not torch.allclose(batch.positions, positions_before), (
            "Samples should have been updated without status field"
        )

    def test_step_masking_without_velocities(self) -> None:
        """Verify masking works when batch has no velocities field.

        Some dynamics (e.g., optimization) may not use velocities.
        The masking logic should handle missing velocities gracefully.
        """
        # Use BaseDynamics directly since DemoDynamics always creates velocities
        dynamics = BaseDynamics(model=self.model, exit_status=1)

        batch = self._make_multi_batch(n_graphs=3, n_atoms_per_graph=2)
        batch["status"] = torch.tensor([[0], [1], [1]])  # 1 active, 2 graduated

        # Remove velocities attribute
        del batch["velocities"]

        # Clone positions
        converged_positions_before = batch.positions[2:6].clone()  # graphs 1 and 2

        # Should run without errors
        dynamics.step(batch)

        # Converged positions should be preserved
        assert torch.allclose(batch.positions[2:6], converged_positions_before), (
            "Converged samples' positions should be preserved even without velocities"
        )

    def test_step_with_squeezable_status(self) -> None:
        """Verify masking works with 2D status tensor that needs squeezing."""
        dynamics = DemoDynamics(model=self.model, n_steps=1, dt=1.0, exit_status=1)

        batch = self._make_multi_batch(n_graphs=3, n_atoms_per_graph=2)
        # 2D status tensor (B, 1) — should be squeezed internally
        batch["status"] = torch.tensor([[0], [0], [1]])

        converged_positions_before = batch.positions[4:6].clone()

        dynamics.step(batch)

        # Converged sample should be preserved
        assert torch.allclose(batch.positions[4:6], converged_positions_before), (
            "Converged sample should be preserved with 2D status"
        )

    def test_exit_status_default_is_one(self) -> None:
        """Verify default exit_status is 1."""
        dynamics = BaseDynamics(model=self.model)
        assert dynamics.exit_status == 1

    def test_exit_status_custom_value(self) -> None:
        """Verify custom exit_status can be set."""
        dynamics = BaseDynamics(model=self.model, exit_status=3)
        assert dynamics.exit_status == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
