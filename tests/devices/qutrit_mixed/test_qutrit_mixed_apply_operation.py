# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for create_initial_state in devices/qutrit_mixed/apply_operation."""

import os
import pytest
import numpy as np
from scipy.stats import unitary_group
import pennylane as qml
from pennylane import math
from pennylane.operation import Channel
from pennylane.devices.qutrit_mixed.apply_operation import (
    apply_operation_einsum,
    apply_operation,
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]

subspaces = [(0, 1), (0, 2), (1, 2)]


def test_custom_operator_with_matrix(one_qutrit_state):
    """Test that apply_operation works with any operation that defines a matrix."""
    mat = np.array(
        [
            [-0.35546532 - 0.03636115j, -0.19051888 - 0.38049108j, 0.07943913 - 0.8276115j],
            [-0.2766807 - 0.71617593j, -0.1227771 + 0.61271557j, -0.0872488 - 0.11150285j],
            [-0.2312502 - 0.47894201j, -0.04564929 - 0.65295532j, -0.3629075 + 0.3962342j],
        ]
    )

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operation):
        num_wires = 1

        def matrix(self):
            return mat

    new_state = apply_operation(CustomOp(0), one_qutrit_state)
    assert qml.math.allclose(new_state, mat @ one_qutrit_state @ np.conj(mat).T)


class TestTwoQutritStateSpecialCases:
    """Test the special cases on a two qutrit state.  Also tests the special cases for einsum application method
    for additional testing of these generic matrix application methods."""

    # pylint: disable=too-few-public-methods
    # Currently not added as special cases, but will be in future
    # TODO add special cases as they are added
    pass


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize(
    "state,shape", [("two_qutrit_state", (9, 9)), ("two_qutrit_batched_state", (2, 9, 9))]
)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    class Debugger:  # pylint: disable=too-few-public-methods
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    @pytest.mark.usefixtures("two_qutrit_state")
    def test_no_debugger(
        self, ml_framework, state, shape, request
    ):  # pylint: disable=unused-argument
        """Test nothing happens when there is no debugger"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        new_state = apply_operation(qml.Snapshot(), initial_state, is_state_batched=len(shape) != 2)
        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded properly when there is no tag"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = self.Debugger()
        new_state = apply_operation(
            qml.Snapshot(), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == shape
        assert math.allclose(debugger.snapshots[0], math.reshape(initial_state, shape))

    def test_provided_tag(self, ml_framework, state, shape, request):
        """Test a snapshot is recorded property when provided a tag"""
        state = request.getfixturevalue(state)
        initial_state = math.asarray(state, like=ml_framework)

        debugger = self.Debugger()
        tag = "dense"
        new_state = apply_operation(
            qml.Snapshot(tag), initial_state, debugger=debugger, is_state_batched=len(shape) != 2
        )

        assert new_state.shape == initial_state.shape
        assert math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == shape
        assert math.allclose(debugger.snapshots[tag], math.reshape(initial_state, shape))


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations (not channels) are applied correctly."""

    broadcasted_ops = [
        qml.TRX(np.array([np.pi, np.pi / 2]), wires=2, subspace=(0, 1)),
        qml.TRY(np.array([np.pi, np.pi / 2]), wires=2, subspace=(0, 1)),
        qml.TRZ(np.array([np.pi, np.pi / 2]), wires=2, subspace=(1, 2)),
        qml.QutritUnitary(
            np.array([unitary_group.rvs(27), unitary_group.rvs(27)]),
            wires=[0, 1, 2],
        ),
    ]
    unbroadcasted_ops = [
        qml.THadamard(wires=2),
        qml.TClock(wires=2),
        qml.TShift(wires=2),
        qml.TAdd(wires=[1, 2]),
        qml.TRX(np.pi / 3, wires=2, subspace=(0, 2)),
        qml.TRY(2 * np.pi / 3, wires=2, subspace=(1, 2)),
        qml.TRZ(np.pi / 6, wires=2, subspace=(0, 1)),
        qml.QutritUnitary(unitary_group.rvs(27), wires=[0, 1, 2]),
    ]
    num_qutrits = 3
    num_batched = 2
    dims = (3**num_qutrits, 3**num_qutrits)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, ml_framework, three_qutrit_state):
        """Tests that batched operations are applied correctly to an unbatched state."""

        state = three_qutrit_state
        flattened_state = state.reshape(self.dims)

        res = apply_operation(op, qml.math.asarray(state, like=ml_framework))
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mats = [
            np.kron(np.eye(3**missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(self.num_batched)
        ]
        expected = []

        for i in range(self.num_batched):
            expanded_mat = expanded_mats[i]
            adjoint_mat = np.conj(expanded_mat).T
            expected.append(
                (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * (self.num_qutrits * 2))
            )

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, ml_framework, three_qutrit_batched_state):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = three_qutrit_batched_state

        res = apply_operation(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = self.num_qutrits - len(op.wires)
        mat = op.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat
        adjoint_mat = np.conj(expanded_mat).T
        expected = []

        for i in range(self.num_batched):
            flattened_state = state[i].reshape(self.dims)
            expected.append(
                (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * (self.num_qutrits * 2))
            )

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, ml_framework, three_qutrit_batched_state):
        """Tests that batched operations are applied correctly to a batched state."""
        state = three_qutrit_batched_state

        res = apply_operation(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = self.num_qutrits - len(op.wires)
        mat = op.matrix()
        expanded_mats = [
            np.kron(np.eye(3**missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(self.num_batched)
        ]
        expected = []

        for i in range(self.num_batched):
            expanded_mat = expanded_mats[i]
            adjoint_mat = np.conj(expanded_mat).T
            flattened_state = state[i].reshape(self.dims)
            expected.append(
                (expanded_mat @ flattened_state @ adjoint_mat).reshape([3] * (self.num_qutrits * 2))
            )
        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, ml_framework, one_qutrit_state):
        """Tests that the batch_size is set on an operator if it was missing before."""
        param = qml.math.asarray([0.1, 0.2], like=ml_framework)
        state = one_qutrit_state
        op = qml.TRX(param, 0)
        op._batch_size = None  # pylint:disable=protected-access
        state = apply_operation(op, state)
        assert state.shape == (self.num_batched, 3, 3)
        assert op.batch_size == self.num_batched


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestChannels:  # pylint: disable=too-few-public-methods
    """Tests that Channel operations are applied correctly."""

    class CustomChannel(Channel):
        num_params = 1
        num_wires = 1

        def __init__(self, p, wires, id=None):
            super().__init__(p, wires=wires, id=id)

        @staticmethod
        def compute_kraus_matrices(p):
            K0 = (np.sqrt(1 - p) * math.cast_like(np.eye(3), p)).astype(complex)
            K1 = (
                np.sqrt(p) * math.cast_like(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), p)
            ).astype(complex)
            return [K0, K1]

    def test_non_broadcasted_state(self, ml_framework, two_qutrit_state):
        """Tests that Channel operations are applied correctly to a state."""
        state = two_qutrit_state
        test_channel = self.CustomChannel(0.3, wires=1)
        res = apply_operation(test_channel, math.asarray(state, like=ml_framework))
        flattened_state = state.reshape(9, 9)

        mat = test_channel.kraus_matrices()

        expanded_mats = [np.kron(np.eye(3), mat[i]) for i in range(len(mat))]
        expected = np.zeros((9, 9)).astype(complex)
        for i in range(len(mat)):
            expanded_mat = expanded_mats[i]
            adjoint_mat = np.conj(expanded_mat).T
            expected += expanded_mat @ flattened_state @ adjoint_mat
        expected = expected.reshape([3] * 4)

        assert qml.math.get_interface(res) == ml_framework
        assert res.shape == expected.shape
        assert qml.math.allclose(res, expected)

    def test_broadcasted_state(self, ml_framework, two_qutrit_batched_state):
        """Tests that Channel operations are applied correctly to a batched state."""
        state = two_qutrit_batched_state
        num_batched = two_qutrit_batched_state.shape[0]

        test_channel = self.CustomChannel(0.3, wires=1)
        res = apply_operation(test_channel, math.asarray(state, like=ml_framework))

        mat = test_channel.kraus_matrices()
        expanded_mats = [np.kron(np.eye(3), mat[i]) for i in range(len(mat))]
        expected = [np.zeros((9, 9)).astype(complex) for _ in range(num_batched)]
        for i in range(num_batched):
            flattened_state = state[i].reshape(9, 9)
            for j in range(len(mat)):
                expanded_mat = expanded_mats[j]
                adjoint_mat = np.conj(expanded_mat).T
                expected[i] += expanded_mat @ flattened_state @ adjoint_mat
            expected[i] = expected[i].reshape([3] * 4)
        expected = np.array(expected)

        assert qml.math.get_interface(res) == ml_framework
        assert res.shape == expected.shape
        assert qml.math.allclose(res, expected)
