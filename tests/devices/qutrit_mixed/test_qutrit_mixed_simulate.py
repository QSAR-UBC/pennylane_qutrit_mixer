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
"""Unit tests for simulate in devices/qutrit_mixed."""

import pytest
import numpy as np
import pennylane as qml
from pennylane import math

# TODO change when added to __init__
from pennylane.devices.qutrit_mixed.simulate import simulate, get_final_state, measure_final_state


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""

        # qs = qml.tape.QuantumScript(measurements=[qml.sample(wires=0)])
        # with pytest.raises(NotImplementedError):
        #     simulate(qs)
        pass


def test_custom_operation():
    """Test execution works with a manually defined operator if it has a matrix."""

    # pylint: disable=too-few-public-methods
    # class MyOperator(qml.operation.Operator):
    #     num_wires = 1
    #
    #     @staticmethod
    #     def compute_matrix():
    #         return qml.PauliX.compute_matrix()
    #
    # qs = qml.tape.QuantumScript([MyOperator(0)], [qml.expval(qml.PauliZ(0))])
    #
    # result = simulate(qs)
    # assert qml.math.allclose(result, -1.0)
    pass


class TestStatePrepBase:
    """Tests integration with various state prep methods."""

    # TODO: 1 test

    def test_basis_state(self):
        """Test that the BasisState operator prepares the desired state."""
        # qs = qml.tape.QuantumScript(
        #     ops=[qml.BasisState([0, 1], wires=(0, 1))], measurements=[qml.probs(wires=(0, 1, 2))]
        # )
        # probs = simulate(qs)
        # expected = np.zeros(8)
        # expected[2] = 1.0
        # assert qml.math.allclose(probs, expected)
        pass


class TestBasicCircuit:
    """Tests a basic circuit with one rx gate and two simple expectation values."""

    @staticmethod
    def expected_circ_expval_values(phi, subspace):
        """TODO helper"""
        if subspace == (0, 1):
            gellmann_2 = -np.sin(phi)
            gellmann_3 = np.cos(phi)
            gellmann_5 = 0
            gellmann_8 = np.sqrt(1 / 3)
        if subspace == (0, 2):
            gellmann_2 = 0
            gellmann_3 = np.cos(phi / 2) ** 2
            gellmann_5 = -np.sin(phi)
            gellmann_8 = np.sqrt(1 / 3) * (np.cos(phi) - np.sin(phi / 2) ** 2)
        return np.array([gellmann_2, gellmann_3, gellmann_5, gellmann_8])

    @staticmethod
    def expected_circ_expval_jacobians(phi, subspace):
        """TODO helper"""
        if subspace == (0, 1):
            gellmann_2 = -np.cos(phi)
            gellmann_3 = -np.sin(phi)
            gellmann_5 = 0
            gellmann_8 = 0
        if subspace == (0, 2):
            gellmann_2 = 0
            gellmann_3 = -np.sin(phi) / 2
            gellmann_5 = -np.cos(phi)
            gellmann_8 = np.sqrt(1 / 3) * -(1.5 * np.sin(phi))
        return np.array([gellmann_2, gellmann_3, gellmann_5, gellmann_8])

    @staticmethod
    def get_basic_quantum_script(phi, subspace):
        ops = [qml.TRX(phi, wires=0, subspace=subspace)]
        obs = [
            qml.expval(qml.GellMann(0, 2)),
            qml.expval(qml.GellMann(0, 3)),
            qml.expval(qml.GellMann(0, 5)),
            qml.expval(qml.GellMann(0, 8)),
        ]
        return qml.tape.QuantumScript(ops, obs)

    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_basic_circuit_numpy(self, subspace):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = self.get_basic_quantum_script(phi, subspace)
        result = simulate(qs)
        print(result)

        expected_measurements = self.expected_circ_expval_values(phi, subspace)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert np.allclose(result, expected_measurements)

        state, is_state_batched = get_final_state(qs)
        result = measure_final_state(qs, state, is_state_batched)

        # find expected state
        expected_vector = np.array([0, 0, 0], dtype=complex)
        expected_vector[subspace[0]] = np.cos(phi / 2)
        expected_vector[subspace[1]] = -1j * np.sin(phi / 2)
        expected_state = np.outer(expected_vector, np.conj(expected_vector))

        assert np.allclose(state, expected_state)
        assert not is_state_batched

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert np.allclose(result, expected_measurements)

    @pytest.mark.autograd
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_autograd_results_and_backprop(self, subspace):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = self.get_basic_quantum_script(x, subspace)
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = self.expected_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = self.expected_circ_expval_jacobians(phi, subspace)
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_jax_results_and_backprop(self, use_jit, subspace):
        """Tests exeuction and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = self.get_basic_quantum_script(x, subspace)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        expected = self.expected_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        g = jax.jacobian(f)(phi)
        expected = self.expected_circ_expval_jacobians(phi, subspace)
        assert qml.math.allclose(g, expected)

    @pytest.mark.torch
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_torch_results_and_backprop(self, subspace):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = self.get_basic_quantum_script(x, subspace)
            return simulate(qs)

        result = f(phi)
        expected = self.expected_circ_expval_values(phi.detach().numpy(), subspace)
        assert qml.math.allclose(result[0], expected[0])
        assert qml.math.allclose(result[1], expected[1])
        assert qml.math.allclose(result[2], expected[2])
        assert qml.math.allclose(result[3], expected[3])

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        expected = self.expected_circ_expval_jacobians(phi.detach().numpy(), subspace)
        assert qml.math.allclose(g[0], expected[0])
        assert qml.math.allclose(g[1], expected[1])
        assert qml.math.allclose(g[2], expected[2])
        assert qml.math.allclose(g[3], expected[3])

    # TODO check if necessary pylint
    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    @pytest.mark.parametrize("subspace", [(0, 1), (0, 2)])
    def test_tf_results_and_backprop(self, subspace):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = self.get_basic_quantum_script(phi, subspace)
            result = simulate(qs)

        expected = self.expected_circ_expval_values(phi, subspace)
        assert qml.math.allclose(result, expected)

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])
        grad2 = grad_tape.jacobian(result[2], [phi])
        grad3 = grad_tape.jacobian(result[3], [phi])

        expected = self.expected_circ_expval_jacobians(phi, subspace)
        assert qml.math.allclose(grad0[0], expected[0])
        assert qml.math.allclose(grad1[0], expected[1])
        assert qml.math.allclose(grad2[0], expected[2])
        assert qml.math.allclose(grad3[0], expected[3])

    @pytest.mark.jax
    @pytest.mark.parametrize("op", [qml.TRX(np.pi, 0), qml.QutritBasisState([1], 0)])
    def test_result_has_correct_interface(self, op):
        """Test that even if no interface parameters are given, result is correct."""
        qs = qml.tape.QuantumScript([op], [qml.expval(qml.GellMann(0, 3))])
        res = simulate(qs, interface="jax")
        assert qml.math.get_interface(res) == "jax"
        assert qml.math.allclose(res, -1)


class TestBroadcasting:
    """Test that simulate works with broadcasted parameters"""

    # TODO:
    #  - 1 broad-casted state, from prep, check measurements
    #  - 1 broadcasted state, from operation, check measurements
    #  - 1 broad-casted state, from prep, check sample
    #  - 1 broadcasted state, from operation, check sample
    #  - 1 broadcasting with extra measurement wires
    # TODO total = 5 funcs

    def test_broadcasted_prep_state(self):
        """Test that simulate works for state measurements
        when the state prep has broadcasted parameters"""
        # x = np.array(1.2)
        #
        # ops = [qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        # measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        # prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]
        #
        # qs = qml.tape.QuantumScript(prep + ops, measurements)
        # res = simulate(qs)
        #
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]))
        # assert np.allclose(res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]))
        #
        # state, is_state_batched = get_final_state(qs)
        # res = measure_final_state(qs, state, is_state_batched)
        # expected_state = np.array(
        #     [
        #         [np.cos(x / 2), 0, 0, np.sin(x / 2)],
        #         [0, np.cos(x / 2), np.sin(x / 2), 0],
        #         [-np.sin(x / 2), 0, 0, np.cos(x / 2)],
        #         [0, -np.sin(x / 2), np.cos(x / 2), 0],
        #     ]
        # ).reshape((4, 2, 2))
        #
        # assert np.allclose(state, expected_state)
        # assert is_state_batched
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]))
        # assert np.allclose(res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]))
        pass

    def test_broadcasted_op_state(self):
        """Test that simulate works for state measurements
        when an operation has broadcasted parameters"""
        # x = np.array([0.8, 1.0, 1.2, 1.4])
        #
        # ops = [qml.PauliX(wires=1), qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        # measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        #
        # qs = qml.tape.QuantumScript(ops, measurements)
        # res = simulate(qs)
        #
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(res[0], np.cos(x))
        # assert np.allclose(res[1], -np.cos(x))
        #
        # state, is_state_batched = get_final_state(qs)
        # res = measure_final_state(qs, state, is_state_batched)
        #
        # expected_state = np.zeros((4, 2, 2))
        # expected_state[:, 0, 1] = np.cos(x / 2)
        # expected_state[:, 1, 0] = np.sin(x / 2)
        #
        # assert np.allclose(state, expected_state)
        # assert is_state_batched
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(res[0], np.cos(x))
        # assert np.allclose(res[1], -np.cos(x))
        pass

    def test_broadcasted_prep_sample(self):
        """Test that simulate works for sample measurements
        when the state prep has broadcasted parameters"""
        # x = np.array(1.2)
        #
        # ops = [qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        # measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        # prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]
        #
        # qs = qml.tape.QuantumScript(prep + ops, measurements, shots=qml.measurements.Shots(10000))
        # res = simulate(qs, rng=123)
        #
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(
        #     res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]), atol=0.05
        # )
        # assert np.allclose(
        #     res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]), atol=0.05
        # )
        #
        # state, is_state_batched = get_final_state(qs)
        # res = measure_final_state(qs, state, is_state_batched, rng=123)
        # expected_state = np.array(
        #     [
        #         [np.cos(x / 2), 0, 0, np.sin(x / 2)],
        #         [0, np.cos(x / 2), np.sin(x / 2), 0],
        #         [-np.sin(x / 2), 0, 0, np.cos(x / 2)],
        #         [0, -np.sin(x / 2), np.cos(x / 2), 0],
        #     ]
        # ).reshape((4, 2, 2))
        #
        # assert np.allclose(state, expected_state)
        # assert is_state_batched
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(
        #     res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]), atol=0.05
        # )
        # assert np.allclose(
        #     res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]), atol=0.05
        # )
        pass

    def test_broadcasted_op_sample(self):
        """Test that simulate works for sample measurements
        when an operation has broadcasted parameters"""
        # x = np.array([0.8, 1.0, 1.2, 1.4])
        #
        # ops = [qml.PauliX(wires=1), qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        # measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        #
        # qs = qml.tape.QuantumScript(ops, measurements, shots=qml.measurements.Shots(10000))
        # res = simulate(qs, rng=123)
        #
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(res[0], np.cos(x), atol=0.05)
        # assert np.allclose(res[1], -np.cos(x), atol=0.05)
        #
        # state, is_state_batched = get_final_state(qs)
        # res = measure_final_state(qs, state, is_state_batched, rng=123)
        #
        # expected_state = np.zeros((4, 2, 2))
        # expected_state[:, 0, 1] = np.cos(x / 2)
        # expected_state[:, 1, 0] = np.sin(x / 2)
        #
        # assert np.allclose(state, expected_state)
        # assert is_state_batched
        # assert isinstance(res, tuple)
        # assert len(res) == 2
        # assert np.allclose(res[0], np.cos(x), atol=0.05)
        # assert np.allclose(res[1], -np.cos(x), atol=0.05)
        pass

    def test_broadcasting_with_extra_measurement_wires(self, mocker):
        """Test that broadcasting works when the operations don't act on all wires."""
        # I can't mock anything in `simulate` because the module name is the function name
        # spy = mocker.spy(qml, "map_wires")
        # x = np.array([0.8, 1.0, 1.2, 1.4])
        #
        # ops = [qml.PauliX(wires=2), qml.RY(x, wires=1), qml.CNOT(wires=[1, 2])]
        # measurements = [qml.expval(qml.PauliZ(i)) for i in range(3)]
        #
        # qs = qml.tape.QuantumScript(ops, measurements)
        # res = simulate(qs)
        #
        # assert isinstance(res, tuple)
        # assert len(res) == 3
        # assert np.allclose(res[0], 1.0)
        # assert np.allclose(res[1], np.cos(x))
        # assert np.allclose(res[2], -np.cos(x))
        # assert spy.call_args_list[0].args == (qs, {2: 0, 1: 1, 0: 2})
        pass


class TestPostselection:  # TODO, necessesary?
    """Tests for applying projectors as operations."""

    # TODO:
    #  -
    #  -
    #  -
    # TODO total = ? funcs, ? repeats

    pass


class TestDebugger:
    """Tests that the debugger works for a simple circuit"""

    # TODO:
    #  - test debugger, all-interfaces + numpy
    # TODO total = 1 func, 5 repeats

    class Debugger:
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

    def test_debugger_numpy(self):
        """Test debugger with numpy"""
        # phi = np.array(0.397)
        # ops = [qml.Snapshot(), qml.RX(phi, wires=0), qml.Snapshot("final_state")]
        # qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        #
        # debugger = self.Debugger()
        # result = simulate(qs, debugger=debugger)
        #
        # assert isinstance(result, tuple)
        # assert len(result) == 2
        #
        # assert np.allclose(result[0], -np.sin(phi))
        # assert np.allclose(result[1], np.cos(phi))
        #
        # assert list(debugger.snapshots.keys()) == [0, "final_state"]
        # assert np.allclose(debugger.snapshots[0], [1, 0])
        # assert np.allclose(
        #     debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        # )
        pass

    @pytest.mark.autograd
    def test_debugger_autograd(self):
        """Tests debugger with autograd"""
        # phi = qml.numpy.array(-0.52)
        # debugger = self.Debugger()
        #
        # def f(x):
        #     ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
        #     qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        #     return qml.numpy.array(simulate(qs, debugger=debugger))
        #
        # result = f(phi)
        # expected = np.array([-np.sin(phi), np.cos(phi)])
        # assert qml.math.allclose(result, expected)
        #
        # assert list(debugger.snapshots.keys()) == [0, "final_state"]
        # assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        # assert qml.math.allclose(
        #     debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        # )
        pass

    @pytest.mark.jax
    def test_debugger_jax(self):
        """Tests debugger with JAX"""
        # import jax
        #
        # phi = jax.numpy.array(0.678)
        # debugger = self.Debugger()
        #
        # def f(x):
        #     ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
        #     qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        #     return simulate(qs, debugger=debugger)
        #
        # result = f(phi)
        # assert qml.math.allclose(result[0], -np.sin(phi))
        # assert qml.math.allclose(result[1], np.cos(phi))
        #
        # assert list(debugger.snapshots.keys()) == [0, "final_state"]
        # assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        # assert qml.math.allclose(
        #     debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        # )
        pass

    @pytest.mark.torch
    def test_debugger_torch(self):
        """Tests debugger with torch"""

        # import torch
        #
        # phi = torch.tensor(-0.526, requires_grad=True)
        # debugger = self.Debugger()
        #
        # def f(x):
        #     ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
        #     qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        #     return simulate(qs, debugger=debugger)
        #
        # result = f(phi)
        # assert qml.math.allclose(result[0], -torch.sin(phi))
        # assert qml.math.allclose(result[1], torch.cos(phi))
        #
        # assert list(debugger.snapshots.keys()) == [0, "final_state"]
        # assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        # print(debugger.snapshots["final_state"])
        # assert qml.math.allclose(
        #     debugger.snapshots["final_state"],
        #     torch.tensor([torch.cos(phi / 2), -torch.sin(phi / 2) * 1j]),
        # )
        pass

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_debugger_tf(self):
        """Tests debugger with tensorflow."""
        # import tensorflow as tf
        #
        # phi = tf.Variable(4.873)
        # debugger = self.Debugger()
        #
        # ops = [qml.Snapshot(), qml.RX(phi, wires=0), qml.Snapshot("final_state")]
        # qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        # result = simulate(qs, debugger=debugger)
        #
        # assert qml.math.allclose(result[0], -tf.sin(phi))
        # assert qml.math.allclose(result[1], tf.cos(phi))
        #
        # assert list(debugger.snapshots.keys()) == [0, "final_state"]
        # assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        # assert qml.math.allclose(
        #     debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        # )
        pass


class TestSampleMeasurements:
    """Tests circuits with sample-based measurements"""

    # TODO:
    #  - 1 broadcasted state
    #  -
    #  -
    #  -
    # TODO total = 7 funcs
    def test_invalid_samples(self):
        """TODO: expval, probs, var"""
        pass

    def test_single_sample(self):
        """Test a simple circuit with a single sample measurement"""
        # x = np.array(0.732)
        # qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)
        # result = simulate(qs)
        #
        # assert isinstance(result, np.ndarray)
        # assert result.shape == (10000, 2)
        # assert np.allclose(
        #     np.sum(result, axis=0).astype(np.float32) / 10000, [np.sin(x / 2) ** 2, 0], atol=0.1
        # )
        pass

    def test_multi_measurements(self):
        """Test a simple circuit containing multiple measurements"""
        # x, y = np.array(0.732), np.array(0.488)
        # qs = qml.tape.QuantumScript(
        #     [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
        #     [qml.expval(qml.PauliZ(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
        #     shots=10000,
        # )
        # result = simulate(qs)
        #
        # assert isinstance(result, tuple)
        # assert len(result) == 3
        # assert isinstance(result[0], np.float64)
        # assert isinstance(result[1], np.ndarray)
        # assert isinstance(result[2], np.ndarray)
        #
        # assert np.allclose(result[0], np.cos(x), atol=0.1)
        #
        # assert result[1].shape == (4,)
        # assert np.allclose(
        #     result[1],
        #     [
        #         np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
        #         np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
        #         np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
        #         np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
        #     ],
        #     atol=0.1,
        # )
        #
        # assert result[2].shape == (10000, 2)

        # TODO remove invalid things
        pass

    shots_data = [
        [10000, 10000],
        [(10000, 2)],
        [10000, 20000],
        [(10000, 2), 20000],
        [(10000, 3), 20000, (30000, 2)],
    ]

    @pytest.mark.parametrize("shots", shots_data)
    def test_sample_shot_vector(self, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        # x = np.array(0.732)
        # shots = qml.measurements.Shots(shots)
        # qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)
        # result = simulate(qs)
        #
        # assert isinstance(result, tuple)
        # assert len(result) == len(list(shots))
        #
        # assert all(isinstance(res, np.ndarray) for res in result)
        # assert all(res.shape == (s, 2) for res, s in zip(result, shots))
        # assert all(
        #     np.allclose(
        #         np.sum(res, axis=0).astype(np.float32) / s, [np.sin(x / 2) ** 2, 0], atol=0.1
        #     )
        #     for res, s in zip(result, shots)
        # )
        pass

    @pytest.mark.parametrize("shots", shots_data)
    def test_counts_shot_vector(self, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        # x = np.array(0.732)
        # shots = qml.measurements.Shots(shots)
        # qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)
        # result = simulate(qs)
        #
        # assert isinstance(result, tuple)
        # assert len(result) == len(list(shots))
        #
        # assert all(isinstance(res, np.ndarray) for res in result)
        # assert all(res.shape == (s, 2) for res, s in zip(result, shots))
        # assert all(
        #     np.allclose(
        #         np.sum(res, axis=0).astype(np.float32) / s, [np.sin(x / 2) ** 2, 0], atol=0.1
        #     )
        #     for res, s in zip(result, shots)
        # )
        pass

    @pytest.mark.parametrize("shots", shots_data)
    def test_multi_measurement_shot_vector(self, shots):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        # x, y = np.array(0.732), np.array(0.488)
        # shots = qml.measurements.Shots(shots)
        # qs = qml.tape.QuantumScript(
        #     [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
        #     [qml.expval(qml.PauliZ(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
        #     shots=shots,
        # )
        # result = simulate(qs)
        #
        # assert isinstance(result, tuple)
        # assert len(result) == len(list(shots))
        #
        # for shot_res, s in zip(result, shots):
        #     assert isinstance(shot_res, tuple)
        #     assert len(shot_res) == 3
        #
        #     assert isinstance(shot_res[0], np.float64)
        #     assert isinstance(shot_res[1], np.ndarray)
        #     assert isinstance(shot_res[2], np.ndarray)
        #
        #     assert np.allclose(shot_res[0], np.cos(x), atol=0.1)
        #
        #     assert shot_res[1].shape == (4,)
        #     assert np.allclose(
        #         shot_res[1],
        #         [
        #             np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
        #             np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
        #             np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
        #             np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
        #         ],
        #         atol=0.1,
        #     )
        #
        #     assert shot_res[2].shape == (s, 2)

        # TODO change measurements to valid
        pass

    def test_custom_wire_labels(self):
        """Test that custom wire labels works as expected"""
        # x, y = np.array(0.732), np.array(0.488)
        # qs = qml.tape.QuantumScript(
        #     [qml.RX(x, wires="b"), qml.CNOT(wires=["b", "a"]), qml.RY(y, wires="a")],
        #     [
        #         qml.expval(qml.PauliZ("b")),
        #         qml.probs(wires=["a", "b"]),
        #         qml.sample(wires=["b", "a"]),
        #     ],
        #     shots=10000,
        # )
        # result = simulate(qs)
        #
        # assert isinstance(result, tuple)
        # assert len(result) == 3
        # assert isinstance(result[0], np.float64)
        # assert isinstance(result[1], np.ndarray)
        # assert isinstance(result[2], np.ndarray)
        #
        # assert np.allclose(result[0], np.cos(x), atol=0.1)
        #
        # assert result[1].shape == (4,)
        # assert np.allclose(
        #     result[1],
        #     [
        #         np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
        #         np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
        #         np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
        #         np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
        #     ],
        #     atol=0.1,
        # )
        #
        # assert result[2].shape == (10000, 2)

        # TODO change measurements
        pass


class TestOperatorArithmetic:  # TODO check if necessary
    """TODO add docstring"""

    # TODO:
    #  - testing op arithmatic all-interfaces
    # TODO total = 1 funcs, 5 repeats

    def test_numpy_op_arithmetic(self):
        """Test an operator arithmetic circuit with non-integer wires with numpy."""
        # phi = 1.2
        # ops = [
        #     qml.PauliX("a"),
        #     qml.PauliX("b"),
        #     qml.ctrl(qml.RX(phi, "target") ** 2, ("a", "b", -3), control_values=[1, 1, 0]),
        # ]
        #
        # qs = qml.tape.QuantumScript(
        #     ops,
        #     [
        #         qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
        #         qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
        #     ],
        # )
        #
        # results = simulate(qs)
        # assert qml.math.allclose(results[0], -np.sin(2 * phi) - 1)
        # assert qml.math.allclose(results[1], 3 * np.cos(2 * phi))
        pass

    @pytest.mark.autograd
    def test_autograd_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with autograd."""

        # phi = qml.numpy.array(1.2)
        #
        # def f(x):
        #     ops = [
        #         qml.PauliX("a"),
        #         qml.PauliX("b"),
        #         qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        #     ]
        #
        #     qs = qml.tape.QuantumScript(
        #         ops,
        #         [
        #             qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
        #             qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
        #         ],
        #     )
        #
        #     return qml.numpy.array(simulate(qs))
        #
        # results = f(phi)
        # assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        # assert qml.math.allclose(results[1], 3 * np.cos(phi))
        #
        # g = qml.jacobian(f)(phi)
        # assert qml.math.allclose(g[0], -np.cos(phi))
        # assert qml.math.allclose(g[1], -3 * np.sin(phi))
        pass

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_op_arithmetic(self, use_jit):
        """Test operator arithmetic circuit with non-integer wires works with jax."""
        # import jax
        #
        # phi = jax.numpy.array(1.2)
        #
        # def f(x):
        #     ops = [
        #         qml.PauliX("a"),
        #         qml.PauliX("b"),
        #         qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        #     ]
        #
        #     qs = qml.tape.QuantumScript(
        #         ops,
        #         [
        #             qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
        #             qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
        #         ],
        #     )
        #
        #     return simulate(qs)
        #
        # if use_jit:
        #     f = jax.jit(f)
        #
        # results = f(phi)
        # assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        # assert qml.math.allclose(results[1], 3 * np.cos(phi))
        #
        # g = jax.jacobian(f)(phi)
        # assert qml.math.allclose(g[0], -np.cos(phi))
        # assert qml.math.allclose(g[1], -3 * np.sin(phi))
        pass

    @pytest.mark.torch
    def test_torch_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with torch."""
        # import torch
        #
        # phi = torch.tensor(-0.7290, requires_grad=True)
        #
        # def f(x):
        #     ops = [
        #         qml.PauliX("a"),
        #         qml.PauliX("b"),
        #         qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        #     ]
        #
        #     qs = qml.tape.QuantumScript(
        #         ops,
        #         [
        #             qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
        #             qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
        #         ],
        #     )
        #
        #     return simulate(qs)
        #
        # results = f(phi)
        # assert qml.math.allclose(results[0], -torch.sin(phi) - 1)
        # assert qml.math.allclose(results[1], 3 * torch.cos(phi))
        #
        # g = torch.autograd.functional.jacobian(f, phi)
        # assert qml.math.allclose(g[0], -torch.cos(phi))
        # assert qml.math.allclose(g[1], -3 * torch.sin(phi))
        pass

    @pytest.mark.tf
    def test_tensorflow_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with tensorflow."""
        # import tensorflow as tf
        #
        # phi = tf.Variable(0.4203)
        #
        # def f(x):
        #     ops = [
        #         qml.PauliX("a"),
        #         qml.PauliX("b"),
        #         qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
        #     ]
        #
        #     qs = qml.tape.QuantumScript(
        #         ops,
        #         [
        #             qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
        #             qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
        #         ],
        #     )
        #
        #     return simulate(qs)
        #
        # with tf.GradientTape(persistent=True) as tape:
        #     results = f(phi)
        #
        # assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        # assert qml.math.allclose(results[1], 3 * np.cos(phi))
        #
        # g0 = tape.gradient(results[0], phi)
        # assert qml.math.allclose(g0, -np.cos(phi))
        # g1 = tape.gradient(results[1], phi)
        # assert qml.math.allclose(g1, -3 * np.sin(phi))
        pass


qml.ApproxTimeEvolution


class TestQInfoMeasurements:  # TODO: maybe add some basics here
    pass
