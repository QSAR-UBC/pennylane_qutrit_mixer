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
"""
Code relevant for performing measurements on a qutrit mixed state.
"""
from typing import Callable
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
    StateMeasurement,
    MeasurementProcess,
    ExpectationMP,
)
from pennylane.operation import Observable
from pennylane.typing import TensorLike
from string import ascii_letters as alphabet
from pennylane.wires import Wires

from .apply_operation import apply_operation
from .utils import get_einsum_indices_two, resquare_state, get_probs, qudit_dim, get_num_wires

alphabet_array = math.asarray(list(alphabet))


def apply_observable_einsum(obs: Observable, state, is_state_batched: bool = False):
    r"""Applies an observable to a density matrix rho, giving obs@state

    Args:
        obs (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        TensorLike: the result of obs@state
    """
    indices = get_einsum_indices_two(obs, state, is_state_batched)
    einsum_indices = f"{indices['op1']},...{indices['state']}->...{indices['new_state']}"
    print(einsum_indices)

    obs_mat = math.cast(obs.matrix(), complex)
    return math.einsum(einsum_indices, obs_mat, state)


def trace_method(
    measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Measure the expectation value of an observable by finding the trace of obs@rho.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not
    Returns:
        TensorLike: the result of the measurement
    """
    obs = measurementprocess.obs
    rho_mult_obs = apply_observable_einsum(obs, state, is_state_batched)
    resquare_state(rho_mult_obs, get_num_wires(state, is_state_batched))
    return math.real(math.trace(rho_mult_obs))


def state_diagonalizing_gates(
    measurementprocess: StateMeasurement, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Apply a measurement to state when the measurement process has an observable with diagonalizing gates.

    Args:
        measurementprocess (StateMeasurement): measurement to apply to the state
        state (TensorLike): state to apply the measurement to
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    for op in measurementprocess.diagonalizing_gates():
        state = apply_operation(op, state, is_state_batched=is_state_batched)

    num_wires = get_num_wires(state, is_state_batched)
    wires = Wires(range(num_wires))
    # resquared state or probs?
    probs = get_probs(state, num_wires)
    return measurementprocess.process_state(probs, wires)


def get_measurement_function(
    measurementprocess: MeasurementProcess, state: TensorLike
) -> Callable[[MeasurementProcess, TensorLike], TensorLike]:
    """Get the appropriate method for performing a measurement.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        Callable: function that returns the measurement result
    """
    if isinstance(measurementprocess, StateMeasurement):
        if isinstance(measurementprocess, ExpectationMP):
            # TODO add faster methods
            # TODO add suport for sparce Hamiltonians
            if isinstance(measurementprocess.obs, Hamiltonian):
                return sum_of_terms_method
            if isinstance(measurementprocess.obs, Sum):
                backprop_mode = math.get_interface(state, *measurementprocess.obs.data) != "numpy"
                if backprop_mode:
                    return sum_of_terms_method
            if measurementprocess.obs.name == "THermitian":
                return trace_method
        if measurementprocess.obs is None or measurementprocess.obs.has_diagonalizing_gates:
            return state_diagonalizing_gates

    raise NotImplementedError


def measure(
    measurementprocess: MeasurementProcess, state: TensorLike, is_state_batched: bool = False
) -> TensorLike:
    """Apply a measurement process to a state.

    Args:
        measurementprocess (MeasurementProcess): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        Tensorlike: the result of the measurement
    """
    return get_measurement_function(measurementprocess, state)(
        measurementprocess, state, is_state_batched
    )


def sum_of_terms_method(  # TODO this is copied code, should this borrow from qubit?
    measurementprocess: ExpectationMP,
    state: TensorLike,
    is_state_batched: bool = False,
    measure_func=measure,
) -> TensorLike:
    """Measure the expecation value of the state when the measured observable is a ``Hamiltonian`` or ``Sum``
    and it must be backpropagation compatible.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not
        measure_func (function): measure function to use

    Returns:
        TensorLike: the result of the measurement
    """
    if isinstance(measurementprocess.obs, Sum):
        # Recursively call measure on each term, so that the best measurement method can
        # be used for each term
        return sum(
            measure_func(ExpectationMP(term), state, is_state_batched=is_state_batched)
            for term in measurementprocess.obs
        )
    # else hamiltonian
    return sum(
        c * measure_func(ExpectationMP(t), state, is_state_batched=is_state_batched)
        for c, t in zip(*measurementprocess.obs.terms())
    )
