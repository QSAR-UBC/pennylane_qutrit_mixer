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
"""Functions and variables to be utilized by qutrit mixed state simulator."""
import functools
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import numpy as np

alphabet_array = np.array(list(alphabet))
qudit_dim = 3  # specifies qudit dimension


def get_einsum_indices(op: qml.operation.Operator, state, is_state_batched: bool = False):
    r"""Finds the indices for einsum to multiply three matrices

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        dict: indices used by einsum to multiply 3 matrices
    """
    num_ch_wires = len(op.wires)
    num_wires = int((len(qml.math.shape(state)) - is_state_batched) / 2)
    rho_dim = 2 * num_wires

    # Tensor indices of the state. For each qutrit, need an index for rows *and* columns
    state_indices = alphabet[:rho_dim]

    # row indices of the quantum state affected by this operation
    row_wires_list = op.wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim : rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]

    # new state indices replace row and column indices with new ones
    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(col_indices + row_indices, new_col_indices + new_row_indices),
        state_indices,
    )

    op_1_indices = f"{kraus_index}{new_row_indices}{row_indices}"
    op_2_indices = f"{kraus_index}{col_indices}{new_col_indices}"
    indices = {
        "op1": op_1_indices,
        "state": state_indices,
        "op2": op_2_indices,
        "new_state": new_state_indices,
    }
    return indices