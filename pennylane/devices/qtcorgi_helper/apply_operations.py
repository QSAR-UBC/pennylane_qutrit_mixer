import time
import jax
import jax.numpy as jnp
from jax.lax import scan

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import pennylane as qml
from string import ascii_letters as alphabet

import numpy as np
from functools import partial, reduce

alphabet_array = np.array(list(alphabet))


def swap_axes(op, start, fin):
    axes = jnp.arange(op.ndim)
    for s, f in zip(start, fin):
        axes = axes.at[f].set(s)
        axes = axes.at[s].set(f)
    indices = jnp.mgrid[tuple(slice(s) for s in op.shape)]  # TODO can do
    indices = indices[axes]
    return op[tuple(indices[i] for i in range(indices.shape[0]))]


def get_einsum_mapping(wires, state):
    r"""Finds the indices for einsum to apply kraus operators to a mixed state

    Args:
        wires
        state (array[complex]): Input quantum state

    Returns:
        tuple(tuple(int)): Indices mapping that defines the einsum
    """
    num_ch_wires = len(wires)
    num_wires = int(len(qml.math.shape(state)) / 2)
    rho_dim = 2 * num_wires

    # Tensor indices of the state. For each qutrit, need an index for rows *and* columns
    # TODO this may need to be an input from above
    state_indices_list = list(range(rho_dim))
    state_indices = (...,) + tuple(state_indices_list)

    # row indices of the quantum state affected by this operation
    row_indices = tuple(wires)

    # column indices are shifted by the number of wires
    # TODO replace
    col_indices = tuple(w + num_wires for w in wires)  # TODO: Should I do an array?

    # indices in einsum must be replaced with new ones
    new_row_indices = tuple(range(rho_dim, rho_dim + num_ch_wires))
    new_col_indices = tuple(range(rho_dim + num_ch_wires, rho_dim + 2 * num_ch_wires))

    # index for summation over Kraus operators
    kraus_index = (rho_dim + 2 * num_ch_wires,)

    # apply mapping function
    op_1_indices = (...,) + kraus_index + new_row_indices + row_indices
    op_2_indices = (...,) + kraus_index + col_indices + new_col_indices

    new_state_indices = get_new_state_einsum_indices(
        old_indices=col_indices + row_indices,
        new_indices=new_col_indices + new_row_indices,
        state_indices=state_indices_list,
    )
    # index mapping for einsum, e.g., (...0,1,2,3), (...0,1,2,3), (...0,1,2,3), (...0,1,2,3)
    return op_1_indices, state_indices, op_2_indices, new_state_indices


def get_new_state_einsum_indices(old_indices, new_indices, state_indices):
    """Retrieves the einsum indices string for the new state

    Args:
        old_indices tuple(int): indices that are summed
        new_indices tuple(int): indices that must be replaced with sums
        state_indices tuple(int): indices of the original state

    Returns:
        tuple(int): The einsum indices of the new state
    """
    for old, new in zip(old_indices, new_indices):
        state_indices[old] = new
    return (...,) + tuple(state_indices)


def apply_operation_einsum(kraus, swap_inds, state, qudit_dim, num_wires):
    state = swap_axes(state, *swap_inds)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + ([qudit_dim] * num_wires * 2)

    kraus = jnp.stack(kraus)
    kraus_dagger = jnp.conj(jnp.stack(jnp.moveaxis(kraus, source=-1, destination=-2)))

    kraus = jnp.reshape(kraus, kraus_shape)
    kraus_dagger = jnp.reshape(kraus_dagger, kraus_shape)
    op_1_indices, state_indices, op_2_indices, new_state_indices = get_einsum_mapping(
        list(range(num_wires)), state
    )
    state = jnp.einsum(
        kraus, op_1_indices, state, state_indices, kraus_dagger, op_2_indices, new_state_indices
    )
    return swap_axes(state, *swap_inds)


def apply_single_qudit_operation(kraus, wire, state, qudit_dim):
    num_wires = state.ndim // 2
    swap_inds = (wire, wire + num_wires), (0, num_wires)
    return apply_operation_einsum(kraus, swap_inds, state, qudit_dim, 1)


def apply_two_qudit_operation(kraus, wires, state, qudit_dim):
    num_wires = state.ndim // 2
    start = (wires[0], wires[0] + num_wires, wires[1], wires[1] + num_wires)
    fin = jax.lax.cond(
        wires[0] > wires[1],
        lambda: (0, num_wires, 1, 1 + num_wires),
        lambda: (1, 1 + num_wires, 0, num_wires),
    )
    return apply_operation_einsum(kraus, (start, fin), state, qudit_dim, 2)


single_qubit_ops = [
    qml.RX.compute_matrix,
    qml.RY.compute_matrix,
    qml.RZ.compute_matrix,
    lambda _param: qml.Hadamard.compute_matrix(),
]
qubits_qutrit_ops = [
    lambda _param: qml.THadamard.compute_matrix(),
    qml.TRX.compute_matrix,
    qml.TRY.compute_matrix,
    qml.TRZ.compute_matrix,
    partial(qml.TRX.compute_matrix, subspace=[0, 2]),
    partial(qml.TRY.compute_matrix, subspace=[0, 2]),
    partial(qml.TRZ.compute_matrix, subspace=[0, 2]),
    partial(qml.TRX.compute_matrix, subspace=[1, 2]),
    partial(qml.TRY.compute_matrix, subspace=[1, 2]),
    partial(qml.TRZ.compute_matrix, subspace=[1, 2]),
]


def get_qutrit_op_as_qubits(param, op_type):
    new_mat = jnp.eye(4)
    new_mat[:3, :3] = jax.lax.switch(op_type - 1, qubits_qutrit_ops, param)
    return new_mat


def apply_single_qubit_unitary(state, op_info):
    wires, param = op_info["wires"][:1], op_info["param"][0]
    kraus_mat = [jax.lax.switch(op_info["type_indices"][1], single_qubit_ops, param)]
    return apply_single_qudit_operation(kraus_mat, wires, state, 2)


def apply_two_qubit_unitary(state, op_info):
    wires, param = op_info["wires"], op_info["param"]
    op_type = op_info["type_indices"][1]
    kraus_mats = [
        jax.lax.cond(
            op_type == 0,
            lambda *_args: qml.CNOT.compute_matrix,
            get_qutrit_op_as_qubits,
            param,
            op_type,
        )
    ]
    return apply_two_qudit_operation(kraus_mats, wires, state, 2)


def apply_qubit_depolarizing_channel(state, op_info):
    wire, param = op_info["wires"][0], op_info["param"][0]
    kraus_mats = qml.DepolarizingChannel.compute_kraus_matrices(param)
    return apply_single_qudit_operation(kraus_mats, wire, state, 2)


def apply_qubit_flipping_channel(state, op_info):
    wire, param = op_info["wires"][0], op_info["param"]
    kraus_mats = jax.lax.cond(
        op_info["type_indices"][1] == 1,
        qml.AmplitudeDamping.compute_kraus_matrices,
        qml.BitFlip.compute_kraus_matrices,
        param,
    )
    return apply_single_qudit_operation(kraus_mats, wire, state, 2)


def apply_single_qubit_channel(state, op_info):
    return jax.lax.cond(
        op_info["type_indices"][1] == 0,
        apply_qubit_depolarizing_channel,
        apply_qubit_flipping_channel,
        state,
        op_info,
    )


qubit_branches = [apply_single_qubit_unitary, apply_two_qubit_unitary, apply_single_qubit_channel]

single_qutrit_ops_subspace_01 = [
    lambda _param: qml.THadamard.compute_matrix(subspace=[0, 1]),
    qml.TRX.compute_matrix,
    qml.TRY.compute_matrix,
    qml.TRZ.compute_matrix,
]
single_qutrit_ops_subspace_02 = [
    lambda _param: qml.THadamard.compute_matrix(subspace=[0, 2]),
    partial(qml.TRX.compute_matrix, subspace=[0, 2]),
    partial(qml.TRY.compute_matrix, subspace=[0, 2]),
    partial(qml.TRZ.compute_matrix, subspace=[0, 2]),
]
single_qutrit_ops_subspace_12 = [
    lambda _param: qml.THadamard.compute_matrix(subspace=[1, 2]),
    partial(qml.TRX.compute_matrix, subspace=[1, 2]),
    partial(qml.TRY.compute_matrix, subspace=[1, 2]),
    partial(qml.TRZ.compute_matrix, subspace=[1, 2]),
]
single_qutrit_ops = [
    lambda _op_type, _param: qml.THadamard.compute_matrix(),
    lambda op_type, param: jax.lax.switch(op_type, single_qutrit_ops_subspace_01, param),
    lambda op_type, param: jax.lax.switch(op_type, single_qutrit_ops_subspace_02, param),
    lambda op_type, param: jax.lax.switch(op_type, single_qutrit_ops_subspace_12, param),
]

two_qutrits_ops = [qml.TAdd.compute_matrix, lambda: jnp.conj(qml.TAdd.compute_matrix().T)]


def apply_single_qutrit_unitary(state, op_info):
    wire, param = op_info["wires"][0], op_info["params"][0]
    subspace_index, op_type = op_info["wires"][1], op_info["type_indices"][1]
    kraus_mats = [jax.lax.switch(subspace_index, single_qutrit_ops, op_type, param)]
    return apply_single_qudit_operation(kraus_mats, wire, state, 3)


def apply_two_qutrit_unitary(state, op_info):
    wires = op_info["wires"]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], two_qutrits_ops)]
    return apply_two_qudit_operation(kraus_mats, wires, state, 3)


def apply_qutrit_depolarizing_channel(state, op_info):
    wire, param = op_info["wires"][0], op_info["params"][0]
    kraus_mats = qml.QutritDepolarizingChannel.compute_kraus_matrices(param)
    return apply_single_qudit_operation(kraus_mats, wire, state, 3)


def apply_qutrit_subspace_channel(state, op_info):
    wire, params = op_info["wires"][0], op_info["params"]
    kraus_mats = jax.lax.cond(
        op_info["type_indices"][1] == 1,
        qml.QutritAmplitudeDamping.compute_kraus_matrices,
        qml.TritFlip.compute_kraus_matrices,
        *params
    )
    return apply_single_qudit_operation(kraus_mats, wire, state, 3)


def apply_single_qutrit_channel(state, op_info):
    return jax.lax.cond(
        op_info["type_indices"][1] == 0,
        apply_qutrit_depolarizing_channel,
        apply_qutrit_subspace_channel,
        state,
        op_info,
    )


qutrit_branches = [
    apply_single_qutrit_unitary,
    apply_single_qutrit_channel,
    apply_two_qutrit_unitary,
]
