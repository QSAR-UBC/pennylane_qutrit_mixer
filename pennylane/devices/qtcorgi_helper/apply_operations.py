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
    for s,f in zip(start, fin):
        axes = axes.at[f].set(s)
        axes = axes.at[s].set(f)
    indices = jnp.mgrid[tuple(slice(s) for s in op.shape)] #TODO can do
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
    # return reduce(  # TODO, redo
    #     lambda old_indices, idx_pair: old_indices[idx_pair[0]],
    #     zip(old_indices, new_indices),
    #     state_indices,
    # )


QUDIT_DIM = 3


def apply_single_qudit_operation(kraus, wire, state):
    num_wires = state.ndim // 2
    start, fin = (wire, wire+num_wires), (0, num_wires)
    state = swap_axes(state, start, fin)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * 2

    kraus = jnp.stack(kraus)
    kraus_dagger = jnp.conj(jnp.stack(jnp.moveaxis(kraus, source=-1, destination=-2)))

    kraus = jnp.reshape(kraus, kraus_shape)
    kraus_dagger = jnp.reshape(kraus_dagger, kraus_shape)
    op_1_indices, state_indices, op_2_indices, new_state_indices = get_einsum_mapping([0], state)  # TODO fix
    state = jnp.einsum(kraus, op_1_indices, state, state_indices, kraus_dagger, op_2_indices, new_state_indices)
    return swap_axes(state, fin, start)


def get_swap_indices(num_wires):
    return (0, num_wires, 1, 1 + num_wires)


def get_swap_indices_opposite(num_wires):
    return (1, 1 + num_wires, 0, num_wires)


def apply_two_qudit_operation(kraus, wires, state):
    num_wires = state.ndim//2
    start = (wires[0], wires[0]+num_wires, wires[1], wires[1]+num_wires)
    fin = jax.lax.cond(wires[0] > wires[1], get_swap_indices, get_swap_indices_opposite, num_wires)
    state = swap_axes(state, start, fin)
    state = swap_axes(state, start, fin)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * 4  # 2 * num_wires = 4

    kraus = jnp.stack(kraus)
    kraus_dagger = jnp.conj(jnp.stack(jnp.moveaxis(kraus, source=-1, destination=-2)))

    kraus = jnp.reshape(kraus, kraus_shape)
    kraus_dagger = jnp.reshape(kraus_dagger, kraus_shape)
    op_1_indices, state_indices, op_2_indices, new_state_indices = get_einsum_mapping([0, 1], state)
    state = jnp.einsum(kraus, op_1_indices, state, state_indices, kraus_dagger, op_2_indices, new_state_indices)
    return swap_axes(state, fin, start)


def apply_operation_einsum(kraus, wires, state):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator.

    Args:
        kraus (??): TODO
        wires
        state (array[complex]): Input quantum state

    Returns:
        array[complex]: output_state
    """
    op_1_indices, state_indices, op_2_indices, new_state_indices = get_einsum_mapping(wires, state)

    num_ch_wires = len(wires)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_ch_wires * 2

    kraus = jnp.stack(kraus)
    kraus_dagger = jnp.conj(jnp.stack(jnp.moveaxis(kraus, source=-1, destination=-2)))

    kraus = jnp.reshape(kraus, kraus_shape)
    kraus_dagger = jnp.reshape(kraus_dagger, kraus_shape)

    return jnp.einsum(
        kraus, op_1_indices, state, state_indices, kraus_dagger, op_2_indices, new_state_indices
    )


def get_two_qubit_unitary_matrix(param):
    # TODO
    pass


def get_CNOT_matrix(_param):
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


single_qubit_ops = [qml.RX.compute_matrix, qml.RY.compute_matrix, qml.RZ.compute_matrix]
two_qubit_ops = [get_CNOT_matrix, get_two_qubit_unitary_matrix]
single_qubit_channels = [
    qml.DepolarizingChannel.compute_kraus_matrices,
    qml.AmplitudeDamping.compute_kraus_matrices,
    qml.BitFlip.compute_kraus_matrices,
]


def apply_single_qubit_unitary(state, op_info):
    wires, param = op_info["wires"][:1], op_info["params"][0]
    kraus_mat = jax.lax.switch(op_info["type_indices"][1], single_qubit_ops, param)
    return apply_operation_einsum(kraus_mat, wires, state)


def apply_two_qubit_unitary(state, op_info):
    wires, params = op_info["wires"], op_info["params"]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], two_qubit_ops, params)]
    return apply_operation_einsum(kraus_mats, wires, state)


def apply_single_qubit_channel(state, op_info):
    wires, param = op_info["wires"][:1], op_info["params"][0]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], single_qubit_channels, param)]
    return apply_operation_einsum(kraus_mats, wires, state)


qubit_branches = [apply_single_qubit_unitary, apply_two_qubit_unitary, apply_single_qubit_channel]

single_qutrit_ops_subspace_01 = [
    qml.TRX.compute_matrix,
    qml.TRY.compute_matrix,
    qml.TRZ.compute_matrix,
    lambda _param: qml.THadamard.compute_matrix(subspace=[0, 1]),
]
single_qutrit_ops_subspace_02 = [
    partial(qml.TRX.compute_matrix, subspace=[0, 2]),
    partial(qml.TRY.compute_matrix, subspace=[0, 2]),
    partial(qml.TRZ.compute_matrix, subspace=[0, 2]),
    lambda _param: qml.THadamard.compute_matrix(subspace=[0, 2]),
]
single_qutrit_ops_subspace_12 = [
    partial(qml.TRX.compute_matrix, subspace=[1, 2]),
    partial(qml.TRY.compute_matrix, subspace=[1, 2]),
    partial(qml.TRZ.compute_matrix, subspace=[1, 2]),
    lambda _param: qml.THadamard.compute_matrix(subspace=[0, 2]),
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
    return apply_single_qudit_operation(kraus_mats, wire, state)


def apply_two_qutrit_unitary(state, op_info):
    wires = op_info["wires"]
    kraus_mats = [jax.lax.switch(op_info["type_indices"][1], two_qutrits_ops)]
    return apply_two_qudit_operation(kraus_mats, wires, state)


def apply_qutrit_depolarizing_channel(state, op_info):
    wire, param = op_info["wires"][0], op_info["params"][0]
    kraus_mats = qml.QutritDepolarizingChannel.compute_kraus_matrices(param)
    return apply_single_qudit_operation(kraus_mats, wire, state)


def apply_qutrit_subspace_channel(state, op_info):
    wire, params = op_info["wires"][0], op_info["params"]
    print(params)
    kraus_mats = jax.lax.cond(op_info["type_indices"][1] == 1, qml.QutritAmplitudeDamping.compute_kraus_matrices, qml.TritFlip.compute_kraus_matrices, *params)
    return apply_single_qudit_operation(kraus_mats, wire, state)


def apply_single_qutrit_channel(state, op_info):
    return jax.lax.cond(op_info["type_indices"][1] == 0, apply_qutrit_depolarizing_channel,
                        apply_qutrit_subspace_channel, state, op_info)

qutrit_branches = [
    apply_single_qutrit_unitary,
    apply_single_qutrit_channel,
    apply_two_qutrit_unitary,
]
