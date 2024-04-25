import pennylane as qml


# print()
mixed_device = qml.device("default.qutrit", wires=2)
@qml.qnode(mixed_device)
def sample_state(x, y):
    qml.TRX(x, 0)
    qml.TRY(y, 1)
    return qml.expval(qml.GellMann(0, 3)@qml.GellMann(1, 1))

print(sample_state(0.4, .67))

print(qml.matrix(sample_state)(0.4, .67))