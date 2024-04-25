import pennylane as qml
import functools

# mixed_device = qml.device("default.qutrit.mixed")
# vec_device = qml.device("default.qutrit.mixed")
#
#
# def circ(x, y):
#     qml.TRX(x, 0)
#     qml.TRY(y, 1)
#     return qml.expval(qml.GellMann(0, 3)@qml.GellMann(1, 1))
#
#
# mixed_circ = qml.qnode(mixed_device, circ)
# vec_circ = qml.qnode(vec_device, circ)

# print()
mixed_device = qml.device("default.qutrit.mixed", measurement_error=(0,0.2,0.14))
@qml.qnode(mixed_device)
def sample_state(x, y):
    qml.TRX(x, 0)
    qml.TRY(y, 1)
    return qml.expval(qml.GellMann(0, 3)@qml.GellMann(1, 1))

print(sample_state(0.4, .67))

