import pennylane as qml

# def circ(x):
#     qml.RX(x, 0)
#     return qml.expval(qml.PauliZ(0))
#
# qnode = qml.QNode(circ, qml.device("default.qubit", wires=1), diff_method="parameter-shift")
# grad = qml.grad(qnode)
# x = qml.numpy.array(1.3, requires_grad=True)
# print(grad(x))
#
#
# def circ(x):
#     qml.TRX(x, 0)
#     return qml.expval(qml.GellMann(0, 3))
#
# qnode = qml.QNode(circ, qml.device("default.qutrit.mixed", wires=1), diff_method="parameter-shift")
# grad = qml.grad(qnode)
# x = qml.numpy.array(1.3, requires_grad=True)
# print(grad(x))


# """
# QNode checks if qfunc uses shots sets arguments
# """
# qnode()
#
# """
# qnode():
# - self.construct(args, kwargs) #construct tape
# - if new deice type:
#     config = _make_execution_config(self)
#     device_transform_program, config = self.device.preprocess(execution_config=config)
#     full_transform_program = self.transform_program + device_transform_program
# - if old device type:
#     full_transform_program = qml.transforms.core.TransformProgram(self.transform_program)
# -
# -
# -
# -
# """
# grad = qml.grad(qnode)
# grad()
#
# print(type(qml.GellMann(0, 1) @ qml.GellMann(1, 2)))
# qml.TensorN


# H = qml.Hamiltonian([1,2,3], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(0), qml.PauliX(1)])
# print(H)
# print(type(H))
#
#
# H = qml.Hamiltonian([1,2,3], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(0), qml.PauliX(1)])
# print()
# print(H)
# print(type(H))


from pennylane.devices.execution_config import ExecutionConfig

ExecutionConfig