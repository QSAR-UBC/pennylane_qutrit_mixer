import pytest
import pennylane as qml

@pytest.mark.usefixture("use_legacy_and_new_opmath")
class TestHelper:
    def test_op_math(self):
        H = qml.Hamiltonian([1,2,3], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(0), qml.PauliX(1)])
        print(dir(H))
        print(type(H))
        assert True

    @pytest.mark.usefixtures("use_legacy_opmath")
    def test_op_math_legacy(self):
        H = qml.Hamiltonian([1,2,3], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(0), qml.PauliX(1)])
        print(dir(H))
        print(type(H))
        assert True