from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt

x1 = Parameter("x1")
x2 = Parameter("x2")

qc = QuantumCircuit(2)

qc.h(0)
qc.h(1)

qc.ry(x1, 0)
qc.ry(x2, 1)

qc.cx(0, 1)

qc.draw("mpl")
plt.show()
