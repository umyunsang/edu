from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


x1 = 0.5
x2 = 1.2

qc = QuantumCircuit(2)

qc.h(0)
qc.h(1)

qc.ry(x1, 0)
qc.ry(x2, 1)

qc.cx(0, 1)

qc.rz(x1 * x2, 1)

qc.draw("mpl")
plt.show()