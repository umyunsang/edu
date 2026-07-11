from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

qc = QuantumCircuit(2)

age = np.pi / 3
salary = np.pi / 4

qc.ry(age, 0)
qc.ry(salary, 1)

qc.h(0)
qc.h(1)

qc.rz(np.pi/6,0)
qc.rx(np.pi/8,1)

#qc.cx(0,1)

qc.ry(np.pi/5,0)
qc.rz(np.pi/7,1)

qc.draw("mpl")

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)
print(qc.count_ops())
print(qc.depth())
print(qc)

plot_histogram(counts)
plt.show()
