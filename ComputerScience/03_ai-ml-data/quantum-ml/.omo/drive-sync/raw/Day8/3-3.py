from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc3 = QuantumCircuit(2)

qc3.h(0)

qc3.cx(0, 1)

qc3.measure_all()

qc3.draw("mpl")

sim = AerSimulator()

job = sim.run(qc3, shots=1000)

result = job.result()

counts3 = result.get_counts()

print(counts3)

plot_histogram(counts3)

import matplotlib.pyplot as plt
plt.show()