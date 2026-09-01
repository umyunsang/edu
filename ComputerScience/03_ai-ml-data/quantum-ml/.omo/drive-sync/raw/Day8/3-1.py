from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc1 = QuantumCircuit(2)

qc1.cx(0, 1)

qc1.measure_all()

qc1.draw("mpl")

sim = AerSimulator()

job = sim.run(qc1, shots=1000)

result = job.result()

counts1 = result.get_counts()

print(counts1)

plot_histogram(counts1)

import matplotlib.pyplot as plt
plt.show()