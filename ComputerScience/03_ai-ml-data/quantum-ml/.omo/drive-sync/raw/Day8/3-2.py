from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc2 = QuantumCircuit(2)

qc2.h(0)

qc2.measure_all()

qc2.draw("mpl")

sim = AerSimulator()

job = sim.run(qc2, shots=1000)

result = job.result()

counts2 = result.get_counts()

print(counts2)

plot_histogram(counts2)

import matplotlib.pyplot as plt
plt.show()