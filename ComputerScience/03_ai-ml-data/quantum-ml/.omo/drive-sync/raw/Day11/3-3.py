from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

qc = QuantumCircuit(2)

qc.h(0)

qc.rx(0.8,0)

qc.ry(1.2,1)

qc.cx(0,1)

qc.rz(0.6,0)

qc.x(1)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc,shots=1000)

result = job.result()

counts = result.get_counts()
print(counts)

plot_histogram(counts)
plt.show()
