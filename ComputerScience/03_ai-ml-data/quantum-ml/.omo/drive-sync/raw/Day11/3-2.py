from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

qc = QuantumCircuit(2)

# Superposition 생성
qc.h(0)

# Rotation
qc.rx(0.8, 0)

# Entanglement
qc.cx(0, 1)

# Measurement
qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

plt.show()