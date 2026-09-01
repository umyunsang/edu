from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(1)

qc.z(0)

qc.measure_all()

sim = AerSimulator()

compiled = transpile(qc, sim)

result = sim.run(compiled, shots=1024).result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)
from matplotlib import pyplot as plt
plt.show()