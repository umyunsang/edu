from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(2)

qc.h(0)

qc.cx(0,1)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

plot_histogram(dict(counts))
from matplotlib import pyplot as plt
plt.show()
