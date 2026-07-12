from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()

qc = QuantumCircuit(1)
qc.h(0)
qc.measure_all()

job = sim.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()

print(counts)
plot_histogram(counts)
