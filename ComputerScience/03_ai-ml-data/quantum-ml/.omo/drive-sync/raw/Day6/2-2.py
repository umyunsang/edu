from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(1)

qc.h(0)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=100)

result = job.result()

counts = result.get_counts()

print(counts)