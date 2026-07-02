from qiskit import QuantumCircuit

qc = QuantumCircuit(2)

qc.h(0)
qc.cx(0, 1)

qc.measure_all()

qc.draw("mpl")

from qiskit_aer import AerSimulator

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)


