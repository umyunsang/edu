from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

sim = AerSimulator()

# Circuit C: H -> X
qc_c = QuantumCircuit(1)
qc_c.h(0)
qc_c.x(0)
qc_c.measure_all()

job_c = sim.run(qc_c, shots=1000)
result_c = job_c.result()
counts_c = result_c.get_counts()

print("Circuit C:", counts_c)


# Circuit D: X -> H
qc_d = QuantumCircuit(1)
qc_d.x(0)
qc_d.h(0)
qc_d.measure_all()

job_d = sim.run(qc_d, shots=1000)
result_d = job_d.result()
counts_d = result_d.get_counts()

print("Circuit D:", counts_d)