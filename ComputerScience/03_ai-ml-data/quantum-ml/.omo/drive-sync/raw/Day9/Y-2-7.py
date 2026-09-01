from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc_x = QuantumCircuit(1)

qc_x.x(0)

qc_x.measure_all()

sim = AerSimulator()

compiled_x = transpile(qc_x, sim)

result_x = sim.run(compiled_x, shots=1024).result()

counts_x = result_x.get_counts()

print("X Gate:", counts_x)

qc_y = QuantumCircuit(1)

qc_y.y(0)

qc_y.measure_all()

compiled_y = transpile(qc_y, sim)

result_y = sim.run(compiled_y, shots=1024).result()

counts_y = result_y.get_counts()

print("Y Gate:", counts_y)
