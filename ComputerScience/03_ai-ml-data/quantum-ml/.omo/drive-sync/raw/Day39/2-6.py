from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Parameter 생성
gamma = Parameter("γ")
beta = Parameter("β")

# Circuit 생성
num_qubits = 2
qc = QuantumCircuit(num_qubits)

# Initial State
for qubit in range(num_qubits):
    qc.h(qubit)

qc.barrier(label="Initial")

# Cost Layer
qc.rzz(2 * gamma, 0, 1)

qc.barrier(label="Cost")

# Mixer Layer
for qubit in range(num_qubits):
    qc.rx(2 * beta, qubit)

qc.barrier(label="Mixer")

# Measurement
qc.measure_all()

print("=== Parameterized QAOA Circuit ===")
print(qc.draw())

# Parameter Binding
bound_qc = qc.assign_parameters({
    gamma: 0.6,
    beta: 0.4
})

print("\n=== Bound QAOA Circuit ===")
print(bound_qc.draw())