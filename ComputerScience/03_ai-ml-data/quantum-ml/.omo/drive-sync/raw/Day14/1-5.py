import numpy as np
import pandas as pd

data = pd.DataFrame({
    "Customer": ["C001", "C002", "C003", "C004"],
    "Gender": [0, 1, 0, 1],
    "Member": [1, 1, 0, 0],
    "Age": [25, 35, 45, 28],
    "Income": [3000, 5000, 7000, 4000],
    "PurchaseCount": [2, 5, 7, 3],
    "Buy": [0, 1, 1, 0]
})

print(data)

binary_features = ["Gender", "Member"]
continuous_features = ["Age", "Income", "PurchaseCount"]

X_binary = data[binary_features].values
X_continuous = data[continuous_features].values
y = data["Buy"].values

print("Binary Features")
print(X_binary)

print("Continuous Features")
print(X_continuous)

print("Target")
print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, np.pi))

X_continuous_scaled = scaler.fit_transform(X_continuous)

print("Scaled Continuous Features")
print(X_continuous_scaled)


from qiskit import QuantumCircuit

def basis_encoding(binary_values):
    """
    binary_values: 0 또는 1로 구성된 리스트 또는 배열
    """
    num_qubits = len(binary_values)
    qc = QuantumCircuit(num_qubits)

    for i, value in enumerate(binary_values):
        if value == 1:
            qc.x(i)

    return qc

sample_binary = X_binary[1]   # C002
qc_basis = basis_encoding(sample_binary)

print(qc_basis)
qc_basis.draw("mpl")


def angle_encoding(angle_values):
    """
    angle_values: 0 ~ pi 범위로 정규화된 연속형 데이터
    """
    num_qubits = len(angle_values)
    qc = QuantumCircuit(num_qubits)

    for i, value in enumerate(angle_values):
        qc.ry(value, i)

    return qc

sample_angle = X_continuous_scaled[1]   # C002
qc_angle = angle_encoding(sample_angle)

print(qc_angle)
qc_angle.draw("mpl")


def hybrid_encoding(binary_values, angle_values):
    """
    binary_values: Binary Feature
    angle_values: 정규화된 Continuous Feature
    """
    num_binary = len(binary_values)
    num_angle = len(angle_values)
    num_qubits = num_binary + num_angle

    qc = QuantumCircuit(num_qubits)

    # 1. Basis Encoding
    for i, value in enumerate(binary_values):
        if value == 1:
            qc.x(i)

    # 2. Angle Encoding
    offset = num_binary
    for j, value in enumerate(angle_values):
        qc.ry(value, offset + j)

    return qc

idx = 1   # C002

sample_binary = X_binary[idx]
sample_angle = X_continuous_scaled[idx]

qc_hybrid = hybrid_encoding(sample_binary, sample_angle)

print(qc_hybrid)
qc_hybrid.draw("mpl")


circuits = []

for i in range(len(data)):
    qc = hybrid_encoding(
        X_binary[i],
        X_continuous_scaled[i]
    )
    circuits.append(qc)

for i, qc in enumerate(circuits):
    print(f"\nCustomer {data.loc[i, 'Customer']}")
    print(qc)

from qiskit_aer import AerSimulator
from qiskit import transpile

simulator = AerSimulator()

qc_measure = qc_hybrid.copy()
qc_measure.measure_all()

compiled_circuit = transpile(qc_measure, simulator)
result = simulator.run(compiled_circuit, shots=1024).result()

counts = result.get_counts()

print(counts)

from qiskit.quantum_info import Statevector

qc_state = qc_hybrid.copy()

state = Statevector.from_instruction(qc_state)

print(state)

probabilities = state.probabilities_dict()

for basis_state, prob in probabilities.items():


    if prob > 0.001:


        print(basis_state, round(prob, 4))



print(qc_hybrid.count_ops())
print(qc_hybrid.depth())
print(qc_hybrid.num_qubits)



X_continuous_binary = (X_continuous > X_continuous.mean(axis=0)).astype(int)

X_all_binary = np.concatenate([X_binary, X_continuous_binary], axis=1)

qc_basis_only = basis_encoding(X_all_binary[1])


X_all_angle = np.concatenate([X_binary, X_continuous_scaled], axis=1)

qc_angle_only = angle_encoding(X_all_angle[1])


qc_hybrid = hybrid_encoding(X_binary[1], X_continuous_scaled[1])


encoding_methods = {
    "Basis Only": qc_basis_only,
    "Angle Only": qc_angle_only,
    "Hybrid": qc_hybrid
}

for name, qc in encoding_methods.items():
    print(f"\n{name}")
    print("Qubits:", qc.num_qubits)
    print("Depth:", qc.depth())
    print("Gate Count:", qc.count_ops())