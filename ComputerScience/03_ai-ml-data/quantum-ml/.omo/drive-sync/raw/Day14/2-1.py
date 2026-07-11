"""
===========================================================
Workflow 직접 설계하기
QML End-to-End Workflow
===========================================================
"""

import numpy as np

from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import real_amplitudes

from qiskit_aer import AerSimulator

from qiskit import transpile

from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

# =====================================================
# STEP 1
# Classical Data 준비
# =====================================================

print("=" * 60)
print("STEP 1. Classical Data")
print("=" * 60)

# 예제 입력 데이터
age = 25
income = 280

print(f"Age    : {age}")
print(f"Income : {income}")

# =====================================================
# STEP 2
# Feature 정규화
# =====================================================

print("\n")
print("=" * 60)
print("STEP 2. Normalize")
print("=" * 60)

# 간단한 정규화 예시
x1 = age / 100
x2 = income / 400

x = [x1, x2]

print("Input Feature")
print(x)

# =====================================================
# STEP 3
# Feature Map 생성
# =====================================================

print("\n")
print("=" * 60)
print("STEP 3. Feature Map")
print("=" * 60)

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=1
)

print(feature_map)

# =====================================================
# STEP 4
# Variational Circuit 생성
# =====================================================

print("\n")
print("=" * 60)
print("STEP 4. Variational Circuit")
print("=" * 60)

ansatz = real_amplitudes(
    num_qubits=2,
    reps=1
)

print(ansatz)

# =====================================================
# STEP 5
# Pipeline 결합
# =====================================================

print("\n")
print("=" * 60)
print("STEP 5. Compose Circuit")
print("=" * 60)

qml_circuit = feature_map.compose(ansatz)

print(qml_circuit)

# =====================================================
# STEP 6
# Parameter 확인
# =====================================================

print("\n")
print("=" * 60)
print("STEP 6. Parameters")
print("=" * 60)

for p in qml_circuit.parameters:
    print(p)

# =====================================================
# STEP 7
# Parameter Binding
# =====================================================

print("\n")
print("=" * 60)
print("STEP 7. Bind Parameters")
print("=" * 60)

parameter_values = {}

for parameter in qml_circuit.parameters:

    if parameter.name.startswith("x"):

        index = int(parameter.name[2:-1])

        parameter_values[parameter] = x[index]

    else:

        parameter_values[parameter] = np.random.random()

bound_circuit = qml_circuit.assign_parameters(parameter_values)

print(bound_circuit)

# =====================================================
# STEP 8
# Measurement 추가
# =====================================================

print("\n")
print("=" * 60)
print("STEP 8. Measurement")
print("=" * 60)

bound_circuit.measure_all()

print(bound_circuit)

# =====================================================
# STEP 9
# Circuit Draw
# =====================================================

print("\n")
print("=" * 60)
print("STEP 9. Circuit")
print("=" * 60)

bound_circuit.draw("mpl")

plt.show()

# =====================================================
# STEP 10
# Simulation
# =====================================================

print("\n")
print("=" * 60)
print("STEP 10. Simulation")
print("=" * 60)

simulator = AerSimulator()

compiled = transpile(bound_circuit, simulator)

job = simulator.run(compiled, shots=1024)

result = job.result()

counts = result.get_counts()

print(counts)

# =====================================================
# STEP 11
# Histogram
# =====================================================

plot_histogram(counts)

plt.show()

# =====================================================
# STEP 12
# Prediction
# =====================================================

print("\n")
print("=" * 60)
print("STEP 12. Prediction")
print("=" * 60)

zero_probability = (
    counts.get("00", 0)
    + counts.get("01", 0)
) / 1024

one_probability = (
    counts.get("10", 0)
    + counts.get("11", 0)
) / 1024

print(f"Class 0 Probability : {zero_probability:.3f}")
print(f"Class 1 Probability : {one_probability:.3f}")

if zero_probability > one_probability:

    prediction = "Loan Approved"

else:

    prediction = "Loan Denied"

print()
print("Prediction")
print("------------------------")
print(prediction)

print("\n")
print("=" * 60)
print("QML Workflow Finished")
print("=" * 60)