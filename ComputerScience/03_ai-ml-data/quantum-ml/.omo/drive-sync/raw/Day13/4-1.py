"""
===========================================================
Mini QML Pipeline
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
purchase_score = 82

print(f"Age            : {age}")
print(f"Purchase Score : {purchase_score}")

# =====================================================
# STEP 2
# Feature 정규화
# =====================================================

print("\n")
print("=" * 60)
print("STEP 2. Normalize")
print("=" * 60)

# 0~1 범위로 정규화
x1 = age / 100
x2 = purchase_score / 100

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

feature_map.draw("mpl")
plt.show()

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

ansatz.draw("mpl")
plt.show()

# =====================================================
# STEP 5
# QML Pipeline 생성
# =====================================================

print("\n")
print("=" * 60)
print("STEP 5. Compose Circuit")
print("=" * 60)

qml_circuit = feature_map.compose(ansatz)

print(qml_circuit)

qml_circuit.draw("mpl")
plt.show()

# =====================================================
# STEP 6
# Measurement 추가
# =====================================================

print("\n")
print("=" * 60)
print("STEP 6. Measurement")
print("=" * 60)

qml_circuit.measure_all()

qml_circuit.draw("mpl")
plt.show()

# =====================================================
# STEP 7
# Parameter 확인
# =====================================================

print("\n")
print("=" * 60)
print("STEP 7. Parameter List")
print("=" * 60)

parameters = list(qml_circuit.parameters)

for p in parameters:
    print(p)

# =====================================================
# STEP 8
# Parameter Binding
# =====================================================

print("\n")
print("=" * 60)
print("STEP 8. Parameter Binding")
print("=" * 60)

parameter_values = {}

for p in parameters:
    value = np.random.random()

    parameter_values[p] = value

    print(f"{str(p):20} -> {value:.3f}")

bound_circuit = qml_circuit.assign_parameters(
    parameter_values
)

# =====================================================
# STEP 9
# Circuit 실행
# =====================================================

print("\n")
print("=" * 60)
print("STEP 9. Simulation")
print("=" * 60)

simulator = AerSimulator()

compiled = transpile(
    bound_circuit,
    simulator
)

job = simulator.run(
    compiled,
    shots=1024
)

result = job.result()

counts = result.get_counts()

print("Measurement Result")

print(counts)

# =====================================================
# STEP 10
# Histogram
# =====================================================

print("\n")
print("=" * 60)
print("STEP 10. Histogram")
print("=" * 60)

plot_histogram(counts)

plt.show()

# =====================================================
# STEP 11
# 실습 종료
# =====================================================

print("\n")
print("=" * 60)
print("Mini QML Pipeline Completed")
print("=" * 60)