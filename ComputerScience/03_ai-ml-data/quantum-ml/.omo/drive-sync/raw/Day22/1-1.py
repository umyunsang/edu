"""
============================================================
Classical Kernel에서 Quantum Kernel으로
============================================================
"""

import numpy as np

# 두 개의 샘플 데이터
x1 = np.array([0.3, 0.8])

x2 = np.array([1.2, 0.5])

print("=" * 60)
print("STEP 1. Classical Data")
print("=" * 60)

print("Sample A :", x1)
print("Sample B :", x2)


print("=" * 60)
print("STEP 2. Feature Vector")
print("=" * 60)

print("Feature Dimension :", len(x1))

for i, value in enumerate(x1):
    print(f"x1[{i}] = {value}")

print()

for i, value in enumerate(x2):
    print(f"x2[{i}] = {value}")


from qiskit.circuit.library import zz_feature_map

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=1
)

print("=" * 60)
print("STEP 3. ZZFeatureMap")
print("=" * 60)

print(feature_map)


circuit = feature_map.assign_parameters(x1)

print("=" * 60)
print("STEP 4. Data Encoding")
print("=" * 60)

print(circuit)


from qiskit.quantum_info import Statevector

state = Statevector.from_instruction(circuit)

print("=" * 60)
print("STEP 5. Quantum State")
print("=" * 60)

print(state)


print("=" * 60)
print("STEP 6. Think Again")
print("=" * 60)

print("Classical Kernel")

print()

print("Input : Feature Vector")

print()

print("Quantum Kernel")

print()

print("Input : Quantum State")

print()




