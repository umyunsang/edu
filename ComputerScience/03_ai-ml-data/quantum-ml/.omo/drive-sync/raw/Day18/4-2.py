"""
==========================================================
Lab 6-1
Frequency별 Expectation Curve 생성
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# =====================================================
# STEP 1
# Time Axis 생성
# =====================================================

print("=" * 60)
print("STEP 1. Time Axis")
print("=" * 60)

time = np.linspace(
    0,
    1,
    100
)

print(f"Time Sample : {len(time)}")

# =====================================================
# STEP 2
# Frequency 정의
# =====================================================

print("\n")
print("=" * 60)
print("STEP 2. Frequency")
print("=" * 60)

frequency_list = [
    5,
    10,
    20,
    40
]

print(frequency_list)

# =====================================================
# STEP 3
# Expectation 계산
# =====================================================

print("\n")
print("=" * 60)
print("STEP 3. Expectation Curve")
print("=" * 60)

plt.figure(figsize=(10,6))

for f in frequency_list:

    expectation = []

    print(f"\nFrequency : {f} Hz")

    for t in time:

        theta = 2 * np.pi * f * t

        qc = QuantumCircuit(1)

        qc.ry(theta,0)

        state = Statevector.from_instruction(qc)

        z = state.expectation_value([[1,0],[0,-1]])

        expectation.append(np.real(z))

    plt.plot(
        time,
        expectation,
        linewidth=2,
        label=f"{f} Hz"
    )

# =====================================================
# STEP 4
# Graph
# =====================================================

plt.title(
    "Frequency Encoding Expectation Curve"
)

plt.xlabel("Time (sec)")

plt.ylabel("<Z> Expectation")

plt.grid(True)

plt.legend()

plt.show()

print("\n실습 완료")
