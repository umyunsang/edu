"""
==========================================================
Lab 6-2
Time Evolution과 Statevector 분석
==========================================================

목적
- Time Evolution 이해
- Frequency에 따른 Quantum State 변화 확인
- Statevector 분석
- Bloch Sphere 변화 확인
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

# =====================================================
# STEP 1
# Frequency 설정
# =====================================================

print("=" * 60)
print("STEP 1. Frequency")
print("=" * 60)

frequency = 10

print(f"Frequency : {frequency} Hz")

# =====================================================
# STEP 2
# Time Axis 생성
# =====================================================

print("\n")
print("=" * 60)
print("STEP 2. Time")
print("=" * 60)

time_list = np.linspace(
    0,
    0.5,
    6
)

print(time_list)

# =====================================================
# STEP 3
# Time Evolution
# =====================================================

for index, t in enumerate(time_list):

    print("\n")
    print("=" * 60)
    print(f"Time Step {index+1}")
    print("=" * 60)

    print(f"Time : {t:.3f} sec")

    # --------------------------------------------
    # Rotation Angle
    # --------------------------------------------

    theta = 2 * np.pi * frequency * t

    print(f"Rotation : {theta:.4f} rad")
    print(f"Rotation : {np.degrees(theta):.2f} degree")

    # --------------------------------------------
    # Quantum Circuit
    # --------------------------------------------

    qc = QuantumCircuit(1)

    qc.ry(theta,0)

    print("\nQuantum Circuit")

    print(qc.draw())

    # --------------------------------------------
    # Statevector
    # --------------------------------------------

    state = Statevector.from_instruction(qc)

    print("\nStatevector")

    print(state)

    amp = state.data

    print("\nAmplitude")

    print(f"|0> : {amp[0]}")
    print(f"|1> : {amp[1]}")

    # --------------------------------------------
    # Probability
    # --------------------------------------------

    p0 = abs(amp[0])**2
    p1 = abs(amp[1])**2

    print("\nProbability")

    print(f"P(0) = {p0:.4f}")
    print(f"P(1) = {p1:.4f}")

    # --------------------------------------------
    # Expectation
    # --------------------------------------------

    expectation = p0 - p1

    print("\nExpectation")

    print(f"<Z> = {expectation:.4f}")

    # --------------------------------------------
    # Bloch Sphere
    # --------------------------------------------

    print("\nBloch Sphere")

    plot_bloch_multivector(state)

    plt.show()

print("\n실습 완료")
