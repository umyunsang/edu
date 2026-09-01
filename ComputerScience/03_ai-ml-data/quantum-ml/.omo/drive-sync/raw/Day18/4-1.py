"""
==========================================================
Frequency Encoding 구현 실습
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.visualization import plot_bloch_multivector

# -------------------------------------------------------
# STEP 1
# 입력 데이터
# -------------------------------------------------------

print("=" * 60)
print("STEP 1. Input Feature")
print("=" * 60)

x = 0.25

print(f"Input Feature x = {x}")

# -------------------------------------------------------
# STEP 2
# Frequency 목록
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("STEP 2. Frequency List")
print("=" * 60)

frequency_list = [1, 2, 3, 4, 5]

print(frequency_list)

# 결과 저장용

expectation_list = []
theory_list = []

backend = AerSimulator()

# -------------------------------------------------------
# STEP 3
# Frequency 반복
# -------------------------------------------------------

for f in frequency_list:

    print("\n")
    print("=" * 60)
    print(f"Frequency = {f} Hz")
    print("=" * 60)

    # -----------------------------------------------
    # STEP 4
    # Rotation Angle
    # -----------------------------------------------

    theta = 2 * np.pi * f * x

    print(f"Rotation Angle = {theta:.4f} rad")
    print(f"Rotation Angle = {np.degrees(theta):.2f} degree")

    # -----------------------------------------------
    # STEP 5
    # Quantum Circuit
    # -----------------------------------------------

    qc = QuantumCircuit(1)

    # Frequency Encoding
    qc.ry(theta, 0)

    print("\nQuantum Circuit")
    print(qc.draw())

    # -----------------------------------------------
    # STEP 6
    # Statevector 확인
    # -----------------------------------------------

    state = Statevector.from_instruction(qc)

    print("\nStatevector")

    print(state)

    # Bloch Sphere

    plot_bloch_multivector(state)

    plt.show()

    # -----------------------------------------------
    # STEP 7
    # Measurement
    # -----------------------------------------------

    qc.measure_all()

    tqc = transpile(qc, backend)

    job = backend.run(tqc, shots=4096)

    result = job.result()

    counts = result.get_counts()

    print("\nMeasurement")

    print(counts)

    # -----------------------------------------------
    # STEP 8
    # Probability 계산
    # -----------------------------------------------

    shots = sum(counts.values())

    p0 = counts.get('0', 0) / shots
    p1 = counts.get('1', 0) / shots

    print("\nProbability")

    print(f"P(0) = {p0:.4f}")
    print(f"P(1) = {p1:.4f}")

    # -----------------------------------------------
    # STEP 9
    # Expectation 계산
    # -----------------------------------------------

    expectation = p0 - p1

    theory = np.cos(theta)

    expectation_list.append(expectation)

    theory_list.append(theory)

    print("\nExpectation")

    print(f"Experiment : {expectation:.4f}")
    print(f"Theory     : {theory:.4f}")

# -------------------------------------------------------
# STEP 10
# 결과 출력
# -------------------------------------------------------

print("\n")
print("=" * 60)
print("Summary")
print("=" * 60)

print("Frequency\tExperiment\tTheory")

for f, e, t in zip(frequency_list,
                   expectation_list,
                   theory_list):

    print(f"{f}\t\t{e:.4f}\t\t{t:.4f}")

# -------------------------------------------------------
# STEP 11
# 그래프
# -------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    frequency_list,
    expectation_list,
    'bo-',
    linewidth=2,
    label="Experiment"
)

plt.plot(
    frequency_list,
    theory_list,
    'r--',
    linewidth=2,
    label="Theory"
)

plt.xlabel("Frequency (Hz)")

plt.ylabel("<Z> Expectation")

plt.title("Frequency Encoding Result")

plt.grid(True)

plt.legend()

plt.show()

print("\n실습 완료")
