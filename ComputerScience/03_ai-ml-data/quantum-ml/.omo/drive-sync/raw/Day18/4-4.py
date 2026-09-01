"""
==========================================================
Lab 6-3
Measurement Probability 변화 비교
==========================================================

목적
- Time Evolution에 따른 Measurement 변화 확인
- Probability Oscillation 분석
- Expectation Value 계산
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

# =====================================================
# STEP 1
# Frequency 설정
# =====================================================

print("=" * 60)
print("STEP 1. Frequency")
print("=" * 60)

frequency = 5

print(f"Frequency : {frequency} Hz")

# =====================================================
# STEP 2
# Time Axis
# =====================================================

print("\n")
print("=" * 60)
print("STEP 2. Time")
print("=" * 60)

time_list = np.linspace(
    0,
    1,
    30
)

print(f"Time Sample : {len(time_list)}")

# =====================================================
# STEP 3
# Simulator
# =====================================================

backend = AerSimulator()

p0_list = []
p1_list = []
expectation_list = []

# =====================================================
# STEP 4
# Measurement
# =====================================================

print("\n")
print("=" * 60)
print("STEP 3. Measurement")
print("=" * 60)

for t in time_list:

    theta = 2 * np.pi * frequency * t

    qc = QuantumCircuit(1)

    qc.ry(theta,0)

    qc.measure_all()

    tqc = transpile(
        qc,
        backend
    )

    job = backend.run(
        tqc,
        shots=4096
    )

    result = job.result()

    counts = result.get_counts()

    shots = sum(counts.values())

    p0 = counts.get("0",0)/shots
    p1 = counts.get("1",0)/shots

    expectation = p0-p1

    p0_list.append(p0)
    p1_list.append(p1)
    expectation_list.append(expectation)

# =====================================================
# STEP 5
# Probability Graph
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    time_list,
    p0_list,
    label="P(0)",
    linewidth=2
)

plt.plot(
    time_list,
    p1_list,
    label="P(1)",
    linewidth=2
)

plt.xlabel("Time (sec)")

plt.ylabel("Probability")

plt.title("Measurement Probability")

plt.grid(True)

plt.legend()

plt.show()

# =====================================================
# STEP 6
# Expectation Graph
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    time_list,
    expectation_list,
    color="red",
    linewidth=2,
    label="<Z>"
)

plt.xlabel("Time (sec)")

plt.ylabel("Expectation")

plt.title("Expectation Value")

plt.grid(True)

plt.legend()

plt.show()

# =====================================================
# STEP 7
# Summary
# =====================================================

print("\n")
print("=" * 60)
print("Summary")
print("=" * 60)

print(" Time    P(0)    P(1)    <Z>")

for t,p0,p1,e in zip(
    time_list,
    p0_list,
    p1_list,
    expectation_list
):

    print(
        f"{t:5.2f}   "
        f"{p0:6.3f}   "
        f"{p1:6.3f}   "
        f"{e:6.3f}"
    )

print("\n실습 완료")
