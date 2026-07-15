"""
============================================================
Lab 3.
Statevector 및 Bloch Sphere 분석
============================================================

실습 순서
--------------------------------------------------
STEP 1. Import Library
STEP 2. Sensor Data 준비
STEP 3. Normalization
STEP 4. Phase Angle 계산
STEP 5. Phase Encoding Circuit 생성
STEP 6. Statevector 생성
STEP 7. Complex Amplitude 분석
STEP 8. Probability 계산
STEP 9. Bloch Sphere 출력
STEP 10. Phase 정보 확인
STEP 11. 결과 분석
============================================================
"""

import numpy as np

from qiskit import QuantumCircuit

from qiskit.quantum_info import Statevector

from qiskit.visualization import plot_bloch_multivector

import matplotlib.pyplot as plt

# ============================================================
# STEP 1.
# Smart Factory Sensor Data
# ============================================================

print("="*80)
print("STEP 1. Smart Factory Sensor Data")
print("="*80)

sensor = {

    "Temperature":72,
    "Pressure":6.5,
    "Vibration":40

}

print(sensor)

# ============================================================
# STEP 2.
# Normalization
# ============================================================

print()
print("="*80)
print("STEP 2. Normalization")
print("="*80)

temp = (sensor["Temperature"]-20)/(120-20)

pressure = (sensor["Pressure"]-1)/(10-1)

vibration = sensor["Vibration"]/100

print(f"Temperature : {temp:.4f}")
print(f"Pressure    : {pressure:.4f}")
print(f"Vibration   : {vibration:.4f}")

# ============================================================
# STEP 3.
# Phase Angle
# ============================================================

print()
print("="*80)
print("STEP 3. Phase Angle")
print("="*80)

phase_temp = temp*np.pi

phase_pressure = pressure*np.pi

phase_vibration = vibration*np.pi

print(f"T Phase : {phase_temp:.4f}")

print(f"P Phase : {phase_pressure:.4f}")

print(f"V Phase : {phase_vibration:.4f}")

# ============================================================
# STEP 4.
# Quantum Circuit
# ============================================================

print()
print("="*80)
print("STEP 4. Phase Encoding Circuit")
print("="*80)

qc = QuantumCircuit(3)

qc.h(range(3))

qc.p(phase_temp,0)
qc.p(phase_pressure,1)
qc.p(phase_vibration,2)

print(qc.draw())

# ============================================================
# STEP 5.
# Statevector 생성
# ============================================================

print()
print("="*80)
print("STEP 5. Statevector")
print("="*80)

state = Statevector.from_instruction(qc)

print(state)

# ============================================================
# STEP 6.
# Complex Amplitude
# ============================================================

print()
print("="*80)
print("STEP 6. Complex Amplitude")
print("="*80)

for i, amp in enumerate(state.data):

    print(f"|{i:03b}> : {amp}")

# ============================================================
# STEP 7.
# Magnitude & Phase
# ============================================================

print()
print("="*80)
print("STEP 7. Magnitude & Phase")
print("="*80)

for i, amp in enumerate(state.data):

    magnitude = np.abs(amp)

    phase = np.angle(amp)

    print(
        f"|{i:03b}>"
        f" Magnitude={magnitude:.4f}"
        f" Phase={phase:.4f}"
    )

# ============================================================
# STEP 8.
# Probability
# ============================================================

print()
print("="*80)
print("STEP 8. Probability")
print("="*80)

prob = state.probabilities_dict()

for key,value in prob.items():

    print(

        f"|{key}>"

        f" : {value:.4f}"

    )

# ============================================================
# STEP 9.
# Probability Sum
# ============================================================

print()
print("="*80)
print("STEP 9. Probability Sum")
print("="*80)

print(

    f"Total = {sum(prob.values()):.4f}"

)

# ============================================================
# STEP 10.
# Bloch Sphere
# ============================================================

print()
print("="*80)
print("STEP 10. Bloch Sphere")
print("="*80)

plot_bloch_multivector(state)

plt.show()


print("="*80)
print("Lab 3 Complete")
print("="*80)
