"""
============================================================
Lab 2.
Phase Encoding Circuit 생성
============================================================

실습 순서
---------------------------------------------
STEP 1. Import Library
STEP 2. Sensor Data 준비
STEP 3. Normalization
STEP 4. Phase Angle 계산
STEP 5. Quantum Circuit 생성
STEP 6. Hadamard Gate 적용
STEP 7. Phase Encoding
STEP 8. Barrier 추가
STEP 9. Circuit 정보 확인
STEP 10. Circuit 시각화
STEP 11. Gate 개수 확인
STEP 12. 결과 분석
============================================================
"""

import numpy as np

from qiskit import QuantumCircuit

# ============================================================
# STEP 1.
# Sensor Data
# ============================================================

print("=" * 80)
print("STEP 1. Smart Factory Sensor Data")
print("=" * 80)

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
print("=" * 80)
print("STEP 2. Normalization")
print("=" * 80)

temp = (sensor["Temperature"]-20)/(120-20)

pressure = (sensor["Pressure"]-1)/(10-1)

vibration = sensor["Vibration"]/100

print(f"Temperature : {temp:.4f}")

print(f"Pressure    : {pressure:.4f}")

print(f"Vibration   : {vibration:.4f}")

# ============================================================
# STEP 3.
# Phase Angle 계산
# ============================================================

print()
print("=" * 80)
print("STEP 3. Phase Angle")
print("=" * 80)

phase_temp = temp*np.pi

phase_pressure = pressure*np.pi

phase_vibration = vibration*np.pi

print(f"Temperature Phase : {phase_temp:.4f}")

print(f"Pressure Phase    : {phase_pressure:.4f}")

print(f"Vibration Phase   : {phase_vibration:.4f}")

# ============================================================
# STEP 4.
# Quantum Circuit 생성
# ============================================================

print()
print("=" * 80)
print("STEP 4. Quantum Circuit")
print("=" * 80)

qc = QuantumCircuit(3)

print("3-Qubit Quantum Circuit 생성 완료")

# ============================================================
# STEP 5.
# Hadamard Gate
# ============================================================

print()
print("=" * 80)
print("STEP 5. Hadamard Gate")
print("=" * 80)

qc.h(0)
qc.h(1)
qc.h(2)

print("Superposition 생성")

# ============================================================
# STEP 6.
# Phase Encoding
# ============================================================

print()
print("=" * 80)
print("STEP 6. Phase Encoding")
print("=" * 80)

qc.p(
    phase_temp,
    0
)

qc.p(
    phase_pressure,
    1
)

qc.p(
    phase_vibration,
    2
)

print("Temperature → q0")

print("Pressure    → q1")

print("Vibration   → q2")

# ============================================================
# STEP 7.
# Barrier
# ============================================================

print()
print("=" * 80)
print("STEP 7. Barrier")
print("=" * 80)

qc.barrier()

print("Barrier 추가")

# ============================================================
# STEP 8.
# Circuit 정보
# ============================================================

print()
print("=" * 80)
print("STEP 8. Circuit Information")
print("=" * 80)

print(f"Qubits : {qc.num_qubits}")

print(f"Depth  : {qc.depth()}")

print(f"Size   : {qc.size()}")

# ============================================================
# STEP 9.
# Gate 개수
# ============================================================

print()
print("=" * 80)
print("STEP 9. Gate Count")
print("=" * 80)

print(qc.count_ops())

# ============================================================
# STEP 10.
# Circuit 출력
# ============================================================

print()
print("=" * 80)
print("STEP 10. Circuit")
print("=" * 80)

print(qc.draw())

# ============================================================
# STEP 11.
# 저장된 Feature 확인
# ============================================================

print()
print("=" * 80)
print("STEP 11. Feature Mapping")
print("=" * 80)

print("q0 ← Temperature")

print("q1 ← Pressure")

print("q2 ← Vibration")

print()

print("Stored Phase")

print(f"q0 : {phase_temp:.4f}")

print(f"q1 : {phase_pressure:.4f}")

print(f"q2 : {phase_vibration:.4f}")


print("=" * 80)
print("Lab 2 Complete")
print("=" * 80)
