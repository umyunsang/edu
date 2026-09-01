"""
=========================================================
 Lab 1. Parameterized Quantum Circuit 구성
---------------------------------------------------------
학습 목표
 1. Feature Map 생성
 2. Variational Circuit 생성
 3. Quantum Circuit 결합
 4. Parameterized Quantum Circuit 분석
=========================================================
"""

# =========================================================
# STEP 1. Library Import
# =========================================================

from qiskit.circuit.library import (
    zz_feature_map,
    real_amplitudes,
)

print("=" * 60)
print("STEP 1. Library Import")
print("=" * 60)
print("Library Import 완료\n")


# =========================================================
# STEP 2. Qubit 설정
# =========================================================

NUM_QUBITS = 2

print("=" * 60)
print("STEP 2. Qubit 설정")
print("=" * 60)

print(f"Qubit 수 : {NUM_QUBITS}\n")


# =========================================================
# STEP 3. Feature Map 생성
# =========================================================

feature_map = zz_feature_map(
    feature_dimension=NUM_QUBITS,
    reps=1,
)

print("=" * 60)
print("STEP 3. Feature Map 생성")
print("=" * 60)

print(feature_map)
print()


# =========================================================
# STEP 4. Feature Map 정보 확인
# =========================================================

print("=" * 60)
print("STEP 4. Feature Map 정보")
print("=" * 60)

print(f"Qubit 수        : {feature_map.num_qubits}")
print(f"Parameter 수    : {feature_map.num_parameters}")
print(f"Parameter 목록  :")

for p in feature_map.parameters:
    print(" ", p)

print()


# =========================================================
# STEP 5. Variational Circuit 생성
# =========================================================

ansatz = real_amplitudes(
    num_qubits=NUM_QUBITS,
    reps=1,
)

print("=" * 60)
print("STEP 5. Variational Circuit 생성")
print("=" * 60)

print(ansatz)
print()


# =========================================================
# STEP 6. Ansatz 정보 확인
# =========================================================

print("=" * 60)
print("STEP 6. Ansatz 정보")
print("=" * 60)

print(f"Qubit 수        : {ansatz.num_qubits}")
print(f"Parameter 수    : {ansatz.num_parameters}")
print("Parameter 목록 :")

for p in ansatz.parameters:
    print(" ", p)

print()


# =========================================================
# STEP 7. Parameterized Quantum Circuit 생성
# =========================================================

circuit = feature_map.compose(ansatz)

print("=" * 60)
print("STEP 7. Parameterized Quantum Circuit")
print("=" * 60)

print(circuit)
print()


# =========================================================
# STEP 8. Circuit 정보 확인
# =========================================================

print("=" * 60)
print("STEP 8. Circuit 정보")
print("=" * 60)

print(f"Qubit 수       : {circuit.num_qubits}")
print(f"Parameter 수   : {circuit.num_parameters}")

print("\n전체 Parameter")

for p in circuit.parameters:
    print(" ", p)

print()


# =========================================================
# STEP 9. Input Parameter 확인
# =========================================================

input_params = list(feature_map.parameters)

print("=" * 60)
print("STEP 9. Input Parameter")
print("=" * 60)

for p in input_params:
    print(" ", p)

print()


# =========================================================
# STEP 10. Weight Parameter 확인
# =========================================================

weight_params = list(ansatz.parameters)

print("=" * 60)
print("STEP 10. Weight Parameter")
print("=" * 60)

for p in weight_params:
    print(" ", p)

print()


# =========================================================
# STEP 11. Parameter 분류 확인
# =========================================================

print("=" * 60)
print("STEP 11. Parameter 분류 확인")
print("=" * 60)

print(f"Input Parameter 수  : {len(input_params)}")
print(f"Weight Parameter 수 : {len(weight_params)}")

print()


# =========================================================
# STEP 12. Parameterized Circuit 요약
# =========================================================

print("=" * 60)
print("STEP 12. Parameterized Quantum Circuit Summary")
print("=" * 60)

print(f"""
Feature Map
    └─ Qubit      : {feature_map.num_qubits}
    └─ Parameters : {feature_map.num_parameters}

Ansatz
    └─ Qubit      : {ansatz.num_qubits}
    └─ Parameters : {ansatz.num_parameters}

Parameterized Circuit
    └─ Total Qubit      : {circuit.num_qubits}
    └─ Total Parameter  : {circuit.num_parameters}

Input Parameters
    {input_params}

Weight Parameters
    {weight_params}
""")

print("=" * 60)
print("Lab 1 완료")
print("=" * 60)