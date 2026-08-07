"""
=========================================================
 Lab 2. EstimatorQNN 생성 준비
---------------------------------------------------------
학습 목표

1. Input Parameter 추출
2. Weight Parameter 추출
3. Observable 생성
4. StatevectorEstimator 생성
5. EstimatorQNN 생성 준비 완료
=========================================================
"""

# =========================================================
# STEP 1. Library Import
# =========================================================

from qiskit.circuit.library import (
    zz_feature_map,
    real_amplitudes,
)

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

print("=" * 60)
print("STEP 1. Library Import")
print("=" * 60)
print("Library Import 완료\n")


# =========================================================
# STEP 2. Parameterized Quantum Circuit 생성
# =========================================================

NUM_QUBITS = 2

feature_map = zz_feature_map(
    feature_dimension=NUM_QUBITS,
    reps=1,
)

ansatz = real_amplitudes(
    num_qubits=NUM_QUBITS,
    reps=1,
)

circuit = feature_map.compose(ansatz)

print("=" * 60)
print("STEP 2. Parameterized Quantum Circuit")
print("=" * 60)

print(circuit)
print()


# =========================================================
# STEP 3. Input Parameter 추출
# =========================================================

input_params = list(feature_map.parameters)

print("=" * 60)
print("STEP 3. Input Parameter")
print("=" * 60)

print(f"Input Parameter 개수 : {len(input_params)}")

for p in input_params:
    print(" ", p)

print()


# =========================================================
# STEP 4. Weight Parameter 추출
# =========================================================

weight_params = list(ansatz.parameters)

print("=" * 60)
print("STEP 4. Weight Parameter")
print("=" * 60)

print(f"Weight Parameter 개수 : {len(weight_params)}")

for p in weight_params:
    print(" ", p)

print()


# =========================================================
# STEP 5. Parameter 비교
# =========================================================

print("=" * 60)
print("STEP 5. Parameter 비교")
print("=" * 60)

print(f"""
Input Parameter
----------------
개수 : {len(input_params)}

Weight Parameter
----------------
개수 : {len(weight_params)}
""")

print()


# =========================================================
# STEP 6. Observable 생성
# =========================================================

observable = SparsePauliOp.from_list(
    [
        ("ZZ", 1.0)
    ]
)

print("=" * 60)
print("STEP 6. Observable 생성")
print("=" * 60)

print(observable)
print()


# =========================================================
# STEP 7. Observable 정보 확인
# =========================================================

print("=" * 60)
print("STEP 7. Observable 정보")
print("=" * 60)

print("Observable Type")

print(type(observable))

print()

print("Observable Matrix")

print(observable.to_matrix())

print()


# =========================================================
# STEP 8. StatevectorEstimator 생성
# =========================================================

estimator = StatevectorEstimator()

print("=" * 60)
print("STEP 8. StatevectorEstimator 생성")
print("=" * 60)

print(estimator)
print()


# =========================================================
# STEP 9. Estimator 정보 확인
# =========================================================

print("=" * 60)
print("STEP 9. Estimator 정보")
print("=" * 60)

print("Estimator Type")

print(type(estimator))

print()


# =========================================================
# STEP 10. 생성 준비 확인
# =========================================================

print("=" * 60)
print("STEP 10. EstimatorQNN 생성 준비")
print("=" * 60)

print("Circuit")
print(circuit)
print()

print("Input Parameters")

for p in input_params:
    print(" ", p)

print()

print("Weight Parameters")

for p in weight_params:
    print(" ", p)

print()

print("Observable")

print(observable)

print()

print("Estimator")

print(estimator)

print()


# =========================================================
# STEP 11. 생성 준비 요약
# =========================================================

print("=" * 60)
print("EstimatorQNN 생성 준비 Summary")
print("=" * 60)

print(f"""
Circuit
----------------------------------------
Qubit 수               : {circuit.num_qubits}

전체 Parameter 수      : {circuit.num_parameters}

Input Parameter 수     : {len(input_params)}

Weight Parameter 수    : {len(weight_params)}

Observable
----------------------------------------
{observable}

Estimator
----------------------------------------
{type(estimator).__name__}
""")

print("=" * 60)
print("EstimatorQNN 생성 준비 완료")
print("=" * 60)