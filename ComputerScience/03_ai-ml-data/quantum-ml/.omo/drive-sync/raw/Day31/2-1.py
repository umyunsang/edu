"""
=========================================================
 Lab QNN 구성 요소 확인 실습
---------------------------------------------------------
실습 목표
1. Feature Map 생성
2. Input Parameter 확인
3. Ansatz 생성
4. Weight Parameter 확인
5. Observable 생성
6. QNN 구성 요소 비교
=========================================================
"""

# =========================================================
# STEP 1. Library Import
# =========================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

print("Library Import 완료\n")


# =========================================================
# STEP 2. Feature Map 생성
# =========================================================

print("=" * 70)
print("STEP 2. Feature Map 생성")
print("=" * 70)

feature_map = ZZFeatureMap(
    feature_dimension=2,
    reps=1
)

print(feature_map)

print()


# =========================================================
# STEP 3. Feature Map 회로 출력
# =========================================================

print("=" * 70)
print("STEP 3. Feature Map 회로")
print("=" * 70)

print(feature_map.draw())

print()


# =========================================================
# STEP 4. Input Parameter 확인
# =========================================================

print("=" * 70)
print("STEP 4. Input Parameter")
print("=" * 70)

print(feature_map.parameters)

print()


# =========================================================
# STEP 5. Input Parameter 개수
# =========================================================

print("=" * 70)
print("STEP 5. Input Parameter 개수")
print("=" * 70)

print("Parameter 개수 :", feature_map.num_parameters)

print()


# =========================================================
# STEP 6. Ansatz 생성
# =========================================================

print("=" * 70)
print("STEP 6. Ansatz 생성")
print("=" * 70)

ansatz = RealAmplitudes(
    num_qubits=2,
    reps=1
)

print(ansatz)

print()


# =========================================================
# STEP 7. Ansatz 회로 출력
# =========================================================

print("=" * 70)
print("STEP 7. Ansatz 회로")
print("=" * 70)

print(ansatz.draw())

print()


# =========================================================
# STEP 8. Weight Parameter 확인
# =========================================================

print("=" * 70)
print("STEP 8. Weight Parameter")
print("=" * 70)

print(ansatz.parameters)

print()


# =========================================================
# STEP 9. Weight Parameter 개수
# =========================================================

print("=" * 70)
print("STEP 9. Weight Parameter 개수")
print("=" * 70)

print("Weight Parameter :", ansatz.num_parameters)

print()


# =========================================================
# STEP 10. Observable 생성
# =========================================================

print("=" * 70)
print("STEP 10. Observable 생성")
print("=" * 70)

observable = SparsePauliOp.from_list(
    [
        ("ZZ", 1.0)
    ]
)

print(observable)

print()


# =========================================================
# STEP 11. Observable 비교
# =========================================================

print("=" * 70)
print("STEP 11. Observable 비교")
print("=" * 70)

observable_zi = SparsePauliOp.from_list(
    [
        ("ZI", 1.0)
    ]
)

observable_iz = SparsePauliOp.from_list(
    [
        ("IZ", 1.0)
    ]
)

observable_zz = SparsePauliOp.from_list(
    [
        ("ZZ", 1.0)
    ]
)

print("ZI")
print(observable_zi)

print()

print("IZ")
print(observable_iz)

print()

print("ZZ")
print(observable_zz)

print()


# =========================================================
# STEP 12. Parameter 비교
# =========================================================

print("=" * 70)
print("STEP 12. Parameter 비교")
print("=" * 70)

print("Feature Map Parameter")
print(feature_map.parameters)

print()

print("Ansatz Parameter")
print(ansatz.parameters)

print()

print("Feature Parameter 수 :", feature_map.num_parameters)
print("Weight Parameter 수 :", ansatz.num_parameters)

print()


# =========================================================
# STEP 13. reps 증가 실험
# =========================================================

print("=" * 70)
print("STEP 13. reps 증가 실험")
print("=" * 70)

ansatz_large = RealAmplitudes(
    num_qubits=2,
    reps=3
)

print(ansatz_large.draw())

print()

print("Weight Parameter 수 :", ansatz_large.num_parameters)

print()


# =========================================================
# STEP 14. Feature 수 증가 실험
# =========================================================

print("=" * 70)
print("STEP 14. Feature 증가 실험")
print("=" * 70)

feature_large = ZZFeatureMap(
    feature_dimension=3,
    reps=1
)

print(feature_large.draw())

print()

print(feature_large.parameters)

print()

print("Input Parameter 수 :", feature_large.num_parameters)

print()


# =========================================================
# STEP 15. QNN 구성 요소 정리
# =========================================================

print("=" * 70)
print("STEP 15. QNN 구성 요소")
print("=" * 70)

print("""
Input Data
      │
      ▼
Input Parameter
      │
      ▼
Feature Map
      │
      ▼
Encoded Quantum State
      │
      ▼
Weight Parameter
      │
      ▼
Ansatz
      │
      ▼
Quantum State
      │
      ▼
Observable
      │
      ▼
Expectation Value
""")

print()

print("=" * 70)
print("실습 완료")
print("=" * 70)

print()

