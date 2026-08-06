"""
=========================================================
 Lab 3. EstimatorQNN 생성
---------------------------------------------------------
학습 목표

1. EstimatorQNN 생성
2. Constructor 이해
3. QNN 구조 확인
4. 입력/출력 구조 분석
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

from qiskit_machine_learning.neural_networks import (
    EstimatorQNN,
)

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
print("STEP 2. Quantum Circuit 생성")
print("=" * 60)

print(circuit)

print()


# =========================================================
# STEP 3. Parameter 준비
# =========================================================

input_params = list(feature_map.parameters)

weight_params = list(ansatz.parameters)

print("=" * 60)
print("STEP 3. Parameter 준비")
print("=" * 60)

print(f"Input Parameter  : {len(input_params)}")

print(f"Weight Parameter : {len(weight_params)}")

print()


# =========================================================
# STEP 4. Observable 생성
# =========================================================

observable = SparsePauliOp.from_list(
    [
        ("ZZ", 1.0)
    ]
)

print("=" * 60)
print("STEP 4. Observable 생성")
print("=" * 60)

print(observable)

print()


# =========================================================
# STEP 5. Estimator 생성
# =========================================================

estimator = StatevectorEstimator()

print("=" * 60)
print("STEP 5. StatevectorEstimator")
print("=" * 60)

print(estimator)

print()


# =========================================================
# STEP 6. EstimatorQNN 생성
# =========================================================

qnn = EstimatorQNN(

    circuit=circuit,

    estimator=estimator,

    observables=observable,

    input_params=input_params,

    weight_params=   weight_params,

    input_gradients= False,

)

print("=" * 60)
print("STEP 6. EstimatorQNN 생성")
print("=" * 60)

print("EstimatorQNN 생성 완료\n")


# =========================================================
# STEP 7. 객체 출력
# =========================================================

print("=" * 60)
print("STEP 7. EstimatorQNN 출력")
print("=" * 60)

print(qnn)

print()


# =========================================================
# STEP 8. 타입 확인
# =========================================================

print("=" * 60)
print("STEP 8. 객체 타입 확인")
print("=" * 60)

print(type(qnn))

print()


# =========================================================
# STEP 9. QNN 구조 확인
# =========================================================

print("=" * 60)
print("STEP 9. QNN 구조 확인")
print("=" * 60)

print(f"Input 수       : {qnn.num_inputs}")

print(f"Weight 수      : {qnn.num_weights}")

print(f"Output Shape   : {qnn.output_shape}")

print()


# =========================================================
# STEP 10. Circuit 정보 확인
# =========================================================

print("=" * 60)
print("STEP 10. Circuit 정보")
print("=" * 60)

print(f"Qubit 수       : {circuit.num_qubits}")

print(f"Parameter 수   : {circuit.num_parameters}")

print()


# =========================================================
# STEP 11. Parameter 확인
# =========================================================

print("=" * 60)
print("STEP 11. Parameter 확인")
print("=" * 60)

print("Input Parameters")

for p in input_params:

    print(" ", p)

print()

print("Weight Parameters")

for p in weight_params:

    print(" ", p)

print()


# =========================================================
# STEP 12. Observable 확인
# =========================================================

print("=" * 60)
print("STEP 12. Observable 확인")
print("=" * 60)

print(observable)

print()


# =========================================================
# STEP 13. Estimator 확인
# =========================================================

print("=" * 60)
print("STEP 13. Estimator 확인")
print("=" * 60)

print(estimator)

print()


# =========================================================
# STEP 14. EstimatorQNN Summary
# =========================================================

print("=" * 60)
print("EstimatorQNN Summary")
print("=" * 60)

print(f"""
EstimatorQNN 생성 성공

Input Dimension
------------------------
{qnn.num_inputs}

Weight 개수
------------------------
{qnn.num_weights}

Output Shape
------------------------
{qnn.output_shape}

Circuit
------------------------
Qubit 수 : {circuit.num_qubits}

Observable
------------------------
{observable}

Estimator
------------------------
{type(estimator).__name__}
""")

print("=" * 60)
print("QNN 내부 정보")
print("=" * 60)

print(f"Input Dimension : {qnn.num_inputs}")
print(f"Weight Count    : {qnn.num_weights}")
print(f"Output Shape    : {qnn.output_shape}")
print(f"Sparse          : {qnn.sparse}")
print(f"Input Gradients : {qnn.input_gradients}")



print("=" * 60)
print("Lab 3 완료")
print("=" * 60)