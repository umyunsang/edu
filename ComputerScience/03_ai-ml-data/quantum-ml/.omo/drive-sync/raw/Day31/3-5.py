"""
=========================================================
 Lab 5. Forward 결과 분석
---------------------------------------------------------
학습 목표

1. Input 변화에 따른 Output 분석
2. Weight 변화에 따른 Output 분석
3. Batch Forward 수행
4. EstimatorQNN 출력 특성 이해
=========================================================
"""

# =========================================================
# STEP 1. Library Import
# =========================================================

import numpy as np

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


# =========================================================
# STEP 2. EstimatorQNN 생성
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

observable = SparsePauliOp.from_list(
    [("ZZ", 1.0)]
)

estimator = StatevectorEstimator()

qnn = EstimatorQNN(

    circuit=circuit,

    estimator=estimator,

    observables=observable,

    input_params=feature_map.parameters,

    weight_params=ansatz.parameters

)

print("EstimatorQNN 생성 완료\n")


# =========================================================
# STEP 3. Weight 고정
# =========================================================

weights = np.zeros(
    qnn.num_weights
)

print("=" * 60)
print("STEP 3. Weight 고정")
print("=" * 60)

print(weights)

print()


# =========================================================
# STEP 4. Input 변경 실험
# =========================================================

print("=" * 60)
print("STEP 4. Input 변경")
print("=" * 60)

inputs = [

    np.array([0.1, 0.1]),

    np.array([0.3, 0.5]),

    np.array([0.5, 0.7]),

    np.array([0.9, 0.2])

]

print(f"{'Input':<20} {'Output':>12}")

print("-" * 40)

for x in inputs:

    output = qnn.forward(

        input_data=x,

        weights=weights

    )

    print(
        f"{str(x):<20} "
        f"{output.item():>12.6f}"
    )

print()


# =========================================================
# STEP 5. 결과 분석
# =========================================================

print("=" * 60)
print("STEP 5. 분석")
print("=" * 60)

print("""
같은 Weight를 사용해도

입력 데이터가 바뀌면

Expectation Value가 달라진다.

→ Feature Map이 다른 Quantum State를 생성하기 때문이다.
""")

print()


# =========================================================
# STEP 6. Input 고정
# =========================================================

input_data = np.array(
    [0.2, 0.8]
)

print("=" * 60)
print("STEP 6. Input 고정")
print("=" * 60)

print(input_data)

print()


# =========================================================
# STEP 7. Weight 변경
# =========================================================

print("=" * 60)
print("STEP 7. Weight 변경")
print("=" * 60)

weight_list = [

    np.zeros(
        qnn.num_weights
    ),

    np.ones(
        qnn.num_weights
    ) * 0.2,

    np.ones(
        qnn.num_weights
    ) * (-0.2),

    np.random.rand(
        qnn.num_weights
    )

]

print(f"{'Weight':<35} {'Output':>12}")

print("-" * 55)

for w in weight_list:

    output = qnn.forward(

        input_data=input_data,

        weights=w

    )

    print(

        f"{np.round(w,3)}"

        f" {output.item():>12.6f}"

    )

print()


# =========================================================
# STEP 8. 결과 분석
# =========================================================

print("=" * 60)
print("STEP 8. 분석")
print("=" * 60)

print("""
같은 입력이라도

Weight가 달라지면

Expectation Value가 달라진다.

→ Ansatz가 Quantum State를 변경하기 때문이다.
""")

print()


# =========================================================
# STEP 9. Batch Input 생성
# =========================================================

batch_input = np.array(

    [

        [0.1, 0.1],

        [0.3, 0.5],

        [0.5, 0.7],

        [0.9, 0.2]

    ]

)

print("=" * 60)
print("STEP 9. Batch Input")
print("=" * 60)

print(batch_input)

print()


# =========================================================
# STEP 10. Batch Forward
# =========================================================

batch_output = qnn.forward(

    input_data=batch_input,

    weights=np.zeros(
        qnn.num_weights
    )

)

print("=" * 60)
print("STEP 10. Batch Forward")
print("=" * 60)

print(batch_output)

print()

print(batch_output.shape)

print()


# =========================================================
# STEP 11. Batch 결과 출력
# =========================================================

print("=" * 60)
print("STEP 11. Batch 결과")
print("=" * 60)

print(f"{'Sample':<8} {'Output':>12}")

print("-" * 25)

for i, value in enumerate(batch_output):

    print(

        f"{i:<8}"

        f"{value.item():>12.6f}"

    )

print()


print("=" * 60)
print("Lab 5 완료")
print("=" * 60)