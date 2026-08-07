"""
=========================================================
 Lab 4. Forward Pass 구현
---------------------------------------------------------
학습 목표

1. Input Data 준비
2. Weight 준비
3. Forward Pass 수행
4. Expectation Value 확인
5. Output Shape 분석
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

print("Library Import 완료\n")


# =========================================================
# STEP 2. Quantum Circuit 생성
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

input_params = list(feature_map.parameters)

weight_params = list(ansatz.parameters)

observable = SparsePauliOp.from_list(
    [
        ("ZZ", 1.0)
    ]
)

estimator = StatevectorEstimator()


# =========================================================
# STEP 3. EstimatorQNN 생성
# =========================================================

qnn = EstimatorQNN(

    circuit=circuit,

    estimator=estimator,

    observables=observable,

    input_params=input_params,

    weight_params=weight_params

)

print("=" * 60)
print("STEP 3. EstimatorQNN 생성")
print("=" * 60)

print("EstimatorQNN 생성 완료\n")


# =========================================================
# STEP 4. QNN 구조 확인
# =========================================================

print("=" * 60)
print("STEP 4. QNN 구조")
print("=" * 60)

print(f"Input 수     : {qnn.num_inputs}")

print(f"Weight 수    : {qnn.num_weights}")

print(f"Output Shape : {qnn.output_shape}")

print()


# =========================================================
# STEP 5. Input Data 준비
# =========================================================

input_data = np.array(
    [
        0.2,
        0.8,
    ]
)

print("=" * 60)
print("STEP 5. Input Data")
print("=" * 60)

print(input_data)

print()

print("Input Shape")

print(input_data.shape)

print()


# =========================================================
# STEP 6. Weight 준비
# =========================================================

weights = np.zeros(
    qnn.num_weights
)

print("=" * 60)
print("STEP 6. Weight")
print("=" * 60)

print(weights)

print()

print("Weight Shape")

print(weights.shape)

print()


# =========================================================
# STEP 7. Shape 확인
# =========================================================

print("=" * 60)
print("STEP 7. Shape Check")
print("=" * 60)

assert len(input_data) == qnn.num_inputs

assert len(weights) == qnn.num_weights

print("Input Shape 정상")

print("Weight Shape 정상")

print()


# =========================================================
# STEP 8. Forward Pass
# =========================================================

output = qnn.forward(

    input_data=input_data,

    weights=weights

)

print("=" * 60)
print("STEP 8. Forward Pass")
print("=" * 60)

print("Forward 성공\n")


# =========================================================
# STEP 9. Output 확인
# =========================================================

print("=" * 60)
print("STEP 9. Output")
print("=" * 60)

print(output)

print()

print("Output Shape")

print(output.shape)

print()


# =========================================================
# STEP 10. Expectation Value
# =========================================================

value = output.item()

print("=" * 60)
print("STEP 10. Expectation Value")
print("=" * 60)

print(f"{value:.6f}")

print()


# =========================================================
# STEP 11. Prediction 예제
# =========================================================

prediction = 1 if value >= 0 else 0

print("=" * 60)
print("STEP 11. Prediction")
print("=" * 60)

print(f"Prediction : {prediction}")

print()


# =========================================================
# STEP 12. Forward Summary
# =========================================================

print("=" * 60)
print("Forward Summary")
print("=" * 60)

print(f"""
Input Data
------------------------
{input_data}

Weights
------------------------
{weights}

Output Shape
------------------------
{output.shape}

Expectation Value
------------------------
{value:.6f}

Prediction
------------------------
{prediction}
""")

print("=" * 60)
print("Lab 4 완료")
print("=" * 60)