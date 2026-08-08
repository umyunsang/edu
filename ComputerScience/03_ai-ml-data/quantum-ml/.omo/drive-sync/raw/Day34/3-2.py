"""
=========================================================
Lab 02

02_create_qnn.py (Part 1)

EstimatorQNN 생성

STEP 1 ~ STEP 6

=========================================================

실습 목표

1. Feature Map 생성

2. Ansatz 생성

3. Quantum Circuit 생성

4. Observable 생성

EstimatorQNN을 구성하기 위한
기본 요소를 단계적으로 생성한다.

=========================================================
"""

# =========================================================
# STEP 1. Import
# =========================================================

print("=" * 70)
print("STEP 1. Import")
print("=" * 70)

import random
import numpy as np

from qiskit import QuantumCircuit

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit.quantum_info import SparsePauliOp

print("라이브러리 Import 완료")
print()


# =========================================================
# STEP 2. Random Seed
# =========================================================

print("=" * 70)
print("STEP 2. Random Seed")
print("=" * 70)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print(f"Random Seed : {SEED}")
print()

print("Random Seed 설정 완료")
print()


# =========================================================
# STEP 3. Feature Map 생성
# =========================================================

print("=" * 70)
print("STEP 3. Feature Map 생성")
print("=" * 70)

feature_map = ZZFeatureMap(

    feature_dimension=2,

    reps=1

)

print("Feature Map 생성 완료")
print()

print(feature_map)

print()

print("-" * 70)
print("Feature Map 정보")
print("-" * 70)

print("Feature Dimension :", feature_map.feature_dimension)

print("Parameter 개수    :", len(feature_map.parameters))

print("Parameter 목록")

for idx, parameter in enumerate(feature_map.parameters):

    print(f"  [{idx}] {parameter}")

print()

print("Draw")

print(feature_map.draw("text"))

print()


# =========================================================
# STEP 4. Ansatz 생성
# =========================================================

print("=" * 70)
print("STEP 4. Ansatz 생성")
print("=" * 70)

ansatz = RealAmplitudes(

    num_qubits=2,

    reps=1

)

print("Ansatz 생성 완료")
print()

print(ansatz)

print()

print("-" * 70)
print("Ansatz 정보")
print("-" * 70)

print("Qubit 수        :", ansatz.num_qubits)

print("Parameter 개수  :", len(ansatz.parameters))

print("Parameter 목록")

for idx, parameter in enumerate(ansatz.parameters):

    print(f"  [{idx}] {parameter}")

print()

print("Draw")

print(ansatz.draw("text"))

print()


# =========================================================
# STEP 5. Quantum Circuit 생성
# =========================================================

print("=" * 70)
print("STEP 5. Quantum Circuit 생성")
print("=" * 70)

circuit = QuantumCircuit(2)

circuit.compose(

    feature_map,

    inplace=True

)

circuit.compose(

    ansatz,

    inplace=True

)

print("Quantum Circuit 생성 완료")
print()

print(circuit)

print()

print("-" * 70)
print("Quantum Circuit 정보")
print("-" * 70)

print("Qubit 수")

print(circuit.num_qubits)

print()

print("Parameter 개수")

print(len(circuit.parameters))

print()

print("전체 Parameter")

for idx, parameter in enumerate(circuit.parameters):

    print(f"  [{idx}] {parameter}")

print()

print("Circuit Draw")

print(circuit.draw("text"))

print()


# =========================================================
# STEP 6. Observable 생성
# =========================================================

print("=" * 70)
print("STEP 6. Observable 생성")
print("=" * 70)

observable = SparsePauliOp.from_list(

    [

        ("ZZ", 1.0)

    ]

)

print("Observable 생성 완료")
print()

print(observable)

print()

print("-" * 70)
print("Observable 정보")
print("-" * 70)

print("Pauli Operator")

print(observable.paulis)

print()

print("Coefficient")

print(observable.coeffs)

print()

print("=" * 70)
print("Part 1 완료")
print("=" * 70)



# =========================================================
# STEP 7. StatevectorEstimator 생성
# =========================================================

print("=" * 70)
print("STEP 7. StatevectorEstimator 생성")
print("=" * 70)

from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()

print("StatevectorEstimator 생성 완료")
print()

print(estimator)

print()


# =========================================================
# STEP 8. EstimatorQNN 생성
# =========================================================

print("=" * 70)
print("STEP 8. EstimatorQNN 생성")
print("=" * 70)

from qiskit_machine_learning.neural_networks import EstimatorQNN

qnn = EstimatorQNN(

    circuit=circuit,

    estimator=estimator,

    observables=observable,

    input_params=feature_map.parameters,

    weight_params=ansatz.parameters

)

print("EstimatorQNN 생성 완료")

print()

print(qnn)

print()


# =========================================================
# STEP 9. QNN 정보 확인
# =========================================================

print("=" * 70)
print("STEP 9. QNN 정보 확인")
print("=" * 70)

print("Input 수")

print(qnn.num_inputs)

print()

print("Weight 수")

print(qnn.num_weights)

print()

print("Output Shape")

print(qnn.output_shape)

print()

print("Input Parameters")

for i, p in enumerate(feature_map.parameters):

    print(f"[{i}] {p}")

print()

print("Weight Parameters")

for i, p in enumerate(ansatz.parameters):

    print(f"[{i}] {p}")

print()

print("Circuit")

print(qnn.circuit.draw("text"))

print()


# =========================================================
# STEP 10. Forward Test
# =========================================================

print("=" * 70)
print("STEP 10. Forward Test")
print("=" * 70)

import numpy as np

# ------------------------------------------
# 입력 데이터
# ------------------------------------------

input_data = np.array(

    [0.20, 0.70]

)

print("Input")

print(input_data)

print()

# ------------------------------------------
# Weight 생성
# ------------------------------------------

weight = np.random.rand(

    qnn.num_weights

)

print("Weight")

print(weight)

print()

# ------------------------------------------
# Forward
# ------------------------------------------

output = qnn.forward(

    input_data=input_data,

    weights=weight

)

print("Forward 완료")

print()

print("Output")

print(output)

print()

print("Output Shape")

print(output.shape)

print()

print("Output Type")

print(type(output))

print()

# ------------------------------------------
# 입력 변경
# ------------------------------------------

print("-" * 70)
print("입력 변경 실험")
print("-" * 70)

input2 = np.array(

    [0.80, 0.10]

)

output2 = qnn.forward(

    input_data=input2,

    weights=weight

)

print("Input 2")

print(input2)

print()

print("Output 2")

print(output2)

print()

print("Output Difference")

print(output2 - output)

print()




print("=" * 70)
print("Lab 02 완료")
print("=" * 70)
