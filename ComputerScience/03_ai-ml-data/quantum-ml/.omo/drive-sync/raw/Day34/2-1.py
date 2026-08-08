"""
===========================================================
Lab.
TorchConnector를 이용한 PyTorch 연동 구조 이해
===========================================================

실습 목표

1. EstimatorQNN 생성
2. TorchConnector 생성
3. PyTorch Model 변환 확인
4. Tensor 입력
5. Forward 수행
6. Output 확인
7. Parameter 확인

===========================================================
"""

############################################################
# STEP 1. Library Import
############################################################

import torch

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit.primitives import StatevectorEstimator

from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

print("=" * 70)
print("STEP 1. Import Complete")
print("=" * 70)


############################################################
# STEP 2. Feature Map 생성
############################################################

NUM_QUBITS = 2

feature_map = ZZFeatureMap(
    feature_dimension=NUM_QUBITS,
    reps=1
)

print()
print("=" * 70)
print("STEP 2. Feature Map")
print("=" * 70)

print(feature_map)


############################################################
# STEP 3. Variational Circuit 생성
############################################################

ansatz = RealAmplitudes(

    num_qubits=NUM_QUBITS,

    reps=1

)

print()
print("=" * 70)
print("STEP 3. Ansatz")
print("=" * 70)

print(ansatz)


############################################################
# STEP 4. Quantum Circuit 생성
############################################################

qc = QuantumCircuit(NUM_QUBITS)

qc.compose(feature_map, inplace=True)

qc.compose(ansatz, inplace=True)

print()
print("=" * 70)
print("STEP 4. Quantum Circuit")
print("=" * 70)

print(qc.draw())


############################################################
# STEP 5. Estimator 생성
############################################################

estimator = StatevectorEstimator()

print()
print("=" * 70)
print("STEP 5. StatevectorEstimator")
print("=" * 70)

print(estimator)


############################################################
# STEP 6. Observable 생성
############################################################

observable = SparsePauliOp.from_list(

    [("ZZ", 1)]

)

print()
print("=" * 70)
print("STEP 6. Observable")
print("=" * 70)

print(observable)


############################################################
# STEP 7. EstimatorQNN 생성
############################################################

qnn = EstimatorQNN(

    circuit=qc,

    estimator=estimator,

    observables=observable,

    input_params=feature_map.parameters,

    weight_params=ansatz.parameters

)

print()
print("=" * 70)
print("STEP 7. EstimatorQNN")
print("=" * 70)

print(qnn)


############################################################
# STEP 8. EstimatorQNN 정보 확인
############################################################

print()
print("=" * 70)
print("STEP 8. QNN Information")
print("=" * 70)

print("Number of Inputs")

print(qnn.num_inputs)

print()

print("Number of Weights")

print(qnn.num_weights)

print()

print("Output Shape")

print(qnn.output_shape)


############################################################
# STEP 9. TorchConnector 생성
############################################################

model = TorchConnector(qnn)

print()
print("=" * 70)
print("STEP 9. TorchConnector")
print("=" * 70)

print(model)


############################################################
# STEP 10. PyTorch Model 확인
############################################################

print()
print("=" * 70)
print("STEP 10. PyTorch Model")
print("=" * 70)

print("Model Type")

print(type(model))

print()

print(model)


############################################################
# STEP 11. Input Tensor 생성
############################################################

x = torch.tensor(

    [[0.25, 0.80]],

    dtype=torch.float32

)

print()
print("=" * 70)
print("STEP 11. Input Tensor")
print("=" * 70)

print(x)

print()

print("Shape")

print(x.shape)


############################################################
# STEP 12. Forward 수행
############################################################

output = model(x)

print()
print("=" * 70)
print("STEP 12. Forward")
print("=" * 70)

print(output)


############################################################
# STEP 13. Output 확인
############################################################

print()
print("=" * 70)
print("STEP 13. Output")
print("=" * 70)

print("Output Tensor")

print(output)

print()

print("Output Shape")

print(output.shape)


############################################################
# STEP 14. Parameter 확인
############################################################

print()
print("=" * 70)
print("STEP 14. Quantum Parameters")
print("=" * 70)

for idx, parameter in enumerate(model.parameters()):

    print(f"Parameter {idx}")

    print(parameter)

    print()


############################################################
# STEP 15. Forward 구조 분석
############################################################

print()
print("=" * 70)
print("STEP 15. Forward Flow")
print("=" * 70)

print("""
Input Tensor
      │
      ▼
TorchConnector
      │
      ▼
EstimatorQNN
      │
      ▼
Quantum Circuit
      │
      ▼
Observable
      │
      ▼
Expectation Value
      │
      ▼
Output Tensor
""")

