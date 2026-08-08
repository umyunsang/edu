"""
=========================================================
Lab 03

03_torch_connector.py (Part 1)

TorchConnector 적용

=========================================================

STEP 1 ~ STEP 5

STEP 1. Import
STEP 2. EstimatorQNN 생성
STEP 3. TorchConnector 적용
STEP 4. nn.Module 확인
STEP 5. Model 구조 출력

=========================================================

실습 목표

1. EstimatorQNN 생성

2. TorchConnector 적용

3. PyTorch Module 변환 확인

4. Quantum Layer 생성

5. Model 구조 확인

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
import torch
import torch.nn as nn

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

print("Import 완료")
print()

# =========================================================
# STEP 2. Random Seed
# =========================================================

print("=" * 70)
print("Random Seed")
print("=" * 70)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Seed : {SEED}")
print()

# =========================================================
# STEP 3. Feature Map
# =========================================================

print("=" * 70)
print("Feature Map")
print("=" * 70)

feature_map = ZZFeatureMap(

    feature_dimension=2,

    reps=1

)

print(feature_map.draw("text"))

print()

# =========================================================
# STEP 4. Ansatz
# =========================================================

print("=" * 70)
print("Ansatz")
print("=" * 70)

ansatz = RealAmplitudes(

    num_qubits=2,

    reps=1

)

print(ansatz.draw("text"))

print()

# =========================================================
# STEP 5. Quantum Circuit
# =========================================================

print("=" * 70)
print("Quantum Circuit")
print("=" * 70)

circuit = feature_map.compose(ansatz)

print(circuit.draw("text"))

print()

# =========================================================
# STEP 6. Observable
# =========================================================

print("=" * 70)
print("Observable")
print("=" * 70)

observable = SparsePauliOp.from_list(

    [

        ("ZZ",1.0)

    ]

)

print(observable)

print()

# =========================================================
# STEP 7. StatevectorEstimator
# =========================================================

print("=" * 70)
print("StatevectorEstimator")
print("=" * 70)

estimator = StatevectorEstimator()

print(estimator)

print()

# =========================================================
# STEP 8. EstimatorQNN 생성
# =========================================================

print("=" * 70)
print("STEP 2. EstimatorQNN 생성")
print("=" * 70)

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

print("-" * 70)
print("QNN 정보")
print("-" * 70)

print("Input")

print(qnn.num_inputs)

print()

print("Weight")

print(qnn.num_weights)

print()

print("Output Shape")

print(qnn.output_shape)

print()

# =========================================================
# STEP 9. TorchConnector 적용
# =========================================================

print("=" * 70)
print("STEP 3. TorchConnector 적용")
print("=" * 70)

model = TorchConnector(qnn)

print("TorchConnector 생성 완료")

print()

print(model)

print()

# =========================================================
# STEP 10. nn.Module 확인
# =========================================================

print("=" * 70)
print("STEP 4. nn.Module 확인")
print("=" * 70)

print("Model Type")

print(type(model))

print()

print("nn.Module 여부")

print(isinstance(model, nn.Module))

print()

print("Module Name")

print(model.__class__.__name__)

print()


# =========================================================
# STEP 11. Model 구조 출력
# =========================================================

print("=" * 70)
print("STEP 5. Model 구조 출력")
print("=" * 70)

print(model)

print()

print("-" * 70)
print("Model 정보")
print("-" * 70)

print("Class")

print(model.__class__)

print()

print("Training Mode")

print(model.training)

print()

print("Device")

print(next(model.parameters()).device)

print()

print("=" * 70)
print("Part 1 완료")
print("=" * 70)


"""
=========================================================
Lab 03

03_torch_connector.py (Part 2)

TorchConnector 적용

=========================================================

STEP 6 ~ STEP 10

STEP 6. Weight 확인
STEP 7. Parameter 확인
STEP 8. Initial Weight 지정
STEP 9. Model 검증
STEP 10. 다음 실습 안내


"""

import torch
import torch.nn as nn

# =========================================================
# STEP 6. Weight 확인
# =========================================================

print("=" * 70)
print("STEP 6. Weight 확인")
print("=" * 70)

print("Quantum Layer Weight")

print(model.weight)

print()

print("Weight Shape")

print(model.weight.shape)

print()

print("Weight Type")

print(type(model.weight))

print()

print("requires_grad")

print(model.weight.requires_grad)

print()

print("""
TorchConnector는

EstimatorQNN의 Weight를

torch.nn.Parameter 형태로

자동 생성합니다.

""")

print()

# =========================================================
# STEP 7. Parameter 확인
# =========================================================

print("=" * 70)
print("STEP 7. Parameter 확인")
print("=" * 70)

print("model.parameters()")

print()

parameter_count = 0

total_parameter = 0

for parameter in model.parameters():

    parameter_count += 1

    total_parameter += parameter.numel()

    print(parameter)

    print()

print("-" * 70)

print("Parameter 개수")

print(parameter_count)

print()

print("전체 Parameter 수")

print(total_parameter)

print()

print("=" * 70)

print("named_parameters()")

print("=" * 70)

for name, parameter in model.named_parameters():

    print("Name")

    print(name)

    print()

    print("Shape")

    print(parameter.shape)

    print()

    print("Value")

    print(parameter)

    print()

    print("requires_grad")

    print(parameter.requires_grad)

    print()

# =========================================================
# STEP 8. Initial Weight 지정
# =========================================================

print("=" * 70)
print("STEP 8. Initial Weight 지정")
print("=" * 70)

print("기존 Weight")

print(model.weight)

print()

custom_weight = torch.tensor(

    [0.10, 0.20, 0.30, 0.40],

    dtype=torch.float32

)

print("사용자 지정 Weight")

print(custom_weight)

print()

with torch.no_grad():

    model.weight.copy_(custom_weight)

print("변경 후 Weight")

print(model.weight)

print()

print("""
Weight를 직접 지정하면

Forward 결과도

달라질 수 있습니다.

""")

print()

# =========================================================
# STEP 9. Model 검증
# =========================================================

print("=" * 70)
print("STEP 9. Model 검증")
print("=" * 70)

print("Model Type")

print(type(model))

print()

print("nn.Module 여부")

print(isinstance(model, nn.Module))

print()

print("Training Mode")

print(model.training)

print()

print("Weight 존재 여부")

print(hasattr(model, "weight"))

print()

print("Parameter 존재 여부")

print(len(list(model.parameters())) > 0)

print()

print("Weight Shape")

print(model.weight.shape)

print()

print("Model Ready")

print("PASS")

print()


