"""
=========================================================
 Lab14-04_check_weight.py (Part 1)

 Lab. TorchConnector를 이용한 Quantum Layer 실행

 STEP 4. Weight 및 Parameter 확인 (Part 1)

=========================================================

[실습 목표]

1. EstimatorQNN 생성
2. TorchConnector 생성
3. Quantum Weight 확인
4. Parameter 확인
5. requires_grad 확인

=========================================================
"""

# =========================================================
# STEP 1. Library Import
# =========================================================

import random
import numpy as np
import torch
import torch.nn as nn

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# =========================================================
# STEP 2. Random Seed
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("STEP 2. Random Seed")
print("=" * 70)

print("Seed :", SEED)
print()

# =========================================================
# STEP 3. Feature Map
# =========================================================

print("=" * 70)
print("STEP 3. Feature Map")
print("=" * 70)

feature_map = ZZFeatureMap(
    feature_dimension=2,
    reps=1
)

print(feature_map)
print()

# =========================================================
# STEP 4. Ansatz
# =========================================================

print("=" * 70)
print("STEP 4. Ansatz")
print("=" * 70)

ansatz = RealAmplitudes(
    num_qubits=2,
    reps=1
)

print(ansatz)
print()

# =========================================================
# STEP 5. Quantum Circuit
# =========================================================

print("=" * 70)
print("STEP 5. Quantum Circuit")
print("=" * 70)

circuit = feature_map.compose(ansatz)

print(circuit)
print()

# =========================================================
# STEP 6. Observable
# =========================================================

print("=" * 70)
print("STEP 6. Observable")
print("=" * 70)

observable = SparsePauliOp.from_list(
    [
        ("ZZ", 1)
    ]
)

print(observable)
print()

# =========================================================
# STEP 7. Estimator 생성
# =========================================================

print("=" * 70)
print("STEP 7. StatevectorEstimator")
print("=" * 70)

estimator = StatevectorEstimator()

print(estimator)
print()

# =========================================================
# STEP 8. EstimatorQNN 생성
# =========================================================

print("=" * 70)
print("STEP 8. EstimatorQNN")
print("=" * 70)

qnn = EstimatorQNN(
    circuit=circuit,
    estimator=estimator,
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    input_gradients=True
)

print("EstimatorQNN 생성 완료")
print()

# =========================================================
# STEP 9. TorchConnector 생성
# =========================================================

print("=" * 70)
print("STEP 9. TorchConnector")
print("=" * 70)

model = TorchConnector(qnn)

print(model)
print()

# =========================================================
# STEP 10. Weight 확인
# =========================================================

print("=" * 70)
print("STEP 10. Weight")
print("=" * 70)

print("model.weight")
print()

print(model.weight)
print()

print("Type")
print(type(model.weight))
print()

print("Shape")
print(model.weight.shape)
print()

# =========================================================
# STEP 11. Parameter 확인
# =========================================================

print("=" * 70)
print("STEP 11. Parameters")
print("=" * 70)

parameters = list(model.parameters())

print("Parameter 개수")

print(len(parameters))
print()

for idx, param in enumerate(parameters):

    print("-" * 50)

    print(f"Parameter {idx}")

    print()

    print(param)

    print()

    print("Shape")

    print(param.shape)

    print()

    print("dtype")

    print(param.dtype)

    print()

# =========================================================
# STEP 12. Named Parameter 확인
# =========================================================

print("=" * 70)
print("STEP 12. Named Parameters")
print("=" * 70)

for name, param in model.named_parameters():

    print(f"Name : {name}")

    print("Tensor")

    print(param)

    print()

# =========================================================
# STEP 13. requires_grad 확인
# =========================================================

print("=" * 70)
print("STEP 13. requires_grad")
print("=" * 70)

for name, param in model.named_parameters():

    print(f"{name}")

    print("requires_grad :", param.requires_grad)

    print()


"""
=========================================================
 Lab14-04_check_weight.py (Part 2)

 STEP 7 ~ STEP 12

 Quantum Weight 분석

=========================================================
"""

import torch
import numpy as np

print("=" * 70)
print("STEP 7. Weight Shape 분석")
print("=" * 70)

print("Weight Tensor")

print(model.weight)

print()

print("Tensor Shape")

print(model.weight.shape)

print()

print("Dimension")

print(model.weight.ndim)

print()

print("Element Count")

print(model.weight.numel())

print()

# =====================================================
# STEP 8
# Weight -> NumPy
# =====================================================

print("=" * 70)
print("STEP 8. NumPy 변환")
print("=" * 70)

weight_numpy = model.weight.detach().numpy()

print(weight_numpy)

print()

print(type(weight_numpy))

print()

# =====================================================
# STEP 9
# Weight Statistics
# =====================================================

print("=" * 70)
print("STEP 9. Weight 통계")
print("=" * 70)

print("Mean")

print(weight_numpy.mean())

print()

print("Std")

print(weight_numpy.std())

print()

print("Minimum")

print(weight_numpy.min())

print()

print("Maximum")

print(weight_numpy.max())

print()

# =====================================================
# STEP 10
# Weight Backup
# =====================================================

print("=" * 70)
print("STEP 10. Weight Backup")
print("=" * 70)

backup_weight = model.weight.detach().clone()

print("Backup Complete")

print()

print(backup_weight)

print()

# =====================================================
# STEP 11
# Initial Weight 변경
# =====================================================

print("=" * 70)
print("STEP 11. Weight 변경")
print("=" * 70)

print("Before")

print(model.weight)

print()

with torch.no_grad():

    model.weight.fill_(0.5)

print("After")

print(model.weight)

print()

# =====================================================
# STEP 12
# Weight Restore
# =====================================================

print("=" * 70)
print("STEP 12. Weight Restore")
print("=" * 70)

with torch.no_grad():

    model.weight.copy_(backup_weight)

print("Restore Complete")

print()

print(model.weight)

print()

# =====================================================
# Optimizer가 사용하는 Parameter 확인
# =====================================================

print("=" * 70)
print("Optimizer Parameter")
print("=" * 70)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.01

)

for idx, group in enumerate(optimizer.param_groups):

    print(f"Group {idx}")

    print()

    for param in group["params"]:

        print(param.shape)

        print(param.requires_grad)

        print()

