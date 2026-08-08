"""
=========================================================
 Lab14-01_environment.py

 Lab. TorchConnector를 이용한 Quantum Layer 실행

 STEP 1. 실습 환경 준비
=========================================================

[실습 목표]

- PyTorch 라이브러리 Import
- Qiskit Machine Learning Import
- TorchConnector Import
- Random Seed 설정
- 실행 환경 확인


=========================================================
"""

# =========================================================
# STEP 1. Python 기본 라이브러리
# =========================================================

import random
import numpy as np

print("=" * 60)
print("STEP 1. Python Library Import")
print("=" * 60)

print("Python Library Import 완료")
print()

# =========================================================
# STEP 2. PyTorch Import
# =========================================================

import torch
import torch.nn as nn

print("=" * 60)
print("STEP 2. PyTorch Import")
print("=" * 60)

print("PyTorch Version")
print(torch.__version__)
print()

# =========================================================
# STEP 3. Qiskit Import
# =========================================================

import qiskit

print("=" * 60)
print("STEP 3. Qiskit Import")
print("=" * 60)

print("Qiskit Version")
print(qiskit.__version__)
print()

# =========================================================
# STEP 4. Qiskit Machine Learning Import
# =========================================================

import qiskit_machine_learning

print("=" * 60)
print("STEP 4. Qiskit Machine Learning")
print("=" * 60)

print("Qiskit Machine Learning Version")
print(qiskit_machine_learning.__version__)
print()

# =========================================================
# STEP 5. 필요한 클래스 Import
# =========================================================

print("=" * 60)
print("STEP 5. 필요한 클래스 Import")
print("=" * 60)

# Feature Map
from qiskit.circuit.library import ZZFeatureMap

# Ansatz
from qiskit.circuit.library import RealAmplitudes

# Primitive
from qiskit.primitives import StatevectorEstimator

# Observable
from qiskit.quantum_info import SparsePauliOp

# EstimatorQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN

# TorchConnector
from qiskit_machine_learning.connectors import TorchConnector

print("Import 완료")
print()

# =========================================================
# STEP 6. Random Seed 설정
# =========================================================

print("=" * 60)
print("STEP 6. Random Seed")
print("=" * 60)

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

print(f"Seed : {SEED}")
print()

# =========================================================
# STEP 7. CUDA 확인
# =========================================================

print("=" * 60)
print("STEP 7. 실행 장치(Device)")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device :", device)
print()

# =========================================================
# STEP 8. Tensor 생성 테스트
# =========================================================

print("=" * 60)
print("STEP 8. Tensor 생성")
print("=" * 60)

x = torch.tensor(
    [0.1, 0.2],
    dtype=torch.float32
)

print("Tensor")
print(x)

print()

print("Shape")

print(x.shape)

print()

print("dtype")

print(x.dtype)

print()

# =========================================================
# STEP 9. nn.Module 확인
# =========================================================

print("=" * 60)
print("STEP 9. nn.Module 확인")
print("=" * 60)


class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


dummy = DummyModel()

print(dummy)

print()

print("PyTorch Module 여부")

print(isinstance(dummy, nn.Module))

print()

# =========================================================
# STEP 10. TorchConnector Import 확인
# =========================================================

print("=" * 60)
print("STEP 10. TorchConnector 확인")
print("=" * 60)

print(TorchConnector)

print()

# =========================================================
# STEP 11. 실습 환경 확인
# =========================================================

print("=" * 60)
print("STEP 11. 환경 점검")
print("=" * 60)

print("Python Library       : OK")
print("PyTorch              : OK")
print("Qiskit               : OK")
print("Qiskit ML            : OK")
print("TorchConnector       : OK")
print("Random Seed          : OK")
print("Tensor               : OK")
print("PyTorch Module       : OK")

print()




print("=" * 60)
print("Lab14-01 완료")
print("=" * 60)