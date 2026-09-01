"""
============================================================
Lab. QSVM Performance Evaluation

실습 환경 준비
------------------------------------------------------------
이번 실습에서는 QSVM 성능 평가에 필요한 라이브러리와
실행 환경을 확인한다.

실습 목표
1. Python 실행 환경 확인
2. Qiskit 설치 여부 확인
3. Scikit-Learn 설치 여부 확인
4. Pandas / NumPy / Matplotlib 확인
5. 출력 폴더 생성
============================================================
"""

# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

import os
import sys
import platform
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import qiskit
import qiskit_machine_learning

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    auc
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

print("Library Import Completed")

# ============================================================
# STEP 2. Python Environment
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Python Environment")
print("=" * 70)

print(f"Python Version : {sys.version.split()[0]}")
print(f"Platform       : {platform.system()}")
print(f"Processor      : {platform.processor()}")

# ============================================================
# STEP 3. Package Version
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Package Version")
print("=" * 70)

print(f"NumPy                     : {np.__version__}")
print(f"Pandas                    : {pd.__version__}")
print(f"Matplotlib                : {plt.matplotlib.__version__}")
print(f"Scikit-Learn              : {sklearn.__version__}")
print(f"Qiskit                    : {qiskit.__version__}")
print(f"Qiskit Machine Learning   : {qiskit_machine_learning.__version__}")

# ============================================================
# STEP 4. Output Directory
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Output Directory")
print("=" * 70)

OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Output Directory : {OUTPUT_DIR}")
print("Directory Ready")

# ============================================================
# STEP 5. Iris Dataset 확인
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Iris Dataset Test")
print("=" * 70)

iris = load_iris()

print(f"Dataset Name : Iris")
print(f"Samples      : {iris.data.shape[0]}")
print(f"Features     : {iris.data.shape[1]}")
print(f"Classes      : {len(iris.target_names)}")

print("\nClass Names")

for idx, name in enumerate(iris.target_names):
    print(f"{idx} : {name}")

# ============================================================
# STEP 6. Quantum Library 확인
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. Quantum Library Test")
print("=" * 70)

feature_map = zz_feature_map(feature_dimension=2)

print(feature_map)

print("\nFeature Map Creation Success")

# ============================================================
# STEP 7. Environment Summary
# ============================================================

print("\n" + "=" * 70)
print("ENVIRONMENT SUMMARY")
print("=" * 70)

print("Python Environment        : OK")
print("Scikit-Learn              : OK")
print("Qiskit                    : OK")
print("Qiskit Machine Learning   : OK")
print("Matplotlib                : OK")
print("Output Directory          : OK")
print("Iris Dataset              : OK")
print("Quantum Feature Map       : OK")

print("=" * 70)
print("Environment Ready")
print("Next Step : 2-2.py")
print("=" * 70)