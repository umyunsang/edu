"""
============================================================
Lab 15. QSVM Performance Evaluation


QSVM 생성 및 학습
------------------------------------------------------------
이번 실습에서는

1. 저장된 Dataset 불러오기
2. Quantum Feature Map 생성
3. Fidelity Quantum Kernel 생성
4. QSVC 생성
5. QSVM 학습
6. 학습 완료 모델 저장

============================================================
"""

# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

import os
import time
import pickle
import numpy as np

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

print("Library Import Completed")

# ============================================================
# STEP 2. Dataset Load
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Load Dataset")
print("=" * 70)

X_train = np.load("output/X_train.npy")
X_test = np.load("output/X_test.npy")

y_train = np.load("output/y_train.npy")
y_test = np.load("output/y_test.npy")

print("Training Shape :", X_train.shape)
print("Test Shape     :", X_test.shape)

# ============================================================
# STEP 3. Quantum Feature Map
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Quantum Feature Map")
print("=" * 70)

num_features = X_train.shape[1]

feature_map = zz_feature_map(
    feature_dimension=num_features,
    reps=2,
    entanglement="linear"
)

print(feature_map)

# ============================================================
# STEP 4. Fidelity Quantum Kernel
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Fidelity Quantum Kernel")
print("=" * 70)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print("Quantum Kernel Created")

# ============================================================
# STEP 5. QSVC 생성
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. QSVC Model")
print("=" * 70)

qsvc = QSVC(
    quantum_kernel=quantum_kernel
)

print(qsvc)

# ============================================================
# STEP 6. QSVM Training
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. QSVM Training")
print("=" * 70)

start = time.perf_counter()

qsvc.fit(
    X_train,
    y_train
)

end = time.perf_counter()

training_time = end - start

print("Training Completed")

print(f"Training Time : {training_time:.4f} sec")

# ============================================================
# STEP 7. Model Information
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Model Information")
print("=" * 70)

print("Model Type        :", type(qsvc).__name__)
print("Feature Count     :", num_features)
print("Training Samples  :", len(X_train))
print("Kernel Type       :", type(quantum_kernel).__name__)

# ============================================================
# STEP 8. Save Model
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Save Model")
print("=" * 70)

os.makedirs("output", exist_ok=True)

with open("output/qsvc_model.pkl", "wb") as f:
    pickle.dump(qsvc, f)

print("Saved")

print("output/qsvc_model.pkl")

# ============================================================
# STEP 9. Save Training Information
# ============================================================

print("\n" + "=" * 70)
print("STEP 9. Save Training Information")
print("=" * 70)

training_info = {

    "feature_count": num_features,

    "training_samples": len(X_train),

    "test_samples": len(X_test),

    "training_time": training_time,

    "feature_map": "ZZFeatureMap",

    "kernel": "FidelityQuantumKernel",

    "classifier": "QSVC"

}

with open("output/training_info.pkl", "wb") as f:

    pickle.dump(training_info, f)

print("output/training_info.pkl")

# ============================================================
# STEP 10. Summary
# ============================================================

print("\n" + "=" * 70)
print("QSVM TRAINING SUMMARY")
print("=" * 70)

print("Dataset             : Iris")

print("Training Samples    :", len(X_train))

print("Test Samples        :", len(X_test))

print("Feature Count       :", num_features)

print("Feature Map         : ZZFeatureMap")

print("Kernel              : FidelityQuantumKernel")

print("Classifier          : QSVC")

print(f"Training Time       : {training_time:.4f} sec")

print("=" * 70)
print("QSVM Training Completed")
print("Next Step : 2-4.py")
print("=" * 70)