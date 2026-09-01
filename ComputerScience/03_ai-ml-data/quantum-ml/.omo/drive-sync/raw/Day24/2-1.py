# ============================================================
# QSVC (Quantum Support Vector Classifier) Example
# ============================================================

# ============================================================
# STEP 1. Library Import
# ============================================================

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# ============================================================
# STEP 2. Iris Dataset 준비
# ============================================================

iris = load_iris()

# Feature 2개만 사용
X = iris.data[:, :2]

# Binary Classification
X = X[iris.target != 2]
y = iris.target[iris.target != 2]

print("=" * 60)
print("Dataset")
print("=" * 60)
print("X Shape :", X.shape)
print("y Shape :", y.shape)

# ============================================================
# STEP 3. Feature Scaling
# Quantum Rotation Gate 입력 범위 : [0, π]
# ============================================================

scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)

# ============================================================
# STEP 4. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data :", len(X_train))
print("Test Data     :", len(X_test))

# ============================================================
# STEP 5. Quantum Feature Map 생성
# ============================================================

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=2,
    entanglement="linear"
)

print("\nFeature Map 생성 완료")

# ============================================================
# STEP 6. Fidelity Quantum Kernel 생성
# ============================================================

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print("Quantum Kernel 생성 완료")

# ============================================================
# STEP 7. QSVC 모델 생성
# ============================================================

model = QSVC(
    quantum_kernel=quantum_kernel
)

print("QSVC 모델 생성 완료")

# ============================================================
# STEP 8. 모델 학습
# ============================================================

print("\nModel Training...")

model.fit(X_train, y_train)

print("Training Complete!")

# ============================================================
# STEP 9. Prediction 수행
# ============================================================

prediction = model.predict(X_test)

print("\nPrediction")
print(prediction)

# ============================================================
# STEP 10. Accuracy 계산
# ============================================================

accuracy = model.score(X_test, y_test)

print("\nAccuracy : {:.4f}".format(accuracy))

# ============================================================
# STEP 11. 실제 Label 비교
# ============================================================

print("\nActual Label")
print(y_test)

print("\nPrediction")
print(prediction)

# ============================================================
# STEP 12. Support Vector 정보 출력
# ============================================================

print("\nSupport Vector Index")
print(model.support_)

print("\nSupport Vector 개수")
print(model.n_support_)

print("\nClass")
print(model.classes_)
