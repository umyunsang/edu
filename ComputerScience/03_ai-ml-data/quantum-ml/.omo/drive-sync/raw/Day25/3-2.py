# ============================================================
# QSVC 모델 생성
# ============================================================

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

# ============================================================
# STEP 1. Lab 1 결과 확인
# ============================================================

# ============================================================
# STEP 1. Iris Dataset 불러오기
# ============================================================

print("=" * 80)
print("STEP 1. Iris Dataset 불러오기")
print("=" * 80)

iris = load_iris()

X = iris.data
y = iris.target

print(f"전체 Sample 수 : {X.shape[0]}")
print(f"Feature 수     : {X.shape[1]}")
print(f"Class 수       : {len(np.unique(y))}")

print("\nFeature 이름")
print(iris.feature_names)

print("\nClass 이름")
print(iris.target_names)


# ============================================================
# STEP 2. Feature 선택
# Petal Length, Petal Width 사용
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Feature 선택")
print("=" * 80)

# Petal Length, Petal Width
X = X[:, 2:4]

print("선택된 Feature")
print("- Petal Length")
print("- Petal Width")

print(f"\n변경된 Feature Shape : {X.shape}")


# ============================================================
# STEP 3. Binary Classification 구성
# Setosa(0), Versicolor(1)만 사용
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Binary Classification 구성")
print("=" * 80)

mask = y != 2

X = X[mask]
y = y[mask]

print(f"Sample 수 : {X.shape[0]}")
print(f"Class : {np.unique(y)}")

print("\nClass Distribution")

unique, counts = np.unique(y, return_counts=True)

for c, n in zip(unique, counts):
    print(f"Class {c} : {n}")


# ============================================================
# STEP 4. Train / Test Split
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. Train / Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")


# ============================================================
# STEP 5. Feature Scaling
# Quantum Feature Map 입력 범위 : 0 ~ π
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Feature Scaling")
print("=" * 80)

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling 완료")

print("\nTrain Data")

print("Minimum")

print(X_train_scaled.min(axis=0))

print("Maximum")

print(X_train_scaled.max(axis=0))


# ============================================================
# STEP 6. 데이터 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. 전처리 데이터 확인")
print("=" * 80)

print(f"X_train_scaled : {X_train_scaled.shape}")

print(f"X_test_scaled  : {X_test_scaled.shape}")

print(f"y_train        : {y_train.shape}")

print(f"y_test         : {y_test.shape}")

print("\nTraining Sample (첫 5개)")

df_train = pd.DataFrame(
    X_train_scaled,
    columns=[
        "Petal Length",
        "Petal Width"
    ]
)

print(df_train.head())

print("\nTraining Label")

print(y_train[:5])


print("=" * 80)
print("STEP 1. Lab 1 결과 확인")
print("=" * 80)

print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
print(f"y_train        : {y_train.shape}")
print(f"y_test         : {y_test.shape}")

feature_dimension = X_train_scaled.shape[1]

print()
print(f"Feature Dimension : {feature_dimension}")



# ============================================================
# QSVC 모델 생성 시작
# ============================================================


# ============================================================
# STEP 2. Quantum Feature Map 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Quantum Feature Map 생성")
print("=" * 80)

feature_map = zz_feature_map(
    feature_dimension=feature_dimension,
    reps=2,
    entanglement="linear"
)

print(feature_map)

print()

print("Qubit 수 :", feature_map.num_qubits)
print("Parameter 수 :", feature_map.num_parameters)


# ============================================================
# STEP 3. Quantum Feature Map 회로 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Feature Map 회로")
print("=" * 80)

print(
    feature_map.draw(output="text")
)

# matplotlib 환경이라면
# feature_map.draw(output="mpl")


# ============================================================
# STEP 4. Quantum Kernel 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. Quantum Kernel 생성")
print("=" * 80)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print(type(quantum_kernel))

print()

print(
    "Feature Map Qubits :",
    quantum_kernel.feature_map.num_qubits
)


# ============================================================
# STEP 5. Quantum Kernel 확인
# (교육용 Sample만 사용)
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Sample Kernel Matrix")
print("=" * 80)

sample = X_train_scaled[:3]

kernel_matrix = quantum_kernel.evaluate(
    x_vec=sample
)

print(kernel_matrix)

print()

print("Kernel Matrix Shape :", kernel_matrix.shape)


# ============================================================
# STEP 6. QSVC 객체 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. QSVC 객체 생성")
print("=" * 80)

qsvc = QSVC(
    quantum_kernel=quantum_kernel
)

print(qsvc)

print()

print(type(qsvc))


# ============================================================
# STEP 7. 객체 연결 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. QSVC 구성 확인")
print("=" * 80)

print("Feature Dimension :", feature_dimension)

print("Feature Map :", type(feature_map).__name__)

print("Quantum Kernel :", type(quantum_kernel).__name__)

print("Classifier :", type(qsvc).__name__)


# ============================================================
# STEP 8. 실습 완료
# ============================================================

print("\n" + "=" * 80)
print("Lab 2 완료")
print("=" * 80)

print("생성된 객체")

print("- feature_map")

print("- quantum_kernel")

print("- qsvc")

print()

