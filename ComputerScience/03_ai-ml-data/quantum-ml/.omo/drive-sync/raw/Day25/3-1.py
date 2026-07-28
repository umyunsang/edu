# ============================================================
# Iris Dataset을 이용한 QSVM 학습 데이터 준비
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


# ============================================================
# STEP 7. Class Distribution 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. Class Distribution")
print("=" * 80)

train_unique, train_counts = np.unique(
    y_train,
    return_counts=True
)

test_unique, test_counts = np.unique(
    y_test,
    return_counts=True
)

print("Train")

for c, n in zip(train_unique, train_counts):
    print(f"Class {c} : {n}")

print()

print("Test")

for c, n in zip(test_unique, test_counts):
    print(f"Class {c} : {n}")


# ============================================================
# STEP 8. 실습 완료
# ============================================================

print("\n" + "=" * 80)
print("Lab 1 완료")
print("=" * 80)

print("QSVM 학습을 위한 데이터 준비가 완료되었습니다.")

print()

print("생성된 데이터")

print("X_train_scaled")

print("X_test_scaled")

print("y_train")

print("y_test")

print()

