"""
============================================================
Lab. QSVM Performance Evaluation


데이터 준비
------------------------------------------------------------
이번 실습에서는 QSVM 학습을 위한 데이터를 준비한다.

실습 목표

1. Iris Dataset 불러오기
2. Binary Classification 구성
3. Feature 선택
4. Train/Test Split
5. Feature Scaling
6. 결과 확인

============================================================
"""

# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 70)
print("STEP 1. Library Import")
print("=" * 70)

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("Library Import Completed")

# ============================================================
# STEP 2. Iris Dataset 준비
# ============================================================

print("\n" + "=" * 70)
print("STEP 2. Iris Dataset")
print("=" * 70)

iris = load_iris()

X = iris.data
y = iris.target

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)
print("Class Names   :", iris.target_names)

# ============================================================
# STEP 3. Binary Classification
# ============================================================

print("\n" + "=" * 70)
print("STEP 3. Binary Classification")
print("=" * 70)

binary_mask = y < 2

X_binary = X[binary_mask]
y_binary = y[binary_mask]

print("Binary Feature Shape :", X_binary.shape)
print("Binary Target Shape  :", y_binary.shape)
print("Classes              :", np.unique(y_binary))

# ============================================================
# STEP 4. Feature Selection
# ============================================================

print("\n" + "=" * 70)
print("STEP 4. Feature Selection")
print("=" * 70)

# Petal Length
# Petal Width

X_selected = X_binary[:, 2:4]

print("Selected Features")
print("---------------------------")

for feature in iris.feature_names[2:4]:
    print(feature)

print()

print("Selected Shape :", X_selected.shape)

# ============================================================
# STEP 5. Train / Test Split
# ============================================================

print("\n" + "=" * 70)
print("STEP 5. Train / Test Split")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y_binary,
    test_size=0.30,
    random_state=42,
    stratify=y_binary
)

print("Training Shape :", X_train.shape)
print("Test Shape     :", X_test.shape)

print()

print("Training Class Distribution")
print(np.bincount(y_train))

print()

print("Test Class Distribution")
print(np.bincount(y_test))

# ============================================================
# STEP 6. Feature Scaling
# ============================================================

print("\n" + "=" * 70)
print("STEP 6. Feature Scaling")
print("=" * 70)

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Training Minimum :", X_train_scaled.min())
print("Training Maximum :", X_train_scaled.max())

print("Test Minimum     :", X_test_scaled.min())
print("Test Maximum     :", X_test_scaled.max())

# ============================================================
# STEP 7. Preview
# ============================================================

print("\n" + "=" * 70)
print("STEP 7. Data Preview")
print("=" * 70)

train_df = pd.DataFrame(
    X_train_scaled,
    columns=[
        "Petal Length",
        "Petal Width"
    ]
)

train_df["Label"] = y_train

print(train_df.head())

# ============================================================
# STEP 8. Save Dataset
# ============================================================

print("\n" + "=" * 70)
print("STEP 8. Save Dataset")
print("=" * 70)

os.makedirs("output", exist_ok=True)

np.save("output/X_train.npy", X_train_scaled)
np.save("output/X_test.npy", X_test_scaled)

np.save("output/y_train.npy", y_train)
np.save("output/y_test.npy", y_test)

print("Saved Files")

print("- output/X_train.npy")
print("- output/X_test.npy")
print("- output/y_train.npy")
print("- output/y_test.npy")

# ============================================================
# STEP 9. Summary
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPARATION SUMMARY")
print("=" * 70)

print("Dataset              : Iris")
print("Classification       : Binary")

print("Training Samples     :", len(X_train))
print("Test Samples         :", len(X_test))

print("Feature Count        :", X_train.shape[1])

print("Scaling              : MinMax (0 ~ π)")

print("=" * 70)
print("Data Preparation Completed")
print("Next Step : 2-3.py")
print("=" * 70)