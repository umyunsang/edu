# ============================================================
# Lab. 좋은 Quantum Kernel 평가 실습
# ============================================================

# ============================================================
# STEP 1. Library Import
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# ============================================================
# STEP 2. Iris Dataset 준비
# ============================================================

iris = load_iris()

X = iris.data[:, :2]
y = iris.target


# ============================================================
# STEP 3. Binary Classification
# ============================================================

mask = y != 2

X = X[mask]
y = y[mask]

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

# ============================================================
# STEP 5. Feature Scaling
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# STEP 6. Feature Map
# ============================================================

feature_map = ZZFeatureMap(
    feature_dimension=2,
    reps=2
)

# ============================================================
# STEP 7. Quantum Kernel
# ============================================================

kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

# ============================================================
# STEP 8. Train Kernel Matrix
# ============================================================

kernel_matrix = kernel.evaluate(
    X_train
)

print(kernel_matrix.shape)


# ============================================================
# STEP 9. Validation
# ============================================================

print("Diagonal")

print(
    np.diag(kernel_matrix)
)

print()

print("Symmetric")

print(
    np.allclose(
        kernel_matrix,
        kernel_matrix.T
    )
)

print()

print("Minimum")

print(kernel_matrix.min())

print()

print("Maximum")

print(kernel_matrix.max())


# ============================================================
# STEP 10. Heatmap
# ============================================================

plt.figure(figsize=(8,6))

plt.imshow(
    kernel_matrix,
    cmap="Blues"
)

plt.colorbar(
    label="Similarity"
)

plt.title(
    "Quantum Kernel Matrix"
)

plt.show()


# ============================================================
# STEP 11. Same-Class Similarity
# ============================================================

same_similarity = []

for i in range(len(y_train)):
    for j in range(i+1, len(y_train)):

        if y_train[i] == y_train[j]:

            same_similarity.append(
                kernel_matrix[i,j]
            )

print(np.mean(same_similarity))


# ============================================================
# STEP 12. Different-Class Similarity
# ============================================================

different_similarity = []

for i in range(len(y_train)):
    for j in range(i+1, len(y_train)):

        if y_train[i] != y_train[j]:

            different_similarity.append(
                kernel_matrix[i,j]
            )

print(
    np.mean(
        different_similarity
    )
)

# ============================================================
# STEP 13. Similarity Gap
# ============================================================

gap = (
    np.mean(same_similarity)
    -
    np.mean(different_similarity)
)

print()

print("Similarity Gap")

print(gap)


# ============================================================
# STEP 14. Kernel Quality
# ============================================================

if gap > 0.4:

    print("Excellent Kernel")

elif gap > 0.2:

    print("Good Kernel")

else:

    print("Poor Kernel")


# ============================================================
# STEP 15. Final Report
# ============================================================

print("="*60)

print("Quantum Kernel Evaluation")

print("="*60)

print(
    f"Same-Class Similarity : {np.mean(same_similarity):.3f}"
)

print(
    f"Different-Class Similarity : {np.mean(different_similarity):.3f}"
)

print(
    f"Similarity Gap : {gap:.3f}"
)

print("="*60)






















