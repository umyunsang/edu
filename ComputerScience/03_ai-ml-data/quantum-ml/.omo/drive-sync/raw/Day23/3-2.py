# ============================================================
# PauliFeatureMap Quantum Kernel 구현
# ============================================================

import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import pauli_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# STEP 1. Library Import
# ============================================================

print("=" * 80)
print("STEP 1. Library Import")
print("=" * 80)

print("Library Loaded Successfully")


# ============================================================
# STEP 2. Iris Dataset 준비
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Iris Dataset")
print("=" * 80)

iris = load_iris()

X = iris.data[:, :2]
y = iris.target

print("Original Dataset Shape :", X.shape)

print("\nFirst 5 Samples")
print(X[:5])


# ============================================================
# STEP 3. Binary Classification
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Binary Classification")
print("=" * 80)

mask = y != 2

X = X[mask]
y = y[mask]

print("Filtered Dataset Shape :", X.shape)

print("Classes :", np.unique(y))


# ============================================================
# STEP 4. Sample 선택
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. Sample Selection")
print("=" * 80)

samples_per_class = 6

X_class0 = X[y == 0][:samples_per_class]
X_class1 = X[y == 1][:samples_per_class]

y_class0 = y[y == 0][:samples_per_class]
y_class1 = y[y == 1][:samples_per_class]

X_small = np.concatenate([X_class0, X_class1])

y_small = np.concatenate([y_class0, y_class1])

print("Selected Samples :", X_small.shape)

print("Labels")

print(y_small)


# ============================================================
# STEP 5. Feature Scaling
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Feature Scaling")
print("=" * 80)

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_scaled = scaler.fit_transform(X_small)

print("Scaled Data")

print(np.round(X_scaled[:5], 3))


# ============================================================
# STEP 6. PauliFeatureMap 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. PauliFeatureMap")
print("=" * 80)

feature_map = pauli_feature_map(
    feature_dimension=2,
    reps=2,
    paulis=["Z", "ZZ"],
    entanglement="full"
)

print(feature_map)


# ============================================================
# STEP 7. Circuit 출력
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. Circuit Information")
print("=" * 80)

decomposed = feature_map.decompose()

print("Number of Qubits")

print(feature_map.num_qubits)

print()

print("Number of Parameters")

print(feature_map.num_parameters)

print()

print("Circuit Depth")

print(decomposed.depth())

print()

print("Gate Counts")

print(decomposed.count_ops())

# Notebook 환경
# decomposed.draw("mpl")


# ============================================================
# STEP 8. FidelityQuantumKernel 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 8. Fidelity Quantum Kernel")
print("=" * 80)

kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print(kernel)


# ============================================================
# STEP 9. Kernel Matrix 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 9. Kernel Matrix")
print("=" * 80)

kernel_matrix = kernel.evaluate(
    x_vec=X_scaled
)

print("Kernel Matrix Shape")

print(kernel_matrix.shape)


# ============================================================
# STEP 10. Kernel Matrix 출력
# ============================================================

print("\n" + "=" * 80)
print("STEP 10. Kernel Matrix Output")
print("=" * 80)

print(np.round(kernel_matrix, 4))


# ============================================================
# STEP 11. Kernel Matrix 검증
# ============================================================

print("\n" + "=" * 80)
print("STEP 11. Kernel Matrix Validation")
print("=" * 80)

print("Shape")

print(kernel_matrix.shape)

print()

print("Symmetric")

print(
    np.allclose(
        kernel_matrix,
        kernel_matrix.T
    )
)

print()

print("Diagonal")

print(
    np.round(
        np.diag(kernel_matrix),
        4
    )
)

print()

print("Minimum Similarity")

print(kernel_matrix.min())

print()

print("Maximum Similarity")

print(kernel_matrix.max())

print()

print("All Finite")

print(np.isfinite(kernel_matrix).all())


# ============================================================
# STEP 12. Same / Different Class Similarity
# ============================================================

print("\n" + "=" * 80)
print("STEP 12. Class Similarity Analysis")
print("=" * 80)


def analyze_similarity(kernel_matrix, labels):

    same = []
    different = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            if labels[i] == labels[j]:
                same.append(kernel_matrix[i, j])

            else:
                different.append(kernel_matrix[i, j])

    same = np.array(same)
    different = np.array(different)

    return (
        np.mean(same),
        np.mean(different)
    )


same_mean, different_mean = analyze_similarity(
    kernel_matrix,
    y_small
)

print("Same Class Similarity")

print(round(same_mean, 4))

print()

print("Different Class Similarity")

print(round(different_mean, 4))

print()

print("Similarity Gap")

print(round(same_mean - different_mean, 4))


# ============================================================
# 실습 종료
# ============================================================

print("\n" + "=" * 80)
print("Lab Completed")
print("=" * 80)

print(f"Feature Number      : {X_small.shape[1]}")
print(f"Qubit Number        : {feature_map.num_qubits}")
print(f"Kernel Matrix Size  : {kernel_matrix.shape}")

print(
    "Kernel Symmetric    :",
    np.allclose(
        kernel_matrix,
        kernel_matrix.T
    )
)

print(
    "Average Diagonal    :",
    round(
        np.mean(np.diag(kernel_matrix)),
        4
    )
)

print(
    "Same Class Mean     :",
    round(same_mean, 4)
)

print(
    "Different Class Mean:",
    round(different_mean, 4)
)

print("\nPauliFeatureMap Quantum Kernel 생성 완료!")