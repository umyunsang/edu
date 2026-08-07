# ============================================================
# Qiskit Fidelity Quantum Kernel 구현
# ============================================================

import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import zz_feature_map

from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# STEP 1. 데이터 준비
# ============================================================

print("=" * 80)
print("STEP 1. 데이터 준비")
print("=" * 80)

iris = load_iris()

# Iris Dataset
# Feature 2개만 사용
X = iris.data[:, :2]
y = iris.target

# Binary Classification
mask = y != 2

X = X[mask]
y = y[mask]

print(f"Feature Shape : {X.shape}")
print(f"Label Shape   : {y.shape}")

print("\nFirst 5 Samples")
print(X[:5])


# ============================================================
# STEP 2. Feature Scaling
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. Feature Scaling")
print("=" * 80)

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_scaled = scaler.fit_transform(X)

print("Scaled Data")
print(np.round(X_scaled[:5], 3))


# ============================================================
# STEP 3. Quantum Feature Map 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. Quantum Feature Map")
print("=" * 80)

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=2
)

print(feature_map)

print("\nCircuit Depth :", feature_map.decompose().depth())

print("\nGate Counts")
print(feature_map.decompose().count_ops())


# ============================================================
# STEP 4. Fidelity Quantum Kernel 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 4. Fidelity Quantum Kernel 생성")
print("=" * 80)

kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print(kernel)


# ============================================================
# STEP 5. Kernel Matrix 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Kernel Matrix 생성")
print("=" * 80)

kernel_matrix = kernel.evaluate(
    x_vec=X_scaled
)

print("Kernel Matrix Shape")

print(kernel_matrix.shape)


# ============================================================
# STEP 6. Kernel Matrix 출력
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. Kernel Matrix")
print("=" * 80)

print(np.round(kernel_matrix, 3))


# ============================================================
# STEP 7. Kernel Matrix 검증
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. Kernel Matrix 검증")
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
        3
    )
)

print()

print("Minimum")

print(kernel_matrix.min())

print()

print("Maximum")

print(kernel_matrix.max())


# ============================================================
# STEP 8. 일부 Similarity 확인
# ============================================================

print("\n" + "=" * 80)
print("STEP 8. Similarity 확인")
print("=" * 80)

print(f"K(x1,x1) = {kernel_matrix[0,0]:.3f}")

print(f"K(x1,x2) = {kernel_matrix[0,1]:.3f}")

print(f"K(x1,x10)= {kernel_matrix[0,9]:.3f}")


# ============================================================
# STEP 9. 실습 결과 정리
# ============================================================

print("\n" + "=" * 80)
print("STEP 9. 실습 결과")
print("=" * 80)

print(f"Feature 수            : {X.shape[1]}")

print(f"Qubit 수              : {feature_map.num_qubits}")

print(f"Kernel Matrix 크기    : {kernel_matrix.shape}")

print(
    "Kernel Matrix 대칭 여부 :",
    np.allclose(
        kernel_matrix,
        kernel_matrix.T
    )
)

print(
    "Diagonal 평균 :",
    np.mean(
        np.diag(kernel_matrix)
    )
)

print(
    "Minimum Similarity :",
    round(kernel_matrix.min(), 4)
)

print(
    "Maximum Similarity :",
    round(kernel_matrix.max(), 4)
)

print("\n실습 완료!")