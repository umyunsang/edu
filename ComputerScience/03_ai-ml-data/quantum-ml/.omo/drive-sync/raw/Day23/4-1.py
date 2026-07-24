# ============================================================
# Lab. Quantum Kernel Matrix Validation
# ============================================================

import numpy as np

# ============================================================
# STEP 1. 예제 Kernel Matrix
# ============================================================

kernel_matrix = np.array([
    [1.00, 0.88, 0.32, 0.25],
    [0.88, 1.00, 0.36, 0.29],
    [0.32, 0.36, 1.00, 0.84],
    [0.25, 0.29, 0.84, 1.00]
])

labels = np.array([0, 0, 1, 1])

# ============================================================
# STEP 2. 기본 정보
# ============================================================

print("=" * 70)
print("Quantum Kernel Matrix")
print("=" * 70)

print(kernel_matrix)

# ============================================================
# STEP 3. 구조 검증
# ============================================================

print("\n[Structure Validation]")

rows, cols = kernel_matrix.shape

print("Shape :", kernel_matrix.shape)
print("Dimension :", kernel_matrix.ndim)

print("Square Matrix :", rows == cols)

print("Training Samples :", len(labels))
print("Matrix Size Match :", rows == len(labels))

# ============================================================
# STEP 4. 값 검증
# ============================================================

print("\n[Value Validation]")

print("Contains NaN :", np.isnan(kernel_matrix).any())

print("Contains Infinity :", np.isinf(kernel_matrix).any())

print("Minimum Value :", np.min(kernel_matrix))

print("Maximum Value :", np.max(kernel_matrix))

print(
    "Kernel Range [0,1] :",
    np.min(kernel_matrix) >= 0 and np.max(kernel_matrix) <= 1
)

# ============================================================
# STEP 5. Diagonal Validation
# ============================================================

print("\n[Diagonal Validation]")

diagonal = np.diag(kernel_matrix)

print(diagonal)

print(
    "Diagonal ≈ 1 :",
    np.allclose(diagonal, np.ones(rows))
)

# ============================================================
# STEP 6. Symmetry Validation
# ============================================================

print("\n[Symmetry Validation]")

print(
    "Symmetric :",
    np.allclose(kernel_matrix, kernel_matrix.T)
)

error = np.max(
    np.abs(
        kernel_matrix -
        kernel_matrix.T
    )
)

print("Maximum Error :", error)

# ============================================================
# STEP 7. Eigenvalue Validation
# ============================================================

print("\n[Eigenvalue Validation]")

eigenvalues = np.linalg.eigvalsh(kernel_matrix)

print(eigenvalues)

print(
    "Minimum Eigenvalue :",
    np.min(eigenvalues)
)

print(
    "Positive Semi-definite :",
    np.min(eigenvalues) >= -1e-8
)

# ============================================================
# STEP 8. Label Validation
# ============================================================

print("\n[Label Validation]")

print("Labels :", labels)

print(
    "Label Size Match :",
    len(labels) == rows
)

# ============================================================
# STEP 9. 최종 결과
# ============================================================

print("\n" + "=" * 70)
print("Validation Summary")
print("=" * 70)

check_items = {
    "2D Matrix":
        kernel_matrix.ndim == 2,

    "Square Matrix":
        rows == cols,

    "No NaN":
        not np.isnan(kernel_matrix).any(),

    "No Infinity":
        not np.isinf(kernel_matrix).any(),

    "Kernel Range":
        np.min(kernel_matrix) >= 0 and
        np.max(kernel_matrix) <= 1,

    "Diagonal":
        np.allclose(
            diagonal,
            np.ones(rows)
        ),

    "Symmetric":
        np.allclose(
            kernel_matrix,
            kernel_matrix.T
        ),

    "PSD":
        np.min(eigenvalues) >= -1e-8,

    "Label Match":
        len(labels) == rows
}

for key, value in check_items.items():

    result = "PASS" if value else "FAIL"

    print(f"{key:20} : {result}")

# ============================================================
# STEP 10. 최종 판정
# ============================================================

if all(check_items.values()):

    print("\nKernel Matrix는 정상입니다.")
    print("→ Heatmap 생성 가능")
    print("→ Similarity 분석 가능")
    print("→ QSVM 입력 가능")

else:

    print("\nKernel Matrix를 다시 확인해야 합니다.")