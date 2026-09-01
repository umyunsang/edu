"""
============================================================
Target과 Loss Curve 생성
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# STEP 1. Parameter 생성
# ============================================================

print("=" * 80)
print("STEP 1. Parameter Sweep")
print("=" * 80)

theta_values = np.linspace(
    0,
    2 * np.pi,
    100
)

print(f"Number of Parameters : {len(theta_values)}")

# ============================================================
# STEP 2. Target 설정
# ============================================================

print("\n")
print("=" * 80)
print("STEP 2. Target 설정")
print("=" * 80)

target = -1.0

print(f"Target : {target}")

# ============================================================
# STEP 3. Prediction 계산
# ============================================================

print("\n")
print("=" * 80)
print("STEP 3. Prediction 계산")
print("=" * 80)

observable = SparsePauliOp.from_list(
    [("Z", 1)]
)

predictions = []

for theta in theta_values:

    qc = QuantumCircuit(1)

    qc.ry(theta, 0)

    state = Statevector.from_instruction(qc)

    prediction = np.real(
        state.expectation_value(observable)
    )

    predictions.append(prediction)

predictions = np.array(predictions)

print("Prediction 계산 완료")

# ============================================================
# STEP 4. Loss 계산
# ============================================================

print("\n")
print("=" * 80)
print("STEP 4. Loss 계산")
print("=" * 80)

losses = (
    predictions - target
) ** 2

print("Loss 계산 완료")

# ============================================================
# STEP 5. 최소 Loss 찾기
# ============================================================

print("\n")
print("=" * 80)
print("STEP 5. 최소 Loss 탐색")
print("=" * 80)

best_index = np.argmin(losses)

best_theta = theta_values[best_index]

best_prediction = predictions[best_index]

best_loss = losses[best_index]

print(f"Best Theta      : {best_theta:.6f}")
print(f"Prediction      : {best_prediction:.6f}")
print(f"Loss            : {best_loss:.8f}")

# ============================================================
# STEP 6. 결과 테이블 출력
# ============================================================

print("\n")
print("=" * 80)
print("STEP 6. 일부 결과")
print("=" * 80)

print(
    f"{'Theta':>10}"
    f"{'Prediction':>15}"
    f"{'Target':>15}"
    f"{'Loss':>15}"
)

print("-"*55)

sample_index = np.linspace(
    0,
    len(theta_values)-1,
    10,
    dtype=int
)

for i in sample_index:

    print(
        f"{theta_values[i]:10.4f}"
        f"{predictions[i]:15.4f}"
        f"{target:15.4f}"
        f"{losses[i]:15.6f}"
    )

# ============================================================
# STEP 7. Prediction Curve
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    theta_values,
    predictions,
    linewidth=2,
    label="Prediction"
)

plt.axhline(
    target,
    linestyle="--",
    color="red",
    label="Target"
)

plt.scatter(
    best_theta,
    best_prediction,
    color="red",
    s=80,
    label="Best Parameter"
)

plt.xlabel("Theta")

plt.ylabel("Prediction")

plt.title("Prediction vs Target")

plt.grid(True)

plt.legend()

plt.show()

# ============================================================
# STEP 8. Loss Curve
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    theta_values,
    losses,
    linewidth=2,
    color="green"
)

plt.scatter(
    best_theta,
    best_loss,
    color="red",
    s=80,
    label="Minimum Loss"
)

plt.xlabel("Theta")

plt.ylabel("Loss")

plt.title("Loss Landscape")

plt.grid(True)

plt.legend()

plt.show()

# ============================================================
# STEP 9. Prediction과 Loss 동시 비교
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    theta_values,
    predictions,
    label="Prediction"
)

plt.plot(
    theta_values,
    losses,
    label="Loss"
)

plt.axhline(
    target,
    linestyle="--",
    color="red",
    label="Target"
)

plt.xlabel("Theta")

plt.ylabel("Value")

plt.title("Prediction and Loss")

plt.grid(True)

plt.legend()

plt.show()

# ============================================================
# STEP 10. 결과 요약
# ============================================================

print("\n")
print("=" * 80)
print("STEP 10. Summary")
print("=" * 80)

print(f"Target              : {target}")

print(f"Best Parameter θ    : {best_theta:.6f}")

print(f"Prediction          : {best_prediction:.6f}")

print(f"Minimum Loss        : {best_loss:.8f}")

print("\nLoss Curve 생성 완료")
