"""
============================================================
Parameter Sweep을 통한 Prediction 변화 관찰
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# STEP 1. Target 설정
# ============================================================

print("=" * 70)
print("STEP 1. Target 설정")
print("=" * 70)

target = -1.0

print(f"Target : {target}")

# ============================================================
# STEP 2. Parameter Sweep
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2. Parameter Sweep")
print("=" * 70)

theta_values = np.linspace(
    0,
    2 * np.pi,
    100
)

print(f"Number of Parameters : {len(theta_values)}")

# ============================================================
# STEP 3. Prediction 계산
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3. Prediction 계산")
print("=" * 70)

predictions = []

observable = SparsePauliOp.from_list([("Z", 1)])

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
print("=" * 70)
print("STEP 4. Loss 계산")
print("=" * 70)

losses = (
    predictions - target
) ** 2

print("Loss 계산 완료")

# ============================================================
# STEP 5. 최적 Parameter 찾기
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5. 최적 Parameter")
print("=" * 70)

best_index = np.argmin(losses)

best_theta = theta_values[best_index]

best_prediction = predictions[best_index]

best_loss = losses[best_index]

print(f"Best Theta      : {best_theta:.4f}")

print(f"Prediction      : {best_prediction:.4f}")

print(f"Loss            : {best_loss:.6f}")

# ============================================================
# STEP 6. 일부 결과 출력
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6. Sample Results")
print("=" * 70)

sample_index = np.linspace(
    0,
    len(theta_values)-1,
    10,
    dtype=int
)

print(
    f"{'Theta':>10}"
    f"{'Prediction':>15}"
    f"{'Loss':>15}"
)

print("-"*45)

for i in sample_index:

    print(
        f"{theta_values[i]:10.4f}"
        f"{predictions[i]:15.4f}"
        f"{losses[i]:15.6f}"
    )

# ============================================================
# STEP 7. Prediction Curve
# ============================================================

plt.figure(figsize=(10,5))

plt.plot(
    theta_values,
    predictions,
    linewidth=2
)

plt.axhline(
    target,
    color='red',
    linestyle='--',
    label='Target'
)

plt.scatter(
    best_theta,
    best_prediction,
    color='red',
    s=80
)

plt.title("Prediction vs Parameter")

plt.xlabel("Theta")

plt.ylabel("Prediction")

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
    linewidth=2
)

plt.scatter(
    best_theta,
    best_loss,
    color='red',
    s=80
)

plt.title("Loss Landscape")

plt.xlabel("Theta")

plt.ylabel("Loss")

plt.grid(True)

plt.show()

# ============================================================
# STEP 9. 결과 요약
# ============================================================

print("\n")
print("=" * 70)
print("STEP 9. Summary")
print("=" * 70)

print(f"Target               : {target}")

print(f"Best Parameter θ     : {best_theta:.4f}")

print(f"Prediction           : {best_prediction:.4f}")

print(f"Minimum Loss         : {best_loss:.6f}")

print("\nParameter Sweep 완료")
