"""
============================================================
Loss Function 이해하기
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 1. Target 설정")
print("=" * 70)

target = 0.8

print(f"Target : {target}")

print("\n")
print("=" * 70)
print("STEP 2. Prediction 생성")
print("=" * 70)

predictions = np.linspace(0, 1, 11)

print(predictions)

print("\n")
print("=" * 70)
print("STEP 3. Loss 계산")
print("=" * 70)

losses = (predictions - target) ** 2

print("Loss 계산 완료")


print("\n")
print("=" * 70)
print("STEP 4. Prediction vs Loss")
print("=" * 70)

print(f"{'Prediction':>15}{'Target':>15}{'Loss':>15}")
print("-" * 45)

for prediction, loss in zip(predictions, losses):

    print(
        f"{prediction:15.2f}"
        f"{target:15.2f}"
        f"{loss:15.6f}"
    )


print("\n")
print("=" * 70)
print("STEP 5. Minimum Loss")
print("=" * 70)

best_index = np.argmin(losses)

best_prediction = predictions[best_index]

best_loss = losses[best_index]

print(f"Best Prediction : {best_prediction:.2f}")
print(f"Minimum Loss    : {best_loss:.6f}")



plt.figure(figsize=(8, 4))

plt.plot(
    predictions,
    marker='o',
    linewidth=2,
    label="Prediction"
)

plt.axvline(
    target,
    color='red',
    linestyle='--',
    label="Target"
)

plt.title("Prediction")

plt.xlabel("Sample")

plt.ylabel("Prediction")

plt.grid(True)

plt.legend()

plt.show()


plt.figure(figsize=(8, 4))

plt.plot(
    predictions,
    losses,
    marker='o',
    linewidth=2,
    color='green'
)

plt.scatter(
    best_prediction,
    best_loss,
    color='red',
    s=100,
    label="Minimum Loss"
)

plt.title("Loss Function")

plt.xlabel("Prediction")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.show()


plt.figure(figsize=(8, 4))

plt.plot(
    predictions,
    predictions,
    label="Prediction",
    linewidth=2
)

plt.plot(
    predictions,
    losses,
    label="Loss",
    linewidth=2
)

plt.axvline(
    target,
    linestyle='--',
    color='red',
    label="Target"
)

plt.grid(True)

plt.legend()

plt.title("Prediction vs Loss")

plt.show()


print("\n")
print("=" * 70)
print("STEP 6. Summary")
print("=" * 70)

print(f"Target            : {target}")
print(f"Best Prediction   : {best_prediction:.2f}")
print(f"Minimum Loss      : {best_loss:.6f}")

print("\nLoss Function 이해 완료")









