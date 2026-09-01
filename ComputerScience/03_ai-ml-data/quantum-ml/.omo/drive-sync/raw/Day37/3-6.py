"""
==========================================================
Lab 6. Cost 변화 시각화
==========================================================

학습 목표

1. Optimization History 저장
2. Cost 변화 확인
3. Parameter 변화 확인
4. Convergence 분석

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


print("=" * 60)
print("STEP 1. History 생성")
print("=" * 60)

theta_history = []
cost_history = []


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Objective Function")
print("=" * 60)


def objective(theta):

    cost = np.cos(theta[0])

    theta_history.append(theta[0])
    cost_history.append(cost)

    print(
        f"Theta = {theta[0]:8.5f}"
        f"   Cost = {cost:8.5f}"
    )

    return cost


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. Optimization")
print("=" * 60)

initial_theta = [0.5]

result = minimize(

    objective,

    initial_theta,

    method="COBYLA"

)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. Optimization Result")
print("=" * 60)

print("Optimal Theta")

print(result.x[0])

print()

print("Minimum Cost")

print(result.fun)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. Optimization History")
print("=" * 60)

print(
    "{:<10}{:<15}{:<15}".format(
        "Iter",
        "Theta",
        "Cost"
    )
)

print("-" * 40)

for i in range(len(theta_history)):

    print(
        "{:<10}{:<15.6f}{:<15.6f}".format(
            i,
            theta_history[i],
            cost_history[i]
        )
    )


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6. Cost Graph")
print("=" * 60)

plt.figure(figsize=(8,5))

plt.plot(

    range(len(cost_history)),

    cost_history,

    marker="o",

    linewidth=2

)

plt.title("Cost History")

plt.xlabel("Iteration")

plt.ylabel("Cost")

plt.grid(True)

plt.tight_layout()

plt.show()


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7. Parameter Graph")
print("=" * 60)

plt.figure(figsize=(8,5))

plt.plot(

    range(len(theta_history)),

    theta_history,

    marker="o",

    linewidth=2

)

plt.title("Parameter Update")

plt.xlabel("Iteration")

plt.ylabel("Theta")

plt.grid(True)

plt.tight_layout()

plt.show()


