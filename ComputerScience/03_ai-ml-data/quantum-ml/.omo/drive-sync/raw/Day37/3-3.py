"""
==========================================================
Lab 3. Objective Function 정의
==========================================================

학습 목표

1. Objective Function 이해
2. Parameter에 따른 Cost 계산
3. Cost가 최소가 되는 Parameter 확인
4. Variational Evaluation 과정 이해

"""

import numpy as np

print("=" * 60)
print("STEP 1. Parameter 준비")
print("=" * 60)

parameter_values = [

    0,
    np.pi / 4,
    np.pi / 2,
    3 * np.pi / 4,
    np.pi

]

for value in parameter_values:

    print(f"{value:.6f}")


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Objective Function 정의")
print("=" * 60)


def objective(theta):

    """
    Objective Function

    실제 Quantum Circuit에서는
    Measurement 결과를 사용하지만,

    이번 실습에서는

    <Z> = cos(theta)

    를 사용한다.
    """

    return np.cos(theta)


print("Objective Function")

print("Cost(theta) = cos(theta)")


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. Cost 계산")
print("=" * 60)

results = []

for theta in parameter_values:

    cost = objective(theta)

    results.append(
        (theta, cost)
    )

    print(
        f"Theta = {theta:.6f} "
        f" Cost = {cost:.6f}"
    )


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. 최소 Cost 찾기")
print("=" * 60)

best_theta = None
best_cost = 999

for theta, cost in results:

    if cost < best_cost:

        best_theta = theta
        best_cost = cost

print("Optimal Theta")

print(f"{best_theta:.6f}")

print("\nMinimum Cost")

print(f"{best_cost:.6f}")


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. 결과 표")
print("=" * 60)

print(
    "{:<15}{:<15}".format(
        "Theta(rad)",
        "Cost"
    )
)

print("-" * 30)

for theta, cost in results:

    print(
        "{:<15.6f}{:<15.6f}".format(
            theta,
            cost
        )
    )


