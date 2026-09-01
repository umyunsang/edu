"""
==========================================================
Lab 4. Classical Optimizer 연결
==========================================================

학습 목표

1. Objective Function 최소화
2. Classical Optimizer 사용
3. Parameter Update 과정 확인
4. Variational Learning 이해

"""

import numpy as np

from scipy.optimize import minimize


print("=" * 60)
print("STEP 1. Objective Function 정의")
print("=" * 60)


def objective(theta):

    """
    Objective Function

    Cost(theta)

    = cos(theta)

    """

    cost = np.cos(theta[0])

    print(
        f"Theta : {theta[0]:8.5f}   "
        f"Cost : {cost:8.5f}"
    )

    return cost


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Initial Parameter")
print("=" * 60)

initial_theta = [0.5]

print(initial_theta)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. Optimization 시작")
print("=" * 60)

result = minimize(

    objective,

    initial_theta,

    method="COBYLA"

)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. Optimization 완료")
print("=" * 60)

print(result)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. 결과 확인")
print("=" * 60)

print("Optimal Theta")

print(result.x[0])

print()

print("Minimum Cost")

print(result.fun)


