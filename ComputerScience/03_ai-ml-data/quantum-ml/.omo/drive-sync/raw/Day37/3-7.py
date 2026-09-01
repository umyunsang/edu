"""
==========================================================
추가 실험 1.
초기 Parameter 변경
==========================================================

학습 목표

1. 초기 Parameter 변경
2. Optimization 과정 비교
3. Evaluation 횟수 비교
4. Initialization 영향 분석

"""

import numpy as np

from scipy.optimize import minimize


print("=" * 60)
print("STEP 1. Initial Parameter 목록")
print("=" * 60)

initial_parameters = [

    0.1,
    1.0,
    2.0,
    4.0,
    5.5

]

for value in initial_parameters:

    print(value)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Objective Function")
print("=" * 60)


def objective(theta):

    return np.cos(theta[0])


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. Optimization")
print("=" * 60)

results = []

for initial_theta in initial_parameters:

    history = []

    def objective_with_history(theta):

        cost = np.cos(theta[0])

        history.append(cost)

        return cost

    result = minimize(

        objective_with_history,

        [initial_theta],

        method="COBYLA"

    )

    results.append({

        "initial": initial_theta,

        "optimal": result.x[0],

        "cost": result.fun,

        "evaluation": len(history)

    })


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. 결과 출력")
print("=" * 60)

print(
    "{:<12}{:<15}{:<15}{:<15}".format(

        "Initial",

        "Optimal",

        "Cost",

        "Eval"

    )
)

print("-" * 60)

for item in results:

    print(

        "{:<12.2f}{:<15.6f}{:<15.6f}{:<15}".format(

            item["initial"],

            item["optimal"],

            item["cost"],

            item["evaluation"]

        )

    )


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. 결과 분석")
print("=" * 60)

best = min(results, key=lambda x: x["evaluation"])

print("가장 적은 Evaluation")

print(best)

print()

worst = max(results, key=lambda x: x["evaluation"])

print("가장 많은 Evaluation")

print(worst)


# ----------------------------------------------------------
