"""
==========================================================
추가 실험 2.
Optimizer 변경
==========================================================

학습 목표

1. 다양한 Optimizer 사용
2. Optimization 결과 비교
3. Evaluation 횟수 비교
4. Optimizer 특성 이해

"""

import time
import numpy as np

from scipy.optimize import minimize


print("=" * 60)
print("STEP 1. Optimizer 목록")
print("=" * 60)

optimizers = [

    "COBYLA",
    "Nelder-Mead",
    "Powell",
    "BFGS"

]

for optimizer in optimizers:

    print(optimizer)


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

initial_theta = [0.5]

for optimizer in optimizers:

    evaluation_count = 0

    def objective_count(theta):

        nonlocal_evaluation[0] += 1

        return np.cos(theta[0])

    nonlocal_evaluation = [0]

    start = time.perf_counter()

    result = minimize(

        objective_count,

        initial_theta,

        method=optimizer

    )

    end = time.perf_counter()

    results.append({

        "optimizer": optimizer,

        "theta": result.x[0],

        "cost": result.fun,

        "evaluation": nonlocal_evaluation[0],

        "time": end - start

    })


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. 결과 출력")
print("=" * 60)

print(
    "{:<15}{:<12}{:<12}{:<12}{:<12}".format(

        "Optimizer",

        "Theta",

        "Cost",

        "Eval",

        "Time(s)"

    )
)

print("-" * 70)

for item in results:

    print(

        "{:<15}{:<12.6f}{:<12.6f}{:<12}{:<12.5f}".format(

            item["optimizer"],

            item["theta"],

            item["cost"],

            item["evaluation"],

            item["time"]

        )

    )


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. 가장 적은 Evaluation")
print("=" * 60)

best = min(

    results,

    key=lambda x: x["evaluation"]

)

print(best)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6. 가장 빠른 Optimizer")
print("=" * 60)

fastest = min(

    results,

    key=lambda x: x["time"]

)

print(fastest)


