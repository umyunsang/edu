"""
==========================================================
추가 실험 3.
Objective Function 변경
==========================================================

학습 목표

1. Objective Function 변경
2. 최적 Parameter 비교
3. Cost Surface 변화 확인
4. QAOA Cost Hamiltonian 이해

"""

import numpy as np

from scipy.optimize import minimize


print("=" * 60)
print("STEP 1. Objective Function 정의")
print("=" * 60)


# ----------------------------------------------------------
# Objective Function 1
# ----------------------------------------------------------

def objective_1(theta):

    """
    Original

    Minimum

    theta = pi

    """

    return np.cos(theta[0])


# ----------------------------------------------------------
# Objective Function 2
# ----------------------------------------------------------

def objective_2(theta):

    """
    Shifted Cost

    """

    return (np.cos(theta[0]) + 0.5) ** 2


# ----------------------------------------------------------
# Objective Function 3
# ----------------------------------------------------------

def objective_3(theta):

    """
    Different Landscape

    """

    return np.sin(theta[0]) ** 2


objective_functions = [

    ("Cos(theta)", objective_1),

    ("(Cos(theta)+0.5)^2", objective_2),

    ("Sin(theta)^2", objective_3)

]


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Optimization")
print("=" * 60)

initial_theta = [0.5]

results = []

for name, function in objective_functions:

    evaluation = [0]

    def objective(theta):

        evaluation[0] += 1

        return function(theta)

    result = minimize(

        objective,

        initial_theta,

        method="COBYLA"

    )

    results.append({

        "name": name,

        "theta": result.x[0],

        "cost": result.fun,

        "evaluation": evaluation[0]

    })


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. 결과 출력")
print("=" * 60)

print(

    "{:<22}{:<15}{:<15}{:<10}".format(

        "Objective",

        "Optimal Theta",

        "Minimum Cost",

        "Eval"

    )

)

print("-" * 70)

for item in results:

    print(

        "{:<22}{:<15.6f}{:<15.6f}{:<10}".format(

            item["name"],

            item["theta"],

            item["cost"],

            item["evaluation"]

        )

    )


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. 결과 비교")
print("=" * 60)

for item in results:

    print()

    print("Objective Function")

    print(item["name"])

    print("Optimal Theta")

    print(item["theta"])

    print("Minimum Cost")

    print(item["cost"])


# ----------------------------------------------------------
