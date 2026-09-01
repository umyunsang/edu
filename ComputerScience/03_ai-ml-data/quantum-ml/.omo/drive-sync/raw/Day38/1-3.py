"""
======================================================================
Lab 3. 첫 번째 QAOA 실행
File : 1-3.py

실습 목표
----------------------------------------------------------------------
1. Optimization Problem 생성
2. QAOA 실행
3. Optimization Result 확인
4. 첫 번째 QAOA 성공적으로 실행
======================================================================
"""

from __future__ import annotations

from qiskit.primitives import StatevectorSampler

from qiskit_optimization import QuadraticProgram

from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
)

from qiskit_optimization.minimum_eigensolvers import (
    QAOA,
)

from qiskit_optimization.optimizers import (
    COBYLA,
)


# ==========================================================
# 출력 함수
# ==========================================================

def print_header(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# Optimization Problem
# ==========================================================

def create_problem() -> QuadraticProgram:

    problem = QuadraticProgram(
        name="First QAOA"
    )

    problem.binary_var("x")

    problem.binary_var("y")

    problem.minimize(

        linear={
            "x": -1,
            "y": -2,
        },

        quadratic={
            ("x", "y"): 2,
        }

    )

    return problem


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    # ------------------------------------------------------
    # STEP 1
    # Optimization Problem
    # ------------------------------------------------------

    print_header(
        "STEP 1. Optimization Problem"
    )

    problem = create_problem()

    print(problem.prettyprint())

    # ------------------------------------------------------
    # STEP 2
    # Sampler
    # ------------------------------------------------------

    print_header(
        "STEP 2. StatevectorSampler"
    )

    sampler = StatevectorSampler()

    print("StatevectorSampler Created")

    # ------------------------------------------------------
    # STEP 3
    # Optimizer
    # ------------------------------------------------------

    print_header(
        "STEP 3. COBYLA Optimizer"
    )

    optimizer = COBYLA(
        maxiter=100
    )

    print("COBYLA Created")

    # ------------------------------------------------------
    # STEP 4
    # QAOA
    # ------------------------------------------------------

    print_header(
        "STEP 4. QAOA"
    )

    qaoa = QAOA(

        sampler=sampler,

        optimizer=optimizer,

        reps=1,

    )

    print("QAOA Created")

    # ------------------------------------------------------
    # STEP 5
    # MinimumEigenOptimizer
    # ------------------------------------------------------

    print_header(
        "STEP 5. MinimumEigenOptimizer"
    )

    solver = MinimumEigenOptimizer(
        qaoa
    )

    print("MinimumEigenOptimizer Created")

    # ------------------------------------------------------
    # STEP 6
    # Solve
    # ------------------------------------------------------

    print_header(
        "STEP 6. Solve"
    )

    result = solver.solve(problem)

    print("Optimization Complete")

    # ------------------------------------------------------
    # STEP 7
    # Result
    # ------------------------------------------------------

    print_header(
        "STEP 7. Optimization Result"
    )

    print(result.prettyprint())

    print()

    print(f"Solution  : {result.x}")

    print(f"Objective : {result.fval}")

    print(f"Status    : {result.status.name}")

    # ------------------------------------------------------
    # STEP 8
    # Finish
    # ------------------------------------------------------

    print_header(
        "STEP 8. Finish"
    )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()