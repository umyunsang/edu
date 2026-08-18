"""
Lab 4.
첫 번째 QAOA 실행

실습 목표
--------------------------------------------------------
1. Optimization Problem 생성
2. Classical Enumeration 수행
3. QAOA 생성
4. Optimization 실행
5. Solution 비교
"""

from __future__ import annotations

from itertools import product

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
from qiskit_optimization.utils import (
    algorithm_globals,
)


# ============================================================
# STEP 출력
# ============================================================

def print_header(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Optimization Problem 생성
# ============================================================

def create_problem() -> QuadraticProgram:

    problem = QuadraticProgram(
        name="First QAOA Problem"
    )

    problem.binary_var(name="x")
    problem.binary_var(name="y")

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


# ============================================================
# 목적 함수
# ============================================================

def objective_function(
        x: int,
        y: int,
) -> float:

    return -x - (2 * y) + (2 * x * y)


# ============================================================
# Classical Enumeration
# ============================================================

def solve_classically():

    print_header(
        "STEP 1. Classical Enumeration"
    )

    best_solution = None

    best_value = float("inf")

    for x, y in product(
            [0, 1],
            repeat=2,
    ):

        value = objective_function(x, y)

        print(
            f"x={x}, "
            f"y={y}, "
            f"Objective={value:.1f}"
        )

        if value < best_value:

            best_value = value

            best_solution = (x, y)

    print()

    print(
        f"Best Solution : {best_solution}"
    )

    print(
        f"Objective     : {best_value:.1f}"
    )

    return best_solution, best_value


# ============================================================
# QAOA Solver
# ============================================================

def solve_with_qaoa(
        problem: QuadraticProgram,
):

    print_header(
        "STEP 2. QAOA 생성"
    )

    algorithm_globals.random_seed = 42

    sampler = StatevectorSampler(
        seed=42
    )

    optimizer = COBYLA(
        maxiter=100
    )

    qaoa = QAOA(

        sampler=sampler,

        optimizer=optimizer,

        reps=1,

        initial_point=[
            0.5,
            0.5,
        ],

    )

    solver = MinimumEigenOptimizer(
        qaoa
    )

    print("Sampler    : StatevectorSampler")

    print("Optimizer  : COBYLA")

    print("QAOA Depth : 1")

    print("Initial γβ : [0.5, 0.5]")

    return solver.solve(problem)


# ============================================================
# Solution Sample 출력
# ============================================================

def print_solution_samples(
        result,
):

    print_header(
        "STEP 3. Solution Samples"
    )

    print(
        "Variable Order :",
        [
            variable.name
            for variable
            in result.variables
        ]
    )

    print()

    for sample in result.samples:

        values = [
            int(v)
            for v
            in sample.x
        ]

        print(

            f"Solution    : {values}"

        )

        print(

            f"Objective   : {sample.fval:.4f}"

        )

        print(

            f"Probability : {sample.probability:.4f}"

        )

        print(

            f"Status      : {sample.status.name}"

        )

        print("-" * 50)


# ============================================================
# 결과 비교
# ============================================================

def compare_result(

        classical_solution,
        classical_value,
        result,

):

    print_header(
        "STEP 4. Result Comparison"
    )

    qaoa_solution = tuple(

        int(v)

        for v

        in result.x

    )

    qaoa_value = float(
        result.fval
    )

    print(
        f"Classical Solution : {classical_solution}"
    )

    print(
        f"QAOA Solution      : {qaoa_solution}"
    )

    print()

    print(
        f"Classical Value : {classical_value:.1f}"
    )

    print(
        f"QAOA Value      : {qaoa_value:.1f}"
    )

    print()

    print(
        "Same Solution :",
        qaoa_solution == classical_solution,
    )

    print(
        "Same Objective :",
        abs(
            qaoa_value
            - classical_value
        ) < 1e-9,
    )


# ============================================================
# Main
# ============================================================

def main():

    print_header(
        "Optimization Problem"
    )

    problem = create_problem()

    print(
        problem.prettyprint()
    )

    classical_solution, classical_value = (
        solve_classically()
    )

    print_header(
        "STEP 2. QAOA Execution"
    )

    result = solve_with_qaoa(
        problem
    )

    print()

    print(
        result.prettyprint()
    )

    print_solution_samples(
        result
    )

    compare_result(

        classical_solution,

        classical_value,

        result,

    )


if __name__ == "__main__":

    main()