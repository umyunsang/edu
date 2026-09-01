"""
======================================================================
Lab 2. QAOA Import 확인
File : 02_import.py

실습 목표
----------------------------------------------------------------------
1. QAOA 실습에 필요한 모듈 Import
2. Import 성공 여부 확인
3. 기본 객체 생성 확인
4. 다음 실습(QAOA 실행) 준비
======================================================================
"""

from __future__ import annotations


# ==============================================================
# STEP 출력 함수
# ==============================================================

def print_header(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==============================================================
# STEP 1
# Import
# ==============================================================

print_header("STEP 1. Import Module")

from qiskit_optimization import QuadraticProgram

from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
)

from qiskit.primitives import (
    StatevectorSampler,
)

from qiskit_optimization.minimum_eigensolvers import (
    QAOA,
)

from qiskit_optimization.optimizers import (
    COBYLA,
)

print("All modules imported successfully.")


# ==============================================================
# STEP 2
# Import 확인
# ==============================================================

print_header("STEP 2. Imported Classes")

print(f"{'QuadraticProgram':<30} : OK")
print(f"{'MinimumEigenOptimizer':<30} : OK")
print(f"{'StatevectorSampler':<30} : OK")
print(f"{'QAOA':<30} : OK")
print(f"{'COBYLA':<30} : OK")


# ==============================================================
# STEP 3
# 기본 객체 생성
# ==============================================================

print_header("STEP 3. Object Creation")

problem = QuadraticProgram()

sampler = StatevectorSampler()

optimizer = COBYLA()

qaoa = QAOA(
    sampler=sampler,
    optimizer=optimizer,
)

print("QuadraticProgram Created")
print("StatevectorSampler Created")
print("COBYLA Created")
print("QAOA Object Created")


# ==============================================================
# STEP 4
# 객체 타입 확인
# ==============================================================

print_header("STEP 4. Object Type")

print(f"Problem   : {type(problem).__name__}")
print(f"Sampler   : {type(sampler).__name__}")
print(f"Optimizer : {type(optimizer).__name__}")
print(f"QAOA      : {type(qaoa).__name__}")


# ==============================================================
# STEP 5
# 객체 역할 확인
# ==============================================================

print_header("STEP 5. Component Role")

roles = [

    (
        "QuadraticProgram",
        "Optimization Problem 정의",
    ),

    (
        "StatevectorSampler",
        "Quantum Circuit 실행",
    ),

    (
        "COBYLA",
        "Classical Parameter 최적화",
    ),

    (
        "QAOA",
        "Hybrid Quantum Optimization",
    ),

    (
        "MinimumEigenOptimizer",
        "Optimization Problem과 QAOA 연결",
    ),

]

for name, role in roles:

    print(f"{name:<30} -> {role}")


# ==============================================================
# STEP 6
# 실행 준비 확인
# ==============================================================

print_header("STEP 6. Ready for Next Lab")

print("Environment              : READY")
print("Import                   : READY")
print("Object Creation          : READY")
print("QAOA Execution           : NOT YET")

print("\nNext Step")
print("-> 1-3.py")
print("-> First QAOA Execution")