"""
==============================================================
Lab 1. 개발 환경 확인
File : 01_environment.py

실습 목표
--------------------------------------------------------------
1. Python 실행 환경 확인
2. Qiskit 설치 여부 확인
3. QAOA 관련 패키지 확인
4. 필수 모듈 Import 확인
5. 실습 환경 준비 완료 여부 확인
==============================================================
"""

from __future__ import annotations

import platform
import sys


# ==========================================================
# 출력 함수
# ==========================================================

def print_title(title: str) -> None:

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# STEP 1
# Python 환경 확인
# ==========================================================

def check_python_environment() -> None:

    print_title("STEP 1. Python Environment")

    print(f"Python Version : {platform.python_version()}")
    print(f"Python Path    : {sys.executable}")
    print(f"Platform       : {platform.system()} ({platform.machine()})")


# ==========================================================
# STEP 2
# Qiskit 설치 확인
# ==========================================================

def check_qiskit() -> None:

    print_title("STEP 2. Qiskit")

    try:

        import qiskit

        print(f"Qiskit Version : {qiskit.__version__}")
        print("Status          : OK")

    except Exception as error:

        print("Status          : FAIL")
        print(error)
        raise


# ==========================================================
# STEP 3
# Algorithms 확인
# ==========================================================

def check_algorithms() -> None:

    print_title("STEP 3. Qiskit Algorithms")

    try:

        import qiskit_algorithms

        print(
            f"Algorithms Version : "
            f"{qiskit_algorithms.__version__}"
        )

        print("Status              : OK")

    except Exception as error:

        print("Status              : FAIL")
        print(error)
        raise


# ==========================================================
# STEP 4
# Optimization 확인
# ==========================================================

def check_optimization() -> None:

    print_title("STEP 4. Qiskit Optimization")

    try:

        import qiskit_optimization

        print(
            f"Optimization Version : "
            f"{qiskit_optimization.__version__}"
        )

        print("Status               : OK")

    except Exception as error:

        print("Status               : FAIL")
        print(error)
        raise


# ==========================================================
# STEP 5
# 필수 모듈 Import 확인
# ==========================================================

def check_required_modules() -> None:

    print_title("STEP 5. Required Module Import")

    modules = [

        (
            "QuadraticProgram",
            "from qiskit_optimization import QuadraticProgram",
        ),

        (
            "MinimumEigenOptimizer",
            "from qiskit_optimization.algorithms import "
            "MinimumEigenOptimizer",
        ),

        (
            "StatevectorSampler",
            "from qiskit.primitives import StatevectorSampler",
        ),

        (
            "QAOA",
            "from qiskit_optimization.minimum_eigensolvers import QAOA",
        ),

        (
            "COBYLA",
            "from qiskit_optimization.optimizers import COBYLA",
        ),

    ]

    for name, statement in modules:

        try:

            exec(statement)

            print(f"{name:<25} : OK")

        except Exception as error:

            print(f"{name:<25} : FAIL")

            print(error)

            raise


# ==========================================================
# STEP 6
# 프로젝트 구조 안내
# ==========================================================

def print_project_structure() -> None:

    print_title("STEP 6. Project Structure")

    print(
        """
QAOA/

│
├── 1-1.py
├── 1-2.py
├── 1-3.py
├── requirements.txt
└── results/
"""
    )


# ==========================================================
# STEP 7
# 환경 점검 결과
# ==========================================================

def print_summary() -> None:

    print_title("SUMMARY")

    print("Python Environment           : READY")
    print("Qiskit                       : READY")
    print("Qiskit Algorithms            : READY")
    print("Qiskit Optimization          : READY")
    print("Required Module Import       : READY")

    print("\nEnvironment Check Complete.")
    print("You are ready to run QAOA!")


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    check_python_environment()

    check_qiskit()

    check_algorithms()

    check_optimization()

    check_required_modules()

    print_project_structure()

    print_summary()


if __name__ == "__main__":

    main()