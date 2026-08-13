"""
==========================================================
Lab 2. Parameter를 직접 변화시켜 보기
==========================================================

학습 목표

1. Parameter Binding 이해
2. 여러 θ 값을 Circuit에 적용
3. Circuit 구조 비교
4. Parameter가 Quantum Circuit를 변경하는 과정 이해

"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

print("=" * 60)
print("STEP 1. Parameter 생성")
print("=" * 60)

theta = Parameter("θ")

print(theta)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Parameterized Circuit 생성")
print("=" * 60)

qc = QuantumCircuit(1)

qc.ry(theta, 0)

print(qc.draw())


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. 사용할 Parameter 값")
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
print("STEP 4. Parameter Binding")
print("=" * 60)

for i, value in enumerate(parameter_values):

    print(f"\nCase {i + 1}")
    print("-" * 40)

    print(f"Theta : {value:.6f} rad")

    bound_qc = qc.assign_parameters(
        {
            theta: value
        }
    )

    print(bound_qc.draw())


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. Degree 확인")
print("=" * 60)

for value in parameter_values:

    degree = np.degrees(value)

    print(
        f"{value:.6f} rad  ->  {degree:.1f} degree"
    )


