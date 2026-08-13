"""
==========================================================
Lab 1. Parameterized Quantum Circuit 만들기
==========================================================

학습 목표
1. Parameter 객체 생성
2. Parameterized Quantum Circuit 생성
3. Parameter 확인
4. Quantum Circuit 구조 확인

"""

print("=" * 60)
print("STEP 1. Import")
print("=" * 60)

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


print("\nImport 완료")


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2. Parameter 생성")
print("=" * 60)

theta = Parameter("θ")

print("Parameter")
print(theta)

print("\nParameter Type")
print(type(theta))


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3. Quantum Circuit 생성")
print("=" * 60)

qc = QuantumCircuit(1)

print("Qubit 수 :", qc.num_qubits)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4. Parameterized Gate 추가")
print("=" * 60)

qc.ry(theta, 0)

print("RY(θ) Gate 추가 완료")


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5. Parameter 확인")
print("=" * 60)

print("Circuit Parameters")

for parameter in qc.parameters:

    print(parameter)


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6. Quantum Circuit")
print("=" * 60)

print(qc.draw())


# ----------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7. Circuit 정보")
print("=" * 60)

print(qc)

print("\nDepth")
print(qc.depth())

print("\nSize")
print(qc.size())

print("\nWidth")
print(qc.width())


