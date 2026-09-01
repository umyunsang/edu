# ============================================================
# Part 2. Cost Hamiltonian 구현
# Lab 3. Circuit 출력
# ============================================================

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# ------------------------------------------------------------
# Step 1. Cost Parameter 생성
# ------------------------------------------------------------

gamma = Parameter("γ")


# ------------------------------------------------------------
# Step 2. 2-Qubit Quantum Circuit 생성
# ------------------------------------------------------------

qc = QuantumCircuit(2)


# ------------------------------------------------------------
# Step 3. Cost Layer 생성
#
# Cost Hamiltonian
# H_C = Z0 Z1
#
# Z0 Z1
#    ↓
# RZZ(2γ)
# ------------------------------------------------------------

qc.rzz(
    2 * gamma,
    0,
    1
)


# ------------------------------------------------------------
# Step 4. Circuit Diagram 출력
# ------------------------------------------------------------

print("=== Cost Layer Circuit ===")
print(qc.draw())


# ------------------------------------------------------------
# Step 5. Circuit Parameter 확인
# ------------------------------------------------------------

print("\n=== Circuit Parameters ===")
print(qc.parameters)


# ------------------------------------------------------------
# Step 6. Gate 개수 확인
# ------------------------------------------------------------

print("\n=== Gate Count ===")
print(qc.count_ops())


# ------------------------------------------------------------
# Step 7. Lab 결과 정리
# ------------------------------------------------------------

print("\n=== Lab 3 Result ===")
print("Qubit Count      : 2")
print("Cost Parameter   : γ")
print("Cost Hamiltonian : Z0 Z1")
print("Cost Gate        : RZZ")
print("Rotation Angle   : 2γ")
print("Applied Qubits   : q0, q1")
print("Circuit Check    : Complete")