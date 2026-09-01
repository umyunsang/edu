"""
==========================================================
Lab. Mixer Hamiltonian을 Mixer Layer로 구현하기
==========================================================

학습 목표
----------------------------------------------------------
1. Mixer Hamiltonian을 RX Gate로 구현한다.
2. β(Parameter)를 생성한다.
3. 모든 Qubit에 Mixer Layer를 적용한다.
4. β가 Rotation Angle에 반영되는 위치를 확인한다.
5. Cost Layer와 Mixer Layer를 하나의 Circuit로 연결한다.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

print("=" * 70)
print("STEP 1. β(Parameter) 생성")
print("=" * 70)

beta = Parameter("β")

print(f"Parameter : {beta}")

print("\n" + "=" * 70)
print("STEP 2. Mixer Hamiltonian 확인")
print("=" * 70)

print("Mixer Hamiltonian")
print("Hm = X0 + X1")

print("\nHamiltonian")
print("        ↓")
print("Mixer Unitary")
print("        ↓")
print("RX Gate")
print("        ↓")
print("Mixer Layer")

print("\n" + "=" * 70)
print("STEP 3. Quantum Circuit 생성")
print("=" * 70)

num_qubits = 2

qc = QuantumCircuit(num_qubits)

print(qc.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 4. Mixer Layer 시작")
print("=" * 70)

qc.barrier(label="Mixer Layer")

print("Barrier 추가 완료")

print("\n" + "=" * 70)
print("STEP 5. 모든 Qubit에 RX Gate 추가")
print("=" * 70)

for qubit in range(num_qubits):

    qc.rx(
        2 * beta,
        qubit
    )

    print(f"Qubit {qubit} → RX(2β) 추가")

print("\n" + "=" * 70)
print("STEP 6. Mixer Layer 종료")
print("=" * 70)

qc.barrier(label="End Mixer")

print("Barrier 추가 완료")

print("\n" + "=" * 70)
print("STEP 7. Parameterized Mixer Layer")
print("=" * 70)

print(qc.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 8. β Binding")
print("=" * 70)

beta_values = [
    0.0,
    0.2,
    0.5,
    1.0
]

for beta_value in beta_values:

    print("\n" + "=" * 70)
    print(f"β = {beta_value}")
    print("=" * 70)

    bound_qc = qc.assign_parameters(
        {
            beta: beta_value
        }
    )

    print(bound_qc.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 9. β = 0 실험")
print("=" * 70)

zero_qc = qc.assign_parameters(
    {
        beta: 0.0
    }
)

print(zero_qc.draw(output="text"))

print("\nβ = 0")
print("→ RX(0)")
print("→ Mixer Layer가 상태를 변화시키지 않음")

print("\n" + "=" * 70)
print("STEP 10. 3-Qubit Mixer Layer 생성")
print("=" * 70)

qc3 = QuantumCircuit(3)

for qubit in range(3):

    qc3.rx(
        2 * beta,
        qubit
    )

print(qc3.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 11. Mixer Layer 함수 구현")
print("=" * 70)


def add_mixer_layer(circuit, beta_parameter):

    circuit.barrier(label="Mixer")

    for qubit in range(circuit.num_qubits):
        circuit.rx(
            2 * beta_parameter,
            qubit
        )

    return circuit


qc_func = QuantumCircuit(3)

add_mixer_layer(
    qc_func,
    beta
)

print(qc_func.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 12. Cost Layer + Mixer Layer")
print("=" * 70)

gamma = Parameter("γ")

qaoa = QuantumCircuit(2)

# ------------------------------------------------
# Initial State
# ------------------------------------------------

qaoa.h(0)
qaoa.h(1)

qaoa.barrier(label="Initial")

# ------------------------------------------------
# Cost Layer
# ------------------------------------------------

qaoa.rzz(
    2 * gamma,
    0,
    1
)

qaoa.barrier(label="Cost")

# ------------------------------------------------
# Mixer Layer
# ------------------------------------------------

for qubit in range(2):

    qaoa.rx(
        2 * beta,
        qubit
    )

qaoa.barrier(label="Mixer")

print(qaoa.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 13. γ, β 동시 Binding")
print("=" * 70)

bound_qaoa = qaoa.assign_parameters(
    {
        gamma: 0.6,
        beta: 0.4
    }
)

print(bound_qaoa.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 14. 실습 정리")
print("=" * 70)

print("Initial State")
print("      ↓")
print("Cost Layer")
print("      ↓")
print("Mixer Layer")
print("      ↓")
print("QAOA Layer")

print("\n이번 실습에서는")
print("• Mixer Hamiltonian → RX Gate")
print("• β → Rotation Angle")
print("• 모든 Qubit에 동일한 Mixer 적용")
print("• Cost Layer + Mixer Layer 결합")
print("을 확인하였다.")