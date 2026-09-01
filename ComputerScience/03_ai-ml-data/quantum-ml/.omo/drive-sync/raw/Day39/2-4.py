"""
==============================================================
Lab. γ(Parameter) 변화 분석
==============================================================

학습 목표
--------------------------------------------------------------
1. Parameterized QAOA Circuit를 생성한다.
2. γ(Parameter)를 변경한다.
3. β는 고정한다.
4. γ가 Cost Layer(RZZ)에만 적용되는 것을 확인한다.
5. Circuit 구조는 동일하고 Rotation Angle만 변경됨을 확인한다.
6. Measurement 결과가 달라질 수 있음을 확인한다.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

print("=" * 70)
print("STEP 1. γ / β Parameter 생성")
print("=" * 70)

gamma = Parameter("γ")
beta = Parameter("β")

print("γ :", gamma)
print("β :", beta)

print("\n" + "=" * 70)
print("STEP 2. Parameterized QAOA Circuit 생성")
print("=" * 70)

qc = QuantumCircuit(2)

# ---------------------------------------------------
# Initial State
# ---------------------------------------------------

qc.h(0)
qc.h(1)

qc.barrier(label="Initial")

# ---------------------------------------------------
# Cost Layer
# ---------------------------------------------------

qc.rzz(
    2 * gamma,
    0,
    1
)

qc.barrier(label="Cost")

# ---------------------------------------------------
# Mixer Layer
# ---------------------------------------------------

qc.rx(
    2 * beta,
    0
)

qc.rx(
    2 * beta,
    1
)

qc.barrier(label="Mixer")

# ---------------------------------------------------
# Measurement
# ---------------------------------------------------

qc.measure_all()

print(qc.draw(output="text"))

print("\n" + "=" * 70)
print("STEP 3. γ 실험값 설정")
print("=" * 70)

gamma_values = [
    0.0,
    0.2,
    0.5,
    1.0
]

fixed_beta = 0.4

print("γ :", gamma_values)
print("β :", fixed_beta)

print("\n" + "=" * 70)
print("STEP 4. Aer Simulator 준비")
print("=" * 70)

simulator = AerSimulator()

shots = 1024

print("Simulator Ready")

print("\n" + "=" * 70)
print("STEP 5. γ 변화 분석")
print("=" * 70)

for gamma_value in gamma_values:

    print("\n" + "=" * 70)
    print(f"γ = {gamma_value}")
    print("=" * 70)

    bound_qc = qc.assign_parameters(
        {
            gamma: gamma_value,
            beta: fixed_beta
        }
    )

    print(bound_qc.draw(output="text"))

    transpiled = transpile(
        bound_qc,
        simulator
    )

    result = simulator.run(
        transpiled,
        shots=shots
    ).result()

    counts = result.get_counts()

    print("\nMeasurement Counts")

    print(counts)

print("\n" + "=" * 70)
print("STEP 6. γ = 0 특별 실험")
print("=" * 70)

bound_zero = qc.assign_parameters(
    {
        gamma: 0.0,
        beta: fixed_beta
    }
)

print(bound_zero.draw(output="text"))

print("""
γ = 0

↓

RZZ(0)

↓

Cost Layer 영향 없음
""")

print("\n" + "=" * 70)
print("STEP 7. Circuit 구조 비교")
print("=" * 70)

print("""
변하지 않는 것

• Qubit 수
• Gate 종류
• Gate 순서
• Mixer Layer
• Measurement

변하는 것

• RZZ Rotation Angle
• Measurement 결과(Counts)
""")

print("\n" + "=" * 70)
print("STEP 8. γ 변화 기록표")
print("=" * 70)

print("{:<10}{:<15}{:<15}".format(
    "γ",
    "RZZ Angle",
    "관찰"
))

print("-" * 45)

for value in gamma_values:

    angle = 2 * value

    print("{:<10}{:<15}{:<15}".format(
        value,
        angle,
        ""
    ))



print("\n" + "=" * 70)
print("STEP 9. 실습 정리")
print("=" * 70)

print("""
γ

↓

Cost Layer

↓

RZZ Rotation Angle

↓

Phase 변화

↓

Measurement 변화

(Optimizer는 아직 사용하지 않음)
""")