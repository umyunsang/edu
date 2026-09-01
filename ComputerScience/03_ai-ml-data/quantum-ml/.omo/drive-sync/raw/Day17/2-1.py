"""
==========================================================
RealAmplitudes Circuit 생성 및 구조 분석
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Statevector

from qiskit_aer import AerSimulator
from qiskit import transpile

from qiskit.visualization import plot_histogram

# ======================================================
# STEP 1
# RealAmplitudes 생성
# ======================================================

print("="*70)
print("STEP 1. RealAmplitudes 생성")
print("="*70)

ansatz = real_amplitudes(
    num_qubits=3,
    reps=2,
    entanglement="linear"
)

print(ansatz)

print()

# ======================================================
# STEP 2
# 회로 시각화
# ======================================================

print("="*70)
print("STEP 2. Circuit Draw")
print("="*70)

figure = ansatz.draw(
    output="mpl",
    fold=-1
)

plt.show()

# ======================================================
# STEP 3
# 회로 기본 정보
# ======================================================

print("="*70)
print("STEP 3. Circuit Information")
print("="*70)

print("Qubit 수")
print(ansatz.num_qubits)

print()

print("Parameter 수")
print(ansatz.num_parameters)

print()

print("Circuit Depth")
print(ansatz.depth())

print()

print("Gate 개수")
print(ansatz.size())

print()

print("Gate 종류")
print(ansatz.count_ops())

print()

# ======================================================
# STEP 4
# Parameter 확인
# ======================================================

print("="*70)
print("STEP 4. Parameter")
print("="*70)

for i, p in enumerate(ansatz.parameters):

    print(f"Parameter {i}")

    print(p)

print()

# ======================================================
# STEP 5
# reps 비교
# ======================================================

print("="*70)
print("STEP 5. reps 비교")
print("="*70)

for reps in [1,2,3]:

    circuit = real_amplitudes(

        num_qubits=3,

        reps=reps

    )

    print("------------------------------")

    print(f"reps = {reps}")

    print("Parameter :", circuit.num_parameters)

    print("Depth     :", circuit.depth())

    print("Gate      :", circuit.size())

print()

# ======================================================
# STEP 6
# Qubit 수 비교
# ======================================================

print("="*70)
print("STEP 6. Qubit 비교")
print("="*70)

for n in [2,3,4]:

    circuit = real_amplitudes(

        num_qubits=n,

        reps=2

    )

    print("------------------------------")

    print(f"Qubit = {n}")

    print("Parameter :", circuit.num_parameters)

    print("Depth     :", circuit.depth())

print()

# ======================================================
# STEP 7
# Entanglement 비교
# ======================================================

print("="*70)
print("STEP 7. Entanglement 비교")
print("="*70)

types = [

    "linear",

    "reverse_linear",

    "circular",

    "full"

]

for ent in types:

    circuit = real_amplitudes(

        num_qubits=4,

        reps=2,

        entanglement=ent

    )

    print("------------------------------")

    print(ent)

    print(circuit.count_ops())

print()

# ======================================================
# STEP 8
# Barrier 사용
# ======================================================

print("="*70)
print("STEP 8. Barrier")
print("="*70)

barrier_circuit = real_amplitudes(

    num_qubits=3,

    reps=2,

    insert_barriers=True

)

barrier_circuit.draw(

    output="mpl",

    fold=-1

)

plt.show()

# ======================================================
# STEP 9
# Parameter 값 생성
# ======================================================

print("="*70)
print("STEP 9. Parameter Assign")
print("="*70)

parameter_values = np.linspace(

    0.1,

    2.0,

    ansatz.num_parameters

)

print(parameter_values)

parameter_map = dict(

    zip(

        ansatz.parameters,

        parameter_values

    )

)

bound = ansatz.assign_parameters(

    parameter_map

)

print()

print("Assign 완료")

print()

# ======================================================
# STEP 10
# Statevector
# ======================================================

print("="*70)
print("STEP 10. Statevector")
print("="*70)

state = Statevector.from_instruction(

    bound

)

print(state)

print()

print("Probability")

print(state.probabilities_dict())

print()

# ======================================================
# STEP 11
# Measurement
# ======================================================

print("="*70)
print("STEP 11. Measurement")
print("="*70)

measure = bound.copy()

measure.measure_all()

sim = AerSimulator()

compiled = transpile(

    measure,

    sim

)

result = sim.run(

    compiled,

    shots=4096

).result()

counts = result.get_counts()

print(counts)

print()

# ======================================================
# STEP 12
# Histogram
# ======================================================

print("="*70)
print("STEP 12. Histogram")
print("="*70)

plot_histogram(

    counts,

    title="RealAmplitudes"

)

plt.show()



print("="*70)
print("실습 종료")
print("="*70)
