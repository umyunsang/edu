"""
============================================================
Parameter 변화 실험 (Part 1)

STEP 1. 실험 조건 정의
STEP 2. Parameterized Circuit 생성
STEP 3. Parameter Sweep
STEP 4. Statevector 분석
STEP 5. Probability 계산
STEP 6. Measurement 수행
============================================================
"""

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

try:
    from qiskit_aer import AerSimulator
    simulator = AerSimulator()
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False

print("=" * 80)
print("STEP 1. 실험 조건 정의")
print("=" * 80)

angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

print("Parameter 목록")
for angle in angles:
    print(f"{angle:.4f} rad")

print("\\n" + "=" * 80)
print("STEP 2. Parameterized Circuit 생성")
print("=" * 80)

theta = Parameter("θ")
qc = QuantumCircuit(1)
qc.ry(theta, 0)
print(qc.draw())

print("\\n" + "=" * 80)
print("STEP 3. Parameter Sweep")
print("=" * 80)

bound_circuits = []
for angle in angles:
    bound = qc.assign_parameters({theta: angle})
    bound_circuits.append(bound)
    print("-"*60)
    print(f"θ = {angle:.4f}")
    print(bound.draw())

print("\\n" + "=" * 80)
print("STEP 4. Quantum State 분석")
print("=" * 80)

statevectors = []
for angle in angles:
    bound = qc.assign_parameters({theta: angle})
    state = Statevector.from_instruction(bound)
    statevectors.append(state)
    print("-"*60)
    print(f"θ = {angle:.4f}")
    print(state)

print("\\n" + "=" * 80)
print("STEP 5. Probability 계산")
print("=" * 80)

probability_table = []
for angle in angles:
    bound = qc.assign_parameters({theta: angle})
    probs = Statevector.from_instruction(bound).probabilities()
    probability_table.append([angle, probs[0], probs[1]])
    print("-"*60)
    print(f"θ = {angle:.4f}")
    print(f"P(0) = {probs[0]:.4f}")
    print(f"P(1) = {probs[1]:.4f}")

print("\\n" + "=" * 80)
print("STEP 6. Measurement 수행")
print("=" * 80)

counts_list = []
if AER_AVAILABLE:
    for angle in angles:
        bound = qc.assign_parameters({theta: angle})
        measure = bound.copy()
        measure.measure_all()
        compiled = transpile(measure, simulator)
        result = simulator.run(compiled, shots=2048).result()
        counts = result.get_counts()
        counts_list.append(counts)
        print("-"*60)
        print(f"θ = {angle:.4f}")
        print(counts)
else:
    print("AerSimulator가 설치되어 있지 않습니다.")

print("\\n" + "=" * 80)
print("Probability Summary")
print("=" * 80)

df = pd.DataFrame(probability_table, columns=["Theta(rad)", "P(0)", "P(1)"])
print(df)


"""
============================================================
Parameter 변화 실험 (Part 2)

STEP 7. Histogram 비교
STEP 8. Quantum State 변화 분석
STEP 9. Prediction 연결
STEP 10. Machine Learning 연결
STEP 11. 결과 분석
============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

print("\n" + "=" * 80)
print("STEP 7. Histogram 비교")
print("=" * 80)

if AER_AVAILABLE:
    for angle, counts in zip(angles, counts_list):
        print(f"\nθ = {angle:.4f}")
        fig = plot_histogram(counts, title=f"Theta = {angle:.2f}")
        plt.show()
else:
    print("AerSimulator가 없습니다.")

print("\n" + "=" * 80)
print("STEP 8. Quantum State 변화 분석")
print("=" * 80)

for angle in angles:
    bound = qc.assign_parameters({theta: angle})
    state = Statevector.from_instruction(bound)
    probs = state.probabilities()

    print("-" * 60)
    print(f"θ = {angle:.4f}")
    print(state)
    print(f"P(0) = {probs[0]:.4f}")
    print(f"P(1) = {probs[1]:.4f}")

    if probs[0] > probs[1]:
        print("→ |0> 방향에 가까운 상태")
    elif probs[1] > probs[0]:
        print("→ |1> 방향에 가까운 상태")
    else:
        print("→ Superposition 상태")

print("\n" + "=" * 80)
print("STEP 9. Prediction 연결")
print("=" * 80)

prediction_table = []

for angle in angles:
    bound = qc.assign_parameters({theta: angle})
    probs = Statevector.from_instruction(bound).probabilities()

    prediction = 1 if probs[1] >= 0.5 else 0

    prediction_table.append(
        [angle, probs[0], probs[1], prediction]
    )

prediction_df = pd.DataFrame(
    prediction_table,
    columns=["Theta(rad)", "P(0)", "P(1)", "Prediction"]
)

print(prediction_df)

print("\n" + "=" * 80)
print("STEP 10. Machine Learning 연결")
print("=" * 80)

print("""
Machine Learning

Weight
   ↓
Prediction

Quantum Machine Learning

Parameter θ
      ↓
Quantum State
      ↓
Measurement
      ↓
Prediction
""")

for row in prediction_table:
    print(f"Theta={row[0]:.3f} -> Prediction={row[3]}")

plt.figure(figsize=(8,5))
plt.plot(df["Theta(rad)"], df["P(0)"], marker="o", label="P(0)")
plt.plot(df["Theta(rad)"], df["P(1)"], marker="s", label="P(1)")
plt.xlabel("Theta (rad)")
plt.ylabel("Probability")
plt.title("Probability Change")
plt.grid(True)
plt.legend()
plt.show()

print("\n" + "=" * 80)
print("STEP 11. 결과 분석")
print("=" * 80)
