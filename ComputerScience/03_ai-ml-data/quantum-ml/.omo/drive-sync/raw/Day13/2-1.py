import numpy as np

from qiskit import transpile

from qiskit.circuit.library import (
    zz_feature_map,
    real_amplitudes
)

from qiskit_aer import AerSimulator

from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt


# ======================================================
# STEP 1.
# Classical Data
# ======================================================

print("=" * 70)
print("STEP 1. Classical Data")
print("=" * 70)

x = np.array([0.3, 0.7])

print("Input Feature")
print(x)
print()


# ======================================================
# STEP 2.
# Feature Map 생성
# ======================================================

print("=" * 70)
print("STEP 2. Feature Map")
print("=" * 70)

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=1
)

print(feature_map)

print()

feature_map.draw("mpl")
plt.show()


# ======================================================
# STEP 3.
# Variational Circuit 생성
# ======================================================

print("=" * 70)
print("STEP 3. Variational Circuit")
print("=" * 70)

ansatz = real_amplitudes(
    num_qubits=2,
    reps=1
)

print(ansatz)

print()

ansatz.draw("mpl")
plt.show()


# ======================================================
# STEP 4.
# Circuit 결합
# ======================================================

print("=" * 70)
print("STEP 4. Compose Circuit")
print("=" * 70)

qml_circuit = feature_map.compose(ansatz)

print(qml_circuit)

print()

qml_circuit.draw("mpl")
plt.show()


# ======================================================
# STEP 5.
# Measurement 추가
# ======================================================

print("=" * 70)
print("STEP 5. Measurement")
print("=" * 70)

qml_circuit.measure_all()

qml_circuit.draw("mpl")
plt.show()


# ======================================================
# STEP 6.
# Parameter 확인
# ======================================================

print("=" * 70)
print("STEP 6. Parameter")
print("=" * 70)

print("Parameter List")

for p in qml_circuit.parameters:
    print(p)

print()

print("Parameter Count :", len(qml_circuit.parameters))

print()


# ======================================================
# STEP 7.
# Parameter Binding
# ======================================================

print("=" * 70)
print("STEP 7. Parameter Binding")
print("=" * 70)

parameter_values = {}

for p in qml_circuit.parameters:

    value = np.random.uniform(0, np.pi)

    parameter_values[p] = value

    print(f"{p!s:15} -> {value:.3f}")

print()

bound_circuit = qml_circuit.assign_parameters(
    parameter_values
)


# ======================================================
# STEP 8.
# Simulator 실행
# ======================================================

print("=" * 70)
print("STEP 8. Simulation")
print("=" * 70)

simulator = AerSimulator()

compiled = transpile(
    bound_circuit,
    simulator
)

job = simulator.run(
    compiled,
    shots=1024
)

result = job.result()

counts = result.get_counts()

print(counts)

print()


# ======================================================
# STEP 9.
# Histogram
# ======================================================

print("=" * 70)
print("STEP 9. Measurement Result")
print("=" * 70)

plot_histogram(counts)

plt.show()



