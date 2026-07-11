# ==========================================================
# Amplitude Encoding
# ==========================================================

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

# ==========================================================
# STEP 2. Input Data
# ==========================================================

customer = np.array([
    35,      # Age
    5200,    # Income
    12,      # Purchase Count
    40       # Visit Count
], dtype=float)

print("=" * 60)
print("STEP 2. Input Data")
print("=" * 60)

print(customer)

# ==========================================================
# STEP 3. Feature Scaling
# ==========================================================

scaled = customer / np.max(customer)

print("=" * 60)
print("STEP 3. Feature Scaling")
print("=" * 60)

print(scaled)

# ==========================================================
# STEP 4. Vector Length
# ==========================================================

length = len(scaled)

print("=" * 60)
print("STEP 4. Vector Length")
print("=" * 60)

print("Vector Length :", length)

qubits = int(np.log2(length))

print("Required Qubits :", qubits)

# ==========================================================
# STEP 5. Padding Function
# ==========================================================

def padding(x):

    length = len(x)

    next_power = 2 ** int(np.ceil(np.log2(length)))

    padded = np.zeros(next_power)

    padded[:length] = x

    return padded

# ==========================================================
# STEP 6. Normalization
# ==========================================================

norm = np.linalg.norm(scaled)

normalized = scaled / norm

print("=" * 60)
print("STEP 6. Normalization")
print("=" * 60)

print(normalized)

print()

print("Norm Check")

print(np.sum(np.abs(normalized) ** 2))


# ==========================================================
# STEP 7. Statevector
# ==========================================================

state = Statevector(normalized)

print("=" * 60)
print("STEP 7. Statevector")
print("=" * 60)

print(state)

# ==========================================================
# STEP 8. Amplitude Analysis
# ==========================================================

basis = [
    "|00>",
    "|01>",
    "|10>",
    "|11>"
]

print("=" * 60)
print("STEP 8. Amplitude")
print("=" * 60)

for b, amp in zip(basis, state.data):

    print(
        f"{b:5}"
        f"Amplitude : {amp.real:.6f}"
    )

# ==========================================================
# STEP 9. Probability
# ==========================================================

prob = np.abs(state.data) ** 2

print("=" * 60)
print("STEP 9. Probability")
print("=" * 60)

for b, p in zip(basis, prob):

    print(f"{b:5} {p:.6f}")

# ==========================================================
# STEP 10. Circuit
# ==========================================================

qc = QuantumCircuit(qubits)

qc.initialize(
    normalized,
    qc.qubits
)

print("=" * 60)
print("STEP 10. Circuit")
print("=" * 60)

print(qc)

# ==========================================================
# STEP 11. Measurement
# ==========================================================

qc.measure_all()

qc.draw(
    "mpl",
    fold=-1
)

# ==========================================================
# STEP 12. Simulation
# ==========================================================

simulator = AerSimulator()

compiled = transpile(
    qc,
    simulator
)

result = simulator.run(
    compiled,
    shots=1024
).result()

counts = result.get_counts()

print("=" * 60)
print("STEP 12. Counts")
print("=" * 60)

print(counts)

# ==========================================================
# STEP 13. Histogram
# ==========================================================

plot_histogram(counts)

plt.show()




# ==========================================================
# STEP 14. Circuit Complexity
# ==========================================================

optimized = transpile(
    qc,
    basis_gates=["u", "cx"]
)

print("=" * 60)
print("STEP 15. Circuit Complexity")
print("=" * 60)

print("Depth")

print(optimized.depth())

print()

print("Gate Count")

print(
    optimized.count_ops()
)




