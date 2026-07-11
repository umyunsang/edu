from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --------------------------
# Original Feature
# --------------------------

age = 0.8
income = 1.5

# --------------------------
# Quantum Circuit
# --------------------------

qc = QuantumCircuit(2,2)

# --------------------------
# Feature Encoding
# --------------------------

qc.ry(age,0)
qc.ry(income,1)

# --------------------------
# Feature Interaction
# --------------------------

qc.cx(0,1)

# --------------------------
# Feature Transformation
# --------------------------

qc.rz(age*income,1)

# --------------------------
# Measurement
# --------------------------

qc.barrier()

qc.measure([0,1],[0,1])

# --------------------------
# Circuit
# --------------------------

print(qc)

qc.draw("mpl")

# --------------------------
# Simulation
# --------------------------

sim = AerSimulator()

job = sim.run(qc,shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)
plt.show()
