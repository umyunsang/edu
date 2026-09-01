from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt
from math import pi

qc = QuantumCircuit(4)

for qubit in range(4):
    qc.h(qubit)

qc.rx(pi/4, 0)

qc.ry(pi/3, 1)

qc.rz(pi/5, 2)

qc.rx(pi/6, 3)

qc.cx(0,1)

qc.cx(1,2)

qc.cx(2,3)

qc.ry(pi/8,0)

qc.rz(pi/7,1)

qc.rx(pi/9,2)

qc.ry(pi/10,3)

qc.cx(0,2)

qc.cx(1,3)

print(qc)

qc.draw("mpl")

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

sorted_counts = sorted(
    counts.items(),
    key=lambda x:x[1],
    reverse=True
)

print("Top 5 States")

for state,count in sorted_counts[:5]:
    print(state,count)

print("Number of Measured States :", len(counts))

print("Most Frequent State :", sorted_counts[0])



plot_histogram(counts)

plt.show()







