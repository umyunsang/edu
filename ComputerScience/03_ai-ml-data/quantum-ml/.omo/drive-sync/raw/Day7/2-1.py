from qiskit import QuantumCircuit
from matplotlib import pyplot as plt

qc = QuantumCircuit(2)

qc.h(0)

qc.cx(0, 1)

qc.measure_all()

from qiskit_aer import AerSimulator

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)


from qiskit.visualization import plot_histogram

plot_histogram(counts)

from matplotlib import pyplot as plt

qc.draw("mpl")
plt.show()
