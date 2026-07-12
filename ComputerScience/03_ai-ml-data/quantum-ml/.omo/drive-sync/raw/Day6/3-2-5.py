from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()

def run_and_get_prob_vector(qc, shots=1000):
    sim = AerSimulator()
    job = sim.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    total = sum(counts.values())
    p0 = counts.get('0', 0) / total
    p1 = counts.get('1', 0) / total

    return counts, [p0, p1]

qc_h = QuantumCircuit(1)
qc_h.h(0)
qc_h.measure_all()

counts_h, prob_h = run_and_get_prob_vector(qc_h)

print(counts_h)
print(prob_h)
