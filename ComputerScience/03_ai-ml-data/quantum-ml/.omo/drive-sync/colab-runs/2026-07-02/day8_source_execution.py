import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qiskit",
        "qiskit-aer",
        "pylatexenc",
    ]
)

SCRIPTS = [
    (
        "4-2-1.py",
        r'''
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(2)

qc.h(0)
qc.cx(0,1)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

from matplotlib import pyplot as plt

qc.draw("mpl")
plt.show()
''',
    ),
    (
        "4-2-2.py",
        r'''
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(2)

qc.cx(0,1)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

from matplotlib import pyplot as plt

qc.draw("mpl")
plt.show()
''',
    ),
    (
        "4-2-3.py",
        r'''
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(2)

qc.h(0)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

from matplotlib import pyplot as plt

qc.draw("mpl")
plt.show()
''',
    ),
    (
        "4-3-1.py",
        r'''
import random
from collections import Counter
from qiskit.visualization import plot_histogram

results = []

for _ in range(1000):

    a = random.randint(0,1)

    b = random.randint(0,1)

    results.append(str(a)+str(b))

counts = Counter(results)

print(counts)

plot_histogram(dict(counts))
from matplotlib import pyplot as plt
plt.show()

''',
    ),
    (
        "4-3-2.py",
        r'''
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(2)

qc.h(0)

qc.cx(0,1)

qc.measure_all()

sim = AerSimulator()

job = sim.run(qc, shots=1000)

result = job.result()

counts = result.get_counts()

print(counts)

plot_histogram(counts)

plot_histogram(dict(counts))
from matplotlib import pyplot as plt
plt.show()
''',
    ),
]

for name, source in SCRIPTS:
    print(f"COLAB_SOURCE_BEGIN {name}")
    exec(compile(source, name, "exec"), {"__name__": "__main__"})
    print(f"COLAB_SOURCE_OK {name}")

print("COLAB_DAY8_SOURCE_EXECUTION_OK")
