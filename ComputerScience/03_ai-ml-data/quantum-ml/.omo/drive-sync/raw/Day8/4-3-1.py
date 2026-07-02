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
