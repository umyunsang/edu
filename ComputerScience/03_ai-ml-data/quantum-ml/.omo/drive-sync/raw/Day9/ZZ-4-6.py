from qiskit.circuit.library import zz_feature_map

feature_map = zz_feature_map(
    feature_dimension=2,
    reps=3
)

print(feature_map)
feature_map.count_ops()

feature_map.draw("mpl")

from matplotlib import pyplot as plt
plt.show()