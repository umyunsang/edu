import numpy as np

from qiskit import transpile
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

# =====================================================
# Step 1. Feature Map 생성
# =====================================================
feature_map = zz_feature_map(
    feature_dimension=2,
    reps=1
)

# =====================================================
# Step 2. Variational Circuit 생성
# =====================================================
ansatz = real_amplitudes(
    num_qubits=2,
    reps=1
)

# =====================================================
# Step 3. Circuit 결합
# =====================================================
qml_circuit = feature_map.compose(ansatz)

# =====================================================
# Step 4. Measurement 추가
# =====================================================
qml_circuit.measure_all()

# =====================================================
# Step 5. Parameter 확인
# =====================================================
print("Parameters")
print("---------------------------")

for p in qml_circuit.parameters:
    print(p)

# =====================================================
# Step 6. Parameter 값 할당
# =====================================================
parameter_values = {
    param: np.random.random()
    for param in qml_circuit.parameters
}

bound_circuit = qml_circuit.assign_parameters(parameter_values)

# =====================================================
# Step 7. Simulator 실행
# =====================================================
sim = AerSimulator()

compiled_circuit = transpile(bound_circuit, sim)

job = sim.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts()

print("\nMeasurement Counts")
print("---------------------------")
print(counts)

# =====================================================
# Step 8. Histogram 출력
# =====================================================
plot_histogram(counts)

plt.show()





