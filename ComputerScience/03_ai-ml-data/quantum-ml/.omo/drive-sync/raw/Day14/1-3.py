from qiskit import QuantumCircuit

# 입력 데이터
x = [0.2, 0.5, 0.8, 0.6]

# feature 개수만큼 큐비트 생성
num_qubits = len(x)
print(num_qubits)
qc = QuantumCircuit(num_qubits)

# 각 feature를 각 qubit에 encoding
for i, value in enumerate(x):
    qc.ry(value, i)

# 측정 추가
qc.measure_all()

# 회로 출력
print(qc)

# 회로 시각화
qc.draw("mpl")