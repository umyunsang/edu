from qiskit import QuantumCircuit

# 입력 데이터
x = [0.3, 0.7]

# 2개의 큐비트를 가진 회로 생성
qc = QuantumCircuit(2)

# 각 feature를 각 큐비트의 회전각으로 사용
qc.ry(x[0], 0)
qc.ry(x[1], 1)

# 측정 추가
qc.measure_all()

# 회로 출력
print(qc)

# 회로 시각화
qc.draw("mpl")