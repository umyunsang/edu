from qiskit import QuantumCircuit

# 입력 데이터
x = 0.5

# 1개의 큐비트를 가진 회로 생성
qc = QuantumCircuit(1)

# x 값을 RY 회전각으로 사용
qc.ry(x, 0)

# 측정 추가
qc.measure_all()

# 회로 출력
print(qc)

# 회로 시각화
qc.draw("mpl")