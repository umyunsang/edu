## --- [Page 1] ---
상태변화분석

## --- [Page 2] ---
상태변화분석> 상태변화분석개념

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 3] ---
상태변화분석> 상태변화분석개념

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 4] ---
상태변화분석> 상태변화분석실습1 : Gate 추가시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 5] ---
상태변화분석> 상태변화분석실습1 : Gate 추가시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

1. 기본상태(Basis,State) 확인
2. 시뮬레이션실행

from,qiskit,import,QuantumCircuit

qc,=,QuantumCircuit(1)

qc.measure_all()

qc.draw("mpl")

from,qiskit_aer,import,AerSimulator

sim,=,AerSimulator()

job,=,sim.run(qc,,shots=1000)

result,=,job.result()

counts,=,result.get_counts()

print(counts)

3. 결과

{'0':,1000}

•
초기상태가|0⟩이기때문에0이나옴

## --- [Page 6] ---
상태변화분석> 상태변화분석실습1 : Gate 추가시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

3. 중첩생성
4. 실행결과

from,qiskit,import,QuantumCircuit

qc,=,QuantumCircuit(1)

qc.h(0)

qc.measure_all()

job,=,sim.run(qc,,shots=100)

result,=,job.result()

counts,=,result.get_counts()

print(counts)

{'0':,48,,'1':,52}

•
큐비트를|0⟩상태에서|0⟩+,|1⟩형태의중첩상태로
바꾸는역할을수행

## --- [Page 7] ---
상태변화분석> 상태변화분석실습2 : Gate 반복적용시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 8] ---
상태변화분석> 상태변화분석실습3 : Gate 순서변경시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

1. 중첩생성
2. 실행결과

from,qiskit,import,QuantumCircuit

qc,=,QuantumCircuit(1)

qc.h(0)
qc.h(0)

qc.measure_all()

job,=,sim.run(qc,,shots=1000)

result,=,job.result()

counts,=,result.get_counts()

print(counts)

{'0':,1000}

•
왜H,Gate를두번적용했는데원래상태로돌아왔을까요?

•
첫번째H -> 중첩생성-> 두번째H -> 중첩제거-> 원래
상태복원

## --- [Page 9] ---
상태변화분석> 상태변화분석실습2 : Gate 반복적용시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

1. 회로A,:,H,Gate 적용후, X,Gate 적용

from,qiskit,import,QuantumCircuit
from,qiskit_aer,import,AerSimulator
from,qiskit.visualization,import,plot_histogram
import,matplotlib.pyplot,as,plt

#,회로생성
qc1,=,QuantumCircuit(1)

#,H,Gate
qc1.h(0)

#,X,Gate
qc1.x(0)

#,측정
qc1.measure_all()

#,회로출력
print(qc1.draw())

#,시뮬레이터실행
sim,=,AerSimulator()

job,=,sim.run(qc1,,shots=1000)

result,=,job.result()

counts1,=,result.get_counts()

print(counts1)

#,Histogram
plot_histogram(counts1)
plt.show()

## --- [Page 10] ---
상태변화분석> 상태변화분석실습2 : Gate 반복적용시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

실행결과

{{'0':,492, '1':,508}

•
Hadamard,Gate가먼저적용되어큐비트가중첩(Super
position),상태가된다.,

•
이후X,Gate를적용해도측정확률은약50%,:,50%,로유
지된다.,

•
즉,,Gate를추가해도중첩상태는유지될수있음을확인할
수있다.

## --- [Page 11] ---
상태변화분석> 상태변화분석실습2 : Gate 반복적용시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

2. 회로B,:,X,Gate 적용후, H,Gate 적용

from,qiskit,import,QuantumCircuit
from,qiskit_aer,import,AerSimulator
from,qiskit.visualization,import,plot_histogram
import,matplotlib.pyplot,as,plt

qc2,=,QuantumCircuit(1)

#,X,Gate
qc2.x(0)

#,H,Gate
qc2.h(0)

qc2.measure_all()

print(qc2.draw())

sim,=,AerSimulator()

job,=,sim.run(qc2,,shots=1000)

result,=,job.result()

counts2,=,result.get_counts()

print(counts2)

plot_histogram(counts2)
plt.show()

## --- [Page 12] ---
상태변화분석> 상태변화분석실습2 : Gate 반복적용시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

실행결과

{'1':,492,,'0':,508}

•
X,Gate로먼저|1⟩상태를만든후Hadamard,Gate를적
용하여중첩상태로변환한다.,

•
측정결과는역시약50%,:,50%,로나타난다.,

•
즉,,Gate의적용순서는상태변화과정에영향을주지만,,
측정결과만으로는모든상태의차이를구별할수는없다.

## --- [Page 13] ---
상태변화분석> 상태변화분석실습3 : Gate 순서변경시, 상태변화

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 14] ---