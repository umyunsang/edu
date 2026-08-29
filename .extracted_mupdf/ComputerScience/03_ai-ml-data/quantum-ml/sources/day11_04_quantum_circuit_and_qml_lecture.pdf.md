## --- [Page 1] ---
Quantum Circuit과QML

## --- [Page 2] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML > 지금까지만든Quantum Circuit은어디에사용될까?

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 3] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML > 지금까지만든Quantum Circuit은어디에사용될까?

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 4] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML > Variational Circuit은학습되는Circuit 이다.

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 5] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

1) Feature)Map 생성하기

import)numpy)as)np

from)qiskit)import)transpile
from)qiskit.circuit.library)import)zz_feature_map,)real
_amplitudes
from)qiskit_aer)import)AerSimulator
from)qiskit.visualization)import)plot_histogram

import)matplotlib.pyplot)as)plt

feature_map)=)zz_feature_map(

feature_dimension=2,
reps=1
)

feature_dimension=2 →)입력Feature가2개라는의미

feature_dimension=2는두개의Feature를두개의Qub
it에인코딩한다는의미

reps=1 →)Feature)Map)구조를1번반복한다는의미.

## --- [Page 6] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

2) Variational)Circuit 생성

ansatz)=)real_amplitudes(

num_qubits=2,
reps=1
)

Variational)Circuit은학습가능한Parameter를가진Quantu
m)Circuit입니다.

Feature)Map이데이터를Quantum)State로변환하면,)Variat
ional)Circuit은그상태를학습가능한방식으로다시변환합니
다.

um_qubits=2 →)2개의Qubit으로구성된Variational)Circuit

reps=1 →)회로Layer를1번반복

RealAmplitudes는내부적으로RY)Rotation)Gate, Entangle
ment)Gate, 학습Parameter 으로구성되어있습니다.

## --- [Page 7] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 8] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 9] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

3) Circuit 결합

qml_circuit)=)feature_map.compose(ansatz)

print(qml_circuit)

Feature)Map 이고데이터를Quantum)State로변환,
Variational)Circuit 은학습가능한Parameter로상태변환
으로두Circuit을연결해야하나의QML)Circuit이됩니다.

compose()는두Quantum)Circuit을순서대로연결합니다.

먼저feature_map)실행후, 그다음ansatz)실행합니다.

결합구조
•
q0)─)Feature)Map)─)Ansatz)─)Measurement
•
q1)─)Feature)Map)─)Ansatz)─)Measurement

이구조에서앞부분은데이터를넣는부분이고,)뒷부분은학습하
는부분입니다.

## --- [Page 10] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 11] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

4) Measurement 추가

qml_circuit.measure_all()

print(qml_circuit)

Measurement를추가하면모든Qubit이Classical)Bit로측정
됩니다.

측정결과는다음과같은Bitstring으로나옵니다.

## --- [Page 12] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

5) Simulator 실행

sim)=)AerSimulator()

compiled_circuit)=)transpile(bound_circuit,)sim)

job)=)sim.run(compiled_circuit,)shots=1000)

result)=)job.result()

counts)=)result.get_counts()

print("\nMeasurement)Counts")
print("---------------------------")
print(counts)

AerSimulator →)양자회로를시뮬레이션하는도구

shots=1000 →)회로를1000번반복실행

counts →)측정결과별횟수

## --- [Page 13] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

6) Parameter 확인

print("Parameters")
print("---------------------------")

for)p)in)qml_circuit.parameters:

print(p)

회로에포함된Parameter)목록을확인할수있습니다.

예상출력예시
ParameterView([x[0],)x[1],)θ[0],)θ[1],)...])

## --- [Page 14] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

7) Parameter 값할당

parameter_values)=){

param:)np.random.random()
for)param)in)qml_circuit.parameters
}

bound_circuit)=)qml_circuit.assign_parameters(para
meter_values)

각Parameter에임의의숫자를넣는다.

assign_parameters()로Parameter가채워진Circuit을만든다.

bound_circuit은실행가능한Circuit이다.

## --- [Page 15] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

8) Histogram 시각화

plot_histogram(counts)

plt.show()

Histogram에서는각Bitstring의측정확률을확인할수
있습니다.

## --- [Page 16] ---
© 2026 Kangwuk Heo. All Rights Reserved.

Quantum Circuit과QML >. 실습: 첫번째QML Circuit 구성하기

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

실행결과

{'01':)115,)'00':)583,)'11':)23,)'10':)279}

랜덤Parameter가적용된현재Circuit에서는|00⟩상태가가장
높은확률로측정되었습니다.

이는현재Gate들의조합이Quantum)State를|00⟩방향으로
가장많이변환했다는의미입니다.

## --- [Page 17] ---
© 2026 Kangwuk Heo. All Rights Reserved.