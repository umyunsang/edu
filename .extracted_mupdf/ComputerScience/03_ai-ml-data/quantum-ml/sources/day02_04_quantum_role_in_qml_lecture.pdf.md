## --- [Page 1] ---
QML에서Quantum의역할

## --- [Page 2] ---
QML에서Quantum의역할> QML 구조이해

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 3] ---
QML에서Quantum의역할> QML 구조이해

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 4] ---
QML에서Quantum의역할> Quantum Encoding 이해

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 5] ---
QML에서Quantum의역할> Feature Map 이해

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 6] ---
QML에서Quantum의역할> QML 구조이해

사전준비사항

•
필수라이브러리설치

from qiskit.circuit.library import zz_feature_map
import matplotlib.pyplot as plt

pip1install1pylatexenc1matplotlib

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 7] ---
QML에서Quantum의역할> QML 구조이해

2. Feature Map 생성

from1qiskit.circuit.library1import1zz_feature_map

feature_map1=1zz_feature_map(feature_dimension=2)

print(feature_map)

실행결과

q_0:1─H──P(x[0])──■────────

│
q_1:1─H──P(x[1])──■────────

방금생성한회로는학습회로가아닙니다.”

"이회로는데이터를양자공간으로보내기위한회로입니다.”

즉, Data1->1Feature1Map 단계만수행합니다.

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 8] ---
QML에서Quantum의역할> QML 구조이해

from1qiskit.circuit.library1import1zz_feature_map
import1matplotlib.pyplot1as1plt

feature_map1=1zz_feature_map(feature_dimension=2)

feature_map.draw("mpl")

plt.show()

3. 회로시각화

실행결과

학습하는회로가아니라데이터를변환하는회로

H 게이트는데이터를넣기전에양자상태를준비하는단계

P 게이트가데이터를회로안에넣어주는역할

CNOT는각데이터특징이서로관계를가지도록만드는과정

큰P 게이트는단순히데이터를넣는것이아니라, 데이터를
사이의관계도함께표현

같은변환을여러번반복하여더복잡한표현공간을구성

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 9] ---
QML에서Quantum의역할> Feature Map 이해

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 10] ---
QML에서Quantum의역할> Measurement 이해

Source : IBM Quantum Learning, Qiskit Machine Learning Documentation

## --- [Page 11] ---