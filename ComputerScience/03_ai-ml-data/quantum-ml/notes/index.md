# Index

## lecture

* [00. quantum-ml 강의 흐름 지도 - 양자 머신러닝 전 단원 로드맵 (Curriculum Map)](./00.%20quantum-ml%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 양자 머신러닝(Quantum Machine Learning) 교과목의 4대 핵심 영역(표현력 한계와 큐비트 기초, 양자 특성 공간과 임베딩, 아다마르/벨 상태 전이와 게이트 대수, 양자 회로 설계와 VQC 최적화)을 인터랙티브 커리큘럼 맵으로 조망한다.
* [01. 표현력의 한계 - 고전 모델의 한계와 고차원 사상 (Expressive Power & Classical Limits)](./01.%20%ED%91%9C%ED%98%84%EB%A0%A5%EC%9D%98%20%ED%95%9C%EA%B3%84.md) - 고전 머신러닝 및 심층 신경망의 비선형 표현력 한계, 차원의 저주(Curse of Dimensionality), 커널 트릭의 계산 복잡도 및 양자 힐베르트 공간으로의 확장 필연성을 인터랙티브 비선형 분리 시뮬레이터로 심층 학습한다.
* [02. 왜 양자 컴퓨팅인가 - 무어의 법칙 한계와 3대 양자 원리 (Why Quantum Computing?)](./02.%20%EC%99%9C%20%EC%96%91%EC%9E%90%20%EC%BB%B4%ED%93%A8%ED%8C%85%EC%9D%B8%EA%B0%80.md) - 반도체 미세화와 무어의 법칙(Moore's Law) 한계, 양자 중첩(Superposition), 양자 얽힘(Entanglement), 양자 간섭(Interference)의 3대 원리와 $2^n$ 지수적 상태 공간 확장을 인터랙티브 큐비트 지수 팽창 계산기로 심층 학습한다.
* [03. Bit와 Qubit - 이진 비트와 양자 큐비트의 물리적 차이 (Classical Bit vs Quantum Qubit)](./03.%20Bit%EC%99%80%20Qubit.md) - 고전 이진 스위치(0 or 1)와 양자 큐비트(확률 진폭과 위상을 갖는 2준위 양자계)의 근본적 차이, 블로흐 구(Bloch Sphere) 좌표계와 측정 붕괴 원리를 인터랙티브 큐비트 위상 각도기로 심층 학습한다.
* [04. Quantum Feature Space - 양자 특성 공간과 3대 데이터 임베딩 (Quantum Feature Maps)](./04.%20Quantum%20Feature%20Space.md) - 고전 데이터를 고차원 힐베르트 상태로 변환하는 각도 임베딩(Angle Embedding), 진폭 임베딩(Amplitude Embedding), IQP 비선형 얽힘 임베딩의 수학적 원리와 회로 복잡도를 인터랙티브 임베딩 시뮬레이터로 심층 학습한다.
* [05. QML에서 Quantum의 역할 - 양자 가속과 하이브리드 QNN (Quantum Advantage in Machine Learning)](./05.%20QML%EC%97%90%EC%84%9C%20Quantum%EC%9D%98%20%EC%97%AD%ED%95%A0.md) - 기계학습 파이프라인에서 양자 프로세서의 3대 핵심 역할(고차원 커널 평가, 변분 파라미터 최적화, 샘플링 생성 모델)과 양자 신경망(QNN)의 역전파 메커니즘을 인터랙티브 QNN 파이프라인 시뮬레이터로 심층 학습한다.
* [06. Hadamard Gate - 양자 중첩 생성과 기저 변환 (Hadamard Gate & Basis Change)](./06.%20Hadamard%20Gate.md) - 양자 중첩(Superposition)을 생성하는 핵심 단일 큐비트 아다마르(Hadamard, H) 게이트의 행렬 대수, 기저 변환(|0⟩/|1⟩ ➔ |+⟩/|-⟩), 자기 수반성(H²=I), n-큐비트 균등 중첩 생성을 인터랙티브 아다마르 시뮬레이터로 심층 학습한다.
* [07. 상태변화 분석 - 2-큐비트 얽힘과 벨 상태 생성 (Two-Qubit State Evolution & Bell States)](./07.%20%EC%83%81%ED%83%9C%EB%B3%80%ED%99%94%20%EB%B6%84%EC%84%9D.md) - 단일 큐비트 중첩에서 CNOT 제어 게이트를 거쳐 2-큐비트 벨 상태(Bell State)로 진화하는 양자 상태 벡터의 궤적, 확률 진폭 변화 및 얽힘(Entanglement) 생성 메커니즘을 인터랙티브 상태 전이 시뮬레이터로 심층 학습한다.
* [08. Quantum Gate 개념 - 단일 큐비트 회전과 범용 게이트 셋 (Universal Quantum Gates)](./08.%20Quantum%20Gate%20%EA%B0%9C%EB%85%90.md) - 파울리 게이트, 아다마르(H), 위상 게이트(S, T), 단일 큐비트 임의 3축 회전 게이트(Rx, Ry, Rz)의 오일러 각도 분해 및 범용 양자 게이트 셋(Universal Gate Set) 구성을 인터랙티브 3축 회전 시뮬레이터로 심층 학습한다.
* [09. Quantum Circuit - 양자 회로 설계와 합성 복잡도 (Quantum Circuit Synthesis & Depth)](./09.%20Quantum%20Circuit.md) - 양자 회로의 와이어, 게이트 배치, 회로 깊이(Circuit Depth), 2-큐비트 게이트 수, 병렬화 가능성 및 텐서 곱/행렬 곱 대수를 인터랙티브 회로 깊이 계산기로 심층 학습한다.
* [10. Quantum Circuit과 QML - 파라미터화 회로와 Barren Plateaus 현상 (PQC & Barren Plateaus)](./10.%20Quantum%20Circuit%EA%B3%BC%20QML.md) - 파라미터화된 양자 회로(PQC)의 엔사츠 설계, 힐베르트 공간의 척박한 고원(Barren Plateaus) 현상의 수학적 원인(지수적 분산 소실) 및 해결 전략을 인터랙티브 그래디언트 분산 시뮬레이터로 심층 학습한다.
* [양자 ML 과정 - 160시간 양자 머신러닝 집중 과정 종합 정리 (Quantum ML 160H Curriculum)](./%EC%96%91%EC%9E%90%20ML%20%EA%B3%BC%EC%A0%95.md) - 고전 표현력 한계 극복부터 큐비트 대수, 양자 임베딩, 아다마르/벨 상태 전이, 범용 회로 합성, VQC 및 Barren Plateaus 극복까지 160시간 양자 머신러닝 전 과정을 총괄 정리한다.
