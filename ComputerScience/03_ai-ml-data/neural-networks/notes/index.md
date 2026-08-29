# Index

## lecture

* [01. 퍼셉트론과 다층 신경망의 논리 게이트](./01.%20%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.md) - 단일 퍼셉트론(Perceptron)의 가중치·편향 판별식, AND·NAND·OR 선형 분리 한계와 민스키의 XOR 비선형 문제, 다층 퍼셉트론(MLP)의 은닉층 조합 원리를 인터랙티브 퍼셉트론 결정 경계 시뮬레이터로 심층 학습한다.
* [02. 인공신경망과 활성화 함수 - 시그모이드·ReLU와 소프트맥스 수치 안정성 (Neural Networks & Activation Functions)](./02.%20%EC%9D%B8%EA%B3%B5%EC%8B%A0%EA%B2%BD%EB%A7%9D%EA%B3%BC%20%ED%99%9C%EC%84%B1%ED%99%94%20%ED%95%A8%EC%88%98.md) - 계단 함수, 로지스틱 시그모이드(Sigmoid σ(x) = 1/(1+e⁻ˣ)), ReLU 및 Leaky ReLU의 도함수 수식화, 소프트맥스(Softmax) 오버플로 방지 상한 정규화(z_k - max(z))를 인터랙티브 활성화 함수 비교기로 심층 학습한다.
* [03. 신경망 학습과 손실 함수·경사하강법](./03.%20%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%ED%95%99%EC%8A%B5.md) - 교차 엔트로피 오차(Cross Entropy Error)와 평균 제곱 오차(MSE), 미니배치 샘플링, 중심 차분 수치 미분(Numerical Gradient) 및 경사하강법(Gradient Descent) 수렴 과정을 인터랙티브 2D 손실 곡면 시뮬레이터로 심층 학습한다.
* [04. 오차역전파법 - 계산 그래프 연쇄법칙과 행렬 어파인 계층 미분 (Backpropagation & Affine Layer)](./04.%20%EC%98%A4%EC%B0%A8%EC%97%AD%EC%A0%84%ED%8C%8C%EB%B2%95.md) - 계산 그래프(Computational Graph)의 국소적 연쇄법칙(Chain Rule), 행렬 곱 어파인(Affine) 계층 미분(∂L/∂X = (∂L/∂Y)Wᵀ, ∂L/∂W = Xᵀ(∂L/∂Y)), Softmax-with-Loss 역전파(y - t)를 인터랙티브 역전파 계산기로 심층 학습한다.
* [05. 학습 기술들 - 최적화 알고리즘·가중치 초기화와 배치 정규화 (Optimization Techniques & Batch Normalization)](./05.%20%ED%95%99%EC%8A%B5%20%EA%B8%B0%EC%88%A0%EB%93%A4.md) - SGD, 모멘텀(Momentum), AdaGrad, Adam의 갱신 점화식, Xavier와 He 가중치 초기화(Weight Initialization), 배치 정규화(Batch Normalization)의 순·역전파 대수학을 인터랙티브 최적화 알고리즘 비교기로 심층 학습한다.
* [neural-networks 강의 흐름 지도](./00.%20neural-networks%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 퍼셉트론의 한계 극복에서 인공신경망 순전파, 경사하강법 학습, 계산 그래프 역전파 및 최신 최적화·정규화 기술까지 5단계 신경망 커리큘럼 로드맵을 인터랙티브 가이드로 제공한다.
