# Index

## lecture

* [01. Perceptron 이론 - 신경망 구성과 의사결정 경계](./01.%20Perceptron%20%EC%9D%B4%EB%A1%A0%20-%20%EC%8B%A0%EA%B2%BD%EB%A7%9D%20%EA%B5%AC%EC%84%B1.md) - 기호주의와 연결주의의 역사적 대립, 생물학적 뉴런과 인공 퍼셉트론의 구조적 대응, 회귀(Regression)와 분류(Classification) 응용, 2차원 결정 초평면(Decision Boundary)의 기하학적 유도를 인터랙티브 퍼셉트론 분류 시뮬레이터로 심층 학습한다.
* [02. Perceptron 이론 - 활성화 함수와 손실 최적화](./02.%20Perceptron%20%EC%9D%B4%EB%A1%A0%20-%20%ED%99%9C%EC%84%B1%ED%99%94%EC%99%80%20%EC%B5%9C%EC%A0%81%ED%99%94.md) - 불연속 계단 함수에서 연속 시그모이드(Sigmoid)로의 확장, 이진 교차 엔트로피(BCE) 손실 함수의 수학적 유도, 경사하강법(Gradient Descent) 가중치 편미분 유도를 인터랙티브 시그모이드 확률·손실 계산기로 심층 학습한다.
* [03. Perceptron 논리 게이트 실습과 선형 분리성 (Colab Lab)](./03.%20Perceptron%20%EB%85%BC%EB%A6%AC%20%EA%B2%8C%EC%9D%B4%ED%8A%B8%20%EC%8B%A4%EC%8A%B5.md) - Google Colab 환경에서의 AND·OR·NAND 단일 퍼셉트론 가중치·편향 설계, 진리표(Truth Table) 기반 2D 결정 경계 분리 검증 및 XOR 다층 결합 파이프라인을 인터랙티브 논리 게이트 시뮬레이터로 실습한다.
* [04. MLP 이론 - 다층 퍼셉트론과 전결합 계층 (Fully Connected Layer)](./04.%20MLP%20%EC%9D%B4%EB%A1%A0%20-%20%EB%8B%A4%EC%B8%B5%20%ED%91%9C%ED%98%84.md) - 다층 퍼셉트론(MLP)의 전결합 계층(FC Layer) 텐서 연산, 계층별 가중치·편향 매개변수 개수 산출 공식, ReLU·Leaky ReLU 및 소프트맥스(Softmax) 수식 유도를 인터랙티브 MLP 구조·매개변수 계산기로 심층 학습한다.
* [05. MLP 실습 - MNIST 데이터셋과 PyTorch 모델 구성 (Model Architecture)](./05.%20MLP%20%EC%8B%A4%EC%8A%B5%20-%20%EB%AA%A8%EB%8D%B8%20%EA%B5%AC%EC%84%B1.md) - MNIST 28x28 이미지의 784차원 1D 평탄화(Flatten) 전처리, PyTorch nn.Module 상속 기반 단층(SLP) 및 2층 MLP 신경망 클래스 설계, 은닉층 활성화 매핑을 인터랙티브 텐서 차원 변환기로 실습한다.
* [06. MLP 실습 - 훈련 루프와 평가 파이프라인 (Training & Evaluation)](./06.%20MLP%20%EC%8B%A4%EC%8A%B5%20-%20%ED%95%99%EC%8A%B5%EA%B3%BC%20%ED%8F%89%EA%B0%80.md) - PyTorch의 DataLoader 미니배치 순회, CrossEntropyLoss와 SGD 옵티마이저, 4단계 그래디언트 갱신 루프(zero_grad -> forward -> backward -> step) 및 무기울기 평가(torch.no_grad)를 인터랙티브 훈련 손실·정확도 시뮬레이터로 실습한다.
* [07. Optimization 이론 - 데이터 정규화와 경사하강법 (Gradient Descent)](./07.%20Optimization%20%EC%9D%B4%EB%A1%A0%20-%20%EC%86%90%EC%8B%A4%EA%B3%BC%20%EA%B2%BD%EC%82%AC.md) - Min-Max 및 표준 정규화(Standardization)의 손실 곡면 왜곡 보정 원리, 전체 배치(BGD)·확률적(SGD)·미니배치(MGD) 경사하강법 비교, 학습률(Learning Rate) 선정 기준을 인터랙티브 정규화 전후 손실 등고선 시뮬레이터로 심층 학습한다.
* [08. Optimizer - 관성과 진동 억제의 모멘텀 (Momentum Optimization)](./08.%20Optimizer%20-%20%EB%AA%A8%EB%A9%98%ED%85%80.md) - 물리적 관성(Inertia)을 활용한 모멘텀(Momentum) 및 Nesterov 가속 경사(NAG)의 수학적 원리, 협곡형 손실 곡면에서의 진동 억제와 안장점 탈출 메커니즘을 인터랙티브 모멘텀 궤적 시뮬레이터로 심층 학습한다.
* [09. Optimizer - 적응적 학습률 알고리즘 (AdaGrad, RMSProp, Adam)](./09.%20Optimizer%20-%20%EC%A0%81%EC%9D%91%EC%A0%81%20%ED%95%99%EC%8A%B5%EB%A5%A0.md) - 매개변수별 맞춤형 학습률 감쇠 원리, AdaGrad의 조기 학습 정체 문제와 RMSProp의 지수 이동 평균(EMA) 해결책, Adam의 1·2차 모멘트 결합 및 편향 보정 수식을 인터랙티브 적응형 옵티마이저 비교기로 심층 학습한다.
* [10. Overfitting - 과적합 진단과 편향-분산 트레이드오프 (Bias-Variance Tradeoff)](./10.%20Overfitting%20-%20%EC%9D%BC%EB%B0%98%ED%99%94%20%EC%A7%84%EB%8B%A8.md) - 과소적합(Underfitting)과 과적합(Overfitting)의 정의, 학습 손실(Train Loss)과 검증 손실(Validation Loss)의 발산 진단, 편향-분산 분해(Bias-Variance Decomposition) 수학적 유도를 인터랙티브 과적합 곡선 시뮬레이터로 심층 학습한다.
* [11. Overfitting - 규제 기법과 드롭아웃·조기 종료 (Regularization & Dropout)](./11.%20Overfitting%20-%20%EA%B7%9C%EC%A0%9C%EC%99%80%20%EC%A1%B0%EA%B8%B0%20%EC%A2%85%EB%A3%8C.md) - 데이터 증식(Data Augmentation), L1(Lasso)과 L2(Weight Decay) 가중치 감쇠의 기하학적 제약 조건 비교, 드롭아웃(Dropout)의 앙상블 효과 및 조기 종료(Early Stopping) 알고리즘을 인터랙티브 L1/L2 규제 시뮬레이터로 심층 학습한다.
* [12. Backpropagation - 연쇄 법칙과 계산 그래프 (Chain Rule)](./12.%20Backpropagation%EC%9D%98%20%EC%97%B0%EC%87%84%20%EB%B2%95%EC%B9%99.md) - 다변수 합성함수의 연쇄 법칙(Chain Rule), 국소적 미분(Local Gradient)의 역방향 누적 곱셈 원리, 덧셈·곱셈 노드의 순전파 캐싱과 역전파를 인터랙티브 연쇄 법칙 계산기 시뮬레이터로 심층 학습한다.
* [13. Backpropagation - 다층 신경망 가중치 역전파와 델타 오차 (Delta Error Rule)](./13.%20Backpropagation%EC%9D%98%20%EA%B0%80%EC%A4%91%EC%B9%98%20%EA%B0%B1%EC%8B%A0.md) - 다층 퍼셉트론(MLP)의 오차 항(Delta Vector) 역전파 유도, 가중치 행렬 및 편향 벡터의 전치 행렬 텐서 미분 공식, 2층 신경망 수치 예제를 인터랙티브 가중치 역전파 계산기로 심층 학습한다.
* [AI 아바타 만들기 실습](./26.%20AI%20%EC%95%84%EB%B0%94%ED%83%80%20%EB%A7%8C%EB%93%A4%EA%B8%B0%20%EC%8B%A4%EC%8A%B5.md) - AI 아바타 만들기 실습의 입력·계산·검증 흐름을 정리한다.
* [artificial-intelligence 강의 흐름 지도](./00.%20artificial-intelligence%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 29개의 순서·쪽수·학습 점검을 연결한다.
* [CIFAR-10 분류 실습](./25.%20CIFAR-10%20%EB%B6%84%EB%A5%98%20%EC%8B%A4%EC%8A%B5.md) - CIFAR-10 분류 실습의 입력·계산·검증 흐름을 정리한다.
* [CNN Backpropagation - AlexNet 구조](./22.%20CNN%20Backpropagation%20-%20AlexNet%20%EA%B5%AC%EC%A1%B0.md) - CNN Backpropagation - AlexNet 구조의 입력·계산·검증 흐름을 정리한다.
* [CNN Backpropagation - AlexNet 정규화](./23.%20CNN%20Backpropagation%20-%20AlexNet%20%EC%A0%95%EA%B7%9C%ED%99%94.md) - CNN Backpropagation - AlexNet 정규화의 입력·계산·검증 흐름을 정리한다.
* [CNN 분류 실습](./21.%20CNN%20%EB%B6%84%EB%A5%98%20%EC%8B%A4%EC%8A%B5.md) - CNN 분류 실습의 입력·계산·검증 흐름을 정리한다.
* [CNN 설계 모듈 실습 - Conv2d 매개변수](./19.%20CNN%20%EC%84%A4%EA%B3%84%20%EB%AA%A8%EB%93%88%20%EC%8B%A4%EC%8A%B5%20-%20Conv2d%20%EB%A7%A4%EA%B0%9C%EB%B3%80%EC%88%98.md) - CNN 설계 모듈 실습 - Conv2d 매개변수의 입력·계산·검증 흐름을 정리한다.
* [CNN 설계 모듈 실습 - 연결 구조](./20.%20CNN%20%EC%84%A4%EA%B3%84%20%EB%AA%A8%EB%93%88%20%EC%8B%A4%EC%8A%B5%20-%20%EC%97%B0%EA%B2%B0%20%EA%B5%AC%EC%A1%B0.md) - CNN 설계 모듈 실습 - 연결 구조의 입력·계산·검증 흐름을 정리한다.
* [CNN 주요 설계 모듈 이론](./18.%20CNN%20%EC%A3%BC%EC%9A%94%20%EC%84%A4%EA%B3%84%20%EB%AA%A8%EB%93%88%20%EC%9D%B4%EB%A1%A0.md) - CNN 주요 설계 모듈 이론의 입력·계산·검증 흐름을 정리한다.
* [CNN의 공간 크기와 채널](./17.%20CNN%EC%9D%98%20%EA%B3%B5%EA%B0%84%20%ED%81%AC%EA%B8%B0%EC%99%80%20%EC%B1%84%EB%84%90.md) - CNN의 공간 크기와 채널의 입력·계산·검증 흐름을 정리한다.
* [CNN의 합성곱 원리](./16.%20CNN%EC%9D%98%20%ED%95%A9%EC%84%B1%EA%B3%B1%20%EC%9B%90%EB%A6%AC.md) - CNN의 합성곱 원리의 입력·계산·검증 흐름을 정리한다.
* [Vanishing Gradient Effect](./14.%20Vanishing%20Gradient%20Effect.md) - Vanishing Gradient Effect의 핵심 개념과 학습 판단을 정리한다.
* [Vanishing Gradient 완화](./15.%20Vanishing%20Gradient%20%EC%99%84%ED%99%94.md) - Vanishing Gradient 완화의 핵심 개념과 학습 판단을 정리한다.
* [VGGNet 실습](./24.%20VGGNet%20%EC%8B%A4%EC%8A%B5.md) - VGGNet 실습의 핵심 개념과 학습 판단을 정리한다.
* [중간고사 - MLP 기반 CIFAR-10 분류](./27.%20%EC%A4%91%EA%B0%84%EA%B3%A0%EC%82%AC%20-%20MLP%20%EA%B8%B0%EB%B0%98%20CIFAR-10%20%EB%B6%84%EB%A5%98.md) - 중간고사 - MLP 기반 CIFAR-10 분류의 핵심 개념과 학습 판단을 정리한다.
* [필기 과제 2 텍스트 추출 한계](./29.%20%ED%95%84%EA%B8%B0%20%EA%B3%BC%EC%A0%9C%202%20%ED%85%8D%EC%8A%A4%ED%8A%B8%20%EC%B6%94%EC%B6%9C%20%ED%95%9C%EA%B3%84.md) - 필기 과제 2 텍스트 추출 한계의 핵심 개념과 학습 판단을 정리한다.
* [필기 과제 자료의 텍스트 추출 한계](./28.%20%ED%95%84%EA%B8%B0%20%EA%B3%BC%EC%A0%9C%20%EC%9E%90%EB%A3%8C%EC%9D%98%20%ED%85%8D%EC%8A%A4%ED%8A%B8%20%EC%B6%94%EC%B6%9C%20%ED%95%9C%EA%B3%84.md) - 필기 과제 자료의 텍스트 추출 한계의 핵심 개념과 학습 판단을 정리한다.
