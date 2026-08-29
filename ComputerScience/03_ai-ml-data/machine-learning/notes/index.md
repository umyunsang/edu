# Index

## lecture

* [CNN과 LeNet-5 분류](./10.%20CNN%EA%B3%BC%20LeNet-5%20%EB%B6%84%EB%A5%98.md) - 평탄화의 한계에서 출발해 합성곱 특징 추출과 CIFAR-10 LeNet-5 분류 구조를 정리한다.
* [KNN 분류·회귀와 직접 구현](./09.%20KNN%20%EB%B6%84%EB%A5%98%C2%B7%ED%9A%8C%EA%B7%80%EC%99%80%20%EC%A7%81%EC%A0%91%20%EA%B5%AC%ED%98%84.md) - KNN의 거리 척도, K 선택, 다수결·평균 집계와 직접 구현 실습을 정리한다.
* [machine-learning 강의 흐름 지도](./00.%20machine-learning%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 0개의 순서·핵심 단서·학습 점검을 연결한다.
* [RNN과 LSTM의 순환 구조](./13.%20RNN%EA%B3%BC%20LSTM%EC%9D%98%20%EC%88%9C%ED%99%98%20%EA%B5%AC%EC%A1%B0.md) - 시계열 의존성, RNN의 shared parameter와 BPTT, LSTM gate·상태 구조를 원본 강의 흐름으로 정리한다.
* [SVM 구현 - 경사 하강법(GD)과 QP](./06.%20SVM%20%EA%B5%AC%ED%98%84%20-%20%EA%B2%BD%EC%82%AC%20%ED%95%98%EA%B0%95%EB%B2%95%28GD%29%EA%B3%BC%20QP.md) - SVM의 힌지 손실 경사 하강법과 QP 기반 최적화 절차를 원본 실습 흐름으로 비교한다.
* [SVM 최대 마진과 힌지 손실](./05.%20SVM%20%EC%B5%9C%EB%8C%80%20%EB%A7%88%EC%A7%84%EA%B3%BC%20%ED%9E%8C%EC%A7%80%20%EC%86%90%EC%8B%A4.md) - 두 클래스를 가르는 초평면과 마진 조건, 힌지 손실 기반 갱신을 직접 구현한 자료를 정리한다.
* [Transformer Self-Attention과 블록 구성](./17.%20Transformer%20Self-Attention%EA%B3%BC%20%EB%B8%94%EB%A1%9D%20%EA%B5%AC%EC%84%B1.md) - Transformer encoder의 positional encoding, Q·K·V attention, multi-head와 residual·FFN 블록을 정리한다.
* [Transformer 언어 모델링과 인코더-디코더](./16.%20Transformer%20%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8%EB%A7%81%EA%B3%BC%20%EC%9D%B8%EC%BD%94%EB%8D%94-%EB%94%94%EC%BD%94%EB%8D%94.md) - 언어 모델의 확률 예측에서 Transformer 기계번역 encoder·decoder와 positional encoding을 연결한다.
* [U-Net 기반 특징 압축과 분류](./11.%20U-Net%20%EA%B8%B0%EB%B0%98%20%ED%8A%B9%EC%A7%95%20%EC%95%95%EC%B6%95%EA%B3%BC%20%EB%B6%84%EB%A5%98.md) - U-Net의 encoder·decoder와 skip connection, 전치 합성곱을 CIFAR-10 분류 경로와 함께 정리한다.
* [Word2Vec와 단어 임베딩](./15.%20Word2Vec%EC%99%80%20%EB%8B%A8%EC%96%B4%20%EC%9E%84%EB%B2%A0%EB%94%A9.md) - one-hot의 차원·거리 한계에서 출발해 Word2Vec embedding, 독립 학습, CBOW와 Skip-Gram을 정리한다.
* [다중 선형 회귀와 주택 가격](./03.%20%EB%8B%A4%EC%A4%91%20%EC%84%A0%ED%98%95%20%ED%9A%8C%EA%B7%80%EC%99%80%20%EC%A3%BC%ED%83%9D%20%EA%B0%80%EA%B2%A9.md) - kc_house_data의 상관관계와 특성 조합을 비교해 최소제곱 주택 가격 예측을 구성한다.
* [선형 회귀의 두 해법](./01.%20%EC%84%A0%ED%98%95%20%ED%9A%8C%EA%B7%80%20%EA%B8%B0%EC%B4%88%EC%99%80%20%EB%91%90%20%ED%95%B4%EB%B2%95.md) - 관측 데이터의 관계를 선형식으로 나타내고 최소제곱법과 경사 하강법으로 파라미터를 찾는 과정을 비교한다.
* [엔트로피와 결정 트리](./08.%20%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC%EC%99%80%20%EA%B2%B0%EC%A0%95%20%ED%8A%B8%EB%A6%AC.md) - 엔트로피로 불확실성을 재고, 낮은 엔트로피 분기를 재귀적으로 쌓는 결정 트리 실습을 정리한다.
* [우버 요금 다중 선형 회귀](./04.%20%EC%9A%B0%EB%B2%84%20%EC%9A%94%EA%B8%88%20%EB%8B%A4%EC%A4%91%20%EC%84%A0%ED%98%95%20%ED%9A%8C%EA%B7%80.md) - 우버 좌표로 L1·L2 이동거리 특성을 만들고 승객 수와 요금을 최소제곱 회귀로 연결한다.
* [중간고사 대비 - SVM·KNN 영화 추천](./07.%20%EC%A4%91%EA%B0%84%EA%B3%A0%EC%82%AC%20%EB%8C%80%EB%B9%84%20-%20SVM%C2%B7KNN%20%EC%98%81%ED%99%94%20%EC%B6%94%EC%B2%9C.md) - Action·Romance 점수로 세 영화 클래스를 분류하는 SVM·KNN 평가 문제와 제공된 해설을 정리한다.
* [초해상도와 SRCNN](./12.%20%EC%B4%88%ED%95%B4%EC%83%81%EB%8F%84%EC%99%80%20SRCNN.md) - 보간·예제 기반 초해상도의 한계를 짚고 SRCNN의 데이터 구성, 학습, PSNR 평가를 정리한다.

## practice

* [RNN·LSTM 실습](./14.%20RNN%C2%B7LSTM%20%EC%8B%A4%EC%8A%B5.md) - 훈민정음 서문을 one-hot sequence로 변환하고 RNN·LSTM으로 예측하는 실습 절차를 정리한다.
* [단일 선형 회귀 실습 - LSM과 GDM](./02.%20%EB%8B%A8%EC%9D%BC%20%EC%84%A0%ED%98%95%20%ED%9A%8C%EA%B7%80%20%EC%8B%A4%EC%8A%B5%20-%20LSM%EA%B3%BC%20GDM.md) - kc_house_data의 sqft_living과 price를 정규화·분할한 뒤 LSM과 GDM을 구현하는 실습 흐름이다.
