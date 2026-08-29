### [Page 1]
1/37
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능



### [Page 2]
2/37
Contents
1.
Perceptron을이용한분류실습
2.
Multi-layer Perceptron 실습



### [Page 3]
3/37
Contents
1.
Perceptron을이용한분류실습
2.
Multi-layer Perceptron 실습



### [Page 4]
4/37
Perceptron을이용한분류실습- 실습일정공지

Perceptron을이용한MNIST 손글씨데이터셋분류
…
1x
2x
3x
784
x



…
Activation function
0y
1y
9y
…
Single Layer Perceptron 실습
…
1x
2x
3x
784
x
…
Activation function
0y
1y
9y
…
…






Multi Layer Perceptron 실습

Layer1 (Input: 784, Out: 10)

Activation function: Softmax

Loss function: Cross Entropy

Layer1 (Input: 784, Out: 100)

Layer2 (Input: 100, Out: 10)

Activation function: Softmax, Sigmoid

Loss function: Cross Entropy



### [Page 5]
5/37
Perceptron을이용한분류실습- 데이터셋소개

MNIST 데이터베이스(Modified National Institute of Standards and Technology)
•
손으로쓴숫자들로이루어진대형데이터베이스
•
Train dataset 60,000개, Test dataset 10,000개로구성됨



### [Page 6]
6/37
Perceptron을이용한분류실습- 실습목표

실습목표: MNIST 손글씨데이터를분류하는단층perceptron 모델학습
•
입력: 손글씨이미지(28x28x1)
•
출력: 0~9까지숫자들의정답확률
Perceptron
예측결과
28 x 28 x 1
정답확률
클래스
0.1666
0
0.0889
1
…
…
0.750
5
…
…
0.0008
8
0.0113
9


| 클래스 | 정답 확률 |
| --- | --- |
| 0 | 0.1666 |
| 1 | 0.0889 |
| … | … |
| 5 | 0.750 |
| … | … |
| 8 | 0.0008 |
| 9 | 0.0113 |


### [Page 7]
7/37
Perceptron을이용한분류실습

데이터셋다운로드
•
(1) 구글드라이브실행
•
(2) 데이터셋및파라미터저장을위한폴더생성(Ex. dataset, parameters)
•
(3) 새로운노트북파일생성및실행(Ex. MLP_Experiment_Part1.ipynb)
1
2
Colab 실행화면
실습코드보관용폴더
데이터셋보관용폴더
파라미터보관용폴더


| 실습코드 데이터셋 보 파라미터 |  |  |
| --- | --- | --- |
|  | 파라미터 | 보관용 폴더 |


### [Page 8]
8/37
Perceptron을이용한분류실습

데이터셋다운로드
•
(4) Colab에서구글드라이브연결
•
(5) 데이터셋을저장할경로가져오기변수로저장
1
구글드라이브연결
폴더경로확인
폴더경로저장
3
2
4



### [Page 9]
9/37
Perceptron을이용한분류실습

데이터셋다운로드
•
(6) 패키지선언및MNIST 데이터셋저장
1
2



### [Page 10]
10/37
Perceptron을이용한분류실습

데이터셋다운로드
•
(7) MNIST 데이터셋저장여부확인
•
(8) MNIST 데이터셋형상확인
데이터셋저장여부확인
60000
1 x 28 x 28
5



### [Page 11]
11/37
Perceptron을이용한분류실습

[참고] MNIST 데이터셋형상
Ref.: https://joongheon.github.io



### [Page 12]
12/37
Perceptron을이용한분류실습

[참고] MNIST 데이터셋형상
MNIST train dataset
img0
5
0
4
img1
img2
img
59999
8
Image
Label
…
…
MNIST test dataset
img0
7
2
1
img1
img2
img
9999
6
Image
Label
…
…



### [Page 13]
13/37
Perceptron을이용한분류실습

MNIST 데이터셋의perceptron 입력방법
•
Perceptron의각노드는한번에1개의값을입력받을수있음
•
따라서2D 형태이미지의전처리가필요함
28 x 28 이미지
입력불가!



### [Page 14]
14/37
Perceptron을이용한분류실습

MNIST 데이터셋의perceptron 입력방법
•
Perceptron의각노드는한번에1개의값을입력받을수있음
•
따라서2D 형태이미지의전처리가필요함평탄화
28 x 28 x 1
784 x 1
평탄화



### [Page 15]
15/37
Perceptron을이용한분류실습

MNIST 데이터셋의perceptron 입력방법
•
Perceptron의각노드는한번에1개의값을입력받을수있음
•
따라서2D 형태이미지의전처리가필요함평탄화
평탄화
Perceptron
예측결과
28 x 28
784 x 1
10 x 1
0~9까지숫자들의예측결과



### [Page 16]
16/37
Perceptron을이용한분류실습

MNIST 데이터셋의perceptron 입력방법
•
Perceptron의각노드는한번에1개의값을입력받을수있음
•
따라서2D 형태이미지의전처리가필요함평탄화
torch.Tensor.view 함수를통해tensor의형태를변경할수있음
(https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)



### [Page 17]
17/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(1) 단층perceptron 모델정의

Pytorch에서구현하는모델은반드시2가지함수를선언해야함: __init__, forward



### [Page 18]
18/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(1) 단층perceptron 모델정의

Pytorch에서구현하는모델은반드시2가지함수를선언해야함: __init__, forward
__init__() 함수에포함해야하는정보
Parameter를가지는layer 정보
(Fully connected layer, Convolutional layer 등)
Activation function
[참고] torch.nn.Linear = Fully connected layer



### [Page 19]
19/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(1) 단층perceptron 모델정의

Pytorch에서구현하는모델은반드시2가지함수를선언해야함: __init__, forward
forward() 함수에포함해야하는정보
모델의동작순서
각layer, 함수의입출력관계



### [Page 20]
20/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(2) Hyper-parameter 지정
실습에사용되는hyper-parameter
Batch size: 100
Learning rate: 0.1
Epoch: 15회학습
Loss function: Cross entropy error
Optimizer: SGD



### [Page 21]
21/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(2) Hyper-parameter 지정
[참고사항] torch.nn.CrossEntropyLoss() 함수
①
예측값들에대해자동으로softmax 적용
②
정답값과예측값을이용해cross entropy loss 측정



### [Page 22]
22/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(2) Hyper-parameter 지정
Batch 단위학습을위해DataLoader 함수사용



### [Page 23]
23/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(3) Perceptron 학습을위한반복문선언
전체데이터에대한반복: epoch
1 epoch 내의배치에대한반복: iteration
모든배치에대한평균loss 값계산
1) 입력이미지에대해forward pass
2) 예측값, 정답을이용해loss 계산
3) 모든weight에대해편미분값계산
4) 파라미터업데이트
(1)
(2)
(3)
(4)



### [Page 24]
24/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(3) Perceptron 학습을위한반복문선언



### [Page 25]
25/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(4) 학습이완료된weight parameter 저장및확인
Weight parameter 저장
저장된weight parameter 불러오기(예시)



### [Page 26]
26/37
Perceptron을이용한분류실습

MNIST 분류를위한단층perceptron 모델학습
•
(5) MNIST test dataset 분류성능확인
예측값이가장높은숫자(0~9)와
정답데이터가일치한지확인
정답률: 88.9%



### [Page 27]
27/37
Contents
1.
Perceptron을이용한분류실습
2.
Multi-layer Perceptron 실습



### [Page 28]
28/37
Multi-layer Perceptron 실습

MNIST 분류를위한Multi-layer Perceptron 모델학습: 2-layer
•
(1) 패키지선언및MNIST 데이터셋다운(이전실습자료와동일)
1
2
3



### [Page 29]
29/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(2) MLP 모델정의

fc1의출력노드와fc2의입력노드가반드시동일하여야함
Multi-layer Perceptron 실습



### [Page 30]
30/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(3) Hyper-parameter 지정
실습에사용되는hyper-parameter
Batch size: 100
Learning rate: 0.1
Epoch: 15회학습
Loss function: Cross entropy error
Optimizer: SGD
Multi-layer Perceptron 실습



### [Page 31]
31/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(3) Network training 을위한반복문선언
전체데이터에대한반복: epoch
1 epoch 내의배치에대한반복: iteration
모든배치에대한평균loss 값계산
1) 입력이미지에대해forward pass
2) 예측값, 정답을이용해loss 계산
3) 모든weight에대해편미분값계산
4) 파라미터업데이트
Multi-layer Perceptron 실습


|  전체 데이 | 터에 대한 반복: epoch |
| --- | --- |
|  1 epoch | 내의 배치에 대한 반복: iteration |


### [Page 32]
32/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(3) Network training 을위한반복문선언결과확인
Multi-layer Perceptron 실습



### [Page 33]
33/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(4) 학습이완료된Network의Weight parameter 저장및확인
Multi-layer Perceptron 실습



### [Page 34]
34/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(5) MNIST Test dataset 분류성능확인

Single layer의성능보다약5% 높은성능
Multi-layer Perceptron 실습
정답률: 94.4%



### [Page 35]
35/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(6) 예측결과값확인
Multi-layer Perceptron 실습
Test dataset 중첫번째Image에
대한예측값확인



### [Page 36]
36/37

MNIST 분류를위한Multi-layer Perceptron (MLP) 모델학습: 2-layer
•
(7) 정답이미지확인
Multi-layer Perceptron 실습



### [Page 37]
37/37
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Dept. of AI
Dong-A University, Busan, Rep. of Korea

