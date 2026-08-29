## --- [Page 1] ---
1/21

Dong-A Univ. (ISPL)

컴퓨터AI공학부AI학과

2024년1학기인공지능

## --- [Page 2] ---
2/21

11주차실습(LeNet-5)

2

## --- [Page 3] ---
3/21

Convolutional Neural Network (CNN) 이론


CNN을이용한classification model 설계시주의사항

•
일반적으로CNN의feature map을평탄화한이후fully connected layer에입력함

3

LeNet-5 구조

Convolution 1
Convolution 2
Pooling
Pooling

평탄화& FC1

FC2

FC3

Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition

## --- [Page 4] ---
4/21

LeNet-5 구조

4

C1

Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition

S2

S4

C5

F6

## --- [Page 5] ---
5/21
5

LeNet-5 C3

## --- [Page 6] ---
6/21

Review – Convolutional layer


3D 이미지(RGB) 입력에대한2D convolution 연산

(1)

Input image

4x4x3
Conv. filter

3x3x3


|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 |  |
| --- | --- |
|  |  |

## --- [Page 7] ---
7/21

Review – Convolutional layer


3D 이미지(RGB) 입력에대한2D convolution 연산

Output
feature map

2x2x1

Input image

4x4x3
Conv. filter

3x3x3


|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  |  | 4 |  | 2 |  | 1 |  | 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 3 |  | 0 |  | 6 |  | 5 |  | 4 |
| 1 |  | 2 |  | 3 |  | 0 |  | 3 | 2 |
| 0 |  | 1 |  | 2 |  | 3 |  | 0 | 5 |
| 3 |  | 0 |  | 1 |  | 2 |  | 1 |  |
| 2 |  | 3 |  | 0 |  | 1 |  |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 |  |
| --- | --- |
|  |  |

| 63 | 55 |
| --- | --- |
|  |  |

|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  | 4 | 2 | 1 | 2 |
| --- | --- | --- | --- | --- | --- |
|  | 3 | 0 | 6 | 5 | 4 |
| 1 | 2 | 3 | 0 | 3 | 2 |
| 0 | 1 | 2 | 3 | 0 | 5 |
| 3 | 0 | 1 | 2 | 1 |  |
| 2 | 3 | 0 | 1 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

|  |  | 4 | 0 | 2 |
| --- | --- | --- | --- | --- |
|  | 0 | 1 | 3 | 0 |
| 2 | 0 | 1 | 2 | 2 |
| 0 | 1 | 2 | 0 |  |
| 1 | 0 | 2 |  |  |

| 63 | 55 |
| --- | --- |
| 18 |  |

| 63 | 55 |
| --- | --- |
| 18 | 51 |

## --- [Page 8] ---
8/21

Review – Convolutional layer


3D 이미지(RGB) 입력에대한2D convolution 연산

•
Output feature map의채널수는filter의개수(N)와같음

Input image

Conv. filters

[3x3x3]xN

4x4x3
3x3x3 filters

*

2x2x1

Output
feature map

…

filter 1

#filters (N)

…

2x2x1

filter N

2x2xN

## --- [Page 9] ---
9/21

Review – Convolutional layer


Stride: Convolution 연산의step size

•
Stride가커질수록feature map의크기는작아짐

Stride=1

Input image
Filter

Output
feature map

Stride=2

Input image

Filter
Output
feature map


| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 15 |  |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 |  |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |
| 2 | 3 | 0 | 1 | 2 | 3 | 0 |
| 1 | 2 | 3 | 0 | 1 | 2 | 3 |
| 0 | 1 | 2 | 3 | 0 | 1 | 2 |
| 3 | 0 | 1 | 2 | 3 | 0 | 1 |

| 15 | 16 |  |  |  |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 | 17 |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |

## --- [Page 10] ---
10/21

Review – Convolutional layer


Padding: Input image 주변값을특정값(주로0) 으로채워줌

•
Convolution 연산으로boundary 정보가소실되는문제를방지


| 1 | 2 | 3 | 0 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 3 | 0 | 1 | 2 |
| 2 | 3 | 0 | 1 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

| 15 | 16 |
| --- | --- |
| 6 | 15 |

| 0 | 0 | 0 | 0 | 0 | 0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 3 | 0 | 0 |
| 0 | 0 | 1 | 2 | 3 | 0 |
| 0 | 3 | 0 | 1 | 2 | 0 |
| 0 | 2 | 3 | 0 | 1 | 0 |
| 0 | 0 | 0 | 0 | 0 | 0 |

| 7 | 12 | 10 | 2 |
| --- | --- | --- | --- |
| 4 | 15 | 16 | 10 |
| 10 | 6 | 15 | 6 |
| 8 | 4 | 7 | 3 |

| 2 | 0 | 1 |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 0 | 2 |

## --- [Page 11] ---
11/21

Review – Convolutional layer

Convolution 연산의출력크기계산

•
출력의크기는filter size, stride, padding size에따라달라짐

•
출력크기는정수로나누어떨어져야함

2
Output Height
1
H
P
FH
OH
S






2
Output Width
1
W
P
FW
OW
S






(H, W): Input data size

(FW, FH): Filter size

P: Padding size

S: Stride

## --- [Page 12] ---
12/21

Implementation of Convolutional Layer

torch.nn.Conv2d() 함수를이용한합성곱계층구현

Ref.: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d

①in_channels: 입력특징맵의채널개수

②out_channels: 출력특징맵의채널개수

③kernel_size: 커널크기

④stride: stride 크기

⑤padding: padding 크기

## --- [Page 13] ---
13/21

Implementation of Convolutional Layer

torch.nn.Conv2d() 함수를이용한합성곱계층구현

Ref.: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d

Convolutional layer

[KW x KH x C] x N, stride, padding

self.conv =  nn.Conv2d (in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)

[5x5x6]x16, s1, p0

Input feature map
Output feature map

6
16

## --- [Page 14] ---
14/21

CIFAR-10 분류실습– CNN을이용한분류

CIFAR-10 dataset 형상

•
32x32x3 (RGB) 이미지, 10개의클래스

•
Train: 50,000개, Test: 10,000개

CIFAR-10 dataset 예시

LeNet-5 신경망구조

## --- [Page 15] ---
15/21

CIFAR-10 분류실습– CNN을이용한분류

입출력구조확인

평탄화
FC layers
예측결과

32 x 32 x 3
3072 x 1
10 x 1

Conv. 
layers
예측결과

32 x 32 x 3
10 x 1

## --- [Page 16] ---
16/21

CIFAR-10 분류실습– CNN을이용한분류

입출력구조확인

CNN model 예시

32 x 32 x 3

Conv. 1
[5x5x3]x32

28x28x32

Conv. 2
[5x5x32]x32

24x24x32

출력의형태가cube 이므로

예측결과를확인할수없음

…

## --- [Page 17] ---
17/21

CIFAR-10 분류실습– CNN을이용한분류

입출력구조확인

평탄화
FC layers
예측결과

32 x 32 x 3
3072 x 1
10 x 1

Conv. 
layers

32 x 32 x 3
Feature maps

평탄화
FC layers
예측결과

10 x 1
Feature vector

## --- [Page 18] ---
18/21

CIFAR-10 분류실습– CNN을이용한분류

LeNet-5 신경망구조

•
2개의convolutional layer, 3개의fully connected layer로구성

Input image (CIFAR-10)

32x32x3
Conv 1
[5x5x3]x6, s1, p0

Conv 2
[5x5x6]x16, s1, p0

FC3

Output classes

10x1

Avg
Pooling


|  | FC2 |
| --- | --- |
| FC1 ng |  |

## --- [Page 19] ---
19/21

CIFAR-10 분류실습– CNN을이용한분류

10주차LMS 강의콘텐츠에업로드되어있는base code 다운로드

## --- [Page 20] ---
20/21

CIFAR-10 분류실습– CNN을이용한분류

10주차LMS 강의콘텐츠에업로드되어있는base code 다운로드

LeNet5 모델구조에맞추어코드작성

## --- [Page 21] ---
21/21

CIFAR-10 분류실습– CNN을이용한분류

LeNet-5 모델구조작성참고사항

•
참고자료: https://pytorch.org/docs/stable/nn.html

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

## --- [Page 22] ---
22/21

CIFAR-10 분류실습– CNN을이용한분류

LeNet-5 모델구조작성참고사항

•
Filter size: 5x5, Stride: 1, Padding: 0

LeNet-5 구조

32x32x3

Input image

[5x5x3]x6, s1, p0

ReLU

AvgPool, k2, s2

[5x5x6]x16, s1, p0

ReLU

AvgPool, k2, s2

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer
평
탄
화

84x10

ReLU

120x84

ReLU

400x120

예측결과

10x1
[KW x KH x C] x N

## --- [Page 23] ---
23/21

CIFAR-10 분류실습– CNN을이용한분류

Convolution 연산의출력크기계산

•
출력의크기는filter size, stride, padding size에따라달라짐

•
출력크기는정수로나누어떨어져야함

2
Output Height
1
H
P
FH
OH
S






2
Output Width
1
W
P
FW
OW
S






(H, W): Input data size

(FW, FH): Filter size

P: Padding size

S: Stride

## --- [Page 24] ---
24/21

CIFAR-10 분류실습– CNN을이용한분류

LeNet-5 모델구조작성참고사항

•
Filter size: 5x5, Stride: 1, Padding: 0

LeNet-5 구조

32x32x3

Input image

[5x5x3]x6, s1, p0

ReLU

AvgPool, k2, s2

[5x5x6]x16, s1, p0

ReLU

AvgPool, k2, s2

평
탄
화

84x10

ReLU

120x84

ReLU

400x120

예측결과

10x1

5x5x16

특징맵평탄화

5x5x16 400x1

## --- [Page 25] ---
25/21

Appendix – 더높은정확도를가지는LeNet-5 설계

1. Pooling layer 변경: Average pooling Max pooling

2. Convolutional layer channel 개수변경: 6 32, 64

LeNet-5 구조

32x32x3

Input image

[5x5x3]x64, s1, p0

ReLU

MaxPool, k2, s2

[5x5x64]x16, s1, p0

ReLU

MaxPool, k2, s2

평
탄
화

84x10

ReLU

120x84

ReLU

400x120

예측결과

10x1

## --- [Page 26] ---
26/21

Appendix – 더높은정확도를가지는LeNet-5 설계

3. Learning rate control

•
1 ~ 74 epoch: 0.001

•
75 ~ 149 epoch: 0.0005

•
150 ~ 200 epoch: 0.00025

# hyper-parameter 변경

training_epochs = 200

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)

## --- [Page 27] ---
27/21

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea