## --- [Page 1] ---
Dong-A Univ. (ISPL)


|  |  |
| --- | --- |
|  |  |
|  | 1/30 |

## --- [Page 2] ---
2/30

딥러닝주요모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)

Skip connection
Dense connection
Channel attention

## --- [Page 3] ---
3/30

<VGG-16 구조>

3 x 3 x 3 x 64
(S1/P1)

3 x 3 x 64 x 64
(S1/P1)

2x2 Max Pooling (S2)
112 x 112 x 64

3 x 3 x 64 x 128
(S1/P1)

3 x 3 x 128 x 128
(S1/P1)

Orig. Network - VGGNet(VGG-16)

기존VGGNet을사용하여실습시많은시간소요금일실습시간소화된모델사용

## --- [Page 4] ---
4/30

Wrap-up

torch.nn.Conv2d() 함수를이용한합성곱계층구현

Ref.: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d

①in_channels: 입력특징맵의채널개수

②out_channels: 출력특징맵의채널개수

③kernel_size: 커널크기

④stride: stride 크기

⑤padding: padding 크기

## --- [Page 5] ---
5/30

Wrap-up

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

## --- [Page 6] ---
6/30

Modified Network - VGGNet(VGG-16)

기존VGG16을CNN layer를6개로간소화

CIFAR-10 데이터셋분류를위해네트워크구조변경(32x32x3, 10 classes)

실제VGG 네트워크는Max Pooling을사용

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

[3x3x64]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

예측결과

10x1

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

AvgPool

[3x3x32]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3

## --- [Page 7] ---
7/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

VGG 간소화모델코드공유

•
LMS 12주차VGG base code 다운로드

•
실습시[3] Model 구조선언부분만수정

## --- [Page 8] ---
8/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

실습Network base 구조(Stride와Padding size는1로고정)

CIFAR-10 데이터셋분류를위해네트워크구조변경(32x32x3, 10 classes)

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

[3x3x64]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

예측결과

10x1

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

AvgPool

[3x3x32]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3

## --- [Page 9] ---
9/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

실습Network base 구조(Stride와Padding size는1로고정)

CIFAR-10 데이터셋분류를위해네트워크구조변경(32x32x3, 10 classes)

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

[3x3x64]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

예측결과

10x1

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

AvgPool

[3x3x32]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3

## --- [Page 10] ---
10/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

실습Network base 구조(Stride와Padding size는1로고정)

CIFAR-10 데이터셋분류를위해네트워크구조변경(32x32x3, 10 classes)

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

[3x3x64]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

예측결과

10x1

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

AvgPool

[3x3x32]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3


하이퍼파라미터

•
Training epoch: 20

•
Batch size: 100

•
Learning rate: 0.1

•
Loss function: Cross Entropy Loss

•
Optimizer: SGD

## --- [Page 11] ---
11/30

딥러닝주요모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)

Skip connection
Dense connection
Channel attention

## --- [Page 12] ---
12/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

[3x3x64]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

AvgPool

[3x3x32]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3

+

주의사항: Skip connection은Width, Height, Channel이모두같아야사용가능

+
+

32x32x3

[3x3x3]x32

[3x3x32]x64

[3x3x64]x256

32x32x32
16x16x32
16x16x64
8x8x64
8x8x256

## --- [Page 13] ---
13/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험

•
Skip connection을위한convolution layer 선언

## --- [Page 14] ---
14/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험

•
Skip connection 적용

## --- [Page 15] ---
15/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험

Skip connection적용을위해Conv. 입력저장

## --- [Page 16] ---
16/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험

Width, Height, Channel을맞춰주기위한Conv. 적용

## --- [Page 17] ---
17/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험

Skip connection 적용코드

## --- [Page 18] ---
18/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Skip connection 추가실험결과확인

Training 결과
Test 결과

## --- [Page 19] ---
19/30

딥러닝주요모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)

Skip connection
Dense connection
Channel attention

## --- [Page 20] ---
20/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Dense connection 추가실험

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

주의사항: Dense connection (torch.cat)은width, height이동일해야적용가능

[3x3x99]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

AvgPool

[3x3x35]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3

C
C

32x32x3
16x16x35

Concat. 이후출력크기

32x32x35
Concat. 이후출력크기

16x16x99

## --- [Page 21] ---
21/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Dense connection 추가실험

•
Dense 추가로인한Input channels 변경

## --- [Page 22] ---
22/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Dense connection 추가실험

•
Dense를위한Concat. 코드추가

out1 = torch.cat([x, out1], dim=1)

Feature map 형상: (Batch_size, Channel, Width, Height)

dim:          0                1            2         3

## --- [Page 23] ---
23/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Dense connection 추가실험결과확인

Training 결과
Test 결과

## --- [Page 24] ---
24/30

딥러닝주요모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)

Skip connection
Dense connection
Channel attention

## --- [Page 25] ---
25/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention (CA) 추가실험

: Convolution layer

: Activation function

: Pooling layer

: Fully connected layer

[3x3x64]x128

ReLU

[3x3x128]x256

ReLU

AvgPool

평
탄
화

4096x512

ReLU

256x10

[3x3x3]x16

ReLU

[3x3x16]x32

[3x3x32]x32

ReLU

[3x3x32]x64

ReLU

AvgPool

32x32x3

CA

σ

Output
Input

σ
: Sigmoid 함수

: Channel 간곱

GAP

ReLU

[1x1x64]x64

[1x1x64]x64


| ReLU | AvgPool |
| --- | --- |


| 512x256 | ReLU |
| --- | --- |


## --- [Page 26] ---
26/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention (CA) 추가실험

•
CA 구성요소정의

## --- [Page 27] ---
27/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention (CA) 추가실험

•
CA 동작코드작성

ex)

4x4x4
1x1x4

GAP

## --- [Page 28] ---
28/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention (CA) 추가실험

•
CA 동작코드작성

ex)

1x1x4
1x1x4

Weight

## --- [Page 29] ---
29/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention (CA) 추가실험

•
CA 동작코드작성

ex)

1x1x4
4x4x4

1
1
1
1
1

1
1
1
1

1
1
1
1

1
1
1
1

## --- [Page 30] ---
30/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention (CA) 추가실험

•
CA 동작코드작성

ex)

4x4x4 (CA_map)
4x4x4 (input)

Output

## --- [Page 31] ---
31/30

CIFAR10 분류실습- CNN을이용한분류(VGGNet)

Channel attention 추가실험결과확인

Training 결과
Test 결과

## --- [Page 32] ---
32/30

CNN 구성요소조합실험

조합실험: Skip, Dense Connection, Channel Attention

: Convolution layer

: Activation function

: Max Pooling layer

: Fully connected layer

주의사항(1): Skip connection은Width, Height, Channel이모두같아야사용가능

주의사항(2): Dense connection (torch.cat)은width, height이동일해야적용가능

[3x3x99]x128

ReLU

[3x3x128]x256

ReLU

MaxPool

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

[3x3x3]x16

ReLU

[3x3x16]x32

ReLU

MaxPool

[3x3x35]x32

ReLU

[3x3x32]x64

ReLU

MaxPool

32x32x3

C
C
+
+

Channel Atten.

[3x3x3]x32

Dense connection

Skip
connection

Dense connection

[3x3x35]x64

Skip connection

+

[3x3x99]x256

Skip connection

ReLU

ReLU

ReLU

## --- [Page 33] ---
33/30

CNN 구성요소조합실험

Resnet

: Convolution layer

: Activation function

: Max Pooling layer

: Fully connected layer

ReLU

[3x3x3]x32

평
탄
화

4096x512

ReLU

512x256

ReLU

256x10

32x32x3

+

Residual 
block (32)

Residual 
block (32)

Residual 
block (32)

ReLU

[3x3x32]x64

Residual 
block (64)

Residual 
block (64)

Residual 
block (64)

ReLU

[3x3x3]x32

ReLU

[3x3x3]x32

## --- [Page 34] ---
34/30

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea