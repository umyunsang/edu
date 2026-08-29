## --- [Page 1] ---
12주차이론
(CNN 주요설계모듈)

1

## --- [Page 2] ---
Convolutional Neural Networks - 3D 데이터의Convolution 연산


3D 이미지(RGB) 입력에대한2D convolution 연산

Input image
Filter
(Kernel)

4x4x3
3x3x3

*

2x2x1

Output
feature map

## --- [Page 3] ---
Convolution Layer

Convolutional neural network (CNN)

3

32

April 18, 2017

Fei-Fei Li & Justin Johnson & Serena Yeung

출처: cs231n_2017_lecture5

32x32x3 image  
5x5x3 filter

32


Overview

Wx+b

## --- [Page 4] ---
Convolution Layer

Convolutional neural network (CNN)

3

32

April 18, 2017

Fei-Fei Li & Justin Johnson & Serena Yeung

출처: cs231n_2017_lecture5

32

One number = ReLU(Wx+b)


Overview

## --- [Page 5] ---
Convolution Layer

Convolutional neural network (CNN)


Overview

April 18, 2017

Fei-Fei Li & Justin Johnson & Serena Yeung

출처: cs231n_2017_lecture5

32

32

3

Convolution Layer

activation maps

6

28

28

•
For example, if we had 6 5x5x3 filters, we`ll get 6 separate activation maps

## --- [Page 6] ---
Convolution Layer

Convolutional neural network (CNN)


Overview

April 18, 2017

Fei-Fei Li & Justin Johnson & Serena Yeung

출처: cs231n_2017_lecture5

•
ConvNet is a sequence of Convolutional Layers, interspersed with activation functions

32

32

3

CONV,  
ReLU
e.g.
6
5x5x3
filters
28

28

6

CONV,  
ReLU
e.g. 10  
5x5x6  
filters

CONV,  
ReLU

….

10

24

24

## --- [Page 7] ---
Example of graphical CNN representation

CNN trains the multiple filters (Kernel).

## --- [Page 8] ---
Convolutional Neural Network (CNN) 이론


CNN을이용한classification model 설계시주의사항

•
일반적으로CNN의feature map을평탄화한이후fully connected layer에입력함

8

LeNet-5 구조

Convolution 1
Convolution 2
Pooling
Pooling

평탄화& FC1

FC2

FC3

Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition

## --- [Page 9] ---
LeNet-5 구조

9

C1

Ref.: [Proc. on IEEE 1998] Gradient-based learning applied to document recognition

S2

S4

C5

F6

## --- [Page 10] ---
10/21
10

LeNet-5 C3

## --- [Page 11] ---
딥러닝주요모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)


Bottleneck Layer


구성요소적용예시

11

## --- [Page 12] ---
딥러닝모델구성요소

12


ImageNet Large Scale Visual Recognition Challenge (ILSVRC)

•
대용량의이미지셋(1000개의클래스) 에대한이미지분류알고리즘성능평가대회

Ref.: https://bskyvision.com/425
첫번째CNN 적용

Top-5 error

## --- [Page 13] ---
딥러닝모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)


Bottleneck Layer


구성요소적용예시

13

## --- [Page 14] ---
딥러닝모델구성요소- Skip connection

14


[CVPR 2015] Deep Residual Learning for Image Recognition (Kaiming He, Microsoft Research)

•
ImageNet dataset에대해20-layer, 56-layer 모델의성능비교

깊은모델의성능이더떨어지는것을확인

Ref.: cs231n.stanford.edu, Lecture 9

## --- [Page 15] ---
딥러닝모델구성요소- Skip connection

15


[CVPR 2015] Deep Residual Learning for Image Recognition (Kaiming He, Microsoft Research)

•
잔차신호(Residual)을학습하게설계함으로써문제해결시도

Ref.: cs231n.stanford.edu, Lecture 9

잔차신호(Residual)

Skip connection

## --- [Page 16] ---
딥러닝모델구성요소- Skip connection

16


[CVPR 2015] ResNet (Kaiming He, Microsoft Research)

•
3x3 convolution 2개,

skip connection으로구성된Residual block 제안

•
여러개의Residual block을이용해제안기법인ResNet을구현

Ref.: cs231n.stanford.edu, Lecture 9

Inference

진행순서

Residual block

## --- [Page 17] ---
딥러닝모델구성요소- Skip connection

17


[CVPR 2015] ResNet (Kaiming He, Microsoft Research)

•
Inference가진행됨에따라

feature map의width, height은감소됨

feature map의channel은증가됨

Ref.: cs231n.stanford.edu, Lecture 9

Feature map size: 7x7x512

Feature map size: 112x112x64

## --- [Page 18] ---
딥러닝모델구성요소- Skip connection

18


[CVPR 2015] ResNet (Kaiming He, Microsoft Research)

•
Convolution layer의최종출력은

Global Average Pooling (GAP)을통해

Fully connected layer에입력됨

Ref.: cs231n.stanford.edu, Lecture 9

C = 512

H = 7

W = 7

GAP
C = 512

H = 1

W = 1

C: Channel
W: Width
H: Height
GAP: Global Average Pooling

## --- [Page 19] ---
19/29

Skip Connection (ResNet)

## --- [Page 20] ---
딥러닝모델구성요소- Skip connection

20


[CVPR 2015] ResNet (Kaiming He, Microsoft Research)

•
2015년ImageNet 대회에서는여러개의Layer에대해실험한결과를제안함

Ref.: cs231n.stanford.edu, Lecture 9

5개의모델에대해실험결과를제시

각모델에대한복잡도

## --- [Page 21] ---
딥러닝모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)


Bottleneck Layer


구성요소적용예시

21

## --- [Page 22] ---
딥러닝모델구성요소- Dense connection

22


[CVPR 2017] Densely Connected Convolutional Networks (Gao Huang, Cornell University)

•
이전Layer의출력feature map을이후layer에서재사용

N x M x 64

Conv1

[3×3×64]

×64

N x M x 64

Conv2

[3×3×64]

×64

N x M x 64

[3×3×128]

×64

Dense connections

[3×3×192]

×64


|  |  |  |
| --- | --- | --- |
|  |  |  |
| N x M x 64 | N x M x 64 | N x M x 64 |

## --- [Page 23] ---
딥러닝모델구성요소- Dense connection

23


[CVPR 2017] Densely Connected Convolutional Networks (Gao Huang, Cornell University)

•
이전Layer의출력feature map을이후layer에서재사용

•
Layer가깊어짐에따라파라미터의개수가증가함

N x M x 64

Conv1

[3×3×64]

×64

N x M x 64

Conv2

[3×3×64]

×64

N x M x 64

[3×3×128]

×64

Dense connections

[3×3×192]

×64


|  |  |  |
| --- | --- | --- |
|  |  |  |
| N x M x 64 | N x M x 64 | N x M x 64 |

## --- [Page 24] ---
딥러닝모델구성요소- Dense connection

24


[CVPR 2017] Densely Connected Convolutional Networks (Gao Huang, Cornell University)

•
여러개의Convolution layer를가지는Dense Block을정의,

Dense Block 들로구성되는DenseNet을제안

DenseNet 네트워크구조예시

## --- [Page 25] ---
25/29

Dense Connection

## --- [Page 26] ---
딥러닝모델구성요소- Dense connection

26


[CVPR 2017] Densely Connected Convolutional Networks (Gao Huang, Cornell University)

•
ResNet보다깊은구조에대해학습을진행

5개의모델에대해실험결과를제시

## --- [Page 27] ---
딥러닝모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)


Bottleneck Layer


구성요소적용예시

27

## --- [Page 28] ---
딥러닝모델구성요소- Channel attention


[CVPR 2018] Squeeze-and-Excitation Networks

•
Feature map의각채널에대해중요도를부여

28

일반적인feature map
Attention을통해중요도가부여된

feature map

## --- [Page 29] ---
딥러닝모델구성요소- Channel attention


[CVPR 2018] Squeeze-and-Excitation Networks

•
일반적으로GAP를통해작아진feature map에대해

Fully connected layer 또는1x1 convolution이사용됨

29

## --- [Page 30] ---
딥러닝모델구성요소- Channel attention


[CVPR 2018] Squeeze-and-Excitation Networks

•
기존에제안된모델들에대해Channel attention을적용하여결과제시

30

Channel attention을통한성능향상

## --- [Page 31] ---
딥러닝모델구성요소- Channel attention


[CVPR 2018] Squeeze-and-Excitation Networks

•
기존에제안된모델들에대해Channel attention을적용하여결과제시

31

복잡도는거의증가되지않음

## --- [Page 32] ---
딥러닝모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)


Bottleneck Layer


구성요소적용예시

32

## --- [Page 33] ---
33/29

Bottleneck Layer


1x1 컨볼루션사용feature map의개수를줄이는목적으로사용

Ref) https://bskyvision.com/539


Memory: [(5 x 5 x 480) + 1] x 48 + (14 x 14 x 48)


연산횟수: (14 x 14 x 48) x (5 x 5 x 480) = 약112.9M


Memory: [(1 x 1 x 480) + 1] x 16 + [(5 x 5 x 16) + 1] x 48 +

(14 x 14 x 16) + (14 x 14 x 48)


연산횟수: (14 x 14 x 16)*(1 x 1 x 480) + (14 x 14 x 48)*(5 x 5 x 16) = 약5.3M

## --- [Page 34] ---
딥러닝모델구성요소


Skip connection (ResNet, 2015)


Dense connection (DenseNet, 2017)


Channel attention (SENet, 2018)


Bottleneck Layer


구성요소적용예시

34

## --- [Page 35] ---
딥러닝모델구성요소- 구성요소적용예시

[CVPR 2016] Accurate Image Super-Resolution Using Very Deep Convolutional Networks (VDSR)

•
3x3 Convolution 20개사용, SR 분야에대해최초로Residual Learning 적용

35

VDSR 구조
Input conv. layer

[3x3x1]x64

Output conv. layer

[3x3x64]x1

Conv. layer 18개

[3x3x64]x64

Interpolation된

image 입력

(Bi-cubic)

## --- [Page 36] ---
딥러닝모델구성요소- 구성요소적용예시

[CVPR 2017] Image Super-Resolution Using Dense Skip Connections (SR-DenseNet)

•
Dense connection을이용해Dense block 8의출력단은1,000개이상의feature map을사용

36