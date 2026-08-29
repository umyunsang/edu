### [Page 1]
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능


|  |  |
| --- | --- |
|  |  |
|  | 컴퓨터AI 2024년 1 |


### [Page 2]
주요CNN 구조소개

ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
•
대용량의이미지셋(1000개의클래스) 에대한이미지분류알고리즘성능평가대회



### [Page 3]
AlexNet(2012)
Ref) https://bskyvision.com/425

2개의GPU로병렬연산을수행하기위해병렬적인구조로설계

총8개의layer로구성(5개의컨볼루션레이어, 3개의Fully connected layer) 
Convolution layers
Fully connected layers



### [Page 4]
AlexNet(2012)
Ref) https://bskyvision.com/425

Activation function으로ReLU함수를사용(TanH함수를사용할때마다6배빠름)

Over-fitting을막기위해Dropout을사용



### [Page 5]
AlexNet(2012)

3x3 Overlapping pooling 사용(Stride = 2)

Data augmentation으로데이터양을증가overfitting 문제줄임
Ref) https://bskyvision.com/425



### [Page 6]
AlexNet(2012)
Ref) https://bskyvision.com/425
Convolution layers
Fully connected layers
[11 x 11 x 3] x 48
(S4/P0)

AlexNet (3x3 Max Pooling@S2, ReLU)
Max Pooling
27 x 27 x 48 [5 x 5 x 48] x 128
(S1/P2)
Max Pooling
13 x 13 x 128
Concat
13 x 13 x 256
Input
227 x 227 x 3
[3 x 3 x 256] x 192
(S1/P1)
[3 x 3 x 192] x 192
(S1/P1)
[3 x 3 x 192] x 128
(S1/P1)
Concat
6 x 6 x 256
Max Pooling
6 x 6 x 128
= 9216
Softmax



### [Page 7]



### [Page 8]



### [Page 9]
주요CNN 구조소개

ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
•
대용량의이미지셋(1000개의클래스) 에대한이미지분류알고리즘성능평가대회



### [Page 10]
GooGleNet(2014)

VGGNet을이기고우승을차지한알고리즘(Inception)

총22개layer로구성
<GoogLeNet 구조>
Ref) https://bskyvision.com/539



### [Page 11]
주요CNN 구조소개

ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
•
대용량의이미지셋(1000개의클래스) 에대한이미지분류알고리즘성능평가대회



### [Page 12]
VGGNet(VGG-16 / VGG-19)

Weight parameter의개수와성능에대한trade-off를탐색

Network의깊이가깊어짐에따라높은성능을보임을증명(이후부터네트워크레이어를증가시키는추세가활발히이루어짐)

필터크기는3x3으로고정& ReLU 사용
<VGG-16 구조>
<VGGNet 실험설정예시>
Ref) https://bskyvision.com/504
Conv1_1
Conv1_2
Conv2_1
Conv2_2



### [Page 13]
VGGNet(VGG-16, 2014)

기존높은필터사용을없애고3x3필터로통일
Parameter 수를줄임(3x3x2=18개, 5x5=25개) -> Light Memory
Fast Training
Ref) https://bskyvision.com/504



### [Page 14]
VGGNet(VGG-16 / VGG-19)

Weight parameter의개수와성능에대한trade-off를탐색

Network의깊이가깊어짐에따라높은성능을보임을증명(이후부터네트워크레이어를증가시키는추세가활발히이루어짐)

필터크기는3x3으로고정& ReLU 사용
<VGG-16 구조>
<VGGNet 실험설정예시>
Ref) https://bskyvision.com/504
3 x 3 x 3 x 64
(S1/P1)
3 x 3 x 64 x 64
(S1/P1)
2x2 Max Pooling (S2)
112 x 112 x 64
3 x 3 x 64 x 128
(S1/P1)
3 x 3 x 128 x 
128
(S1/P1)



### [Page 15]
REVIEW (PART1 MLP)

MLP(FC Layer) Forward Propagation: Perceptron, Activation Functions, L1/L2 Loss Functions

MLP(FC Layer) Backward Propagation: Gradient Descent(GD) Method

Various Techniques & Implementations
- Overfitting Problem
- Vanishing Gradient
- Data Argumentation
- Optimization
- Drop-out
- Hyper-parament Control (Ex., Adaptive Learning Rate, mini-batch, epoch, etc…)
- Ablation Works



### [Page 16]
REVIEW (PART2 CNN)

CNN Forward Propagation: Convolution, Max/Avg Pooling, Padding, Stride, Kernel(Filter) 

CNN Backward Propagation

CNN Network Design Schemes: SKIP Connection, Dense Connection, Channel Attention, Bottleneck Layer

Popular CNN Networks: LeNet5, AlexNet, VGG, ResNet

CNN Implementation: LeNet5, VGG



### [Page 17]
CNN 1D BACKPROPAGATION



### [Page 18]
CNN 2D CONVOLUTION OPERATION (FORWARD PASS@S1/P0)



### [Page 19]
CNN 1D CONVOLUTION OPERATION (FORWARD PASS@S1/P0)



### [Page 20]
CNN 1D CONVOLUTION OPERATION (BACKPROPAGATION)
Local Gradient
Global Gradient



### [Page 21]



### [Page 22]
CNN 1D CONVOLUTION OPERATION (FORWARD PASS@S1/P0)



### [Page 23]
CNN 1D CONVOLUTION OPERATION (BACKPROPAGATION)
Local Gradient
Global Gradient



### [Page 24]
CNN 1D CONVOLUTION OPERATION (FORWARD PASS@S1/P0)



### [Page 25]
CNN 1D CONVOLUTION OPERATION (BACKPROPAGATION)
GRADIENT W3?



### [Page 26]
CNN 1D CONVOLUTION OPERATION (BACKPROPAGATION)



### [Page 27]
CNN 1D CONVOLUTION OPERATION (BACKPROPAGATION)
GRADIENT X?



### [Page 28]
CNN 1D CONVOLUTION OPERATION (BACKPROPAGATION)
GRADIENT X?



### [Page 29]
CNN 2D BACKPROPAGATION



### [Page 30]



### [Page 31]
참고
Average Pooling
Max Pooling



### [Page 32]



### [Page 33]



### [Page 34]

