## --- [Page 1] ---
1/14

Dong-A Univ. (ISPL)

컴퓨터AI공학부AI학과

2024년1학기인공지능

## --- [Page 2] ---
2/39

Overall Architecture of Deep Learning

Training Input

x

x

x

Drop-out / BN

Step Function

Sigmoid

ReLU

Parametric ReLU

⋮

Activation Function

Fully Connected Layer

Training Ground Truth

Loss Function

Mean Absolute Error (MAE)

Mean Square Error (MSE)

⋮

Network

Optimization

Gradient Descent

Momentum

Adam

⋮

Forward

Backward (Backpropagation)
Vanishing Gradient
(Activation Function)

𝒉(ȉ)

Test Input
Trained Network

Evaluation Metric

PSNR

SSIM

Total Memory

⋮

Training

Test

Overfitting
(Drop-out / BN)

•
ReLU: Rectified Linear Unit
•
Adam: Adaptive Moment Estimation
•
PSNR: Peak Signal-to-Noise Ratio
•
SSIM: Structural Similarity Index Measure

## --- [Page 3] ---
3/14

Normalization (정규화)

## --- [Page 4] ---
Normalization (정규화)

(1) Min-Max 정규화

(2) 표준정규화

## --- [Page 5] ---
Min-Max 정규화

## --- [Page 6] ---
표준정규화

## --- [Page 7] ---
정규분포(표준정규분포)

## --- [Page 8] ---
8/14

Contents

1.
Gradient Descent

2.
Stochastic Gradient Descent

3.
Momentum

4.
AdaGrad

5.
RMSProp

6.
Adam

## --- [Page 9] ---
9/14

[Review] Descent Method (하강법)


주어진어떤지점에서부터오차가더작은곳으로이동하려는방법


하강법을위해Backpropagation으로가중치와편향을수정하는과정을확인

< Descent Method >

𝐿(𝑦, 𝑦ො)

𝒘
∆𝒘𝒕

𝒘𝒕+ ∆𝒘𝒕→𝒘𝒕ା𝟏

∆𝒘𝒕ା𝟏

𝒘𝒕ା𝟏+ ∆𝒘𝒕ା𝟏→𝒘𝒕ା𝟐

1.
낮은곳으로조금씩이동

2.
𝒘 는계속변경

3.
최소지점까지이동

𝒘𝒏ି𝟏+ ∆𝒘𝒏ି𝟏→𝒘𝒏

…

𝒘𝒕ା𝟏
𝒘𝒕
𝒘𝒕ା𝟐

Start

f
𝒛

𝒙

𝒚

𝝏𝒛
𝝏𝒙

𝝏𝒛
𝝏𝒚

local gradient

𝝏𝑳
𝝏𝒛

gradients

Optimization

𝒘𝒕ା𝟏←𝒘𝒕−𝝁𝛁𝒇(𝒘𝒕),

학습률(learning rate, step size)

탐색방향

𝒘𝒉𝒆𝒓𝒆  𝛁𝒇𝒘𝒕∆𝒘< 𝟎

## --- [Page 10] ---
10/14

[Review] Descent Method (하강법)


하강법보다빠르게내려가려면어떻게할까?


초기위치에따라도착지점이정해지는단점을어떻게해결할까?

𝐿(𝑦, 𝑦ො)

𝒘
𝒘𝒂

Start

Start

𝒘𝒃
𝒘′
𝒘′′

Start

𝐿(𝑦, 𝑦ො)

𝒘
𝒘′
𝒘𝒕

Optimization

## --- [Page 11] ---
11/14

[Review] Descent Method (하강법)


하강법보다빠르게내려가려면어떻게할까?


초기위치에따라도착지점이정해지는단점을어떻게해결할까?

𝐿(𝑦, 𝑦ො)

𝒘
𝒘𝒂

Start

Start

𝒘𝒃
𝒘′
𝒘′′

Start

𝐿(𝑦, 𝑦ො)

𝒘
𝒘′
𝒘𝒕

Optimization

## --- [Page 12] ---
12/14

Gradient Descent (GD)

전체학습데이터를하나의Batch로묶어서Gradient 값을한번만계산하여가중치를갱신함

※ 일반적인Batch 단위가아닌전체데이터셋이라는점을유의할것


전체학습데이터에대해한번의업데이트가이루어지므로전체적인연산횟수가적음


전체학습데이터에대해Gradient를계산하므로, 수렴이안정적으로진행


한Step에모든학습데이터를사용하기때문에학습이오래걸림


Local optimal 상태가되면빠져나오기어려움


모델파라미터의업데이터가이루어지기전까지모든학습데이터에대해

저장하므로많은메모리가필요


Pros.


Cons.

Optimization

## --- [Page 13] ---
13/14

Mini-batch Gradient Descent (Mini-batch GD)

전체학습데이터를여러개의Batch 단위로나누어서Gradient 값을계산하여가중치를갱신함

GD
Mini-batch GD

Mini-batch

Mini-batch

Mini-batch

Mini-batch

Mini-batch

Mini-batch


GD보다local optimal에빠질위험이적음


병렬처리에유리


전체학습데이터가아닌일부분만사용하므로메모리에부하가낮음


적절한크기의Batch size를설정하여야함


Pros.


Cons.

Optimization


| Training Data |  |
| --- | --- |


| Training Data |  |
| --- | --- |


| Training Data |  |
| --- | --- |


| Training Data |  |
| --- | --- |


| Training Data |  |
| --- | --- |


| Training Data |  |
| --- | --- |


| Training Data |  |
| --- | --- |


## --- [Page 14] ---
14/14

Stochastic Gradient Descent (SGD)

학습데이터세트들중하나의데이터를Random하게선택하여Gradient를갱신함


하나의랜덤한데이터를고려하여가중치를갱신하므로메모리요구량이낮음


GD에비해학습속도가빠름


가중치의학습이불안정할수있음


GD에비해정확도가낮을수있음


Pros.


Cons.

Optimization

## --- [Page 15] ---
15/14

Momentum

기울기방향으로힘을받아물체가가속된다는물리법칙을적용

Momentum을적용하여학습방향이바로변하지않고, 일정한방향을유지하며움직임

: 갱신할가중치매개변수

: W에대한손실함수의기울기

* 𝜂: 학습률


기존에이동하는방향에대한관성을이용하여Local minimum을

빠져나올수있음


기존의변수들외에도과거에이동하는양을별도로저장하므로

메모리요구량이증가


Pros.


Cons.

Optimization

𝜕𝐿
𝜕𝑊

## --- [Page 16] ---
16/14

Adaptive Gradient (AdaGrad)

Ref) https://morioh.com/p/0831c4986e98

AdaGrad는개별매개변수에적응적으로Learning rate를조정하면서학습을진행

ℎ 는기존기울기값을제곱하여계속더함

매개변수를갱신할때1/ℎ를곱해학습률조정

동일한Learning rate을사용하여학습
Learning rate를서서히감소하며학습

: 갱신할가중치매개변수

: W에대한손실함수의기울기

* 𝜂: 학습률

ℎ: 기존기울기값


랜덤으로들어오는변수들에대해효율적으로학습시켜최적점을

빨리찾을수있음


학습을진행하며학습이많이된변수라면최적점가까이갔다고판단


학습이아직덜된변수라면학습을더빨리해야한다고판단


기존기울기값이계속누적되어값이커지게되며, 전체값이

작아지게되고학습을할수없게되는문제발생


Pros.


Cons.

Optimization

𝜕𝐿
𝜕𝑊

## --- [Page 17] ---
17/14

Root Mean Square Propagation (RMSprop)

AdaGrad를개선한기법으로갱신된기울기에weight를달리주는기법

𝜶값은사용자가별도로지정

과거의기울기를균일하게더하지않고새로운기울기정보를크게반영

: 갱신할가중치매개변수

: W에대한손실함수의기울기

* 𝜂: 학습률

ℎ: 기존기울기값

Optimization

𝜕𝐿
𝜕𝑊

## --- [Page 18] ---
18/14

Adaptive Moment Estimation (Adam)

Momentum + RMSProp

Momentum의지난gradient의지수감소평균을사용

물체가가속된다는물리법칙을적용

RMSProp의지난gradient의제곱지수감소를사용

갱신된기울기에weight를달리적용

: 갱신할가중치매개변수

: W에대한손실함수의기울기

* 𝜂: 학습률

ℎ: 기존기울기값

Momentum

RMSProp

Optimization

𝜕𝐿
𝜕𝑊

## --- [Page 19] ---
19/14

대표기법비교

Adam
Momentum
AdaGrad
SGD

Optimization

## --- [Page 20] ---
20/14

대표기법비교


GD


SGD


Momentum


NAG


Adagrad


RMSProp


Adam

Ref) https://truman.tistory.com/164

Optimization

## --- [Page 21] ---
21/76

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Division of Computer〮AI Engineering

Dong-A University, Busan, Rep. of Korea