## --- [Page 1] ---
1/18

Dong-A Univ. (ISPL)

컴퓨터AI공학부AI학과

2023년1학기인공지능

## --- [Page 2] ---
2/18

Learning Rate Control


Review: Gradient decent algorithm

X (W, b): Trainable parameters
Y: Loss function

Gradient = 0.8

new
prev

L
W
W
W





Learning rate

Gradient

Gradient decent algorithm

①현재지점에서미분을이용해gradient 계산

②Gradient에learning rate를곱하고

반대방향으로weight update

Learning rate = 0.1

## --- [Page 3] ---
3/18

Learning Rate Control


Learning rate control (decay)

•
빠른시간내에정확도높은학습파라미터를구하기위해learning rate를조절하는방법

{
, }
W b


( )
L 

{
, }
W b


( )
L 

{
, }
W b


( )
L 

Learning rate가큰경우
Learning rate가작은경우

## --- [Page 4] ---
4/18

Learning Rate Control


이전실습까지고정된learning rate를사용해학습수행(LR: 0.1)

5주차overfitting 실습자료중일부(Hyper-parameter 지정)

## --- [Page 5] ---
5/18

Learning Rate Control


Pytorch에서제공하는learning rate control 기능

•
StepLR: 지정된step (epoch) 단위마다learning rate 조절

•
ExponentialLR: 매step (epoch) 단위마다learning rate 조절

•
etc…

StepLR
(step_size: 10, gamma: 0.5)

ExponentialLR
(step_size: 10, gamma: 0.5)

Epoch

Learning rate

Epoch

Learning rate

## --- [Page 6] ---
6/18

Learning Rate Control


금일은Single Layer Perceptron (SLP)을이용해실습진행

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

## --- [Page 7] ---
7/18

Learning Rate Control


[실습1] Learning rate control을수행하지않는SLP 학습

1) 6주차LMS에업로드된기본소스코드다운로드

2) 구글드라이브마운트(파라미터및데이터셋경로확인필요)

3) 전체셀실행

## --- [Page 8] ---
8/18

Learning Rate Control


[실습1] Learning rate control을수행하지않는SLP 학습

1) 6주차LMS에업로드된기본소스코드다운로드

2) 구글드라이브마운트(파라미터및데이터셋경로확인필요)

3) 전체셀실행

4) 결과확인

## --- [Page 9] ---
9/18

Learning Rate Control


[실습2] StepLR 기법을이용한SLP 학습

•
[5] Hyper-parameter 및[6] Training loop 일부수정

StepLR 매5epoch 마다learning rate에0.5 곱하기

## --- [Page 10] ---
10/18

Learning Rate Control


[실습2] StepLR 기법을이용한SLP 학습

•
결과확인

고정된learning rate 사용시

Learning rate control 사용시

1.3% 성능향상

## --- [Page 11] ---
11/18

Learning Rate Control


[실습3] ExponentialLR 기법을이용한SLP 학습

•
[5] Hyper-parameter 일부수정

## --- [Page 12] ---
12/18

Learning Rate Control


[실습3] ExponentialLR 기법을이용한SLP 학습

•
결과확인

고정된learning rate 사용(lr: 0.1)

StepLR (step_size: 5, gamma: 0.5)

ExponentialLR (gamma: 0.8)

## --- [Page 13] ---
13/18

Optimizer


대표적인optimizer 기법비교

Adam
Momentum
AdaGrad
SGD

f(x,y): loss function으로가정

x, y: trainable parameter로가정

: Starting point

: Optimal point

## --- [Page 14] ---
14/18

1

1

t
t
t

t
t

t

W
W
v

L
v
v
W















Optimizer


[실습1] Momentum 기법을이용한SLP 학습

•
[5] Hyper-parameter 및[6] Training loop 부분변경

momentum 계수

## --- [Page 15] ---
15/18

Optimizer


[실습1] Momentum 기법을이용한SLP 학습

•
[5] Hyper-parameter 및[6] Training loop 부분변경

## --- [Page 16] ---
16/18

Optimizer


[실습1] Momentum 기법을이용한SLP 학습

•
결과확인

SGD

SGD + Momentum

## --- [Page 17] ---
17/18

Optimizer


[실습2] Adam 기법을이용한SLP 학습

•
[5] Hyper-parameter 변경

## --- [Page 18] ---
18/18

Optimizer


[실습2] Adam 기법을이용한SLP 학습

•
결과확인

SGD

SGD + Momentum

Adam

## --- [Page 19] ---
19/18

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Division of Computer〮AI Engineering

Dong-A University, Busan, Rep. of Korea