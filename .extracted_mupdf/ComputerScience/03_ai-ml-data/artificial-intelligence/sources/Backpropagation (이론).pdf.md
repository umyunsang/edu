## --- [Page 1] ---
1/85

Dong-A Univ. (ISPL)

컴퓨터AI공학부

2024년1학기인공지능

## --- [Page 2] ---
2/85

Backpropagation

Review: MLP by Linear Algebra

SLP MLP (One of the Deep Neural Network: DNN)

행렬에대한Forward propagation

𝒙∈𝑹𝒎q ∈𝑹𝒏

𝑾∈𝑹𝒎×𝒏=

𝑤ଵଵ
⋯
𝑤ଵ௡
⋮
⋱
⋮
𝑤௠ଵ
⋯
𝑤௠௡

,
𝑾𝑻∈𝑹𝒏×𝒎=

𝑤ଵଵ
⋯
𝑤௠ଵ
⋮
⋱
⋮
𝑤ଵ௡
⋯
𝑤௠௡

,
𝒙∈𝑹𝒎=

𝑥ଵ

⋮
𝑥௠

,

𝒒= 𝝈(𝑾𝑻ȉ 𝒙+ 𝒃) = 𝝈

𝑊ଵ,ଵ𝑥ଵ+ ⋯+ 𝑊௠,ଵ𝑥௠

⋮
𝑊ଵ,௡𝑥ଵ+ ⋯+ 𝑊௠,௡𝑥௠

+

𝑏ଵ

⋮
𝑏௡

=

𝑞1

⋮
𝑞𝑛

𝒃∈𝑹𝒏=

𝑏ଵ

⋮
𝑏௡

∈𝑹𝒏

## --- [Page 3] ---
3/85

Backpropagation

Introduction

Backpropagation(역전파)

모험가이야기: 어떤모험가는여행을하며가장깊고낮은골짜기를찾아가려한다.


조건1: 지도를보지않을것


조건2: 눈가리개를쓰는것


어떻게모험가는가장깊고낮을골짜기를찾아갈수있을까?

## --- [Page 4] ---
4/85

GD Method (역전파를위한최적화: Optimization for Backpropagation)

하강법에서접선의기울기와𝒘 는항상다른부호를가짐

∆𝒘 를통해방향을결정

Gradient Descent Method (GD Method / 경사하강법)

## --- [Page 5] ---
5/85

Contents

1.
Introduction

2.
Concept of Backpropagation

: Gradient Descent(GD) Method (경사하강법)

## --- [Page 6] ---
6/85

Backpropagation

Introduction

Backpropagation: 계산결과와정답의오차를구해서오차에관여하는노드값들의가중치와편향을수정하는데,

오차가작아지는방향으로반복해서수정하는기법

Input 𝑋
Model

𝑓(w)
Output 𝑦ො

Loss 𝐿(𝑦, 𝑦ො)

Update 𝑊

min 𝐿(𝑦, 𝑦ො)

𝑤 = (𝑤ଵ, 𝑤ଶ, … , 𝑤௡, 𝑏ଵ,𝑏ଶ, …,𝑏௠)
𝑤: 가중치
𝑏: 편향

Loss를최소화

b

𝑥ଵ
𝑠ଵ

<Multi-Layer Perceptron>

𝑥ଶ

b

𝑠ଶ

out

𝒃𝟐𝟏

𝒘𝟐𝟏

𝒘𝟐𝟐

𝒃𝟏𝟏

𝒃𝟏𝟐

𝒘𝟏𝟏

𝒘𝟏𝟐

𝒘𝟏𝟑

𝒘𝟏𝟒

Input layer
Hidden layer
Output layer

Error back 
propagation

## --- [Page 7] ---
7/85

Backpropagation

Introduction

b

𝑥ଵ
𝑠ଵ

<Multi-Layer Perceptron>

𝑥ଶ

b

𝑠ଶ

out

𝒃𝟐𝟏

𝒘𝟐𝟏

𝒘𝟐𝟐

𝒃𝟏𝟏

𝒃𝟏𝟐

𝒘𝟏𝟏

𝒘𝟏𝟐

𝒘𝟏𝟑

𝒘𝟏𝟒

Input layer
Hidden layer
Output layer

Error back 
propagation

Parameter (w, b) 업데이트: per Mini-batch or Epoch (Iteration)

Iteration 수가늘어나면정확성을증가됨시간多

Iteration 수가줄어들면정확성을떨어짐시간小

Backpropagation: 계산결과와정답의오차를구해서오차에관여하는노드값들의가중치와편향을수정하는데,

오차가작아지는방향으로반복해서수정하는기법

## --- [Page 8] ---
8/85

Backpropagation

Introduction

모험가이야기: 어떤모험가는여행을하며가장깊고낮은골짜기를찾아가려한다.


조건1: 지도를보지않을것


조건2: 눈가리개를쓰는것


어떻게모험가는가장깊고낮을골짜기를찾아갈수있을까?

- 특정포인트w에서w가커질수록함수값이커지는중이라면(즉, 기울기의부호는양수) 음의방향으로w를이동
- 반대로특정포인트w에서w가커질수록함수값이작아지는중이라면(즉, 기울기의부호는음수) 양의방향으로w를이동
- 이동거리Gradient 크기를이용

## --- [Page 9] ---
9/85

Backpropagation

Introduction

모험가이야기: 어떤모험가는여행을하며가장깊고낮은골짜기를찾아가려한다.


조건1: 지도를보지않을것


조건2: 눈가리개를쓰는것


어떻게모험가는가장깊고낮을골짜기를찾아갈수있을까?

???

1. 전진/후진

2. 보폭

두가지선택이필요

1. 방향설정

2. 움직임

## --- [Page 10] ---
10/85

Backpropagation

Introduction

모험가이야기: 어떤모험가는여행을하며가장깊고낮은골짜기를찾아가려한다.


조건1: 지도를보지않을것


조건2: 눈가리개를쓰는것


어떻게모험가는가장깊고낮을골짜기를찾아갈수있을까?

기울기를활용

## --- [Page 11] ---
11/85

Backpropagation

Introduction

모험가이야기: 어떤모험가는여행을하며가장깊고낮은골짜기를찾아가려한다.


조건1: 지도를보지않을것


조건2: 눈가리개를쓰는것


어떻게모험가는가장깊고낮을골짜기를찾아갈수있을까? Optimization (최적화)

기울기를활용

## --- [Page 12] ---
12/85

이동거리에사용할값을gradient의크기와비례하는factor를이용하면현재xx의값이극소값에서
멀때는많이이동하고, 극소값에가까워졌을때는조금씩이동할수있게된다.

## --- [Page 13] ---
13/85

Contents

1.
Introduction

2.
Derivative Review (미분/편미분/체인룰)

3.
Concept of Backpropagation

: Gradient Descent(GD) Method (경사하강법)

## --- [Page 14] ---
14/85

Backpropagation

Descent Method(하강법)

주어진어떤지점에서부터오차가더작은곳으로이동하려는방법

Start

𝐿(𝑦, 𝑦ො)

𝒘

𝒘𝒔𝒖𝒃ି𝒐𝒑𝒕𝒊𝒎𝒂𝒍

현재위치한곳에서낮은곳으로이동하려면?

𝒘𝒕

𝛁𝑳(𝒘𝒕): 접선의기울기

## --- [Page 15] ---
15/85

Backpropagation

Descent Method(하강법)

주어진어떤지점에서부터오차가더작은곳으로이동하려는방법

𝐿(𝑦, 𝑦ො)

𝒘

𝒘𝒕+ ∆𝒘𝒕→𝒘𝒕ା𝟏

∆𝒘𝒕

1. 일정부분이동

2. 이전위치의𝒘𝒕 와변화된위치의 ∆𝒘𝒕를더하면

현재위치의𝒘𝒕ା𝟏 생성

𝒘𝒕

Start

𝒘𝒕ା𝟏

## --- [Page 16] ---
16/85

Backpropagation

Descent Method(하강법)

주어진어떤지점에서부터오차가더작은곳으로이동하려는방법

𝐿(𝑦, 𝑦ො)

𝒘

𝒘𝒕+ ∆𝒘𝒕→𝒘𝒕ା𝟏

∆𝒘𝒕ା𝟏

𝒘𝒕ା𝟏+ ∆𝒘𝒕ା𝟏→𝒘𝒕ା𝟐

1. 낮은곳으로조금씩이동

2.
𝒘 는계속변경

3. 최소지점까지이동

𝒘𝒏ି𝟏+ ∆𝒘𝒏ି𝟏→𝒘𝒏

⋮

𝒘𝒕ା𝟏
𝒘𝒕
𝒘𝒕ା𝟐

Start

∆𝒘𝒕

## --- [Page 17] ---
17/85

Backpropagation

Descent Method(하강법)

주어진어떤지점에서부터오차가더작은곳으로이동하려는방법

𝐿(𝑦, 𝑦ො)

𝒘

𝛁𝑳𝒘𝒕∆𝒘< 𝟎

1. 하강법에서접선의기울기와𝒘 는항상다른부호를가짐∆𝒘 를통해방향을결정

𝛁𝑳(𝒘𝒕): 접선의기울기

𝒘𝒕ା𝟏
𝒘𝒕

Start

∆𝒘𝒕

## --- [Page 18] ---
18/85

Backpropagation

Descent Method(하강법)

주어진어떤지점에서부터오차가더작은곳으로이동하려는방법

𝐿(𝑦, 𝑦ො)

𝒘

Start

1. 하강법에서접선의기울기와𝒘 는항상다른부호를가짐∆𝒘 를통해방향을결정

2. 하강법은초기위치에따라도착지점이변경Weight Initialization

Start

𝒘𝒏
𝒘𝒐𝒑𝒕𝒊𝒎𝒂𝒍
𝒘𝒕

𝛁𝑳(𝒘𝒕): 접선의기울기

𝒘𝒔𝒖𝒃ି𝒐𝒑𝒕𝒊𝒎𝒂𝒍

## --- [Page 19] ---
19/85

Backpropagation

Backpropagation 동작원리

operation

𝒛

𝒙

𝒚

Chain rule

## --- [Page 20] ---
20/85

Backpropagation

Backpropagation 동작원리

operation

𝒛

𝒙

𝒚

𝝏𝒛
𝝏𝒙

𝝏𝒛
𝝏𝒚

Local gradient

Chain rule

## --- [Page 21] ---
21/85

Backpropagation

Backpropagation 동작원리

operation

𝒛

𝒙

𝒚

𝝏𝒛
𝝏𝒙

𝝏𝒛
𝝏𝒚

Local gradient

𝝏𝑳
𝝏𝒛

Global gradient

Chain rule

## --- [Page 22] ---
22/85

Backpropagation

Backpropagation 동작원리

operation

𝒛

𝒙

𝒚

𝝏𝒛
𝝏𝒙

𝝏𝒛
𝝏𝒚

Local gradient

𝝏𝑳
𝝏𝒛

Global gradient

Chain rule

## --- [Page 23] ---
23/85

Backpropagation

Backpropagation 동작원리

operation

𝒛

𝒙

𝒚

𝝏𝒛
𝝏𝒙

𝝏𝒛
𝝏𝒚

Local gradient

𝝏𝑳
𝝏𝒛

Chain rule

Global gradient

## --- [Page 24] ---
24/85

Backpropagation

Backpropagation 동작원리

operation

𝒛

𝒙

𝒚

𝝏𝒛
𝝏𝒙

𝝏𝒛
𝝏𝒚

Local gradient

𝝏𝑳
𝝏𝒛

Global gradient

## --- [Page 25] ---
25/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 26] ---
26/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 27] ---
27/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 28] ---
28/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 29] ---
29/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 30] ---
30/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 31] ---
31/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 32] ---
32/85

Backpropagation 동작원리

Forward propagation (pass)

## --- [Page 33] ---
33/85

Backpropagation 동작원리

## --- [Page 34] ---
34/85

Backpropagation 동작원리

## --- [Page 35] ---
35/85

Backpropagation 동작원리

## --- [Page 36] ---
36/85

Backpropagation 동작원리

## --- [Page 37] ---
37/85

Backpropagation 동작원리

## --- [Page 38] ---
38/85

Backpropagation 동작원리

## --- [Page 39] ---
39/85

Backpropagation 동작원리

## --- [Page 40] ---
40/85

Backpropagation

Backpropagation 동작원리

예제1: 슈퍼에서사과를2개, 귤을3개구매시지불금액은?

사과는1개100원, 귤은1개150원

소비세10%

x

x

100

150

+

사과의개수
2

3

x

소비세
1.1

사과의단가

귤의단가

귤의개수

𝒘𝟏

𝒉

𝒙𝟏

𝒙𝟐

𝒘𝟐

## --- [Page 41] ---
41/85

Backpropagation

Backpropagation 동작원리

예제1: 슈퍼에서사과를2개, 귤을3개구매시지불금액은?

사과는1개100원, 귤은1개150원

소비세10%

(
)
T
y
h W X
b



…



1x

2x

3x

10
x

1
w

2
w

3
w

10
w

1

b

y

x

x

100

150

+

사과의개수
2

귤의개수
3

x

소비세
1.1

( )
h 

𝒘𝟏

𝒉

𝒙𝟏

𝒙𝟐

𝒘𝟐

## --- [Page 42] ---
42/85

Backpropagation

Backpropagation 동작원리

예제1: 슈퍼에서사과를2개, 귤을3개구매시지불금액은?

사과는1개100원, 귤은1개150원

소비세10%

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Forward propagation
사과의개수

소비세

사과의단가

귤의단가

귤의개수

𝒘𝟏

𝒉

𝒙𝟏

𝒙𝟐

𝒘𝟐

## --- [Page 43] ---
43/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과의개수

소비세

사과의단가

귤의단가

귤의개수

𝒘𝟏

𝒉

𝒙𝟏

𝒙𝟐

𝒘𝟐

## --- [Page 44] ---
44/85

Backpropagation

Backpropagation 동작원리

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝑳

사과의개수

소비세

사과의단가

귤의단가

귤의개수

𝒘𝟏

𝒉

𝒙𝟏

𝒙𝟐

𝒘𝟐

## --- [Page 45] ---
45/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

사과의개수
2

3

x

200

450

650

소비세
1.1

715

Backward propagation

귤의개수

𝒘𝟏

𝒅

𝒉

𝒆

𝒇
𝑳

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

사과의단가

𝒙𝟏

귤의단가

𝒙𝟐

𝒘𝟐

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

## --- [Page 46] ---
46/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

𝝏𝑳
𝝏𝑳= 𝟏

1

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

## --- [Page 47] ---
47/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

𝝏𝑳
𝝏𝒉= 𝒇

1

650

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가
𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

## --- [Page 48] ---
48/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

𝝏𝑳
𝝏𝒇= 𝒉

1

650

1.1

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

## --- [Page 49] ---
49/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

𝝏𝑳
𝝏𝒆= 𝝏𝑳

𝝏𝒇

𝝏𝒇
𝝏𝒆

1

650

1.1

1.1

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

## --- [Page 50] ---
50/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

𝝏𝑳
𝝏𝒅= 𝝏𝑳

𝝏𝒇

𝝏𝒇
𝝏𝒅

1

650

1.1

1.1

1.1

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

## --- [Page 51] ---
51/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

1
1.1

1.1

1.1
2.2

3.3

110

165

650

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

## --- [Page 52] ---
52/85

Backpropagation

Backpropagation 동작원리

x

x

100

150

+

2

3

x

200

450

650

1.1

715

Backward propagation

𝒅

𝒆

𝒇
𝑳

1
1.1

1.1

1.1
2.2

3.3

110

165

650

∴𝝏𝑳

𝝏𝒘𝟏 = 𝝏𝑳

𝝏𝒇 𝝏𝒇

𝝏𝒅 𝝏𝒅

𝝏𝒘𝟏 = 𝟏𝟏𝟎

Chain rule

예제2: 사과개수가변하면최종금액에어떤영향을끼칠까?

사과개수: 𝑤ଵ

지불금액: 𝐿

𝝏𝑳
𝝏𝒘𝟏

사과개수가증가했을때지불금액이얼마나증가하는지표시

𝒅= 𝒘𝟏𝒙𝟏

𝝏𝒅

𝝏𝒘𝟏= 𝒙𝟏,

𝝏𝒅

𝝏𝒙𝟏= 𝒘𝟏

𝑳(𝒘𝟏, 𝒘𝟐, 𝒙𝟏, 𝒙𝟐) = 𝒄(𝒘𝟏𝒙+ 𝒘𝟐𝒙𝟐)

𝒆= 𝒘𝟐𝒙𝟐

𝝏𝒆

𝝏𝒘𝟐= 𝒙𝟐,

𝝏𝒆

𝝏𝒙𝟐= 𝒘𝟐

𝒇= 𝒅+ 𝒆

𝝏𝒇

𝝏𝒅= 𝟏,

𝝏𝒇

𝝏𝒆= 𝟏

𝑳= 𝒉𝒇

𝝏𝑳

𝝏𝒉= 𝒇,

𝝏𝑳

𝝏𝒇= 𝒉

사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

## --- [Page 53] ---
53/85

Backpropagation

Backpropagation 동작원리

Weight Update (a: 사과의개수)

x

x

100

150

+

3

3

x

300

450

750

1.1

715

825

Forward propagation
사과의개수

소비세

귤의개수

𝒘𝟏

𝒉

사과의단가

𝒙𝟏

𝒙𝟐

𝒘𝟐

귤의단가

∴𝝏𝑳

𝝏𝒘𝟏 = 𝝏𝑳

𝝏𝒇 𝝏𝒇

𝝏𝒅 𝝏𝒅

𝝏𝒘𝟏 = 𝟏𝟏𝟎

Chain rule

## --- [Page 54] ---
54/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

## --- [Page 55] ---
55/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐
-2.00

0.73

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

2.00

<Sigmoid>
𝒂

𝒄

𝒃

𝒅
𝒆
𝑳

Sigmoid

## --- [Page 56] ---
56/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73

2.00

𝒂

𝒄

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 57] ---
57/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73

Sigmoid gate

2.00

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝒆

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 58] ---
58/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73

2.00

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

## --- [Page 59] ---
59/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
1.00

2.00

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝑳

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 60] ---
60/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
1.00

−𝟏
𝟏. 𝟑𝟕𝟐
= −𝟎. 𝟓𝟑

2.00

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒉

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

## --- [Page 61] ---
61/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53

−𝟎. 𝟓𝟑
𝟏= −𝟎. 𝟓𝟑

2.00

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒈= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

## --- [Page 62] ---
62/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53

−𝟎. 𝟓𝟑(𝟏) 𝒆ି𝟏= −𝟎. 𝟐𝟎

-0.20

2.00

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒇= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝝏𝒈

𝝏𝒇

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

## --- [Page 63] ---
63/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53
0.20

2.00

-0.20

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒆= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝝏𝒈

𝝏𝒇

𝝏𝒇
𝝏𝒆

−𝟎. 𝟓𝟑(𝟏) 𝒆ି𝟏(−𝟏) = 𝟎. 𝟐𝟎

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

## --- [Page 64] ---
64/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53
0.20

0.20

2.00

-0.20
0.20

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒅= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝝏𝒈

𝝏𝒇

𝝏𝒇
𝝏𝒆

𝝏𝒆
𝝏𝒅

𝝏𝑳
𝝏𝒃= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝝏𝒈

𝝏𝒇

𝝏𝒇
𝝏𝒆

𝝏𝒆
𝝏𝒃

−𝟎. 𝟓𝟑(𝟏) 𝒆ି𝟏(−𝟏)(𝟏) = 𝟎. 𝟐𝟎

−𝟎. 𝟓𝟑(𝟏) 𝒆ି𝟏(−𝟏)(𝟏) = 𝟎. 𝟐𝟎

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 65] ---
65/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53
0.20

0.20

0.20

0.20

2.00

-0.20
0.20

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒂= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝝏𝒈

𝝏𝒇

𝝏𝒇
𝝏𝒆

𝝏𝒆
𝝏𝒅

𝝏𝒅
𝝏𝒂

−𝟎. 𝟓𝟑(𝟏) 𝒆ି𝟏(−𝟏)(𝟏)(𝟏) = 𝟎. 𝟐𝟎

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 66] ---
66/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+
+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53
0.20

0.20

0.20

0.20

0.40

2.00
-0.20

-0.20
0.20

𝒂

𝒅
𝒆
𝒇
𝒈
𝒉
𝑳

𝝏𝑳
𝝏𝒘𝟏

= 𝝏𝑳

𝝏𝒉

𝝏𝒉
𝝏𝒈

𝝏𝒈

𝝏𝒇

𝝏𝒇
𝝏𝒆

𝝏𝒆
𝝏𝒅

𝝏𝒅
𝝏𝒂

𝝏𝒂
𝝏𝒘𝟏

−𝟎. 𝟓𝟑
𝟏
𝒆ି𝟏
−𝟏
𝟏
𝟏
−𝟏= −𝟎. 𝟐𝟎

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒄

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 67] ---
67/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+

2.00

+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53
0.20

0.20

0.20

0.20

-0.20

0.40

-0.40

-0.60

-0.20
0.20

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 68] ---
68/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation

x

x

-1.00

-3.00

+

2.00

+

-2.00

6.00

4.00

-3.00

1.00

-2.00

x (-1)
-1.00
exp
0.37
+1
1.37
𝟏
𝒙

0.73
-0.53
-0.53
0.20

0.20

0.20

0.20

-0.20

0.40

-0.40

-0.60

-0.20
0.20

𝒘𝒕ା𝟏←𝒘𝒕−𝝁𝛁𝒇(𝒘𝒕)

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

1.00

=
𝟏
𝟏+ 𝒆ି(𝒘𝟎𝒙𝟎ା𝒘𝟏𝒙𝟏ା𝒃)
𝝈𝒙=
𝟏
𝟏+ 𝒆ି𝒙

<Sigmoid>

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 69] ---
69/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation (Ex., Learning Rate = 0.5)

x

x

-1.00

-3.00 -2.80

+

2.00 2.10

+

-3.00 -3.10

-2.00

x (-1)
exp
+1
𝟏
𝒙

0.20

-0.20

-0.40

𝒘𝒕ା𝟏←𝒘𝒕−𝝁𝛁𝒇(𝒘𝒕)

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏
𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 70] ---
70/85

Backpropagation

Backpropagation 동작원리

Sigmoid 계층의backpropagation (Ex., Learning Rate = 0.5)

x

x

-1.00

-2.80

+

2.10

+

-2.10

5.60

3.50

-3.10

0.40

-2.00

x (-1)
-0.40
exp
0.67
+1
1.67
𝟏
𝒙

0.60

𝒇𝒙= 𝒆𝒙
𝝏𝒇
𝝏𝒙= 𝒆𝒙
𝒇𝒙=

𝟏
𝒙

𝝏𝒇
𝝏𝒙= −

𝟏
𝒙𝟐

𝒇𝒙= 𝒙+ 𝒄

𝝏𝒇
𝝏𝒙= 𝟏

0.73

𝒇𝒙= 𝜶𝒙

𝝏𝒇
𝝏𝒙= 𝜶

𝒘𝟏

𝒙𝟏

𝒙𝟐

𝒘𝟐

𝒃

## --- [Page 71] ---
71/85

Backpropagation

Backpropagation 동작원리

Single Layer Perceptron (SLP)

Forward propagation (pass)

## --- [Page 72] ---
72/85

Backpropagation

Backpropagation 동작원리

SLP Deep Neural Network (MLP)

행렬에대한Backpropagation

𝒙∈𝑹𝒎q ∈𝑹𝒏

𝑾∈𝑹𝒎×𝒏=

𝑤ଵଵ
⋯
𝑤ଵ௡
⋮
⋱
⋮
𝑤௠ଵ
⋯
𝑤௠௡

,
𝑾𝑻∈𝑹𝒏×𝒎=

𝑤ଵଵ
⋯
𝑤௠ଵ
⋮
⋱
⋮
𝑤ଵ௡
⋯
𝑤௠௡

,
𝒙∈𝑹𝒎=

𝑥ଵ

⋮
𝑥௠

,

𝒒= 𝝈(𝑾𝑻ȉ 𝒙+ 𝒃) = 𝝈

𝑊ଵ,ଵ𝑥ଵ+ ⋯+ 𝑊௠,ଵ𝑥௠

⋮
𝑊ଵ,௡𝑥ଵ+ ⋯+ 𝑊௠,௡𝑥௠

+

𝑏ଵ

⋮
𝑏௡

=

𝑞1

⋮
𝑞𝑛

𝒃∈𝑹𝒏=

𝑏ଵ

⋮
𝑏௡

∈𝑹𝒏

## --- [Page 73] ---
73/85

Backpropagation

Backpropagation 동작원리

SLP Deep Neural Network (MLP)

행렬에대한Backpropagation

𝒙∈𝑹𝒎q ∈𝑹𝒏

𝑾∈𝑹𝒎×𝒏=

𝑤ଵଵ
⋯
𝑤ଵ௡
⋮
⋱
⋮
𝑤௠ଵ
⋯
𝑤௠௡

,
𝑾𝟐𝑻∈𝑹𝒏×𝒎=

𝑤ଵଵ
⋯
𝑤௠ଵ
⋮
⋱
⋮
𝑤ଵ௡
⋯
𝑤௠௡

,
𝒙∈𝑹𝒎=

𝑥ଵ

⋮
𝑥௠

,

𝒚= 𝝈(𝑾𝑻ȉ 𝒙+ 𝒃) = 𝝈

𝑊ଵ,ଵ𝑥ଵ+ ⋯+ 𝑊௠,ଵ𝑥௠

⋮
𝑊ଵ,௡𝑥ଵ+ ⋯+ 𝑊௠,௡𝑥௠

+

𝑏ଵ

⋮
𝑏௡

=

𝑦1

⋮
𝑦𝑛

𝒃∈𝑹𝒏=

𝑏ଵ

⋮
𝑏௡

## --- [Page 74] ---
74/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

𝒒= 𝝈(𝑾𝑻ȉ 𝒙+ 𝒃) = 𝝈

𝑊ଵ,ଵ𝑥ଵ+ ⋯+ 𝑊௠,ଵ𝑥௠

⋮
𝑊ଵ,௡𝑥ଵ+ ⋯+ 𝑊௠,௡𝑥௠

+

𝑏ଵ

⋮
𝑏௡

=

𝑞1

⋮
𝑞𝑛

𝒒= 𝑾𝑻ȉ 𝒙=

𝑊ଵ,ଵ𝑥ଵ+ ⋯+ 𝑊௠,ଵ𝑥௠

⋮
𝑊ଵ,௡𝑥ଵ+ ⋯+ 𝑊௠,௡𝑥௠

=

𝑞1

⋮
𝑞𝑛

간소화

## --- [Page 75] ---
75/85

Backpropagation

Backpropagation 동작원리

L2

𝑾𝑻
0.1
0.5
−0.3
0.8

0.22
0.26

0.2
0.4

0.116

(𝟎. 𝟐𝟐)𝟐+(𝟎. 𝟐𝟔)𝟐 = 𝟎. 𝟏𝟏𝟔

𝒙

𝒇

행렬의Backpropagation (Example)

행렬곱

(𝑾𝑻𝒙)

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝒒= 𝑾𝑻𝒙

## --- [Page 76] ---
76/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝑾𝑻
0.1
0.5
−0.3
0.8

𝟎. 𝟐𝟐
𝟎. 𝟐𝟔

0.2
0.4

0.116

1.00
𝟎. 𝟒𝟒
𝟎. 𝟓𝟐

𝒙

𝒇

행렬곱

(𝑾𝑻𝒙)

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊

𝛁𝒒𝒊𝒇= 𝟐𝒒𝒊

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝒒= 𝑾𝑻𝒙

## --- [Page 77] ---
77/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝑾𝑻
𝟎. 𝟏
𝟎. 𝟓
−𝟎. 𝟑
𝟎. 𝟖

0.22
0.26
0.116

1.00

𝒙

𝒇

행렬곱

(𝑾𝑻𝒙)

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝝏𝒇
𝝏𝑾𝑻= 𝝏𝒇

𝝏𝒒ȉ 𝒙𝑻

𝜕𝑓
𝜕𝑥= 𝑊ȉ 𝜕𝑓

𝜕𝑞

0.2
0.4

𝒒= 𝑾𝑻𝒙

𝝏𝒇
𝝏𝑾𝑻= 𝜕𝑓

𝜕𝑞

𝜕𝑞
𝜕𝑊்
= 2𝑞𝒙𝑻

𝑛× 𝑚= 𝑛× 1 ȉ 𝑚× 1்

= 𝑛× 1 ȉ 1 × 𝑚

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊= 𝟎. 𝟒𝟒

𝟎. 𝟓𝟐

## --- [Page 78] ---
78/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝑾𝑻
𝟎. 𝟏
𝟎. 𝟓
−𝟎. 𝟑
𝟎. 𝟖

0.22
0.26
0.116

1.00

𝒙

𝒇

행렬곱

(𝑾𝑻𝒙)

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝝏𝒇
𝝏𝑾𝑻= 𝝏𝒇

𝝏𝒒ȉ 𝒙𝑻

𝜕𝑓
𝜕𝑥= 𝑊ȉ 𝜕𝑓

𝜕𝑞

0.2
0.4

𝒒= 𝑾𝑻𝒙

𝝏𝒇
𝝏𝑾𝑻= 𝜕𝑓

𝜕𝑞

𝜕𝑞
𝜕𝑊்
= 2𝑞𝒙𝑻

𝑛× 𝑚= 𝑛× 1 ȉ 𝑚× 1்

= 𝑛× 1 ȉ 1 × 𝑚

= 0.44

0.52
0.2
0.4 = 𝟎. 𝟎𝟖𝟖
𝟎. 𝟏𝟕𝟔
𝟎. 𝟏𝟎𝟒
𝟎. 𝟐𝟎𝟖

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊= 𝟎. 𝟒𝟒

𝟎. 𝟓𝟐

## --- [Page 79] ---
79/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝑾𝑻
𝟎. 𝟏
𝟎. 𝟓
−𝟎. 𝟑
𝟎. 𝟖

0.22
0.26
0.116

1.00

𝒙

𝒇

행렬곱

(𝑾𝑻𝒙)

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝝏𝒇
𝝏𝑾𝑻= 𝝏𝒇

𝝏𝒒ȉ 𝒙𝑻

𝜕𝑓
𝜕𝑥= 𝑊ȉ 𝜕𝑓

𝜕𝑞

0.2
0.4

𝒒= 𝑾𝑻𝒙

𝝏𝒇
𝝏𝑾𝑻= 𝜕𝑓

𝜕𝑞

𝜕𝑞
𝜕𝑊்
= 2𝑞𝒙𝑻= 0.44

0.52
0.2
0.4 = 𝟎. 𝟎𝟖𝟖
𝟎. 𝟏𝟕𝟔
𝟎. 𝟏𝟎𝟒
𝟎. 𝟐𝟎𝟖

𝜵𝑾𝑻𝒇= 𝜵𝒒𝒇ȉ 𝜵𝑾𝑻𝒒

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊= 𝟎. 𝟒𝟒

𝟎. 𝟓𝟐

## --- [Page 80] ---
80/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝒙

𝑾𝑻
0.1
0.5
−0.3
0.8

0.22
0.26

𝟎. 𝟐
𝟎. 𝟒

0.116

1.00

𝟎. 𝟎𝟖𝟖
𝟎. 𝟏𝟕𝟔
𝟎. 𝟏𝟎𝟒
𝟎. 𝟐𝟎𝟖

𝒇

행렬곱

(𝑾𝑻𝒙)

𝜕𝑓
𝜕𝑊்
= 𝜕𝑓

𝜕𝑞ȉ 𝑥்

𝝏𝒇
𝝏𝒙= 𝑾ȉ 𝝏𝒇

𝝏𝒒

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝝏𝒇
𝝏𝒙= 𝜕𝑓

𝜕𝑞

𝜕𝑞
𝜕𝑥= 𝑾𝑻𝑻ȉ 2𝑞

𝑚× 1 = 𝑛× 𝑚்
ȉ 𝑛× 1

= 𝑚× 𝑛ȉ 𝑛× 1

𝒒= 𝑾𝑻𝒙

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊= 𝟎. 𝟒𝟒

𝟎. 𝟓𝟐

## --- [Page 81] ---
81/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝒙

𝑾𝑻
0.1
0.5
−0.3
0.8

0.22
0.26

𝟎. 𝟐
𝟎. 𝟒

0.116

1.00

𝟎. 𝟎𝟖𝟖
𝟎. 𝟏𝟕𝟔
𝟎. 𝟏𝟎𝟒
𝟎. 𝟐𝟎𝟖

𝒇

행렬곱

(𝑾𝑻𝒙)

𝜕𝑓
𝜕𝑊்
= 𝜕𝑓

𝜕𝑞ȉ 𝑥்

𝝏𝒇
𝝏𝒙= 𝑾ȉ 𝝏𝒇

𝝏𝒒

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝝏𝒇
𝝏𝒙= 𝜕𝑓

𝜕𝑞

𝜕𝑞
𝜕𝑥= 𝑾ȉ 2𝑞

𝑚× 1 = 𝑛× 𝑚்
ȉ 𝑛× 1

= 𝑚× 𝑛ȉ 𝑛× 1

= 0.1
−0.3
0.5
0.8
0.44
0.52 = −𝟎. 𝟏𝟏𝟐
𝟎. 𝟔𝟑𝟔

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊= 𝟎. 𝟒𝟒

𝟎. 𝟓𝟐

𝒒= 𝑾𝑻𝒙

## --- [Page 82] ---
82/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝒙

𝑾𝑻
0.1
0.5
−0.3
0.8

0.22
0.26

𝟎. 𝟐
𝟎. 𝟒

0.116

1.00

𝟎. 𝟎𝟖𝟖
𝟎. 𝟏𝟕𝟔
𝟎. 𝟏𝟎𝟒
𝟎. 𝟐𝟎𝟖

𝒇

행렬곱

(𝑾𝑻𝒙)

𝜕𝑓
𝜕𝑊்
= 𝜕𝑓

𝜕𝑞ȉ 𝑥்

𝝏𝒇
𝝏𝒙= 𝑾ȉ 𝝏𝒇

𝝏𝒒

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

𝝏𝒇
𝝏𝒙= 𝜕𝑓

𝜕𝑞

𝜕𝑞
𝜕𝑥= 𝑾ȉ 2𝑞
= 0.1
−0.3
0.5
0.8
0.44
0.52 = −𝟎. 𝟏𝟏𝟐
𝟎. 𝟔𝟑𝟔

𝜵𝒙𝒇= 𝜵𝒒𝒇ȉ 𝜵𝒙𝒒

𝝏𝒇
𝝏𝒒𝒊

= 𝟐𝒒𝒊= 𝟎. 𝟒𝟒

𝟎. 𝟓𝟐

𝒒= 𝑾𝑻𝒙

## --- [Page 83] ---
83/85

Backpropagation

Backpropagation 동작원리

행렬의Backpropagation (Example)

L2

𝒙

𝑾𝑻
0.1
0.5
−0.3
0.8

0.22
0.26

0.2
0.4

0.116

1.00
𝟎. 𝟒𝟒
𝟎. 𝟓𝟐

𝟎. 𝟎𝟖𝟖
𝟎. 𝟏𝟕𝟔
𝟎. 𝟏𝟎𝟒
𝟎. 𝟐𝟎𝟖

𝒇

행렬곱

(𝑾𝑻𝒙)

𝒒

𝒒=

𝑞1

⋮
𝑞𝑛

∈𝑹𝒏

−𝟎. 𝟏𝟏𝟐

𝟎. 𝟔𝟑𝟔

𝒘𝒕ା𝟏←𝒘𝒕−𝝁𝛁𝒇(𝒘𝒕)

## --- [Page 84] ---
84/85

Backpropagation

Descent Method(하강법)

주어진어떤지점에서부터오차가더작은곳으로이동하려는방법

탐색방법∆𝒘에따라하강법이결정

종류: 경사하강법, 모멘텀등Optimization(최적화)

𝒘𝒕ା𝟏←𝒘𝒕−𝝁𝛁𝑳(𝒘𝒕),

학습률(learning rate, step size)

탐색방향

𝒘𝒉𝒆𝒓𝒆  𝛁𝑳𝒘𝒕∆𝒘< 𝟎

## --- [Page 85] ---
85/85

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Division of Computer〮AI Engineering

Dong-A University, Busan, Rep. of Korea