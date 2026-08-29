## --- [Page 1] ---
Dong-A Univ. (ISPL)

머신러닝


|  |  |
| --- | --- |
|  |  |
|  | 컴퓨터 |

## --- [Page 2] ---
2/21

Linear Regression

회귀(Regression )

•
관측된데이터를통해변수사이의숨어있는관계를추정하는것

•
변수y를오염도, x를약품투입량으로가정했을때, 기존의데이터를이용하여𝒚= 𝒘𝒙+ 𝒃수식생성


| 변수 x | 변수 y |
| --- | --- |
| 사람의 키 | 사람의 몸무게 |
| 주택의 크기 | 주택의 가격 |
| 공부 시간 | 시험 점수 |
| 약품 투입량 | 오염도 |
| … | … |

## --- [Page 3] ---
3/21

Linear Regression

회귀(Regression )

•
관측된데이터를통해변수사이의숨어있는관계를추정하는것

•
변수y를오염도, x를약품투입량으로가정했을때, 기존의데이터를이용하여𝒚= 𝒘𝒙+ 𝒃수식생성

•
이후신규데이터를이용하여오염도예측

회귀분석을통해두개의수식중

오차가더작은수식탐색

## --- [Page 4] ---
4/21

Linear Regression

선형회귀(Linear Regression )

•
선형회귀: 변수사이의숨어있는관계를1차원수식으로표현(𝒚= 𝒘𝒙+ 𝒃)

•
비선형회귀: 변수사이의숨어있는관계를2차원이상의수식으로표현(𝒚= 𝒘𝟏𝒙𝒏+ 𝒘𝟐𝒙𝒏ି𝟏… 𝒘𝒏𝒙+ 𝒃)

## --- [Page 5] ---
5/21

Linear Regression

문제해결방법

•
Ordinary Least Squares method (OLS)

= Least Squares Method (LSM)

= Normal Equation

•
Gradient Descent method (GD)

x

y
y = wx + b

단일선형회귀예시

## --- [Page 6] ---
6/21

Linear Regression

Ordinary Least Squares method (OLS)

•
아래예시data를이용하여선형회귀진행(최적의𝜃 𝑤, 𝑏탐색)

(x1, y1)

(x2, y2)

(x3, y3)

(x4, y4)

𝑋=

−3
−1

1
3

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x1
4x1
2x1

## --- [Page 7] ---
7/21

Linear Regression

Ordinary Least Squares method (OLS)

•
아래예시data를이용하여선형회귀진행(최적의𝜃 𝑤, 𝑏탐색)

(x1, y1)

(x2, y2)

(x3, y3)

(x4, y4)

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

𝑋𝜃= 𝑌෠

bias

최종목표: 𝑌= 𝑌෠가되는𝜃탐색

## --- [Page 8] ---
8/21

Linear Regression

Ordinary Least Squares method (OLS)

•
아래예시data를이용하여선형회귀진행(최적의𝜃 𝑤, 𝑏탐색)

(x1, y1)

(x2, y2)

(x3, y3)

(x4, y4)

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

𝑋𝜃= 𝑌෠

bias

최종목표: 𝑌= 𝑌෠가되는𝜃탐색

𝜃= 𝑋ିଵȉ 𝑌
X가정방행렬이아니기때문에불가능

## --- [Page 9] ---
9/21

Linear Regression

Ordinary Least Squares method (OLS)

•
아래예시data를이용하여선형회귀진행(최적의𝜃 𝑤, 𝑏탐색)

(x1, y1)

(x2, y2)

(x3, y3)

(x4, y4)

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

𝑋𝜃= 𝑌෠

bias

최종목표: 𝑌= 𝑌෠가되는𝜃탐색

(𝑋்ȉ 𝑋)𝜃= 𝑋்ȉ 𝑌

## --- [Page 10] ---
10/21

Linear Regression

Ordinary Least Squares method (OLS)

•
아래예시data를이용하여선형회귀진행(최적의𝜃 𝑤, 𝑏탐색)

(x1, y1)

(x2, y2)

(x3, y3)

(x4, y4)

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

𝑋𝜃= 𝑌෠

bias

최종목표: 𝑌= 𝑌෠가되는𝜃탐색

(𝑋்ȉ 𝑋)𝜃= 𝑋்ȉ 𝑌

𝑋்ȉ 𝑋ି
ଵȉ (𝑋்ȉ 𝑋)𝜃= 𝑋்ȉ 𝑋ି
ଵȉ (𝑋்ȉ 𝑌)

𝑋்ȉ 𝑋ି
ଵ를양변에곱하여𝜃계산

## --- [Page 11] ---
11/21

Linear Regression

Ordinary Least Squares method (OLS)

•
아래예시data를이용하여선형회귀진행(최적의𝜃 𝑤, 𝑏탐색)

(x1, y1)

(x2, y2)

(x3, y3)

(x4, y4)

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

bias

𝜃= 𝑋்ȉ 𝑋ି
ଵȉ (𝑋்ȉ 𝑌)

𝜃= 0.8

1

## --- [Page 12] ---
12/21

Linear Regression

Gradient Descent method (GD)

•
예측값(𝑌෠) 과정답값(𝑌) 간의차이를이용하여𝜃 를업데이트하며최적의𝜃 를탐색하는방법

θ

Loss(θ)

optimal θ

초기지점

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

bias

𝑋𝜃= 𝑌෠

𝐿𝑜𝑠𝑠= 1

𝑁෍𝑌−𝑌෠ଶ

평균제곱오차(Mean Square Error (MSE))

## --- [Page 13] ---
13/21

Linear Regression

Gradient Descent method (GD)

•
예측값(𝑌෠) 과정답값(𝑌) 간의차이를이용하여𝜃 를업데이트하며최적의𝜃 를탐색하는방법

θ

Loss(θ)

optimal θ

Loss가최소인위치의
θ를구하는것이목표

𝑋=

−3
1
−1
1
1
1
3
1

𝑌=

−1
−1

3
3

𝜃= 𝑤

𝑏

4x2
4x1
2x1

bias

𝑋𝜃= 𝑌෠

𝐿𝑜𝑠𝑠= 1

𝑁෍𝑌−𝑌෠ଶ

평균제곱오차(Mean Square Error (MSE))

## --- [Page 14] ---
14/21

Linear Regression

Gradient Descent method (GD)

•
현재지점에서Loss 값을𝜃 에대한편미분을통해gradient 계산

•
Gradient에learning rate를곱하고반대방향으로weight 업데이트

θ

Loss(θ)

optimal θ

Gradient = 0.8

Gradient = 0
1
t
t

t

L










Learning rate
Gradient

## --- [Page 15] ---
15/21

Linear Regression

Gradient Descent method (GD)

•
현재지점에서Loss 값을𝜃 에대한편미분을통해gradient 계산

•
Gradient에learning rate를곱하고반대방향으로weight 업데이트

θ

Loss(θ)

optimal θ

1
t
t

t

L










Gradient = 0.8

Learning rate = 0.1

0.08
t



## --- [Page 16] ---
16/21

Linear Regression

Gradient Descent method (GD)

•
현재지점에서Loss 값을𝜃 에대한편미분을통해gradient 계산

•
Gradient에learning rate를곱하고반대방향으로weight 업데이트

•
Learning rate: 파라미터를얼마나업데이트할지정하는하이퍼파라미터

α: Learning rate

## --- [Page 17] ---
17/21

Linear Regression

Gradient Descent method (GD)

𝒀෡= 𝒘𝒙+ 𝒃

𝑳𝒐𝒔𝒔= 𝟏

𝑵෍𝒀−𝒀෡𝟐

= 𝟏

𝑵෍𝒀−𝒘𝒙𝒊−𝒃𝟐

𝒀෡= 𝒘𝒙+ 𝒃대입

## --- [Page 18] ---
18/21

Linear Regression

Gradient Descent method (GD)

𝒀෡= 𝒘𝒙+ 𝒃

𝑳𝒐𝒔𝒔= 𝟏

𝑵෍𝒀−𝒀෡𝟐

= 𝟏

𝑵෍𝒀−𝒘𝒙𝒊−𝒃𝟐

𝝏𝑳
𝝏𝒘= 𝟏

𝑵× 𝟐× ෍𝒀−𝒘𝒙𝒊−𝒃× −𝒙𝒊

≈𝟐

𝑵෍𝒀−𝒀෡× −𝑿

w에대한편미분

## --- [Page 19] ---
19/21

Linear Regression

Gradient Descent method (GD)

𝒀෡= 𝒘𝒙+ 𝒃

𝑳𝒐𝒔𝒔= 𝟏

𝑵෍𝒀−𝒀෡𝟐

= 𝟏

𝑵෍𝒀−𝒘𝒙𝒊−𝒃𝟐

𝝏𝑳
𝝏𝒘= 𝟏

𝑵× 𝟐× ෍𝒀−𝒘𝒙𝒊−𝒃× −𝒙𝒊

≈𝟐

𝑵෍𝒀−𝒀෡× −𝑿

b에대한편미분

𝝏𝑳
𝝏𝒃= 𝟏

𝑵× 𝟐× ෍𝒀−𝒘𝒙𝒊−𝒃 × −𝟏

≈𝟐

𝑵෍𝒀−𝒀෡ × −𝟏

## --- [Page 20] ---
20/21

Linear Regression

Gradient Descent method (GD)

𝒀෡= 𝒘𝒙+ 𝒃

𝑳𝒐𝒔𝒔= 𝟏

𝑵෍𝒀−𝒀෡𝟐

= 𝟏

𝑵෍𝒀−𝒘𝒙𝒊−𝒃𝟐

𝑤௧ାଵ= 𝑤௧−𝛼× 𝜕L

𝜕w

𝑏௧ାଵ= 𝑏௧−𝛼× 𝜕L

𝜕b

𝝏𝑳
𝝏𝒘= 𝟏

𝑵× 𝟐× ෍𝒀−𝒘𝒙𝒊−𝒃× −𝒙𝒊

≈𝟐

𝑵෍𝒀−𝒀෡× −𝑿

𝝏𝑳
𝝏𝒃= 𝟏

𝑵× 𝟐× ෍𝒀−𝒘𝒙𝒊−𝒃 × −𝟏

≈𝟐

𝑵෍𝒀−𝒀෡ × −𝟏

## --- [Page 21] ---
21/21

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea