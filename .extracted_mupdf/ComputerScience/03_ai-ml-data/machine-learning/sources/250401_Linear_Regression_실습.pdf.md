## --- [Page 1] ---
1/33

Dong-A Univ. (ISPL)

컴퓨터AI공학부

2025년1학기머신러닝

## --- [Page 2] ---
2/33


실습목적: 선형회귀기법을통해주어진데이터셋에서데이터간의관계를분석하고예측하는모델을구현

실습개요

181 cm
172 cm
? cm

price: ?

## --- [Page 3] ---
3/33


Linear Regression (선형회귀)

•
Single Linear Regression (단일선형회귀)

평수

가격

단일선형회귀예시

실습개요

10평

50만원

20평

100만원

12평

? 만원

## --- [Page 4] ---
4/33


Linear Regression (선형회귀)

•
Single Linear Regression (단일선형회귀)


Solutions

•
Ordinary Least Squares (OLS)

= Least Squares Method (LSM)

= Normal Equation

•
Gradient Descent Method

x

y
y = wx + b

단일선형회귀예시

실습개요

## --- [Page 5] ---
5/33

실습개요


Dataset: kc_house_data

•
2014 ~ 2015년사이판매된주택가격데이터셋

•
21개변수, 21,613개데이터로구성됨
가격
침실
화장실
크기
…

…

https://www.kaggle.com/datasets/shivachandel/kc-house-data/

## --- [Page 6] ---
6/33

Google Colaboratory (Colab)

•
딥러닝, 머신러닝모델등을실행할수있는무료클라우드서비스

•
모델학습을위해GPU를일정시간동안무료로사용할수있음

•
구글계정으로로그인해사용가능(colab.research.google.com)

실습환경구성

## --- [Page 7] ---
7/33

[1] 웹브라우저에서Google 로그인

[2] 구글드라이브접속(drive.google.com)

[3] 구글colab 설치(아래사진참고)

1

2
3

4

5

6

실습환경구성

## --- [Page 8] ---
8/33

[4] 구글드라이브내실습코드를보관할폴더생성(띄어쓰기, 한글사용X)

[5] 구글Colab 실행

1

2

3

4

Colab 실행화면

실습환경구성

## --- [Page 9] ---
9/33

LMS 강의콘텐츠5주차1차시Base code 및데이터셋다운로드

1

2

실행

실습환경구성

데이터셋저장확인

## --- [Page 10] ---
10/33

Single Linear Regression을위한데이터셋확인

실습환경구성

•
Y: 정답데이터(price)

•
X: 입력데이터(sqft_living)

## --- [Page 11] ---
11/33

Single Linear Regression을위한데이터셋확인

X, Y 값의scale이너무큰경우학습이잘안될수있음
Gaussian 정규화수행

실습환경구성

## --- [Page 12] ---
12/33

Single Linear Regression을위한데이터셋확인

Train dataset : Test dataset = 8 : 2 로분할

실습환경구성

## --- [Page 13] ---
13/33

Single Linear Regression을위한데이터셋확인

실습환경구성

## --- [Page 14] ---
14/33

Review – Least Squares Method


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

## --- [Page 15] ---
15/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

Review – Least Squares Method

※최종목표: 𝑌= 𝑌෠가되는𝜃 탐색
𝑋𝜃= 𝑌

## --- [Page 16] ---
16/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

𝜃= 𝑋ିଵȉ 𝑌
X가정방행렬이아니기때문에불가능

Review – Least Squares Method

𝑋𝜃= 𝑌

## --- [Page 17] ---
17/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

(𝑋்ȉ 𝑋)𝜃= 𝑋்ȉ 𝑌

Review – Least Squares Method

𝑋𝜃= 𝑌

## --- [Page 18] ---
18/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

𝑋𝜃= 𝑌

bias

(𝑋்ȉ 𝑋)𝜃= 𝑋்ȉ 𝑌

𝑋்ȉ 𝑋ି
ଵȉ (𝑋்ȉ 𝑋)𝜃= 𝑋்ȉ 𝑋ି
ଵȉ (𝑋்ȉ 𝑌)

Review – Least Squares Method

## --- [Page 19] ---
19/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

실습– Least Squares Method்ି

ଵ்

## --- [Page 20] ---
20/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

실습– Least Squares Method

## --- [Page 21] ---
21/33

Review – Gradient Descent


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

𝐿𝑜𝑠𝑠= ෍𝑌−𝑌෠

## --- [Page 22] ---
22/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

𝐿𝑜𝑠𝑠= ෍𝑌−𝑌෠

Review – Gradient Descent

## --- [Page 23] ---
23/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

Learning rate
Gradient

Gradient Descent algorithm

①현재지점에서미분을이용해gradient 계산

②Gradient에learning rate를곱하고

반대방향으로weight update
Gradient = 0.8

Gradient = 0

Review – Gradient Descent

## --- [Page 24] ---
24/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method

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

Gradient Descent algorithm

①현재지점에서미분을이용해gradient 계산

②Gradient에learning rate를곱하고

반대방향으로weight update
Gradient = 0.8

Learning rate = 0.1

0.08
t



Review – Gradient Descent

## --- [Page 25] ---
25/33


Solutions

•
Ordinary Least Squares (OLS) = Least Squares Method (LSM) = Normal Equation

•
Gradient Descent Method


Learning rate: 파라미터를얼마나업데이트할지정하는하이퍼파라미터

α: Learning rate

Review – Gradient Descent

## --- [Page 26] ---
26/33

𝟐

𝒊
𝟐


Gradient Descent Method (Parameter update)

Review – Gradient Descent

## --- [Page 27] ---
27/33


Gradient Descent Method (Parameter update)

𝟐

𝒊
𝟐

𝒊
𝒊

Review – Gradient Descent

## --- [Page 28] ---
28/33

𝒊

𝒊
𝒊


Gradient Descent Method (Parameter update)

𝟐

𝒊
𝟐

Review – Gradient Descent

## --- [Page 29] ---
29/33

𝒊
𝒊


Gradient Descent Method (Parameter update)

𝟐

𝒊
𝟐

௧ାଵ
௧

௧ାଵ
௧

Review – Gradient Descent

𝒊

## --- [Page 30] ---
30/33

Single Linear Regression 모델작성

실습– Gradient Descent

## --- [Page 31] ---
31/33

Single Linear Regression 모델작성

실습– Gradient Descent

𝑋=

−3
1
−1
1
1
1
3
1

4x2

bias

## --- [Page 32] ---
32/33

Single Linear Regression 모델작성

실습– Gradient Descent

## --- [Page 33] ---
33/33

Single Linear Regression 모델작성

𝝏𝑳
𝝏𝒃= 𝟏

𝑵෍𝒀−𝒀෡ × −𝟏

𝝏𝑳
𝝏𝒘= 𝟏

𝑵෍𝒀−𝒀෡× −𝑿

𝒕ା𝟏
𝒕

𝒕ା𝟏
𝒕

실습– Gradient Descent

## --- [Page 34] ---
34/33

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea