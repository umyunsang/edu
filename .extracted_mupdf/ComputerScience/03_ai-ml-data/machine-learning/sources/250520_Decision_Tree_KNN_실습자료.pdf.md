## --- [Page 1] ---
1/22

Dong-A Univ. (ISPL)

컴퓨터AI공학부

2025년 1학기 머신러닝


| K Nearest Neighbors – | 실습 |  |
| --- | --- | --- |


## --- [Page 2] ---
2/22

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Ex) 야구, 배구, 축구, 농구 분류

## --- [Page 3] ---
3/22

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Entropy가 감소하는 Threshold (기준점)을찾아내는 것이 목적

## --- [Page 4] ---
4/22

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Entropy가 감소하는 Threshold (기준점)을찾아내는 것이 목적

Entropy 낮음
Entropy 높음

Threshold

Entropy 낮음
Entropy 낮음

Threshold

## --- [Page 5] ---
5/22

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Ex) 2개의 특성, 3개의 class를 가지는 데이터를 Decision tree를 이용해 분류

x1

x2

## --- [Page 6] ---
6/22

Entropy 높음

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Ex) 2개의 특성, 3개의 class를 가지는 데이터를 Decision tree를 이용해 분류

x1

x2

x1

x2

Threshold

Threshold

Entropy 높음

Entropy 높음


|  | En |
| --- | --- |
|  |  |

## --- [Page 7] ---
7/22

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Ex) 2개의 특성, 3개의 class를 가지는 데이터를 Decision tree를 이용해 분류

x1

x2

Threshold

Entropy 높음


|  | En |
| --- | --- |
|  |  |

## --- [Page 8] ---
8/22

x1

x2

Threshold

Review – Decision tree

▪
목적: 데이터에 있는 규칙을 학습을 통해 찾아내 Tree 기반의 분류 규칙 생성

▪
Ex) 2개의 특성, 3개의 class를 가지는 데이터를 Decision tree를 이용해 분류

## --- [Page 9] ---
9/22

[실습] Decision Tree (DT)

▪
Basecode 다운로드

## --- [Page 10] ---
10/22

[실습] Decision Tree (DT)

▪
Basecode 다운로드

feature

Right

Threshold

Value

## --- [Page 11] ---
11/22

[실습] Decision Tree (DT)

▪
Decision tree 코드 작성

▪
재귀 구조를 이용하여 코드 작성

x1

x2

Threshold

Entropy 높음

x1

x2

Threshold


|  | En |
| --- | --- |
|  |  |

## --- [Page 12] ---
12/22

[실습] Decision Tree (DT)

▪
Decision tree 코드 작성

▪
재귀 구조를 이용하여 코드 작성

## --- [Page 13] ---
13/22

[실습] Decision Tree (DT)

▪
Decision tree 코드 작성

▪
재귀 구조를 이용하여 코드 작성

## --- [Page 14] ---
14/22

[실습] Decision Tree (DT)

▪
Decision tree 코드 작성

▪
재귀 구조를 이용하여 코드 작성

Left

Right
Right

Right

## --- [Page 15] ---
15/22

[실습] Decision Tree (DT)

▪
Decision tree 코드 작성

▪
재귀 구조를 이용하여 코드 작성

## --- [Page 16] ---
16/22

Review – K Nearest Neighbors

▪
목적: 새로운 샘플에서 가장 인접한 k개 샘플의 class에 따라 현재 class 분류

Ref.: https://towardsdatascience.com/

Euclidean distance (L2 distance)

## --- [Page 17] ---
17/22

Review – K Nearest Neighbors

▪
목적: 새로운 샘플에서 가장 인접한 k개 샘플의 class에 따라 현재 class 분류

Ref.: https://medium.com/swlh/k-nearest-neighbor-ca2593d7a3c4

## --- [Page 18] ---
18/22

[실습] K Nearest Neighbors (KNN)

▪
Basecode 다운로드

## --- [Page 19] ---
19/22

[실습] K Nearest Neighbors (KNN)

▪
Euclidian Distance 함수, KNN 모델 작성

Euclidean distance (L2 distance)

## --- [Page 20] ---
20/22

[실습] K Nearest Neighbors (KNN)

▪
예측 및 성능 평가

## --- [Page 21] ---
21/22

[실습] K Nearest Neighbors (KNN)

▪
예측 결과 시각화

•
Label: 정답 데이터

•
Prediction: 예측 값

## --- [Page 22] ---
22/22

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea