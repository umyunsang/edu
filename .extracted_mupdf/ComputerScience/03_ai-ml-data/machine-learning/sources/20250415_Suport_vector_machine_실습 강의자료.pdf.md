## --- [Page 1] ---
Dong-A Univ. (ISPL)

컴퓨터AI공학부


|  |  |
| --- | --- |
|  |  |
|  | 1/31 |

## --- [Page 2] ---
2/31

Review – Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기

머리길이

목소리톤

: 남성

: 여성

머리길이

목소리톤

## --- [Page 3] ---
3/31

Review – Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기

: 남성

: 여성

머리길이

목소리톤

머리길이

목소리톤

## --- [Page 4] ---
4/31

Review – Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기

Support

vector

Maximum

margin

: 남성

: 여성

머리길이

목소리톤

머리길이

목소리톤

## --- [Page 5] ---
5/31

Review – Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기

Maximum

margin

0
T
W X
b



(
)
T
f X
W X
b



optimal separating hyperplane

: 남성

: 여성

머리길이

목소리톤

## --- [Page 6] ---
6/31

Review – Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기

Maximum

margin

0
T
W X
b



(
)
T
f X
W X
b



Weight (vector)
Normal vector

Margin 방향의unit vector

W
W

W



optimal separating hyperplane

: 남성

: 여성

머리길이

목소리톤

## --- [Page 7] ---
7/31

Review – Support Vector Machine


임의의데이터xi 에대해separating hyperplane과의거리: 𝝆

Separating hyperplane

Xi

Xp

x1

x2

W
W

?

?

Margin 방향의unit vector

## --- [Page 8] ---
8/31

Review – Support Vector Machine


임의의데이터xi 에대해separating hyperplane과의거리: 𝝆

Separating hyperplane

Xi

Xp

x1

x2

W
W


i
p

W
x
x
W





(
)
0
T
p
p
f x
W x
b







2

( )
T
i
i

T

p

T

p

f x
W x
b

W
W
x
b
W

W
W x
b
W



























( )
f x

W




0

1

2

3

4

## --- [Page 9] ---
9/31

Review – Support Vector Machine


Binary classification

1

2

3
1
: Optimal separating hyperplane

2
: Support vector (negative)

3
: Support vector (positive)

(
)
0
T
f X
W X
b




(
)
1
T
f X
W X
b




(
)
1
T
f X
W X
b




## --- [Page 10] ---
10/31

Review – Support Vector Machine


Binary classification

1
: Optimal separating hyperplane

2
: Support vector (negative)

3
: Support vector (positive)

(
)
0
T
f X
W X
b




(
)
1
T
f X
W X
b




(
)
1
T
f X
W X
b




1
T
W X
b



1
T
W X
b



1

2

3

## --- [Page 11] ---
11/31

Review – Support Vector Machine


Margin 계산

x
x
W






1

(
)
1

1

T

T

T
T

W x
b

W
x
W
b

W x
b
W
W






















3

-1

2

T
W W




x

2

## --- [Page 12] ---
12/31

Review – Support Vector Machine


Margin 계산
1

(
)
1

1

T

T

T
T

W x
b

W
x
W
b

W x
b
W
W





















2

T
W W




2
2

2

Margin
distance(
,
)

2
2
T
T

x
x

x
x
W

W W
W W
W



















x
x
W






3

x

2

## --- [Page 13] ---
13/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기

(
)
T
f X
W X
b



## --- [Page 14] ---
14/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)

(
)
T
f X
W X
b



## --- [Page 15] ---
15/31

[실습] Support Vector Machine


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)

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

Gradient decent algorithm

①현재지점에서미분을이용해gradient 계산

②Gradient에learning rate를곱하고

반대방향으로weight update
Gradient = 0.8

Learning rate = 0.1

0.08
t



## --- [Page 16] ---
16/31

[실습] Support Vector Machine


Loss function (Cost function): Hinge loss

1
T

i
W X
b



1
T

i
W X
b




1
iy 

1
iy 


Label
Prediction

(
)
0
T
i
i
y W x
b



Label
Prediction

이조건을만족하는경우정상적으로분류성공

1
T
W X
b



1
T
W X
b



1

2

3

## --- [Page 17] ---
17/31

[실습] Support Vector Machine


Loss function (Cost function): Hinge loss

1
T

i
W X
b



1
T

i
W X
b




1
iy 

1
iy 


Label
Prediction

(
)
0
T
i
i
y W x
b



Label
Prediction

이조건을만족하는경우정상적으로분류성공

Hinge loss

max(0,1
(
))
T
i
i
Loss
y W x
b




## --- [Page 18] ---
18/31

[실습] Support Vector Machine


Loss function (Cost function): Hinge loss

1
T

i
W X
b



1
T

i
W X
b




1
iy 

1
iy 


Label
Prediction

(
)
0
T
i
i
y W x
b



Label
Prediction

이조건을만족하는경우정상적으로분류성공

Hinge loss

max(0,1
(
))
T
i
i
Loss
y W x
b




(
)
1
T
i
i
y W x
b



(
)
0
T
i
i
y W x
b



(
)
0.5
T
i
i
y W x
b



Loss = +2

Loss = +1

(
)
1
T
i
i
y W x
b



Loss = +0.5

Loss = 0

## --- [Page 19] ---
19/31

[실습] Support Vector Machine


Loss function (Cost function): Hinge loss Gradient

Hinge loss

max(0,1
(
))
T
i
i
Loss
y W x
b




(
)
1
T
i
i
y W x
b



otherwise

1

2
1
(
)
T
i
i
Loss
y W x
b



0
Loss 

## --- [Page 20] ---
20/31

[실습] Support Vector Machine


Loss function (Cost function): Hinge loss Gradient

Hinge loss

max(0,1
(
))
T
i
i
Loss
y W x
b




(
)
1
T
i
i
y W x
b



otherwise

1

2
1
(
)
T
i
i
Loss
y W x
b



0
Loss 

(
)
1
T
i
i
y W x
b


1

0
L
W



0
L
b





Update 수행X

otherwise
2

i
i
L
y x
W



i
L
y
b





## --- [Page 21] ---
21/31

[실습] Support Vector Machine


Basecode 다운로드: LMS 강의콘텐츠13주차

## --- [Page 22] ---
22/31

[실습] Support Vector Machine


데이터셋생성: sklearn.datasets

Ref.: https://datascienceschool.net/

## --- [Page 23] ---
23/31

[실습] Support Vector Machine


데이터셋생성: sklearn.datasets

## --- [Page 24] ---
24/31

[실습] Support Vector Machine


SVM 모델작성및gradient decent 코드작성

## --- [Page 25] ---
25/31

[실습] Support Vector Machine


SVM 모델training 및도출된W, b값확인

2
2

2

Margin
distance(
,
)

2
2
T
T

x
x

x
x
W

W W
W W
W



















## --- [Page 26] ---
26/31

[실습] Support Vector Machine


Visualization

Margin

## --- [Page 27] ---
27/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)


목적함수: 𝑚𝑖𝑛

ଵ

ଶ𝑤ଶ


제약조건: subject to  𝑦௜(𝑊்𝑥௜+ 𝑏) ≥1

## --- [Page 28] ---
28/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)


목적함수: 𝑚𝑖𝑛

ଵ

ଶ𝑤ଶ


제약조건: subject to  𝑦௜(𝑊்𝑥௜+ 𝑏) ≥1

## --- [Page 29] ---
29/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)


목적함수: 𝑚𝑖𝑛

ଵ

ଶ𝑤ଶ


제약조건: subject to  𝑦௜(𝑊்𝑥௜+ 𝑏) ≥1

linear

C

## --- [Page 30] ---
30/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)


목적함수: 𝑚𝑖𝑛

ଵ

ଶ𝑤ଶ


제약조건: subject to  𝑦௜(𝑊்𝑥௜+ 𝑏) ≥1

## --- [Page 31] ---
31/31

[실습] Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


Solution

•
Gradient Decent Method (GD) Optimal W, b

•
Qudratic Programming (2차계획법)


목적함수: 𝑚𝑖𝑛

ଵ

ଶ𝑤ଶ


제약조건: subject to  𝑦௜(𝑊்𝑥௜+ 𝑏) ≥1

Ref: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

커널행렬생성
Step1

Qudratic Programming 문제정의
Step2

Convex Optimization
Step3

𝝀ᵢ > 0인경우support vector 선택
Step4

w, b 계산
Step5

fit 함수

Margin

## --- [Page 32] ---
32/31

Questions & Answers

Dongsan Jun (dsjun@dau.ac.kr)

Image Signal Processing Laboratory (www.donga-ispl.kr)

Dept. of Computer Engineering

Dong-A University, Busan, Rep. of Korea

## --- [Page 33] ---
33/31

Review – Support Vector Machine


목적: Margin을최대화하는optimal separating hyperplane (decision boundary) 구하기


예시: 스팸메일

: 스팸메일

: 일반메일

𝒙𝟐

𝒙𝟏

Maximum

margin