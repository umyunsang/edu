### [Page 1]
1/59
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능



### [Page 2]
2/59
Overall Architecture of Deep Learning 
Training Input
x
x
x
Drop-Out
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
•
ReLU: Rectified Linear Unit
•
Adam: Adaptive Moment Estimation
•
PSNR: Peak Signal-to-Noise Ratio
•
SSIM: Structural Similarity Index Measure



### [Page 3]
3/59
Contents
1. Introduction
2. Applications
①Regression
②Classification
③Logic gate
3. Multi-layer Perceptron (MLP)



### [Page 4]
4/59
Introduction
Dartmouth Conference (1956년)
•
목적: 지능을가진기계연구인공지능(Artificial Intelligence)
•
주요성과: 초기인공지능의개념및분야확립
Dartmouth 학술회의참여연구자
Dartmouth 학술회의개최지
(미국뉴햄프셔Dartmouth 대학)



### [Page 5]
5/59
Introduction
기호주의인공지능vs. 연결주의인공지능
•
두연구자를중심으로서로다른인공지능학습방법을주장
<Marvin Minsky>
<Frank Rosenblatt>
인공지능은컴퓨터
작동방식에가까운
기호관계를정의해
학습해야한다.
실제뇌가정보를
처리하는방식에
가까운인공신경망
으로학습해야한다.



### [Page 6]
6/59
Introduction
<Marvin Minsky>
<Frank Rosenblatt>
인공지능은컴퓨터
작동방식에가까운
기호관계를정의해
학습해야한다.
실제뇌가정보를
처리하는방식에
가까운인공신경망
으로학습해야한다.
기호주의인공지능
연결주의인공지능
기호주의인공지능vs. 연결주의인공지능
•
두연구자를중심으로서로다른인공지능학습방법을주장



### [Page 7]
7/59
Introduction
기호주의인공지능(Symbolism AI)
•
컴퓨터의작동방식에맞추어규칙기반(Rule-based)으로인공지능을학습
•
(1) 인간의지식을기호화(2) 기호간관계정의(3) 연역적추론
<Marvin Minsky>
Knowledge1
Knowledge2
Inference
Result
Knowledge1: 사람은죽는다.
Knowledge2: 소크라테스는사람이다.
Inference: 소크라테스는죽는가?
Result: 죽는다.



### [Page 8]
8/59
Introduction
20
100
52
173
기호주의인공지능의한계
•
현실세계의모든형상/규칙/개념을기호화하는것은불가능함
•
Ex. 기호주의기법으로고양이를판별하는인공지능
[1] Training
Cat = [ 20, 100, 52, 173 ]
픽셀값예시


| 20 | 100 |
| --- | --- |
| 52 | 173 |


### [Page 9]
9/59
Introduction
20
100
52
173
120
200
152
255
기호주의인공지능의한계
•
현실세계의모든형상/규칙/개념을기호화하는것은불가능함
•
Ex. 기호주의기법으로고양이를판별하는인공지능
[1] Training
[2] Inference
픽셀값예시
픽셀값예시
Cat = [ 20, 100, 52, 173 ]
[ 120, 200, 152, 255 ] is Cat? No


| 20 | 100 |
| --- | --- |
| 52 | 173 |

| 120 | 200 |
| --- | --- |
| 152 | 255 |


### [Page 10]
10/59
Introduction
기호주의인공지능의한계
•
현실세계의모든형상/규칙/개념을기호화하는것은불가능함
•
Ex. 기호주의기법으로고양이를판별하는인공지능
밝기, 자세, 품종, 크기등모든자연형상을고려하지못함
Challenge(1): Illumination
Challenge(2): Deformation
Challenge(3): Background Clutter
Challenge(4): Intraclass Variation
Ref.: Stanford Univ., “CS231N Lecture2: Image Classification Pipeline”, F.F. Li et al.



### [Page 11]
11/59
Introduction
연결주의인공지능(Connection AI)
•
인간이학습하는방법을구현/모방해신경망(Neural Network) 기반으로인공지능을학습
•
대표적인알고리즘으로perceptron이존재함(1958)
<Frank Rosenblatt>
신경세포(Neuron)
인공신경망예시(Perceptron)



### [Page 12]
12/59
Introduction
Perceptron 출력과정
1.
입력신호와가중치값을곱해준뒤편향값을더함
2.
더한값이임계값을넘으면1 그렇지않으면0을출력
<Perceptron>
•
𝑥ଵ, 𝑥ଶ: 입력값
•
𝑤ଵ, 𝑤ଶ: 가중치
•
b : 편향
•
𝜽: 임계값(0으로설정)
•
𝑦: 출력값
𝑤ଵ
1
𝑥ଵ
Σ
𝜽=0
𝑥ଶ
𝑤ଶ
𝑏
𝑦
𝑦= ቊ1   𝑏+ 𝑤ଵ𝑥ଵ+ 𝑤ଶ𝑥ଶ> 0
0   𝑏+ 𝑤ଵ𝑥ଵ+ 𝑤ଶ𝑥ଶ≤0
1
𝜽=0
0.5
0.5
-0.2
1
1
0.8
1
각가중치값과편향값을통해출력값조정
1
𝜽=0
-0.5
-0.5
0.7
1
1
-0.3
0



### [Page 13]
13/59
Introduction
XOR 문제와AI winter (1969 ~ 1985)
•
기호주의인공지능을지지하던M. Minsky는perceptron의한계를증명한책을출판
Perceptron은어떤방법으로도XOR 문제를풀수없음
•
이후연결주의인공지능연구는관심을받지못함AI winter
Minsky’s perceptrons
XOR 문제예시
1개의직선으로0과1을구분할수없음
: 0
: 1



### [Page 14]
14/59
Introduction
Multi-layer Perceptron (MLP, 1986)
•
여러층의perceptron으로구성된MLP와오차역전파법이개발됨
•
XOR 문제를해결함으로써인공지능연구가다시활발해지는계기가됨
<D. Rumelhart, G. Hinton, and R. Wiliams>
out
<Multi-Layer Perceptron>



### [Page 15]
15/59
Introduction
인공지능역사의흐름
1956
Perceptron 발표
Dartmouth Conference
XOR 문제
Multi-layer Perceptron,
Backpropagation 발표
1957
1969
1986
AI winter
Deep Neural
Network



### [Page 16]
16/59
Introduction
Breakthrough for Artificial Intelligence
•
(1) 오차역전파법개발(Back-propagation)
•
(2) 대규모데이터셋확보(Big data)
•
(3) 하드웨어기술발전
(1) 오차역전파법
(2) 대규모데이터셋
(3) GPU를이용한학습



### [Page 17]
17/59
Introduction
이미지넷대규모시각인식대회(Image Large Scale Visual Recognition Challenge, ILSVR)
•
대용량이미지데이터셋(1,000개클래스)에대한이미지분류알고리즘성능평가대회
•
딥러닝기반기법이제안된이후분류성능이급격히높아짐
딥러닝기반분류모델



### [Page 18]
18/59
Contents
1. Introduction
2. Applications
①Regression
②Classification
③Logic gate
3. Multi-layer Perceptron (MLP)



### [Page 19]
19/59
Applications
Supervised Learning (지도학습)
•
입력데이터와대응되는정답데이터를함께이용해학습하는기법
•
대표적인응용분야로Regression (회귀)와Classification (분류)이존재함
Supervised
Learning
Regression (output: real number)
Classification
Binary Classification
(output: 0 or 1)
Multi-label Classification
(output: 0, 1, 2, …, N)



### [Page 20]
20/59
Applications
Perceptron을이용한regression 예시
•
Ex1. 공부한시간으로시험점수예측
입력데이터: 공부한시간(x)
출력데이터: 시험점수(y)
Output
y (score)
Input
x (hours)
90
10
80
9
50
3
30
2
학습용데이터셋예시


| Input x (hours) | Output y (score) |
| --- | --- |
| 10 | 90 |
| 9 | 80 |
| 3 | 50 |
| 2 | 30 |


### [Page 21]
21/59
Applications
Perceptron을이용한regression 예시
•
Ex1. 공부한시간으로시험점수예측
입력데이터: 공부한시간(x)
출력데이터: 시험점수(y)
Output
y (score)
Input
x (hours)
90
10
80
9
50
3
30
2
?
5


| Input x (hours) | Output y (score) |
| --- | --- |
| 10 | 90 |
| 9 | 80 |
| 3 | 50 |
| 2 | 30 |
| 5 | ? |


### [Page 22]
22/59
Applications
Perceptron을이용한regression 예시
•
Ex1. 공부한시간으로시험점수예측
입력데이터: 공부한시간(x)
출력데이터: 시험점수(y)
Output
y (score)
Input
x (hours)
90
10
80
9
50
3
30
2
65
5
x (hours)
y (score)
2
3
9
10
30
50
80
90
: 정답데이터
: 예측데이터
5
65


| Input x (hours) | Output y (score) |
| --- | --- |
| 10 | 90 |
| 9 | 80 |
| 3 | 50 |
| 2 | 30 |
| 5 | 65 |


### [Page 23]
23/59
Applications
Perceptron을이용한regression 예시
•
Ex1. 공부한시간으로시험점수예측
입력데이터: 공부한시간(x)
출력데이터: 시험점수(y)
x (hours)
y (score)
8
10
y
x


1
Σ
시험점수예측을위한perceptron 구조
w = 8
b = 10
8
10



### [Page 24]
24/59
Applications
Perceptron을이용한regression 예시
•
Ex2. 주택가격예측
입력데이터: 주택가격에영향을미치는정보(x)
출력데이터: 주택가격(y)
주택가격에영향을미치는정보
①넓이(x1)
②교통(x2)
③세대수(x3)
④층수(x4)
예상주택가격(y)



### [Page 25]
25/59
Applications
Perceptron을이용한regression 예시
•
Ex2. 주택가격예측
입력데이터: 주택가격에영향을미치는정보(x)
출력데이터: 주택가격(y)
x1
y
x2
y
x3
y
x4
y
넓이
교통
세대수
층수
주택가격에영향을미치는정보
①넓이(x1)
②교통(x2)
③세대수(x3)
④층수(x4)
학습용데이터셋예시
: 정답데이터



### [Page 26]
26/59
Applications
Perceptron을이용한regression 예시
•
Ex2. 주택가격예측
입력데이터: 주택가격에영향을미치는정보(x)
출력데이터: 주택가격(y)
x1
y
x2
y
x3
y
x4
y
넓이
교통
세대수
층수
주택가격에영향을미치는정보
①넓이(x1)
②교통(x2)
③세대수(x3)
④층수(x4)



### [Page 27]
27/59
Applications
Perceptron을이용한regression 예시
•
Ex2. 주택가격예측
입력데이터: 주택가격에영향을미치는정보(x)
출력데이터: 주택가격(y)
x1
y
x2
y
x3
y
x4
y
넓이
교통
세대수
층수
1 1
y
w x
b


2
2
y
w x
b


4
4
y
w x
b


3
3
y
w x
b


1
1
Σ
𝑤1
𝑏
주택가격예측을위한perceptron 구조
2
4
𝑤2
𝑤4
3
𝑤3



### [Page 28]
28/59
Contents
1. Introduction
2. Applications
①Regression
②Classification
③Logic gate
3. Multi-layer Perceptron (MLP)



### [Page 29]
29/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex1. 중간/기말고사점수로Pass/Non-pass (P/N) 분류
입력데이터: 중간점수(x1), 기말점수(x2)
출력데이터: Pass or Non-pass (y)
Output
Input
P/N (y)
기말(x2)
중간(x1)
1 (P)
80
90
1 (P)
60
75
0 (N)
50
40
0 (N)
25
25
학습용데이터셋예시


| Input |  | Output |
| --- | --- | --- |
| 중간 (x ) 1 | 기말 (x ) 2 | P/N (y) |
| 90 | 80 | 1 (P) |
| 75 | 60 | 1 (P) |
| 40 | 50 | 0 (N) |
| 25 | 25 | 0 (N) |


### [Page 30]
30/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex1. 중간/기말고사점수로Pass/Non-pass (P/N) 분류
입력데이터: 중간점수(x1), 기말점수(x2)
출력데이터: Pass or Non-pass (y)
Output
Input
P/N (y)
기말(x2)
중간(x1)
1 (P)
90
75
1 (P)
60
80
0 (N)
50
40
0 (N)
25
25
: 1 (Pass)
: 0 (Non-pass)
중간(x1)
기말(x2)
학습용데이터셋예시


| Input |  | Output |
| --- | --- | --- |
| 중간 (x ) 1 | 기말 (x ) 2 | P/N (y) |
| 75 | 90 | 1 (P) |
| 80 | 60 | 1 (P) |
| 40 | 50 | 0 (N) |
| 25 | 25 | 0 (N) |


### [Page 31]
31/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex1. 중간/기말고사점수로Pass/Non-pass (P/N) 분류
입력데이터: 중간점수(x1), 기말점수(x2)
출력데이터: Pass or Non-pass (y)
: 1 (Pass)
: 0 (Non-pass)
중간(x1)
기말(x2)
Decision boundary
Perceptron은weight, bias를이용해
decision boundary를구현함



### [Page 32]
32/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex1. 중간/기말고사점수로Pass/Non-pass (P/N) 분류
입력데이터: 중간점수(x1), 기말점수(x2)
출력데이터: Pass or Non-pass (y)
: 1 (Pass)
: 0 (Non-pass)
중간(x1)
기말(x2)
Decision boundary
새로운데이터예시pass로예측할수있음



### [Page 33]
33/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex1. 중간/기말고사점수로Pass/Non-pass (P/N) 분류
입력데이터: 중간점수(x1), 기말점수(x2)
출력데이터: Pass or Non-pass (y)
: 1 (Pass)
: 0 (Non-pass)
중간(x1)
기말(x2)
Decision boundary
1
1
Σ
1
Pass/Non-pass 예측을위한perceptron 구조
1 1
2
2
w x
w x
b


2
2
𝒔> 𝟎?



### [Page 34]
34/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
: 1 (고양이)
: 0 (개)
x1
x2
: 2 (자동차)
학습용데이터셋예시



### [Page 35]
35/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
: 1 (고양이)
: 0 (개)
: 2 (자동차)
개를판단하는perceptron
고양이를판단하는perceptron
자동차를판단하는perceptron
개
개아님
고양이
고양이아님
자동차
자동차아님



### [Page 36]
36/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
: 1 (고양이)
: 0 (개)
: 2 (자동차)
Perceptron (개)
Perceptron (고양이)
Perceptron (자동차)



### [Page 37]
37/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
Perceptron (개)
Perceptron (고양이)
Perceptron (자동차)
1
Σ
1
1
1
2
𝒔> 𝟎?



### [Page 38]
38/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
Perceptron (개)
Perceptron (고양이)
Perceptron (자동차)
1
Σ
1
1
1
Σ
2
2
2
𝒔> 𝟎?
𝒔> 𝟎?



### [Page 39]
39/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
Perceptron (개)
Perceptron (고양이)
Perceptron (자동차)
1
Σ
1
1
1
Σ
2
2
2
Σ
3
3
𝒔> 𝟎?
𝒔> 𝟎?
𝒔> 𝟎?



### [Page 40]
40/59
Applications of Perceptron
Perceptron을이용한classification 예시
•
Ex2. 입력데이터x1, x2로개/고양이/자동차분류
입력데이터: x1, x2
출력데이터: 개or 고양이or 자동차(y)
1
Σ
1
1
1
Σ
2
2
2
Σ
3
3
𝒔> 𝟎?
𝒔> 𝟎?
𝒔> 𝟎?
0: 자동차아님
1: 자동차
0: 고양이아님
1: 고양이
0: 개아님
1: 개



### [Page 41]
41/59
Contents
1. Introduction
2. Applications
①Regression
②Classification
③Logic gate
3. Multi-layer Perceptron (MLP)



### [Page 42]
42/59
 𝐲
 𝒙𝟐
𝒙𝟏
0
0
0
0
1
0
0
0
1
1
1
1
[Preliminary Study] Logic Gate
논리회로(Logic gate)
•
입력값에대해논리연산을수행하여하나의출력값을얻는전자회로
•
대표적인논리회로로AND, OR, NOT gate가존재함
𝑥ଵ
𝑥ଶ
𝑦
AND gate
𝐲
 𝒙𝟐
𝒙𝟏
0
0
0
1
1
0
1
0
1
1
1
1
𝑥ଵ
𝑥ଶ
𝑦
OR gate
𝐲
𝐱
1
0
0
1
𝑦
NOT gate
𝑥
1
2
x
x
y


1
2
x
x
y


x
y



| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

| 𝐱 | 𝐲 |
| --- | --- |
| 0 | 1 |
| 1 | 0 |


### [Page 43]
43/59
[Preliminary Study] Logic Gate
논리회로(Logic gate)
•
입력값에대해논리연산을수행하여하나의출력값을얻는전자회로
•
대표적인논리회로로AND, OR, NOT gate가존재함
 𝐲
 𝒙𝟐
𝒙𝟏
1
0
0
1
1
0
1
0
1
0
1
1
𝑥ଵ
𝑥ଶ
𝑦
NAND gate
𝐲
 𝒙𝟐
𝒙𝟏
1
0
0
0
1
0
0
0
1
0
1
1
𝑥ଵ
𝑥ଶ
𝑦
NOR gate
1
2
x
x
y


1
2
x
x
y




| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 1 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |


### [Page 44]
44/59
[Preliminary Study] Logic Gate
논리회로(Logic gate)
•
입력값에대해논리연산을수행하여하나의출력값을얻는전자회로
 𝐲
 𝒙𝟐
𝒙𝟏
0
0
0
1
1
0
1
0
1
0
1
1
𝑥ଵ
𝑥ଶ
𝑦
XOR gate
1
2
x
x
y




| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |


### [Page 45]
45/59
Logic Gate using Perceptron
일반적인방법을통한logic gate 구현예시- AND gate
 𝐲
 𝒙𝟐
𝒙𝟏
0
0
0
0
1
0
0
0
1
1
1
1


| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |


### [Page 46]
46/59
Logic Gate using Perceptron
Perceptron을이용한논리회로구현
•
Binary classification (이진분류)을통해논리회로를구현할수있음
x1
x2
: y = 1
: y = 0
Decision boundary
 𝐲
 𝒙𝟐
𝒙𝟏
0
0
0
0
1
0
0
0
1
1
1
1
AND gate에대한이진분류예시
1
0
1
0


| 𝒙 𝟏 | 𝒙 𝟐 | 𝐲 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |


### [Page 47]
47/59
Logic Gate using Perceptron
Perceptron을이용한논리회로구현- AND gate
•
적정한weight (w1, w2), bias (b)를지정해야함
•
Ex. 𝑤ଵ= 0.5, 𝑤ଶ= 0.5, b = -0.7
1
ଵ
Σ
s > 0 ?
ଶ
ଵ
ଶ
(1,1)
(1,0)
(0,0)
(0,1)



### [Page 48]
48/59
Logic Gate using Perceptron
Perceptron을이용한논리회로구현- AND gate
•
적정한weight (w1, w2), bias (b)를지정해야함
•
Ex. 𝑤ଵ= 0.5, 𝑤ଶ= 0.5, b = -0.7
1
ଵ
Σ
s > 0 ?
ଶ
ଵ
ଶ
0.5
0.5
-0.7
𝐲
 𝐬
 𝒙𝟐
𝒙𝟏
0
-0.7
0
0
0
-0.2
1
0
0
-0.2
0
1
1
0.3
1
1
Input
Output
1
1
2
2
(1,1)
(1,0)
(0,0)
(0,1)


| 𝒙 𝟏 | 𝒙 𝟐 | 𝐬 | 𝐲 |
| --- | --- | --- | --- |
| 0 | 0 | -0.7 | 0 |
| 0 | 1 | -0.2 | 0 |
| 1 | 0 | -0.2 | 0 |
| 1 | 1 | 0.3 | 1 |


### [Page 49]
49/59
Logic Gate using Perceptron
Perceptron을이용한논리회로구현- OR gate
•
적정한weight (w1, w2), bias (b)를지정해야함
•
Ex. 𝑤ଵ= 0.5, 𝑤ଶ= 0.5, b = -0.2
1
ଵ
Σ
s > 0 ?
ଶ
ଵ
ଶ
0.5
0.5
-0.2
𝐲
 𝐬
 𝒙𝟐
𝒙𝟏
0
-0.2
0
0
1
0.3
1
0
1
0.3
0
1
1
0.8
1
1
Input
Output
1
1
2
2
(1,1)
(1,0)
(0,0)
(0,1)


| 𝒙 𝟏 | 𝒙 𝟐 | 𝐬 | 𝐲 |
| --- | --- | --- | --- |
| 0 | 0 | -0.2 | 0 |
| 0 | 1 | 0.3 | 1 |
| 1 | 0 | 0.3 | 1 |
| 1 | 1 | 0.8 | 1 |


### [Page 50]
50/59
Logic Gate using Perceptron
Perceptron을이용한논리회로구현- NAND gate
•
적정한weight (w1, w2), bias (b)를지정해야함
•
Ex. 𝑤ଵ= -0.5, 𝑤ଶ= -0.5, b = 0.7
1
ଵ
Σ
s > 0 ?
ଶ
ଵ
ଶ
-0.5
-0.5
0.7
𝐲
 𝐬
 𝒙𝟐
𝒙𝟏
1
0.7
0
0
1
0.2
1
0
1
0.2
0
1
0
-0.3
1
1
Input
Output
1
1
2
2
(1,1)
(1,0)
(0,0)
(0,1)


| 𝒙 𝟏 | 𝒙 𝟐 | 𝐬 | 𝐲 |
| --- | --- | --- | --- |
| 0 | 0 | 0.7 | 1 |
| 0 | 1 | 0.2 | 1 |
| 1 | 0 | 0.2 | 1 |
| 1 | 1 | -0.3 | 0 |


### [Page 51]
51/59
Logic Gate using Perceptron
Perceptron을이용한논리회로구현- XOR gate
•
적정한weight (w1, w2), bias (b)를지정해야함
•
XOR gate는단층perceptron을이용해구현할수없음
1
ଵ
Σ
s > 0 ?
ଶ
ଵ
ଶ
?
?
?
 𝑦
 𝑥ଶ
𝑥ଵ
0
0
0
1
1
0
1
0
1
0
1
1
(1,1)
(1,0)
(0,0)
(0,1)
?


| 𝑥 ଵ | 𝑥 ଶ | 𝑦 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |


### [Page 52]
52/59
Contents
1. Introduction
2. Applications
①Regression
②Classification
③Logic gate
3. Multi-layer Perceptron (MLP)



### [Page 53]
53/59
Multi-Layer Perceptron : 여러개의층을가진Perceptron XOR Gate 구현가능
Multi-Layer Perceptron
<Multi-Layer Perceptron>
𝑦
𝑥ଵ
𝑥ଶ
(1,1)
(1,0)
(0,0)
(0,1)
 𝑦
 𝑥ଶ
𝑥ଵ
0
0
0
1
1
0
1
0
1
0
1
1


| 𝑥 ଵ | 𝑥 ଶ | 𝑦 |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |


### [Page 54]
54/59
XOR Gate
•
NAND Gate, OR Gate, AND Gate 서로연결하여XOR Gate 구현가능
Multi-Layer Perceptron
NAND
OR
AND
𝑦
𝑠ଶ
𝑠ଵ
 𝑥ଶ
𝑥ଵ
0
0
1
0
0
1
1
1
1
0
1
1
1
0
1
0
1
0
1
1

ଵ: ଵNAND
ଶ

ଶ: ଵOR
ଶ

: ଵAND
ଶ
𝑦


| 𝑥 ଵ | 𝑥 ଶ | 𝑠 ଵ | 𝑠 ଶ | 𝑦 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0 | 0 |
| 0 | 1 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 | 1 |
| 1 | 1 | 0 | 1 | 0 |


### [Page 55]
55/59
XOR Gate
•
NAND Gate, OR Gate, AND Gate 서로연결하여XOR Gate 구현가능
•
Perceptron 에서층을쌓아서로연결하여XOR Gate 구현가능
Multi-Layer Perceptron
NAND
OR
AND
𝑦
1
𝑥ଵ
𝑠ଵ
<Multi-Layer Perceptron>
𝑥ଶ
1
𝑠ଶ
𝑠ଷ
𝑦
𝒃𝟏𝟎
𝒘𝟏𝟎
𝒘𝟏𝟏
𝒃𝟎𝟎
𝒃𝟎𝟏
𝒘𝟎𝟎
𝒘𝟎𝟏
𝒘𝟎𝟑
𝒘𝟎𝟒



### [Page 56]
56/59
XOR Gate
•
NAND Gate, OR Gate, AND Gate 서로연결하여XOR Gate 구현가능
•
Perceptron 에서층을쌓아서로연결하여XOR Gate 구현가능
Multi-Layer Perceptron
NAND
OR
AND
𝑦
1
𝑥ଵ
𝑠ଵ
<Multi-Layer Perceptron>
𝑥ଶ
1
𝑠ଶ
𝑠ଷ
𝑦
𝒃𝟏𝟎
𝒘𝟏𝟎
𝒘𝟏𝟏
𝒃𝟎𝟎
𝒃𝟎𝟏
𝒘𝟎𝟎
𝒘𝟎𝟏
𝒘𝟎𝟑
𝒘𝟎𝟒
NAND
OR
AND



### [Page 57]
57/59
XOR Gate
Multi-Layer Perceptron
1
𝑥ଵ
𝑠ଵ
<Multi-Layer Perceptron>
𝑥ଶ
1
𝑠ଶ
𝑠ଷ
𝑦
𝒃𝟏𝟎
𝒘𝟏𝟎
𝒘𝟏𝟏
𝒃𝟎𝟎
𝒃𝟎𝟏
𝒘𝟎𝟎
𝒘𝟎𝟏
𝒘𝟎𝟐
𝒘𝟎𝟑
NAND
OR
AND



### [Page 58]
58/59
XOR Gate
•
𝑤଴଴= -0.5, 𝑤଴ଶ= -0.5, 𝑏଴଴= -0.7 (NAND Gate 구현)
•
S1 : NAND Gate
Multi-Layer Perceptron
1
𝑥ଵ
𝑠ଵ
<Multi-Layer Perceptron>
𝑥ଶ
1
𝑠ଶ
𝑠ଷ
𝑦
𝒘𝟏𝟎
𝒘𝟏𝟏
𝒃𝟎𝟎
𝒃𝟎𝟏
𝒘𝟎𝟎
𝒘𝟎𝟏
𝒘𝟎𝟐
𝒘𝟎𝟑
NAND
OR
AND
𝑥ଵ
𝑥ଶ
(1,1)
(1,0)
(0,0)
(0,1)
NAND (𝑠ଵ)
𝒃𝟏𝟎
-0.5
-0.5
0.7



### [Page 59]
59/59
XOR Gate
•
𝑤଴ଵ= 0.5, 𝑤଴ଷ= 0.5, 𝑏଴ଵ= -0.2 (OR Gate 구현)
•
S2 : OR Gate
Multi-Layer Perceptron
𝑥ଵ
𝑥ଶ
(1,1)
(1,0)
(0,0)
(0,1)
𝑥ଵ
(1,1)
(1,0)
(0,0)
(0,1)
𝑥ଶ
1
𝑥ଵ
𝑠ଵ
<Multi-Layer Perceptron>
𝑥ଶ
1
𝑠ଶ
𝑠ଷ
𝑦
𝒘𝟏𝟎
𝒘𝟏𝟏
𝒃𝟎𝟎
𝒃𝟎𝟏
𝒘𝟎𝟎
𝒘𝟎𝟏
𝒘𝟎𝟐
𝒘𝟎𝟑
NAND
OR
AND
𝒃𝟏𝟎
NAND (𝑠ଵ)
OR (𝑠ଶ)
0.5
0.5
-0.2



### [Page 60]
60/59
XOR Gate
•
𝑤ଵ଴= 0.5, 𝑤ଵଵ= 0.5, 𝑏ଵ଴= -0.7 (AND Gate 구현) 
•
S1 AND S2 : S3 (XOR Gate)
Multi-Layer Perceptron
𝑥ଵ
𝑥ଶ
(1,1)
(1,0)
(0,0)
(0,1)
𝑥ଵ
𝑥ଶ
(1,1)
(1,0)
(0,0)
(0,1)
1
𝑥ଵ
𝑠ଵ
<Multi-Layer Perceptron>
𝑥ଶ
1
𝑠ଶ
𝑠ଷ
𝑦
𝒘𝟏𝟎
𝒘𝟏𝟏
𝒃𝟎𝟎
𝒃𝟎𝟏
𝒘𝟎𝟎
𝒘𝟎𝟏
𝒘𝟎𝟐
𝒘𝟎𝟑
NAND
OR
AND
𝒃𝟏𝟎
NAND (𝑠ଵ)
AND
𝑥ଵ
𝑥ଶ
(1,1)
(1,0)
(0,0)
(0,1)
OR (𝑠ଶ)
XOR (𝑠ଷ)
0.5
0.5
-0.7
두개의선형분류기를AND시켜
두개의분류선구현



### [Page 61]
61/59
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Dept. of AI
Dong-A University, Busan, Rep. of Korea

