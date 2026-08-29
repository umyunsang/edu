### [Page 1]
1/23
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능


|  |  |
| --- | --- |
|  |  |
|  | 1/23 |


### [Page 2]
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



### [Page 3]
3/23
Multi Layer Perceptron
다층퍼셉트론(Multi Layer Perceptron, MLP)
•
여러개의층으로구성된perceptron 모델
•
입력층, 은닉층, 출력층으로구성됨
Perceptron 구조예시
Multi Layer Perceptron (MLP) 예시



### [Page 4]
4/23
Multi Layer Perceptron
다층퍼셉트론(Multi Layer Perceptron, MLP)
•
각입력(x)에대응되는weight(w), 1개의노드에입력되는bias(b) 존재
•
가중합으로얻어진결과치에활성화함수(h) 적용
…

( )
h 
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
1 1
2
2
10
10
(
...
)
y
h w x
w x
w x
b







1
2
10
1
2
10
,
...
...
T
w
w
W
w
W
w
w
w













1개노드계산예시
1
2
10
...
x
x
X
x











்
𝑋
= Input Data
𝑊்
= Weight
𝑏 
= Bias
ℎ
= Activation Function



### [Page 5]
5/23
Multi Layer Perceptron
다층퍼셉트론(Multi Layer Perceptron, MLP)
•
각입력(x)에대응되는weight(w), 1개의노드에입력되는bias(b) 존재
•
가중합으로얻어진결과치에활성화함수(h) 적용
…
1x
2x
3x
10
x
1,1
w
1
1y
1
1,1 1
2,1
2
10,1 10
1
2
1,2
1
2,2
2
10,2
10
2
(
...
)
(
...
)
y
h w x
w x
w
x
b
y
h w x
w
x
w
x
b










1개노드계산예시
2y
1,2
w
2,1
w
3,1
w
10,1
w
2,2
w
3,2
w
10,2
w
1,1
1,2
2,1
2,2
10,1
10,2
1,1
2,1
10,1
1,2
2,2
10,2
,
...
...
...
...
T
w
w
w
w
W
w
w
w
w
w
W
w
w
w


















1b
2b
1
2
10
...
x
x
X
x













( )
h 

( )
h 
1-layer perceptron 계산்
𝑋
= Input Data
𝑊்
= Weight
𝑏 
= Bias
ℎ
= Activation Function



### [Page 6]
6/23
Multi Layer Perceptron
다층퍼셉트론(Multi Layer Perceptron, MLP)
•
각입력(x)에대응되는weight(w), 1개의노드에입력되는bias(b) 존재
•
가중합으로얻어진결과치에활성화함수(h) 적용
1
1,1
2,1
10,1
2
1
1,2
2,2
10,2
2
10
...
...
...
x
w
w
w
x
b
h
w
w
w
b
x













































1,1 1
2,1
2
10,1 10
1
1
2
1,2
1
2,2
2
10,2
10
2
...
...
h w x
w x
w
x
b
y
y
h w x
w
x
w
x
b























2 x 10
10 x 1
2 x 1
1-layer perceptron 계산்
𝑋
= Input Data
𝑊்
= Weight
𝑏 
= Bias
ℎ
= Activation Function



### [Page 7]
7/23
Multi Layer Perceptron
다층퍼셉트론(Multi Layer Perceptron, MLP)
•
각입력(x)에대응되는weight(w), 1개의노드에입력되는bias(b) 존재
•
가중합으로얻어진결과치에활성화함수(h) 적용
…
1x
2x
3x
N
x
1
…
X
𝑾𝟏, 𝒃𝟏
𝑾𝟐, 𝒃𝟐
2y
1y

( )
h 

( )
h 

( )
h 

( )
h 
1

( )
h 

( )
h 
Y
𝑌= ℎ(𝑊ଶ்ℎ(𝑊ଵ்𝑋+ 𝒃𝟏) + 𝒃𝟐)
2-layer perceptron 계산
𝑋
= Input Data
𝑊்
= Weight
𝑏 
= Bias
ℎ
= Activation Function



### [Page 8]
8/23
Multi Layer Perceptron - Fully Connected Layer
전결합계층(Fully Connected layer, FC layer)
•
각층별로모든노드가연결되어weight를가지는계층을FC layer라고함
Fully connected layer



### [Page 9]
9/23
Multi Layer Perceptron - Fully Connected Layer
전결합계층(Fully Connected layer, FC layer)
•
각층별로모든노드가연결되어weight를가지는계층을FC layer라고함
Fully connected layer
Q1. 2-layer perceptron에서가지는weight (w) 개수?
Q2. 2-layer perceptron 가가지는bias (b) 개수?
Num. weight 300
Num. bias 30
Num. Input nodes = 10
Num. Hidden nodes = 10
Num. Output nodes = 20



### [Page 10]
10/23
Multi Layer Perceptron - Activation Function
활성화함수(Activation function)
•
입력값들의가중합을통해노드의활성화여부를판단하는함수
(
)
T
Y
h W X
b


Activation function
…

( )
h 
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



### [Page 11]
11/23
-10
-5
0
5
10
X
-1
-0.5
0
0.5
1
1.5
2
Step function
•
0 이하이면0, 0보다크면1을출력
•
역전파를통한학습불가능(미분불가)
•
이진분류시출력층에서사용하기적합함
Linear function
•
입력값을그대로출력으로내보내는함수
•
Regression (회귀)문제의출력층에서주로사용됨
Multi Layer Perceptron - Activation Function
0
0
0
1
x
y
x







Tanh function 출력결과
y
x

Linear function 출력결과
-10
-5
0
5
10
X
-10
-8
-6
-4
-2
0
2
4
6
8
10



### [Page 12]
12/23
1
1
x
x
e
y
e




-10
-5
0
5
10
X
-2
-1.5
-1
-0.5
0
0.5
1
1.5
2
Multi Layer Perceptron - Activation Function
Hypergolic tangent (tanh) function
•
-1~1 사이실수를출력
•
Sigmoid 함수보다출력의범위가큼
•
Sigmoid 함수보다최대기울기가큼
Tanh function 출력결과
-10
-5
0
5
10
X
-1
-0.5
0
0.5
1
1.5
2
1
( )
1
x
y
x
e




Sigmoid function
•
0~1 사이실수를출력확률로해석가능
•
기울기가발생하지않는지점이존재함
•
exp 연산의속도가느림
Sigmoid function 출력결과



### [Page 13]
13/23
Multi Layer Perceptron - Activation Function
Leaky ReLU function
•
0 이하일때0.1을곱한값이출력됨
•
ReLU 함수의음수구간기울기소실문제를보완
Leaky ReLU function 출력결과
Rectified Linear Unit (ReLU) function
•
0 이하일때0, 0 보다크면값을그대로출력
•
단순한연산으로학습속도가빠름
•
입력값이0이하인경우기울기는항상0
ReLU function 출력결과
-10
-5
0
5
10
X
-2
0
2
4
6
8
10
-10
-5
0
5
10
X
-2
0
2
4
6
8
10
max(0, )
y
x

max(0.1 , )
y
x x




### [Page 14]
14/23
Multi Layer Perceptron - Activation Function
Parametric ReLU (PReLU) function
•
Leaky ReLU와유사
•
음수구간기울기를결정하는p는학습가능한파라미터
Parametric ReLU function 출력결과
max(
, )
y
x x


Trainable parameter

( )
h 

( )
h 

( )
h 
Perceptron의PReLU 사용예시
1
0.2

2
0.05

3
0.1

-10
-5
0
5
10
X
-2
0
2
4
6
8
10



### [Page 15]
15/23
Multi Layer Perceptron - Activation Function
Softmax function
•
여러개의입력을받아각각의확률값으로출력
•
0~1 사이실수를출력하고, 모든출력의합은1 (확률로해석)
•
Multi-label classification 모델의출력층에주로사용됨
Softmax 함수사용예시(개/고양이/자동차분류)
…
1x
2x
N
x
1
…

( )
h 

( )
h 

( )
h 
1



<점수>
2
4
-1
(개)
(고양이)
(자동차)
1
i
j
x
N
x
j
e
y
e





### [Page 16]
16/23
Multi Layer Perceptron - Activation Function
Softmax function
•
여러개의입력을받아각각의확률값으로출력
•
0~1 사이실수를출력하고, 모든출력의합은1 (확률로해석)
•
Multi-label classification 모델의출력층에주로사용됨
Softmax 함수사용예시(개/고양이/자동차분류)
…
1x
2x
N
x
1
…

( )
h 

( )
h 

( )
h 
1



<점수>
2
4
-1
Softmax
(개)
(고양이)
(자동차)
1
i
j
x
N
x
j
e
e


<확률>
0.1185
0.8756
0.0059
(개)
(고양이)
(자동차)
1
i
j
x
N
x
j
e
y
e





### [Page 17]
17/23
Multi Layer Perceptron - Activation Function
Softmax function
•
여러개의입력을받아각각의확률값으로출력
•
0~1 사이실수를출력하고, 모든출력의합은1 (확률로해석)
•
Multi-label classification 모델의출력층에주로사용됨
2
4
-1
0.1185
0.8756
0.0059
1
i
j
x
N
x
j
e
y
e


1exp(
)
62.3551
N
j
j
x



exp( )
exp( )
exp( )
7.3891
54.5982
0.3679
62.3551

62.3551

62.3551

Softmax 함수계산예시
입력
결과(확률)



### [Page 18]
18/23
Multi Layer Perceptron - Activation Function
활성화함수를비선형함수로사용하는이유
•
문제의형상이비선형형태로되어있는경우
out
<Classification Problem>
<Regression Problem>
<Multi-Layer Perceptron>



### [Page 19]
19/23
Multi Layer Perceptron - Activation Function
활성화함수를비선형함수로사용하는이유
•
문제의형상이비선형형태로되어있는경우비선형함수를거쳐표현및해결가능
f  
f 
f 
out
<Multi-Layer Perceptron>
<Classification Problem>
< Regression Problem>



### [Page 20]
20/23
Multi Layer Perceptron – Loss Function

Loss function: 학습모델이얼마나잘못예측하고있는지는표현하는지표
•
값이낮을수록모델이정확하게예측했다고해석할수있음

평균제곱오차(Mean Squared Error, MSE)
2
1
1
( , ')
(
')
N
i
i
i
MSE y y
y
y
N





교차엔트로피오차(Cross Entropy Error, CEE)
1
( , ')
log(
')
N
i
i
i
CEE y y
y
y





평균절대오차(Mean Absolute Error, MAE)
1
1
( , ')
|
'|
N
i
i
i
MAE y y
y
y
N





y: 정답값

y’: 예측값
Input 𝑋
Model
𝑓(w)
Output 𝑦′
Loss 𝐿(𝑦, 𝑦′)
Update 𝑊
min 𝐿(𝑦, 𝑦′)
𝑤 = (𝑤ଵ, 𝑤ଶ, … , 𝑤௡, 𝑏ଵ,𝑏ଶ, …,𝑏௠)
𝑤: Weight
𝑏: Bias



### [Page 21]
21/23
Multi Layer Perceptron – Loss Function

Loss function: 학습모델이얼마나잘못예측하고있는지는표현하는지표
•
값이낮을수록모델이정확하게예측했다고해석할수있음
•
Ex. Cross Entropy Error (CEE) 계산방법
1
CEE( , ')
log(
')
N
i
i
i
y y
y
y




0
0
1
0
0
0
0
0
0
0
정답값(y, one-hot)
0
0
0.8
0
0
0
0.1
0
0.1
0
예측확률(y’)
CEE( , ')
(1 log(0.8))
0.2231
y y


CEE = 0.2231
0
1
2
3
4
5
6
7
8
9
Model A의예측결과

y: 정답값

y’: 예측값



### [Page 22]
22/23
Multi Layer Perceptron – Loss Function

Loss function: 학습모델이얼마나잘못예측하고있는지는표현하는지표
•
값이낮을수록모델이정확하게예측했다고해석할수있음
•
Ex. Cross Entropy Error (CEE) 계산방법
1
CEE( , ')
log(
')
N
i
i
i
y y
y
y





y: 정답값

y’: 예측값
0
0
1
0
0
0
0
0
0
0
정답값(y, one-hot)
0
0
0.8
0
0
0
0.1
0
0.1
0
예측확률(y’)
CEE = 0.2231
0
1
2
3
4
5
6
7
8
9
Model A의예측결과
CEE( , '')
(1 log(0.2))
1.6094
y y


0
0
0.2
0
0
0
0.2
0
0.6
0
예측확률(y’’)
CEE = 1.6094
Model B의예측결과



### [Page 23]
23/23
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Dept. of AI
Dong-A University, Busan, Rep. of Korea

