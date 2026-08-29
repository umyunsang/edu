### [Page 1]
1/49
Dong-A Univ. (ISPL)
컴퓨터AI공학부
2024년1학기인공지능



### [Page 2]
2/49
Contents
1.
Vanishing Gradient 현상이란? 
2.
Vanishing Gradient 원인
3.
Vanishing Gradient 해결법



### [Page 3]
3/49
Vanishing Gradient
기울기소실(Vanishing gradient) 현상이란?
•
일반적으로신경망을깊게설계할수록더좋은성능을기대함
Deep Neural Network
Neural Network



### [Page 4]
4/49
Vanishing Gradient
기울기소실(Vanishing gradient) 현상이란?
•
일반적으로신경망을깊게설계할수록더좋은성능을기대함
•
하지만, 깊은신경망설계시Vanishing gradient 현상발생
Deep Neural Network
Neural Network
Problem!
𝒘𝒕ା𝟏←𝒘𝒕−𝝁𝛁𝑳(𝒘𝒕),   
학습률(learning rate, step size)
탐색방향
𝒘𝒉𝒆𝒓𝒆  𝛁𝑳𝒘𝒕∆𝒘< 𝟎



### [Page 5]
5/49
기울기소실(Vanishing gradient) 현상이란?
•
Backpropagation 과정시입력층에가까워질수록기울기값이점점0으로수렴
Vanishing Gradient
Deep Neural Network
Vanishing Gradient



### [Page 6]
6/49
기울기소실(Vanishing gradient) 현상이란?
•
Backpropagation 과정시입력층에가까워질수록기울기값이점점0으로수렴기울기값소실
Vanishing Gradient
Deep Neural Network
Vanishing Gradient
기울기값소실



### [Page 7]
7/49
기울기소실(Vanishing gradient) 현상이란?
•
Backpropagation 과정시입력층에가까워질수록기울기값이점점0으로수렴
•
소실된기울기값에의해신경망의가중치를업데이트하는데문제발생
Vanishing Gradient
Deep Neural Network
Vanishing Gradient
*
L
W
W
W





가중치업데이트수식
Learning rate
Gradient



### [Page 8]
8/49
기울기소실(Vanishing gradient) 현상이란?
•
Backpropagation 과정시입력층에가까워질수록기울기값이점점0으로수렴
•
소실된기울기값에의해신경망의가중치를업데이트하는데문제발생최적의모델을찾을수없음
Vanishing Gradient
Deep Neural Network
Vanishing Gradient
*
L
W
W
W





가중치업데이트수식
Learning rate
Gradient



### [Page 9]
9/49
Contents
1.
Vanishing Gradient 현상이란?
2.
Vanishing Gradient 원인
3.
Vanishing Gradient 해결법



### [Page 10]
10/49
기울기소실(Vanishing gradient) 원인
•
(1) 활성화함수의미분값의최대치가1보다작은경우
•
(2) 가중치초기화가제대로되지않은경우
Vanishing Gradient



### [Page 11]
11/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
활성화함수의미분값의최대치가1보다작은경우Sigmoid, tanh (Hyperbolic tangent)
Vanishing Gradient



### [Page 12]
12/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
활성화함수의미분값의최대치가1보다작은경우Sigmoid, tanh (Hyperbolic tangent)
•
Sigmoid 함수의경우미분값의최대치가0.25이므로Vanishing gradient 현상발생
Vanishing Gradient
함수값
미분값



### [Page 13]
13/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
활성화함수의미분값의최대치가1보다작은경우Sigmoid, tanh (Hyperbolic tangent)
•
Sigmoid 함수의경우미분값의최대치가0.25이므로Vanishing gradient 현상발생
•
Tanh 함수의경우미분값이여전히1보다작은값을가지므로Vanishing gradient 현상발생
Vanishing Gradient
함수값
미분값
-
-
-
-



### [Page 14]
14/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
Example
Vanishing Gradient
𝑥: Input
𝑦: Output
𝑤௜: 가중치
𝑧௜: 은닉층의입력값
𝑎௜: 은닉층의출력값
𝜎: Sigmoid function
𝑥
×
𝑤ଵ
𝑧ଵ
𝜎
𝑎ଵ
𝑤ଶ
×
𝑧ଶ
𝜎
𝑎ଶ
𝑤ଷ
×
𝑧ଷ
𝜎
𝑎ଷ
𝑤ସ
𝑦
×



### [Page 15]
15/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
Example
Vanishing Gradient
𝑥: Input
𝑦: Output
𝑤௜: 가중치
𝑧௜: 은닉층의입력값
𝑎௜: 은닉층의출력값
𝜎: Sigmoid function
𝑥
×
𝑤ଵ
𝑧ଵ
𝜎
𝑎ଵ
𝑤ଶ
×
𝑧ଶ
𝜎
𝑎ଶ
𝑤ଷ
×
𝑧ଷ
𝜎
𝑎ଷ
𝑤ସ
𝑦
×
Backpropagation
4
4
y
y
y
w
y
w











### [Page 16]
16/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
Example
Vanishing Gradient
𝑥: Input
𝑦: Output
𝑤௜: 가중치
𝑧௜: 은닉층의입력값
𝑎௜: 은닉층의출력값
𝜎: Sigmoid function
𝑥
×
𝑤ଵ
𝑧ଵ
𝜎
𝑎ଵ
𝑤ଶ
×
𝑧ଶ
𝜎
𝑎ଶ
𝑤ଷ
×
𝑧ଷ
𝜎
𝑎ଷ
𝑤ସ
𝑦
×
Backpropagation
3
3
3
3
3
2
a
z
y
y
y
w
y
a
z
a














Sigmoid function의미분값< 1



### [Page 17]
17/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
Example
Vanishing Gradient
𝑥: Input
𝑦: Output
𝑤௜: 가중치
𝑧௜: 은닉층의입력값
𝑎௜: 은닉층의출력값
𝜎: Sigmoid function
𝑥
×
𝑤ଵ
𝑧ଵ
𝜎
𝑎ଵ
𝑤ଶ
×
𝑧ଶ
𝜎
𝑎ଶ
𝑤ଷ
×
𝑧ଷ
𝜎
𝑎ଷ
𝑤ସ
𝑦
×
Backpropagation
3
3
2
2
1
1
1
3
3
2
2
1
1
1
a
z
a
z
a
z
y
y
y
w
y
a
z
a
z
a
z
w


























Sigmoid function의미분값< 1



### [Page 18]
18/49
기울기소실(Vanishing gradient) 원인: 활성화함수(Activation function)
•
Example
•
가중치의기울기를구할때이전층들의미분값이사용
•
결론적으로, Backpropagation 시입력층에가까워질수록활성화함수의미분값에의해기울기소실현상발생
Vanishing Gradient
𝑥: Input
𝑦: Output
𝑤௜: 가중치
𝑧௜: 은닉층의입력값
𝑎௜: 은닉층의출력값
𝜎: Sigmoid function
𝑥
×
𝑤ଵ
𝑧ଵ
𝜎
𝑎ଵ
𝑤ଶ
×
𝑧ଶ
𝜎
𝑎ଶ
𝑤ଷ
×
𝑧ଷ
𝜎
𝑎ଷ
𝑤ସ
𝑦
×
Backpropagation
3
3
2
2
1
1
1
3
3
2
2
1
1
1
a
z
a
z
a
z
y
y
y
w
y
a
z
a
z
a
z
w





























### [Page 19]
19/49
기울기소실(Vanishing gradient) 원인: 가중치초기화(Weight initialization)
•
가중치를모두0으로초기화한경우
•
가중치를평균이0, 표준편차가1인정규분포로초기화한경우
•
가중치를평균이0, 표준편차가0.01인정규분포로초기화한경우
Vanishing Gradient



### [Page 20]
20/49
기울기소실(Vanishing gradient) 원인: 가중치초기화(Weight initialization)
•
가중치를모두0으로초기화한경우
Backpropagation시곱셈과정에서문제신경망학습안됨
Vanishing Gradient



### [Page 21]
21/49
기울기소실(Vanishing gradient) 원인: 가중치초기화(Weight initialization)
•
가중치를모두0으로초기화한경우
•
가중치를평균이0, 표준편차가1인정규분포로초기화한경우
출력값들이-1과1로치우침Vanishing gradient 문제발생
Vanishing Gradient
각층의활성화값분포
Activation Function: tanh
𝑦= tanh (𝑊× 𝑥+ 𝑏)
표준정규분포
Weight 값이커짐으로인해출력값증가
출력값
미분값



### [Page 22]
22/49
기울기소실(Vanishing gradient) 원인: 가중치초기화(Weight initialization)
•
가중치를모두0으로초기화한경우
•
가중치를평균이0, 표준편차가1인정규분포로초기화한경우
•
가중치를평균이0, 표준편차가0.01인정규분포로초기화한경우
Vanishing gradient는해결되지만출력값이0으로치우치는현상발생신경망학습안됨
Vanishing Gradient
각층의활성화값분포
Activation Function: tanh
𝑦= tanh (𝑊× 𝑥+ 𝑏)
표준정규분포
Weight 값이작아짐으로인해출력값감소
출력값
미분값
0.01



### [Page 23]
23/49
Contents
1.
Vanishing Gradient 현상이란? 
2.
Vanishing Gradient 원인
3.
Vanishing Gradient 해결법



### [Page 24]
24/49
기울기소실(Vanishing gradient) 해결법
•
활성화함수변경
•
가중치초기화설정
Vanishing Gradient



### [Page 25]
25/49
기울기소실(Vanishing gradient) 해결법: 활성화함수변경
•
Sigmoid, tanh ReLU, PReLU, Leaky ReLU, etc…
•
미분값의최대치가1보다큰활성화함수를사용하여기울기소실문제해결가능
Vanishing Gradient
함수값
미분값
미분값
함수값



### [Page 26]
26/49
기울기소실(Vanishing gradient) 해결법: 가중치초기화(Weight initialization)
•
Xavier, He initialization 등을사용하여가중치값을초기화입력Node 개수가wight를초기화하는데영향을줌
Vanishing Gradient



### [Page 27]
27/49
기울기소실(Vanishing gradient) 해결법: 가중치초기화(Weight initialization)
•
Xavier, He initialization 등을사용하여가중치값을초기화입력Node 개수가wight를초기화하는데영향을줌
Vanishing Gradient
입력노드가2개인경우weight 예시)
입력노드가3개인경우weight 예시)
1
1
1
1
1
0.5
0.5
input
output
input
output
𝑤1
𝑥1
𝑥2
𝑥1
𝑥2
𝑥3
𝑦
𝑦
𝑤2
𝑤1
𝑤2
𝑤3



### [Page 28]
28/49
기울기소실(Vanishing gradient) 해결법: 가중치초기화(Weight initialization)
•
Xavier, He initialization 등을사용하여가중치값을초기화입력node 개수에따라weight를초기화해줘야함
Vanishing Gradient
입력노드가2개인경우weight 예시)
입력노드가3개인경우weight 예시)
1
1
1
1
1
0.5
0.5
input
output
input
output
0.25
0.25
𝑦 = 𝑥1 ∗𝑤1 + (𝑥2 ∗𝑤2)
0.5 = 1 ∗0.25 + (1 ∗0.25)
𝑥1
𝑥2
𝑥1
𝑥2
𝑥3
𝑦
𝑦
𝑦 = 𝑥1 ∗𝑤1 + 𝑥2 ∗𝑤2 + (𝑥3 ∗𝑤3)
0.5 = 1 ∗0.166 + 1 ∗0.166 + (1 ∗0.166)
0.166..
0.166..
0.166..



### [Page 29]
29/49
기울기소실(Vanishing gradient) 해결법: 가중치초기화(Weight initialization)
•
Xavier, He initialization 등을사용하여가중치값을초기화
Xavier initialization: 이전층과다음층의노드의수를반영
Vanishing Gradient
2
(
)
in
out
Var W
n
n



Xavier Normal Initialization
𝑛௜௡:이전layer (input)의노드의수
𝑛௢௨௧: 다음layer의노드의수
~
(0,
(
))
W
N
Var W
Activation Function: tanh



### [Page 30]
30/49
기울기소실(Vanishing gradient) 해결법: 가중치초기화(Weight initialization)
•
Xavier, He initialization 등을사용하여가중치값을초기화
Xavier initialization: 이전층과다음층의노드의수를반영ReLU 활성화함수를사용시문제발생
Vanishing Gradient
2
(
)
in
out
Var W
n
n



Xavier Normal Initialization
𝑛௜௡:이전layer (input)의노드의수
𝑛௢௨௧: 다음layer의노드의수
~
(0,
(
))
W
N
Var W
Activation Function: ReLU



### [Page 31]
31/49
기울기소실(Vanishing gradient) 해결법: 가중치초기화(Weight initialization)
•
Xavier, He initialization 등을사용하여가중치값을초기화
Xavier initialization: 이전층과다음층의노드의수를반영
He initialization: 이전층의노드수만반영ReLU함수에적합한초기화방법
Vanishing Gradient
𝑛௜௡:이전layer (input)의노드의수
~
(0,
(
))
W
N
Var W

He Normal Initialization
2
(
)
in
Var W
n




### [Page 32]
32/12
<실습>
- 오버피팅-



### [Page 33]
33/49
Overfitting 문제실습

Overfitting 이주로일어나는경우

매개변수가많은표현력이높은모델

훈련데이터가적음
…
…
Activation function
0y
1y
9y
…
…
…
…
…






### [Page 34]
34/49
Vanishing Gradient 문제실습

Vanishing Gradient 문제확인: 깊은신경망설계
…
…
Activation function
0y
1y
9y
…
…
Vanishing Gradient 실습

Layer1 (Input: 784, Out: 100)

Layer2 (Input: 100, Out: 100)

Layer3 (Input: 100, Out: 100)

Layer4 (Input: 100, Out: 100)

Layer5 (Input: 100, Out: 10)

Activation function: Sigmoid

Loss function: Cross Entropy
…
…
…






### [Page 35]
35/49

Vanishing Gradient 문제확인: 깊은신경망설계
•
(1) MLP 모델정의
Vanishing Gradient 문제실습
Network 이름변경
Layer 5개추가(Input, Output node 개수주의)



### [Page 36]
36/49

Vanishing Gradient 문제확인: 깊은신경망설계
•
(2) Hyper-parameter 지정(2-layer 환경이랑동일)
Vanishing Gradient 문제실습
Network 이름변경주의



### [Page 37]
37/49

Vanishing Gradient 문제확인: 깊은신경망설계
•
(3) MLP 학습을위한반복문선언(2-layer 환경이랑동일)
Vanishing Gradient 문제실습



### [Page 38]
38/49

Vanishing Gradient 문제확인: 깊은신경망설계
•
(3) MLP 학습을위한반복문선언(2-layer 환경이랑동일) 결과확인
Vanishing Gradient 문제실습
Vanishing Gradient 문제발생!



### [Page 39]
39/49

Vanishing Gradient 문제확인: 깊은신경망설계
•
(4) 학습완료된Weight parameter 저장및확인
•
(5) MNIST Test dataset 분류성능확인
Vanishing Gradient 문제실습
4
5
정답률: 11.3%



### [Page 40]
40/49

Vanishing Gradient 문제확인: 깊은신경망설계
•
(6) 예측결과값및정답이미지확인
Vanishing Gradient 문제실습
1로예측하였지만실제결과는7



### [Page 41]
41/49

Vanishing Gradient 문제해결(1): 활성화함수변경
•
(1) MLP 모델재정의
•
Sigmoid 함수ReLU 함수
Vanishing Gradient 문제실습
ReLU 함수로변경



### [Page 42]
42/49

Vanishing Gradient 문제해결(1): 활성화함수변경
•
(2) Hyper-parameter 지정및Training 진행
Vanishing Gradient 문제실습



### [Page 43]
43/49

Vanishing Gradient 문제해결(1): 활성화함수변경
•
(3) Network Training 결과확인
Vanishing Gradient 문제실습
Vanishing Gradient 문제해결



### [Page 44]
44/49

Vanishing Gradient 문제해결(1): 활성화함수변경
•
(4) Weight parameter 저장및불러오기
•
(5) MNIST Test dataset 분류성능확인
Vanishing Gradient 문제실습
4
5
정답률: 97.1%



### [Page 45]
45/49

Vanishing Gradient 문제해결(2): 가중치초기화사용
•
(1) MLP 모델재정의
•
Sigmoid 함수사용
•
Xavier normal initialization 사용
Vanishing Gradient 문제실습
Sigmoid 함수정의
각Layer마다Xavier 초기화적용



### [Page 46]
46/49

Vanishing Gradient 문제해결(2): 가중치초기화사용
•
(2) Hyper-parameter 지정및Training 진행
Vanishing Gradient 문제실습



### [Page 47]
47/49

Vanishing Gradient 문제해결(2): 가중치초기화사용
•
(3) Network Training 결과확인
Vanishing Gradient 문제실습
Vanishing Gradient 문제해결



### [Page 48]
48/49

Vanishing Gradient 문제해결(2): 가중치초기화사용
•
(4) Weight parameter 저장및불러오기
•
(5) MNIST Test dataset 분류성능확인
Vanishing Gradient 문제실습
4
5
정답률: 88.5%



### [Page 49]
49/49
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Division of Computer〮AI Engineering
Dong-A University, Busan, Rep. of Korea

