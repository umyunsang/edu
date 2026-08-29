### [Page 1]
1/12
Dong-A Univ. (ISPL)
컴퓨터AI공학부
2024년1학기인공지능


|  |  |
| --- | --- |
|  |  |
|  | 2 1/12 |


### [Page 2]
2/12
Overfitting
오버피팅(Overfitting): 학습데이터를과하게학습하여그외의데이터에는대응하지못하는상태
오버피팅이주로일어나는경우
매개변수가많은모델
학습데이터가적음
학습데이터
시험데이터
예측값
Underfitting
Adequate
Overfitting



### [Page 3]
3/12
Overfitting
해결방법1 데이터확보



### [Page 4]
4/12
Overfitting
해결방법데이터증식(Data Augmentation)

입력이미지(학습이미지)를‘인위적’으로확장

회전(Rotation), 이동(Move), 자르기(Crop), 대칭(Symmetry) 등



### [Page 5]
5/12
Overfitting
해결방법2 조기종료(Early Stopping)

Epoch, Iteration을많이돌린후, 특정시점에서멈추는것
Loss
Iteration
Early Stopping 
Point
학습데이터
시험데이터



### [Page 6]
6/12
Overfitting
해결방법3 L1, L2 정규화(L1, L2 Regularization)
𝑾∗
𝑾𝟏
1. Optimization을통해Training dataset에최적화된상태



### [Page 7]
7/12
Overfitting
해결방법L1, L2 정규화(L1, L2 Regularization)
𝑾∗
𝑾𝟏
𝑳𝟏
𝑳𝟐
𝑾𝟐
1. Optimization을통해Training dataset에최적화된상태
2. Loss Function을이용하여Optimization을못하게설정Loss Function Boundary 추가



### [Page 8]
8/12
Overfitting
해결방법L1, L2 정규화(L1, L2 Regularization)
𝑾∗
𝑾𝟏
𝑳𝟏
𝑳𝟐
𝑾𝟐
𝑳෨= 𝑳𝒚, 𝒚ෝ+ 𝝀𝛀𝒘,𝝀≥𝟎
𝑳𝟏: 𝛀𝒘= ෍
|𝒘𝒊|
𝒏
𝒊ୀ𝟏
𝑳𝟐: 𝛀𝒘= ෍
𝒘𝒊
𝟐
𝒏
𝒊ୀ𝟏
1. Optimization을통해Training dataset에최적화된상태
2. Loss Function을이용하여Optimization을못하게설정Loss Function Boundary 추가
3. Loss Function뒤에정규화Term(Loss Function Boundary)을추가



### [Page 9]
9/12
Overfitting
해결방법L1, L2 정규화(L1, L2 Regularization)
𝑾∗
𝑾𝟏
𝑳𝟏
𝑳𝟐
𝑾𝟐
𝑽∗
𝑳෨= 𝑳𝒚, 𝒚ෝ+ 𝝀𝛀𝒘,𝝀≥𝟎
𝑳𝟏: 𝛀𝒘= ෍
|𝒘𝒊|
𝒏
𝒊ୀ𝟏
𝑳𝟐: 𝛀𝒘= ෍
𝒘𝒊
𝟐
𝒏
𝒊ୀ𝟏
1. Optimization을통해Training dataset에최적화된상태
4. Loss Function Boundary 내에서Weight가존재
2. Loss Function을이용하여Optimization을못하게설정Loss Function Boundary 추가
3. Loss Function뒤에정규화Term(Loss Function Boundary)을추가



### [Page 10]
10/12
Overfitting
해결방법4 드롭아웃(Dropout)
Backpropagation시모든Weight가업데이트되는것을방지
일부Node를랜덤하게제거
Dropout 적용전
x
x
x
Dropout 적용후
x
x



### [Page 11]
11/12
Overfitting
해결방법5 드롭커넥트(Drop Connect)
Backpropagation시모든Weight가업데이트되는것을방지
일부Weight를랜덤하게제거
Drop Connect 적용전
Drop Connect 적용후



### [Page 12]
12/12
Overfitting
해결방법6 배치정규화(Batch Normalization)
출력값을정규화하는작업
Fully 
Connected
ReLU
Fully 
Connected
ReLU
Fully 
Connected
ReLU
Fully 
Connected
Batch Norm
ReLU
Fully 
Connected
Batch Norm
ReLU
Fully 
Connected
Batch Norm
ReLU
학습과정에서계층별로입력의데이터분포가달라지는현상을
내부공변량변화(Internal Covariate Shift)라고함
학습과정에서배치별로평균과분산을이용해정규화하는계층을
배치정규화계층이라함
<배치정규화적용전>
<배치정규화적용후>
Ex)
Ex)



### [Page 13]
13/12
실습



### [Page 14]
14/12
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



### [Page 15]
15/12

Overfitting 문제확인
•
(1) MNIST train dataset 개수변경60000 300
Overfitting 문제실습
MNIST train dataset
img0
5
0
4
img1
img2
img
59999
8
Image
Label
…
…
MNIST train dataset
img0
5
0
4
img1
img2
img
300
6
Image
Label
…
…




### [Page 16]
16/12

Overfitting 문제확인
•
(1) MNIST train dataset 개수변경60000 300
Overfitting 문제실습
Image 데이터개수300으로설정
Label 데이터개수300으로설정
데이터개수확인



### [Page 17]
17/12

Overfitting 문제확인
•
(2) MLP 모델정의
Overfitting 문제실습



### [Page 18]
18/12

Overfitting 문제확인
•
(3) Hyper-parameter 지정
Overfitting 문제실습
Epoch 100 설정
Network 이름확인



### [Page 19]
19/12

Overfitting 문제확인
•
(4) MLP 학습을위한반복문선언
Overfitting 문제실습



### [Page 20]
20/12

Overfitting 문제확인
•
(4) MLP 학습을위한반복문선언
Overfitting 문제실습



### [Page 21]
21/12

Overfitting 문제확인
•
(5) MNIST Train dataset, MNIST Test dataset 분류성능확인
Overfitting 문제실습
Train Data 적용
정답률: 100%
정답률: 76.3%



### [Page 22]
22/12

Overfitting 문제확인
•
(5) MNIST Train dataset, MNIST Test dataset 분류성능확인
Overfitting 문제실습
Overfitting 문제발생!
학습데이터
시험데이터
예측값
Overfitting
학습데이터정답률: 100%
시험데이터정답률: 76.3%



### [Page 23]
23/12

Overfitting 문제해결(1): Batch Normalization
•
(1) MLP 모델재정의
•
Batch Normalization 선언및적용
Overfitting 문제실습
Batch Normalization 수행선언(100 features)
Batch Normalization 적용
Fully 
Connected
Batch Norm
ReLU



### [Page 24]
24/12

Overfitting 문제해결(1): Batch Normalization
•
(2) Hyper-parameter 지정및Training 진행
Overfitting 문제실습



### [Page 25]
25/12

Overfitting 문제해결(1): Batch Normalization
•
(3) MNIST Test dataset 분류성능확인
Overfitting 문제실습
정답률: 79.3%



### [Page 26]
26/12

Overfitting 문제해결(2): Dropout
•
(1) MLP 모델재정의
•
Dropout 선언및적용
Overfitting 문제실습
Dropout 선언(0.2 비율)
Dropout 적용



### [Page 27]
27/12

Overfitting 문제해결(2): Dropout
•
(2) Hyper-parameter 지정및Training 진행
Overfitting 문제실습



### [Page 28]
28/12

Overfitting 문제해결(2): Dropout
•
(3) MNIST Test dataset 분류성능확인
Overfitting 문제실습
정답률: 79.3%
Network Dropout 비활성



### [Page 29]
29/12

Overfitting 문제해결(3): Data Augmentation
•
(1) MNIST Train Data Rotation 변환
Overfitting 문제실습
반시계방향15도Rotation 수행
반시계방향30도Rotation 수행
시계방향15도Rotation 수행
시계방향30도Rotation 수행



### [Page 30]
30/12

Overfitting 문제해결(3): Data Augmentation
•
(2) rotation_data 형태확인
•
(3) rotation 수행된데이터MNIST Train Dataset 에합치기
Overfitting 문제실습
2
3



### [Page 31]
31/12

Overfitting 문제해결(3): Data Augmentation
•
(4) Train Dataset label 늘리기
Overfitting 문제실습
같은Tensor 데이터5배증가



### [Page 32]
32/12

Overfitting 문제해결(3): Data Augmentation
•
300 Train Dataset 1500 Train Dataset
Overfitting 문제실습
MNIST train dataset
img0
5
0
4
img1
img2
img
300
5
Image
Label
…
…
…
…
img
600
5
img
900
5
…
…
…
…
img
1200
5
…
…
img
1499
6



### [Page 33]
33/12

Overfitting 문제해결(3): Data Augmentation
•
300 Train Dataset 1500 Train Dataset
Overfitting 문제실습
MNIST train dataset
img0
5
0
4
img1
img2
img
300
5
Image
Label
…
…
…
…
img
600
5
img
900
5
…
…
…
…
img
1200
5
…
…
img
1499
6



### [Page 34]
34/12

Overfitting 문제해결(3): Data Augmentation
•
(5) MLP 모델재정의
Overfitting 문제실습



### [Page 35]
35/12

Overfitting 문제해결(3): Data Augmentation
•
(6) Hyper-parameter 지정및Training 진행
Overfitting 문제실습



### [Page 36]
36/12

Overfitting 문제해결(3): Data Augmentation
•
(7) MNIST Test dataset 분류성능확인
Overfitting 문제실습
정답률: 80.73%



### [Page 37]
37/17
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Division of Computer〮AI Engineering
Dong-A University, Busan, Rep. of Korea

