### [Page 1]
1/5
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능



### [Page 2]
2/5
CIFAR-10 dataset 형상
•
32x32x3 (RGB) 이미지, 10개의클래스
•
Train: 50,000개, Test: 10,000개



### [Page 3]
3/5

패키지선언

Dataset 선언CIFAR-10 dataset으로변경

Accuracy 측정코드



### [Page 4]
4/5
Requirement
Batch size: ~200
Learning rate: ~0.1
Learning rate decay
Activation function: ReLU
Loss function: Cross Entropy
Node: ~500
Layer: ~6
Epoch: ~20
Batch normalization
Drop-out
Weight initialization
Optimization



### [Page 5]
5/5
Backbone Network
Batch size: 200
Learning rate: 0.1
Learning rate decay : 10Epoch x0.1
Activation function: ReLU
Loss function: Cross Entropy
Node: 500
Layer: 6
Epoch: 20
Batch normalization: o (ALL)
Drop-out: o (0.1)
Weight initialization: o (ALL)
Optimization: SGD
55%



### [Page 6]
6/5
Code
1
2
1. CIFAR10 다운로드
2. 모델정의



### [Page 7]
7/5
Code
3
4
5
3. Hyper-parameter 지정
4. 학습을위한반복문선언
5. 정답률확인



### [Page 8]
8/5
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Division of Computer〮AI Engineering
Dong-A University, Busan, Rep. of Korea

