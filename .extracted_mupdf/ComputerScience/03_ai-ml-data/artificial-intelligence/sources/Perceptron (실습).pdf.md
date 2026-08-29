### [Page 1]
1/7
Dong-A Univ. (ISPL)
컴퓨터AI공학부AI학과
2024년1학기인공지능



### [Page 2]
2/7
Contents
1. Perceptron 기반Gate 구현
A. AND Gate
B. OR Gate
C. NAND Gate
D. XOR Gate



### [Page 3]
3/7
실습환경: Google Colab
실습환경실행
1.
구글계정생성및로그인후구글사이트에서구글코랩검색후접속
2.
좌측상단파일새노트Jupyter Notebook 실행
Logic Gate of Perceptron



### [Page 4]
4/7
AND Gate Perceptron 구현– 실습
1. 새로운노트북파일생성및실행(EX. Perceptron_Gate.ipynb)
2.
AND Gate Perceptron 구현을위한코드작성
3.
AND Gate Perceptron 실행및결과확인
Logic Gate of Perceptron
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
𝑠= 𝑤1 ∗𝑥1 + 𝑤2 ∗𝑥2 + 𝑏
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


### [Page 5]
5/7
OR Gate Perceptron 구현– 실습
1. 새로운노트북파일생성및실행(EX. Perceptron_Gate.ipynb)
2.
OR Gate Perceptron 구현을위한코드작성
3.
OR Gate Perceptron 실행및결과확인
Logic Gate of Perceptron
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
𝑠= 𝑤1 ∗𝑥1 + 𝑤2 ∗𝑥2 + 𝑏
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


### [Page 6]
6/7
NAND Gate Perceptron 구현– 실습
1. 새로운노트북파일생성및실행(EX. Perceptron_Gate.ipynb)
2.
NAND Gate Perceptron 구현을위한코드작성
3.
NAND Gate Perceptron 실행및결과확인
Logic Gate of Perceptron
Input
Output
𝑠= 𝑤1 ∗𝑥1 + 𝑤2 ∗𝑥2 + 𝑏
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


### [Page 7]
7/7
XOR Gate Perceptron 구현– 실습
1. 새로운노트북파일생성및실행(EX. Perceptron_Gate.ipynb)
2.
XOR Gate Perceptron 구현을위한코드작성
•
기존의NAND , OR , AND Gate Perceptron 서로연결
3.
XOR Gate Perceptron 실행및결과확인
Logic Gate of Perceptron
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


### [Page 8]
8/7
Questions & Answers
Dongsan Jun (dsjun@dau.ac.kr)
Image Signal Processing Laboratory (www.donga-ispl.kr)
Dept. of AI
Dong-A University, Busan, Rep. of Korea

