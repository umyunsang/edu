---
aliases: []
course: ml-projects
created: '2024-08-01'
date: '2024-08-01'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ml
- type/lecture
title: Numpy 기초
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/ML 프로젝트 인터페이스|ML 프로젝트 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/Queue|Queue]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬|정렬]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트|1. 배열과 리스트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/스택|스택]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/ML 프로젝트 근거 인덱스|ML 프로젝트 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/구구단 프로그램|구구단 프로그램]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/knn|knn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/svm|svm]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
#### 배열 생성 및 속성 확인

```python
a = np.array([2, 3, 4])
print(a.shape)      # (3,) - 배열의 형태 (1차원, 요소 3개)
print(a.ndim)       # 1 - 배열의 차원 수
print(a.dtype)      # int64 - 배열 요소의 데이터 타입
print(a.itemsize)   # 8 - 배열 요소 하나의 크기 (바이트 단위)
print(a.size)       # 3 - 배열의 전체 요소 수
```

#### 2차원 배열 생성
```python
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)      # (2, 3) - 배열의 형태 (2차원, 2x3)
```

#### 배열 연산

```python
a = np.array([10, 20, 30])
b = np.array([1, 2, 3])
print(a - b)        # [9, 18, 27] - 배열의 요소별 뺄셈
print(a + b)        # [11, 22, 33] - 배열의 요소별 덧셈
print(a / b)        # [10.0, 10.0, 10.0] - 배열의 요소별 나눗셈
```

#### 데이터 타입 지정
```python
a = np.array([10, 20, 30], dtype=np.int32)
print(a.dtype)      # int32 - 배열 요소의 데이터 타입
```

#### 업 캐스팅 (Up Casting)
```python
b = np.array([10, 20.1, 40])
print(b.dtype)      # float64 - 배열 요소의 데이터 타입
```

#### 브로드캐스팅 (Broadcasting)
```python
print(a * 10)       # [100, 200, 300] - 배열의 요소별 곱셈
print(a / 2)        # [5.0, 10.0, 15.0] - 배열의 요소별 나눗셈
print(a + 10)       # [20, 30, 40] - 배열의 요소별 덧셈
```

2차원 배열과 1차원 배열의 브로드캐스팅:
```python
b = np.array([[10, 20, 30], [40, 50, 60]])
c = np.array([2, 3, 4])
print(b + c)        # [[12, 23, 34], [42, 53, 64]] - 브로드캐스팅 덧셈
print(b * c)        # [[20, 60, 120], [80, 150, 240]] - 브로드캐스팅 곱셈
```

#### 배열 초기화 함수
```python
print(np.zeros((2, 3)))          # 2x3 배열, 모든 요소 0
print(np.ones((2, 3)))           # 2x3 배열, 모든 요소 1
print(np.full((2, 3), 100))      # 2x3 배열, 모든 요소 100
print(np.eye(3))                 # 3x3 단위 행렬
print(np.eye(10))                # 10x10 단위 행렬
```

#### 연속 값 배열 생성
```python
print(np.arange(0, 10))          # [0, 1, ..., 9]
print(np.arange(0, 10, 2))       # [0, 2, 4, 6, 8] - 2씩 증가
print(np.arange(0.0, 1.0, 0.2))  # [0.0, 0.2, ..., 0.8] - 0.2씩 증가
```

#### 특정 구간을 나눈 값 생성
```python
print(np.linspace(0, 10, 5))     # [0, 2.5, 5, 7.5, 10] - 5등분
print(np.linspace(0, 10, 4))     # [0, 3.33, 6.67, 10] - 4등분
```

#### 지수, 로그
```python
a = np.logspace(0, 5, 6)         # [1.0, 10.0, ..., 100000.0] - 로그스케일 값
print(a)
print(np.log10(a))               # [0.0, 1.0, ..., 5.0] - 상기 배열의 로그값
```

#### 배열에 요소 삽입
```python
a = np.array([1, 3, 4])
print(np.insert(a, 1, 2))        # [1, 2, 3, 4] - 인덱스 1에 2 삽입
b = np.array([[1, 1], [2, 2], [3, 3]])
print(np.insert(b, 1, 4, axis=0))# [[1, 1], [4, 4], [2, 2], [3, 3]] - 행 추가
c = np.array([[1, 2, 3], [4, 5, 6]])
print(c)
print(np.flip(c, axis=0))        # 배열을 행 기준으로 뒤집기
```

#### 배열 인덱싱
```python
print(c[1][1])       # 5 - 인덱스 [1][1]의 요소
print(c[1, 1])       # 5 - 인덱스 [1, 1]의 요소
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]])
print(arr_2d)
print(arr_2d[2])     # [7, 8, 9] - 2번째 행
print(arr_2d[2, 2])  # 9 - 인덱스 [2, 2]의 요소
print(arr_2d[1:3, 1])# [5, 8] - 인덱스 1~2 행, 1열
print(arr_2d[2:, :2])# [[7, 8], [0, 1]] - 2번째 행 이후, 처음 두 열
```

#### 배열의 최소, 최대, 평균값
```python
a = np.array([10, 20, 30])
print(a.min())       # 10
print(a.max())       # 30
print(a.mean())      # 20.0
```

#### 배열 평탄화 및 전치
```python
print(arr_2d.flatten()) # [1, 2, ..., 2] - 1차원 배열로 변환
print(arr_2d.T)         # 배열 전치
```

#### 배열 정렬
```python
c = np.array([35, 24, 55, 69, 19, 9, 4, 1, 11])
c.sort()
print(c)               # [1, 4, 9, ..., 69] - 오름차순 정렬
print(c[::-1])         # [69, 55, ..., 1] - 내림차순 정렬

d = np.array([[35, 24, 55], [69, 19, 9], [4, 1, 11]])
d.sort()
print(d)               # 각 행 기준 정렬
d.sort(axis=0)
print(d)               # 각 열 기준 정렬
```

#### 배열 요소 추가
```python
a = np.array([1, 2, 3])
b = np.array([[4, 5, 6], [7, 8, 9]])
print(np.append(a, b))        # [1, 2, 3, 4, ..., 9] - 1차원 배열로 추가
print(np.append([a], b, axis=0))# [[1, 2, 3], [4, 5, 6], [7, 8, 9]] - 2차원으로 추가
```

#### 배열 연산 및 행렬 곱
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[10, 20], [30, 40]])
print(a * b)            # [[10, 40], [90, 160]] - 요소별 곱셈
print(a @ b)            # [[70, 100], [150, 220]] - 행렬 곱
```

#### 난수 생성 및 배열
```python
np.random.seed(42)      # 난수 시드 설정
rnd = np.random.rand(5) # [0.37, 0.95, ..., 0.06] - 0~1 사이 난수 5개
print(rnd)
rnd = np.random.rand(5) * 10 + 165 # [166.0, 168.6, ..., 172.1] - 범위 조정
print(rnd.round(2))     # 소수점 둘째 자리까지 반올림

print(rnd.astype(int))  # 정수로 변환

nums = np.random.normal(loc=165, scale=10, size=(3, 4)).round(2)
print(nums)             # 정규분포를 따르는 난수 배열 (평균 165, 표준편차 10)
```

#### 논리 인덱싱
```python
np_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(np_array % 2 == 0)        # 짝수인 경우 True
print(np_array[np_array % 2 == 0]) # 짝수 요소만 추출
print(np_array[np_array >= 5])  # 5 이상인 요소만 추출
```

#### 배열의 리덕션
```python
arr = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(np.sum(arr, axis=0))      # 각 열의 합
print(np.sum(arr, axis=1))      # 각 행의 합
print(np.sum(arr))              # 전체 요소의 합
```

#### 배열 형태 변환
```python
y = np.arange(12)
print(y.reshape(3, 4))          # 3x4 배열로 변환
print(y.reshape(6, -1))         # 6x2 배열로 변환 (자동 계산)
```

#### 배열 병합
```python
a = np.arange(10, 18).reshape(2, 4)
print(a)
b = np.arange(12).reshape(3, 4)
print(b)
c = np.arange(8).reshape(2, 4)
print(c)
d = list(range(4))
print(d)

print(np.concatenate((a, b)))   # 행 기준 병합
print(np.vstack((a, b)))        # 행 기준 병합
print(np.vstack((a, d)))        # 행 기준 병합 (리스트 포함)
print(np.vstack((a, b, d)))     # 행 기준 병합 (리스트 포함)
print(np.hstack((a, c)))        # 열 기준 병합
```

#### 배열 결합
```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[10, 20, 30], [40, 50, 60]])
print(np.r_[a, b])              # [1, 2, 3, 10, 20, 30] - 배열 결합
print(np.r_[c, d])              # [[1, 2, 3], [4, 5, 6], [10, 20, 30], [40, 50, 60]] - 배열 결합
print(np.r_[a, 4, 5, 6, b])     # [1, 2, 3, 4, 5, 6, 10, 20, 30] - 배열과 스칼라 결합
print(np.r_[[0] * 3, 5, 6])     # [0, 0, 0, 5, 6] - 배열 결합
```

---
