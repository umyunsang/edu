---
aliases: []
course: ml-projects
created: '2024-08-05'
date: '2024-08-05'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ml
- type/lecture
title: 데이터 포인트 개수
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/ML 프로젝트 인터페이스|ML 프로젝트 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/ML 프로젝트 근거 인덱스|ML 프로젝트 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/구구단 프로그램|구구단 프로그램]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/knn|knn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/svm|svm]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 포인트 개수
N = 30

# 랜덤한 x 좌표와 y 좌표 생성
x = np.random.rand(N)  # 0과 1 사이의 랜덤 값 N개
y = np.random.rand(N)  # 0과 1 사이의 랜덤 값 N개

# 점의 색상을 위한 랜덤 값 생성
color = np.random.rand(N)  # 0과 1 사이의 랜덤 값 N개

# 점의 크기를 위한 랜덤 값 생성
# (30 * 랜덤 값)^2로 점의 크기를 결정
area = (30 * np.random.rand(N)) ** 2  # 0부터 30 사이의 크기

# 산점도 그래프 생성
# s: 점의 크기, c: 점의 색상, alpha: 투명도
plt.scatter(x, y, s=area, c=color, alpha=0.5)

# 그래프 출력
plt.show()
```

- **데이터 생성**:
  - `N = 30`: 30개의 데이터 포인트를 생성합니다.
  - `x = np.random.rand(N)`: 0과 1 사이의 랜덤 값을 가지는 x 좌표를 생성합니다.
  - `y = np.random.rand(N)`: 0과 1 사이의 랜덤 값을 가지는 y 좌표를 생성합니다.
  - `color = np.random.rand(N)`: 점의 색상을 위한 랜덤 값을 생성합니다.
  - `area = (30 * np.random.rand(N)) ** 2`: 점의 크기를 위한 랜덤 값을 생성합니다. 점의 크기는 \( (30 \times \text{랜덤 값})^2 \)로 계산됩니다.

- **산점도 그래프 생성**:
  - `plt.scatter(x, y, s=area, c=color, alpha=0.5)`: 산점도를 생성합니다.
    - `x`와 `y`: 데이터 포인트의 위치입니다.
    - `s`: 점의 크기를 설정합니다. `area` 배열로 크기가 조절됩니다.
    - `c`: 점의 색상을 설정합니다. `color` 배열로 색상이 조절됩니다.
    - `alpha`: 점의 투명도를 설정합니다. 0은 완전 투명, 1은 불투명입니다.

- **그래프 출력**:
  - `plt.show()`: 그래프를 화면에 출력합니다.

---

### `scatter` 함수의 파라미터 설명

| **파라미터** | **설명**                                           | **예시 값**                         |
|--------------|----------------------------------------------------|-------------------------------------|
| `x`          | 데이터 포인트의 x 좌표                           | `np.random.rand(N)`                 |
| `y`          | 데이터 포인트의 y 좌표                           | `np.random.rand(N)`                 |
| `s`          | 데이터 포인트의 크기 (면적)                       | `(30 * np.random.rand(N)) ** 2`     |
| `c`          | 데이터 포인트의 색상                             | `np.random.rand(N)`                 |
| `alpha`      | 데이터 포인트의 투명도 (0은 완전 투명, 1은 불투명) | `0.5`                               |

- **`s` (Size)**: 점의 크기를 제어합니다. 크기는 면적으로 표시되므로 제곱값으로 지정하는 것이 일반적입니다.
- **`c` (Color)**: 점의 색상을 설정합니다. 색상은 숫자 배열, RGB/RGBA 튜플, 색상 이름 등으로 설정할 수 있습니다.
- **`alpha`**: 점의 투명도를 설정합니다. 0이면 완전히 투명하고, 1이면 불투명합니다.

---
