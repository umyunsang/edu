---
aliases: []
course: ml-projects
created: '2024-08-07'
date: '2024-08-07'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 3-1
source: ''
status: seedling
tags:
- cs/ml
- type/lecture
title: DataFrame 생성
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/ML 프로젝트 인터페이스|ML 프로젝트 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/ML 프로젝트 근거 인덱스|ML 프로젝트 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/구구단 프로그램|구구단 프로그램]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/knn|knn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/svm|svm]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

#### 1. DataFrame 생성 및 초기화

먼저, 상품, 재질, 가격 정보를 포함하는 DataFrame을 생성합니다.

```python
import pandas as pd
import numpy as np

# DataFrame 생성
df = pd.DataFrame({
    '상품': ['시계', '반지', '반지', '목걸이', '팔찌'],
    '재질': ['금', '은', '백금', '금', '은'],
    '가격': [500000, 20000, 350000, 300000, 60000]
})
print(df)
```

출력:
```
    상품  재질      가격
0  시계   금  500000
1  반지   은   20000
2  반지  백금  350000
3  목걸이  금  300000
4  팔찌   은   60000
```

---

#### 2. Pivot 테이블 생성

`pivot` 함수를 사용하여 '상품'을 행, '재질'을 열, '가격'을 값으로 하는 새로운 DataFrame을 생성합니다. 결측치는 0으로 채웁니다.

```python
# pivot을 사용하여 '상품'을 행, '재질'을 열, '가격'을 값으로 하는 새로운 DataFrame 생성
new_df = df.pivot(index='상품', columns='재질', values='가격')
# 결측치를 0으로 채움
new_df = new_df.fillna(value=0)
print(new_df)
```

출력:
```
재질       금      백금       은
상품                            
목걸이  300000.0    0.0      0.0
반지         0.0  350000.0  20000.0
시계    500000.0    0.0      0.0
팔찌         0.0    0.0     60000.0
```

---

#### 3. DataFrame 결합

세로로 결합할 두 개의 DataFrame을 생성합니다.

```python
# 첫 번째 DataFrame 생성
df_1 = pd.DataFrame({
    'A': ['a10', 'a11', 'a12'],
    'B': ['b10', 'b11', 'b12'],
    'C': ['c10', 'c11', 'c12']
}, index=['가', '나', '다'])

# 두 번째 DataFrame 생성
df_2 = pd.DataFrame({
    'B': ['b23', 'b24', 'b25'],
    'C': ['c23', 'c24', 'c25'],
    'D': ['d23', 'd24', 'd25']
}, index=['다', '라', '마'])
```

각 DataFrame의 내용을 출력해 봅니다.

출력 `df_1`:
```
     A   B   C
가  a10 b10 c10
나  a11 b11 c11
다  a12 b12 c12
```

출력 `df_2`:
```
     B   C   D
다  b23 c23 d23
라  b24 c24 d24
마  b25 c25 d25
```

---

#### 4. DataFrame 세로로 결합

두 DataFrame을 세로로 결합합니다.

```python
# 두 DataFrame을 세로로 결합
df_3 = pd.concat([df_1, df_2])
print(df_3)
```

출력:
```
      A    B    C    D
가    a10  b10  c10  NaN
나    a11  b11  c11  NaN
다    a12  b12  c12  NaN
다    NaN  b23  c23  d23
라    NaN  b24  c24  d24
마    NaN  b25  c25  d25
```

---

#### 5. 공통 열만 포함하도록 결합

두 DataFrame을 공통 열만 포함하도록 결합합니다.

```python
# 두 DataFrame을 공통 열만 포함하도록 결합
df_4 = pd.concat([df_1, df_2], join='inner')
print(df_4)
```

출력:
```
      B    C
가    b10  c10
나    b11  c11
다    b12  c12
다    b23  c23
라    b24  c24
마    b25  c25
```

---

#### 6. merge를 이용한 조인 연산

각 유형의 조인 연산을 수행하여 결과를 비교합니다.

```python
# merge를 이용한 조인 연산
# left outer join: df_1 기준
print('left outer \n', df_1.merge(df_2, how='left', on='B'))

# right outer join: df_2 기준
print('right outer \n', df_1.merge(df_2, how='right', on='B'))

# full outer join: df_1과 df_2의 모든 데이터를 포함
print('full outer \n', df_1.merge(df_2, how='outer', on='B'))

# inner join: 공통된 데이터만 포함
print('inner \n', df_1.merge(df_2, how='inner', on='B'))
```

출력:
```
left outer 
     A   B   C_x   C_y    D
0  a10  b10  c10  NaN   NaN
1  a11  b11  c11  NaN   NaN
2  a12  b12  c12  c23  d23

right outer 
     A   B   C_x   C_y    D
0  a12  b23  c12  c23  d23
1  NaN  b24  NaN  c24  d24
2  NaN  b25  NaN  c25  d25

full outer 
     A   B   C_x   C_y    D
0  a10  b10  c10  NaN   NaN
1  a11  b11  c11  NaN   NaN
2  a12  b12  c12  c23  d23
3  NaN  b24  NaN  c24  d24
4  NaN  b25  NaN  c25  d25

inner 
     A   B   C_x   C_y    D
0  a12  b23  c12  c23  d23
```

- **Left Outer Join**: `df_1`을 기준으로 조인.
- **Right Outer Join**: `df_2`를 기준으로 조인.
- **Full Outer Join**: `df_1`과 `df_2`의 모든 데이터를 포함.
- **Inner Join**: 공통된 데이터만 포함.

---

### 요약

- **pivot**: 데이터 재구조화.
- **concat**: 데이터프레임 결합.
- **merge**: 다양한 방식으로 데이터프레임 병합.

---
