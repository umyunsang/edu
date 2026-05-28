---
aliases: []
course: ml-projects
created: '2024-08-05'
date: '2024-08-05'
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
title: 랜덤 시드 설정 (재현성을 위해)
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/ML 프로젝트 인터페이스|ML 프로젝트 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/ML 프로젝트 근거 인덱스|ML 프로젝트 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/구구단 프로그램|구구단 프로그램]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/knn|knn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ml-projects/svm|svm]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

```python
import matplotlib.pyplot as plt
import numpy as np

# 랜덤 시드 설정 (재현성을 위해)
np.random.seed(42)

# 데이터 생성
data1 = np.random.normal(100, 10, 200)  # 평균 100, 표준편차 10, 200개의 데이터
data2 = np.random.normal(100, 40, 200)  # 평균 100, 표준편차 40, 200개의 데이터
data3 = np.random.normal(80, 40, 200)   # 평균 80, 표준편차 40, 200개의 데이터
data4 = np.random.normal(80, 60, 200)   # 평균 80, 표준편차 60, 200개의 데이터

# 박스 플롯 생성
# 여러 데이터 세트를 비교하기 위해 리스트 형태로 전달
plt.boxplot([data1, data2, data3, data4])

# 그래프 출력
plt.show()
```

- **데이터 생성**:
  - `np.random.seed(42)`: 랜덤 시드를 설정하여 결과의 재현성을 보장합니다. 동일한 시드를 사용하면 같은 랜덤 데이터가 생성됩니다.
  - `data1`, `data2`, `data3`, `data4`: 각각 평균과 표준편차가 다른 정규 분포를 가진 데이터 세트입니다. 총 200개의 데이터 포인트를 포함합니다.

- **박스 플롯 생성**:
  - `plt.boxplot([data1, data2, data3, data4])`: `boxplot` 함수를 사용하여 박스 플롯을 생성합니다. 데이터 세트를 리스트 형태로 전달하여 여러 데이터 세트를 동시에 시각화합니다.

- **그래프 출력**:
  - `plt.show()`: 그래프를 화면에 출력합니다.

---

### `boxplot` 함수의 파라미터 설명

| **파라미터**   | **설명**                                                                                  | **예시 값**                       |
|----------------|-------------------------------------------------------------------------------------------|-----------------------------------|
| `x`            | 박스 플롯을 생성할 데이터 세트입니다. 배열 또는 리스트 형태로 전달됩니다.                   | `[data1, data2, data3, data4]`   |
| `notch`        | 박스의 중간값에 노치를 추가할지 여부를 결정합니다. (기본값: `False`)                       | `True`, `False`                   |
| `vert`         | 박스 플롯의 방향을 설정합니다. (세로 방향: `True`, 가로 방향: `False`)                      | `True`, `False`                   |
| `patch_artist` | 박스의 내부를 색칠할지 여부를 결정합니다. (기본값: `False`)                                 | `True`, `False`                   |
| `showfliers`   | 이상치를 표시할지 여부를 결정합니다. (기본값: `True`)                                      | `True`, `False`                   |

- **`x`**: 박스 플롯을 생성할 데이터 세트입니다. 각 데이터 세트는 박스 플롯의 하나의 박스를 나타냅니다.
- **`notch`**: 박스의 중간값에 노치를 추가하여 중앙값의 신뢰 구간을 표시할 수 있습니다. 노치를 추가하면 데이터의 분포에 대한 추가적인 정보를 제공합니다.
- **`vert`**: 박스 플롯의 방향을 설정합니다. 기본적으로 세로 방향으로 생성되며, `False`로 설정하면 가로 방향으로 생성됩니다.
- **`patch_artist`**: 박스의 내부를 색칠할지 여부를 결정합니다. 색칠하면 데이터의 시각적 표현이 더 명확해집니다.
- **`showfliers`**: 이상치를 표시할지 여부를 결정합니다. 기본값은 `True`로, 이상치를 시각적으로 강조할 수 있습니다.

---
