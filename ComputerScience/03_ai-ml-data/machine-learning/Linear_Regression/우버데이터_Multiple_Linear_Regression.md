---
aliases: []
course: machine-learning
created: '2025-04-09'
date: '2025-04-09'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ml
- type/lecture
title: Multiple Linear Regression with Uber Dataset
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/머신러닝 인터페이스|머신러닝 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]]
prerequisites:: [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]]
related:: [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/06_algorithms-graphics/computer-graphics/좌표공간과-카메라|좌표공간과-카메라]], [[LGAimer/LG Aimers 9기 평가 및 제출 가이드|LG Aimers 9기 평가 및 제출 가이드]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/스택|스택]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[LGAimer/LG Aimers 9기 지원서 초안|LG Aimers 9기 지원서 초안]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/머신러닝 지식그래프|머신러닝]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/머신러닝 지식그래프|머신러닝]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/머신러닝 근거 인덱스|머신러닝 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/rep. of korea|rep. of korea]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/년 1학기 머신러닝|년 1학기 머신러닝]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/dong a univ|dong a univ]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/learning rate|learning rate]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/SVM|SVM]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
# Multiple Linear Regression with Uber Dataset

>[!note] 다중 선형 회귀란?
>- 여러 개의 독립 변수를 사용하여 종속 변수를 예측하는 통계적 방법
>- 수식: $$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon$$
>- $\beta_0$ : 절편, $\beta_1$ ~ $\beta_n$ : 회귀 계수, $\epsilon$ : 오차항

## Import packages

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=6, suppress=True)
```

## 데이터셋 로딩

>[!info] Modified Uber Dataset
>- **종속 변수**: fare_amount (우버 요금 지불 금액)
>- **독립 변수**:
>    - pickup_x: 승객 탑승 x 좌표
>    - pickup_y: 승객 탑승 y 좌표
>    - dropoff_x: 승객 하차 x 좌표
>    - dropoff_y: 승객 하차 y 좌표
>    - passenger_count: 탑승 승객 수

```python
# Download dataset file
# !wget "https://dongaackr-my.sharepoint.com/:x:/g/personal/sjkim_donga_ac_kr/EYMXYk25h2VBtPXtu3QaBqoBJK4cK-TI9mamHoFGJRYn5Q?e=8x2MTn&download=1" -q -O modified_uber.csv

# Load dataset file
data = pd.read_csv('modified_uber.csv')
data = data.drop(['Unnamed: 0.1', 'Unnamed: 0', 'key', 'pickup_datetime'], axis=1)
data = data.dropna()
```

## L1, L2 이동거리 계산 및 열 추가

**L1 Distance (맨해튼 거리):**

L1_distance = |pickup_x - dropoff_x| + |pickup_y - dropoff_y|

**L2 Distance (유클리드 거리):**

L2_distance = √((pickup_x - dropoff_x)² + (pickup_y - dropoff_y)²)

- 절댓값 계산 함수: np.abs()
- 제곱근 (square root) 계산 함수: np.sqrt()

```python
data['L1_distance'] = np.abs(data['pickup_x'] - data['dropoff_x']) + np.abs(data['pickup_y'] - data['dropoff_y'])
data['L2_distance'] = np.sqrt((data['pickup_x'] - data['dropoff_x'])**2 + (data['pickup_y'] - data['dropoff_y'])**2)

data # L1/L2_distance 열이 정상적으로 계산/추가되었는지 확인
```
## 데이터셋 전처리

>[!important] 가우시안 정규화 (Gaussian Normalization)
>- 데이터를 평균이 0, 표준편차가 1인 정규분포 형태로 변환
>- 수식: $$x' = \frac{x - \mu}{\sigma}$$
>- 목적: 
>    1. 서로 다른 스케일의 특성들을 동일한 스케일로 변환
>    2. 모델의 학습 안정성과 성능 향상
>    3. 이상치의 영향 감소

```python
data = data.dropna()

# 데이터셋 가우시안 정규화
data_normalized = (data - data.mean()) / data.std()
data_normalized
```

## 택시요금 예측에 사용할 데이터 지정

```python
# 예측에 사용할 데이터들에 대한 2차원 행렬 변환
X = np.array(data_normalized[['L1_distance', 'L2_distance', 'passenger_count']]) # 입력 데이터 설정
Y = np.array(data_normalized[['fare_amount']])

# Train dataset / Test dataset 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# Train dataset 형상 확인
print(X_train.shape)
print(Y_train.shape)
```

## Least Square Method 기반 선형 회귀 모델 구현

>[!note] 최소 제곱법 (Least Square Method)
>- 예측값과 실제값의 차이(오차)의 제곱합을 최소화하는 방법
>>[!important] 수식
>>```python
>>XT = X.T
>>```
>>$$X^\top$$
>>```python
>>XTX = np.dot(XT, X)
>>```
>>$$X^\top X$$
>>```python
>>XTX_inverse = np.linalg.inv(XTX)
>>```
>>$$\left(X^\top X\right)^{-1}$$
>>```python
>>XTY = np.dot(XT, Y)
>>```
>>$$X^\top Y$$
>>```python
>>self.theta = np.dot(XTX_inverse, XTY)
>>```
>>$$\theta = \left(X^\top X\right)^{-1} X^\top Y$$

```python
class LinearRegression_LSM():
    def __init__(self):
        self.theta = None

    def fit(self, X, Y):
        N = X.shape[0] # N = 입력 데이터 개수

        # 입력 X에 대해 bias 차원 추가
        bias = np.ones((N, 1))    # N x 1
        X = np.hstack([X, bias])  # N x 2

        # theta (W, b) 저장을 위한 배열 초기화
        self.theta = np.zeros(X.shape[1])

        # Least Square Method 수행
        XT = X.T 
        XTX = np.dot(XT, X)
        XTX_inverse = np.linalg.inv(XTX)
        XTY = np.dot(XT, Y)

        self.theta = np.dot(XTX_inverse, XTY)
        return self.theta

    def predict(self, X):
        N = X.shape[0] # N = 입력 데이터 개수

        # 입력 X에 대해 bias 차원 추가
        bias = np.ones((N, 1)) # N x 1
        X = np.hstack([X, bias]) # N x 2

        Y_hat = np.dot(X, self.theta)
        return Y_hat
```

## 모델 학습 및 평가

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

```python
# X_train, Y_train 데이터를 이용한 linear regression 수행 (학습)
model_LSM = LinearRegression_LSM()
theta = model_LSM.fit(X_train, Y_train)

## X_test, Y_test 데이터를 이용한 linear regression 성능 검증 (테스트)
# MSE 계산 함수 정의
def MSE(Y, Y_hat):
    error = Y - Y_hat
    mse = np.mean(error ** 2)
    return mse

# 테스트 데이터에 대한 예측 및 평가
Y_hat = model_LSM.predict(X_test)
mse = MSE(Y_test, Y_hat)
```

## 임의 데이터 X 입력 시 Y_hat 예측

**가우시안 정규화:**

$$
x' = \frac{x - \mu}{\sigma}
$$

**가우시안 역정규화:**

$$
x = x' \cdot \sigma + \mu
$$

>[!example] 새로운 데이터에 대한 예측
>```python
># 가우시안 정규화/역정규화를 위한 평균, 표준편차 저장
>mean_array = data.mean()
>std_array = data.std()
>
>mean_L1 = mean_array['L1_distance']
>mean_L2 = mean_array['L2_distance']
>mean_passenger = mean_array['passenger_count']
>mean_fare = mean_array['fare_amount']
>
>std_L1 = std_array['L1_distance']
>std_L2 = std_array['L2_distance']
>std_passenger = std_array['passenger_count']
>std_fare = std_array['fare_amount']
>
># 임의 데이터 X 생성
>L1_distance = 250
>L2_distance = 197
>passenger_count = 2
>
># 각 입력 변수 X에 대한 정규화 수행
>L1_distance_norm = (L1_distance - mean_L1) / std_L1
>L2_distance_norm = (L2_distance - mean_L2) / std_L2
>passenger_count_norm = (passenger_count - mean_passenger) / std_passenger
>
>X_new = np.array(\[\[L1_distance_norm, L2_distance_norm, passenger_count_norm\]\])
>
># 학습한 모델 theta를 이용해 Y_hat 예측
>Y_hat = model_LSM.predict(X_new)
>
># 출력 변수 Y에 대한 역정규화 수행
>Y_hat = (Y_hat * std_fare) + mean_fare
>
>print(f"Y_hat = {Y_hat}")
>```

>[!warning] 주의사항
>1. 새로운 데이터 예측 시 반드시 정규화 필요
>2. 예측 결과는 역정규화하여 해석
>3. L1, L2 거리는 실제 거리와 비례하도록 계산
>4. 승객 수는 정수값으로 입력
