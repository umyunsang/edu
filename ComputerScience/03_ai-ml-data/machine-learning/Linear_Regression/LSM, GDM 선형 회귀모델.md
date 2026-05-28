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
title: Simple Linear Regression
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/머신러닝 인터페이스|머신러닝 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]]
prerequisites:: [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]]
related:: [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬|정렬]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/Stack|Stack]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/리스트|리스트]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[LGAimer/LG Aimers 9기 평가 및 제출 가이드|LG Aimers 9기 평가 및 제출 가이드]], [[LGAimer/LG Aimers 9기 지원서 초안|LG Aimers 9기 지원서 초안]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/머신러닝 지식그래프|머신러닝]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/머신러닝 지식그래프|머신러닝]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/머신러닝 근거 인덱스|머신러닝 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/rep. of korea|rep. of korea]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/image signal processing laboratory|image signal processing laboratory]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/년 1학기 머신러닝|년 1학기 머신러닝]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/learning rate|learning rate]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/machine-learning/SVM|SVM]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
# Simple Linear Regression

### 데이터셋 로딩 및 전처리
- kc_house_data: 미국 워싱턴주 시애틀 지역의 주택 가격 데이터를 포함한 공개 데이터셋
- **price: 주택의 판매 가격 (종속 변수, 목표 값).**
- **sqft_living: 주택의 실내 면적 (평방 피트).**

```python
# # Download dataset file

# Load dataset file
data = pd.read_csv('kc_house_data.csv')

# Single linear regression 실습에 사용할 데이터 열만 수집 (price (정답), sqft_living (입력))
X, Y = data['sqft_living'], data['price']

# 데이터 값 확인
df = data[['sqft_living', 'price']]
print(df)

# Numpy 배열로 전환
X = np.array(X) # sqft_living
Y = np.array(Y) # price

# X, Y 각각에 대한 평균과 표준편차 계산
X_mean = np.mean(X)
Y_mean = np.mean(Y)
X_std = np.std(X)
Y_std = np.std(Y)

# 평균, 표준편차를 이용한 Gaussian 정규화 수행
X = (X - X_mean) / X_std
Y = (Y - Y_mean) / Y_std

# 2차원 행렬 변환
X = np.expand_dims(X, 1)
Y = np.expand_dims(Y, 1)

# Train dataset / Test dataset 분할 (8:2 비율)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# Test dataset 시각화
fig = plt.figure()
plt.scatter(X_test, Y_test, color='b', marker='o', s=15)
plt.xlabel("X_test (sqft_living)")
plt.ylabel("Y_test (price)")
plt.show()

# 데이터 형상 확인
print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
```

>[!success]
>![[3-1_machine-learning__Pasted image 20250421120641.png]]
>X_train: (17290, 1), Y_train: (17290, 1)
X_test: (4323, 1), Y_test: (4323, 1)

## Least Square Method 기반 선형 회귀 모델 작성

- 크기를 입력 받고 모두 1로 채워진 행렬 생성:
  ```python
  arr = np.ones(size)
  ```
- 2개 이상 행렬을 가로로 쌓기:
  ```python
  arr = np.hstack([a, b])
  ```
- 행렬 곱 (dot product):
  ```python
  arr = np.dot(a, b)
  ```
- 전치 행렬:
  ```python
  arr = a.T
  ```
- 역 행렬:
  ```python
  arr = np.linalg.inv(a)
  ```

- Least Square Method:
$$\theta = (X^T \cdot X)^{-1} \cdot (X^T \cdot Y)$$

>[!important] 
> 최소제곱법은 데이터의 잔차 제곱합을 최소화하여 최적의 파라미터를 찾는 방법입니다.

>[!warning]
> 행렬 $X^T \cdot X$ 가 비가역 행렬일 경우 역행렬을 구할 수 없으므로 주의가 필요합니다.

```python
class LinearRegression_LSM():
    """
    최소제곱법(Least Square Method)을 이용한 선형 회귀 모델 클래스
    
    최소제곱법 공식: θ = (X^T · X)^(-1) · (X^T · Y)을 이용하여 
    최적의 파라미터(θ)를 계산하는 선형 회귀 모델입니다.
    
    Attributes:
        theta: 학습된 모델 파라미터 (가중치와 편향)
    """
    
    # 클래스 초기화 함수
    def __init__(self):
        # 학습된 파라미터(theta)를 저장할 변수 초기화
        self.theta = None

    # 학습 함수 - Least Squares Method(최소제곱법)를 이용해 theta를 계산
    def fit(self, X, Y):
        N = X.shape[0]  # 입력 샘플 개수 (행 개수)

        # Bias term 추가를 위해 모든 샘플에 대해 1을 추가 (상수항을 위한 열)
        bias = np.ones((N, 1))      # (N x 1) 크기의 배열 생성
        X = np.hstack([X, bias])    # (N x 2) bias 열 추가

        # --------- Least Squares Method 수식 구현 ---------
        # Normal Equation: θ = (XᵀX)^(-1) XᵀY
        # X의 전치 행렬
        X_T = X.T

        # XᵀX 계산
        XTX = np.dot(X_T, X)

        # 역행렬 계산: (XᵀX)^(-1)
        XTX_inv = np.linalg.inv(XTX)

        # XᵀY 계산
        XTY = np.dot(X_T, Y)

        # 최종 파라미터 θ 계산: θ = (XᵀX)^(-1) XᵀY
        self.theta = np.dot(XTX_inv, XTY)

        return self.theta

    def predict(self, X):
        # 입력 X에 bias 항(상수항) 추가
        bias = np.ones((X.shape[0], 1))    # (N x 1)
        X = np.hstack([X, bias])           # (N x 2)

        # 예측값 계산: ŷ = Xθ
        pred = np.dot(X, self.theta)

        return pred
```

###  X_train, Y_train 데이터를 이용한 linear regression 수행 (학습)

```python
model_LSM = LinearRegression_LSM()
theta = model_LSM.fit(X_train, Y_train)

print(f"W = {theta[0]}, b = {theta[1]}")
```
>[!success]
>W = [0.70406843], b = [0.00267388]

### X_test, Y_test 데이터를 이용한 linear regression 성능 검증 (테스트)
```python
Y_pred = model_LSM.predict(X_test)

# 시각화
fig = plt.figure()
plt.scatter(X_test, Y_test, color='b', marker='o', s=15)
plt.plot(X_test, Y_pred, color='r')
plt.xlabel("X_test (sqft_living)")
plt.ylabel("Y_test / Y_pred (price)")
plt.show()
```

>[!success]
>![[3-1_machine-learning__Pasted image 20250421120853.png]]

---
## Gradient Descent Method 기반 선형 회귀 모델 작성

- **Parameters:**
  - `iteration`: 경사하강법의 반복 횟수
  - `learning_rate`: 학습률

- **Attributes:**
  - `theta`: 학습된 모델의 파라미터

>[!important]
> 경사하강법은 반복적인 업데이트를 통해 비용 함수를 최소화하는 방법으로, 학습률과 반복 횟수가 중요합니다.

>[!warning]
> 학습률이 너무 크면 발산할 수 있고, 너무 작으면 수렴 속도가 느려질 수 있습니다. 적절한 학습률을 선택하는 것이 중요합니다.

```python

class LinearRegression_GDM():
  """
  경사하강법(Gradient Descent Method)을 사용한 선형 회귀 모델 클래스입니다.
  
  Parameters
  ----------
  iteration : int, default=1000
      경사하강법의 반복 횟수
  learning_rate : float, default=1e-4  
      학습률(learning rate)
      
  Attributes
  ----------
  theta : ndarray
      학습된 모델의 파라미터 (w, b)
  """

  #def __init__(self, iteration=1000, learning_rate=0.1):
  def __init__(self, iteration=1000, learning_rate=1e-4):
    self.iteration = iteration              # 반복 횟수 설정
    self.learning_rate = learning_rate      # 학습률 설정
    self.theta = None                       # 학습된 파라미터 저장 변수

  def fit(self, X, Y):
    N = X.shape[0]                          # 데이터 개수

    # 행렬 X에 bias 열 추가
    bias = np.ones((N, 1))                  # (N x 1)
    X = np.hstack([X, bias])                # (N x 2)

    # w, b 초기값 설정
    w = 0.0
    b = 0.0

    for i in range(self.iteration):
      # [[w],
      #  [b]] 형태로 theta 행렬 생성
      theta = [w, b]
      theta = np.array([w, b]).reshape(2, 1)

      # y_hat 계산: 예측값 = X @ theta
      y_hat = np.dot(X, theta)

      # dw, db 계산
      # dw = (2/N) * sum((y - y_hat) * -X)
      dw = (2/N) * sum((Y - y_hat) * (-X[:, [0]]))

      # db = (2/N) * sum((y - y_hat) * -1)
      db = (2/N) * sum((Y - y_hat) * -1)

      # w, b 업데이트
      # w_t+1 = w_t - learning_rate * dw
      w = w - self.learning_rate * dw

      # b_t+1 = b_t - learning_rate * db
      b = b - self.learning_rate * db

    # 최종 학습된 theta 저장
    self.theta = np.array([w, b])  # (1차원 배열)

    return self.theta

  def predict(self, X):
    # 예측을 위한 bias 열 추가
    bias = np.ones((X.shape[0], 1))     # (N x 1)
    X = np.hstack([X, bias])            # (N x 2)

    # 예측값 계산
    pred = np.dot(X, self.theta)        # (N x 1)

    return pred
```

### 학습
```python
model_GDM = LinearRegression_GDM(iteration=1000, learning_rate=0.1)
theta = model_GDM.fit(X_train, Y_train)

print(f"W = {theta[0]}, b = {theta[1]}")
```
### 성능 검증
```python
Y_pred = model_GDM.predict(X_test)

# 시각화
fig = plt.figure()
plt.scatter(X_test, Y_test, color='b', marker='o', s=15)
plt.plot(X_test, Y_pred, color='r')
plt.xlabel("X_test (sqft_living)")
plt.ylabel("Y_test / Y_pred (price)")
plt.show()
```
