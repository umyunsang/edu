---
aliases: []
course: probability-statistics
created: '2026-03-19'
date: '2026-03-19'
semester: 2-1
source: ''
status: seedling
tags:
- math/probability
- math/statistics
- type/lecture
title: Random Variables
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]]
related:: [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/02_math-theory/discrete-mathematics/1. 수학적 모델과 논리/수학적 모델과 논리|수학적 모델과 논리]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/02_math-theory/mathematical-logic/논리학 개론|논리학 개론]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/ASD Feature 발굴|ASD Feature 발굴]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/02_math-theory/mathematical-logic/동아설계도|동아설계도]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/파라미터 추정|파라미터 추정]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/joint RVs|joint RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Independent RVs|Independent RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/random variables|random variables]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Continuous Joint|Continuous Joint]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

#ComputerScience #확률과통계 

---
1. **랜덤 변수의 정의**:
   - 랜덤 변수는 표본 공간의 각 표본에 대해 실수 값을 가지는 함수입니다. 즉, 표본 공간의 각 원소에 대해 실수 값을 할당하는 매핑입니다.
   - 수식: 만약 확률 실험의 표본 공간을 $S$라고 하고, 랜덤 변수를 $X$로 표현한다면, $X: S \rightarrow \mathbb{R}$로 나타낼 수 있습니다.

2. **이산형 랜덤 변수 (Discrete Random Variables)**:
   - 이산형 랜덤 변수는 유한하거나 셀 수 있는 값을 가지는 랜덤 변수를 의미합니다.
   - 수식: 이산형 랜덤 변수 $X$의 확률 질량 함수(PMF)는 $P(X = x)$로 표시됩니다. 여기서 $x$는 $X$가 취할 수 있는 값입니다.

3. **연속형 랜덤 변수 (Continuous Random Variables)**:
   - 연속형 랜덤 변수는 특정 구간 내의 모든 값을 가질 수 있는 랜덤 변수를 의미합니다.
   - 수식: 연속형 랜덤 변수 $X$의 확률 밀도 함수(PDF)는 $f(x)$로 표시됩니다. 여기서 $x$는 $X$의 가능한 값이며, $f(x)$는 해당 값에서의 확률 밀도입니다.

1. **누적 분포 함수 (Cumulative Distribution Function, CDF)**:
   - 랜덤 변수의 확률 분포를 설명하는 데 사용되는 함수로, 랜덤 변수가 특정 값보다 작거나 같은 확률을 제공합니다.
   - 수식: 이산형 랜덤 변수의 경우 $F(x) = P(X \leq x)$로 표시되며, 연속형 랜덤 변수의 경우 $F(x) = \int_{-\infty}^{x} f(t) dt$로 정의됩니다.

2. **확률 질량 함수 (Probability Mass Function, PMF)**:
	- 이산형 랜덤 변수의 경우, 각 가능한 값에 대한 확률을 나타내는 함수입니다.
	- PMF는 각 특정 값이 발생할 확률을 나타내며, 이를 통해 이산형 확률 변수의 분포를 완전히 설명할 수 있습니다.
	- 보통 $P(X = x)$로 표기하며, $x$는 랜덤 변수가 취할 수 있는 값입니다.
	- PMF는 다음과 같은 특징을 가집니다:
	  1. 모든 가능한 값에 대해 확률은 0 또는 양수여야 합니다: $P(X = x) \geq 0$.
	  2. 모든 가능한 값에 대한 확률의 합은 1이어야 합니다: $\sum_{x} P(X = x) = 1$.
