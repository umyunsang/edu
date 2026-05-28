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
title: Binomial Distribution
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]]
related:: [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/파라미터 추정|파라미터 추정]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/joint RVs|joint RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Independent RVs|Independent RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/random variables|random variables]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Continuous Joint|Continuous Joint]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

#ComputerScience #확률과통계

---
**이항 분포(Binomial Distribution)**:

이항 분포는 성공과 실패를 가진 이진 시행을 $n$ 번 반복하여 성공의 횟수를 나타내는 확률 분포입니다. 각 시행은 독립적이며, 동일한 확률로 $k$번 성공 또는 $(n-k)$번 실패합니다.

**주요 속성**:
- **확률 변수**:  $X \sim \text{Bin}(n, p)$ (이항 확률 변수)
- **설명**: $n$번의 독립적인 이진 시행에서 성공의 횟수를 나타내는 변수
- **매개 변수**: $n$ (시행 횟수), $p$ (각 시행의 성공 확률)
- **지원 범위**:  $k = 0, 1, 2, ..., n$
- **PMF 방정식**: $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
- **기대값**: 
	$E[X] = np$
- **분산**: 
	$\text{Var}(X) = np(1-p)$

이항 분포의 주요 특징은 시행 횟수 $n$ 과 각 시행의 성공 확률 $p$ 을 사용하여 성공 횟수의 기대값과 분산을 계산할 수 있다는 것입니다. 

**이항 분포 예시:**

예를 들어, 공정한 동전을 5번 던져서 앞면이 나오는 횟수를 알아보고자 합니다. 이 경우, 이항 확률 변수 $Y$ 는 다음과 같이 정의됩니다:$$Y \sim \text{Bin}(5, 0.5)$$
여기서 $n = 5$ 는 동전을 던지는 횟수이고, $p = 0.5$ 는 각 동전 던지기에서 앞면이 나올 확률입니다.

**풀이:**

1. $P(Y = 0)$ : 앞면이 한 번도 나오지 않는 경우$$P(Y = 0) = \binom{5}{0} (0.5)^0 (1-0.5)^{5-0} = 1 \times 1 \times 0.5^5 = 0.03125$$
2. $P(Y = 1)$ : 앞면이 한 번 나오는 경우$$P(Y = 1) = \binom{5}{1} (0.5)^1 (1-0.5)^{5-1} = 5 \times 0.5 \times 0.5^4 = 0.15625$$
3. $P(Y = 2)$ : 앞면이 두 번 나오는 경우$$P(Y = 2) = \binom{5}{2} (0.5)^2 (1-0.5)^{5-2} = 10 \times 0.25 \times 0.5^3 = 0.3125$$
4. $P(Y = 3)$ : 앞면이 세 번 나오는 경우$$P(Y = 3) = \binom{5}{3} (0.5)^3 (1-0.5)^{5-3} = 10 \times 0.125 \times 0.5^2 = 0.3125$$
5. $P(Y = 4)$ : 앞면이 네 번 나오는 경우$$P(Y = 4) = \binom{5}{4} (0.5)^4 (1-0.5)^{5-4} = 5 \times 0.0625 \times 0.5^1 = 0.15625$$
6. $P(Y = 5)$ : 앞면이 다섯 번 나오는 경우$$P(Y = 5) = \binom{5}{5} (0.5)^5 (1-0.5)^{5-5} = 1 \times 0.03125 \times 0.5^0 = 0.03125$$
따라서, 이러한 경우들의 확률을 계산할 수 있습니다.
