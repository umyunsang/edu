---
aliases: []
course: probability-statistics
created: '2026-03-19'
date: '2026-03-19'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-1
source: ''
status: seedling
tags:
- math/probability
- math/statistics
- type/lecture
title: Bayes theorem 문제 풀이
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]]
related:: [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/중간 퀴즈|중간 퀴즈]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 퀴즈|기말 퀴즈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/이상 탐지(ASD)를 위한 최적의 Feature Engineering|이상 탐지(ASD)를 위한 최적의 Feature Engineering]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/02_math-theory/discrete-mathematics/3. 관계와 함수/관계와 함수|관계와 함수]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/과제 번역|과제 번역]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/파라미터 추정|파라미터 추정]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/joint RVs|joint RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Independent RVs|Independent RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/random variables|random variables]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Continuous Joint|Continuous Joint]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
### **퀴즈 1: 스팸 이메일 탐지**
```
Detecting spam email
- 60 % of all email in 2016 is spam.
- 20 % of spam has the word "Dear"
- 1 % of non-spam (aka ham) has the word "Dear"

You get an email with the word "Dear" in it.
What is the probability that the email is spam?
```

주어진 정보:
- 스팸 이메일 확률: $P(F)=0.6$
- 스팸에서 "Dear" 단어가 포함된 확률: $P(E|F)=0.2$
- 스팸이 아닌 이메일에서 "Dear" 단어가 포함된 확률: $P(E|F^c)=0.01$

우리가 구하려는 것:
"Dear" 단어가 포함된 이메일이 스팸일 확률인 $P(F|E)$

베이즈 정리를 사용하여 계산합니다:
$$P(F|E) = \frac{P(E|F) \cdot P(F)}{P(E)}$$

전체 확률의 법칙을 사용하여 $P(E)$를 계산합니다:
	$P(E) = P(E|F) \cdot P(F) + P(E|F^c) \cdot P(F^c)$

$P(E)$를 계산해보겠습니다:
	$P(E) = 0.2 \times 0.6 + 0.01 \times 0.4 = 0.122$

이제 베이즈 정리에 값을 대입하여 $P(F|E)$를 구합니다:
	$P(F|E) = \frac{0.2 \times 0.6}{0.122} = \frac{0.12}{0.122} \approx 0.9836$

---
### **퀴즈 2: 지카 바이러스 테스트**
```
Zika Testing
- A test is 98%, effective at detecting Zika("true positive")
- However, the test has a "false positive" rate of 1%.
- 0.5% of the US population has Zika.

What is the likelihood you have Zika if you test positive?
why would you expect this number?
```

주어진 정보:
- 지카 바이러스 감염 확률: $P(F)=0.005$
- 지카 바이러스에 감염되었을 때 양성 반응이 나올 확률: $P(E|F)=0.98$
- 지카 바이러스에 감염되지 않았을 때 양성 반응이 나올 확률: $P(E|F^c)=0.01$

우리가 구하려는 것:
양성 반응이 나왔을 때 실제로 지카 바이러스에 감염되어 있을 확률인 $P(F|E)$

베이즈 정리를 사용하여 계산합니다:
	$P(F|E) = \frac{P(E|F) \cdot P(F)}{P(E)}$

전체 확률의 법칙을 사용하여 $P(E)$를 계산합니다:
	$P(E) = P(E|F) \cdot P(F) + P(E|F^c) \cdot P(F^c)$

$P(E)$를 계산해보겠습니다:
	$P(E) = 0.98 \times 0.005 + 0.01 \times 0.995 = 0.0149$

이제 베이즈 정리에 값을 대입하여 $P(F|E)$를 구합니다:
	$P(F|E) = \frac{0.98 \times 0.005}{0.0149} \approx \frac{0.0049}{0.0149} \approx 0.3289$

---
### **퀴즈 2-1: What is $P(F|E^c)$ ?**

우리가 구하려는 것은 음성 반응이 나왔을 때 실제로 지카 바이러스에 감염되어 있을 확률인 $P(F|E^c)$입니다.

베이즈 정리를 사용하여 계산합니다:
$$P(F|E^c) = \frac{P(E^c|F) \cdot P(F)}{P(E^c)}$$

전체 확률의 법칙을 사용하여 $P(E^c)$를 계산합니다:
	$P(E^c) = P(E^c|F) \cdot P(F) + P(E^c|F^c) \cdot P(F^c)$

$P(E^c)$를 계산해보겠습니다:
	$P(E^c) = (1-P(E|F)) \cdot P(F) + (1-P(E|F^c)) \cdot P(F^c)$

이제 베이즈 정리에 값을 대입하여 $P(F|E^c)$를 구하겠습니다.

$P(F|E^c)$를 구하기 위해 다음과 같이 계산합니다.

먼저, 전체 확률의 법칙을 사용하여 $P(E^c)$를 계산합니다:
	$P(E^c) = (1-P(E|F)) \cdot P(F) + (1-P(E|F^c)) \cdot P(F^c)$

여기서,
- $P(E|F)$는 지카 바이러스에 감염되어 있을 때 양성 반응이 나올 확률이므로 0.98입니다.
- $P(E|F^c)$는 지카 바이러스에 감염되어 있지 않을 때 양성 반응이 나올 확률이므로 0.01입니다.
- $P(F)$는 지카 바이러스 감염 확률이므로 0.005입니다.
- $P(F^c)$는 지카 바이러스 미감염 확률로 1에서 $P(F)$를 뺀 값, 즉 $1 - 0.005 = 0.995$입니다.

이제 위의 값들을 대입하여 $P(E^c)$를 계산합니다.

	$P(E^c) = (1 - 0.98) \cdot 0.005 + (1 - 0.01) \cdot 0.995$
	
	$P(E^c) = (0.02) \cdot 0.005 + (0.99) \cdot 0.995$
	
	$P(E^c) = 0.001 + 0.985$
	
	$P(E^c) = 0.986$

이제 $P(E^c)$를 사용하여 $P(F|E^c)$를 계산합니다. 베이즈 정리에 따르면,

$$P(F|E^c) = \frac{P(E^c|F) \cdot P(F)}{P(E^c)}$$

여기서,
- $P(E^c|F)$는 지카 바이러스에 감염되어 있을 때 음성 반응이 나올 확률이므로 1에서 0.98을 뺀 값, 즉 $1 - 0.98 = 0.02$입니다.

따라서 $P(F|E^c)$는 다음과 같이 계산됩니다:
	
	$P(F|E^c) = \frac{0.02 \cdot 0.005}{0.986}$
	
	$P(F|E^c) ≈ \frac{0.0001}{0.986}$
	
	$P(F|E^c) ≈ 0.0001014$

따라서, 지카 바이러스에 감염되어 있지 않은데 양성 반응이 나왔을 때, 즉 $P(F|E^c)$는 약 0.01014%입니다.
