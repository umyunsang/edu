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
title: Maximum A Posteriori
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]]
related:: [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/신호 특징 분석 결과|신호 특징 분석 결과]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/파라미터 추정|파라미터 추정]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/joint RVs|joint RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Independent RVs|Independent RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/random variables|random variables]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Continuous Joint|Continuous Joint]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
# Maximum A Posteriori

MAP(Most A Posteriori) 추정은 파라미터를 추정하는 또 다른 방법으로, MLE(Maximum Likelihood Estimator)와는 조금 다른 접근 방식을 취합니다. MLE가 데이터가 주어졌을 때 파라미터 값이 가장 가능성이 높은 값을 선택한다면, MAP는 주어진 데이터를 고려할 때 파라미터의 가장 가능성이 높은 값을 선택하는 것입니다. 수식적으로는 다음과 같이 표현됩니다.
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } f(\theta | X_1, X_2, \dots, X_n)$$

여기서 우리는 관측되지 않은 랜덤 변수의 조건부 확률을 계산하려고 합니다. 이럴 때, Bayes의 정리를 사용할 수 있습니다. 식을 Bayes의 정리를 사용하여 확장해 보겠습니다.
Intuition with Bayes' Theorem:$$P(\theta|data)=\frac{P(data|\theta)P(\theta)}{P(data)}$$
Solving for ${\theta}_{MAP}$ :
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } f(\theta | X_1, X_2, \dots, X_n)$$
$$= \underset{\theta}{\operatorname{argmax }} \text{ } \frac{f(X_1, X_2, \dots, X_n | \theta) g(\theta)}{h(X_1, X_2, \dots X_n)}$$

여기서 $f, g$ 및 $h$는 모두 확률 밀도입니다. 우리는 $f$가 다른 기능일 수 있음을 명시적으로 만들기 위해 다른 기호를 사용했습니다. 데이터가 IID로 가정되기 때문에 데이터가 주어진 $\theta$의 밀도를 분해할 수 있습니다. 또한 분모는 $\theta$에 대해 상수이므로 argmax에 영향을 미치지 않으며 해당 항목을 삭제할 수 있습니다. 수학적으로 표현하면 다음과 같습니다.
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } \prod_{i=1}^n f(X_i | \theta) g(\theta)$$

이전과 마찬가지로 MAP 함수의 로그의 argmax를 찾는 것이 더 편리할 것입니다. 이것이 파라미터의 MAP 추정의 최종 형태를 제공합니다.
$$\theta_{\text{MAP}} = \underset{\theta}{\operatorname{argmax }} \text{ } \left( \log (g(\theta)) + \sum_{i=1}^n \log(f(X_i | \theta)) \right)$$

베이지안 용어를 사용하면, MAP 추정은 $\theta$ 에 대한 "사후 분포"의 모드입니다. MLE 방정식과 MAP 방정식을 나란히 비교하면 MAP가 로그의 사전항을 더한 정확히 동일한 함수의 argmax임을 알 수 있습니다.
- log prior + log-likelihood

![[probability-statistics__Maximum A Posteriori.png]]

Laplace estimates : 각각의 Observe 에 +1 을 함  

## Conjugate distributions

위 내용은 MAP(Maximum A Posteriori) 추정을 준비하기 위해 사전 분포(prior distributions)에 대해 다룹니다. 각각의 다른 매개변수에 대한 합리적인 분포가 필요합니다. 예를 들어, 포아송 분포를 예측한다면 𝜆에 대한 사전 분포의 적절한 확률 변수 유형은 무엇일까요?

다음은 각각의 다른 매개변수 및 그들의 사전 분포로 가장 자주 사용되는 분포 목록입니다:

| Parameter                     | Distribution  |
| ----------------------------- | ------------- |
| Bernoulli (베르누이 분포) $p$       | Beta          |
| Binomial (이항 분포) $p$          | Beta          |
| Poisson (포아송 분포) $\lambda$    | Gamma         |
| Exponential (지수 분포) $\lambda$ | Gamma         |
| Multinomial $p_i$             | Dirichlet     |
| Normal $\mu$                  | Normal        |
| Normal $\sigma^2$             | Inverse Gamma |
