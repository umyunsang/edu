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
title: MLE
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]]
related:: [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/문제풀이|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/Discrete mathematics Assignment|Discrete mathematics Assignment]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/02_math-theory/mathematical-logic/동아설계도|동아설계도]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/cnn|cnn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/svm|svm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/cifar10|cifar10]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/nand|nand]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/mle|mle]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

### 최대 우도 추정 (Maximum Likelihood Estimation, )

최대 우도 추정(MLE)은 관찰된 데이터를 가장 잘 설명하는 파라미터 $\theta$를 선택하는 알고리즘입니다. 우리가 파라미터를 추정하는 데 사용할 데이터는 $n$개의 독립적이고 동일하게 분포된(IID) 샘플 $X_1, X_2, \dots, X_n$입니다.

#### 우도 (Likelihood)

데이터가 동일하게 분포된다고 가정했으므로, 이들은 같은 확률 질량 함수(PMF) 또는 같은 확률 밀도 함수(PDF)를 가집니다. 우리는 $f(X|\theta)$로 이 공유된 PMF 또는 PDF를 나타낼 것입니다. 이 표기법은 두 가지 중요한 점을 포함합니다. 첫째, 우도가 파라미터 $\theta$에 따라 달라진다는 것을 나타내기 위해 조건부를 포함했습니다. 둘째, 이산 분포와 연속 분포 모두에 대해 동일한 기호 $f$를 사용합니다.

우도와 확률의 차이는 무엇일까요? 이산 분포의 경우, 우도는 데이터의 확률 질량 또는 결합 확률 질량과 동의어입니다. 연속 분포의 경우, 우도는 데이터의 확률 밀도를 나타냅니다.

각 데이터 포인트가 독립적이라고 가정했기 때문에, 모든 데이터의 우도는 각 데이터 포인트의 우도의 곱입니다. 수학적으로, 파라미터 $\theta$가 주어진 데이터의 우도는 다음과 같습니다:

$$
L(\theta) = \prod_{i=1}^n f(X_i|\theta)
$$

다른 파라미터 값에 대해 데이터의 우도는 달라집니다. 올바른 파라미터를 가지고 있다면, 데이터는 잘 맞을 것이고 그렇지 않은 경우는 그렇지 않을 것입니다. 따라서 우리는 우도를 파라미터 $\theta$의 함수로 작성합니다.

#### 최대화 (Maximization)

MLE의 목표는 이전 섹션의 우도 함수를 최대화하는 파라미터 $\theta$ 값을 선택하는 것입니다. 우리는 파라미터의 최적 선택을 나타내기 위해 $\hat{\theta}$를 사용할 것입니다. 형식적으로, MLE는 다음과 같이 가정합니다:

$$
\hat{\theta} = \underset{\theta}{\operatorname{argmax }} \text{ }L(\theta)
$$

argmax는 함수가 최대화되는 도메인의 값을 의미합니다. 이는 어떤 차원의 도메인에도 적용될 수 있습니다.

argmax의 멋진 속성은 로그가 단조 함수이기 때문에 함수의 로그의 argmax가 함수의 argmax와 동일하다는 것입니다. 로그는 수학을 더 간단하게 만들어줍니다. 따라서 MLE를 위해 먼저 로그 우도 함수(LL)를 작성합니다:

$$
LL(\theta) = \log L(\theta) = \log \prod_{i=1}^n f(X_i|\theta) = \sum_{i=1}^n \log f(X_i|\theta)
$$

최대 우도 추정기를 사용하려면, 먼저 주어진 파라미터의 로그 우도를 작성하고, 로그 우도 함수를 최대화하는 파라미터 값을 선택합니다. Argmax는 여러 가지 방법으로 계산할 수 있으며, 이 수업에서 다루는 모든 방법은 함수의 1차 도함수를 계산하는 것을 포함합니다.

### 베르누이 MLE 추정

첫 번째 예로, 우리는 베르누이 분포의 $p$ 파라미터를 추정하는 MLE를 사용할 것입니다. 우리는 $n$개의 데이터 포인트를 기반으로 추정할 것이며, 이를 IID 랜덤 변수 $X_1, X_2, \dots, X_n$으로 참조할 것입니다. 이 랜덤 변수들은 모두 동일한 베르누이 분포에서 샘플링된 것으로 가정됩니다: $X_i \sim \text{Ber}(p)$. 우리는 이 $p$ 값을 알아내고자 합니다.

MLE의 첫 번째 단계는 최적화할 수 있는 함수로서 베르누이의 우도를 작성하는 것입니다. 베르누이는 이산 분포이므로, 우도는 확률 질량 함수입니다.

베르누이 $X$의 확률 질량 함수는 다음과 같이 쓸 수 있습니다: $f(x) = p^x(1-p)^{1-x}$.

이제 MLE 추정을 해봅시다:

$$
L(\theta) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i}
$$

$$
LL(\theta) = \sum_{i=1}^n \log p^{x_i}(1-p)^{1-x_i}$$
$$= \sum_{i=1}^n x_i (\log p) + (1 - x_i) \log(1-p)$$ 
$$= Y \log p + (n - Y) \log(1-p)
$$

여기서 $Y = \sum_{i=1}^n x_i$입니다.

이제 로그 우도 방정식을 얻었으므로, 로그 우도를 최대화하는 $p$ 값을 선택해야 합니다. 이를 위해 함수의 1차 도함수를 찾아 0으로 설정합니다:

$$
\frac{\partial LL(p)}{\partial p} = Y \frac{1}{p} + (n - Y) \frac{-1}{1-p} = 0
$$

따라서,

$$
\hat{p} = \frac{Y}{n} = \frac{\sum_{i=1}^n x_i}{n}$$

결국, MLE 추정값은 단순히 샘플 평균이 됩니다.

### 정규 MLE 추정

다음으로, 정규 분포의 최적 파라미터 값을 추정해 봅시다. 우리는 $n$개의 정규 분포에서 샘플링된 IID 랜덤 변수 $X_1, X_2, \dots, X_n$에 접근할 수 있습니다. 각 $X_i$는 $\mu = \theta_0, \sigma^2 = \theta_1$ 인 $N(\mu, \sigma^2)$에서 샘플링된 것으로 가정합니다. 이 경우 $\theta$는 평균( $\mu$ ) 및 분산( $\sigma^2$ )이라는 두 값을 가진 벡터입니다.

$$
L(\theta) = \prod_{i=1}^n f(X_i|\theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\theta_1}} e^{-\frac{(X_i - \theta_0)^2}{2\theta_1}}
$$

$$
LL(\theta) = \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\theta_1}} e^{-\frac{(X_i - \theta_0)^2}{2\theta_1}} = \sum_{i=1}^n \left[ - \log(\sqrt{2\pi\theta_1}) - \frac{1}{2\theta_1}(X_i - \theta_0)^2 \right]
$$

이제, 로그 우도 함수를 최대화하는 $\theta$ 값을 선택해야 합니다. 이를 위해 $LL$ 함수에 대해 $\theta_0$ 및 $\theta_1$에 대한 편미분을 계산하고 두 방정식을 모두 0으로 설정한 다음 $\theta$ 값을 구합니다. 그 결과는 다음과 같습니다:

$$
{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n X_i, \quad {\sigma^2}_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - {\mu}_{MLE})^2
$$
