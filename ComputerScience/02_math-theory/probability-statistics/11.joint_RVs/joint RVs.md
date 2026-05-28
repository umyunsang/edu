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
title: Continuity correction
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]]
related:: [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/문제풀이|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/과제 번역|과제 번역]], [[ComputerScience/02_math-theory/discrete-mathematics/4. 그래프/그래프|그래프]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/02_math-theory/discrete-mathematics/1. 수학적 모델과 논리/수학적 모델과 논리|수학적 모델과 논리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/cnn|cnn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/svm|svm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/cifar10|cifar10]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/nand|nand]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/mle|mle]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
## **Normal Approximation**
### Website testing 문제
```
Quiz: Website testing
- 100 people are presented with a new wesite design.
- X = # people whose time on site increases
- PM assumes design has no effect. so assume P(stickier) = 0.5 independently.
- CEO will endorse the new design if X>=65.

What is P(CEO endorses change)? Give a numerical approximation.
```
**방법 1: 이항 분포**

$X$를 사이트에서 시간이 증가한 사람들의 수로 정의합니다. 우리는 $X$가 $n=100$ (시도 횟수)와 $p=0.5$ (성공 확률 - 이 경우 사이트를 더 잘 만드는 새로운 디자인이 시간을 더 많이 보내도록 만드는 확률)인 이항 분포를 따른다는 것을 알고 있습니다.

$P(\text{CEO가 변경을 지지함})$를 찾으려면 $X \geq 65$의 확률을 계산해야 합니다.

$$P(\text{CEO가 변경을 지지함}) = P(X \geq 65) = 1 - P(X < 65)$$

이제 $P(X < 65)$를 계산해 봅시다:

$$P(X < 65) = \sum_{x=0}^{64} \binom{100}{x} (0.5)^x (0.5)^{100-x}$$

이 식을 계산해서 계산 결과를 얻을 수 있습니다.

**방법 2: 이항 분포의 정규 근사**

이 방법은 이항 분포를 정규 분포로 근사하는 것입니다. 이를 위해 이항 분포의 평균 $\mu = np$와 분산 $\sigma^2 = np(1-p)$를 계산합니다. 여기서 $n=100$, $p=0.5$입니다.

따라서 이항 분포의 정규 근사는 평균 $\mu = 100 \times 0.5 = 50$ 및 표준 편차 $\sigma = \sqrt{100 \times 0.5 \times (1 - 0.5)} = 5$를 갖습니다.

이제 우리는 이 정규 분포를 사용하여 $X \geq 65$를 계산할 수 있습니다. 이를 위해 표준 정규 분포의 누적 분포 함수를 사용합니다.

$$P(X \geq 65) = 1 - P\left(Z < \frac{65 - \mu}{\sigma}\right)$$

여기서 $Z$는 표준 정규 분포를 나타냅니다. $\mu = 50$, $\sigma = 5$로 대입하여 $P(X \geq 65)$를 계산할 수 있습니다.

# Continuity correction
두 번째 방법의 결과가 첫 번째 방법의 결과와 살짝 다를 수 있는 이유는 연속성 수정(Continuity correction) 때문입니다.

연속성 수정은 이항 분포를 정규 분포로 근사할 때, 이산적인 이항 분포를 연속적인 정규 분포로 근사하는 과정에서 발생하는 근사 오차를 보정하는데 사용됩니다. 이 항상 정확하지는 않지만, 일반적으로 좀 더 정확한 근사를 제공합니다.

첫 번째 방법에서는 이항 분포를 정확하게 사용하여 $P(X \geq 65)$를 계산합니다. 하지만 두 번째 방법에서는 정규 분포를 사용하여 이항 분포를 근사하므로 연속성 수정을 적용해야 합니다. 

연속성 수정을 적용하면 $X \geq 65$를 $X > 64.5$로 취급합니다. 즉, 이항 분포의 확률을 정규 분포의 확률로 근사할 때, 64.5와 같은 연속적인 값을 기준으로 하여 계산을 합니다.

따라서 두 번째 방법에서는 $P(X \geq 65)$를 $P(X > 64.5)$로 근사하여 계산하므로, 조금 다른 결과를 얻을 수 있습니다.

# Discrete Joint RVs
이산형 결합 분포는 주로 다음과 같은 표기법을 사용하여 설명됩니다:

1. 두 개의 이산형 확률 변수를 $X$와 $Y$로 정의합니다.
2. 각 확률 변수는 특정 값들 중 하나를 취할 수 있으며, 이를 각각 $x_1, x_2, ..., x_n$ 및 $y_1, y_2, ..., y_m$으로 나타냅니다.
3. 이 두 확률 변수가 함께 발생하는 경우를 나타내기 위해 결합 확률 질량 함수 $P(X=x_i, Y=y_j)$를 사용합니다. 여기서 $P$는 확률을 나타내며, $X=x_i$는 확률 변수 $X$가 값 $x_i$를 가질 때를 의미하고, $Y=y_j$는 확률 변수 $Y$가 값 $y_j$를 가질 때를 의미합니다.
4. 결합 확률 질량 함수는 모든 가능한 $X$와 $Y$의 값에 대한 확률을 제공합니다. 즉, $P(X=x_i, Y=y_j)$는 모든 $i$와 $j$에 대해 정의됩니다.

### **The marginal distributions**
마진 분포(Marginal distribution)는 결합 분포에서 특정 확률 변수에 대한 확률 분포를 나타냅니다. 이것은 다른 모든 변수에 대한 정보를 무시하고 해당 변수에만 집중합니다. 

수학적으로, 두 개의 이산형 확률 변수 $X$와 $Y$의 결합 확률 질량 함수를 $P(X=x_i, Y=y_j)$로 표현할 때, $X$의 마진 분포는 다음과 같이 정의됩니다:
$$P(X=x_i) = \sum_{j} P(X=x_i, Y=y_j)$$

즉, $X$의 각 값 $x_i$에 대한 확률은 해당 값을 가지고 있을 때 $Y$의 모든 가능한 값에 대한 결합 확률을 합산한 것입니다. 마찬가지로, $Y$의 마진 분포는 다음과 같이 정의됩니다:
$$P(Y=y_j) = \sum_{i} P(X=x_i, Y=y_j)$$
이것은 $Y$의 각 값 $y_j$에 대한 확률을 해당 값을 가지고 있을 때 $X$의 모든 가능한 값에 대한 결합 확률을 합산한 것입니다.

## Multinomial Random Variable
다양한 범주 또는 결과가 있는 실험을 고려할 때, Multinomial Random Variable은 각 범주에 속하는 사건의 수를 나타내는 이산형 확률 변수입니다. 이는 이항 분포를 일반화한 것으로 볼 수 있습니다.

수학적으로, Multinomial Random Variable은 다음과 같이 정의됩니다:

1. 실험을 $n$번 시행합니다.
2. 각 시행에서는 여러 개의 범주 중 하나를 선택합니다. 이때, 각 범주에 대한 선택 확률은 각 시행마다 동일합니다.
3. 범주들은 상호 배타적이며, 한 시행에서는 하나의 범주만 선택됩니다.
4. Multinomial Random Variable은 각 범주에 속하는 사건의 수를 나타냅니다.

수식적으로, $k$개의 범주가 있다고 가정하고 $X_1, X_2, ..., X_k$를 각 범주에 대한 Multinomial Random Variable이라고 합시다. 또한 $p_1, p_2, ..., p_k$를 각 범주가 선택될 확률이라고 하면, 이러한 변수들은 다음과 같은 특성을 갖습니다:

1. 각 $X_i$는 범주 $i$에 속하는 사건의 수를 나타냅니다.
2. 모든 $X_i$의 합은 실험의 총 시행 횟수 $n$과 같아야 합니다. 즉, $X_1 + X_2 + ... + X_k = n$이어야 합니다.
3. 각 $X_i$는 이항 분포를 따릅니다. 따라서 $X_i$의 확률 질량 함수는 다음과 같습니다:
$$P(X_i = x_i) = \binom{n}{x_i} \cdot p_i^{x_i} \cdot (1 - p_i)^{n - x_i}$$
여기서 $\binom{n}{x_i}$는 $n$개 중에서 $x_i$개를 선택하는 조합을 의미하며, $p_i$는 범주 $i$가 선택될 확률을 나타냅니다.
