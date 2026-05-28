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
title: Poisson Distribution
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]]
related:: [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/02_math-theory/discrete-mathematics/3. 관계와 함수/관계와 함수|관계와 함수]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/Discrete mathematics Assignment|Discrete mathematics Assignment]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/파라미터 추정|파라미터 추정]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/joint RVs|joint RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Independent RVs|Independent RVs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/random variables|random variables]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/Continuous Joint|Continuous Joint]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

#ComputerScience #확률과통계 

---
**포아송 분포(Poisson Distribution):**

**표기:** $X \sim \text{Poi}(\lambda)$

**설명:**
포아송 분포는 <mark style="background: yellow;">일정한 시간</mark> 또는 공간 간격 내에서 발생하는 사건의 수를 모델링하는 이산 확률 분포로, 해당 사건들의 평균 발생률을 고려합니다. 이는 사건들이 독립적으로 발생하며 주어진 간격 내에서 일정한 평균 발생률을 가정하는 상황에서 사용됩니다.

**주요속성**:
- **매개변수:**
	- $\lambda$ (람다): 주어진 간격 내에서 사건의 평균 발생률.

- **지원:**
	포아송 랜덤 변수의 지원은 비음수 정수 집합 $\{0, 1, 2, 3, \ldots \}$으로, 사건의 수를 나타냅니다.

- **PMF 식:**
	포아송 랜덤 변수 $X$의 확률 질량 함수(PMF)는 다음과 같습니다:
$$ P(X = k) = \frac{{e^{-\lambda} \lambda^k}}{{k!}}, \quad \text{for } k = \{0, 1, 2, \ldots\}$$

- **기대값:**
	포아송 랜덤 변수의 기대값 또는 평균 ($\mu$)은 매개변수와 동일합니다:
$$ E(X) = \mu = \lambda $$

- **분산:**
	포아송 랜덤 변수의 분산 ($\sigma^2$) 또한 매개변수와 동일합니다:
$$ \text{Var}(X) = \sigma^2 = \lambda $$

- **PMF 그래프:**
	포아송 분포의 PMF 그래프는 확률 질량 함수 플롯으로, x-축은 사건의 수 $k$를 나타내고 y-축은 확률 $P(X = k)$를 나타냅니다. 일반적으로 그래프는 평균 발생률 매개변수 $\lambda$를 중심으로 오른쪽으로 치우친 모양을 보입니다.

## Poisson Proof

Binomial 분포의 PMF는 다음과 같습니다:
$$ P(X = k) = \binom{n}{k} p^k (1-p)^{n-k} $$

Poisson 분포의 PMF는 다음과 같습니다:
$$ P(X = k) = \frac{{e^{-\lambda} \lambda^k}}{{k!}} $$

Binomial 분포의 매개변수 $n$과 $p$가 Poisson 분포의 매개변수 $\lambda$와 관련된 경우를 살펴보겠습니다. $np = \lambda$ 라고 가정하겠습니다.

이제 $n$이 충분히 크고 $p$가 충분히 작은 경우, $n$을 큰 값으로, $p$를 작은 값으로 극한을 취하면:

$$ \lim_{n \to \infty} P(X = k) = \lim_{n \to \infty} \binom{n}{k} p^k (1-p)^{n-k} $$

$$ = \lim_{n \to \infty} \frac{{n!}}{{k! (n-k)!}} p^k (1-p)^{n-k} $$

$$ = \lim_{n \to \infty} \frac{{n \cdot (n-1) \cdot (n-2) \cdot \ldots \cdot (n-k+1)}}{{k!}} p^k \left(1-\frac{{\lambda}}{n}\right)^n \left(1-\frac{{\lambda}}{n}\right)^{-k} $$

$$ = \frac{{\lambda^k}}{{k!}} \cdot \lim_{n \to \infty} \left(1-\frac{{\lambda}}{n}\right)^n $$

$$ = \frac{{\lambda^k}}{{k!}} \cdot e^{-\lambda} $$

위의 극한은 Poisson 분포의 PMF와 동일한 형태를 가지므로, Binomial 분포의 극한이 Poisson 분포가 됨을 증명했습니다.

### Poisson 분포 응용 예시
- 일정 주어진 시간 동안에 도착한 고객의 수
- 1킬로미터 도로에 있는 흠집의 수
- 일정 주어진 생산시간 동안 발생하는 불량 수
- 하룻동안 발생하는 출생자 수
- 어떤 시간 동안 롤게이트를 통과하는 차량의 수
- 어떤 페이지 하나를 완성하는 데 발생하는 오타의 발생률
- 어떤 특정 진도 이상의 지진이 발생하는 수 

Example)
Probability of k requests from this area in the next 1minute?
On average, $\lambda = 5$ requests per minute
$$P(X = k) = \frac{{e^{-5} 5^k}}{{k!}}$$
