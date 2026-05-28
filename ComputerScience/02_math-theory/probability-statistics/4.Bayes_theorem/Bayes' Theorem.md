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
title: Bayes' Theorem
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/1단계 기초 구축 인터페이스|1단계 기초 구축 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/확률통계 인터페이스|확률통계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]]
related:: [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/문제풀이|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/02_math-theory/mathematical-logic/동아설계도|동아설계도]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/과제 번역|과제 번역]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/ASD Feature 발굴|ASD Feature 발굴]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/02_math-theory/discrete-mathematics/1. 수학적 모델과 논리/수학적 모델과 논리|수학적 모델과 논리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/확률통계 근거 인덱스|확률통계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/cnn|cnn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/svm|svm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/cifar10|cifar10]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/nand|nand]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/probability-statistics/mle|mle]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

#ComputerScience #확률과통계 

---
### Bayes' 이론

- **핵심내용**:
  - Bayes' 이론은 주어진 증거를 고려하여 사건의 사후 확률을 추론하는 방법입니다.
  - 주어진 증거 $E$가 주어졌을 때, 사건 $B$의 발생 확률을 업데이트합니다.

- **수식**:
  - Bayes' 이론은 다음과 같이 표현됩니다:
    $$ P(B|E) = \frac{P(E|B) \cdot P(B)}{P(E)} $$
    여기서:
    - $P(B|E)$: 사후 확률 (증거 $E$가 주어졌을 때 사건 $B$의 확률)
    - $P(E|B)$: 가능도 (사건 $B$가 주어졌을 때 증거 $E$의 확률)
    - $P(B)$: 사전 확률 (사건 $B$의 확률)
    - $P(E)$: 주변 확률 (증거 $E$의 확률)

- **사후 확률 계산**:
  1. **가능도 $P(E|B)$ 계산**: 주어진 사건 $B$에 대한 증거 $E$의 발생 확률을 계산합니다.
  2. **사전 확률 $P(B)$ 계산**: 사전 정보를 바탕으로 사건 $B$의 발생 확률을 계산합니다.
  3. **주변 확률 $P(E)$ 계산**: 증거 $E$가 발생할 확률을 계산합니다.
  4. **Bayes' 이론을 적용하여 사후 확률 $P(B|E)$ 계산**:
     $$ P(B|E) = \frac{P(E|B) \cdot P(B)}{P(E)} $$

- **사후 확률의 의미**:
  - Bayes' 이론은 증거를 통해 사건의 확률을 업데이트하여 더 나은 추론을 할 수 있도록 합니다. 증거 $E$가 주어졌을 때, 사건 $B$의 발생 확률을 추론합니다.

### 주변 확률 $P(E)$ 계산

- **주변 확률**:
  - 주변 확률은 특정 사건이 발생할 확률입니다.
  - 여러 사건이 동시에 발생할 수 있는 경우, 모든 가능한 경우의 확률을 합한 것입니다.

- **계산 방법**:
  - 주변 확률 $P(E)$를 계산하기 위해 베이즈 이론을 활용하여 다음과 같이 계산할 수 있습니다:
    $$ P(E) = P(E|B) \cdot P(B) + P(E|B') \cdot P(B') $$
    여기서 $B'$는 사건 $B$의 여집합을 나타냅니다.

### Bayes' 이론의 확장: Unknown Normalization Constant

#### 핵심내용:
Bayes' 이론은 주어진 증거를 바탕으로 사후 확률을 추론하는데 사용됩니다. 그러나 때로는 증거의 주변 확률이 계산하기 어려운 경우가 있습니다. 이때 사용되는 Unknown Normalization Constant는 이 확률을 추정하는 데 도움이 됩니다.

#### 수식:
Unknown Normalization Constant를 적용한 Bayes' 이론의 수식은 다음과 같이 표현됩니다:
$$ P(B|E) = \frac{P(E|B) \cdot P(B)}{\sum_{i} P(E|B_i) \cdot P(B_i)} $$
여기서:
- $P(B|E)$: 사후 확률 (증거 $E$가 주어졌을 때 사건 $B$의 확률)
- $P(E|B)$: 가능도 (사건 $B$가 주어졌을 때 증거 $E$의 확률)
- $P(B)$: 사전 확률 (사건 $B$의 확률)
- $\sum_{i} P(E|B_i) \cdot P(B_i)$: 주변 확률의 추정치로서, 모든 가능한 사건 $B_i$에 대한 가능도와 사전 확률의 합

### Bayes' 이론과 일반적인 총 확률의 법칙

#### 핵심내용:
Bayes' 이론은 일반적인 총 확률의 법칙과 결합하여 복잡한 문제를 해결하는 데 사용됩니다. 이 법칙은 주어진 증거의 조건 하에서 사전 확률을 계산하는 데 유용합니다.

#### 수식:
Bayes' 이론과 총 확률의 법칙을 결합한 수식은 다음과 같습니다:
$$ P(B|E) = \frac{P(E|B) \cdot P(B)}{P(E)} $$
$$ P(E) = \sum_{i} P(E|B_i) \cdot P(B_i) $$
여기서:
- $P(B|E)$: 사후 확률 (증거 $E$가 주어졌을 때 사건 $B$의 확률)
- $P(E|B)$: 가능도 (사건 $B$가 주어졌을 때 증거 $E$의 확률)
- $P(B)$: 사전 확률 (사건 $B$의 확률)
- $\sum_{i} P(E|B_i) \cdot P(B_i)$: 증거 $E$의 주변 확률으로서, 모든 가능한 사건 $B_i$에 대한 가능도와 사전 확률의 합

### 예시: 병 검사

가정:
- 어떤 질병에 걸렸을 때 특정 검사는 양성 또는 음성 결과를 보여줍니다.
- 검사는 95%의 정확도를 가지며, 질병에 걸린 사람이 양성 결과를 보일 확률은 99%입니다.
- 전체 인구 중 1%가 해당 질병에 걸렸다고 가정합니다.

주어진 정보:
- 사전 확률 $P(A)$ : 질병에 걸렸을 확률 = 0.01
- 가능도 $P(B|A)$ : 양성 결과가 나올 확률 = 0.99
- 가능도 $P(B|A')$ : 음성 결과가 나올 확률 = 0.05 (검사의 정확도로부터 계산됨)

우리는 주어진 검사 결과가 양성인 경우, 실제로 질병에 걸렸을 확률을 계산하고 싶습니다. 이를 Bayes' 이론을 사용하여 계산할 수 있습니다.

#### 1. 사후 확률 $P(A|B)$ 계산:

Bayes' 이론을 적용하여 주어진 검사 결과가 양성인 경우, 질병에 걸렸을 확률인 $P(A|B)$를 계산합니다.

$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

#### 2. 주변 확률 $P(B)$ 계산:

주변 확률은 양성 결과가 나올 전체 확률로서, 다음과 같이 계산할 수 있습니다.

$P(B) = P(B|A) \cdot P(A) + P(B|A') \cdot P(A')$

여기서,
- $P(A')$ : 질병에 걸리지 않을 확률 = 1 - $P(A)$
- $P(B|A')$ : 걸리지 않은 사람이 양성 결과를 보일 확률 = 1 - 검사의 정확도

#### 3. 주어진 데이터를 바탕으로 계산:

주어진 정보를 바탕으로 위의 수식에 값을 대입하여 $P(A|B)$를 계산합니다.

이렇게 계산된 $P(A|B)$는 양성 결과가 주어졌을 때 실제로 질병에 걸렸을 확률을 나타냅니다.

이제 이를 수식으로 계산해보겠습니다.

#### 1. 주변 확률 $P(B)$ 계산:

$P(B) = P(B|A) \cdot P(A) + P(B|A') \cdot P(A')$
$P(B) = 0.99 \times 0.01 + 0.05 \times 0.99$
$P(B) = 0.0099 + 0.0495$
$P(B) = 0.0594$

#### 2. 사후 확률 $P(A|B)$ 계산:

$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$
$P(A|B) = \frac{0.99 \times 0.01}{0.0594}$
$P(A|B) = \frac{0.0099}{0.0594}$
$P(A|B) ≈ 0.1667$

따라서, 주어진 검사 결과가 양성인 경우, 실제로 질병에 걸렸을 확률은 약 0.1667 또는 약 16.67%입니다.
