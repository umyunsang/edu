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
title: 23_naive_bayes
type: lecture
updated: '2026-05-05'
---


domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
up:: [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]]
related:: [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/문제풀이|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/신호 특징 분석 결과|신호 특징 분석 결과]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]]

---
### 나이브 베이즈 분류기 (Naive Bayes Classifier)

나이브 베이즈 분류기는 **분류 작업**을 위한 기계 학습 알고리즘입니다. 이 알고리즘은 "나이브 베이즈 가정"이라 불리는, 모든 특징들이 주어진 분류 레이블에 대해 서로 독립적이라는 실질적인 가정을 합니다. 이 가정은 실제로는 틀릴 수 있지만, 빠르고 간단한 알고리즘을 가능하게 하여 유용한 경우가 많습니다. 나이브 베이즈를 구현하기 위해 모델을 학습시키는 방법과 학습된 모델을 사용하여 예측하는 방법을 알아야 합니다.

#### 1. 학습 (모수 추정)

학습의 목표는 모든 특징 $X_i$에 대해 $P(Y)$와 $P(X_i | Y)$의 확률을 추정하는 것입니다. 여기서 $\hat{p}$는 확률의 추정치를 나타냅니다.

##### 최대 우도 추정 (MLE) 사용:
$$\hat{p}(X_i = x_i | Y = y) = \frac{ \text{Count}(X_i = x_i \text{ and } Y = y)}{\text{Count}(Y = y)}$$

##### 라플라스 MAP 추정 사용:
$$\hat{p}(X_i = x_i | Y = y) = \frac{ \text{Count}(X_i = x_i \text{ and } Y = y) + 1 }{\text{Count}(Y = y) + 2}$$

##### 최대 우도 추정 (MLE) 사용한 Y의 사전 확률:
$$\hat{p}(Y = y) = \frac{ \text{Count}(Y = y)}{\text{Total count of examples}}$$

#### 2. 예측

특징 벡터 $x = [x_1, x_2, \dots, x_m]$에 대해 $y$의 값을 다음과 같이 추정합니다:
$$\hat{y} = argmax_{y = \{0, 1\}} \left( \log \hat{p}(Y = y) + \sum_{i=1}^m \log \hat{p}(X_i = x_i | Y = y) \right)$$

작은 데이터셋의 경우 로그 버전의 $argmax$를 사용하지 않아도 됩니다. 

#### 3. 이론

분류에서 예측을 할 때 우리는 $P(Y = y | X = x)$를 최대화하는 $y$ 값을 선택하고자 합니다.
$$\hat{y} = argmax_{y = \{0, 1\}} P(Y = y | \mathbf{X} = \mathbf{x})$$

베이즈 정리를 사용하면 다음과 같이 전개됩니다:
$$\hat{y} = argmax_{y = \{0, 1\}} \frac{P(Y = y)P(\mathbf{X} = \mathbf{x} | Y = y)}{P(\mathbf{X} = \mathbf{x})}$$

여기서 $P(\mathbf{X} = \mathbf{x})$는 모든 $Y$에 대해 상수이므로 무시할 수 있습니다:
$$\hat{y} = argmax_{y = \{0, 1\}} P(Y = y)P(\mathbf{X} = \mathbf{x} | Y = y)$$

나이브 베이즈 가정을 사용하면 다음과 같이 간소화됩니다:
$$\hat{y} = argmax_{y = \{0, 1\}} P(Y = y) \prod_{i} P(X_i = x_i | Y = y)$$

로그를 취하면:
$$\hat{y} = argmax_{y = \{0, 1\}} \left( \log P(Y = y) + \sum_{i} \log P(X_i = x_i | Y = y) \right)$$

이러한 알고리즘은 학습과 예측 시 빠르고 안정적입니다.

#### 나이브 베이즈 가정

나이브 베이즈 가정은 각 특징 $x_i$가 주어진 $y$에 대해 서로 독립적이라는 것입니다. 이 가정은 실제로는 틀릴 수 있지만, 큰 특징 공간에서 데이터를 학습하고 예측하는 데 유용합니다. 이 가정을 통해 알고리즘을 간소화하고, 계산 복잡성을 줄일 수 있습니다.

---
![[Pasted image 20240610191037.png]]
![[Pasted image 20240610191323.png]]
![[Pasted image 20240610191347.png]]

---
![[Pasted image 20240610200513.png]]
![[Pasted image 20240610200631.png]]
![[Pasted image 20240610200658.png]]

---
![[Pasted image 20240610200738.png]]
![[Pasted image 20240610200757.png]]
