---
aliases: []
course: artificial-intelligence
created: '2024-05-11'
date: '2024-05-11'
semester: 2-1
source: ''
status: seedling
tags:
- cs/ai
- cs/dl
- cs/ml
- type/project
title: Training dataset 다운로드
type: project
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/인공지능 인터페이스|인공지능 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]]
up:: [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability|Probability]]
related:: [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/02_math-theory/probability-statistics/22.map/Maximum A Posteriori|Maximum A Posteriori]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous RVs|Continuous RVs]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/연습문제 풀이|연습문제 풀이]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/Continuous Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes theorem 문제 풀이|문제 풀이]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/02_math-theory/probability-statistics/3.Probability/Probability 문제 풀이|문제 풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/Joint Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Expectation|Expectation]], [[ComputerScience/02_math-theory/probability-statistics/11.joint_RVs/joint RVs|joint RVs]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/02_math-theory/probability-statistics/4.Bayes_theorem/Bayes' Theorem|Bayes' Theorem]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Distribution|Normal Distribution]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance|Variance]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/19_sampling|19_sampling]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/지진 문제|문제풀이]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/7-0.Variance/Variance 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/02_math-theory/probability-statistics/10.Normal_RV/Normal Random Variable 문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/23.naive_bayes/23_naive_bayes|23_naive_bayes]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Sampling|Sampling]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/02_math-theory/probability-statistics/15.General_inference/16.Continous_joint_probability-1/Continuous Joint|Continuous Joint]], [[ComputerScience/02_math-theory/probability-statistics/7-1.Bernoulli_RV/Bernoulli Distribution|Bernoulli Distribution]], [[ComputerScience/02_math-theory/probability-statistics/1.Counting/Counting|Counting]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence|Independence]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/More Discrete Distributions (시험 X)|More Discrete Distributions (시험 X)]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/02_math-theory/probability-statistics/2.Combinations/Combinations|Combinations]], [[ComputerScience/02_math-theory/probability-statistics/6.Random_Variables/Random Variables|Random Variables]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/02_math-theory/probability-statistics/5.Independence/Independence 문제 풀이|문제 풀이]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/02_math-theory/probability-statistics/7-2.Binomial_RV/Binomial Distribution|Binomial Distribution]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/파라미터 추정|파라미터 추정]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/인공지능 지식그래프|인공지능]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/인공지능 지식그래프|인공지능]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/인공지능 근거 인덱스|인공지능 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/artificial-intelligence/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/artificial-intelligence/learning rate|learning rate]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/artificial-intelligence/loss function|loss function]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/artificial-intelligence/cnn|cnn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/artificial-intelligence/CIFAR10|CIFAR10]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---

모멘텀(Momentum)은 경사 하강법(Gradient Descent) 최적화 알고리즘의 한 종류로, 기울기 갱신 시 직전 기울기를 고려하여 업데이트하는 방법입니다. 이는 기존의 경사 하강법보다 빠르게 수렴하고, 지역 최솟값(local minimum)에 빠지지 않고 전역 최솟값(global minimum)으로 빠르게 수렴하는 데 도움이 됩니다.

모멘텀은 다음과 같이 계산됩니다.

$$
\begin{align*}
v_t &= \beta v_{t-1} + \alpha \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t - v_t
\end{align*}
$$

여기서,
- $v_t$는 시간 $t$에서의 모멘텀(momentum) 벡터입니다.
- $\beta$ 는 모멘텀 상수(momentum coefficient)로, 일반적으로 0.9와 같은 값으로 설정됩니다.
- $\alpha$ 는 학습률(learning rate)입니다.
- $nabla J(\theta_t$) 는 현재 매개변수에 대한 손실 함수의 기울기(gradient)입니다.
- $\theta_t$ 는 시간 $t$ 에서의 매개변수(parameters)입니다.

모멘텀은 이전의 속도 벡터 $v_{t-1}$ 와 현재의 기울기 $\nabla J(\theta_t)$ 를 고려하여 새로운 속도 벡터 $v_t$ 를 계산합니다. 이후, 속도 벡터 $v_t$ 를 사용하여 매개변수를 업데이트합니다.

모멘텀을 사용하면 경사 하강법이 효율적으로 수렴하고 지역 최솟값에 빠지지 않는 것을 도와줍니다. 특히, 비등방성(anisotropic) 및 길쭉한(sloped) 기울기 표면에서 경사 하강법을 가속화하여 빠르게 수렴할 수 있습니다.

종합하면, 모멘텀은 경사 하강법 최적화 알고리즘의 한 종류로서, 기울기 갱신 시 직전 기울기를 고려하여 업데이트하여 수렴 속도를 높이고 안정성을 향상시키는 데 사용됩니다.

```python
import torch  
import torch.nn as nn  
import torchvision.datasets as dataset  
import torchvision.transforms as transform  
from torch.utils.data import DataLoader  
  
# Training dataset 다운로드  
mnist_train = dataset.MNIST(root="./",  # 데이터셋을 저장할 위치  
                            train=True,  
                            transform=transform.ToTensor(),  
                            download=True)  
# Testing dataset 다운로드  
mnist_test = dataset.MNIST(root='./',  
                           train=False,  
                           transform=transform.ToTensor(),  
                           download=True)  
  
  
# Single Layer Perceptron 모델 정의  
class SLP(nn.Module):  
    def __init__(self):  
        super(SLP, self).__init__()  
  
        self.fc = nn.Linear(in_features=784, out_features=10)  
  
    def forward(self, x):  
        x = x.view(-1, 28 * 28)  
        y = self.fc(x)  
  
        return y  
  
  
# Hyper-parameters 지정  
batch_size = 100  
learning_rate = 0.1  
training_epochs = 15  
loss_function = nn.CrossEntropyLoss()  
network = SLP()  
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum= 0.9)  
  
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  
  
data_loader = DataLoader(dataset=mnist_train,  
                         batch_size=batch_size,  
                         shuffle=True,  
                         drop_last=True)  
  
# Perceptron 학습을 위한 반복문 선언  
for epoch in range(training_epochs):  
    avg_cost = 0  
    total_batch = len(data_loader)  
  
    for img, label in data_loader:  
        pred = network(img)  
  
        loss = loss_function(pred, label)  
        optimizer.zero_grad()  # gradient 초기화  
        loss.backward()  
        optimizer.step()  
  
        avg_cost += loss / total_batch  
  
    print('Epoch: %d  LR: %f  Loss = %f' % (epoch + 1, optimizer.param_groups[0]['lr'], avg_cost))  
    # scheduler.step()  
  
print('Learning finished')  
  
# 학습이 완료된 모델을 이용해 정답률 확인  
with torch.no_grad():  # test에서는 기울기 계산 제외  
  
    img_test = mnist_test.data.float()  
    label_test = mnist_test.targets  
  
    prediction = network(img_test)  # 전체 test data를 한번에 계산  
  
    correct_prediction = torch.argmax(prediction, 1) == label_test  
    accuracy = correct_prediction.float().mean()  
    print('Accuracy:', accuracy.item())
```
