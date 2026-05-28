---
aliases: []
course: large-language-models
created: '2025-01-20'
date: '2025-01-20'
semester: elective
source: ''
status: seedling
tags:
- cs/llm
- cs/nlp
- type/lecture
title: LLM 모델 이해
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/ASD Feature 발굴|ASD Feature 발굴]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
## LLM 모델 이해
#### 자연어처리(NLP) 기술의 발전
![[elective_LLM__Pasted image 20250120101836.png]]
- **n-gram 모델** : 연속된 단어의 그룹을 기반으로 확률을 계산하는 모델
- **Hidden Markov Model (HMM)** : 관찰할 수 없는(숨겨진) 상태와 관찰 가능한 데이터 사이
의 확률 관계를 모델링
- **Recurrent Neural Networks (RNN)** : 순차적 데이터(텍스트, 음성, 동영상 등)를 처리하기
위해 설계된 신경망 구조

![[elective_LLM__Pasted image 20250120103621.png]]
- **Sequence-to-Sequence (Seq2Seq)** : Encoder(문맥이해, 압축)-Decoder(문장 생성) 아키텍 처 기반, 입력 시퀀스를 출력 시퀀스로 변환하는 모델 구조 
- **Attention** : 입력 시퀀스 내의 중요한 부분에 집중하여, 시퀀스 처리 작업의 성능을 향상 시키는 신경망 구조 
- **Transformer** : 어텐션 메커니즘을 활용하여 순차적 계산을 최소화하고 병렬 처리를 극대화 한 신경망 아키텍처

#### 자연어처리(NLP) Process

1. **Cleaning** : 불필요한 데이터 제거(예: HTML 태그, 특수 문자), 노이즈 감소 
	-  데이터의 품질을 보장하고 다음 단계의 처리를 용이하게 합니다. 
2. **Tokenization** : 텍스트를 작은 단위(토큰)로 나누는 과정 
3. **정규화** (Normalization) : 데이터의 일관성을 높이고, 처리 과정의 복잡성을 줄이고 텍스트 데이터의 변형을 최소화 
	-  어간 추출 (Stemming) : 단어에서 접사를 제거하여 기본 형태(어간)를 찾는 과정 
	-  표제어 추출 (Lemmatization): 단어를 사전에 등록된 형태(표제어)로 변환. 
	-  형태소 분석 (Morphological Analysis): 단어를 형태소로 분석하여 그 구조를 이해하는 과정. 
4. **품사 태깅** (Part-of-Speech Tagging): 각 토큰에 품사(명사, 동사 등)를 할당하는 과정 
	-  구문 분석이나 의미 분석에 중요한 정보를 제공 
5. **임베딩** (Embedding) : 토큰을 벡터 공간에 매핑하여 컴퓨터가 처리할 수 있는 수치적 형태로 변환. 
6. **학습** (Training) : 주어진 태스크(예: 분류, 번역)에 맞게 모델을 학습시키는 과정
	-  데이터를 기반으로 패턴을 학습하고 모델 파라미터를 최적화 
7. **생성 및 분류** (Generation and Classification)
#### LLM의 언어 처리 Task Model
##### Sequence-to-Sequence (Seq2Seq)
![[elective_LLM__Pasted image 20250120111923.png]]
- 하나의 입력 시퀀스를 받아 다른 출력 시퀀스로 변환하는 신경망 구조 
- Encoder(문맥이해, 압축)-Decoder(문장 생성) 아키텍처 기반 
- 컨텍스트 벡터를 기반으로 다음 단어를 예측 
- 두 개의 RNN(인코더와 디코더)를 연결하여 구현 
- **입력과 출력 시퀀스의 길이가 달라도 처리 가능** 
- 초기 Seq2Seq 모델에서는 모든 입력 정보를 고정 길이 벡터 C로 압축하므로, 긴 시퀀스 에서는 정보 손실이 발생 (장기 의존성 문제) 
- 정보 손실을 해결하기 위해 Attention 메커니즘 도입

`Encoder`
- 입력 시퀀스를 고정된 길이의 벡터(컨텍스트 벡터) 로 변환 
- 입력 데이터를 단계별로 처리하여 입력 시퀀스의 정보를 함축적으로 표현 
- RNN, LSTM, GRU 등의 순환 신경망이 사용됨
`Decoder`
- Encoder에서 생성된 컨텍스트 벡터를 사용하여 출력 시퀀스를 생성 
- RNN 계열 모델로 구성 
- 생성된 단어(또는 토큰)와 컨텍스트 벡터를 기반으로 Decoder가 생성한 단어들을 하나씩 결 합하면서 다음 단어를 예측하며, 생성해야 할 단어의 수만큼 예측 반복
##### Attention Mechanism
![[elective_LLM__Pasted image 20250120112057.png]]
-  Seq2Seq 모델에서 Decoder가 입력 시퀀스의 모든 단어를 동적으로 참조할 수 있도록 도와주는 방법 
- Seq2Seq 모델에서 발생하는 정보 손실 문제를 해결하기 위해 도입 
- Decoder 가 출력 단어를 생성할 때 입력의 특정 부분에 더 집중(Attention) 하도록 함 
- Context Vector 는 첫 단어의 예측에 가장 많은 영향을 미치는 단어에 대한 정보가 담겨 있다 (단어별로 계산된 Attention Score에 Softmax를 적용한 가중치의 표현) 
- 모든 디코더의 RNN이 인코더의 각 타임 스텝마다 RNN의 결과 값인 출력을 참고하는 방식
##### Transformer
- 각 토큰을 시퀀스에 존재하는 다른 토큰들과 연관시킨다 
- 순차적인 데이터 처리의 필요성을 최소화하고 전체 시퀀스를 한 번에 처리할 수 있게 합니다.
- Self-Attention Mechanism
	- 입력 데이터의 모든 부분이 서로 상호 작용할 수 있도록 하여, 문맥 이해를 극대화합니다. 
	- 각 단어가 문장 내 다른 단어와의 관계를 동시에 고려하여, 풍부하고 상세한 텍스트 표현을 생성합니다.
-  Multi-Head Attention 
	- 여러 개의 어텐션 메커니즘(헤드)을 동시에 사용하여 다양한 부분에서 정보를 추출하고, 입력시퀀스의 다양한 표현을 학습합니다. 
	- 모델이 더 넓은 범위의 정보를 포착하고, 다양한 관점에서 데이터를 해석할 수 있도록 돕습니다
-  Positional Encoding 
	- 시퀀스의 순서 정보를 모델에 포함시키기 위해 위치 인코딩을 사용합니다. 
	- 단어의 위치에 따른 문맥적 의미를 모델이 이해할 수 있도록 돕습니다
- Layered Architecture 
	- 여러 개의 인코더와 디코더 층으로 구성되어 있으며, 각 층은 복잡한 패턴과 관계를 학습
>[!Transformer]
>Text & Position Embed : 텍스트 데이터와 해당 위치 정보를 포함하는 임베딩 벡터를 생성 12x Transformer Block : 
>- Masked Multi Self Attention: 입력 시퀀스 내의 각 요소가 서로 상호 작용하며 중요한 정보에 집중할 수 있도록 합니다. 
>- Layer Norm: 층 정규화를 통해 학습 과정을 안정화하고 더 빠르게 수렴하도록 돕습니다. 
>- Feed Forward: 각 위치에서 독립적으로 적용되는 신경망으로, 비선형 변환을 추가합니다.

##### 자기 회귀 언어 모델(Autoregressive Language Model)
- 문장에서 이전 토큰만을 기반으로 다음 토큰을 예측 
- 알려진 어휘에서 주어진 문장의 바로 다음에 가장 가능성 있는 토큰을 생성하도록 모델 에 요청 
- Transformer Model의 Decoder 
- Attention Head가 앞서 온 토큰만 볼 수 있도록 전체 문장에 mask가 적용 
- 텍스트 생성에 이상적
##### 자동 인코딩 언어 모델(Autoencoding Language Model)
-  손상된 버전의 입력 내용으로부터 기존 문장을 재구성하도록 훈련 
- 알려진 어휘에서 문장의 어느 부분이든 누락된 단어를 채우도록 모델에 요청 
- Transformer Model의 Encoder 
- 전체 문장의 양방향 표현을 생성 
- 마스크 없이 전체 입력에 전근할 수 있습니다 
- 텍스트 생성과 같은 다양한 작업에 파인 튜닝 될 수 있습니다.
##### 거대 언어 모델(Large Language Model)
- 자기 회귀 이거나 자동 인코딩 또는 두 가지의 조합이 될 수 있는 언어 모델 
- 인터넷의 텍스트, 책, 논문, 기사 등 **다양하고 방대한 양의 텍스트 데이터로부터 학습** 
- **언어를 이해하고 생성하는데 특화되어 질문에 답변하고, 글을 작성하고, 대화를 나누는 생성적 작업 수행** 
- **Transformer Architecture 기반** 
- 특화된 분야에 정교하게 사용될 수 있도록 파인 튜닝 할 수 있습니다. 
- 수천억 개 이상의 파라미터를 갖는 모델
- 텍스트 생성 및 분류와 같은 복잡한 언어 작업을 파인 튜닝이 필요 없을 만큼 높은 정확 도로 수행 
- 단계적으로 추론을 유도하여 복잡한 문제의 논리적 추론을 수행
#### Large Language Model 비교

>[!BERT-Large]
>2018, Google AI, Transformer 기반의 양방향 모델 
>전체 문장 내에서 단어의 좌우, 즉, 양방향으로 문맥을 고려하면서 학습하기 때문 에, 문장의 의미를 포괄적으로 이해함. 맥락 상 단어의 의미를 파악하는 것이 중요 한 자연어 처리 작업에 강점을 지님. 
>Fine-tuning을 통해 다양한 NLP 태스크에 적용 가능 
>문서 분류, 감성 분석, 개체명 인식, Q&A, 기계 번역 등의 분야에 활용

>[!GPT]
>2019, OpenAI, **Transformer 기반의 자기 회귀 모델**
> Transformer의decoder 부분만 중첩된 구조의 모델
> 일방향으로 나아가면서 다음 단어를 예측하는 방식으로 학습하기 때문에, 순차적 으로 일관되게 문장을 생성함. 
> 인간처럼 일관되고 연관성이 높은 언어를 구사하여 대화형 작업에 강점을 지님 큰 규모의 데이터셋에서 미세조정 없이도 강력한 성능을 발휘, 일반화 능력이 뛰어남 
> 문장 완성,요약,대화형 챗봇,창작 글쓰기(소설, 시 등),프로그래밍 코드 작성 등의 분야에 활용

>[!T5]
>2019, Google Research 
>Transformer 기반, 모든 NLP 문제를 텍스트 생성 문제로 변환하여 처리 
>파인튜닝 없이 한 번에 여러 작업을 해결하는 데 잠재력을 보인 최초의 LLM 
>기계 번역, 요약, 텍스트 분류 등의 분야에 활용

>[!RoBERTa]
>2019, Facebook AI, BERT의 개선된 버전 
>BERT보다 더 긴 훈련 시간, 더 큰 데이터셋, 더 큰 배치 사이즈로 성능 개선 
>텍스트 분류, 자연어 이해, 질문응답 등의 분야에 활용

>[!ELECTRA]
>2020, Google Research, Transformer 기반, Discriminator와 Generator로 구성 
>기존 BERT보다 훈련 효율성을 높이며, 낮은 계산 비용으로 성능을 향상 
>감정 분석, 텍스트 분류 등

---
