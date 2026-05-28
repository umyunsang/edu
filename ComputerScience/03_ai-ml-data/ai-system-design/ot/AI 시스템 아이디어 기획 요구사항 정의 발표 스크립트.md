---
aliases: []
course: ai-system-design
created: '2025-03-17'
date: '2025-03-17'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ai
- cs/se
- type/lecture
title: 'AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트'
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/AI 시스템 설계 인터페이스|AI 시스템 설계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]]
prerequisites:: [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_ax_archive_draft|self_intro_ax_archive_draft]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/07_professional-humanities/degree-portfolio/GovOn 온프레미스 AI 발표 스크립트|GovOn 온프레미스 AI 발표 스크립트]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/deco_extracurricular_evidence|deco_extracurricular_evidence]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/애플 M4 CPU/애플 M4 CPU|애플 M4 CPU]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/ASD Feature 발굴|ASD Feature 발굴]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/시험정리|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/AI 시스템 설계 지식그래프|AI 시스템 설계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/AI 시스템 설계 근거 인덱스|AI 시스템 설계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/cnn|cnn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/cifar10|cifar10]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/sql|sql]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
**AI 시스템 아이디어 기획 및 요구사항 정의 발표 스크립트**

안녕하세요. 저희 팀의 AI 시스템 아이디어를 발표하겠습니다. 저는 발표를 맡은 김수미입니다. 저희 팀은 정보 교류의 단절 문제를 해결하고 빠른 정보 교환이 가능한 AI 시스템을 제안합니다.

### 1. 아이디어 개요

저희가 제안하는 AI 시스템은 **각 개인별 정보를 학습한 AI 시스템이 다른 사람의 AI 시스템과 정보를 교류할 수 있는 시스템**입니다.

현재 많은 사람들이 정보를 주고받는 과정에서 여러 단절이 발생합니다. 예를 들어, 회의에서 상대방이 어떤 정보들을 가지고 있는지 알지 못하면 의사소통이 원활하지 않을 수 있습니다. 또한, 개인이 미처 기억하지 못한 중요한 정보를 놓치는 경우도 많습니다.

저희 AI 시스템은 이러한 문제를 해결하여, 개인의 정보를 학습한 후, 필요할 때 신속하고 효율적으로 다른 AI 시스템과 교류함으로써 정보의 단절을 최소화합니다. 이를 통해 보다 빠르고 정확한 의사결정이 가능해집니다.

### 2. 주요 기능 (기능적 요구사항)

이 AI 시스템이 제공하는 주요 기능은 다음과 같습니다.

1. 사용자가 제공한 데이터를 기반으로 새로운 데이터를 분석하고 이를 전송합니다.
2. 사용자가 설정한 정보 공유 기준에 따라 특정 정보만 교류할 수 있도록 합니다.
3. 회의나 일정 관리 시 AI가 필요한 정보를 자동으로 정리하여 제공하며, 사용자에게 필요한 인사이트를 제공합니다.
4. AI 간 실시간 데이터 교류를 통해 정보 전달 속도를 극대화합니다.

### 3. 시스템의 성능 및 제약 조건 (비기능적 요구사항)

AI 시스템이 원활하게 동작하기 위해 몇 가지 성능 기준과 제약 조건을 설정하였습니다.

- **시스템의 처리 속도**: AI는 데이터 크기당 최대 4초 이내에 정보를 처리하고 분석하여 전달해야 합니다.
- **데이터 보안 요구사항**: 개인정보 유출을 방지하고, 사용자가 허용한 정보만 공유될 수 있도록 엄격한 보안 기준을 준수합니다.
- **AI 모델의 정확도 기준**: AI는 반드시 사실에 기반한 데이터를 제공하여 신뢰성을 확보해야 합니다.
- **사용 환경**: 모바일 기반으로 동작하여, 스마트폰에서 손쉽게 활용할 수 있도록 설계되었습니다.

### 4. 예상 사용자 및 활용 시나리오

이 AI 시스템은 **남녀노소 누구나** 활용할 수 있으며, 특히 회사 미팅과 같은 상황에서 매우 유용하게 사용될 수 있습니다.

예를 들어, 회사 미팅을 진행할 때, 팀원들이 서로 어떤 데이터를 가지고 있는지 알지 못해 정보 공유가 원활하지 않은 경우가 많습니다. 하지만 이 AI 시스템을 사용하면 **각 팀원의 AI가 미팅 중 필요한 정보를 빠르게 분석하고 교류하여, 보다 효과적인 의사결정을 돕게 됩니다.**

또한, 개인 일정 관리에도 활용할 수 있습니다. AI가 사용자의 일정과 관련 정보를 자동으로 정리하고 중요한 미팅이나 업무를 미리 알려주므로, 더욱 체계적인 일정 관리가 가능해집니다.

### 5. 기대 효과 및 결론

이 AI 시스템이 도입되었을 때, 기대할 수 있는 효과는 다음과 같습니다.

1. **빠른 정보 교류**: AI 간 자동 정보 교환을 통해 기존보다 훨씬 빠른 정보 공유가 가능합니다.
2. **효율적인 일정 관리**: AI가 사용자의 업무 및 일정을 자동으로 정리하여, 보다 체계적인 관리가 가능합니다.
3. **의사결정의 정확성 증가**: 필요한 정보를 즉각적으로 제공함으로써, 보다 정확한 의사결정을 내릴 수 있습니다.

결론적으로, 저희 AI 시스템은 개인 간 정보 교류의 속도를 획기적으로 높이고, 보다 효율적인 일정 및 업무 관리를 가능하게 합니다. 이를 통해 AI 기반 정보 교환 시스템이 다양한 산업과 일상생활에서 혁신적인 변화를 가져올 것이라 확신합니다.

이상으로 저희 팀의 AI 시스템 아이디어 발표를 마치겠습니다. 경청해 주셔서 감사합니다.
