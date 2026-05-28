---
aliases: []
course: ai-system-design
created: '2025-03-21'
date: '2025-03-21'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ai
- cs/se
- type/lecture
title: 'AI 챗봇 특허 저작권 보호 전략 발표 스크립트'
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/AI 시스템 설계 인터페이스|AI 시스템 설계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]]
prerequisites:: [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/이론/Backpropagation|Backpropagation]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/07_professional-humanities/intellectual-property/1. 소개/소개|소개]], [[ComputerScience/07_professional-humanities/intellectual-property/2. 저작권제도와 등록요건/저작권 제도와 등록요건|저작권 제도와 등록요건]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/07_professional-humanities/intellectual-property/5. 특허/특허 제도|특허 제도]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/07_professional-humanities/degree-portfolio/GovOn 온프레미스 AI 발표 스크립트|GovOn 온프레미스 AI 발표 스크립트]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/07_professional-humanities/intellectual-property/기출문제/processed/지식재산권_processed|지식재산권_processed]], [[ComputerScience/07_professional-humanities/intellectual-property/기출문제/1. IPAT 기출문제 지식재산권_답안통합|1. IPAT 기출문제 지식재산권_답안통합]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/시험정리|시험정리]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/AI 시스템 설계 지식그래프|AI 시스템 설계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/AI 시스템 설계 근거 인덱스|AI 시스템 설계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/cnn|cnn]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/cifar10|cifar10]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/sql|sql]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
### **📢 AI 챗봇 기반 가상 고객센터 – 특허 vs 저작권 보호 전략**

**발표자: 엄윤상**  
**팀명: 전화위복 팀**

---

## **1. 발표 시작 & 팀 소개**

안녕하세요, **전화위복 팀**의 발표를 맡은 엄윤상입니다.  
저희 팀은 **AI 챗봇 기반 가상 고객센터** 시스템을 주제로 선택하였습니다.
이 기술을 **특허로 보호할지, 저작권으로 보호할지** 분석했습니다.

오늘 발표에서는  
1️⃣ AI 챗봇 시스템이 무엇인지,  
2️⃣ 어떤 문제를 해결하는지,  
3️⃣ 특허 보호가 가능한지,  
4️⃣ 저작권 보호가 가능한지,  
5️⃣ 그리고 기업이 실제로 사용할 보호 전략까지  
차례대로 설명해 드리겠습니다.

---

## **2. AI 챗봇 기반 가상 고객센터란?**

저희가 개발하는 AI 챗봇은 단순한 FAQ 시스템이 아닙니다.  
**각 회사의 방침과 매뉴얼을 반영하여 맞춤형 응답을 제공하는 AI 시스템**입니다.

기술적으로는 **LLM 기반 RAG(Retrieval-Augmented Generation) 방식**을 활용하여,  
고객 질문을 분석하고, **회사 내부 문서를 실시간으로 검색하여**  
보다 정확하고 자연스러운 답변을 제공합니다.

이 시스템을 통해 기업은 **운영 비용을 절감하고**,  
**24시간 자동화된 고객 지원**을 제공할 수 있습니다.

---

## **3. 특허 보호 가능성 분석**

그렇다면, 이 AI 챗봇 시스템을 **특허로 보호할 수 있을까요?**

저희는 **다음 네 가지 기준**으로 특허 가능성을 검토했습니다.

1️⃣ **혁신성이 있는가?**

- ❌ 기존 RAG 기반 기술과 유사한 방식이 많아 **혁신성이 부족함**

2️⃣ **경쟁사가 쉽게 모방할 수 있는가?**

- ❌ 오픈소스 모델이 많아, **경쟁사가 쉽게 유사한 기능을 개발 가능**

3️⃣ **20년 동안 독점할 가치가 있는가?**

- ❌ 기술 발전 속도가 빨라, **특허 보호의 실효성이 낮음**

4️⃣ **특허 등록이 현실적으로 가능한가?**

- ❌ 기존 특허들과 유사한 내용이 많아, **등록 가능성이 낮음**

🔴 **결론적으로, 특허 보호는 어렵습니다.**  
기술의 혁신성이 부족하고, **경쟁사가 쉽게 모방할 가능성이 높기 때문입니다.**

---

## **4. 저작권 보호 가능성 분석**

그렇다면, **저작권 보호는 가능할까요?**

저작권은 **창작물(코드, 디자인, 데이터 등)을 보호**하는 방식입니다.  
저희 기술을 저작권 보호 기준에 따라 분석해 보았습니다.

✅ **소프트웨어 코드, UI 디자인, 데이터셋은 창작물인가?**

- ✅ **네, 저작권 보호 가능!**

✅ **등록하지 않아도 자동 보호되는가?**

- ✅ **네, 저작권은 자동으로 보호됨!**

✅ **다른 기업이 유사한 기술을 개발해도 보호할 수 있는가?**

- ✅ **UI 디자인과 대화 데이터는 보호 가능!**
- ❌ 하지만, **알고리즘 자체는 저작권 보호 대상이 아님.**

✅ **알고리즘 자체를 보호할 수 있는가?**

- ❌ 아니요, **알고리즘 자체는 보호되지 않음.**
- 하지만, **학습 데이터와 UI를 보호하여 간접적으로 방어 가능!**

🔵 **결론적으로, 저작권 보호가 가능합니다.**  
하지만, **알고리즘 자체를 보호할 수는 없다는 한계**가 있습니다.

---

## **5. 최종 결론 – 저작권 보호 전략**

저희 팀은 최종적으로 **저작권 보호 전략을 선택**했습니다.

🟢 **선택한 이유:**  
✔ AI 챗봇의 **소스 코드, UI 디자인, 대화 데이터셋**은 저작권 보호 가능  
✔ 특허를 등록하기에는 **기존 기술과 차별성이 부족**  
✔ 저작권 보호를 통해 **경쟁사의 무단 사용을 방지 가능**

🔴 **특허를 선택하지 않은 이유:**  
✔ 특허 등록이 어렵고, **기술 발전 속도가 너무 빠름**  
✔ 경쟁사가 **조금만 변형해도 회피 가능**

📌 **따라서, 저작권을 적극 활용하여 보호하는 전략을 채택하였습니다.**

---

## **6. 기업이 실제로 사용할 보호 전략**

그렇다면, **실제 기업에서 어떻게 보호할 수 있을까요?**

🔒 **저작권 등록:**  
✔ 챗봇의 **UI 디자인과 코드**를 저작권으로 보호  
✔ 대화 데이터베이스를 **저작권 보호 대상으로 포함**

🔒 **비공개 데이터 관리:**  
✔ **중요한 AI 학습 데이터 및 응답 모델**을 비공개(Trade Secret)로 관리  
✔ **경쟁사의 접근을 차단**하여 무단 복제 방지

💡 **즉, 저작권과 비공개 데이터 전략을 병행하여 보호하는 것이 최적의 방법입니다!**

---

## **7. 마무리 및 Q&A**

정리하면,  
💡 **특허 보호는 어려운 반면, 저작권 보호는 가능**하며,  
💡 **비공개 데이터 관리 전략을 추가로 활용하면 보다 강력한 보호가 가능**합니다.

이상으로 발표를 마치겠습니다.  
경청해 주셔서 감사합니다. 😊  
질문 있으시면 자유롭게 해주세요! 🎤

---

📌 **이 대본은 발표자가 화면을 보지 않고도 자연스럽게 읽을 수 있도록 구성했습니다.**  
추가 수정이 필요하면 말씀해주세요! 🚀😊
