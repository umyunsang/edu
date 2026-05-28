---
aliases: []
course: ai-system-design
created: '2025-03-31'
date: '2025-03-31'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ai
- cs/se
- type/lecture
title: Architecture
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/5단계 통합 프로젝트 인터페이스|5단계 통합 프로젝트 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/AI 시스템 설계 인터페이스|AI 시스템 설계 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/산출물 포트폴리오 브리지|산출물 포트폴리오 브리지]]
up:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
prerequisites:: [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_assignment2_starl_skeleton|self_intro_assignment2_starl_skeleton]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/이론/MLP 이론|MLP 이론]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/AI 시스템 설계 지식그래프|AI 시스템 설계]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/AI 시스템 설계 지식그래프|AI 시스템 설계]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/AI 시스템 설계 근거 인덱스|AI 시스템 설계 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/데이터 모델링|데이터 모델링]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/ai-system-design/db|db]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
# Architecture
## **1. 시스템 개요**  

카페나 상점에서 고객의 **구매 기록**과 **AI 분석**을 활용하여 맞춤형 추천을 제공하고, **결제까지 한 번에 진행**할 수 있도록 하는 시스템을 설계합니다. 또한, **판매 관리자가 판매하는 메뉴 정보를 직접 입력**하면, 해당 데이터가 **벡터 저장소(Vector Database)** 에 저장되고, **RAG(Retrieval-Augmented Generation) 방식**을 활용하여 AI 챗봇이 더욱 정밀한 응답을 생성하도록 합니다.

### **🎯 주요 목표**  
- 고객의 취향과 구매 패턴을 분석하여 **개인 맞춤형 메뉴 추천**  
- 자연어 처리(NLP) 기반의 **대화형 AI 챗봇**을 활용한 주문 인터페이스  
- **결제 기능 통합**으로 원스톱 구매 프로세스 구축  
- LLM(RAG 또는 Fine-Tuning)을 활용한 고객 응대 및 추천 개선  
- **판매 관리자가 메뉴 데이터를 입력하여 AI 응답을 최적화**  
- 접근성 향상(디지털 소외계층 고려)  

---

## **2. 시스템 구성 요소**  

### **🛠 기술 스택**  
| 구성 요소 | 기술 선택 |
|-----------|------------|
| 백엔드 API | Python (Flask / FastAPI) |
| 데이터베이스 | PostgreSQL / Firebase Firestore |
| AI 모델 | LLM (GPT-4, Llama2, Mistral) + RAG or Fine-Tuning |
| 추천 시스템 | 협업 필터링, 콘텐츠 기반 필터링 (Scikit-learn, TensorFlow) |
| 음성 인터페이스 | OpenAI Whisper, Google Speech-to-Text |
| 프론트엔드 | React (Next.js) |
| 결제 시스템 | Stripe, Toss Payments, Kakao Pay |
| 벡터 저장소 | ChromaDB, Weaviate, Pinecone |

---

## **3. 시스템 아키텍처**  

<div class="mermaid-container" style="transform: scale(0.6); transform-origin: top left;">

```mermaid
sequenceDiagram
    판매관리자->>벡터저장소: 메뉴 데이터 입력
    벡터저장소-->>AI챗봇: 데이터 활용
    사용자->>AI챗봇: 자연어 입력
    AI챗봇-->>사용자: 메뉴 추천
    사용자->>결제시스템: 주문 요청
    결제시스템-->>사용자: 결제 진행
    사용자->>데이터저장: 주문 완료
```
</div>

1️⃣ **판매 관리자가 메뉴 정보를 입력하면 벡터 저장소에 저장**  
2️⃣ **고객이 AI 챗봇을 통해 음성/텍스트 입력으로 메뉴를 검색**  
3️⃣ **AI가 벡터 저장소의 정보를 기반으로 최적의 추천을 제공**  
4️⃣ **고객이 메뉴를 선택하고 주문을 진행**  
5️⃣ **결제 모듈과 연동하여 결제 진행 (QR 결제, 카드, 간편결제 지원)**  
6️⃣ **주문 완료 후 고객 피드백 수집 및 데이터 저장**  

---

## **4. AI 모델 활용 방안**  

### **🔹 LLM을 활용한 자연어 기반 추천**  
- 고객의 **자연어 입력 분석** (예: "오늘 단 게 땡겨", "커피 말고 다른 음료 있어?")  
- **카페 메뉴 정보**를 벡터 저장소에 저장하고, RAG 또는 Fine-Tuning하여 LLM이 자연스럽게 추천  

### **🔹 추천 시스템 (AI 기반 추천)**  
1️⃣ **협업 필터링 (CF, Collaborative Filtering)**  
   - 비슷한 고객의 구매 패턴을 분석하여 추천  
   - "A 고객이 선호하는 메뉴를 B 고객도 선호할 가능성이 높음"  

2️⃣ **콘텐츠 기반 필터링 (CBF, Content-Based Filtering)**  
   - 고객의 과거 구매한 메뉴와 유사한 특징을 가진 메뉴 추천  
   - "고객이 바닐라 라떼를 자주 주문했다면, 헤이즐넛 라떼 추천"  

3️⃣ **하이브리드 모델 (CF + CBF + LLM)**  
   - LLM이 자연어 입력을 해석하고, 추천 알고리즘과 결합하여 정밀한 추천 제공  

---

## **5. 데이터베이스 및 벡터 저장소 설계 (예시)**  

### **🔹 주요 테이블**  

#### **① 고객 테이블 (`customers`)**  
| Column | Type | Description |
|--------|------|-------------|
| customer_id | INT | 고객 고유 ID |
| name | VARCHAR | 고객 이름 |
| phone | VARCHAR | 전화번호 |
| preferences | JSON | 선호하는 메뉴 및 태그 |

#### **② 주문 테이블 (`orders`)**  
| Column | Type | Description |
|--------|------|-------------|
| order_id | INT | 주문 고유 ID |
| customer_id | INT | 고객 ID (FK) |
| menu_id | INT | 주문한 메뉴 ID |
| total_price | DECIMAL | 총 결제 금액 |
| order_date | TIMESTAMP | 주문 시간 |

#### **③ 메뉴 테이블 (`menu`)**  
| Column | Type | Description |
|--------|------|-------------|
| menu_id | INT | 메뉴 고유 ID |
| name | VARCHAR | 메뉴명 |
| category | VARCHAR | 메뉴 카테고리 (커피, 디저트 등) |
| price | DECIMAL | 가격 |
| tags | JSON | 태그 정보 (예: {"달콤한": true, "고소한": false}) |

#### **④ 벡터 저장소 (`vector_menu`)**  
| Column | Type | Description |
|--------|------|-------------|
| menu_id | INT | 메뉴 고유 ID (FK) |
| embedding | VECTOR | 메뉴 설명 및 특징을 임베딩한 벡터 |
| metadata | JSON | 추가적인 메타데이터 (예: 재료, 맛 등) |

---

# DEV

## 🚀 **개발 진행 현황 및 우선순위**  

### ✅ **현재까지 개발된 부분**  

#### 1️⃣ **사용자 관리**  
- ✅ **회원가입** (일반 사용자, 관리자)  
- ✅ **로그인/인증** (JWT 토큰 기반)  
- ✅ **사용자 선호도 저장** (단맛, 신맛, 쓴맛)  

#### 2️⃣ **메뉴 관리**  
- ✅ **메뉴 데이터베이스 구축** (SQLite)  
- ✅ **메뉴 조회 기능**  

#### 3️⃣ **장바구니 & 주문 시스템**  
- ✅ **장바구니 계산 기능**  
- ✅ **주문 생성 및 조회**  
- ✅ **주문 상태 관리** (Pending, Paid, Completed, Cancelled)  

#### 4️⃣ **AI 추천 시스템**  
- ✅ **LLM 기반 자연어 처리** (Mistral)  
- ✅ **개인화된 메뉴 추천**  
- ✅ **협업 필터링 & 콘텐츠 기반 필터링**  
- ✅ **벡터 스토어 기반 검색** (RAG + 파인튜닝)  

#### 5️⃣ **관리자 기능**  
- ✅ **메뉴 관리 (CRUD)**  
- ✅ **주문 관리 및 상태 업데이트**  

---

### ❗ **개발이 필요한 부분**  

#### 🔹 **1. 결제 시스템**  
- ❌ **결제 모듈 연동** (Naver Pay, Toss, Kakao Pay)  
- ❌ **결제 상태 관리**  
- ❌ **영수증 발행**  

#### 🔹 **2. 음성 인터페이스**  
- ❌ **음성 인식 기능** (OpenAI Whisper)  
- ❌ **음성 출력 기능**  

#### 🔹 **3. 프론트엔드 (웹 UI/UX 개발)**  
- ❌ **웹 인터페이스 구축** (Next.js)  
- ❌ **반응형 디자인 적용**  
- ❌ **접근성 고려 (디지털 소외 계층 배려)**  

#### 🔹 **4. 관리자 기능 확장**  
- ❌ **매출 통계 대시보드**  
- ❌ **고객 데이터 분석 기능**  

#### 🔹 **5. 보안 강화**  
- ❌ **결제 정보 암호화**  
- ❌ **API 접근 제한**  
- ❌ **로그 관리 시스템**  

#### 🔹 **6. 기타 기능**  
- ❌ **알림 시스템 (주문 완료 알림 등)**  
- ❌ **리뷰/평점 시스템**  
- ❌ **포인트/적립 시스템**  

---

# TEST

# 카페 추천 AI 시스템 API 테스트 결과
테스트 시작 시간: 2024-03-30 15:57:32

## 1. 발견된 문제점 및 해결
1. URL 리다이렉트 문제
   - 문제: 307 Temporary Redirect 에러 발생
   - 해결: URL 끝에 슬래시(/) 추가 (예: /api/v1/menus/ )

2. 관리자 권한 문제
   - 문제: 잘못된 관리자 계정 정보 사용
   - 해결: config.py에 설정된 올바른 관리자 계정 사용
     - 이메일: admin@example.com
     - 비밀번호: admin123

## 2. API 테스트 결과

### 2.1 회원가입 테스트 (일반 사용자)
**요청 (Request):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/auth/register' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "test2@example.com",
    "name": "테스트2",
    "password": "test1234!",
    "preferences": {
      "taste": {
        "sweet": 3,
        "sour": 2,
        "bitter": 4
      }
    }
  }'
```

**응답 (Response):**
```json
{
  "email": "test2@example.com",
  "name": "테스트2",
  "preferences": {
    "taste": {
      "sweet": 3,
      "sour": 2,
      "bitter": 4
    }
  },
  "is_active": true,
  "is_admin": false,
  "id": 6,
  "created_at": "2025-03-30T15:48:06",
  "updated_at": null
}
```

### 2.2 관리자 로그인 테스트
**요청 (Request):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/auth/login' \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "password": "admin123"
  }'
```

**응답 (Response):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDQwNDE0NDAsInN1YiI6ImFkbWluQGV4YW1wbGUuY29tIn0.R2_7-87Nd3b6KMeleKKSZqnJYRQVS_VdU0feT9wPQgo",
  "token_type": "bearer"
}
```

### 2.3 메뉴 추가 테스트 (관리자)
**요청 (Request):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/menus/' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NDQwNDE0NDAsInN1YiI6ImFkbWluQGV4YW1wbGUuY29tIn0.R2_7-87Nd3b6KMeleKKSZqnJYRQVS_VdU0feT9wPQgo' \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "카푸치노",
    "description": "에스프레소와 스팀 밀크, 우유 거품의 완벽한 조화",
    "price": 5000,
    "category": "커피",
    "attributes": {
      "taste": {
        "sweet": 2,
        "sour": 3,
        "bitter": 3
      },
      "temperature": "hot",
      "size": "regular"
    }
  }'
```

**응답 (Response):**
```json
{
  "name": "카푸치노",
  "description": "에스프레소와 스팀 밀크, 우유 거품의 완벽한 조화",
  "price": 5000.0,
  "category": "커피",
  "image_url": null,
  "sweetness": 0.0,
  "sourness": 0.0,
  "bitterness": 0.0,
  "temperature": "both",
  "properties": null,
  "id": 5,
  "order_count": 0,
  "created_at": "2025-03-30T15:57:32.432824",
  "updated_at": "2025-03-30T15:57:32.432830",
  "rating": 0.0
}
```

## 3. 최종 테스트 결과
1. 회원가입 (일반 사용자) - ✅ 성공
2. 일반 사용자 로그인 - ✅ 성공
3. 메뉴 목록 조회 - ✅ 성공
4. 일반 사용자 메뉴 추가 - ❌ 실패 (권한 없음, 예상된 동작)
5. 관리자 로그인 - ✅ 성공
6. 관리자 메뉴 추가 - ✅ 성공
