---
aliases: []
course: large-language-models
created: '2025-01-21'
date: '2025-01-21'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: elective
source: ''
status: seedling
tags:
- cs/llm
- cs/nlp
- type/lecture
title: OpenAI API Key 설정
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

#### 1. Embedding이란?

- **정의**: 텍스트 데이터를 고차원 벡터(부동소수점 배열)로 변환하는 기술.
- **특징**:
    - 유사한 의미를 가진 단어나 문장은 벡터 거리가 가깝게 설계.
    - 유사하지 않은 의미는 벡터 간 거리가 멀어지도록 구성.
- **활용 사례**:
    - **데이터 분석**: 클러스터링, 추천 시스템, 검색 최적화.
    - **AI 응용**: 유사도 비교, 주제 분석, 군집화.
- **참고 자료**:  
    [Embedding 공식 문서](https://platform.openai.com/docs/guides/embeddings)

---

#### 2. Embedding 클래스 및 주요 함수

**A. Embedding 클래스**

- **역할**: 입력 텍스트에 대해 고차원 벡터 표현 생성.
- **응용 사례**:
    - 텍스트를 의미론적으로 유사한 텍스트끼리 가까운 위치로 매핑.
    - 텍스트 의미와 구조를 벡터로 변환해 컨텍스트 정보 캡처.

**B. Embedding.create() 함수**

- **설명**: 주어진 입력 텍스트에 대한 임베딩 생성.
- **매개변수**:
    - `model` (필수): 사용할 모델 이름 (예: `"text-embedding-3-small"`).
    - `input` (필수): 임베딩 생성 대상 (텍스트 또는 텍스트 리스트).
    - `user` (선택적): 사용자 정의 데이터 제공 시 사용.
    - `**kwargs` (선택적): 추가 설정이나 API 기능 구성.

---

#### 3. Embedding 기반 유사도 검색 실습

**목적**: Facebook의 오픈소스 벡터 데이터베이스 **Faiss**를 활용하여 유사도 검색 수행.

```python
import openai
import numpy as np
import faiss

# OpenAI API Key 설정
OPENAI_API_KEY = "sk-proj-hbsi47ndwg_2I1zuYAa0p9ODzEM 4vApXF27cJkcA............."
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 입력 텍스트에 대한 임베딩 생성
in_text = "오늘은 눈이 오지 않아서 다행입니다"
response = client.embeddings.create(
    input=in_text,
    model="text-embedding-3-sm"
)

# 응답에서 임베딩 추출 및 변환
in_embeds = [record.embedding for record in response.data]
ndarray_embeds = np.array(in_embeds).astype("float32")  # 효율적 계산을 위해 numpy 배열 변환

# 비교 대상 텍스트 리스트
target_texts = [
    "좋아하는 음식은 무엇인가요?", 
    "어디에 살고 계신가요?", 
    "아침 전철은 혼잡하네요", 
    "오늘 날씨가 화창합니다", 
    "요즘 경기가 좋지 않습니다."
]

# 비교 대상 텍스트에 대한 임베딩 생성
response2 = client.embeddings.create(
    input=target_texts,
    model="text-embedding-3-small"
)
target_embeds = [record.embedding for record in response2.data]
ndarray_embeds2 = np.array(target_embeds).astype("float32")

# Faiss 인덱스 생성 및 벡터 추가
index = faiss.IndexFlatL2(1536)  # 벡터 차원 지정 (예: 1536차원 모델)
index.add(ndarray_embeds2)  # 벡터 추가 (ndarray는 float32 형식이어야 함)

# 유사도 검색 수행
D, I = index.search(ndarray_embeds, 1)  # 가장 가까운 벡터 1개 검색

# 결과 출력
print(D)  # D: 쿼리 벡터와 가장 가까운 이웃 벡터 간 거리
print(I)  # I: 가장 가까운 이웃 벡터의 인덱스
print(target_texts[I[0][0]])  # 가장 가까운 이웃 텍스트 출력
```

---

#### 4. 주요 실행 결과

- **`D` (거리)**: 입력 벡터와 가장 가까운 이웃 벡터 간의 거리 정보.
- **`I` (인덱스)**: 입력 벡터와 가장 가까운 이웃 텍스트의 인덱스.
- **출력 텍스트**: 가장 유사한 의미를 가진 텍스트를 반환.

---

**활용**: 추천 시스템, 유사 텍스트 검색, 주제 기반 군집화 등 다양한 머신러닝 및 NLP 응용.  
**참고**: `faiss`는 대규모 벡터 데이터를 처리할 때 효율적이며, GPU 가속도 가능.
