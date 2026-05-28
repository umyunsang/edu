---
aliases: []
course: large-language-models
created: '2025-01-22'
date: '2025-01-22'
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
title: Vector store
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
### Vector Store (VectorDB)

- **Vector Store**는 데이터를 임베딩(벡터 표현)으로 저장하여 벡터 공간 내에서 빠른 검색을 구현하기 위한 데이터베이스입니다.
- **임베딩 벡터**는 텍스트, 이미지, 소리 등 다양한 형태의 데이터를 벡터 공간에 매핑한 것으로, 데이터의 의미적, 시각적, 오디오적 특성을 수치적으로 표현합니다.
- **유사도 측정 방법**: 코사인 유사도, 유클리드 거리, 맨해튼 거리 등 다양한 유사도 측정 방법을 제공합니다.

#### 벡터 스토어 패키지:

- **Faiss**: Facebook AI Research에서 개발한 오픈소스 라이브러리로, 대규모 데이터셋에서 효율적인 유사도 검색을 지원 ([Faiss GitHub](https://github.com/facebookresearch/faiss))
- **Qdrant**: 비동기 작업을 지원하는 벡터 데이터베이스 ([Qdrant](https://qdrant.tech/))
- **Chroma**: 오픈소스 벡터 데이터베이스로, 로컬 환경에서 쉽게 사용할 수 있음 ([Chroma Docs](https://docs.trychroma.com/))
- **Milvus**: 벡터 데이터베이스로, 고차원 데이터의 효율적 저장과 검색을 제공 ([Milvus](https://milvus.io/))

#### 클라우드 서비스:

- **Pinecone**: 클라우드 기반 벡터 데이터베이스 서비스로, 대규모 데이터 처리에 적합 ([Pinecone](https://www.pinecone.io/))
- **Weaviate**: 클라우드 기반 벡터 데이터베이스 서비스 ([Weaviate](https://weaviate.io/))

---

### 비정형 데이터 처리

- **비정형 데이터 처리**: 텍스트, 이미지 등 다양한 형태의 비정형 데이터를 효율적으로 저장하고 검색할 수 있습니다.
- **고차원 데이터 처리**: 수백에서 수천 차원의 고차원 벡터 데이터를 빠르게 처리할 수 있습니다.
- **유사도 기반 검색**: 벡터 간 유사도를 계산하여 가장 유사한 데이터를 빠르게 찾아낼 수 있습니다.
- **LLM 성능 향상**: 방대한 양의 정보를 효율적으로 검색하여 LLM에 제공함으로써 성능을 크게 향상시킬 수 있습니다.

---

### Vector Store - VectorDB 설명

#### Faiss

- **Faiss**는 Facebook AI Research에서 개발한 라이브러리로, 고차원 벡터 간의 유사성 검색을 빠르고 효율적으로 수행할 수 있도록 설계되었습니다.
- **지원**: CPU와 GPU 모두에서 사용 가능, 최적화된 인덱싱 구조 사용
- **용도**: 수백만 또는 수십억 개의 벡터를 처리할 수 있으며, 벡터 간의 유사성 검색을 매우 빠르게 수행합니다.
- **다양한 인덱싱 메커니즘 제공**:
    - **Flat 인덱스**: 정확한 검색을 위한 L2 거리(유클리디안 거리) 계산 (모든 벡터를 저장하고 모든 벡터와의 거리를 계산하여 가장 가까운 이웃을 찾음)
    - **IVF (Inverted File Indexing)**: 클러스터로 나누어 검색 속도를 높이고 정확도를 유지
    - **PQ (Product Quantization)**: 저장 공간을 절약하는 양자화 인덱스
- **확장성**: 단일 머신 및 클러스터 환경에서 확장 가능

---

### Vector Store 알고리즘 설명

- **IndexFlatL2**:
    - 유클리디안 거리(L2)를 사용하여 모든 벡터와의 거리를 계산하는 브루트포스 방식의 검색 인덱스입니다.
    - 정확도는 매우 높지만, 벡터 수가 많을 경우 계산 비용이 증가합니다.
    
- **IndexFlatIP**:
    - 내적(dot product)을 기반으로 검색을 수행하는 인덱스입니다.
    - 벡터 간의 각도 또는 방향성을 비교하는 데 유용합니다.
    - 추천 시스템이나 텍스트 임베딩에서 사용됩니다.
    
- **IndexIVFFlat**:
    - 데이터베이스를 여러 개의 클러스터로 나누고, 쿼리 벡터는 가장 가까운 클러스터를 빠르게 찾고, 그 안에서 정밀한 검색을 수행합니다.
    - 대규모 데이터셋에 대해 높은 검색 속도와 적당한 정확도를 유지할 수 있습니다.
    
- **IndexIVFPQ**:
    - **Product Quantization**을 사용하여 데이터를 더 효율적으로 저장합니다.
    - 메모리 사용량을 줄이고 빠른 검색 속도를 제공합니다.
    - 대규모 데이터셋에 적합합니다.
    
- **IndexLSH (Locality-Sensitive Hashing)**:
    - 고차원 데이터를 저차원 공간으로 해싱하여, 유사한 데이터 포인트가 같은 해시 버킷에 떨어지도록 합니다.
    - 빠른 검색 속도를 제공하지만, 정확도는 다른 방법에 비해 낮을 수 있습니다.
    
- **IndexHNSW (Hierarchical Navigable Small World)**:    
    - 그래프 기반 검색 알고리즘으로, 다수의 계층에서 효율적인 경로를 통해 빠르게 근접 이웃을 찾습니다.
    - 높은 차원의 데이터에서도 우수한 검색 성능과 정확도를 제공합니다.
    
- **IndexPQ**:    
    - 벡터를 여러 부분으로 나누고 각 부분을 양자화하여 저장합니다.
    
- **IndexSQ (Scalar Quantizer)**:
    - 벡터의 각 차원을 독립적으로 양자화하여 저장합니다.
    - 차원별로 양자화를 수행하여 더 세밀한 제어가 가능합니다.
    

---

### MMR (Maximum Marginal Relevance) 검색

- **MMR**은 유사성과 다양성의 균형을 맞추어 검색 결과의 품질을 향상시키는 알고리즘입니다.
    - **query**: 사용자로부터 입력받은 검색 쿼리
    - **k**: 최종적으로 선택할 문서의 수 (반환할 문서 개수)
    - **fetch_k**: MMR 알고리즘을 수행할 때 고려할 상위 문서의 수
    - **lambda_mult**: 쿼리와의 유사성, 선택된 문서 간의 다양성 사이의 균형을 조절 (λ=1은 유사성만 고려, λ=0은 다양성만 고려)

MMR은 **유사성**과 **다양성** 사이의 균형을 맞추어 사용자에게 더 다양한 정보를 제공합니다.
