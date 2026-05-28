---
aliases: []
course: large-language-models
created: '2024-06-12'
date: '2024-06-12'
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
title: Llama Index
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Adam|Adam]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
#### LlamaIndex란?
LlamaIndex는 LLM에서 학습되지 않은 데이터를 활용하여 질의응답 AI를 쉽게 개발할 수 있는 오픈소스 라이브러리입니다. 외부 데이터를 LLM에 주입해 더 정확하고 최신의 응답을 생성할 수 있으며, RAG(Retrieval-Augmented Generation) 작업 흐름을 간단한 Python 코드로 구현할 수 있도록 지원합니다. 이 라이브러리는 내부적으로 LangChain을 활용하며, 다음과 같은 특징과 기능을 제공합니다:

- **다양한 데이터 소스 활용**  
    텍스트, PDF, API 등 다양한 형태의 데이터를 처리할 수 있어, 고객 지원 챗봇, 학술 연구, 의료 정보 제공 등 다양한 응용 분야에 적용 가능합니다.
    
- **효율적인 데이터 처리 및 쿼리 기능**  
    데이터를 적재, 구조화, 쿼리, 통합하는 과정을 지원하며, 고급 검색 및 쿼리 인터페이스를 제공합니다. 복잡한 RAG 파이프라인도 간단히 구현 가능하여 사용자 요구에 맞춘 솔루션을 쉽게 개발할 수 있습니다.
    
- **유연성과 확장성**  
    다양한 모델 및 프레임워크와 통합 가능하도록 설계되어, 개인 데이터를 가져오고 구조화하거나 고급 검색 인터페이스를 생성하는 데 유용합니다. ChatGPT와의 통합도 지원합니다.
    
- **참고 자료**  
    관련 내용은 [GitHub](https://github.com/run-llama/llama_index)와 [공식 문서](https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader//) 에서 확인할 수 있습니다.
---
#### LlamaIndex 주요 기능

- **다양한 데이터 소스 지원**
    - 텍스트, PDF, ePub, 워드, 파워포인트, 오디오를 비롯한 다양한 파일 형식과 트위터, 슬랙, 위키피디아 등 웹 서비스를 자체 데이터로 지정 가능
    
- **벡터 임베딩 및 인덱싱**
    - 로드한 데이터를 벡터 임베딩으로 변환하고 효율적인 검색을 위해 인덱싱
    
- **효율적인 검색 알고리즘**
    - 사용자의 쿼리에 대해 가장 관련성이 높은 문서나 데이터 조각을 검색
    
- **LLM 기반 응답 생성**
    - 검색된 관련 문서를 바탕으로 LLM을 활용하여 사용자 쿼리에 대한 정확하고 상세한 응답 생성
    
- **모듈화된 구조 설계**
    - 각 컴포넌트를 필요에 따라 커스터마이즈하거나 교체 가능
    
- **다양한 데이터베이스 및 모델 지원**
    - 벡터 데이터베이스, 임베딩 모델, LLM 등을 유연하게 지원
    
- **쿼리 최적화**
    - 복잡한 쿼리를 자동으로 분해하고 최적화하여 더 정확한 응답 생성
    
- **멀티모달 데이터 처리**
    - 텍스트뿐만 아니라 이미지, 오디오 등 다양한 형태의 데이터를 처리 가능

---
#### LlamaIndex 아키텍처

- **다양한 데이터 소스 지원**
    1. 데이터는 "인덱스" 형태로 변환되어 쿼리에 사용될 수 있도록 준비됨
    2. 사용자가 질문을 입력하면, 쿼리는 인덱스에 전달됨
    3. 인덱스는 사용자의 쿼리와 가장 관련성 높은 데이터를 필터링
    4. 필터링된 관련 데이터, 원래 쿼리, 적절한 프롬프트가 LLM에 전달
    5. 정보를 바탕으로 LLM이 응답 생성
    6. 생성된 응답이 사용자에게 전달됨

---

#### LlamaIndex 주요 구성 요소

- **Loading**
    - 텍스트 파일, PDF 파일, 웹 사이트, 데이터베이스, API 등의 데이터를 파이프라인에 넣는 단계
    - LlamaHub에 다양한 **Connector**가 존재
- **Document**
    - 데이터 소스를 담는 컨테이너 역할
- **Node**
    - LlamaIndex 데이터의 한 단위이며 Document를 작은 청크로 나눔
    - Node에는 메타데이터 포함
- **Connectors**
    - **Reader**라고도 불리며, 다양한 소스에서 데이터를 받아 Document와 Node를 생성
- **Indexing**
    - 데이터를 쿼리할 수 있는 자료 구조를 생성
    - LLM에서는 주로 벡터 임베딩을 생성
- **Indexes**
    - 데이터를 ingest한 뒤 검색(retrieve)하기 쉬운 구조로 변경
    - 주로 벡터 임베딩을 생성하며, 데이터에 대한 메타데이터를 포함할 수 있음
- **Embeddings**
    - LLM은 숫자로 표현된 데이터(임베딩)를 생성
- **Storing**
    - 생성된 인덱스를 저장하여 재사용 가능
- **Querying**
    - 저장된 인덱스를 활용해 효율적인 데이터 검색 및 쿼리 수행
---
