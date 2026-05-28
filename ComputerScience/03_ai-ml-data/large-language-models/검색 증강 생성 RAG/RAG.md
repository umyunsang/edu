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
title: 필요한 라이브러리 임포트
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/07_professional-humanities/creative-writing/docs/CODEX_RAG_NOTES|CODEX_RAG_NOTES]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/07_professional-humanities/creative-writing/assignments/drafts/self_intro_ax_archive_draft|self_intro_ax_archive_draft]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/Momentum|Momentum]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/computer-vision/markdown_midterm/Computer_Vision_1_overview_v1.1|Computer_Vision_1_overview_v1.1]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
### RAG (Retrieval-Augmented Generation) 개요

![[elective_LLM__Pasted image 20250121150551.png]]

**RAG**는 생성형 AI 모델에 실시간 정보 검색 능력을 결합한 접근 방식입니다. 대형 언어 모델(LLM)이 외부 지식 소스와 연계하여 모델의 범용성과 적응력을 유지하면서도 정확하고 신뢰할 수 있는 답변을 생성할 수 있게 합니다. 주요 응용 분야로는 질문 응답 시스템(QA), 지식 검색, 고객 지원 등이 있습니다.

---

### 챗GPT의 한계와 RAG의 해결책

#### 챗GPT의 한계

1. **고정된 지식**: GPT-3.5까지의 챗GPT는 학습 데이터의 기준 시점에 고정되어 최신 정보를 반영하지 못함.
2. **한정된 도메인 지식**: 특정 분야의 깊이 있는 전문 지식을 제공하는 데 한계가 있음.
3. **맥락 이해의 한계**: 사용자의 특정 상황이나 맥락을 완벽히 이해하고 반영하기 어려움.
4. **환각(Hallucination)**: 잘못된 정보를 사실인 것처럼 제시하는 경우가 있음.

#### RAG의 해결책

1. **최신 정보 접근**: 실시간으로 외부 데이터베이스를 검색하여 최신 정보를 반영 가능.
2. **전문 지식 강화**: 특정 도메인의 전문 데이터베이스를 연결하여 깊이 있는 전문 지식 제공.
3. **맥락 인식 향상**: 사용자의 질문과 관련된 구체적인 정보를 검색하여 더 정확한 맥락 이해 가능.
4. **정보의 신뢰성 향상**: 검색된 실제 데이터를 바탕으로 답변을 생성하여 환각 현상 감소.

---

### RAG 파이프라인

1. **로딩 (Loading)**: 다양한 소스(텍스트 파일, PDF, 웹사이트, 데이터베이스, API 등)에서 데이터를 가져와 파이프라인에 입력.
    - LlamaHub에서 제공하는 다양한 커넥터를 활용할 수 있음.
    
2. **인덱싱 (Indexing)**: 데이터를 쿼리 가능한 구조로 변환.
    - 주로 벡터 임베딩을 생성하여 데이터의 의미를 수치화하고, 관련 메타데이터와 함께 저장.
    
3. **저장 (Storing)**: 생성된 인덱스와 메타데이터를 저장하여 재사용 가능하게 함.
    
4. **쿼리 (Querying)**: LLM과 LlamaIndex 데이터 구조를 활용하여 데이터를 검색.
    - 서브쿼리, 다단계 쿼리, 하이브리드 전략 등 다양한 방식으로 데이터를 검색.
    
5. **평가 (Evaluation)**: 파이프라인의 효과성을 객관적으로 측정.
    - 응답의 정확성, 충실도, 속도 등을 평가.

---

### RAG 작동 원리

1. **색인 작업 (Indexing)**
    - 다양한 외부 데이터 소스(예: 코드 파일, PDF, 텍스트 문서, 이미지 등)에서 정보를 추출.
    - **로드** → **분할** → **임베딩**: 분할된 데이터를 벡터 형태로 변환하고, 벡터 스토어에 저장.
    
2. **Retrieval (검색 단계)**
    - **질문 분석 (Question Analysis)**: 사용자의 질문을 벡터 형태로 변환.
    - **벡터 검색(Vector Search)**: 입력 문장과 관련 있는 문서를 검색 (유사도 검색, TF-IDF, BM25 등).
    
3. **Augmentation (정보 보강 단계)**
    - 검색된 정보를 언어 모델의 입력으로 제공.
    - LLM이 기본적으로 이해하는 언어적 지식에 검색된 데이터를 보강.
    
4. **Generation (생성 단계)**
    - LLM이 보강된 데이터를 바탕으로 자연스러운 텍스트 생성.
    - 응답은 단순히 검색된 내용을 복사하는 것이 아니라, 문맥을 바탕으로 생성.

---

### RAG 장점

1. **시간과 비용 절감**: Fine-tuning에 비해 적은 시간과 비용이 소요됨. 외부 데이터베이스를 활용하므로 별도의 학습 데이터 준비가 필요 없음.
2. **모델의 일반성 유지**: 특정 도메인에 국한되지 않고 다양한 분야에 대한 질문에 답변할 수 있음.
3. **답변의 근거 제공**: 답변과 함께 정보 출처를 제공하여 신뢰도를 높일 수 있음.
4. **할루시네이션 가능성 감소**: 외부 데이터를 기반으로 답변을 생성하여 모델 자체의 오류나 편향을 줄일 수 있음.

---

### Data Load 클래스 (참고)

**데이터 소스에서 정보를 읽어 들여 처리하는 다양한 클래스들**

- **WebBaseLoader**: 웹 기반 데이터 소스로부터 데이터를 로드 (웹 페이지나 API에서 사용).
- **TextLoader**: 일반 텍스트 데이터를 읽어서 LangChain에서 사용할 수 있는 형태로 변환.
- **DirectoryLoader**: 프로젝트 폴더나 지정된 경로 내의 다수 파일을 일괄적으로 처리.
- **CSVLoader**: 테이블 형태의 데이터를 처리하거나 데이터 분석 작업에 사용.
- **PyPDFLoader**: PDF 파일에서 텍스트를 추출 (PyPDF2 라이브러리 사용).
- **UnstructuredPDFLoader**: 복잡하거나 일정하지 않은 레이아웃을 가진 PDF에서 텍스트 추출.
- **PyMuPDFLoader**: PyMuPDF (Fitz) 라이브러리를 사용하여 PDF 파일로부터 텍스트 추출.
- **OnlinePDFLoader**: 웹 URL을 통해 접근 가능한 PDF 문서 처리.
- **PyPDFDirectoryLoader**: 디렉토리 내의 모든 PDF 파일을 로드하여 데이터를 추출.

---

### RAG: 데이터 로딩 및 텍스트 분할

RAG(Retrieval-Augmented Generation)은 정보를 검색한 뒤 이를 활용해 자연어 생성 모델의 응답을 향상시키는 기술입니다. 이 과정에서 데이터 로딩과 텍스트 분할은 매우 중요한 초기 단계입니다. 아래 코드는 이 두 과정을 예제와 함께 설명합니다.

#### 데이터 로딩 (Data Load)

먼저, 웹페이지에서 텍스트 데이터를 추출해 문서 객체 리스트로 변환하는 과정을 살펴봅니다.

```python
# 필요한 라이브러리 임포트
from langchain_community.document_loaders import WebBaseLoader

# 데이터 로드할 URL 지정
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'

# WebBaseLoader를 사용하여 URL에서 데이터 로드
loader = WebBaseLoader(url)
docs = loader.load()

# 로드된 문서 정보 출력
print(len(docs))  # 문서 객체의 개수
print(len(docs[0].page_content))  # 첫 번째 문서의 내용 길이
print(docs[0].page_content[5000:6000])  # 첫 번째 문서의 일부 내용 출력
```

- `WebBaseLoader(url)`: 주어진 URL의 내용을 텍스트로 로드하는 클래스입니다.
- `docs`: 로드된 문서 리스트입니다. 각 문서는 `Document` 객체로 저장됩니다.
    - `docs[0].page_content`: 첫 번째 문서의 전체 텍스트 내용입니다.
    - `docs[0].metadata`: 문서의 메타데이터(예: URL, 제목 등)를 포함합니다.

---
#### 텍스트 분할 (Text Split)

긴 텍스트 문서를 일정한 크기의 청크(chunk)로 분할합니다. 이렇게 하면 각 청크를 독립적으로 처리할 수 있어 검색과 생성 단계에서 더 효율적입니다.

```python
# 텍스트 분할을 위한 클래스 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 텍스트 분할기 설정: 1000자 크기의 청크 생성, 200자 중복 유지
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)

# 로드된 문서를 작은 청크로 분할
splits = text_splitter.split_documents(docs)

# 결과 확인
print(len(splits))  # 생성된 청크 개수
print(splits[10].page_content)  # 11번째 청크의 텍스트 내용
print(splits[10].metadata)  # 11번째 청크의 메타데이터
```

- **`chunk_size=1000`**: 각 청크의 최대 길이를 1000자로 설정합니다.
- **`chunk_overlap=200`**: 청크 간 200자를 중복시켜 문맥 연결을 유지합니다.
    - 중복은 청크를 연결했을 때 문맥이 끊기지 않도록 돕습니다.
- **`text_splitter.split_documents(docs)`**: `docs` 리스트에 있는 문서를 지정된 설정에 따라 청크로 분할합니다.
- **`splits`**: 분할된 청크 리스트입니다.

---
### RAG: 인덱싱 및 임베딩 생성

RAG(Retrieval-Augmented Generation)에서 **인덱싱과 임베딩 생성**은 텍스트 데이터를 검색하기 쉽게 변환하는 과정입니다. 이를 통해 모델이 필요한 정보를 빠르게 찾고, 질문에 적절히 응답할 수 있습니다.

#### 1. 인덱싱 (Indexing)

텍스트 데이터를 임베딩(벡터 표현)으로 변환하고 벡터 저장소에 저장하여 유사성 검색(similarity search)을 수행합니다.

```python
# 필요한 라이브러리 임포트
from langchain_community.vectorstores import Chroma  # 벡터 저장소 관리
from langchain_openai import OpenAIEmbeddings        # 임베딩 생성

# Chroma 벡터 저장소에 문서 저장 (임베딩 생성 및 저장 과정 포함)
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

# 유사성 검색: 질문과 가장 관련 있는 문서 검색
docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")

# 검색 결과 확인
print(len(docs))  # 검색 결과로 나온 문서 개수
print(docs[0].page_content)  # 가장 관련 있는 문서 내용 출력
```

1. **`OpenAIEmbeddings`**: 텍스트 데이터를 벡터(숫자 배열)로 변환하는 도구입니다.
    - 예: 텍스트 `["hello"]` → 벡터 `[0.1, 0.3, ...]` (1536차원 배열)
2. **`Chroma`**: 벡터 데이터를 저장하고 검색할 수 있는 벡터 저장소입니다.
    - `from_documents`: 텍스트를 임베딩으로 변환한 뒤, 저장소에 저장합니다.
3. **유사성 검색**: `similarity_search`는 입력 질문과 가장 유사한 문서를 반환합니다.
    - 입력 질문과 저장된 벡터 간의 유사도를 계산합니다(코사인 유사도 등 사용).

---
#### 2. 벡터 임베딩 전략

##### **임베딩이란?**

임베딩은 텍스트를 숫자 벡터로 변환하여 컴퓨터가 의미를 이해하고 비교할 수 있도록 돕는 과정입니다. 이를 효율적으로 처리하기 위한 전략은 다음과 같습니다:

1. **모델 선택**:
    
    - 고성능 모델 사용: 예를 들어 `text-embedding-ada-002`는 OpenAI에서 제공하는 강력한 임베딩 생성 모델입니다.
    - 이 모델은 일반적으로 1536차원의 벡터를 생성하며, 높은 품질의 유사성 검색을 지원합니다.
2. **차원 최적화**:
    
    - 1536차원은 임베딩 생성에 자주 사용되는 표준입니다.
    - 차원 수가 너무 크거나 작으면 검색 성능이 저하될 수 있으므로, 적절한 차원을 유지하는 것이 중요합니다.
3. **정규화 및 배치 처리**:
    
    - **정규화**: 벡터를 크기(길이)가 1이 되도록 정규화하여 계산 안정성을 확보합니다.
    - **배치 처리**: 대규모 데이터셋을 처리할 때는 데이터를 나누어(배치) 임베딩을 생성해 메모리 사용량을 최적화합니다.

---
### RAG: 검색 및 생성 (Retrieval & Generation)

**RAG**의 검색 및 생성 과정은 사용자 질문에 대해 관련 정보를 먼저 검색한 후, 이를 기반으로 LLM(Large Language Model)을 활용하여 답변을 생성하는 단계입니다. 이 과정을 통해 질문에 맞는 정확한 답변을 제공할 수 있습니다.

---
1. **검색 (Retrieval)**:
    - 사용자의 질문과 관련된 정보를 벡터 데이터베이스(VectorDB)에서 검색.
    - 검색된 문서(문맥)를 바탕으로 질문에 대한 배경 정보를 제공합니다.
    
2. **생성 (Generation)**:
    - 검색된 문서를 LLM에 입력하여 자연스럽고 명확한 답변을 생성.

```python
# RAG : 검색 및 생성 (Retrieval & Generation)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

template = '''Answer the question based only on the following context: 
{context} 

Question: {question}'''
# Template기반의 Prompt객체 (사용자의 질문 포함)
prompt = ChatPromptTemplate.from_template(template) 
# 검색결과 기반으로 응답 텍스트 생성할 LLM모델 생성
model = ChatOpenAI(model='gpt-4o-mini', temperature=0) 
# VectorDB의 검색 엔진 객체 생성
retriever = vectorstore.as_retriever()                 

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)
    
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt 
    | model 
    | StrOutputParser()
)
rag_chain.invoke("격하 과정에 대해서 설명해주세요.")
```

---

### RAG: 검색기(Retriever) 최적화

**검색기 최적화**는 사용자가 제시한 질문에 대해 더 적절하고 관련성이 높은 문서를 반환하는 것을 목표로 합니다. 각 검색기 전략은 검색 요구사항에 맞게 활용할 수 있는 특성을 가지고 있습니다. 아래에서 각 전략을 상세히 설명하겠습니다.

---
#### `BM25 Retriever`
- **특징**:
    - 고전적인 정보 검색 알고리즘으로 정확한 단어 매칭에 강점.
    - 단어 빈도(TF)와 역문서 빈도(IDF)를 활용하여 질의와 문서의 관련성을 계산.
- **장점**:
    - 검색 대상과 사용자의 질문이 단어 수준에서 일치하는 경우 높은 성능을 발휘.
- **적용 예**:
    - 법률 문서나 기술 문서처럼 정확한 용어가 중요한 데이터.

---
#### `Vector Store Retriever`
- **특징**:
    - 텍스트 데이터를 벡터로 변환한 후, 의미론적 유사성을 기반으로 검색.
    - 임베딩 모델(예: `text-embedding-ada-002`)을 사용하여 문장의 의미를 벡터로 표현.
- **장점**:
    - 질의와 문서가 정확히 일치하지 않아도 유사한 의미를 가지는 문서를 검색 가능.
- **적용 예**:
    - 사용자 질의가 자연어 형태로 다양하게 표현될 때 유용.

---
#### `MultiQuery Retriever`
- **특징**:
    - 단일 질의를 다양한 방식으로 변형하여 다수의 질의를 생성.
    - 여러 질의를 사용해 더 많은 관련 문서를 검색.
- **장점**:
    - 질문의 뉘앙스나 표현 방식이 다양할 때도 고유한 관련 문서를 찾을 수 있음.
- **적용 예**:
    - 동일한 질문에 대해 다양한 측면의 답변이 필요할 때.

---
#### `Ensemble Retriever`
- **특징**:
    - 여러 검색 방법(BM25, Vector Store 등)을 결합하여 복합적으로 검색.
    - 각 검색기의 결과를 조합하여 최종 순위를 결정.
- **장점**:
    - 다양한 검색 방식의 장점을 통합하여 높은 검색 품질 제공.
- **적용 예**:
    - 다중 도메인 데이터나 복잡한 검색 시나리오에서 최적.

---
#### `MMR (Maximal Marginal Relevance)`
- **특징**:
    - 검색된 문서의 **다양성**과 **관련성** 사이의 균형을 유지.
    - 이미 선택된 문서와의 중복을 피하면서 새로운 정보를 포함하는 문서를 선택.
- **장점**:
    - 중복된 정보 없이 다각도의 정보를 제공.
- **적용 예**:
    - 뉴스 기사 검색, 요약 생성, 다중 문서 분석.

---

|전략|주된 사용 사례|장점|한계|
|---|---|---|---|
|**BM25 Retriever**|정확한 단어 기반 매칭이 중요한 경우|높은 정확도|단어 수준의 검색 한계|
|**Vector Store Retriever**|의미론적 검색이 중요한 경우|자연어 질의에 강점|임베딩 품질에 의존|
|**MultiQuery Retriever**|질의 변형이 필요한 경우|다양한 각도의 정보 제공 가능|추가 연산 비용 발생|
|**Ensemble Retriever**|복합적인 검색 요구사항|검색 품질 최적화|구현 복잡성 증가|
|**MMR**|중복 없는 결과가 중요한 경우|정보 다양성과 관련성 조화|연산 비용 증가|

---

#### 추천 활용 시나리오

1. **BM25**: 법률 문서, 데이터베이스 검색처럼 구조화된 데이터 검색.
2. **Vector Store**: 자연어 기반 대화형 검색.
3. **MultiQuery**: 질의가 추상적이거나 모호할 때.
4. **Ensemble**: 이종 데이터셋을 통합 검색해야 할 때.
5. **MMR**: 뉴스, 학술 연구에서 다각도 분석이 필요할 때.

---

### RAG: 순위 재조정 (Rerank)

- **Rerank**: 검색된 문서들의 순위를 재조정하여 더 관련성 높은 문서를 상위에 배치.
- **한국어 특화 Reranker**: 한국어 데이터에 특화된 'Dongjin-kr/ko-reranker' 모델 사용.

---
### RAG 기반 Q&A 웹서비스 개발

#### 1. 필요한 라이브러리 설치

RAG (Retrieval-Augmented Generation) 기반 Q&A 웹서비스를 개발하려면 필요한 라이브러리들을 설치해야 합니다. 아래 명령어로 필요한 라이브러리들을 설치합니다:

```bash
pip install langchain langchain_openai chromadb streamlit Wikipedia langchain_community
```

- `WikipediaLoader` : Wikipedia에서 문서를 로드합니다. 
- `RecursiveCharacterTextSplitter` : 긴 문서를 작은 청크로 분할합니다. 
- `OpenAIEmbeddings` : 텍스트를 벡터로 변환합니다. 
- `Chroma` : 벡터 데이터베이스로 사용됩니다. 
- `ChatOpenAI` : OpenAI의 GPT 모델을 사용합니다. 
- `RetrievalQA` : 검색-질문 응답 체인을 생성합니다. 
- `PromptTemplate` : 사용자 정의 프롬프트를 생성합니다.

---
#### 2. Streamlit 예제

Streamlit을 사용하면 파이썬 코드를 이용해 간단하게 웹 애플리케이션을 만들 수 있습니다. 예를 들어, 아래와 같은 코드를 사용하여 랜덤 데이터를 표시하고, 간단한 차트를 그릴 수 있습니다.

```python
import streamlit as st  # streamlit 라이브러리 임포트
import pandas as pd  # 데이터 처리를 위한 pandas 임포트
import numpy as np  # 숫자 계산을 위한 numpy 임포트

# Streamlit 애플리케이션의 제목 설정
st.title('Streamlit Example')

# 랜덤 데이터를 생성하여 DataFrame으로 변환
data = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])

# 생성된 데이터 표시
st.write("Here's our random data:")
st.write(data)

# 생성된 데이터로 라인 차트 그리기
chart = st.line_chart(data)
```

- `st.title('Streamlit Example')`: 애플리케이션의 제목을 설정합니다.
- `pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])`: 랜덤한 숫자 데이터 50개, 3개의 열을 생성하고 DataFrame 형식으로 저장합니다.
- `st.write(data)`: DataFrame을 웹 페이지에 표시합니다.
- `st.line_chart(data)`: DataFrame의 데이터를 기반으로 라인 차트를 그립니다.

---

#### 3. Streamlit 애플리케이션 실행

Streamlit 애플리케이션을 실행하려면 아래 명령어를 사용합니다. `your_script.py`는 사용자가 작성한 파이썬 스크립트 파일을 의미합니다.

```bash
streamlit run your_script.py
```

위 명령어를 입력하면, Streamlit 웹 애플리케이션이 자동으로 실행되어 웹 브라우저에서 애플리케이션을 확인할 수 있습니다.

---
