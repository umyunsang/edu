---
aliases: []
course: large-language-models
created: '2025-01-21'
date: '2025-01-21'
semester: elective
source: ''
status: seedling
tags:
- cs/llm
- cs/nlp
- type/lecture
title: 음성 파일 저장
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

### 1. Text to Speech (TTS)

**TTS 모델**은 텍스트를 음성으로 변환하며, 다양한 언어 및 음성 스타일을 제공합니다.  
즉각적인 오디오 피드백과 실시간 스트리밍 기능도 지원합니다.

#### 주요 특징

- **지원 음성**: `alloy`, `ash`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`.
- **지원 출력 형식**: 기본은 `mp3`, 추가적으로 `opus`, `aac`, `flac`, `pcm` 가능.
- **지원 언어**: Whisper 모델과 동일한 언어 지원.

#### TTS 실습 코드

```python
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="2025년은 푸른 뱀의 해입니다."
)

# 음성 파일 저장
response.stream_to_file("output.mp3")
```

---

### 2. Speech to Text (STT

**STT 모델**은 Whisper를 기반으로 하며, 오디오를 텍스트로 변환(Transcription)하거나 번역(Translation)합니다.

#### 주요 특징

- **지원 모델**: `large-v2 Whisper`.
- **파일 크기 제한**: 최대 25MB.
- **지원 입력 파일 형식**: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`.
- **고급 기능**:
    - 타임스탬프를 포함한 JSON 구조화된 출력 가능.
    - PyDub 라이브러리를 통해 큰 파일을 분할 처리.

---

### 2.1. Transcriptions (오디오 -> 텍스트 변환

#### 실습 코드

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 오디오 파일 열기
audio_file = open("output.mp3", "rb")

# Whisper 모델로 텍스트 변환
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)

# 변환된 텍스트 출력
print(transcription)
```

---

### 2.2. Translations (오디오 -> 번역 텍스트 변환

#### 실습 코드

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 오디오 파일 열기
audio_file = open("output.mp3", "rb")

# Whisper 모델로 번역 수행
translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
)

# 번역된 텍스트 출력
print(translation.text)
```

---

### 활용 시나리오

1. **TTS 활용**:
    
    - 콘텐츠 음성화(예: 뉴스 읽기, 챗봇 음성 응답).
    - 스트리밍 기반의 실시간 피드백 서비스.
2. **STT 활용**:
    
    - 음성 기록 텍스트화(회의 기록, 강의 녹음).
    - 멀티언어 오디오 번역(국제 컨퍼런스).

**참고**: [OpenAI TTS/STT Documentation](https://platform.openai.com/docs/guides/text-to-speech)
