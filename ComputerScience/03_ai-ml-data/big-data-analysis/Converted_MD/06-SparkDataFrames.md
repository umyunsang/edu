---
aliases: []
course: big-data-analysis
created: '2025-09-24'
date: '2025-09-24'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 3-2
source: ''
status: seedling
tags:
- cs/db
- cs/ml
- type/lecture
title: 06. Spark DataFrames - 구조화된 데이터 처리
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/빅데이터분석 인터페이스|빅데이터분석 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[LGAimer/LG Aimers 9기 평가 및 제출 가이드|LG Aimers 9기 평가 및 제출 가이드]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[LGAimer/LG Aimers 9기 지원서 초안|LG Aimers 9기 지원서 초안]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/신호 특징 분석 결과|신호 특징 분석 결과]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/빅데이터분석 근거 인덱스|빅데이터분석 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/decision tree|decision tree]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/sql|sql]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/ai|ai]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

# 06. Spark DataFrames - 구조화된 데이터 처리

## 📚 개요

Spark DataFrames는 대용량 구조화된 데이터를 효율적으로 처리하기 위한 고수준 API입니다. 이 실습에서는 DataFrames의 핵심 개념과 실제 활용법을 학습합니다.

### 🎯 학습 목표
- Spark DataFrames의 기본 개념과 특징 이해
- RDD와 DataFrames의 차이점 파악
- 구조화된 데이터 처리 방법 학습
- 실제 데이터 분석 작업에 DataFrames 적용

## 🚀 Spark DataFrames 소개

### 📋 **DataFrames의 핵심 특징**
- **접근성 향상**: "빅데이터" 엔지니어가 아닌 더 넓은 사용자층이 분산 처리의 힘을 활용
- **영감의 원천**: R과 Python(Pandas)의 데이터 프레임에서 영감을 받아 설계
- **현대적 설계**: 현대적인 빅데이터와 데이터 사이언스 애플리케이션을 지원하도록 처음부터 설계
- **API 확장**: 기존 RDD API의 확장으로 더욱 강력한 기능 제공

## 📚 참고 자료

### 🔗 **추가 학습 자료**
- [Spark SQL, DataFrames and Datasets Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
  - Apache Spark 공식 문서: DataFrames와 Datasets 가이드
- [Introduction to DataFrames - Python](https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html)
  - Databricks 공식 가이드: Python DataFrames 소개
- [PySpark Cheat Sheet: Spark DataFrames in Python](https://www.datacamp.com/community/blog/pyspark-sql-cheat-sheet)
  - DataCamp 치트시트: PySpark DataFrames 핵심 문법

## 📊 DataFrames의 핵심 개념

### 🎯 **DataFrames란?**
- **선호되는 추상화**: Spark에서 가장 선호되는 데이터 추상화
- **강타입 컬렉션**: 분산된 요소들의 강타입 컬렉션
- **RDD 기반**: Resilient Distributed Datasets(RDD) 위에 구축
- **불변성**: 생성 후 변경할 수 없는 불변 데이터 구조

### 🔧 **DataFrames의 주요 기능**
- **계보 추적**: 데이터 계보 정보를 추적하여 손실된 데이터를 효율적으로 재계산
- **병렬 처리**: 요소 컬렉션에 대한 병렬 연산 지원
- **최적화**: 지능적인 최적화와 코드 생성으로 성능 향상

### 🏗️ **DataFrames 생성 방법**
- **기존 컬렉션 병렬화**: Pandas DataFrames 등 기존 컬렉션을 병렬화
- **기존 DataFrames 변환**: 기존 DataFrames를 변환하여 새로운 DataFrames 생성
- **파일에서 로드**: HDFS나 기타 스토리지 시스템의 파일에서 로드 (예: Parquet)

### 🚀 **DataFrames의 주요 특징**
- **확장성**: 단일 노트북의 킬로바이트부터 대규모 클러스터의 페타바이트까지 확장
- **다양한 형식 지원**: 광범위한 데이터 형식과 스토리지 시스템 지원
- **통합성**: Spark를 통한 모든 빅데이터 도구 및 인프라와의 원활한 통합
- **다국어 지원**: Python, Java, Scala, R을 위한 API 제공

### 🔄 **DataFrames vs RDDs**
- **사용자 친화적**: 다른 프로그래밍 언어의 데이터 프레임에 익숙한 사용자에게 친숙한 API
- **프로그래밍 용이성**: 기존 Spark 사용자에게 RDD보다 더 쉬운 프로그래밍 환경 제공
- **성능 향상**: 지능적인 최적화와 코드 생성을 통한 성능 개선

## 🐍 PySpark Shell - 개발 환경 설정

### 🚀 **PySpark Shell 실행**

#### 📋 **기본 실행 방법**
```bash
pyspark
```

#### 💡 **PySpark Shell의 특징**
- **대화형 환경**: REPL(Read-Eval-Print Loop) 방식의 대화형 개발 환경
- **자동 설정**: SparkContext와 SparkSession이 자동으로 생성
- **즉시 실행**: 코드를 입력하면 즉시 실행되어 결과 확인 가능
- **디버깅**: 실시간으로 코드를 테스트하고 디버깅 가능

### 📊 **PySpark Shell 출력 예시**

다음과 유사한 출력이 표시되며, `>>>` REPL 프롬프트가 나타납니다:

```
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56)
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
2018-09-18 17:13:13 WARN  NativeCodeLoader:62 - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.3.1
      /_/

Using Python version 3.6.5 (default, Apr 29 2018 16:14:56)
SparkSession available as 'spark'.
>>>
```

### 💡 **출력 해석**
- **Python 버전**: Python 3.6.5 사용
- **경고 메시지**: 네이티브 Hadoop 라이브러리 로드 실패 (정상적인 현상)
- **로그 레벨**: 기본 로그 레벨이 "WARN"으로 설정
- **SparkSession**: `spark` 변수로 SparkSession 사용 가능
- **REPL 프롬프트**: `>>>` 프롬프트에서 코드 입력 가능

## 📊 데이터 읽기 및 DataFrames 생성

### 🔧 **CSV 파일 읽기**

```python
# CSV 파일을 DataFrames로 읽기
df = sqlContext.read.csv("/tmp/irmar.csv", sep=';', header=True)
```

~~~
>>> df2.show()
+---+--------------------+------------+------+------------+--------+-----+---------+--------+
|_c0|                name|       phone|office|organization|position|  hdr|    team1|   team2|
+---+--------------------+------------+------+------------+--------+-----+---------+--------+
|  0|      Alphonse Paul |+33223235223|   214|          R1|     DOC|False|      EDP|      NA|
|  1|        Ammari Zied |+33223235811|   209|          R1|      MC| True|      EDP|      NA|
.
.
.
| 18|    Bernier Joachim |+33223237558|   214|          R1|     DOC|False|   ANANUM|      NA|
| 19|   Berthelot Pierre |+33223236043|   601|          R1|      PE| True|       GA|      NA|
+---+--------------------+------------+------+------------+--------+-----+---------+--------+
only showing top 20 rows
~~~

## 🔄 Transformations, Actions, Laziness - DataFrames의 핵심

### 📋 **DataFrames의 지연 실행**

#### 🎯 **Lazy Evaluation의 특징**
- **RDD와 동일**: DataFrames도 RDD와 마찬가지로 지연 실행(lazy)
- **쿼리 계획**: Transformations은 쿼리 계획에 기여하지만 실제로는 아무것도 실행하지 않음
- **실행 트리거**: Actions가 호출될 때 쿼리가 실행됨
- **최적화**: 지연 실행을 통해 불필요한 계산을 피하고 성능을 최적화

#### 💡 **Lazy Evaluation의 장점**
- **성능 최적화**: 불필요한 중간 계산을 피하여 효율성 증대
- **메모리 절약**: 필요한 데이터만 메모리에 유지
- **자동 최적화**: Spark가 자동으로 최적의 실행 계획 수립
- **유연성**: 복잡한 변환 체인을 쉽게 구성 가능

### 🔧 **Transformation 예제**

#### 📋 **주요 Transformation 함수들**
- **`filter()`**: 조건을 만족하는 행만 필터링
- **`select()`**: 특정 컬럼만 선택
- **`drop()`**: 특정 컬럼 제거
- **`intersect()`**: 두 DataFrames의 교집합 계산
- **`join()`**: 두 DataFrames를 조인

### 🎯 **Action 예제**

#### 📋 **주요 Action 함수들**
- **`count()`**: 행의 개수 계산
- **`collect()`**: 모든 데이터를 드라이버에 수집 
- **`show()`**: 데이터를 테이블 형태로 표시
- **`head()`**: 처음 몇 개 행 반환
- **`take()`**: 지정된 개수만큼 행 반환

## 🏗️ Python에서 DataFrames 생성

### 📋 **DataFrames 생성 방법**
- **기존 컬렉션**: Python 리스트나 딕셔너리에서 생성
- **파일 읽기**: CSV, JSON, Parquet 등 다양한 형식의 파일에서 로드
- **데이터베이스**: JDBC를 통한 데이터베이스 연결
- **변환**: 기존 DataFrames를 변환하여 새로운 DataFrames 생성

### 🔧 **기본 설정 및 라이브러리**

```python
import sys, subprocess
import os

# PySpark Python 경로 설정
os.environ["PYSPARK_PYTHON"] = sys.executable
```

### 🚀 **SparkSession 생성**

```python
from pyspark.sql import SparkSession

# SparkSession 생성
spark = SparkSession.builder \
    .appName("DataFramesExample") \
    .getOrCreate()

# SparkContext 생성
sc = spark.sparkContext
```

### 💡 **설정 설명**
- **PYSPARK_PYTHON**: PySpark가 사용할 Python 인터프리터 경로 설정
- **SparkSession**: DataFrames 작업을 위한 진입점
- **SparkContext**: RDD 작업을 위한 진입점

### 🔧 **추가 설정 (PySpark Shell에서는 불필요)**

```python
from pyspark import SparkContext, SparkConf, SQLContext

# 다음 세 줄은 PySpark Shell에서는 불필요
# PySpark Shell에서는 자동으로 설정됨
# Spark 설정
conf = SparkConf().setAppName("people").setMaster("local[*]") 
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")  # 로그 레벨을 ERROR로 설정
sqlContext = SQLContext(sc)  # SQLContext 생성
```

### 💡 **설정 설명**
- **SparkConf**: Spark 애플리케이션 설정
- **setAppName**: 애플리케이션 이름 설정
- **setMaster**: 마스터 노드 설정 (local[*]는 로컬의 모든 코어 사용)
- **setLogLevel**: 로그 레벨을 ERROR로 설정하여 불필요한 출력 제거
- **SQLContext**: DataFrames 작업을 위한 SQL 컨텍스트

## 📊 데이터 로드 및 DataFrames 생성

### 🔧 **JSON 파일에서 DataFrames 생성**

```python
# JSON 파일에서 DataFrames 생성
df = sqlContext.read.json("data/people.json")
```

### 💡 **데이터 로드 방법**
- **JSON 파일**: `read.json()` 메소드 사용
- **CSV 파일**: `read.csv()` 메소드 사용
- **Parquet 파일**: `read.parquet()` 메소드 사용
- **텍스트 파일**: `read.text()` 메소드 사용

### 📋 **데이터 확인**

```python
# 처음 24개 행 표시
df.show(24)
```

### 💡 **show() 메소드의 특징**
- **기본값**: `show()`는 처음 20개 행만 표시
- **개수 지정**: `show(n)`으로 표시할 행 수 지정 가능
- **테이블 형태**: 데이터를 테이블 형태로 깔끔하게 표시
- **컬럼 정렬**: 컬럼이 자동으로 정렬되어 표시

## 🔍 Schema Inference - 스키마 추론

### 📋 **스키마 추론이란?**
- **자동 스키마**: Spark가 데이터의 구조를 자동으로 분석하여 스키마 생성
- **구조화된 데이터**: JSON, Parquet 등 구조화된 형식에서 자동으로 스키마 추론
- **비구조화 데이터**: 텍스트 파일 등에서는 수동으로 스키마 정의 필요

### 🎯 **실습: irmar.txt 파일 분석**

#### 📋 **데이터 특징**
- **파일명**: `irmar.txt`
- **구조**: 구조화된 데이터이지만 자체 설명 스키마가 없음
- **형식**: JSON이 아니므로 Spark가 자동으로 스키마를 추론할 수 없음
- **목표**: RDD를 생성하고 파일의 처음 몇 행을 확인

### 🔧 **텍스트 파일 읽기 및 데이터 확인**

```python
# 텍스트 파일을 RDD로 읽기
rdd = sc.textFile("data/irmar.csv")

# 처음 10개 행 출력
for line in rdd.take(10):
    print(line)
```

### 💡 **코드 설명**
- **`textFile()`**: 텍스트 파일을 RDD로 읽기
- **`take(10)`**: 처음 10개 행만 가져오기
- **`print(line)`**: 각 행을 출력하여 데이터 구조 확인

## 🎯 실습 연습문제

### 📋 **연습문제 목표**
- **DataFrames 생성**: 다양한 방법으로 DataFrames 생성
- **데이터 조작**: 필터링, 선택, 변환 등 기본 연산 수행
- **스키마 이해**: 데이터의 구조와 타입 파악
- **실제 활용**: 실제 데이터 분석 작업에 DataFrames 적용

### 📚 **참고 자료**
- [DataFrames API documentation](http://spark.apache.org/docs/2.3.1/api/python/index.html)
  - Apache Spark 공식 DataFrames API 문서

### 🎯 **실습 데이터: irmar.csv 파일**

#### 📋 **데이터 구조**
- **파일 경로**: `/tmp/irmar.csv`
- **데이터 형식**: 각 행은 한 사람에 대한 동일한 정보를 포함
- **구조화된 데이터**: CSV 형식의 구조화된 데이터

#### 📊 **데이터 필드**
- **name**: 사람의 이름
- **phone**: 전화번호
- **office**: 사무실 번호
- **organization**: 소속 조직
- **position**: 직책
- **hdr**: 헤더 정보
- **team1**: 팀1 정보
- **team2**: 팀2 정보

### 🔧 **데이터 처리 코드**

```python
from collections import namedtuple

# CSV 파일을 RDD로 읽기
rdd = sc.textFile("data/irmar.csv")
```

### 💡 **코드 설명**
- **`namedtuple`**: 구조화된 데이터를 처리하기 위한 Python 클래스
- **`textFile()`**: 텍스트 파일을 RDD로 읽기
- **CSV 처리**: 쉼표로 구분된 데이터를 처리

### 🔧 **데이터 구조 정의**

```python
# Person 구조체 정의
Person = namedtuple('Person', ['name', 'phone', 'office', 'organization', 
                               'position', 'hdr', 'team1', 'team2'])

# 문자열을 불린으로 변환하는 함수
def str_to_bool(s):
    if s == 'True': return True
    return False
```

### 💡 **코드 설명**
- **`namedtuple`**: 구조화된 데이터를 처리하기 위한 Python 클래스
- **`Person`**: 각 필드를 가진 구조체 정의
- **`str_to_bool()`**: 문자열을 불린 값으로 변환

### 🔧 **데이터 매핑 함수**

```python
# CSV 라인을 Person 객체로 변환하는 함수
def map_to_person(line):
    cols = line.split(";")
    return Person(name         = cols[0],
                  phone        = cols[1],
                  office       = cols[2],
                  organization = cols[3],
                  position     = cols[4], 
                  hdr          = str_to_bool(cols[5]),
                  team1        = cols[6],
                  team2        = cols[7])
```

### 💡 **코드 설명**
- **`split(";")`**: 세미콜론으로 구분된 데이터를 분리
- **`Person` 객체**: 각 필드를 가진 구조화된 데이터
- **`str_to_bool()`**: hdr 필드를 불린 값으로 변환
    
### 🔧 **RDD를 DataFrame으로 변환**

```python
# RDD를 Person 객체로 변환
people_rdd = rdd.map(map_to_person)

# RDD를 DataFrame으로 변환
df = people_rdd.toDF()
```

### 💡 **코드 설명**
- **`map(map_to_person)`**: 각 라인을 Person 객체로 변환
- **`toDF()`**: RDD를 DataFrame으로 변환
- **DataFrame**: 구조화된 데이터를 처리하기 위한 고수준 API

~~~

### 🔧 **DataFrame 표시**

```python
# DataFrame 내용 표시
df.show()
```

### 💡 **코드 설명**
- **`show()`**: DataFrame의 내용을 테이블 형태로 표시
- **기본값**: 처음 20개 행을 표시
- **옵션**: `show(n)`으로 표시할 행 수 지정 가능

~~~

### 🔍 **Schema - 데이터 구조 확인**

#### **스키마란?**
- **정의**: DataFrame의 컬럼 구조와 데이터 타입을 정의
- **역할**: 데이터 검증, 최적화, SQL 쿼리 지원
- **자동 추론**: Spark가 데이터를 분석하여 스키마를 자동으로 생성

### 🔧 **스키마 출력**

```python
# DataFrame의 스키마 출력
df.printSchema()
```

### 💡 **코드 설명**
- **`printSchema()`**: DataFrame의 스키마를 트리 형태로 출력
- **컬럼 정보**: 컬럼명, 데이터 타입, null 허용 여부
- **중첩 구조**: 복잡한 데이터 타입의 중첩 구조도 표시

### 📊 **Display - 데이터 시각화**

#### **Display란?**
- **정의**: Jupyter Notebook에서 DataFrame을 시각적으로 표시
- **장점**: 테이블 형태의 깔끔한 출력
- **기능**: 정렬, 필터링, 페이지네이션 지원

### 🔧 **Display 사용**

```python
# DataFrame을 시각적으로 표시
display(df)
```

~~~

### 🔍 **Select - 컬럼 선택**

#### **Select란?**
- **정의**: DataFrame에서 특정 컬럼을 선택하는 연산
- **용도**: 필요한 컬럼만 추출하여 메모리 사용량 최적화
- **성능**: 불필요한 데이터를 제거하여 처리 속도 향상

### 🔧 **Select 연산**

```python
# 특정 컬럼만 선택
df.select(df["name"], df["position"], df["organization"])
```

### 💡 **코드 설명**
- **`select()`**: 지정된 컬럼만 선택
- **컬럼 지정**: `df["컬럼명"]` 형태로 컬럼 지정
- **결과**: 선택된 컬럼만 포함된 새로운 DataFrame

### 🔧 **Select 결과 표시**

```python
# 선택된 컬럼의 결과를 표시
df.select(df["name"], df["position"], df["organization"]).show()
```

### 💡 **코드 설명**
- **체이닝**: `select()`와 `show()`를 연결하여 사용
- **결과**: 선택된 컬럼만 포함된 테이블 출력
- **성능**: 필요한 데이터만 처리하여 효율성 향상

### 🔍 **Filter - 데이터 필터링**

#### **Filter란?**
- **정의**: DataFrame에서 조건에 맞는 행만 선택하는 연산
- **용도**: 특정 조건을 만족하는 데이터만 추출
- **성능**: 불필요한 데이터를 제거하여 처리 속도 향상

### 🔧 **Filter 연산**

```python
# 특정 조직의 데이터만 필터링
df.filter(df["organization"] == "R2").show()
```

~~~

### 🔗 **Filter + Select - 복합 연산**

#### **복합 연산이란?**
- **정의**: Filter와 Select를 조합하여 사용하는 연산
- **용도**: 조건에 맞는 데이터에서 필요한 컬럼만 선택
- **성능**: 데이터 처리 단계를 최적화하여 효율성 향상

### 🔧 **Filter + Select 연산**

```python
# R2 조직의 데이터에서 이름과 팀1 정보만 선택
df2 = df.filter(df["organization"] == "R2").select(df['name'], df['team1'])
```

### 💡 **코드 설명**
- **`filter()`**: organization이 "R2"인 행만 선택
- **`select()`**: name과 team1 컬럼만 선택
- **체이닝**: 두 연산을 연결하여 효율적으로 처리

### 🔧 **결과 표시**

```python
# 필터링된 결과 표시
df2.show()
```

### 💡 **코드 설명**
- **`show()`**: 필터링된 DataFrame의 내용을 표시
- **결과**: R2 조직의 이름과 팀1 정보만 출력
- **효율성**: 필요한 데이터만 처리하여 성능 최적화

### 📊 **OrderBy - 데이터 정렬**

#### **OrderBy란?**
- **정의**: DataFrame의 행을 특정 컬럼 기준으로 정렬하는 연산
- **용도**: 데이터를 특정 순서로 정렬하여 분석 용이성 향상
- **성능**: 정렬 연산은 비용이 높으므로 필요한 경우에만 사용

### 🔧 **OrderBy 연산**

```python
# R2 조직의 데이터를 이름 순으로 정렬
(df.filter(df["organization"] == "R2")
   .select(df["name"], df["position"])
   .orderBy("position")).show()
```

### 💡 **코드 설명**
- **`filter()`**: R2 조직의 데이터만 선택
- **`select()`**: 이름과 직책 컬럼만 선택
- **`orderBy()`**: 직책 기준으로 정렬
- **체이닝**: 여러 연산을 연결하여 효율적으로 처리

### 📊 **GroupBy - 데이터 그룹화**

#### **GroupBy란?**
- **정의**: DataFrame의 행을 특정 컬럼 기준으로 그룹화하는 연산
- **용도**: 그룹별 집계 연산 수행
- **성능**: 그룹화 연산은 비용이 높으므로 신중하게 사용

### 🔧 **GroupBy 연산**

```python
# hdr 컬럼 기준으로 그룹화
df.groupby(df["hdr"])
```

~~~

### 🔧 **GroupBy + Count 연산**

```python
# hdr 컬럼 기준으로 그룹화하고 개수 계산
df.groupby(df["hdr"]).count().show()
```

### 💡 **코드 설명**
- **`groupby()`**: hdr 컬럼 기준으로 그룹화
- **`count()`**: 각 그룹의 개수 계산
- **`show()`**: 결과를 테이블 형태로 표시

~~~

### ⚠️ **중요한 주의사항**

#### **GroupedData.count() vs DataFrame.count()**
- **`GroupedData.count()`**: 그룹화된 데이터의 개수를 계산하는 **Transformation**
- **`DataFrame.count()`**: 전체 DataFrame의 행 수를 계산하는 **Action**
- **차이점**: GroupedData.count()는 지연 실행, DataFrame.count()는 즉시 실행

### 🔧 **DataFrame.count() 예제**

```python
# hdr가 True인 행의 개수 계산 (Action)
df.filter(df["hdr"]).count()
```

### 💡 **코드 설명**
- **`filter()`**: hdr가 True인 행만 선택
- **`count()`**: 선택된 행의 개수를 즉시 계산 (Action)
- **결과**: 정수 값 반환

### 🔧 **Filter + Select 예제**

```python
# hdr가 True인 행의 이름만 선택하여 표시
df.filter(df['hdr']).select("name").show()
```

### 💡 **코드 설명**
- **`filter()`**: hdr가 True인 행만 선택
- **`select()`**: name 컬럼만 선택
- **`show()`**: 결과를 테이블 형태로 표시

### 🔧 **Organization별 그룹화**

```python
# 조직별 개수 계산
df.groupBy(df["organization"]).count().show()
```

### 💡 **코드 설명**
- **`groupBy()`**: organization 컬럼 기준으로 그룹화
- **`count()`**: 각 그룹의 개수 계산
- **`show()`**: 결과를 테이블 형태로 표시

### 🎯 **실습 연습문제**

#### **연습문제 목표**
- **INSA 교수 수 계산**: INSA 소속 교수(PR+MC)의 총 수
- **STATS 팀 MC 수 계산**: STATS 팀의 MC(마스터 코스) 학생 수

#### **해결 방법**
1. **필터링**: 특정 조건에 맞는 데이터만 선택
2. **집계**: 선택된 데이터의 개수 계산
3. **결과 확인**: 계산된 결과를 표시
#### **추가 연습문제**
- **HDR 보유자 수**: MC+CR 중 HDR을 보유한 사람의 수
- **지도 비율 계산**: 학생 지도 비율 (DOC / HDR)
- **조직별 인원 수**: 각 조직의 총 인원 수
- **팀별 HDR 수**: 각 팀의 HDR 보유자 수
- **최대 HDR 팀**: HDR을 가장 많이 보유한 팀 찾기

#### **해결 힌트**
- **필터링**: `filter()` 함수로 조건에 맞는 데이터 선택
- **그룹화**: `groupBy()` 함수로 데이터 그룹화
- **집계**: `count()`, `sum()`, `avg()` 등으로 통계 계산
- **정렬**: `orderBy()` 함수로 결과 정렬
#### **고급 연습문제**
- **조직별 DOC 학생 수**: 각 조직의 DOC 학생 수
- **최대 DOC 팀**: DOC 학생을 가장 많이 보유한 팀
- **CNRS 비연구직**: CNRS 소속이지만 CR나 DR이 아닌 사람들

### 🔧 **실습 코드 템플릿**

```python
# 기본 템플릿
sc.stop()
```

### 💡 **코드 설명**
- **`sc.stop()`**: SparkContext 종료
- **리소스 해제**: 메모리와 연결 리소스 해제
- **중요**: 프로그램 종료 시 반드시 호출해야 함

### 🔧 **실습 시작**

```python
# 실습을 위한 기본 설정
```

### 🎯 **학습 목표 달성**

#### **이번 실습에서 배운 내용**
- **DataFrame 생성**: RDD를 DataFrame으로 변환
- **데이터 조작**: select, filter, groupBy, orderBy 연산
- **집계 연산**: count, sum, avg 등 통계 함수
- **체이닝**: 여러 연산을 연결하여 효율적으로 처리

#### **다음 단계**
- **SQL 쿼리**: DataFrame을 SQL로 쿼리하는 방법
- **고급 집계**: 복잡한 집계 연산 수행
- **성능 최적화**: DataFrame 연산의 성능 최적화
