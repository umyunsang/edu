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
title: 08-PandaDataframes
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/빅데이터분석 인터페이스|빅데이터분석 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[LGAimer/LG Aimers 9기 지원서 초안|LG Aimers 9기 지원서 초안]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/openai API 활용|openai API 활용]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[LGAimer/LG Aimers 9기 평가 및 제출 가이드|LG Aimers 9기 평가 및 제출 가이드]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/구구단 프로그램|구구단 프로그램]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/빅데이터분석 근거 인덱스|빅데이터분석 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/decision tree|decision tree]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/sql|sql]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/ai|ai]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

# 08-PandaDataframes

# 📊 **Pandas DataFrames - 2차원 데이터 구조**

## 🎯 **개요**

### **학습 목표**
- **DataFrame 이해**: 2차원 데이터 구조의 핵심 개념
- **데이터 생성**: 다양한 방법으로 DataFrame 생성
- **데이터 조작**: 인덱싱, 필터링, 정렬, 집계
- **데이터 변환**: stack, unstack, transpose
- **실제 데이터**: CSV, HTML에서 데이터 로드
- **그룹화**: groupby를 활용한 데이터 집계

### **DataFrame이란?**
- **정의**: 행과 열로 구성된 2차원 데이터 구조
- **구조**: Series의 집합체 (각 컬럼이 Series)
- **용도**: 테이블 형태의 데이터 처리
- **장점**: SQL과 유사한 데이터 조작 가능

### 🔧 **환경 설정**

```python
# Jupyter Notebook 설정
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# 필수 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 표시 옵션 설정
pd.set_option("display.max_rows", 8)
plt.rcParams['figure.figsize'] = (9, 6)
```

### 💡 **설정 설명**
- **`%matplotlib inline`**: 그래프를 노트북에서 인라인으로 표시
- **`%config InlineBackend.figure_format = 'retina'`**: 고해상도 그래프
- **`pd.set_option("display.max_rows", 8)`**: 표시할 최대 행 수 제한
- **`plt.rcParams['figure.figsize'] = (9, 6)`**: 그래프 크기 설정

## 🔧 **DataFrame 생성**

### **방법 1: 날짜 인덱스와 랜덤 데이터**

```python
# 날짜 범위 생성
dates = pd.date_range('20130101', periods=6)

# 랜덤 데이터로 DataFrame 생성
pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
```

### 💡 **코드 설명**
- **`pd.date_range()`**: 2013-01-01부터 6일간의 날짜 범위 생성
- **`np.random.randn(6,4)`**: 6행 4열의 정규분포 랜덤 데이터
- **`index=dates`**: 행 인덱스를 날짜로 설정
- **`columns=list('ABCD')`**: 열 이름을 A, B, C, D로 설정

### **방법 2: 딕셔너리로 DataFrame 생성**

```python
# 다양한 데이터 타입으로 DataFrame 생성
pd.DataFrame({'A' : 1.,
              'B' : pd.Timestamp('20130102'),
              'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
              'D' : np.arange(4,dtype='int32'),
              'E' : pd.Categorical(["test","train","test","train"]),
              'F' : 'foo' })
```

### 💡 **코드 설명**
- **`'A' : 1.`**: 부동소수점 값
- **`'B' : pd.Timestamp('20130102')`**: 날짜/시간 타입
- **`'C' : pd.Series(...)`**: Series 객체
- **`'D' : np.arange(4,dtype='int32')`**: 정수 배열
- **`'E' : pd.Categorical(...)`**: 범주형 데이터
- **`'F' : 'foo'`**: 문자열 값

## 📁 **CSV 파일에서 데이터 로드**

### **실제 데이터 사용**
- **데이터**: 프랑스 도시별 온도 데이터
- **출처**: MOOC 강의 데이터셋
- **형식**: CSV 파일 (세미콜론 구분)

### 🔧 **CSV 파일 로드**

```python
# 프랑스 도시 온도 데이터 로드
url = "https://www.fun-mooc.fr/c4x/agrocampusouest/40001S03/asset/AnaDo_JeuDonnees_TemperatFrance.csv"
french_cities = pd.read_csv(url, delimiter=";", encoding="latin1", index_col=0)
french_cities
```

### 💡 **코드 설명**
- **`url`**: 원격 CSV 파일 URL
- **`delimiter=";"`**: 세미콜론으로 구분된 데이터
- **`encoding="latin1"`**: 라틴-1 인코딩 (프랑스어 지원)
- **`index_col=0`**: 첫 번째 컬럼을 인덱스로 사용

## 👀 **데이터 탐색**

### **데이터 미리보기**

```python
# 처음 5개 행 표시
french_cities.head()
```

```python
# 마지막 5개 행 표시
french_cities.tail()
```

### 💡 **데이터 탐색의 중요성**
- **데이터 구조 파악**: 컬럼과 행의 개수 확인
- **데이터 타입 확인**: 각 컬럼의 데이터 타입 파악
- **결측값 확인**: 누락된 데이터 식별
- **이상값 탐지**: 비정상적인 값 확인

## 🏷️ **인덱스 관리**

### **인덱스 확인**

```python
# DataFrame의 인덱스 확인
french_cities.index
```

### **인덱스 이름 설정**

```python
# 인덱스에 이름 설정
french_cities.index.name = "City"
french_cities.head()
```

### 💡 **인덱스 관리의 중요성**
- **데이터 식별**: 각 행을 고유하게 식별
- **빠른 접근**: 인덱스 기반의 효율적인 데이터 접근
- **의미있는 라벨**: 숫자 대신 의미있는 이름 사용
- **데이터 정렬**: 인덱스 기준으로 데이터 정렬 가능

### **컬럼 이름 변경**

```python
import locale
import calendar
 
# 로케일 설정
locale.setlocale(locale.LC_ALL,'C')
 
# 월 약어 가져오기
months = calendar.month_abbr
print(*months)
 
# 월 이름을 영어로 변경
french_cities.rename(
  columns={ old : new 
           for old, new in zip(french_cities.columns[:12], months[1:])
          if old != new },
  inplace=True)
 
# 'Moye'를 'Mean'으로 변경
french_cities.rename(columns={'Moye':'Mean'}, inplace=True)
french_cities
```

### 💡 **컬럼 이름 변경의 중요성**
- **가독성 향상**: 이해하기 쉬운 컬럼명 사용
- **국제화**: 영어 컬럼명으로 표준화
- **일관성**: 전체 데이터셋의 컬럼명 통일
- **분석 편의성**: 의미있는 이름으로 분석 용이

#### **연습문제: DataFrame 월 이름을 영어로 변경**

### 🌐 **HTML 파일에서 데이터 로드**

#### **해수면 관측소 데이터**
- **출처**: [PSMSL 웹사이트](http://www.psmsl.org/)
- **데이터**: 전 세계 해수면 관측소 정보
- **형식**: HTML 테이블
- **용도**: 해수면 변화 모니터링

### 🔧 **HTML 테이블 읽기**

```python
# 필요한 패키지: lxml, beautifulSoup4, html5lib
table_list = pd.read_html("http://www.psmsl.org/data/obtaining/")
```

```python
# 페이지에 있는 테이블 중 첫 번째 테이블 선택
# 해수면 관측소 메타데이터 포함
local_sea_level_stations = table_list[0]
local_sea_level_stations
```

### 💡 **HTML 데이터 로드의 장점**
- **자동 파싱**: HTML 테이블을 자동으로 DataFrame으로 변환
- **다양한 소스**: 웹사이트에서 직접 데이터 수집
- **실시간 데이터**: 최신 데이터에 접근 가능
- **구조화**: HTML 테이블의 구조를 유지

## 🔍 **DataFrame 인덱싱**

### **컬럼 접근**

```python
# 컬럼 접근 (Series 반환)
french_cities['Lati']  # 위도 컬럼 선택
```

### **고급 인덱싱**

#### **`.loc` - 라벨 기반 인덱싱**

```python
# 특정 행과 열의 값 접근
french_cities.loc['Rennes', "Sep"]
```

```python
# 여러 컬럼 선택
french_cities.loc['Rennes', ["Sep", "Dec"]]
```

```python
# 컬럼 범위 선택
french_cities.loc['Rennes', "Sep":"Dec"]
```

### 💡 **인덱싱 방법 비교**
- **`[]`**: 컬럼 접근 (Series 반환)
- **`.loc`**: 라벨 기반 인덱싱 (행, 열 모두 라벨 사용)
- **`.iloc`**: 위치 기반 인덱싱 (숫자 인덱스 사용)

## 🎭 **마스킹 (Masking)**

### **불린 마스크 사용**

```python
# 불린 마스크 생성 (True/False 패턴)
mask = [True, False] * 6 + 5 * [False]
print(french_cities.iloc[:, mask])
```

```python
# 특정 행에 마스크 적용
print(french_cities.loc["Rennes", mask])
```

### 💡 **마스킹의 활용**
- **조건부 선택**: 특정 조건을 만족하는 데이터만 선택
- **컬럼 필터링**: 필요한 컬럼만 선택
- **데이터 정제**: 불필요한 데이터 제거
- **성능 최적화**: 필요한 데이터만 처리

## ➕ **새 컬럼 추가**

### **표준편차 컬럼 생성**

```python
# 월별 온도의 표준편차 계산하여 새 컬럼 추가
french_cities["std"] = french_cities.iloc[:,:12].std(axis=1)
french_cities
```

### **컬럼 제거**

```python
# 새로 추가한 컬럼 제거
french_cities = french_cities.drop("std", axis=1)
french_cities
```

### 💡 **컬럼 조작의 중요성**
- **파생 변수**: 기존 데이터로부터 새로운 정보 생성
- **통계 계산**: 표준편차, 평균 등 통계량 계산
- **데이터 정제**: 불필요한 컬럼 제거
- **메모리 관리**: 사용하지 않는 컬럼 제거로 메모리 절약

## ⚠️ **다중 인덱싱으로 DataFrame 수정**

### **잘못된 방법**

```python
# 이 방법은 작동하지 않고 DataFrame을 손상시킴
# french_cities['Rennes']['Sep'] = 25  # ❌ 잘못된 방법
```

### **올바른 방법**

```python
# 올바른 방법: .loc 사용
french_cities.loc['Rennes']['Sep']  # = 25  # ✅ 올바른 방법
```

```python
# 수정된 DataFrame 확인
french_cities
```

### 💡 **인덱싱 방법의 차이점**
- **`df['row']['col']`**: 체이닝 방식 (권장하지 않음)
- **`df.loc['row', 'col']`**: 명시적 인덱싱 (권장)
- **안전성**: `.loc` 사용 시 데이터 무결성 보장
- **성능**: 명시적 인덱싱이 더 효율적

## 🔄 **데이터셋 변환**

### **통계 계산**

```python
# 평균 온도의 최솟값과 진폭의 최댓값 계산
french_cities['Mean'].min(), french_cities['Ampl'].max()
```

### 💡 **데이터 변환의 중요성**
- **요약 통계**: 데이터의 전체적인 특성 파악
- **이상값 탐지**: 최솟값/최댓값으로 극값 확인
- **데이터 품질**: 통계량을 통한 데이터 품질 검증
- **분석 준비**: 데이터 변환을 통한 분석 준비

## 🔧 **Apply 함수**

### **온도 단위 변환**
- **목표**: 섭씨 온도를 화씨 온도로 변환
- **공식**: F = C × 9/5 + 32

### **Apply 함수 사용**

```python
# 화씨 변환 함수 정의
fahrenheit = lambda T: T*9/5+32

# 평균 온도를 화씨로 변환
french_cities['Mean'].apply(fahrenheit)
```

### 💡 **Apply 함수의 활용**
- **함수 적용**: 각 값에 함수를 적용
- **데이터 변환**: 단위 변환, 스케일링 등
- **조건부 처리**: 복잡한 조건에 따른 데이터 처리
- **벡터화**: 전체 컬럼에 함수를 한 번에 적용

## 📊 **정렬 (Sort)**

### **위도 기준 정렬**

```python
# 위도 기준으로 오름차순 정렬
french_cities.sort_values(by='Lati')
```

### **내림차순 정렬**

```python
# 위도 기준으로 내림차순 정렬
french_cities = french_cities.sort_values(by='Lati', ascending=False)
french_cities
```

### 💡 **정렬의 활용**
- **데이터 탐색**: 정렬을 통한 패턴 발견
- **분석 준비**: 특정 기준으로 데이터 정렬
- **시각화**: 정렬된 데이터로 더 나은 시각화
- **비교 분석**: 정렬을 통한 데이터 비교

## 🔄 **Stack과 Unstack**

### **데이터 구조 변환**
- **목표**: 월별 데이터를 1차원으로 변환
- **변환**: 2차원 → 1차원 (월별 데이터를 세로로 배치)
- **용도**: 시계열 분석, 시각화에 유용

### **Unstack 연산**

```python
# 표시 옵션 변경 (더 많은 행 표시)
pd.set_option("display.max_rows", 20)

# 월별 데이터를 1차원으로 변환
unstacked = french_cities.iloc[:,:12].unstack()
unstacked
```

### **데이터 타입 확인**

```python
# 변환된 데이터의 타입 확인
type(unstacked)
```

### 💡 **Stack/Unstack의 활용**
- **데이터 재구성**: 2차원 ↔ 1차원 변환
- **시계열 분석**: 시간 순서대로 데이터 정렬
- **시각화**: 선 그래프 등에 적합한 형태로 변환
- **통계 분석**: 시계열 통계 분석에 유용

## 🔄 **Transpose (전치)**

### **데이터 전치의 필요성**
- **문제**: Unstack 결과가 잘못된 순서로 그룹화됨
- **해결**: DataFrame을 전치하여 올바른 순서로 정렬
- **효과**: 도시별 월별 온도 데이터를 올바르게 정렬

### **전치 및 시각화**

```python
# DataFrame 전치 (행과 열 바꾸기)
city_temp = french_cities.iloc[:,:12].transpose()

# 도시별 온도 변화 시각화
city_temp.plot()
```

### **박스플롯 시각화**

```python
# 도시별 온도 분포 박스플롯
city_temp.boxplot(rot=90)
```

### 💡 **Transpose의 활용**
- **데이터 재구성**: 행과 열의 역할 바꾸기
- **시각화 최적화**: 그래프에 적합한 형태로 변환
- **분석 편의성**: 특정 관점에서 데이터 분석
- **통계 계산**: 다른 축 기준으로 통계 계산

## 📊 **데이터 설명 (Describing)**

### **범주형 데이터 분석**

```python
# 지역 컬럼의 기본 통계
french_cities['Région'].describe()
```

### **고유값 확인**

```python
# 지역의 고유값들 확인
french_cities['Région'].unique()
```

### **값의 빈도 계산**

```python
# 각 지역별 도시 수 계산
french_cities['Région'].value_counts()
```

### **메모리 최적화**

```python
# 메모리 절약을 위해 범주형 데이터로 변환
french_cities["Région"] = french_cities["Région"].astype("category")
```

### **메모리 사용량 확인**

```python
# 각 컬럼의 메모리 사용량 확인
french_cities.memory_usage()
```

### 💡 **데이터 설명의 중요성**
- **데이터 이해**: 데이터의 구조와 특성 파악
- **품질 검증**: 데이터의 품질과 완전성 확인
- **메모리 최적화**: 효율적인 데이터 타입 사용
- **분석 준비**: 분석에 필요한 정보 수집

## 📊 **데이터 집계 및 요약**

## 🔄 **GroupBy - 그룹화**

### **지역별 그룹화**

```python
# 지역별로 데이터 그룹화
fc_grouped_region = french_cities.groupby("Région")
type(fc_grouped_region)
```

### **그룹별 데이터 확인**

```python
# 각 그룹의 데이터 확인
for group_name, subdf in fc_grouped_region:
    print(group_name)
    print(subdf)
    print("")
```

### 💡 **GroupBy의 활용**
- **그룹별 분석**: 특정 기준으로 데이터 분할
- **집계 연산**: 그룹별 통계 계산
- **비교 분석**: 그룹 간 차이점 분석
- **패턴 발견**: 그룹별 특성 파악

## 🎯 **최종 연습문제: 발전소 데이터 분석**

### **데이터셋 소개**
- **출처**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
- **데이터**: 발전소 측정 기록 (2006-2011, 6년간)
- **크기**: 10,000개 데이터 포인트
- **목표**: 전력 출력을 다른 변수의 함수로 모델링

### **변수 설명**
- **AT**: 대기 온도 (°C)
- **V**: 배기 진공 속도
- **AP**: 대기압
- **RH**: 상대 습도
- **PE**: 전력 출력

### **연습문제 목표**
1. **Excel 파일 읽기**: `pd.read_excel()` 함수 사용
2. **데이터 통합**: 5개 시트의 데이터를 하나로 통합
3. **상관관계 분석**: 변수 간 최대 상관계수 계산
4. **병렬 처리**: `concurrent.futures`로 루프 병렬화

### **해결 방법**
- **데이터 로드**: Excel 파일의 모든 시트 읽기
- **데이터 통합**: `select` 함수로 모든 관측값 통합
- **상관분석**: `corr()` 함수로 상관계수 계산
- **성능 최적화**: 병렬 처리로 계산 속도 향상

## 🎯 **학습 목표 달성**

### **이번 실습에서 배운 내용**
- **DataFrame 생성**: 다양한 방법으로 DataFrame 생성
- **데이터 조작**: 인덱싱, 필터링, 정렬, 집계
- **데이터 변환**: stack, unstack, transpose
- **그룹화**: groupby를 활용한 데이터 집계
- **실제 데이터**: CSV, HTML, Excel에서 데이터 로드

### **다음 단계**
- **고급 분석**: 더 복잡한 데이터 분석 기법
- **머신러닝**: 데이터를 활용한 머신러닝 모델 구축
- **시각화**: 고급 데이터 시각화 기법
- **성능 최적화**: 대용량 데이터 처리 최적화
