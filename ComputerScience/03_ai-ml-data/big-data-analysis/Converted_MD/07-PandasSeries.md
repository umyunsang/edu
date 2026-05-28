---
aliases: []
course: big-data-analysis
created: '2025-09-24'
date: '2025-09-24'
semester: 3-2
source: ''
status: seedling
tags:
- cs/db
- cs/ml
- type/lecture
title: 07-PandasSeries
type: lecture
updated: '2026-05-05'
---



domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
up:: [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/시험정리|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/boxplot 주식 그래프|boxplot 주식 그래프]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/scatter 산점도 그래프|scatter 산점도 그래프]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/신호 특징 분석 결과|신호 특징 분석 결과]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot 생성|subplot 생성]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/그래프 스타일|그래프 스타일]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/subplot  예제|subplot  예제]], [[LGAimer/LG Aimers 9기 지원서 초안|LG Aimers 9기 지원서 초안]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[LGAimer/LG Aimers 9기 평가 및 제출 가이드|LG Aimers 9기 평가 및 제출 가이드]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/bar 막대 그래프|bar 막대 그래프]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/SLP (Single Layer Perceptron)|SLP (Single Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/기본 그래프|기본 그래프]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/pie 원형 그래프|pie 원형 그래프]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/computer-vision/중간고사_컴퓨터비전_정밀분석_정리|중간고사_컴퓨터비전_정밀분석_정리]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]]

# 07-PandasSeries

# 📊 **Pandas Series - 1차원 데이터 구조**

## 🎯 **개요**

### **학습 목표**
- **Pandas Series 이해**: 1차원 데이터 구조의 핵심 개념
- **데이터 조작**: Series 생성, 인덱싱, 필터링
- **시계열 데이터**: 시간 기반 데이터 처리
- **데이터 시각화**: Series 데이터의 시각화
- **데이터 저장**: HDF5 형식으로 데이터 저장 및 로드

### **Pandas란?**
- **개발자**: Wes MacKinney (2011년 첫 릴리스)
- **기반**: NumPy를 기반으로 한 데이터 분석 라이브러리
- **영감**: R 언어의 데이터 조작 도구에서 영감
- **라이선스**: 오픈 소스 (BSD)
- **용도**: 데이터 분석, 정리, 시각화의 핵심 도구

### **Pandas의 핵심 특징**
- **고성능**: NumPy 기반의 빠른 데이터 처리
- **사용 편의성**: 직관적인 데이터 구조
- **자체 설명**: 데이터 구조가 스스로를 설명
- **파일 지원**: 다양한 파일 형식 지원
- **시각화**: 내장된 플롯팅 함수
- **통계 도구**: 기본적인 통계 분석 기능

### **Pandas 설치 및 임포트**

```python
import pandas as pd
```

### **Pandas의 장점**
- **데이터 구조**: 자체 설명적 데이터 구조
- **파일 처리**: 다양한 파일 형식 지원
- **시각화**: 플롯팅 함수 내장
- **통계 분석**: 기본적인 통계 도구 제공

### 🔧 **환경 설정**

```python
# Jupyter Notebook 설정
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# 필수 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화 설정
sns.set()
pd.set_option("display.max_rows", 8)
plt.rcParams['figure.figsize'] = (9, 6)
```

### 💡 **설정 설명**
- **`%matplotlib inline`**: Jupyter Notebook에서 그래프를 인라인으로 표시
- **`%config InlineBackend.figure_format = 'retina'`**: 고해상도 그래프 출력
- **`sns.set()`**: Seaborn 스타일 적용
- **`pd.set_option("display.max_rows", 8)`**: DataFrame 표시 행 수 제한
- **`plt.rcParams['figure.figsize'] = (9, 6)`**: 그래프 크기 설정

## 📊 **Series - 1차원 데이터 구조**

### **Series란?**
- **정의**: 1차원 배열 데이터와 인덱스 라벨의 조합
- **구조**: 데이터 + 인덱스 (라벨)
- **용도**: 단일 컬럼 데이터를 효율적으로 처리
- **장점**: 인덱스 기반의 빠른 데이터 접근

### **Series의 핵심 특징**
- **1차원 배열**: 단일 차원의 데이터 구조
- **인덱스**: 각 데이터에 대한 라벨 (숫자, 문자열, 날짜/시간)
- **시계열**: 인덱스가 시간 값일 때 시계열 데이터
- **길이 일치**: 인덱스와 데이터의 길이가 동일해야 함
- **자동 생성**: 인덱스가 없으면 `range(len(data))`로 자동 생성

### **Series의 장점**
- **빠른 접근**: 인덱스 기반의 O(1) 데이터 접근
- **유연한 인덱싱**: 다양한 타입의 인덱스 지원
- **시계열 처리**: 시간 기반 데이터 처리에 최적화
- **메모리 효율**: 1차원 구조로 메모리 사용량 최적화

### 🔧 **Series 생성 예제**

```python
# 기본 Series 생성
pd.Series([1,3,5,np.nan,6,8], dtype=np.float64)
```

### 💡 **코드 설명**
- **`pd.Series()`**: Pandas Series 생성 함수
- **데이터**: `[1,3,5,np.nan,6,8]` - 숫자와 NaN 값 포함
- **`dtype=np.float64`**: 데이터 타입을 float64로 지정
- **인덱스**: 자동으로 0, 1, 2, 3, 4, 5 생성

### 🔧 **시계열 Series 생성**

```python
# 날짜 범위를 인덱스로 하는 Series 생성
pd.Series(index=pd.period_range('09/11/2017', '09/18/2017', freq="D"), dtype=np.int8)
```

### 💡 **코드 설명**
- **`pd.period_range()`**: 날짜 범위 생성 (2017년 9월 11일 ~ 18일)
- **`freq="D"`**: 일별 주기로 데이터 생성
- **`dtype=np.int8`**: 데이터 타입을 int8로 지정
- **인덱스**: 날짜가 인덱스로 사용됨

### 🎯 **실습 연습문제**

#### **연습문제 1: 텍스트 생성 및 단어 빈도 계산**
- **목표**: `lorem` 라이브러리를 사용하여 텍스트 생성
- **단계**: 
  1. `lorem` 라이브러리로 텍스트 생성
  2. `collections.Counter`로 단어 빈도 계산
  3. 결과를 딕셔너리로 저장

#### **연습문제 2: Series 생성**
- **목표**: 단어 빈도 결과를 Pandas Series로 변환
- **요구사항**: 
  - Series 이름: `latin_series`
  - 인덱스: 단어를 알파벳 순으로 정렬
  - 데이터: 각 단어의 빈도

### 🔧 **Series 생성 코드**

```python
# 딕셔너리를 Series로 변환
df = pd.Series(result)
df
```

### 💡 **코드 설명**
- **`pd.Series(result)`**: 딕셔너리를 Pandas Series로 변환
- **인덱스**: 딕셔너리의 키가 Series의 인덱스가 됨
- **데이터**: 딕셔너리의 값이 Series의 데이터가 됨
- **결과**: 단어 빈도가 Series 형태로 표시됨

#### **연습문제 3: Series 시각화**
- **목표**: Series 데이터를 막대 그래프로 시각화
- **방법**: `plot()` 함수의 `kind='bar'` 옵션 사용
- **효과**: 단어 빈도를 막대 그래프로 확인

#### **연습문제 4: 인덱싱 연습**
- **목표**: Pandas의 인덱싱 함수 `loc`과 `iloc` 사용
- **`loc`**: 라벨 기반 인덱싱
  - 'dolore' 단어의 빈도 표시
- **`iloc`**: 위치 기반 인덱싱
  - 인덱스의 마지막 단어 빈도 표시

#### **연습문제 5: 정렬 및 시각화**
- **목표**: 단어를 빈도 순으로 정렬하고 시각화
- **단계**:
  1. 단어를 빈도 순으로 정렬
  2. 정렬된 Series를 막대 그래프로 시각화
- **효과**: 빈도가 높은 단어부터 확인 가능

## 🌍 **전 지구 온도 데이터 분석 (1901-2000)**

### **데이터 개요**
- **기간**: 1901년 ~ 2000년 (100년간)
- **데이터**: 전 지구 평균 온도
- **형식**: 텍스트 파일
- **목표**: 데이터 정리 및 시계열 변환

### **데이터 처리 단계**
1. **텍스트 파일 읽기**: pandas로 데이터 로드
2. **데이터 정리**: 결측값 처리 및 데이터 타입 변환
3. **시계열 변환**: DataFrame을 시계열 Series로 변환
4. **시각화**: 온도 변화 추이 확인

### 🔧 **데이터 로드**

```python
import os
here = os.getcwd()

filename = os.path.join(here,"data","monthly.land.90S.90N.df_1901-2000mean.dat.txt")

df = pd.read_table(filename, sep="\s+", 
                   names=["year", "month", "mean temp"])
df
```

### 💡 **코드 설명**
- **`os.getcwd()`**: 현재 작업 디렉토리 확인
- **`os.path.join()`**: 파일 경로 생성
- **`pd.read_table()`**: 텍스트 파일을 DataFrame으로 읽기
- **`sep="\s+"`**: 공백으로 구분된 데이터 파싱
- **`names`**: 컬럼명 지정 (year, month, mean temp)

#### **연습문제 6: 데이터 전처리**
- **목표**: DataFrame을 시계열 Series로 변환
- **단계**:
  1. **컬럼 추가**: "day" 컬럼을 값 1로 추가 (`.insert` 사용)
  2. **인덱스 변환**: DataFrame 인덱스를 datetime으로 변환 (`pd.to_datetime`)
  3. **Series 변환**: "mean temp" 컬럼만 포함하는 Series 생성

#### **연습문제 7: 데이터 탐색**
- **목표**: 데이터의 시작과 끝 부분 확인
- **`.head()`**: 데이터의 처음 5개 행 표시
- **`.tail()`**: 데이터의 마지막 5개 행 표시
- **용도**: 데이터 구조와 내용 파악

### ⚠️ **결측값 처리**

#### **결측값 표시**
- **데이터셋**: -999.00은 해당 연도에 값이 없음을 의미
- **처리 방법**: -999.00을 `np.nan`으로 변환

#### **연습문제 8: 결측값 처리**
- **목표**: 결측값을 식별하고 처리
- **단계**:
  1. **결측값 확인**: -999 값이 있는지 `.values`로 확인
  2. **결측값 변환**: -999.000을 `np.nan`으로 대체

#### **결측값 제거**
- **변환 후**: `np.nan`으로 변환된 결측값은 제거 가능
- **제거 방법**: `.dropna()` 함수 사용

#### **연습문제 9: 결측값 제거**
- **목표**: 결측값이 포함된 행 제거
- **방법**: `.dropna()` 함수 사용
- **효과**: 완전한 데이터만 남김

#### **연습문제 10: 데이터 시각화**
- **목표**: 기본적인 시각화 생성
- **방법**: `.plot()` 함수 사용
- **효과**: 온도 변화 추이를 그래프로 확인

#### **연습문제 11: 시계열 변환**
- **목표**: 인덱스를 timestamp에서 period로 변환
- **이유**: 월별 평균 데이터이므로 period가 더 의미있음
- **방법**: `to_period` 메서드 사용

## 📊 **Resampling - 시계열 데이터 리샘플링**

### **Resampling이란?**
- **정의**: 시계열 데이터의 주기를 변경하는 작업
- **용도**: 데이터의 시간 해상도를 조정하여 분석 용이성 향상
- **종류**: 
  - **Downsampling**: 고주파 데이터를 저주파로 변환 (예: 일별 → 월별)
  - **Upsampling**: 저주파 데이터를 고주파로 변환 (예: 월별 → 일별)

### **주파수 지정**
- **문자열 형식**: "us", "ms", "S", "T", "H", "D", "B", "W", "M", "A"
- **복합 형식**: "3min", "2h20" 등
- **참고**: [Pandas 주파수 별칭](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

### **Resampling의 장점**
- **데이터 압축**: 대용량 데이터를 효율적으로 처리
- **트렌드 분석**: 장기적 패턴 파악
- **노이즈 제거**: 단기 변동을 평활화

#### **연습문제 12: 10년 단위 리샘플링**
- **목표**: Series를 10년 단위로 리샘플링
- **방법**: `resample` 메서드 사용
- **효과**: 100년 데이터를 10개 구간으로 압축하여 장기 트렌드 파악

## 💾 **데이터 저장 및 로드**

### **HDF5 파일 형식**
- **정의**: 계층적 데이터 형식 (Hierarchical Data Format version 5)
- **특징**: 바이너리 데이터 저장에 최적화된 강력한 파일 형식
- **용도**: Series와 DataFrame 모두 저장 가능
- **장점**: 
  - **고성능**: 빠른 읽기/쓰기 속도
  - **압축**: 효율적인 데이터 압축
  - **계층구조**: 복잡한 데이터 구조 지원

### 🔧 **데이터 저장**

```python
# HDF5 파일에 Series 저장
with pd.HDFStore("data/pandas_series.h5") as writer:
    df.to_hdf(writer, "/temperatures/full_globe")
```

### 💡 **코드 설명**
- **`pd.HDFStore()`**: HDF5 파일 저장소 생성
- **`with` 문**: 파일 자동 닫기 보장
- **`to_hdf()`**: DataFrame/Series를 HDF5 형식으로 저장
- **경로**: "/temperatures/full_globe" - 계층적 경로 지정

### 🔧 **데이터 로드**

```python
# HDF5 파일에서 Series 읽기
with pd.HDFStore("data/pandas_series.h5") as store:
    df = store["/temperatures/full_globe"]
```

### 💡 **코드 설명**
- **`pd.HDFStore()`**: HDF5 파일 저장소 열기
- **`store[경로]`**: 지정된 경로의 데이터 읽기
- **자동 복원**: 원본 Series/DataFrame 구조 완전 복원
- **효율성**: 바이너리 형식으로 빠른 로드

## 🎯 **학습 목표 달성**

### **이번 실습에서 배운 내용**
- **Series 기본**: 1차원 데이터 구조와 인덱싱
- **데이터 처리**: 결측값 처리, 정렬, 시각화
- **시계열 분석**: 날짜 기반 데이터 처리
- **리샘플링**: 시간 해상도 변경
- **데이터 저장**: HDF5 형식으로 효율적 저장

### **다음 단계**
- **DataFrame**: 2차원 데이터 구조 학습
- **고급 시계열**: 더 복잡한 시계열 분석 기법
- **성능 최적화**: 대용량 데이터 처리 최적화
