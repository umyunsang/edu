---
aliases: []
course: big-data-analysis
created: '2025-12-03'
date: '2025-12-03'
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
title: Docker 컨테이너 실행
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/빅데이터분석 인터페이스|빅데이터분석 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]]
related:: [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[LGAimer/LG Aimers 9기 평가 및 제출 가이드|LG Aimers 9기 평가 및 제출 가이드]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[LGAimer/LG Aimers 9기 지원서 초안|LG Aimers 9기 지원서 초안]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/빅데이터분석 근거 인덱스|빅데이터분석 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/decision tree|decision tree]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/ml|ml]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/sql|sql]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/big-data-analysis/ai|ai]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

## 1. 실행한 maxBins 및 maxDepth 값

총 **5개의 runs**를 다음과 같은 파라미터 조합으로 실행했습니다:

| Run | maxBins | maxDepth | 설명 |
|-----|---------|----------|------|
| 1   | 20      | 4        | 기본 설정 |
| 2   | 24      | 6        | 중간 복잡도 |
| 3   | 28      | 7        | 높은 복잡도 |
| 4   | 30      | 8        | 최대 깊이 증가 |
| 5   | 40      | 5        | 최대 Bins 증가 |

**변경 사항**:
- PDF 예제에서는 (32,5), (16,3), (16,5), (32,10) 조합을 사용했으나,
- 본 과제에서는 **(20,4), (24,6), (28,7), (30,8), (40,5)** 조합으로 변경하여 실행

---

## 2. 실행 환경 설정

### 2.1 Docker 환경
```bash
# Docker 컨테이너 실행
docker run -d -p 5001:5000 \
  -v /Users/um-yunsang/Documents/mlflow_example:/home/jovyan/work \
  --name mlflow-container \
  jupyter/pyspark-notebook:latest sleep infinity

# MLFlow 설치
docker exec mlflow-container conda install -c conda-forge mlflow -y

# MLFlow 서버 실행
docker exec -d mlflow-container bash -c \
  "mlflow server --backend-store-uri sqlite:///mlflow.db \
   --default-artifact-root /home/jovyan/work \
   --host 0.0.0.0 --port 5000"
```

### 2.2 환경 정보
- **Docker Image**: jupyter/pyspark-notebook:latest
- **MLFlow Version**: 3.6.0
- **Spark Version**: 3.5.0
- **Python Version**: 3.11
- **MLFlow UI Port**: 5001 (로컬) → 5000 (컨테이너)

---

## 3. 실행 명령어

### 3.1 환경변수 설정
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3.2 5개의 Runs 실행
```bash
# Run 1: maxBins=20, maxDepth=4
docker exec mlflow-container bash -c \
  'export MLFLOW_TRACKING_URI=http://localhost:5000 && \
   spark-submit --master local[*] /home/jovyan/work/code/mlflow_example.py 20 4 run1'

# Run 2: maxBins=24, maxDepth=6
docker exec mlflow-container bash -c \
  'export MLFLOW_TRACKING_URI=http://localhost:5000 && \
   spark-submit --master local[*] /home/jovyan/work/code/mlflow_example.py 24 6 run2'

# Run 3: maxBins=28, maxDepth=7
docker exec mlflow-container bash -c \
  'export MLFLOW_TRACKING_URI=http://localhost:5000 && \
   spark-submit --master local[*] /home/jovyan/work/code/mlflow_example.py 28 7 run3'

# Run 4: maxBins=30, maxDepth=8
docker exec mlflow-container bash -c \
  'export MLFLOW_TRACKING_URI=http://localhost:5000 && \
   spark-submit --master local[*] /home/jovyan/work/code/mlflow_example.py 30 8 run4'

# Run 5: maxBins=40, maxDepth=5
docker exec mlflow-container bash -c \
  'export MLFLOW_TRACKING_URI=http://localhost:5000 && \
   spark-submit --master local[*] /home/jovyan/work/code/mlflow_example.py 40 5 run5'
```

---

## 4. 실행 결과

### 4.1 터미널 실행 화면

실제 터미널에서 명령어를 실행한 결과입니다:

![[3-2_bigdata-analysis__터미널_실행결과_1705817_엄윤상.png]]

**주요 실행 내용:**
- Spark 3.5.0 버전으로 실행
- Random Forest Classifier 학습 완료
- Test Area Under ROC: **0.838** (Run 1 기준)
- MLFlow에 성공적으로 기록됨

### 4.2 실행 출력 요약

![[3-2_bigdata-analysis__MLFlow_실행결과_1705817_엄윤상.png]]
**5개 Runs 실행 결과:**
- ✅ Run 1: maxBins=20, maxDepth=4
- ✅ Run 2: maxBins=24, maxDepth=6
- ✅ Run 3: maxBins=28, maxDepth=7
- ✅ Run 4: maxBins=30, maxDepth=8
- ✅ Run 5: maxBins=40, maxDepth=5

### 4.3 상세 실행 로그

![[3-2_bigdata-analysis__MLFlow_실행로그_1705817_엄윤상.png]]

**로그에서 확인 가능한 정보:**
- Spark Job 실행 상태
- MLFlow Run 등록 확인
- 모델 학습 완료 메시지
- Experiment URL: http://localhost:5000/#/experiments/0

---

## 5. MLFlow UI 접속 정보

### 5.1 접속 방법
- **URL**: http://localhost:5001
- **Experiments ID**: 0 (Default)

### 5.2 기록된 정보
각 run에는 다음 정보가 자동으로 기록되었습니다:

![[Pasted image 20251202160528.png]]

**Parameters:**
- `maxBins`: 사용한 최대 Bin 값
- `maxDepth`: 사용한 최대 깊이 값

**Metrics:**
- `ROC`: Area Under ROC 값 (모델 성능 지표)

**Artifacts:**
- `spark-model`: 학습된 Random Forest 모델
- `MLmodel`: 모델 메타데이터
- `conda.yaml`: 환경 설정
- `requirements.txt`: Python 패키지 의존성

---

## 6. 코드 수정 사항

### 6.1 오류 수정
원본 `mlflow_example.py` 파일의 79번 줄에 오타가 있어 수정했습니다:

**수정 전:**
```python
evaluator = BinaryClassificationEvSaluator()  # 오타: EvSaluator
```

**수정 후:**
```python
evaluator = BinaryClassificationEvaluator()  # 정상: Evaluator
```

---

## 7. 실행 결과 분석

### 7.1 모델 성능
- **ROC Score**: 약 0.838 (Run 1 기준)
- **데이터셋**: bank-full.csv (은행 마케팅 데이터)
- **알고리즘**: Random Forest Classifier
- **Train/Test Split**: 70% / 30%

### 7.2 파라미터 영향
- maxBins 증가: 더 세밀한 feature 분할 가능
- maxDepth 증가: 더 복잡한 패턴 학습 가능
- 하지만 과적합 위험도 증가

---
## 8. 결론

본 과제를 통해 다음을 학습하고 실습했습니다:

1. **MLFlow 설치 및 설정**: Docker 환경에서 MLFlow 서버 구축
2. **MLFlow Tracking**: 실험 파라미터와 메트릭 자동 기록
3. **Spark ML**: Random Forest 모델 학습 및 평가
4. **실험 관리**: 여러 파라미터 조합을 체계적으로 비교

모든 실행이 정상적으로 완료되었으며, MLFlow UI를 통해 5개의 runs를 확인할 수 있습니다.

---
