---
aliases: []
course: bigdata-analysis
created: '2025-12-03'
date: '2025-12-03'
semester: 3-2
source: ''
status: seedling
tags:
- type/lecture
title: Docker 컨테이너 실행
type: lecture
updated: '2026-05-05'
---

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

![[터미널_실행결과_1705817_엄윤상.png]]

**주요 실행 내용:**
- Spark 3.5.0 버전으로 실행
- Random Forest Classifier 학습 완료
- Test Area Under ROC: **0.838** (Run 1 기준)
- MLFlow에 성공적으로 기록됨

### 4.2 실행 출력 요약

![[MLFlow_실행결과_1705817_엄윤상.png]]
**5개 Runs 실행 결과:**
- ✅ Run 1: maxBins=20, maxDepth=4
- ✅ Run 2: maxBins=24, maxDepth=6
- ✅ Run 3: maxBins=28, maxDepth=7
- ✅ Run 4: maxBins=30, maxDepth=8
- ✅ Run 5: maxBins=40, maxDepth=5

### 4.3 상세 실행 로그

![[MLFlow_실행로그_1705817_엄윤상.png]]

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
