# Index

## lecture

* [01. 빅데이터 분석의 범위와 데이터 생태계 (Big Data Ecosystem & 5V)](./01.%20%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%EC%84%9D%EC%9D%98%20%EB%B2%94%EC%9C%84.md) - 빅데이터의 핵심 5V 특성(Volume, Velocity, Variety, Veracity, Value), 정형·반정형·비정형 데이터의 구조적 분류, DIKW(Data-Information-Knowledge-Wisdom) 지식 피라미드 및 분석 라이프사이클을 인터랙티브 5V 차원 분석기로 심층 학습한다.
* [02. 하둡과 분산 아키텍처 - HDFS와 MapReduce (Hadoop Ecosystem)](./02.%20%ED%95%98%EB%91%A1%EA%B3%BC%20%EB%B6%84%EC%82%B0%20%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98.md) - 하둡 분산 파일 시스템(HDFS NameNode/DataNode, 128MB 블록 복제, 랙 인지 정책), MapReduce 분산 연산 4단계(Map-Shuffle/Sort-Reduce), YARN 리소스 매니저를 인터랙티브 MapReduce 워크스루 시뮬레이터로 심층 학습한다.
* [03. Apache Spark 미리보기 - 인메모리 분산 컴퓨팅 아키텍처 (Spark Overview)](./03.%20Apache%20Spark%20%EB%AF%B8%EB%A6%AC%EB%B3%B4%EA%B8%B0.md) - Apache Spark의 마스터-워커 분산 구조(Driver, Cluster Manager, Executor, Tasks), 하둡 MapReduce 대비 100배 빠른 In-Memory DAG 엔진의 원리를 인터랙티브 Spark 클러스터 실행 시뮬레이터로 심층 학습한다.
* [04-1. Apache Spark의 배경 - 분산 컴퓨팅의 진화와 통합 스택 (Evolution & Unified Stack)](./04-1.%20Apache%20Spark%EC%9D%98%20%EB%B0%B0%EA%B2%BD.md) - MapReduce의 한계를 극복한 Apache Spark의 탄생 배경, 통일된 API(Unified Engine) 생태계(Spark SQL, Structured Streaming, MLlib, GraphX)와 Catalyst 최적화 엔진을 인터랙티브 통합 스택 탐색기로 심층 학습한다.
* [04-2. Spark RDD와 워크플로 - 지연 평가와 좁은/넓은 의존성 (RDD & Workflow)](./04-2.%20Spark%20RDD%EC%99%80%20%EC%9B%8C%ED%81%AC%ED%94%8C%EB%A1%9C.md) - RDD(Resilient Distributed Dataset)의 5대 핵심 특성, 불변성(Immutability)과 계통도(Lineage) 장애 복구 원리, 지연 평가(Lazy Evaluation), 좁은 의존성(Narrow)과 넓은 의존성(Wide)에 따른 셔플 발생 메커니즘을 인터랙티브 RDD DAG 시뮬레이터로 심층 학습한다.
* [05. 데이터 통계 기초 - 가설 검정과 중심 극한 정리 (Statistical Inference & Hypothesis Testing)](./05.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%86%B5%EA%B3%84%20%EA%B8%B0%EC%B4%88.md) - 표본 추출과 중심 극한 정리(CLT), 귀무가설(H_0)과 대립가설(H_1), 유의수준(alpha)과 p-value, 1종·2종 오류의 수학적 정의 및 t-검정을 인터랙티브 가설 검정 p-value 계산기로 심층 학습한다.
* [06. 변수 선택 - 필터·래퍼·임베디드 기법과 희소성 (Feature Selection)](./06.%20%EB%B3%80%EC%88%98%20%EC%84%A0%ED%83%9D.md) - 차원의 저주(Curse of Dimensionality) 극복, 필터 기법(ANOVA F-test, Chi2), 래퍼 기법(RFE, 전진/후진 선택), 임베디드 기법(Lasso L1 정규화)의 장단점 및 수학적 원리를 인터랙티브 변수 선택 시뮬레이터로 심층 학습한다.
* [07. 스트리밍 알고리즘](./07.%20%EC%8A%A4%ED%8A%B8%EB%A6%AC%EB%B0%8D%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98.md) - 끝나는 시점을 모르고 재방문하기 어려운 데이터에서 샘플링·Bloom filter·고유값 추정을 수행하는 방법을 정리한다.
* [08. 다목적 최적화](./08.%20%EB%8B%A4%EB%AA%A9%EC%A0%81%20%EC%B5%9C%EC%A0%81%ED%99%94.md) - 상충하는 복수 목적과 제약 조건을 우월성·파레토 프론트·가중합·ε-제약의 관점에서 정리한다.
* [09. MLFlow 설치와 실행](./09.%20MLFlow%20%EC%84%A4%EC%B9%98%EC%99%80%20%EC%8B%A4%ED%96%89.md) - 반복 실행을 추적하고 모델을 비교·등록하는 MLflow 과제 흐름을 환경·run·모델 기록의 관점에서 정리한다.
* [big-data-analysis 강의 흐름 지도](./00.%20big-data-analysis%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 19개의 순서·쪽수·학습 점검을 연결한다.
