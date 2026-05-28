---
aliases: []
course: database-systems
created: '2024-09-09'
date: '2024-09-09'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-2
source: ''
status: seedling
tags:
- cs/db
- type/lecture
title: DB 시스템
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/데이터베이스 인터페이스|데이터베이스 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/뷰(view)|뷰(view)]], [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]], [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스 확인문제|확인문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/03_ai-ml-data/ml-projects/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/1. 기억 장치 시스템의 개요|1. 기억 장치 시스템의 개요]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[certifications/체크리스트|체크리스트]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/05_software-engineering/programming-languages/교재/6장_교재_문제|6장_교재_문제]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/05_software-engineering/programming-languages/과제/9장 레포트|9장 레포트]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/05_software-engineering/programming-languages/필기/0. 명령어 집합|0. 명령어 집합]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/05_software-engineering/programming-languages/7장-12장 연습문제 종합|7장-12장 연습문제 종합]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[certifications/information-processing/실기/C언어 실기 오답노트|오답노트]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/데이터베이스 지식그래프|데이터베이스]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/데이터베이스 지식그래프|데이터베이스]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/데이터베이스 근거 인덱스|데이터베이스 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/thank you|thank you]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/sql|sql]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/관계 데이터 모델|관계 데이터 모델]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/데이터 모델링|데이터 모델링]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/데이터베이스 설계|데이터베이스 설계]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

---
![[database-systems__DB 시스템__OEWWIOA AlABy.png]]

## 1. 데이터베이스 시스템의 정의

#### 데이터베이스 시스템(DBS; DataBase System)
- 데이터베이스에 데이터를 저장하고, 이를 관리하여 조직에 필요한 정보를 생성해주는 시스템

## 2. 데이터베이스의 구조

#### ==스키마와 인스턴스==
- **스키마(schema)**
	- 데이터베이스에 저장되는 데이터 구조와 제약조건을 정의한 것 
- **인스턴스(instance)** 
	- 스키마에 따라 데이터베이스에 실제로 저장된 값
	- CPU에 점유되어 처리되고있는 객체를 인스턴스라 한다
	![[database-systems__DB 시스템__스키마와 인스턴스.png]]

#### ==3단계 데이터베이스 구조(시험: 용어 위주)==
- 데이터베이스를 쉽게 이해하고 이용할 수 있도록 하나의 데이터베이스를 관점에 따라 세 단계로 나눈 것 
- **외부 단계(external level)** : **개별 사용자 관점** 
	- 데이터베이스를 개별 사용자 관점에서 이해하고 표현하는 단계 
	- 하나의 데이터베이스에 외부 스키마가 여러 개 존재할 수 있음 
		- **외부 스키마(external schema)**
			- 외부 단계에서 사용자에게 필요한 데이터베이스를 정의한 것 
			- 각 사용자가 생각하는 데이터베이스의 모습, 즉 논리적 구조로 사용자마다 다름 
			- **서브 스키마(sub schema)라고도 함**
- **개념 단계(conceptual level)** : **조직 전체의 관점(통합)**
	- 데이터베이스를 조직 전체의 관점에서 이해하고 표현하는 단계 
	- 하나의 데이터베이스에 개념 스키마가 하나만 존재함 
		- **개념 스키마(conceptual schema)**
			- 개념 단계에서 전체 데이터베이스의 논리적 구조를 정의한 것 
			- **조직 전체의 관점에서 생각하는 데이터베이스의 모습** 
			- 전체 데이터베이스에 어떤 데이터가 저장되는지, 데이터들 간에는 어떤 관계가 존재 하고 어떤 제약조건이 있는지에 대한 정의뿐만 아니라, 데이터에 대한 보안 정책이나 접근 권한에 대한 정의도 포함
- **내부 단계(internal level)** : **저장 장치의 관점**
	- 데이터베이스를 저장 장치의 관점에서 이해하고 표현하는 단계 
	- 하나의 데이터베이스에 내부 스키마가 하나만 존재함 
		- **내부 스키마(internal schema)** 
			- 전체 데이터베이스가 저장 장치에 실제로 저장되는 방법을 정의한 것 
			- 레코드 구조, 필드 크기, 레코드 접근 경로 등 물리적 저장 구조를 정의
- 3단계 데이터베이스 구조의 예
	![[database-systems__DB 시스템__3단계 데이터베이스 구조(시험 용어 위주.png]]

#### ==데이터 독립성(data independency)(시험: 용어 위주)==
데이터베이스를 3단계 구조로 나누고 단계별로 스키마를 유지하며 스키마 사이의 대응 관계를 정의하는 궁극적인 목적 -> **데이터 독립성의 실현**
- 하위 스키마를 변경하더라도 상위 스키마가 영향을 받지 않는 특성
- 미리 정의된 사상 정보를 이용해 사용자가 원하는 데이터에 접근
- **논리적 데이터 독립성 (외부/개념 사상)** : **응용 인터페이스(application interface)**
	- 개념 스키마가 변경되어도 외부 스키마는 영향을 받지 않음 
	- 개념 스키마가 변경되면 관련된 외부/개념 사상만 정확하게 수정해주면 됨 
- **물리적 데이터 독립성 (개념/내부 사상)** : **저장 인터페이스(storage interface)**
	- 내부 스키마가 변경되어도 개념 스키마는 영향을 받지 않음 
	- 내부 스키마가 변경되면 관련된 개념/내부 사상만 정확하게 수정해주면 됨

---
## 용어만 알아두기
#### 데이터 사전(data dictionary)
- 시스템 카탈로그(system catalog)라고도 함 
- 데이터베이스에 저장되는 데이터에 관한 정보, 즉 메타 데이터를 유지 하는 시스템 데이터베이스 
	- 메타 데이터(meta data) : 데이터에 대한 데이터 
- 스키마, 사상 정보, 다양한 제약조건 등을 저장 
- 데이터베이스 관리 시스템이 스스로 생성하고 유지함 
- 일반 사용자도 접근이 가능하지만 저장 내용을 검색만 할 수 있음

#### 데이터 디렉터리(data directory)
- 데이터 사전에 있는 데이터에 실제로 접근하는 데 필요한 위치 정보를 저장하는 시스템 데이터베이스 
- 일반 사용자의 접근은 허용되지 않음

#### 사용자 데이터베이스(user database)
- 사용자가 실제로 이용하는 데이터가 저장되어 있는 일반 데이터베이스

---
## ==4. 데이터 언어==
#### 데이터 언어
- 사용자와 데이터베이스 관리 시스템 간의 통신 수단 
- 사용 목적에 따라 데이터 정의어, 데이터 조작어, 데이터 제어어로 구분
	![[database-systems__DB 시스템__데이터 언어.png]]
- **데이터 정의어(DDL; Data Definition Language)** 
	- 스키마를 정의하거나, 수정 또는 삭제하기 위해 사용 
- **데이터 조작어(DML; Data Manipulation Language)** 
	- 데이터의 삽입·삭제·수정·검색 등의 처리를 요구하기 위해 사용 
	- 절차적 데이터 조작어와 비절차적 데이터 조작어로 구분 
		- **절차적 데이터 조작어**(procedural DML) 
			- 사용자가 어떤(what) 데이터를 원하고 그 데이터를 얻으려면 어떻게(how) 처리해야 하는지도 설명(정확하게 지시)
		- **비절차적 데이터 조작어**(nonprocedural DML) 
			- 사용자가 어떤(what) 데이터를 원하는지만 설명 
			- 선언적 언어(declarative language)라고도 함
- **데이터 제어어(DCL; Data Control Language)** 
	- 내부적으로 필요한 규칙이나 기법을 정의하기 위해 사용 
	- 사용 목적 
		- 무결성 : 정확하고 유효한 데이터만 유지 
		- 보안 : 허가받지 않은 사용자의 데이터 접근 차단, 허가된 사용자에게 권한 부여
		- 회복 : 장애가 발생해도 데이터 일관성 유지 
		- 동시성 제어 : 데이터 동시 공유 지원

## 데이터베이스 관리 시스템의 구성

![[database-systems__DB 시스템__데이터베이스 관리 시스템의 구성.png]]
- DML 프리 컴파일러(용어 알아두기)
- 트랜잭션 : 명령어의 답안들로 묶어진 하나의 명령어

---
#### 1. 데이터베이스 시스템
- 데이터베이스에 데이터를 저장하고, 이를 관리하여 조직에 필요한 정보를 생성해주는 시스템이다.
- 사용자, 데이터 언어, 데이터베이스 관리 시스템, 데이터베이스, 컴퓨터로 구성된다.

#### 2. 스키마와 인스턴스
- 스키마 : 데이터베이스에 저장되는 데이터 구조와 제약조건을 정의한 것 이다.
- 인스턴스 : 스키마에 따라 데이터베이스에 실제로 저장된 값이다.

#### 3. 3단계 데이터베이스 구조
데이터베이스를 쉽게 이해하고 이용할 수 있도록 하나의 데이터베이스를 관점에 따라 세 단계(외부 단계, 개념 단계, 내부 단계)로 나눈 것이다.
- 외부 단계 : 데이터베이스를 개별 사용자 관점에서 이해하고 표현한다. 사용자에게 필요한 데이터베이스를 정의한 외부 스키마가 여러 개 존재할 수 있다.
- 개념 단계 : 데이터베이스를 조직 전체의 관점에서 이해하고 표현한다. 데이터베이스 전체의 논리적 구조를 정의하는 개념 스키마가 하나만 존재한다.
- 내부 단계 : 데이터베이스를 저장 장치의 관점에서 이해하고 표현한다. 데이터베이스가 저장 장치에 저장되는 방법을 정의한 내부 스키마가 하나만 존재한다.

#### 4. 데이터 독립성
3단계 데이터베이스 구조의 목적은 데이터 독립성을 실현하는 데 있다. 데이터 독립성에는 논리적 데이터 독립성과 물리적 데이터 독립성이 존재한다.
- 논리적 데이터 독립성 (외부/개념 사상) : 개념 스키마가 변경되어도 외부 스키마는 영향을 받지 않는다
- 물리적 데이터 독립성 (개념/내부 사상) : 내부 스키마가 변경되어도 개념 스키마는 영향을 받지 않는다.

#### 5. 데이터 사전(시스템 카탈로그)
데이터베이스에 저장되는 데이터에 관한 정보, 즉 메타 데이터를 유지하는 시스템 데이터베이스다.

#### 6. 데이터베이스 사용자
데이터베이스를 이용하기 위해 접근하는 모든 사람을 의미한다. 데이터베이스 관리자, 최종 사용자, 응용 프로그래머로 나뉜다.
- 데이터베이스 관리자 : 데이터베이스 시스템을 운영, 관리한다.
- 최종 사용자 : 데이터베이스에 접근하여 데이터를 조작한다.
- 응용 프로그래머 : 데이터 언어를 삽입하여 응용 프로그램을 작성한다.

#### 7. 데이터 언어
사용자와 데이터베이스 관리 시스템 간의 통신 수단이다. 데이터 정의어, 데이터 조작어, 데이터 제어어로 나뉜다.
- 데이터 정의어(DDL) : 스키마를 정의하거나, 수정 또는 삭제하기 위해서 사용한다.
- 데이터 조작어(DML) : 데이터의 삽입, 삭제, 수정, 검색 등의 처리를 요구하기 위해서 사용한다.
- 데이터 제어어(DCL) : 동시 공유가 가능하면서도 무결성과 일관성을 유지하도록 내부적으로 필요한 규칙이나 기법들을 정의하기 위해서 사용한다.

#### 8. 데이터베이스 관리 시스템
주요 기능은 데이터베이스 관리와 데이터 처리 요구에 대한 수행이다. 질의 처리기와 저장 데이터 관리자로 나뉜다.
- 질의 처리기 : 사용자의 데이터 처리 요구를 해석하여 처리한다.
- 저장 데이터 관리자 : 디스크에 저장된 데이터베이스와 데이터 사전을 관리하고, 여기에 실제로 접근한다.
