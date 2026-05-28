---
aliases: []
course: database-systems
created: '2024-10-28'
date: '2024-10-28'
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
title: 뷰(view)
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/데이터베이스 인터페이스|데이터베이스 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/05_software-engineering/database-systems/6. 관계 데이터 연산/관계 데이터 연산|관계 데이터 연산]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/05_software-engineering/database-systems/7. 데이터베이스 언어 SQL/데이터 베이스 언어 SQL|데이터 베이스 언어 SQL]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/7장 문제|7장 문제]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상(답)|중간 주관식 예상(답)]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/중간 주관식 예상|중간 주관식 예상]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/데이터베이스 연습문제|데이터베이스 연습문제]], [[ComputerScience/05_software-engineering/database-systems/11. 보안과 권한 관리/보안과 권한 관리|보안과 권한 관리]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/05_software-engineering/database-systems/5. 관계 데이터 모델/관계 데이터 모델 (용어 암기)|관계 데이터 모델 (용어 암기)]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/정규화|정규화]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/05_software-engineering/database-systems/12. 데이터베이스 응용 기술/데이터베이스 응용 기술|데이터베이스 응용 기술]], [[ComputerScience/05_software-engineering/database-systems/4. 데이터 모델링/데이터 모델링|데이터 모델링]], [[ComputerScience/05_software-engineering/database-systems/13. 데이터 과학과 빅데이터/데이터 과학과 빅데이터|데이터 과학과 빅데이터]], [[ComputerScience/05_software-engineering/database-systems/10. 회복과 병행제어/회복과 병행 제어|회복과 병행 제어]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/레포트|레포트]], [[ComputerScience/05_software-engineering/database-systems/8. 데이터베이스 설계/데이터베이스 설계|데이터베이스 설계]], [[ComputerScience/05_software-engineering/database-systems/0. 시험/기말시험 범위 및 연습문제|기말시험 범위 및 연습문제]], [[ComputerScience/05_software-engineering/database-systems/9. 정규화/고급 정규형|고급 정규형]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/01-WordCount|01-WordCount]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_CSE408_Pandas,_Geopandas|DAU_CSE408_Pandas,_Geopandas]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/05_software-engineering/programming-languages/7장-12장 연습문제 종합|7장-12장 연습문제 종합]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 확인문제|확인문제]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/05_software-engineering/programming-languages/과제/9장 레포트|9장 레포트]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 실습|HTML 기초 실습]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/3주차 발표자료|3주차 발표자료]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/빅데이터 분석 시험 대비 총정리 (실전 예시 중심|시험정리]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/07-PandasSeries|07-PandasSeries]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션 확인문제|확인문제]], [[certifications/체크리스트|체크리스트]], [[ComputerScience/03_ai-ml-data/ai-system-design/3주차/AI 챗봇 특허 저작권 보호 전략 발표 스크립트|AI 챗봇 특허 저작권 보호 전략 발표 스크립트]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/개념문제_풀이|개념문제_풀이]], [[ComputerScience/05_software-engineering/programming-languages/교재/5장_교재_문제|5장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습|Spring Boot 기초 실습]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/programming-languages/교재/6장_교재_문제|6장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/03_ai-ml-data/ai-system-design/ot/AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트|AI 시스템 아이디어 기획 요구사항 정의 발표 스크립트]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/K-POP 아티스트 인기도 분석 시스템|K-POP 아티스트 인기도 분석 시스템]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[certifications/information-processing/실기/C언어 실기 오답노트|오답노트]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초2 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 연습문제|연습문제]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/이론정리|이론정리]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램|아키텍처 다이어그램]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/아키텍처 다이어그램 텍스트 버전|아키텍처 다이어그램 텍스트 버전]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/데이터베이스 지식그래프|데이터베이스]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/데이터베이스 지식그래프|데이터베이스]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/데이터베이스 근거 인덱스|데이터베이스 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/thank you|thank you]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/sql|sql]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/관계 데이터 모델|관계 데이터 모델]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/데이터 모델링|데이터 모델링]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/database-systems/데이터베이스 설계|데이터베이스 설계]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

---
## 4. 뷰
#### 뷰(view)
- 다른 테이블을 기반으로 만들어진 **가상 테이블** 
	- 데이터를 실제로 저장하지 않고 **논리적으로만 존재하는 테이블** 
- 일반 테이블과 동일한 방법으로 사용 
- 뷰를 통해 기본 테이블의 내용을 **쉽게 검색**할 수는 있지만, 기본 테이블의 **내용을 바꾸는 작업은 제한적으로 이루어짐** 
	- **기본 테이블**: 뷰를 만드는데 기반이 되는 **물리적인 테이블** 
- 다른 뷰를 기반으로 새로운 뷰를 만드는 것도 가능
- 뷰는 **기본 테이블을 들여다 볼 수 있는 창 역할**을 담당 (뷰의 목적)

#### 뷰 생성 : CREATE VIEW 문
```sql
CREATE VIEW 뷰_이름[(속성_리스트)]
AS SELECT 문
[WITH CHECK OPTION];
```
- CREATE VIEW 키워드와 함께 생성할 뷰의 이름과 속성의 이름을 나열 
	- **속성 리스트를 생략하면 SELECT 절에 나열된 속성의 이름을 그대로 사용** 
- AS 키워드와 함께 기본 테이블에 대한 SELECT 문 제시 
	- SELECT 문은 생성하려는 뷰의 정의를 표현하며 **ORDER BY는 사용 불가** 
		- 오라클과 같은 일부 DBMS에서는 ORDER BY를 허용하기도 함 
- WITH CHECK OPTION 
	- 뷰에 삽입이나 수정 연산을 할 때 **SELECT 문에서 제시한 뷰의 정의 조건을 위반하면 수행되지 않도록 하는 제약조건을 지정**

![[database-systems__뷰(view__뷰 생성 CREATE VIEW 문.png]]
```SQL
CREATE VIEW 우수고객(고객아이디, 고객이름, 나이, 등급)
AS SELECT 고객아이디, 고객이름, 나이, 등급
	FROM 고객
	WHERE 등급='VIP'
WITH CHECK OPTION;

/* 생략 가능 */
CREATE VIEW 우수고객
AS SELECT 고객아이디, 고객이름, 나이, 등급
	FROM 고객
	WHERE 등급='VIP'
WITH CHECK OPTION;
```
- 뷰가 생성된 후에 우수고객 뷰에 ‘vip’ 등급이 아닌 고객 데이터를 삽입하거나 뷰의 정의 조건을 위반하는 수정 및 삭제 연산을 시도하면 실행을 거부함 (WITH CHECK OPTION 때문)

![[database-systems__뷰(view__뷰 생성 CREATE VIEW 문 2.png]]
```SQL
CREATE VIEW 업체별제품수(제조업체, 제품수)
AS SELECT 제조업체, COUNT(*)
	FROM 제품
	GROUP BY 제조업체
WITH CHECK OPTION;
```
- **제품수 속성**은 기본 테이블인 제품 테이블에 원래 있던 속성이 아니라 집계 함수를 통해 **새로 계산된 것**이므로 **속성의 이름을 명확히 제시해야 함**

#### 뷰 활용 : SELECT 문
- 일반 테이블과 같은 방법으로 원하는 데이터를 검색할 수 있음 
	- 뷰에 대한 SELECT 문이 내부적으로는 기본 테이블에 대한 SELECT 문으로 변환되어 수행
- **검색 연산은 모든 뷰에 수행 가능**

![[database-systems__뷰(view__뷰 활용 SELECT 문.png]]
```SQL
SELECT * FROM 우수고객 WHERE 나이>=20;
```

#### 뷰 활용 : INSERT, UPDATE, DELETE 문
- 뷰에 대한 삽입, 수정, 삭제 연산 가능 
	- 실제로 기본 테이블에 수행되므로 결과적으로는 기본 테이블이 변경됨
- 뷰에 대한 삽입, 수정, 삭제 연산은 제한적으로 수행됨 
	- 변경 가능한 뷰 vs. 변경 불가능한 뷰
- ==**변경 불가능한 뷰의 특징**==
	- 기본 테이블의 **기본키를 구성하는 속성이 포함되어 있지 않은 뷰** 
		- 개체 무결성의 이유
	- 기본 테이블에서 **NOT NULL로 지정된 속성이 포함되어 있지 않은 뷰** 
	- 기본 테이블에 있던 내용이 아닌 집계 함수로 **새로 계산된 내용을 포함하는 뷰**
	- **DISTINCT** 키워드를 포함하여 정의한 뷰 
	- **GROUP BY 절**을 포함하여 정의한 뷰 
	- 여러 개의 테이블을 **조인하여 정의한 뷰**는 변경이 불가능한 경우가 많음

#### ==뷰의 장점==
- 질의문을 좀 더 **쉽게 작성**할 수 있다
	- GROUP BY, 집계 함수, 조인 등을 이용해 미리 뷰를 만들어 놓으면, 복잡한 SQL 문 대신 SELECT 절과 **FROM 절만으로 원하는 데이터 검색이 가능** 
- 데이터의 **보안 유지**에 도움이 된다
	- 자신에게 제공된 뷰를 통해서만 데이터에 접근하도록 **권한 설정이 가능** 
- 데이터를 좀 더 편리하게 관리할 수 있다
	- 제공된 뷰와 관련이 없는 다른 내용에 대해 사용자가 신경 쓸 필요가 없음

#### 뷰 삭제 : DROP VIEW 문
```SQL
DROP VIEW 뷰_이름;
```
- 뷰를 삭제해도 기본 테이블은 영향을 받지 않음
- 만약, 삭제할 뷰를 참조하는 **제약조건**이 존재한다면? 
	- 뷰 삭제가 수행되지 않음 
	- **관련된 제약조건을 먼저 삭제해야 함** 
	- 예) 삭제할 뷰를 이용해 만들어진 다른 뷰가 존재하는 경우

## 5. 삽입 SQL
- 삽입 SQL(ESQL; Embedded SQL) 
	- 프로그래밍 언어로 작성된 응용 프로그램 안에 삽입하여 사용하는 SQL 문 
- 주요 특징 
	- 프로그램 안에서 일반 명령문이 위치할 수 있는 곳이면 어디든 삽입 가능 
	- 일반 명령문과 구별하기 위해 삽입 SQL 문 앞에 EXEC SQL을 붙임 
	- 프로그램에 선언된 일반 변수를 **삽입 SQL 문에서 사용할 때는 이름 앞에 콜론(:)을** 붙여서 구분함 
- 커서(cursor) 
	- 수행 결과로 반환된 여러 행을 한 번에 하나씩 가리키는 **포인터** 
	- 여러 개의 행을 결과로 반환하는 SELECT 문을 프로그램에서 사용할 때 필요
- 삽입 SQL 문에서 사용할 변수 선언 방법 
	- BEGIN DECLARE SECTION과 END DECLARE SECTION 사이에 선언 
- 커서가 필요 없는 삽입 SQL 
	- CREATE TABLE 문, INSERT 문, DELETE 문, UPDATE 문 
	- 결과로 행 하나만 반환하는 SELECT 문

```C
int main() {
	EXEC SQL BEGIN DECLARE SECTION;
		char p_no[4], p_name[21];
		int price;
	EXEC SQL END DECLARE SECTION;
	
	printf('제품번호를 입력하세요 : ');
	scanf('%s', p_no);
	
	EXEC SQL SELECT 제품명, 단가 INTO :p_name, :price
		FROM 제품
		WHERE 제품번호 = :p_no;
		
	printf('\n 제품명 = %s', p_name);
	printf('\n 단가 = %d', price);
	
	return 0;
}
```
