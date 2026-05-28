---
aliases: []
course: web-programming
created: '2024-04-09'
date: '2024-04-09'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-1
source: ''
status: seedling
tags:
- cs/se
- skill/javascript
- type/lecture
title: Spring Boot 개발 환경 세팅 확인문제
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/05_software-engineering/소프트웨어 엔지니어링 인터페이스|소프트웨어 엔지니어링 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/웹 프로그래밍 인터페이스|웹 프로그래밍 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 실습|HTML 기초 실습]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/05_software-engineering/web-programming/1. HTML 기초/HTML 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 실습|Spring Boot 기초 실습]], [[ComputerScience/05_software-engineering/web-programming/3. Spring Boot 기초/Spring Boot 기초 확인문제|확인문제]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/데이터베이스|데이터베이스]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초 실습2|HTML 기초 실습2]], [[ComputerScience/05_software-engineering/web-programming/7. 웹 시스템 제작/웹 시스템 제작|웹 시스템 제작]], [[ComputerScience/05_software-engineering/web-programming/6. HTML 기초2/HTML 기초2 문제 풀이|문제 풀이]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션|쿠키와 세션]], [[ComputerScience/05_software-engineering/web-programming/5. 데이터베이스/MYSQL|MYSQL]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/Framework|Framework]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/이벤트 이해하기|이벤트 이해하기]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/HTML JavaScript 기초 연습문제|연습문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/3. 문서 객체 모델/문서 객체 모델(DOM)|문서 객체 모델(DOM)]], [[ComputerScience/05_software-engineering/open-source-software/0. Html. javascript 기초/html, javascript 기초|html, javascript 기초]], [[ComputerScience/05_software-engineering/open-source-software/1. 이벤트 이해하기/연습 문제|연습 문제]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/배경사진 요구사항|배경사진 요구사항]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/1. 음성 인식 요구 사항|1. 음성 인식 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/TTS 요구 사항|TTS 요구 사항]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/slot 요구사항|slot 요구사항]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기 연습문제|연습문제]], [[ComputerScience/05_software-engineering/open-source-software/2. 자바스크립트 객체 다루기/자바스크립트 객체 다루기|자바스크립트 객체 다루기]], [[ComputerScience/05_software-engineering/open-source-software/고객님 요구사항/음성 인식 고객 추가 요구사항|음성 인식 고객 추가 요구사항]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/05_software-engineering/database-systems/1. 기본 개념/기본 개념|기본 개념]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/웹 프로그래밍 지식그래프|웹 프로그래밍]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/웹 프로그래밍 지식그래프|웹 프로그래밍]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/웹 프로그래밍 근거 인덱스|웹 프로그래밍 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/doctype html|doctype html]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/input type|input type]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/웹 프로그래밍|웹 프로그래밍]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/html|html]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/web-programming/model mo|model mo]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]

---

**Quiz #1:** 
spring은 java 기반 웹 애플리케이션을 만들 수 있는 framework입니다. 이번 학기 우리가 배우는 ( )는 복잡한 설정 없이 spring 개발을 할 수 있도록 도와줍니다. 괄호 안에 뭐가 들어갈까요?
- (1) php
- (2) jsp
- (3) node.js
- (4) spring boot

**정답:** (4) spring boot

---

**Quiz #2:** 
front-end VS back-end 구분하세요.
- html
- spring boot
- javascript
- php
- jsp
- css
- node.js
- django

**정답:**
- Front-end: html, javascript, css
- Back-end: spring boot, php, jsp, node.js, django

---

**Quiz #3:** OX 문제
1) spring boot를 개발하려면 꼭 jdk를 설치해야 한다. (O)
2) spring boot를 개발하려면 꼭 eclipse를 설치해야 한다. (X)
3) python웹 및 go웹은 백엔드 아니고 프론트엔드이다. (X)
4) spring boot 개발할 때 우리 컴퓨터가 웹서버 역할을 할 수 있으려면 아파치톰캣을 별도로 설치해야 한다. (X)

---

**Quiz #4:** 
2장에서 우리가 설치한 두 가지가 무엇이며, 각각 왜 설치했는지 설명하세요.
(jdk와 이클립스 설치했음)

**정답:**
- JDK(Java Development Kit): 자바 언어로 프로그램을 개발하고 실행할 수 있도록 필요한 도구들을 제공하는 개발 키트입니다. Spring Boot 프레임워크를 사용하기 위해서는 Java 개발 환경이 필요하므로 JDK를 설치했습니다.
- 이클립스(Eclipse): 통합 개발 환경(IDE)으로, Java 프로그래밍을 위한 코드 편집, 디버깅, 컴파일 등의 기능을 제공합니다. Spring Boot 프로젝트를 효율적으로 관리하고 개발하기 위해 이클립스를 설치했습니다.
