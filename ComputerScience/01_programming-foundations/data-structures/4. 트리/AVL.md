---
aliases: []
course: data-structures
created: '2024-05-14'
date: '2024-05-14'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-1
source: ''
status: seedling
tags:
- cs/algorithms
- type/lecture
title: AVL
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/01_programming-foundations/프로그래밍 기초 인터페이스|프로그래밍 기초 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/자료구조 인터페이스|자료구조 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/01_programming-foundations/data-structures/4. 트리/트리 (TREE)|트리 (TREE)]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/01_programming-foundations/data-structures/시험/기말/기말_데이터구조 답지|기말_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/스택|스택]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬|정렬]], [[ComputerScience/01_programming-foundations/data-structures/시험/중간/중간_데이터구조 답지|중간_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/1705817_엄윤상_데이터구조_4주차과제|1705817_엄윤상_데이터구조_4주차과제]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/Queue|Queue]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/Stack|Stack]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/리스트|리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/큐|큐]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/전위, 후위 표기법|전위, 후위 표기법]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/실습과제/트리 만들기|트리 만들기]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/DecisionTree/Decision_Tree|Decision_Tree]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/5. 스택과 큐|5. 스택과 큐]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트|1. 배열과 리스트]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/3. 투 포인터|3. 투 포인터]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 2 풀이|Pop Quiz 2 풀이]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/4. 슬라이딩 윈도우|4. 슬라이딩 윈도우]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/2. 구간 합|2. 구간 합]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리|기말고사_정리]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/05_software-engineering/programming-languages/교재/4장_교재_문제|4장_교재_문제]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/coding-test/정렬/1. 버블 정렬|1. 버블 정렬]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/java-programming/4. 연산자|4. 연산자]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/연산자|연산자]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]], [[ComputerScience/05_software-engineering/programming-languages/과제/9장 레포트|9장 레포트]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/05_software-engineering/programming-languages/교재/3장_교재_문제|3장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론|3. 구문론]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/05_software-engineering/programming-languages/교재/5장_교재_문제|5장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/7장-12장 연습문제 종합|7장-12장 연습문제 종합]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[ComputerScience/05_software-engineering/programming-languages/필기/4. 재귀 하강 파싱|4. 재귀 하강 파싱]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 연습문제 (과제)|3장 연습문제 (과제)]], [[ComputerScience/05_software-engineering/programming-languages/필기/0. 명령어 집합|0. 명령어 집합]], [[ComputerScience/05_software-engineering/programming-languages/교재/6장_교재_문제|6장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 제출용|3장 제출용]], [[ComputerScience/05_software-engineering/programming-languages/과제/4장 재귀 하강 파서 연습문제|4장 재귀 하강 파서 연습문제]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/자료구조 근거 인덱스|자료구조 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/데이터 구조|데이터 구조]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/스택과 큐|스택과 큐]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
#### AVL 트리
**이진 탐색 트리
- 자료를 탐색하는데 최적화된 트리
- 자료가 오름차순이나 내림차순과 같은 순서대로 입력 될 경우 연결리스트와 같은 형태 -> 빠른 검색속도를 이용하지 못함.
-> 해결알고리즘 : 완전 이진 탐색 트리

**완전 이진 탐색 트리
- 트리에 자료가 삽입될 때 마다 완전 이진 탐색 트리 형태 유지를 위해 모양 변경 동작 발생
- 삽입할 때 많은 시간 소요
- 삽입이 적고 탐색이 많은 경우 유리 ->삽입 동작이 빈도수가 높아질수록 효율성 감소
-> 해결알고리즘 : AVL 트리

**AVL 트리
- 1962년 Adelson-Velskii 와 Landis에 의해 제안
- 트리 내의 모든 노드에 대해 왼쪽 서브 트리의 높이와 오른쪽 서브 트리의 높이가 1이상 차이가 나지 않는 균형 이진 트리 (height balanced binary tree)
- 이진 탐색 트리 내의 임의의 노드 N에 대해서 균형 인수 BF(N)가 -1, 0, 1의 값을 갖는 트리

**AVL 균형인수
- 왼쪽 서브트리의 높이와 오른쪽 서브트리의 높이 차를 말함
- 균형인수는 보통 BF(T)로 표기
- BF(T) = height(left tree) - height(right tree)

#### AVL 트리의 회전 연산
**AVL 트리의 회전 연산
- AVL 트리에서의 삽입, 삭제 과정은 이진 탐색 트리에서의 삽입 삭제 과정과 동일 
- 삽입, 삭제 후 균형 인수에 따라 트리를 재조정하는 과정이 필요 
- 균형인수(BF)를 -1이상에서 1 이하로 조절하는 재조정 동작

| 회전 유형 | 삽입에 따른 회전 종류/방식 설명                                                                                             |
| ----- | -------------------------------------------------------------------------------------------------------------- |
| LL 회전 | 노드 N이 A의 왼쪽 서브 트리의 왼쪽 서브 트리에 삽입되는 경우 Single Right Rotation<br>(A를 기준으로 한번의 시계 방향 회전)                           |
| LR 회전 | 노드 N이 A의 왼쪽 서브 트리의 오른쪽 서브 트리에 삽입되는 경우<br>Left-Right Rotation<br>(A의 왼쪽 자식을 기준으로 반시계 방향 회전 후 A를 기준으로 시계 방향 회전)  |
| RR 회전 | 노드 N이 A의 오른쪽 서브 트리의 오른쪽 서브 트리에 삽입되는 경우 Single Left Rotation<br>(A를 기준으로 한번의 반시계 방향 회전)                         |
| RL 회전 | 노드 N이 A의 오른쪽 서브 트리의 왼쪽 서브 트리에 삽입되는 경우<br>Right-Left Rotation<br>(A의 오른쪽 자식을 기준으로 시계 방향 회전 후 A를 기준으로 반시계 방향 회전) |
**LL 회전
- 왼쪽 서브 트리의 왼쪽 서브 트리에 노드가 추가되면서 불균형이 발생했을 때 사용

	![[data-structures__AVL__AVL 트리의 회전 연산.png]]

**RR 회전
- 오른쪽 서브 트리의 오른쪽 서브 트리에 노드가 추가되면서 불균형이 발생했을 때 사용

	![[data-structures__AVL__AVL 트리의 회전 연산 2.png]]

**LR 회전
- 왼쪽 서브 트리의 오른쪽 서브 트리에 노드가 추가되면서 불균형이 발생했을 때 사용

	![[data-structures__AVL__AVL 트리의 회전 연산 3.png]]

**RL 회전
- 오른쪽 서브 트리의 왼쪽 서브 트리에 노드가 추가되면서 불균형이 발생했을 때 사용

	![[data-structures__AVL__AVL 트리의 회전 연산 4.png]]

## AVL 트리 코드 구현
#### AVl 트리 코드 구현

![[data-structures__AVL__AVl 트리 코드 구현.png]]
![[data-structures__AVL__AVl 트리 코드 구현 2.png]]
![[data-structures__AVL__AVl 트리 코드 구현 3.png]]
![[data-structures__AVL__AVl 트리 코드 구현 4.png]]
