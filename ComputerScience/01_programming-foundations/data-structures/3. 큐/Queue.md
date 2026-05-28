---
aliases: []
course: data-structures
created: '2024-08-16'
date: '2024-08-16'
semester: 2-1
source: ''
status: seedling
tags:
- cs/algorithms
- type/lecture
title: Queue
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/01_programming-foundations/프로그래밍 기초 인터페이스|프로그래밍 기초 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/자료구조 인터페이스|자료구조 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/01_programming-foundations/data-structures/3. 큐/큐|큐]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/01_programming-foundations/data-structures/2. 스택/스택|스택]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/리스트|리스트]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/Stack|Stack]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/시험/중간/중간_데이터구조 답지|중간_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬|정렬]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/1705817_엄윤상_데이터구조_4주차과제|1705817_엄윤상_데이터구조_4주차과제]], [[ComputerScience/01_programming-foundations/data-structures/4. 트리/트리 (TREE)|트리 (TREE)]], [[ComputerScience/01_programming-foundations/data-structures/4. 트리/AVL|AVL]], [[ComputerScience/01_programming-foundations/data-structures/시험/기말/기말_데이터구조 답지|기말_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/전위, 후위 표기법|전위, 후위 표기법]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/5. 스택과 큐|5. 스택과 큐]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/3. 투 포인터|3. 투 포인터]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/05_software-engineering/programming-languages/과제/9장 레포트|9장 레포트]], [[ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론|3. 구문론]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트|1. 배열과 리스트]], [[ComputerScience/01_programming-foundations/coding-test/정렬/1. 버블 정렬|1. 버블 정렬]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/4. 슬라이딩 윈도우|4. 슬라이딩 윈도우]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/2. 구간 합|2. 구간 합]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/전역변수, 지역변수|전역변수, 지역변수]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/문법|문법]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리|기말고사_정리]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/연산자|연산자]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/05_software-engineering/programming-languages/과제/4장 재귀 하강 파서 연습문제|4장 재귀 하강 파서 연습문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/05_software-engineering/programming-languages/필기/4. 재귀 하강 파싱|4. 재귀 하강 파싱]], [[ComputerScience/05_software-engineering/programming-languages/교재/5장_교재_문제|5장_교재_문제]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/05_software-engineering/programming-languages/교재/6장_교재_문제|6장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/0. 명령어 집합|0. 명령어 집합]], [[ComputerScience/05_software-engineering/programming-languages/7장-12장 연습문제 종합|7장-12장 연습문제 종합]], [[ComputerScience/05_software-engineering/programming-languages/교재/3장_교재_문제|3장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/교재/4장_교재_문제|4장_교재_문제]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 2 풀이|Pop Quiz 2 풀이]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/실습과제/트리 만들기|트리 만들기]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 제출용|3장 제출용]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 연습문제 (과제)|3장 연습문제 (과제)]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/자료구조 근거 인덱스|자료구조 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/데이터 구조|데이터 구조]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/스택과 큐|스택과 큐]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
## 큐

- 큐(Queue): 삽입과 삭제가 양 끝에서 각각 수행되는 자료구조 

- 일상생활의 관공서, 은행, 우체국, 병원 등에서 번호표를 이용한 줄서기 

- 선입 선출(First-In First-Out, FIFO)

#### `파이썬 리스트 큐`

```python
def add(item): # 삽입연산
	q.append(item)

def remove(): # 삭제연산
	if len(q) != 0:
		item = q.pop(0)
		return item

def print_q(): # 큐출력
	print('front -> ', end='')
	for i in range(len(q)):
		print('{!s:<8}'.format(q[i]), end='')
	print(' <- rear')
```

---
#### `단순 연결 리스트 큐`

```python
class Node:  
    def __init__(self, item, n):  
        self.item = item  
        self.next = n  
  
  
front = None  
rear = None  
size = 0  
  
  
def add(item):  # 삽입 연산  
    global size  
    global front  
    global rear  
    new_node = Node(item, None)  
    if size == 0:  
        front = new_node  
    else:  
        rear.next = new_node  
    rear = new_node  
    size += 1  
  
  
def remove():  # 삭제 연산  
    global size  
    global front  
    global rear  
    if size != 0:  
        fitem = front.item  
        front = front.next  
        size -= 1  
        if size == 0:  
            rear = None  
        return fitem  
  
  
def print_q():  # 큐출력  
    p = front  
    print('front: ', end='')  
    while p:  
        if p.next is not None:  
            print(p.item, '-> ', end='')  
        else:  
            print(p.item, end='')  
        p = p.next  
    print(' : reear')
```

#### 수행 시간

- 리스트로 구현한 큐의 add와 remove 연산: 각각 O(1) 시간 
	- 리스트 크기를 확대/축소시키는 경우에 큐의 모든 항목을 새 리스트에 복사해야 하므로 O(n) 시간 
- 단순 연결 리스트 큐의 add와 remove 연산은 각각 O(1) 시간 
	- 삽입 또는 삭제 연산이 rear 와 front로 인해 연결 리스트의 다른 노드를 방문할 필요 없음

---
## 데크

![](../../../../image/Pasted%20image%2020240816190547.png)

- 데크(Double-ended Queue, Deque): 양쪽 끝에서 삽입과 삭제를 허용하는 자료구조 

- 데크는 스택과 큐 자료구조를 혼합한 자료구조 

- 따라서 데크는 스택과 큐를 동시에 구현하는데 사용

- 데크를 이중 연결 리스트로 구현하는 것이 편리

- 단순 연결 리스트는 노드의 이전 노드의 레퍼런스를 알아야 삭제

- 파이썬에는 데크가 Collections 패키지에 정의되어 있음

- 삽입, 삭제 등의 연산은 파이썬의 리스트의 연산과 매우 유사

```python
from collections import deque     
  
dq = deque('data')  
for elem in dq:  
    print(elem.upper(), end='')  
print()  
  
dq.append('r')  
dq.appendleft('k')  
print(dq)  
  
dq.pop()  
dq.popleft()  
print(dq[-1])  
print('x' in dq)      
  
dq.extend('structure')  
dq.extendleft(reversed('python'))  
print(dq)
```

#### 수행 시간

- 데크를 배열이나 이중 연결 리스트로 구현한 경우, 스택과 큐의 수행 시간과 동일 
- 양 끝에서 삽입과 삭제가 가능하므로 프로그램이 다소 복잡
- 이중 연결 리스트로 구현한 경우는 더 복잡함
---
## 요약

- 큐는 삽입과 삭제가 양 끝에서 각각 수행되는 선입 선출(FIFO) 자료구조 
- 큐는 CPU의 태스크 스케줄링, 네트워크 프린터, 실시간 시스템의 인터럽트 처리, 다양한 이벤트 구동 방식 컴퓨터 시뮬레이션, 콜 센터의 전화 서비스 처리 등에 사용되며, 이진트리의 레벨순회와 그래프의 BFS에 사용 
- 데크는 양쪽 끝에서 삽입과 삭제를 허용하는 자료구조로서 스택과 큐 자료구조를 혼합한 자료구조 
- 데크는 스크롤, 문서 편집기의 undo 연산, 웹 브라우저의 방문 기록 등에 사용
