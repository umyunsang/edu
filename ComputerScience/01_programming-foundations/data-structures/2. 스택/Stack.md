---
aliases: []
course: data-structures
created: '2024-04-03'
date: '2024-04-03'
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
title: 5개의 웹 주소를 스택에 추가합니다.
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/01_programming-foundations/프로그래밍 기초 인터페이스|프로그래밍 기초 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/자료구조 인터페이스|자료구조 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/01_programming-foundations/data-structures/2. 스택/스택|스택]]
prerequisites:: [[ComputerScience/01_programming-foundations/python-programming/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/리스트|리스트]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/큐|큐]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/전위, 후위 표기법|전위, 후위 표기법]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/Queue|Queue]], [[ComputerScience/01_programming-foundations/data-structures/시험/중간/중간_데이터구조 답지|중간_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬|정렬]], [[ComputerScience/01_programming-foundations/data-structures/시험/기말/기말_데이터구조 답지|기말_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/1705817_엄윤상_데이터구조_4주차과제|1705817_엄윤상_데이터구조_4주차과제]], [[ComputerScience/01_programming-foundations/data-structures/4. 트리/트리 (TREE)|트리 (TREE)]], [[ComputerScience/01_programming-foundations/data-structures/4. 트리/AVL|AVL]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 1-10|프로그래머스 Python 기초 문제 1-10]], [[ComputerScience/01_programming-foundations/python-programming/문제풀이/프로그래머스 Python 기초 문제 11-20|프로그래머스 Python 기초 문제 11-20]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/5. 스택과 큐|5. 스택과 큐]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/3. 투 포인터|3. 투 포인터]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/01_programming-foundations/python-programming/5. 리스트, 튜플, 딕셔너리|5. 리스트, 튜플, 딕셔너리]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/01_programming-foundations/python-programming/지뢰찾기/지뢰찾기|지뢰찾기]], [[ComputerScience/01_programming-foundations/python-programming/3. 반복문|3. 반복문]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/Discrete mathematics Assignment|Discrete mathematics Assignment]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/01_programming-foundations/python-programming/7. 파일 읽기와 쓰기|7. 파일 읽기와 쓰기]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[certifications/information-processing/실기/C언어 실기 오답노트|오답노트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/이상 탐지(ASD)를 위한 최적의 Feature Engineering|이상 탐지(ASD)를 위한 최적의 Feature Engineering]], [[ComputerScience/05_software-engineering/programming-languages/과제/9장 레포트|9장 레포트]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론|3. 구문론]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트|1. 배열과 리스트]], [[ComputerScience/01_programming-foundations/java-programming/2. 변수와 자료형|2. 변수와 자료형]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/01_programming-foundations/coding-test/정렬/1. 버블 정렬|1. 버블 정렬]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 2 풀이|Pop Quiz 2 풀이]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/4. 슬라이딩 윈도우|4. 슬라이딩 윈도우]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/2. 구간 합|2. 구간 합]], [[ComputerScience/01_programming-foundations/python-programming/중간시험 범위|중간시험 범위]], [[ComputerScience/01_programming-foundations/python-programming/2. 연산자|2. 연산자]], [[ComputerScience/01_programming-foundations/python-programming/중간/답지|답지]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/문법|문법]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/실습과제/트리 만들기|트리 만들기]], [[ComputerScience/01_programming-foundations/java-programming/5. 조건문|5. 조건문]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/05_software-engineering/programming-languages/교재/5장_교재_문제|5장_교재_문제]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/전역변수, 지역변수|전역변수, 지역변수]], [[ComputerScience/01_programming-foundations/java-programming/3. Scanner|3. Scanner]], [[ComputerScience/05_software-engineering/programming-languages/과제/4장 재귀 하강 파서 연습문제|4장 재귀 하강 파서 연습문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/05_software-engineering/programming-languages/7장-12장 연습문제 종합|7장-12장 연습문제 종합]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]], [[ComputerScience/05_software-engineering/programming-languages/필기/4. 재귀 하강 파싱|4. 재귀 하강 파싱]], [[ComputerScience/01_programming-foundations/coding-basics/4. 아두이노/아두이노 실습|아두이노 실습]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[알고리즘_기말고사_정리|기말고사_정리]], [[ComputerScience/05_software-engineering/programming-languages/교재/4장_교재_문제|4장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/교재/6장_교재_문제|6장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/05_software-engineering/programming-languages/교재/3장_교재_문제|3장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/0. 명령어 집합|0. 명령어 집합]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 제출용|3장 제출용]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 연습문제 (과제)|3장 연습문제 (과제)]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/자료구조 근거 인덱스|자료구조 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/데이터 구조|데이터 구조]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/data-structures/스택과 큐|스택과 큐]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
스택(Stack)은 데이터를 저장하는 선형 자료구조 중 하나로, 데이터를 한쪽 끝에서만 삽입하고 삭제할 수 있는 구조를 갖습니다. 이러한 특성 때문에 "후입선출" (Last-In-First-Out, LIFO) 구조라고도 합니다. 

스택은 일상 생활에서 쉽게 비유할 수 있습니다. 예를 들어, 책상 위에 책을 쌓아놓을 때 가장 위에 있는 책부터 차례대로 가져다 쓰게 되는데, 이것이 스택의 동작 방식과 비슷합니다.

## Stack 개념

![[data-structures__5개의 웹 주소를 스택에 추가합니다__Stack 개념.png]]

-  LIFO (Last In First Out) 또는 FILO (First In Last Out) 데이터 출입 통로가 하나인 구조 
- "**TOP**" : 데이터 최상위 데이터 위치를 의미함. 
- “**PUSH**” 동작 : 스택 구조에 데이터를 추가하는 동작 (APPEND) 데이터가 추가할 때마다 “TOP” 변경 
- “**POP**” 동작 : 스택 구조에 데이터를 빼내는 동작 (DELETE) 데이터가 빼낼 때마다 “TOP” 변경 
- 스택 예시: 웹 브라우저 뒤로 가기, 생산성 도구(문서편집 등)의 “Undo” (Ctrl+z) 
- 스택 구현 방법: 
	- NODE를 이용한 직접 코딩 
	- 파이썬의 list 자료형을 활용한 코딩

## 스택 구현 - Python list 자료형 활용

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            return "Stack is empty"

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            return "Stack is empty"

    def is_empty(self):
        return len(self.items) == 0
```

1. **push(item)**: 스택의 맨 위에 새로운 데이터를 추가합니다. 리스트의 append() 메서드를 사용하여 구현하였습니다.

2. **pop()**: 스택의 맨 위에 있는 데이터를 제거하고 반환합니다. 리스트의 pop() 메서드를 사용하여 구현하였습니다.

3. **peek()**: 스택의 맨 위에 있는 데이터를 반환하지만 제거하지는 않습니다. 이 연산을 통해 스택의 맨 위에 어떤 데이터가 있는지 확인할 수 있습니다.

1. **is_empty()**: 스택이 비어있는지 여부를 확인합니다. 스택이 비어있으면 True를 반환하고, 그렇지 않으면 False를 반환합니다.

## 스택 구현 - 단순연결리스트 NODE 활용

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.top = None
        
    def push(self, data):
        new_node = Node(data)
        if self.top is None:
            self.top = new_node
            return
        new_node.next = self.top
        self.top = new_node

    def pop(self):
        if self.is_empty():
            return "Stack is empty"
        else:
            popped = self.top.data     # TOP에 대한 임시변수 지정
            self.top = self.top.next
            return popped

    def peek(self):
        if self.top is None:
            return "Stack is empty"
        else:
            return self.top.data
```

- 단순연결리스트 NODE 활용 

- “PUSH” 동작: 
	 ① newNode 생성 
	 ② newNode.next = self.top 
	 ③ self.top = newNode 
	 ④ 스택크기 증가 
	 
 - “POP” 동작: 
	 ① TOP 에 대한 임시변수 지정 
	 ② self.top = self.top.next 
	 ③ 임시변수.next = None 
	 ④ 스택크기 감소 
	 
- 스택 출력: 단순연결리스트 출력과 동일

## 스택 활용
### **TRY**
1) 웹 주소를 저장할 수 있는 변수를 만들고, 5개 웹 주소를 스택 구조에 push 하세요.
2) 키보드 이벤트를 활용하여 왼쪽 화살표 방향키 (←)를 입력하면 해당 웹 주소를 출력하고, 스택에서 제거하세요. 
3) hint) 
```python
	import keyboard
	import time ## 키보드 입력 대기를 위해 무한반복이 필수이며, 
				## 무한 반복시 키보드 값이 계속 입력되므로 time.sleep 을 줘야 함. 
```

아래는 요구사항에 맞게 작성된 파이썬 코드입니다.

```python
import keyboard
import time

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            return "Stack is empty"

    def is_empty(self):
        return len(self.items) == 0

web_address_stack = Stack()

# 5개의 웹 주소를 스택에 추가합니다.
web_address_stack.push("www.google.com")
web_address_stack.push("www.github.com")
web_address_stack.push("www.stackoverflow.com")
web_address_stack.push("www.youtube.com")
web_address_stack.push("www.facebook.com")

print("Pushed 5 web addresses to stack.")

while True:
    if keyboard.is_pressed('left'):
        time.sleep(0.2)  # 키보드 입력에 대한 대기 시간 설정
        if not web_address_stack.is_empty():
            web_address = web_address_stack.pop()
            print("Popped web address:", web_address)
        else:
            print("Stack is empty")
``` 

### 수업 자료
```python
import keyboard  
import time  
from getkey import getkey  
  
class stack_list:  
    def __init__(self):  
        self.items = []  
  
    def push(self, item):  
        self.items.append(item)  
  
    def pop(self):  
        if len(self.items) == 0:  
            return print("Stack is Empty")  
        return self.items.pop()  
  
    def topItem(self):  
        return self.items[-1]  
  
    def isEmpty(self):  
        return not self.items  
  
  
stk = stack_list()  
  
stk.push("http://www.naver.com")  
stk.push("http://www.google.com")  
stk.push("http://www.daum.net")  
stk.push("http://www.tistory.com")  
stk.push("http://www.donga.ac.kr")  
  
print(stk.items)  
  
while True:  
   if keyboard.is_pressed('left'):  
       time.sleep(0.1)  
       stk.pop()  
       print(stk.items)  
   elif  keyboard.is_pressed('q'):  
       time.sleep(0.1)  
       break;
```

## 스택 응용문제 - 프로그래밍에서 수식 표현 방식

#### 수식 중위(Infix) 표기 / 후위(Postfix) 표기 
- 중위 표기 : 수식에서 +, -, *, / 와 같은 이항 연산자는 2개의 피 연산자 사이에 위치 
- 후위 표기 : 컴파일러는 중위 표기 수식을 후위 표기로 바꾼다. 
	- 후위 표기 방식은 괄호없이 중위 표기 수식을 표현할 수 있음. 
- 전위(Prefix) 표기: 연산자를 피 연산자들 앞에 두는 표기법

![[data-structures__5개의 웹 주소를 스택에 추가합니다__수식 중위(Infix) 표기 후위(Postfix) 표기.png]]

**다음 수식을 입력 받아 후위 표기로 출력하는 코드를 작성하세요. 
문제1. 입력 => A * ( B + C / D ) 
문제2. 입력 => ( A * B ) + ( C / D )**

```python

st = cStack()  # cStack 클래스의 인스턴스 생성
prec = {'*': 3, '/': 3, '+': 2, '-': 2, '(': 1}  # 연산자의 우선순위를 저장한 딕셔너리

formula = '(A*B)/(C-D)'  # 변환할 중위 표기법의 수식
postfix = ''  # 후위 표기법으로 변환된 수식을 저장할 변수

# 중위 표기법 수식을 한 글자씩 읽어가면서 처리
for w in formula:
    if w in prec:  # 현재 글자가 연산자인 경우
        if st.isEmpty():  # 스택이 비어있는 경우
            st.push(w)  # 현재 연산자를 스택에 추가
        else:  # 스택이 비어있지 않은 경우
            if w == '(':  # 현재 연산자가 여는 괄호인 경우
                st.push(w)  # 스택에 추가
            else:  # 현재 연산자가 닫는 괄호가 아닌 경우
                # 스택의 top 연산자와 현재 연산자의 우선순위를 비교하여 처리
                while prec.get(w) <= prec.get(st.topItem()):
                    postfix += st.pop()  # 스택의 top 연산자를 후위 표기법에 추가
                    if st.isEmpty():  # 스택이 비어있으면 반복문 종료
                        break
                st.push(w)  # 현재 연산자를 스택에 추가
    elif w == ')':  # 현재 글자가 닫는 괄호인 경우
        # 스택의 top 연산자가 여는 괄호가 나올 때까지 스택에서 pop하여 후위 표기법에 추가
        while st.topItem() != '(':
            postfix += st.pop()
        st.pop()  # 여는 괄호를 스택에서 pop하여 제거
    else:  # 현재 글자가 피연산자인 경우
        postfix += w  # 후위 표기법에 추가

# 스택에 남은 모든 연산자를 후위 표기법에 추가
while not st.isEmpty():
    postfix += st.pop()

print(postfix)  # 후위 표기법으로 변환된 수식 출력

```
