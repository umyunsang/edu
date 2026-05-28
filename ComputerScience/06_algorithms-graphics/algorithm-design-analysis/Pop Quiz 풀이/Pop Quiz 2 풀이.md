---
aliases: []
course: algorithm-design-analysis
created: '2026-03-23'
date: '2026-03-23'
semester: 4-1
source: ''
status: 정리완료
tags:
- cs/algorithms
- type/lecture
- 시험대비
- 알고리즘
- 트리순회
- 해싱
title: 'Pop Quiz #2 문제 풀이'
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/06_algorithms-graphics/알고리즘 그래픽스 인터페이스|알고리즘 그래픽스 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/4단계 시스템 실전 인터페이스|4단계 시스템 실전 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/알고리즘 설계와 분석 인터페이스|알고리즘 설계와 분석 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/수학 알고리즘 브리지|수학 알고리즘 브리지]]
up:: [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]]
prerequisites:: [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/정렬|정렬]], [[ComputerScience/02_math-theory/discrete-mathematics/4. 그래프/그래프|그래프]]
related:: [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/실습과제/트리 만들기|트리 만들기]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리|기말고사_정리]], [[ComputerScience/01_programming-foundations/data-structures/4. 트리/트리 (TREE)|트리 (TREE)]], [[ComputerScience/05_software-engineering/web-programming/4. 쿠키와 세션/쿠키와 세션 확인문제|확인문제]], [[ComputerScience/01_programming-foundations/data-structures/시험/기말/기말_데이터구조 답지|기말_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/시험/중간/중간_데이터구조 답지|중간_데이터구조 답지]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/스택|스택]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/전위, 후위 표기법|전위, 후위 표기법]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/Stack|Stack]], [[ComputerScience/01_programming-foundations/data-structures/4. 트리/AVL|AVL]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/5. 스택과 큐|5. 스택과 큐]], [[ComputerScience/06_algorithms-graphics/computer-graphics/지오매트리|지오매트리]], [[ComputerScience/01_programming-foundations/data-structures/5. 정렬/1705817_엄윤상_데이터구조_4주차과제|1705817_엄윤상_데이터구조_4주차과제]], [[ComputerScience/01_programming-foundations/coding-test/정렬/1. 버블 정렬|1. 버블 정렬]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/1. 배열과 리스트|1. 배열과 리스트]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/Discrete mathematics Assignment|Discrete mathematics Assignment]], [[ComputerScience/02_math-theory/discrete-mathematics/3. 관계와 함수/관계와 함수|관계와 함수]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/2. 구간 합|2. 구간 합]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/3. 투 포인터|3. 투 포인터]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/단순 연결 리스트|단순 연결 리스트]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/이중 연결 리스트|이중 연결 리스트]], [[ComputerScience/01_programming-foundations/coding-test/자료구조/4. 슬라이딩 윈도우|4. 슬라이딩 윈도우]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/원형 연결 리스트|원형 연결 리스트]], [[ComputerScience/02_math-theory/discrete-mathematics/과제/과제 번역|과제 번역]], [[ComputerScience/02_math-theory/discrete-mathematics/2. 집합 및 집합 연산/집합 및 집합 연산|집합 및 집합 연산]], [[ComputerScience/02_math-theory/discrete-mathematics/1. 수학적 모델과 논리/수학적 모델과 논리|수학적 모델과 논리]], [[ComputerScience/01_programming-foundations/data-structures/1. 리스트/리스트|리스트]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/Queue|Queue]], [[ComputerScience/01_programming-foundations/data-structures/3. 큐/큐|큐]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/알고리즘 설계와 분석 지식그래프|알고리즘 설계와 분석]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/알고리즘 설계와 분석 지식그래프|알고리즘 설계와 분석]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/알고리즘 설계와 분석 근거 인덱스|알고리즘 설계와 분석 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/algorithm-design-analysis/억지 기법|억지 기법]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/algorithm-design-analysis/시간 복잡도|시간 복잡도]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/algorithm-design-analysis/근사 알고리즘|근사 알고리즘]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/algorithm-design-analysis/복잡도 분석|복잡도 분석]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/algorithm-design-analysis/해시 함수|해시 함수]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

# Pop Quiz #2 문제 풀이

### [문제 1/2] 해싱 (Hashing)
> [!question]
> **문제 1 (01분반)**: Linear probing 방식으로 마지막 키 18이 저장되는 곳의 인덱스는?
> **문제 2 (02분반)**: Quadratic probing 방식으로 마지막 키 24가 저장되는 곳의 인덱스는?

> [!summary] 풀이 (PDF 원문)
> **문제 1**: $P = (h+i) \% M$. 44(5), 37(11), 31(5), 63(11), 14(1), 58(6), 45(6), 18(5). 키 31(5) 삽입 시 5번 방 충돌 $\to$ 6번 방 저장. 키 63(11) $\to$ 12번 방. 키 58(6) $\to$ 7번 방. 키 45(6) $\to$ 8번 방. 마지막 키 18(5)는 충돌을 거쳐 **9번 방**에 저장된다. 정답: 9.
> 
> **문제 2**: $P = (h+i^2) \% M$. 45(6), 38(12), 32(6), 64(12), 15(2), 59(7), 46(7), 24(11). 키 32(6) 충돌 시 $i=1$일 때 $6+1=7$번 방 충돌 $\to i=2$일 때 $6+4=10$번 방 저장. 마지막 키 24(11)은 $i=4$일 때 **1번 방** 비어있으므로 저장된다. 정답: 1.

> [!example] 상세 해설
> 선형 조사법(Linear Probing)은 충돌 시 단순히 다음 빈칸(+1)을 찾지만, 이차 조사법(Quadratic Probing)은 충돌 횟수의 제곱($1, 4, 9, \dots$)만큼 건너뛰며 빈칸을 찾습니다. 이 방식은 데이터가 뭉치는 클러스터링 현상을 방지하는 데 더 효과적입니다.

### [문제 3/4] 숫자 야구 (Number Baseball)
> [!question]
> 예측 수 "3689"에 대해 힌트가 주어졌을 때 가능한 수의 개수는?
> 3) "2 strike 1 ball" / 4) "0 strike 4 ball" / 5) "9135"에 대해 "1 strike 2 ball"

> [!summary] 풀이 (PDF 원문)
> *   **2 strike 1 ball**: 가능한 경우 총 **60가지**.
> *   **0 strike 4 ball**: 총 **9가지** (첫째 수가 6인 경우 세 가지, 8인 경우 세 가지, 9인 경우 세 가지).
> *   **1 strike 2 ball**: 가능한 모든 경우는 $4 \times 45 = \mathbf{180}$가지.

> [!example] 상세 해설
> 숫자 야구에서 0S 4B는 모든 숫자가 정답에 포함되지만 위치만 모두 틀린 '교란 순열(Derangement)' 유형입니다. 각 숫자가 원래 위치에 오지 않도록 배열하는 경우의 수를 따지면 9가지가 나옵니다. 1S 2B는 스트라이크가 될 숫자 하나를 정하고($\binom{4}{1}$), 나머지 위치에 숫자를 배치하는 경우의 수($45$)를 곱해 180가지가 도출됩니다.

### [문제 5/6] 트리 순회 및 복원 (Tree Traversal)
> [!question]
> 중위/전위 또는 중위/후위 순회 결과가 주어졌을 때 나머지 순회 결과는?

> [!summary] 풀이 (PDF 원문)
> **문제 5**: 중위(2 3 1 5 6 7 4), 전위(3 2 7 5 1 6 4). 전위의 첫 노드 3이 Root. 중위에서 3 기준 좌(2)/우(1,5,6,7,4) 분할. 결과: **2 1 6 5 4 7 3** (후위).
> 
> **문제 6**: 중위(3 6 1 2 7 5 4), 후위(6 1 3 7 4 5 2). 후위의 마지막 노드 2가 Root. 중위에서 2 기준 좌(3,6,1)/우(7,5,4) 분할. 결과: **2 3 6 1 5 7 4** (전위).

> [!example] 상세 해설
> 트리를 복원할 때 가장 먼저 할 일은 전체 트리의 Root를 찾는 것입니다. 전위 순회(Root-좌-우)는 맨 앞에, 후위 순회(좌-우-Root)는 맨 뒤에 Root가 옵니다. 찾은 Root를 중위 순회 결과에 대입하면 좌/우 자식들이 물리적으로 구분되므로 이를 반복하여 트리를 완성합니다.

### [문제 8/9] 기타 문제
> [!question]
> 8) 개구리가 가장 멀리 뛰어야 하는 최솟값은?
> 9) "KAKARBAR"에 대해 A, B, K, T, R 각각의 Shift table 값은?

> [!summary] 풀이 (PDF 원문)
> **문제 8**: 동쪽/서쪽으로 이동할 때 작은 돌을 번갈아 밟아야 멀리 뛰는 거리를 최소화할 수 있다. 그림처럼 뛰는 거리 중 가장 긴 **30**이 답이다.
> 
> **문제 9**: Shift Table 결과 A(1), B(2), K(5), T(8), R(3). 모두 이어 붙인 5자리 수는 **12583**이다.

> [!example] 상세 해설
> 개구리 문제는 모든 점프 거리 중 '최댓값'을 '최소화'하는 전략을 묻습니다. 한쪽으로만 가면 구간이 길어지므로 지그재그로 돌을 밟아 균형을 맞춥니다. 호스풀 알고리즘의 이동 테이블은 $m-1-i$ 공식을 쓰되, 패턴의 마지막 글자는 제외하고 우측 끝에서부터 첫 출현 위치를 찾아 거리를 계산합니다.
