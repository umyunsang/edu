# Discrete Mathematics: Homework I 

### Prof. Chun 

November 4, 2024 

## Before solving problems, 

`본 과제는 총` 18 `문항으로 각 문제당` 10 `점` ( `총` 180 `점` ) `입니다` . `각 문제에 대해서 적절한 답을작성하시면 됩니다` . `답안지는 수기로 작성된` ( `전자` ) `문서 혹은 그와 유사한 형태로 작성할 수 있으며` , `답안지만 최종 결과물로` LMS `에 제출하시면 됩니다` . `답안지 과제제출시스캔` , `촬영` , pdf `위에 재쓰기 등이 모두 가능하며` , `꼭 하나의 파일로 압축 혹은 합침하여 가상대학` (LMS) `에 제출하기 바랍니다` . `또한 다음과같이 파일명을 준수하기 바랍니다` . 

`파일명` : `학번` - `이름` . `확장자명` (pdf, zip `등` ), `예로` 2022100011- `아무개` .zip `답안지 양식은 다음과같이 학생정보와 답안` ( `대응하는 문제 번호를 포함` ) `을 기입할 수 있는 형태이면 됩니다` . 

( `주의사항` ) ChatGPT `등의 거대언어모델 사용` , `다른 학우에 의한 도움 등은 가능하지만` , `출처를 남기지 않는 경우는 과제점수는` 0 `점 처리함` . `수기형태로 작성된 문서가 아닌` hwp, word `등의 타이핑` (Typing) `을 통한 문서 작성은` 0 `점으로 처리함` . 

```
답안지양식
```

Class No.( `분반번호` ): Student ID( `학번` ): Student name( `이름` ): 

<u>Question</u> no. Answer 1.1 - 1) 1.1 - 2) 1.1 - 3) .... ... 

1 

## 1 Airline Routes and Strong Connections 

Scenario: An airline has a network of airports where each airport is a node, and each directed edge represents a one-way flight between airports. We are interested in understanding the network’s connectivity. 

### 1.1 Graph Connectivity 

The airline’s network consists of airports A, B, C, D, and E. There are direct flights as follows: 

A → B, B → C, C → D, D → E, and E → A 

1. Represent this network as a directed graph. 

2. Based on the structure, determine if the graph is strongly connected. Explain the conditions for strong connectivity in this context. 

3. Identify the strongly connected components in the graph, if any. 

### 1.2 Shortest Path Using BFS 

1. Suppose there are additional flights from A to D and from B to E. Use BFS to find the shortest path (in terms of the number of flights) from A to E. 

2. Explain each step and the rationale behind using BFS in this context. 

### 1.3 Transitive Closure and Reachability 

1. Using Warshall’s algorithm, find the transitive closure of the graph. Interpret the result to determine if every airport is reachable from every other airport. 

2. Explain how Warshall’s algorithm helps in identifying reachability within directed graphs like this one. 

## 2 Social Media Following 

Scenario: In a social media network, each user is a node, and directed edges represent ”follows” between users. We are interested in understanding relationships and possible recommendations. 

2 

### 2.1 Relation Properties and Graph Representation 

A small network includes the following users and follow relationships: 

- Alice follows Bob, 

- Bob follows Charlie, 

- Charlie follows Alice, 

- Alice also follows Dana. 

1. Represent this network as a directed graph. 

2. Discuss if the ”follow” relationship in this network is symmetric, antisymmetric, or neither. Justify your answer. 

3. Identify any cycles in the graph and explain what these cycles imply about the relationships among users. 

### 2.2 Transitive Follows and Recommendation System 

1. Using the transitive property, determine if it is possible for the system to recommend Charlie to Alice as someone she might want to follow. 

2. Describe how transitivity is applied here to generate potential follow recommendations. 

## 3 Company Email System 

Scenario: A company’s email system is represented as a graph, where employees are nodes, and directed edges indicate that one employee can email another. Each employee has permission to email every other employee, except for themselves. 

### 3.1 Graph Completeness 

Suppose the network includes employees E1, E2, E3, E4. An adjacency matrix represents whether an employee can email another (1 indicates permission, and 0 indicates no permission). 

1. Construct the adjacency matrix for this system. 

2. Based on the adjacency matrix, explain if this graph is complete. Define what makes a directed graph complete in this context. 

3 

- 3.2 Role of Reflexive and Symmetric Properties in the Email System 

   1. Describe whether this email system is reflexive, symmetric, or both. Explain why or why not, using definitions for reflexive and symmetric relations. 

   2. Explain the implications if the system allowed reflexivity (i.e., employees could email themselves). Would this affect the completeness of the graph? 

### 3.3 Graph Traversal for Information Flow 

Imagine employee E1 wants to send an announcement to all other employees through email chains, where each recipient must pass on the message. 

1. Discuss whether the system’s completeness allows for efficient information flow without any employee being left out. 

2. If one direct link (e.g., E1 to E2) is temporarily unavailable, explain how this might affect the connectivity of the system and whether all employees could still receive the message. 

4 

