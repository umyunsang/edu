### [Page 1]
Discrete Mathematics: Homework I
Prof. Chun
November 4, 2024
Before solving problems,
본과제는총18문항으로각문제당10점(총180점)입니다. 각문제에대해서
적절한답을작성하시면됩니다. 답안지는수기로작성된(전자)문서혹은그와
유사한형태로작성할수있으며, 답안지만최종결과물로LMS에제출하시면
됩니다. 답안지과제제출시스캔, 촬영, pdf위에재쓰기등이모두가능하며,
꼭하나의파일로압축혹은합침하여가상대학(LMS)에제출하기바랍니다.
또한다음과같이파일명을준수하기바랍니다.
파일명: 학번-이름.확장자명(pdf, zip 등), 예로2022100011-아무개.zip
답안지양식은다음과같이학생정보와답안(대응하는문제번호를포함)
을기입할수있는형태이면됩니다.
(주의사항) ChatGPT 등의거대언어모델사용, 다른학우에의한도움등은
가능하지만, 출처를남기지않는경우는과제점수는0점처리함. 수기형태로
작성된문서가아닌hwp, word 등의타이핑(Typing)을통한문서작성은0
점으로처리함.
답안지양식
Class No.(분반번호):
Student ID(학번):
Student name(이름):
Question no.
Answer
1.1 - 1)
1.1 - 2)
1.1 - 3)
....
...
1


| Question no. | Answer |
| --- | --- |
| 1.1 - 1) |  |
| 1.1 - 2) |  |
| 1.1 - 3) |  |
| .... | ... |


### [Page 2]
1
Airline Routes and Strong Connections
Scenario: An airline has a network of airports where each airport is a node,
and each directed edge represents a one-way ﬂight between airports. We are
interested in understanding the network’s connectivity.
1.1
Graph Connectivity
The airline’s network consists of airports A, B, C, D, and E. There are direct
ﬂights as follows:
A →B, B →C, C →D, D →E, and E →A
1. Represent this network as a directed graph.
2. Based on the structure, determine if the graph is strongly connected.
Explain the conditions for strong connectivity in this context.
3. Identify the strongly connected components in the graph, if any.
1.2
Shortest Path Using BFS
1. Suppose there are additional ﬂights from A to D and from B to E. Use
BFS to ﬁnd the shortest path (in terms of the number of ﬂights) from
A to E.
2. Explain each step and the rationale behind using BFS in this context.
1.3
Transitive Closure and Reachability
1. Using Warshall’s algorithm, ﬁnd the transitive closure of the graph.
Interpret the result to determine if every airport is reachable from
every other airport.
2. Explain how Warshall’s algorithm helps in identifying reachability
within directed graphs like this one.
2
Social Media Following
Scenario: In a social media network, each user is a node, and directed
edges represent ”follows” between users. We are interested in understanding
relationships and possible recommendations.
2



### [Page 3]
2.1
Relation Properties and Graph Representation
A small network includes the following users and follow relationships:
• Alice follows Bob,
• Bob follows Charlie,
• Charlie follows Alice,
• Alice also follows Dana.
1. Represent this network as a directed graph.
2. Discuss if the ”follow” relationship in this network is symmetric, anti-
symmetric, or neither. Justify your answer.
3. Identify any cycles in the graph and explain what these cycles imply
about the relationships among users.
2.2
Transitive Follows and Recommendation System
1. Using the transitive property, determine if it is possible for the system
to recommend Charlie to Alice as someone she might want to follow.
2. Describe how transitivity is applied here to generate potential follow
recommendations.
3
Company Email System
Scenario: A company’s email system is represented as a graph, where em-
ployees are nodes, and directed edges indicate that one employee can email
another. Each employee has permission to email every other employee, ex-
cept for themselves.
3.1
Graph Completeness
Suppose the network includes employees E1, E2, E3, E4. An adjacency ma-
trix represents whether an employee can email another (1 indicates permis-
sion, and 0 indicates no permission).
1. Construct the adjacency matrix for this system.
2. Based on the adjacency matrix, explain if this graph is complete. De-
ﬁne what makes a directed graph complete in this context.
3



### [Page 4]
3.2
Role of Reﬂexive and Symmetric Properties in the Email
System
1. Describe whether this email system is reﬂexive, symmetric, or both.
Explain why or why not, using deﬁnitions for reﬂexive and symmetric
relations.
2. Explain the implications if the system allowed reﬂexivity (i.e., employ-
ees could email themselves). Would this aﬀect the completeness of the
graph?
3.3
Graph Traversal for Information Flow
Imagine employee E1 wants to send an announcement to all other employees
through email chains, where each recipient must pass on the message.
1. Discuss whether the system’s completeness allows for eﬃcient infor-
mation ﬂow without any employee being left out.
2. If one direct link (e.g., E1 to E2) is temporarily unavailable, explain
how this might aﬀect the connectivity of the system and whether all
employees could still receive the message.
4

