### [Page 1]
2024-03-01
1
Spark: Resilient Distributed Datasets 
as Workflow System
빅데이터분석
천세진
동아대학교
컴퓨터AI공학부
2
Where is MapReduce Inefficient?


| Where is MapReduce Inefficient? |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 2 | 동아대학교 |


### [Page 2]
2024-03-01
2
동아대학교
컴퓨터AI공학부
Long pipelines sharing data
Interactive applications
Streaming applications
Iterative algorithms ( optimization problems )
3
Where is MapReduce Inefficient?
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
4
Spark의Big Idea


| Where is MapReduce Inefficient?  Long pipelines sharing data  Interactive applications  Streaming applications  Iterative algorithms ( optimization problems ) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 3 | 동아대학교 |

| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 4 | 동아대학교 |


### [Page 3]
2024-03-01
3
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
5
Spark의Big Idea
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
6
Spark의Big Idea


| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 5 | 동아대학교 |

| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 6 | 동아대학교 |


### [Page 4]
2024-03-01
4
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
7
Spark의Big Idea
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
Faster communication and I/O
On-the-fly 형태로데이터셋을rebuilding이가능함
disk에중간결과(Intermediate datasets)가저장되지않음
Only in-memory
8
Spark의Big Idea


| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 7 | 동아대학교 |

| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s).  Faster communication and I/O  On-the-fly 형태로 데이터셋을 rebuilding이 가능함  disk에 중간결과(Intermediate datasets)가저장되지 않음  Only in-memory |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 8 | 동아대학교 |


### [Page 5]
2024-03-01
5
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
9
Spark의Big Idea
Stable Storage
Other RDDs
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
10
Spark의Big Idea
map
filter
join


| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). Stable Storage Other RDDs |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 9 | 동아대학교 |

| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). map filter join |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 10 | 동아대학교 |


### [Page 6]
2024-03-01
6
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
11
Spark의Big Idea
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
12
Spark의Big Idea


| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 11 | 동아대학교 |

| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 12 | 동아대학교 |


### [Page 7]
2024-03-01
7
동아대학교
컴퓨터AI공학부
Resilient Distributed Datasets(RDDs)
Read-only  partitioned collection of records (like a DFS)
But, 어떻게데이터가생성되었는지에대한레코드를가짐
Combination of transformations from other dataset(s).
13
Spark의Big Idea
동아대학교
컴퓨터AI공학부
14
Transformations: RDD to RDD


| Spark의 Big Idea  Resilient Distributed Datasets(RDDs) Read-only partitioned collection of records (like a DFS)  But, 어떻게 데이터가생성되었는지에대한 레코드를가짐  Combination of transformations from other dataset(s). |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 13 | 동아대학교 |

| Transformations: RDD to RDD |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 14 | 동아대학교 |


### [Page 8]
2024-03-01
8
동아대학교
컴퓨터AI공학부
15
Transformations: RDD to Value Object, or Storage
동아대학교
컴퓨터AI공학부
filter, map, flatMap, reduceByKey, groupByKey
collect, count, take
16
Current Transformation and Actions
eager 와 lazy 실행의 설명


| Transformations: RDD to Value Object, or Storage eager 와 lazy 실행의 설명 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 15 | 동아대학교 |

| Current Transformation and Actions  filter, map, flatMap, reduceByKey, groupByKey  collect, count, take |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 16 | 동아대학교 |


### [Page 9]
2024-03-01
9
동아대학교
컴퓨터AI공학부
17
Example
동아대학교
컴퓨터AI공학부
18
Example


| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 17 | 동아대학교 |

| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 18 | 동아대학교 |


### [Page 10]
2024-03-01
10
동아대학교
컴퓨터AI공학부
19
Example
동아대학교
컴퓨터AI공학부
20
Example


| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 19 | 동아대학교 |

| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 20 | 동아대학교 |


### [Page 11]
2024-03-01
11
동아대학교
컴퓨터AI공학부
21
Example
동아대학교
컴퓨터AI공학부
22
Example


| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 21 | 동아대학교 |

| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 22 | 동아대학교 |


### [Page 12]
2024-03-01
12
동아대학교
컴퓨터AI공학부
23
Example
동아대학교
컴퓨터AI공학부
24
Example


| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 23 | 동아대학교 |

| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 24 | 동아대학교 |


### [Page 13]
2024-03-01
13
동아대학교
컴퓨터AI공학부
25
Example
동아대학교
컴퓨터AI공학부
More efficient failure recovery
More efficient grouping of tasks and scheduling
Integration of programming language features:
Loops (not a “cyclic” workflow system)
Function libraries
26
Workflow System의장점


| Example |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 25 | 동아대학교 |

| Workflow System의 장점  More efficient failure recovery  More efficient grouping of tasks and scheduling  Integration of programming language features:  Loops (not a “cyclic” workflow system)  Function libraries |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 26 | 동아대학교 |


### [Page 14]
2024-03-01
14
동아대학교
컴퓨터AI공학부
27
The Spark Programming Model
동아대학교
컴퓨터AI공학부
28
The Spark Programming Model


| The Spark Programming Model |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 27 | 동아대학교 |

| The Spark Programming Model |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 28 | 동아대학교 |


### [Page 15]
2024-03-01
15
동아대학교
컴퓨터AI공학부
29
The Spark Programming Model
동아대학교
컴퓨터AI공학부
30
Lazy Evaluation


| The Spark Programming Model |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 29 | 동아대학교 |

| Lazy Evaluation |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 30 | 동아대학교 |


### [Page 16]
2024-03-01
16
동아대학교
컴퓨터AI공학부
31
Lazy Evaluation
동아대학교
컴퓨터AI공학부
32
Broadcast Variables
rdd(T) -> rdd(T)


| Lazy Evaluation rdd(T) -> rdd(T) |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 31 | 동아대학교 |

| Broadcast Variables |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 32 | 동아대학교 |


### [Page 17]
2024-03-01
17
동아대학교
컴퓨터AI공학부
33
Broadcast Variables
동아대학교
컴퓨터AI공학부
34
Accumulators


| Broadcast Variables |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 33 | 동아대학교 |

| Accumulators |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 34 | 동아대학교 |


### [Page 18]
2024-03-01
18
동아대학교
컴퓨터AI공학부
35
Accumulators
동아대학교
컴퓨터AI공학부
36
Spark System: Review


| Accumulators |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 35 | 동아대학교 |

| Spark System: Review |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 36 | 동아대학교 |


### [Page 19]
2024-03-01
19
동아대학교
컴퓨터AI공학부
37
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
38
Spark System: Hierarchy


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 37 | 동아대학교 |

| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 38 | 동아대학교 |


### [Page 20]
2024-03-01
20
동아대학교
컴퓨터AI공학부
39
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
40
Spark System: Hierarchy


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 39 | 동아대학교 |

| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 40 | 동아대학교 |


### [Page 21]
2024-03-01
21
동아대학교
컴퓨터AI공학부
41
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
42
Spark System: Hierarchy


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 41 | 동아대학교 |

| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 42 | 동아대학교 |


### [Page 22]
2024-03-01
22
동아대학교
컴퓨터AI공학부
43
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
44
Spark System: Hierarchy


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 43 | 동아대학교 |

| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 44 | 동아대학교 |


### [Page 23]
2024-03-01
23
동아대학교
컴퓨터AI공학부
45
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
46
Spark System: Hierarchy


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 45 | 동아대학교 |

| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 46 | 동아대학교 |


### [Page 24]
2024-03-01
24
동아대학교
컴퓨터AI공학부
47
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
48
Spark System: Hierarchy


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 47 | 동아대학교 |

| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 48 | 동아대학교 |


### [Page 25]
2024-03-01
25
동아대학교
컴퓨터AI공학부
49
Spark System: Hierarchy
동아대학교
컴퓨터AI공학부
50
MapReduce or Spark?


| Spark System: Hierarchy |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 49 | 동아대학교 |

| MapReduce or Spark? |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 50 | 동아대학교 |


### [Page 26]
2024-03-01
26
동아대학교
컴퓨터AI공학부
51
MapReduce or Spark?


| MapReduce or Spark? |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 51 | 동아대학교 |
