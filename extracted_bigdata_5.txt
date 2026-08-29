### [Page 1]
2024-03-01
1
History & Essentials
Apache Spark
Why do we learn? 
Apache Spark



### [Page 2]
2024-03-01
2
동아대학교
컴퓨터AI공학부
3
배우면좋은이유? 빅데이터엔지니어의핵심
Spark is more for mainstream developers
동아대학교
컴퓨터AI공학부
4
배우면좋은이유? 빅데이터엔지니어의핵심


| 배우면 좋은 이유? 빅데이터 엔지니어의 핵심 Spark is more for mainstream developers |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 3 | 동아대학교 |

| 배우면 좋은 이유? 빅데이터 엔지니어의 핵심 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 4 | 동아대학교 |


### [Page 3]
2024-03-01
3
A Brief History
Apache Spark
동아대학교
컴퓨터AI공학부
6
A Brief History


| A Brief History |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 6 | 동아대학교 |


### [Page 4]
2024-03-01
4
동아대학교
컴퓨터AI공학부
7
Spark의방향
동아대학교
컴퓨터AI공학부
8
Spark의방향


| Spark의 방향 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 7 | 동아대학교 |

| Spark의 방향 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 8 | 동아대학교 |


### [Page 5]
2024-03-01
5
동아대학교
컴퓨터AI공학부
9
Spark의방향
동아대학교
컴퓨터AI공학부
10
Spark의방향


| Spark의 방향 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 9 | 동아대학교 |

| Spark의 방향 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 10 | 동아대학교 |


### [Page 6]
2024-03-01
6
동아대학교
컴퓨터AI공학부
11
Spark의방향
Essentials
Apache Spark


| Spark의 방향 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 11 | 동아대학교 |


### [Page 7]
2024-03-01
7
동아대학교
컴퓨터AI공학부
Entry point to Spark
Creation
Spark RDD
Accumulators
Broadcast variables
Configuration
appName
Master URL
13
SparkContext
Source: spark.apache.org
동아대학교
컴퓨터AI공학부
14
SparkContext


| SparkContext  Entry point to Spark  Creation  Spark RDD  Accumulators  Broadcast variables  Configuration  appName  Master URL Source: spark.apache.org |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 13 | 동아대학교 |

| SparkContext |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 14 | 동아대학교 |


### [Page 8]
2024-03-01
8
동아대학교
컴퓨터AI공학부
15
SparkContext
동아대학교
컴퓨터AI공학부
16
Master


| SparkContext |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 15 | 동아대학교 |

| Master |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 16 | 동아대학교 |


### [Page 9]
2024-03-01
9
동아대학교
컴퓨터AI공학부
1. cluster manager에연결
응용간에리소스를할당
2.  cluster nodes 위에executor를획득
Worker는computation 실행과data 저장을담당
3. executor에app code를전송
4. 실행하려는executor를위한task 전송
17
Master
동아대학교
컴퓨터AI공학부
18
RDD
Resilient Distributed Datasets (RDD) 는Spark 기본추상화
병렬처리위에서운영될수있는
데이터요소들의Fault-tolerant collection
두가지타입
Parallelized collections: Scala collection을가지고
병렬적으로해당collection 위에서function을실행
Hadoop datasets: Hadoop 내file의record에
function을적용하여실행


| Master  1. cluster manager에 연결  응용 간에 리소스를 할당  2. cluster nodes 위에 executor를 획득  Worker는 computation 실행과 data 저장을 담당  3. executor에 app code를 전송  4. 실행하려는 executor를 위한 task 전송 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 17 | 동아대학교 |

| RDD  Resilient Distributed Datasets (RDD) 는 Spark 기본 추상화  병렬 처리 위에서 운영될 수 있는 데이터 요소들의 Fault-tolerant collection  두가지 타입  Parallelized collections: Scala collection을 가지고 병렬적으로 해당 collection 위에서 function을 실행  Hadoop datasets: Hadoop 내 file의 record에 function을 적용하여 실행 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 18 | 동아대학교 |


### [Page 10]
2024-03-01
10
동아대학교
컴퓨터AI공학부
19
RDD
Two types of operations on RDDs:
Transformations and actions
Transformations are lazy
(not computed immediately)
The transformed RDD gets recomputed
when an action is run on it (default)
However, an RDD can be persisted into storage
in memory or disk
동아대학교
컴퓨터AI공학부
20
RDD


| RDD  Two types of operations on RDDs:  Transformations and actions  Transformations are lazy  (not computed immediately)  The transformed RDD gets recomputed when an action is run on it (default)  However, an RDD can be persisted into storage in memory or disk |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 19 | 동아대학교 |

| RDD |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 20 | 동아대학교 |


### [Page 11]
2024-03-01
11
동아대학교
컴퓨터AI공학부
21
RDD
동아대학교
컴퓨터AI공학부
22
RDD


| RDD |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 21 | 동아대학교 |

| RDD |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 22 | 동아대학교 |


### [Page 12]
2024-03-01
12
동아대학교
컴퓨터AI공학부
23
Transformations
동아대학교
컴퓨터AI공학부
24
Transformations


| Transformations |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 23 | 동아대학교 |

| Transformations |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 24 | 동아대학교 |


### [Page 13]
2024-03-01
13
동아대학교
컴퓨터AI공학부
25
Transformations
동아대학교
컴퓨터AI공학부
26
Transformations


| Transformations |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 25 | 동아대학교 |

| Transformations |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 26 | 동아대학교 |


### [Page 14]
2024-03-01
14
동아대학교
컴퓨터AI공학부
27
Transformations
동아대학교
컴퓨터AI공학부
28
Transformations
Map() vs. flatMap()간결과비교


| Transformations |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 27 | 동아대학교 |

| Transformations Map() vs. flatMap()간 결과 비교 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 28 | 동아대학교 |


### [Page 15]
2024-03-01
15
동아대학교
컴퓨터AI공학부
29
Actions
동아대학교
컴퓨터AI공학부
30
Actions


| Actions |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 29 | 동아대학교 |

| Actions |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 30 | 동아대학교 |


### [Page 16]
2024-03-01
16
동아대학교
컴퓨터AI공학부
31
Actions
동아대학교
컴퓨터AI공학부
Spark는연산간내데이터셋을인메모리에persist(혹은
cache)함]
각노드는데이터셋의일부를메모리에저장
계산하고다른action에대해재사용함
다른future action에대해10x배이상빨라지게함
Cache는fault-tolerant
RDD가손실되더라도, transformation을사용하여
자동으로재계산함
32
Persistence


| Actions |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 31 | 동아대학교 |

| Persistence  Spark는 연산간 내 데이터셋을 인메모리에 persist(혹은 cache)함]  각 노드는 데이터셋의 일부를 메모리에 저장  계산하고 다른 action에 대해 재사용함  다른 future action에 대해 10x배 이상 빨라지게 함  Cache는 fault-tolerant  RDD가 손실되더라도, transformation을 사용하여 자동으로 재계산함 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 32 | 동아대학교 |


### [Page 17]
2024-03-01
17
동아대학교
컴퓨터AI공학부
33
Persistence
동아대학교
컴퓨터AI공학부
34
Persistence


| Persistence |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 33 | 동아대학교 |

| Persistence |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 34 | 동아대학교 |


### [Page 18]
2024-03-01
18
동아대학교
컴퓨터AI공학부
35
Broadcast variables
프로그래머에게각머신(machine)에
cached read-only variables을유지
예로, large 입력데이터셋의copy를모든노드에
효율적으로주기위해
Spark는효율적인broadcast algorithm을사용하여
broadcast 변수를배포함
커뮤니케이션비용을줄임
동아대학교
컴퓨터AI공학부
36
Broadcast variables


| Broadcast variables  프로그래머에게 각 머신(machine)에 cached read-only variables을 유지  예로, large 입력 데이터셋의 copy를 모든 노드에 효율적으로 주기위해  Spark는 효율적인 broadcast algorithm을 사용하여 broadcast 변수를 배포함  커뮤니케이션 비용을 줄임 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 35 | 동아대학교 |

| Broadcast variables |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 36 | 동아대학교 |


### [Page 19]
2024-03-01
19
동아대학교
컴퓨터AI공학부
37
Accumulators
Associative operation을통해추가될수있는변수
병렬적으로, counters과sums을구현하기위해주로사용됨
기본적으로, numeric type과standard mutable collection을
지원함. 또한, 프로그래머가새로운타입을위해확장할수
있음
Driver 프로그램만accumulator의값을읽을수(read)가있
음
동아대학교
컴퓨터AI공학부
38
Accumulators


| Accumulators  Associative operation을 통해 추가될 수 있는 변수  병렬적으로, counters과 sums을 구현하기 위해 주로 사용됨  기본적으로, numeric type과 standard mutable collection을 지원함. 또한, 프로그래머가 새로운 타입을 위해 확장할 수 있음  Driver 프로그램만 accumulator의 값을 읽을 수(read)가 있 음 |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 37 | 동아대학교 |

| Accumulators |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 38 | 동아대학교 |


### [Page 20]
2024-03-01
20
동아대학교
컴퓨터AI공학부
39
Accumulators


| Accumulators |  |  |
| --- | --- | --- |
| 컴퓨터AI공학부 | 39 | 동아대학교 |
