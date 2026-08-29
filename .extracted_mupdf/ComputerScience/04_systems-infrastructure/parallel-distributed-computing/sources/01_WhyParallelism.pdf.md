## --- [Page 1] ---
분산처리

1

Computer & Ai

Department of Computer Engineering

Parallel Computing Stanford CS149, Fall 2024 수업 자료 참고

## --- [Page 2] ---
왜 병렬 처리일까요?
왜 효율성일까요?

2

성능과 효율성을 
중요하게 고려함
이를 얻기 위해 여러 처

리 요소를 사용할 것

## --- [Page 3] ---
Mandelbrot Fractal

3

## --- [Page 4] ---
•

•

•
➔ 따라서 communication 시간을 줄이면 성능이 자연스레 오른다.

Communication limited the maximum speedup achieve

•
➔ task 를 잘 배분하면 성능이 오른다.

## --- [Page 5] ---
▪

▪

## --- [Page 6] ---
⚫

➢
➢

⚫

➢

➢

## --- [Page 7] ---
▪

▪

▪

▪

joonojoono.tistory.com/30

## --- [Page 8] ---

## --- [Page 9] ---
⚫싱글 스레드 CPU 성능 2배 증가 ~ 18개월마다 증가

⚫시사점: 코드 병렬화 작업은 종종 시간 대비 가치가 없었습니다.

➢
- 소프트웨어 개발자가 아무것도 하지 않아도 내년에는 코드가 더 빨라집니다. 우와!

## --- [Page 10] ---
역사적 맥락: 병렬 처리를 회피한 이유는 무엇인가요?

⚫15년 전까지만 해도 프로세서 성능 개선의 두 가지 중요한 이유는 다음과 같습니다.

1. 명령어 수준의 병렬 처리(instruction-level parallelism )(슈퍼스칼라 실행) 활용

2. CPU 클럭 주파수 증가

Instruction level parallelism (ILP) example

•
ILP

➢한 번에 병렬 처리할 수 있는 instructions 을 구분하기 위한 level 값

•
왼쪽과 같은 예제가 있다고 하자. 
•
총 5개의 instructions 로 이루어져있음.

## --- [Page 11] ---

## --- [Page 12] ---
⚫프로그램은 프로세서 명령어의  리스트일 뿐

## --- [Page 13] ---
⚫프로그램은 프로세서 명령어의  리스트일 뿐

## --- [Page 14] ---

## --- [Page 15] ---
프로세서는 명령어를 실행

Very Simple Processor

•
다음에 실행할 명령어를 결정.

•
실행 단위: 명령어로 기술된 작업을 수행하며, 프로세서 레지
스터 또는 컴퓨터 메모리의 값을 수정할 수 있음

•
레지스터: 프로그램 상태 유지: 연산에 입력 및 출력으로 사용
되는 변수의 값을 저장.

## --- [Page 16] ---
한 가지 예시: 숫자 두 개 더하기

Very Simple Processor

Step 1:
▪프로세서가 메모리에서 다음 프로그램 명령을 가져옵니다(“프로세서가 다음에

수행해야 할 작업 파악”).

“레지스터 R0의 내용을 레지스터 R1의 내용에 더하고 그 결과를 레지스터 R0에 넣어주세요.”

Step 2:
▪레지스터에서 연산 입력을 가져옵니다.
▪실행 유닛에 입력된 R0의 내용:
▪실행 유닛에 입력된 R1의 내용:

Step 3:
▪더하기 작업을 수행
▪실행 단위가 산술을 수행하면 결과는

## --- [Page 17] ---
한 가지 예시: 숫자 두 개 더하기

Very Simple Processor

Step 1:
▪프로세서가 메모리에서 다음 프로그램 명령을 가져옵니다(“프로세서가 다음에

수행해야 할 작업 파악”).

“레지스터 R0의 내용을 레지스터 R1의 내용에 더하고 그 결과를 레지스터 R0에 넣어주세요.”

Step 2:
▪레지스터에서 연산 입력을 가져옵니다.
▪실행 유닛에 입력된 R0의 내용:
▪실행 유닛에 입력된 R1의 내용:

Step 3:
▪더하기 작업을 수행
▪실행 단위가 산술을 수행하면 결과는

Step 4:
▪결과 
을 저장하고 R0에 Register(등록).

## --- [Page 18] ---
프로그램 실행

아주 간단한 프로세서: 클럭당 하나의 명령어 실행

## --- [Page 19] ---
프로그램 실행

아주 간단한 프로세서: 클럭당 하나의 명령어 실행

## --- [Page 20] ---
프로그램 실행

아주 간단한 프로세서: 클럭당 하나의 명령어 실행

## --- [Page 21] ---
프로그램 실행

아주 간단한 프로세서: 클럭당 하나의 명령어 실행

## --- [Page 22] ---
컴퓨터 작동 원리 살펴보기...

• 컴퓨터 프로그램이란 무엇인가요? (프로세서의 관점에서)

➢실행할 명령어 목록!

• 명령어란 무엇인가요?

➢프로세서가 수행해야 할 작업을 기술한 것.

➢명령어를 실행하면 일반적으로 컴퓨터의 상태가 변경됨.

• 컴퓨터의 “상태(state)”란 무엇을 의미하나요?

➢프로세서의 레지스터나 메모리에 저장된 프로그램 데이터의 값을 말함.

## --- [Page 23] ---
아주 간단한 코드를 예로 들어 보자

• 이 프로그램에는 다섯 개의 명령어가 있으
므로 실행하는 데 다섯 번의 클럭이 걸리겠
죠?
• 더 좋은 방법이  있을까요?

## --- [Page 24] ---
아주 간단한 코드를 예로 들어 보자

## --- [Page 25] ---
아주 간단한 코드를 예로 들어 보자

## --- [Page 26] ---
아주 간단한 코드를 예로 들어 보자

## --- [Page 27] ---

## --- [Page 28] ---
Instruction level parallelism (ILP) example

•ILP 를 계산하여 같은 ILP를 갖는 연산들은 한 번에 같이 수행할 수 있지만 그렇지 않으면 기다려야한다.

## --- [Page 29] ---
▪이 예제에서는 명령어 1, 2, 3을 프로그램 정확성에 영향을 주지 않고 병렬로 실행 가능.
(슈퍼스칼라 프로세서상에서는 종속성이 존재하지 않는다고 판단).

▪하지만 명령어 4는 명령어 1과 2 다음에 실행되어야 함.

▪그리고 명령어 5는 명령어 4 이후에 실행되어야 함.

서로 독립적인 instructions 를 자동으로 찾아서 이를 
multi-processes 에 잘 배분하여 수행함.

* 또는 컴파일 시 컴파일러가 독립적인 명령어를 컴파일하고 컴파일된 
바이너리에 종속성을 명시적으로 인코딩함.

슈퍼스칼라 실행: 프로세서가 명령어 시퀀스에서 독
립적인 명령어를 자동으로 찾아서 시퀀스에서 독립적
인 명령어를 생성하고 여러 실행 단위에서 병렬로 실
행할 수 있습니다.

## --- [Page 30] ---
이 프로세서는 클럭당 최대 2개의 명령어를 디코딩하고 실행가능

## --- [Page 31] ---

## --- [Page 32] ---

## --- [Page 33] ---
•

## --- [Page 34] ---

## --- [Page 35] ---
•

## --- [Page 36] ---

## --- [Page 37] ---
•

•
➢

## --- [Page 38] ---

## --- [Page 39] ---

## --- [Page 40] ---

## --- [Page 41] ---

## --- [Page 42] ---

## --- [Page 43] ---

## --- [Page 44] ---

## --- [Page 45] ---
모든

## --- [Page 46] ---

## --- [Page 47] ---
• 일정 시간 동안 더 높은 성능으로 실행

• 더 오랜 시간 동안 충분한 성능으로 실행

전력 = 열
칩이 너무 뜨거워지면 반드시 클럭을 낮추
어 식혀야 함

전원 = 배터리
긴 배터리 수명은 모바일 디바이스에서 바
람직한 기능

## --- [Page 48] ---

## --- [Page 49] ---
•
모바일 시스템에서는 특수 처리(Specialized processing) 가 보편화되고 있음

## --- [Page 50] ---
• 고효율 달성은 이 강의의 핵심 주제입니다.

• 최신 시스템은 많은 처리 장치를 사용할 뿐만 아니라 특수 처리 장치를 활용하여 높은 수준의 전력 효율을 달성
하는 방법에 대해 학습

Google TPU pods
TPU = Tensor Processing Unit: specialized processor for ML computations

Image Credit: TechInsights Inc.


| Specialization |  | for |  | datacenter | - | scale |  | applications |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |


## --- [Page 51] ---

## --- [Page 52] ---

## --- [Page 53] ---

## --- [Page 54] ---
프로그램의 메모리 주소 공간

• 컴퓨터의 메모리는 바이트 배열로 구성

• 각 바이트는 메모리 내 '주소'로 식별
(이 배열에서의 위치)로 식별.
(메모리는 바이트 주소 지정이 가능하다고 가정.)

“0x8 주소에 저장된 바이트의 값은 32입니다.”

“주소 0x10(16)에 저장된 바이트의 값은 128입니다.”

오른쪽 그림에서 프로그램의 메모리 주소 공간은 32바이트 크기
 (따라서 유효한 주소 범위는 0x0에서 0x1F까지).

## --- [Page 55] ---
Load: 메모리 내용(the contents of memory)에 액세스하기 위한 명령어

“레지스터 R2에 저장된 주소부터 시작하여 4바이트 값을 메모리에 로드하
고 이 값을 레지스터 R0에 넣으십시오

## --- [Page 56] ---
Terminology

• Memory access latency
➢메모리 시스템이 프로세서에 데이터를 제공하는 데 걸리는 시간.
➢Example: 100 clock cycles, 100 nsec

## --- [Page 57] ---
Stalls

• 프로세서는 다음 명령어가 아직 완료되지 않은 이전 명령어에 
의존하기 때문에 명령어 스트림에서 다음 명령어를 실행할 수 
없을 때 “멈춤(Stalls)”(진행이 불가능)됩니다.

• 메모리 액세스는 지연(Stalls)의 주요 원인

종속성(Dependency): mem[r2] 및 mem[r3]의 데이터
가 메모리에서 로드될 때까지 'add' 명령을 실행할 수 없음

• Memory access times ~ 100’s of cycles

➢메모리 '액세스 시간'은 지연 시간을 측정하는 척도

## --- [Page 58] ---
Stalls

• 프로세서는 다음 명령어가 아직 완료되지 않은 이전 명령어에 
의존하기 때문에 명령어 스트림에서 다음 명령어를 실행할 수 
없을 때 “멈춤(Stalls)”(진행이 불가능)됩니다.

CPU가 명령어를 처리하는 과정에서 특정 이유로 인해 일시적으로 멈추거나 지연되는 현상

• 메모리 액세스는 지연(Stalls)의 주요 원인

종속성(Dependency): mem[r2] 및 mem[r3]의 데이터
가 메모리에서 로드될 때까지 'add' 명령을 실행할 수 없음

• Memory access times ~ 100’s of cycles

➢메모리 '액세스 시간'은 지연 시간을 측정하는 척도

## --- [Page 59] ---
What are caches?

• 리콜 메모리는 값의 배열일 뿐

• 그리고 프로세서에는 메모리에서 레지스터로 데이터를 
이동(로드)하고 레지스터에서 메모리로 데이터를 저장
(스토어)하는 명령어가 있음

## --- [Page 60] ---
What are caches?

•
캐시는 프로그램의 출력에는 영향을 주지 않고 성능에만 영향을 주는 하드웨어 구현 세부 사항.
•
캐시는 메모리에 값의  subset 사본을 유지하는 온칩 스토리지.
•
Address(주소)가 “캐시에” 저장되면 프로세서는 데이터가 DRAM에만 있는 경우보다 더 빠르게 이 주소로 
로드/저장할 수 있음.
•
캐시는 '캐시 라인'이라는 세분화(granularity)된 단위로 작동.

- Has a capacity of 2 lines
- Each line holds 4 bytes of data

## --- [Page 61] ---
프로세서는 어떤 데이터를 캐시에 보관할지 어떻게 결정하나요?

•
이 강좌의 범위를 벗어나지만 다음 용어를 구글에서 검색해 보시기 바랍니다...

- 직접 매핑 캐시
- 집합 연관 캐시
- 캐시 라인

•
지금은 N바이트 크기의 캐시가 마지막으로 액세스한 N개의 주소에 대한 값을 저장한다고 가정합니다. -
LRU 교체 정책(“가장 최근에 사용된”) - 새 데이터를 위한 공간을 확보하기 위해 가장 오래 전에 액세스한 데
이터를 캐시에 있는 데이터를 버립니다.

## --- [Page 62] ---
Cache example 1

이 시퀀스에는 두 가지 형태의 “데이터 로캘리티”가 있음:

①
공간적 위치: 캐시 라인에 데이터를 로드하면 같은 
라인의 다른 주소에 대한 후속 액세스에 필요한 데
이터를 '미리 로드'하여 캐시 히트로 이어짐.
②
시간적 위치: 동일한 주소에 반복적으로 액세스하면 
히트가 발생.

## --- [Page 63] ---
Cache example 2

## --- [Page 64] ---
캐시로 지연 시간 감소(메모리 액세스 지연 시간 감소)

프로세서는 캐시에 상주하는 데이터에 액세스할 때 효율적으로 실행됩니다.
캐시는 프로세서가 최근에 액세스한 데이터에 액세스할 때 메모리 액세스 대기 시간을 줄입니다. 메모리 
액세스 지연을 줄입니다! *

## --- [Page 65] ---
최신 컴퓨터에서 선형 메모리 주소 공간 추상화를 구현하는 것은 복잡함

“주소 X에 저장된 값을 레지스터 R0에 로드하라"는 명령에는 여러 데이터 캐시
에 의한 복잡한 연산 시퀀스와 DRAM에 대한 액세스가 포함될 수 있음.

## --- [Page 66] ---
Data access times(Kaby Lake CPU)

## --- [Page 67] ---
데이터 이동에는 높은 에너지 비용이 소요됨

•
최신 시스템 설계의 경험 법칙: 항상 컴퓨터에서 데이터 이동량을 줄이려고 노력

•
“야구장” 숫자

Integer op: ~ 1 pJ *
- Floating point op: ~20 pJ *
- Reading 64 bits from small local SRAM (1mm away on chip): ~ 26 pJ
- Reading 64 bits from low power mobile DRAM (LPDDR): ~1200 pJ

•
Implications

- Reading 10 GB/sec from memory: ~1.6 watts
- Entire power budget for mobile GPU: ~1 watt

(remember phone is also running CPU, display, radios, etc.)
- iPhone 6 battery: ~7 watt-hours (note: my Macbook Pro laptop: 99 watt-hour battery)
- Exploiting locality matters!!!

[Sources: Bill Dally (NVIDIA), Tom Olson (ARM)]

*명령어 디코딩, 레지스터에서 데이터 로드 등의 오버헤드를 계산하지 않고 논리적 연산만 수행하는 데 드는 비용입니다.

## --- [Page 68] ---
데이터 활용 관점에서 모
든 분야의 경쟁 영역이 변
화

데이터수집➔해석➔처
리 사이클

2) IT기술로 새로운 비즈니스사이클 출현

1. 왜 병렬(분산)처리 인가

## --- [Page 69] ---
3) 4차 산업 혁명의 핵심 기술: Linux 기반으로 운영

69

IOT Computing
Cloud Computing
AI Computing

1. 왜 병렬(분산)처리 인가

## --- [Page 70] ---
70

IOT Computing
• ARM은 (Advanced RISC Machine) 약자이
며1985년 영국 캠브릿지 대학의 연구진들
이 시작한 벤처 기업으로 시작한 회사.

• ARM은 반도체 회사지만 반도체를 직접 만
들지 않는 팸리스 회사이며, MCU의 아키텍
처 개발만 하는 전문 회사.

• 타 반도체 회사에서ARM IP(아키텍처)를 갖
고 자신만의 반도체를 만드는 회사가 대표
적으로
TI, STM, 삼성, 프리스 케일,
Nvidia, 퀄컴 등이 있음.

• 그리고, ARM는2016년에 일본 소프트뱅크
에서36조원 전액 현금으로 인수한 회사 .

• 통계적으로 전세계 스마트폰의90% 이상
이ARM 반도체로 만들어져 있음.

• 스마트폰 뿐만 아니라, 태블릿, 스마트 와
치, 저장장치 컨트롤러, 차량용 메인 컨트롤
러, 무선통신기기 등 다양한 사업군의 기기
에서ARM 반도체로 사용

• 다양한 산업군에 사용되고 있는ARM
Architecture 의 종류로는
Cortex
A,
Cortex R, Cortex M 포트폴리오로 갖고 있
음.

Cortex-M를 구동시킬 수 있는OS
가Mbed OS 입니다.

1. 왜 병렬(분산)처리 인가

3) 4차 산업 혁명의 핵심 기술: Linux 기반으로 운영
ARM

## --- [Page 71] ---
Cloud Computing

가상화

클라우드

가상화는 단일한 물리 하드웨어 시스템에서 여러 시뮬레이션 환경이나 
전용 리소스를 생성할 수 있는 기술

클라우드는 네트워크 전체에서 확장 가능한 리소스를 추상화하고 풀링
하는 IT 환경

• 가상화는 하이퍼바이저라 불리는 소프트웨어가 하드웨어에 직접 연결되며 1
개의 시스템을가상 머신(VM)이라는 별도의 고유하고 안전한 환경으로 분할. 
• 이러한 VM은 하이퍼바이저의 기능을 사용하여 머신의 리소스를 하드웨어에
서 분리한 후 적절하게 배포

1. 왜 병렬(분산)처리 인가

3) 4차 산업 혁명의 핵심 기술: Linux 기반으로 운영

## --- [Page 72] ---
AI Computing

추론

학습

리눅스(RHEL)
TOP 1: Fugaku

Linux OS

1. 왜 병렬(분산)처리 인가

3) 4차 산업 혁명의 핵심 기술: Linux 기반으로 운영

## --- [Page 73] ---
활용 영역 확대

1. 클라우드 비용
2. 대기업의 리눅스 채택
3. 사물인터넷
(Internet of Things)

4. 보안 문제
(Security concerns)

5. 비용 민감도
(Cost Sensitivity)
미래형 운영체제?

U2L이란유닉스 플랫폼 환경(하드웨어, OS, DBMS, 미들웨어, 애플
리케이션 등)을 리눅스 환경으로 마이그레이션(Migration)하는 방
법론

1. 왜 병렬(분산)처리 인가

3) 4차 산업 혁명의 핵심 기술: Linux 기반으로 운영

## --- [Page 74] ---
2) UNIX일화

켄톰프슨
데니스 리치

1. 왜 병렬(분산)처리 인가

## --- [Page 75] ---
2) UNIX / Linux 역사

▪
UNIX 개발

•
1969년 AT&T Bell Labs, Ken Thompson, Dennis Ritchie, 
Douglas Mcllroy, Brian Kernighan( Unics )

•
Multics (Multiplexed Information and Computing 
Service) 프로젝트에서 파생

•
Open System : License with source

▪
개발 의도

•
Portable

•
Multi-Tasking

•
Multi-User

•
Time-Sharing

•
Network 와 Security 개념은 없었다.

1. 왜 병렬(분산)처리 인가

## --- [Page 76] ---
프로세스(process) 관리

프로세스(process)란

## --- [Page 77] ---
•
실행중인 프로그램을 프로세스(process)라고 부른다. 
•
각 프로세스는 유일한 프로세스 번호 PID를 갖는다.
•
각 프로세스는 부모 프로세스에 의해 생성된다.

✓하나의 프로세스에서 다수의 프로세스(프로그램)이 생성되는 경우도 많이 있음(네트워크 프로그램)
✓Linux 데스크탑 환경에서는 100개 이상의 프로세스가 동작하고 있음.

프로세스(process)

프로세스(process) 관리

[Memo]
•
Linux 커널은 Linux 운영 체
제(OS)의 주요 구성 요소이
며 컴퓨터 하드웨어와 프로
세스를 잇는 핵심 인터페이
스임

## --- [Page 78] ---
프로세스(process)

리눅스의 프로세스(process) 관리

## --- [Page 79] ---
프로세스(process) 관리

프로세스(process)란

new: 프로세스가 만들어지는 과정의 상태
terminated:  프로세스가 다 수행되어서 종료할 때 생기는 상
태
이 외의 상태 3개(running, waiting, ready)가 돌아 가면서 프
로세스가 수행 
running: CPU에서 수행되고 있는 상태
Ready: CPU에서 언제든지 수행할 수 있도록 대기하고 있는 상태
waiting: I/O나 다른 이벤트가 발생하기를 기다리고 있는 상태

## --- [Page 80] ---
프로세스(process)란

프로세스(process) 관리

사전적 의미

•“컴퓨터에서 연속적으로 실행되고 있는 컴퓨터 프로그램”메모리에 올라와 실행되고 있는 프로그램의 인스턴스(독립적
인 개체)
•운영체제로부터 시스템 자원을 할당받는 작업의 단위
•즉, 동적인 개념으로는 실행된 프로그램을 의미한다.

참고 할당받는 시스템 자원의 예

•CPU 시간
•운영되기 위해 필요한 주소 공간
•Code, Data, Stack, Heap의 구조로 되어 있는 독립된 메모리 영역

특징

•프로세스는 각각 독립된 메모리 영역(Code, Data, Stack, Heap의 구조)을 할당받는다.
•기본적으로 프로세스당 최소 1개의 스레드(메인 스레드)를 가지고 있다.
•각 프로세스는 별도의 주소 공간에서 실행되며, 한 프로세스는 다른 프로세스의 변수나 자료구조에 접근할 수 없다.
•한 프로세스가 다른 프로세스의 자원에 접근하려면 프로세스 간의 통신(IPC, inter-process communication)을 사용해
야 한다.Ex. 파이프, 파일, 소켓 등을 이용한 통신 방법 이용

인스턴스는 일반적으로 실행 
중인 임의의 프로세스

## --- [Page 81] ---
스레드(Thread) 란

리눅스의 프로세스(process) 관리

•“프로세스 내에서 실행되는 여러 흐름의 단위”
•프로세스의 특정한 수행 경로
•프로세스가 할당받은 자원을 이용하는 실행의 단위

•스레드는 프로세스 내에서 각각 Stack만 따로 할당받고 Code, Data, Heap 영역은 공유한다.
•스레드는 한 프로세스 내에서 동작되는 여러 실행의 흐름으로, 프로세스 내의 주소 공간이나 자원
들(힙 공간 등)을 같은 프로세스 내에 스레드끼리 공유하면서 실행된다.
•같은 프로세스 안에 있는 여러 스레드들은 같은 힙 공간을 공유한다. 반면에 프로세스는 다른 프로
세스의 메모리에 직접 접근할 수 없다.
•각각의 스레드는 별도의 레지스터와 스택을 갖고 있지만, 힙 메모리는 서로 읽고 쓸 수 있다.
•한 스레드가 프로세스 자원을 변경하면, 다른 이웃 스레드(sibling thread)도 그 변경 결과를 즉시 
볼 수 있다.

▪
특징

▪사전적 의미

## --- [Page 82] ---
프로세스(process)

리눅스의 프로세스(process) 관리

## --- [Page 83] ---
$ ps
PID TTY TIME CMD
8695 pts/3 00:00:00 bash
8720 pts/3 00:00:00 ps

$ ps u
USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
chang 8695 0.0 0.0 5252 1728 pts/3 Ss 11:12 0:00 bash
chang 8793 0.0 0.0 4252  940 pts/3 R+ 11:15 0:00 ps u

➢예

1) 프로세스관리 명령어 - ps

프로세스(process) 관리


| ps 출력 정보 |  |
| --- | --- |
| 항목 | 설명 |
| PID | 프로세스의 아이디, 식별변호 |
| PPID | 부모 프로세스 ID |
| UID | SYSTEM V계열에서 나타나는 항목으로 프로세스 소유자의 이름 |
| TTY | 프로세스를 제어하는 수단, 프로세스와 연결된 터미널로 콘솔접속시 "tty숫자" 행태로 표시되며, 원격이나 에뮬레이터 접속시 "pts/숫자" 형태로 표시 |
| TIME | 프로세스에 사용된 CPU 시간 |
| CMD | 프로세스 실행 명령어 |
| COMMAND | 프로세스의 실행 명령행 |
| USER | BSD계열에서 나타나는 항목으로 프로세스 소유자의 이름 |
| %CPU | CPU 사용 비율의 추정치(BSD) |
| %MEM | 메모리의 사용 비율의 추정치 (BSD) |
| VSZ | K단위 또는 페이지 단위의 가상메모리 사용량 |
| RSS | 실제 메모리 사용량 (Resident Set Size) |
| S, STAT | 현재 프로세스의 상태 코드 (S: Sys V, STAT: BSD) |
| STIME | 프로세스가 시작된 시간 혹은 날짜 |
| C, CP | 짧은 기간 동안의 CPU 사용률 (C: Sys V, CP: BSD) |
| F | 프로세스의 플래그 |
| PRI | 실제 실행 우선순위 |
| NI | nice 우선순위 번호 |

## --- [Page 84] ---
프로세스(process) 관리

## --- [Page 85] ---
프로세스(process) 관리

## --- [Page 86] ---
1) 프로세스관리 명령어 - ps

프로세스(process) 관리

## --- [Page 87] ---
Computation Server

In the cold and dark server r
oom!

8
7

Run Linux/Unix 
Operating System

Linux & C언어

## --- [Page 88] ---
Client/Server and SSH (Secure Shell)

8
8

Linux & C언어

## --- [Page 89] ---
Machine for Development for OpenMP and MPI

8
9

• Linux machines in Swearingen 1D39 and 3D22
– All CSCE students by default have access to these machine using t

heir standard login credentials

• Let me know if you, CSCE or not, cannot access
– Remote access is also available via SSH over port
222. Naming schema is as follows:
• l-1d39-01.cse.sc.edu through l-1d39-26.cse.sc.edu
• l-3d22-01.cse.sc.edu through l-3d22-20.cse.sc.edu
• Restricted to 2GB of data in their home folder (~/).
– For more space, create a directory in /scratch on the login machine,

however that data is not shared and it will only be available on that 
specific machine.

Linux & C언어

## --- [Page 90] ---
It is all about dealing with files and folders
Linux folder: /acct/yanyh/…

•
ls (list files in the current folder)
–
$ ls -l
–
$ ls -a
–
$ ls -la
–
$ ls -l --sort=time
–
$ ls -l --sort=size –r
•
cd (change directory to)
–
$ cd /usr/bin
•
pwd (show current folder name)
–
$ pwd
•
~ (home folder)
–
$ cd ~
•
~user (home folder of a user)
–
$ cd ~weesan

•
What will “cd ~/weesan” do?

• rm (remove a filer/folder)
– $ rm foo
– $ rm -rf foo
– $ rm -i foo
– $ rm -- -foo
• cat (print the file contents to 
terminal)
– $ cat /etc/motd
– $ cat /proc/cpuinfo
• cp (create a copy of a file/folder)
– $ cp foo bar
– $ cp -a foo bar
• mv (move a file/folder to anot
her location. Used also for ren
aming)
– $ mv foo bar
• mkdir (create a folder)
– $ mkdir foo

Linux & C언어

Linux Basic Commands

## --- [Page 91] ---
•
df (Disk usage)
– $ df -h /
– $ du -sxh ~/
• man (manual)
– $ man ls
– $ man 2 mkdir
– $ man man
– $ man -k mkdir
• Manpage sections
– 1
User-level cmds and apps
• /bin/mkdir
– 2
System calls
• int mkdir(const char *, …);
– 3
Library calls
• int printf(const char *, …);

Search a command or a file

•
which
–
$ which ls
•
whereis
–
$ whereis ls
•
locate
–
$ locate stdio.h
–
$ locate iostream
•
find
–
$ find / | grep stdio.h
–
$ find /usr/include | grep stdio.h

Smarty:
1. [Tab] key: auto-complete the command 
sequence
2.key: to find previous command
3. [Ctl]+r key: to search previous command

Linux & C언어

Basic Commands (cont)

## --- [Page 92] ---
• 2 modes
– Input mode

• ESC to back to cmd mode
– Command mode

• Cursor movement
– h (left), j (down), k (up), l (right)
– ^f (page down)
– ^b (page up)
– ^ (first char.)
– $ (last char.)
– G (bottom page)
– :1 (goto first line)
• Swtch to input mode
– a (append)
– i (insert)
– o (insert line after
– O (insert line before)

• Delete
– dd (delete a line)
– d10d (delete 10 lines)
– d$ (delete till end of line)
– dG (delete till end of file)
– x (current char.)
• Paste
– p (paste after)
– P (paste before)
• Undo
– u
• Search
– /
• Save/Quit
– :w (write)
– :q (quit)
– :wq (write and quit)
– :q! (give up changes)

Linux & C언어

Editing a File: Vi/Vim

## --- [Page 93] ---
•
vi hello.c
•
Switch to editing mode: i or a
•
Switching to control mode: ESC
•
Save a file: in control mode, :w
•
To quit, in control mode, :q
•
To quit without saving, :q!
•
Copy/paste a line: in control model, “yy” and then “p”, both from the current 
cursor
– 5 line: 5yy and then p
•
To delete a whole line, in control mode, : dd

•
vi hello.c
•
ls hello.c
•
gcc hello.c –o hello
•
ls
•
./hello

1
1

#include <stdio.h>

/* The simplest C Program */

int main(int argc, char **argv) {

printf(“Hello World\n”);

return 0;

}

Linux & C언어

C Hello World

## --- [Page 94] ---
#include <stdio.h>

/* The simplest C Program */ i

nt main(int argc, char **argv)

{

printf(“Hello World!\n”);

return 0;

}

The main() function is always 
where your program starts r
unning.

#include inserts another file. “.h” files are called “header” 
files. They contain declarations/definitions needed to int
erface to libraries and code in other “.c” files.

A comment, ignored by the compiler

Blocks of code (“lexical scop
es”) are marked by { … }

Return ‘0’ from this function

What do the < > 
mean?

12

Linux & C언어

C Syntax and Hello World

## --- [Page 95] ---
Compiling/Building Process in C to  Generate Executables

• Compiling/Building process: gcc hello.c –o hello
– Constructing an executable image for an application
– FOUR stages
– Command:

gcc <options> <source_file.c>

• Compiler Tool
– gcc (GNU Compiler)

• man gcc (on Linux m/c)

– icc (Intel C compiler)

Linux & C언어

## --- [Page 96] ---
4 Stages of Compiling Process

1. Preprocessing (Those with # …)
– Expansion of Header files (#include … )
– Substitute macros and inline functions (#define …)
2. Compilation (the most important one)
– Generates assembly language
– Verification of functions usage using prototypes
– Header files: Prototypes declaration (-I option to provide header

folder)
3. Assembling
– Generates re-locatable object file (contains m/c instructions)
– nm app.o: To list functions/symbols provided by a an object file

0000000000000000 T main
U puts
– objdump to view object files and disassembly

Linux & C언어

## --- [Page 97] ---
4. Linking
– Generates executable file (nm tool used to view exe file)
– Binds appropriate libraries

• Static Linking
• Dynamic Linking (default)

• Loading and Execution (of an executable file)
– Evaluate size of code and data segment
– Allocates address space in the user mode and transfers them

into memory
– Load dependent libraries needed by program and links them
– Invokes Process Manager →Program registration

Linux & C언어

4 Stages of Compiling Process (contd..)

## --- [Page 98] ---
View the output of each stage using vi editor: e.g. vim hello.i
Preprocessing
gcc –E hello.c –o hello.i 
hello.c →hello.i

Compilation (after preprocessing)

gcc –S hello.i –o hello.s

Assembling (after compilation)

gcc –c hello.s –o hello.o

Linking object files

gcc hello.o –o hello

Output →Executable (a.out) 
Run →./hello (Loader)

Linux & C언어

4 Stages of Compiling Process

## --- [Page 99] ---
• gcc <options> program_name.c

• Options:

-Wall: Shows all warnings
-o output_file_name: By default a.out executable file is 
created when we compile our program with gcc. Instead, 
we can specify the output file name using "-o" option.
-g: Include debugging information in the binary.

• man gcc

Four stages into one

Linux & C언어

Compiling a C Program

## --- [Page 100] ---
• Two programs, prog1.c and prog2.c for one single task
– To make single executable file using following instructions

First, compile these two files with option "-c" gcc -c prog1.c
gcc -c prog2.c

-c: Tells gcc to compile and assemble the code, but not link. We get two file

s as output, prog1.o and prog2.o
Then, we can link these object files into single executable file using below ins
truction.

gcc -o prog prog1.o prog2.o

Now, the output is prog executable file. We can run our
program using
./prog

Linux & C언어

Linking multiple files to make executable file

## --- [Page 101] ---
• Normally, compiler will read/link libraries from /usr/lib 
directory to our program during compilation process.
– Library are precompiled object files

• To link our programs with libraries like pthreads and 
realtime libraries (rt library).
– gcc <options> program_name.c -lpthread -lrt

-lpthread: Link with pthread library →libpthread.so file
-lrt: Link with rt library
→librt.so file 
Option here is "-l<library>"

Another option "-L<dir>" used to tell gcc compiler search for 
library file in given <dir> directory.

Linux & C언어

Linking with other libraries

## --- [Page 102] ---
source

file 1

source

file 2

source

file N

object

file 1

object

file 2

object

file N

library
object
file 1

library
object
file M

load

file

usually performed by a compiler, usually in one uninterrupted sequence

http://www.tenouk.com/ModuleW.html

linking (relo

cation +  lin

king)
compilation

Linux & C언어

Compilation, Linking, Execution of C/C++ Programs

## --- [Page 103] ---
10
3

• nm: e.g. “nm a.out”, “nm libc.so”
– list symbols from object files

• Symbols: function name, global variables that are exposed or refer
ence by an object file.
• ldd: e.g. “ldd a.out”, or “ldd hello”
– List the name and the path of the dynamic library needed by a progr

am

– LD_LIBRARY_PATH: env for setting runtime lib path

• export LD_LIBRARY_PATH=/acct/yanyh/usr/lib:$LD_LIBRARY_PATH
• objdump: objdump –d a.out
– dump information about object files, including disassembly

Linux & C언어

Three Useful Commands

## --- [Page 104] ---
10
4

• Download the file:
– wget https://passlab.github.io/CSCE569/resources/sum.c
• gcc sum.c –o sum
• ./sum 102400

• vi sum.c
• ldd sum
• nm sum

• Other system commands:
– cat /proc/cpuinfo to show the CPU and #cores
– top command to show system usage and memory

Or step by step 
gcc -E sum.c -o sum.i g
cc -S sum.i -o sum.s gcc 
-c sum.c -o sum.o gcc s
um.o -o sum

Linux & C언어

sum.c

## --- [Page 105] ---
sum (exe)

sum.o
main.o

sum.c
sum.h
sum.h
main.c

Linux & C언어

Makefile

## --- [Page 106] ---
sum: main.o sum.o

gcc –o sum main.o sum.o

main.o: main.c sum.h

gcc –c main.c

sum.o: sum.c sum.h

gcc –c sum.c

Linux & C언어

Makefile

## --- [Page 107] ---
main.o: main.c sum.h

gcc –c main.c

tab

dependency
action

Rule

Linux & C언어

Rule syntax

## --- [Page 108] ---
• Provides single-sourcing for build systems
• Knowledge of many platforms and tools
• Users configure builds through a GUI

CMakeLists.txt
CMake
Native Build System

Native Build Tools
Executables  
Libraries

Linux & C언어

cmake: Makefile/Build-System Generator

## --- [Page 109] ---
• Memory is a sequentially accessed using the address of 
each byte/word

10
9

Linux & C언어

Sequential Memory Regions vs Multi- dimensional Array

## --- [Page 110] ---
• C has row-major storage for multiple dimensional array
– A[2,2] is followed by A[2,3]

char A[4][4]

• 3-dimensional array
– B[3][100][100]

Memory address of A[2][3]
= A[0][0] + offset
= A + sizeof (char) * (2 * # columns + 3)
= 0 + 1 * (2 * 4 + 3) = 11

11
0

Linux & C언어

Vector/Matrix and Array in C

## --- [Page 111] ---
C storage type
Address of element A[1][2]?

Address of element A[1][2]?
= A + sizeof (int) * (1 * 4 + 2)
= A + 4 * 6 = A + 24

Linux & C언어

Store Array in Memory in Row Major


| 8 | 6 | 5 | 4 |
| --- | --- | --- | --- |
| 2 | 1 | 9 | 7 |
| 3 | 6 | 4 | 2 |

## --- [Page 112] ---
11
2

3x4 matrix

Linux & C언어

Store Array in Memory in Column Major


| 8 | 6 | 5 | 4 |
| --- | --- | --- | --- |
| 2 | 1 | 9 | 7 |
| 3 | 6 | 4 | 2 |

## --- [Page 113] ---
3 X 4

3 X 4

11
3

Linux & C언어

For a Memory Region to Store Data for an Array in Either Row or Col Major


| 8 | 6 | 5 | 4 |
| --- | --- | --- | --- |
| 2 | 1 | 9 | 7 |
| 3 | 6 | 4 | 2 |

| 8 | 4 | 9 | 6 |
| --- | --- | --- | --- |
| 6 | 2 | 7 | 4 |
| 5 | 1 | 3 | 2 |

## --- [Page 114] ---
C Programming

Linux & C언어

## --- [Page 115] ---
• Linux/Unix Introduction
– http://www.ee.surrey.ac.uk/Teaching/Unix/
• VI Editor
– https://www.cs.colostate.edu/helpdocs/vi.html
• C Programming Tutorial
– http://www.cprogramming.com/tutorial/c-tutorial.html
• Compiler, Assembler, Linker and Loader: A Brief Story
– http://www.tenouk.com/ModuleW.html

63

Linux & C언어

## --- [Page 116] ---
linux library 만들기(static, shared, dynamic)

linux library 만들기
(static, shared, dynamic)

## --- [Page 117] ---
➢ 정적 라이브러리와 동적 라이브러리(Static Library, Dynamic Library)

•
라이브러리에 의존하는 프로그램은 시스템에 필요한 라이브러리가 설치되지 않으면 동작하지 않는다.

•
실제로 실행될 때는 라이브러리가 제공하는 코드와 링크되어야 프로그램 코드가 동작할 수 있다.

•
라이브러리는 오브젝트 코드와 결합하는 방법에 따라 정적(Static) 라이브러리와 공유(Shared) 라이브러
리로 나뉜다.

•
공유(shared) 라이브러리는 다시 일반적인 동적 링크(Dynamic link) 라이브러리와 동적 로드(Dynamic 
load) 라이브러리로나뉜다.

Library와 Link

## --- [Page 118] ---
•
프로그램 빌드 시에 라이브러리가 제공하는 코드를 실행 파일에 넣는 방식의 라이브러리를 의미한다.

•
이 방식의 장점은 시스템 환경이 변해도 애플리케이션에 아무런 영향이 없고, 완성된 애플리케이션을 
안정적으로 사용할 수 있다는 점이 있다.

•
반면에 사용하는 모든 오브젝트 코드를 실행 파일에 내장하기 때문에 메모리에 로드되는 애플리케이션 
코드 크기가 커진다는 단점이 있다.

•
유닉스/리눅스에서는 확장자로 .a가 붙는다.

정적 라이브러리(Static Library)란?

## --- [Page 119] ---
•
어떤 라이브러리가 제공하는 기능을 다른 애플리케이션에서 사용하고 싶을 때 라이브러리 코드를 
메모리에 하나만 두고 각 애플리케이션에서 이를 공유하는 방식의 라이브러리를 의미한다.

공유 라이브러리(Shared Library)란?

## --- [Page 120] ---
•
공유 라이브러리는 프로그램 실행 시 라이브러리의

코드와 애플리케이션의 코드가 메모리에 로드되는 시

점에 링크된다.

•
그렇기 때문에 라이브러리를 이용하는 애플리케이션

에는 호출할 라이브러리 함수의 정보만 들어 있다. 애

플리케이션이 실행되어 메모리에 로드된 시점에서야

그 함수가 메모리의 어디에 있는지 알 수 있고, 그곳

으로 포인터가 쓰여지면서 함수의 호출을 실현한다.

•
이 구조를 이용하면 한 라이브러리의 코드를 여러 애

플리케이션에서 공유할 수 있기 때문에 메모리를 효

율적으로 이용할 수 있다. 윈도우에서는 확장자

로 .DLL이 붙고, 리눅스에서는 확장자로 .so가 붙는다.

공유 라이브러리(Shared Library)란?

## --- [Page 121] ---
C Programming: The Four Stages of Compilation

121

gcc C 컴파일러 드라이버 옵션
-E : 전처리 과정의 결과를 화면에 보이는 옵션
-save -temps 옵션을 사용해 like.i 파일을 읽어보는 것이 더 좋은 방법.
-S : 어셈블리 파일만 생성하고 컴파일 과정을 멈춘다.
-c : 오브젝트 파일까지만 생성하고 컴파일 과정을 멈춘다.
-v : gcc가 컴파일 과정을 어떤 식으로 수행하는지를 화면에 출력한다.
-save-temps : 컴파일 과정에서 생성되는 중간 파일인 전처리 파일(*.i)과 어셈블리 파일(*.s), 오브젝트 파
일(*.o)을 지우지 않고 현재 디렉토리에 저장한다.

gcc -E hello.c -o hello.i

## --- [Page 122] ---
C Programming: The Four Stages of Compilation

gcc -E hello.c -o hello.i

gcc -S hello.c

gcc -c hello.c

gcc hello.c -o hello

전처리 단계:c 파일을 gcc 컴파일러로 컴파일 할 경우 전처리 단계가 진행되는데 결과로 i 파일을 만듬

컴파일 단계:전처리된 파일로 컴파일을 진행하여 어셈블리어로 된 파일인 s 파일을 생성

어셈블 단계:어셈블리어로 쓰여진 s 파일을 컴퓨터가 이해할 수 있는 기계어로 된 파일인 o 파일로 변환

링크 단계: 라이브러리 함수와 여러 오브젝트 파일들을 연결해서 실행 파일인 a.out을 생성

## --- [Page 123] ---
동적 로드 라이브러리(Dynamic Load Library)란?

• 동적 링크 방식에서는 애플리케이션의 실행 시, 실행 파일과 관련된 라이브러리 코드를 모두

메모리에 읽어들여 호출. 관계를 조정한 다음 애플리케이션이 실행된다.

• 동적 로드는 애플리케이션 실행 시에 읽어들이지 않은 라이브러리를 이용할 수도 있기 때문에

동적 링크보다 더욱 자유도가 높은 라이브러리의 활용 방식을 사용한다고 할 수 있다. 심지어

애플리케이션을 빌드 할 때 존재하지 않았던 라이브러리도 사용이 가능하다.

• 동적 로드에서는 프로그램 내부에서 라이브러리를 로드하는데, 어느 라이브러리의 어느 함수

를 이용할지는 프로그램의 동작 상황에 따라 변할 수 있다. 대다수 애플리케이션에서 자주 이

용되는 플러그인에 의한 기능 확장은 동적 로드를 이용해 구현된다.

## --- [Page 124] ---
•
 라이브러리는 함수, 구조체, 클래스 등을 포함하고 있는 컴파일된 파일이며, 그 종류는 다음과 같이 3가지가 존

재한다.

* 정적라이브러리 (*.a, *.lib)

* 공유라이브러리 (*.so, *.dll)

* 동적라이브러리 (*.so, *.dll)

⚫정적 라이브러리는 컴파일시에 해당 정적 라이브러리의 내용이 실행 바이

너리안에 포함되어지는 특징을 갖고 있다. 즉, 실행 파일 배포시에 정적라

이브러리를 함께 배포하지 않아도 된다.

라이브러리종류

## --- [Page 125] ---
라이브러리종류

⚫반면, 동적 라이브러리는 컴파일시에 해당 동적 라이브러리의 내용이 실행 파일안에 포함되지 않

기 때문에, 반드시실행 파일 배포시에는 동적 라이브러리 역시 함깨 배포해야 한다.

⚫유닉스 계열의 OS에서는ar 시스템 유틸리티를 사용하여 정적라이브러리를 생성할 수 있다.

⚫참고로 정적 라이브러리 생성시에 반드시 사용되어지는 옵션은 다음과 같다.  -c create archive

file -r insert object file 예를 들면 다음과 같다.

⚫ar rc libmymath.a mymath.o 그리고, 생성된 정적 라이브러리 파일내에 컴파일 된 Object List를

보기 위해서는 다음과 같이 -t 옵션을 사용할 수 있다.

⚫ar t libmymath.a

## --- [Page 126] ---
ar : 정적 라이브러리 작성 명령어옵션

## --- [Page 127] ---
63

왜 분산처리(병행처리)처리인가.??

반도체 전자 장치에서 Dennard 스케일링(MOSFET 스케일링이라고도 함)은 트랜지스터가 작아
질수록 전력 밀도가 일정하게 유지되어 전력 사용이 면적에 비례하여 유지된다는 스케일링 법
칙

Dennard scaling

## --- [Page 128] ---
63

왜 분산처리(병행처리)처리인가.??

반도체 전자 장치에서 Dennard 스케일링(MOSFET 스케일링이라고도 함)은 트랜지스터가 작아
질수록 전력 밀도가 일정하게 유지되어 전력 사용이 면적에 비례하여 유지된다는 스케일링 법
칙

Dennard scaling

The End of Dennard 
Scaling

## --- [Page 129] ---
왜 분산처리(병행처리)처리인가.??

Second Phase of Moore’s Law

## --- [Page 130] ---
왜 분산처리(병행처리)처리인가.??

The End of Single-Die Scaling

## --- [Page 131] ---
왜 분산처리(병행처리)처리인가.??

레티클 한계(Reticle Limit)  극복: Heterogeneous Hardware

## --- [Page 132] ---
왜 분산처리(병행처리)처리인가.??

Heterogeneous Software

Software is also made up of heterogeneous components written in many different 
langauges

## --- [Page 133] ---
왜 분산처리(병행처리)처리인가.??

Heterogeneous Computing

Combining processors of different types, each specializing in different types of 
execution

## --- [Page 134] ---
왜 분산처리(병행처리)처리인가.??

Third Phase of Moore’s Law

## --- [Page 135] ---
왜 분산처리(병행처리)처리인가.??

Multi-Die Heterogeneous Devices

Grace/Hopper Superchip

## --- [Page 136] ---
왜 분산처리(병행처리)처리인가.??

Grace/Hopper Superchip

Grace/Hopper Superchip

## --- [Page 137] ---
왜 분산처리(병행처리)처리인가.??

NVLink Connects Up To 256 Superchips

Grace/Hopper Superchip

## --- [Page 138] ---
왜 분산처리(병행처리)처리인가.??

Multi-Chip Systems

Grace/Hopper Superchip

## --- [Page 139] ---
왜 분산처리(병행처리)처리인가.??

There Is No Future That Is Not Multi-Node

Grace/Hopper Superchip