---
aliases: []
course: operating-systems
created: '2024-09-04'
date: '2024-09-04'
semester: 2-2
source: ''
status: seedling
tags:
- cs/systems
- type/lecture
title: OS의 시작과 발전
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/운영체제 인터페이스|운영체제 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다|쿠다]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/1. 프로세스 구성과 동작|1. 프로세스 구성과 동작]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/1. Why Parallelism|1. Why Parallelism]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/2. 제어 장치의 종류|2. 제어 장치의 종류]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/5. 파이프 라이닝|5. 파이프 라이닝]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/2. 주기억 장치|2. 주기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/3. 레지스터|3. 레지스터]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/1. 제어 장치의 기능|1. 제어 장치의 기능]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/4. 가상 기억 장치|4. 가상 기억 장치]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/5. 주소 지정 방식|5. 주소 지정 방식]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/4. 컴퓨터 명령어|4. 컴퓨터 명령어]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/3. 명령어 사이클|3. 명령어 사이클]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/3. 캐시 기억 장치|3. 캐시 기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/1. 기억 장치 시스템의 개요|1. 기억 장치 시스템의 개요]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/6. CISC와 RISC|6. CISC와 RISC]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/05_software-engineering/programming-languages/교재/5장_교재_문제|5장_교재_문제]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/4. 프로세서 제어|4. 프로세서 제어]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/4. 디지털 코드|4. 디지털 코드]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/애플 M4 CPU/애플 M4 CPU|애플 M4 CPU]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/전역변수, 지역변수|전역변수, 지역변수]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/5. 에러 검출 코드|5. 에러 검출 코드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/2. 정수 표현|2. 정수 표현]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/2. 불 대수|2. 불 대수]], [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/1. 진법과 진법 변환|1. 진법과 진법 변환]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/1. 논리 게이트|1. 논리 게이트]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/3. 실수 표현|3. 실수 표현]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/2. 산술 논리 연산 장치|2. 산술 논리 연산 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/중간 시험 범위|중간 시험 범위]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/3. 카르노 맵|3. 카르노 맵]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로|4. 조합 논리 회로]], [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/04_systems-infrastructure/computer-networks/2. 네트워크 분류와 계층 모델/네트워크 분류와 계층 모델|네트워크 분류와 계층 모델]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/04_systems-infrastructure/computer-networks/12. 네트워크 계층 작업과 프로토콜/네트워크 계층 작업과 프로토콜|네트워크 계층 작업과 프로토콜]], [[ComputerScience/04_systems-infrastructure/computer-networks/4. 유선 및 무선 데이터 전송/유선 및 무선 데이터 전송|유선 및 무선 데이터 전송]], [[ComputerScience/04_systems-infrastructure/computer-networks/9. 네트워크 계층/네트워크 계층|네트워크 계층]], [[ComputerScience/04_systems-infrastructure/computer-networks/8. 무선통신 시스템/무선통신 시스템|무선통신 시스템]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/5. 통신망과 특징/통신망과 특징|통신망과 특징]], [[certifications/체크리스트|체크리스트]], [[ComputerScience/04_systems-infrastructure/computer-networks/11. 인터넷 프로토콜 라우팅 알고리즘/인터넷 프로토콜(IP)|인터넷 프로토콜(IP)]], [[ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍|TCP와 소켓 프로그래밍]], [[certifications/information-processing/실기/오답노트|오답노트]], [[ComputerScience/04_systems-infrastructure/computer-networks/3. 신호 처리/신호 처리|신호 처리]], [[ComputerScience/04_systems-infrastructure/computer-networks/10. 라우팅 알고리즘/라우팅 알고리즘|라우팅 알고리즘]], [[ComputerScience/04_systems-infrastructure/computer-networks/7. LAN의 특징과 규격/LAN의 특징과 규격|LAN의 특징과 규격]]

kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/운영체제 지식그래프|운영체제]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/운영체제 근거 인덱스|운영체제 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/cpu|cpu]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/fcfs|fcfs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/cuda|cuda]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/srtf|srtf]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/os|os]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
## 운영체제의 정의에서 핵심 단어
1. **운영체제는 컴퓨터의 모든 자원(resource) 관리** 
	- 자원 
		-  하드웨어 자원 – CPU, 캐시, 메모리, 키보드, 마우스, 디스플레이,하드디스크,프린터 
		-  소프트웨어 자원 - 응용프로그램 
		-  데이터 자원 - 파일, 데이터베이스 등 
2. **운영체제는 자원에 대한 독점(exclusive) 권한 소유** 
	-  자원에 대한 모든 관리 권한 운영체제에게 있음 
		-  자원 할당, 자원 공유, 자원 액세스, 자원 입출력 등 
		-  예) 파일 생성 – 디스크의 빈 공간 관리, 파일 저장 위치 관리, 파일 입출력 등 
3. **운영체제는 관리자(supervisor)** 
	-  실행중인 프로그램 관리, 메모리 관리, 
	-  파일과 디스크 장치 관리, 입출력 장치 관리, 사용자 계정 등 관리 등 
5. **운영체제는 소프트웨어(software)** 
	-  커널(kernel)이라고 불리는 핵심 코드와, 
	-  UI를 비롯한 도구 프로그램들(tool/utility), 
		-  예: 탐색기(explorer), 작업 관리자(task manager), 제어판(control panel) 등 
	-  장치를 제어하는 디바이스 드라이버들로 구성

## 운영체제와 응용소프트웨어

![](../../../../image/Pasted%20image%2020240904103832.png)

## 운영체제의 역사

#### 1. 고정 프로그래밍 방식
- 1940년대, 전자식 디지털 컴퓨터가 만들어지기 시작한 시대
- 운영체제에 대한 개념 없음
- 소프트웨어와 하드웨어로 제작

#### 2. 내장 프로그래밍 방식
- 1945년 폰 노이만에 의해 제안
- 1951년 EDVAC 컴퓨터를 만들 때 적용
- CPU와 메모리 분리
- 소프트웨어와 하드웨어 분리
	![](../../../../image/Pasted%20image%2020240904104950.png)

#### 3. 프로그램 로더의 발견
 1. **로더의 역할**
   - **프로그램 적재(Loading)**: 로더는 실행 파일을 메모리에 적재합니다. 이 과정에서 실행 파일의 코드, 데이터, 초기화되지 않은 데이터 등이 각각 메모리의 적절한 위치에 배치됩니다.
   - **메모리 할당**: 실행 파일을 적재할 메모리 공간을 할당합니다. 메모리에서 프로그램이 사용할 공간을 예약하고, 코드와 데이터를 배치합니다.
   - **주소 바인딩(Address Binding)**: 실행 파일이 사용하고 있는 가상 주소를 실제 메모리 주소로 변환하는 과정을 관리합니다. 컴파일러가 생성한 실행 파일은 가상 주소를 기반으로 작성되기 때문에, 이를 물리적 주소로 변환해 실제 메모리에 적재할 수 있어야 합니다.
   - **초기화**: 프로그램의 초기화 과정을 담당합니다. 여기에는 전역 변수의 초기화, 스택 초기화 등이 포함됩니다.
   - **제어권 전달**: 프로그램의 초기 진입점(entry point)으로 제어권을 넘깁니다. 이는 보통 프로그램의 `main()` 함수나 실행 시작 지점으로 연결됩니다.

 2. **로더의 유형**
   - **절대 로더(Absolute Loader)**: 컴파일러나 어셈블러가 생성한 실행 파일을 메모리의 특정 위치에 고정된 주소로 적재합니다. 프로그램이 메모리 내에서 고정된 위치에서만 실행될 수 있습니다.
   - **재배치 가능한 로더(Relocating Loader)**: 실행 파일을 적재할 때 메모리의 어느 위치에든 로드될 수 있도록 주소를 조정합니다. 이 방법은 프로그램을 메모리의 다양한 위치에서 실행할 수 있도록 유연성을 제공합니다.
   - **동적 로더(Dynamic Loader)**: 프로그램 실행 중에 필요한 모듈이나 라이브러리를 동적으로 메모리에 적재합니다. 이를 통해 프로그램은 실제로 필요할 때만 메모리를 할당받아 더 효율적인 메모리 사용이 가능합니다.

---
## 배치 운영체제

![](../../../../image/Pasted%20image%2020240909091810.png)

- 출현 배경
	- 컴퓨터의 노는 시간을 줄여 컴퓨터의 활용률 향상
- 배치 운영체제 컴퓨터 시스템
	- 개발자와 관리자의 구분
	- 개발자는 펀치 카드를 입력 데크에 두고 결과 기다림
	- 배치 운영체제는 자동으로 데이프 장치에 대기중인 프로그램을 한 번에 하나씩 적재하고, 실행

## 다중 프로그래밍 운영체제

![](../../../../image/Pasted%20image%2020240909091830.png)

다중프로그래밍은 여러 프로그램을 메모리에 올려놓고, CPU가 한 프로그램을 실 행하다 I/O가 발생하면, 입출력이 완료될 때까지 CPU가 메모리에 적재된 다른 프로그램을 실행하는 식으로 CPU의 노는 시간을 줄이는 기법이다.

- 출현 배경
	- 프로그램의 실행 형태로 인한 CPU의 유휴시간(idle 시간) 발생
		- 프로그램 실행 형태 : CPU 작업 - I/O 작업 - CPU 작업 - I/O 작업의 반복
		- 배치 작업은 1번에 1개의 프로그램만 실행하므로, I/O 작업이 이루어지는 동안 CPU는 놀면서 대기, CPU의 많은 시간 낭비
- 다중프로그래밍 기법 출현
	- 미리 여러 프로그램을 메모리에 적재
	- **프로그램 실행 도중 I/O가 발생하면, CPU에게 메모리에 적재된 다른 프로그램 실행시킴**
- 문제점
	- 큰 메모리 이슈
		- 여러 프로그램을 동시에 메모리에 올려놓기 위해 메모리의 크기가 필요
	- 프로그램의 메모리 할당 및 관리 이슈
		- 몇 개의 프로그램 적재? 메모리 어디에 적재? 프로그램 당 할당하는 메모리 크기?
	- 메모리 보호 이슈
		- 프로그램이 다른 프로그램의 영역을 침범하지 못하게 막는 방법 필요
	- CPU 스케줄링과 컨텍스트 스위칭
		- 실행시킬 프로그램 선택하는 스케줄링 필요
		- 프로그램의 실행 상태를 저장할 컨텍스트 정의
		- 컨텍스트 스위칭 필요
	- **인터럽트 개념 도입**
		- 운영체제가 I/O 장치로 부터 입출력 완료를 전달받는 방법 필요
	- 동기화
		- 예) A프로그램이 완료가 되어야 B프로그램을 실행할 수 있을 때
		- 여러 프로그램이 동일한 자원을 동시에 액세스할 때 발생하는 문제 해결
	- 교착 상태 해결 (DeadLock)
		- 프로세스들이 상대가 가진 자원을 서로 요청하면서 무한대기하는 교착상태 해결

![](../../../../image/Pasted%20image%2020240904111128.png)

## 시분할 다중프로그래밍 운영체제 (Time Sharing Multiprogramming)

![](../../../../image/Pasted%20image%2020240909091851.png)

- 출현 배경 
	- 다중프로그래밍 운영체제와 거의 동시에 연구 시작 
	- 배치 처리와 당시 다중프로그래밍의 다음 2가지 문제점 인식 
		- 비 대화식 처리방식(non-interactive processing) 
		- 느린 응답시간, 오랜 대기 시간 
			- 프로그램을 제출하고 하루 후에 결과 보기, 사용자의 즉각적인 대응 없음 
- 시분할 운영체제의 시작 
	- 1959년 MIT 대학, John McCarty 교수에 의해 
	- 빠른 프로그래밍 디버깅 필요 
		- McCarty 교수의 당면한 문제였음 
	- 사용자에게 빠른 응답을 제공하는 **대화식 시스템** 제안 
		- **터미널**이란? 키보드+모니터+전화선+모뎀으로 구성된 장치 
		- 사용자는 자신의 터미널을 이용하여 메인 컴퓨터에 원격 접속 
		- 운영체제는 **시간을 나누어 돌아가면서 각 터미널의 명령** 처리 
	- CTSS(Compatible Time Sharing System) 시분할 시스템 개발 
		- 1962년 MIT
