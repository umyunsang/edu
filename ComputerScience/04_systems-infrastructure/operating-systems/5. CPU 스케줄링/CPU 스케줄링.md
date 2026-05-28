---
aliases: []
course: operating-systems
created: '2024-10-07'
date: '2024-10-07'
semester: 2-2
source: ''
status: seedling
tags:
- cs/systems
- type/lecture
title: CPU 스케줄링
type: lecture
updated: '2026-05-05'
---



domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
up:: [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다|쿠다]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/애플 M4 CPU/애플 M4 CPU|애플 M4 CPU]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/1. 기억 장치 시스템의 개요|1. 기억 장치 시스템의 개요]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/1. Why Parallelism|1. Why Parallelism]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/5. 파이프 라이닝|5. 파이프 라이닝]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/1. 제어 장치의 기능|1. 제어 장치의 기능]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/3. 명령어 사이클|3. 명령어 사이클]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/1. 프로세스 구성과 동작|1. 프로세스 구성과 동작]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/4. 컴퓨터 명령어|4. 컴퓨터 명령어]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/2. 주기억 장치|2. 주기억 장치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/6. CISC와 RISC|6. CISC와 RISC]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/2. 제어 장치의 종류|2. 제어 장치의 종류]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/3. 캐시 기억 장치|3. 캐시 기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/4. 가상 기억 장치|4. 가상 기억 장치]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/5. 주소 지정 방식|5. 주소 지정 방식]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/2. 정수 표현|2. 정수 표현]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/5. 에러 검출 코드|5. 에러 검출 코드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/4. 프로세서 제어|4. 프로세서 제어]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/3. 레지스터|3. 레지스터]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/2. 산술 논리 연산 장치|2. 산술 논리 연산 장치]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/4. 디지털 코드|4. 디지털 코드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/1. 진법과 진법 변환|1. 진법과 진법 변환]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/3. 실수 표현|3. 실수 표현]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/1. 논리 게이트|1. 논리 게이트]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/3. 카르노 맵|3. 카르노 맵]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/2. 불 대수|2. 불 대수]], [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/computer-networks/10. 라우팅 알고리즘/라우팅 알고리즘|라우팅 알고리즘]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로|4. 조합 논리 회로]], [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/중간 시험 범위|중간 시험 범위]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/8. 무선통신 시스템/무선통신 시스템|무선통신 시스템]], [[certifications/information-processing/실기/오답노트|오답노트]], [[ComputerScience/04_systems-infrastructure/computer-networks/11. 인터넷 프로토콜 라우팅 알고리즘/인터넷 프로토콜(IP)|인터넷 프로토콜(IP)]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/04_systems-infrastructure/computer-networks/6. 데이터 링크 계층의 작업/데이터 링크 계층의 작업 (2 계층)|데이터 링크 계층의 작업 (2 계층)]], [[ComputerScience/04_systems-infrastructure/computer-networks/4. 유선 및 무선 데이터 전송/유선 및 무선 데이터 전송|유선 및 무선 데이터 전송]], [[certifications/체크리스트|체크리스트]]

---
## 1. CPU 스케줄링 개요
- 스케줄링은 왜 필요할까? 
	- 자원에 대한 경쟁이 있는 곳에서 경쟁자 중 하나 선택 
	- 자원 : CPU, 디스크, 프린트, 파일, 데이터베이스 등 
- 컴퓨터 시스템 내 다양한 스케줄링
	- 작업( job) 스케줄링 
		- 배치시스템에서 
		- 대기중인 배치 작업(Job) 중 메모리에 적재할 작업 결정 
	- **CPU 스케줄링** 
		- 프로세스/스레드 중에 하나를 선택하여 CPU 할당 
		- 오늘날 CPU 스케줄링은 스레드 스케줄링 
	- 디스크 스케줄링 
		- 디스크 장치 내에서 
		- 디스크 입출력 요청 중 하나 선택 
	- 프린터 스케줄링 
		- 프린팅 작업 중 하나 선택하여 프린터 할당
#### CPU burst와 I/O burst
- 프로그램의 실행 특성 
	- CPU 연산 작업과 I/O 작업(화면 출력, 키보드, 입력, 파일 입출력 등)이 순차적으로 섞여 
	- CPU-burst – I/O burst – CPU-burst – I/O burst의 반복 ... 
- CPU burst 
	- 프로그램 실행 중 CPU 연산(계산 작업)이 연속적으로 실행되는 상황 
- I/O burst 
	- 프로그램 실행 중 I/O 장치의 입출력이 이루어지는 상황
#### CPU 스케줄링의 정의와 목표
- CPU 스케줄링 
	- 정의 
		- 실행 준비 상태(Ready)의 스레드 중 하나를 선택하는 과정 
	- 기본 목표 
		- CPU 활용률 향상 -> 컴퓨터 시스템 처리율 향상 
	- 컴퓨터 시스템에 따라 CPU 스케줄링의 목표가 다를 수 있다
#### CPU 스케줄링의 기준(criteria)
- 스케줄링 알고리즘의 다양한 목표와 평가 기준 
	- CPU 활용률(CPU utilization) 
		- 전체 시간 중 CPU의 사용 시간 비율, 운영체제 입장 
	- 처리율(throughput) 
		- 단위 시간당 처리하는 스레드 개수, 운영체제 입장 
	- 공평성(fairness) 
		- CPU를 스레드들에게 공평하게 배분, 사용자 입장 
		- 시분할로 스케줄링 
		- 무한정 대기하는 기아 스레드(starving thread)가 생기지 않도록 스케줄 
	- 응답시간(response time) 
		- 대화식 사용자의 경우, 사용자에 대한 응답 시간, 사용자 입장 
	- **대기시간(waiting time)** 
		- 스레드가 준비 큐에서 머무르는 시간, 운영체제와 사용자 입장 
	- 소요 시간(turnaround time) 
		- 프로세스(스레드)가 컴퓨터 시스템에 도착한 후(혹은 생성된 후) 완료될 때까지 걸린 시간, 사용자 입장 
		- 배치 처리 시스템에서 주된 스케줄링의 기준 
	- 시스템 정책(policy enforcement) 우선 
		- 컴퓨터 시스템의 특별한 목적을 달성하기 위한 스케줄링, 운영체제 입장 
		- 예) 실시간 시스템에서는 스레드가 완료 시한(deadline) 내에 이루어지도록 하는 정책 
		- 예) 급여 시스템에서는 안전을 관리하는 스레드를 우선 실행하는 정책 등 
	- 자원 활용률(resource efficiency)

## 2. CPU 스케줄링 기본

#### CPU 스케줄링이 실행되는 4가지 상황
1. 스레드가 시스템 호출 끝에 I/O를 요청하여 블록될 때 
	-  스레드를 블록 상태로 만들고 스케줄링 
	-  (CPU 활용률 향상 목적) 
2. 스레드가 자발적으로 CPU를 반환할 때
	-  yield() 시스템 호출 등을 통해 스레드가 자발적으로 CPU 반환 
	-  커널은 현재 스레드를 준비 리스트에 넣고, 새로운 스레드 선택 
	-  (CPU의 자발적 양보) 
3. 스레드의 타임 슬라이스가 소진되어 타이머 인터럽트 발생 
	-  (균등한 CPU 분배 목적) 
4. 더 높은 순위의 스레드가 요청한 입출력 작업 완료, 인터럽트 발생 
	-  현재 스레드를 강제 중단(preemption)시켜 준비 리스트에 넣고 
	-  높은 순위의 스레드를 깨워 스케줄링 
	-  (우선순위를 지키기 위한 목적)
#### CPU 스케줄링과 디스패치(dispatch)

#### 선점 스케줄링과 비선점 스케줄링

![](../../../../image/Pasted%20image%2020241007095445.png)
#### 기아와 에이징

## 3. CPU 스케줄링 알고리즘

#### FCFS(First Come First Served)(비선점 스케줄링)
**도착한 순서대로 처리**
- 알고리즘 
	- 선입선처리 
		- 먼저 도착한(큐의 맨 앞에 있는) 스레드 먼저 스케줄링 
- 스케줄링 파라미터 : 스레드 별 큐 도착 시간 
- 스케줄링 타입 : 비선점 스케줄링 
- 스레드 우선순위 : 없음 
- 기아 : 발생하지 않음 
	- 스레드가 오류로 인해 무한 루프를 실행한다면, 뒤 스레드 기아 발생 
- 성능 이슈 
	- 처리율 낮음 
	- **호위 효과(convoy effect) 발생** 
		- 긴 스레드가 CPU를 오래 사용하면, 늦게 도착한 짧은 스레드 오래 대기

![](../../../../image/Pasted%20image%2020241007100617.png)
#### Shortest Job First(SJF; 비선점 스케줄링)
**가장 짧은 스레드 우선 처리**
- 알고리즘 
	- 최단 작업 우선 스케줄링 
	- **실행 시간(예상 실행 시간)이 가장 짧은 스레드 선택** 
	- 스레드가 도착할 때, 
		- 실행 시간이 짧은 순으로 큐 삽입, 큐의 맨 앞에 있는 스레드 선택 
- 스케줄링 파라미터 : **스레드 별 예상 실행 시간** 
	- **스레드의 실행 시간을 아는 것은 불가능**. 비현실적 
- 스케줄링 타입 : **비선점 스케줄링** (cpu를 뺏기지 않음음)
- 스레드 우선순위 : **없음** 
- **기아 : 발생 가능** 
	- 짧은 스레드가 계속 도착하면, 긴 스레드는 실행 기회를 언제 얻을 지 예측할 수 없음 
- 성능 이슈 
	- 가장 짧은 스레드가 먼저 실행되므로 **평균 대기 시간 최소화**
- 문제점 
	- 실행 시간의 **예측이 불가능**하므로 현실에서는 거의 사용되지 않음

![](../../../../image/Pasted%20image%2020241012121203.png)

#### Shortest Remaining Time First(SRTF; 선점 스케줄링) 
**남은 시간이 짧은 스레드가 준비 큐에 들어오면 이를 우선 처리**
- 알고리즘 
	- 최소 잔여 시간 우선 스케줄링 
	- 남은 실행 시간이 가장 짧은 스레드 선택 
	- SJF의 선점 스케줄링 버전 
		- 한 스레드가 끝나거나 **실행 시간이 더 짧은 스레드가 도착할 때**, 남은 실행 시간이 가장 짧은 스레드 선택 
		- 실행 시간에 짧은 순으로 스레드들을 큐에 삽입- > 큐 맨 앞에 있는 스레드 선택 
- 스케줄링 파라미터 : 스레드 별 예상 실행 시간과 남은 실행 시간 값 
	- 이 시간을 아는 것은 불가능. 비현실적 
- 스케줄링 타입 : **선점 스케줄링**(cpu를 뺏길 수 있다)
- 스레드 우선순위 : 없음 
- 기아 : 발생 가능 
	- 짧은 스레드가 계속 도착하면, 긴 스레드는 실행 기회를 언제 얻을 지 모름 
- 성능 이슈 
	- 실행 시간이 **가장 짧은 스레드가 먼저 실행되므로 평균 대기 시간 최소화** 
- 문제점 
	- 실행 시간 예측이 불가능하므로 현실에서는 거의 사용되지 않음

![](../../../../image/Pasted%20image%2020241012122320.png)
#### ==Round-Robin(RR; 선점 스케줄링)==
스레드들을 돌아가면서 할당된 시간(타임 슬라이스)만큼 실행
- 알고리즘 
	- 스레드들에게 공평한 실행 기회를 주기 위해 
	- 큐에 대기중인 스레드들을 타임 슬라이스 주기로 돌아가면서 선택 
	- 스레드는 **도착하는 순서대로 큐에 삽입** 
	- 스레드가 타임 슬라이스를 소진하면 큐 끝으로 이동 
- 스케줄링 파라미터 : **타임 슬라이스** 
- 스케줄링 타입 : 선점 스케줄링 
- 스레드 우선순위 : 없음 
- 기아 : **없음** 
	- 스레드의 우선순위가 없고, 타임 슬라이스가 정해져 있어, 일정 시간 후에 스레드는 반드시 실행 
- 성능 이슈 
	- 공평하고, 기아 현상 없고, 구현이 쉬움 
	- 잦은 스케줄링으로 전체 스케줄링 오버헤드 큼. 특히 타임 슬라이스가 작을 때 더욱 큼 
	- 균형된 처리율 : **타임슬라이스가 크면 FCFS에 가까움, 적으면 SJF/SRTF에 가까움** 늦게 도착한 짧은 스레드는 FCFS보다 빨리 완료되고, 긴 스레드는 SJF보다 빨리 완료됨
- 타임 슬라이스 = 1ms 일 때
	![](../../../../image/Pasted%20image%2020241012123755.png)
- 타임 슬라이스 = 2ms 일 때
	![](../../../../image/Pasted%20image%2020241012123828.png)

#### Priority Scheduling(ready Queue 1개)
우선 순위를 기반으로 하는 스케줄링. 가장 높은 순위의 스레드 먼저 실행
- 알고리즘 
	- 우선순위에 따라 스레드를 실행시키기 위한 목적 
	- 가장 높은 순위의 스레드 선택 
		- 현재 스레드가 종료되거나 더 높은 순위의 스레드가 도착할 때, 가장 높은 순위의 스레드 선택 
	- 모든 스레드에 **고정 우선순위 할당**, 종료 때까지 바뀌지 않음 
	- 도착하는 스레드는 우선순위 순으로 큐에 삽입 
- 스케줄링 파라미터 : 스레드 별 고정 우선순위 
- 스케줄링 타입 : 선점 스케줄링/비선점 스케줄링 
	- 선점 스케줄링 : 더 높은 순위의 스레드가 도착할 때 현재 스레드 강제 중단하고 스케줄링 
	- 비선점 스케줄링 : 현재 실행 중인 스레드가 종료될 때 스케줄링 
- 스레드 우선순위 : 있음 
- 기아 : **발생 가능** 
	- 높은 순위의 스레드가 계속 도착하는 경우, 실행 기회를 언제 얻을 지 예상할 수 없음 
	- 큐 대기 시간에 비례하여 일시적으로 우선순위를 높이는 **에이징 방법으로 해결** 가능 
- 성능 이슈 
	- 높은 우선순위의 스레드일수록 대기 혹은 응답시간 짧음 
- 특징 
	- 스레드별 고정 우선순위를 가지는 실시간 시스템에서 사용
#### Multilevel queue scheduling(MLQ; ready queue 여러개)
스레드와 큐 모두 n개의 우선순위 레벨로 할당, 스레드는 자신의 레벨과 동일한 큐에 삽입 
 높은 순위의 큐에서 스레드 스케줄링, 높은 순위의 큐가 빌 때 아래 순위의 큐에서 스케줄링 
스레드는 다른 큐로 이동하지 못함 
	예) background process, Foreground process
- 설계 의도 
	- 스레드들을 n개의 **우선순위 레벨**로 구분, 레벨이 높은 스레드를 우선 처리하는 목적 
- 알고리즘 
	- **고정된 n 개의 큐** 사용, 각 큐에 **고정 우선순위 할당** 
	- 스레드들의 우선순위도 n개의 레벨로 분류 
	- 각 큐는 나름대로의 기법으로 스케줄링 
	- 스레드는 도착 시 우선순위에 따라 해당 레벨 큐에 삽입. 다른 큐로 이동할 수 없음 
	- 가장 높은 순위의 큐가 빌 때, 그 다음 순위의 큐에서 스케줄링 
- 스케줄링 파라미터 : 스레드의 고정 우선순위 
- 스케줄링 타입 : **비선점/선점 모두 가능** 
	- 비선점 스케줄링 : 현재 실행중인 스레드가 종료할 때 스케줄링 
	- 선점 스케줄링 : 높은 레벨의 큐에 스레드가 도착하면 중단하고 높은 레벨 큐에서 스케줄링 
- 스레드 우선순위 : 있음 
- 기아 : 발생 가능 
	- 높은 순위의 스레드가 계속 도착하는 경우 실행 기회를 언제 얻을 지 예상할 수 없음 
- 성능 이슈와 활용 사례 
	- **스레드의 고정 순위를 가진 시스템에서 활용** 
	- 예) 전체 스레드를 백그라운드 스레드와 포그라운드 스레드의 2개의 그룹으로 구성 
	- 예) 시스템 스레드, 대화식 스레드, 배치 스레드 등 3개의 레벨로 나누고 시스템 스레드를 우선적으로 스케줄링

![](../../../../image/Pasted%20image%2020241012125223.png)

#### Multilevel feedback queue scheduling(MLFQ; 선점/비선점 스케줄링 둘 다 구현 가능)
큐만 n개의 우선순위 레벨을 둠. 스레드는 동일한 우선순위 
스레드는 제일 높은 순위의 큐에 진입하고 큐타임슬라이스가 다하면 아래 레벨의 큐로 이동 
낮은 레벨의 큐에 오래 있으면 높은 레벨의 큐로 이동
- 설계 의도 
	- 1962년에 개발된 알고리즘 
	- 기아를 없애기 위해 여러 레벨의 큐 사이에 스레드 이동 가능하도록 설계 
	- 짧은 스레드와 I/O가 많은 스레드, **대화식 스레드의 우선 처리**. 스레드 평균대기시간 줄임 
- n개의 레벨 큐 
	- n개의 고정 큐. 큐마다 우선순위 다름. **큐마다 서로 다른 스케줄링** 알고리즘 
	- 큐는 준비 상태(Ready 상태)의 스레드 저장 
	- 큐마다 스레드가 머무를 수 있는 큐 타임슬라이스 있음. 낮은 레벨의 큐일수록 더 긴 타임 슬라이스 
	- I/O 집중 스레드(대화식 스레드)는 높은 순위의 큐에 있을 가능성 높음 
- 알고리즘 
	- 스레드는 도착 시 최상위 레벨 큐에 삽입 
	- 가장 높은 레벨 큐에서 스레드 선택. 비어 있으며 그 아래의 큐에서 스레드 선택 
	- 스레드의 CPU-burst가 큐 타임 슬라이스를 초과하면 강제로 아래 큐로 이동시킴 
	- **스레드가 자발적으로 중단한 경우, 현재 큐 끝에 삽입** 
	- **스레드가 I/O로 실행이 중단된 경우, I/O가 끝나면 동일 레벨 큐 끝에 삽입** 
	- **큐에 있는 시간이 오래되면 기아를 막기 위해 하나 위 레벨 큐로 이동** (에이징)
		- 단, 최하위 레벨로 일단 떨어지면 위로 이동 못함
	- 최하위 레벨 큐는 주로 FCFS나 긴 타임 슬라이스의 RR로 스케줄. 스레드들은 다른 큐로 이동 못함 
- 스케줄링 파라미터 : 각 큐의 큐 타임슬라이스 
- 스케줄링 타입 : 선점 스케줄링 
- 스레드 우선 순위 : 없음 
- 기아 : 발생하지 않음, 큐에 대기하는 시간에 오래되면, 더 높은 레벨의 큐로 이동시킴(에이징 기법) 
- 성능 이슈 
	- 짧거나 입출력이 빈번한 스레드, 혹은 대화식 스레드를 높은 레벨의 큐에서 빨리 실행 -> CPU 활용률이 높음

![](../../../../image/Pasted%20image%2020241012125429.png)

---
## 실전 스케줄링 사례

![](../../../../image/Pasted%20image%2020241012130824.png)
![](../../../../image/Pasted%20image%2020241012132530.png)

## 4. 멀티 코어 cpu에서의 스케줄링
- 멀티코어 시스템에서 싱글 코어 CPU 스케줄링을 사용할 때 문제점 
	- (문제 1) 컨텍스트 스위칭 후 오버헤드 문제 
		- 이전에 실행된 적이 없는 코어에 스레드가 배치될 때, 
		- 컨텍스트 스위칭 후, 실행 중에 새로운 스레드의 코드와 데이터가 캐시에 채워지는 긴 경 과 시간 
	- (문제 2) 코어별 부하 불균형 문제 
		- 스레드를 무작위로 코어에 할당하면, 코어마다 처리할 스레드 수의 불균형 발생 
- 컨텍스트 스위칭 후 오버헤드 문제 해결 
	- CPU 친화성(CPU affinity) 적용 
		- 스레드를 동일한 코어에서만 실행하도록 스케줄링 
		- 코어 친화성(Core affinity), CPU 피닝(pinning), 캐시 친화성(cache affinity)라고도 부름 
	- 코어 당 스레드 큐 사용 
- 코어별 부하 불균형 문제 해결 
	- 부하 균등화 기법으로 해결 
		1) 푸시 마이그레이션(push migration) 기법 
			-  감시 스레드가, 짧거나 빈 큐를 가진 코어에 다른 큐의 스레드를 옮겨놓는 기법 
		2) 풀 마이그레이션(pull migration) 기법 
			-  코어가 처리할 스레드가 없게 되면, 다른 코어의 스레드 큐에서 자신이 큐로 가져와 실행 시키는 기법
