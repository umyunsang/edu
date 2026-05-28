---
aliases: []
course: operating-systems
created: '2024-09-09'
date: '2024-09-09'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-2
source: ''
status: seedling
tags:
- cs/systems
- type/lecture
title: 컴퓨터 시스템과 OS
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/운영체제 인터페이스|운영체제 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/1. 프로세스 구성과 동작|1. 프로세스 구성과 동작]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/1. 제어 장치의 기능|1. 제어 장치의 기능]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다|쿠다]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/4. 컴퓨터 명령어|4. 컴퓨터 명령어]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/2. 제어 장치의 종류|2. 제어 장치의 종류]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/3. 레지스터|3. 레지스터]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/3. 명령어 사이클|3. 명령어 사이클]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/5. 파이프 라이닝|5. 파이프 라이닝]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/3. 캐시 기억 장치|3. 캐시 기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/2. 주기억 장치|2. 주기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/4. 프로세서 제어|4. 프로세서 제어]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/5. 주소 지정 방식|5. 주소 지정 방식]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/4. 가상 기억 장치|4. 가상 기억 장치]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/1. 기억 장치 시스템의 개요|1. 기억 장치 시스템의 개요]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/1. Why Parallelism|1. Why Parallelism]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/6. CISC와 RISC|6. CISC와 RISC]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/애플 M4 CPU/애플 M4 CPU|애플 M4 CPU]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/4. 디지털 코드|4. 디지털 코드]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/04_systems-infrastructure/computer-networks/2. 네트워크 분류와 계층 모델/네트워크 분류와 계층 모델|네트워크 분류와 계층 모델]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/05_software-engineering/database-systems/3. DB 시스템/DB 시스템|DB 시스템]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/05_software-engineering/database-systems/2. 관리 시스템/관리 시스템|관리 시스템]], [[ComputerScience/04_systems-infrastructure/computer-networks/8. 무선통신 시스템/무선통신 시스템|무선통신 시스템]], [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/5. 에러 검출 코드|5. 에러 검출 코드]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/03_Hadoop|03_Hadoop]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/2. 정수 표현|2. 정수 표현]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/04-ParallelComputation|04-ParallelComputation]], [[ComputerScience/04_systems-infrastructure/computer-networks/7. LAN의 특징과 규격/LAN의 특징과 규격|LAN의 특징과 규격]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/9. 네트워크 계층/네트워크 계층|네트워크 계층]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/2. 산술 논리 연산 장치|2. 산술 논리 연산 장치]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/1. 진법과 진법 변환|1. 진법과 진법 변환]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/3. 실수 표현|3. 실수 표현]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/3. 카르노 맵|3. 카르노 맵]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍|TCP와 소켓 프로그래밍]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/1. 논리 게이트|1. 논리 게이트]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/2. 불 대수|2. 불 대수]], [[ComputerScience/04_systems-infrastructure/computer-architecture/중간 시험 범위|중간 시험 범위]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로|4. 조합 논리 회로]], [[certifications/information-processing/실기/C언어 실기 오답노트|오답노트]], [[ComputerScience/04_systems-infrastructure/computer-networks/5. 통신망과 특징/통신망과 특징|통신망과 특징]], [[ComputerScience/04_systems-infrastructure/computer-networks/12. 네트워크 계층 작업과 프로토콜/네트워크 계층 작업과 프로토콜|네트워크 계층 작업과 프로토콜]], [[ComputerScience/04_systems-infrastructure/computer-networks/6. 데이터 링크 계층의 작업/데이터 링크 계층의 작업 (2 계층)|데이터 링크 계층의 작업 (2 계층)]], [[ComputerScience/04_systems-infrastructure/computer-networks/3. 신호 처리/신호 처리|신호 처리]], [[ComputerScience/04_systems-infrastructure/computer-networks/11. 인터넷 프로토콜 라우팅 알고리즘/인터넷 프로토콜(IP)|인터넷 프로토콜(IP)]], [[ComputerScience/04_systems-infrastructure/computer-networks/4. 유선 및 무선 데이터 전송/유선 및 무선 데이터 전송|유선 및 무선 데이터 전송]], [[certifications/체크리스트|체크리스트]], [[ComputerScience/04_systems-infrastructure/computer-networks/16. 보안/네트워크 보안|네트워크 보안]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/운영체제 지식그래프|운영체제]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/운영체제 지식그래프|운영체제]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/운영체제 근거 인덱스|운영체제 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/cpu|cpu]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/fcfs|fcfs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/메모리 관리|메모리 관리]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/srtf|srtf]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/파일 시스템 관리|파일 시스템 관리]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
## 컴퓨터 하드웨어 구성

![[operating-systems__컴퓨터 시스템과 OS__컴퓨터 하드웨어 구성.png]]
- CPU(Central Processing Unit) 
	- 프로그램 코드(기계 명령)를 해석하여 실행하는 중앙처리장치 
	- 컴퓨터의 가장 핵심 장치 
	- 전원이 공급될 때 작동 시작, 메모리에 적재된 프로그램 실행 
- 메모리 
	- CPU에 의해 실행되는 프로그램 코드와 데이터가 적재되는 공간 
	- 반도체 메모리 RAM 
	- 프로그램은 실행되기 위해 반드시 메모리에 적재되어야 함 
- 캐시 메모리(Cache Memory) 
	- CPU 처리속도가 메모리 속도에 비해 빠르게 발전 -> 느린 메모리 때문에 CPU의 대기 시 간이 늘게 되었음 
	- CPU의 프로그램 실행 속도를 높이기 위해, CPU와 메모리 사이에 소량의 빠른 메모리(고가의 메모리)를 설치하게 되었음 
- 온칩 캐시(on-chip) – CPU 내부에 만들어진 캐시, 오늘날 대부분 온칩 캐시 
- 옵칩 캐시(off-chip) – CPU 외부에 설치되는 캐시 
- 캐시 메모리가 있는 경우 CPU는 캐시 메모리에서만 프로그램 실행 
	- 실행하고자하는 프로그램과 데이터는 메모리에 먼저 적재되고 다시 캐시로 옮겨져야 함 
	- 캐시는 용량이 작기 때문에 현재 실행할 코드와 데이터의 극히 일부분만 저장
- 버스(bus) 
	- 하드웨어들이 데이터를 주고받기 위해 0과 1의 디지털 신호가 지나가는 여 러 가닥의 선을 다발로 묶어 부르는 용어 
	 - 버스의 종류(버스에 지나다니는 정보에 따라) 
		 - 주소 버스(address bus) - 주소 신호가 지나다니는 버스(도로) 
		 - 데이터 버스(data bus) - 데이터 신호가 지나다니는 버스(도로) 
		 - 제어 버스(control bus) - 제어 신호가 지나다니는 버스(도로) 
	- 주소 
		- 메모리, 입출력 장치나 저장 장치 내에 있는 저장소(레지스터들)에 대한 번지 
		- 0번지에서 시작하는 양수 
		- 주소 버스는 주소 값이 전달되는 여러 선의 다발 
		- CPU는 메모리나 입출력 장치에 값을 쓰거나 읽을 때 반드시 주소를 발생시킴
	- 시스템 버스(system bus) 
		- CPU, 캐시 메모리, 메모리 등 빠른 하드웨어들 사이에 데이터 전송 
		- 고속도로에 비유 
	- 입출력 버스(I/O bus) 
		- 상대적으로 느린 입출력 장치들로부터 입출력 데이터 전송 
		- 일반 도로에 비유
- I/O controllers & control circuit 
	- 입출력 장치들을 제어하기 위한 여러 하드웨어 
		- 입출력 장치에게 명령 하달 
		- 메모리와 입출력 장치 사이에 혹은 CPU와 입출력 장치 사이에 데이터 전달 중계
		- DMAC(Direct Memory Access Controller

#### 컨텍스트(Context)
- 컨텍스트(문맥이라고도 표현)
	- 한 프로그램이 실행 중인 일체의 상황 혹은 상황 정보
		- 메모리
			- 프로그램 코드와 데이터, 스택, 동적할당 받아 저장한 값
		- CPU 레지스터들의 값
			- PC에는 코드의 주소
			- SP에는 스택의 주소
			- 다른 레지스터는 이전의 실행 결과나 현재 실행에 사용되는 데이터 들
	- 축소 정의
		- 현재 CPU에 들어 있는 레지스터의 값들
			- 메모리에 저장된 상황 정보는 그대로 있다
- 컨텍스트 스위칭
	- 발생
		- CPU가 현재 프로그램의 실행을 중지하고 다른 프로그램을 실행할 때
	- 과정
		- 현재 실행중인 프로그램의 컨텍스트(CPU레지스터들의 값)를 메모리에 저장
		- 새로 실행시킬 프로그램의 저장된 컨텍스트(CPU레지스터들의 값)를 CPU에 복귀

## 2. 컴퓨터 시스템의 계층 구조와 운영체제 인터페이스
#### 컴퓨터 시스템 계층 구조
![[operating-systems__컴퓨터 시스템과 OS__컴퓨터 시스템 계층 구조.png]]
- 응용프로그램이 하드웨어를 사용하고자 할 때 
	- 반드시 운영체제에게 요청 -> 운영체제가 대신하여 하드웨어 조작 
	- 유일한 요청 방법 – ==시스템 호출(system call)==

#### 운영체제의 전체 기능
- 프로세스와 스레드 관리 
	- 프로세스/스레드의 실행, 일시 중단, 종료, 스케줄링, 컨텍스트 스위칭, 동기화 
- 메모리 관리 
	- 프로세스나 스레드에게 메모리 할당, 메모리 반환, 다른 프로세스/스레드로부터의 메모리 보호 
	- 메모리를 하드 디스크의 영역까지 확장하는 가상 메모리 기술 
- 파일 관리 혹은 파일 시스템 관리 
	- 파일 생성, 저장, 읽기, 복사, 삭제, 이동, 파일 보호 
- 장치 관리 
	- 키보드, 마우스, 프린터 등 입출력 장치, 하드 디스크 등 저장 장치 제어 
	- 입출력 
- 사용자 인터페이스 
	- 라인 기반 명령 입출력 창, 마우스와 그래픽 사용 GUI 인터페이스 제공 
- 네트워킹 
	- 네트워크 인지, 연결, 닫기, 데이터 송수신 
- 보호 및 보안 
	- 바이러스나 웜, 멀웨어(malware), 해킹 등의 외부 공격이나 무단 침입으로부터 보호

#### 운영체제 커널 인터페이스 : 시스템 호출과 인터럽트
- 커널이 제공하는 2개 인터페이스 : 시스템 호출과 인터럽트 
	- 응용프로그램과 하드웨어 사이의 중계 역할 
- **시스템 호출(system call)** 
	- 커널과 응용프로그램 사이의 인터페이스 
	- 응용프로그램에서 **커널 기능을 사용할 수 있는 유일한 방법** 
	- 시스템 호출 라이브러리를 통해 다양한 시스템 호출 함수 제공 
		- 시스템 호출 라이브러리는 운영체제 패키지에 포함됨 
		- 예) 파일 읽기, 메모리 할당, 프로세스 정보 보기, 프로세스 생성 등
- **인터럽트(interrupt)** 
	- 커널과 하드웨어 장치 사이의 인터페이스 
	- **장치들이 입출력 완료, 타이머 완료 등을 CPU에게 알리는 하드웨어적 방법** 
		- 인터럽트를 알리는 하드웨어 신호가 직접 CPU에 전달 
	- 인터럽트가 발생하면 
		- CPU는 하는 일을 중단하고 인터럽트 서비스 루틴 실행 
		- 인터럽트 서비스 루틴은 대부분 디바이스 드라이버 내에 있음 
			- 예) 키를 입력하면 커널의 키보드 인터럽트 서비스 루틴 실행, 키를 읽어 커널 버퍼에 저장 
		- 인터럽트 서비스 루틴은 커널 영역에 적재 
		- 인터럽트 서비스 루틴의 실행을 마치면 하던 작업 계속 

## 3. 커널과 시스템 호출

#### 사용자 공간과 커널 공간
- 운영체제는 컴퓨터 메모리를 두 공간으로 분리 
	- **사용자 공간**(user space) : 모든 응용프로그램들이 나누어 사용하는 공간 
		- 응용프로그램들이 적재되는 공간 
	- **커널 공간**(kernel space) : 커널만 사용할 수 있는 공간 
		- 커널 코드, 커널 데이터 등 커널에 의해 배타적으로 사용되는 공간 
		- 디바이스 드라이버 포함 
- 분리 이유 
	- 커널 코드와 데이터를 악의적인 응용프로그램이나 코딩 실수로부터 지키기 위함

#### 사용자 모드와 커널 모드
- CPU는 사용자 모드와 커널 모드 중 한 모드로 실행 
	- CPU 내부에 모드 상태를 나타내는 ‘모드 레지스터’ 있음 
- **사용자 모드**(user mode) 
	- CPU의 모드 비트 = 1 
	- CPU는 사용자 공간에 있는 코드나 데이터를 액세스하는 중 
	- CPU의 커널 공간 접근 불허 -> 응용프로그램으로부터 커널영역 보호 
	- ==특권 명령==(privileged instruction) 실행 불허 
		- 특권 명령 – 입출력 장치 등 하드웨어나 시스템 중단 등 시스템 관련 처리를 위해 설계된 특별한 명령 
- **커널 모드**(kernel mode, supervisor mode) 
	- CPU의 모드 비트 = 0 
	- CPU가 커널 공간에서 실행하는 중, 혹은 사용자 코드를 실행하는 중 
	- 특권 명령 사용 가능
- 사용자 모드에서 커널 모드로 전환하는 경우 
	- 오직 2 가지 경우 - 시스템 호출과 인터럽트 발생 
- 시스템 호출 
	- 시스템 호출을 실행하는 특별한 기계 명령에 의해 진행 
		- 예) int 0x80/sysenter/trap/syscall 등 CPU마다 다름 
	- 기계 명령이 CPU의 모드 비트를 커널 모드로 전환 
- 인터럽트 
	- CPU가 인터럽트를 수신하면 커널 모드로 자동 전환 
		- 인터럽트 서비스 루틴이 커널 공간에 있기 때문 
	- CPU는 인터럽트 서비스 루틴 실행 
	- 인터럽트 서비스 루틴이 끝나면 CPU는 사용자 모드로 자동 전환
	![[operating-systems__컴퓨터 시스템과 OS__사용자 모드와 커널 모드 2.png]]
- 사용자 모드와 커널 모드의 비교
	![[operating-systems__컴퓨터 시스템과 OS__사용자 모드와 커널 모드.png]]

#### 특권 명령
- 특권 명령 
	- 입출력 장치로부터의 입출력(I/O), 시스템 중단, 컨텍스트 스위칭, 인터럽트 금지 등 특별한 목적으로 설계된 CPU 명령 
	- 커널 모드에서만 실행 
- 특권 명령 종류 
	- I/O 명령 
		- 하드웨어 제어 및 장치로부터의 입출력 
		- 사례) in eax, 300 ; I/O 포트 300 번지에서 값을 읽어 eax 레지스터에 저장 
		- out 301, eax ; eax 레지스터에 있는 값을 I/O 포트 301 번지에 쓰기 
	- Halt 명령 
		- CPU의 작동을 중지시키는 명령. CPU를 유휴 상태로 만듦 
	- 인터럽트 플래그를 켜고 끄는 명령 
		- CPU 내에 있는 인터럽트 플래그 비트를 제어하여 CPU가 인터럽트를 허용하거나 무시하도록 지시 
		- 사례) cli/sti 명령
	- 타이머 설정 명령 
	- 컨텍스트 스위칭 명령 
	- 메모리 지우기 명령 
	- 장치 상태 테이블 수정 등의 명령

#### 퀴즈 - 특권 명령에 대한 이해 (읽어 보기)
- Q. 다음 보기 중 어떤 명령이 특권 명령일까? 이유도 설명하라
	1) 사용자 모드에서 커널 모드로 전환시키는 명령 
		- 특권 명령이 아니다
		- 왜냐하면 시스템 호출을 위해 모든 응용 프로그램에게 허용되어야 하는 명령이기 때문이 다
	2) 시계 읽기 
		- 특권 명령이 아니다
		- 모든 응용프로그램에서 시계를 읽을 수 있어야 하기 때문이다
	3) 가상 메모리에서 메모리 지우기
		- 특권 명령이 아니다
		- 프로그램이 자신의 메모리 부분을 지우는 것은 다른 프로세스의 영역을 침범하지 않기 때문이다
	4) 인터럽트 끄기 
		- 특권 명령이다
		- 인터럽트를 끄는 행위는 CPU 내부의 인터럽트 플래그(IF)를 끄는 행동으로, 인터럽트가 꺼지면 외부에서 인터럽트가 발생해도 CPU는 인터럽트의 발생을 체크하지 않는다. 한 프로그램이 인터럽트를 끄면 CPU는 인터럽트가 꺼진 상태로 계속 있게 되어, CPU가 다른 프로그램을 실행하더라도 인터럽트를 받을 수 없게 된다. CPU는 한 프로그램에게 독점될 수 없기 때문에 이 명령은 특권 명령으로 응용프로그램이 실행할 수 없다

#### 커널의 실체
- 커널은 부팅 시에 커널 공간에 적재된 함수들과 데이터 집합 
	- 커널은 컴파일된 바이너리 형태, 하드디스크 특정 영역에 저장, 부팅 시에 커널 공간의 메모리에 적재 
- 커널 코드는 함수들의 집합 
	- 커널의 존재 - 커널 모드에서 실행되는 함수들과 데이터들의 집합
- 커널은 스스로 실행되는 프로세스인가? NO 
	- 커널은 함수들의 단순 집합, 시스템 호출을 통해 호출되는 함수들 
	- 커널이 스케줄링한다(x) 
		- 커널 프로세스가 실행되면서 주기적으로 스케줄링한다(x) 
		- 시스템 호출과 인터럽트 서비스 루틴에 의해 커널 내 스케줄러 함수가 호출되어 실행(0) 
- 커널은 실행 중이다? NO 
	- 커널은 프로세스도 스레드도 아니므로 NO 
	- 커널이 실행 중이다(x) 
		- 응용프로그램이 시스템 호출을 하여 커널 코드를 실행하고 있다(0) 
		- 인터럽트가 발생하여 인터럽트 서비스 루틴이 실행되고 있다(0) 
- 커널은 스택이나 힙을 가지는가? NO 
	- 커널은 스택이나 힙을 가지는 주체가 아니다. 그러므로 NO 
	- 스택이나 힙을 가지는 주체는 프로세스나 스레드  
	- 스레드마다 사용자 스택과 커널 스택 소유 
		- 스레드가 생성될 때 프로세스의 사용자 공간에 사용자 스택 할당 
			- 사용자 코드가 실행되는 동안 활용 
		- 스레드가 생성될 때 커널 공간에 커널 스택 할당 
		- 스레드가 시스템 호출로 커널 코드를 실행할 때 스택으로 활용
#### 응용프로그램 빌딩
- 라이브러리(library) 
	- 응용프로그램에서 활용하도록 미리 작성된 함수들, 컴파일되어 바이너리 형태로 제공되는 파일 
	- 개발자는 라이브러리 활용없이 응용프로그램 작성 불가능 
- 응용프로그램이 활용하는 라이브러리는 2가지 유형 
	- **표준 라이브러리(Standard Library)**
		- 사용자가 작성하기 힘든 함수 제공 
		- 운영체제나 컴퓨터 하드웨어에 상관없이 이름과 사용법 동일 
			- 운영체제나 하드웨어, 컴파일러에 관계없이 호환 
	- **시스템 호출 라이브러리(System Call Library)** 
		- 시스템 호출 함수들 포함
		- 시스템 호출 함수들은 시스템 호출을 진행하여 커널 모드로 바꾸고 커널로 진입하 여 커널에 만들어진 함수 실행(커널의 다양한 기능 수행) 
		- 운영체제마다 시스템 호출 함수의 이름이 서로 다름
		- 시스템 호출 함수를 **커널 AP**I(Application Programming Interface)라고 부름

	![[operating-systems__컴퓨터 시스템과 OS__응용프로그램 빌딩.png]]

#### 시스템 호출
- 시스템 호출 
	- 사용자 공간의 코드에서 커널 서비스를 요청하는 과정 
		- 사용자 공간의 코드가 커널 함수를 호출하는 과정 
		- 커널 콜(kernel call), **트랩**(trap) = (소프트웨어 인터럽트) 로도 불림 
		- 응용프로그램에서 커널 기능을 활용하도록 만들어 놓은 기능 

![[operating-systems__컴퓨터 시스템과 OS__시스템 호출.png]]

#### 시스템 호출에 따른 비용 정리
- **시스템 호출은 함수 호출에 비해 많은 시간 비용** 
	- 시스템 호출을 많이 할수록 프로그램 실행 속도 저하
- 시스템 호출은 필연적이지만, 시스템 호출 횟수를 줄여야 응용프로그램의 실행 시간이 짧아지고, 시스템 입장에서 더 많은 프로그램을 실행시킬 수 있는 시간 확보 -> 시스템의 처리율 향상
	![[operating-systems__컴퓨터 시스템과 OS__시스템 호출에 따른 비용 정리.png]]

## 4. 운영체제와 인터럽트
#### 인터럽트
- 인터럽트 
	- 입출력 장치들이 비동기적 사건을 CPU에게 알리는 행위 
		- 비동기적이란 예정되지 않거나 발생시간을 예측할 수 없는 사건 
		- (키보드 입력, 네트워크로부터 데이터 도착 등) 
	- 하드웨어 인터럽트 
		- 장치들이 어떤 상황 발생을 CPU에게 알리는 하드웨어 신호 
		- CPU는 인터럽트를 수신하면 인터럽트 서비스 루틴 실행 
	- 소프트웨어 인터럽트 
		- CPU 명령으로 발생시키는 인터럽트 
		- 하드웨어 인터럽트를 수신한 것과 동일하게 처리
- 인터럽트가 없다면 
	- 다중프로그래밍 운영체제의 구현은 사실상 거의 불가능
