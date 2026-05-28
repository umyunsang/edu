---
aliases: []
course: computer-networks
created: '2024-10-16'
date: '2024-10-16'
semester: 2-2
source: ''
status: seedling
tags:
- cs/systems
- type/lecture
title: LAN의 특징과 규격
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/4단계 시스템 실전 인터페이스|4단계 시스템 실전 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/컴퓨터네트워크 인터페이스|컴퓨터네트워크 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/computer-networks/6. 데이터 링크 계층의 작업/데이터 링크 계층의 작업 (2 계층)|데이터 링크 계층의 작업 (2 계층)]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]]
related:: [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/9. 네트워크 계층/네트워크 계층|네트워크 계층]], [[ComputerScience/04_systems-infrastructure/computer-networks/2. 네트워크 분류와 계층 모델/네트워크 분류와 계층 모델|네트워크 분류와 계층 모델]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍|TCP와 소켓 프로그래밍]], [[ComputerScience/04_systems-infrastructure/computer-networks/8. 무선통신 시스템/무선통신 시스템|무선통신 시스템]], [[ComputerScience/04_systems-infrastructure/computer-networks/11. 인터넷 프로토콜 라우팅 알고리즘/인터넷 프로토콜(IP)|인터넷 프로토콜(IP)]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/04_systems-infrastructure/computer-networks/12. 네트워크 계층 작업과 프로토콜/네트워크 계층 작업과 프로토콜|네트워크 계층 작업과 프로토콜]], [[ComputerScience/04_systems-infrastructure/computer-networks/4. 유선 및 무선 데이터 전송/유선 및 무선 데이터 전송|유선 및 무선 데이터 전송]], [[ComputerScience/04_systems-infrastructure/computer-networks/5. 통신망과 특징/통신망과 특징|통신망과 특징]], [[ComputerScience/04_systems-infrastructure/computer-networks/3. 신호 처리/신호 처리|신호 처리]], [[ComputerScience/04_systems-infrastructure/computer-networks/10. 라우팅 알고리즘/라우팅 알고리즘|라우팅 알고리즘]], [[ComputerScience/04_systems-infrastructure/computer-networks/16. 보안/네트워크 보안|네트워크 보안]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 퀴즈|기말 퀴즈]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/Routing Information Protocol (RIP)|Routing Information Protocol (RIP)]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/중간 퀴즈|중간 퀴즈]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/4. 프로세서 제어|4. 프로세서 제어]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/2. 제어 장치의 종류|2. 제어 장치의 종류]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/1. 제어 장치의 기능|1. 제어 장치의 기능]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/1. 프로세스 구성과 동작|1. 프로세스 구성과 동작]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/5. 주소 지정 방식|5. 주소 지정 방식]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/3. 레지스터|3. 레지스터]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/2. 주기억 장치|2. 주기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/4. 컴퓨터 명령어|4. 컴퓨터 명령어]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/5. 파이프 라이닝|5. 파이프 라이닝]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/4. 가상 기억 장치|4. 가상 기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로|4. 조합 논리 회로]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/5. 에러 검출 코드|5. 에러 검출 코드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/6. CISC와 RISC|6. CISC와 RISC]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다|쿠다]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/1. 논리 게이트|1. 논리 게이트]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/1. 기억 장치 시스템의 개요|1. 기억 장치 시스템의 개요]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/3. 캐시 기억 장치|3. 캐시 기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/2. 정수 표현|2. 정수 표현]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/중간 시험 범위|중간 시험 범위]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/3. 명령어 사이클|3. 명령어 사이클]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/1. Why Parallelism|1. Why Parallelism]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/2. 불 대수|2. 불 대수]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/3. 카르노 맵|3. 카르노 맵]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/4. 디지털 코드|4. 디지털 코드]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/1. 진법과 진법 변환|1. 진법과 진법 변환]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/3. 실수 표현|3. 실수 표현]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/2. 산술 논리 연산 장치|2. 산술 논리 연산 장치]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/애플 M4 CPU/애플 M4 CPU|애플 M4 CPU]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컴퓨터네트워크 지식그래프|컴퓨터네트워크]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컴퓨터네트워크 지식그래프|컴퓨터네트워크]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/컴퓨터네트워크 근거 인덱스|컴퓨터네트워크 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/computer-networks/kakao id|kakao id]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/computer-networks/s03 301 01호실|s03 301 01호실]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/computer-networks/네트워크 계층|네트워크 계층]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/computer-networks/전송 계층|전송 계층]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/computer-networks/cpu|cpu]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
## 1. LAN 계층
#### LAN의 계층 구조
![](../../../../image/Pasted%20image%2020241016173259.png)
- 5계층이나 OSI 7계층은 인터넷이 보급되고 난 이후에 만들어진 구분법
- LAN의 계층을 세분화 하면 **논리 연결 제어**(LLC)와 **매체 접근 제어**(MAC)로 나눌 수 있음
- 논리 연결 제어(Logical Link Control)는 LLC로, 매체 접근 제어(Media Acess Control)는 MAC로 표시
- 논리 연결 제어(LLC)는 802.2에 정의되어 있으며, 802.3부터 802.22까지가 매체 접근 제어(MAC)에 해당

#### 논리 연결 제어(LLC)
- **논리 연결 제어(LLC)** 하부계층은 **두 노드를 논리적으로 연결**하는 계층
- LLC는 **프레임**을 송수신하는 방식을 정하고 **상위계층**(네트워크 계층)에 있는 프로토콜과의 인터페이스를 제공
- LLC 계층의 중요한 역할은 **프레임을 에러없이** 전달하면서도 **프레임 전송률**을 높이는 것
- LLC 계층에 사용되는 프로토콜에 따라 연결 서비스 혹은 비연결 서비스를 지원하며 연결 서비스의 유지관리를 담당
- LLC는 사용방식에 따라 다음과 같이 3가지의 종류로 나뉨
	![](../../../../image/Pasted%20image%2020241028152241.png)

#### 매체 접근 제어(MAC)
![](../../../../image/Pasted%20image%2020241028152835.png)
- 매체 접근 제어(MAC) 하부계층은 **여러 종류 LAN**의 연결형태, 데이터 **전송방법, 헤더**들을 정의하는 계층
- ==**MAC에 대한 설명의 대부분은 이더넷을 기준으로 함**==
- 이더넷은 별형태 + **CSMA/CD** 프로토콜로 정의. CSMA/CD는 경쟁방식의 프로토콜
	- CSMA/CD : carrier sense multiple access with collision detection
	- CSMA/CA : WIFI
- MAC 주소 -> 6바이트로 구성
- **==IP 주소는 사용자가 변경할 수 있는 값이지만, MAC 주소는 통신기기가 만들어질 때 제조사가 임의로 부여한 값으로 변경 불가==**
- **MAC 주소는 주민등록번호**처럼 한번 만들어지면 바꿀 수 없는 값이기 때문에 **물리적 주소**라고 부름
	- IP 주소의 예 : 핸드폰번호나 집 주소 
	- **MAC 주소는 일관성이나 표준성이 없다**
- 모든 통신기기는 MAC 주소와 IP 주소를 같이 가지고 있음

## 2. 이더넷과 토큰 링
#### 이더넷에 대하여
- **이더넷(Ethernet)은** 컴퓨터 네트워크 기술의 하나로, 일반적으로 LAN에서 가장 많이 활용되는 기술 규격이며 **IEEE 802.3**에 정의되어 있음
- 이더넷이 만들어진 초기에는 버스형태로 구성되었으며, 현재는 스타형태로 구성
- 이더넷에서 각 호스트를 유선으로 연결하는 장치를 허브라 부름
- 유선 뿐 아니라 무선으로 연결할 수 있는 장치의 규격이 와이파이며 흔히 무선 공유기라 부름

####  CSMA/CD
![](../../../../image/Pasted%20image%2020241028154955.png)
- CSMA/CD에 참여하는 모든 호스트는 선이 사용 중인지 아닌지를 계속해서 듣고 있음(**스누핑**) 
- 선을 사용하는 호스트가 없을 때 전송을 시작
- 선을 사용하지 않을 때 호스트들이 경쟁적으로 데이터를 보내려 하기 때문에 경쟁방식이라 부름

![](../../../../image/Pasted%20image%2020241028155031.png)
- 선을 사용하고 있지 않을 경우, 두 개 이상의 호스트가 거의 동시에 데이터를 보내면 신호가 충돌 할 수 있음
	- 충돌 상황이 잦은경우 : 인터넷트래픽이 높다
- CSMA/CD 방식에서 충돌이 발생하는 경우 데이터 전송을 즉각 멈춤
- 전체 호스트에게 충돌이 일어났음을 알리는 신호를 보냄
- ==**충돌을 일으킨 호스트들은 무작위 수를 만들어 일정시간 기다린 후 재 전송**==

#### 토큰 링
![](../../../../image/Pasted%20image%2020241028155501.png)
- 토큰링에서 호스트들을 원형으로 연결 -> 토큰-token이라는 빈 패킷이 한쪽 방향으로 계속 회전
- 전송을 하려는 호스트가 있다면 빈 토큰을 가져가서 토큰에 주소와 데이터를 채운 후 전송
- **목적지 주소**에 토큰이 도착하면 해당 호스트는 내용을 **복사한 후 계속 토큰을 전달**
- 토큰이 회전하여 처음 데이터를 보냈던 곳에 돌아오면 
	-  해당 호스트는 토큰을 회수한 후 내용물을 지우고 **빈 패킷을 다시 옆으로 전송** 
	-  ==**토큰에 데이터를 채워서 보낸 호스트가 데이터를 지우고 빈 패킷을 만듬**==
- 여러개의 호스트들이 동시에 데이터를 보내는 경우 
	-  토큰링 방식에서는 하나의 호스트가 데이터를 보내고 지운 후 빈 토큰을 옆으로 전송 
	-  여러개의 호스트들은 순서대로 돌아가면서 데이터를 보냄 
	-  예약을 통한 충돌회피 방식

## 3. 데이터 링크 계층 프레임 분석
#### HDLC 프레임
![](../../../../image/Pasted%20image%2020241028160442.png)
- 데이터 링크 계층에는 **HDLC**, LAP, LAPB, LAPD, LAPF, ATM, PPP와 같은 많은 종류의 프로토콜들이 있음
- HDLC 프로토콜은 **동기식** 전송을 사용하며, 동기식 중 비트 방식을 사용. **비트 스터핑** 사용
- HDLC는 통신방식으로 **유니케스트, 멀티 케스트, 브로트 케스트**를 모두 지원
- HDLC 프로토콜은 **흐름제어**로 슬라이딩 윈도우 프로토콜을 사용하며, Go-Back-N ARQ와 Selective Reapeat ARQ를 모두 지원
- **HDLC 프레임은 데이터의 길이가 정해져 있지 않아 포스트앰블(Flag)가 꼭 필요**
- **FCS**(Frame Check Sequence)는 오류제어를 위해 사용되는 필드이며 CRC-16을 사용
	- **트레일러, 테일**라고도 함
- HDLC 프레임은 Information frame. Supervisory frame, Unnumbered frame의 총 3가지 종류가 있음
	- 정보(Information) 프레임(I frame)은 사용자 정보와 제어 정보를 모두 포함하는 일반 프레임을 의미
	- 감시(Supervisory) 프레임(S frame)은 제어 정보만 가지고 있는 프레임
	- 비번호(Unnumbered) 프레임(U frame)은 **연결 관리정보**를 포함하는 프레임을 의미

#### 이더넷 프레임
![](../../../../image/Pasted%20image%2020241030164634.png)
- 그림은 LAN의 대부분을 차지하는 이더넷의 프레임 구조
- Data + padding 필드에는 LLC 계층으로 부터 받은 **LLC 프레임**(2.5계층 데이터)이 들어감
- 이더넷 프레임은 **데이터의 크기**는 최대 1500바이트로 **한정**
- 상위에서 받은 **데이터가 46바이트 보다 작은 경우에는 패딩(padding)을 붙여서 46바이트(전체 64바이트)로 맞춤**
- 이더넷 프레임의 크기는 최소 64바이트에서 최대 1518바이트로 한정되어 있음. 프레임이 언제 끝날지 예측이 가능하기 때문에 **포스트앰블을 사용하지 않음**
- 프리앰블은 프리앰블 7바이트와 SFD 1바이트, 총 8바이트로 구성
	- 프리앰블은 10101010을 7번 전송하는데 이것이 통신의 시작을 알림
	- 7바이트의 프리앰블이 끝나고 난 후, 프레임의 시작을 알려주는 것이 SFD(10101011)
	- (10101010) * 7 + SFD(10101011)를 보내고 데이터를 전송
- ==**Destination address = 목적지 주소 Source address = 호스트 주소 (6바이트의 MAC 주소)**==
- Length/Type 필드의 경우 데이터가 1500바이트 이하이면 데이터의 길이(Length)를 나타냄
- Length/Type 필드의 값이 1500이상이 Type으로 해석
- FCS 필드는 이더넷 프레임의 오류 탐색을 위한 것으로 CRC-32 사용
	- 트레일러, 테일

---
