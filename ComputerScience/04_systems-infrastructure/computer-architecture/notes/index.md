# Index

## lecture

* [01. 컴퓨터구조 서론 - 폰 노이만 아키텍처와 암달의 법칙 대수학 (Computer Architecture & Amdahl's Law)](./01.%20%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B5%AC%EC%A1%B0%20%EC%84%9C%EB%A1%A0.md) - 폰 노이만 구조의 내장 프로그램 방식, 명령어 인출-해석-실행 사이클, CPI와 MIPS 성능 지표, 병렬 가속 상한을 규정하는 암달의 법칙(Amdahl's Law S = 1 / ((1-f) + f/s))을 인터랙티브 암달 가속기 시뮬레이터로 심층 학습한다.
* [02. 데이터의 표현 - 2의 보수 대수학과 IEEE 754 부동소수점 표준 (Data Representation & IEEE 754)](./02.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%9D%98%20%ED%91%9C%ED%98%84.md) - 기수 변환, 2의 보수(2's Complement) 감산 및 오버플로 조건(C_in ⊕ C_out), IEEE 754 32비트 단정밀도 부동소수점 수식(V = (-1)ˢ × 2ᴱ⁻¹²⁷ × 1.M)을 인터랙티브 부동소수점 변환기로 심층 학습한다.
* [04. 조합 논리 회로](./04.%20%EC%A1%B0%ED%95%A9%20%EB%85%BC%EB%A6%AC%20%ED%9A%8C%EB%A1%9C.md) - 현재 입력만으로 출력을 결정하는 조합 논리 회로
* [05. 순서 논리 회로](./05.%20%EC%88%9C%EC%84%9C%20%EB%85%BC%EB%A6%AC%20%ED%9A%8C%EB%A1%9C.md) - 상태와 클록을 이용해 시간에 따른 동작을 만드는 순서 논리 회로
* [06. 집적 회로](./06.%20%EC%A7%91%EC%A0%81%20%ED%9A%8C%EB%A1%9C.md) - 논리 회로를 집적해 복잡한 기능을 구현하는 집적 회로
* [07. 기억 장치 - 메모리 계층 구조와 캐시 사상·유효 접근 시간(EAT) 대수학 (Memory Hierarchy & Cache EAT)](./07.%20%EA%B8%B0%EC%96%B5%20%EC%9E%A5%EC%B9%98.md) - SRAM과 DRAM의 회로적 차이, 직접 사상(Direct Mapped)과 세트 연관(Set-Associative) 캐시 주소 분해, 적중률(h)에 따른 유효 메모리 접근 시간(EAT = h·t_c + (1-h)·t_m)을 인터랙티브 캐시 EAT 계산기로 심층 학습한다.
* [08. 제어 장치](./08.%20%EC%A0%9C%EC%96%B4%20%EC%9E%A5%EC%B9%98.md) - 명령 해석과 제어 신호 생성으로 데이터 경로를 조정하는 제어 장치
* [09. 중앙 처리 장치 - 5단계 파이프라이닝과 해저드 해결 대수학 (CPU Architecture & Pipelining)](./09.%20%EC%A4%91%EC%95%99%20%EC%B2%98%EB%A6%AC%20%EC%9E%A5%EC%B9%98.md) - CPU 내부 레지스터 조직, 5단계 명령어 파이프라인(IF-ID-EX-MEM-WB)의 처리 시간 수식(T = (k+n-1)τ), 데이터·제어·구조적 해저드(Hazard)와 포워딩 해결책을 인터랙티브 파이프라인 가속기 시뮬레이터로 심층 학습한다.
* [computer-architecture 강의 흐름 지도](./00.%20computer-architecture%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 14개의 순서·쪽수·학습 점검을 연결한다.
