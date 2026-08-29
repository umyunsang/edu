# Index

## lecture

* [01. 리눅스 개요와 디렉터리 계층 구조(FHS) - 유닉스 철학, 핵심 CLI 명령어와 경로 체계](./01.%20%EB%A6%AC%EB%88%85%EC%8A%A4%20%EA%B0%9C%EC%9A%94%EC%99%80%20%EB%94%94%EB%A0%89%ED%84%B0%EB%A6%AC%20%EA%B3%84%EC%B8%B5%20%EA%B5%AC%EC%A1%B0%28FHS%29.md) - 리눅스 OS 커널과 유닉스(Unix) 설계 철학('모든 것은 파일이다'), 파일 시스템 계층 표준(FHS: /, /bin, /etc, /var, /proc), 절대/상대 경로, 그리고 필수 CLI 명령어(ls, cd, pwd, cp, mv, rm, find)를 인터랙티브 FHS 탐색기로 학습한다.
* [02. 파일 시스템 권한 체계와 계정 관리 - 8진수 권한(rwx), umask, 특수 권한(SUID·SGID·Sticky)과 계정 관리](./02.%20%ED%8C%8C%EC%9D%BC%20%EC%8B%9C%EC%8A%A4%ED%85%9C%20%EA%B6%8C%ED%95%9C%20%EC%B2%B4%EA%B3%84%EC%99%80%20%EA%B3%84%EC%A0%95%20%EA%B4%80%EB%A6%AC.md) - 리눅스 파일 소유권(User/Group/Others), 8진수 권한(chmod rwx), 파일 생성 기본 마스크(umask), 특수 권한(SetUID, SetGID, Sticky Bit), 그리고 사용자 및 그룹 관리 명령어(useradd, usermod, /etc/passwd, /etc/shadow)를 인터랙티브 권한 계산기로 학습한다.
* [03. Vim 텍스트 편집기와 3대 모드 - 일반·입력·명령행 모드, 버퍼 조작과 정규표현식 치환](./03.%20Vim%20%ED%85%8D%EC%8A%A4%ED%8A%B8%20%ED%8E%B8%EC%A7%91%EA%B8%B0%EC%99%80%203%EB%8C%80%20%EB%AA%A8%EB%93%9C.md) - Vim의 3대 동작 모드(일반 모드 Command, 입력 모드 Insert, 명령행 모드 Last-Line/Ex), 커서 이동 및 텍스트 조작 단축키(yy, dd, p, u, Ctrl+r), 정규표현식 문자열 치환(:%s/old/new/g), 그리고 .vimrc 환경 설정을 인터랙티브 Vim 키 매핑 훈련기로 학습한다.
* [04. 셸 환경과 Bash 스크립트 프로그래밍 - 표준 입출력 리다이렉션, 파이프(|), 변수와 제어문](./04.%20%EC%85%B8%20%ED%99%98%EA%B2%BD%EA%B3%BC%20Bash%20%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D.md) - 리눅스 셸(Bash)의 환경 변수, 표준 입출력(stdin, stdout, stderr) 리다이렉션(<, >, >>, 2>&1), 파이프라인(|), 조건문(if-else, test/[[]]), 반복문(for, while), 그리고 매개변수($1, $#, $?)를 인터랙티브 Bash 파이프라인 시뮬레이터로 학습한다.
* [05. 프로세스 관리와 백그라운드 실행·시그널 - nohup, jobs/fg/bg, kill 시그널(SIGKILL/SIGTERM)과 systemd](./05.%20%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4%20%EA%B4%80%EB%A6%AC%EC%99%80%20%EB%B0%B1%EA%B7%B8%EB%9D%BC%EC%9A%B4%EB%93%9C%20%EC%8B%A4%ED%96%89%C2%B7%EC%8B%9C%EA%B7%B8%EB%84%90.md) - 리눅스 프로세스 모니터링(ps aux, top), 백그라운드 실행(&)과 세션 종료 방지(nohup, SIGHUP 무시), 작업 제어(jobs, fg, bg), 핵심 시그널(SIGINT 2, SIGKILL 9, SIGTERM 15), cron 정기 예약, 그리고 systemd 서비스 유닛 제어를 인터랙티브 프로세스 관리자로 학습한다.
* [CLI와 리눅스 디렉터리](./02.%20CLI%EC%99%80%20%EB%A6%AC%EB%88%85%EC%8A%A4%20%EB%94%94%EB%A0%89%ED%84%B0%EB%A6%AC.md) - 문자열 명령으로 시스템과 상호작용하고 파일 계층을 탐색하는 기초
* [Flask 회원 기능](./08.%20Flask%20%ED%9A%8C%EC%9B%90%20%EA%B8%B0%EB%8A%A5.md) - 회원 가입·로그인·세션·로그아웃 흐름을 구성하는 Flask 기초
* [Flask와 REST](./07.%20Flask%EC%99%80%20REST.md) - Flask의 요청·응답 처리와 리소스·HTTP 메서드 중심 REST API 기초
* [linux 강의 흐름 지도](./00.%20linux%20%EA%B0%95%EC%9D%98%20%ED%9D%90%EB%A6%84%20%EC%A7%80%EB%8F%84.md) - 원본 PDF 16개의 순서·쪽수·학습 점검을 연결한다.
* [SSH와 리눅스 네트워킹](./05.%20SSH%EC%99%80%20%EB%A6%AC%EB%88%85%EC%8A%A4%20%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%82%B9.md) - 원격 로그인·보안 전송과 가상머신 네트워크 구성의 기초
* [vi와 텍스트 편집](./03.%20vi%EC%99%80%20%ED%85%8D%EC%8A%A4%ED%8A%B8%20%ED%8E%B8%EC%A7%91.md) - 일반·입력·명령 모드를 오가며 파일을 편집하는 vi의 기초
* [데이터베이스 기초](./10.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4%20%EA%B8%B0%EC%B4%88.md) - SQL·MySQL·테이블 조작을 중심으로 관계형 데이터베이스를 다루는 기초
* [데이터베이스 요약](./08.%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4%20%EC%9A%94%EC%95%BD.md) - 공유 데이터를 통합 관리하는 데이터베이스와 관계형·NoSQL 모델의 개요
* [도커 요약](./10.%20%EB%8F%84%EC%BB%A4%20%EC%9A%94%EC%95%BD.md) - 이미지·컨테이너·Dockerfile·레지스트리로 실행 환경을 묶는 도커 개요
* [도커 컨테이너](./13.%20%EB%8F%84%EC%BB%A4%20%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88.md) - 이미지·컨테이너의 실행 수명주기와 네트워크·볼륨·Compose 관리 기초
* [리눅스 소개와 설치](./01.%20%EB%A6%AC%EB%88%85%EC%8A%A4%20%EC%86%8C%EA%B0%9C%EC%99%80%20%EC%84%A4%EC%B9%98.md) - 리눅스 커널·배포판·가상 환경 설치의 기초
* [사용자·그룹과 권한](./06.%20%EC%82%AC%EC%9A%A9%EC%9E%90%C2%B7%EA%B7%B8%EB%A3%B9%EA%B3%BC%20%EA%B6%8C%ED%95%9C.md) - 다중 사용자 리눅스에서 root·그룹·rwx 권한을 나누어 관리하는 방법
* [셸과 셸 프로그래밍](./04.%20%EC%85%B8%EA%B3%BC%20%EC%85%B8%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D.md) - 명령·프로그램 실행 인터페이스와 Bash 스크립트의 작성·실행 기초
* [오픈소스 기반 프로젝트 관리](./90.%20%EC%98%A4%ED%94%88%EC%86%8C%EC%8A%A4%20%EA%B8%B0%EB%B0%98%20%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%EA%B4%80%EB%A6%AC.md) - 소스코드 공유·라이선스·커뮤니티와 프로젝트 관리의 관계를 정리
* [프로세스 기초](./11.%20%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4%20%EA%B8%B0%EC%B4%88.md) - 프로세스 상태 코드와 작업 제어·백그라운드 명령을 중심으로 보는 기초
* [프로세스 요약](./09.%20%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4%20%EC%9A%94%EC%95%BD.md) - 실행 중인 프로그램의 식별자·조회·종료·백그라운드 실행 기초
