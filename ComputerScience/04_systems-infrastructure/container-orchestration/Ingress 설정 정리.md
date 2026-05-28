---
aliases: []
course: container-orchestration
created: '2025-01-17'
date: '2025-01-17'
semester: elective
source: ''
status: seedling
tags:
- cs/devops
- skill/docker
- type/lecture
title: Ingress 설정 정리
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/4단계 시스템 실전 인터페이스|4단계 시스템 실전 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/컨테이너 오케스트레이션 인터페이스|컨테이너 오케스트레이션 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]]
related:: [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍|TCP와 소켓 프로그래밍]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/컨테이너 오케스트레이션 근거 인덱스|컨테이너 오케스트레이션 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/os|os]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/cpu|cpu]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
### **Ingress 설정 정리**

### 1. **Ingress-NGINX 설치**

#### 1.1. **Ingress Controller 설치**

- Bare Metal 환경에서 Ingress-NGINX 설치:

```bash
wget https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.2/deploy/static/provider/baremetal/deploy.yaml
```

- `deploy.yaml` 파일의 **366행**에서 `type` 수정:

```yaml
type: LoadBalancer  # NodePort에서 LoadBalancer로 변경
```

- 수정한 파일 적용:

```bash
kubectl apply -f deploy.yaml
```

#### 1.2. **설치 상태 확인**

```bash
kubectl get ns                   # 네임스페이스 확인
kubectl get all -n ingress-nginx # 리소스 확인
kubectl get svc -n ingress-nginx # 서비스 정보 확인
```

- **EXTERNAL-IP** 확인 (예: `192.168.11.231`):

```plaintext
NAME                        TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                      AGE
ingress-nginx-controller    LoadBalancer   10.102.195.216   192.168.11.231   80:32218/TCP,443:30533/TCP   101s
```

---

### 2. **디플로이먼트와 서비스 생성**

#### 2.1. **기본 NGINX 서비스**

```bash
kubectl create deploy nginx-main --image=nginx
kubectl expose deploy nginx-main --name nginx-main-svc --port 80
```

#### 2.2. **색상별 NGINX 서비스**

- **블루 버전**:

```bash
kubectl create deploy nginx-blue --image=thekoguryo/nginx-hello:blue
kubectl expose deploy nginx-blue --name nginx-blue-svc --port 80
```

- **그린 버전**:

```bash
kubectl create deploy nginx-green --image=thekoguryo/nginx-hello:green
kubectl expose deploy nginx-green --name nginx-green-svc --port 80
```

#### 2.3. **HTTPD 서비스**

```bash
kubectl create deploy httpd-main --image=httpd
kubectl expose deploy httpd-main --name httpd-main-svc --port 80
```

- **상태 확인**:

```bash
kubectl get deploy,svc,po
```

---

### 3. **Ingress 설정**

#### 3.1. **Ingress 매니페스트 작성**

`ig.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myingress
spec:
  rules:
  - host: www.uys1998.com
    http:
      paths:
      - pathType: Prefix
        path: "/"
        backend:
          service:
            name: nginx-main-svc
            port:
              number: 80
      - pathType: Prefix
        path: "/blue"
        backend:
          service:
            name: nginx-blue-svc
            port:
              number: 80
      - pathType: Prefix
        path: "/green"
        backend:
          service:
            name: nginx-green-svc
            port:
              number: 80
  - host: www.guru2025.co.kr
    http:
      paths:
      - pathType: Prefix
        path: "/"
        backend:
          service:
            name: httpd-main-svc
            port:
              number: 80
```

#### 3.2. **Ingress 리소스 적용**

```bash
kubectl apply -f ig.yaml
kubectl get ing                  # 생성된 Ingress 확인
kubectl describe ing myingress   # 상세 정보 확인
```

---

### 4. **로컬 네임 해석 설정**

#### 4.1. **호스트 파일 수정**

- **Windows**: `C:\Windows\System32\drivers\etc\hosts`
    
- **Linux/Mac**: `/etc/hosts`
    
- 파일의 **마지막 줄**에 다음 내용 추가:
    
```
192.168.11.231 www.uys1998.com www.guru2025.co.kr
```

- 수정 후 저장.

---

### 5. **Ingress 동작 확인**

- **URL 접속**:
    - [http://www.uys1998.com](http://www.uys1998.com/) → 기본 `nginx-main`
    - [http://www.uys1998.com/blue](http://www.uys1998.com/blue) → `nginx-blue`
    - [http://www.uys1998.com/green](http://www.uys1998.com/green) → `nginx-green`
    - [http://www.guru2025.co.kr](http://www.guru2025.co.kr/) → `httpd-main`

---
