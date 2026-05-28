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
title: NodePort 서비스 설정 및 가이드
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/4단계 시스템 실전 인터페이스|4단계 시스템 실전 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/컨테이너 오케스트레이션 인터페이스|컨테이너 오케스트레이션 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]]
related:: [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍|TCP와 소켓 프로그래밍]], [[ComputerScience/04_systems-infrastructure/computer-networks/5. 통신망과 특징/통신망과 특징|통신망과 특징]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/04_systems-infrastructure/computer-networks/9. 네트워크 계층/네트워크 계층|네트워크 계층]], [[ComputerScience/04_systems-infrastructure/computer-networks/11. 인터넷 프로토콜 라우팅 알고리즘/인터넷 프로토콜(IP)|인터넷 프로토콜(IP)]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/컨테이너 오케스트레이션 근거 인덱스|컨테이너 오케스트레이션 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/os|os]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/cpu|cpu]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
### NodePort 서비스 설정 및 가이드

#### 1. **파드 생성**

NodePort 서비스를 위한 파드 생성:

```bash
kubectl run myweb2 --image nginx:1.12 --labels "app=web-svc2" --port 80
```

---

#### 2. **NodePort 서비스 매니페스트 작성**

NodePort를 사용하는 서비스 매니페스트 파일(`nodeport.yaml`) 작성:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: svc-nodeport  # 서비스 이름
spec:
  selector:
    app: web-svc2  # 'web-svc2' 라벨을 가진 파드를 선택
  type: NodePort  # 클러스터 외부에서 접근 가능한 서비스
  ports:
  - port: 80          # 서비스가 노출하는 포트
    targetPort: 80    # 파드 내부 컨테이너 포트
    nodePort: 30080   # 노드의 특정 포트 (30000~32767 범위에서 선택 가능)
    protocol: TCP
```

---

#### 3. **NodePort 서비스 적용 및 확인**

1. **매니페스트 파일 적용**:
    
    ```bash
    kubectl apply -f nodeport.yaml
    ```
    
2. **서비스 상태 확인**:
    
    ```bash
    kubectl get svc
    kubectl describe svc svc-nodeport
    ```
    
    - `NodePort`에 설정된 포트를 확인합니다.

---

#### 4. **서비스 테스트**

1. **서비스 및 파드 확인**:
    
    ```bash
    kubectl get svc
    kubectl get po -o wide
    ```
    
    - 각 노드의 IP 주소와 `NodePort`를 확인.
2. **외부 접속 테스트**:
    
    - 브라우저 또는 `curl` 명령어를 사용하여 서비스 확인:
        
        ```bash
        curl http://<노드의-외부-IP>:30080
        ```
        
    - `<노드의-외부-IP>`는 클러스터 노드의 IP 주소.

---

#### 5. **클라이언트 파드 생성**

클러스터 내부에서 통신을 테스트하기 위한 클라이언트 파드 생성:

```bash
kubectl run clientpod --image nginx
```

---

#### 6. **클라이언트 파드에서 서비스 테스트**

1. **클라이언트 파드에 접속**:
    
    ```bash
    kubectl exec -it clientpod -- bash
    ```
    
2. **서비스 통신 테스트**:
    
    - **클러스터 IP 사용**:
        
        ```bash
        curl http://<서비스의-클러스터-IP>
        ```
        
    - **NodePort 사용**:
        
        ```bash
        curl http://<노드의-외부-IP>:30080
        ```
        

---

#### 7. **NodePort 서비스의 구조**

- **동작 원리**:
    - 클러스터의 모든 노드에서 지정된 `NodePort`를 통해 서비스를 노출.
    - 외부에서 클러스터 노드의 IP 주소와 `NodePort`를 사용하여 접근.
- **DNS 역할**:
    - 클러스터 내부에서는 CoreDNS를 통해 서비스 이름으로 통신.
    - 외부에서는 클러스터 IP 대신 노드 IP와 `NodePort`를 사용.

---

#### 8. **Endpoints 확인**

서비스와 연결된 파드의 IP를 확인:

```bash
kubectl describe svc svc-nodeport
```

- `Endpoints` 섹션에서 서비스와 매핑된 파드의 IP를 확인할 수 있습니다.

---
