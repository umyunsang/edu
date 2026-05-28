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
title: quiz.yaml 파일을 클러스터에 적용
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/4단계 시스템 실전 인터페이스|4단계 시스템 실전 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/컨테이너 오케스트레이션 인터페이스|컨테이너 오케스트레이션 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]]
related:: [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/04_systems-infrastructure/computer-networks/14. TCP와 소켓 프로그래밍/TCP와 소켓 프로그래밍|TCP와 소켓 프로그래밍]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/9. 네트워크 계층/네트워크 계층|네트워크 계층]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/컨테이너 오케스트레이션 근거 인덱스|컨테이너 오케스트레이션 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/os|os]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/cpu|cpu]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---

#### 1. **파드 생성**

ClusterIP 서비스를 위한 파드 생성:

```bash
kubectl run web --image nginx:1.12 --labels "app=websvr" --port 80
```

---

#### 2. **ClusterIP 서비스 매니페스트 작성**

ClusterIP 서비스 매니페스트 파일(`cluster.yaml`) 작성:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: svc  # 서비스 이름
spec:
  selector:
    app: websvr  # 'websvr' 라벨을 가진 파드를 선택
  type: ClusterIP  # 클러스터 내부에서만 접근 가능한 서비스
  ports:
  - port: 80        # 서비스가 노출하는 포트
    targetPort: 80  # 파드 내부 컨테이너 포트
    protocol: TCP
```

---

#### 3. **ClusterIP 서비스 적용 및 확인**

1. **매니페스트 파일 적용**:
    
    ```bash
    kubectl apply -f cluster.yaml
    ```
    
2. **서비스 및 파드 상태 확인**:
    
    ```bash
    kubectl get svc
    kubectl get po -o wide
    ```
    
    - 서비스의 클러스터 IP 및 포트를 확인합니다.
3. **서비스의 상세 정보 확인**:
    
    ```bash
    kubectl describe svc svc
    ```
    
    - `Endpoints` 섹션에서 서비스와 연결된 파드의 IP 주소를 확인합니다.

---

#### 4. **클라이언트 파드 생성**

클러스터 내부에서 서비스를 테스트하기 위한 클라이언트 파드 생성:

```bash
kubectl run clientpod --image nginx
```

---

#### 5. **클라이언트 파드에서 서비스 테스트**

1. **클라이언트 파드에 접속**:
    
    ```bash
    kubectl exec -it clientpod -- bash
    ```
    
2. **서비스 통신 테스트**:
    
    - **클러스터 IP 사용**:
        
        ```bash
        curl http://<서비스의-클러스터-IP-주소>
        ```
        
    - **서비스 이름 사용**:
        
        ```bash
        curl svc
        ```
        

---

#### 6. **클러스터 내 DNS 서비스 확인**

1. **CoreDNS 관련 서비스 확인**:
    
    ```bash
    kubectl get svc -n kube-system --show-labels
    ```
    
2. **CoreDNS 파드 확인**:
    
    ```bash
    kubectl get po -n kube-system | grep coredns
    ```
    
    - CoreDNS는 클러스터 내의 DNS 서비스로, 서비스 이름을 IP 주소로 변환하는 역할을 합니다.

---

#### 7. **ClusterIP의 구조**

- **동작 원리**:
    - ClusterIP 서비스는 클러스터 내부에서만 접근 가능하며, 외부에서 직접 접근할 수 없습니다.
    - CoreDNS를 통해 서비스 이름을 사용하여 파드와 통신할 수 있습니다.
- **DNS 역할**:
    - 클러스터 내에서 서비스 이름을 통해 통신이 가능하며, CoreDNS가 이를 처리합니다.

---

#### 8. **Endpoints 확인**

서비스와 연결된 파드의 IP 주소를 확인:

```bash
kubectl describe svc svc
```

- `Endpoints` 섹션에서 서비스와 매핑된 파드의 IP 주소를 확인할 수 있습니다.

---

###### Quiz. **아래의 조건에 해당하는 deployment 와  service 설정을  quiz.yaml  파일에 설정하세요.**
```
deployment 이름:  quiz-deploy
pod 의 초기 개수: 5
pod 의 label:   nginx-testbed
pod 의 이름:  www
pod 의 이미지: nginx:1.14
pod 의 port 번호:  80 / TCP
  
service 의 이름: quiz-svc
service 의 type: clusterip
targetport: 80
```

>[! deployment와 service설정을 quiz..yaml파일로 생성]
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quiz-deploy
spec:
  replicas: 5  # 초기 파드 개수
  selector:
    matchLabels:
      app: nginx-testbed  # 파드의 라벨
  template:
    metadata:
      labels:
        app: nginx-testbed  # 파드의 라벨
    spec:
      containers:
      - name: www  # 파드 이름
        image: nginx:1.14  # 파드의 이미지
        ports:
        - containerPort: 80  # 파드의 포트 번호

---
apiVersion: v1
kind: Service
metadata:
  name: quiz-svc  # 서비스 이름
spec:
  selector:
    app: nginx-testbed  # 'nginx-testbed' 라벨을 가진 파드를 선택
  type: ClusterIP  # 클러스터 내부에서만 접근 가능한 서비스
  ports:
    - port: 80        # 서비스가 노출하는 포트
      targetPort: 80  # 파드 내부 컨테이너 포트
      protocol: TCP

```

>[!적용 확인]
```bash
# quiz.yaml 파일을 클러스터에 적용
kubectl apply -f quiz.yaml
# 생성된 서비스 목록 확인
kubectl get svc
# 생성된 파드 목록 및 상세 정보 확인
kubectl get po -o wide

# 클라이언트 역할을 할 파드 생성
kubectl run clientpod --image nginx
# 클라이언트 파드 터미널 접속
kubectl exec -it clientpod -- bash
# curl로 연결 확인
@clientpod:/~$ curl quiz-svc
```
---
