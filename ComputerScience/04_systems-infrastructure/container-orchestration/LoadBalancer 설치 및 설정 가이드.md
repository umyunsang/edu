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
title: LoadBalancer 설치 및 설정 가이드
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/4단계 시스템 실전 인터페이스|4단계 시스템 실전 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/컨테이너 오케스트레이션 인터페이스|컨테이너 오케스트레이션 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]], [[ComputerScience/00_graph-interfaces/bridges/오픈소스 delivery 브리지|오픈소스 delivery 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]]
related:: [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/MLFlow 과제|MLFlow 과제]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/07_professional-humanities/degree-portfolio/PDF_인쇄_완전가이드|PDF_인쇄_완전가이드]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week0 - Orientation|Week0 - Orientation]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week6 - GitHub Actions|Week6 - GitHub Actions]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week3 - Teamwork & Collaborative Development|Week3 - Teamwork & Collaborative Development]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Banker/Banker Algorithm 구현 과제|Banker Algorithm 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week5 - Open & Inner Source Software Delivery|Week5 - Open & Inner Source Software Delivery]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/computer-networks/1. 통신과 컴퓨터네트워크/통신과 컴퓨터 네트워크|통신과 컴퓨터 네트워크]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week2 - Plan, Track & Visualize|Week2 - Plan, Track & Visualize]], [[ComputerScience/04_systems-infrastructure/computer-networks/0. Quiz/기말 암기 정리|기말 암기 정리]], [[ComputerScience/04_systems-infrastructure/computer-networks/16. 보안/네트워크 보안|네트워크 보안]], [[ComputerScience/04_systems-infrastructure/computer-networks/13. 전송 계층/전송 계층|전송 계층]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week1 - Metrics That Matter|Week1 - Metrics That Matter]], [[ComputerScience/05_software-engineering/aioss-open-source-delivery/md/Week4 - Asynchronous Work|Week4 - Asynchronous Work]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/컨테이너 오케스트레이션 근거 인덱스|컨테이너 오케스트레이션 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/os|os]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/container-orchestration/cpu|cpu]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
### LoadBalancer 설치 및 설정 가이드

#### 1. **Kube-Proxy 설정 변경**

`kube-proxy`의 설정을 편집하여 `strictARP`를 활성화합니다.

```bash
kubectl edit configmap -n kube-system kube-proxy
```

- `ipvs` 섹션에서 `strictARP`를 `true`로 변경:
    
    ```yaml
    ipvs:
      excludeCIDRs: null
      minSyncPeriod: 0s
      scheduler: ""
      strictARP: true  # 기존 false에서 true로 변경
    ```
    

---

#### 2. **MetalLB 다운로드 및 설치**

1. **MetalLB 아카이브 다운로드**
    
    ```bash
    wget https://github.com/metallb/metallb/archive/refs/tags/v0.12.1.tar.gz
    ```
    
2. **아카이브 압축 해제 및 디렉토리 이동**
    
    ```bash
    tar -xvzf v0.12.1.tar.gz
    cd metallb-0.12.1/manifests
    ```
    
3. **MetalLB 네임스페이스 및 구성 적용**
    
    ```bash
    kubectl apply -f namespace.yaml
    kubectl apply -f metallb.yaml
    ```
    
4. **MetalLB 리소스 상태 확인**
    
    ```bash
    kubectl get all -n metallb-system
    ```
    

---

#### 3. **Layer 2 Configuration 설정**

1. **`example-layer2-config.yaml` 파일 편집**
    
    - `addresses`를 네트워크 대역에 맞게 수정:
        
        ```yaml
        addresses:
          - 192.168.11.230-192.168.11.233  # 네트워크 대역에 맞게 수정
        ```
        
2. **Layer 2 Configuration 적용**
    
    ```bash
    kubectl apply -f example-layer2-config.yaml
    ```
    

---

#### 4. **LoadBalancer 서비스 생성**

1. **테스트 파드 생성**
    
    ```bash
    kubectl run myweb3 --image nginx:1.12 --labels "app=web-svc3" --port 80
    ```
    
2. **`load.yaml` 파일 작성**
    
    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: sv3
    spec:
      selector:
        app: web-svc3
      type: LoadBalancer
      ports:
      - port: 80
        targetPort: 80
    ```
    
3. **LoadBalancer 서비스 적용**
    
    ```bash
    kubectl apply -f load.yaml
    ```
    
4. **서비스 확인**
    
    ```bash
    kubectl get svc
    ```
    

---
