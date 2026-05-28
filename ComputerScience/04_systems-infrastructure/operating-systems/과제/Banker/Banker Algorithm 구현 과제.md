---
aliases: []
course: operating-systems
created: '2024-12-01'
date: '2024-12-01'
semester: 2-2
source: ''
status: seedling
tags:
- cs/systems
- type/project
title: 'Banker Algorithm 구현 과제'
type: project
updated: '2026-05-05'
---

domain:: [[ComputerScience/04_systems-infrastructure/시스템 인프라 인터페이스|시스템 인프라 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/2단계 전공 핵심 인터페이스|2단계 전공 핵심 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/운영체제 인터페이스|운영체제 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/시스템 운영 브리지|시스템 운영 브리지]]
up:: [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/프로세스와 프로세스 관리|프로세스와 프로세스 관리]]
prerequisites:: [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/과제_CacheFriendly코딩실습|과제_CacheFriendly코딩실습]], [[ComputerScience/04_systems-infrastructure/linux/1. 리눅스의 기본|1. 리눅스의 기본]]
related:: [[ComputerScience/04_systems-infrastructure/operating-systems/과제/MemoryAlloc/메모리 할당 알고리즘 구현 과제|메모리 할당 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/Page/페이지 교체 알고리즘 구현 과제|페이지 교체 알고리즘 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SJF/SJF CPU 스케줄링 구현 과제|SJF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/SRTF/SRTF CPU 스케줄링 구현 과제|SRTF CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/FCFS/FCFS CPU 스케줄링 구현 과제|FCFS CPU 스케줄링 구현 과제]], [[ComputerScience/04_systems-infrastructure/operating-systems/과제/sum/sum.c|sum.c]], [[ComputerScience/04_systems-infrastructure/operating-systems/4. 스레드와 멀티테스킹/스레드와 멀티테스킹|스레드와 멀티테스킹]], [[ComputerScience/04_systems-infrastructure/operating-systems/7. 교착상태/교착상태|교착상태]], [[ComputerScience/04_systems-infrastructure/operating-systems/5. CPU 스케줄링/CPU 스케줄링|CPU 스케줄링]], [[ComputerScience/04_systems-infrastructure/operating-systems/2. 컴퓨터 시스템과 운영체제/컴퓨터 시스템과 OS|컴퓨터 시스템과 OS]], [[ComputerScience/04_systems-infrastructure/operating-systems/1. OS의 시작과 발전/OS의 시작과 발전|OS의 시작과 발전]], [[ComputerScience/04_systems-infrastructure/operating-systems/9. 페이징 메모리 관리/페이징 메모리 관리|페이징 메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/시험/기말 정리|기말 정리]], [[ComputerScience/04_systems-infrastructure/operating-systems/6. 스레드 동기화/스레드 동기화|스레드 동기화]], [[ComputerScience/04_systems-infrastructure/operating-systems/11. 파일 시스템 관리/파일 시스템 관리|파일 시스템 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/8. 메모리관리/메모리 관리|메모리 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/10. 가상 메모리/가상 메모리|가상 메모리]], [[ComputerScience/04_systems-infrastructure/operating-systems/12. 저장 장치 관리/대용량 저장 장치 관리|대용량 저장 장치 관리]], [[ComputerScience/04_systems-infrastructure/operating-systems/3. 프로세스와 프로세스 관리/3장문제|3장문제]], [[ComputerScience/04_systems-infrastructure/linux/8. 데이터베이스|8. 데이터베이스]], [[ComputerScience/04_systems-infrastructure/linux/3. 리눅스 셸|3. 리눅스 셸]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/3. 명령어 사이클|3. 명령어 사이클]], [[ComputerScience/04_systems-infrastructure/container-orchestration/LoadBalancer 설치 및 설정 가이드|LoadBalancer 설치 및 설정 가이드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/3. 레지스터|3. 레지스터]], [[ComputerScience/04_systems-infrastructure/linux/5. 플라스크|5. 플라스크]], [[ComputerScience/04_systems-infrastructure/container-orchestration/Ingress 설정 정리|Ingress 설정 정리]], [[ComputerScience/04_systems-infrastructure/computer-architecture/중간 시험 범위|중간 시험 범위]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/1. 기억 장치 시스템의 개요|1. 기억 장치 시스템의 개요]], [[ComputerScience/04_systems-infrastructure/linux/7. 회원 가입 및 로그인|7. 회원 가입 및 로그인]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/4. 조합 논리 회로|4. 조합 논리 회로]], [[ComputerScience/04_systems-infrastructure/linux/0. 리눅스 소개|0. 리눅스 소개]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/2. 산술 논리 연산 장치|2. 산술 논리 연산 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/1. 진법과 진법 변환|1. 진법과 진법 변환]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/1. 제어 장치의 기능|1. 제어 장치의 기능]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/1. 프로세스 구성과 동작|1. 프로세스 구성과 동작]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/쿠다|쿠다]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/6. CISC와 RISC|6. CISC와 RISC]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/1. Why Parallelism|1. Why Parallelism]], [[ComputerScience/04_systems-infrastructure/linux/2. 리눅스 VI|2. 리눅스 VI]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/2. 정수 표현|2. 정수 표현]], [[ComputerScience/04_systems-infrastructure/linux/4. 리눅스 권한|4. 리눅스 권한]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/3. 실수 표현|3. 실수 표현]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/4. 디지털 코드|4. 디지털 코드]], [[ComputerScience/04_systems-infrastructure/linux/9. 프로세스|9. 프로세스]], [[ComputerScience/04_systems-infrastructure/computer-architecture/1. 데이터의 표현/5. 에러 검출 코드|5. 에러 검출 코드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/4. 프로세서 제어|4. 프로세서 제어]], [[ComputerScience/04_systems-infrastructure/linux/6. REST|6. REST]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/3. 캐시 기억 장치|3. 캐시 기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/2. 주기억 장치|2. 주기억 장치]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/2. 불 대수|2. 불 대수]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/3. 카르노 맵|3. 카르노 맵]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/애플 M4 CPU/애플 M4 CPU|애플 M4 CPU]], [[ComputerScience/04_systems-infrastructure/linux/10. 도커|10. 도커]], [[ComputerScience/04_systems-infrastructure/computer-architecture/5. 기억 장치/4. 가상 기억 장치|4. 가상 기억 장치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/NodePort 서비스 설정 및 가이드|NodePort 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/CUDA 프로그램 연습 및 CUDA API 이해|CUDA 프로그램 연습 및 CUDA API 이해]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/5. 주소 지정 방식|5. 주소 지정 방식]], [[ComputerScience/04_systems-infrastructure/computer-architecture/3. 중앙 처리 장치/4. 컴퓨터 명령어|4. 컴퓨터 명령어]], [[ComputerScience/04_systems-infrastructure/computer-architecture/2. 디지털 논리 회로/1. 논리 게이트|1. 논리 게이트]], [[ComputerScience/04_systems-infrastructure/container-orchestration/ClusterIP 서비스 설정 및 가이드|ClusterIP 서비스 설정 및 가이드]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/5. 파이프 라이닝|5. 파이프 라이닝]], [[ComputerScience/04_systems-infrastructure/computer-architecture/4. 제어 장치/2. 제어 장치의 종류|2. 제어 장치의 종류]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커|도커]], [[ComputerScience/04_systems-infrastructure/container-orchestration/도커 기초|도커 기초]], [[ComputerScience/04_systems-infrastructure/container-orchestration/쿠버네티스 설치|쿠버네티스 설치]], [[ComputerScience/04_systems-infrastructure/container-orchestration/파드(Pod)|파드(Pod)]], [[certifications/information-processing/필기/1. 프로그래밍 언어 활용|1. 프로그래밍 언어 활용]], [[certifications/체크리스트|체크리스트]], [[certifications/information-processing/실기/C언어 실기 오답노트|오답노트]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/운영체제 지식그래프|운영체제]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/운영체제 지식그래프|운영체제]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/운영체제 근거 인덱스|운영체제 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/cpu|cpu]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/fcfs|fcfs]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/메모리 관리|메모리 관리]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/srtf|srtf]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/operating-systems/파일 시스템 관리|파일 시스템 관리]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]]

---
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX 50
  
typedef struct Command {
    int Play_Process;
    int *resource;
    struct Command *next;
} Command;
  
typedef struct Queue {
    Command *front, *rear;
    int Q_count;
} Queue;
  
void enQueue(Queue *q, Command item, int R_num) {
    int i;
    Command *temp = (Command*)malloc(sizeof(Command));
    temp->resource = (int*)malloc(sizeof(int)*R_num);
  
    temp->Play_Process = item.Play_Process;
    for(i = 0 ; i < R_num ; i++) {
        temp->resource[i] = item.resource[i];
    }
  
    temp->next = NULL;
  
    if(q->Q_count == 0) {
        q->front = temp;
        q->rear = temp;
    }
    else {
        q->rear->next = temp;
        q->rear = temp;
    }
    q->Q_count++;
}
  
Command deQueue(Queue *q, int R_num) {
    Command *temp = q->front;
    Command item;
    int i;
  
    item.resource = (int*)malloc(sizeof(int)*R_num);
    item.Play_Process = temp->Play_Process;
    q->front = q->front->next;
  
    for(i = 0 ; i < R_num ; i++) {
        item.resource[i] = temp->resource[i];
    }
    q->Q_count--;
    return item;
}
  
void inti_Q(Queue *q) {
    q->front = NULL;
    q->rear = NULL;
    q->Q_count = 0;
}
  
int Av_Re(int Av[], int Re_Ne[], int c) {
    int i;
    for(i = 0 ; i < c ; i++) {
        if(Re_Ne[i] > Av[i])
            return 0;
    }
    return 1;
}
  
void Array_copy1(int temp[], int origin[], int num) {
    int i;
    for(i = 0 ; i < num ; i++) {
        temp[i] = origin[i];
    }
}
  
void Array_copy2(int (*temp)[50], int (*origin)[50], int P_num, int R_num) {
    int i, j;
    for(i = 0 ; i < P_num ; i++) {
        for(j = 0 ; j < R_num ; j++) {
            temp[i][j] = origin[i][j];
        }
    }
}
  
int Request_Check(int (*All)[50], int (*Ne)[50], int Av[], Command Re, int P_num, int R_num) {
    int i, j;
    int All_temp[MAX][MAX];
    int N_temp[MAX][MAX];
    int Av_temp[MAX];
    int flag[MAX];
    int Next = 1;
  
    for(i = 0 ; i < P_num ; i++) {
        flag[i] = 0;
    }
  
    if(Av_Re(Ne[Re.Play_Process], Re.resource, R_num) == 0) {
        return 2;
    }
  
    Array_copy2(All_temp, All, P_num, R_num);
    Array_copy2(N_temp, Ne, P_num, R_num);
    Array_copy1(Av_temp, Av, R_num);
  
    for(i = 0 ; i < R_num ; i++) {
        N_temp[Re.Play_Process][i] -= Re.resource[i];
        Av_temp[i] -=  Re.resource[i];
    }
    for(i = 0 ; i < R_num ; i++) {
        All_temp[Re.Play_Process][i] += Re.resource[i];
    }
  
    while(Next) {
        Next = 0;
        for(i = 0 ; Next == 0 && i < P_num ; i++) {
            if(flag[i] == 0) {
                if(Av_Re(Av_temp, N_temp[i], R_num) == 0) {
                }
                else {
                    Next = 1;
                    for(j = 0 ; j < R_num ; j++)
                        Av_temp[j] += All_temp[i][j];
                    flag[i] = 1;
                }
            }
        }
    }
  
    for(i = 0 ; i < P_num ; i++) {
        if(flag[i] == 0) {
            return 0;
        }
    }
  
    for(i = 0 ; i < R_num ; i++) {
        Ne[Re.Play_Process][i] -= Re.resource[i];
        Av[i] -=  Re.resource[i];
    }
    for(i = 0 ; i < R_num ; i++) {
        All[Re.Play_Process][i] += Re.resource[i];
    }
    return 1;
}
  
void Release(int (*All)[50], int (*Ne)[50], int Av[], Command Re, int P_num, int R_num) {
    int i;
    for(i = 0 ; i < R_num ; i++) {
        All[Re.Play_Process][i] -= Re.resource[i];
    }
    for(i = 0 ; i < R_num ; i++) {
        Ne[Re.Play_Process][i] += Re.resource[i];
        Av[i] += Re.resource[i];
    }
}
  
int main() {
    int Process_num, Resource_num;
    int Resource_MAX[MAX];
    int Available[MAX];
    int Allocated[MAX][MAX];
    int Max[MAX][MAX];
    int Need[MAX][MAX];
  
    int i, j, temp, Q_count;
    char word[8];
  
    Queue Wait_Queue;
    Command C_temp;
  
    FILE *file = fopen("banker.inp", "rt");
    FILE *file2 = fopen("banker.out", "wt");
    fscanf(file, "%d%d", &Process_num, &Resource_num);
    inti_Q(&Wait_Queue);
  
    for(i = 0; i < Resource_num; i++) {
        fscanf(file, "%d", &Resource_MAX[i]);
    }
  
    for(i = 0; i < Process_num ; i++) {
        for(j = 0; j < Resource_num ; j++) {
            fscanf(file, "%d", &Max[i][j]);
        }
    }
  
    for(i = 0; i < Process_num ; i++) {
        for(j = 0; j < Resource_num ; j++) {
            fscanf(file, "%d", &Allocated[i][j]);
        }
    }
  
    for(i = 0; i < Process_num ; i++) {
        for(j = 0; j < Resource_num ; j++) {
            Need[i][j] = Max[i][j] - Allocated[i][j];
        }
    }
  
    for(i = 0; i < Resource_num; i++) {
        temp = 0;
        for(j = 0; j < Process_num; j++) {
            temp += Allocated[j][i];
        }
        Available[i] = Resource_MAX[i] - temp;
    }
  
    C_temp.resource = (int*)malloc(sizeof(int)*Resource_num);
  
    while(1) {
        fscanf(file, "%s", &word);
        fscanf(file, "%d", &C_temp.Play_Process);
        for(i = 0; i < Resource_num ; i++) {
            fscanf(file, "%d", &C_temp.resource[i]);
        }
  
        if(strcmp(word, "request") == 0) {
            if(Av_Re(Available, C_temp.resource, Resource_num)) {
                temp = Request_Check(Allocated, Need, Available, C_temp, Process_num, Resource_num);
                if(temp == 1 || temp == 2) {
                }
                else {
                    enQueue(&Wait_Queue, C_temp, Resource_num);
                }
  
                for(i = 0; i < Resource_num ; i++) {
                    printf("%d ", Available[i]);
                    fprintf(file2, "%d ", Available[i]);
                }
                printf("\n");
                fprintf(file2, "\n");
            }
            else {
                enQueue(&Wait_Queue, C_temp, Resource_num);
                for(i = 0; i < Resource_num ; i++) {
                    printf("%d ", Available[i]);
                    fprintf(file2, "%d ", Available[i]);
                }
                printf("\n");
                fprintf(file2, "\n");
            }
        }
        else if(strcmp(word, "release") == 0) {
            int Q_count_temp;
            Release(Allocated, Need, Available, C_temp, Process_num, Resource_num);
  
            Q_count_temp = Wait_Queue.Q_count;
            for(i = 0 ; i < Q_count_temp ; i++) {
                C_temp = deQueue(&Wait_Queue, Resource_num);
                if(Av_Re(Available, C_temp.resource, Resource_num) == 0) {
                    enQueue(&Wait_Queue, C_temp, Resource_num);
                    continue;
                }
                temp = Request_Check(Allocated, Need, Available, C_temp, Process_num, Resource_num);
                if(temp == 0) {
                    enQueue(&Wait_Queue, C_temp, Resource_num);
                }
                else {
                }
            }
  
            for(i = 0; i < Resource_num ; i++) {
                fprintf(file2, "%d ", Available[i]);
                printf("%d ", Available[i]);
            }
            printf("\n");
            fprintf(file2, "\n");
        }
        else
            break;
  
        C_temp.Play_Process = -1;
        for(i = 0 ; i < Resource_num ; i++)
            C_temp.resource[i] = -1;
    }
  
    fclose(file);
    fclose(file2);
    return 0;
}
```
