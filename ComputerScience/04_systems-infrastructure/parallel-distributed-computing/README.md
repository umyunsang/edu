---
title: 병렬 · 분산처리
description: 'Stanford CS149 기반. 병렬성의 동기, 멀티코어 아키텍처, ISPC, 작업 분배와 스케줄링, 지역성·통신·경합,
  GPU/CUDA, MPI. 학기: 3-1.'
type: course-index
tags:
- course
- 3-1
course: parallel-distributed-computing
semester: 3-1
status: draft
created: '2026-08-28'
updated: '2026-08-28'
---

> [!abstract] 이 과목은
> Stanford CS149 기반. 병렬성의 동기, 멀티코어 아키텍처, ISPC, 작업 분배와 스케줄링, 지역성·통신·경합, GPU/CUDA, MPI. 학기: 3-1.

## 학습 경로

번호는 강의 진도 순이다. 앞 문서를 읽었다는 전제로 다음 문서가 쓰인다.

```mermaid
flowchart LR
    N0["01. 왜 병렬 처리인가"]
    N1["02. 병렬 컴퓨터의 기본 아키텍처"]
    N2["03. 멀티코어 아키텍처 II - 지연…"]
    N3["04. 병렬 프로그래밍 기본"]
    N4["05. 성능 최적화 I - 작업 분배와…"]
    N5["06. 지역성·통신·경합"]
    N6["07. CUDA 설치와 개발 환경"]
    N7["08. GPU 아키텍처와 CUDA 프로…"]
    N8["09. GPU 아키텍처와 CUDA 프로…"]
    N9["11. MPI 병렬 프로그래밍 I"]
    N10["12. MPI 병렬 프로그래밍 II"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N7
    N7 --> N8
    N8 --> N9
    N9 --> N10
```

## 정리문서

모두 `notes/` 에 있다. 총 11편.

| 문서 | 다루는 내용 |
| :-- | :-- |
| [01. 왜 병렬 처리인가](<./notes/01. 왜 병렬 처리인가.md>) | — |
| [02. 병렬 컴퓨터의 기본 아키텍처](<./notes/02. 병렬 컴퓨터의 기본 아키텍처.md>) | — |
| [03. 멀티코어 아키텍처 II - 지연·대역폭과 ISPC](<./notes/03. 멀티코어 아키텍처 II - 지연·대역폭과 ISPC.md>) | — |
| [04. 병렬 프로그래밍 기본](<./notes/04. 병렬 프로그래밍 기본.md>) | — |
| [05. 성능 최적화 I - 작업 분배와 스케줄링](<./notes/05. 성능 최적화 I - 작업 분배와 스케줄링.md>) | — |
| [06. 지역성·통신·경합](<./notes/06. 지역성·통신·경합.md>) | — |
| [07. CUDA 설치와 개발 환경](<./notes/07. CUDA 설치와 개발 환경.md>) | — |
| [08. GPU 아키텍처와 CUDA 프로그래밍 II](<./notes/08. GPU 아키텍처와 CUDA 프로그래밍 II.md>) | — |
| [09. GPU 아키텍처와 CUDA 프로그래밍 III](<./notes/09. GPU 아키텍처와 CUDA 프로그래밍 III.md>) | — |
| [11. MPI 병렬 프로그래밍 I](<./notes/11. MPI 병렬 프로그래밍 I.md>) | — |
| [12. MPI 병렬 프로그래밍 II](<./notes/12. MPI 병렬 프로그래밍 II.md>) | — |

## 원본 자료

교수가 배포한 자료다. `sources/` 에 있고 수정하지 않는다. 총 21건.

- `01_WhyParallelism.pdf`
- `02_기본아키텍츠_update 2.pdf`
- `02_기본아키텍츠_update.pdf`
- `03_multicore2-ispc_update 2.pdf`
- `03_multicore2-ispc_update.pdf`
- `03_multicore2-update_0416 2.pdf`
- `03_multicore2-update_0416.pdf`
- `03_multicore2-강의자료업데이트_0415.pdf`
- `04_Parallel Programming 기본 2.pdf`
- `04_Parallel Programming 기본.pdf`
- `05_Performance Optimization I Work Distribution and Scheduling.pdf`
- `06_Locality_Communication_Contention.pdf`
- `07_GPU Architecture & CUDAProgramming_v01.pdf`
- `08_GPU Architecture & CUDAProgramming_v02.pdf`
- `09_GPU Architecture & CUDAProgramming_v03 2.pdf`
- `09_GPU Architecture & CUDAProgramming_v03.pdf`
- `11_MPI 프로그래밍_V_01 2.pdf`
- `11_MPI 프로그래밍_V_01.pdf`
- `12_MPI 프로그래밍_V_02 2.pdf`
- `12_MPI 프로그래밍_V_02.pdf`
- `쿠다.pdf`

## 관련 과목

> [!note] 아직 비어 있다
> 다른 과목과의 관계는 지식그래프 4단계에서 관계 타입(`prerequisite` · `elaborates` · `contrasts` · `applies` · `evidences`)과 함께 채운다. 근거 없이 미리 이어두지 않는다.
