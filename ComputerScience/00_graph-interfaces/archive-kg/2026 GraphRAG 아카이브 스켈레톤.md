---
aliases: []
course: archive-kg
created: '2026-05-28'
date: '2026-05-28'
kg_graph_size: 176
kg_layer_label: L0 skeleton
kg_level: 0
kg_role: skeleton
semester: meta
source: ''
status: evergreen
tags:
- type/interface
- pkm/kg-skeleton
title: 2026 GraphRAG 아카이브 스켈레톤
type: interface
updated: '2026-05-28'
---

method:: [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/GraphRAG-Bench 2026 리더보드|GraphRAG-Bench 2026 리더보드]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/AutoPrunedRetriever 최소 추론 서브그래프|AutoPrunedRetriever 최소 추론 서브그래프]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/Youtu-GraphRAG 4단계 지식 트리|Youtu-GraphRAG 4단계 지식 트리]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/FalkorDB GraphRAG-SDK 파이프라인|FalkorDB GraphRAG-SDK 파이프라인]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/LinearRAG 관계 추출 과잉 회피|LinearRAG 관계 추출 과잉 회피]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/BRINK KG-RAG 근거 감사|BRINK KG-RAG 근거 감사]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/Atomic Educational GraphRAG|Atomic Educational GraphRAG]], [[ComputerScience/00_graph-interfaces/archive-kg/methods-2026/WildGraphBench 현실 코퍼스 평가|WildGraphBench 현실 코퍼스 평가]]
query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Creative Generation|Creative Generation]]
community:: [[ComputerScience/00_graph-interfaces/archive-kg/communities/프로그래밍 기초 커뮤니티|프로그래밍 기초 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/수학 이론 커뮤니티|수학 이론 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/AI ML 데이터 커뮤니티|AI ML 데이터 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/시스템 인프라 커뮤니티|시스템 인프라 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/소프트웨어 엔지니어링 커뮤니티|소프트웨어 엔지니어링 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/알고리즘 그래픽스 커뮤니티|알고리즘 그래픽스 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/전문 교양 커뮤니티|전문 교양 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/외부 프로그램 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/자격증 검증 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/공유 미디어 커뮤니티]], [[ComputerScience/00_graph-interfaces/archive-kg/communities/아카이브 운영 커뮤니티]]

# 2026 GraphRAG 아카이브 스켈레톤

이 스켈레톤은 2026년 기준 GraphRAG-Bench와 ICLR/EACL/AAAI 계열 KG-RAG 연구에서 반복적으로 확인되는 구조를 Obsidian vault에 맞게 옮긴 것입니다.

## 적용 원칙

- **Top-down interface first**: 허브와 스켈레톤에서 방법론, 질의 모드, 분야 커뮤니티, 과목 프로필, 근거 파일 순서로 내려갑니다.
- **Full source extraction**: 원문 노트 전체, PDF 전체 텍스트, 텍스트/코드/노트북/Office 산출물 텍스트를 추출한 뒤 개념을 계산합니다.
- **Atomic evidence**: 과목별 근거 인덱스가 원문 파일을 모아 출처와 적용 범위를 보존합니다.
- **Minimal reasoning subgraph**: 일반 연구/스택 링크를 모든 노트에 붙이지 않고, 과목별 핵심 개념과 공유 개념을 연결합니다.
- **4-level knowledge tree**: method/query, community, course/concept, evidence/source 계층으로 탐색합니다.
- **Resolution before linking**: 파일명만 보지 않고 heading, frontmatter title, PDF 전체 텍스트, 코드/산출물 내용을 함께 보고 개념을 합칩니다.
- **Query-mode view**: GraphRAG-Bench의 fact, complex, contextual, creative 질의 유형을 아카이브 탐색 인터페이스로 둡니다.

## Top-down 계층

1. 허브/스켈레톤: 전체 GraphRAG 방식과 탐색 계층
2. 방법론/질의 모드: 2026 연구와 retrieval 목적별 관점
3. 커뮤니티 리포트: 프로그래밍, 수학, AI, 시스템, 소프트웨어 등 분야별 인터페이스
4. 과목 프로필/개념 노드: 과목별 reasoning subgraph와 concept codebook
5. 근거 인덱스/원문 파일: 강의 노트, PDF, 코드/노트북, 과제 산출물, PNG 첨부

## 과목 프로필

- [[ComputerScience/00_graph-interfaces/archive-kg/courses/AI 시스템 설계 지식그래프|AI 시스템 설계]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/AIOSS 오픈소스 delivery 지식그래프|AIOSS 오픈소스 delivery]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/Java 프로그래밍 지식그래프|Java 프로그래밍]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/LG Aimers 지식그래프|LG Aimers]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/Linux 지식그래프|Linux]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/ML 프로젝트 지식그래프|ML 프로젝트]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/Python 프로그래밍 지식그래프|Python 프로그래밍]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/고전 읽기 지식그래프|고전 읽기]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/공유 미디어 지식그래프|공유 미디어]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/뉴럴네트워크 지식그래프|뉴럴네트워크]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/데이터베이스 지식그래프|데이터베이스]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/머신러닝 지식그래프|머신러닝]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/병렬 분산처리 지식그래프|병렬 분산처리]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/빅데이터분석 지식그래프|빅데이터분석]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/생성형 AI 파인튜닝 지식그래프|생성형 AI 파인튜닝]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/수리논리학 지식그래프|수리논리학]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/아카이브 운영 지식그래프|아카이브 운영]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/알고리즘 설계와 분석 지식그래프|알고리즘 설계와 분석]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/오픈소스 소프트웨어 지식그래프|오픈소스 소프트웨어]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/운영체제 지식그래프|운영체제]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/웹 프로그래밍 지식그래프|웹 프로그래밍]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/이산수학 지식그래프|이산수학]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/인공지능 지식그래프|인공지능]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/자격증 지식그래프|자격증]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/자료구조 지식그래프|자료구조]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/지식재산 지식그래프|지식재산]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/창의적 글쓰기 지식그래프|창의적 글쓰기]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/최적화 수학 지식그래프|최적화 수학]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/컨테이너 오케스트레이션 지식그래프|컨테이너 오케스트레이션]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/컴퓨터구조 지식그래프|컴퓨터구조]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/컴퓨터그래픽스 지식그래프|컴퓨터그래픽스]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/컴퓨터네트워크 지식그래프|컴퓨터네트워크]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/컴퓨터비전 지식그래프|컴퓨터비전]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/코딩 기초 지식그래프|코딩 기초]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/코딩 테스트 지식그래프|코딩 테스트]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/프로그래밍언어론 지식그래프|프로그래밍언어론]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/학점 포트폴리오 지식그래프|학점 포트폴리오]]
- [[ComputerScience/00_graph-interfaces/archive-kg/courses/확률통계 지식그래프|확률통계]]
