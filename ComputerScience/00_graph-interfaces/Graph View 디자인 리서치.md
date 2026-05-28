---
aliases: []
course: archive-kg
created: '2026-05-28'
date: '2026-05-28'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: meta
source: Obsidian Help, Obsidian Stats, Obsidian Community plugins, 2026 KG visualization research
status: evergreen
tags:
- type/interface
- pkm/hub
- pkm/kg-skeleton
title: Graph View 디자인 리서치
type: interface
updated: '2026-05-28'
---

kg_parent:: [[ComputerScience/00_graph-interfaces/지식그래프 허브|지식그래프 허브]]
kg_skeleton:: [[ComputerScience/00_graph-interfaces/archive-kg/2026 GraphRAG 아카이브 스켈레톤|2026 GraphRAG 아카이브 스켈레톤]]
design_target:: [[2026 GraphRAG 아카이브.canvas]], [[지식그래프 레벨 인터페이스.canvas]], [[커리큘럼 관계 그래프.canvas]]

# Graph View 디자인 리서치

## 적용 판단

- Obsidian 기본 Graph View는 [공식 도움말](https://obsidian.md/help/plugins/graph) 기준으로 필터, 색상 그룹, 표시 옵션, 힘 배치를 제어합니다. 계층별 개별 노드 크기를 직접 지정하는 기능은 기본 기능에 없습니다.
- Graph View 외형은 [CSS snippets](https://obsidian.md/help/snippets)와 Graph 전용 [CSS 변수](https://obsidian-developer-docs.pages.dev/Reference/CSS-variables/Plugins/Graph)로 기본 노드, 선, 텍스트, 첨부파일 색상을 제어합니다.
- 확장 플러그인 평가 기준에서는 [Extended Graph](https://www.obsidianstats.com/plugins/extended-graph)가 메타데이터 기반 노드/링크 크기, 이미지, 모양, 복수 뷰를 지원하고 가장 직접적으로 요구사항에 맞습니다.
- [Juggl](https://community.obsidian.md/plugins/juggl)은 스타일과 계층 레이아웃이 강하지만, 현재 vault에는 설치되어 있지 않고 마지막 업데이트가 오래되어 기본 적용 대상에서 제외합니다.
- 2026년 KG 시각화 연구인 [Context-KG](https://arxiv.org/abs/2604.10384)는 단순 force-directed 배치보다 온톨로지, 사용자 의도, type-aware region 기반 배치를 권장합니다.
- 데이터 시각화 접근성 기준은 색상만으로 의미를 전달하지 않고, 대비와 clutter 관리를 요구합니다. 이 vault에서는 색상 그룹과 Canvas의 고정 계층 크기를 함께 사용합니다.

## 적용 방식

- 기본 Graph View는 전체 관계 탐색용 overview로 둡니다.
- 태그 노드는 숨기고 첨부파일 노드는 보이게 둡니다.
- unresolved 노드는 숨겨서 존재하지 않는 링크가 지식그래프 노드처럼 보이지 않게 합니다.
- 전체 노드 크기와 링크 두께를 낮춰 degree가 큰 하위 노드가 상위 인터페이스처럼 보이지 않게 합니다.
- 분야별 색상 그룹은 유지하되 Graph View 기본 색상은 CSS snippet으로 차분하게 낮춥니다.
- 계층별 노드 크기와 top-down 우선순위는 Canvas에서 명시합니다.

## 디자인 계층

1. Level 0: [[ComputerScience/00_graph-interfaces/지식그래프 허브|지식그래프 허브]], [[ComputerScience/00_graph-interfaces/archive-kg/2026 GraphRAG 아카이브 스켈레톤|2026 GraphRAG 아카이브 스켈레톤]]
2. Level 1: 단계 인터페이스, 2026 방법론, 질의 모드
3. Level 2: 분야 인터페이스와 커뮤니티
4. Level 3: 과목 프로필과 브리지
5. Level 4: 개념, 근거 인덱스, 원문 파일, PNG 첨부

## 적용 파일

- `.obsidian/graph.json`: overview용 Graph View 설정
- `.obsidian/snippets/graph-view-design.css`: Graph View 색상/대비 토큰
- `지식그래프 레벨 인터페이스.canvas`: 단계와 분야 계층 시각화
- `2026 GraphRAG 아카이브.canvas`: 2026 GraphRAG 스켈레톤 시각화
- `커리큘럼 관계 그래프.canvas`: 기존 관계 그래프의 top-down 크기 보정

## Extended Graph 적용

- 설치 후보는 [ElsaTam/obsidian-extended-graph 2.7.7](https://github.com/ElsaTam/obsidian-extended-graph/releases/tag/2.7.7)로 확정했습니다. 릴리스 에셋의 `manifest.json`은 upstream 기준 `2.7.6` 버전을 유지합니다.
- 노드 크기는 연결 수가 아니라 `kg_graph_size` frontmatter 속성으로 계산합니다.
- 계층 레이어는 `kg_level`을 사용합니다. Level 0은 허브/스켈레톤, Level 1은 단계/방법론/질의 모드, Level 2는 분야/커뮤니티/브리지, Level 3은 과목/근거/미디어 인덱스, Level 4는 개념과 원문 노트입니다.
- 전역 Graph View는 첨부파일 노드를 숨겨 PNG가 독립 노드로 그래프를 흐리지 않게 했습니다. PNG는 정합한 노트의 embed와 Local Graph 이미지 표시를 통해 확인합니다.
- 대형 vault 성능을 위해 `maxNodes`는 3200, 초기화 지연은 900ms, 통계 재계산은 수동/초기화 중심으로 둡니다.
