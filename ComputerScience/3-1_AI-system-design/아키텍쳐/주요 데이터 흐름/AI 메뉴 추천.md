---
aliases: []
course: AI-system-design
created: '2025-04-10'
date: '2025-04-10'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ai
- cs/se
- type/lecture
title: AI 메뉴 추천
type: lecture
updated: '2026-05-05'
---

up:: [[커리큘럼 관계 정리|[3-1] AI시스템개발및설계]]

siblings:: [[ComputerScience/3-1_AI-system-design/아키텍쳐/주요 데이터 흐름/메뉴 조회|메뉴 조회]], [[ComputerScience/3-1_AI-system-design/아키텍쳐/주요 데이터 흐름/장바구니에 메뉴 추가|장바구니에 메뉴 추가]], [[ComputerScience/3-1_AI-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]]
---
```mermaid
graph LR
    A["Client: Send Chat Message"] --> B("Backend: /chat Router");
    B --> C{"Chat Logic (Likely involves NLP/Embedding)"};
    C -- "Generate Query Embedding" --> D["Vector Store: Search Similar Menu Embeddings"];
    D -- "Return Similar Menu IDs" --> C;
    C -- "Get Menu Details" --> E["CRUD: Get Menus by IDs"];
    E --> F{"DB: Select Menus"};
    F --> E;
    E --> C;
    C -- "Format Recommendation" --> B;
    B --> G["Backend: Return Recommended Menus"];
    G --> H["Client: Display Recommendations"];

    subgraph "Backend Interaction"
        B; C; E; G;
    end
    subgraph "Database Interaction"
        F;
    end
    subgraph "Vector DB Interaction"
        D;
    end
```
