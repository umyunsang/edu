---
aliases: []
course: ai-system-design
created: '2025-04-10'
date: '2025-04-10'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 3-1
source: ''
status: draft
tags:
- ai
- se
- lecture
title: AI 메뉴 추천
type: lecture
updated: '2026-05-05'
---

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
