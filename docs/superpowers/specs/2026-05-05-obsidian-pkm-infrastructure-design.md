---
title: Obsidian PKM Infrastructure — Full Refactor (Option C)
date: 2026-05-05
status: approved
owner: umyunsang
branch: pkm-refactor
---

# Obsidian PKM Infrastructure — Full Refactor

## 0. Context

Computer Science / AI 학부 커리큘럼 볼트 (479 .md 노트). 사용자가 "전면 재정비 + 옵시디언 기능 최대 활용"을 명시 승인. AI 작업은 Claude Code 스킬 측에서만 수행 (옵시디언 AI 플러그인 미사용).

**현재 상태 (audit 결과):**
- 479 노트 (ComputerScience 436, root 4, certifications 4, LGAimer 2)
- 91% (435/479) frontmatter 보유 — 스키마: `title/date/tags/aliases`
- 폴더명 100% `학기_영문약어` 규칙 준수
- 산재 이미지 50개, 루트 잡파일 6개 (54MB), 파일명 충돌 7건, 깨진 위키링크 다수
- 활성 플러그인 9개, 비활성 7개

**목표:** MOC-first Evergreen PKM 방법론 + 5축 태그 분류 + 확장 frontmatter + 그래프 뷰 최적화를 옵시디언 네이티브 기능으로 구현.

---

## 1. 아키텍처 (5계층)

```
[Theme Layer]      Minimal + Style Settings + Pretendard·JetBrains Mono
[Plugin Layer]     기존 9개 + 신규 7개(Tier1) + 활성화 4개 + 선택 3개
[Data Layer]       확장 frontmatter (기존 4필드 + 신규 7필드, 비파괴)
[Navigation Layer] MOCs/ 폴더 + 15개 MOC + 36개 README→MOC 업그레이드
[Workflow Layer]   Templater 6종 + CC 스킬 트리거 + Linter 규칙 + 그래프 프리셋
```

**원칙:**
- 기존 폴더 구조와 frontmatter 4필드는 100% 보존
- 모든 변경은 `pkm-refactor` 브랜치, Phase별 단일 커밋, 롤백 가능
- 스크립트는 dry-run → 사용자 확인은 생략(승인 받음) → 실행
- 한국어 텍스트 안전성 최우선

---

## 2. Plugin Layer

### 2.1 신규 설치 (Tier 1, 7개)

| 플러그인 ID | 역할 | 비고 |
|---|---|---|
| `dataview` | MOC 자동 쿼리, exam-queue, status 대시보드 | 본 디자인의 쿼리 엔진 |
| `obsidian-excalidraw-plugin` | 알고리즘·신경망·시스템 손그림 다이어그램 | |
| `obsidian-spaced-repetition` | SM-2 플래시카드 | `#flashcards/*` 태그 기반 |
| `table-editor-obsidian` | Advanced Tables — 표 자동정렬·수식·탭 네비 | |
| `obsidian-icon-folder` | Iconize — 폴더/파일 아이콘 | 학기 시각 구분 |
| `graph-analysis` | 중심성·커뮤니티 검출 | 진짜 허브 노트 측정 |
| `excalibrain` | 방사형 계층 그래프 | MOC 구조 시각화 핵심 |

### 2.2 기존 비활성 → 활성화 (4개)

| 플러그인 | 활성화 이유 |
|---|---|
| `mermaid-tools` | MOC 구조 다이어그램용 |
| `pdf-plus` | 강의자료 PDF 주석·인용 |
| `quick-latex` | 수식 입력 가속 |
| `obsidian-latex` | Extended MathJax 매크로 |

### 2.3 Tier 2 (선택, 즉시 설치)

| 플러그인 | 역할 |
|---|---|
| `obsidian-style-settings` | 테마 GUI 커스터마이징 (Minimal 의존성) |
| `periodic-notes` | 일/주/월간 노트 |
| `obsidian-hover-editor` | 링크 미리보기 인라인 편집 |

### 2.4 유지 (수정 없음, 9개)

obsidian-git, obsidian-linter, templater-obsidian, tag-wrangler, auto-note-mover, find-unlinked-files, consistent-attachments-and-links, better-export-pdf, terminal

### 2.5 폐기/제외

- 클라우드 AI 플러그인 일체 (Smart Connections, Copilot for Obsidian, Text Generator) — 제외
- `marker-api`, `ink` — 비활성 유지 (필요 시점에 활성)

---

## 3. Theme & CSS Layer

### 3.1 테마

**Minimal** (kepano) 설치 + 활성화. **Style Settings** 플러그인을 통해 GUI 조정.

### 3.2 CSS 스니펫 (`.obsidian/snippets/`)

```
fonts.css           — Pretendard(한글) + JetBrains Mono(코드)
callouts.css        — type별 색상 (lecture=파랑, permanent=초록, project=주황)
status-badges.css   — seedling/budding/evergreen 시각 배지 (frontmatter status 기반)
moc-styling.css     — type:MOC 노트 본문 좌측 컬러 바
```

### 3.3 Style Settings 적용

- Minimal 테마의 CSS 변수 노출 활용
- 다크/라이트 동일 가독성

---

## 4. Data Layer (Frontmatter)

### 4.1 확장 스키마

```yaml
---
# === 기존 4필드 (모든 노트 보존) ===
title: 머신러닝 기초
date: 2026-03-12
tags:
  - cs/ml          # 신규 도메인 태그 (자동 추론)
  - type/lecture   # 신규 type 태그 (자동 추론)
aliases:
  - ML 기초

# === 신규 7필드 (일괄 주입) ===
type: lecture            # lecture | literature | permanent | project | MOC | index
status: seedling         # seedling | budding | evergreen
semester: "3-1"          # 폴더명에서 추론, 따옴표 필수 (YAML 날짜 파싱 방지)
course: machine-learning # 폴더 슬러그
created: 2026-03-12      # 기존 date 또는 git log 기반
updated: 2026-05-05      # Linter가 매 저장마다 갱신
source: ""               # 빈값 허용, 필드 존재 보장
---
```

### 4.2 자동 추론 규칙

**`type` 추론:**
- 파일명이 `README.md` → `MOC`
- 폴더에 `과제`, `프로젝트`, `실습` 포함 → `project`
- 폴더에 `papers`, `reading`, `교재` 포함 → `literature`
- 그 외 일반 노트 → `lecture` (사용자가 추후 점진 분류)
- `MOCs/` 디렉터리 노트 → `MOC`

**`status` 기본값:** 모두 `seedling` (사용자가 점진 승급)

**`semester` 추론:**
- `3-1_*/...` → `"3-1"`
- `elective_*/...` → `"elective"`
- `certifications/...` → `"cert"`
- `LGAimer/...` → `"extracurricular"`

**`course` 추론:** 폴더명 prefix 제거 (`3-1_machine-learning` → `machine-learning`)

**`created` 추론:** 기존 `date` 필드 우선, 없으면 `git log --diff-filter=A --follow --format=%ai -- <file> | tail -1` 결과의 ISO 날짜

### 4.3 보존 규칙

- 기존 `title/date/tags/aliases` 절대 삭제·수정 금지
- 기존 `tags` 배열에 새 태그 **추가만** (중복 제거)
- 기존 `date` 필드는 그대로 두고 `created` 별도 추가

---

## 5. Tag Taxonomy (5축, ~40개)

### 5.1 분류 트리

```
type/
├─ lecture
├─ literature
├─ permanent
├─ project
├─ MOC
└─ index

cs/
├─ ml, dl, ai, llm, nlp, cv
├─ algorithms, db, se, security
├─ systems, distributed, devops, open-source

math/
├─ linalg, calculus, probability, statistics, discrete

skill/
├─ python, pytorch, cuda, sql, docker, git, latex, java

meta/
├─ exam, portfolio, question, cert
```

### 5.2 자동 주입 매핑 (폴더 → 태그)

| 폴더 패턴 | 주입 태그 |
|---|---|
| `*_machine-learning` | `cs/ml` |
| `*_AIOSS` | `cs/open-source`, `cs/ai` |
| `*_LLM` | `cs/llm`, `cs/nlp` |
| `*_docker-k8s` | `skill/docker`, `cs/devops` |
| `*_probability-statistics` | `math/probability`, `math/statistics` |
| `*_distributed-computing` | `cs/systems`, `cs/distributed` |
| `*_computer-vision` | `cs/cv`, `cs/dl` |
| `*_algorithm*` | `cs/algorithms` |
| `*_database*` | `cs/db` |
| `*_operating-system*` | `cs/systems` |
| `*_network*` | `cs/systems` |
| `*_software-engineering` | `cs/se` |
| `*_security*` | `cs/security` |
| `*_java` | `skill/java` |
| `*_coding-basics` | `skill/python` |
| `*_web*` | `cs/se`, `skill/javascript` |
| `*_intellectual-property` | `meta/cert` |
| `certifications/*` | `meta/cert` |
| `LGAimer/*` | `cs/ml`, `meta/extracurricular` |

(Phase 5 스크립트는 38개 폴더 전수 매핑 표를 사용)

### 5.3 규칙

- `type/*` 1개 필수 (모든 노트)
- `cs/*` 또는 `math/*` 1개 이상 권장 (자동 추론으로 보장)
- `skill/*`, `meta/*` 선택
- 폴더가 이미 의미하는 건 태그하지 않음 (`#semester/3-1` 금지)

---

## 6. Navigation Layer (MOC)

### 6.1 MOCs/ 디렉터리 (15개)

```
MOCs/
├── Home MOC.md
├── Machine Learning MOC.md
├── Deep Learning MOC.md
├── Algorithms MOC.md
├── Systems MOC.md
├── Computer Vision MOC.md
├── LLM & NLP MOC.md
├── AI Open Source MOC.md
├── Math Foundations MOC.md
├── Database MOC.md
├── Cloud & Containers MOC.md
├── Security MOC.md
├── Software Engineering MOC.md
├── Certifications MOC.md
├── Portfolio MOC.md
└── Open Questions MOC.md
```

### 6.2 표준 MOC 구조

```markdown
---
type: MOC
status: evergreen
tags: [type/MOC, cs/ml]
semester: "all"
course: cross-curriculum
created: 2026-05-05
updated: 2026-05-05
aliases: [ML MOC]
---

up:: [[Home MOC]]
central:: [[Machine Learning MOC]]
children:: [[Deep Learning MOC]], [[Computer Vision MOC]], [[LLM & NLP MOC]]

# Machine Learning MOC

## Foundations
- [[Bias-Variance Tradeoff]]
- [[Maximum Likelihood Estimation]]
- [[Regularization]]

## Supervised
- [[Linear Regression]] · [[Logistic Regression]] · [[SVM]]

## Unsupervised
- [[PCA]] · [[K-Means]] · [[GMM]]

## Open Questions
- [[Why does double descent happen?]]

## All ML notes (auto)
\`\`\`dataview
TABLE status, file.mtime as updated
FROM #cs/ml
WHERE type != "MOC"
SORT file.mtime DESC
LIMIT 50
\`\`\`

## Recently created
\`\`\`dataview
LIST FROM #cs/ml
WHERE type != "MOC"
SORT file.cday DESC
LIMIT 10
\`\`\`
```

### 6.3 README → MOC 업그레이드

기존 36개 코스별 `README.md`는:
- frontmatter `type: MOC` 추가
- 본문 보존
- 하단에 Dataview 블록 자동 삽입 (`FROM #cs/<inferred>` AND `course = "<course>"`)
- 상단에 `up:: [[<적절한 도메인 MOC>]]` 추가

---

## 7. Workflow Layer

### 7.1 Templater 템플릿 (`_templates/`, 6개)

| 템플릿 | 트리거 | 자동 채움 |
|---|---|---|
| `lecture-note.md` | 강의 직후 | type/status/semester/course (현재 폴더 추론), created/updated, tags |
| `literature-note.md` | 논문/책 정리 | type=literature, source 필수 |
| `permanent-note.md` | evergreen 아이디어 | type=permanent, status=budding |
| `project-note.md` | 과제·프로젝트 | type=project, source=과제 명세 링크 |
| `moc.md` | 새 MOC 생성 | type=MOC, status=evergreen, up:: 입력 프롬프트 |
| `daily-note.md` | periodic-notes 연동 | 일자, exam-queue 쿼리 임베드 |

### 7.2 Linter 규칙 (한국어 안전)

`.obsidian/plugins/obsidian-linter/data.json`:

**활성:**
- `yaml-key-sort` (alphabetical, 신규 7필드 + 기존 4필드 모두 정렬)
- `format-tags-in-yaml` (array form, leading hyphen)
- `format-yaml-array` (multi-line)
- `yaml-timestamp` (`updated` 자동 갱신, 한국 시간대)
- `trailing-spaces`
- `consecutive-blank-lines`
- `heading-blank-lines`
- `empty-line-around-blockquotes`
- `empty-line-around-code-fences`
- `space-after-list-markers`

**비활성 (한글 파괴 위험):**
- `capitalize-headings`
- `capitalize-headings-with-articles`
- `english-spelling`
- `punctuation-conversion`
- `quote-style` (한글 따옴표 보존)
- `paragraph-blank-lines` (한국어 단락 관행 충돌)

### 7.3 Graph View 최적화

**`.obsidian/graph.json` 색상 그룹:**

```json
{
  "colorGroups": [
    {"query": "tag:#type/MOC",                              "color": {"a":1, "rgb":15225154}},
    {"query": "tag:#cs/ml OR tag:#cs/dl",                   "color": {"a":1, "rgb":3447003}},
    {"query": "tag:#cs/systems OR tag:#cs/devops",          "color": {"a":1, "rgb":10181046}},
    {"query": "tag:#cs/algorithms",                          "color": {"a":1, "rgb":15967746}},
    {"query": "tag:#cs/ai OR tag:#cs/llm OR tag:#cs/nlp",   "color": {"a":1, "rgb":1751474}},
    {"query": "tag:#math",                                   "color": {"a":1, "rgb":3066993}},
    {"query": "tag:#skill",                                  "color": {"a":1, "rgb":9807270}},
    {"query": "tag:#meta/portfolio",                         "color": {"a":1, "rgb":15844367}},
    {"query": "tag:#meta/question",                          "color": {"a":1, "rgb":15105570}}
  ]
}
```

**필터 프리셋 (워크스페이스 저장):**
- `graph-overview` — 전체 + tag별 색상, attachments 제외
- `graph-semester` — 현재 학기 (`["semester":"3-1"]`)
- `graph-portfolio` — `tag:#meta/portfolio OR tag:#type/permanent`

**Excalibrain 활용:**
- 모든 MOC에 `central::` 인라인 필드 자동 삽입
- `parents::`, `children::` 관계 명시 → 방사형 그래프

### 7.4 Spaced Repetition 워크플로우

- 플러그인: `obsidian-spaced-repetition` (SM-2)
- 카드 위치: lecture/permanent 노트 본문 인라인 (`Question::Answer` 또는 `?` 구분)
- 덱: `#flashcards/cs/ml`, `#flashcards/math/linalg`, `#flashcards/cert/정보처리기사` 형태
- 글로벌 큐: 매일 10장 (학기무관 핵심)
- 시험 큐: `#meta/exam` 추가 시 활성

### 7.5 CC 스킬 통합 패턴

| 작업 | 호출 스킬 |
|---|---|
| 새 강의노트 작성 | `/obsidian-markdown` + Templater |
| 웹 자료 캡처 | `/defuddle` → literature 폴더 |
| PDF/DOCX 임포트 | `/docx-to-markdown`, `/epub-to-markdown` |
| 다이어그램 작성 | `/obsidian-mermaid`, `/obsidian-canvas` |
| 발표자료 생성 | `/markdown-slides` from MOC |
| 볼트 정합성 점검 | `/obsidian-cli` 검색 + 검증 |

---

## 8. Refactor Operations (11 Phase)

| Phase | 작업 | 영향 파일 | 시간 | 리스크 | 롤백 |
|---|---|---|---|---|---|
| 1 | 플러그인 설치/활성화 + 테마 + 템플릿 + 루트 잡파일 6개 삭제 | ~10 | 30분 | 낮음 | git revert |
| 2 | MOCs/ + 15개 MOC 작성 | +15 | 2시간 | 낮음 | rm -rf MOCs/ |
| 3 | 미보유 44개 노트에 frontmatter 주입 | 44 | 30분 | 중간 | git revert |
| 4 | 기존 435개에 신규 7필드 추가 | 435 | 1시간 | 중간 | git revert (드라이런 검증 후 실행) |
| 5 | 폴더 → 도메인 태그 자동 주입 | 479 | 1시간 | 중간 | git revert |
| 6 | 산재 이미지 50개 → `/image/{course}_*` + 위키링크 갱신 | ~150 | 2시간 | **높음** (위키링크 깨질 시) | git revert + 검증 |
| 7 | 파일명 충돌 7건 해결 + 위키링크 갱신 | ~20 | 1시간 | **높음** | git revert |
| 8 | 36개 README.md → MOC 업그레이드 | 36 | 1시간 | 낮음 | git revert |
| 9 | 깨진 위키링크 스캔·수정 | ~30 | 1시간 | 중간 | 개별 검토 |
| 10 | Linter 전체 적용 + 한글 안전성 검증 | 479 | 30분 | 중간 | dry-run 우선 |
| 11 | 검증 (5% 샘플) + 최종 검토 | — | 30분 | 낮음 | — |

**총 영향: ~600~700 파일, 작업시간 ~12시간 (자동화 70%, 검토 30%)**

---

## 9. 안전장치

- **브랜치 격리**: 전 작업을 `pkm-refactor` 브랜치에서 수행 (이미 생성됨)
- **Phase별 단일 커밋**: 각 Phase 완료 시 명확한 커밋 메시지로 단일 커밋 → 롤백 가능
- **이중 백업**: obsidian-git 자동백업이 main 브랜치에서 계속 실행되어 안전망 역할
- **Dry-run 우선**: 모든 일괄 변경 스크립트는 `--dry-run` 모드를 먼저 실행, diff 출력 후 실제 적용
- **랜덤 검증**: 각 Phase 후 5% 랜덤 샘플 (전체 ~25개 파일) 수동 점검
- **한글 안전성**: 모든 텍스트 변환에 UTF-8 강제, NFC 정규화, BOM 미사용

---

## 10. 산출물

### 10.1 본 spec에서 만들 파일

1. `docs/superpowers/specs/2026-05-05-obsidian-pkm-infrastructure-design.md` (이 문서)

### 10.2 plan 단계에서 만들 파일

2. `docs/superpowers/plans/2026-05-05-pkm-refactor-plan.md` — Phase별 구체 명령·스크립트
3. `scripts/pkm/inject_frontmatter.py` — Phase 3·4
4. `scripts/pkm/inject_tags.py` — Phase 5
5. `scripts/pkm/consolidate_images.py` — Phase 6
6. `scripts/pkm/resolve_filename_collisions.py` — Phase 7
7. `scripts/pkm/upgrade_readmes_to_moc.py` — Phase 8
8. `scripts/pkm/scan_broken_wikilinks.py` — Phase 9
9. `scripts/pkm/folder_tag_map.json` — 폴더→태그 매핑 데이터

### 10.3 implement 단계에서 만들 파일

10. 신규 디렉터리: `MOCs/`, `_templates/`, `image/`(이미 존재, 통합)
11. 15개 MOC 파일
12. 6개 Templater 템플릿
13. 4개 CSS 스니펫
14. `.obsidian/graph.json` 업데이트
15. `.obsidian/community-plugins.json` 업데이트
16. 모든 노트 frontmatter 확장 (479 파일)

---

## 11. 검증 기준

작업 완료 = 다음 모두 성립:

- [ ] 활성 플러그인 16개 (기존 9 + 신규 7), Tier 2 3개는 사용자 판단
- [ ] 모든 479 노트가 신규 7필드 보유 (스크립트 검증)
- [ ] 기존 4필드(title/date/tags/aliases)는 1건도 손실 없음 (diff 검증)
- [ ] `MOCs/` 폴더 15개 MOC 존재, 각 Dataview 블록이 1개 이상 결과 반환
- [ ] `image/` 외부에 위치한 이미지 0개 (find 검증)
- [ ] 파일명 충돌 0건 (find -print | sort | uniq -d)
- [ ] 그래프 뷰가 hub-and-spoke로 시각화됨 (스크린샷 첨부)
- [ ] obsidian-spaced-repetition이 카드 0개 이상 인식 (한국어 카드 호환성 검증)
- [ ] obsidian-git이 main 브랜치 동기화 정상

---

## 12. 가정과 비범위

### 가정

- macOS 환경 (Darwin 25.2.0), Python 3 사용 가능
- 옵시디언 1.9 이상 (Bases 호환 가능하나 본 디자인은 Dataview 우선)
- iCloud 동기화로 인한 파일 잠금이 일시적으로 발생할 수 있음 → 재시도 로직 포함
- 사용자가 작업 중 옵시디언 앱을 종료한 상태로 유지

### 비범위 (이번 spec 제외)

- 모바일 앱 워크플로우 최적화
- Obsidian Sync 유료 기능 도입
- AI 임베딩·RAG 통합 (CC 스킬 측에서 별도 처리)
- Obsidian Publish 배포
- Anki 양방향 동기화 (obsidian-spaced-repetition만 사용)

---

## 13. 향후 후속 spec 후보

본 spec 완료 후 별도 처리:

- `pkm-content-curation` — evergreen 노트 추출 워크플로우, status 승급 정책
- `pkm-portfolio-pipeline` — Quartz/Hugo 정적 사이트 생성
- `pkm-mobile-optimization` — iOS 옵시디언 워크플로우
- `pkm-ai4pkm-integration` — ai4pkm-helper 오케스트레이터 자동화

---

## 변경 이력

- 2026-05-05: 초안 작성, 사용자 승인 (Option C 확정, 자율 진행 권한 부여)
