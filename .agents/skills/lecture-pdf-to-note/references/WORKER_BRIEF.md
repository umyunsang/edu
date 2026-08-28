# 워커 브리프 — 강의 PDF 정리문서 1개를 작성할 때

이 문서는 `SKILL.md` 를 실제 작업 지시로 풀어 쓴 것이다.
정리문서를 만드는 에이전트는 **이 문서를 처음부터 끝까지 지킨다.**

## 0. 먼저 읽을 것

1. `.agents/skills/lecture-pdf-to-note/SKILL.md` — 변환 규격
2. `ComputerScience/04_systems-infrastructure/parallel-distributed-computing/01. 왜 병렬 처리인가.md`
   — **품질 기준 레퍼런스 노트.** 밀도·구성·시각화 수준을 이 노트에 맞춘다.

## 1. 근거 자료 읽기

추출 번들은 `.ok/local/pdf-extract/<slug>/` 에 이미 있다.

| 파일 | 내용 |
|:--|:--|
| `text.md` | 페이지별 원문. ` ```page N chars=M ` 블록. **유일한 사실 근거** |
| `meta.json` | `sha256`, `pages`, `pdf_created`, `page_chars` |

`text.md` 가 크면 빈 페이지를 걸러서 읽는다.

```bash
cd <repo>/.ok/local/pdf-extract/<slug>
python3 - <<'EOF'
import re
t = open('text.md', encoding='utf-8').read()
for n, c, body in re.findall(r'```page (\d+) chars=(\d+)[^\n]*\n(.*?)\n```', t, re.S):
    if int(c) == 0:
        continue
    b = re.sub(r'\n{3,}', '\n', body.strip())
    b = re.sub(r'[ \t]{3,}', '  ', b)
    print(f'--[p{n} c={c}]--')
    print(b[:900])
EOF
```

`chars` 가 낮은 페이지는 내용이 그림 안에 있다는 뜻이다.
그중 정말 중요한 도해만 `![[pdf/<원본파일명>.pdf#page=N]]` 로 임베드하고,
**임베드 앞뒤에 반드시 텍스트 설명을 붙인다.** LLM은 PDF를 읽지 못한다.
노트 1개당 임베드는 1~3개면 충분하다.

## 2. 프론트매터 (알파벳 순 유지)

```yaml
---
aliases: []
authority: primary
course: <수업 폴더명>
created: '<meta.json 의 pdf_created 를 YYYY-MM-DD 로>'
date: '<동일>'
review_state: active
semester: '<학기, 예: 3-1>'
source: pdf/<원본 PDF 파일명>
source_pages: <meta.json pages>
source_sha256: <meta.json sha256 앞 16자>
source_type: lecture-slides
status: seedling
tags:
  - type/lecture
  - cs/<도메인>
title: <노트 제목>
type: lecture
updated: '<오늘 날짜>'
---
```

프론트매터 바로 아래에 인라인 그래프 필드를 둔다.

```
domain:: [[ComputerScience/<field>/<분야 인터페이스>|<분야 인터페이스>]]
module:: [[ComputerScience/00_graph-interfaces/courses/<과목 인터페이스>|<과목 인터페이스>]]
source:: `pdf/<파일명>.pdf` (<N>p)
```

## 3. 시각화 요구사항 — 가장 중요

한 노트에 **최소 4종류 이상**의 시각 표현을 섞는다.
글 덩어리로만 된 노트는 반려된다.

- **마크다운 표** — 비교/분류/스펙/용어. 최소 2개
- **Mermaid 다이어그램** — 최소 3개, **서로 다른 종류로**
  `flowchart` · `sequenceDiagram` · `mindmap` · `timeline` · `stateDiagram-v2`
  · `quadrantChart` · `xychart-beta` · `block-beta` · `gantt` · `pie` · `classDiagram`
- **콜아웃**
  - `> [!abstract] 한 줄 요약` — 맨 위 **필수**
  - `> [!info] 정의` / `> [!example] 예시` / `> [!warning] 주의` / `> [!tip]` / `> [!note] 보충`
  - `> [!question]- 스스로 점검` — 맨 아래 **필수**, `-` 를 붙여 접힌 상태로
- **LaTeX** — 성능식·복잡도·주소 계산은 `$...$` / `$$...$$`
- **코드 블록** — 언어 태그 필수 (`c`, `cuda`, `cpp`, `bash`, `python`)

### Mermaid 문법 주의

- 라벨에 `(` `)` `:` `,` `/` 가 들어가면 **반드시 큰따옴표**로 감싼다 → `A["ILP (슈퍼스칼라)"]`
- 줄바꿈은 `<br/>`
- 한 다이어그램당 노드 12개 이하
- `quadrantChart` 의 축/사분면 라벨에는 특수문자를 넣지 않는다
- 한글 라벨 사용 가능

## 4. 노트 구조

1. 프론트매터 + 인라인 필드
2. `# 제목`
3. `> [!abstract] 한 줄 요약`
4. `## 이 강의의 지도` — mermaid `mindmap`
5. `## 1. ~` 본문 섹션들 — 각 섹션 시작에 `> [!quote] 슬라이드 근거` 로 페이지 범위 표기
6. `## 핵심 정리` — 개념 / 한 줄 정의 / 왜 중요한가 3열 표
7. `> [!question]- 스스로 점검` — Q&A 5~6개
8. `## 슬라이드 근거` — 섹션 ↔ 페이지 매핑 표
9. `## 관련 노트` — 같은 과목 노트 위키링크 3~5개

위키링크는 전체 경로 형식으로 쓴다.

```
[[ComputerScience/04_systems-infrastructure/parallel-distributed-computing/02. 병렬 컴퓨터의 기본 아키텍처|02. 병렬 컴퓨터의 기본 아키텍처]]
```

## 5. 금지 사항

1. **슬라이드에 없는 내용을 지어내지 않는다.** 보충이 필요하면 `> [!note] 보충` 으로 분리 표기
2. 슬라이드를 그대로 옮기지 않는다. **재구성해서 설명**하고, 원문의 어색한 기계번역 투는 자연스러운 한국어로 고친다
3. 고유명사·API·명령어·수식 기호는 원문 유지
4. `![[이미지.png]]` 형태의 새 이미지 임베드를 만들지 않는다 (그 파일은 존재하지 않는다)
5. 분량은 레퍼런스 노트(약 400줄) 수준을 목표로 하되, 내용이 적은 덱이면 무리해서 늘리지 않는다

## 6. 작성 후 검증 (실패 0이 될 때까지)

```bash
cd <repo>
node scripts/check_mermaid.mjs --modules "$MERMAID_MODULES" "<만든 노트 경로>"
```

실패가 남은 채로 작업을 끝내지 않는다.
