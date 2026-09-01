---
name: creative-writing-assignment
description: Use for 창의적글쓰기 수업의 실습/과제 글쓰기. Trigger when drafting, revising, diagnosing, structuring, or finalizing writing assignments in the creative-writing course, especially 자기소개서, 성장과정, 성격 장단점, STARL, 힘글쓰기, Power Writing, or PDF-grounded revision requests.
allowed-tools:
  - Read
  - Edit
  - Glob
  - Grep
  - Bash
---

# Creative Writing Assignment

이 스킬은 창의적글쓰기 과제 글쓰기를 `수업 PDF 기반 RAG처럼` 수행하기 위한 스킬이다. 실제 벡터 RAG가 아니어도, 매번 PDF 추출본을 검색하고 직접 확인한 뒤 그 내용에 맞춰 구조와 문장을 만든다.

핵심 원칙: **구조를 먼저 만들고 내용을 끼워 넣지 않는다.** 반드시 `PDF 검색 -> 원문 확인 -> 사용자 기록 수집 -> 근거 등급화 -> 구조 설계 -> 문장 작성/수정 -> 검증` 순서로 진행한다.

## Scope

사용자가 다음 중 하나를 요청하면 이 스킬을 사용한다.

- 창의적글쓰기 과제 작성, 수정, 검토, 최종본 제작
- 자기소개서, 성장과정, 성격상의 장단점, 지원동기, 입사 후 포부
- STARL, 힘글쓰기, Power Writing, Hi Five, MECE, 3C, 두괄식/미괄식 적용
- `수업자료 PDF를 바탕으로`, `PDF 내용 확인`, `RAG처럼`, `아카이브 근거`, `글쓰기기법 적용` 요청

## Required Source Paths

작업 기준 디렉터리는 보통 다음이다.

```bash
cd "/Users/um-yunsang/Library/Mobile Documents/iCloud~md~obsidian/Documents/edu/ComputerScience/4-1_creative-writing"
```

주요 자료:

- 원본 PDF: `pdf/26-1창의적글쓰기 강의자료_중핵교양.pdf`
- PDF 추출본: `knowledge/source_text/pages/26-1창의적글쓰기_강의자료_중핵교양/page-###.txt`
- 검색 도구: `tools/search_sources.py`
- 수업 절차 문서: `docs/WRITING_WORKFLOW.md`
- 과제 초안: `assignments/drafts/`
- 최종본: `assignments/final/`

## Non-Negotiable Workflow

### 1. 과제 조건을 먼저 고정한다

수정하거나 작성하기 전에 문항, 글자수, 제출 형식, 필수 조건을 요약한다.

자기소개서 2차 과제 기준:

- 문항 1: `성장과정을 중심으로 자신에 대해 자유롭게 표현하시오.` 700자 이상
- 문항 2: `성격상의 장단점 및 단점 개선 노력에 대해 기술하시오.` 800자 이상
- 공통: 제목, 과거-현재-미래, 에피소드, STARL, 공백 포함 글자수, 표절률 20% 미만
- 제출: 원본 DOCX에 10pt 글자로 입력, PDF 제출 금지

### 2. PDF를 반드시 검색하고 원문 페이지를 읽는다

사용자 요청이 단순 문장 교체가 아니라 구조/내용/방향 수정이면, 바로 쓰지 말고 먼저 검색한다.

기본 검색:

```bash
python3 tools/search_sources.py "자기소개서 작성" -n 12
python3 tools/search_sources.py "성장과정 작성시 고려사항" -n 12
python3 tools/search_sources.py "성격의 장단점 작성시 고려사항" -n 12
python3 tools/search_sources.py "Power Writing 힘 글쓰기" -n 12
python3 tools/search_sources.py "서론 시작하는 방법" -n 12
python3 tools/search_sources.py "MECE" -n 12
python3 tools/search_sources.py "3C Correct Concise Clear" -n 12
```

자기소개서 작업에서는 최소한 아래 페이지를 직접 확인한다.

| 목적 | 필수 페이지 |
|---|---|
| 자기소개서 기본 | p.87 |
| 사실 확인/원인-과정-결과 | p.93 |
| 소제목 + STARL | p.97 |
| 성장과정 | p.104, p.108, p.115 |
| 성격 장단점 | p.117, p.126, p.127 |
| 서론 시작 방식 | p.58~67 중 관련 페이지 |
| 힘글쓰기/Power Writing | p.44 |
| MECE | p.50 |
| Technical Writing 3C | p.14 |

원문 확인 예:

```bash
nl -ba knowledge/source_text/pages/26-1창의적글쓰기_강의자료_중핵교양/page-087.txt
nl -ba knowledge/source_text/pages/26-1창의적글쓰기_강의자료_중핵교양/page-115.txt
nl -ba knowledge/source_text/pages/26-1창의적글쓰기_강의자료_중핵교양/page-126.txt
```

PDF에서 확인하지 않은 교수 의도, 채점 기준, 글쓰기 규칙을 단정하지 않는다.

### 3. 사용자 기록과 아카이브를 찾는다

PDF 규칙을 확인한 뒤, 그 규칙에 맞는 사용자 근거를 찾는다. 구조에 맞춰 내용을 끼우지 말고, 근거가 실제로 있는 내용만 구조에 올린다.

우선 확인할 파일:

```bash
rg --files assignments/drafts
sed -n '1,220p' assignments/drafts/self_intro_assignment2_starl_skeleton.md
sed -n '1,180p' assignments/drafts/deco_extracurricular_evidence.md
find .. -maxdepth 2 -type d | sort
```

사용자 근거 등급:

| 등급 | 의미 | 예시 | 사용 규칙 |
|---|---|---|---|
| A | 파일/시스템으로 확인되는 객관 기록 | DECO 수료 기록, 전공 폴더, 프로젝트 파일, Obsidian/GitHub 기록 | 문단 중심 근거로 사용 가능 |
| B | 사용자 제공 고유명사 경험 | 성신공업 지게차공장, 하남돼지집 정직원, 맥도날드 멀티 정직원, 대학가 아르바이트 | 에피소드 근거로 사용하되 과장 금지 |
| C | 사용자 자기해석/회고 | 해설을 먼저 보지 않는 학습 방식, 어머니가 수학 선생님, 오래 생각하는 습관 | 핵심 문단 금지. 배경 설명으로 짧게만 사용 |
| D | 외부 최신 자료 | AX, AI, GraphRAG, 미래 역량 자료 | 본문에는 짧게 반영. 시대비판 과장 금지 |

금지:

- C등급 자료만으로 한 문단을 만들지 않는다.
- A등급 자료를 목록처럼 나열하고 끝내지 않는다. 반드시 `무엇을 증명하는지`를 붙인다.
- DECO 미수료/탈락 항목은 제출 본문에 쓰지 않는다.
- `기존 방식의 인재는 필요 없다`처럼 타인을 깎는 문장은 쓰지 않는다.

### 4. 구조는 PDF 규칙과 근거 매핑 후에 만든다

작성 전 반드시 짧은 매핑표를 만든다. 사용자가 “바로 적용”을 요청해도, 내부적으로 이 표를 먼저 채운 뒤 수정한다.

```md
| 문항 | 문단 | 핵심 기능 | PDF 근거 | 사용할 사용자 근거 | 등급 | 적용 기법 | 위험 |
|---|---:|---|---|---|---|---|---|
| 1 | 1 | 현재의 객관 기록 | p.87, p.93 | DECO, Obsidian/GitHub | A | 두괄식, Power 1/2 | 목록처럼 보일 위험 |
```

문항 1 권장 구조:

- 1문단: 현재의 객관 기록 제시
- 2문단: 컴퓨터공학/AI 관심의 배경 설명
- 3문단: 관심이 행동으로 바뀐 과정
- 4문단: 미래 방향 재주장

문항 2 권장 구조:

- 1문단: 장점 두괄식
- 2문단: 현장 에피소드
- 3문단: 현재 IT 학습 방식으로 확장
- 4문단: 단점과 개선 행동

단, 이 구조는 고정 템플릿이 아니다. PDF와 사용자 근거를 확인한 뒤 더 자연스러운 흐름이 있으면 바꾼다.

### 5. 글쓰기기법을 적용한다

문장화할 때 적용 순서:

1. 도입 방식 선택
   - 객관 기록 중심: 구체 자료 제시형 또는 주제 제시형
   - 경험 중심: 자신의 경험/생각으로 시작
   - 문항명 반복 시작 금지

2. STARL 숨은 구조
   - 제출문에 S/T/A/R/L 표기를 드러내지 않는다.
   - 상황, 과제, 행동, 결과, 배움이 자연스럽게 읽혀야 한다.

3. Power Writing
   - Power 1: 주제/주장
   - Power 2: 근거/방법
   - Power 3: 증명/사례/자료
   - Power 4: 결론/재강조

4. MECE
   - 같은 층위 문단은 중복 없이, 누락 없이 나눈다.
   - 같은 말이 반복되면 삭제하거나 문단 기능을 바꾼다.

5. 3C
   - Correct: 사실과 자기해석 분리
   - Concise: 긴 나열 압축
   - Clear: 한 문장에 한 주장

### 6. 검증 후에만 완료라고 말한다

완료 전 확인:

- PDF 근거를 실제로 검색하고 원문 페이지를 읽었는가?
- 각 문항이 p.87, p.93, p.97, p.115 또는 p.126 중 관련 규칙에 맞는가?
- 주요 문단이 A/B 근거를 포함하는가?
- C등급 자료만 있는 문단이 없는가?
- STARL의 사건성이 보이는가?
- Power Writing의 주제-근거-증명-결론 기능이 있는가?
- MECE 중복이 없는가?
- 문장 길이가 과도하게 길지 않은가?
- 글자수 조건을 넘는가?
- 제출문에 출처표기, 근거 메모, Markdown 기호가 섞이지 않았는가?

글자수 확인 예:

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('assignments/drafts/self_intro_assignment2_full_draft.md')
s=p.read_text()
parts=s.split('## 2. 성격상의 장단점 및 단점 개선 노력에 대해 기술하시오.')
q1=parts[0].split('###',2)[2].split('최종 글자수:',1)[0].strip()
q2=parts[1].split('###',1)[1].split('최종 글자수:',1)[0].strip()
print(len(q1), len(q2))
PY
```

## Output Rules

- 제출용 본문에는 강의자료 출처를 넣지 않는다.
- 근거와 출처는 스켈레톤, 근거 메모, 또는 작업 설명에만 남긴다.
- 사용자가 초안 수정을 요청하면 파일을 직접 수정한다.
- 사용자가 “먼저 제안”을 요청하면 적용하지 말고 후보와 근거만 제시한다.
- 수업 PDF 확인 없이 “자연스럽게 고쳤다”고 말하지 않는다.

## Common Failure Modes

- 문항 이름을 첫 문장에 반복해서 부자연스럽게 시작함
- 구조에 맞추려고 없는 내용을 끼워 넣음
- 사용자 자기해석(C등급)을 객관 근거처럼 사용함
- DECO 수료 기록을 이력서처럼 나열하고 사건으로 만들지 못함
- 성격 장단점에서 장점을 너무 많이 벌림
- 단점 개선을 “노력하고 있습니다”로만 쓰고 실제 행동을 쓰지 않음
- AX 시대 담론이 길어져 자기소개서가 설명문처럼 변함
