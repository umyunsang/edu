---
title: PKM Refactor Completion Report
date: 2026-05-05
status: complete
branch: pkm-refactor
related_spec: 2026-05-05-obsidian-pkm-infrastructure-design.md
related_plan: 2026-05-05-pkm-refactor-plan.md
---

# PKM Refactor Completion Report

작업: 옵시디언 학부 커리큘럼 볼트(479+ 노트)에 MOC-first Evergreen PKM 인프라 전면 재정비 (Option C, 사용자 자율 진행 권한 부여 모드).

## 요약

| 항목 | 결과 |
|---|---|
| 총 노트 수 | **461개** (`.md`, vault tooling/skill 제외) |
| frontmatter 11필드 완비율 | **100% (461/461)** |
| 신규 MOC | **16개** (Home + 15 도메인) — `MOCs/` 디렉터리 |
| README → MOC 업그레이드 | **33개** 코스별 README |
| 산재 이미지 통합 | **129개** → `image/{course}_*.png` |
| 파일명 충돌 해결 | **28건** rename (course-prefix) |
| 도메인 태그 자동 주입 | **438 파일에 602 태그 추가** |
| 깨진 위키링크 (실제) | **15건** (대부분 stub, 사용자 점진 작성) |
| 신규 Templater 템플릿 | **6종** (lecture/literature/permanent/project/MOC/daily) |
| CSS 스니펫 | **4종** (fonts, callouts, status-badges, moc-styling) |
| 활성 플러그인 | **9 → 23** (신규 7 Tier-1 + 활성화 4 + Tier-2 3) |
| 그래프 색상 그룹 | **9개** (태그 기반 클러스터) |
| Linter 한국어 안전 룰 | **16 활성 / 6 비활성** |

## frontmatter 분포

**type:**
- `lecture`: 365
- `MOC`: 50 (16 도메인 MOC + 33 README + 1 Home)
- `project`: 42
- `literature`: 4

**status:**
- `seedling`: 408 (점진 큐레이션 후보)
- `evergreen`: 49 (MOC + 사용자 기존)
- `정리완료`, `복습필요`, `암기필요`: 4 (사용자 기존 status 보존됨)

**semester:**
- `2-1`: 146 / `2-2`: 86 / `3-1`: 92 / `3-2`: 37 / `4-1`: 15
- `1-2`: 11
- `elective`: 45 / `cert`: 4 / `extracurricular`: 9 / `all`: 16

## 태그 분포 Top 25

```
365  type/lecture
 87  cs/systems
 62  cs/ml
 56  cs/se
 50  type/MOC
 46  cs/ai
 43  cs/db
 42  type/project
 36  math/probability
 36  math/statistics
 30  skill/python
 28  cs/algorithms
 23  cs/open-source
 20  meta/cert
 18  cs/llm / cs/nlp
 17  cs/devops
 15  math/discrete
 15  skill/javascript
 12  cs/cv / skill/linux
 10  skill/docker / meta/extracurricular
  7  cs/dl
```

## Git 커밋 이력 (`pkm-refactor` 브랜치)

```
spec: Obsidian PKM infrastructure full refactor design (Option C)
plan: PKM refactor implementation plan (19 tasks, 11 phases)
chore(pkm): scaffold scripts/pkm/ for refactor automation
feat(pkm): lib_frontmatter with Korean-safe NFC normalization + tests
data(pkm): folder→tag mapping covering all 36 course directories
feat(pkm): Phase 1 — plugins + theme + CSS + Templater + root cleanup
feat(pkm): Phase 2 — 16 MOCs (Home + 15 domain MOCs with Dataview blocks)
feat(pkm): Phase 3+4 — inject/extend frontmatter on 445 notes
feat(pkm): Phase 5 — inject domain tags from folder mapping
refactor(pkm): Phase 6 — consolidate 129 scattered images into /image/
refactor(pkm): Phase 7 — resolve filename collisions with course-prefix rename
feat(pkm): Phase 8 — upgrade course READMEs to MOC type with Dataview blocks
feat(pkm): Phase 9 — broken wikilink scanner + report
chore(pkm): Phase 10 — re-serialize 461 notes with sorted YAML keys
docs(pkm): completion report
```

## 다음 단계 (사용자 점진 작업)

### 즉시 가능
- [ ] 옵시디언 시작 → 신규 플러그인 다운로드 자동 처리됨
- [ ] Settings → About → Theme → "Minimal" 검색 후 설치 (kepano 테마)
- [ ] Iconize 플러그인에서 폴더별 아이콘 지정
- [ ] Excalibrain에서 Home MOC를 central로 설정 → 방사형 그래프 확인
- [ ] Graph Analysis로 중심성 측정 → 진짜 허브 노트 식별

### 점진 큐레이션 (다음 학기 동안)
- [ ] `seedling` → `budding` 승급: 강의 후 정리한 노트 재독 + 자기 언어로 재서술
- [ ] `budding` → `evergreen` 승급: 한 문장으로 제목이 가능한 idea-shaped 노트로 정련
- [ ] 15개 깨진 위키링크 정리 (`docs/superpowers/specs/2026-05-05-broken-wikilinks-report.txt` 참조)
- [ ] obsidian-spaced-repetition 카드 작성 시작 (`#flashcards/cs/ml` 등)
- [ ] MOC 본문의 빈 섹션 채우기 (Foundations / Supervised / Unsupervised 등)

### 후속 spec 후보
- `pkm-content-curation` — evergreen 추출 워크플로우, status 승급 정책
- `pkm-portfolio-pipeline` — Quartz/Hugo 정적 사이트 생성
- `pkm-mobile-optimization` — iOS 옵시디언 워크플로우
- `pkm-ai4pkm-integration` — ai4pkm-helper 오케스트레이터로 자동화

## 검증 결과

✅ frontmatter 완비율 100% (461/461)
✅ 모든 MOC가 `type: MOC` + `type/MOC` 태그 보유
✅ 산재 이미지 0개 (`/image/` 외부)
✅ 파일명 충돌 0건 (README 제외)
✅ 폴더 명명 규칙 100% 준수 (`학기_영문약어`)
✅ 한글 텍스트 UTF-8 무결성 (NFC 정규화)
✅ obsidian-git 자동백업 main 브랜치에서 정상 작동
✅ pkm-refactor 브랜치는 main과 분리 — 언제든 git revert 가능

## 머지 전 체크리스트

- [ ] 옵시디언 앱에서 vault 열어 그래프 뷰 시각 확인
- [ ] 5% 랜덤 샘플 (20개 노트) 본문 보존 확인
- [ ] Dataview 블록이 결과 반환하는지 MOC 1~2개에서 확인
- [ ] obsidian-git가 새 파일 변경을 추적하는지 확인
- [ ] `git checkout main && git diff main..pkm-refactor --stat` 로 변경 규모 확인
- [ ] 머지: `git checkout main && git merge pkm-refactor` (또는 PR 생성)

## 영향 통계

```
총 변경 파일: ~700
총 커밋: 14
신규 디렉터리: MOCs/, _templates/, image/_archive/, scripts/pkm/, docs/superpowers/
신규 파일: 16 MOC + 6 템플릿 + 7 스크립트 + 4 CSS + 6 문서 = 39
수정 파일: 461 노트 (frontmatter 확장) + 21 노트 (이미지 위키링크 갱신) = ~480
이동 파일: 129 이미지 + 28 노트 rename + 5 root archive = 162
삭제 파일: 1 (...md, malformed)
```

---

**Status: Complete and ready for user review.**
