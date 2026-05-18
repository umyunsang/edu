# 시험 학습용 샘플 문제 (3가지)

시험에 익숙해지기 위한 축약된 샘플 문제입니다. 각 샘플은 3단계로 구성되어 있고, 풀이 코드는 별도 폴더에 있습니다.

샘플별 성공 기준 체크리스트 모음은 `exam/SUCCESS_SAMPLE.md` 에서 한 번에 확인할 수 있습니다.

## 샘플 1️⃣: 협업 워크플로우 기초 (10분)
**원본 시험:** 문항 1 협업 워크플로우 구성 (축약)

### 🎯 목표
GitHub Flow를 따르는 간단한 협업 작업 완료

### 3단계 가이드
1. **Step 1: 브랜치 전략 선언**
   - GitHub Flow 선택 이유를 1문장으로 작성
   - 예: "메인 브랜치 중심, 빠른 배포"

2. **Step 2: 브랜치 생성 및 작업**
   - `feature/add-greeting` 브랜치 생성
   - `app/greeting.py` 파일 수정 (1-2줄)

3. **Step 3: PR 생성 및 체크리스트**
   - PR 제목: `[Feature] Add greeting endpoint`
   - PR 설명에 변경사항 3줄 + 테스트 결과 1줄 포함

### ✅ 성공 기준
- [ ] PR 링크 1개
- [ ] 브랜치 전략 설명 1문장
- [ ] 체크리스트 마크업 포함

### 💾 풀이 코드 위치
→ `sample-solutions/sample-1-collaboration/`

### 🧩 TODO minimal 버전
→ `sample-solutions-minimal/sample-1-collaboration/`

---

## 샘플 2️⃣: CI 파이프라인 기초 (15분)
**원본 시험:** 문항 2 CI 파이프라인 구축 (축약)

### 🎯 목표
GitHub Actions 기본 CI 워크플로우 작성

### 3단계 가이드
1. **Step 1: 트리거 설정**
   - `.github/workflows/simple-ci.yml` 생성
   - `push`, `pull_request` 이벤트 트리거

2. **Step 2: 단일 Job 작성**
   - 체크아웃 → 의존성 설치 → Lint (3단계)
   - Python 3.10 단일 버전만 사용

3. **Step 3: 실행 및 로그 확인**
   - 워크플로우 실행 (push 또는 수동)
   - 성공 로그 스크린샷 또는 링크 저장

### ✅ 성공 기준
- [ ] YML 파일 작성
- [ ] Actions 실행 링크 1개
- [ ] 성공 로그 확인

### 💾 풀이 코드 위치
→ `sample-solutions/sample-2-ci-basics/.github/workflows/ci.yml`

### 🧩 TODO minimal 버전
→ `sample-solutions-minimal/sample-2-ci-basics/`

---

## 샘플 3️⃣: 테스트 기초 (15분)
**원본 시험:** 문항 3 Shift-left 테스트 (축약)

### 🎯 목표
단위 테스트 2개 작성 및 TDD 사이클 경험

### 3단계 가이드
1. **Step 1: 실패 테스트 작성**
   - `tests/test_greeting.py` 생성
   - `test_greeting_not_empty()` - 일부러 실패
   - `pytest` 실행 후 실패 확인

2. **Step 2: 구현으로 테스트 통과**
   - `app/greeting.py` 간단한 함수 구현
   - 테스트 다시 실행 → 성공

3. **Step 3: 테스트 추가 작성**
   - `test_greeting_contains_name()` - 조건 검증
   - TDD 사이클 1회 완료 기록

### ✅ 성공 기준
- [ ] 테스트 2개 작성
- [ ] pytest 통과 로그
- [ ] 실패→성공 흐름이 보이는 커밋 2개

### 💾 풀이 코드 위치
→ `sample-solutions/sample-3-testing/tests/` + `app/`

### 🧩 TODO minimal 버전
→ `sample-solutions-minimal/sample-3-testing/`

---

## 🚀 샘플 학습 순서 추천
1. **샘플 1** (협업): Git/GitHub 기본 익숙해지기 → 5~10분
2. **샘플 2** (CI): GitHub Actions 최소 구조 이해 → 10~15분
3. **샘플 3** (테스트): 테스트 코드 작성 경험 → 15~20분

**총 학습 시간: 30~45분**

---

## 📝 실제 시험 진행 팁
- 샘플 완료 후 본 시험 문항과 비교
- 각 샘플의 "3단계"를 확장하면 시험 수준
- 실패/에러는 학습 과정 → 기록하면서 진행