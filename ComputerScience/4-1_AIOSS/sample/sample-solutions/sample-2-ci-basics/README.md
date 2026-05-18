# 샘플 2: CI 파이프라인 기초

## 📌 문제 설명
GitHub Actions 기본 CI 워크플로우를 작성합니다. Lint 1개 단계만 포함한 최소 버전입니다.

## 🎯 요구사항 (3단계)

### Step 1: 트리거 및 기본 구조 설정
`.github/workflows/ci.yml` 생성:

```yaml
name: Simple CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    # 다음 단계에서 추가
```

### Step 2: 작업 단계 추가
위 파일에 steps 추가:

```yaml
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install ruff
      
      - name: Lint check
        run: ruff check .
```

### Step 3: 실행 및 확인
- 파일 저장 후 `git push origin main` 또는 PR 생성
- GitHub → Actions 탭에서 실행 확인
- 성공 로그 스크린샷 저장 또는 링크 기록

## ✅ 성공 기준
- [ ] `.github/workflows/ci.yml` 파일 생성
- [ ] Actions 탭에서 실행 기록 1개 이상
- [ ] 성공 상태(✓ 초록색) 확인

## 💾 풀이 보기
[-> 완전한 워크플로우](./complete_ci.yml)

