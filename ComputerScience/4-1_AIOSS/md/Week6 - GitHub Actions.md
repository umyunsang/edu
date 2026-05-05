---
course: AIOSS
created: '2026-04-06'
semester: 4-1
source: ''
status: seedling
tags:
- AIOSS
- CI-CD
- DevOps
- YAML
- automation
- github-actions
- type/lecture
- workflow
title: Automation with GitHub Actions
type: lecture
updated: '2026-05-05'
week: 6
---

# Automation with GitHub Actions

## 학습 목표

- **아키텍처 이해**: Workflow, Job, Step, Action 등 핵심 구성 요소와 작동 원리 파악
- **YAML 작성**: 워크플로우 정의를 위한 YAML 문법을 익히고, 실제 실행 가능한 자동화 스크립트 작성
- **트리거 & 보안**: Push, PR 등 다양한 이벤트 트리거 활용법과 Secrets를 통한 민감 정보 관리 방법 습득
- **CI 파이프라인**: Node.js, Python 등 실제 프로젝트에 적용 가능한 기본 CI(지속적 통합) 파이프라인 구축

---

## GitHub Actions란?

GitHub 저장소에서 직접 실행되는 **자동화 플랫폼**으로, 개발 워크플로우를 저장소 내에서 이벤트 기반으로 자동 실행한다.

**주요 용도:**
- CI/CD (지속적 통합/배포)
- 테스트 자동화
- 코드 품질 체크
- 배포 자동화
- 이슈/PR 관리
- 기타 모든 자동화

### 왜 GitHub Actions인가?

**주요 장점:**
- GitHub 네이티브 통합
- Public 레포 무료
- 강력한 Marketplace 생태계
- 다양한 이벤트 트리거
- Matrix 빌드 지원
- 간단한 YAML 설정

| 기능 | GitHub Actions | Jenkins | GitLab CI | CircleCI |
|------|---------------|---------|-----------|----------|
| 설정 난이도 | 매우 쉬움 | 복잡 | 쉬움 | 쉬움 |
| 호스팅 | GitHub 제공 | 자체 호스팅 | GitLab 제공 | 클라우드 |
| 가격 | 무료* | 무료 | 무료* | 무료* |
| VCS 통합 | GitHub 전용 | 모든 VCS | GitLab 전용 | 모든 VCS |
| 생태계 | 매우 큼 | 큼 | 중간 | 중간 |

---

## GitHub Actions 아키텍처

### 핵심 개념

```
Repository > Workflow (.yml) > Event (Trigger) > Job (Execution Unit) > Steps (Sequential Tasks) > Runner
```

### Workflow 예시 구조

```yaml
# .github/workflows/example.yml
name: My Workflow           # 워크플로우 이름
on: [push, pull_request]    # 트리거 이벤트

jobs:                       # 작업 정의
  build:
    runs-on: ubuntu-latest  # 러너 환경
    steps:                  # 단계
    - uses: actions/checkout@v3
    - run: echo "Hello World"
```

**각 키워드 설명:**

| 키워드 | 설명 |
|--------|------|
| `name` | 워크플로우 식별 이름 |
| `on` | 이벤트 트리거 (push, pull_request, schedule, workflow_dispatch, release, issues 등) |
| `jobs` | 작업 정의 집합, 기본 병렬 실행 |
| `runs-on` | 실행 환경(Runner) |
| `steps` | Job 내 순차 실행 명령. `uses`(외부 Action) 또는 `run`(셸 명령) |

### Event (이벤트) 트리거 종류

| 이벤트 | 설명 |
|-------|------|
| `push` | 코드 커밋을 저장소에 푸시하거나 태그를 푸시할 때. 가장 기본적인 CI 트리거 |
| `pull_request` | PR이 생성되거나 업데이트될 때. 코드 리뷰 전 자동 테스트 |
| `schedule` | POSIX cron 구문으로 주기적 실행 |
| `workflow_dispatch` | GitHub Actions 탭에서 수동 실행. inputs 파라미터 지원 |
| `release` | 새로운 릴리스가 생성/게시될 때. 배포 자동화용 |
| `issues` | 이슈 생성, 수정, 삭제, 라벨 변경 시 |

### Job & Runner

**Job** — 워크플로우 내 독립적 작업 단위
- 기본 병렬 실행
- `needs`로 순차 실행 가능
- 별도 Runner에서 실행

**Runner** — Job을 실제로 실행하는 서버 환경
- **GitHub-hosted**: `ubuntu-latest`, `windows-latest`, `macos-latest`
- **Self-hosted**: 자체 서버 사용

### Step vs Action

**Step** — Job 내 순차 실행 개별 작업 단위
- `uses`: 외부 Action 사용
- `run`: 셸 명령 실행

**Action** — 복잡한 작업을 캡슐화한 재사용 가능 단위 (Plugin)
- Public Actions (Marketplace)
- Private Actions
- Docker Container Actions
- JavaScript / Composite Actions

---

## 첫 워크플로우 작성하기

### Hello World 워크플로우

```yaml
# .github/workflows/hello.yml
name: Hello World

on: push  # push 이벤트 시 실행

jobs:
  greet:
    runs-on: ubuntu-latest
    steps:
    - name: Say Hello
      run: echo "Hello, GitHub Actions!"
```

### 파일 위치 및 실행 과정

```
repository/
  .github/
    workflows/
      hello.yml
      test.yml
      deploy.yml
  src/
  README.md
```

**실행 과정:**

1. 코드 푸시 (Push)
2. `.github/workflows/` 확인
3. 이벤트 매칭 워크플로우 검색
4. Runner 할당
5. Job 실행 (병렬/순차)
6. Step별 명령/액션 수행
7. 결과 리포트 및 로그 저장
8. 자동화 완료 (Success/Fail)

> [!important] 워크플로우 파일은 반드시 `.github/workflows/` 경로에 위치해야 인식됩니다.

---

## 이벤트 트리거

### Push 이벤트 패턴

**모든 Push:**

```yaml
on: push
```

**특정 브랜치 지정:**

```yaml
on:
  push:
    branches:
    - 'main'
    - 'develop'
    - 'releases/**'  # 와일드카드
```

**파일 경로 필터링:**

```yaml
on:
  push:
    paths:
    - 'src/**'
    - '**.js'
    - '!docs/**'  # 문서 제외
```

**제외 조건 (Ignore):**

```yaml
on:
  push:
    branches-ignore:
    - 'feature/experimental'
    paths-ignore:
    - '**.md'
```

### pull_request 트리거 설정

```yaml
# 1. 기본: PR 생성, 업데이트, 재오픈 시
on: pull_request

# 2. 활동 유형(types) 상세 제어
on:
  pull_request:
    types:
    - opened
    - synchronize
    - reopened

# 3. 대상 브랜치 제한
on:
  pull_request:
    branches:
    - main
    - 'releases/**'
```

### Schedule (Cron) 트리거

```yaml
on:
  schedule:
  - cron: '0 0 * * *'    # 매일 자정 (UTC)
  - cron: '0 9 * * 1'    # 매주 월요일 오전 9시
  - cron: '0 0 1 * *'    # 매월 1일 자정
  - cron: '30 5 * * 1-5' # 평일 오전 5시 30분
```

> [!warning] UTC 시간대 주의
> 한국 시간(KST)은 UTC보다 9시간 빠릅니다. 예: 한국 오전 9시 = UTC 0시 (`'0 0 * * *'`)

### 수동 실행 (workflow_dispatch)

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production
      log_level:
        description: 'Log level'
        required: false
        default: 'info'
```

### 복합 이벤트 조건

```yaml
# OR 조건 (여러 이벤트 중 하나)
on: [push, pull_request]

# AND 조건 (필터 조건 모두 만족)
on:
  push:
    branches:
    - main
    paths:
    - 'src/**'
```

> [!tip] Best Practice
> 불필요한 빌드 실행을 방지하여 GitHub Actions 사용량을 절약하려면 `paths`나 `branches` 필터를 적극 활용하세요.

---

## 기본 Actions 활용

### actions/checkout@v3

```yaml
steps:
  # 1. 기본 코드 체크아웃 (필수!)
  - uses: actions/checkout@v3

  # 2. 전체 히스토리 가져오기
  - uses: actions/checkout@v3
    with:
      fetch-depth: 0

  # 3. 특정 브랜치 체크아웃
  - uses: actions/checkout@v3
    with:
      ref: develop
```

### Node.js 설정 및 캐싱

```yaml
steps:
  # 기본 버전 설정
  - uses: actions/setup-node@v3
    with:
      node-version: '18'

  # 의존성 캐싱 (추천!)
  - uses: actions/setup-node@v3
    with:
      node-version: '18'
      cache: 'npm'

  # 버전 범위 지정
  - uses: actions/setup-node@v3
    with:
      node-version: '18.x'  # 최신 18.x 자동 선택
```

### Python 환경 설정

```yaml
steps:
  - name: Setup Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.10'
      cache: 'pip'

  - name: Install dependencies
    run: |
      python -m pip install --upgrade pip
      pip install -r requirements.txt
```

### 아티팩트 업/다운로드

```yaml
# Job 1: Build & Upload
jobs:
  build:
    steps:
    - run: npm run build
    - uses: actions/upload-artifact@v3
      with:
        name: build-output
        path: dist/

# Job 2: Deploy & Download
  deploy:
    needs: build
    steps:
    - uses: actions/download-artifact@v3
      with:
        name: build-output
        path: dist/
```

---

## Secrets 및 환경 변수

### Secrets 설정

1. **Settings** > **Secrets and variables** > **Actions**
2. **New repository secret** 버튼 클릭
3. Name: `API_KEY`, Secret 값 입력

> [!warning] 저장된 Secret 값은 다시 볼 수 없으므로, 생성 시 신중하게 입력하세요.

### Secrets 사용 예시

```yaml
steps:
- name: Use secret
  run: |
    echo "API Key: ${{ secrets.API_KEY }}"
  env:
    API_KEY: ${{ secrets.API_KEY }}
```

> [!warning] 보안 주의사항
> 로그 출력 시 Secrets 값은 자동으로 마스킹(`***`) 처리되지만, 가능한 한 값을 직접 출력하는 행위는 피해야 합니다.

### 환경 변수와 Context

```yaml
# 변수 스코프: Workflow > Job > Step
env:
  NODE_ENV: production       # Workflow 레벨

jobs:
  build:
    env:
      DB_URL: postgres://localhost  # Job 레벨
    runs-on: ubuntu-latest
    steps:
    - name: Build Step
      env:
        API_KEY: 123456      # Step 레벨 (Highest priority)
      run: npm run build
```

**GitHub 컨텍스트 활용:**

```yaml
- name: Print Context
  run: |
    echo "Repo: ${{ github.repository }}"
    echo "Branch: ${{ github.ref }}"
    echo "Actor: ${{ github.actor }}"
    echo "SHA: ${{ github.sha }}"
    echo "Event: ${{ github.event_name }}"
```

---

## CI 워크플로우 실전 예시

### Node.js 프로젝트 CI

```yaml
# .github/workflows/ci.yml
name: Node.js CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]

    steps:
    - uses: actions/checkout@v3
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    - name: Install dependencies
      run: npm ci
    - name: Run tests
      run: npm test
```

### Python 프로젝트 CI

```yaml
name: Python CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pylint
    - name: Run Lint
      run: pylint src/
    - name: Run Tests
      run: pytest --cov
```

### Go 프로젝트 CI

```yaml
name: Go CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
        cache: true
    - name: Install dependencies
      run: go mod download
    - name: Run tests
      run: go test -race -coverprofile=coverage.out ./...
    - name: Build
      run: go build ./...
```

---

## 고급 워크플로우 패턴

### Job 의존성 설정 (needs)

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - run: echo "Building..."

  test:
    needs: build  # build 완료 후 실행
    runs-on: ubuntu-latest
    steps:
    - run: echo "Testing..."

  deploy:
    needs: [build, test]  # 둘 다 성공 시 실행 (Fan-in)
    runs-on: ubuntu-latest
    steps:
    - run: echo "Deploying..."
```

### 조건부 실행 (Conditional Execution)

```yaml
steps:
- name: Run only on main
  if: github.ref == 'refs/heads/main'
  run: echo "Main branch"

- name: Run only on PR
  if: github.event_name == 'pull_request'
  run: echo "Pull Request"

- name: Run on success
  if: success()  # 기본값 (생략 가능)
  run: echo "Previous steps succeeded"

- name: Run on failure
  if: failure()
  run: echo "Something failed"

- name: Always run
  if: always()
  run: echo "This always runs"
```

### 컨테이너 실행 환경 구성

```yaml
jobs:
  container-job:
    runs-on: ubuntu-latest
    container:
      image: node:18
      env:
        NODE_ENV: production
    steps:
    - uses: actions/checkout@v3
    - run: node --version
    - run: npm install
    - run: npm test
```

### 서비스 컨테이너 (Service Containers)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
        - 5432:5432
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
        ports:
        - 6379:6379
    steps:
    - uses: actions/checkout@v3
    - run: npm test
      env:
        DATABASE_URL: postgres://postgres:postgres@localhost:5432/test
```

---

## 실습 과제

### 과제 1: 기본 CI 구축 (20 Mins, Easy)

1. 프로젝트 생성 (Node.js 또는 Python)
2. `.github/workflows/ci.yml` 파일 생성, `on: push` 트리거
3. Lint + Test를 실행하는 Job 정의
4. GitHub에 Push 후 Actions 탭에서 성공 확인

### 과제 2: Matrix 빌드 (25 Mins, Medium)

1. Node 버전 매트릭스: `node-version: [16, 18, 20]`
2. OS 매트릭스 확장: `os: [ubuntu-latest, windows-latest]`
3. `strategy` 키워드로 매트릭스 변수 정의
4. Actions 탭에서 N x M 조합 확인 및 결과 비교 분석

### 과제 3: Secrets 활용 (15 Mins, Medium)

1. GitHub Secrets 설정 (Settings > Secrets and variables > Actions)
2. 워크플로우에서 `${{ secrets.API_KEY }}` 문법으로 Secret 호출
3. `env` 키워드로 환경 변수 매핑
4. 실행 로그에서 Secret 값이 `***`로 마스킹되는지 보안 확인

### 과제 4: 복합 워크플로우 (40 Mins, Hard)

1. 3단계 파이프라인: `build`, `test`, `deploy` 3개 별도 Job 정의
2. `needs` 키워드로 순차적 의존성 설정 (build → test → deploy)
3. Deploy Job에 `if` 조건: main 브랜치일 때만 배포 실행
4. `upload-artifact` / `download-artifact`로 빌드 결과물 전달

---

## 참고 자료

### 필수 읽기

- **GitHub Actions Docs** — [docs.github.com/actions](https://docs.github.com/actions)
- **Workflow Syntax** — YAML 파일 작성 구문 레퍼런스
- **Actions Marketplace** — [github.com/marketplace?type=actions](https://github.com/marketplace?type=actions)

### 유용한 Actions & 리소스

#### Essential Actions

- `actions/checkout` — 저장소 코드를 Runner로 내려받는 필수 액션
- `actions/setup-node` — Node.js 환경 설정 및 npm 캐싱
- `actions/cache` — 의존성 파일을 캐싱하여 빌드 속도 단축
- `codecov/codecov-action` — 테스트 커버리지 리포트 자동 업로드

#### Learning Resources

- [GitHub Skills](https://skills.github.com)
- [Awesome Actions](https://github.com/sdras/awesome-actions)
- [Actions Marketplace](https://github.com/marketplace?type=actions)

---

## 핵심 요약

### Workflow Structure
Event(트리거) + Jobs(작업) + Steps(단계)의 계층 구조를 명확히 이해해야 한다.

### Checkout First
대부분의 작업에서 `actions/checkout@v3`는 필수적인 첫 단계이다.

### Security First
API 키나 토큰은 절대 코드에 하드코딩하지 말고 `Secrets`를 사용한다.

### Matrix Builds
다양한 OS와 언어 버전에서 동시에 테스트하여 호환성을 확보한다.

### Best Practices

- 워크플로우는 작고 집중적으로 유지
- 재사용 가능한 Actions 적극 활용
- Fail Fast 전략으로 리소스 절약
- 의존성 캐싱으로 빌드 속도 향상
- 명확한 Job/Step 네이밍 사용

> [!quote] "Automate everything you can." — DevOps Principle

---

## 다음 단계 및 예고

- **CI 워크플로우 구축**: 실제 프로젝트에 적용
- **Marketplace 탐색**: 유용한 검증된 Actions 찾기
- **커스텀 Action 제작**: 재사용 가능한 나만의 Action 모듈화
- **워크플로우 최적화**: 불필요한 단계 줄이고 캐싱 활용

---

> *"Automate everything you can."* — DevOps Principle
