# 샘플 문제 학습 패키지

이 폴더는 AI OSS 실습기반 시험 전에 미리 연습할 수 있도록 만든 샘플 문제 모음입니다.

샘플은 총 3개이며, 각 샘플은 다음 2가지 버전으로 제공됩니다.

- 완성본: 정답 예시와 동작하는 코드 확인용
- minimal 버전: TODO를 직접 채우는 실습용

## 폴더 구성

```text
sample/
├── README.md
├── 공지사항_게시판용.md
├── SAMPLE_PROBLEMS.md
├── SUCCESS_SAMPLE.md
├── sample-solutions/
└── sample-solutions-minimal/
```

## 파일 설명

### 1. SAMPLE_PROBLEMS.md
샘플 1~3의 문제 설명, 3단계 가이드, 성공 기준이 정리된 문서입니다.

### 2. SUCCESS_SAMPLE.md
샘플 1~3의 성공 기준을 한 번에 체크할 수 있는 체크리스트 문서입니다.

### 3. 공지사항_게시판용.md
수업 게시판에 바로 올릴 수 있도록 정리한 공지문 초안입니다.

### 4. sample-solutions/
각 샘플의 완성 예시 코드입니다.

- sample-1-collaboration: 협업 워크플로우 예시
- sample-2-ci-basics: GitHub Actions CI 예시
- sample-3-testing: 테스트와 TDD 예시

### 5. sample-solutions-minimal/
학생이 직접 TODO를 채우는 minimal 실습 버전입니다.

- sample-1-collaboration: PR 템플릿과 greeting TODO
- sample-2-ci-basics: CI YAML TODO
- sample-3-testing: calculator 구현 및 테스트 TODO

## 권장 학습 순서

1. SAMPLE_PROBLEMS.md를 읽고 전체 흐름을 이해합니다.
2. sample-solutions-minimal에서 TODO를 직접 채웁니다.
3. 막히는 경우 sample-solutions의 완성본과 비교합니다.
4. SUCCESS_SAMPLE.md에서 성공 기준을 체크합니다.

## 샘플별 목표

### 샘플 1. 협업 워크플로우 기초
- GitHub Flow 브랜치 전략 이해
- 브랜치 생성, 커밋, PR 작성 연습
- 체크리스트 기반 설명 작성 연습

### 샘플 2. CI 파이프라인 기초
- GitHub Actions 기본 구조 이해
- 트리거, job, step 작성 연습
- 실행 결과와 로그 확인 연습

### 샘플 3. 테스트 기초
- 실패하는 테스트 먼저 작성하는 흐름 경험
- 최소 구현으로 테스트 통과시키기
- TDD의 RED -> GREEN -> REFACTOR 감각 익히기

## 빠른 시작

### 가장 추천하는 방식
1. sample-solutions-minimal/sample-1-collaboration 부터 시작
2. sample-solutions-minimal/sample-2-ci-basics 진행
3. sample-solutions-minimal/sample-3-testing 진행
4. 각 단계 완료 후 SUCCESS_SAMPLE.md 체크

### 완성본을 먼저 보고 싶은 경우
sample-solutions 폴더를 먼저 읽고, 이후 minimal 버전을 다시 직접 구현해보면 됩니다.

## 참고

이 폴더는 본 시험 전체가 아니라 시험 핵심 개념을 축약한 연습 세트입니다. 실제 시험에서는 더 많은 증빙, 더 긴 실행 흐름, 더 엄격한 체크가 필요할 수 있습니다.