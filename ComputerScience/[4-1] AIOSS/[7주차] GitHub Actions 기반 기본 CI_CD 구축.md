---
tags:
  - AIOSS
  - GitHub-Actions
  - CI-CD
  - Automation
status: In Progress
---

# [7주차] GitHub Actions 기반 기본 CI/CD 구축

## 1. CI(지속적 통합) 워크플로우 구성
코드 변경 시마다 품질을 자동으로 검증하는 체계를 구축합니다.

### 자동 검증 항목
*   **Lint**: 정적 분석 도구(ESLint, Flake8 등)를 활용한 코드 스타일 및 잠재 오류 체크
*   **Test**: 단위 테스트(Unit Test) 프레임워크(Jest, Pytest 등)를 통한 로직 검증 자동 실행

---

## 2. 효율적인 테스트 및 보안 설정
다양한 환경에서의 호환성을 확인하고 보안 사고를 예방합니다.

### 주요 기술 적용
*   **Matrix 전략**: 여러 버전(Node.js 18/20 등)과 다양한 운영체제(Ubuntu, Windows, macOS) 조합을 한 번에 테스트
*   **GitHub Secrets**: API 키, 데이터베이스 접속 정보 등 민감 정보를 암호화하여 안전하게 주입

---

## 3. 복합 워크플로우(Complex Workflow) 설계
단계별 의존성을 정의하여 안정적인 배포 파이프라인을 완성합니다.

### 워크플로우 단계 (Jobs)
*   **Build**: 애플리케이션 빌드 및 실행 파일 생성
*   **Test**: 빌드된 아티팩트를 활용하여 테스트 수행 (Build 단계에 의존)
*   **Deploy**: 모든 테스트 통과 시 운영 환경으로 배포 수행
*   **Artifacts**: 단계 간 데이터 전달을 위해 빌드 결과물(Artifacts)의 업로드 및 다운로드 과정 포함

---

## 4. 제출 및 확인 (GitHub Link)
*   **YAML 파일**: `.github/workflows/*.yml` 설정 파일 링크
*   **Actions 실행 내역**: 실제 성공적으로 수행된 워크플로우 실행 결과(Run history)

> [!important]
> **빌드 실패 처리**: 하나라도 테스트가 실패할 경우 배포(Deploy) 단계가 실행되지 않도록 `needs` 구문을 활용해 엄격한 의존성을 설정해야 합니다.
