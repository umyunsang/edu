---
tags:
  - AIOSS
  - GitHub-Packages
  - Docker
  - Security
  - Dependabot
status: In Progress
---

# [9주차] GitHub Packages 및 의존성 보안 자동화

## 1. 패키지 배포 및 버전 관리 (GitHub Packages)
코드의 재사용성을 높이기 위해 라이브러리 형태의 결과물을 배포하고 관리합니다.

### 배포 및 업데이트 실습
*   **npm 패키지 배포**: 프로젝트 결과물을 GitHub Packages(GPR)에 게시
*   **버전 업데이트**: 패치(Patch) 업데이트를 통해 버전 정보 갱신 (예: `1.0.0` → `1.0.1`)
*   **Docker 이미지 관리**: 애플리케이션의 컨테이너 이미지를 자동 빌드 후 GitHub Container Registry(GHCR)로 푸시 및 로컬 실행 검증

---

## 2. 의존성 관리 및 보안 자동화 (Shift-Left Security)
개발 초기 단계에서 보안 취약점을 발견하고 조치하는 'Shift-Left' 보안을 실천합니다.

### Dependabot 운영 정책
*   **업데이트 스케줄**: 매주/매월 주기적인 의존성 버전 체크 설정
*   **그룹 업데이트**: 관련 있는 패키지들을 하나의 PR로 묶어 관리 비용 절감
*   **자동 머지 조건**: 패치 버전 등 안전한 업데이트에 대해 자동 병합 규칙 수립

### 취약점 스캔 및 리포팅
*   **npm audit / Snyk**: 외부 라이브러리의 보안 취약점 실시간 검사
*   **자동 리포팅**: 발견된 취약점을 이슈(Issue)로 생성하거나 주기적인 보안 리포트 문서로 자동 업데이트

---

## 3. 결과 확인 (GitHub Link)
*   **Packages**: 배포된 패키지 및 컨테이너 이미지 링크
*   **Security Tab**: Dependabot 알림 및 스캔 결과 화면 내역

> [!warning]
> **보안 주의**: GitHub Packages 배포 시 사용되는 토큰(`GITHUB_TOKEN` 등)의 권한 범위를 최소화하여 보안 사고를 예방해야 합니다.
