"""
샘플 2 CI 파이프라인 상세 설명

Step 1: 트리거 설정
- on: push, pull_request → main 브랜치에서만 실행
- 이는 main에 변경이 생기면 자동으로 CI 시작

Step 2: Job 및 Steps
- runs-on: ubuntu-latest → 리눅스 환경에서 실행
- 의존성 설치 (ruff, pytest)
- Lint 체크 (ruff)
- 테스트 실행 (pytest)

Step 3: 결과 확인
- Actions 탭에서 워크플로우 실행 기록 확인
- 성공하면 PR에 초록 체크(✓) 표시
- 실패하면 빨강 X(✗) 표시 + 에러 로그

성공 시나리오:
✓ Checkout code
✓ Set up Python 3.10
✓ Install dependencies
✓ Run Lint (ruff check)
✓ Run Tests (pytest)

실패 시나리오 (수정하고 다시 시도):
✗ Run Lint (ruff check) → 코드 스타일 오류 수정 필요
- 에러 메시지 읽기
- 코드 수정
- 다시 커밋/푸시
"""
