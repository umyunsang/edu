# 샘플 1: 협업 워크플로우 기초

## 📌 문제 설명
GitHub Flow 기반 간단한 협업 작업을 경험합니다.

## 🎯 요구사항 (3단계)

### Step 1: GitHub Flow 전략 선언
브랜치 전략을 GitHub Flow로 선택하고, 선택 이유를 1문장으로 작성하세요.

**예시:**
```
GitHub Flow를 선택합니다. 이유: main 중심 개발로 빠른 배포가 가능하고 릴리스 브랜치 관리 없이 간단합니다.
```

### Step 2: 브랜치 생성 및 작업
```bash
# 기본 main에서 브랜치 생성
git checkout -b feature/add-greeting

# 파일 수정 (app/greeting.py)
echo 'def get_greeting(name: str) -> str:
    return f"Hello, {name}!"' > app/greeting.py

# 커밋
git add app/greeting.py
git commit -m "feat: add greeting function"
```

### Step 3: PR 생성
GitHub에서 PR을 생성할 때 다음을 포함하세요:

```markdown
## 변경 사항
- greeting 함수 추가
- 사용자 이름 기반 인사말 반환

## 테스트
- 로컬 테스트 완료
- 함수 호출 확인: get_greeting("student") → "Hello, student!"

## 체크리스트
- [x] 코드가 작동함
- [x] 테스트 완료
- [x] 문서 확인
```

## ✅ 성공 기준
- PR 링크 1개 제출
- GitHub Flow 선택 이유 1문장
- 체크리스트 마크업 포함

## 💾 풀이 보기
[-> 완전한 예제 코드](./app/greeting.py)

