---
aliases: []
course: LLM
created: '2025-01-21'
date: '2025-01-21'
semester: elective
source: ''
status: seedling
tags:
- cs/llm
- cs/nlp
- type/lecture
title: OpenAI 클라이언트 초기화
type: lecture
updated: '2026-05-05'
---

up:: [[LLM & NLP MOC]]

siblings:: [[Audio Generation]], [[ChatGPT API]], [[ChatGPT 모델 이해와 활용]], [[Embedding]], [[Fine-Tuning]], [[Fine-Tuning 실습]], [[openai API 활용]], [[TTS, STT]]
### 주요 특징

1. 텍스트 및 이미지 콘텐츠 검토 지원.
2. 콘텐츠 필터링과 문제 발생 방지를 위한 시정 조치 수행.
3. **Moderation 엔드포인트**는 무료로 제공.

---

### 모델 종류

1. **omni-moderation-latest**
    
    - 최신 모델로, 더 많은 분류 옵션 제공.
    - 멀티 모달 입력(텍스트와 이미지)을 지원.
2. **text-moderation-latest (Legacy)**
    
    - 구형 모델로, 텍스트 입력만 지원.
    - 분류 옵션이 제한적.

---

### Moderation 엔드포인트 활용 실습

```python
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 콘텐츠 검토 요청
response = client.moderations.create(
    model="omni-moderation-latest",  # 최신 Moderation 모델 사용
    input="...text to classify goes here...",  # 검토할 텍스트 입력
)

# 응답 출력
print(response)
```

---

### 참고 링크

- Moderation 가이드: [OpenAI Moderation Documentation](https://platform.openai.com/docs/guides/moderation)
