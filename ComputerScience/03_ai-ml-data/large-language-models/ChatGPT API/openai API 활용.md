---
aliases: []
course: large-language-models
created: '2025-01-21'
date: '2025-01-21'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: elective
source: ''
status: seedling
tags:
- cs/llm
- cs/nlp
- type/lecture
title: 데이터프레임 로드
type: lecture
updated: '2026-05-05'
---

domain:: [[ComputerScience/03_ai-ml-data/AI ML 데이터 인터페이스|AI ML 데이터 인터페이스]]
stage:: [[ComputerScience/00_graph-interfaces/stages/3단계 AI 데이터 심화 인터페이스|3단계 AI 데이터 심화 인터페이스]]
module:: [[ComputerScience/00_graph-interfaces/courses/LLM 인터페이스|LLM 인터페이스]]
bridge:: [[ComputerScience/00_graph-interfaces/bridges/AI 구현 브리지|AI 구현 브리지]], [[ComputerScience/00_graph-interfaces/bridges/데이터 서비스 브리지|데이터 서비스 브리지]]
up:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT API|ChatGPT API]]
prerequisites:: [[ComputerScience/03_ai-ml-data/machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/03_ai-ml-data/neural-networks/md/신경망_핵심이론_시험정리|신경망_핵심이론_시험정리]]
related:: [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Embedding|Embedding]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Audio Generation|Audio Generation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/TTS, STT|TTS, STT]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Moderation|Moderation]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/추론 모델|추론 모델]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning|Fine-Tuning]], [[ComputerScience/03_ai-ml-data/large-language-models/LLM 이해/LLM 모델 이해|LLM 모델 이해]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/RAG|RAG]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Llama Index|Llama Index]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/large-language-models/검색 증강 생성 RAG/Vector store|Vector store]], [[ComputerScience/03_ai-ml-data/large-language-models/환경 구성|환경 구성]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/실력과제|실력과제]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/03_ai-ml-data/ml-projects/Numpy/Numpy 기초|Numpy 기초]], [[ComputerScience/03_ai-ml-data/machine-learning/중간/대비문제|대비문제]], [[ComputerScience/01_programming-foundations/python-programming/4. 조건문|4. 조건문]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/03_ai-ml-data/ml-projects/Python 기초/성적입력 프로그램|성적입력 프로그램]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/AI 메뉴 추천|AI 메뉴 추천]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/2. MLP(Multi Layer Perceptron)/실습/MLP (Multi Layer Perceptron)|MLP (Multi Layer Perceptron)]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/LSM, GDM 선형 회귀모델|LSM, GDM 선형 회귀모델]], [[ComputerScience/03_ai-ml-data/ai-system-design/주문 및 결제 AI 시스템 개발|주문 및 결제 AI 시스템 개발]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/1. Perceptron/실습/AND, NAND, OR 게이트 실습|AND, NAND, OR 게이트 실습]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/KNNR/KNN 회귀|KNN 회귀]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/4. Optimization/실습/LearningRateControl|LearningRateControl]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/중간시험/CIFAR10 MLP 이미지 분류 중간 실습시험|CIFAR10 MLP 이미지 분류 중간 실습시험]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Dropout|Dropout]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Data Augmentation|Data Augmentation]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Classifier/KNNC/KNN 분류|KNN 분류]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/CIFAR10/CIFAR10|CIFAR10]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Multiple_Linear_Regression|Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/우버데이터_Multiple_Linear_Regression|우버데이터_Multiple_Linear_Regression]], [[ComputerScience/03_ai-ml-data/neural-networks/md/학습기술 이론|학습기술 이론]], [[ComputerScience/03_ai-ml-data/neural-networks/AIE309_HW1_풀이|AIE309_HW1_풀이]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/08-PandaDataframes|08-PandaDataframes]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGG|VGG]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGDense|VGGDense]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Series|Series]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Overfitting 해결/Batch Normalization|Batch Normalization]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/Civitai LoRA 실내공간 스타일 생성 과제|Civitai LoRA 실내공간 스타일 생성 과제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/소감문 작성|소감문 작성]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGCA|VGGCA]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/SDG/SGDClassifier|SGDClassifier]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 이미지 스타일 변환 과제|생성형 AI 이미지 스타일 변환 과제]], [[ComputerScience/03_ai-ml-data/generative-ai-fine-tuning/생성형 AI 파인튜닝 프로젝트 주제|생성형 AI 파인튜닝 프로젝트 주제]], [[ComputerScience/03_ai-ml-data/neural-networks/md/Ch5. 오차역전파법 수학적 증명|Ch5. 오차역전파법 수학적 증명]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/SVM|SVM]], [[ComputerScience/03_ai-ml-data/neural-networks/md/2장 퍼셉트론 상세 정리|2장 퍼셉트론 상세 정리]], [[ComputerScience/03_ai-ml-data/neural-networks/md/4장 신경망 학습과 경사 하강법|4장 신경망 학습과 경사 하강법]], [[ComputerScience/03_ai-ml-data/machine-learning/SVM/QP SVM|QP SVM]], [[ComputerScience/03_ai-ml-data/machine-learning/Linear_Regression/Linear Regression|Linear Regression]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/VGGskip|VGGskip]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/기말시험/시험 예상 문제|시험 예상 문제]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/VGGNet/UMNet|UMNet]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/pooling|pooling]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Multiple/다중 선형 회귀|다중 선형 회귀]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/05-PySpark|05-PySpark]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/CNN 모듈|CNN 모듈]], [[ComputerScience/03_ai-ml-data/ai-system-design/스마트 오더 플랫폼 B2B 어드민 기능 제안|스마트 오더 플랫폼 B2B 어드민 기능 제안]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/LeNet/LR control|LR control]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API|BDA_Hands_on_Numerical_and_Textual_Data_Analytics_using_Youtube_API]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/시스템 구성도|시스템 구성도]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Pivot|Pivot]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/06-SparkDataFrames|06-SparkDataFrames]], [[ComputerScience/03_ai-ml-data/big-data-analysis/md/연습문제_풀이|연습문제_풀이]], [[ComputerScience/03_ai-ml-data/ai-system-design/아키텍쳐/주요 데이터 흐름/주문 생성|주문 생성]]

kg_parent:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_profile:: [[ComputerScience/00_graph-interfaces/archive-kg/courses/LLM 지식그래프|LLM]]
kg_evidence:: [[ComputerScience/00_graph-interfaces/archive-kg/evidence/LLM 근거 인덱스|LLM 근거 인덱스]]
kg_concepts:: [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ai|ai]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/llm|llm]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/api|api]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/rag|rag]], [[ComputerScience/00_graph-interfaces/archive-kg/concepts/large-language-models/ChatGPT 모델 이해와 활용|ChatGPT 모델 이해와 활용]]
kg_query_mode:: [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Fact Retrieval|Fact Retrieval]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Complex Reasoning|Complex Reasoning]], [[ComputerScience/00_graph-interfaces/archive-kg/query-modes/Contextual Summarize|Contextual Summarize]]

---
#### openai API  활용
    
>[!<문제1>]
>post 요청 endpoint url [https://api.openai.com/v1/embeddings](https://api.openai.com/v1/embeddings) 임베딩 요청을 합니다 
>- 로컬의 엑셀파일(naver_news.xlsx)에서 0번 행의 제목 셀을 임베딩 요청하고 응답 임베딩 결과를 출력하는 코드를 구현하시오
```python
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

## openai apikey 입력
OPENAI_API_KEY = "YOUR_API_KEY"

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ").replace("\u00A0", " ")  # Non-breaking Space 제거
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + OPENAI_API_KEY
    }
    json_data = {
        'input': text,
        'model': model,
    }
    try:
        response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=json_data)
        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            return embedding
        else:
            print(f"임베딩 가져오기 오류: HTTP {response.status_code} - {response.text}")
            return None
    except Exception as e:
        # 다른 예외 처리
        print(f"임베딩 가져오기 오류: {e}")
        return None

# 데이터프레임 로드
df = pd.read_excel('./data/naver_news.xlsx')

# 첫 번째 행의 제목 추출
sample_text = df.loc[0, '제목']
print("Sample Text:", sample_text)

# 임베딩 추출
embedding = get_embedding(sample_text)
print("Embedding:", embedding)
```

---
>[!<문제2>]
>openai  SDK를 사용해서 말하는 고양이에 대한 짧은 이야기를  창작한 결과를 출력하는 코드를 구현하시오
```python
import openai
api_key = OPENAI_API_KEY

# OpenAI API 클라이언트 초기화
openai.api_key = api_key

response = openai.chat.completions.create(
     model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "당신은 창작 작가 입니다."},
        {"role": "user", "content": "말하는 고양이에 대한 짧은 이야기를 작성하시오"}
    ],
    temperature=0.8,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=1.0,
    presence_penalty=1.0
)

print("Assistant Response:")
print(response.choices[0].message)
```

>[!<문제3>]
>openai  SDK를 사용해서 로컬에서 nomuhun.txt파일을 읽어서 3문장으로 요약결과를 출력하는 코드를 구현하시오
```python
with open('./data/nomuhun.txt', 'r', encoding='utf-8') as file:
    file_content = file.read()

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Generate an executive summary focusing on key points and actionable insights."},
        {"role": "user", "content": f"문서를 3문장으로 요약해주세요: {file_content}"}
    ],
    temperature=0.2,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
) 

print("Original Text:\n", file_content)
print("Summary Text:\n", response.choices[0].message.content)
```

>[!<문제4>]
>사용자로부터 나이 , 성별, 선호 장르, 한국 영화 또는 외국영화, 감정상태를 입력받아서 prompt로 생성한 후 ChatGPT 영화추천을 요청하고 결과를 출력하는 코드를 구현하시오

```python
def get_movie_recommendation(age, gender, genre, country, mood) :
    prompt = f"""
    당신은 영화 추천 전문가입니다. 아래 사용자의 정보를 기반으로 개인 맞춤형 영화 5편을 추천해주세요
    각 영화의 제목, 개봉 연도, 감독, 주연배우, 간단한 줄거리를 포함해주세요
    - 나이 : {age}
    - 성별 : {gender}
    - 선호 장르 : {genre}
    - 한국영화 또는 외국영화 : {country}
    - 감정 상태 : {mood}
    
    개인 맞춤형 영화 추천해주세요"""

    try :
        response = openai.chat.completions.create(
            model ="gpt-4o-mini",
            messages =[ { "role": "system", "content" : "당신은 영화 추천 전문가입니다"},
                        { "role" : "user", "content" : prompt}  ]
        )
        recommendations = response.choices[0].message.content
        return recommendations
    except  openai.error.OpenAIError as e :
        return f"오류가 발생했습니다 : {str(e)}"       
```
```bash
age = input("나이를 입력하세요")
gender = input("성별 입력하세요 (예: 남성, 여성)")
genre = input("선호하는 장르를 입력하세요(예 : 액션, 코미디, 로맨스)")
country = input("어느 나라 영화를 선호하시나요? (예: 한국, 미국, 중국, 일본)")
mood = input("현재 감정상태를 입력하세요(예: 행복, 우울, ")
recommendations = get_movie_recommendation(age, gender, genre, country, mood)
print("**** 개인 맞춤 영화 추천 리스트 ****")
print(recommendations)
```
