---
aliases: []
course: discrete-mathematics
created: '2024-11-30'
date: '2024-11-30'
semester: 2-2
source: ''
status: seedling
tags:
- math/discrete
- type/project
title: Decorator to log execution details
type: project
updated: '2026-05-05'
---


domain:: [[ComputerScience/02_math-theory/수학 이론 인터페이스|수학 이론 인터페이스]]
up:: [[ComputerScience/02_math-theory/discrete-mathematics/과제/과제 번역|과제 번역]]
prerequisites:: [[ComputerScience/01_programming-foundations/coding-basics/중간고사|중간고사]]
related:: [[ComputerScience/02_math-theory/discrete-mathematics/3. 관계와 함수/관계와 함수|관계와 함수]], [[ComputerScience/02_math-theory/discrete-mathematics/4. 그래프/그래프|그래프]], [[ComputerScience/02_math-theory/discrete-mathematics/1. 수학적 모델과 논리/수학적 모델과 논리|수학적 모델과 논리]], [[ComputerScience/02_math-theory/discrete-mathematics/2. 집합 및 집합 연산/집합 및 집합 연산|집합 및 집합 연산]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/Dataframe|Dataframe]], [[ComputerScience/01_programming-foundations/python-programming/6. 함수|6. 함수]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/DAU_BigDataAnalytics_06_(Variable_Selection)|DAU_BigDataAnalytics_06_(Variable_Selection)]], [[ComputerScience/03_ai-ml-data/ml-projects/LangChain/LangChain|LangChain]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/이상 탐지(ASD)를 위한 최적의 Feature Engineering|이상 탐지(ASD)를 위한 최적의 Feature Engineering]], [[ComputerScience/03_ai-ml-data/artificial-intelligence/3. Backpropagation/실습/Vanishing Gradient 해결/활성화 함수 변경|활성화 함수 변경]], [[ComputerScience/01_programming-foundations/data-structures/2. 스택/Stack|Stack]], [[ComputerScience/05_software-engineering/programming-languages/필기/6. 데이터 타입|6. 데이터 타입]], [[ComputerScience/03_ai-ml-data/large-language-models/ChatGPT API/Fine-Tuning 실습|Fine-Tuning 실습]], [[ComputerScience/03_ai-ml-data/big-data-analysis/Converted_MD/02-MapReduce|02-MapReduce]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/2차 컨펌|2차 컨펌]], [[ComputerScience/03_ai-ml-data/ml-projects/Pandas/데이터 분석 및 처리 과정 요약|데이터 분석 및 처리 과정 요약]], [[ComputerScience/01_programming-foundations/python-programming/8. 객체 지향 프로그래밍|8. 객체 지향 프로그래밍]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/1차 컨펌|1차 컨펌]], [[ComputerScience/05_software-engineering/programming-languages/과제/9장 레포트|9장 레포트]], [[ComputerScience/03_ai-ml-data/ml-projects/Sklearn/Regression/Logistic/logistic|logistic]], [[ComputerScience/02_math-theory/mathematical-logic/논리학 개론|논리학 개론]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/신호 특징 분석 결과|신호 특징 분석 결과]], [[ComputerScience/02_math-theory/mathematical-logic/프로젝트/ASD Feature 발굴|ASD Feature 발굴]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제|5장 연습문제]], [[ComputerScience/02_math-theory/mathematical-logic/동아설계도|동아설계도]], [[ComputerScience/05_software-engineering/programming-languages/과제/5장 연습문제 (제출용)|5장 연습문제 (제출용)]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/20_mle|20_mle]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 1 풀이|Pop Quiz 1 풀이]], [[ComputerScience/02_math-theory/probability-statistics/20.mle/MLE|MLE]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/중간고사_정리|중간고사_정리]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/전역변수, 지역변수|전역변수, 지역변수]], [[ComputerScience/02_math-theory/optimization-math/1. Matrix/1. Matrix|1. Matrix]], [[ComputerScience/05_software-engineering/programming-languages/필기/5. 이름, 바인딩, 영역|5. 이름, 바인딩, 영역]], [[ComputerScience/02_math-theory/probability-statistics/8.Poisson/Poisson Distribution|Poisson Distribution]], [[ComputerScience/05_software-engineering/programming-languages/필기/1. 기본사항|1. 기본사항]], [[ComputerScience/05_software-engineering/programming-languages/교재/5장_교재_문제|5장_교재_문제]], [[ComputerScience/02_math-theory/optimization-math/MSC087_HW2_풀이|MSC087_HW2_풀이]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/문법|문법]], [[ComputerScience/02_math-theory/probability-statistics/12.Independent_RVs/Independent RVs|Independent RVs]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/Pop Quiz 풀이/Pop Quiz 2 풀이|Pop Quiz 2 풀이]], [[ComputerScience/02_math-theory/probability-statistics/9.Continuous_RVs/문제 풀이|문제 풀이]], [[ComputerScience/02_math-theory/probability-statistics/19.sampling_bootstrap/Bootstrapping|Bootstrapping]], [[ComputerScience/05_software-engineering/programming-languages/교재/3장_교재_문제|3장_교재_문제]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/기말고사_정리|기말고사_정리]], [[ComputerScience/05_software-engineering/programming-languages/7장-12장 연습문제 종합|7장-12장 연습문제 종합]], [[ComputerScience/05_software-engineering/programming-languages/필기/2. 프로그래밍 언어의 발전사|2. 프로그래밍 언어의 발전사]], [[ComputerScience/01_programming-foundations/coding-basics/1. 컴퓨터에서의 정보의 표현/컴퓨터에서의 정보의 표현(시험문제 안나옴)|컴퓨터에서의 정보의 표현(시험문제 안나옴)]], [[ComputerScience/05_software-engineering/programming-languages/필기/3. 구문론|3. 구문론]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 연습문제 (과제)|3장 연습문제 (과제)]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/변수와 배열|변수와 배열]], [[ComputerScience/05_software-engineering/programming-languages/과제/3장 제출용|3장 제출용]], [[ComputerScience/05_software-engineering/programming-languages/교재/6장_교재_문제|6장_교재_문제]], [[ComputerScience/05_software-engineering/programming-languages/교재/4장_교재_문제|4장_교재_문제]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/순서도 작성|순서도 작성]], [[ComputerScience/05_software-engineering/programming-languages/과제/4장 재귀 하강 파서 연습문제|4장 재귀 하강 파서 연습문제]], [[ComputerScience/05_software-engineering/programming-languages/필기/4. 재귀 하강 파싱|4. 재귀 하강 파싱]], [[ComputerScience/01_programming-foundations/coding-basics/4. 아두이노/아두이노 입출력|아두이노 입출력]], [[ComputerScience/01_programming-foundations/coding-basics/3. 알고리즘과 프로그래밍 언어/연산자|연산자]], [[ComputerScience/05_software-engineering/programming-languages/필기/0. 명령어 집합|0. 명령어 집합]], [[ComputerScience/01_programming-foundations/coding-basics/4. 아두이노/아두이노 실습|아두이노 실습]], [[ComputerScience/06_algorithms-graphics/algorithm-design-analysis/실습과제/트리 만들기|트리 만들기]]

### **작업 1: 데이터 정리 함수 정의**

1. 다음 데이터 정리 단계를 구현하세요:

- **`strip_whitespace(df)`**: 모든 문자열 열에서 앞뒤 공백을 제거합니다.
- **`convert_columns_to_lowercase(df)`**: 모든 문자열 열을 소문자로 변환합니다.
- **`normalize_age_column(df)`**: 나이(age) 열에서 공백을 제거하고 정수로 변환합니다.

2. 이러한 함수들을 함수 합성을 사용하여 단일 파이프라인으로 결합하세요.

---

### **작업 2: 실행 세부 정보를 로그하는 데코레이터 사용**

1. `log_execution`이라는 데코레이터를 작성하세요. 이 데코레이터는 다음 작업을 수행합니다:

- 실행 중인 함수의 이름을 로그에 기록합니다.
- DataFrame의 작업 전후의 형태(shape)를 로그에 기록합니다.
- 함수의 실행 시간을 로그에 기록합니다.

2. 이 데코레이터를 모든 데이터 정리 함수에 적용하세요.

---

### **작업 3: 함수 합성과 데코레이터 결합**

1. 함수 합성과 데코레이터를 결합하여 견고한 데이터 정리 파이프라인을 생성하세요.
2. 각 함수가 실행 세부 정보를 기록하며 순차적으로 실행되도록 해야 합니다.
3. 처리된 DataFrame(정리된 결과)을 출력하세요.

## Exaple Code Structure
```python
import pandas as pd
import time
from functools import reduce

# Decorator to log execution details  
def log_execution(func):
    def wrapper(df, *args, **kwargs):
        # Log start time, shape {df.shape}, and function name {func.__name__}
        # Execute the function
        # Log end time and final shape {end_time - start_time:.4f}
        pass
    return wrapper

# Data cleaning functions  
@log_execution  
def strip_whitespace(df):
    # Your implementation
    pass

@log_execution  
def convert_columns_to_lowercase(df):
    # Your implementation
    pass

# Decorator to log execution details
def log_execution(func):
    def wrapper(df, *args, **kwargs):
        start_time = time.time()
        print(f"Executing {func.__name__}...")
        print(f"Initial DataFrame shape: {df.shape}")
        result = func(df, *args, **kwargs)
        end_time = time.time()
        print(f"Final DataFrame shape: {result.shape}")
        print(f"Execution time: {end_time - start_time:.4f} seconds\n")
        return result
    return wrapper

# Data cleaning functions
@log_execution
def strip_whitespace(df):
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].str.strip()
    return df

@log_execution
def convert_columns_to_lowercase(df):
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].str.lower()
    return df

@log_execution
def normalize_age_column(df):
    if "age" in df.columns:
        df["age"] = df["age"].str.strip().astype(int)
    return df

# Function composition utility
def compose(*functions):
    def composed_function(df):
        return reduce(lambda acc, func: func(acc), functions, df)
    return composed_function

# Define the pipeline
pipeline = compose(strip_whitespace, convert_columns_to_lowercase, normalize_age_column)

# Example DataFrame
data = {
    "name": [" Alice ", "BOB ", "   Carol   "],
    "age": [" 25", "30 ", " 35 "],
    "city": [" New York", "Los Angeles ", " Chicago "],
}

df = pd.DataFrame(data)

# Execute the pipeline
cleaned_df = pipeline(df)

# Print the final DataFrame
print(cleaned_df)

```
