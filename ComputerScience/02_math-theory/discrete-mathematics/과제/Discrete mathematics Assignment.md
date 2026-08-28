---
aliases: []
course: discrete-mathematics
created: '2024-11-30'
date: '2024-11-30'
kg_graph_size: 62
kg_layer_label: L4 source
kg_level: 4
kg_role: source-note
semester: 2-2
source: ''
status: draft
tags:
- discrete
- project
title: Decorator to log execution details
type: project
updated: '2026-05-05'
---

### **작업 1: 데이터 정리 함수 정의**

1. 다음 데이터 정리 단계를 구현하세요:

- **`strip_whitespace(df)`**: 모든 문자열 열에서 앞뒤 공백을 제거합니다.
- **`convert_columns_to_lowercase(df)`**: 모든 문자열 열을 소문자로 변환합니다.
- **`normalize_age_column(df)`**: 나이(age) 열에서 공백을 제거하고 정수로 변환합니다.

1. 이러한 함수들을 함수 합성을 사용하여 단일 파이프라인으로 결합하세요.

---

### **작업 2: 실행 세부 정보를 로그하는 데코레이터 사용**

1. `log_execution`이라는 데코레이터를 작성하세요. 이 데코레이터는 다음 작업을 수행합니다:

- 실행 중인 함수의 이름을 로그에 기록합니다.
- DataFrame의 작업 전후의 형태(shape)를 로그에 기록합니다.
- 함수의 실행 시간을 로그에 기록합니다.

1. 이 데코레이터를 모든 데이터 정리 함수에 적용하세요.

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
