---
aliases: []
course: ML-project
created: '2024-08-05'
date: '2024-08-05'
semester: 3-1
source: ''
status: seedling
tags:
- cs/ml
- type/lecture
title: 2x3 그리드로 서브플롯 생성
type: lecture
updated: '2026-05-05'
---





up:: [[ComputerScience/3-1_ML-project/LangChain/ML-project__LangChain__LangChain|ML-project__LangChain__LangChain]]
prerequisites:: [[ComputerScience/3-1_machine-learning/머신러닝 핵심 수학 개념|머신러닝 핵심 수학 개념]], [[ComputerScience/2-1_python/1. 변수와 자료형|1. 변수와 자료형]]
related:: [[ComputerScience/3-1_ML-project/Matplotlib/subplot  예제|subplot  예제]], [[ComputerScience/3-1_ML-project/Matplotlib/imshow 이미지 그래프|imshow 이미지 그래프]], [[ComputerScience/3-1_ML-project/Matplotlib/크기조정 및 그리드|크기조정 및 그리드]]

---
### 서브플롯을 활용한 텍스트 표시 예제

아래는 `matplotlib`의 `subplots` 기능을 사용하여 2x3 그리드로 서브플롯을 생성하고, 각 서브플롯에 텍스트를 표시하는 코드입니다. 이 코드는 서브플롯의 기본 구조를 이해하는 데 도움이 됩니다.

#### **코드 설명 및 주석**

```python
import matplotlib.pyplot as plt

# 2x3 그리드로 서브플롯 생성
fig, ax = plt.subplots(2, 3)

# 각 서브플롯에 텍스트 표시
for i in range(2):
    for j in range(3):
        # (0.3, 0.5) 위치에 (i, j) 튜플을 텍스트로 표시
        # fontsize=11로 폰트 크기를 설정
        ax[i, j].text(0.3, 0.5, str((i, j)), fontsize=11)

# 모든 서브플롯을 화면에 출력
plt.show()
```

- **서브플롯 생성**:
  - `fig, ax = plt.subplots(2, 3)`: 2행 3열의 서브플롯을 생성합니다.
  - `fig`: 전체 그림 객체를 나타냅니다.
  - `ax`: 서브플롯의 축 배열을 나타냅니다. `ax[i, j]`로 각 서브플롯에 접근할 수 있습니다.

- **텍스트 표시**:
  - `for i in range(2):`와 `for j in range(3):` 반복문을 사용하여 각 서브플롯을 순회합니다.
  - `ax[i, j].text(0.3, 0.5, str((i, j)), fontsize=11)`: 각 서브플롯의 `(0.3, 0.5)` 위치에 `(i, j)` 튜플을 텍스트로 표시합니다. `fontsize=11`로 텍스트의 크기를 설정합니다.

- **그래프 출력**:
  - `plt.show()`: 설정된 모든 서브플롯을 화면에 출력합니다.

### 추가 정보

- **텍스트 위치**: `text(x, y, text)`에서 `(x, y)`는 서브플롯의 좌표계에서 텍스트의 위치를 설정합니다. 여기서 `(0.3, 0.5)`는 서브플롯의 좌표계에서 상대적 위치를 의미합니다.
- **폰트 크기**: `fontsize=11`로 폰트 크기를 설정합니다. 원하는 크기로 조정할 수 있습니다.

---
