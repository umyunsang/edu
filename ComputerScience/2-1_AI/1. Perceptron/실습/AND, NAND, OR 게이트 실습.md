---
aliases: []
course: AI
created: '2024-04-10'
date: '2024-04-10'
semester: 2-1
source: ''
status: seedling
tags:
- cs/ai
- cs/dl
- cs/ml
- type/project
title: AND, NAND, OR 게이트 실습
type: project
updated: '2026-05-05'
---





up:: [[ComputerScience/2-1_AI/3. Backpropagation/이론/Backpropagation|Backpropagation]]
prerequisites:: [[ComputerScience/2-1_python/1. 변수와 자료형|1. 변수와 자료형]], [[ComputerScience/2-1_probability-statistics/3.Probability/Probability|Probability]]
related:: [[ComputerScience/2-1_AI/1. Perceptron/이론/AND, NAND, OR 게이트|AND, NAND, OR 게이트]], [[ComputerScience/2-1_AI/5. CNN/실습/ResNet/ResNet|ResNet]], [[ComputerScience/2-1_AI/5. CNN/실습/VGGNet/UMNet|UMNet]]

---

### AND 게이트
```python
import numpy as np  
  
def AND (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
x1 = 1  
x2 = 0  
print(AND(x1, x2))  
  
x1 = 1  
x2 = 1  
print(AND(x1, x2))
```

---

### NAND 게이트

```python
import numpy as np  
  
def NAND (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([-0.5, -0.5])  
  b = 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
x1 = 1  
x2 = 0  
print(NAND(x1, x2))  
  
x1 = 1  
x2 = 1  
print(NAND(x1, x2))
```

---

### OR 게이트

```python
import numpy as np  
  
def OR (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.2  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
x1 = 1  
x2 = 0  
print(OR(x1, x2))  
  
x1 = 1  
x2 = 1  
print(OR(x1, x2))
```

---

### XOR 게이트

```python
import numpy as np  
  
def AND (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
def NAN

D (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([-0.5, -0.5])  
  b = 0.7  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
def OR (x1, x2):  
  x = np.array([x1, x2])  
  w = np.array([0.5, 0.5])  
  b = - 0.2  
  
  s = np.sum(x*w) + b  
  
  if s <= 0:  
    return 0  
  else :  
    return 1  
  
def XOR(x1, x2):  
    s1 = NAND(x1, x2)  
    s2 = OR(x1, x2)  
    s3 = AND(s1, s2)  
    return s3  
  
print(XOR(0,0))  
print(XOR(0,1))  
print(XOR(1,0))  
print(XOR(1,1))
```
