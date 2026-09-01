"""
=========================================================
Lab 14

06_input_change.py

입력 변화 실험

=========================================================

실습 목표

1. 입력 Tensor 생성

2. 입력 변화 실험

3. Output 비교

4. Batch Input 비교

5. Feature Map 역할 확인

=========================================================
"""

import torch

from create_qnn import create_qnn
from torch_connector import create_torch_connector


# =========================================================
# STEP 1
# QNN 생성
# =========================================================

print("=" * 70)
print("STEP 1. EstimatorQNN")
print("=" * 70)

qnn = create_qnn()

model = create_torch_connector(qnn)

print("Quantum Layer Ready")

print()

# =========================================================
# STEP 2
# Weight 확인
# =========================================================

print("=" * 70)
print("STEP 2. Current Weight")
print("=" * 70)

print(model.weight)

print()

# =========================================================
# STEP 3
# Input 1
# =========================================================

print("=" * 70)
print("STEP 3. Input 1")
print("=" * 70)

x1 = torch.tensor(

    [[0.10, 0.20]],

    dtype=torch.float32

)

print(x1)

print()

y1 = model(x1)

print("Output")

print(y1)

print()

# =========================================================
# STEP 4
# Input 2
# =========================================================

print("=" * 70)
print("STEP 4. Input 2")
print("=" * 70)

x2 = torch.tensor(

    [[0.60, 0.70]],

    dtype=torch.float32

)

print(x2)

print()

y2 = model(x2)

print("Output")

print(y2)

print()

# =========================================================
# STEP 5
# Input 3
# =========================================================

print("=" * 70)
print("STEP 5. Input 3")
print("=" * 70)

x3 = torch.tensor(

    [[0.95, 0.10]],

    dtype=torch.float32

)

print(x3)

print()

y3 = model(x3)

print("Output")

print(y3)

print()

# =========================================================
# STEP 6
# Output 비교
# =========================================================

print("=" * 70)
print("STEP 6. Output Comparison")
print("=" * 70)

outputs = [

    y1.item(),

    y2.item(),

    y3.item()

]

for idx, value in enumerate(outputs):

    print(f"Sample {idx+1}")

    print(value)

    print()

print("Difference")

print(max(outputs)-min(outputs))

print()

# =========================================================
# STEP 7
# 여러 입력 비교
# =========================================================

print("=" * 70)
print("STEP 7. Multiple Input")
print("=" * 70)

samples = [

    [0.10,0.20],

    [0.20,0.30],

    [0.30,0.40],

    [0.40,0.50],

    [0.50,0.60],

    [0.60,0.70]

]

for sample in samples:

    x = torch.tensor(

        [sample],

        dtype=torch.float32

    )

    y = model(x)

    print("----------------------------")

    print("Input")

    print(x.numpy())

    print()

    print("Output")

    print(y.item())

    print()

# =========================================================
# STEP 8
# Batch Input
# =========================================================

print("=" * 70)
print("STEP 8. Batch Input")
print("=" * 70)

batch = torch.tensor(

    [

        [0.10,0.20],

        [0.20,0.30],

        [0.40,0.60],

        [0.80,0.90]

    ],

    dtype=torch.float32

)

print(batch)

print()

batch_output = model(batch)

print(batch_output)

print()

# =========================================================
# STEP 9
# Mapping
# =========================================================

print("=" * 70)
print("STEP 9. Input → Output")
print("=" * 70)

for i in range(batch.shape[0]):

    print("----------------------------")

    print("Input")

    print(batch[i])

    print()

    print("Output")

    print(batch_output[i])

    print()

