"""
=========================================================
Lab 14

09_batch_forward.py

Batch 입력 실행

=========================================================

실습 목표

1. Batch Tensor 생성

2. Batch Forward

3. Batch Output 확인

4. Sample별 Output 확인

5. Batch Loss 계산

6. Batch Gradient 계산

=========================================================
"""

import torch
import torch.nn as nn

from create_qnn import create_qnn
from torch_connector import create_torch_connector


# =========================================================
# STEP 1
# Quantum Layer 생성
# =========================================================

print("=" * 70)
print("STEP 1. Quantum Layer")
print("=" * 70)

qnn = create_qnn()

model = create_torch_connector(qnn)

print("Quantum Layer Ready")

print()

# =========================================================
# STEP 2
# Batch Tensor 생성
# =========================================================

print("=" * 70)
print("STEP 2. Batch Input")
print("=" * 70)

batch_x = torch.tensor(

    [

        [0.10,0.20],

        [0.30,0.40],

        [0.50,0.60],

        [0.70,0.80]

    ],

    dtype=torch.float32

)

print(batch_x)

print()

print("Shape")

print(batch_x.shape)

print()

print("Batch Size")

print(batch_x.shape[0])

print()

# =========================================================
# STEP 3
# Batch Forward
# =========================================================

print("=" * 70)
print("STEP 3. Batch Forward")
print("=" * 70)

batch_output = model(batch_x)

print(batch_output)

print()

# =========================================================
# STEP 4
# Output Shape
# =========================================================

print("=" * 70)
print("STEP 4. Output Shape")
print("=" * 70)

print(batch_output.shape)

print()

print(type(batch_output))

print()

print(torch.is_tensor(batch_output))

print()

# =========================================================
# STEP 5
# Sample Mapping
# =========================================================

print("=" * 70)
print("STEP 5. Sample Mapping")
print("=" * 70)

for idx in range(batch_x.shape[0]):

    print("-"*40)

    print(f"Sample {idx+1}")

    print()

    print("Input")

    print(batch_x[idx])

    print()

    print("Output")

    print(batch_output[idx])

    print()

# =========================================================
# STEP 6
# Batch Target
# =========================================================

print("=" * 70)
print("STEP 6. Target")
print("=" * 70)

target = torch.tensor(

    [

        [1.0],

        [0.0],

        [1.0],

        [0.0]

    ],

    dtype=torch.float32

)

print(target)

print()

# =========================================================
# STEP 7
# Batch Loss
# =========================================================

print("=" * 70)
print("STEP 7. Batch Loss")
print("=" * 70)

criterion = nn.MSELoss()

loss = criterion(

    batch_output,

    target

)

print(loss)

print()

# =========================================================
# STEP 8
# Gradient 계산
# =========================================================

print("=" * 70)
print("STEP 8. Backward")
print("=" * 70)

model.zero_grad()

loss.backward()

print("Gradient Complete")

print()

print(model.weight.grad)

print()

# =========================================================
# STEP 9
# Gradient Shape
# =========================================================

print("=" * 70)
print("STEP 9. Gradient Shape")
print("=" * 70)

print(model.weight.grad.shape)

print()

# =========================================================
# STEP 10
# Optimizer
# =========================================================

print("=" * 70)
print("STEP 10. Optimizer")
print("=" * 70)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.01

)

print("Optimizer Ready")

print()

before = model.weight.detach().clone()

optimizer.step()

after = model.weight.detach().clone()

print("Weight Updated")

print()

print("Before")

print(before)

print()

print("After")

print(after)

print()



