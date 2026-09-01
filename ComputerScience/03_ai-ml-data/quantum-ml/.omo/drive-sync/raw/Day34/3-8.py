"""
=========================================================
Lab 14

08_gradient.py

Gradient 계산

=========================================================

실습 목표

1. Forward 수행

2. Loss 생성

3. Backward 수행

4. Gradient 확인

5. Gradient Shape 확인

6. Optimizer 연결 확인

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
# Input 생성
# =========================================================

print("=" * 70)
print("STEP 2. Input")
print("=" * 70)

x = torch.tensor(

    [[0.20,0.70]],

    dtype=torch.float32

)

target = torch.tensor(

    [[1.0]],

    dtype=torch.float32

)

print("Input")

print(x)

print()

print("Target")

print(target)

print()

# =========================================================
# STEP 3
# Forward
# =========================================================

print("=" * 70)
print("STEP 3. Forward")
print("=" * 70)

prediction = model(x)

print(prediction)

print()

# =========================================================
# STEP 4
# Loss 생성
# =========================================================

print("=" * 70)
print("STEP 4. Loss")
print("=" * 70)

criterion = nn.MSELoss()

loss = criterion(

    prediction,

    target

)

print(loss)

print()

# =========================================================
# STEP 5
# Gradient 초기화
# =========================================================

print("=" * 70)
print("STEP 5. Zero Gradient")
print("=" * 70)

model.zero_grad()

print("Gradient Reset")

print()

# =========================================================
# STEP 6
# Backward
# =========================================================

print("=" * 70)
print("STEP 6. Backward")
print("=" * 70)

loss.backward()

print("Backward Complete")

print()

# =========================================================
# STEP 7
# Gradient 확인
# =========================================================

print("=" * 70)
print("STEP 7. Gradient")
print("=" * 70)

print(model.weight.grad)

print()

# =========================================================
# STEP 8
# Gradient Shape
# =========================================================

print("=" * 70)
print("STEP 8. Gradient Shape")
print("=" * 70)

print(model.weight.grad.shape)

print()

# =========================================================
# STEP 9
# Gradient Statistics
# =========================================================

print("=" * 70)
print("STEP 9. Gradient Statistics")
print("=" * 70)

grad = model.weight.grad

print("Mean")

print(grad.mean())

print()

print("Std")

print(grad.std())

print()

print("Max")

print(grad.max())

print()

print("Min")

print(grad.min())

print()

# =========================================================
# STEP 10
# Parameter + Gradient
# =========================================================

print("=" * 70)
print("STEP 10. Parameter & Gradient")
print("=" * 70)

for name, param in model.named_parameters():

    print(name)

    print()

    print("Parameter")

    print(param)

    print()

    print("Gradient")

    print(param.grad)

    print()

# =========================================================
# STEP 11
# Optimizer
# =========================================================

print("=" * 70)
print("STEP 11. Optimizer")
print("=" * 70)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.01

)

print(optimizer)

print()

print("Optimizer Ready")

print()

# =========================================================
# STEP 12
# Optimizer Step
# =========================================================

print("=" * 70)
print("STEP 12. Optimizer Step")
print("=" * 70)

before = model.weight.detach().clone()

optimizer.step()

after = model.weight.detach().clone()

print("Before")

print(before)

print()

print("After")

print(after)

print()

print("Difference")

print(after-before)

print()

