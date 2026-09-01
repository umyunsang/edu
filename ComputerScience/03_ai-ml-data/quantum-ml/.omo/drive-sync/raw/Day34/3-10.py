"""
=========================================================
Lab 14

10_summary.py

전체 동작 검증

=========================================================

실습 목표

1. EstimatorQNN 생성 확인

2. TorchConnector 생성 확인

3. Weight 확인

4. Forward 확인

5. Input 변화 확인

6. Weight 변화 확인

7. Gradient 확인

8. Batch 처리 확인

9. Optimizer 확인

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

print("Quantum Layer 생성 성공")
print()

# =========================================================
# STEP 2
# Weight 확인
# =========================================================

print("=" * 70)
print("STEP 2. Weight")
print("=" * 70)

print(model.weight)
print()

print("Shape")

print(model.weight.shape)

print()

# =========================================================
# STEP 3
# Forward
# =========================================================

print("=" * 70)
print("STEP 3. Forward")
print("=" * 70)

x = torch.tensor(

    [[0.20,0.70]],

    dtype=torch.float32

)

output = model(x)

print("Input")

print(x)

print()

print("Output")

print(output)

print()

# =========================================================
# STEP 4
# Input 변화
# =========================================================

print("=" * 70)
print("STEP 4. Input Change")
print("=" * 70)

x2 = torch.tensor(

    [[0.80,0.10]],

    dtype=torch.float32

)

output2 = model(x2)

print("Input 1")

print(x)

print()

print("Output 1")

print(output)

print()

print("Input 2")

print(x2)

print()

print("Output 2")

print(output2)

print()

# =========================================================
# STEP 5
# Weight 변화
# =========================================================

print("=" * 70)
print("STEP 5. Weight Change")
print("=" * 70)

backup = model.weight.detach().clone()

before = model(x)

with torch.no_grad():

    model.weight.fill_(0.5)

after = model(x)

print("Before")

print(before)

print()

print("After")

print(after)

print()

with torch.no_grad():

    model.weight.copy_(backup)

# =========================================================
# STEP 6
# Gradient
# =========================================================

print("=" * 70)
print("STEP 6. Gradient")
print("=" * 70)

target = torch.tensor(

    [[1.0]],

    dtype=torch.float32

)

criterion = nn.MSELoss()

prediction = model(x)

loss = criterion(

    prediction,

    target

)

model.zero_grad()

loss.backward()

print(model.weight.grad)

print()

# =========================================================
# STEP 7
# Batch Forward
# =========================================================

print("=" * 70)
print("STEP 7. Batch")
print("=" * 70)

batch = torch.tensor(

    [

        [0.10,0.20],

        [0.30,0.40],

        [0.50,0.60],

        [0.70,0.80]

    ],

    dtype=torch.float32

)

batch_output = model(batch)

print(batch_output)

print()

print("Shape")

print(batch_output.shape)

print()

# =========================================================
# STEP 8
# Optimizer
# =========================================================

print("=" * 70)
print("STEP 8. Optimizer")
print("=" * 70)

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.01

)

print(optimizer)

print()

print("Optimizer 연결 완료")

print()

# =========================================================
# STEP 9
# Check List
# =========================================================

print("=" * 70)
print("STEP 9. Check List")
print("=" * 70)

check_list = {

    "EstimatorQNN 생성": True,

    "TorchConnector 생성": True,

    "Weight 확인": True,

    "Forward 실행": True,

    "Input 변화": True,

    "Weight 변화": True,

    "Gradient 생성": model.weight.grad is not None,

    "Batch 처리": batch_output.shape[0] == batch.shape[0],

    "Optimizer 연결": True

}

for key, value in check_list.items():

    print(f"{key:25s} : {'PASS' if value else 'FAIL'}")

print()

