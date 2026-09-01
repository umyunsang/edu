"""
=========================================================
Lab 14

07_weight_change.py

Weight 변화 실험

=========================================================

실습 목표

1. Weight 확인

2. Weight Backup

3. Weight 변경

4. Forward 비교

5. Output 분석

=========================================================
"""

import torch

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
# 입력 고정
# =========================================================

print("=" * 70)
print("STEP 2. Fixed Input")
print("=" * 70)

x = torch.tensor(

    [[0.30, 0.60]],

    dtype=torch.float32

)

print(x)

print()

# =========================================================
# STEP 3
# 현재 Weight 확인
# =========================================================

print("=" * 70)
print("STEP 3. Current Weight")
print("=" * 70)

print(model.weight)

print()

# =========================================================
# STEP 4
# 최초 Output
# =========================================================

print("=" * 70)
print("STEP 4. Original Output")
print("=" * 70)

original_output = model(x)

print(original_output)

print()

# =========================================================
# STEP 5
# Weight Backup
# =========================================================

print("=" * 70)
print("STEP 5. Backup")
print("=" * 70)

backup_weight = model.weight.detach().clone()

print("Backup Complete")

print()

# =========================================================
# STEP 6
# Weight 변경
# =========================================================

print("=" * 70)
print("STEP 6. Weight Change")
print("=" * 70)

with torch.no_grad():

    model.weight.fill_(0.50)

print(model.weight)

print()

# =========================================================
# STEP 7
# 변경 후 Output
# =========================================================

print("=" * 70)
print("STEP 7. New Output")
print("=" * 70)

new_output = model(x)

print(new_output)

print()

# =========================================================
# STEP 8
# Output 비교
# =========================================================

print("=" * 70)
print("STEP 8. Compare")
print("=" * 70)

print("Original")

print(original_output.item())

print()

print("Modified")

print(new_output.item())

print()

difference = abs(

    original_output.item()

    -

    new_output.item()

)

print("Difference")

print(difference)

print()

# =========================================================
# STEP 9
# 다양한 Weight 실험
# =========================================================

print("=" * 70)
print("STEP 9. Various Weight")
print("=" * 70)

weights = [

    0.0,

    0.2,

    0.4,

    0.6,

    0.8,

    1.0

]

for w in weights:

    with torch.no_grad():

        model.weight.fill_(w)

    y = model(x)

    print("----------------------------")

    print("Weight")

    print(w)

    print()

    print("Output")

    print(y.item())

    print()

# =========================================================
# STEP 10
# Weight Restore
# =========================================================

print("=" * 70)
print("STEP 10. Restore")
print("=" * 70)

with torch.no_grad():

    model.weight.copy_(backup_weight)

print(model.weight)

print()

restore_output = model(x)

print("Restored Output")

print(restore_output)

print()

