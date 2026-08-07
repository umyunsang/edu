"""
============================================================
 Lab. PyTorch 기본 모델 실행
============================================================

실습 목표
------------------------------------------------------------
1. PyTorch Tensor 생성
2. nn.Sequential 모델 생성
3. 모델 구조 확인
4. Forward 수행
5. Output Shape 확인
6. Weight / Bias 확인
7. Trainable Parameter 확인

"""

# ============================================================
# STEP 1. PyTorch Import
# ============================================================

print("=" * 70)
print("STEP 1. PyTorch Import")
print("=" * 70)

import torch
import torch.nn as nn

# 항상 동일한 결과를 얻기 위해 Seed 고정
torch.manual_seed(42)

print("PyTorch Version :", torch.__version__)




# ============================================================
# STEP 2. Input Tensor 생성
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2. Input Tensor 생성")
print("=" * 70)

# Feature가 2개인 Sample 1개
x = torch.tensor(
    [[0.2, 0.7]],
    dtype=torch.float32
)

print("Input Tensor")
print(x)

print("\nTensor Shape")
print(x.shape)

print("\nTensor dtype")
print(x.dtype)




# ============================================================
# STEP 3. Neural Network 생성
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3. Neural Network 생성")
print("=" * 70)

model = nn.Sequential(

    nn.Linear(
        in_features=2,
        out_features=4
    ),

    nn.ReLU(),

    nn.Linear(
        in_features=4,
        out_features=1
    )

)

print("Neural Network 생성 완료")




# ============================================================
# STEP 4. Model 구조 확인
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4. Model 구조 확인")
print("=" * 70)

print(model)




# ============================================================
# STEP 5. Forward 수행
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5. Forward 수행")
print("=" * 70)

output = model(x)

print("Forward 완료")




# ============================================================
# STEP 6. Output 확인
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6. Output 확인")
print("=" * 70)

print("Output Tensor")

print(output)




# ============================================================
# STEP 7. Output Shape 확인
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7. Shape 변화 확인")
print("=" * 70)

print("Input Shape")
print(x.shape)

print()

print("Output Shape")
print(output.shape)

print()

print("Shape 변화")
print(f"{x.shape}  --->  {output.shape}")




# ============================================================
# STEP 8. Weight와 Bias 확인
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8. Weight와 Bias 확인")
print("=" * 70)

first_layer = model[0]

print("First Linear Layer")

print(first_layer)

print("\nWeight")

print(first_layer.weight)

print("\nWeight Shape")

print(first_layer.weight.shape)

print("\nBias")

print(first_layer.bias)

print("\nBias Shape")

print(first_layer.bias.shape)




# ============================================================
# STEP 9. Trainable Parameter 확인
# ============================================================

print("\n")
print("=" * 70)
print("STEP 9. Trainable Parameter 확인")
print("=" * 70)

for name, param in model.named_parameters():

    print("-" * 60)

    print("Parameter Name")

    print(name)

    print()

    print("Shape")

    print(param.shape)

    print()

    print("Requires Grad")

    print(param.requires_grad)

    print()

    print("Value")

    print(param)





# ============================================================
# STEP 10. Parameter 개수 확인
# ============================================================

print("\n")
print("=" * 70)
print("STEP 10. Parameter 개수 확인")
print("=" * 70)

total_parameter = sum(

    p.numel()

    for p in model.parameters()

)

print("Total Trainable Parameters")

print(total_parameter)





# ============================================================
# STEP 11. 전체 실행 결과 요약
# ============================================================

print("\n")
print("=" * 70)
print("STEP 11. 실행 결과 요약")
print("=" * 70)

print(f"""
Input Tensor Shape  : {x.shape}

Output Tensor Shape : {output.shape}

Model

Input
   ↓
Linear(2 → 4)
   ↓
ReLU
   ↓
Linear(4 → 1)
   ↓
Output

Forward 수행 완료

Weight 확인 완료

Bias 확인 완료

Trainable Parameter 확인 완료
""")





