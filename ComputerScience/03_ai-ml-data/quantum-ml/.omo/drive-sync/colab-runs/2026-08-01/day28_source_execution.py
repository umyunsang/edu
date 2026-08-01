"""
============================================================
ㄴNeural Network 구조 분석
============================================================

실습 목표
------------------------------------------------------------
1. PyTorch Neural Network 생성
2. Layer 구조 확인
3. Forward Propagation 수행
4. Layer별 Output 분석
5. Weight / Bias 분석
6. Parameter 개수 계산
7. Prediction 생성 과정 이해
8. Classical NN과 QNN 구조 비교

사용 라이브러리
------------------------------------------------------------
PyTorch

작성자 : Quantum Machine Learning 과정
============================================================
"""

# ==========================================================
# STEP 1. Library Import
# ==========================================================

import torch
import torch.nn as nn

print("=" * 60)
print("STEP 1. Library Import")
print("=" * 60)

print("PyTorch Version :", torch.__version__)

# ==========================================================
# STEP 2. Neural Network 생성
# ==========================================================

print("\n" + "=" * 60)
print("STEP 2. Neural Network 생성")
print("=" * 60)

model = nn.Sequential(

    nn.Linear(
        in_features=2,
        out_features=8
    ),

    nn.ReLU(),

    nn.Linear(
        in_features=8,
        out_features=2
    )

)

print(model)

# ==========================================================
# STEP 3. Input Data 생성
# ==========================================================

print("\n" + "=" * 60)
print("STEP 3. Input Data 생성")
print("=" * 60)

x = torch.tensor(
    [
        [1.4, 0.2],
        [4.7, 1.4],
        [1.5, 0.3]
    ],
    dtype=torch.float32
)

print(x)

print("\nInput Shape")
print(x.shape)

# ==========================================================
# STEP 4. Forward Propagation
# ==========================================================

print("\n" + "=" * 60)
print("STEP 4. Forward Propagation")
print("=" * 60)

output = model(x)

print(output)

print("\nOutput Shape")
print(output.shape)

# ==========================================================
# STEP 5. Layer별 Output 확인
# ==========================================================

print("\n" + "=" * 60)
print("STEP 5. Layer별 Output")
print("=" * 60)

linear1 = model[0](x)

relu = model[1](linear1)

linear2 = model[2](relu)

print("----- Linear Layer 1 -----")
print(linear1)
print(linear1.shape)

print()

print("----- ReLU -----")
print(relu)
print(relu.shape)

print()

print("----- Linear Layer 2 -----")
print(linear2)
print(linear2.shape)

# ==========================================================
# STEP 6. Weight 분석
# ==========================================================

print("\n" + "=" * 60)
print("STEP 6. Weight 분석")
print("=" * 60)

print("Layer 1 Weight")

print(model[0].weight)

print()

print("Shape")

print(model[0].weight.shape)

print()

print("Layer 2 Weight")

print(model[2].weight)

print()

print("Shape")

print(model[2].weight.shape)

# ==========================================================
# STEP 7. Bias 분석
# ==========================================================

print("\n" + "=" * 60)
print("STEP 7. Bias 분석")
print("=" * 60)

print("Layer 1 Bias")

print(model[0].bias)

print(model[0].bias.shape)

print()

print("Layer 2 Bias")

print(model[2].bias)

print(model[2].bias.shape)

# ==========================================================
# STEP 8. Parameter 분석
# ==========================================================

print("\n" + "=" * 60)
print("STEP 8. Parameter 분석")
print("=" * 60)

total_parameters = sum(

    p.numel()

    for p in model.parameters()

)

print("Total Parameters")

print(total_parameters)

print()

print("Parameter Detail")

for name, parameter in model.named_parameters():

    print(f"{name:15s} {list(parameter.shape)}")

# ==========================================================
# STEP 9. Shape 변화 분석
# ==========================================================

print("\n" + "=" * 60)
print("STEP 9. Tensor Shape 변화")
print("=" * 60)

print("Input           :", x.shape)

print("Linear Layer 1  :", linear1.shape)

print("ReLU            :", relu.shape)

print("Linear Layer 2  :", linear2.shape)

# ==========================================================
# STEP 10. Softmax
# ==========================================================

print("\n" + "=" * 60)
print("STEP 10. Softmax")
print("=" * 60)

probability = torch.softmax(

    output,

    dim=1

)

print(probability)

# ==========================================================
# STEP 11. Prediction
# ==========================================================

print("\n" + "=" * 60)
print("STEP 11. Prediction")
print("=" * 60)

prediction = torch.argmax(

    output,

    dim=1

)

print(prediction)

# ==========================================================
# STEP 12. 구조 분석
# ==========================================================

print("\n" + "=" * 60)
print("STEP 12. Model Structure Summary")
print("=" * 60)

print("""
Input Feature
    ↓
2

↓

Linear Layer
2 → 8

↓

ReLU

↓

Linear Layer
8 → 2

↓

Prediction
""")

print("COLAB_SOURCE_OK Day28/1-1.py")
print("COLAB_DAY28_SOURCE_EXECUTION_OK")

