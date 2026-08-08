"""
=========================================================
Lab 14

05_forward.py (Part 1)

Forward 실행

=========================================================

실습 목표

1. EstimatorQNN 생성

2. TorchConnector 생성

3. Input Tensor 생성

4. Forward 수행

5. Output 확인

6. Output Shape 확인

7. Output Type 확인

=========================================================
"""

import random
import numpy as np
import torch

from create_qnn import create_qnn
from torch_connector import create_torch_connector


# =========================================================
# STEP 1. Random Seed
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("STEP 1. Random Seed")
print("=" * 70)

print("Seed :", SEED)
print()

# =========================================================
# STEP 2. EstimatorQNN 생성
# =========================================================

print("=" * 70)
print("STEP 2. EstimatorQNN")
print("=" * 70)

qnn = create_qnn()

print("EstimatorQNN 생성 완료")
print()

print("Input 수")

print(qnn.num_inputs)

print()

print("Weight 수")

print(qnn.num_weights)

print()

print("Output Shape")

print(qnn.output_shape)

print()

# =========================================================
# STEP 3. TorchConnector 생성
# =========================================================

print("=" * 70)
print("STEP 3. TorchConnector")
print("=" * 70)

model = create_torch_connector(qnn)

print(model)

print()

# =========================================================
# STEP 4. Input Tensor 생성
# =========================================================

print("=" * 70)
print("STEP 4. Input Tensor")
print("=" * 70)

x = torch.tensor(
    [[0.20, 0.70]],
    dtype=torch.float32
)

print("Input")

print(x)

print()

print("Shape")

print(x.shape)

print()

print("dtype")

print(x.dtype)

print()

# =========================================================
# STEP 5. Forward 실행
# =========================================================

print("=" * 70)
print("STEP 5. Forward")
print("=" * 70)

output = model(x)

print("Forward 완료")

print()

# =========================================================
# STEP 6. Output 확인
# =========================================================

print("=" * 70)
print("STEP 6. Output")
print("=" * 70)

print(output)

print()

# =========================================================
# STEP 7. Output 분석
# =========================================================

print("=" * 70)
print("STEP 7. Output Analysis")
print("=" * 70)

print("Type")

print(type(output))

print()

print("Shape")

print(output.shape)

print()

print("dtype")

print(output.dtype)

print()

print("Tensor 여부")

print(torch.is_tensor(output))

print()

print("Device")

print(output.device)

print()

print("requires_grad")

print(output.requires_grad)

print()

# =========================================================
# STEP 8. Input / Output 비교
# =========================================================

print("=" * 70)
print("STEP 8. Input -> Output")
print("=" * 70)

print("Input")

print(x)

print()

print("Output")

print(output)

print()

print("Input Shape")

print(x.shape)

print()

print("Output Shape")

print(output.shape)

print()

# =========================================================
# STEP 9. Prediction Value
# =========================================================

print("=" * 70)
print("STEP 9. Prediction")
print("=" * 70)

print("Expectation Value")

print(output.item())

print()

if output.item() >= 0:

    print("Positive Expectation")

else:

    print("Negative Expectation")

print()



"""
=========================================================
Lab 14

05_forward.py (Part 2)

Forward 실행 (Input 변화 및 Batch 처리)

=========================================================

실습 목표

1. 입력 변화에 따른 Output 확인

2. 여러 입력 비교

3. Batch Input 생성

4. Batch Forward 수행

5. Batch Output 분석

=========================================================
"""

import torch

# =========================================================
# STEP 11. 입력 변화 실험
# =========================================================

print("=" * 70)
print("STEP 11. Input Change")
print("=" * 70)

x1 = torch.tensor(
    [[0.20, 0.70]],
    dtype=torch.float32
)

x2 = torch.tensor(
    [[0.80, 0.10]],
    dtype=torch.float32
)

print("Input 1")

print(x1)

print()

print("Input 2")

print(x2)

print()

output1 = model(x1)

output2 = model(x2)

print("Output 1")

print(output1)

print()

print("Output 2")

print(output2)

print()

# =========================================================
# STEP 12. Output 비교
# =========================================================

print("=" * 70)
print("STEP 12. Output Comparison")
print("=" * 70)

print("Expectation 1")

print(output1.item())

print()

print("Expectation 2")

print(output2.item())

print()

difference = abs(
    output1.item() - output2.item()
)

print("Difference")

print(difference)

print()

if difference > 1e-6:

    print("입력이 변경되면 Output도 변경됩니다.")

else:

    print("Output 변화가 거의 없습니다.")

print()

# =========================================================
# STEP 13. 여러 입력 Forward
# =========================================================

print("=" * 70)
print("STEP 13. Multiple Forward")
print("=" * 70)

samples = [

    [0.10, 0.20],

    [0.20, 0.40],

    [0.30, 0.60],

    [0.50, 0.70]

]

for idx, sample in enumerate(samples):

    x = torch.tensor(

        [sample],

        dtype=torch.float32

    )

    y = model(x)

    print("-" * 40)

    print(f"Sample {idx + 1}")

    print()

    print("Input")

    print(x)

    print()

    print("Output")

    print(y)

    print()

# =========================================================
# STEP 14. Batch Input 생성
# =========================================================

print("=" * 70)
print("STEP 14. Batch Input")
print("=" * 70)

batch = torch.tensor(

    [

        [0.10, 0.20],

        [0.30, 0.50],

        [0.60, 0.70],

        [0.90, 0.10]

    ],

    dtype=torch.float32

)

print(batch)

print()

print("Batch Shape")

print(batch.shape)

print()

# =========================================================
# STEP 15. Batch Forward
# =========================================================

print("=" * 70)
print("STEP 15. Batch Forward")
print("=" * 70)

batch_output = model(batch)

print(batch_output)

print()

# =========================================================
# STEP 16. Batch Output 분석
# =========================================================

print("=" * 70)
print("STEP 16. Batch Output")
print("=" * 70)

print("Output Shape")

print(batch_output.shape)

print()

print("Output Type")

print(type(batch_output))

print()

print("Tensor 여부")

print(torch.is_tensor(batch_output))

print()

print("Batch Size")

print(batch.shape[0])

print()

print("Output Count")

print(batch_output.shape[0])

print()

# =========================================================
# STEP 17. Input / Output Mapping
# =========================================================

print("=" * 70)
print("STEP 17. Mapping")
print("=" * 70)

for i in range(batch.shape[0]):

    print("-" * 40)

    print(f"Sample {i + 1}")

    print()

    print("Input")

    print(batch[i])

    print()

    print("Output")

    print(batch_output[i])

    print()

