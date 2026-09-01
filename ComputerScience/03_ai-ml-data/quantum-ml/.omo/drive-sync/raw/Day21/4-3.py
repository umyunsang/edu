"""
============================================================
학습 결과 분석 및 Parameter 저장 
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.optimizers import COBYLA

TARGET = 0.8

# ==========================================================
# STEP 3. Parameter 생성
# ==========================================================
theta = ParameterVector("θ", 4)

# ==========================================================
# STEP 4. Quantum Circuit 생성
# ==========================================================
qc = QuantumCircuit(2)

qc.ry(theta[0], 0)
qc.ry(theta[1], 1)

qc.cx(0, 1)

qc.rz(theta[2], 0)
qc.rz(theta[3], 1)

observable = SparsePauliOp.from_list(
    [("ZI", 1)]
)

## Step 6. Prediction Function

def predict(parameter):

    bind = dict(zip(theta, parameter))

    state = Statevector.from_instruction(
        qc.assign_parameters(bind)
    )

    prediction = np.real(
        state.expectation_value(observable)
    )

    return prediction

## Loss Function

loss_history = []
parameter_history = []

def loss_function(parameter):

    prediction = predict(parameter)

    loss = (prediction - TARGET) ** 2

    loss_history.append(loss)
    parameter_history.append(parameter.copy())

    return loss

## Initial Parameter 설정
initial_parameter = np.array(
    [0.1, 0.1, 0.1, 0.1]
)


## Step 9. 학습 전 상태 확인
initial_prediction = predict(initial_parameter)

initial_loss = loss_function(initial_parameter)

print("=" * 60)
print("Before Learning")
print("=" * 60)

print("Initial Parameter")
print(initial_parameter)

print()

print("Initial Prediction")
print(initial_prediction)

print()

print("Initial Loss")
print(initial_loss)

## Step 10. Optimizer 실행

optimizer = COBYLA(maxiter=80)

result = optimizer.minimize(
    fun=loss_function,
    x0=initial_parameter
)

print("=" * 60)
print("Optimizer Result")
print("=" * 60)

print(result)

## Step 11. Optimizer 분석


optimal_parameter = result.x
final_loss = result.fun

evaluation = result.nfev

iteration = getattr(result, "nit", "N/A")
success = getattr(result, "success", "N/A")
message = getattr(result, "message", "N/A")

print("=" * 60)
print("Result Analysis")
print("=" * 60)

print("Optimal Parameter")
print(optimal_parameter)

print()

print("Final Loss")
print(final_loss)

print()

print("Evaluation")
print(evaluation)

print()

print("Iteration")
print(iteration)

print()

print("Success")
print(success)

print()

print("Message")
print(message)

## Step 12. Before / After 비교

before_prediction = predict(initial_parameter)
after_prediction = predict(optimal_parameter)

before_loss = loss_function(initial_parameter)
after_loss = loss_function(optimal_parameter)

final_prediction = predict(
    optimal_parameter
)

final_loss = loss_function(
    optimal_parameter
)


print("=" * 60)
print("After Learning")
print("=" * 60)

print("Optimal Parameter")
print(optimal_parameter)

print()

print("Final Prediction")
print(final_prediction)

print()

print("Final Loss")
print(final_loss)

## Step 13. Parameter 저장

import json

model = {

    "optimizer": "COBYLA",

    "parameter": optimal_parameter.tolist(),

    "loss": float(final_loss),

    "evaluation": int(evaluation),

    "iteration": iteration,

    "success": bool(success),

    "message": str(message)

}

with open(
    "parameter.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model,
        f,
        indent=4,
        ensure_ascii=False
    )


print()

print("parameter.json 저장 완료")


## Step 14. JSON 확인

with open(
    "parameter.json",
    "r",
    encoding="utf-8"
) as f:

    loaded = json.load(f)

print("=" * 60)
print("Saved JSON")
print("=" * 60)

print(json.dumps(
    loaded,
    indent=4,
    ensure_ascii=False
))

## Step 15. Parameter 복원

loaded_parameter = np.array(
    loaded["parameter"]
)

print()

print("Loaded Parameter")

print(loaded_parameter)


## Step 16. Inference
new_inputs = [

    "Data A",

    "Data B",

    "Data C"

]

print("=" * 60)
print("Inference")
print("=" * 60)

for data in new_inputs:

    prediction = predict(
        loaded_parameter
    )

    print(
        f"{data} -> {prediction:.6f}"
    )


## Step 17. Training vs Inference

print("=" * 70)

print("{:<20}{:<20}".format(
    "Training",
    "Inference"
))

print("=" * 70)

print("{:<20}{:<20}".format(
    "Optimizer",
    "사용 안함"
))

print("{:<20}{:<20}".format(
    "Loss 계산",
    "없음"
))

print("{:<20}{:<20}".format(
    "Parameter 변경",
    "고정"
))

print("{:<20}{:<20}".format(
    "반복 수행",
    "한 번 실행"
))


## Step 18. 최종 요약
print("=" * 70)

print("{:<20}{}".format(
    "Optimizer",
    loaded["optimizer"]
))

print("{:<20}{:.6f}".format(
    "Final Loss",
    loaded["loss"]
))

print("{:<20}{}".format(
    "Evaluation",
    loaded["evaluation"]
))

print("{:<20}{}".format(
    "Iteration",
    loaded["iteration"]
))

print("{:<20}{}".format(
    "Success",
    loaded["success"]
))

print("{:<20}{}".format(
    "Parameter",
    loaded_parameter
))
























