"""
============================================================
Lab. Optimizer 비교 실험
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit.quantum_info import Statevector
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SPSA

# ==========================================================
# STEP 2. Target 설정
# ==========================================================

TARGET = 0.8

# ==========================================================
# STEP 3. Parameter 생성
# ==========================================================

theta = ParameterVector("θ", 4)

# ==========================================================
# STEP 4. Quantum Circuit 생성
# ==========================================================
qc = QuantumCircuit(2)

qc.ry(theta[0],0)
qc.ry(theta[1],1)

qc.cx(0,1)

qc.rz(theta[2],0)
qc.rz(theta[3],1)


# ==========================================================
# STEP 5. Prediction Function
# ==========================================================

observable = SparsePauliOp.from_list([("ZI",1)])

def predict(params):

    bind = dict(zip(theta,params))

    state = Statevector.from_instruction(
        qc.assign_parameters(bind)
    )

    prediction = np.real(
        state.expectation_value(observable)
    )

    return prediction


# ==========================================================
# STEP 6. Loss Function
# ==========================================================

def loss_function(params):

    prediction = predict(params)

    loss = (prediction-TARGET)**2

    return loss


# ==========================================================
# STEP 7. History 저장 클래스
# ==========================================================
class History:

    def __init__(self):

        self.loss=[]

        self.parameter=[]

        self.prediction=[]


# ==========================================================
# STEP 8. COBYLA Objective Function
# ==========================================================

cobyla_history = History()

def cobyla_objective(params):

    pred = predict(params)

    loss = (pred-TARGET)**2

    cobyla_history.loss.append(loss)

    cobyla_history.prediction.append(pred)

    cobyla_history.parameter.append(params.copy())

    return loss


# ==========================================================
# STEP 9. SPSA Objective Function
# ==========================================================

spsa_history = History()

def spsa_objective(params):

    pred = predict(params)

    loss = (pred-TARGET)**2

    spsa_history.loss.append(loss)

    spsa_history.prediction.append(pred)

    spsa_history.parameter.append(params.copy())

    return loss


# ==========================================================
# STEP 10. Initial Parameter
# ==========================================================

initial_parameter=np.array(
    [0.1,0.1,0.1,0.1]
)

# ==========================================================
# STEP 11. COBYLA 실행
# ==========================================================

cobyla=COBYLA(maxiter=80)

result_cobyla = cobyla.minimize(
    fun=cobyla_objective,
    x0=initial_parameter
)

# ==========================================================
# STEP 12. SPSA 실행
# ==========================================================

np.random.seed(42)

spsa=SPSA(maxiter=80)

result_spsa = spsa.minimize(
    fun=spsa_objective,
    x0=initial_parameter
)


# ==========================================================
# STEP 13. 결과 출력
# ==========================================================

print("="*60)
print("COBYLA")
print("="*60)

print("Parameter")
print(result_cobyla.x)

print()

print("Loss")
print(result_cobyla.fun)

print()

print("="*60)
print("SPSA")
print("="*60)

print("Parameter")
print(result_spsa.x)

print()

print("Loss")
print(result_spsa.fun)



# ==========================================================
# STEP 14. Loss Curve 비교
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    cobyla_history.loss,
    label="COBYLA"
)

plt.plot(
    spsa_history.loss,
    label="SPSA"
)

plt.xlabel("Evaluation")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()



# ==========================================================
# STEP 15. Prediction 비교
# ==========================================================

print()

print("Target")

print(TARGET)

print()

print("COBYLA Prediction")

print(
    predict(result_cobyla.x)
)

print()

print("SPSA Prediction")

print(
    predict(result_spsa.x)
)


# ==========================================================
# STEP 16. Optimizer 비교표
# ==========================================================

print("="*70)

print("{:<12}{:<15}{:<15}".format(
    "Optimizer",
    "Loss",
    "Prediction"
))

print("="*70)

print("{:<12}{:<15.6f}{:<15.6f}".format(
    "COBYLA",
    result_cobyla.fun,
    predict(result_cobyla.x)
))

print("{:<12}{:<15.6f}{:<15.6f}".format(
    "SPSA",
    result_spsa.fun,
    predict(result_spsa.x)
))







