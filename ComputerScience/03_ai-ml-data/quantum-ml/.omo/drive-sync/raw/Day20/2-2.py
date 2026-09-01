"""
============================================================
수동 Learning Cycle 구현
============================================================
"""


import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


target = 0.80

print("=" * 60)
print("Target Probability")
print("=" * 60)

print(target)


theta = 0.20

print("\nInitial Parameter")
print(theta)

def quantum_prediction(theta):

    qc = QuantumCircuit(1)

    qc.ry(theta,0)

    state = Statevector.from_instruction(qc)

    probabilities = state.probabilities()

    prediction = probabilities[1]

    return prediction

def loss_function(prediction,target):

    return (prediction-target)**2

prediction = quantum_prediction(theta)

loss = loss_function(prediction,target)

print("\nPrediction :",prediction)
print("Loss :",loss)

theta = 0.60

prediction = quantum_prediction(theta)

loss = loss_function(prediction,target)

print("\nParameter :",theta)
print("Prediction :",prediction)
print("Loss :",loss)


theta = 1.20

prediction = quantum_prediction(theta)

loss = loss_function(prediction,target)

print("\nParameter :",theta)
print("Prediction :",prediction)
print("Loss :",loss)

theta = 2.00

prediction = quantum_prediction(theta)

loss = loss_function(prediction,target)

print("\nParameter :",theta)
print("Prediction :",prediction)
print("Loss :",loss)


theta = 0.20

print("="*70)
print("Manual Learning Cycle")
print("="*70)

for step in range(8):

    prediction = quantum_prediction(theta)

    loss = loss_function(prediction,target)

    print(f"Step {step+1}")

    print(f"Theta      : {theta:.2f}")

    print(f"Prediction : {prediction:.4f}")

    print(f"Loss       : {loss:.6f}")

    print("-"*40)

    theta += 0.30


best_theta = None
best_loss = 999

for theta in np.linspace(0, np.pi, 50):

    prediction = quantum_prediction(theta)

    loss = loss_function(prediction,target)

    if loss < best_loss:

        best_loss = loss

        best_theta = theta

print("="*60)
print("Best Parameter")
print("="*60)

print("Theta :",best_theta)

print("Loss :",best_loss)


prediction = quantum_prediction(best_theta)

print("="*60)
print("Final Result")
print("="*60)

print("Target     :",target)

print("Prediction :",prediction)

print("Loss       :",best_loss)






