
"""
==========================================================
Lab. Strongly Entangling Layer 구현 실습
==========================================================
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================================
# STEP 1. Environment
# ==========================================================
np.random.seed(42)

N_QUBITS = 4
N_LAYERS = 2
EPOCHS = 30

dev = qml.device("default.qubit", wires=N_QUBITS)

print("="*70)
print("STEP 1. Environment")
print("="*70)
print(f"Qubits : {N_QUBITS}")

# ==========================================================
# STEP 2. Dataset
# ==========================================================
normal = np.random.normal(
    loc=[60,0.2,5,12],
    scale=[3,0.05,0.4,1],
    size=(40,4)
)

fault = np.random.normal(
    loc=[80,0.7,7,18],
    scale=[3,0.05,0.5,1],
    size=(40,4)
)

X = np.vstack([normal,fault])
y = np.hstack([np.full(40,-1),np.full(40,1)])

print("="*70)
print("STEP 2. Dataset")
print("="*70)
print("Samples :",len(X))

# ==========================================================
# STEP 3. Normalize
# ==========================================================
scaler = MinMaxScaler(feature_range=(0,np.pi))
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

# ==========================================================
# STEP 4. Layers
# ==========================================================
def rotation_layer(weights):
    for i in range(N_QUBITS):
        qml.Rot(*weights[i], wires=i)

def entanglement_layer():
    for i in range(N_QUBITS):
        qml.CNOT(wires=[i,(i+1)%N_QUBITS])

@qml.qnode(dev)
def circuit(x, weights):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for l in range(N_LAYERS):
        rotation_layer(weights[l])
        entanglement_layer()
    return qml.expval(qml.PauliZ(0))

weights = np.random.normal(0,0.1,(N_LAYERS,N_QUBITS,3), requires_grad=True)

def loss(w):
    preds = np.array([circuit(x,w) for x in X_train])
    return np.mean((preds-y_train)**2)

opt = qml.AdamOptimizer(stepsize=0.05)
history=[]

print("="*70)
print("STEP 5. Training")
print("="*70)

for epoch in range(EPOCHS):
    weights = opt.step(loss, weights)
    l = loss(weights)
    history.append(l)
    print(f"Epoch {epoch+1:02d} Loss : {l:.4f}")

pred = np.array([1 if circuit(x,weights)>=0 else -1 for x in X_test])

acc = accuracy_score(y_test,pred)

print("="*70)
print("STEP 6. Evaluation")
print("="*70)
print("Accuracy :",round(acc,4))
print("Confusion Matrix")
print(confusion_matrix(y_test,pred))

print("="*70)
print("STEP 7. Circuit")
print("="*70)
print(qml.draw(circuit)(X_train[0],weights))

plt.figure(figsize=(6,4))
plt.plot(history,marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

