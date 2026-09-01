"""
===========================================
Basis Encoding & Angle Encoding
===========================================
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qiskit_aer import AerSimulator
from qiskit import transpile


print("="*60)
print("STEP 1. Customer Dataset")
print("="*60)

customer = pd.DataFrame({

    "Gender":[0,1,0,1],

    "Member":[1,1,0,0],

    "Age":[25,35,45,28],

    "Income":[3000,5000,7000,4000],

    "Purchase":[2,5,7,3],

    "Label":[0,1,1,0]

})

print(customer)

print("="*60)
print("STEP 2. Feature Split")
print("="*60)

binary_features = customer[

    ["Gender","Member"]

].values

continuous_features = customer[

    ["Age","Income","Purchase"]

].values

labels = customer["Label"].values

print(binary_features)

print(continuous_features)


print("="*60)
print("STEP 3. Scaling")
print("="*60)

scaler = MinMaxScaler(

    feature_range=(0,np.pi)

)

continuous_scaled = scaler.fit_transform(

    continuous_features

)

print(continuous_scaled)

def basis_encoding(binary):

    qc = QuantumCircuit(

        len(binary)

    )

    for i,value in enumerate(binary):

        if value==1:

            qc.x(i)

    return qc

print("="*60)
print("STEP 4. Basis Encoding")
print("="*60)

basis = basis_encoding(

    binary_features[1]

)

print(basis)

print(

    basis.draw("text")

)

def angle_encoding(features):

    qc = QuantumCircuit(

        len(features)

    )

    for i,value in enumerate(features):

        qc.ry(

            value,

            i

        )

    return qc

print("="*60)
print("STEP 5. Angle Encoding")
print("="*60)

angle = angle_encoding(

    continuous_scaled[1]

)

print(

    angle.draw("text")

)

def hybrid_encoding(

        binary,

        continuous

):

    total = len(binary)+len(continuous)

    qc = QuantumCircuit(total)

    #
    # Basis Encoding
    #

    for i,v in enumerate(binary):

        if v==1:

            qc.x(i)

    #
    # Angle Encoding
    #

    offset=len(binary)

    for i,v in enumerate(continuous):

        qc.ry(

            v,

            offset+i

        )

    return qc


print("="*60)
print("STEP 6. Hybrid Encoding")
print("="*60)

hybrid = hybrid_encoding(

    binary_features[1],

    continuous_scaled[1]

)

print(

    hybrid.draw("text")

)

print("="*60)
print("STEP 7. Statevector")
print("="*60)

state = Statevector.from_instruction(

    hybrid

)

print(state)

prob = state.probabilities_dict()

for k,v in prob.items():

    if v>0.01:

        print(

            k,

            round(v,4)

        )

measure = hybrid.copy()

measure.measure_all()

print(

measure.draw("text")

)

print("="*60)
print("STEP 8. Simulation")
print("="*60)

sim = AerSimulator()

compiled = transpile(

    measure,

    sim

)

job = sim.run(

    compiled,

    shots=1024

)

result = job.result()

counts = result.get_counts()

print(counts)

print("="*60)
print("STEP 9. Circuit Analysis")
print("="*60)

print(

"Depth :",

hybrid.depth()

)

print(

"Qubits :",

hybrid.num_qubits

)

print(

"Gate :",

hybrid.count_ops()

)

print("="*60)
print("STEP 10. Batch Encoding")
print("="*60)

circuits=[]

for i in range(

        len(customer)

):

    qc = hybrid_encoding(

        binary_features[i],

        continuous_scaled[i]

    )

    circuits.append(qc)

print(

"Total Circuit :",len(circuits)

)

for i,qc in enumerate(circuits):

    print()

    print(

        "Customer",

        i+1

    )

    print(

        qc.draw("text")

    )

print("="*60)
print("STEP 11. Encoding Comparison")
print("="*60)

print(

"Basis"

)

print(

basis.count_ops()

)

print(

basis.depth()

)

print()

print(

"Angle"

)

print(

angle.count_ops()

)

print(

angle.depth()

)

print()

print(

"Hybrid"

)

print(

hybrid.count_ops()

)

print(

hybrid.depth()

)

print("="*60)
print("FINAL RESULT")
print("="*60)

print(

"Binary Feature → Basis Encoding"

)

print(

"Continuous Feature → Angle Encoding"

)

print(

"Hybrid Encoding Complete"

)

















