"""
==========================================================
Lab 5
Quantum Circuit와 Objective Function 연결
==========================================================
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives import StatevectorEstimator


print("="*60)
print("STEP 1. Parameter 생성")
print("="*60)

theta = Parameter("θ")

print(theta)


# -------------------------------------------------

print("\n"+"="*60)
print("STEP 2. Quantum Circuit 생성")
print("="*60)

qc = QuantumCircuit(1)

qc.ry(theta,0)

print(qc.draw())


# -------------------------------------------------

print("\n"+"="*60)
print("STEP 3. Observable 생성")
print("="*60)

observable = SparsePauliOp("Z")

print(observable)


# -------------------------------------------------

print("\n"+"="*60)
print("STEP 4. Estimator 생성")
print("="*60)

estimator = StatevectorEstimator()

print("Estimator 생성 완료")


# -------------------------------------------------

print("\n"+"="*60)
print("STEP 5. Parameter Sweep")
print("="*60)

parameter_values = [

    0,
    np.pi/4,
    np.pi/2,
    3*np.pi/4,
    np.pi

]

for theta_value in parameter_values:

    job = estimator.run(
        [
            (
                qc,
                observable,
                [theta_value]
            )
        ]
    )

    result = job.result()

    expectation = result[0].data.evs

    print(
        f"Theta = {theta_value:8.5f}"
        f"    <Z> = {expectation:8.5f}"
    )


