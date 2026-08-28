"""
=========================================================
create_qnn.py

공통 라이브러리

EstimatorQNN 생성 함수

=========================================================

이 파일은 교육용 출력이 아닌
실습에서 공통으로 사용하는 라이브러리입니다.

사용 예

from create_qnn import create_qnn

qnn = create_qnn()

=========================================================
"""

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN


def create_qnn():

    """
    EstimatorQNN 생성

    Returns
    -------
    EstimatorQNN
    """

    # -----------------------------------------------------
    # Feature Map
    # -----------------------------------------------------

    feature_map = ZZFeatureMap(

        feature_dimension=2,

        reps=1

    )

    # -----------------------------------------------------
    # Ansatz
    # -----------------------------------------------------

    ansatz = RealAmplitudes(

        num_qubits=2,

        reps=1

    )

    # -----------------------------------------------------
    # Quantum Circuit
    # -----------------------------------------------------

    circuit = feature_map.compose(ansatz)

    # -----------------------------------------------------
    # Observable
    # -----------------------------------------------------

    observable = SparsePauliOp.from_list(

        [

            ("ZZ", 1.0)

        ]

    )

    # -----------------------------------------------------
    # Estimator
    # -----------------------------------------------------

    estimator = StatevectorEstimator()

    # -----------------------------------------------------
    # EstimatorQNN
    # -----------------------------------------------------

    qnn = EstimatorQNN(

        circuit=circuit,

        estimator=estimator,

        observables=observable,

        input_params=feature_map.parameters,

        weight_params=ansatz.parameters

    )

    return qnn


# =========================================================
# 단독 실행 테스트
# =========================================================

if __name__ == "__main__":

    qnn = create_qnn()

    print("=" * 60)
    print("EstimatorQNN Test")
    print("=" * 60)

    print()

    print("Input")

    print(qnn.num_inputs)

    print()

    print("Weight")

    print(qnn.num_weights)

    print()

    print("Output Shape")

    print(qnn.output_shape)

    print()

    print("Circuit")

    print(qnn.circuit.draw("text"))

    print()

    print("create_qnn() 테스트 완료")