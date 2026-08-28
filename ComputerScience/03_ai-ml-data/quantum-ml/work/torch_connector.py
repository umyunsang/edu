"""
=========================================================
torch_connector.py

공통 라이브러리

TorchConnector 생성 함수

=========================================================

이 파일은 교육용 출력이 아닌
실습에서 공통으로 사용하는 라이브러리입니다.

사용 예

from create_qnn import create_qnn
from torch_connector import create_torch_connector

qnn = create_qnn()

model = create_torch_connector(qnn)

=========================================================
"""

import torch
import torch.nn as nn

from qiskit_machine_learning.connectors import TorchConnector


def create_torch_connector(
    qnn,
    initial_weights=None
):
    """
    EstimatorQNN을 TorchConnector로 변환

    Parameters
    ----------
    qnn : EstimatorQNN

    initial_weights : array-like or Tensor, optional

    Returns
    -------
    TorchConnector
    """

    # -----------------------------------------------------
    # Initial Weight 지정
    # -----------------------------------------------------

    if initial_weights is None:

        model = TorchConnector(qnn)

    else:

        model = TorchConnector(

            qnn,

            initial_weights=initial_weights

        )

    return model


# =========================================================
# 단독 실행 테스트
# =========================================================

if __name__ == "__main__":

    from create_qnn import create_qnn

    qnn = create_qnn()

    model = create_torch_connector(qnn)

    print("=" * 60)
    print("TorchConnector Test")
    print("=" * 60)

    print()

    print("Model")

    print(model)

    print()

    print("Type")

    print(type(model))

    print()

    print("nn.Module")

    print(isinstance(model, nn.Module))

    print()

    print("Weight")

    print(model.weight)

    print()

    print("Weight Shape")

    print(model.weight.shape)

    print()

    print("requires_grad")

    print(model.weight.requires_grad)

    print()

    print("Parameters")

    for name, parameter in model.named_parameters():

        print(f"{name}")

        print(parameter)

        print()

    print("=" * 60)
    print("torch_connector.py 테스트 완료")
    print("=" * 60)