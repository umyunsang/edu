# ============================================================
# PauliFeatureMap 구현 실습
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import pauli_feature_map
from qiskit import transpile

try:
    from qiskit_aer import AerSimulator
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False


# ============================================================
# STEP 1. 데이터 준비
# ============================================================

print("=" * 80)
print("STEP 1. 데이터 준비")
print("=" * 80)

iris = load_iris()

# Iris 데이터 중 앞 3개 feature만 사용
X = iris.data[:, :3]
y = iris.target

# 이진 분류 실습을 위해 class 0, 1만 사용
X = X[y != 2]
y = y[y != 2]

# Quantum Feature Map 입력 범위에 맞게 0 ~ pi로 스케일링
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

feature_dim = X_scaled.shape[1]

print("입력 데이터 shape:", X_scaled.shape)
print("라벨 shape:", y.shape)
print("사용 feature 수:", feature_dim)
print("샘플 데이터:")
print(X_scaled[:5])


# ============================================================
# STEP 2. PauliFeatureMap 후보 생성
# ============================================================

print("\n" + "=" * 80)
print("STEP 2. PauliFeatureMap 후보 생성")
print("=" * 80)

feature_maps = {
    "Z": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Z"],
        entanglement="linear"
    ),
    "X": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["X"],
        entanglement="linear"
    ),
    "Y": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Y"],
        entanglement="linear"
    ),
    "ZZ": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["ZZ"],
        entanglement="linear"
    ),
    "Z_ZZ": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Z", "ZZ"],
        entanglement="linear"
    ),
    "X_XX": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["X", "XX"],
        entanglement="linear"
    ),
    "Y_YY": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Y", "YY"],
        entanglement="linear"
    ),
    "Z_X_ZZ": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Z", "X", "ZZ"],
        entanglement="linear"
    ),
    "Z_ZZ_XYZ": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Z", "ZZ", "XYZ"],
        entanglement="linear"
    )
}

print("생성된 Feature Map 목록:")
for name in feature_maps.keys():
    print("-", name)


# ============================================================
# STEP 3. 회로 구조 출력
# ============================================================

print("\n" + "=" * 80)
print("STEP 3. 회로 구조 출력")
print("=" * 80)

for name, fmap in feature_maps.items():
    print("\n" + "-" * 80)
    print(f"Feature Map: {name}")
    print("-" * 80)
    print(fmap.decompose())


# ============================================================
# STEP 4. 회로 복잡도 분석 함수 정의
# ============================================================

def analyze_circuit(name, circuit):
    decomposed = circuit.decompose()
    ops = decomposed.count_ops()

    return {
        "Feature Map": name,
        "Qubits": decomposed.num_qubits,
        "Parameters": len(decomposed.parameters),
        "Depth": decomposed.depth(),
        "Size": decomposed.size(),
        "CX Count": ops.get("cx", 0),
        "H Count": ops.get("h", 0),
        "RX Count": ops.get("rx", 0),
        "RY Count": ops.get("ry", 0),
        "RZ Count": ops.get("rz", 0),
        "P Count": ops.get("p", 0),
        "Gate Summary": dict(ops)
    }


# ============================================================
# STEP 5. Feature Map별 회로 복잡도 비교
# ============================================================

print("\n" + "=" * 80)
print("STEP 5. Feature Map별 회로 복잡도 비교")
print("=" * 80)

analysis_results = []

for name, fmap in feature_maps.items():
    analysis_results.append(analyze_circuit(name, fmap))

df_analysis = pd.DataFrame(analysis_results)

print(df_analysis[
    [
        "Feature Map",
        "Qubits",
        "Parameters",
        "Depth",
        "Size",
        "CX Count",
        "H Count",
        "RX Count",
        "RY Count",
        "RZ Count",
        "P Count"
    ]
])


# ============================================================
# STEP 6. Entanglement 구조 비교
# ============================================================

print("\n" + "=" * 80)
print("STEP 6. Entanglement 구조 비교")
print("=" * 80)

entanglement_types = ["linear", "full", "circular"]
entanglement_maps = {}

for ent in entanglement_types:
    entanglement_maps[f"Z_ZZ_{ent}"] = pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Z", "ZZ"],
        entanglement=ent
    )

ent_results = []

for name, fmap in entanglement_maps.items():
    ent_results.append(analyze_circuit(name, fmap))

df_ent = pd.DataFrame(ent_results)

print(df_ent[
    [
        "Feature Map",
        "Qubits",
        "Parameters",
        "Depth",
        "Size",
        "CX Count",
        "P Count"
    ]
])


# ============================================================
# STEP 7. reps 반복 횟수 비교
# ============================================================

print("\n" + "=" * 80)
print("STEP 7. reps 반복 횟수 비교")
print("=" * 80)

reps_maps = {}

for r in [1, 2, 3]:
    reps_maps[f"Z_ZZ_reps_{r}"] = pauli_feature_map(
        feature_dimension=feature_dim,
        reps=r,
        paulis=["Z", "ZZ"],
        entanglement="linear"
    )

reps_results = []

for name, fmap in reps_maps.items():
    reps_results.append(analyze_circuit(name, fmap))

df_reps = pd.DataFrame(reps_results)

print(df_reps[
    [
        "Feature Map",
        "Qubits",
        "Parameters",
        "Depth",
        "Size",
        "CX Count",
        "P Count"
    ]
])


# ============================================================
# STEP 8. 데이터 샘플 바인딩
# ============================================================

print("\n" + "=" * 80)
print("STEP 8. 데이터 샘플 바인딩")
print("=" * 80)

sample = X_scaled[0]
target_map = feature_maps["Z_ZZ"]

# 파라미터 이름 기준으로 정렬
params = sorted(target_map.parameters, key=lambda p: p.name)

param_dict = {
    param: value
    for param, value in zip(params, sample)
}

print("입력 샘플:")
print(sample)

print("\nParameter Binding 결과:")
for param, value in param_dict.items():
    print(f"{str(param):15} -> {value:.4f}")

bound_circuit = target_map.assign_parameters(param_dict)

print("\n바인딩된 회로:")
print(bound_circuit.decompose())


# ============================================================
# STEP 9. Transpile 후 회로 복잡도 비교
# ============================================================

print("\n" + "=" * 80)
print("STEP 9. Transpile 후 회로 복잡도 비교")
print("=" * 80)

if AER_AVAILABLE:
    backend = AerSimulator()

    transpile_results = []

    for name, fmap in feature_maps.items():
        original = fmap.decompose()

        transpiled = transpile(
            original,
            backend=backend,
            optimization_level=1
        )

        ops_after = transpiled.count_ops()

        transpile_results.append({
            "Feature Map": name,
            "Depth Before": original.depth(),
            "Depth After": transpiled.depth(),
            "Size After": transpiled.size(),
            "CX After": ops_after.get("cx", 0),
            "Gate Summary After": dict(ops_after)
        })

    df_transpile = pd.DataFrame(transpile_results)

    print(df_transpile[
        [
            "Feature Map",
            "Depth Before",
            "Depth After",
            "Size After",
            "CX After"
        ]
    ])

else:
    print("qiskit-aer가 설치되어 있지 않습니다.")
    print("설치 명령어: pip install qiskit-aer")


# ============================================================
# STEP 10. 최종 Feature Map 후보 추천
# ============================================================

print("\n" + "=" * 80)
print("STEP 10. 최종 Feature Map 후보 추천")
print("=" * 80)

# 단순 추천 로직:
# Depth와 CX Count가 낮으면서 Z와 ZZ를 모두 포함한 Feature Map 우선 추천
candidate_df = df_analysis.copy()

candidate_df["Score"] = (
    candidate_df["Depth"] +
    candidate_df["CX Count"] * 2 +
    candidate_df["Size"] * 0.1
)

candidate_df = candidate_df.sort_values(by="Score")

print("복잡도 기준 추천 순위:")
print(candidate_df[
    [
        "Feature Map",
        "Depth",
        "Size",
        "CX Count",
        "Score"
    ]
])
