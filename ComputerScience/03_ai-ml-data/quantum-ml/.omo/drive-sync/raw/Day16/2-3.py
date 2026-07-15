import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)

from qiskit.circuit.library import PauliFeatureMap

# =====================================================
# Feature Map 설정
# =====================================================

feature_dim = 3
reps = 1
entanglement = "linear"

# =====================================================
# Pauli 조합 정의
# =====================================================

combined_paulis = {
    "Z + ZZ": ["Z", "ZZ"],
    "X + XX": ["X", "XX"],
    "Y + YY": ["Y", "YY"],
    "X + Y + Z + ZZ": ["X", "Y", "Z", "ZZ"]
}

# =====================================================
# Feature Map 생성
# =====================================================

combined_maps = {}

for name, paulis in combined_paulis.items():

    combined_maps[name] = PauliFeatureMap(
        feature_dimension=feature_dim,
        reps=reps,
        paulis=paulis,
        entanglement=entanglement
    )

# =====================================================
# 결과 출력
# =====================================================

for name, fmap in combined_maps.items():

    circuit = fmap.decompose()

    print("=" * 60)
    print(name)
    print(circuit)
    print(f"Depth      : {circuit.depth()}")
    print(f"Gate Count : {circuit.count_ops()}")
