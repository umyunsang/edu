from qiskit.circuit.library import pauli_feature_map

feature_dim = 3

single_pauli_maps = {
    "X Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["X"]
    ),
    "Y Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Y"]
    ),
    "Z Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["Z"]
    )
}

for name, fmap in single_pauli_maps.items():
    print("=" * 60)
    print(name)
    print(fmap.decompose())
    print("Depth:", fmap.decompose().depth())
    print("Gate count:", fmap.decompose().count_ops())
