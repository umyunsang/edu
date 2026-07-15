from qiskit.circuit.library import pauli_feature_map

feature_dim = 3

multi_pauli_maps = {
    "ZZ Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["ZZ"],
        entanglement="linear",
    ),

    "XX Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["XX"],
        entanglement="linear",
    ),

    "YY Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["YY"],
        entanglement="linear",
    ),

    "ZX Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["ZX"],
        entanglement="linear",
    ),

    "XYZ Mapping": pauli_feature_map(
        feature_dimension=feature_dim,
        reps=1,
        paulis=["XYZ"],
        entanglement="linear",
    )
}

for name, circuit in multi_pauli_maps.items():
    print("=" * 60)
    print(name)
    print(circuit)
    print("Depth :", circuit.depth())
    print("Gate Count :", circuit.count_ops())
