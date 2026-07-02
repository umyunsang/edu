# Day8 Notebook Processing

## Source

- Google Drive root: `https://drive.google.com/drive/folders/1efBVhTfekDukwBtYe7aPrsa4lNhaeXnl`
- Processed folder: `Day8`

## Files Processed

| Drive file | Drive ID | Local raw copy | Local notebook |
|---|---|---|---|
| `3-1.py` | `1C6bKbtut_BsZSmI3vr5Tm-z4nYGlODfj` | `.omo/drive-sync/raw/Day8/3-1.py` | `1.quantum-ml-overview/entanglement-and-cnot/3_1_cnot_basis_state_practice.ipynb` |
| `3-2.py` | `1V_BdwzK_aqMj1XcQTycxITq1dNo-5_F_` | `.omo/drive-sync/raw/Day8/3-2.py` | `1.quantum-ml-overview/entanglement-and-cnot/3_2_hadamard_superposition_practice.ipynb` |
| `3-3.py` | `1YVazSFPi6JlSEB7Igjarmr9a6JH9Wb5s` | `.omo/drive-sync/raw/Day8/3-3.py` | `1.quantum-ml-overview/entanglement-and-cnot/3_3_bell_state_practice.ipynb` |
| `4-1.py` | `1NsLjgxHgjjCz_CPPbrObcXUHnARyItjs` | `.omo/drive-sync/raw/Day8/4-1.py` | `1.quantum-ml-overview/entanglement-and-cnot/4_1_bell_measurement_practice.ipynb` |

## Processing

- Created the shallow topic folder `1.quantum-ml-overview/entanglement-and-cnot/`.
- Preserved the original Drive scripts under `.omo/drive-sync/raw/Day8/`.
- Converted each script into a Colab-ready tutorial notebook.
- Updated `.omo/drive-sync/quantum-ml-drive-manifest.json`.
- Updated `양자 ML 과정.md`.

## Validation

Validation was run after writing:

- `python3 -m json.tool` for each generated notebook.
- `ast.parse` for every code cell in each generated notebook.

Notebook execution was intentionally not run locally because this course lane keeps Qiskit practice notebooks Colab-ready and avoids local package installation/execution unless explicitly requested.
