# Quantum ML Daily Google Drive Sync

## Source and scope

- Google Drive is observed as `Day1` through `Day39`, including nested `Day39/Project`.
- Source types are Python practice files, lecture PDFs, and the raw-only Netflix CSV.
- Preserve Drive IDs, original titles, raw storage, receipts, and static/Colab validation evidence.

## Public routing

- The only public roots are:
- `01.quantum-foundations/`
- `02.circuits-and-encoding/`
- `03.variational-learning-and-kernels/`
- `04.quantum-kernel-classification/`
- `05.quantum-neural-networks/`
- `06.qaoa-and-combinatorial-optimization/`
- `07.capstone/`
- Future publication destinations are exactly these implemented concept subfolders:
- `01.quantum-foundations/01.why-quantum-and-qml/`
- `01.quantum-foundations/02.bits-qubits-and-state/`
- `01.quantum-foundations/03.gates-measurement-and-entanglement/`
- `02.circuits-and-encoding/01.quantum-circuits-and-qml/`
- `02.circuits-and-encoding/02.feature-encoding/`
- `02.circuits-and-encoding/03.ansatz-and-parameterized-circuits/`
- `03.variational-learning-and-kernels/01.loss-and-optimization/`
- `03.variational-learning-and-kernels/02.quantum-kernels/`
- `04.quantum-kernel-classification/01.iris-qsvm/`
- `05.quantum-neural-networks/01.classical-neural-network-baseline/`
- `05.quantum-neural-networks/02.estimator-qnn/`
- `05.quantum-neural-networks/03.torchconnector-hybrid-qnn/`
- `06.qaoa-and-combinatorial-optimization/01.tsp-classical-baseline/`
- `06.qaoa-and-combinatorial-optimization/02.qaoa-objectives-and-optimizers/`
- `06.qaoa-and-combinatorial-optimization/03.qaoa-layers-and-execution/`
- `07.capstone/01.netflix-qml-project/`
- File each future artifact into its matching destination above; retain Day/source identity in the filename.
- Store downloaded bytes at `.omo/drive-sync/raw/<Day>/` (or `raw/Day39/Project/`) and never move the raw-only CSV into the public tree.

## Processing and publication

- Detect by Drive ID and `modified_time`; preserve the original Drive title in the manifest.
- Validate generated notebooks statically, then use Colab only for approved remote execution. Do not install dependencies locally.
- Record command, runner, log, dependency installs, marker, and result in a dated receipt.
- Stage explicit approved paths only; never broadly stage `.omo/drive-sync/`, the course tree, or the repository.
