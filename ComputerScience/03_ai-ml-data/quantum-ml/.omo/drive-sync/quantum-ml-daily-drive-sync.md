# Quantum ML Daily Google Drive Sync

## Source

- Google Drive folder: https://drive.google.com/drive/folders/1efBVhTfekDukwBtYe7aPrsa4lNhaeXnl
- Current observed structure: `Day1` through `Day8` subfolders.
- File types observed: PDF lecture files and Python practice files.

## Local Destination Rules

- Keep the public course surface in the existing shallow topic folders under this directory.
- Keep raw downloaded Drive files under `.omo/drive-sync/raw/<Day>/` for audit and repeatability.
- Keep sync state under `.omo/drive-sync/quantum-ml-drive-manifest.json`.
- Update `양자 ML 과정.md` when a new processed artifact is added to a topic folder.
- Do not install notebook dependencies locally. Use static validation for generated notebooks before remote execution.
- Execute newly generated Python-practice notebook sources through the Colab CLI after static validation. Colab runtime dependency installation is allowed only on the remote Colab VM and must be recorded in the run log.
- After Drive processing, static validation, and required Colab CLI execution pass, commit and push only the quantum-ml sync artifacts. Preserve unrelated dirty vault changes by staging explicit paths only.

## Processing Rules

- Detect changes by Drive file ID plus `modified_time`.
- Download only files that are new or whose `modified_time` changed since the manifest.
- For PDF files, normalize the filename when needed, then file it into a shallow topic folder that matches the content/title.
- For Python practice files, save the raw `.py`, convert it to a Colab-ready `.ipynb` in the matching topic folder, validate the notebook JSON plus Python syntax statically, then execute the source content through `~/.local/bin/colab run`.
- Because `colab run` executes `.py` scripts rather than `.ipynb` files, use the raw source or a self-contained runner generated from the raw source as the runtime proof for the corresponding notebook artifact.
- Store Colab execution logs under `.omo/drive-sync/colab-runs/YYYY-MM-DD/` and record the command, runner path, log path, dependency installs, success marker, and pass/fail status in the manifest.
- Preserve the original Drive title in the manifest even when the local artifact uses a corrected or normalized name.

## Git Publication Rules

- Commit and push are part of the automation only after the run status is PASS.
- Before committing, run the relevant static validations and confirm Colab CLI execution logs contain the expected success marker.
- Stage explicit paths for this lane only: `.omo/drive-sync/`, `1.quantum-ml-overview/`, and `양자 ML 과정.md`.
- Do not stage unrelated vault changes outside the `quantum-ml` sync lane.
- If push is rejected because the remote moved, run `git pull --rebase origin main`, rerun validation, then push.
- Record the commit hash, push result, and any remaining unrelated dirty worktree changes in the run log.

## Current Routing Hints

- `Hadamard`, `H`, `CX`, Bell-state, and superposition practice belongs under `1.quantum-ml-overview/hadamard-gate/` unless the content clearly belongs elsewhere.
- `state`, gate effects, or measurement probability practice belongs under `1.quantum-ml-overview/state-change-analysis/`.
- loss surface, binary classification dataset, PCA projection, or pipeline practice belongs under `1.quantum-ml-overview/qml-pipeline/`.
- Iris classifier practice belongs under `1.quantum-ml-overview/iris-classification/`.
- QML role material belongs under `1.quantum-ml-overview/quantum-role/`.

## Daily Automation Contract

Each run should leave a concise log under `.omo/drive-sync/runs/YYYY-MM-DD.md` with:

- Drive folders scanned.
- New or changed files detected.
- Local artifacts written or skipped because already current.
- Static validation commands and pass/fail results.
- Colab CLI execution command, pass/fail result, log path, and success marker for each newly generated Python-practice notebook batch.
- Commit hash and push result when publication succeeds.
- Any files that need user review because routing was ambiguous.
