# Repository Guidelines

## Project Structure & Module Organization
Main study notes live in `ComputerScience/`, organized by semester-coded folders such as `[2-1] 데이터 구조`. Each course folder keeps topic subfolders (`강의/`, `실습/`, `과제/`), and note leaves end with `.md`. Shared media stays in `image/`, while course-specific exports (for example `fcfs.c.md`) sit alongside their parent notes. Additional micro-vaults under `Obsidian/` (e.g., `KR_history/`, `constellation/`) should preserve their own internal outlines. Vault settings, plugin configs, and theme snippets remain in `.obsidian/`; touch them only when amending workspace-wide behaviour.

## Build, Test, and Development Commands
This vault has no compile step, but run quick checks before syncing:
- `git status` — confirm the obsidian-git queue matches expectations.
- `rg "\[\[" ComputerScience` — surface new wikilinks to verify each target exists.
- `find image -type f -mtime -1` — list freshly attached media so you can caption or relocate them.

## Coding Style & Naming Conventions
Keep directory names in the `[semester] 과목명` pattern and prefer sentence-case note titles (`CPU Scheduling.md`). Start documents with a top-level heading mirroring the filename, add lightweight tables of contents only when notes grow long, and use Markdown lists instead of ad-hoc numbering. Fence code blocks with language tags (` ```c `) and append `.md` to exported source so syntax highlighting is preserved within the vault. When embedding figures, store the asset under `image/` and reference it with a relative path.

## Testing Guidelines
Open the vault in Obsidian and use the backlinks or graph view to confirm new notes connect to existing material. When adding runnable code, capture a minimal invocation and its output in the same note for reproducibility. Run the `obsidian-git: Check for changes` command inside Obsidian to review diffs, and ensure cross-file embeds render correctly by following each wikilink once.

## Commit & Pull Request Guidelines
Follow the existing `vault backup: YYYY-MM-DD HH:MM.` convention produced by obsidian-git; when a manual summary is clearer, keep it concise (`Reorganize neural network study materials`). Squash incremental edits before opening a PR, describe the touched paths (`ComputerScience/[2-1] ...`), and link the related coursework issue or syllabus page. Attach screenshots for diagram-heavy updates and call out any plugin or `.obsidian` adjustments so reviewers can replicate the setup.
