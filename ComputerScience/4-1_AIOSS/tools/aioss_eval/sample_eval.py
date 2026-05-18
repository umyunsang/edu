#!/usr/bin/env python3
"""Deterministic checks for the AIOSS sample practice package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def run(cmd: list[str], cwd: Path, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def load_module(module_path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_todo_markers(target: Path) -> Check:
    scanned_suffixes = {".py", ".yml", ".yaml"}
    scanned_names = {"PR_TEMPLATE.md", "TDD_CYCLE.md"}
    markers: list[str] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in scanned_suffixes and path.name not in scanned_names:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "TODO" in line or "NotImplementedError" in line:
                rel = path.relative_to(target)
                markers.append(f"{rel}:{line_no}: {line.strip()}")
    detail = "No TODO or NotImplementedError markers in practice files."
    if markers:
        detail = "\n".join(markers[:20])
    return Check("todo-marker-scan", not markers, detail)


def eval_sample1(target: Path) -> list[Check]:
    sample = target / "sample-1-collaboration"
    checks: list[Check] = []
    greeting_path = sample / "app" / "greeting.py"
    if not greeting_path.exists():
        return [Check("sample1-greeting-file", False, f"Missing {greeting_path}")]

    try:
        module = load_module(greeting_path, "aioss_sample1_greeting")
        cases = [("AIOSS", "Hello, AIOSS!"), ("", "Hello, Guest!")]
        failures = []
        for value, expected in cases:
            actual = module.get_greeting(value)
            if actual != expected:
                failures.append(f"get_greeting({value!r}) -> {actual!r}, expected {expected!r}")
        checks.append(
            Check(
                "sample1-functional-greeting",
                not failures,
                "Greeting cases passed." if not failures else "\n".join(failures),
            )
        )
    except Exception as exc:  # noqa: BLE001 - evaluator reports raw failure mode.
        checks.append(Check("sample1-functional-greeting", False, repr(exc)))

    template = sample / "PR_TEMPLATE.md"
    if template.exists():
        text = template.read_text(encoding="utf-8", errors="replace")
        ready = "TODO" not in text and "- [x]" in text and (
            "rollback" in text.lower() or "롤백" in text
        )
        checks.append(
            Check(
                "sample1-pr-template-readiness",
                ready,
                "Template has completed checklist and rollback plan."
                if ready
                else "Template still needs completed checklist, rollback plan, or TODO removal.",
            )
        )
    else:
        checks.append(Check("sample1-pr-template-readiness", False, "Missing PR_TEMPLATE.md"))
    return checks


def eval_sample2(target: Path) -> list[Check]:
    sample = target / "sample-2-ci-basics"
    workflow = sample / ".github" / "workflows" / "ci.yml"
    checks: list[Check] = []
    if not workflow.exists():
        return [Check("sample2-workflow-file", False, f"Missing {workflow}")]

    text = workflow.read_text(encoding="utf-8", errors="replace")
    required = {
        "push trigger": "push:" in text,
        "pull request trigger": "pull_request:" in text,
        "checkout action": "actions/checkout" in text,
        "setup python action": "actions/setup-python" in text,
        "python 3.10": "3.10" in text,
        "ruff check": "ruff check" in text,
    }
    missing = [name for name, ok in required.items() if not ok]
    checks.append(
        Check(
            "sample2-workflow-content",
            not missing,
            "Workflow contains expected CI steps." if not missing else "Missing: " + ", ".join(missing),
        )
    )

    if shutil.which("actionlint"):
        result = run(["actionlint", str(workflow)], cwd=ROOT)
        checks.append(
            Check(
                "sample2-actionlint",
                result.returncode == 0,
                result.stdout.strip() or "actionlint passed.",
            )
        )
    else:
        checks.append(Check("sample2-actionlint", False, "actionlint is not installed."))
    return checks


def eval_sample3(target: Path) -> list[Check]:
    sample = target / "sample-3-testing"
    checks: list[Check] = []
    test_file = sample / "tests" / "test_calculator.py"
    app_file = sample / "app" / "calculator.py"
    if not test_file.exists() or not app_file.exists():
        return [Check("sample3-files", False, "Missing calculator app or test file.")]

    text = test_file.read_text(encoding="utf-8", errors="replace")
    has_add_assert = "add(2, 3)" in text and "== 5" in text
    has_subtract_assert = "subtract(10, 3)" in text and "== 7" in text
    checks.append(
        Check(
            "sample3-test-intent",
            has_add_assert and has_subtract_assert,
            "Tests cover add and subtract happy paths."
            if has_add_assert and has_subtract_assert
            else "Expected add and subtract assertions were not both found.",
        )
    )

    result = run([sys.executable, "-m", "pytest", "tests/test_calculator.py", "-q"], cwd=sample)
    checks.append(Check("sample3-pytest", result.returncode == 0, result.stdout.strip()))

    if shutil.which("ruff"):
        lint = run(["ruff", "check", "app", "tests"], cwd=sample)
        checks.append(Check("sample3-ruff", lint.returncode == 0, lint.stdout.strip() or "ruff passed."))
    else:
        checks.append(Check("sample3-ruff", False, "ruff is not installed."))
    return checks


def write_report(target: Path, label: str, checks: list[Check]) -> Path:
    run_dir = ROOT / ".aioss-eval" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for check in checks if check.passed)
    payload = {
        "label": label,
        "target": str(target),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score": {"passed": passed, "total": len(checks)},
        "checks": [check.to_json() for check in checks],
        "evaluation_lenses": [
            "functional correctness",
            "CI syntax and reproducibility",
            "shift-left testing evidence",
            "open source readiness",
            "TODO debt removal",
        ],
    }
    json_path = run_dir / f"{label}-sample-eval.json"
    md_path = run_dir / f"{label}-sample-eval.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# AIOSS Sample Evaluation: {label}",
        "",
        f"- Target: `{target}`",
        f"- Score: {passed}/{len(checks)}",
        "",
        "## Checks",
    ]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        detail = check.detail.replace("\n", "\n  ")
        lines.append(f"- {status} `{check.name}`: {detail}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="sample-solutions or sample-solutions-minimal directory")
    parser.add_argument("--label", default="sample", help="report label")
    args = parser.parse_args()

    target = (ROOT / args.target).resolve() if not Path(args.target).is_absolute() else Path(args.target)
    checks: list[Check] = []
    checks.extend(eval_sample1(target))
    checks.extend(eval_sample2(target))
    checks.extend(eval_sample3(target))
    checks.append(scan_todo_markers(target))
    report = write_report(target, args.label, checks)

    passed = sum(1 for check in checks if check.passed)
    print(f"{args.label}: {passed}/{len(checks)} checks passed")
    print(f"report: {report}")
    for check in checks:
        if not check.passed:
            print(f"FAIL {check.name}: {check.detail}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
