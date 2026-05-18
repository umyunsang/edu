#!/usr/bin/env python3
"""Best-effort Codex Stop hook summary for AIOSS practice sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def latest_eval(root: Path) -> str:
    runs = root / ".aioss-eval" / "runs"
    if not runs.exists():
        return "No evaluation run yet."
    files = sorted(runs.glob("*/*-sample-eval.json"))
    if not files:
        return "No evaluation JSON report yet."
    latest = files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        score = payload.get("score", {})
        return f"Latest eval {latest.name}: {score.get('passed')}/{score.get('total')} checks passed."
    except Exception as exc:  # noqa: BLE001 - hook must never block a session.
        return f"Latest eval exists but could not be parsed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    print(f"[aioss-hook] {latest_eval(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
