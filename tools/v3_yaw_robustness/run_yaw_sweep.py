#!/usr/bin/env python3
"""Run a sequential yaw chain, using each passed case to seed the next one."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASE_RUNNER = ROOT / "run_yaw_case.py"


def yaw_slug(yaw_deg: int) -> str:
    return f"yaw-{'p' if yaw_deg >= 0 else 'm'}{abs(yaw_deg):02d}"


def parse_yaws(value: str) -> list[int]:
    try:
        yaws = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("yaw list must contain integers") from error
    if not yaws or len(set(yaws)) != len(yaws) or any(not -5 <= yaw <= 5 for yaw in yaws):
        raise argparse.ArgumentTypeError("yaw list must contain unique integers in [-5, 5]")
    return yaws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaws", type=parse_yaws, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--initial-continuation-cache", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    continuation = (
        args.initial_continuation_cache.resolve()
        if args.initial_continuation_cache else None
    )
    completed = []
    for yaw in args.yaws:
        command = [
            "/usr/bin/python3", str(CASE_RUNNER),
            "--yaw-deg", str(yaw),
            "--baseline-cache", str(args.baseline_cache.resolve()),
            "--output-root", str(output_root),
        ]
        if continuation:
            command.extend(("--continuation-cache", str(continuation)))
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)
        case_dir = output_root / yaw_slug(yaw)
        summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
        if not summary.get("validation", {}).get("success"):
            raise RuntimeError(f"yaw {yaw:+d} did not pass validation")
        completed.append(yaw)
        continuation = case_dir / "v322-yaw-plan-cache.json"
        print(f"YAW_SWEEP completed={completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
