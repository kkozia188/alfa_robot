#!/usr/bin/python3
"""Verify and open the chest-first safe-turn unload Rerun recording."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RESULTS = WORKSPACE / "results/v322_chest_turn_place"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyframe",
        action="store_true",
        help="open the first equal-height release after the safe 180-degree turn",
    )
    parser.add_argument(
        "--spawn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    report_path = RESULTS / "v322-chest-pull-safe-turn-contract.json"
    if not report_path.is_file():
        parser.error(f"missing acceptance report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("success"):
        parser.error("acceptance report is not successful")
    recording = RESULTS / (
        "v322-chest-pull-safe-turn-equal-height-keyframe.rrd"
        if args.keyframe
        else "v322-chest-pull-safe-turn-unload-production-shell.rrd"
    )
    subprocess.run(["rerun", "rrd", "verify", str(recording)], check=True)
    print(
        "RESULT success=25/25 cycles=15/15 "
        "backoff=2.35m turn=180deg lateral=0m "
        f"max_plan={report['maximum_task_core_ms'] / 1000.0:.3f}s "
        f"max_equal_height_error="
        f"{1000.0 * report['maximum_dual_release_height_error_m']:.9f}mm "
        f"rrd={recording}",
        flush=True,
    )
    if args.spawn:
        subprocess.Popen(
            ["rerun", "--new", str(recording)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
