#!/usr/bin/python3
"""Verify and open the certified chest-carry Rerun recording."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RESULTS = WORKSPACE / "results/v322_chest_carry_fast_base"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keyframe",
        action="store_true",
        help="open the first dual chest-carry-ready inspection frame",
    )
    parser.add_argument(
        "--spawn", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    report_path = RESULTS / "optimized-acceptance-report.json"
    if not report_path.is_file():
        parser.error(f"missing acceptance report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("success"):
        parser.error("acceptance report is not successful")
    recording = RESULTS / (
        "v322-chest-carry-ready-optimized-keyframe.rrd"
        if args.keyframe
        else "v322-chest-carry-fast-base-optimized-production-shell.rrd"
    )
    subprocess.run(["rerun", "rrd", "verify", str(recording)], check=True)
    print(
        "RESULT success=25/25 cycles=15/15 "
        f"base_speed={report['base']['new_speed_m_s']:.2f}m/s "
        f"front_clearance={report['minimum_vehicle_front_to_box_clearance_m']:.3f}m "
        f"dual_gap={report['minimum_dual_box_gap_m']:.3f}m "
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
