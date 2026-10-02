#!/usr/bin/env python3
"""Record one certified yaw replay with the current V3.2.2 production URDF."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
sys.path.insert(0, str(BASELINE_DIR))
import v3_scoop_5x5_dual_rerun as dual  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_xacro = (
        Path(get_package_share_directory("alfa_robot_description"))
        / "urdf" / "alfa_robot.urdf.xacro"
    )
    urdf_text = subprocess.run(
        ["xacro", str(urdf_xacro)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    if "robot_v3_2_2" not in urdf_text and "part_065_solid_065" not in urdf_text:
        raise RuntimeError("rendered URDF is not the V3.2.2 production model")
    dual.sequence.render_current_urdf = lambda _mappings=None: urdf_text

    prefix = case_dir.name
    replay = case_dir / "v322-yaw-conveyor-replay.json"
    validation = case_dir / "v322-yaw-conveyor-validation.json"
    rrd = output_dir / f"{prefix}-v322-production-shell.rrd"
    summary = output_dir / f"{prefix}-summary.json"
    metrics = output_dir / f"{prefix}-metrics.csv"
    planning = output_dir / f"{prefix}-box-planning.csv"
    sys.argv = [
        sys.argv[0],
        "--input", str(replay),
        "--save", str(rrd),
        "--summary", str(summary),
        "--metrics-csv", str(metrics),
        "--box-planning-csv", str(planning),
        "--validation-report", str(validation),
        "--playback-speed", "2.0",
        "--no-spawn",
    ]
    return dual.main()


if __name__ == "__main__":
    raise SystemExit(main())
