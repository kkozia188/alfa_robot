#!/usr/bin/python3
"""Measure chest pullback TCP length and deviation from its endpoint chord."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
RERUN_DIR = WORKSPACE / "ros2_ws/src/alfa_robot_rerun"
if str(RERUN_DIR) not in sys.path:
    sys.path.insert(0, str(RERUN_DIR))

from alfa_robot_rerun.visualize_rerun import UrdfRobot


DEFAULT_MODEL = Path(
    "/home/tim/alfa_robot-alfa_v3_dev/data/ik_benchmark/v3_scoop_5x5/"
    "range_certification_v322_tool0151/model/alfa-robot-v322-tool0151.urdf"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def frame_tcp(
    robot: UrdfRobot, payload: dict[str, Any], frame: dict[str, Any]
) -> np.ndarray:
    joints = {
        str(name): float(value)
        for name, value in zip(payload["joint_names"], frame["joints"])
    }
    return robot.fk(joints)[str(payload["tool_link"])][:3, 3]


def line_deviation(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    chord = end - start
    squared_length = float(chord @ chord)
    if squared_length <= 1e-18:
        return float(np.linalg.norm(point - start))
    progress = max(0.0, min(1.0, float((point - start) @ chord) / squared_length))
    return float(np.linalg.norm(point - (start + progress * chord)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--model-urdf", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    robot = UrdfRobot(args.model_urdf.resolve().read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for path in sorted(args.payload_dir.resolve().glob("box-*.json")):
        payload = read_json(path)
        if not payload.get("success"):
            continue
        frames = [
            frame for frame in payload["frames"]
            if frame["stage"] == "rrt_to_place"
        ]
        if len(frames) < 2:
            raise ValueError(f"{path} has fewer than two rrt_to_place frames")
        points = [frame_tcp(robot, payload, frame) for frame in frames]
        path_length = sum(
            float(np.linalg.norm(current - previous))
            for previous, current in zip(points, points[1:])
        )
        direct_length = float(np.linalg.norm(points[-1] - points[0]))
        ratio = path_length / direct_length if direct_length > 1e-9 else math.inf
        maximum_deviation = max(
            line_deviation(point, points[0], points[-1]) for point in points
        )
        search = payload.get("segment_search", [{}])[-1]
        cases.append({
            "box_id": int(payload["target_box_id"]),
            "side": str(payload["side"]),
            "mode": str(payload["grasp_mode"]),
            "strategy": str(search.get("strategy", "")),
            "planning_ms": float(payload["total_ms"]),
            "frames": len(frames),
            "tcp_path_m": path_length,
            "tcp_direct_m": direct_length,
            "path_to_direct_ratio": ratio,
            "maximum_line_deviation_m": maximum_deviation,
        })
    cases.sort(key=lambda item: int(item["box_id"]))
    report = {
        "schema": "alfa.v322_chest_pullback_path_analysis.v1",
        "case_count": len(cases),
        "maximum_planning_ms": max(item["planning_ms"] for item in cases),
        "maximum_path_to_direct_ratio": max(
            item["path_to_direct_ratio"] for item in cases
        ),
        "maximum_line_deviation_m": max(
            item["maximum_line_deviation_m"] for item in cases
        ),
        "cases": cases,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.resolve().write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
